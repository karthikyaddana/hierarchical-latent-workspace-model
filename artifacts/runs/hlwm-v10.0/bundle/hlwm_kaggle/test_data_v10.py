"""Tests for the Version 10.0 anchor collator and its data contract.

Row source: the v10 builder's row function
(``scripts/build_hlwm_v100_bundle.behavior_anchor_rows_v10``) with a tiny
count when the repo's ``scripts/`` tree is present (local development and
builds); inside a shipped bundle (Kaggle) the builder is absent, so the rows
come from the shipped ``data/master/test.jsonl``, whose anchors carry the
same payloads.  Both sources produce raw episodes that are normalized with
``normalize_episode`` exactly as training does.

Tokenizer: the end-to-end test runs ``train_kaggle.v10_training_step`` on an
``HLWMConfig.tiny``-scale model whose vocabulary is 41, so the real Qwen
tokenizer (vocab 151k) cannot feed it.  The pragmatic path (documented per
the build spec): ONE deterministic character-level tokenizer whose ids live
inside the tiny vocabulary is used for every test.  Space and the ten digits
get RESERVED unique ids, so token-subsequence leak checks on numeric
literals are exact — a spurious match would need the literal's digit string
to actually appear in the text, which the string-level scan already
excludes; all other characters hash into the remaining slots (collisions
harmless there).  Real-tokenizer guarantees (>=8 trace tokens, no eos/pad in
windows, <=30 tokens/step) are the BUILDER's assertions, exercised by its
build-time verification and by ``scripts/test_hlwm_v10_traces.py``.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import pytest
import torch

BUNDLE = Path(__file__).resolve().parent
if str(BUNDLE) not in sys.path:
    sys.path.insert(0, str(BUNDLE))

from data import (
    V10AnchorCollator,
    V10_FAMILY_INDEX,
    V10_ROUTE_BY_FAMILY,
    encode_trace_segments,
    normalize_episode,
)
from modeling_hlwm import HLWMConfig, HLWMForConditionalGeneration


def _builder_module():
    try:
        root = BUNDLE.parents[4]
    except IndexError:
        return None
    if not (root / "scripts" / "build_hlwm_v100_bundle.py").exists():
        return None
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts import build_hlwm_v100_bundle

    return build_hlwm_v100_bundle


# Step regexes: the trace library's own validators when the repo tree is
# available; verbatim fallback copies inside a shipped bundle (the library
# lives in scripts/ and does not ship).
try:  # pragma: no cover - exercised implicitly by whichever env runs this
    _builder_module()
    from scripts.hlwm_v10_traces import EQUATION_STEP_RE, SORTED_STEP_RE
except Exception:  # pragma: no cover
    EQUATION_STEP_RE = re.compile(r"<<(-?\d+) ([+\-*]) (-?\d+) = (-?\d+)>>")
    SORTED_STEP_RE = re.compile(r"<<sorted: (\d+(?: \d+)*)>>")


class _TinyCharTokenizer:
    """Deterministic char-level tokenizer inside HLWMConfig.tiny's 41 vocab.

    pad=0 / bos=1 / eos=2 match ``HLWMConfig.tiny``; every emitted id is in
    [3, 40], so pad/eos can never appear inside a trace and every id embeds
    in the tiny model.  See the module docstring for the reserved-id design.
    """

    pad_token_id = 0
    eos_token_id = 2
    pad_token = "<pad>"
    eos_token = "<eos>"
    vocab_size = 41

    _RESERVED = {
        " ": 3, "0": 4, "1": 5, "2": 6, "3": 7, "4": 8,
        "5": 9, "6": 10, "7": 11, "8": 12, "9": 13,
    }

    def encode(self, text, add_special_tokens=False):
        ids = []
        for char in str(text):
            reserved = self._RESERVED.get(char)
            ids.append(reserved if reserved is not None else 14 + (ord(char) % 27))
        return ids

    def decode(self, ids, skip_special_tokens=True):
        reverse = {value: key for key, value in self._RESERVED.items()}
        return "".join(reverse.get(int(item), "?") for item in ids)


def _raw_rows(count=32):
    module = _builder_module()
    if module is not None:
        return module.behavior_anchor_rows_v10("train", count, 910, tokenizer=None)
    data_path = BUNDLE / "data" / "master" / "test.jsonl"
    if not data_path.exists():
        pytest.skip("neither the v10 builder nor shipped v10 master data is available")
    rows = []
    with data_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row.get("v10"), dict):
                rows.append(row)
            if len(rows) >= count:
                break
    if len(rows) < count:
        pytest.skip("shipped master data has fewer than %d v10 anchors" % count)
    return rows


def _collator(trace_tokens=192):
    # context_tokens=256 keeps the char-level truncation head (70% = 179
    # chars) wide enough to always cover the request's masked-operand region;
    # at 64 one ordering template's operands start past the head and the
    # masked/full prompts truncate to the same ids (a harness artifact —
    # production runs 256 REAL tokens, several times the whole prompt).
    return V10AnchorCollator(
        _TinyCharTokenizer(),
        latent_thoughts=6,
        trace_tokens=trace_tokens,
        num_lanes=1,
        context_tokens=256,
        canvas_tokens=16,
        brief_tokens=16,
        causal_tokens=96,
    )


_CACHE = {}


def _rows_and_batch():
    if "batch" not in _CACHE:
        rows = [normalize_episode(row, num_lanes=1) for row in _raw_rows(32)]
        _CACHE["rows"] = rows
        _CACHE["batch"] = _collator()(rows)
    return _CACHE["rows"], _CACHE["batch"]


def _tiny_v10_config(**overrides):
    values = {
        "latent_thoughts": 6,
        "kv_prefix_slots": 4,
        "kv_prefix_rank": 8,
        "response_cue_ids": (5, 7),
        "lora_rank": 0,
        "lora_tail_layers": 0,
        "max_position_embeddings": 512,
        "num_lanes": 1,
    }
    values.update(overrides)
    return HLWMConfig.tiny(**values)


def _contains_subsequence(haystack, needle):
    if not needle or len(needle) > len(haystack):
        return False
    for start in range(len(haystack) - len(needle) + 1):
        if haystack[start : start + len(needle)] == needle:
            return True
    return False


# ---------------------------------------------------------------------------
# (a) every contract key, right shapes and dtypes, on 32 generated rows.
# ---------------------------------------------------------------------------

def test_collator_emits_every_contract_key_with_shapes_and_dtypes():
    rows, batch = _rows_and_batch()
    count = len(rows)
    trace_budget = 192
    two_dim = {
        "input_ids", "attention_mask",
        "student_input_ids", "student_attention_mask",
        "target_ids", "target_attention_mask",
        "trace_input_ids", "trace_attention_mask", "teacher_supervised_mask",
        "corrupt_trace_input_ids", "corrupt_trace_attention_mask",
        "premise_ids", "premise_attention_mask", "premise_negative_ids",
    }
    one_dim = {"corrupt_step_index", "family_index", "route_index", "masked_rows"}
    for key in two_dim | one_dim | {"trace_window_targets"}:
        assert key in batch, "missing contract key %s" % key
        tensor = batch[key]
        assert isinstance(tensor, torch.Tensor), key
        assert tensor.shape[0] == count, key
        expected_dtype = torch.bool if key == "masked_rows" else torch.long
        assert tensor.dtype == expected_dtype, key
        assert tensor.dim() == (
            3 if key == "trace_window_targets" else 2 if key in two_dim else 1
        ), key

    for key in (
        "trace_input_ids", "trace_attention_mask", "teacher_supervised_mask",
        "corrupt_trace_input_ids", "corrupt_trace_attention_mask",
    ):
        assert batch[key].shape[1] == trace_budget, key
    windows = batch["trace_window_targets"]
    assert windows.shape[1] == 6

    positions = torch.arange(trace_budget)
    lengths = batch["trace_attention_mask"].sum(dim=1)
    # Right-padded: the attention mask is a clean ones-prefix per row.
    assert torch.equal(
        batch["trace_attention_mask"], (positions[None, :] < lengths[:, None]).long()
    )
    pad_id = _TinyCharTokenizer.pad_token_id
    assert (batch["trace_input_ids"][batch["trace_attention_mask"] == 0] == pad_id).all()
    # Supervision is inside the valid span and always excludes something.
    assert (batch["teacher_supervised_mask"] <= batch["trace_attention_mask"]).all()
    assert (batch["teacher_supervised_mask"].sum(dim=1) < lengths).all()
    # Window targets exactly cover each row's valid trace ids, in order.
    for index in range(count):
        covered = []
        for window in range(6):
            ids = windows[index, window]
            ids = ids[ids != -100]
            assert ids.numel() > 0, "empty window %d on row %d" % (window, index)
            covered.extend(ids.tolist())
        valid = int(lengths[index])
        assert covered == batch["trace_input_ids"][index, :valid].tolist()


def test_collator_rejects_rows_without_v10_payload():
    rows, _ = _rows_and_batch()
    stripped = dict(rows[0])
    stripped["v10"] = None
    with pytest.raises(ValueError):
        _collator()([stripped])


# ---------------------------------------------------------------------------
# (b) teacher_supervised_mask excludes exactly the final step's tokens.
# ---------------------------------------------------------------------------

def test_teacher_supervised_mask_excludes_exactly_the_final_step():
    rows, batch = _rows_and_batch()
    tokenizer = _TinyCharTokenizer()
    saw_multi_step = saw_single_step = False
    for index, row in enumerate(rows):
        payload = row["v10"]
        steps = [str(step) for step in payload["steps"]]
        # Independent expected arithmetic: first step bare, later steps
        # space-prefixed (the compositional encoding contract).
        segment_lengths = [
            len(tokenizer.encode(step if position == 0 else " " + step))
            for position, step in enumerate(steps)
        ]
        valid = int(batch["trace_attention_mask"][index].sum())
        assert sum(segment_lengths) == valid
        supervised_steps = len(steps) - int(payload["teacher_excluded_steps"])
        expected = sum(segment_lengths[:supervised_steps])
        supervised = int(batch["teacher_supervised_mask"][index].sum())
        assert supervised == expected
        # Clean prefix: ones then zeros, never interleaved.
        assert bool(batch["teacher_supervised_mask"][index, :supervised].all())
        assert int(batch["teacher_supervised_mask"][index, supervised:].sum()) == 0
        if len(payload["gold_steps"]) > 1:
            saw_multi_step = True
            assert supervised > 0
        else:
            # Single-step traces: the only step produces the answer, so the
            # teacher mask is empty by contract.
            assert supervised == 0
            saw_single_step = True
    assert saw_multi_step and saw_single_step


# ---------------------------------------------------------------------------
# (c) window boundaries match compressed_gold_thoughts exactly.
# ---------------------------------------------------------------------------

def test_trace_window_targets_match_compressed_gold_thoughts():
    from train_kaggle import compressed_gold_thoughts

    rows, batch = _rows_and_batch()
    torch.manual_seed(73)
    model = HLWMForConditionalGeneration(_tiny_v10_config()).eval()
    windows = 6
    targets = batch["trace_window_targets"]
    with torch.no_grad():
        pooled = compressed_gold_thoughts(
            model, batch["trace_input_ids"], batch["trace_attention_mask"], windows
        )
        expected = torch.zeros_like(pooled)
        for index in range(targets.shape[0]):
            for window in range(windows):
                ids = targets[index, window]
                ids = ids[ids != -100]
                span = model.backbone.embed_tokens(ids)
                expected[index, window] = span.sum(dim=0) / float(ids.numel()) ** 0.5
        expected = model.ground_latents(expected)
    assert torch.allclose(pooled, expected, atol=1.0e-5), (
        "window targets disagree with compressed_gold_thoughts boundaries"
    )
    # And the raw integer boundary arithmetic, row by row.
    for index in range(targets.shape[0]):
        valid = int(batch["trace_attention_mask"][index].sum())
        bounds = [(valid * position) // windows for position in range(windows + 1)]
        widths = [int((targets[index, window] != -100).sum()) for window in range(windows)]
        assert widths == [bounds[position + 1] - bounds[position] for position in range(windows)]


# ---------------------------------------------------------------------------
# (d) masked rows differ and are leak-free; unmasked rows are identical.
# ---------------------------------------------------------------------------

def test_masked_student_prompts_differ_and_carry_no_withheld_literal():
    rows, batch = _rows_and_batch()
    tokenizer = _TinyCharTokenizer()
    masked = batch["masked_rows"]
    assert bool(masked.any()) and bool((~masked).any())
    for index, row in enumerate(rows):
        student = batch["student_input_ids"][index]
        full = batch["input_ids"][index]
        if bool(masked[index]):
            assert not torch.equal(student, full)
            student_ids = student.tolist()
            assert row["withheld_literals"], "masked row without withheld literals"
            for literal in row["withheld_literals"]:
                span = tokenizer.encode(" " + str(literal))
                assert not _contains_subsequence(student_ids, span), (
                    "withheld literal %r leaked into student_input_ids of %s"
                    % (literal, row["episode_id"])
                )
        else:
            assert torch.equal(student, full)
            assert torch.equal(
                batch["student_attention_mask"][index], batch["attention_mask"][index]
            )


# ---------------------------------------------------------------------------
# (e) family and route indices per kind.
# ---------------------------------------------------------------------------

def test_family_and_route_indices_match_the_kind():
    rows, batch = _rows_and_batch()
    assert V10_ROUTE_BY_FAMILY == (0, 0, 1, 1)
    for index, row in enumerate(rows):
        kind = str(row["difficulty_params"]["kind"])
        assert row["v10"]["family"] == kind
        family = V10_FAMILY_INDEX[kind]
        assert int(batch["family_index"][index]) == family
        assert int(batch["route_index"][index]) == V10_ROUTE_BY_FAMILY[family]
    assert set(batch["family_index"].tolist()) == {0, 1, 2, 3}
    assert set(batch["route_index"].tolist()) == {0, 1}


# ---------------------------------------------------------------------------
# (f) corruption: in range, differs at exactly the recorded step.
# ---------------------------------------------------------------------------

def _assert_equation_corruption(payload):
    gold = [str(step) for step in payload["gold_steps"]]
    corrupt = [str(step) for step in payload["corrupt_steps"]]
    index = int(payload["corrupt_step_index"])
    assert 0 <= index < len(gold)
    assert corrupt[:index] == gold[:index]
    assert corrupt[index] != gold[index]
    # Exactly one internally-wrong derivation (the trace-library contract:
    # later steps cascade truthfully from the corrupted running value).
    wrong = []
    for position in range(len(gold)):
        match = EQUATION_STEP_RE.fullmatch(corrupt[position])
        assert match, corrupt[position]
        left, operator, right, result = match.groups()
        left, right, result = int(left), int(right), int(result)
        computed = {"+": left + right, "-": left - right, "*": left * right}[operator]
        if computed != result:
            wrong.append(position)
        if position > 0:
            previous = EQUATION_STEP_RE.fullmatch(corrupt[position - 1])
            assert left == int(previous.group(4))
    assert wrong == [index]


def _assert_ordering_corruption(payload):
    assert int(payload["corrupt_step_index"]) == 0
    gold_match = SORTED_STEP_RE.fullmatch(str(payload["gold_steps"][0]))
    corrupt_match = SORTED_STEP_RE.fullmatch(str(payload["corrupt_steps"][0]))
    assert gold_match and corrupt_match
    gold_positions = [int(token) for token in gold_match.group(1).split(" ")]
    corrupt_positions = [int(token) for token in corrupt_match.group(1).split(" ")]
    assert corrupt_positions != gold_positions
    assert sorted(corrupt_positions) == sorted(gold_positions)  # still a permutation


def test_corruption_is_single_step_in_range_with_abstention_sentinel():
    rows, batch = _rows_and_batch()
    families_checked = set()
    for index, row in enumerate(rows):
        payload = row["v10"]
        family = payload["family"]
        families_checked.add(family)
        if family == "abstention":
            assert int(batch["corrupt_step_index"][index]) == -1
            assert int(batch["corrupt_trace_attention_mask"][index].sum()) == 0
            assert payload["corrupt_steps"] == [] and payload["corrupt_trace"] == ""
            continue
        assert int(batch["corrupt_step_index"][index]) == int(payload["corrupt_step_index"])
        assert int(batch["corrupt_trace_attention_mask"][index].sum()) > 0
        # The corrupted trace tensor differs from the gold trace tensor.
        assert not torch.equal(
            batch["corrupt_trace_input_ids"][index], batch["trace_input_ids"][index]
        )
        if family == "ordering":
            _assert_ordering_corruption(payload)
        else:
            _assert_equation_corruption(payload)
    assert families_checked == {"numeric", "unit", "ordering", "abstention"}


# ---------------------------------------------------------------------------
# shared encoding helper: exact segment boundary under any tokenizer.
# ---------------------------------------------------------------------------

def test_encode_trace_segments_boundary_and_guards():
    tokenizer = _TinyCharTokenizer()
    steps = ["<<417 + 88 = 505>>", "<<505 - 3 = 502>>"]
    ids, excluded = encode_trace_segments(tokenizer, steps, 1)
    assert ids == tokenizer.encode(steps[0]) + tokenizer.encode(" " + steps[1])
    assert excluded == len(tokenizer.encode(" " + steps[1]))
    _, all_excluded = encode_trace_segments(tokenizer, steps, 2)
    assert all_excluded == len(ids)
    with pytest.raises(ValueError):
        encode_trace_segments(tokenizer, steps, 0)
    with pytest.raises(ValueError):
        encode_trace_segments(tokenizer, steps, 3)
    with pytest.raises(ValueError):
        encode_trace_segments(tokenizer, [], 1)


# ---------------------------------------------------------------------------
# (g) end-to-end: real generated rows -> collator -> v10_training_step.
# ---------------------------------------------------------------------------

def test_end_to_end_tiny_model_runs_v10_training_step():
    from train_kaggle import v10_training_step

    rows, _ = _rows_and_batch()
    batch = _collator()(rows[:8])  # two of each family
    torch.manual_seed(79)
    config = _tiny_v10_config(
        mlp_expert_count=2, mlp_expert_rank=4, lora_rank=4, lora_tail_layers=2
    )
    model = HLWMForConditionalGeneration(config).train()

    warm = v10_training_step(
        model, batch, step=0, gamma=10.0, accumulation_scale=1.0, warm_phase=True
    )
    assert all(math.isfinite(value) for value in warm.values()), warm
    assert "recon" in warm and "producer_mse" in warm
    model.zero_grad(set_to_none=True)

    for step in (0, 4):  # block 0: reconstruction; block 1: CoLaR (v10.5 block parity)
        metrics = v10_training_step(
            model, batch, step=step, gamma=10.0, accumulation_scale=1.0
        )
        assert all(math.isfinite(value) for value in metrics.values()), metrics
        assert "student_ce" in metrics and "distill_l1" in metrics
        assert "latent_step_ce" in metrics
        if step == 4:
            assert "colar_segment_ce" in metrics and "colar_answer_ce" in metrics
        else:
            assert "recon" in metrics
        assert "router_probe_ce" in metrics  # family_index + experts enabled
        model.zero_grad(set_to_none=True)
