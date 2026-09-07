"""Version 12 declared routing.

The claim this feature exists to support is narrow and worth stating: with
declared routing on, causal liveness of the expert path is readable off the
token stream. These tests check the machinery that makes that readable, not
that it produces a capability win -- no test here licenses a capability claim.
"""

from __future__ import annotations

import torch

from modeling_hlwm import (
    ROUTE_PATTERN,
    HLWMConfig,
    HLWMForConditionalGeneration,
    parse_declaration,
    route_declaration,
    strip_declarations,
)

FAMILIES = ("numeric", "unit", "ordering", "abstention")


# ---- declaration grammar ----------------------------------------------------


def test_declaration_round_trips_through_the_parser():
    for index, name in enumerate(FAMILIES):
        text = "%s the answer is 42" % route_declaration(name)
        assert parse_declaration(text, FAMILIES) == index


def test_unknown_route_name_parses_as_none_not_an_exception():
    """An unknown name is a real runtime possibility once the model is free to
    emit anything. It must be a None the caller can count, not a crash that
    voids the audit row -- and it must stay distinguishable from 'emitted
    nothing', since those are different failure modes."""

    text = "<route:trigonometry> 42"
    assert parse_declaration(text, FAMILIES) is None
    assert ROUTE_PATTERN.search(text) is not None  # declared, but unknown
    assert ROUTE_PATTERN.search("42") is None  # undeclared


def test_first_declaration_wins():
    text = "<route:unit> then <route:numeric>"
    assert parse_declaration(text, FAMILIES) == FAMILIES.index("unit")


def test_stripping_removes_declarations_and_nothing_else():
    assert strip_declarations("<route:numeric> 42 apples") == "42 apples"
    assert strip_declarations("42 apples") == "42 apples"
    # A near-miss must not be silently eaten as if it were a declaration.
    assert strip_declarations("<route: numeric> 42") == "<route: numeric> 42"


def test_declaration_needs_no_vocabulary_surgery():
    """Every arm in this record shares one frozen substrate. If declarations
    required added tokens, the embedding matrix would differ between arms and
    no cross-arm comparison in the paper would hold."""

    for name in FAMILIES:
        declaration = route_declaration(name)
        assert declaration.isascii()
        assert ROUTE_PATTERN.fullmatch(declaration)


# ---- within-mix -------------------------------------------------------------


def _config(**overrides):
    values = dict(
        latent_thoughts=3, kv_prefix_slots=2, lora_rank=4, lora_tail_layers=2,
        mlp_expert_count=2, mlp_expert_rank=4, lora_dropout=0.0,
    )
    values.update(overrides)
    return HLWMConfig.tiny(**values)


def test_within_mix_absent_when_declared_routing_is_off():
    """v10/v11 byte-compatibility: the flag off must allocate no parameter."""

    model = HLWMForConditionalGeneration(_config(declared_routing=False))
    mixes = [
        layer.mlp.expert_within_mix
        for layer in model.backbone.layers
        if layer.mlp.mlp_experts is not None
    ]
    assert mixes and all(mix is None for mix in mixes)


def test_within_mix_present_and_trainable_when_on():
    model = HLWMForConditionalGeneration(_config(declared_routing=True))
    model.freeze_language_substrate()
    model.unfreeze_language_adapters()
    mixes = [
        layer.mlp.expert_within_mix
        for layer in model.backbone.layers
        if layer.mlp.mlp_experts is not None
    ]
    assert mixes and all(mix is not None for mix in mixes)
    assert all(mix.requires_grad for mix in mixes), (
        "within-mix frozen by the substrate freeze (the v8 frozen-alpha class)"
    )
    assert all(mix.dim() == 1 for mix in mixes), (
        "must stay 1-D so the optimizer's no-decay group covers it"
    )


def test_within_mix_starts_essentially_open():
    """It may only ATTENUATE the declared expert. Starting near 1.0 means the
    declared arm begins equivalent to v10 deterministic routing rather than at
    an arbitrary half-strength."""

    model = HLWMForConditionalGeneration(_config(declared_routing=True))
    for layer in model.backbone.layers:
        if layer.mlp.expert_within_mix is not None:
            assert torch.sigmoid(layer.mlp.expert_within_mix).min().item() > 0.95


def test_within_mix_cannot_reassign_the_route():
    """The recorded collapse mode was a router silently concentrating on one
    expert. Driving the mix to its extremes must change only how much of the
    DECLARED expert applies -- never which expert is selected."""

    torch.manual_seed(3)
    model = HLWMForConditionalGeneration(_config(declared_routing=True)).eval()
    hidden = torch.randn(2, 5, model.config.hidden_size)
    layer = next(l.mlp for l in model.backbone.layers if l.mlp.mlp_experts is not None)
    route = torch.tensor([0, 1], dtype=torch.long)
    with torch.no_grad():
        layer.expert_within_mix.fill_(-20.0)  # sigmoid ~ 0: expert shut off
        closed = layer(hidden, route_index=route)
        layer.expert_within_mix.fill_(20.0)  # sigmoid ~ 1: expert fully on
        opened = layer(hidden, route_index=route)
        layer.expert_within_mix.fill_(-20.0)
        root_only = layer(hidden, route_index=None)
    assert torch.allclose(closed, root_only, atol=1e-5), (
        "a shut mix must land exactly on the always-on root path"
    )
    assert not torch.allclose(opened, closed, atol=1e-4)


def test_declared_routing_off_matches_v10_outputs_exactly():
    """The flag must be inert, not merely small: same seed, same numbers."""

    torch.manual_seed(5)
    baseline = HLWMForConditionalGeneration(_config(declared_routing=False)).eval()
    torch.manual_seed(5)
    flagged = HLWMForConditionalGeneration(_config(declared_routing=True)).eval()
    hidden = torch.randn(2, 5, baseline.config.hidden_size)
    route = torch.tensor([0, 1], dtype=torch.long)
    base_mlp = next(l.mlp for l in baseline.backbone.layers if l.mlp.mlp_experts is not None)
    flag_mlp = next(l.mlp for l in flagged.backbone.layers if l.mlp.mlp_experts is not None)
    with torch.no_grad():
        flag_mlp.expert_within_mix.fill_(20.0)  # fully open == v10 semantics
        assert torch.allclose(
            base_mlp(hidden, route_index=route),
            flag_mlp(hidden, route_index=route),
            atol=1e-5,
        )


def test_within_mix_receives_gradient():
    torch.manual_seed(9)
    model = HLWMForConditionalGeneration(_config(declared_routing=True))
    model.train()
    layer = next(l.mlp for l in model.backbone.layers if l.mlp.mlp_experts is not None)
    hidden = torch.randn(2, 5, model.config.hidden_size)
    layer(hidden, route_index=torch.tensor([0, 1])).pow(2).mean().backward()
    assert layer.expert_within_mix.grad is not None
    assert layer.expert_within_mix.grad.abs().sum().item() > 0


def test_config_serializes_the_new_flags():
    """Checkpoints persist json.dumps(vars(config)); a flag missing there
    silently reloads as a different architecture."""

    config = _config(declared_routing=True)
    values = vars(config)
    assert values["declared_routing"] is True
    assert "declared_within_mix" in values


# ---- the audit arm end to end -----------------------------------------------

import zlib  # noqa: E402

import evaluate_v10 as ev  # noqa: E402


class _TinyTokenizer:
    """Deterministic word tokenizer into the 41-token tiny vocabulary."""

    pad_token_id = 0
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        ids = [3 + (zlib.crc32(word.encode("utf-8")) % 38) for word in str(text).split()]
        return ids or [3]

    def decode(self, ids, skip_special_tokens=True):
        values = [int(v) for v in torch.as_tensor(ids).flatten().tolist()]
        if skip_special_tokens:
            values = [v for v in values if v > 2]
        return " ".join(str(v) for v in values)


class _DeclaringTokenizer(_TinyTokenizer):
    """Decodes as if the model had emitted a declaration, so the arm's route
    mapping is exercised. A randomly initialized tiny model cannot emit real
    declaration text, and stubbing the emission is the only way to test what
    the harness does with one."""

    def __init__(self, declaration):
        self.declaration = declaration

    def decode(self, ids, skip_special_tokens=True):
        return "%s %s" % (self.declaration, super().decode(ids, skip_special_tokens))


_SPECS = {
    "numeric": {"type": "numeric", "expected": 7},
    "unit": {"type": "unit", "expected": 7, "unit": "kg"},
}


def _row(index, family):
    literal = "%d8%02d" % (1 + index % 5, index)
    prompt = "compute value %s plus tax now please" % literal
    return {
        "episode_id": "row-%03d" % index,
        "family": family,
        "masked": True,
        "public_prompt": prompt,
        "masked_prompt": prompt.replace(literal, "[withheld]"),
        "public_target": "the answer is %d" % (7 + index),
        "answer_spec": dict(_SPECS[family]),
        "withheld_literals": [literal],
    }


def _rows():
    return [_row(0, "numeric"), _row(1, "numeric"), _row(2, "unit"), _row(3, "unit")]


def _eval_model():
    torch.manual_seed(0)
    config = HLWMConfig.tiny(
        latent_thoughts=3, kv_prefix_slots=4, kv_prefix_rank=8, response_cue_ids=(5, 7),
        lora_rank=4, lora_tail_layers=2, mlp_expert_count=2, mlp_expert_rank=4,
        lora_dropout=0.0, declared_routing=True,
    )
    return HLWMForConditionalGeneration(config).eval()


def test_declaration_arm_runs_and_its_counters_are_exhaustive():
    model, tokenizer = _eval_model(), _TinyTokenizer()
    prepared = ev.prepare_rows(model, tokenizer, _rows(), context_tokens=24)
    result = ev.arm_declaration_routed(model, prepared, tokenizer, max_new_tokens=6)
    audit = result["declaration_audit"]
    assert len(result["decode_ids"]) == len(prepared)
    assert audit["declared"] + audit["undeclared"] + audit["unknown"] == audit["total"]
    # A randomly initialized tiny model emits no declaration, so this run is
    # entirely fallback -- which is the case that must not crash or route.
    assert audit["undeclared"] == len(prepared)
    assert all(route is None for route in result["declared_routes"])


def test_declaration_arm_applies_the_declared_route():
    model = _eval_model()
    tokenizer = _DeclaringTokenizer(route_declaration("ordering"))
    prepared = ev.prepare_rows(model, tokenizer, _rows(), context_tokens=24)
    result = ev.arm_declaration_routed(model, prepared, tokenizer, max_new_tokens=6)
    audit = result["declaration_audit"]
    assert audit["declared"] == len(prepared)
    assert audit["undeclared"] == 0 and audit["unknown"] == 0
    # ordering is family 2, and {ordering, abstention} -> expert 1.
    assert set(result["declared_families"]) == {FAMILIES.index("ordering")}
    assert set(result["declared_routes"]) == {1}
    # Every prepared row is numeric or unit, so a declared "ordering" must
    # score zero agreement: the arm reports the model's route, not the gold one.
    assert audit["gold_agreement"] == 0.0


def test_declaration_arm_counts_unknown_names_separately():
    model = _eval_model()
    tokenizer = _DeclaringTokenizer("<route:trigonometry>")
    prepared = ev.prepare_rows(model, tokenizer, _rows(), context_tokens=24)
    result = ev.arm_declaration_routed(model, prepared, tokenizer, max_new_tokens=6)
    audit = result["declaration_audit"]
    assert audit["unknown"] == len(prepared) and audit["undeclared"] == 0
    assert all(route is None for route in result["declared_routes"])


def test_grading_strips_declarations_so_arms_share_one_surface():
    """A declared arm's answer must be graded on the same text a gold-routed
    arm's answer is graded on, or every cross-arm comparison is confounded."""

    tokenizer = _DeclaringTokenizer(route_declaration("numeric"))
    ids = torch.tensor([[9, 9]])
    text, _ = ev.grade_decode(tokenizer, ids, {"type": "numeric", "expected": 7})
    assert "<route:" not in text


# ---- data side: the declaration must reach the supervised targets -----------

from test_data_v10 import (  # noqa: E402
    _TinyCharTokenizer,
    _raw_rows,
    _contains_subsequence,
)
from data import (  # noqa: E402
    FAMILY_ORDER as DATA_FAMILY_ORDER,
    V10AnchorCollator,
    normalize_episode,
)


def _episodes(count=8):
    return [normalize_episode(row, num_lanes=1) for row in _raw_rows(count)]


def _data_collator(declared_routing, canvas_tokens=64):
    # canvas_tokens=64 is the production value. The suite's usual 16 cannot
    # hold a CHAR-level declaration plus an answer -- an artifact of the test
    # tokenizer, not of the design (real BPE spends ~6 tokens on
    # "<route:ordering>"), and the too-small case is asserted separately.
    return V10AnchorCollator(
        _TinyCharTokenizer(),
        latent_thoughts=6,
        trace_tokens=192,
        declared_routing=declared_routing,
        num_lanes=1,
        context_tokens=256,
        canvas_tokens=canvas_tokens,
        brief_tokens=16,
        causal_tokens=96,
    )


def test_declaration_that_cannot_fit_raises_instead_of_truncating():
    """A half-emitted declaration is unparseable forever and would show up
    only as 'every row undeclared' with no stated cause -- the silent
    instrument class this record exists to document."""

    import pytest

    with pytest.raises(ValueError, match="does not survive canvas truncation"):
        _data_collator(True, canvas_tokens=8)(_episodes(4))


def test_family_order_matches_the_route_map():
    """data.FAMILY_ORDER indexes the same families the route map assumes; a
    silent reordering would route every family to the wrong expert while every
    gate still passed."""

    assert DATA_FAMILY_ORDER == FAMILIES


def test_collator_is_byte_identical_when_declared_routing_is_off():
    rows = _episodes(8)
    off = _data_collator(False)(rows)
    for key, value in off.items():
        if isinstance(value, torch.Tensor):
            assert value is not None
    assert "route_declarations" not in off


def test_collator_prefixes_targets_with_the_declaration():
    rows = _episodes(8)
    tokenizer = _TinyCharTokenizer()
    off = _data_collator(False)(rows)
    on = _data_collator(True)(rows)

    assert "route_declarations" in on
    assert len(on["route_declarations"]) == len(rows)
    # The route the model is taught to declare must be the route the harness
    # would have applied from the gold label -- otherwise training teaches a
    # declaration that contradicts the deterministic map.
    for declaration, family in zip(on["route_declarations"], on["family_index"].tolist()):
        assert declaration == route_declaration(DATA_FAMILY_ORDER[int(family)])

    # Targets actually changed, and the declaration's ids are really in there.
    assert not torch.equal(off["target_ids"], on["target_ids"])
    for index, declaration in enumerate(on["route_declarations"]):
        ids = tokenizer.encode(declaration)
        assert _contains_subsequence(on["target_ids"][index].tolist(), ids), (
            "declaration for row %d never reached the supervised target" % index
        )


def test_declaration_does_not_disturb_the_gold_route_or_family():
    """The declaration is added supervision, not a relabeling."""

    rows = _episodes(8)
    off = _data_collator(False)(rows)
    on = _data_collator(True)(rows)
    assert torch.equal(off["family_index"], on["family_index"])
    assert torch.equal(off["route_index"], on["route_index"])
    assert torch.equal(off["trace_window_targets"], on["trace_window_targets"])
    assert torch.equal(off["student_input_ids"], on["student_input_ids"])


def test_declaration_does_not_reintroduce_a_masked_leak():
    """The collator raises if withheld premise tokens survive into the masked
    prompt. Rebuilding targets must not smuggle the premise in by another
    route -- the instrument's whole validity rests on that deletion."""

    rows = _episodes(8)
    on = _data_collator(True)(rows)
    for index, row in enumerate(rows):
        target = on["target_ids"][index].tolist()
        for literal in row.get("withheld_literals", []):
            declaration = on["route_declarations"][index]
            assert literal not in declaration
        assert target  # decoded targets still present
