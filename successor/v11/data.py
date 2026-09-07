"""Reasoning9000 reader and tokenizer collator for HLWM Version 5.

The loader consumes master episodes so one training example retains the public
request, distinct lane briefs, private lane artifacts, verification records and
the exact publication target.  It never places one lane's private work inside a
sibling lane's brief. Dataset provenance remains in the package manifest rather
than being injected into the model prompt.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from modeling_hlwm import route_declaration


PROTOTYPE_WARNING = (
    "UNREVIEWED SYNTHETIC PROTOTYPE EPISODE. Use for architecture testing, "
    "not as evidence that claims are correct."
)

LANE_ROLE_GUIDANCE = (
    "Construct a direct solution. Track assumptions, concrete steps, and the final deliverable.",
    "Act as an independent critic. Look for counterexamples, missing evidence, unsafe claims, and fixes.",
    "Seek a materially different solution path and compare its tradeoffs against the first path.",
)


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _list_text(values: Any) -> str:
    if not isinstance(values, list):
        return str(values or "")
    return "\n".join("- " + str(value) for value in values)


def build_context(record: Mapping[str, Any]) -> str:
    source = record.get("input") or {}
    frame = record.get("frame") or {}
    return "\n".join(
        (
            "<|domain|>\n%s / %s"
            % (record.get("domain", "unknown"), record.get("subdomain", "unknown")),
            "<|user_request|>\n" + str(source.get("user_request", "")),
            "<|context|>\n" + _list_text(source.get("context", [])),
            "<|constraints|>\n" + _list_text(source.get("constraints", [])),
            "<|objective|>\n" + str(frame.get("objective", "")),
            "<|requirements|>\n" + _list_text(frame.get("requirements", [])),
            "<|failure_contract|>\n" + _list_text(frame.get("failure_contract", [])),
            "<|response_policy|>\nAnswer the user request directly. Do not repeat the prompt, "
            "invent evidence, or expose private workspace text.",
        )
    )


MASK_MARKER_TEXT = "[withheld]"
"""Replacement text for withheld premise spans on masked rows (Version 9.0).

Masked rows are the information-asymmetry stratum: the workspace builds from
the full context while the answer channel conditions on a context whose
answer-determining operands are replaced by this marker, so the latent prefix
is the only path from the operands to the answer."""


def encode_preserving_ends(
    tokenizer: Any,
    text: str,
    max_length: int,
    head_fraction: float = 0.70,
) -> List[int]:
    """The ONE prompt-encode primitive shared by training, harvest, calibration
    and audit (Version 9.0).

    Version 8.0's harvest generated from the training collator's fixed-length
    right-padded batch (roughly 150 pad tokens between prompt and response cue)
    while the audit encoded prompts unpadded; per-candidate validity on
    generative anchor families was 1-3% on the padded surface against 0.82
    causal validity on the audit surface. Every generation-time caller now
    routes through this unpadded encode.
    """

    ids = tokenizer.encode(str(text), add_special_tokens=True)
    if max_length <= 0:
        raise ValueError("token length must be positive")
    if len(ids) <= max_length:
        return list(ids)
    if max_length == 1:
        return list(ids[:1])
    head = max(1, min(max_length - 1, int(round(max_length * head_fraction))))
    return list(ids[:head]) + list(ids[-(max_length - head):])


RESPONSE_CUE_TEXT = "\n### Response\n"
"""The hard response cue (Version 8.0 layout).

The prompt no longer carries the trailing response header; the model appends
these tokens itself after the (optional) workspace prefix, so both channels
always continue generation from the same real text tokens.  Callers that
prompt an external baseline model directly should append this text to the
prompt string to keep the comparison format-matched."""


def build_public_prompt(record: Mapping[str, Any]) -> str:
    """Create the single public answer prompt used by training and evaluation.

    Version 5 evaluated a long internal serialization with ordinary right-side
    truncation.  That frequently removed the user request or the answer marker.
    This compact prompt keeps the public request first; the response cue is no
    longer part of the prompt (see ``RESPONSE_CUE_TEXT``), which also removes
    any chance of the token-budget truncation clipping it.
    """

    source = record.get("input") or {}
    frame = record.get("frame") or {}
    context = _list_text(source.get("context", []))
    constraints = _list_text(source.get("constraints", []))
    requirements = _list_text(frame.get("requirements", []))
    return "\n".join(
        (
            "### Instruction",
            str(source.get("user_request", "")).strip(),
            "### Relevant context",
            context or "- None supplied",
            "### Constraints",
            constraints or "- None supplied",
            "### Response requirements",
            requirements or "- Give a direct, supported answer.",
            "Do not repeat these instructions. Return only the answer.",
        )
    )


def build_masked_prompt(record: Mapping[str, Any]) -> str:
    """Public prompt with the answer-determining premise withheld (Version 9.0).

    Uses ``input.user_request_masked`` when the generator supplied one;
    otherwise falls back to the full prompt (row is then not maskable).
    """

    masked_request = str((record.get("input") or {}).get("user_request_masked", "")).strip()
    if not masked_request:
        return build_public_prompt(record)
    shadow = dict(record)
    shadow_input = dict(record.get("input") or {})
    shadow_input["user_request"] = masked_request
    shadow["input"] = shadow_input
    return build_public_prompt(shadow)


def build_lane_brief(lane: Mapping[str, Any], lane_index: int) -> str:
    brief = lane.get("brief") or {}
    route = lane.get("route") or ["root"]
    role = LANE_ROLE_GUIDANCE[lane_index % len(LANE_ROLE_GUIDANCE)]
    return "\n".join(
        (
            "<|lane|> %s" % lane.get("lane_id", "lane-%d" % lane_index),
            "<|lane_role|> " + role,
            "<|route|> " + " -> ".join(str(item) for item in route),
            "<|scope|> " + str(brief.get("scope", "")),
            "<|assumptions|>\n" + _list_text(brief.get("assumptions", [])),
            "<|deliverable|>\n" + str(brief.get("deliverable", "")),
            "<|rejection_tests|>\n" + _list_text(brief.get("rejection_tests", [])),
        )
    )


def build_lane_target(lane: Mapping[str, Any]) -> str:
    """Render private work as readable text rather than punctuation-heavy JSON.

    Version 4 serialized artifacts, claims and checkpoints as compact JSON and
    then truncated them to a short diffusion canvas.  That made commas, braces
    and quotes dominate many clean-token targets.  Private canvases are latent
    workspace material, but their clean targets should still have the language
    geometry inherited from Qwen.  This representation preserves provenance
    markers while keeping the actual content natural and independently
    decodable for diagnostics.
    """

    artifacts = lane.get("artifacts") or []
    claims = lane.get("claims") or []
    checkpoints = lane.get("checkpoints") or []
    artifact_lines = []
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            continue
        label = "%s (%s)" % (
            artifact.get("artifact_id", "artifact"),
            artifact.get("type", "text"),
        )
        artifact_lines.append("### %s\n%s" % (label, artifact.get("content", "")))
    claim_lines = []
    for claim in claims:
        if not isinstance(claim, Mapping):
            continue
        references = ", ".join(str(item) for item in claim.get("evidence_refs", []))
        suffix = " Evidence: %s." % references if references else ""
        claim_lines.append("- %s: %s.%s" % (
            claim.get("claim_id", "claim"),
            str(claim.get("statement", "")).rstrip("."),
            suffix,
        ))
    checkpoint_lines = []
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, Mapping):
            continue
        checkpoint_lines.append(
            "- Step %s: %s Decision: %s."
            % (
                checkpoint.get("step", "?"),
                str(checkpoint.get("observable_update", "")).rstrip("."),
                checkpoint.get("decision", "unknown"),
            )
        )
    return "\n".join(
        (
            "<|lane_work|>\n" + ("\n\n".join(artifact_lines) or "No artifact recorded."),
            "<|claims|>\n" + ("\n".join(claim_lines) or "No claims recorded."),
            "<|checkpoints|>\n" + (
                "\n".join(checkpoint_lines) or "No eligible checkpoint recorded."
            ),
            "<|summary|>\n" + str(lane.get("summary", "")),
        )
    )


def build_public_target(record: Mapping[str, Any]) -> str:
    integration = record.get("integration") or {}
    answer = str(integration.get("published_answer", "")).strip()
    if answer:
        return answer
    decision = str((record.get("commitment") or {}).get("decision", "abstain"))
    return "I cannot publish a supported answer. Commitment decision: %s." % decision


def _claim_error_by_lane(record: Mapping[str, Any]) -> Dict[str, float]:
    verdicts = {
        str(item.get("claim_id")): str(item.get("verdict", "insufficient")).lower()
        for item in record.get("verification", [])
        if isinstance(item, Mapping)
    }
    result: Dict[str, float] = {}
    for lane in record.get("lanes", []):
        if not isinstance(lane, Mapping):
            continue
        claim_ids = [
            str(claim.get("claim_id"))
            for claim in lane.get("claims", [])
            if isinstance(claim, Mapping)
        ]
        result[str(lane.get("lane_id", ""))] = float(
            any(verdicts.get(claim_id, "insufficient") != "supported" for claim_id in claim_ids)
        )
    return result


def normalize_episode(record: Mapping[str, Any], num_lanes: int) -> Dict[str, Any]:
    if num_lanes <= 0:
        raise ValueError("num_lanes must be positive")
    lanes = [lane for lane in record.get("lanes", []) if isinstance(lane, Mapping)]
    public_target = build_public_target(record)
    selected: List[Mapping[str, Any]] = lanes[:num_lanes]
    while len(selected) < num_lanes:
        selected.append(
            {
                "lane_id": "fallback-%d" % len(selected),
                "route": ["root"],
                "brief": {
                    "scope": "Produce a conservative root-only answer.",
                    "assumptions": [],
                    "deliverable": "A concise supported answer.",
                    "rejection_tests": ["Do not invent evidence."],
                },
                "artifacts": [{"artifact_id": "fallback", "type": "text", "content": public_target}],
                "claims": [],
                "checkpoints": [],
                "summary": public_target,
            }
        )

    claim_errors = _claim_error_by_lane(record)
    lane_briefs = [build_lane_brief(lane, index) for index, lane in enumerate(selected)]
    lane_targets = [build_lane_target(lane) for lane in selected]
    verification_targets = [
        claim_errors.get(str(lane.get("lane_id", "")), 0.0) for lane in selected
    ]
    halt_targets = []
    route_depth_targets = []
    for lane in selected:
        windows = lane.get("route_windows") or []
        last_decision = str(windows[-1].get("decision", "halt")) if windows else "halt"
        halt_targets.append(float(last_decision == "halt"))
        route_depth_targets.append(max(0, len(lane.get("route") or ["root"]) - 1))

    commitment = record.get("commitment") or {}
    decision = str(commitment.get("decision", "abstain")).lower()
    evaluation = record.get("evaluation") or {}
    generation_metadata = record.get("generation_metadata") or {}
    verification_records = [
        item for item in record.get("verification", []) if isinstance(item, Mapping)
    ]
    risk_target = float(
        any(str(item.get("verdict", "insufficient")).lower() != "supported" for item in verification_records)
        or bool((record.get("barrier") or {}).get("open_claims"))
    )
    # The direct-fast release was not independently judged, so its publish,
    # halt and risk labels are weak metadata rather than production policy
    # supervision. Version 5.4 also masks paired commitment scoring for these
    # rows; only independently adjudicated or narrow programmatically verified
    # records may calibrate publication behavior. A supported statement of
    # missing evidence is a public answer, not a hidden/non-commit decision.
    policy_supervision_eligible = bool(
        generation_metadata.get("independently_adjudicated", False)
        or generation_metadata.get("programmatically_verified", False)
    )
    evaluation_block = record.get("evaluation") or {}
    withheld_literals = [
        str(item) for item in (evaluation_block.get("withheld_literals") or []) if str(item)
    ]
    masked_prompt = build_masked_prompt(record)
    maskable = bool(withheld_literals) and masked_prompt != build_public_prompt(record)
    if maskable:
        for literal in withheld_literals:
            if literal in masked_prompt:
                raise ValueError(
                    "withheld literal %r leaks into the masked prompt of %s"
                    % (literal, record.get("episode_id"))
                )
    return {
        "episode_id": str(record.get("episode_id", "")),
        "domain": str(record.get("domain", "unknown")),
        "context": build_context(record),
        "public_prompt": build_public_prompt(record),
        "masked_prompt": masked_prompt if maskable else build_public_prompt(record),
        "withheld_literals": withheld_literals if maskable else [],
        "maskable": maskable,
        "masked": maskable and bool(evaluation_block.get("masked", False)),
        "difficulty_params": dict(record.get("difficulty_params") or {}),
        "lane_briefs": lane_briefs,
        "lane_targets": lane_targets,
        "public_target": public_target,
        "negative_public_target": str(evaluation.get("negative_answer", "")).strip(),
        "verification_targets": verification_targets,
        "halt_targets": halt_targets,
        "route_depth_targets": route_depth_targets,
        "commitment_target": float(decision == "publish"),
        "risk_target": risk_target,
        "policy_supervision_eligible": policy_supervision_eligible,
        "expected_commit": bool(
            evaluation.get(
                "expected_commit",
                evaluation.get("expected_publish", decision == "publish"),
            )
        ),
        "expected_action": str(
            evaluation.get("expected_action", "answer" if decision == "publish" else "withhold")
        ),
        "answer_spec": dict(evaluation.get("answer_spec") or {}),
        "quality_checks": {
            "required_phrases": [
                str(value).lower() for value in evaluation.get("required_phrases", [])
            ],
            "forbidden_phrases": [
                str(value).lower() for value in evaluation.get("forbidden_phrases", [])
            ],
        },
        # Version 10.0 dense-supervision payload (gold trace, teacher-step
        # exclusion, process negative, family label), attached by the v10
        # builder and consumed by V10AnchorCollator.  Passed through verbatim;
        # None on rows that predate v10.
        "v10": dict(record["v10"]) if isinstance(record.get("v10"), Mapping) else None,
        # Anchor status is intentionally narrower than policy eligibility:
        # Version 5.6 adds execution-verified benchmark episodes that are
        # policy-eligible but must not join the anchor-only probe gates,
        # the anchor oversampler, or the on-policy head-phase emission pool.
        "is_behavior_anchor": bool(
            generation_metadata.get("programmatically_verified", False)
            and str(record.get("domain", "")) == "behavior-anchor"
        ),
    }


class Reasoning9000Dataset(Dataset):
    """In-memory normalized view of a prototype-sized master JSONL split."""

    def __init__(self, path: str | Path, num_lanes: int = 2, limit: Optional[int] = None) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.rows: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                self.rows.append(normalize_episode(json.loads(line), num_lanes))
                if limit is not None and len(self.rows) >= limit:
                    break
        if not self.rows:
            raise ValueError("dataset split is empty: %s" % self.path)
        self.anchor_indices = [
            index
            for index, row in enumerate(self.rows)
            if bool(row.get("is_behavior_anchor", False))
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.rows[index]


class HLWMCollator:
    """Tokenize public context, isolated lane records and causal anchor text."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        num_lanes: int = 2,
        context_tokens: int = 96,
        canvas_tokens: int = 96,
        brief_tokens: int = 64,
        causal_tokens: int = 192,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_lanes = num_lanes
        self.context_tokens = context_tokens
        self.canvas_tokens = canvas_tokens
        self.brief_tokens = brief_tokens
        self.causal_tokens = causal_tokens
        if getattr(tokenizer, "pad_token_id", None) is None:
            if getattr(tokenizer, "eos_token", None) is None:
                raise ValueError("tokenizer needs a pad token or EOS token")
            tokenizer.pad_token = tokenizer.eos_token

    @staticmethod
    def _preserve_ends(ids: Sequence[int], length: int, head_fraction: float = 0.70) -> List[int]:
        """Fit token ids while retaining the instruction head and response cue tail."""

        values = list(ids)
        if length <= 0:
            raise ValueError("token length must be positive")
        if len(values) <= length:
            return values
        if length == 1:
            return values[:1]
        head = max(1, min(length - 1, int(round(length * head_fraction))))
        return values[:head] + values[-(length - head) :]

    def _fixed(self, texts: Sequence[str], length: int) -> Dict[str, Tensor]:
        pad_id = int(self.tokenizer.pad_token_id)
        rows: List[List[int]] = []
        masks: List[List[int]] = []
        for text in texts:
            ids = self.tokenizer.encode(str(text), add_special_tokens=True)
            ids = self._preserve_ends(ids, length)
            padding = length - len(ids)
            rows.append(ids + [pad_id] * padding)
            masks.append([1] * len(ids) + [0] * padding)
        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }

    def _answer_targets(
        self,
        texts: Sequence[str],
        length: int,
        negative_texts: Optional[Sequence[str]] = None,
    ) -> Dict[str, Tensor]:
        """Create clean answer blocks with an explicit EOS and deterministic negatives."""

        pad_id = int(self.tokenizer.pad_token_id)
        eos_id = int(self.tokenizer.eos_token_id)
        vocabulary_size = getattr(self.tokenizer, "vocab_size", None)
        if vocabulary_size is None:
            try:
                vocabulary_size = len(self.tokenizer)
            except (TypeError, AttributeError) as error:
                raise ValueError("tokenizer must expose vocab_size or __len__") from error
        vocabulary_size = max(4, int(vocabulary_size))
        rows: List[List[int]] = []
        masks: List[List[int]] = []
        negatives: List[List[int]] = []
        negative_masks: List[List[int]] = []
        if negative_texts is not None and len(negative_texts) != len(texts):
            raise ValueError("negative texts must align with answer texts")
        for index, text in enumerate(texts):
            content = self.tokenizer.encode(str(text), add_special_tokens=False)
            content = content[: max(0, length - 1)] + [eos_id]
            mask = [1] * len(content)
            padding = length - len(content)
            padded = content + [pad_id] * padding

            # A deterministic hard-negative snapshot preserves length and most
            # lexical content while breaking order.  It supplies the missing
            # reject/risk contrast without treating weak Reasoning9000 policy
            # labels as calibrated truth.
            explicit_negative = (
                str(negative_texts[index]).strip() if negative_texts is not None else ""
            )
            body = content[:-1]
            if explicit_negative:
                negative_body = self.tokenizer.encode(
                    explicit_negative, add_special_tokens=False
                )[: max(1, length - 1)]
            elif len(body) > 1:
                shift = max(1, len(body) // 3)
                negative_body = body[shift:] + body[:shift]
            elif body:
                replacement = (body[0] + 1) % vocabulary_size
                if replacement in (pad_id, eos_id):
                    replacement = (replacement + 2) % vocabulary_size
                negative_body = [replacement]
            else:
                negative_body = [max(3, eos_id + 1) % vocabulary_size]
            negative = (negative_body + [eos_id])[:length]
            negative_length = len(negative)
            negative += [pad_id] * (length - negative_length)
            rows.append(padded)
            masks.append(mask + [0] * padding)
            negatives.append(negative)
            negative_masks.append(
                [1] * negative_length + [0] * (length - negative_length)
            )
        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "negative_input_ids": torch.tensor(negatives, dtype=torch.long),
            "negative_attention_mask": torch.tensor(negative_masks, dtype=torch.long),
        }

    def _premise_supervision(
        self,
        rows: Sequence[Mapping[str, Any]],
        masked_flags: Sequence[bool],
        answer_input_ids: Tensor,
        max_premise_tokens: int = 24,
        negative_pool: int = 32,
    ) -> Dict[str, Tensor]:
        """Premise token ids for the latent-only auxiliary probe (Version 9.0),
        plus the runtime leak assertion for masked rows.

        The leak check is contiguous-subsequence: the exact token sequence of a
        withheld literal must not appear inside the masked answer prompt.
        String-level absence is already asserted at normalization time; this
        catches collation-time regressions.
        """

        batch = len(rows)
        premise_rows: List[List[int]] = []
        negative_rows: List[List[int]] = []
        for index, (row, masked) in enumerate(zip(rows, masked_flags)):
            literals = [str(item) for item in row.get("withheld_literals", [])]
            ids: List[int] = []
            if masked and literals:
                for literal in literals:
                    ids.extend(
                        self.tokenizer.encode(" " + literal, add_special_tokens=False)
                    )
                ids = ids[:max_premise_tokens]
                answer_row = answer_input_ids[index].tolist()
                for literal in literals:
                    span = self.tokenizer.encode(" " + literal, add_special_tokens=False)
                    if len(span) > 0 and len(span) <= len(answer_row):
                        for start in range(len(answer_row) - len(span) + 1):
                            if answer_row[start : start + len(span)] == span:
                                raise ValueError(
                                    "withheld premise tokens leak into the masked "
                                    "answer prompt of %s" % row.get("episode_id")
                                )
            premise_rows.append(ids)
            # Deterministic in-row negatives: prompt tokens that are not
            # premise tokens, cycled to a fixed pool size.
            pool = [
                token
                for token in answer_input_ids[index].tolist()
                if token != int(self.tokenizer.pad_token_id) and token not in ids
            ]
            if not pool:
                pool = [int(self.tokenizer.eos_token_id)]
            negative_rows.append([pool[i % len(pool)] for i in range(negative_pool)])
        premise_ids = torch.full((batch, max_premise_tokens), -100, dtype=torch.long)
        premise_mask = torch.zeros(batch, max_premise_tokens, dtype=torch.long)
        for index, ids in enumerate(premise_rows):
            if ids:
                premise_ids[index, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                premise_mask[index, : len(ids)] = 1
        return {
            "premise_ids": premise_ids,
            "premise_attention_mask": premise_mask,
            "premise_negative_ids": torch.tensor(negative_rows, dtype=torch.long),
        }

    def _causal(self, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Tensor]:
        pad_id = int(self.tokenizer.pad_token_id)
        eos_id = int(self.tokenizer.eos_token_id)
        input_rows: List[List[int]] = []
        label_rows: List[List[int]] = []
        for row in rows:
            prompt = str(row["public_prompt"])
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
            answer_ids = self.tokenizer.encode(str(row["public_target"]), add_special_tokens=False)
            max_answer = max(8, int(self.causal_tokens * 0.65))
            answer_ids = answer_ids[: max_answer - 1] + [eos_id]
            room = max(1, self.causal_tokens - len(answer_ids))
            prompt_ids = self._preserve_ends(prompt_ids, room, head_fraction=0.78)
            combined = (prompt_ids + answer_ids)[: self.causal_tokens]
            labels = [-100] * min(len(prompt_ids), len(combined))
            labels += combined[len(labels) :]
            padding = self.causal_tokens - len(combined)
            input_rows.append(combined + [pad_id] * padding)
            label_rows.append(labels + [-100] * padding)
        ids = torch.tensor(input_rows, dtype=torch.long)
        labels = torch.tensor(label_rows, dtype=torch.long)
        return {
            "causal_input_ids": ids,
            "causal_attention_mask": (ids != pad_id).long(),
            "causal_labels": labels,
        }

    def __call__(self, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        # The private workspace and the public decoder receive the same
        # bounded task statement.  Keeping this aligned removes the Version 5
        # train/evaluation mismatch where one path saw an internal dump and
        # the other saw a differently truncated pseudo-chat prompt.
        contexts = self._fixed(
            [str(row["public_prompt"]) for row in rows], self.context_tokens
        )
        # Version 9.0 masked answer channel: rows flagged ``masked`` condition
        # the answer channel on the withheld-premise prompt while the
        # workspace still reads the full context. Unmasked rows reuse the full
        # prompt so the tensors are always present and shape-stable.
        masked_flags = [bool(row.get("masked", False)) for row in rows]
        answer_contexts = self._fixed(
            [
                str(row.get("masked_prompt", row["public_prompt"]))
                if masked
                else str(row["public_prompt"])
                for row, masked in zip(rows, masked_flags)
            ],
            self.context_tokens,
        )
        premise = self._premise_supervision(rows, masked_flags, answer_contexts["input_ids"])
        targets = self._answer_targets(
            [str(row["public_target"]) for row in rows],
            self.canvas_tokens,
            [str(row.get("negative_public_target", "")) for row in rows],
        )
        flat_briefs = [
            str(brief)
            for row in rows
            for brief in list(row["lane_briefs"])[: self.num_lanes]
        ]
        flat_lane_targets = [
            str(target)
            for row in rows
            for target in list(row["lane_targets"])[: self.num_lanes]
        ]
        briefs = self._fixed(flat_briefs, self.brief_tokens)
        lane_targets = self._answer_targets(flat_lane_targets, self.canvas_tokens)
        batch = len(rows)
        result: Dict[str, Any] = {
            "input_ids": contexts["input_ids"],
            "attention_mask": contexts["attention_mask"],
            "answer_input_ids": answer_contexts["input_ids"],
            "answer_attention_mask": answer_contexts["attention_mask"],
            "masked_rows": torch.tensor(masked_flags, dtype=torch.bool),
            "premise_ids": premise["premise_ids"],
            "premise_attention_mask": premise["premise_attention_mask"],
            "premise_negative_ids": premise["premise_negative_ids"],
            "masked_prompts": [
                str(row.get("masked_prompt", row["public_prompt"])) for row in rows
            ],
            "withheld_literals": [
                [str(item) for item in row.get("withheld_literals", [])] for row in rows
            ],
            "target_ids": targets["input_ids"],
            "target_attention_mask": targets["attention_mask"],
            "negative_target_ids": targets["negative_input_ids"],
            "negative_target_attention_mask": targets["negative_attention_mask"],
            "lane_brief_ids": briefs["input_ids"].reshape(
                batch, self.num_lanes, self.brief_tokens
            ),
            "lane_brief_attention_mask": briefs["attention_mask"].reshape(
                batch, self.num_lanes, self.brief_tokens
            ),
            "lane_target_ids": lane_targets["input_ids"].reshape(
                batch, self.num_lanes, self.canvas_tokens
            ),
            "lane_target_attention_mask": lane_targets["attention_mask"].reshape(
                batch, self.num_lanes, self.canvas_tokens
            ),
            "verification_targets": torch.tensor(
                [row["verification_targets"] for row in rows], dtype=torch.float32
            ),
            "halt_targets": torch.tensor(
                [row["halt_targets"] for row in rows], dtype=torch.float32
            ),
            "commitment_targets": torch.tensor(
                [row["commitment_target"] for row in rows], dtype=torch.float32
            ),
            "risk_targets": torch.tensor(
                [row["risk_target"] for row in rows], dtype=torch.float32
            ),
            "policy_supervision_mask": torch.tensor(
                [row["policy_supervision_eligible"] for row in rows], dtype=torch.bool
            ),
            "route_depth_targets": torch.tensor(
                [row["route_depth_targets"] for row in rows], dtype=torch.long
            ),
            "episode_ids": [str(row["episode_id"]) for row in rows],
            "domains": [str(row["domain"]) for row in rows],
            "raw_targets": [str(row["public_target"]) for row in rows],
            "public_prompts": [str(row["public_prompt"]) for row in rows],
            "expected_commit": torch.tensor(
                [bool(row["expected_commit"]) for row in rows], dtype=torch.bool
            ),
            "expected_actions": [str(row["expected_action"]) for row in rows],
            "answer_specs": [dict(row["answer_spec"]) for row in rows],
            "is_behavior_anchor": torch.tensor(
                [bool(row["is_behavior_anchor"]) for row in rows], dtype=torch.bool
            ),
            "quality_checks": [dict(row["quality_checks"]) for row in rows],
        }
        result.update(self._causal(rows))
        return result


V10_FAMILY_INDEX = {"numeric": 0, "unit": 1, "ordering": 2, "abstention": 3}
"""Version 10.0 family label order (numeric, unit, ordering, abstention)."""

FAMILY_ORDER: tuple[str, ...] = tuple(
    name for name, _ in sorted(V10_FAMILY_INDEX.items(), key=lambda item: item[1])
)
"""Family names by index, derived from the index map so the two cannot drift."""

V10_ROUTE_BY_FAMILY = (0, 0, 1, 1)
"""Deterministic family->expert map: {numeric, unit} -> expert 0,
{ordering, abstention} -> expert 1 (plan section 3; collapse impossible by
construction because the label, not a learned router, selects the expert)."""


def encode_trace_segments(
    tokenizer: Any, steps: Sequence[str], excluded_steps: int
) -> tuple[List[int], int]:
    """Compositional trace encoding shared by the collator and the builder.

    The trace string is ``" ".join(steps)``; encoding it per segment (first
    step bare, every later step with its leading space) instead of as one
    string makes the step/token boundary EXACT by construction: no BPE merge
    can ever straddle two steps, so the teacher-excluded span is a clean
    suffix of the id sequence under ANY tokenizer.  Returns ``(ids,
    excluded_token_count)`` where the excluded count covers the final
    ``excluded_steps`` segments (the answer-producing step plus any
    builder-added padding repeats of it).
    """

    step_list = [str(step) for step in steps]
    if not step_list:
        raise ValueError("trace has no steps")
    excluded_steps = int(excluded_steps)
    if not 1 <= excluded_steps <= len(step_list):
        raise ValueError(
            "excluded_steps=%d must lie in [1, %d]" % (excluded_steps, len(step_list))
        )
    ids: List[int] = []
    excluded = 0
    boundary = len(step_list) - excluded_steps
    for index, step in enumerate(step_list):
        text = step if index == 0 else " " + step
        segment = tokenizer.encode(text, add_special_tokens=False)
        if not segment:
            raise ValueError("trace step %d encodes to zero tokens: %r" % (index, step))
        ids.extend(int(token) for token in segment)
        if index >= boundary:
            excluded += len(segment)
    return ids, excluded


class V10AnchorCollator(HLWMCollator):
    """Version 10.0 anchor collator: the dense-supervision batch contract.

    Consumes normalized rows carrying a ``v10`` payload (attached by
    ``scripts/build_hlwm_v100_bundle.py``:  ``steps`` is the padded gold step
    list, ``teacher_excluded_steps`` = 1 + pad_repeats counts the trailing
    answer-producing segments, ``corrupt_steps``/``corrupt_step_index`` the
    Math-Shepherd process negative, ``family`` the routing label) and emits
    EXACTLY the keys of the "Batch contract for anchor rows" comment in
    ``train_kaggle.py``, on top of every Version 9.0 key from the base
    collator (``premise_ids`` and friends unchanged).

    Teacher-exclusion encoding: the row records step-level structure, never
    token counts, so the boundary survives any tokenizer swap; the collator
    re-derives the excluded token span with its OWN tokenizer through
    ``encode_trace_segments`` (the builder asserts the recorded
    ``teacher_excluded_tokens`` matches this arithmetic under the real pinned
    tokenizer at build time).  Single-step traces are fully excluded
    (``teacher_supervised_mask`` all zero): their only step produces the
    answer, the CODI copy-shortcut this exclusion exists to kill.

    Traces are NEVER truncated (a truncated trace would silently break the
    window arithmetic and the no-eos-in-window guarantee); an over-budget or
    under-``latent_thoughts`` trace raises instead.

    Abstention rows carry no corruptible value (the trace library raises by
    contract), so their corrupt tensors are the documented sentinel:
    all-pad ids, all-zero attention mask, ``corrupt_step_index`` -1.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        latent_thoughts: int,
        trace_tokens: int = 96,
        declared_routing: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(tokenizer, **kwargs)
        if int(latent_thoughts) <= 0:
            raise ValueError("latent_thoughts must be positive")
        if int(trace_tokens) < int(latent_thoughts):
            raise ValueError("trace_tokens cannot be smaller than latent_thoughts")
        self.latent_thoughts = int(latent_thoughts)
        self.trace_tokens = int(trace_tokens)
        self.declared_routing = bool(declared_routing)

    def _encode_trace_row(
        self, episode_id: str, steps: Sequence[str], excluded_steps: int
    ) -> tuple[List[int], int]:
        ids, excluded = encode_trace_segments(self.tokenizer, steps, excluded_steps)
        if len(ids) > self.trace_tokens:
            raise ValueError(
                "trace of %s needs %d tokens > budget %d; traces are never truncated"
                % (episode_id, len(ids), self.trace_tokens)
            )
        if len(ids) < self.latent_thoughts:
            raise ValueError(
                "trace of %s has %d tokens < latent_thoughts %d; the builder must pad it"
                % (episode_id, len(ids), self.latent_thoughts)
            )
        pad_id = int(self.tokenizer.pad_token_id)
        eos_id = int(self.tokenizer.eos_token_id)
        if pad_id in ids or eos_id in ids:
            raise ValueError(
                "pad/eos id inside the trace of %s (the v8.0 root-cause class)"
                % episode_id
            )
        return ids, excluded

    def __call__(self, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        result = super().__call__(rows)
        payloads: List[Mapping[str, Any]] = []
        for row in rows:
            payload = row.get("v10")
            if not isinstance(payload, Mapping) or not payload.get("steps"):
                raise ValueError(
                    "V10AnchorCollator requires rows carrying a v10 trace payload; "
                    "missing on %s" % row.get("episode_id")
                )
            payloads.append(payload)

        batch = len(rows)
        windows = self.latent_thoughts
        pad_id = int(self.tokenizer.pad_token_id)
        trace_ids = torch.full((batch, self.trace_tokens), pad_id, dtype=torch.long)
        trace_mask = torch.zeros(batch, self.trace_tokens, dtype=torch.long)
        supervised = torch.zeros(batch, self.trace_tokens, dtype=torch.long)
        corrupt_ids = torch.full((batch, self.trace_tokens), pad_id, dtype=torch.long)
        corrupt_mask = torch.zeros(batch, self.trace_tokens, dtype=torch.long)
        corrupt_index = torch.full((batch,), -1, dtype=torch.long)
        family_index = torch.zeros(batch, dtype=torch.long)
        row_trace_ids: List[List[int]] = []
        for index, (row, payload) in enumerate(zip(rows, payloads)):
            episode_id = str(row.get("episode_id", "row-%d" % index))
            family = V10_FAMILY_INDEX.get(str(payload.get("family")))
            if family is None:
                raise ValueError(
                    "unknown v10 family %r on %s" % (payload.get("family"), episode_id)
                )
            family_index[index] = family
            ids, excluded = self._encode_trace_row(
                episode_id,
                [str(step) for step in payload["steps"]],
                int(payload.get("teacher_excluded_steps", 1)),
            )
            trace_ids[index, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            trace_mask[index, : len(ids)] = 1
            supervised[index, : len(ids) - excluded] = 1
            row_trace_ids.append(ids)

            corrupt_steps = [str(step) for step in (payload.get("corrupt_steps") or [])]
            if corrupt_steps:
                bad_ids, _ = self._encode_trace_row(
                    episode_id + " (corrupt)", corrupt_steps, 1
                )
                corrupt_ids[index, : len(bad_ids)] = torch.tensor(bad_ids, dtype=torch.long)
                corrupt_mask[index, : len(bad_ids)] = 1
                corrupt_index[index] = int(payload.get("corrupt_step_index", 0))

        # Per-window token targets, [batch, K, W] with -100 padding.  The
        # window boundaries use the EXACT arithmetic of
        # ``train_kaggle.compressed_gold_thoughts`` over each row's VALID
        # length: bounds[i] = (valid * i) // K, sizes differ by at most one
        # and are never empty because valid >= latent_thoughts is asserted.
        width = max(
            (len(ids) + windows - 1) // windows for ids in row_trace_ids
        )
        window_targets = torch.full((batch, windows, width), -100, dtype=torch.long)
        for index, ids in enumerate(row_trace_ids):
            valid = len(ids)
            bounds = [(valid * position) // windows for position in range(windows + 1)]
            for window in range(windows):
                span = ids[bounds[window] : bounds[window + 1]]
                window_targets[index, window, : len(span)] = torch.tensor(
                    span, dtype=torch.long
                )

        if self.declared_routing:
            # Teach the declaration through the ordinary answer CE: the model
            # emits <route:family> at the head of its answer, so the route it
            # is trained to use is a token in its own output stream rather
            # than a gold label the deployment path does not have. Targets are
            # rebuilt here (not at the base call site) because the family is
            # only resolved above; the extra encode costs one pass and is
            # skipped entirely when the flag is off.
            declarations = [
                route_declaration(FAMILY_ORDER[int(value)]) for value in family_index
            ]
            targets = self._answer_targets(
                [
                    "%s %s" % (declaration, str(row["public_target"]))
                    for declaration, row in zip(declarations, rows)
                ],
                self.canvas_tokens,
                [
                    "%s %s" % (declaration, str(row.get("negative_public_target", "")))
                    for declaration, row in zip(declarations, rows)
                ],
            )
            # A declaration that does not survive canvas truncation teaches an
            # unparseable prefix: the model emits half a declaration forever,
            # the audit counts every row undeclared, and nothing in the run
            # says why. That is precisely the silent-instrument class this
            # record exists to document, so it raises here instead.
            for index, declaration in enumerate(declarations):
                needle = self.tokenizer.encode(declaration, add_special_tokens=False)
                haystack = targets["input_ids"][index].tolist()
                if not any(
                    haystack[start : start + len(needle)] == needle
                    for start in range(len(haystack) - len(needle) + 1)
                ):
                    raise ValueError(
                        "route declaration %r does not survive canvas truncation on %s "
                        "(canvas_tokens=%d): raise canvas_tokens or shorten the route name"
                        % (
                            declaration,
                            rows[index].get("episode_id", "row-%d" % index),
                            self.canvas_tokens,
                        )
                    )

            result["target_ids"] = targets["input_ids"]
            result["target_attention_mask"] = targets["attention_mask"]
            result["negative_target_ids"] = targets["negative_input_ids"]
            result["negative_target_attention_mask"] = targets["negative_attention_mask"]
            result["route_declarations"] = declarations

        result.update(
            {
                # The v10 contract name for the answer-channel surface the
                # base collator already builds: masked prompt on masked rows,
                # the full prompt otherwise (Amendment A1 asymmetry).
                "student_input_ids": result["answer_input_ids"],
                "student_attention_mask": result["answer_attention_mask"],
                "trace_input_ids": trace_ids,
                "trace_attention_mask": trace_mask,
                "teacher_supervised_mask": supervised,
                "trace_window_targets": window_targets,
                "corrupt_trace_input_ids": corrupt_ids,
                "corrupt_trace_attention_mask": corrupt_mask,
                "corrupt_step_index": corrupt_index,
                "family_index": family_index,
                "route_index": torch.tensor(
                    [V10_ROUTE_BY_FAMILY[int(value)] for value in family_index],
                    dtype=torch.long,
                ),
            }
        )
        return result


__all__ = [
    "HLWMCollator",
    "MASK_MARKER_TEXT",
    "PROTOTYPE_WARNING",
    "RESPONSE_CUE_TEXT",
    "Reasoning9000Dataset",
    "V10AnchorCollator",
    "V10_FAMILY_INDEX",
    "V10_ROUTE_BY_FAMILY",
    "build_context",
    "build_masked_prompt",
    "build_public_prompt",
    "build_lane_brief",
    "build_lane_target",
    "build_public_target",
    "encode_preserving_ends",
    "encode_trace_segments",
    "normalize_episode",
]
