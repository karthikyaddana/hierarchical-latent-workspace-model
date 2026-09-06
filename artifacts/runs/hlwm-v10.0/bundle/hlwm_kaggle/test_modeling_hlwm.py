from __future__ import annotations

import math
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

# This file's own directory is the source of record. It used to prefer
# ``experiments.kaggle_hlwm``, which is the ARCHIVED pre-hotfix Version 6.0
# tree; that branch became reachable the moment anything put the repo root
# on sys.path (test_data_v10's builder import does exactly that), at which
# point this module silently tested the wrong copy of the code -- the same
# stale-tree failure class the notebook's content-pinned cell 3 closes.
from pathlib import Path as _Path

_BUNDLE = str(_Path(__file__).resolve().parent)
if sys.path[:1] != [_BUNDLE]:
    sys.path.insert(0, _BUNDLE)

from data import HLWMCollator, build_lane_brief, build_public_prompt, normalize_episode
from evaluate_checkpoint import output_quality
from semantic_grading import grade_semantic_answer
from modeling_hlwm import (
    CategoricalDiffusion,
    HLWMConfig,
    HLWMForConditionalGeneration,
    RoutedAdapterBank,
    _masked_token_cross_entropy,
)
from train_kaggle import (
    DeterministicStepBatchSampler,
    fit_policy_thresholds,
    train_policy_heads_on_policy,
)


def tiny_config(**overrides):
    return HLWMConfig.tiny(**overrides)


def sample_batch(config: HLWMConfig, batch: int = 2):
    torch.manual_seed(11)
    input_ids = torch.randint(3, config.vocab_size, (batch, 5))
    target_ids = torch.randint(3, config.vocab_size, (batch, 4))
    attention_mask = torch.ones_like(input_ids)
    target_mask = torch.ones_like(target_ids)
    timesteps = torch.full((batch,), config.diffusion_steps, dtype=torch.long)
    return input_ids, attention_mask, target_ids, target_mask, timesteps


def has_finite_gradient(module: nn.Module) -> bool:
    return any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )


def test_categorical_diffusion_is_normalized_and_has_identity_at_t0():
    schedule = CategoricalDiffusion(
        vocab_size=13, steps=4, beta_start=0.1, beta_end=0.4
    )
    clean = torch.tensor([[1, 2, 3], [4, 5, 6]])
    at_zero = schedule.q_sample(clean, torch.zeros(2, dtype=torch.long))
    assert torch.equal(at_zero, clean)
    noisy = schedule.q_sample(clean, torch.tensor([2, 4]))
    predicted_clean = torch.randn(2, 3, 13).softmax(dim=-1)
    posterior = schedule.posterior_probabilities(
        noisy, predicted_clean, torch.tensor([2, 4])
    )
    assert posterior.shape == (2, 3, 13)
    assert torch.all(posterior >= 0)
    torch.testing.assert_close(
        posterior.sum(dim=-1), torch.ones(2, 3), atol=1.0e-5, rtol=1.0e-5
    )


def test_training_timestep_sampler_uses_equal_categorical_corruption_mass():
    schedule = CategoricalDiffusion(
        vocab_size=11, steps=4, beta_start=0.1, beta_end=0.4
    )
    expected = schedule.alpha_bar[:-1] - schedule.alpha_bar[1:]
    expected = expected / expected.sum()
    torch.testing.assert_close(schedule.training_timestep_probabilities(), expected)
    generator = torch.Generator().manual_seed(7)
    sampled = schedule.sample_training_timesteps(1000, torch.device("cpu"), generator)
    assert sampled.min() >= 1
    assert sampled.max() <= schedule.steps
    assert sampled.unique().numel() == schedule.steps


def test_connected_router_executes_parent_leaf_and_preserves_activation_dtype():
    config = tiny_config(num_experts=4, expert_top_level=2)
    bank = RoutedAdapterBank(config).eval()

    class PromotedExpert(nn.Module):
        def forward(self, hidden):
            return hidden.double()

    bank.experts[2] = PromotedExpert()
    with torch.no_grad():
        bank.router.weight.zero_()
        bank.router.bias.copy_(torch.tensor([-9.0, -9.0, 9.0, -9.0]))
    hidden = torch.randn(2, 4, config.hidden_size, dtype=torch.float32)
    context = torch.randn(2, config.hidden_size, dtype=torch.float32)
    output, selected, _, path_mask = bank(hidden, context)
    assert output.dtype == hidden.dtype
    assert torch.equal(selected, torch.full_like(selected, 2))
    assert torch.all(path_mask[:, 0])
    assert torch.all(path_mask[:, 2])
    assert torch.all(path_mask.sum(dim=-1) == 2)


def test_deterministic_sampler_resumes_and_limits_overfit_prefix():
    complete = list(DeterministicStepBatchSampler(7, 2, 0, 11, seed=23))
    resumed = list(DeterministicStepBatchSampler(7, 2, 5, 11, seed=23))
    assert resumed == complete[5:]
    progressive = list(
        DeterministicStepBatchSampler(
            20, 2, 0, 8, seed=3, overfit_steps=4, overfit_examples=5
        )
    )
    assert all(max(batch) < 5 for batch in progressive[:4])
    assert all(len(batch) == 2 for batch in complete)


def test_priority_sampler_oversamples_anchors_and_resumes_exactly():
    kwargs = dict(
        dataset_size=20,
        batch_size=1,
        total_steps=100,
        seed=41,
        priority_indices=[0, 1],
        priority_ratio=0.35,
    )
    complete = list(DeterministicStepBatchSampler(start_step=0, **kwargs))
    resumed = list(DeterministicStepBatchSampler(start_step=37, **kwargs))
    assert resumed == complete[37:]
    priority_fraction = sum(batch[0] in (0, 1) for batch in complete) / len(complete)
    assert 0.25 <= priority_fraction <= 0.45


def test_validation_threshold_fit_separates_clean_and_corrupt_scores():
    fitted = fit_policy_thresholds(
        positive_commit=[0.82, 0.86, 0.90, 0.94],
        positive_risk=[0.08, 0.12, 0.16, 0.20],
        negative_commit=[0.10, 0.18, 0.24, 0.30],
        negative_risk=[0.70, 0.76, 0.82, 0.90],
        positive_verifier=[0.05, 0.08, 0.10, 0.12],
        negative_verifier=[0.72, 0.78, 0.84, 0.91],
    )
    assert fitted["clean_accept_rate"] == 1.0
    assert fitted["corrupt_reject_rate"] == 1.0
    assert fitted["balanced_accuracy"] == 1.0


def test_prompt_budget_preserves_instruction_and_response_cue():
    record = {
        "domain": "test",
        "subdomain": "prompting",
        "input": {
            "user_request": "Compute 17 + 25 and give only the result.",
            "context": ["x" * 500],
            "constraints": ["Be concise."],
        },
        "frame": {"requirements": ["Return the computed number."]},
    }
    prompt = build_public_prompt(record)
    assert prompt.startswith("### Instruction\nCompute 17 + 25")
    # Version 8.0: the response cue is no longer part of the prompt — the
    # model appends RESPONSE_CUE_TEXT itself after the optional prefix.
    assert "### Response" not in prompt.splitlines()[-1]
    assert prompt.endswith("Return only the answer.")
    from data import RESPONSE_CUE_TEXT

    assert "### Response" in RESPONSE_CUE_TEXT
    kept = HLWMCollator._preserve_ends(list(range(100)), 10)
    assert kept[:7] == list(range(7))
    assert kept[-3:] == [97, 98, 99]


def test_programmatic_anchor_enables_only_verified_policy_and_distinct_lane_roles():
    record = {
        "episode_id": "anchor-1",
        "domain": "behavior-anchor",
        "subdomain": "arithmetic",
        "input": {"user_request": "What is 2 + 3?", "context": [], "constraints": []},
        "frame": {"objective": "Calculate.", "requirements": [], "failure_contract": []},
        "lanes": [],
        "integration": {"published_answer": "5"},
        "commitment": {"decision": "publish"},
        "evaluation": {
            "expected_commit": True,
            "expected_action": "answer",
            "answer_spec": {"type": "numeric", "expected": 5},
            "negative_answer": "6",
        },
        "generation_metadata": {"programmatically_verified": True},
    }
    normalized = normalize_episode(record, num_lanes=2)
    assert normalized["policy_supervision_eligible"] is True
    assert normalized["is_behavior_anchor"] is True
    assert normalized["negative_public_target"] == "6"
    assert normalized["expected_commit"] is True
    assert normalized["answer_spec"]["expected"] == 5
    assert "Construct a direct solution" in normalized["lane_briefs"][0]
    assert "independent critic" in normalized["lane_briefs"][1]
    assert build_lane_brief({}, 0) != build_lane_brief({}, 1)
    # Execution-verified non-anchor rows are policy-eligible but must not be
    # counted as behavior anchors (v5.6 distinction).
    verified_code = dict(record, domain="python-function")
    normalized_code = normalize_episode(verified_code, num_lanes=2)
    assert normalized_code["policy_supervision_eligible"] is True
    assert normalized_code["is_behavior_anchor"] is False


def test_output_quality_rejects_prompt_copy_and_grades_meaning_not_wording():
    row = {
        "quality_checks": {"required_phrases": [], "forbidden_phrases": []},
        "answer_spec": {"type": "numeric", "expected": 42},
    }
    good = output_quality("The answer is 42.", row, ended_with_eos=True)
    copied = output_quality(
        "Human: You are asked to calculate 42.", row, ended_with_eos=False
    )
    wrong = output_quality("The answer is 41.", row, ended_with_eos=True)
    assert good["passed"] is True
    assert copied["passed"] is False and copied["no_prompt_leak"] is False
    assert wrong["passed"] is False


def test_semantic_graders_accept_equivalent_arithmetic_units_order_and_abstention():
    assert grade_semantic_answer(
        "After checking it, the answer is 36360.",
        {"type": "numeric", "expected": 36360},
    )["correct"]
    assert grade_semantic_answer(
        "That converts to 7,200 seconds.",
        {"type": "unit", "expected": 7200, "unit": "seconds"},
    )["correct"]
    assert grade_semantic_answer(
        "In ascending order they are 2, 7, 11, 19.",
        {"type": "ordering", "expected": [2, 7, 11, 19]},
    )["correct"]
    assert grade_semantic_answer(
        "The battery percentage cannot be determined from the provided information.",
        {"type": "abstention", "forbid_numbers": True},
    )["correct"]


def test_semantic_graders_reject_wrong_value_order_unit_and_guessed_abstention():
    assert not grade_semantic_answer(
        "The answer is 36361.", {"type": "numeric", "expected": 36360}
    )["correct"]
    assert not grade_semantic_answer(
        "That converts to 7,200 minutes.",
        {"type": "unit", "expected": 7200, "unit": "seconds"},
    )["correct"]
    assert not grade_semantic_answer(
        "Ascending: 19, 11, 7, 2.",
        {"type": "ordering", "expected": [2, 7, 11, 19]},
    )["correct"]
    assert not grade_semantic_answer(
        "The battery is probably 80%.",
        {"type": "abstention", "forbid_numbers": True},
    )["correct"]


def test_local_denoising_is_exactly_one_private_transition():
    torch.manual_seed(2)
    config = tiny_config(max_refinement_steps=4, diffusion_steps=4)
    model = HLWMForConditionalGeneration(config).train()
    input_ids, attention_mask, target_ids, target_mask, timesteps = sample_batch(config)
    lane_targets = torch.stack(
        (target_ids, torch.roll(target_ids, shifts=1, dims=-1)), dim=1
    )
    briefs = torch.randn(2, config.num_lanes, config.hidden_size)
    calls = []
    handle = model.backbone.layers[0].register_forward_hook(
        lambda module, args, result: calls.append(id(module))
    )
    output = model(
        input_ids,
        attention_mask,
        mode="local_denoise",
        lane_target_ids=lane_targets,
        lane_target_attention_mask=target_mask[:, None, :].expand_as(lane_targets),
        lane_briefs=briefs,
        timesteps=timesteps,
    )
    handle.remove()
    assert output.transition_count == 1
    assert torch.isfinite(output.lane_diversity_loss)
    assert output.route_indices.shape[-1] == 1
    assert len(calls) == 2  # committed context plus one private transition
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert has_finite_gradient(model.timestep_conditioner)
    assert has_finite_gradient(model.expert_bank)


def test_joint_forward_has_autoregressive_workspace_losses_and_gradients():
    torch.manual_seed(3)
    config = tiny_config()
    model = HLWMForConditionalGeneration(config).train()
    input_ids, attention_mask, target_ids, target_mask, timesteps = sample_batch(config)
    lane_targets = torch.stack(
        (target_ids, torch.roll(target_ids, shifts=1, dims=-1)), dim=1
    )
    lane_mask = target_mask[:, None, :].expand_as(lane_targets)
    output = model(
        input_ids,
        attention_mask,
        target_ids=target_ids,
        lane_target_ids=lane_targets,
        target_attention_mask=target_mask,
        lane_target_attention_mask=lane_mask,
        negative_target_ids=torch.roll(target_ids, shifts=2, dims=-1),
        timesteps=timesteps,
        adaptive_halt=False,
    )
    assert output.loss is not None and torch.isfinite(output.loss)
    assert set(output.loss_components) == {
        "denoise",
        "synthesis",
        "verification",
        "commitment",
        "halt",
        "router_balance",
        "router_entropy",
        "lane_diversity",
    }
    assert output.synthesis_logits is not None
    assert output.synthesis_logits.shape == (2, 4, config.vocab_size)
    assert output.negative_commitment_logits is not None
    assert output.commitment_logits.shape == (2, 3)
    assert output.route_path_masks.shape == (
        2,
        config.num_lanes,
        config.max_refinement_steps,
        config.num_experts,
    )
    assert output.workspace_prefix.shape == (
        2,
        config.synthesis_prefix_tokens,
        config.hidden_size,
    )
    torch.testing.assert_close(
        output.final_canvas_probabilities.sum(dim=-1),
        torch.ones(2, config.num_lanes, 4),
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    output.loss.backward()
    assert has_finite_gradient(model.backbone)
    assert has_finite_gradient(model.root_adapter)
    assert has_finite_gradient(model.expert_bank)
    assert has_finite_gradient(model.fast_candidate)
    assert has_finite_gradient(model.slow_candidate)
    assert has_finite_gradient(model.global_candidate)
    assert has_finite_gradient(model.verification_head)
    assert has_finite_gradient(model.workspace_prefix_projection)
    assert has_finite_gradient(model.commitment_head)


def test_unreviewed_rows_do_not_train_commitment_pairing():
    torch.manual_seed(31)
    config = tiny_config()
    model = HLWMForConditionalGeneration(config).train()
    input_ids, attention_mask, target_ids, target_mask, _ = sample_batch(config, batch=1)
    lane_targets = target_ids[:, None, :].expand(-1, config.num_lanes, -1).contiguous()
    lane_mask = target_mask[:, None, :].expand_as(lane_targets)
    output = model(
        input_ids,
        attention_mask,
        target_ids=target_ids,
        lane_target_ids=lane_targets,
        target_attention_mask=target_mask,
        lane_target_attention_mask=lane_mask,
        negative_target_ids=torch.roll(target_ids, shifts=1, dims=-1),
        commitment_supervision_mask=torch.zeros(1, dtype=torch.bool),
        adaptive_halt=False,
    )
    assert float(output.loss_components["commitment"].detach()) == 0.0


def test_private_lanes_are_isolated_until_summary_barrier():
    torch.manual_seed(5)
    config = tiny_config()
    model = HLWMForConditionalGeneration(config).eval()
    input_ids, attention_mask, target_ids, target_mask, timesteps = sample_batch(
        config, batch=1
    )
    noisy = torch.randint(
        0, config.vocab_size, (1, config.num_lanes, target_ids.shape[1])
    )
    briefs = torch.randn(1, config.num_lanes, config.hidden_size)
    changed_briefs = briefs.clone()
    changed_briefs[:, 0] += 7.0
    with torch.no_grad():
        original = model(
            input_ids,
            attention_mask,
            target_ids=target_ids,
            target_attention_mask=target_mask,
            noisy_lane_ids=noisy,
            timesteps=timesteps,
            lane_briefs=briefs,
            adaptive_halt=False,
        )
        changed = model(
            input_ids,
            attention_mask,
            target_ids=target_ids,
            target_attention_mask=target_mask,
            noisy_lane_ids=noisy,
            timesteps=timesteps,
            lane_briefs=changed_briefs,
            adaptive_halt=False,
        )
    assert not torch.allclose(original.denoise_logits[:, 0], changed.denoise_logits[:, 0])
    torch.testing.assert_close(
        original.denoise_logits[:, 1],
        changed.denoise_logits[:, 1],
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    assert original.synthesis_logits is not None and changed.synthesis_logits is not None
    assert not torch.allclose(original.synthesis_logits, changed.synthesis_logits)


def test_full_inference_runs_every_timestep_independent_of_joint_depth():
    config = tiny_config(diffusion_steps=4, max_refinement_steps=1, min_halt_steps=1)
    model = HLWMForConditionalGeneration(config).eval()
    input_ids, attention_mask, _, _, _ = sample_batch(config, batch=1)
    briefs = torch.randn(1, config.num_lanes, config.hidden_size)
    with torch.no_grad():
        output = model(
            input_ids,
            attention_mask,
            canvas_length=4,
            lane_briefs=briefs,
        )
    assert torch.equal(output.reverse_timesteps.cpu(), torch.tensor([[4, 3, 2, 1]]))
    assert output.route_indices.shape[-1] == config.diffusion_steps
    assert output.synthesis_logits is None
    assert output.loss is None


def test_recurrent_depth_reuses_one_parameter_system():
    config_one = tiny_config(
        max_refinement_steps=1, min_halt_steps=1, diffusion_steps=4
    )
    config_four = tiny_config(
        max_refinement_steps=4, min_halt_steps=1, diffusion_steps=4
    )
    model_one = HLWMForConditionalGeneration(config_one)
    model_four = HLWMForConditionalGeneration(config_four)
    assert sum(p.numel() for p in model_one.parameters()) == sum(
        p.numel() for p in model_four.parameters()
    )
    calls = []
    handle = model_four.backbone.layers[0].register_forward_hook(
        lambda module, args, result: calls.append(id(module))
    )
    input_ids, attention_mask, target_ids, target_mask, timesteps = sample_batch(
        config_four, batch=1
    )
    model_four.eval()
    with torch.no_grad():
        model_four(
            input_ids,
            attention_mask,
            target_ids=target_ids,
            target_attention_mask=target_mask,
            timesteps=timesteps,
            adaptive_halt=False,
        )
    handle.remove()
    assert len(calls) == 6  # context, four tied private updates, one answer pass
    assert len(set(calls)) == 1


def test_route_override_pins_every_window_to_the_named_expert():
    torch.manual_seed(23)
    config = tiny_config(diffusion_steps=4, max_refinement_steps=1, min_halt_steps=1)
    model = HLWMForConditionalGeneration(config).eval()
    input_ids, attention_mask, _, _, _ = sample_batch(config, batch=1)
    briefs = torch.randn(1, config.num_lanes, config.hidden_size)
    with torch.no_grad():
        forced = model(
            input_ids,
            attention_mask,
            canvas_length=4,
            lane_briefs=briefs,
            route_override=3,
        )
        free = model(
            input_ids,
            attention_mask,
            canvas_length=4,
            lane_briefs=briefs,
        )
    assert torch.all(forced.route_indices == 3)
    # Expert 3 is a leaf whose deterministic parent is expert 1, so the
    # executed path must include both nodes and never touch expert 0 or 2.
    assert torch.all(forced.route_path_masks[..., 3])
    assert torch.all(forced.route_path_masks[..., 1])
    assert not bool(forced.route_path_masks[..., 0].any())
    assert not bool(forced.route_path_masks[..., 2].any())
    assert not torch.allclose(forced.global_state, free.global_state)


def test_router_marginal_entropy_loss_is_zero_at_uniform_and_positive_when_collapsed():
    model = HLWMForConditionalGeneration(tiny_config(num_experts=4))
    uniform = {
        "router_probabilities": torch.full((2, 2, 3, 4), 0.25),
        "route_indices": torch.zeros(2, 2, 3, dtype=torch.long),
    }
    collapsed = {
        "router_probabilities": torch.tensor([0.97, 0.01, 0.01, 0.01]).expand(
            2, 2, 3, 4
        ),
        "route_indices": torch.zeros(2, 2, 3, dtype=torch.long),
    }
    uniform_loss = model._router_marginal_entropy_loss(uniform)
    collapsed_loss = model._router_marginal_entropy_loss(collapsed)
    assert abs(float(uniform_loss)) < 1.0e-5
    assert float(collapsed_loss) > 0.5


def test_candidate_policy_features_feed_the_commitment_head_exactly():
    torch.manual_seed(29)
    config = tiny_config()
    model = HLWMForConditionalGeneration(config).eval()
    hidden = torch.randn(2, 4, config.hidden_size)
    logits = torch.randn(2, 4, config.vocab_size)
    mask = torch.ones(2, 4, dtype=torch.long)
    global_state = torch.randn(2, config.hidden_size)
    verification_error = torch.rand(2, config.num_lanes)
    features = model.candidate_policy_features(
        hidden, logits, mask, global_state, verification_error
    )
    assert features.shape == (2, config.hidden_size * 2 + 2)
    torch.testing.assert_close(
        model.commitment_head(features),
        model._score_candidate(hidden, logits, mask, global_state, verification_error),
    )


def test_on_policy_head_fit_separates_graded_emissions_and_fails_soft():
    torch.manual_seed(37)
    config = tiny_config()
    model = HLWMForConditionalGeneration(config)
    width = config.hidden_size * 2 + 2
    direction = torch.randn(width)
    positive = direction[None, :] * 1.0 + 0.05 * torch.randn(16, width)
    negative = -direction[None, :] * 1.0 + 0.05 * torch.randn(24, width)
    examples = {
        "positive_features": positive,
        "negative_features": negative,
        "ranking_pairs": [(index % 16, index % 24) for index in range(48)],
        "emitted": 24,
        "valid_emissions": 16,
        "invalid_emissions": 8,
        "semantic_negatives": 16,
        "accuracy_by_grader": {"numeric": 0.7},
        "source_split": "train",
    }
    report = train_policy_heads_on_policy(
        model, examples, epochs=200, learning_rate=1.0e-2
    )
    assert report["trained"] is True
    assert report["last_loss"] < report["first_loss"]
    after = report["separation_after"]
    assert after["pairwise_commit_ranking_accuracy"] > 0.9
    assert (
        after["mean_positive_commit_probability"]
        > after["mean_negative_commit_probability"]
    )

    empty = train_policy_heads_on_policy(
        model,
        {
            "positive_features": torch.empty(0, 0),
            "negative_features": negative,
            "ranking_pairs": [],
            "emitted": 4,
            "valid_emissions": 0,
            "invalid_emissions": 4,
            "semantic_negatives": 4,
            "accuracy_by_grader": {},
            "source_split": "train",
        },
        epochs=10,
        learning_rate=1.0e-2,
    )
    assert empty["trained"] is False


def test_publication_requires_commit_low_risk_and_low_verifier_error():
    model = HLWMForConditionalGeneration(tiny_config())
    good = torch.tensor([[10.0, -10.0, -10.0]])
    high_risk = torch.tensor([[10.0, 10.0, -10.0]])
    high_error = torch.tensor([[10.0, -10.0, 10.0]])
    assert bool(model._commit_mask(good).item())
    assert not bool(model._commit_mask(high_risk).item())
    assert not bool(model._commit_mask(high_error).item())


def test_hlwm_generation_uses_qwen_candidate_not_private_canvas_argmax():
    torch.manual_seed(13)
    config = tiny_config(
        commitment_threshold=0.0,
        risk_threshold=1.0,
        verifier_error_threshold=1.0,
    )
    model = HLWMForConditionalGeneration(config).eval()
    input_ids = torch.tensor([[1, 7, 8]])
    mask = torch.ones_like(input_ids)
    with torch.no_grad():
        result = model.generate_hlwm(
            input_ids,
            mask,
            canvas_length=4,
            max_new_tokens=3,
            do_sample=False,
        )
    assert result.decision == "publish"
    assert torch.equal(result.output_ids, result.candidate_ids)
    assert result.macrocycles == config.diffusion_steps
    assert result.workspace.synthesis_logits is not None
    assert result.workspace.synthesis_logits.shape[1] == result.candidate_ids.shape[1]


def test_normal_causal_answer_path_and_generation_are_available():
    torch.manual_seed(19)
    config = tiny_config()
    model = HLWMForConditionalGeneration(config).eval()
    input_ids = torch.tensor([[1, 7, 8], [1, 9, 10]])
    mask = torch.ones_like(input_ids)
    with torch.no_grad():
        logits = model(input_ids, mask, mode="causal")
        generated = model.generate(
            input_ids,
            mask,
            max_new_tokens=3,
            do_sample=False,
            eos_token_id=config.vocab_size - 1,
        )
    assert logits.shape == (2, 3, config.vocab_size)
    assert generated.shape == (2, 6)
    torch.testing.assert_close(logits, model.causal_logits(input_ids, mask))


class FakeHFCausalLM(nn.Module):
    def __init__(self, source: HLWMForConditionalGeneration) -> None:
        super().__init__()
        config = source.config
        self.config = SimpleNamespace(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=config.attention_bias,
            mlp_bias=config.mlp_bias,
            tie_word_embeddings=config.tie_word_embeddings,
            attention_dropout=config.attention_dropout,
            pad_token_id=config.pad_token_id,
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,
        )
        self.model = source.backbone
        self.lm_head = source.lm_head

    def get_output_embeddings(self):
        return self.lm_head


def test_from_pretrained_transplants_qwen_style_weights_without_network():
    torch.manual_seed(17)
    source_config = tiny_config(max_refinement_steps=1, min_halt_steps=1)
    source = HLWMForConditionalGeneration(source_config)
    fake_hf = FakeHFCausalLM(source)

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(name, **kwargs):
            assert name == "offline-qwen3-0.6b"
            assert kwargs == {"revision": "pinned-test-revision"}
            return fake_hf

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = FakeAutoModel
    with patch.dict(sys.modules, {"transformers": fake_transformers}):
        loaded = HLWMForConditionalGeneration.from_pretrained(
            "offline-qwen3-0.6b",
            revision="pinned-test-revision",
            hlwm_overrides={
                "max_refinement_steps": 2,
                "min_halt_steps": 1,
                "num_lanes": 2,
                "diffusion_steps": 3,
                "num_experts": 3,
                "expert_top_level": 2,
                "expert_bottleneck": 12,
            },
        )
    torch.testing.assert_close(
        loaded.backbone.embed_tokens.weight, source.backbone.embed_tokens.weight
    )
    torch.testing.assert_close(
        loaded.backbone.layers[0].self_attn.q_proj.weight,
        source.backbone.layers[0].self_attn.q_proj.weight,
    )
    torch.testing.assert_close(loaded.lm_head.weight, source.lm_head.weight)
    assert loaded.config.max_refinement_steps == 2
    assert torch.count_nonzero(loaded.root_adapter.up.weight) == 0


def test_only_requested_qwen_tail_is_unfrozen():
    model = HLWMForConditionalGeneration(tiny_config(num_hidden_layers=3))
    model.freeze_language_substrate()
    model.unfreeze_language_tail(1)
    assert not any(parameter.requires_grad for parameter in model.backbone.layers[0].parameters())
    assert not any(parameter.requires_grad for parameter in model.backbone.layers[1].parameters())
    assert all(parameter.requires_grad for parameter in model.backbone.layers[2].parameters())
    assert all(parameter.requires_grad for parameter in model.backbone.norm.parameters())
    assert not any(parameter.requires_grad for parameter in model.lm_head.parameters())


def test_lora_sidecars_start_as_exact_zero_and_are_the_only_unfrozen_qwen_state():
    config = tiny_config(
        num_hidden_layers=3,
        lora_rank=4,
        lora_alpha=8.0,
        lora_tail_layers=1,
    )
    model = HLWMForConditionalGeneration(config).eval()
    input_ids = torch.tensor([[1, 7, 8, 0]])
    mask = torch.tensor([[1, 1, 1, 0]])
    with torch.no_grad():
        with_lora = model.backbone(input_ids=input_ids, attention_mask=mask)
        sidecars = []
        for module in model.backbone.modules():
            if hasattr(module, "enabled") and module.__class__.__name__ == "LoRAResidual":
                sidecars.append((module, module.enabled))
                module.enabled = False
        without_lora = model.backbone(input_ids=input_ids, attention_mask=mask)
        for module, was_enabled in sidecars:
            module.enabled = was_enabled
    torch.testing.assert_close(with_lora, without_lora)
    assert sum(was_enabled for _, was_enabled in sidecars) == 7
    assert torch.isfinite(with_lora).all()
    assert torch.count_nonzero(with_lora[:, -1]) == 0

    model.freeze_language_substrate()
    model.unfreeze_language_adapters()
    trainable_backbone = [
        name for name, parameter in model.backbone.named_parameters() if parameter.requires_grad
    ]
    assert trainable_backbone
    assert all("_lora." in name for name in trainable_backbone)
    assert all(name.startswith("layers.2.") for name in trainable_backbone)


def test_large_vocabulary_loss_is_fp32_and_padding_safe():
    vocabulary = 151_936
    logits = torch.zeros(1, 2, vocabulary, dtype=torch.float16)
    logits[:, 1] = torch.nan
    targets = torch.tensor([[17, 23]])
    mask = torch.tensor([[1, 0]])
    losses = _masked_token_cross_entropy(logits, targets, mask)
    assert losses.dtype == torch.float32
    assert torch.isfinite(losses).all()
    torch.testing.assert_close(
        losses[0, 0], torch.tensor(float(torch.log(torch.tensor(vocabulary)))), rtol=1e-5, atol=1e-5
    )
    assert float(losses[0, 1]) == 0.0


# ----------------------------------------------------------------- v5.6 tests


def test_execution_graders_pass_and_fail_code_sql_and_io():
    from semantic_grading import grade_semantic_answer

    tests_spec = {
        "type": "python_tests",
        "tests": ["assert add(2, 3) == 5", "assert add(-1, 1) == 0"],
        "time_limit_seconds": 6.0,
    }
    good = grade_semantic_answer("```python\ndef add(a, b):\n    return a + b\n```", tests_spec)
    assert good["correct"] is True
    bad = grade_semantic_answer("```python\ndef add(a, b):\n    return a - b\n```", tests_spec)
    assert bad["correct"] is False

    io_spec = {
        "type": "io_tests",
        "inputs": ["3 4\n", "10 5\n"],
        "outputs": ["7\n", "15\n"],
        "max_cases": 3,
        "time_limit_seconds": 6.0,
    }
    good_io = grade_semantic_answer(
        "```python\na, b = map(int, input().split())\nprint(a + b)\n```", io_spec
    )
    assert good_io["correct"] is True and good_io["cases_run"] == 2
    bad_io = grade_semantic_answer(
        "```python\na, b = map(int, input().split())\nprint(a * b)\n```", io_spec
    )
    assert bad_io["correct"] is False

    sql_spec = {"type": "sql_exact", "expected": "SELECT count(*) FROM head WHERE age > 56"}
    assert grade_semantic_answer(
        "```sql\nselect COUNT(*) from head where age > 56;\n```", sql_spec
    )["correct"] is True
    assert grade_semantic_answer(
        "```sql\nselect * from head\n```", sql_spec
    )["correct"] is False


def test_expert_diversity_loss_penalizes_identical_experts():
    torch.manual_seed(11)
    config = tiny_config(num_experts=4, expert_top_level=2)
    bank = RoutedAdapterBank(config)
    probe = torch.randn(6, config.hidden_size)
    with torch.no_grad():
        # Adapters zero-initialize their up projection; give every expert a
        # distinct nonzero function first.
        for expert in bank.experts:
            for parameter in expert.parameters():
                parameter.add_(0.5 * torch.randn_like(parameter))
    distinct = bank.output_diversity(probe)
    with torch.no_grad():
        reference = bank.experts[0].state_dict()
        for expert in bank.experts[1:]:
            expert.load_state_dict(reference)
    collapsed = bank.output_diversity(probe)
    assert float(collapsed) > 0.99  # identical experts are maximally similar
    assert float(collapsed) > float(distinct)
    collapsed.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in bank.experts[0].parameters()
    )


def test_local_denoise_reports_and_weights_expert_diversity():
    torch.manual_seed(3)
    config = tiny_config(
        max_refinement_steps=4, diffusion_steps=4, expert_diversity_weight=0.05
    )
    model = HLWMForConditionalGeneration(config).train()
    with torch.no_grad():
        # Move adapters off their zero initialization so the diversity term
        # is active in this one-step check.
        for expert in model.expert_bank.experts:
            for parameter in expert.parameters():
                parameter.add_(0.5 * torch.randn_like(parameter))
    input_ids, attention_mask, target_ids, target_mask, timesteps = sample_batch(config)
    lane_targets = torch.stack(
        (target_ids, torch.roll(target_ids, shifts=1, dims=-1)), dim=1
    )
    briefs = torch.randn(2, config.num_lanes, config.hidden_size)
    output = model(
        input_ids,
        attention_mask,
        mode="local_denoise",
        lane_target_ids=lane_targets,
        lane_target_attention_mask=target_mask[:, None, :].expand_as(lane_targets),
        lane_briefs=briefs,
        timesteps=timesteps,
    )
    assert torch.isfinite(output.expert_diversity_loss)
    assert float(output.expert_diversity_loss) > 0.0
    output.loss.backward()
    assert has_finite_gradient(model.expert_bank)


def test_policy_label_smoothing_desaturates_head_scores():
    torch.manual_seed(41)
    width = None
    reports = {}
    for smoothing in (0.0, 0.1):
        config = tiny_config()
        model = HLWMForConditionalGeneration(config)
        width = config.hidden_size * 2 + 2
        torch.manual_seed(41)
        direction = torch.randn(width)
        positive = direction[None, :] + 0.05 * torch.randn(16, width)
        negative = -direction[None, :] + 0.05 * torch.randn(16, width)
        examples = {
            "positive_features": positive,
            "negative_features": negative,
            "ranking_pairs": [(index, index) for index in range(16)],
            "emitted": 32,
            "valid_emissions": 16,
            "invalid_emissions": 16,
            "semantic_negatives": 16,
            "accuracy_by_grader": {"numeric": 0.8},
            "source_split": "train",
        }
        reports[smoothing] = train_policy_heads_on_policy(
            model,
            examples,
            epochs=300,
            learning_rate=1.0e-2,
            label_smoothing=smoothing,
        )
    plain = reports[0.0]["separation_after"]
    smoothed = reports[0.1]["separation_after"]
    assert reports[0.1]["label_smoothing"] == 0.1
    # Smoothing must preserve separation while pulling scores off the rails.
    assert smoothed["pairwise_commit_ranking_accuracy"] > 0.9
    assert (
        smoothed["mean_positive_commit_probability"]
        < plain["mean_positive_commit_probability"]
    )
    assert (
        smoothed["mean_negative_commit_probability"]
        > plain["mean_negative_commit_probability"]
    )


def test_v56_converted_episodes_gate_policy_on_executed_checks():
    import pathlib

    script = (
        pathlib.Path(__file__).resolve().parents[2]
        / "scripts"
        / "build_hlwm_v56_dataset.py"
    )
    if not script.exists():  # bundle copy on Kaggle omits repo scripts
        return
    import importlib.util

    spec = importlib.util.spec_from_file_location("v56_dataset", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    packet = {
        "problem": "Write a function double(x) returning 2*x.",
        "reference_answer": "def double(x):\n    return 2 * x",
        "curation": {"dataset_id": "google-research-datasets/mbpp"},
        "license": "cc-by-4.0",
    }
    answer_spec = {
        "type": "python_tests",
        "tests": ["assert double(2) == 4"],
        "setup": "",
        "time_limit_seconds": 6.0,
    }
    verified = module.convert_packet(
        "mbpp-train", packet, answer_spec, True, "executed_reference_tests"
    )
    unverified = module.convert_packet(
        "spider-train",
        {
            "problem": "How many heads are older than 56?",
            "reference_answer": "SELECT count(*) FROM head WHERE age > 56",
            "curation": {"dataset_id": "xlangai/spider"},
            "license": "cc-by-sa-4.0",
        },
        {"type": "sql_exact", "expected": "SELECT count(*) FROM head WHERE age > 56", "db_id": "x"},
        False,
        "official_gold_label",
    )
    eligible = normalize_episode(verified, num_lanes=2)
    ineligible = normalize_episode(unverified, num_lanes=2)
    assert eligible["policy_supervision_eligible"] is True
    assert ineligible["policy_supervision_eligible"] is False
    assert eligible["answer_spec"]["type"] == "python_tests"
    assert ineligible["answer_spec"]["type"] == "sql_exact"
    assert eligible["expected_commit"] is True


def test_causal_control_freeze_leaves_only_lora_trainable():
    torch.manual_seed(7)
    config = tiny_config(
        num_hidden_layers=3, lora_rank=4, lora_alpha=8.0, lora_tail_layers=1
    )
    model = HLWMForConditionalGeneration(config)
    model.freeze_language_substrate()
    model.unfreeze_language_adapters()
    model.unfreeze_language_tail(0)
    # the trainer's --causal-control freeze predicate, verbatim
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and "_lora." not in name and not name.startswith(
            "backbone.layers."
        ):
            parameter.requires_grad_(False)
    trainable = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    assert trainable, "control must still train the LoRA sidecars"
    assert all("_lora." in name for name in trainable)
    assert not any(
        parameter.requires_grad for parameter in model.expert_bank.parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in model.commitment_head.parameters()
    )


def test_expert_init_scale_wakes_the_diversity_penalty_from_step_one():
    torch.manual_seed(19)
    zero_init = RoutedAdapterBank(tiny_config(num_experts=4, expert_top_level=2))
    assert all(
        torch.count_nonzero(expert.up.weight) == 0 for expert in zero_init.experts
    )
    probe = torch.randn(6, zero_init.norm.weight.shape[0])
    assert float(zero_init.output_diversity(probe)) == 0.0  # inert (Study 5)
    seeded = RoutedAdapterBank(
        tiny_config(num_experts=4, expert_top_level=2, expert_init_scale=0.02)
    )
    assert all(
        torch.count_nonzero(expert.up.weight) > 0 for expert in seeded.experts
    )
    live = float(seeded.output_diversity(probe))
    assert live > 0.0
    assert live < 0.99  # independent random experts are not identical


def test_balanced_anchor_order_interleaves_grader_families():
    from train_kaggle import balanced_anchor_order

    rows = []
    anchor_indices = []
    families = ["numeric"] * 12 + ["unit"] * 4 + ["ordering"] * 4 + ["abstention"] * 4
    for index, family in enumerate(families):
        rows.append({"answer_spec": {"type": family}})
        anchor_indices.append(index)
    dataset = SimpleNamespace(rows=rows, anchor_indices=anchor_indices)
    order = balanced_anchor_order(dataset, seed=5)
    assert sorted(order) == list(range(len(families)))
    # the first 16 draws must contain every family exactly 4 times despite the
    # 3x numeric imbalance in the pool
    head = [rows[anchor_indices[position]]["answer_spec"]["type"] for position in order[:16]]
    assert all(head.count(family) == 4 for family in ("numeric", "unit", "ordering", "abstention"))
    assert balanced_anchor_order(dataset, seed=5) == order  # deterministic


class _StubAnchorDataset:
    def __init__(self, rows):
        self.rows = rows
        self.anchor_indices = list(range(len(rows)))

    def __getitem__(self, index):
        return self.rows[index]

    def __len__(self):
        return len(self.rows)


class _StubTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return "42"


def _stub_policy_collator(config):
    def collate(items):
        row = items[0]
        torch.manual_seed(7)
        input_ids = torch.randint(3, config.vocab_size, (1, 5))
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "target_ids": torch.randint(3, config.vocab_size, (1, 4)),
            "policy_supervision_mask": torch.tensor([True]),
            "is_behavior_anchor": torch.tensor([True]),
            "answer_specs": [row["answer_spec"]],
            "negative_target_ids": torch.randint(3, config.vocab_size, (1, 4)),
            "negative_target_attention_mask": torch.ones(1, 4, dtype=torch.long),
        }

    return collate


def test_validity_gated_collection_spends_extra_attempts_on_starved_families():
    from train_kaggle import collect_on_policy_policy_examples

    torch.manual_seed(3)
    config = tiny_config()
    model = HLWMForConditionalGeneration(config).eval()
    # numeric anchors grade "42" as correct; unit anchors require "kg" and fail
    rows = []
    for index in range(16):
        if index % 2 == 0:
            rows.append({"answer_spec": {"type": "numeric", "expected": 42.0}})
        else:
            rows.append({"answer_spec": {"type": "unit", "expected": 42.0, "unit": "kg"}})
    dataset = _StubAnchorDataset(rows)
    result = collect_on_policy_policy_examples(
        model,
        dataset,
        _stub_policy_collator(config),
        _StubTokenizer(),
        torch.device("cpu"),
        max_records=4,
        max_new_tokens=3,
        seed=11,
        min_valid_per_family=3,
        max_attempts=10,
    )
    assert result["validity_floor_met"] is False
    assert result["family_valid_positives"]["numeric"] >= 3
    assert result["family_valid_positives"]["unit"] == 0
    assert 4 < result["emitted"] <= 10  # kept collecting past the base budget
    # once numeric reached its floor, remaining attempts went to unit only
    assert result["family_attempts"]["unit"] >= result["family_attempts"]["numeric"]


def test_validity_gated_collection_stops_at_base_budget_when_floor_met():
    from train_kaggle import collect_on_policy_policy_examples

    torch.manual_seed(3)
    config = tiny_config()
    model = HLWMForConditionalGeneration(config).eval()
    rows = [{"answer_spec": {"type": "numeric", "expected": 42.0}} for _ in range(12)]
    dataset = _StubAnchorDataset(rows)
    result = collect_on_policy_policy_examples(
        model,
        dataset,
        _stub_policy_collator(config),
        _StubTokenizer(),
        torch.device("cpu"),
        max_records=4,
        max_new_tokens=3,
        seed=11,
        min_valid_per_family=3,
        max_attempts=12,
    )
    assert result["validity_floor_met"] is True
    assert result["emitted"] == 4


def test_generation_canary_reports_routing_and_quality_fields():
    from train_kaggle import generation_canary

    torch.manual_seed(5)
    config = tiny_config()
    model = HLWMForConditionalGeneration(config).train()
    collate = _stub_policy_collator(config)
    batches = [
        collate([{"answer_spec": {"type": "numeric", "expected": 42.0}}])
        for _ in range(3)
    ]
    report = generation_canary(
        model,
        batches,
        _StubTokenizer(),
        torch.device("cpu"),
        max_new_tokens=3,
        seed=0,
    )
    assert model.training  # train/eval state restored
    assert report["anchors"] == 3
    assert len(report["route_load"]) == config.num_experts
    assert abs(sum(report["route_load"]) - 1.0) < 1.0e-6
    assert 0.0 <= report["second_route_load"] <= 1.0
    assert 0.0 <= report["route_entropy_normalized"] <= 1.0
    for key in ("prompt_leak_rate", "format_marker_rate", "semantic_valid_rate", "commit_rate"):
        assert 0.0 <= report[key] <= 1.0


def test_decorrelated_policy_labels_by_negative_kind():
    torch.manual_seed(21)
    config = tiny_config()
    model = HLWMForConditionalGeneration(config)
    width = config.hidden_size * 2 + 2
    direction = torch.randn(width)
    positive = direction[None, :] + 0.05 * torch.randn(16, width)
    negative = -direction[None, :] + 0.05 * torch.randn(24, width)
    kinds = (["corrupt"] * 8) + (["invalid"] * 8) + (["semantic"] * 8)
    examples = {
        "positive_features": positive,
        "negative_features": negative,
        "negative_kinds": kinds,
        "ranking_pairs": [(index % 16, index % 24) for index in range(48)],
        "emitted": 24,
        "valid_emissions": 16,
        "invalid_emissions": 8,
        "semantic_negatives": 8,
        "accuracy_by_grader": {"numeric": 0.7},
        "source_split": "train",
    }
    report = train_policy_heads_on_policy(
        model, examples, epochs=150, learning_rate=1.0e-2
    )
    assert report["trained"] is True
    assert report["method"] == "on_policy_train_anchor_head_fit_v2_decorrelated"
    assert report["negative_kind_counts"] == {"corrupt": 8, "invalid": 8, "semantic": 8}
    # risk trains only on the corruption event, verifier only on wrongness
    assert report["label_scheme"]["corrupt"][1] > report["label_scheme"]["corrupt"][2]
    assert report["label_scheme"]["invalid"][2] > report["label_scheme"]["invalid"][1]
    assert report["separation_after"]["pairwise_commit_ranking_accuracy"] > 0.9


def test_domain_stratified_indices_round_robins_domains():
    from train_kaggle import domain_stratified_indices

    rows = [{"domain": ("code", "sql", "prose")[index % 3]} for index in range(30)]
    dataset = SimpleNamespace(rows=rows)
    picked = domain_stratified_indices(dataset, 9, seed=3)
    assert len(picked) == len(set(picked)) == 9
    head_domains = [rows[index]["domain"] for index in picked]
    assert all(head_domains.count(domain) == 3 for domain in ("code", "prose", "sql"))
    assert domain_stratified_indices(dataset, 9, seed=3) == picked  # deterministic


def test_single_expert_workspace_generates_and_diversity_is_zero():
    torch.manual_seed(23)
    config = tiny_config(num_experts=1, expert_top_level=1)
    model = HLWMForConditionalGeneration(config).eval()
    probe = torch.randn(4, config.hidden_size)
    diversity = model.expert_bank.output_diversity(probe)
    assert float(diversity) == 0.0 and torch.isfinite(diversity)
    input_ids = torch.tensor([[1, 7, 8]])
    with torch.no_grad():
        result = model.generate_hlwm(
            input_ids,
            torch.ones_like(input_ids),
            canvas_length=4,
            max_new_tokens=3,
            do_sample=False,
        )
    assert result.candidate_ids.shape[0] == 1
    assert int(result.workspace.route_indices.max()) == 0


def test_latent_readout_memory_widens_prefix_and_ablates():
    torch.manual_seed(31)
    config = tiny_config(workspace_memory_windows=2)
    model = HLWMForConditionalGeneration(config).eval()
    ids = torch.tensor([[1, 7, 8, 9]])
    mask = torch.ones_like(ids)
    with torch.no_grad():
        workspace = model.forward_hlwm(ids, mask, canvas_length=4, full_reverse_schedule=True)
    narrow = config.synthesis_prefix_tokens
    wide = narrow + config.num_lanes * (config.workspace_memory_windows + 1)
    assert workspace.workspace_prefix.shape[1] == wide
    # zero-initialized projections: memory tokens start as near-null values
    memory_tokens = workspace.workspace_prefix[:, narrow:, :]
    prefix_tokens = workspace.workspace_prefix[:, :narrow, :]
    assert float(memory_tokens.norm()) < float(prefix_tokens.norm())
    with torch.no_grad():
        ablated = model.generate_hlwm_nbest(
            ids, mask, canvas_length=4, max_new_tokens=2,
            candidate_temperatures=(0.0,), disable_workspace_memory=True,
            generator=torch.Generator().manual_seed(5),
        )
    assert ablated.candidate_ids[0].shape[0] == 1  # decodes under narrow interface


def test_nbest_selects_highest_publish_score_and_matches_single_greedy():
    torch.manual_seed(37)
    config = tiny_config()
    model = HLWMForConditionalGeneration(config).eval()
    ids = torch.tensor([[1, 7, 8]])
    mask = torch.ones_like(ids)
    with torch.no_grad():
        single = model.generate_hlwm(
            ids, mask, canvas_length=4, max_new_tokens=3, do_sample=False,
            generator=torch.Generator().manual_seed(9),
        )
        workspace_nbest = model.generate_hlwm_nbest(
            ids, mask, canvas_length=4, max_new_tokens=3,
            candidate_temperatures=(0.0,),
            generator=torch.Generator().manual_seed(9),
            candidate_channel="workspace",
        )
        causal_greedy = model.generate(
            ids, mask, max_new_tokens=3, do_sample=False,
        )
        nbest = model.generate_hlwm_nbest(
            ids, mask, canvas_length=4, max_new_tokens=3,
            candidate_temperatures=(0.0, 1.0, 1.3),
            generator=torch.Generator().manual_seed(9),
            return_features=True,
        )
    # The workspace channel matches the single-candidate generator; the
    # default (causal) channel matches the plain causal generator.
    assert torch.equal(workspace_nbest.candidate_ids[0], single.candidate_ids)
    assert torch.equal(
        nbest.candidate_ids[0][0], causal_greedy[0, ids.shape[1]:]
    )
    assert len(nbest.candidate_ids) == 3 == len(nbest.publish_scores)
    assert len(nbest.candidate_features) == 3
    assert len(nbest.candidate_mean_logprobs) == 3
    assert nbest.selected_index == max(
        range(3), key=nbest.publish_scores.__getitem__
    )
    if nbest.decision == "publish":
        assert torch.equal(nbest.output_ids, nbest.candidate_ids[nbest.selected_index])
    else:
        assert int(nbest.output_ids.item()) == config.abstain_token_id


def test_fit_publish_combiner_learns_signs_and_nondegenerate_threshold():
    from train_kaggle import fit_publish_combiner

    torch.manual_seed(41)
    valid = torch.stack(
        (
            torch.rand(64) * 0.3 + 0.7,   # high commit
            torch.rand(64) * 0.3,         # low risk
            torch.rand(64) * 0.3,         # low verifier error
        ),
        dim=1,
    )
    invalid = torch.stack(
        (torch.rand(64) * 0.3, torch.rand(64) * 0.3 + 0.7, torch.rand(64) * 0.3 + 0.7),
        dim=1,
    )
    rows = torch.cat((valid, invalid))
    labels = torch.cat((torch.ones(64), torch.zeros(64)))
    fitted = fit_publish_combiner(rows, labels)
    assert fitted["fitted"] is True
    assert fitted["weights"][0] > 0 > fitted["weights"][1]
    assert fitted["weights"][2] < 0
    assert fitted["feature_count"] == 3
    assert fitted["balanced_accuracy"] > 0.95
    # Version 8.0: a fourth (logprob) feature is accepted and learns a
    # positive sign when it separates the classes.
    logprob_valid = torch.rand(64) * 0.2 + 0.6
    logprob_invalid = torch.rand(64) * 0.2 + 0.1
    rows4 = torch.cat(
        (
            torch.cat((valid, logprob_valid[:, None]), dim=1),
            torch.cat((invalid, logprob_invalid[:, None]), dim=1),
        )
    )
    fitted4 = fit_publish_combiner(rows4, labels)
    assert fitted4["fitted"] is True
    assert fitted4["feature_count"] == 4
    assert fitted4["weights"][3] > 0
    # Version 9.0: calibration fits the four- AND five-feature forms
    # (``for feature_count in (4, 5)``); the fifth column is the within-pool
    # agreement fraction. Session G crashed both seeds because the combiner
    # still rejected five columns, so this case now covers the exact call the
    # calibrator makes.
    agreement_valid = torch.rand(64) * 0.3 + 0.6
    agreement_invalid = torch.rand(64) * 0.3 + 0.05
    rows5 = torch.cat(
        (
            torch.cat((valid, logprob_valid[:, None], agreement_valid[:, None]), dim=1),
            torch.cat((invalid, logprob_invalid[:, None], agreement_invalid[:, None]), dim=1),
        )
    )
    fitted5 = fit_publish_combiner(rows5, labels)
    assert fitted5["fitted"] is True
    assert fitted5["feature_count"] == 5
    assert fitted5["weights"][4] > 0
    try:
        fit_publish_combiner(torch.cat((rows5, rows5[:, :1]), dim=1), labels)
    except ValueError:
        pass
    else:
        raise AssertionError("six-feature rows must be rejected")


def test_conformal_threshold_guarantees_coverage_floor():
    from train_kaggle import conformal_publish_threshold

    torch.manual_seed(47)
    scores = torch.rand(128).tolist()
    result = conformal_publish_threshold(scores, target_coverage=0.35, seed=3)
    assert result["fitted"] is True
    # The order statistic guarantees expected coverage at or above target.
    assert result["guaranteed_expected_coverage"] >= 0.35
    assert result["calibration_empirical_coverage"] >= 0.30
    assert 0.0 < result["threshold"] < 1.0
    # Massive ties at the order statistic trigger the seeded jitter rule.
    tied = [0.5] * 100 + torch.rand(28).tolist()
    tied_result = conformal_publish_threshold(tied, target_coverage=0.35, seed=3)
    assert tied_result["tie_jitter_applied"] is True
    # Too few anchors refuse to fit rather than fake a floor.
    small = conformal_publish_threshold([0.5] * 8, target_coverage=0.35)
    assert small["fitted"] is False


def test_nbest_collection_counts_candidates_and_pairs_within_anchor():
    from train_kaggle import collect_on_policy_policy_examples

    torch.manual_seed(43)
    config = tiny_config()
    model = HLWMForConditionalGeneration(config).eval()
    rows = [{"answer_spec": {"type": "numeric", "expected": 42.0}} for _ in range(6)]
    dataset = _StubAnchorDataset(rows)
    result = collect_on_policy_policy_examples(
        model,
        dataset,
        _stub_policy_collator(config),
        _StubTokenizer(),
        torch.device("cpu"),
        max_records=8,
        max_new_tokens=3,
        seed=11,
        candidate_temperatures=(0.0, 1.2),
    )
    assert result["candidate_temperatures"] == [0.0, 1.2]
    assert result["anchors_consumed"] >= 4
    # stub tokenizer decodes every candidate to "42": all unique candidates are
    # valid numerics, duplicates are counted but stored once
    assert result["emitted"] >= 8
    assert result["valid_emissions"] + result["duplicate_candidates"] == result["emitted"]
    assert result["negative_kinds"].count("corrupt") == result["anchors_consumed"]
    assert len(result["ranking_pairs"]) > 0


# ---------------------------------------------------------------------------
# Version 8.0: channel layout, KV cache, gated prefix, KL anchor, lane views.
# ---------------------------------------------------------------------------


def test_kv_cached_backbone_matches_uncached_forward():
    torch.manual_seed(53)
    config = tiny_config()
    model = HLWMForConditionalGeneration(config).eval()
    ids = torch.randint(3, config.vocab_size, (1, 9))
    mask = torch.ones_like(ids)
    with torch.no_grad():
        full = model.backbone(input_ids=ids, attention_mask=mask, attention_mode="causal")
        # Prefill on the first six tokens, then feed the rest one at a time.
        prefill_hidden, past = model.backbone(
            input_ids=ids[:, :6],
            attention_mask=mask[:, :6],
            attention_mode="causal",
            use_cache=True,
        )
        pieces = [prefill_hidden]
        running = mask[:, :6]
        for position in range(6, 9):
            running = torch.cat((running, torch.ones(1, 1, dtype=mask.dtype)), dim=1)
            step_hidden, past = model.backbone(
                input_ids=ids[:, position : position + 1],
                attention_mask=running,
                attention_mode="causal",
                past_key_values=past,
                use_cache=True,
            )
            pieces.append(step_hidden)
    incremental = torch.cat(pieces, dim=1)
    torch.testing.assert_close(incremental, full, rtol=2.0e-4, atol=2.0e-4)


def test_cached_greedy_generate_matches_uncached_causal_logits():
    torch.manual_seed(59)
    config = tiny_config()
    model = HLWMForConditionalGeneration(config).eval()
    ids = torch.tensor([[3, 7, 11, 5]])
    mask = torch.ones_like(ids)
    with torch.no_grad():
        generated = model.generate(ids, mask, max_new_tokens=4, do_sample=False)
        # Replay the same greedy loop without any cache via causal_logits.
        replay = ids
        replay_mask = mask
        for _ in range(4):
            logits = model.causal_logits(replay, replay_mask)[:, -1]
            token = logits.argmax(dim=-1, keepdim=True)
            replay = torch.cat((replay, token), dim=1)
            replay_mask = torch.cat((replay_mask, torch.ones_like(token)), dim=1)
            if int(token.item()) == config.eos_token_id:
                break
    assert torch.equal(generated, replay[:, : generated.shape[1]])


def test_channel_teacher_force_never_inserts_bos_and_aligns_targets():
    torch.manual_seed(61)
    config = tiny_config(response_cue_ids=(5, 6))
    model = HLWMForConditionalGeneration(config).eval()
    ids = torch.tensor([[3, 7, 11]])
    mask = torch.ones_like(ids)
    targets = torch.tensor([[9, 13, 4]])
    target_mask = torch.ones_like(targets)
    with torch.no_grad():
        logits, target_hidden = model._channel_teacher_force(
            ids, mask, None, targets, target_mask
        )
        # Manual reference: [context][cue][targets] through the causal path;
        # the first target is predicted from the last cue token.
        manual_ids = torch.tensor([[3, 7, 11, 5, 6, 9, 13, 4]])
        manual_mask = torch.ones_like(manual_ids)
        manual_logits = model.causal_logits(manual_ids, manual_mask)
    torch.testing.assert_close(
        logits, manual_logits[:, 4:7], rtol=2.0e-4, atol=2.0e-4
    )
    assert logits.shape == (1, 3, config.vocab_size)
    assert target_hidden.shape[1] == 3


def test_generate_appends_cue_internally_but_returns_prompt_plus_answer():
    torch.manual_seed(67)
    config = tiny_config(response_cue_ids=(5, 6))
    model = HLWMForConditionalGeneration(config).eval()
    ids = torch.tensor([[3, 7, 11]])
    mask = torch.ones_like(ids)
    with torch.no_grad():
        generated = model.generate(ids, mask, max_new_tokens=3, do_sample=False)
        # The cue changes the continuation relative to a cue-free model.
        bare = HLWMForConditionalGeneration(tiny_config()).eval()
        bare.load_state_dict(model.state_dict())
        bare_generated = bare.generate(ids, mask, max_new_tokens=3, do_sample=False)
    assert generated.shape[1] <= ids.shape[1] + 3
    assert torch.equal(generated[:, : ids.shape[1]], ids)
    # The cue tokens themselves never appear in the returned sequence prefix.
    assert generated[0, ids.shape[1] :].tolist() != [5, 6]
    # Same weights, different layout => generally different continuations;
    # at minimum the two calls both return well-formed sequences.
    assert bare_generated.shape[1] <= ids.shape[1] + 3


def test_prefix_gate_zero_nulls_the_prefix_and_gate_is_trainable():
    torch.manual_seed(71)
    config = tiny_config(workspace_memory_windows=2)
    model = HLWMForConditionalGeneration(config).eval()
    ids = torch.tensor([[3, 7, 11, 5]])
    mask = torch.ones_like(ids)
    with torch.no_grad():
        model.prefix_gate.fill_(0.0)
        output = model.forward_hlwm(ids, mask, canvas_length=4)
        assert float(output.workspace_prefix.abs().sum()) == 0.0
        model.prefix_gate.fill_(0.05)
        output = model.forward_hlwm(ids, mask, canvas_length=4)
        assert float(output.workspace_prefix.abs().sum()) > 0.0
    assert model.prefix_gate.requires_grad


def test_synthesis_kl_anchor_appears_in_losses_and_is_finite():
    torch.manual_seed(73)
    config = tiny_config(response_cue_ids=(5, 6), synthesis_kl_weight=0.05)
    model = HLWMForConditionalGeneration(config)
    ids, mask, targets, target_mask, timesteps = sample_batch(config)
    output = model.forward_hlwm(
        ids,
        mask,
        target_ids=targets,
        target_attention_mask=target_mask,
        timesteps=timesteps,
    )
    assert "synthesis_kl" in output.loss_components
    kl_value = float(output.loss_components["synthesis_kl"])
    assert math.isfinite(kl_value) and kl_value >= 0.0
    assert output.loss is not None
    output.loss.backward()
    assert model.prefix_gate.grad is not None


def test_lane_briefs_receive_structurally_distinct_context_views():
    torch.manual_seed(79)
    config = tiny_config()
    model = HLWMForConditionalGeneration(config).eval()
    # A context whose halves differ strongly so the per-lane views separate.
    ids = torch.tensor([[3, 3, 3, 3, 29, 29, 29, 29]])
    mask = torch.ones_like(ids)
    with torch.no_grad():
        context_hidden = model._encode_committed(ids, mask)
        global_state = torch.zeros(1, config.hidden_size)
        briefs = model._prepare_briefs(
            context_hidden,
            mask,
            global_state,
            lane_briefs=None,
            lane_brief_ids=None,
            lane_brief_attention_mask=None,
        )
    assert briefs.shape == (1, config.num_lanes, config.hidden_size)
    difference = (briefs[0, 0] - briefs[0, 1]).abs().mean()
    assert float(difference) > 0.0


def test_four_weight_publish_rule_requires_and_uses_logprob_feature():
    torch.manual_seed(83)
    config = tiny_config(
        publish_weights=(2.0, -1.0, -1.0, 1.5),
        publish_bias=-0.5,
        publish_threshold=0.5,
    )
    model = HLWMForConditionalGeneration(config).eval()
    logits = torch.tensor([[2.0, -2.0, -2.0]])
    feature = torch.tensor([0.9])
    score_high = model.publish_score(logits, feature)
    score_low = model.publish_score(logits, torch.tensor([0.1]))
    assert float(score_high) > float(score_low)
    try:
        model.publish_score(logits)
    except ValueError:
        pass
    else:  # pragma: no cover - the guard must fire
        raise AssertionError("four-weight rule accepted a missing feature")
    # _commit_mask without the feature falls back to the triple rule instead
    # of crashing mid-training.
    mask = model._commit_mask(logits)
    assert mask.dtype == torch.bool


def test_audit_statistics_helpers_behave():
    from evaluate_checkpoint import (
        augrc,
        clopper_pearson_upper,
        difficulty_bin,
        mcnemar_mid_p,
        spearman_correlation,
    )

    # McNemar mid-p: symmetric evidence gives p ~ 0.5+, one-sided wins shrink it.
    assert mcnemar_mid_p(0, 0) == 1.0
    assert mcnemar_mid_p(5, 5) > 0.4
    assert mcnemar_mid_p(9, 1) < 0.05
    assert mcnemar_mid_p(1, 9) > 0.95
    # AUGRC: a perfect confidence ranking beats an inverted one.
    correct = [True, True, False, False]
    good = augrc([0.9, 0.8, 0.2, 0.1], correct)
    bad = augrc([0.1, 0.2, 0.8, 0.9], correct)
    assert good < bad
    # Spearman: monotone agreement is +1, reversal is -1.
    up = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert abs(spearman_correlation(up, up) - 1.0) < 1e-9
    assert abs(spearman_correlation(up, up[::-1]) + 1.0) < 1e-9
    # Clopper-Pearson upper bound is above the point estimate, below one.
    upper = clopper_pearson_upper(2, 20)
    assert 0.1 < upper < 0.4
    assert clopper_pearson_upper(0, 0) == 1.0
    # Difficulty bins split at the preregistered 2/8 and 5/8 boundaries.
    assert difficulty_bin(6 / 8) == "easy"
    assert difficulty_bin(5 / 8) == "medium"
    assert difficulty_bin(2 / 8) == "medium"
    assert difficulty_bin(1 / 8) == "hard"


def test_scaffold_leak_is_detected_mid_answer_not_only_at_the_start():
    """The Version 6.0 defect emitted turn scaffold *after* real answer text.

    ``bos_token_id`` resolving to ``<|endoftext|>`` inserted a document boundary
    mid-sequence, so the model finished the answer and then opened a new turn.
    A start-of-string leak test scores those rows clean, which would let the
    ``no_scaffold_leak`` gate certify a channel that is still broken.
    """

    from evaluate_checkpoint import output_quality
    from train_kaggle import candidate_has_no_prompt_leak

    row = {"answer_spec": {}, "quality_checks": {}}
    clean = "The total cost is 42 dollars."
    assert output_quality(clean, row, True)["no_prompt_leak"]
    assert candidate_has_no_prompt_leak(clean)

    trailing_scaffold = "The total cost is 42 dollars.\nHuman: thanks\nAssistant: sure"
    assert not output_quality(trailing_scaffold, row, True)["no_prompt_leak"]
    assert not candidate_has_no_prompt_leak(trailing_scaffold)

    for variant in ("\nUser: hi", "\n  assistant : ok", "\n> System: reset"):
        assert not candidate_has_no_prompt_leak(clean + variant), variant

    # Prose that merely mentions a role must not be scored as a leak.
    prose = "Ask the assistant: it will confirm the total is 42 dollars."
    assert output_quality(prose, row, True)["no_prompt_leak"]
    assert candidate_has_no_prompt_leak(prose)


def test_public_prompt_omits_the_response_cue_despite_the_requirements_heading():
    """``### Response requirements`` shares a prefix with the removed cue.

    A substring test for ``### Response`` therefore reports a cue that is not
    there; the cue's own text is the only sound check.
    """

    from data import RESPONSE_CUE_TEXT, build_public_prompt

    prompt = build_public_prompt({"input": {"user_request": "ping"}})
    assert RESPONSE_CUE_TEXT not in prompt
    assert not prompt.rstrip().endswith("### Response")
    assert "### Response requirements" in prompt


# ---------------------------------------------------------------------------
# Version 9.0: information-asymmetric channels, premise probe, gist control,
# five-feature publish rule, shared encode primitive, partial AUGRC.


class _WordTokenizer:
    """Deterministic offline word-level tokenizer for collator tests."""

    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    eos_token = "<eos>"
    vocab_size = 50_000

    def encode(self, text, add_special_tokens=False):
        ids = []
        for word in str(text).split():
            ids.append(2 + (hash(word) % (self.vocab_size - 2)))
        return ids

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


def _masked_episode(masked_value="48213", leak=False):
    request = "Calculate %s + 912 and state the exact result." % masked_value
    masked_request = request.replace(masked_value, "[withheld]")
    if leak:
        # Masks the WRONG operand, so the prompt differs from the public one
        # but the withheld literal is still present: a real leak.
        masked_request = request.replace("912", "[withheld]")
    return {
        "episode_id": "anchor-x",
        "domain": "behavior-anchor",
        "input": {"user_request": request, "user_request_masked": masked_request},
        "frame": {},
        "lanes": [],
        "integration": {"published_answer": "The result is 49125."},
        "evaluation": {
            "answer_spec": {"type": "numeric", "expected": 49125},
            "withheld_literals": [masked_value],
            "masked": True,
        },
        "commitment": {"decision": "publish"},
        "generation_metadata": {"programmatically_verified": True},
    }


def test_encode_preserving_ends_matches_collator_policy():
    from data import encode_preserving_ends

    tokenizer = _WordTokenizer()
    text = " ".join("w%d" % index for index in range(100))
    ids = encode_preserving_ends(tokenizer, text, 10)
    full = tokenizer.encode(text)
    assert ids == HLWMCollator._preserve_ends(full, 10)
    short = encode_preserving_ends(tokenizer, "a b c", 10)
    assert short == tokenizer.encode("a b c")


def test_masked_prompt_leak_raises_at_normalization():
    from data import normalize_episode

    clean = normalize_episode(_masked_episode(), num_lanes=1)
    assert clean["maskable"] and clean["masked"]
    assert "[withheld]" in clean["masked_prompt"]
    assert "48213" not in clean["masked_prompt"]
    try:
        normalize_episode(_masked_episode(leak=True), num_lanes=1)
    except ValueError as error:
        assert "leak" in str(error)
    else:  # pragma: no cover - the assert must fire
        raise AssertionError("a leaking masked prompt must raise")


def test_collator_masked_answer_channel_and_premise():
    from data import normalize_episode

    tokenizer = _WordTokenizer()
    collator = HLWMCollator(
        tokenizer,
        num_lanes=1,
        context_tokens=64,
        canvas_tokens=16,
        brief_tokens=16,
        causal_tokens=96,
    )
    masked_row = normalize_episode(_masked_episode(), num_lanes=1)
    unmasked_source = _masked_episode()
    unmasked_source["evaluation"]["masked"] = False
    unmasked_row = normalize_episode(unmasked_source, num_lanes=1)
    batch = collator([masked_row, unmasked_row])
    assert batch["masked_rows"].tolist() == [True, False]
    assert not torch.equal(batch["answer_input_ids"][0], batch["input_ids"][0])
    assert torch.equal(batch["answer_input_ids"][1], batch["input_ids"][1])
    # Premise supervision exists only on the masked row.
    assert int(batch["premise_attention_mask"][0].sum()) > 0
    assert int(batch["premise_attention_mask"][1].sum()) == 0
    # The premise token sequence must not appear inside the masked prompt.
    premise = [
        int(v) for v in batch["premise_ids"][0][batch["premise_attention_mask"][0].bool()]
    ]
    answer_row = batch["answer_input_ids"][0].tolist()
    for start in range(len(answer_row) - len(premise) + 1):
        assert answer_row[start : start + len(premise)] != premise


def test_forward_hlwm_masked_channel_premise_aux_and_kl_skip():
    config = tiny_config(premise_aux_weight=0.1, synthesis_kl_weight=0.1)
    model = HLWMForConditionalGeneration(config)
    input_ids, attention_mask, target_ids, target_mask, _ = sample_batch(config)
    answer_ids = input_ids.clone()
    answer_ids[:, 1] = 2  # masked surface differs from the workspace surface
    premise_ids = input_ids[:, :2].clone()
    premise_mask = torch.ones_like(premise_ids)
    negatives = torch.full((input_ids.shape[0], 8), 3, dtype=torch.long)
    output = model.forward_hlwm(
        input_ids,
        attention_mask,
        target_ids=target_ids,
        target_attention_mask=target_mask,
        answer_input_ids=answer_ids,
        skip_synthesis_kl=True,
        premise_ids=premise_ids,
        premise_attention_mask=premise_mask,
        premise_negative_ids=negatives,
    )
    assert "premise_aux" in output.loss_components
    assert torch.isfinite(output.loss_components["premise_aux"])
    accuracy = float(output.loss_components["premise_aux_accuracy"])
    assert 0.0 <= accuracy <= 1.0
    assert "synthesis_kl" not in output.loss_components  # skipped on masked rows
    with_kl = model.forward_hlwm(
        input_ids,
        attention_mask,
        target_ids=target_ids,
        target_attention_mask=target_mask,
    )
    assert "synthesis_kl" in with_kl.loss_components
    assert torch.isfinite(output.loss)


def test_full_prefix_ablation_equals_causal_greedy():
    config = tiny_config(response_cue_ids=(5, 6))
    model = HLWMForConditionalGeneration(config).eval()
    input_ids = torch.randint(3, config.vocab_size, (1, 6))
    mask = torch.ones_like(input_ids)
    ablated = model.generate_hlwm_nbest(
        input_ids,
        mask,
        canvas_length=4,
        max_new_tokens=4,
        candidate_temperatures=(0.0,),
        generator=torch.Generator().manual_seed(3),
        candidate_channel="workspace",
        disable_workspace_prefix=True,
    )
    causal = model.generate_hlwm_nbest(
        input_ids,
        mask,
        canvas_length=4,
        max_new_tokens=4,
        candidate_temperatures=(0.0,),
        generator=torch.Generator().manual_seed(3),
        candidate_channel="causal",
    )
    assert torch.equal(ablated.candidate_ids[0], causal.candidate_ids[0])


def test_gist_prefix_budget_must_match_and_runs():
    base = tiny_config()
    prefix_tokens = base.synthesis_prefix_tokens + (
        base.workspace_memory_windows * base.num_lanes + base.num_lanes
        if base.workspace_memory_windows > 0
        else 0
    )
    config = tiny_config(gist_prefix_tokens=prefix_tokens)
    model = HLWMForConditionalGeneration(config)
    input_ids, attention_mask, target_ids, target_mask, _ = sample_batch(config)
    output = model.forward_hlwm(
        input_ids,
        attention_mask,
        target_ids=target_ids,
        target_attention_mask=target_mask,
        prefix_source="gist",
    )
    assert torch.isfinite(output.loss)
    bad = tiny_config(gist_prefix_tokens=prefix_tokens + 1)
    bad_model = HLWMForConditionalGeneration(bad)
    try:
        bad_model.forward_hlwm(
            input_ids,
            attention_mask,
            target_ids=target_ids,
            target_attention_mask=target_mask,
            prefix_source="gist",
        )
    except ValueError as error:
        assert "budget" in str(error)
    else:  # pragma: no cover
        raise AssertionError("a mismatched gist budget must raise")


def test_publish_score_five_weights_requires_agreement():
    config = tiny_config(
        publish_weights=(0.5, -0.5, -0.5, 1.0, 0.7), publish_bias=-0.1
    )
    model = HLWMForConditionalGeneration(config)
    logits = torch.zeros(2, 3)
    logprob = torch.tensor([0.5, 0.9])
    agreement = torch.tensor([0.25, 1.0])
    scores = model.publish_score(logits, logprob, agreement)
    assert scores.shape == (2,)
    assert float(scores[1]) > float(scores[0])
    try:
        model.publish_score(logits, logprob)
    except ValueError as error:
        assert "agreement" in str(error)
    else:  # pragma: no cover
        raise AssertionError("five weights without agreement must raise")


def test_partial_augrc_restricts_to_the_band():
    from evaluate_checkpoint import augrc, partial_augrc

    confidences = [0.9, 0.8, 0.7, 0.6]
    correct = [True, True, False, False]
    full = augrc(confidences, correct)
    band = partial_augrc(confidences, correct, low=0.05, high=0.50)
    # In the low-coverage half only the two correct rows are accepted: zero risk.
    assert band == 0.0
    assert full > 0.0


def test_agreement_feature_is_well_formed():
    config = tiny_config(response_cue_ids=(5,))
    model = HLWMForConditionalGeneration(config).eval()
    input_ids = torch.randint(3, config.vocab_size, (1, 6))
    generation = model.generate_hlwm_nbest(
        input_ids,
        torch.ones_like(input_ids),
        canvas_length=4,
        max_new_tokens=3,
        candidate_temperatures=(0.0, 0.9, 0.9),
        generator=torch.Generator().manual_seed(9),
    )
    agreements = generation.candidate_agreements
    assert agreements is not None and len(agreements) == 3
    assert all(0.0 < value <= 1.0 for value in agreements)
    # The greedy candidate agrees at least with itself.
    assert agreements[0] >= 1.0 / 3.0


def test_risk_ucb_gate_is_arithmetically_winnable():
    from evaluate_checkpoint import clopper_pearson_upper

    # Twenty published rows with zero errors must clear the 0.15 bound,
    # otherwise the preregistered gate could never pass at its own floor.
    assert clopper_pearson_upper(0, 20) <= 0.15
    assert clopper_pearson_upper(0, 10) > 0.15  # below the floor it is void


def test_hlwm_loss_masked_batch_runs_with_pure_noise():
    from train_kaggle import hlwm_loss

    config = tiny_config(premise_aux_weight=0.1)
    model = HLWMForConditionalGeneration(config)
    input_ids, attention_mask, target_ids, target_mask, _ = sample_batch(config, batch=1)
    answer_ids = input_ids.clone()
    answer_ids[:, 2] = 2
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_ids": target_ids,
        "target_attention_mask": target_mask,
        "lane_target_ids": target_ids[:, None, :].expand(-1, config.num_lanes, -1),
        "lane_target_attention_mask": target_mask[:, None, :].expand(
            -1, config.num_lanes, -1
        ),
        "negative_target_ids": torch.randint(3, config.vocab_size, target_ids.shape),
        "negative_target_attention_mask": torch.ones_like(target_mask),
        "policy_supervision_mask": torch.tensor([True]),
        "lane_brief_ids": None,
        "lane_brief_attention_mask": None,
        "masked_rows": torch.tensor([True]),
        "answer_input_ids": answer_ids,
        "answer_attention_mask": attention_mask.clone(),
        "premise_ids": input_ids[:, :2].clone(),
        "premise_attention_mask": torch.ones(1, 2, dtype=torch.long),
        "premise_negative_ids": torch.full((1, 8), 3, dtype=torch.long),
        "verification_targets": torch.zeros(1, config.num_lanes),
        "halt_targets": torch.zeros(1, config.num_lanes),
        "commitment_targets": torch.zeros(1),
        "risk_targets": torch.zeros(1),
    }
    loss, metrics = hlwm_loss(model, batch)
    assert torch.isfinite(loss)
    assert metrics["masked_row"] == 1.0
    assert "premise_aux" in metrics
    assert "synthesis_kl" not in metrics


# ----------------------------------------------------------------------
# Version 10.0 dense-supervision channel: blocking preflight tests.


def _v10_config(**overrides):
    values = {
        "latent_thoughts": 3,
        "kv_prefix_slots": 4,
        "kv_prefix_rank": 8,
        "response_cue_ids": (5, 7),
        "lora_rank": 0,
        "lora_tail_layers": 0,
    }
    values.update(overrides)
    return HLWMConfig.tiny(**values)


def _v10_batch(config, seed=11):
    generator = torch.Generator().manual_seed(seed)
    batch = 2
    prompt = torch.randint(3, config.vocab_size, (batch, 9), generator=generator)
    prompt_mask = torch.ones(batch, 9, dtype=torch.long)
    prompt_mask[1, 6:] = 0  # right-padded second row
    prompt[1, 6:] = config.pad_token_id
    answer = torch.randint(3, config.vocab_size, (batch, 5), generator=generator)
    answer_mask = torch.ones(batch, 5, dtype=torch.long)
    return prompt, prompt_mask, answer, answer_mask


def test_v10_config_validation_and_defaults_stay_v9_compatible():
    default = HLWMConfig.tiny()
    assert default.latent_thoughts == 0 and default.kv_prefix_slots == 0
    try:
        HLWMConfig.tiny(kv_prefix_slots=4, prefix_attn_gate_init=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero gate init must be rejected when slots are enabled")
    try:
        HLWMConfig.tiny(latent_thoughts=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative latent_thoughts must be rejected")


def test_v10_gate_closed_equivalence_recovers_no_prefix_model():
    torch.manual_seed(23)
    config = _v10_config()
    model = HLWMForConditionalGeneration(config).eval()
    prompt, prompt_mask, answer, answer_mask = _v10_batch(config)
    with torch.no_grad():
        embeds, states = model.produce_latent_thoughts(prompt, prompt_mask)
        for layer in model.backbone.layers:
            layer.self_attn.prefix_attn_gate.zero_()  # tanh(0) = 0 exactly
        gated = model.student_channel_teacher_force(
            prompt, prompt_mask, embeds, states, answer, answer_mask,
            use_prefix_slots=True,
        )["logits"]
        plain = model.student_channel_teacher_force(
            prompt, prompt_mask, embeds, None, answer, answer_mask,
            use_prefix_slots=False,
        )["logits"]
    # Catches slot double-counting through the word branch exactly: with the
    # gate forced closed, the two paths must be the same computation.
    assert torch.allclose(gated, plain, atol=1e-5), float((gated - plain).abs().max())


def test_v10_gradient_liveness_at_init_gate_and_projector_and_producer():
    torch.manual_seed(29)
    config = _v10_config()
    model = HLWMForConditionalGeneration(config).train()
    prompt, prompt_mask, answer, answer_mask = _v10_batch(config)
    embeds, states = model.produce_latent_thoughts(prompt, prompt_mask)
    out = model.student_channel_teacher_force(
        prompt, prompt_mask, embeds, states, answer, answer_mask,
        use_prefix_slots=True,
    )
    loss = _masked_token_cross_entropy(out["logits"], answer, answer_mask).mean()
    loss.backward()
    gate_grads = [
        layer.self_attn.prefix_attn_gate.grad for layer in model.backbone.layers
    ]
    assert all(grad is not None for grad in gate_grads)
    assert sum(float(grad.abs().sum()) for grad in gate_grads) > 0.0
    projector_norm = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.kv_prefix_projector.parameters()
        if parameter.grad is not None
    )
    assert projector_norm > 0.0, "slot projector received no gradient (frozen-alpha class)"
    producer_norm = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.latent_projection.parameters()
        if parameter.grad is not None
    )
    assert producer_norm > 0.0, "latent projection received no gradient"


def test_v10_latent_perturbation_changes_answer_logits_under_right_padding():
    torch.manual_seed(31)
    config = _v10_config()
    model = HLWMForConditionalGeneration(config).eval()
    prompt, prompt_mask, answer, answer_mask = _v10_batch(config)
    with torch.no_grad():
        embeds, states = model.produce_latent_thoughts(prompt, prompt_mask)
        base = model.student_channel_teacher_force(
            prompt, prompt_mask, embeds, states, answer, answer_mask,
        )["logits"]
        for thought in range(config.latent_thoughts):
            perturbed = embeds.clone()
            perturbed[:, thought] += torch.randn_like(perturbed[:, thought]) * 0.5
            moved = model.student_channel_teacher_force(
                prompt, prompt_mask, perturbed, states, answer, answer_mask,
            )["logits"]
            delta = float((moved - base).abs().max())
            assert delta > 1e-6, "thought %d is invisible to the answer (dead-mask class)" % thought


def test_v10_first_thought_reads_last_valid_position_not_padding():
    torch.manual_seed(37)
    config = _v10_config()
    model = HLWMForConditionalGeneration(config).eval()
    prompt, prompt_mask, _, _ = _v10_batch(config)
    with torch.no_grad():
        _, padded_states = model.produce_latent_thoughts(prompt, prompt_mask)
        short_ids = prompt[1:2, :6]
        short_mask = prompt_mask[1:2, :6]
        _, unpadded_states = model.produce_latent_thoughts(short_ids, short_mask)
    assert torch.allclose(
        padded_states[1, 0], unpadded_states[0, 0], atol=1e-4
    ), "padded row's first thought differs from its unpadded computation"


def test_v10_grounding_matches_embedding_std():
    config = _v10_config()
    model = HLWMForConditionalGeneration(config)
    vectors = torch.randn(3, 5, config.hidden_size) * 7.0
    grounded = model.ground_latents(vectors)
    target = float(model.backbone.embed_tokens.weight.float().std())
    stds = grounded.float().std(dim=-1)
    assert torch.allclose(stds, torch.full_like(stds, target), rtol=0.05)


def test_v10_collect_hidden_states_returns_pre_answer_readout():
    torch.manual_seed(41)
    config = _v10_config()
    model = HLWMForConditionalGeneration(config).eval()
    prompt, prompt_mask, answer, answer_mask = _v10_batch(config)
    with torch.no_grad():
        embeds, states = model.produce_latent_thoughts(prompt, prompt_mask)
        out = model.student_channel_teacher_force(
            prompt, prompt_mask, embeds, states, answer, answer_mask,
            collect_hidden_states=True,
        )
    layer_states = out["pre_answer_layer_states"]
    assert len(layer_states) == config.num_hidden_layers
    assert all(state.shape == (2, config.hidden_size) for state in layer_states)


def test_v10_golden_parity_cached_decode_matches_teacher_force():
    """Cached step-wise decode == uncached full re-forward, slots and all.

    This is the blocking decode-path parity test from the Study 10 plan
    (amendment A6 item 0): greedily decode M tokens through the cached
    channel, then teacher-force exactly those tokens through the uncached
    training path and assert the argmax chain reproduces the decode."""

    torch.manual_seed(43)
    config = _v10_config()
    model = HLWMForConditionalGeneration(config).eval()
    for batch_rows in (1, 2):
        prompt, prompt_mask, _, _ = _v10_batch(config, seed=100 + batch_rows)
        prompt, prompt_mask = prompt[:batch_rows], prompt_mask[:batch_rows]
        with torch.no_grad():
            embeds, states = model.produce_latent_thoughts(prompt, prompt_mask)
            decoded = model.decode_candidate_v10(
                prompt, prompt_mask, embeds, states,
                max_new_tokens=4, temperature=0.0,
            )
            forced = model.student_channel_teacher_force(
                prompt, prompt_mask, embeds, states,
                decoded, torch.ones_like(decoded),
            )["logits"]
        # Position i of the teacher-forced logits predicts decoded token i.
        assert torch.equal(forced.argmax(dim=-1), decoded), (
            "cached decode diverges from the uncached forward (batch %d)" % batch_rows
        )


def test_v10_decode_slots_change_output_and_slotless_matches_plain():
    torch.manual_seed(47)
    config = _v10_config()
    model = HLWMForConditionalGeneration(config).eval()
    prompt, prompt_mask, _, _ = _v10_batch(config, seed=7)
    prompt, prompt_mask = prompt[:1], prompt_mask[:1]
    answer = torch.randint(3, config.vocab_size, (1, 5))
    answer_mask = torch.ones(1, 5, dtype=torch.long)
    with torch.no_grad():
        embeds, states = model.produce_latent_thoughts(prompt, prompt_mask)
        shuffled_states = states + torch.randn_like(states) * 2.0
        slotless_a = model.decode_candidate_v10(
            prompt, prompt_mask, embeds, states,
            use_prefix_slots=False, max_new_tokens=6,
        )
        slotless_b = model.decode_candidate_v10(
            prompt, prompt_mask, embeds, shuffled_states,
            use_prefix_slots=False, max_new_tokens=6,
        )
        base_logits = model.student_channel_teacher_force(
            prompt, prompt_mask, embeds, states, answer, answer_mask,
        )["logits"]
        moved_logits = model.student_channel_teacher_force(
            prompt, prompt_mask, embeds, shuffled_states, answer, answer_mask,
        )["logits"]
    # Slot ablation must be exactly insensitive to the states; the live slot
    # path must be measurably sensitive to them at the logit level (greedy
    # tokens are too coarse at the 0.08 gate init to certify liveness).
    assert torch.equal(slotless_a, slotless_b)
    assert float((base_logits - moved_logits).abs().max()) > 1e-6, (
        "slot path shows no sensitivity to thought states"
    )


def test_v10_family_routed_experts_route_and_swap():
    torch.manual_seed(53)
    config = _v10_config(mlp_expert_count=2, mlp_expert_rank=4, lora_rank=4, lora_tail_layers=2)
    model = HLWMForConditionalGeneration(config).eval()
    prompt, prompt_mask, answer, answer_mask = _v10_batch(config)
    # Identical duplicated row: only the route differs between the rows.
    prompt = prompt[:1].repeat(2, 1)
    prompt_mask = prompt_mask[:1].repeat(2, 1)
    answer = answer[:1].repeat(2, 1)
    answer_mask = answer_mask[:1].repeat(2, 1)
    with torch.no_grad():
        embeds, states = model.produce_latent_thoughts(prompt, prompt_mask)
        same = model.student_channel_teacher_force(
            prompt, prompt_mask, embeds, states, answer, answer_mask,
            route_index=torch.tensor([0, 0]),
        )["logits"]
        crossed = model.student_channel_teacher_force(
            prompt, prompt_mask, embeds, states, answer, answer_mask,
            route_index=torch.tensor([0, 1]),
        )["logits"]
        unrouted = model.student_channel_teacher_force(
            prompt, prompt_mask, embeds, states, answer, answer_mask,
        )["logits"]
    assert torch.allclose(same[0], same[1], atol=1e-5)
    # The cross-routing audit arm depends on expert identity being visible.
    assert float((crossed[0] - crossed[1]).abs().max()) > 1e-6, (
        "expert swap is invisible: routed experts output identically"
    )
    # Experts start nonzero by design, so routing must differ from no route.
    assert float((same - unrouted).abs().max()) > 1e-7


def test_v10_router_probe_is_detached_from_backbone():
    torch.manual_seed(59)
    config = _v10_config(mlp_expert_count=2, mlp_expert_rank=4, lora_rank=4, lora_tail_layers=2)
    model = HLWMForConditionalGeneration(config).train()
    prompt, prompt_mask, _, _ = _v10_batch(config)
    _, states = model.produce_latent_thoughts(prompt, prompt_mask)
    logits = model.route_family_logits(states)
    assert logits.shape == (2, config.router_families)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    loss.backward()
    assert model.family_router.weight.grad is not None
    assert float(model.family_router.weight.grad.abs().sum()) > 0.0
    backbone_grads = [
        parameter.grad for parameter in model.backbone.parameters()
        if parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
    ]
    assert not backbone_grads, "router probe leaked gradient into the backbone"


def _v10_training_batch(config, seed=61):
    generator = torch.Generator().manual_seed(seed)
    batch = 2
    vocab = config.vocab_size
    def ids(length, pad_from=None):
        values = torch.randint(3, vocab, (batch, length), generator=generator)
        mask = torch.ones(batch, length, dtype=torch.long)
        if pad_from is not None:
            values[1, pad_from:] = config.pad_token_id
            mask[1, pad_from:] = 0
        return values, mask
    prompt, prompt_mask = ids(9, pad_from=6)
    student, student_mask = ids(9, pad_from=6)
    answer, answer_mask = ids(4)
    trace, trace_mask = ids(8, pad_from=6)
    supervised = trace_mask.clone()
    supervised[:, -2:] = 0  # final answer-producing step excluded
    windows = config.latent_thoughts
    window_targets = torch.full((batch, windows, 2), -100, dtype=torch.long)
    for row in range(batch):
        valid = int(trace_mask[row].sum())
        bounds = [(valid * index) // windows for index in range(windows + 1)]
        for window in range(windows):
            span = trace[row, bounds[window]:bounds[window + 1]]
            window_targets[row, window, : min(2, span.numel())] = span[:2]
    return {
        "input_ids": prompt, "attention_mask": prompt_mask,
        "student_input_ids": student, "student_attention_mask": student_mask,
        "target_ids": answer, "target_attention_mask": answer_mask,
        "trace_input_ids": trace, "trace_attention_mask": trace_mask,
        "teacher_supervised_mask": supervised,
        "trace_window_targets": window_targets,
        "family_index": torch.tensor([0, 2]),
        "route_index": torch.tensor([0, 1]),
        "masked_rows": torch.tensor([True, False]),
    }


def test_v10_training_step_warm_and_main_produce_finite_grads():
    from train_kaggle import v10_training_step, optimizer_parameter_groups

    torch.manual_seed(67)
    config = _v10_config(mlp_expert_count=2, mlp_expert_rank=4, lora_rank=4, lora_tail_layers=2)
    model = HLWMForConditionalGeneration(config).train()
    batch = _v10_training_batch(config)

    warm = v10_training_step(
        model, batch, step=0, gamma=10.0, accumulation_scale=1.0, warm_phase=True,
    )
    assert all(math.isfinite(value) for value in warm.values()), warm
    assert "recon" in warm and "producer_mse" in warm
    model.zero_grad(set_to_none=True)

    for step in (0, 4):  # block 0: reconstruction; block 1: CoLaR (v10.5 block parity)
        metrics = v10_training_step(
            model, batch, step=step, gamma=10.0, accumulation_scale=1.0,
        )
        assert all(math.isfinite(value) for value in metrics.values()), metrics
        assert "student_ce" in metrics and "distill_l1" in metrics
        assert "latent_step_ce" in metrics and "inter_latent_cosine" in metrics
        if step == 4:
            assert "colar_segment_ce" in metrics
        else:
            assert "recon" in metrics
    # EMA normalizer filled and persisted as a buffer.
    assert float(model.distill_ema_std.min()) > 0.0
    assert "distill_ema_std" in dict(model.named_buffers())
    # Gradients reached the load-bearing modules through the grouped backward.
    for module_name in ("kv_prefix_projector", "latent_projection", "latent_step_head"):
        module = getattr(model, module_name)
        norm = sum(
            float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None
        )
        assert norm > 0.0, "%s received no gradient" % module_name
    gate_norm = sum(
        float(layer.self_attn.prefix_attn_gate.grad.abs().sum())
        for layer in model.backbone.layers
        if layer.self_attn.prefix_attn_gate is not None
        and layer.self_attn.prefix_attn_gate.grad is not None
    )
    assert gate_norm > 0.0
    expert_norm = sum(
        float(p.grad.abs().sum())
        for layer in model.backbone.layers
        if layer.mlp.mlp_experts is not None
        for expert in layer.mlp.mlp_experts
        for p in expert.parameters()
        if p.grad is not None
    )
    assert expert_norm > 0.0, "routed experts received no gradient"

    groups = optimizer_parameter_groups(model, weight_decay=0.1)
    assert groups[0]["weight_decay"] == 0.1 and groups[1]["weight_decay"] == 0.0
    decayed = {id(p) for p in groups[0]["params"]}
    for layer in model.backbone.layers:
        if layer.self_attn.prefix_attn_gate is not None:
            assert id(layer.self_attn.prefix_attn_gate) not in decayed, (
                "gate is being weight-decayed (constant close-the-channel force)"
            )


def test_v10_checkpoint_roundtrip_preserves_buffers():
    from train_kaggle import trainable_state_dict, v10_training_step

    torch.manual_seed(71)
    config = _v10_config()
    model = HLWMForConditionalGeneration(config).train()
    batch = _v10_training_batch(config)
    v10_training_step(model, batch, step=0, gamma=10.0, accumulation_scale=1.0)
    model.zero_grad(set_to_none=True)
    saved = trainable_state_dict(model)
    assert "distill_ema_std" in saved and "latent_embed_std" in saved
    clone = HLWMForConditionalGeneration(_v10_config())
    missing, unexpected = clone.load_state_dict(saved, strict=False)
    assert not unexpected
    assert torch.allclose(clone.distill_ema_std, model.distill_ema_std.cpu())
    assert torch.allclose(clone.latent_embed_std, model.latent_embed_std.cpu())


def test_v10_isolated_group_grad_norms_all_live():
    from train_kaggle import v10_training_step

    torch.manual_seed(73)
    config = _v10_config(mlp_expert_count=2, mlp_expert_rank=4, lora_rank=4, lora_tail_layers=2)
    model = HLWMForConditionalGeneration(config).train()
    batch = _v10_training_batch(config)
    for step, expected in ((0, {"teacher", "student"}), (4, {"teacher", "student", "colar"})):
        probe: dict = {}
        v10_training_step(
            model, batch, step=step, gamma=10.0, accumulation_scale=1.0, grad_probe=probe,
        )
        assert expected <= set(probe), probe
        for group, norm in probe.items():
            assert norm > 1e-8, "group %r has zero isolated gradient (dead-pass class)" % group
        model.zero_grad(set_to_none=True)


def test_v10_resume_continuity_five_plus_five_equals_ten():
    """Real save/load path: buffers, optimizer moments, and RNG must all
    round-trip so a salvage resume is bit-equivalent (the Session G/H
    pattern is a load-bearing path, not a contingency)."""

    import argparse, tempfile
    from pathlib import Path
    from train_kaggle import v10_training_step, trainable_state_dict, save_checkpoint, load_resume

    def build(seed):
        torch.manual_seed(seed)
        config = _v10_config()
        model = HLWMForConditionalGeneration(config).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        return model, optimizer

    def run_steps(model, optimizer, batch, count, start):
        records = []
        for index in range(count):
            torch.manual_seed(10_000 + start + index)
            metrics = v10_training_step(
                model, batch, step=start + index, gamma=10.0, accumulation_scale=1.0,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            records.append(metrics["student_ce"])
        return records

    args = argparse.Namespace(
        model="tiny", revision="tiny", steps=10, gradient_accumulation=1,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    model_a, optimizer_a = build(79)
    config = model_a.config
    batch = _v10_training_batch(config)
    straight = run_steps(model_a, optimizer_a, batch, 10, 0)

    model_b, optimizer_b = build(79)
    first = run_steps(model_b, optimizer_b, batch, 5, 0)
    with tempfile.TemporaryDirectory() as workdir:
        path = Path(workdir) / "checkpoint-step-000005.pt"
        save_checkpoint(
            path, model=model_b, optimizer=optimizer_b, scaler=scaler,
            step=5, optimizer_updates=5, skipped_optimizer_updates=0, args=args,
        )
        model_c, optimizer_c = build(9999)  # deliberately different init
        state = load_resume(path, model_c, optimizer_c, scaler, args)
        assert state["microsteps"] == 5
    second = run_steps(model_c, optimizer_c, batch, 5, 5)
    resumed = first + second
    for index, (a, b) in enumerate(zip(straight, resumed)):
        assert abs(a - b) < 1e-4, (
            "step %d diverged after resume: %.6f vs %.6f" % (index, a, b)
        )


def test_warm_gate_amended_predicate_session_i4():
    """Both session I-4 seeds cleared the 0.50 floor at step 200 and were
    aborted by the unamended 2x growth clause (em_600 >= 2*em_200 is
    unsatisfiable once em_200 > 0.5).  The amendment scopes growth to low
    starts only (plan section 13)."""

    from train_kaggle import warm_gate_passed

    # The two aborted seeds, verbatim from session I-4 metrics.
    assert warm_gate_passed(0.599406528189911, 0.6023738872403561, 0.5)
    assert warm_gate_passed(0.5637982195845698, 0.5905044510385756, 0.5)
    # The floor still binds regardless of the start.
    assert not warm_gate_passed(0.0, 0.45, 0.5)
    assert not warm_gate_passed(0.6, 0.45, 0.5)
    # A start below the floor still owes 2x growth.
    assert not warm_gate_passed(0.30, 0.55, 0.5)
    assert warm_gate_passed(0.25, 0.55, 0.5)
    # Zero start: the floor alone governs (the tiny-stack path).
    assert warm_gate_passed(0.0, 0.55, 0.5)


def test_tripwire_selectors_read_the_normalized_row_schema_session_i5():
    """Session I-5 regression: both binding tripwires selected rows through
    ``row['evaluation']['masked']``, a key ``normalize_episode`` does not
    emit.  The go/no-go therefore returned 0.0 without decoding a single row
    and aborted both seeds at step 1200, and the A1 masked-oversampling
    floor was silently inert for the whole run.  These selectors are tested
    against NORMALIZED rows -- the only schema they ever see in production.
    """

    from types import SimpleNamespace as NS
    import test_data_v10 as datafix
    from data import normalize_episode
    from train_kaggle import v10_anchor_indices, v10_masked_numeric_indices

    rows = [normalize_episode(row, num_lanes=1) for row in datafix._raw_rows(24)]
    dataset = NS(rows=rows)
    # The schema the old selectors read simply does not exist post-normalize.
    assert not any("masked" in (row.get("evaluation") or {}) for row in rows)

    buckets = v10_anchor_indices(dataset)
    masked_keys = sorted(key for key in buckets if key.endswith("|m"))
    assert masked_keys, buckets  # else the A1 floor has nothing to oversample
    assert sum(len(buckets[key]) for key in masked_keys) >= 4

    picked = v10_masked_numeric_indices(dataset, rows=8)
    assert picked, "go/no-go tripwire selected no rows"
    for index in picked:
        row = rows[index]
        assert bool(row.get("masked"))
        assert (row.get("v10") or {}).get("family") == "numeric"
        assert (row.get("answer_spec") or {}).get("expected") is not None
    assert len(v10_masked_numeric_indices(dataset, rows=1)) == 1


def test_go_nogo_excludes_rows_whose_answer_sits_in_their_own_masked_prompt():
    """The tripwire grades by substring containment, so a row whose answer
    already occurs in the masked prompt can be scored a hit by copying.

    Five of the shipped corpus's 2,212 masked rows collide this way (answer
    546 nested inside the distractor 69546), one of them inside the 32-row
    go/no-go sample.  Guess-proofing the numeric family (Amendment A3) is
    pointless if the prompt hands the answer over in a substring.
    """

    from types import SimpleNamespace as NS

    from train_kaggle import v10_masked_numeric_indices

    def row(index, expected, prompt):
        return {
            "v10": {"family": "numeric"},
            "masked": True,
            "masked_prompt": prompt,
            "answer_spec": {"type": "numeric", "expected": expected},
        }

    rows = [
        row(0, 546, "reference values: 69546, 48993. compute the total."),
        row(1, 546, "reference values: 69541, 48993. compute the total."),
        row(2, 7, "reference values: 61773. compute the total."),
    ]
    picked = v10_masked_numeric_indices(NS(rows=rows), rows=8)
    assert picked == [1], picked


def test_go_nogo_raises_instead_of_reporting_zero_on_an_empty_population():
    """An empty tripwire population is an instrument fault, not evidence.

    Returning 0.0 here is what let a broken selector manufacture a
    preregistered scientific abort in session I-5: a perfectly working
    channel would have produced the identical verdict.
    """

    from types import SimpleNamespace as NS
    from train_kaggle import v10_masked_numeric_em

    unmasked_only = NS(
        rows=[
            {"v10": {"family": "numeric"}, "masked": False, "answer_spec": {"expected": 1}}
        ]
    )
    try:
        v10_masked_numeric_em(None, unmasked_only, None, None)
    except ValueError as error:
        assert "vacuous" in str(error)
    else:
        raise AssertionError("an empty go/no-go population must raise")


def test_v10_run_orchestration_end_to_end_tiny():
    """Whole-run integration on the tiny stack: warm phase + gate, main
    phase, sampler masked-oversampling, checkpoints, and the w/o-L1 branch
    with its contamination guard — both the passing and the aborting warm
    gate."""

    import argparse, tempfile
    from pathlib import Path
    from types import SimpleNamespace as NS
    import test_data_v10 as datafix
    from data import normalize_episode
    from train_kaggle import run_v10_training, optimizer_parameter_groups

    rows = [normalize_episode(row, num_lanes=1) for row in datafix._raw_rows(24)]
    dataset = NS(rows=rows)
    collator = datafix._collator()

    def make_args(workdir, warm_em_floor):
        return argparse.Namespace(
            output_dir=Path(workdir), steps=8, warm_steps=4,
            gradient_accumulation=2, seed=3, distill_gamma=5.0,
            telemetry_every=0, masked_fraction_floor=0.5,
            warm_em_floor=warm_em_floor, gonogo_step=10_000,
            gonogo_masked_numeric_em=0.05, save_every=1_000,
            max_runtime_hours=1.0, wo_l1_branch_steps=2,
            learning_rate=1e-3, warmup_updates=1, weight_decay=0.0,
            model="tiny", revision="tiny",
        )

    def build_model():
        torch.manual_seed(83)
        config = datafix._tiny_v10_config(
            mlp_expert_count=2, mlp_expert_rank=4, lora_rank=4, lora_tail_layers=2
        )
        model = HLWMForConditionalGeneration(config).train()
        optimizer = torch.optim.AdamW(
            optimizer_parameter_groups(model, 0.0), lr=1e-3
        )
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        return model, optimizer, scaler

    with tempfile.TemporaryDirectory() as workdir:
        model, optimizer, scaler = build_model()
        verdict = run_v10_training(
            args=make_args(workdir, warm_em_floor=0.0),
            model=model, tokenizer=collator.tokenizer,
            train_data=dataset, validation_data=dataset, collator=collator,
            optimizer=optimizer, scaler=scaler, device=torch.device("cpu"),
            start_step=0, optimizer_updates=0, skipped_optimizer_updates=0,
        )
        assert verdict["warm_gate"]["passed"] is True
        assert verdict["completed_steps"] == 8
        assert verdict["full_checkpoint_sha256"] != verdict["wo_l1_branch_sha256"]
        assert verdict["masked_fraction_seen"] > 0.0
        assert (Path(workdir) / "checkpoint-v10-full.pt").exists()
        assert (Path(workdir) / "checkpoint-v10-wo-l1-branch.pt").exists()

    with tempfile.TemporaryDirectory() as workdir:
        model, optimizer, scaler = build_model()
        verdict = run_v10_training(
            args=make_args(workdir, warm_em_floor=0.99),
            model=model, tokenizer=collator.tokenizer,
            train_data=dataset, validation_data=dataset, collator=collator,
            optimizer=optimizer, scaler=scaler, device=torch.device("cpu"),
            start_step=0, optimizer_updates=0, skipped_optimizer_updates=0,
        )
        assert verdict.get("aborted") == "warm_gate", verdict
        assert verdict["warm_gate"]["passed"] is False
        # Session I-4: the abort must be visible to the notebook via
        # metrics.jsonl, not only via the return value — the notebook
        # audited a warm-only checkpoint as a completed run without it.
        import json as _json

        records = [
            _json.loads(line)
            for line in (Path(workdir) / "metrics.jsonl").read_text().splitlines()
            if line.strip()
        ]
        verdict_records = [r["v10_verdict"] for r in records if "v10_verdict" in r]
        assert verdict_records, "abort left no v10_verdict record in metrics.jsonl"
        assert verdict_records[-1]["aborted"] == "warm_gate"


def test_v10_gonogo_resume_reaborts_instead_of_training_past_the_tripwire():
    """Session I-5 regression: both seeds aborted at the go/no-go and the
    relaunch resumes from the abort checkpoint (scheduled saves land on
    save_every multiples, so landing exactly on the gonogo step implies an
    abort). The resume must re-evaluate the tripwire on the restored
    weights — re-aborting into the failure branch with a v10_verdict record
    — rather than silently continuing the run the tripwire killed."""

    import argparse, json as _json, tempfile
    from pathlib import Path
    from types import SimpleNamespace as NS
    import test_data_v10 as datafix
    from data import normalize_episode
    from train_kaggle import run_v10_training, optimizer_parameter_groups

    rows = [normalize_episode(row, num_lanes=1) for row in datafix._raw_rows(24)]
    dataset = NS(rows=rows)
    collator = datafix._collator()

    def make_args(workdir, resume=None):
        return argparse.Namespace(
            output_dir=Path(workdir), steps=8, warm_steps=2,
            gradient_accumulation=2, seed=3, distill_gamma=5.0,
            telemetry_every=0, masked_fraction_floor=0.5,
            warm_em_floor=0.0, gonogo_step=4,
            gonogo_masked_numeric_em=2.0,  # unsatisfiable: EM <= 1.0
            save_every=1_000, max_runtime_hours=1.0, wo_l1_branch_steps=2,
            learning_rate=1e-3, warmup_updates=1, weight_decay=0.0,
            model="tiny", revision="tiny", resume=resume,
        )

    def build_model():
        torch.manual_seed(83)
        config = datafix._tiny_v10_config(
            mlp_expert_count=2, mlp_expert_rank=4, lora_rank=4, lora_tail_layers=2
        )
        model = HLWMForConditionalGeneration(config).train()
        optimizer = torch.optim.AdamW(
            optimizer_parameter_groups(model, 0.0), lr=1e-3
        )
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        return model, optimizer, scaler

    with tempfile.TemporaryDirectory() as tempdir:
        first_dir = Path(tempdir) / "first"
        model, optimizer, scaler = build_model()
        first = run_v10_training(
            args=make_args(first_dir), model=model, tokenizer=collator.tokenizer,
            train_data=dataset, validation_data=dataset, collator=collator,
            optimizer=optimizer, scaler=scaler, device=torch.device("cpu"),
            start_step=0, optimizer_updates=0, skipped_optimizer_updates=0,
        )
        assert first.get("aborted") == "gonogo", first
        assert first.get("warm_gate", {}).get("passed") is True
        abort_checkpoint = first_dir / "checkpoint-step-000004.pt"
        assert abort_checkpoint.exists()

        resumed_dir = Path(tempdir) / "resumed"
        model, optimizer, scaler = build_model()
        resumed = run_v10_training(
            args=make_args(resumed_dir, resume=abort_checkpoint),
            model=model, tokenizer=collator.tokenizer,
            train_data=dataset, validation_data=dataset, collator=collator,
            optimizer=optimizer, scaler=scaler, device=torch.device("cpu"),
            start_step=4, optimizer_updates=2, skipped_optimizer_updates=0,
        )
        assert resumed.get("aborted") == "gonogo", resumed
        assert resumed.get("resumed_at_gonogo") == 4
        # The rung readout keys on warm_gate_passed; a resume past the warm
        # phase must carry the recorded result from the sibling metrics or
        # the salvage audit would misreport rung 1 (session I-5).
        assert resumed.get("warm_gate", {}).get("passed") is True
        assert resumed["warm_gate"]["carried_from_resume"] is True
        records = [
            _json.loads(line)
            for line in (resumed_dir / "metrics.jsonl").read_text().splitlines()
            if line.strip()
        ]
        final = [
            record["v10_verdict"] for record in records if "v10_verdict" in record
        ]
        assert final and final[-1]["aborted"] == "gonogo"
        assert final[-1]["warm_gate"]["carried_from_resume"] is True
        assert (resumed_dir / "checkpoint-step-000004.pt").exists()


def test_v10_preflight_passes_on_tiny_stack():
    import argparse
    from types import SimpleNamespace as NS
    import test_data_v10 as datafix
    from data import normalize_episode
    from train_kaggle import v10_preflight

    torch.manual_seed(89)
    rows = [normalize_episode(row, num_lanes=1) for row in datafix._raw_rows(16)]
    dataset = NS(rows=rows)
    collator = datafix._collator()
    config = datafix._tiny_v10_config(
        mlp_expert_count=2, mlp_expert_rank=4, lora_rank=4, lora_tail_layers=2
    )
    model = HLWMForConditionalGeneration(config).train()
    args = argparse.Namespace(
        weight_decay=0.0, learning_rate=1e-3, steps=8, wo_l1_branch_steps=2,
    )
    report = v10_preflight(
        args=args, model=model, tokenizer=collator.tokenizer,
        train_data=dataset, validation_data=dataset, collator=collator,
        device=torch.device("cpu"),
    )
    assert report["passed"] is True, report["failed_checks"]
    assert report["calibrated_gamma"] in (5.0, 10.0, 20.0)
    assert report["checks"]["seconds_per_step"] > 0
    assert set(report["checks"]["teacher_ceiling_by_family"]) >= {"numeric"}


def test_v10_loss_schedule_covers_every_family_in_both_parities():
    """v10.5 item 1 (P-SCHED-1): the CoLaR/reconstruction alternation must be
    decorrelated from the sampler's period-4 family rotation — the original
    odd/even schedule trained reconstruction only on two families."""

    from train_kaggle import v10_sample_indices

    buckets = {
        "numeric|u": [0], "unit|u": [1], "ordering|u": [2], "abstention|u": [3],
        "numeric|m": [4], "unit|m": [5], "ordering|m": [6],
    }
    order = v10_sample_indices(buckets, count=512, masked_floor=0.5, seed=7)
    family_of = {0: "numeric", 4: "numeric", 1: "unit", 5: "unit",
                 2: "ordering", 6: "ordering", 3: "abstention"}
    exposure = {"colar": set(), "recon": set()}
    for step, index in enumerate(order):
        block_parity = (step // 4) % 2
        exposure["colar" if block_parity == 1 else "recon"].add(family_of[index])
    assert exposure["colar"] == {"numeric", "unit", "ordering", "abstention"}, exposure
    assert exposure["recon"] == {"numeric", "unit", "ordering", "abstention"}, exposure
    # And demonstrate the defect the fix removes: raw step parity aliases.
    aliased = {"odd": set(), "even": set()}
    for step, index in enumerate(order):
        aliased["odd" if step % 2 else "even"].add(family_of[index])
    assert len(aliased["odd"]) < 4 or len(aliased["even"]) < 4, (
        "expected the raw parity to alias with the family rotation; "
        "if this fails the sampler changed and the schedule needs re-analysis"
    )


def test_v10_trainer_entry_point_is_last_toplevel_statement():
    """Session I regression: main() resolves names at CALL time, so the
    __main__ invocation must be the final top-level statement — appending
    definitions after it crashes the CLI (NameError) while module imports
    (and therefore every other test) stay green."""

    import ast, subprocess, sys
    from pathlib import Path

    source = Path(__file__).with_name("train_kaggle.py")
    tree = ast.parse(source.read_text())
    last = tree.body[-1]
    assert isinstance(last, ast.If), "entry point must be the last top-level statement"
    guard = ast.unparse(last.test)
    assert "__main__" in guard, guard
    result = subprocess.run(
        [sys.executable, str(source), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_v10_audit_entry_point_is_last_toplevel_statement():
    """The audit gained a CLI so the notebook can run one seed per GPU in
    its own process; it inherits the same entry-point discipline, and the
    --help subprocess proves the module is importable as a script (the
    failure mode that aborted session I at 160 seconds)."""

    import ast, subprocess, sys
    from pathlib import Path

    source = Path(__file__).with_name("evaluate_v10.py")
    tree = ast.parse(source.read_text())
    last = tree.body[-1]
    assert isinstance(last, ast.If), "entry point must be the last top-level statement"
    assert "__main__" in ast.unparse(last.test)
    result = subprocess.run(
        [sys.executable, str(source), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr[-2000:]
    for flag in ("--output-dir", "--data-dir", "--seed"):
        assert flag in result.stdout, result.stdout


def _run_naked_forwards_mixed_dtype(reduced_dtype):
    """Session I-2/I-5 regression harness (CUDA-class, reproduced on CPU):
    the runtime regime is a reduced-precision frozen base with fp32
    trainable parameters, and the preflight/eval/audit paths call the
    channel methods WITHOUT autocast. The fp32 mode embedding used to
    promote the assembled embeddings to fp32, crashing the first frozen
    reduced-precision linear — first in the decode paths (I-2, bf16), then
    in the audit's causal logprob through _channel_teacher_force (I-5,
    fp16, 1h45m into the audit)."""

    torch.manual_seed(97)
    config = _v10_config(mlp_expert_count=2, mlp_expert_rank=4, lora_rank=4, lora_tail_layers=2)
    model = HLWMForConditionalGeneration(config)
    model.to(reduced_dtype)
    for parameter in model.parameters():
        parameter.data = parameter.data.float() if parameter.requires_grad else parameter.data
    # Mirror main(): freeze nothing here (tiny config trains everything), so
    # force the mixed regime explicitly — frozen-style reduced precision for
    # the backbone linears, fp32 for HLWM modules.
    for name, parameter in model.named_parameters():
        if name.startswith("backbone.layers.") and "_lora" not in name and "prefix_attn_gate" not in name:
            parameter.data = parameter.data.to(reduced_dtype)
        elif name.startswith("backbone.embed_tokens") or name.startswith("lm_head"):
            parameter.data = parameter.data.to(reduced_dtype)
        else:
            parameter.data = parameter.data.float()
    model.eval()
    prompt, prompt_mask, answer, answer_mask = _v10_batch(config)
    with torch.no_grad():
        embeds, states = model.produce_latent_thoughts(prompt, prompt_mask)
        out = model.student_channel_teacher_force(
            prompt, prompt_mask, embeds, states, answer, answer_mask,
            collect_hidden_states=True,
        )
        assert torch.isfinite(out["logits"].float()).all()
        decoded = model.decode_candidate_v10(
            prompt[:1], prompt_mask[:1], embeds[:1], states[:1], max_new_tokens=3,
        )
        assert decoded.shape[0] == 1
        recon_targets = torch.cat((answer, answer), dim=1)
        recon_mask = torch.cat((answer_mask, answer_mask), dim=1)
        loss = model.reconstruct_from_thoughts(embeds, recon_targets, recon_mask)
        assert torch.isfinite(loss.float())
        plain = model._decode_candidate(
            prompt[:1], prompt_mask[:1], None, max_new_tokens=3,
        )
        assert plain.shape[0] == 1
        # Session I-5: the audit's abstention block scores every row through
        # the plain-causal teacher force — the one channel entry the I-2
        # sweep missed.
        logprob = model.causal_answer_mean_logprob(
            prompt[:1], prompt_mask[:1], answer[:1], answer_mask[:1]
        )
        assert torch.isfinite(logprob.float()).all()


def test_v10_naked_forwards_survive_bf16_base_with_fp32_trainables():
    _run_naked_forwards_mixed_dtype(torch.bfloat16)


def test_v10_naked_forwards_survive_fp16_base_with_fp32_trainables():
    """The session I-5 audit ran the checkpoint in fp16 (c10::Half), not
    bf16; the sweep must hold in the dtype the audit actually uses."""

    _run_naked_forwards_mixed_dtype(torch.float16)


def test_v10_freeze_sequence_leaves_gates_and_experts_trainable():
    """Session I-3 regression: main() runs freeze_language_substrate() then
    unfreeze_language_adapters(); the v10 gates live inside the backbone and
    were left frozen at init by that sequence (the on-device preflight
    caught gate_grad_norm == 0). The freeze path must exempt HLWM-native
    parameters while keeping the transplanted Qwen weights frozen."""

    config = _v10_config(mlp_expert_count=2, mlp_expert_rank=4, lora_rank=4, lora_tail_layers=2)
    model = HLWMForConditionalGeneration(config)
    model.freeze_language_substrate()
    model.unfreeze_language_adapters()
    gates = [
        layer.self_attn.prefix_attn_gate
        for layer in model.backbone.layers
        if layer.self_attn.prefix_attn_gate is not None
    ]
    assert gates and all(gate.requires_grad for gate in gates), (
        "prefix gates frozen by the substrate freeze (v8 frozen-alpha class)"
    )
    expert_params = [
        parameter
        for layer in model.backbone.layers
        if layer.mlp.mlp_experts is not None
        for expert in layer.mlp.mlp_experts
        for parameter in expert.parameters()
    ]
    assert expert_params and all(p.requires_grad for p in expert_params)
    assert not model.backbone.layers[0].self_attn.q_proj.weight.requires_grad
    assert not model.backbone.embed_tokens.weight.requires_grad
