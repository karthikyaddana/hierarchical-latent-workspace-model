from types import SimpleNamespace
import json

import torch
from torch import nn

from data_hlwm8b import DeterministicDistributedBatchSampler, HLWM8BCollator
from checkpointing_hlwm8b import load_training_state, save_checkpoint, sha256, verify_checkpoint
from evaluate_hlwm8b import calibration_for
from modeling_hlwm8b import HLWM8BConfig, HLWMSidecar, QwenHLWM


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [2 + (ord(character) % 29) for character in text]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        return "USER: " + messages[0]["content"] + "\nASSISTANT:"


def tiny_config():
    return HLWM8BConfig(
        latent_dim=48,
        num_lanes=2,
        canvas_tokens=8,
        num_experts=3,
        expert_bottleneck=16,
        diffusion_steps=3,
        refinement_steps=2,
        prefix_tokens=3,
        num_attention_heads=6,
    )


def test_collator_masks_prompt_and_keeps_answer():
    collator = HLWM8BCollator(TinyTokenizer(), max_prompt_tokens=24, max_answer_tokens=12)
    batch = collator(
        [
            {
                "id": "one",
                "prompt": "What is 2+2?",
                "chosen": "4",
                "rejected": "5",
                "difficulty": 1,
            }
        ]
    )
    assert batch["full_input_ids"].shape == (1, 36)
    assert (batch["labels"][:, :24] == -100).all()
    assert batch["difficulty_targets"].item() == 0.0


def test_sidecar_reuses_one_transition_and_produces_prefix():
    config = tiny_config()
    sidecar = HLWMSidecar(64, config)
    context = torch.randn(2, 20, 64)
    context_mask = torch.ones(2, 20, dtype=torch.long)
    target = torch.randn(2, 12, 64)
    target_mask = torch.ones(2, 12, dtype=torch.long)
    output = sidecar(context, context_mask, target, target_mask)
    assert output["prefix_embeddings"].shape == (2, 3, 64)
    assert output["route_indices"].shape == (2, 2, config.refinement_steps)
    assert output["transition_count"].item() == config.refinement_steps
    assert torch.isfinite(output["denoise_loss"])
    transition_ids = {id(module) for module in sidecar.modules() if module is sidecar.transition}
    assert len(transition_ids) == 1


def test_full_reverse_uses_every_timestep():
    config = tiny_config()
    sidecar = HLWMSidecar(64, config)
    output = sidecar(
        torch.randn(1, 8, 64),
        torch.ones(1, 8, dtype=torch.long),
        full_reverse=True,
    )
    assert output["route_indices"].shape[-1] == (
        config.diffusion_steps * config.refinement_steps
    )
    assert output["corruption_fraction"].item() == 1.0


def test_policy_scores_good_and_bad_candidates_with_three_heads():
    sidecar = HLWMSidecar(64, tiny_config())
    summary = torch.randn(2, 48)
    scores = sidecar.score_candidate(
        torch.randn(2, 10, 64), torch.ones(2, 10, dtype=torch.long), summary
    )
    assert scores.shape == (2, 3)


class TinyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=64)
        self.embeddings = nn.Embedding(128, 64)

    def get_input_embeddings(self):
        return self.embeddings

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        hidden = self.embeddings(input_ids) if inputs_embeds is None else inputs_embeds
        return SimpleNamespace(hidden_states=[hidden], loss=hidden.float().square().mean())

    def generate(self, input_ids=None, inputs_embeds=None, max_new_tokens=4, **kwargs):
        batch = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        answer = torch.full((batch, max_new_tokens), 7, dtype=torch.long, device=device)
        sequences = answer if input_ids is None else torch.cat((input_ids, answer), dim=1)
        return SimpleNamespace(sequences=sequences)


def test_generation_compacts_right_padding_and_verifies_candidate():
    model = QwenHLWM(TinyBase(), tiny_config())
    prompt = torch.tensor([[5, 6, 7, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 0, 0]])
    generated = model.generate_hlwm(
        prompt,
        mask,
        max_new_tokens=5,
        force_hlwm=True,
        pad_token_id=0,
        eos_token_id=1,
    )
    assert generated["generated_ids"].shape == (1, 5)
    assert generated["route_indices"].shape[-1] == (
        tiny_config().diffusion_steps * tiny_config().refinement_steps
    )
    verified = model.verify_candidate(
        prompt,
        mask,
        generated["generated_ids"],
        torch.ones_like(generated["generated_ids"]),
    )
    assert verified["verified_score"].shape == (1,)
    assert torch.isfinite(verified["verified_score"]).all()


def test_calibration_uses_validation_labels_and_fails_single_class():
    calibration = calibration_for([0.9, 0.8, 0.2, 0.1], [True, True, False, False])
    assert calibration["passed_research_gate"]
    assert calibration["test_split_used"] is False
    single = calibration_for([0.9, 0.8], [True, True])
    assert not single["passed_research_gate"]


def test_checkpoint_manifest_detects_corruption(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "adapter").mkdir(parents=True)
    member = checkpoint / "adapter" / "adapter.txt"
    member.write_text("valid", encoding="utf-8")
    sidecar = checkpoint / "hlwm-sidecar.safetensors"
    sidecar.write_bytes(b"sidecar")
    files = [
        {"path": str(path.relative_to(checkpoint)), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in (member, sidecar)
    ]
    (checkpoint / "checkpoint-complete.json").write_text(
        json.dumps({"format": "hlwm8b-resumable-v1", "global_update": 1, "files": files}),
        encoding="utf-8",
    )
    verify_checkpoint(checkpoint)
    member.write_text("broken", encoding="utf-8")
    try:
        verify_checkpoint(checkpoint)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted checkpoint was accepted")


def test_distributed_sampler_resumes_exact_microstep_order():
    full = list(
        DeterministicDistributedBatchSampler(
            dataset_size=11,
            batch_size=2,
            start_microstep=0,
            total_microsteps=9,
            seed=17,
            rank=1,
            world_size=2,
        )
    )
    resumed = list(
        DeterministicDistributedBatchSampler(
            dataset_size=11,
            batch_size=2,
            start_microstep=4,
            total_microsteps=9,
            seed=17,
            rank=1,
            world_size=2,
        )
    )
    assert resumed == full[4:]


class SaveableBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(3, 3))

    def save_pretrained(self, path, safe_serialization=True):
        path.mkdir(parents=True, exist_ok=True)
        path.joinpath("adapter.txt").write_text("adapter", encoding="utf-8")


class SaveableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = SaveableBase()
        self.sidecar = nn.Linear(3, 3)
        self.hlwm_config = tiny_config()


class FakeAccelerator:
    is_main_process = True
    process_index = 0
    scaler = None

    def wait_for_everyone(self):
        return None

    def unwrap_model(self, model):
        return model


def test_atomic_checkpoint_roundtrip_restores_sidecar_and_state(tmp_path):
    accelerator = FakeAccelerator()
    model = SaveableModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    before = {name: value.detach().clone() for name, value in model.sidecar.state_dict().items()}
    checkpoint = save_checkpoint(
        accelerator,
        model,
        optimizer,
        scheduler,
        tmp_path,
        {"global_update": 7, "phase": "joint", "joint_microstep": 28},
    )
    verify_checkpoint(checkpoint)
    with torch.no_grad():
        for parameter in model.sidecar.parameters():
            parameter.zero_()
    state = load_training_state(checkpoint, accelerator, model, optimizer, scheduler)
    assert state["global_update"] == 7
    for name, value in model.sidecar.state_dict().items():
        assert torch.equal(value, before[name])
