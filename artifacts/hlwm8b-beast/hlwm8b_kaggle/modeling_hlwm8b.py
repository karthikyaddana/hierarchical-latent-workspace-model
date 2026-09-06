from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class HLWM8BConfig:
    base_model: str = "Qwen/Qwen3-8B"
    base_revision: str = "b968826d9c46dd6066d109eabc6255188de91218"
    latent_dim: int = 384
    num_lanes: int = 2
    canvas_tokens: int = 16
    num_experts: int = 4
    expert_bottleneck: int = 96
    diffusion_steps: int = 4
    refinement_steps: int = 2
    prefix_tokens: int = 4
    num_attention_heads: int = 6
    dropout: float = 0.05
    causal_loss_weight: float = 1.0
    policy_loss_weight: float = 0.35
    denoise_loss_weight: float = 0.30
    route_balance_weight: float = 0.02
    lane_diversity_weight: float = 0.03
    difficulty_loss_weight: float = 0.05
    policy_margin: float = 0.5
    easy_threshold: float = 0.35

    def validate(self, hidden_size: int) -> None:
        positive = {
            "latent_dim": self.latent_dim,
            "num_lanes": self.num_lanes,
            "canvas_tokens": self.canvas_tokens,
            "num_experts": self.num_experts,
            "expert_bottleneck": self.expert_bottleneck,
            "diffusion_steps": self.diffusion_steps,
            "refinement_steps": self.refinement_steps,
            "prefix_tokens": self.prefix_tokens,
            "num_attention_heads": self.num_attention_heads,
        }
        invalid = [key for key, value in positive.items() if int(value) <= 0]
        if invalid:
            raise ValueError("positive HLWM values required: %s" % ", ".join(invalid))
        if self.latent_dim % self.num_attention_heads:
            raise ValueError("latent_dim must be divisible by num_attention_heads")
        if hidden_size <= 0:
            raise ValueError("base hidden size must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExpertAdapter(nn.Module):
    def __init__(self, hidden: int, bottleneck: int, dropout: float) -> None:
        super().__init__()
        self.down = nn.Linear(hidden, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, hidden, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: Tensor) -> Tensor:
        return self.up(self.dropout(F.silu(self.down(value))))


class SharedDenoisingTransition(nn.Module):
    """One root-preserving transition reused across lanes and diffusion time."""

    def __init__(self, config: HLWM8BConfig) -> None:
        super().__init__()
        hidden = config.latent_dim
        self.norm_attention = nn.RMSNorm(hidden)
        self.attention = nn.MultiheadAttention(
            hidden,
            config.num_attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm_root = nn.RMSNorm(hidden)
        self.root = nn.Sequential(
            nn.Linear(hidden, hidden * 2, bias=False),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden * 2, hidden, bias=False),
        )
        self.norm_expert = nn.RMSNorm(hidden)
        self.experts = nn.ModuleList(
            ExpertAdapter(hidden, config.expert_bottleneck, config.dropout)
            for _ in range(config.num_experts)
        )
        self.router = nn.Linear(hidden, config.num_experts, bias=False)

    def forward(self, canvas: Tensor, condition: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # canvas: [B, L, C, D], condition: [B, L, D]
        batch, lanes, tokens, hidden = canvas.shape
        flat = canvas.reshape(batch * lanes, tokens, hidden)
        flat_condition = condition.reshape(batch * lanes, hidden)
        attended_input = self.norm_attention(flat + flat_condition[:, None, :])
        attended, _ = self.attention(
            attended_input,
            attended_input,
            attended_input,
            need_weights=False,
        )
        flat = flat + attended
        flat = flat + self.root(self.norm_root(flat))
        pooled = flat.mean(dim=1) + flat_condition
        route_logits = self.router(pooled)
        route_indices = route_logits.argmax(dim=-1)
        expert_delta = torch.zeros_like(flat)
        normalized = self.norm_expert(flat)
        for expert_index, expert in enumerate(self.experts):
            rows = torch.nonzero(route_indices == expert_index, as_tuple=False).flatten()
            if rows.numel():
                selected = expert(normalized.index_select(0, rows))
                expert_delta.index_copy_(0, rows, selected.to(expert_delta.dtype))
        flat = flat + expert_delta
        return (
            flat.reshape(batch, lanes, tokens, hidden),
            route_logits.reshape(batch, lanes, -1),
            route_indices.reshape(batch, lanes),
        )


class HLWMSidecar(nn.Module):
    def __init__(self, base_hidden_size: int, config: HLWM8BConfig) -> None:
        super().__init__()
        config.validate(base_hidden_size)
        self.base_hidden_size = int(base_hidden_size)
        self.config = config
        hidden = config.latent_dim
        self.context_down = nn.Linear(base_hidden_size, hidden, bias=False)
        self.token_down = nn.Linear(base_hidden_size, hidden, bias=False)
        self.token_up = nn.Linear(hidden, base_hidden_size, bias=False)
        self.lane_embeddings = nn.Parameter(torch.empty(config.num_lanes, hidden))
        self.noise_canvas = nn.Parameter(
            torch.empty(config.num_lanes, config.canvas_tokens, hidden)
        )
        self.time_embeddings = nn.Embedding(config.diffusion_steps + 1, hidden)
        self.refinement_embeddings = nn.Embedding(config.refinement_steps, hidden)
        self.transition = SharedDenoisingTransition(config)
        self.prefix_positions = nn.Parameter(torch.empty(config.prefix_tokens, hidden))
        self.prefix_project = nn.Linear(hidden, base_hidden_size, bias=False)
        self.policy = nn.Sequential(
            nn.RMSNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, 3),
        )
        self.cheap_difficulty = nn.Sequential(
            nn.RMSNorm(base_hidden_size),
            nn.Linear(base_hidden_size, hidden // 2, bias=False),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.lane_embeddings, std=0.02)
        nn.init.normal_(self.noise_canvas, std=0.02)
        nn.init.normal_(self.prefix_positions, std=0.02)

    @staticmethod
    def masked_mean(hidden: Tensor, mask: Tensor) -> Tensor:
        weights = mask.to(hidden.dtype).unsqueeze(-1)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def difficulty_logits(self, prompt_embeddings: Tensor, prompt_mask: Tensor) -> Tensor:
        prompt_embeddings = prompt_embeddings.to(self.cheap_difficulty[0].weight.dtype)
        return self.cheap_difficulty(self.masked_mean(prompt_embeddings, prompt_mask)).squeeze(-1)

    def _compress_target(self, target_embeddings: Tensor, target_mask: Tensor) -> Tensor:
        # Pool a variable answer length into a small private canvas.
        target_embeddings = target_embeddings.to(self.token_down.weight.dtype)
        weights = target_mask.to(target_embeddings.dtype).unsqueeze(-1)
        values = self.token_down(target_embeddings) * weights
        pooled_values = F.adaptive_avg_pool1d(
            values.transpose(1, 2), self.config.canvas_tokens
        ).transpose(1, 2)
        pooled_weights = F.adaptive_avg_pool1d(
            weights.transpose(1, 2), self.config.canvas_tokens
        ).transpose(1, 2)
        pooled = pooled_values / pooled_weights.clamp_min(1.0e-6)
        pooled = pooled * (pooled_weights > 0).to(pooled.dtype)
        return pooled

    def _initial_canvas(
        self,
        batch: int,
        target_embeddings: Optional[Tensor],
        target_mask: Optional[Tensor],
        timesteps: Tensor,
        generator: Optional[torch.Generator],
    ) -> tuple[Tensor, Optional[Tensor], Tensor]:
        noise = self.noise_canvas.unsqueeze(0).expand(batch, -1, -1, -1)
        if target_embeddings is None or target_mask is None:
            corruption = torch.ones(
                batch,
                self.config.num_lanes,
                self.config.canvas_tokens,
                dtype=torch.bool,
                device=noise.device,
            )
            return noise, None, corruption
        target = self._compress_target(target_embeddings, target_mask)
        clean = target[:, None, :, :].expand(-1, self.config.num_lanes, -1, -1)
        probabilities = timesteps.to(clean.dtype) / float(self.config.diffusion_steps)
        probabilities = probabilities[:, None, None, None]
        random_values = torch.rand(
            clean.shape[:-1] + (1,),
            device=clean.device,
            generator=generator,
            dtype=clean.dtype,
        )
        corruption = random_values < probabilities
        canvas = torch.where(corruption, noise, clean)
        return canvas, clean, corruption.squeeze(-1)

    def forward(
        self,
        context_hidden: Tensor,
        context_mask: Tensor,
        target_embeddings: Optional[Tensor] = None,
        target_mask: Optional[Tensor] = None,
        *,
        full_reverse: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, Tensor]:
        batch = context_hidden.shape[0]
        context_hidden = context_hidden.to(self.context_down.weight.dtype)
        context = self.context_down(self.masked_mean(context_hidden, context_mask))
        if full_reverse:
            time_sequence = list(range(self.config.diffusion_steps, 0, -1))
            initial_t = torch.full(
                (batch,), self.config.diffusion_steps, device=context.device, dtype=torch.long
            )
        else:
            initial_t = torch.randint(
                1,
                self.config.diffusion_steps + 1,
                (batch,),
                device=context.device,
                generator=generator,
            )
            time_sequence = [initial_t]
        canvas, clean, corruption = self._initial_canvas(
            batch, target_embeddings, target_mask, initial_t, generator
        )
        routes: List[Tensor] = []
        route_logits: List[Tensor] = []
        for time_value in time_sequence:
            if isinstance(time_value, Tensor):
                timesteps = time_value
            else:
                timesteps = torch.full(
                    (batch,), int(time_value), device=context.device, dtype=torch.long
                )
            for refinement_index in range(self.config.refinement_steps):
                condition = (
                    context[:, None, :]
                    + self.lane_embeddings[None, :, :]
                    + self.time_embeddings(timesteps)[:, None, :]
                    + self.refinement_embeddings.weight[refinement_index][None, None, :]
                )
                canvas, logits, indices = self.transition(canvas, condition)
                routes.append(indices)
                route_logits.append(logits)
        lane_summaries = canvas.mean(dim=2)
        global_summary = lane_summaries.mean(dim=1)
        prefix_latent = global_summary[:, None, :] + self.prefix_positions[None, :, :]
        prefix = self.prefix_project(prefix_latent)
        denoise_loss = canvas.new_zeros(())
        if clean is not None:
            denoise_loss = F.mse_loss(canvas.float(), clean.float())
        normalized = F.normalize(lane_summaries.float(), dim=-1)
        similarity = torch.matmul(normalized, normalized.transpose(-1, -2))
        eye = torch.eye(self.config.num_lanes, device=similarity.device, dtype=torch.bool)
        # Penalize true collapse while allowing lanes to share useful content.
        lane_diversity = F.relu(
            similarity.masked_select(~eye[None]) - 0.85
        ).square().mean()
        probabilities = torch.softmax(route_logits[-1].float(), dim=-1).mean(dim=(0, 1))
        route_balance = (probabilities * probabilities).sum() * self.config.num_experts
        return {
            "canvas": canvas,
            "lane_summaries": lane_summaries,
            "global_summary": global_summary,
            "prefix_embeddings": prefix,
            "denoise_loss": denoise_loss,
            "lane_diversity_loss": lane_diversity,
            "route_balance_loss": route_balance,
            "route_indices": torch.stack(routes, dim=-1),
            "route_logits": torch.stack(route_logits, dim=-2),
            "corruption_fraction": corruption.float().mean(),
            "transition_count": torch.tensor(
                len(routes), device=canvas.device, dtype=torch.long
            ),
        }

    def score_candidate(
        self,
        candidate_embeddings: Tensor,
        candidate_mask: Tensor,
        global_summary: Tensor,
    ) -> Tensor:
        candidate_embeddings = candidate_embeddings.to(self.token_down.weight.dtype)
        candidate = self.token_down(self.masked_mean(candidate_embeddings, candidate_mask))
        return self.policy(torch.cat((candidate, global_summary), dim=-1))


class QwenHLWM(nn.Module):
    """Quantized Qwen language backbone plus trainable recurrent HLWM sidecar."""

    def __init__(self, base_model: nn.Module, config: HLWM8BConfig) -> None:
        super().__init__()
        self.base_model = base_model
        hidden_size = int(base_model.config.hidden_size)
        self.hlwm_config = config
        self.sidecar = HLWMSidecar(hidden_size, config)

    def get_input_embeddings(self) -> nn.Module:
        return self.base_model.get_input_embeddings()

    @contextlib.contextmanager
    def _context_mode(self):
        was_training = self.base_model.training
        self.base_model.eval()
        try:
            with torch.no_grad():
                yield
        finally:
            if was_training:
                self.base_model.train()

    def _context_hidden(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        with self._context_mode():
            output = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        return output.hidden_states[-1].detach()

    def forward(
        self,
        batch: Mapping[str, Any],
        generator: Optional[torch.Generator] = None,
        mode: str = "joint",
    ) -> Dict[str, Tensor]:
        if mode == "on_policy":
            return self.forward_on_policy(batch)
        if mode != "joint":
            raise ValueError("unknown training mode %s" % mode)
        embeddings = self.get_input_embeddings()
        prompt_embeddings = embeddings(batch["prompt_input_ids"])
        difficulty_logits = self.sidecar.difficulty_logits(
            prompt_embeddings.detach(), batch["prompt_attention_mask"]
        )
        context_hidden = self._context_hidden(
            batch["prompt_input_ids"], batch["prompt_attention_mask"]
        )
        chosen_embeddings = embeddings(batch["chosen_ids"])
        rejected_embeddings = embeddings(batch["rejected_ids"])
        workspace = self.sidecar(
            context_hidden,
            batch["prompt_attention_mask"],
            chosen_embeddings.detach(),
            batch["chosen_attention_mask"],
            full_reverse=False,
            generator=generator,
        )

        full_embeddings = embeddings(batch["full_input_ids"])
        prefix = workspace["prefix_embeddings"].to(full_embeddings.dtype)
        hard_rows = (batch["difficulty_targets"] >= 0.5).to(prefix.dtype)
        prefix = prefix * hard_rows[:, None, None]
        synthesis_embeddings = torch.cat((prefix, full_embeddings), dim=1)
        prefix_mask = hard_rows[:, None].expand(prefix.shape[:2]).to(
            device=batch["full_attention_mask"].device,
            dtype=batch["full_attention_mask"].dtype,
        )
        synthesis_mask = torch.cat((prefix_mask, batch["full_attention_mask"]), dim=1)
        prefix_labels = torch.full(
            prefix.shape[:2], -100, device=batch["labels"].device, dtype=batch["labels"].dtype
        )
        synthesis_labels = torch.cat((prefix_labels, batch["labels"]), dim=1)
        language = self.base_model(
            inputs_embeds=synthesis_embeddings,
            attention_mask=synthesis_mask,
            labels=synthesis_labels,
            use_cache=False,
            return_dict=True,
        )

        chosen_policy = self.sidecar.score_candidate(
            chosen_embeddings.detach(), batch["chosen_attention_mask"], workspace["global_summary"]
        )
        rejected_policy = self.sidecar.score_candidate(
            rejected_embeddings.detach(), batch["rejected_attention_mask"], workspace["global_summary"]
        )
        positive_labels = torch.tensor(
            [1.0, 0.0, 0.0], device=chosen_policy.device, dtype=chosen_policy.dtype
        ).expand_as(chosen_policy)
        negative_labels = 1.0 - positive_labels
        policy_bce = F.binary_cross_entropy_with_logits(chosen_policy, positive_labels)
        policy_bce = policy_bce + F.binary_cross_entropy_with_logits(
            rejected_policy, negative_labels
        )
        margin = self.hlwm_config.policy_margin
        policy_rank = (
            F.relu(margin - chosen_policy[:, 0] + rejected_policy[:, 0])
            + F.relu(margin - rejected_policy[:, 1] + chosen_policy[:, 1])
            + F.relu(margin - rejected_policy[:, 2] + chosen_policy[:, 2])
        ).mean()
        policy_loss = policy_bce + 0.5 * policy_rank
        difficulty_loss = F.binary_cross_entropy_with_logits(
            difficulty_logits.float(), batch["difficulty_targets"].float()
        )
        config = self.hlwm_config
        total = (
            config.causal_loss_weight * language.loss
            + config.policy_loss_weight * policy_loss
            + config.denoise_loss_weight * workspace["denoise_loss"]
            + config.route_balance_weight * workspace["route_balance_loss"]
            + config.lane_diversity_weight * workspace["lane_diversity_loss"]
            + config.difficulty_loss_weight * difficulty_loss
        )
        return {
            "loss": total,
            "causal_loss": language.loss.detach(),
            "policy_loss": policy_loss.detach(),
            "denoise_loss": workspace["denoise_loss"].detach(),
            "route_balance_loss": workspace["route_balance_loss"].detach(),
            "lane_diversity_loss": workspace["lane_diversity_loss"].detach(),
            "difficulty_loss": difficulty_loss.detach(),
            "difficulty_probability": torch.sigmoid(difficulty_logits.detach()).mean(),
            "hlwm_training_fraction": hard_rows.detach().float().mean(),
            "chosen_commit_probability": torch.sigmoid(chosen_policy[:, 0].detach()).mean(),
            "rejected_commit_probability": torch.sigmoid(rejected_policy[:, 0].detach()).mean(),
            "chosen_risk_probability": torch.sigmoid(chosen_policy[:, 1].detach()).mean(),
            "rejected_risk_probability": torch.sigmoid(rejected_policy[:, 1].detach()).mean(),
            "route_indices": workspace["route_indices"].detach(),
            "corruption_fraction": workspace["corruption_fraction"].detach(),
        }

    def forward_on_policy(self, batch: Mapping[str, Any]) -> Dict[str, Tensor]:
        embeddings = self.get_input_embeddings()
        prompt_embeddings = embeddings(batch["prompt_input_ids"])
        difficulty_logits = self.sidecar.difficulty_logits(
            prompt_embeddings.detach(), batch["prompt_attention_mask"]
        )
        context_hidden = self._context_hidden(
            batch["prompt_input_ids"], batch["prompt_attention_mask"]
        )
        workspace = self.sidecar(
            context_hidden,
            batch["prompt_attention_mask"],
            full_reverse=True,
        )
        candidate_embeddings = embeddings(batch["on_policy_ids"])
        rejected_embeddings = embeddings(batch["rejected_ids"])
        candidate_policy = self.sidecar.score_candidate(
            candidate_embeddings.detach(),
            batch["on_policy_attention_mask"],
            workspace["global_summary"],
        )
        rejected_policy = self.sidecar.score_candidate(
            rejected_embeddings.detach(),
            batch["rejected_attention_mask"],
            workspace["global_summary"],
        )
        correct = batch["on_policy_correct"].to(candidate_policy.dtype).unsqueeze(-1)
        positive = torch.tensor(
            [1.0, 0.0, 0.0], device=candidate_policy.device, dtype=candidate_policy.dtype
        ).unsqueeze(0)
        negative = 1.0 - positive
        candidate_labels = correct * positive + (1.0 - correct) * negative
        rejected_labels = negative.expand_as(rejected_policy)
        policy_loss = F.binary_cross_entropy_with_logits(candidate_policy, candidate_labels)
        policy_loss = policy_loss + F.binary_cross_entropy_with_logits(
            rejected_policy, rejected_labels
        )
        positive_rows = batch["on_policy_correct"].bool()
        if positive_rows.any():
            good = candidate_policy[positive_rows]
            bad = rejected_policy[positive_rows]
            margin = self.hlwm_config.policy_margin
            policy_loss = policy_loss + 0.5 * (
                F.relu(margin - good[:, 0] + bad[:, 0])
                + F.relu(margin - bad[:, 1] + good[:, 1])
                + F.relu(margin - bad[:, 2] + good[:, 2])
            ).mean()
        difficulty_loss = F.binary_cross_entropy_with_logits(
            difficulty_logits.float(), batch["difficulty_targets"].float()
        )
        total = policy_loss + self.hlwm_config.difficulty_loss_weight * difficulty_loss
        return {
            "loss": total,
            "policy_loss": policy_loss.detach(),
            "difficulty_loss": difficulty_loss.detach(),
            "candidate_commit_probability": torch.sigmoid(
                candidate_policy[:, 0].detach()
            ).mean(),
            "candidate_risk_probability": torch.sigmoid(
                candidate_policy[:, 1].detach()
            ).mean(),
            "candidate_error_probability": torch.sigmoid(
                candidate_policy[:, 2].detach()
            ).mean(),
            "rejected_commit_probability": torch.sigmoid(
                rejected_policy[:, 0].detach()
            ).mean(),
            "correct_fraction": batch["on_policy_correct"].float().mean(),
            "route_indices": workspace["route_indices"].detach(),
        }

    @torch.no_grad()
    def prepare_generation_prefix(
        self, prompt_input_ids: Tensor, prompt_attention_mask: Tensor
    ) -> Dict[str, Tensor]:
        embeddings = self.get_input_embeddings()
        prompt_embeddings = embeddings(prompt_input_ids)
        difficulty = torch.sigmoid(
            self.sidecar.difficulty_logits(prompt_embeddings, prompt_attention_mask)
        )
        context_hidden = self._context_hidden(prompt_input_ids, prompt_attention_mask)
        workspace = self.sidecar(
            context_hidden,
            prompt_attention_mask,
            full_reverse=True,
        )
        return {
            "difficulty_probability": difficulty,
            "prefix_embeddings": workspace["prefix_embeddings"],
            "route_indices": workspace["route_indices"],
            "global_summary": workspace["global_summary"],
        }

    @torch.no_grad()
    def generate_hlwm(
        self,
        prompt_input_ids: Tensor,
        prompt_attention_mask: Tensor,
        *,
        max_new_tokens: int,
        force_hlwm: bool = False,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        # Training batches are right padded, while decoder-only generation must
        # end each row on a real prompt token. Compact and left-pad first.
        active_lengths = prompt_attention_mask.sum(dim=1).to(torch.long)
        maximum_active = int(active_lengths.max().item())
        padding_id = int(pad_token_id if pad_token_id is not None else 0)
        compact_ids = torch.full(
            (prompt_input_ids.shape[0], maximum_active),
            padding_id,
            dtype=prompt_input_ids.dtype,
            device=prompt_input_ids.device,
        )
        compact_mask = torch.zeros_like(compact_ids)
        for row_index, active_length in enumerate(active_lengths.tolist()):
            tokens = prompt_input_ids[row_index][prompt_attention_mask[row_index].bool()]
            compact_ids[row_index, -active_length:] = tokens[-active_length:]
            compact_mask[row_index, -active_length:] = 1
        prompt_input_ids = compact_ids
        prompt_attention_mask = compact_mask
        embeddings = self.get_input_embeddings()
        prompt_embeddings = embeddings(prompt_input_ids)
        difficulty = torch.sigmoid(
            self.sidecar.difficulty_logits(prompt_embeddings, prompt_attention_mask)
        )
        use_hlwm_rows = torch.ones_like(difficulty, dtype=torch.bool) if force_hlwm else (
            difficulty >= self.hlwm_config.easy_threshold
        )
        if prompt_input_ids.shape[0] > 1 and not bool(
            (use_hlwm_rows == use_hlwm_rows[0]).all()
        ):
            rows = [
                self.generate_hlwm(
                    prompt_input_ids[index : index + 1],
                    prompt_attention_mask[index : index + 1],
                    max_new_tokens=max_new_tokens,
                    force_hlwm=False,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                )
                for index in range(prompt_input_ids.shape[0])
            ]
            maximum = max(row["generated_ids"].shape[1] for row in rows)
            padding_id = int(pad_token_id if pad_token_id is not None else 0)
            generated_rows = []
            route_rows = []
            maximum_routes = max(row["route_indices"].shape[-1] for row in rows)
            for row in rows:
                ids = row["generated_ids"]
                if ids.shape[1] < maximum:
                    ids = F.pad(ids, (0, maximum - ids.shape[1]), value=padding_id)
                generated_rows.append(ids)
                routes = row["route_indices"]
                if routes.shape[-1] < maximum_routes:
                    routes = F.pad(
                        routes, (0, maximum_routes - routes.shape[-1]), value=-1
                    )
                route_rows.append(routes)
            return {
                "generated_ids": torch.cat(generated_rows, dim=0),
                "difficulty_probability": torch.cat(
                    [row["difficulty_probability"] for row in rows], dim=0
                ),
                "used_hlwm": torch.cat([row["used_hlwm"] for row in rows], dim=0),
                "route_indices": torch.cat(route_rows, dim=0),
            }
        use_hlwm = bool(use_hlwm_rows[0])
        route_indices = torch.empty(
            prompt_input_ids.shape[0],
            self.hlwm_config.num_lanes,
            0,
            dtype=torch.long,
            device=prompt_input_ids.device,
        )
        if use_hlwm:
            prepared = self.prepare_generation_prefix(prompt_input_ids, prompt_attention_mask)
            prefix = prepared["prefix_embeddings"].to(prompt_embeddings.dtype)
            inputs_embeds = torch.cat((prefix, prompt_embeddings), dim=1)
            prefix_mask = torch.ones(
                prefix.shape[:2],
                dtype=prompt_attention_mask.dtype,
                device=prompt_attention_mask.device,
            )
            attention_mask = torch.cat((prefix_mask, prompt_attention_mask), dim=1)
            generated = self.base_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                return_dict_in_generate=True,
            ).sequences
            route_indices = prepared["route_indices"]
        else:
            generated = self.base_model.generate(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                return_dict_in_generate=True,
            ).sequences
            if generated.shape[1] > prompt_input_ids.shape[1]:
                generated = generated[:, prompt_input_ids.shape[1] :]
        if generated.shape[1] > max_new_tokens:
            generated = generated[:, -max_new_tokens:]
        return {
            "generated_ids": generated,
            "difficulty_probability": difficulty,
            "used_hlwm": torch.full(
                (prompt_input_ids.shape[0],),
                use_hlwm,
                dtype=torch.bool,
                device=prompt_input_ids.device,
            ),
            "route_indices": route_indices,
        }

    @torch.no_grad()
    def verify_candidate(
        self,
        prompt_input_ids: Tensor,
        prompt_attention_mask: Tensor,
        candidate_ids: Tensor,
        candidate_attention_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if candidate_attention_mask is None:
            candidate_attention_mask = torch.ones_like(candidate_ids)
        context_hidden = self._context_hidden(prompt_input_ids, prompt_attention_mask)
        workspace = self.sidecar(
            context_hidden,
            prompt_attention_mask,
            full_reverse=True,
        )
        candidate_embeddings = self.get_input_embeddings()(candidate_ids)
        logits = self.sidecar.score_candidate(
            candidate_embeddings,
            candidate_attention_mask,
            workspace["global_summary"],
        )
        probabilities = torch.sigmoid(logits.float())
        publish, risk, error = probabilities.unbind(dim=-1)
        return {
            "policy_logits": logits,
            "publish_probability": publish,
            "risk_probability": risk,
            "error_probability": error,
            "verified_score": publish * (1.0 - risk) * (1.0 - error),
            "route_indices": workspace["route_indices"],
        }
