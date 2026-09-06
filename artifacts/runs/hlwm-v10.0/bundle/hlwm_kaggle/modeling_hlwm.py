"""Qwen-initializable Hierarchical Latent Workspace model, Version 5.

This module deliberately separates two execution paths:

* ``causal_logits`` / ``generate`` preserve an ordinary autoregressive language
  model path (shared Qwen-like backbone plus the always-on root adapter).
* ``forward_hlwm`` performs private categorical denoising in isolated tensor
  lanes, using the same transformer weights at every recurrent update.  Lane
  summaries cross a synchronization barrier only after private refinement,
  where they update the global state and feed synthesis and commitment heads.

The implementation is a research prototype, not a claim that the randomly
initialized routing, halting, verification, or commitment heads are calibrated.
Those heads affect computation and have explicit supervised targets; none is a
decorative label.  ``from_pretrained`` transplants the compatible language
weights from a Hugging Face Qwen2/Qwen3-style causal LM and leaves only the new
HLWM modules to be learned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class HLWMConfig:
    """Configuration for the Qwen-like substrate and HLWM additions.

    The defaults match the important tensor dimensions of Qwen3-0.6B.  Loading
    a real checkpoint should nevertheless use :meth:`from_pretrained`, which
    reads the authoritative checkpoint config instead of trusting defaults.
    """

    vocab_size: int = 151_936
    hidden_size: int = 1_024
    intermediate_size: int = 3_072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 40_960
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1.0e-6
    attention_bias: bool = False
    mlp_bias: bool = False
    use_qk_norm: bool = True
    tie_word_embeddings: bool = False
    attention_dropout: float = 0.0

    # Low-rank sidecars adapt the language substrate without changing the
    # dtype of any transplanted Qwen weight.  A zero rank preserves the exact
    # checkpoint architecture used by older adapters and tiny tests.
    lora_rank: int = 0
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_tail_layers: int = 0

    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    abstain_token_id: int = 0

    diffusion_steps: int = 8
    diffusion_beta_start: float = 0.05
    diffusion_beta_end: float = 0.35
    num_lanes: int = 3
    max_refinement_steps: int = 8
    slow_update_every: int = 2
    min_halt_steps: int = 2
    halt_threshold: float = 0.5
    halt_compute_cost: float = 0.01
    commitment_threshold: float = 0.70
    risk_threshold: float = 0.30
    verifier_error_threshold: float = 0.35
    synthesis_prefix_tokens: int = 4

    num_experts: int = 6
    expert_top_level: int = 2
    expert_bottleneck: int = 64
    route_window_steps: int = 2
    router_sample_training: bool = True
    router_aux_weight: float = 0.01
    # KL(mean routing probabilities || uniform).  Zero preserves Version 5.4
    # behavior; Version 5.5 enables it because both 5.4 seeds collapsed to one
    # or two routes despite the Switch-style balance term.
    router_entropy_weight: float = 0.0
    # Penalizes pairwise cosine similarity between expert outputs evaluated on
    # the same probe rows.  Version 5.5 showed the marginal-entropy term
    # homogenizes experts (near-uniform soft mixing trains every expert on the
    # same gradient signal); this term applies pressure on the experts
    # themselves rather than on the routing marginal.  Zero preserves
    # Version 5.5 behavior.
    expert_diversity_weight: float = 0.0
    # Standard deviation for the expert up-projection initialization.  Zero
    # keeps the historical zero-output initialization, under which expert
    # deltas start identical (all zero) and the diversity penalty above is
    # inert until the denoise gradient happens to differentiate the experts
    # (Study 5 observed it never waking up).  A small positive scale gives
    # every expert an independent random function from the first step so the
    # penalty has something to push against.
    expert_init_scale: float = 0.0
    denoise_loss_weight: float = 1.0
    synthesis_loss_weight: float = 1.0
    brief_loss_weight: float = 0.10
    lane_diversity_weight: float = 0.05
    verification_loss_weight: float = 0.15
    commitment_loss_weight: float = 0.25
    halt_loss_weight: float = 0.05

    # Version 6.0 latent read-out memory.  Zero preserves the Version 5.x
    # synthesis interface, where the decoder reads the workspace only through
    # ``synthesis_prefix_tokens`` positions that are a linear reshape of three
    # pooled vectors.  A positive value appends, per lane, this many windowed
    # canvas hidden-state tokens (plus one summary token per lane) to the
    # synthesis prefix, so the decoder can attend to what the lanes actually
    # computed.  Hidden states cross the barrier; token identities never do.
    workspace_memory_windows: int = 0

    # Version 6.0 scalar publication rule.  When ``publish_weights`` is set,
    # publication authorizes sigmoid(w . features + b) >= publish_threshold:
    # one calibrated score fitted by logistic regression on generated
    # validation candidates, used both to select among N candidates and to
    # gate publication.  None preserves the Version 5.x triple-threshold
    # rule, whose fitted commitment threshold degenerated to a rail in six
    # separate seed-runs.  Version 8.0 extends the feature vector to four:
    # [commit_p, risk_p, verifier_p, exp(causal mean logprob)] — the fourth
    # feature nests the max-logprob baseline inside the combiner; a
    # three-weight tuple remains accepted for cross-study comparability.
    publish_weights: Optional[Tuple[float, ...]] = None
    publish_bias: float = 0.0
    publish_threshold: float = 0.5

    # Version 8.0 channel layout.  Hard response-cue token ids that the model
    # itself appends after the (optional) workspace prefix, so that decoding
    # always continues from real text tokens rather than from soft vectors or
    # a mid-sequence document boundary.  Callers set this from the tokenizer
    # (the prompt builder's trailing response header, re-tokenized); None
    # appends nothing.  The Version 6.0 answer channel began with
    # ``bos_token_id`` — which ``from_hf_config`` maps to ``eos`` because
    # Qwen3 has no BOS — so every workspace answer opened with an
    # end-of-document marker; the audit measured scaffold emissions
    # ("Human:", "Assistant:") on 12/64 rows and a ~30-point content gap
    # against the causal channel.  Version 8.0 removes that token entirely.
    response_cue_ids: Optional[Tuple[int, ...]] = None
    # Version 8.0 gated prefix: the entire workspace prefix is scaled by
    # tanh(prefix_gate), a learned scalar initialized small-positive (the
    # ControlNeXt "silent phase" contingency, pre-applied) so the channel
    # starts near-null without parking the gradient at exactly zero.
    prefix_gate_init: float = 0.05
    # Version 8.0 KL-to-causal anchor on the synthesis answer span: weight of
    # KL(workspace-conditioned answer logits || detached causal-channel
    # logits on the same targets).  Bounds register drift while the read-out
    # liveness gate guards against the anchor nulling the mechanism.
    synthesis_kl_weight: float = 0.0
    # Version 9.0 information-asymmetric channel: weight of the latent-only
    # auxiliary probe that decodes the withheld premise tokens from the
    # prefix positions (dense latent supervision; never the gold answer).
    premise_aux_weight: float = 0.0
    # Version 9.0 attribution control: token count of the gist prefix, a
    # trivial mean-pool projection of the full context at the same budget as
    # the workspace prefix. Zero disables the module.
    gist_prefix_tokens: int = 0

    # ------------------------------------------------------------------
    # Version 10.0 dense-supervision channel.  Every field defaults OFF so
    # Version 9.0 checkpoints and tests remain loadable unchanged.
    #
    # Number K of autoregressive continuous thoughts produced after the
    # (possibly premise-masked) question: the model's own last hidden state
    # through a 2-layer MLP+LayerNorm projection, re-injected as the next
    # input embedding and rescaled to the base embedding standard deviation,
    # so latents live inside the embedding distribution by construction.
    # Zero disables the channel.
    latent_thoughts: int = 0
    # Per-layer key/value slot count (m): thought states are attention-pooled
    # into m shared slot states and projected by factorized per-layer heads
    # into K/V slots appended at every attention layer.  Slots never pass
    # through RMSNorm as token embeddings, which retires the
    # scale-invariance inertness class structurally.  Zero disables.
    kv_prefix_slots: int = 0
    # Rank of the factorized per-layer K/V slot heads.
    kv_prefix_rank: int = 64
    # Initial tanh(g) of the post-softmax per-layer, per-head mixing gate on
    # the prefix attention branch.  Never zero: at g=0 the gradient into the
    # slot projector is exactly zero (the Study 9 frozen-alpha class); the
    # Study 10 red-team fixed a small-positive init and exempted the gates
    # (parameter name pattern ``prefix_attn_gate``) from weight decay.
    prefix_attn_gate_init: float = 0.08
    # Family-routed MLP experts (Version 10.0): count of routed LoRA experts
    # on the decoder down projections, selected per SEQUENCE by the gold
    # family label (deterministic; balance comes from the stratified
    # dataloader, never from a balance loss).  Zero disables.
    mlp_expert_count: int = 0
    mlp_expert_rank: int = 16
    # Task families the detached router probe classifies (numeric, unit,
    # ordering, abstention).  The probe recovers the routing decision from
    # the prompt; it never gates training-time expert selection.
    router_families: int = 4

    def __post_init__(self) -> None:
        positive = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "max_position_embeddings": self.max_position_embeddings,
            "diffusion_steps": self.diffusion_steps,
            "num_lanes": self.num_lanes,
            "max_refinement_steps": self.max_refinement_steps,
            "slow_update_every": self.slow_update_every,
            "num_experts": self.num_experts,
            "expert_top_level": self.expert_top_level,
            "expert_bottleneck": self.expert_bottleneck,
            "route_window_steps": self.route_window_steps,
            "synthesis_prefix_tokens": self.synthesis_prefix_tokens,
        }
        invalid = [name for name, value in positive.items() if int(value) <= 0]
        if invalid:
            raise ValueError("configuration values must be positive: %s" % ", ".join(invalid))
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for rotary embeddings")
        if not 0.0 < self.diffusion_beta_start <= self.diffusion_beta_end < 1.0:
            raise ValueError("diffusion beta schedule must satisfy 0 < start <= end < 1")
        for token_name in ("pad_token_id", "bos_token_id", "eos_token_id", "abstain_token_id"):
            token_id = int(getattr(self, token_name))
            if not 0 <= token_id < self.vocab_size:
                raise ValueError("%s=%d is outside the vocabulary" % (token_name, token_id))
        if self.min_halt_steps > self.max_refinement_steps:
            raise ValueError("min_halt_steps cannot exceed max_refinement_steps")
        if self.max_refinement_steps > self.diffusion_steps:
            raise ValueError("joint refinement steps cannot exceed diffusion_steps")
        if self.expert_top_level > self.num_experts:
            raise ValueError("expert_top_level cannot exceed num_experts")
        if self.lora_rank < 0 or self.lora_tail_layers < 0:
            raise ValueError("LoRA rank and tail layer count cannot be negative")
        if self.lora_tail_layers > self.num_hidden_layers:
            raise ValueError("LoRA tail layer count exceeds the decoder depth")
        if self.lora_rank and not self.lora_tail_layers:
            raise ValueError("a positive LoRA rank requires at least one tail layer")
        if self.lora_alpha <= 0 or not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("LoRA alpha/dropout are invalid")
        for name in (
            "commitment_threshold",
            "risk_threshold",
            "verifier_error_threshold",
            "publish_threshold",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError("%s must be between zero and one" % name)
        if self.workspace_memory_windows < 0:
            raise ValueError("workspace_memory_windows cannot be negative")
        if self.gist_prefix_tokens < 0:
            raise ValueError("gist_prefix_tokens cannot be negative")
        if self.premise_aux_weight < 0:
            raise ValueError("premise_aux_weight cannot be negative")
        if self.publish_weights is not None:
            self.publish_weights = tuple(float(value) for value in self.publish_weights)
            if len(self.publish_weights) not in (3, 4, 5):
                raise ValueError("publish_weights must hold three, four, or five values")
        if self.response_cue_ids is not None:
            self.response_cue_ids = tuple(int(value) for value in self.response_cue_ids)
            if not self.response_cue_ids:
                raise ValueError("response_cue_ids cannot be empty when set")
            for token_id in self.response_cue_ids:
                if not 0 <= token_id < self.vocab_size:
                    raise ValueError("response cue token %d is outside the vocabulary" % token_id)
        if self.latent_thoughts < 0:
            raise ValueError("latent_thoughts cannot be negative")
        if self.kv_prefix_slots < 0:
            raise ValueError("kv_prefix_slots cannot be negative")
        if self.kv_prefix_slots > 0 and self.kv_prefix_rank <= 0:
            raise ValueError("kv_prefix_rank must be positive when slots are enabled")
        if self.kv_prefix_slots > 0 and not 0.0 < self.prefix_attn_gate_init < 1.0:
            # Zero parks the slot-projector gradient at exactly zero (the
            # Study 9 frozen-alpha failure); one saturates tanh.
            raise ValueError("prefix_attn_gate_init must lie strictly between 0 and 1")
        if self.mlp_expert_count < 0:
            raise ValueError("mlp_expert_count cannot be negative")
        if self.mlp_expert_count > 0 and self.mlp_expert_rank <= 0:
            raise ValueError("mlp_expert_rank must be positive when experts are enabled")
        if self.mlp_expert_count > 0 and self.router_families < self.mlp_expert_count:
            raise ValueError("router_families cannot be fewer than the expert count")

    @classmethod
    def tiny(cls, **overrides: Any) -> "HLWMConfig":
        """Return a small, offline-testable Qwen-like configuration."""

        values: Dict[str, Any] = {
            "vocab_size": 41,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 96,
            "rope_theta": 10_000.0,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "abstain_token_id": 0,
            "diffusion_steps": 3,
            "diffusion_beta_start": 0.10,
            "diffusion_beta_end": 0.35,
            "num_lanes": 2,
            "max_refinement_steps": 3,
            "slow_update_every": 1,
            "min_halt_steps": 2,
            "num_experts": 4,
            "expert_top_level": 2,
            "expert_bottleneck": 12,
            "synthesis_prefix_tokens": 2,
            "route_window_steps": 2,
            "router_sample_training": False,
        }
        values.update(overrides)
        return cls(**values)

    @classmethod
    def from_hf_config(cls, hf_config: Any, **overrides: Any) -> "HLWMConfig":
        """Create an HLWM config from a Qwen-style Hugging Face config."""

        hidden_size = int(hf_config.hidden_size)
        attention_heads = int(hf_config.num_attention_heads)
        head_dim = int(getattr(hf_config, "head_dim", hidden_size // attention_heads))
        eos = getattr(hf_config, "eos_token_id", 2)
        if isinstance(eos, (list, tuple)):
            eos = eos[0]
        pad = getattr(hf_config, "pad_token_id", None)
        if pad is None:
            pad = int(eos)
        bos = getattr(hf_config, "bos_token_id", None)
        if bos is None:
            bos = int(eos)
        values: Dict[str, Any] = {
            "vocab_size": int(hf_config.vocab_size),
            "hidden_size": hidden_size,
            "intermediate_size": int(hf_config.intermediate_size),
            "num_hidden_layers": int(hf_config.num_hidden_layers),
            "num_attention_heads": attention_heads,
            "num_key_value_heads": int(getattr(hf_config, "num_key_value_heads", attention_heads)),
            "head_dim": head_dim,
            "max_position_embeddings": int(getattr(hf_config, "max_position_embeddings", 32_768)),
            "rope_theta": float(getattr(hf_config, "rope_theta", 10_000.0)),
            "rms_norm_eps": float(getattr(hf_config, "rms_norm_eps", 1.0e-6)),
            "attention_bias": bool(getattr(hf_config, "attention_bias", False)),
            "mlp_bias": bool(getattr(hf_config, "mlp_bias", False)),
            "tie_word_embeddings": bool(getattr(hf_config, "tie_word_embeddings", False)),
            "attention_dropout": float(getattr(hf_config, "attention_dropout", 0.0)),
            "pad_token_id": int(pad),
            "bos_token_id": int(bos),
            "eos_token_id": int(eos),
            "abstain_token_id": int(pad),
        }
        values.update(overrides)
        return cls(**values)


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, hidden: Tensor) -> Tensor:
        source_dtype = hidden.dtype
        hidden_float = hidden.float()
        variance = hidden_float.square().mean(dim=-1, keepdim=True)
        normalized = hidden_float * torch.rsqrt(variance + self.eps)
        return (normalized * self.weight.float()).to(source_dtype)


def _rotate_half(hidden: Tensor) -> Tensor:
    first, second = hidden.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        inv_frequency = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_frequency", inv_frequency, persistent=False)

    def forward(self, position_ids: Tensor, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
        frequencies = torch.einsum(
            "bl,d->bld", position_ids.float(), self.inv_frequency.float()
        )
        doubled = torch.cat((frequencies, frequencies), dim=-1).unsqueeze(1)
        return doubled.cos().to(dtype), doubled.sin().to(dtype)


class QwenSelfAttention(nn.Module):
    def __init__(self, config: HLWMConfig, *, enable_lora: bool = False) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.groups = self.num_heads // self.num_key_value_heads
        self.dropout = config.attention_dropout
        q_width = self.num_heads * self.head_dim
        kv_width = self.num_key_value_heads * self.head_dim
        self.q_proj = nn.Linear(config.hidden_size, q_width, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, kv_width, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, kv_width, bias=config.attention_bias)
        self.o_proj = nn.Linear(q_width, config.hidden_size, bias=config.attention_bias)
        rank = config.lora_rank if enable_lora else 0
        self.q_lora = LoRAResidual(config.hidden_size, q_width, rank, config.lora_alpha, config.lora_dropout)
        self.k_lora = LoRAResidual(config.hidden_size, kv_width, rank, config.lora_alpha, config.lora_dropout)
        self.v_lora = LoRAResidual(config.hidden_size, kv_width, rank, config.lora_alpha, config.lora_dropout)
        self.o_lora = LoRAResidual(q_width, config.hidden_size, rank, config.lora_alpha, config.lora_dropout)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps) if config.use_qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps) if config.use_qk_norm else None
        self.rotary = RotaryEmbedding(self.head_dim, config.rope_theta)
        if config.kv_prefix_slots > 0:
            # Version 10.0 post-softmax mixing gate on the prefix branch, one
            # scalar per query head.  It sits AFTER both softmaxes, so no
            # normalization upstream can zero its gradient (the Study 9
            # frozen-alpha class), and it is initialized small-positive so
            # the slot projector receives gradient from step 0.  The trainer
            # exempts parameters matching ``prefix_attn_gate`` from weight
            # decay: decay would otherwise close the gate at a constant rate
            # with no opposing gradient.
            self.prefix_attn_gate: Optional[nn.Parameter] = nn.Parameter(
                torch.full(
                    (self.num_heads,),
                    math.atanh(float(config.prefix_attn_gate_init)),
                )
            )
        else:
            self.prefix_attn_gate = None

    @staticmethod
    def _allowed_pattern(
        length: int,
        mode: str,
        context_length: Optional[int],
        device: torch.device,
    ) -> Tensor:
        if mode == "bidirectional":
            return torch.ones(length, length, dtype=torch.bool, device=device)
        if mode == "causal":
            return torch.ones(length, length, dtype=torch.bool, device=device).tril()
        if mode != "protected":
            raise ValueError("attention mode must be causal, bidirectional, or protected")
        if context_length is None or not 0 < context_length < length:
            raise ValueError("protected attention requires 0 < context_length < sequence length")
        # Committed context remains causal and cannot read private state.  Every
        # private query may read the full committed context and its own private
        # block bidirectionally.
        allowed = torch.zeros(length, length, dtype=torch.bool, device=device)
        allowed[:context_length, :context_length] = torch.ones(
            context_length, context_length, dtype=torch.bool, device=device
        ).tril()
        allowed[context_length:, :] = True
        return allowed

    def forward(
        self,
        hidden: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
        *,
        attention_mode: str,
        context_length: Optional[int] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
        prefix_kv: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tuple[Tensor, Tensor]]]:
        batch, length, _ = hidden.shape
        query_projection = self.q_proj(hidden)
        key_projection = self.k_proj(hidden)
        value_projection = self.v_proj(hidden)
        if self.q_lora.enabled:
            query_projection = query_projection + self.q_lora(hidden).to(query_projection.dtype)
            key_projection = key_projection + self.k_lora(hidden).to(key_projection.dtype)
            value_projection = value_projection + self.v_lora(hidden).to(value_projection.dtype)
        query = query_projection.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key_projection.view(batch, length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = value_projection.view(batch, length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            query = self.q_norm(query)
            key = self.k_norm(key)  # type: ignore[operator]
        cosine, sine = self.rotary(position_ids, query.dtype)
        query = query * cosine + _rotate_half(query) * sine
        key = key * cosine + _rotate_half(key) * sine
        # Keys are cached post-RoPE (absolute position ids are supplied by
        # the caller), values as-is, both before grouped-head expansion.
        past_length = 0
        if past_key_value is not None:
            if attention_mode != "causal":
                raise ValueError("KV caching supports causal attention only")
            past_length = past_key_value[0].shape[2]
            key = torch.cat((past_key_value[0], key), dim=2)
            value = torch.cat((past_key_value[1], value), dim=2)
        present: Optional[Tuple[Tensor, Tensor]] = (key, value) if use_cache else None
        if self.groups > 1:
            key = key.repeat_interleave(self.groups, dim=1)
            value = value.repeat_interleave(self.groups, dim=1)

        total_length = past_length + length
        if past_length > 0:
            # Incremental causal step: query i may attend every cached key
            # plus new keys up to its own position.
            key_positions = torch.arange(total_length, device=hidden.device)
            query_positions = past_length + torch.arange(length, device=hidden.device)
            allowed = key_positions[None, :] <= query_positions[:, None]
        else:
            allowed = self._allowed_pattern(
                length, attention_mode, context_length, hidden.device
            )
        allowed = allowed.view(1, 1, length, total_length)
        key_valid = attention_mask.to(torch.bool).view(batch, 1, 1, total_length)
        allowed = allowed & key_valid
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            # A boolean mask avoids finite-minimum FP16 masks and gives the
            # fused kernel an exact representation of forbidden keys.
            attn_mask=allowed,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        if prefix_kv is not None:
            if self.prefix_attn_gate is None:
                raise ValueError("prefix_kv supplied but kv_prefix_slots is zero")
            prefix_key, prefix_value = prefix_kv
            if prefix_key.shape[1] != self.num_key_value_heads or prefix_key.shape[-1] != self.head_dim:
                raise ValueError("prefix slots must be shaped [batch, kv_heads, slots, head_dim]")
            if self.k_norm is not None:
                # Score-scale compatibility with the QK-normed word branch.
                # The gate sits after the softmax, so this normalization
                # cannot silence it (unlike the Study 5-9 embedding gates).
                prefix_key = self.k_norm(prefix_key)
            if self.groups > 1:
                prefix_key = prefix_key.repeat_interleave(self.groups, dim=1)
                prefix_value = prefix_value.repeat_interleave(self.groups, dim=1)
            # Slots are position-free (no RoPE) and always visible to every
            # query; the word branch keeps its exact baseline computation, so
            # forcing the gate to zero recovers the no-prefix model
            # identically (the gate-closed equivalence preflight relies on
            # this).
            prefix_attended = F.scaled_dot_product_attention(
                query,
                prefix_key.to(query.dtype),
                prefix_value.to(query.dtype),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
            gate = torch.tanh(self.prefix_attn_gate.float()).to(attended.dtype)
            attended = attended + gate.view(1, -1, 1, 1) * prefix_attended
        attended = attended.transpose(1, 2).reshape(batch, length, -1)
        projected = self.o_proj(attended)
        if self.o_lora.enabled:
            projected = projected + self.o_lora(attended).to(projected.dtype)
        attended = projected
        query_valid = attention_mask[:, past_length:]
        attended = attended * query_valid.unsqueeze(-1).to(attended.dtype)
        if use_cache:
            return attended, present  # type: ignore[return-value]
        return attended


class LoRAResidual(nn.Module):
    """Zero-output low-rank sidecar whose base projection stays untouched.

    ``nonzero_init`` (Version 10.0, routed experts only) initializes the up
    projection small-normal instead of zero: a family-routed expert that
    starts at exactly zero loses the gradient race to the always-on shared
    path and can stay at zero all run (the v5.6.2 pattern), so routed
    experts start as small independent functions instead.
    """

    def __init__(
        self,
        input_width: int,
        output_width: int,
        rank: int,
        alpha: float,
        dropout: float,
        *,
        nonzero_init: bool = False,
    ) -> None:
        super().__init__()
        self.enabled = rank > 0
        self.scale = float(alpha) / max(1, rank)
        self.dropout = nn.Dropout(dropout)
        if self.enabled:
            self.down = nn.Linear(input_width, rank, bias=False)
            self.up = nn.Linear(rank, output_width, bias=False)
            nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
            if nonzero_init:
                nn.init.normal_(self.up.weight, std=0.02)
            else:
                nn.init.zeros_(self.up.weight)
        else:
            self.down = None
            self.up = None

    def forward(self, hidden: Tensor) -> Tensor:
        if not self.enabled or self.down is None or self.up is None:
            raise RuntimeError("disabled LoRA sidecar cannot execute")
        working = self.dropout(hidden).to(self.down.weight.dtype)
        return self.up(self.down(working)) * self.scale


class QwenMLP(nn.Module):
    def __init__(self, config: HLWMConfig, *, enable_lora: bool = False) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.mlp_bias
        )
        rank = config.lora_rank if enable_lora else 0
        self.gate_lora = LoRAResidual(config.hidden_size, config.intermediate_size, rank, config.lora_alpha, config.lora_dropout)
        self.up_lora = LoRAResidual(config.hidden_size, config.intermediate_size, rank, config.lora_alpha, config.lora_dropout)
        self.down_lora = LoRAResidual(config.intermediate_size, config.hidden_size, rank, config.lora_alpha, config.lora_dropout)
        if config.mlp_expert_count > 0 and enable_lora:
            # Version 10.0 family-routed experts: sequence-level deterministic
            # routing from gold family labels, so collapse is impossible by
            # construction (nothing about the assignment is learned during
            # training).  Experts live on the down projection only and start
            # as small independent functions (nonzero init), because a
            # zero-output expert loses the gradient race to the always-on
            # shared LoRA and can stay silent all run.
            self.mlp_experts = nn.ModuleList(
                LoRAResidual(
                    config.intermediate_size,
                    config.hidden_size,
                    config.mlp_expert_rank,
                    config.lora_alpha,
                    config.lora_dropout,
                    nonzero_init=True,
                )
                for _ in range(config.mlp_expert_count)
            )
        else:
            self.mlp_experts = None

    def forward(self, hidden: Tensor, route_index: Optional[Tensor] = None) -> Tensor:
        gate = self.gate_proj(hidden)
        up = self.up_proj(hidden)
        if self.gate_lora.enabled:
            gate = gate + self.gate_lora(hidden).to(gate.dtype)
            up = up + self.up_lora(hidden).to(up.dtype)
        activated = F.silu(gate) * up
        output = self.down_proj(activated)
        if self.down_lora.enabled:
            output = output + self.down_lora(activated).to(output.dtype)
        if self.mlp_experts is not None and route_index is not None:
            if route_index.shape[0] != hidden.shape[0]:
                raise ValueError("route_index must supply one expert id per row")
            for expert_id, expert in enumerate(self.mlp_experts):
                selected = route_index == expert_id
                if not bool(selected.any()):
                    continue
                mask = selected.view(-1, 1, 1).to(output.dtype)
                output = output + mask * expert(activated).to(output.dtype)
        return output


class QwenDecoderLayer(nn.Module):
    def __init__(self, config: HLWMConfig, *, enable_lora: bool = False) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = QwenSelfAttention(config, enable_lora=enable_lora)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = QwenMLP(config, enable_lora=enable_lora)

    def forward(
        self,
        hidden: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
        *,
        attention_mode: str,
        context_length: Optional[int] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
        prefix_kv: Optional[Tuple[Tensor, Tensor]] = None,
        route_index: Optional[Tensor] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tuple[Tensor, Tensor]]]:
        residual = hidden
        attended = self.self_attn(
            self.input_layernorm(hidden),
            attention_mask,
            position_ids,
            attention_mode=attention_mode,
            context_length=context_length,
            past_key_value=past_key_value,
            use_cache=use_cache,
            prefix_kv=prefix_kv,
        )
        present: Optional[Tuple[Tensor, Tensor]] = None
        if use_cache:
            attended, present = attended  # type: ignore[misc]
        hidden = residual + attended
        hidden = hidden + self.mlp(
            self.post_attention_layernorm(hidden), route_index=route_index
        )
        if use_cache:
            return hidden, present  # type: ignore[return-value]
        return hidden


class QwenLikeBackbone(nn.Module):
    """Minimal Qwen-compatible decoder with switchable private attention."""

    def __init__(self, config: HLWMConfig) -> None:
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        lora_start = config.num_hidden_layers - config.lora_tail_layers
        self.layers = nn.ModuleList(
            QwenDecoderLayer(config, enable_lora=index >= lora_start)
            for index in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def gradient_checkpointing_enable(self) -> None:
        """Recompute decoder layers during backward to fit a single T4."""

        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    def forward(
        self,
        *,
        input_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mode: str = "causal",
        context_length: Optional[int] = None,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        use_cache: bool = False,
        prefix_kv: Optional[List[Tuple[Tensor, Tensor]]] = None,
        collect_hidden_states: bool = False,
        route_index: Optional[Tensor] = None,
    ) -> Union[Tensor, Tuple[Tensor, List[Tuple[Tensor, Tensor]]], Tuple[Tensor, List[Tensor]]]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        if prefix_kv is not None and len(prefix_kv) != len(self.layers):
            raise ValueError("prefix_kv must supply one (key, value) pair per layer")
        if collect_hidden_states and use_cache:
            raise ValueError("hidden-state collection is a training-time facility only")
        hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if hidden.dtype != self.embed_tokens.weight.dtype:
            # Boundary normalization for the no-autocast eval paths: fp32
            # trainables (mode/positional/workspace pieces) can promote an
            # assembled embedding, and a missed cast in any caller must not
            # reach a frozen reduced-precision linear (session I-2/I-5
            # dtype class). Under autocast this is a no-op.
            hidden = hidden.to(self.embed_tokens.weight.dtype)
        batch, length, _ = hidden.shape
        past_length = past_key_values[0][0].shape[2] if past_key_values else 0
        if past_length + length > self.config.max_position_embeddings:
            raise ValueError("sequence exceeds max_position_embeddings")
        if attention_mask is None:
            # With a cache the mask must cover past plus new positions.
            attention_mask = torch.ones(
                batch, past_length + length, dtype=torch.long, device=hidden.device
            )
        if attention_mask.shape[1] != past_length + length:
            raise ValueError("attention_mask must cover cached and new positions")
        query_mask = attention_mask[:, past_length:].to(hidden.dtype).unsqueeze(-1)
        hidden = hidden * query_mask
        if position_ids is None:
            position_ids = (
                past_length + torch.arange(length, device=hidden.device)
            ).unsqueeze(0).expand(batch, -1)
        presents: List[Tuple[Tensor, Tensor]] = []
        layer_hiddens: List[Tensor] = []
        for index, layer in enumerate(self.layers):
            layer_prefix = prefix_kv[index] if prefix_kv is not None else None
            if self.gradient_checkpointing and self.training and hidden.requires_grad:
                if use_cache:
                    raise ValueError("KV caching is a decode-time facility only")
                from torch.utils.checkpoint import checkpoint

                def run_layer(
                    value: Tensor,
                    current_layer: nn.Module = layer,
                    current_prefix: Optional[Tuple[Tensor, Tensor]] = layer_prefix,
                ) -> Tensor:
                    return current_layer(
                        value,
                        attention_mask,
                        position_ids,
                        attention_mode=attention_mode,
                        context_length=context_length,
                        prefix_kv=current_prefix,
                        route_index=route_index,
                    )

                hidden = checkpoint(run_layer, hidden, use_reentrant=False)
            else:
                hidden = layer(
                    hidden,
                    attention_mask,
                    position_ids,
                    attention_mode=attention_mode,
                    context_length=context_length,
                    past_key_value=past_key_values[index] if past_key_values else None,
                    use_cache=use_cache,
                    prefix_kv=layer_prefix,
                    route_index=route_index,
                )
                if use_cache:
                    hidden, present = hidden  # type: ignore[misc]
                    presents.append(present)
            # Padding must remain a finite zero state.  This prevents invalid
            # padded queries from entering later decoder layers or masked loss.
            hidden = hidden * query_mask
            if collect_hidden_states:
                layer_hiddens.append(hidden)
        hidden = self.norm(hidden)
        if use_cache:
            return hidden, presents
        if collect_hidden_states:
            return hidden, layer_hiddens
        return hidden


class ResidualAdapter(nn.Module):
    """Small SwiGLU-free residual adapter, zero-output at initialization."""

    def __init__(self, width: int, bottleneck: int) -> None:
        super().__init__()
        self.down = nn.Linear(width, bottleneck)
        self.up = nn.Linear(bottleneck, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden: Tensor) -> Tensor:
        # Dtype-tolerant like LoRAResidual: naked (non-autocast) forwards on
        # a bf16 base with fp32 trainables must not crash — Session I-2's
        # preflight caught exactly this class on the first CUDA forward.
        working = hidden.to(self.down.weight.dtype)
        return self.up(F.silu(self.down(working))).to(hidden.dtype)


class RoutedAdapterBank(nn.Module):
    """Sparse connected two-level adapter graph.

    The first ``expert_top_level`` nodes are children of the permanent root.
    Every remaining node has one deterministic top-level parent.  Selecting a
    leaf therefore executes a root-connected path rather than a decorative flat
    expert.  Unselected paths perform no expert forward pass.  Selection can be
    sampled during training for exploration and is deterministic at inference.
    """

    def __init__(self, config: HLWMConfig) -> None:
        super().__init__()
        self.router = nn.Linear(config.hidden_size, config.num_experts)
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.sample_training = bool(config.router_sample_training)
        parents = [-1] * config.expert_top_level
        parents.extend(
            index % config.expert_top_level
            for index in range(config.num_experts - config.expert_top_level)
        )
        self.register_buffer(
            "parent_ids", torch.tensor(parents, dtype=torch.long), persistent=True
        )
        self.experts = nn.ModuleList(
            ResidualAdapter(config.hidden_size, config.expert_bottleneck)
            for _ in range(config.num_experts)
        )
        if getattr(config, "expert_init_scale", 0.0) > 0:
            for expert in self.experts:
                nn.init.normal_(expert.up.weight, std=float(config.expert_init_scale))

    def forward(
        self,
        hidden: Tensor,
        routing_context: Tensor,
        selected_override: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        probabilities = self.router(routing_context).softmax(dim=-1)
        if selected_override is not None:
            selected = selected_override.to(device=hidden.device, dtype=torch.long)
            if selected.shape != (hidden.shape[0],):
                raise ValueError("selected_override must have shape [batch]")
        elif self.training and self.sample_training:
            selected = torch.multinomial(probabilities, 1).squeeze(1)
        else:
            selected = probabilities.argmax(dim=-1)
        selected_probability = probabilities.gather(1, selected.unsqueeze(1)).squeeze(1)
        routed = torch.zeros_like(hidden)
        path_mask = torch.zeros(
            hidden.shape[0], len(self.experts), dtype=torch.bool, device=hidden.device
        )
        normalized = self.norm(hidden)
        for expert_index, expert in enumerate(self.experts):
            rows = torch.nonzero(selected == expert_index, as_tuple=False).flatten()
            if rows.numel() == 0:
                continue
            path = []
            parent = int(self.parent_ids[expert_index].item())
            if parent >= 0:
                path.append(parent)
            path.append(expert_index)
            working = normalized.index_select(0, rows)
            expert_hidden = torch.zeros_like(working)
            for node_index in path:
                node_delta = self.experts[node_index](self.norm(working))
                working = working + node_delta
                expert_hidden = expert_hidden + node_delta
                path_mask[rows, node_index] = True
            gate = selected_probability.index_select(0, rows)
            straight_through_gate = gate / gate.detach().clamp_min(1.0e-6)
            expert_hidden = expert_hidden * straight_through_gate[:, None, None]
            # CUDA autocast may evaluate softmax/router probabilities in FP32
            # while the routed activation buffer remains FP16.  Preserve the
            # activation dtype before the indexed write; the cast still keeps
            # gradients flowing through the expert and straight-through gate.
            expert_hidden = expert_hidden.to(dtype=routed.dtype)
            routed = routed.index_copy(0, rows, expert_hidden)
        return hidden + routed, selected, probabilities, path_mask

    def output_diversity(self, probe: Tensor, max_rows: int = 8) -> Tensor:
        """Mean pairwise squared cosine between expert outputs on shared rows.

        Every expert adapter is evaluated on the same normalized probe rows,
        so the value is differentiable with respect to every expert regardless
        of routing traffic.  Squared cosine targets near-orthogonal expert
        functions instead of rewarding a degenerate anti-correlation.
        """

        if len(self.experts) < 2:
            # A single always-active adapter has no sibling to diverge from;
            # an empty pairwise mean would be NaN and poison the joint loss.
            return probe.new_zeros(())
        if probe.ndim == 3:
            probe = probe.mean(dim=1)
        rows = probe[: max(1, int(max_rows))]
        normalized_probe = self.norm(rows.float())
        deltas = torch.stack(
            [expert(normalized_probe).float() for expert in self.experts], dim=0
        )
        directions = F.normalize(deltas.reshape(len(self.experts), -1), dim=-1)
        similarities = directions @ directions.t()
        off_diagonal = ~torch.eye(
            len(self.experts), dtype=torch.bool, device=probe.device
        )
        return similarities.masked_select(off_diagonal).square().mean()


class ScalarConditioner(nn.Module):
    """Sinusoidal scalar embedding followed by a learned projection."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.projection = nn.Sequential(
            nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width)
        )

    def forward(self, values: Tensor) -> Tensor:
        half = self.width // 2
        if half == 1:
            frequencies = torch.ones(1, device=values.device, dtype=torch.float32)
        else:
            frequencies = torch.exp(
                -math.log(10_000.0)
                * torch.arange(half, device=values.device, dtype=torch.float32)
                / (half - 1)
            )
        angles = values.float().unsqueeze(-1) * frequencies
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if embedding.shape[-1] < self.width:
            embedding = F.pad(embedding, (0, self.width - embedding.shape[-1]))
        return self.projection(embedding.to(self.projection[0].weight.dtype))


class CategoricalDiffusion(nn.Module):
    r"""Multinomial diffusion with ``Q_t=(1-beta_t)I+beta_t 1 nu^T``.

    ``nu`` is uniform by default.  This keeps every transition probability
    positive and makes the exact x0-parameterized posterior numerically stable
    without allocating a ``vocab_size x vocab_size`` transition matrix.
    """

    def __init__(
        self,
        vocab_size: int,
        steps: int,
        beta_start: float,
        beta_end: float,
        noise_probabilities: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        beta = torch.zeros(steps + 1, dtype=torch.float32)
        beta[1:] = torch.linspace(beta_start, beta_end, steps)
        alpha = 1.0 - beta
        alpha_bar = torch.ones_like(alpha)
        alpha_bar[1:] = torch.cumprod(alpha[1:], dim=0)
        if noise_probabilities is None:
            noise_probabilities = torch.full((vocab_size,), 1.0 / vocab_size)
        noise_probabilities = noise_probabilities.float()
        if noise_probabilities.shape != (vocab_size,):
            raise ValueError("noise probabilities must have shape [vocab_size]")
        if torch.any(noise_probabilities <= 0):
            raise ValueError("all categorical noise probabilities must be positive")
        noise_probabilities = noise_probabilities / noise_probabilities.sum()
        self.steps = steps
        self.vocab_size = vocab_size
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("noise_probabilities", noise_probabilities)

    def training_timestep_probabilities(self) -> Tensor:
        """Return equal-corruption-mass weights for timesteps 1..T.

        For the multinomial process, ``alpha_bar[t-1]-alpha_bar[t]`` is the
        newly corrupted clean-token mass assigned to interval t.  Sampling by
        this mass is the categorical analogue of the equal-training-mass
        partition used by DiffusionBlocks; it avoids copying Gaussian
        log-normal boundaries into a discrete process.
        """

        mass = (self.alpha_bar[:-1] - self.alpha_bar[1:]).clamp_min(0.0)
        return mass / mass.sum().clamp_min(1.0e-12)

    def sample_training_timesteps(
        self,
        batch: int,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        if batch <= 0:
            raise ValueError("batch must be positive")
        sampled = torch.multinomial(
            self.training_timestep_probabilities().to(device),
            batch,
            replacement=True,
            generator=generator,
        )
        return sampled + 1

    def _validate_timesteps(self, timesteps: Tensor, batch: int) -> Tensor:
        timesteps = timesteps.to(dtype=torch.long)
        if timesteps.shape != (batch,):
            raise ValueError("timesteps must have shape [batch]")
        if bool(((timesteps < 0) | (timesteps > self.steps)).any()):
            raise ValueError("diffusion timestep is outside the configured schedule")
        return timesteps

    def sample_noise(
        self, shape: Sequence[int], device: torch.device, generator: Optional[torch.Generator] = None
    ) -> Tensor:
        samples = torch.multinomial(
            self.noise_probabilities.to(device),
            int(math.prod(shape)),
            replacement=True,
            generator=generator,
        )
        return samples.reshape(*shape)

    def q_sample(
        self, clean_ids: Tensor, timesteps: Tensor, generator: Optional[torch.Generator] = None
    ) -> Tensor:
        timesteps = self._validate_timesteps(timesteps, clean_ids.shape[0])
        keep_probability = self.alpha_bar[timesteps].to(clean_ids.device)
        keep = torch.rand(
            clean_ids.shape, device=clean_ids.device, generator=generator
        ) < keep_probability[:, None]
        noise = self.sample_noise(clean_ids.shape, clean_ids.device, generator)
        return torch.where(keep, clean_ids, noise)

    def posterior_probabilities(
        self, noisy_ids: Tensor, predicted_clean_probabilities: Tensor, timesteps: Tensor
    ) -> Tensor:
        """Compute exact ``p_theta(x_{t-1}|x_t)`` under x0 parameterization.

        This is the D3PM posterior marginalized over the model's predicted x0
        distribution.  The closed form exploits the rank-one categorical noise
        kernel and therefore uses O(vocabulary) rather than O(vocabulary^2)
        memory per token.
        """

        batch, length = noisy_ids.shape
        timesteps = self._validate_timesteps(timesteps, batch)
        if predicted_clean_probabilities.shape != (batch, length, self.vocab_size):
            raise ValueError("predicted clean probabilities have the wrong shape")
        if bool((timesteps == 0).any()):
            if bool((timesteps != 0).any()):
                raise ValueError("mixed zero and nonzero posterior timesteps are unsupported")
            return predicted_clean_probabilities

        probabilities = predicted_clean_probabilities.float()
        device = probabilities.device
        beta_t = self.beta[timesteps].to(device)[:, None, None]
        alpha_t = self.alpha[timesteps].to(device)[:, None, None]
        alpha_bar_t = self.alpha_bar[timesteps].to(device)[:, None, None]
        alpha_bar_previous = self.alpha_bar[timesteps - 1].to(device)[:, None, None]
        noise = self.noise_probabilities.to(device)
        observed_noise_probability = noise[noisy_ids].unsqueeze(-1)

        # q_bar_t(x_t | x0=i), for every possible clean i.
        denominator = ((1.0 - alpha_bar_t) * observed_noise_probability).expand(
            -1, -1, self.vocab_size
        ).clone()
        denominator.scatter_add_(
            -1,
            noisy_ids.unsqueeze(-1),
            alpha_bar_t.expand(batch, length, 1),
        )
        weighted_clean = probabilities / denominator.clamp_min(1.0e-20)
        previous_marginal = (
            alpha_bar_previous * weighted_clean
            + (1.0 - alpha_bar_previous)
            * noise.view(1, 1, -1)
            * weighted_clean.sum(dim=-1, keepdim=True)
        )

        # Q_t[k, observed] as a function of the candidate previous token k.
        likelihood = (beta_t * observed_noise_probability).expand(
            -1, -1, self.vocab_size
        ).clone()
        likelihood.scatter_add_(
            -1, noisy_ids.unsqueeze(-1), alpha_t.expand(batch, length, 1)
        )
        posterior = likelihood * previous_marginal
        posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(1.0e-20)
        return posterior.to(predicted_clean_probabilities.dtype)

    def reverse_step(
        self,
        noisy_ids: Tensor,
        predicted_clean_probabilities: Tensor,
        timesteps: Tensor,
        *,
        sample: bool,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[Tensor, Tensor]:
        posterior = self.posterior_probabilities(
            noisy_ids, predicted_clean_probabilities, timesteps
        )
        if sample:
            ids = torch.multinomial(
                posterior.detach().reshape(-1, self.vocab_size),
                1,
                generator=generator,
            ).reshape(noisy_ids.shape)
            state = F.one_hot(ids, self.vocab_size).to(posterior.dtype)
        else:
            # A differentiable mean-field state is used for training.  The next
            # exact posterior observes its argmax token; sampled evaluation can
            # request a discrete reverse chain with ``sample=True``.
            ids = posterior.argmax(dim=-1)
            state = posterior
        return ids, state


@dataclass
class HLWMOutput:
    loss: Optional[Tensor]
    loss_components: Dict[str, Tensor]
    denoise_logits: Tensor
    synthesis_logits: Optional[Tensor]
    verification_logits: Tensor
    commitment_logits: Tensor
    negative_commitment_logits: Optional[Tensor]
    halt_logits: Tensor
    route_indices: Tensor
    router_probabilities: Tensor
    route_path_masks: Tensor
    lane_summaries: Tensor
    fast_state: Tensor
    slow_state: Tensor
    global_state: Tensor
    final_canvas_probabilities: Tensor
    corrupted_ids: Tensor
    timesteps: Tensor
    reverse_timesteps: Tensor
    commit_mask: Tensor
    committed_ids: Tensor
    workspace_prefix: Tensor


@dataclass
class HLWMLocalDenoiseOutput:
    loss: Tensor
    denoise_loss: Tensor
    brief_loss: Tensor
    router_loss: Tensor
    router_entropy_loss: Tensor
    expert_diversity_loss: Tensor
    lane_diversity_loss: Tensor
    denoise_logits: Tensor
    corrupted_ids: Tensor
    timesteps: Tensor
    route_indices: Tensor
    route_path_masks: Tensor
    transition_count: int


@dataclass
class HLWMGeneration:
    """Externally safe generation result plus auditable private diagnostics."""

    output_ids: Tensor
    candidate_ids: Tensor
    decision: str
    commit_probability: float
    risk_probability: float
    verifier_error_probability: float
    private_verifier_error_probability: float
    macrocycles: int
    workspace: HLWMOutput


@dataclass
class HLWMNBestGeneration:
    """Verified fan-in result: N candidates from one workspace, one published.

    The workspace runs once; each candidate is decoded from the same latent
    read-out at its own temperature, scored by the candidate-level policy
    heads, and the highest calibrated publish score is published if and only
    if it clears the fitted publication rule.  Everything else stays private.
    """

    output_ids: Tensor
    selected_index: int
    decision: str
    candidate_ids: List[Tensor]
    candidate_features: Optional[List[Tensor]]
    commit_probabilities: List[float]
    risk_probabilities: List[float]
    verifier_error_probabilities: List[float]
    publish_scores: List[float]
    candidate_mean_logprobs: List[float]
    temperatures: List[float]
    private_verifier_error_probability: float
    macrocycles: int
    workspace: HLWMOutput
    candidate_agreements: Optional[List[float]] = None


def _masked_token_cross_entropy(
    logits: Tensor, targets: Tensor, attention_mask: Tensor
) -> Tensor:
    """Return per-token FP32 CE while never evaluating padded vocabulary rows."""

    if logits.shape[:-1] != targets.shape or targets.shape != attention_mask.shape:
        raise ValueError("token logits, targets and attention mask do not align")
    flat_mask = attention_mask.reshape(-1).to(torch.bool)
    if not bool(flat_mask.any()):
        raise ValueError("token loss needs at least one supervised position")
    vocabulary = logits.shape[-1]
    active_logits = logits.reshape(-1, vocabulary)[flat_mask].float()
    active_targets = targets.reshape(-1)[flat_mask]
    active_losses = F.cross_entropy(active_logits, active_targets, reduction="none")
    return torch.zeros(
        targets.numel(), device=logits.device, dtype=torch.float32
    ).masked_scatter(flat_mask, active_losses).reshape_as(targets)


class KVPrefixProjector(nn.Module):
    """Version 10.0 interface: latent thoughts to per-layer K/V slots.

    The thought states are attention-pooled into ``kv_prefix_slots`` shared
    slot states by learned queries, passed through a small trunk, then
    projected by factorized per-layer heads (shared down-projection, batched
    per-layer up-projections of rank ``kv_prefix_rank``) into key and value
    slots for every decoder layer.  Slots are appended directly to each
    layer's attention K/V, so they never pass through RMSNorm as token
    embeddings: the Study 5--9 scale-invariance inertness class cannot recur
    on this path.  Every projection is small-normal initialized (never zero;
    zero-output initialization made auxiliary gradients provably inert in
    Studies 5--8) and the near-null start lives in the post-softmax
    per-layer, per-head attention gates instead.
    """

    def __init__(self, config: HLWMConfig) -> None:
        super().__init__()
        width = config.hidden_size
        rank = config.kv_prefix_rank
        self.num_layers = config.num_hidden_layers
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.slots = config.kv_prefix_slots
        kv_width = self.num_key_value_heads * self.head_dim
        self.slot_queries = nn.Parameter(torch.randn(self.slots, width) * 0.02)
        self.trunk = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.down = nn.Linear(width, rank, bias=False)
        # [layers, 2 (key/value), rank, kv_width], one factorized head pair
        # per layer, applied with a single einsum.
        self.up = nn.Parameter(
            torch.randn(self.num_layers, 2, rank, kv_width) * 0.02
        )
        nn.init.normal_(self.down.weight, std=0.02)

    def forward(self, thought_states: Tensor) -> List[Tuple[Tensor, Tensor]]:
        if thought_states.ndim != 3:
            raise ValueError("thought states must be [batch, thoughts, width]")
        batch = thought_states.shape[0]
        states = thought_states.float()
        scores = torch.einsum(
            "sw,bkw->bsk", self.slot_queries.float(), states
        ) / float(states.shape[-1]) ** 0.5
        pooled = torch.einsum("bsk,bkw->bsw", scores.softmax(dim=-1), states)
        slot_states = self.down(self.trunk(pooled))
        projected = torch.einsum("bsr,ltrw->bltsw", slot_states, self.up.float())
        projected = projected.view(
            batch,
            self.num_layers,
            2,
            self.slots,
            self.num_key_value_heads,
            self.head_dim,
        ).permute(1, 2, 0, 4, 3, 5)
        dtype = thought_states.dtype
        return [
            (projected[layer, 0].to(dtype), projected[layer, 1].to(dtype))
            for layer in range(self.num_layers)
        ]


class HLWMForConditionalGeneration(nn.Module):
    """Single-model causal and private-diffusion HLWM prototype."""

    MODE_CAUSAL = 0
    MODE_PRIVATE = 1
    MODE_SYNTHESIS = 2
    MODE_BRIEF = 3

    def __init__(self, config: HLWMConfig) -> None:
        super().__init__()
        self.config = config
        width = config.hidden_size
        self.backbone = QwenLikeBackbone(config)
        self.lm_head = nn.Linear(width, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.backbone.embed_tokens.weight

        self.diffusion = CategoricalDiffusion(
            config.vocab_size,
            config.diffusion_steps,
            config.diffusion_beta_start,
            config.diffusion_beta_end,
        )
        self.mode_embedding = nn.Embedding(4, width)
        nn.init.zeros_(self.mode_embedding.weight)
        self.lane_embedding = nn.Embedding(config.num_lanes, width)
        self.brief_queries = nn.Parameter(torch.randn(config.num_lanes, width) * 0.02)
        self.brief_projection = nn.Sequential(
            nn.Linear(width * 3, width), nn.SiLU(), nn.Linear(width, width)
        )

        self.root_norm = RMSNorm(width, config.rms_norm_eps)
        self.root_adapter = ResidualAdapter(width, config.expert_bottleneck)
        self.expert_bank = RoutedAdapterBank(config)
        self.route_context_projection = nn.Sequential(
            nn.Linear(width * 3, width), nn.SiLU(), nn.Linear(width, width)
        )

        self.timestep_conditioner = ScalarConditioner(width)
        self.depth_conditioner = ScalarConditioner(width)
        self.self_condition_projection = nn.Linear(width, width, bias=False)
        self.fast_initialization = nn.Linear(width * 3, width)
        self.fast_candidate = nn.Linear(width * 2, width)
        self.fast_gate = nn.Linear(width * 2, width)
        self.slow_initialization = nn.Linear(width * 2, width)
        self.slow_candidate = nn.Linear(width * 3, width)
        self.slow_gate = nn.Linear(width * 3, width)

        self.denoise_norm = RMSNorm(width, config.rms_norm_eps)
        self.summary_head = nn.Sequential(
            nn.Linear(width * 3, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.verification_head = nn.Linear(width * 2, 1)
        self.halt_head = nn.Sequential(
            nn.Linear(width * 2, width), nn.SiLU(), nn.Linear(width, 1)
        )

        self.global_candidate = nn.Linear(width * 2, width)
        self.global_gate = nn.Linear(width * 2, width)
        self.synthesis_lane_score = nn.Sequential(
            nn.Linear(width * 2 + 1, width), nn.SiLU(), nn.Linear(width, 1)
        )
        self.workspace_prefix_projection = nn.Sequential(
            nn.Linear(width * 3, width * 2),
            nn.SiLU(),
            nn.Linear(width * 2, width * config.synthesis_prefix_tokens),
        )
        self.workspace_prefix_norm = RMSNorm(width, config.rms_norm_eps)
        # Version 8.0: the near-null start lives in one scalar gate on the
        # whole prefix instead of zero-initialized projections.  Moving the
        # zero out of the projections gives diversity and orthogonality
        # pressure live gradients from step 0 (zero-init outputs made every
        # such penalty provably inert in Studies 5-8).
        self.prefix_gate = nn.Parameter(
            torch.tensor(float(config.prefix_gate_init))
        )
        if config.workspace_memory_windows > 0:
            # Latent read-out memory (Version 6.0 interface, Version 8.0
            # init): small random projections behind the shared tanh gate.
            self.workspace_memory_projection = nn.Linear(width, width)
            nn.init.normal_(self.workspace_memory_projection.weight, std=0.02)
            nn.init.zeros_(self.workspace_memory_projection.bias)
            self.workspace_summary_projection = nn.Linear(width, width)
            nn.init.normal_(self.workspace_summary_projection.weight, std=0.02)
            nn.init.zeros_(self.workspace_summary_projection.bias)
            self.workspace_memory_position = nn.Parameter(
                torch.randn(config.workspace_memory_windows, width) * 0.02
            )
            self.workspace_memory_lane = nn.Parameter(
                torch.randn(config.num_lanes, width) * 0.02
            )
            self.workspace_memory_norm = RMSNorm(width, config.rms_norm_eps)
        if config.premise_aux_weight > 0:
            # Version 9.0 latent-only probe: reads prefix positions and must
            # decode the withheld premise token ids. Supervises the latent
            # densely without ever seeing the gold answer.
            self.premise_probe = nn.Linear(width, width, bias=False)
            nn.init.normal_(self.premise_probe.weight, std=0.02)
        if config.gist_prefix_tokens > 0:
            # Version 9.0 attribution control: same prefix budget, no canvas,
            # no diffusion, no refinement. If this ties the workspace at
            # audit, compression suffices and the canvas is retired.
            self.gist_projection = nn.Sequential(
                nn.Linear(width, width * 2),
                nn.SiLU(),
                nn.Linear(width * 2, width * config.gist_prefix_tokens),
            )
            self.gist_norm = RMSNorm(width, config.rms_norm_eps)
        if config.latent_thoughts > 0:
            # Version 10.0 continuous-thought producer: the model's own last
            # hidden state through a 2-layer MLP with LayerNorm, re-injected
            # as the next input embedding after rescaling to the base
            # embedding standard deviation, so thoughts live inside the
            # embedding distribution by construction.
            self.latent_projection = nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, width),
                nn.SiLU(),
                nn.Linear(width, width),
            )
            # Lazily filled with the frozen embedding table's global std on
            # first use; registered as a buffer so resume restores it and the
            # smooth-L1 scale never silently changes mid-salvage.
            self.register_buffer(
                "latent_embed_std", torch.zeros(()), persistent=True
            )
            # Per-latent step supervision head (SIM-CoT, budget form): latent
            # k's state must decode its gold-trace window through the frozen
            # lm_head.  Training-only; discarded at inference.  This is also
            # the anti-homogenization term: six latents with six different
            # decode targets cannot collapse to one effective latent.
            self.latent_step_head = nn.Linear(width, width)
            nn.init.normal_(self.latent_step_head.weight, std=0.02)
            nn.init.zeros_(self.latent_step_head.bias)
            # Reconstruction cue: a learned embedding vector, never a resized
            # vocabulary row (an unadded token id returns None and a resized
            # row would sit untrained outside LoRA).
            self.recon_cue_embedding = nn.Parameter(torch.randn(width) * 0.02)
            # Per-layer EMA of the teacher's pre-answer hidden std, the
            # normalizer of the distillation smooth-L1.  A persistent buffer:
            # if this were a loop-local float, resume would silently rescale
            # the distillation term by 10-100x mid-salvage.
            self.register_buffer(
                "distill_ema_std",
                torch.zeros(config.num_hidden_layers),
                persistent=True,
            )
        if config.kv_prefix_slots > 0:
            self.kv_prefix_projector = KVPrefixProjector(config)
        if config.mlp_expert_count > 0:
            # Detached router probe (StableMoE stage 1): learns to PREDICT
            # the deterministic family route from the prompt representation
            # and never shapes it during training.  Small-normal init — no
            # zero-init anywhere on the router path.
            self.family_router = nn.Linear(width, config.router_families)
            nn.init.normal_(self.family_router.weight, std=0.02)
            nn.init.zeros_(self.family_router.bias)
        self.commitment_head = nn.Sequential(
            nn.Linear(width * 2 + 2, width), nn.SiLU(), nn.Linear(width, 3)
        )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.backbone.embed_tokens

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def gradient_checkpointing_enable(self) -> None:
        self.backbone.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:
        self.backbone.gradient_checkpointing_disable()

    def freeze_language_substrate(self) -> None:
        """Freeze transplanted Qwen weights while leaving HLWM modules trainable.

        This is the default prototype regime on a 16 GB T4.  Gradients still
        pass through the frozen substrate into timestep, state, routing and
        adapter modules; the language weights simply receive no optimizer state.
        """

        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        for parameter in self.lm_head.parameters():
            parameter.requires_grad_(False)

    def unfreeze_language_adapters(self) -> None:
        """Enable only the configured low-rank Qwen sidecars.

        Base Qwen projections stay FP16 and frozen.  The small sidecars can use
        FP32 optimizer parameters safely because they are separate residual
        branches rather than a dtype-promoted pretrained decoder block.
        """

        for module in self.backbone.modules():
            if isinstance(module, LoRAResidual) and module.enabled:
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        # Version 10.0 gates live INSIDE the backbone but are HLWM-native,
        # never transplanted Qwen weights: the substrate freeze must not
        # reach them.  Session I-3's on-device preflight caught them frozen
        # at init — the v8 frozen-alpha catastrophe reborn through the
        # freeze path, invisible to every test that skips main()'s
        # freeze/unfreeze sequence.
        for name, parameter in self.backbone.named_parameters():
            if "prefix_attn_gate" in name:
                parameter.requires_grad_(True)

    def unfreeze_language_tail(self, layer_count: int = 1) -> None:
        """Tune complete decoder blocks for controlled backward compatibility.

        New T4 packages use :meth:`unfreeze_language_adapters` instead.  Full
        block tuning is retained for old checkpoints and explicit ablations.
        """

        if not 0 <= layer_count <= len(self.backbone.layers):
            raise ValueError("language tail layer count is out of range")
        if layer_count:
            for layer in self.backbone.layers[-layer_count:]:
                for parameter in layer.parameters():
                    parameter.requires_grad_(True)
            for parameter in self.backbone.norm.parameters():
                parameter.requires_grad_(True)

    def trainable_parameter_summary(self) -> Dict[str, int]:
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in self.parameters())
        return {"trainable": trainable, "total": total, "frozen": total - trainable}

    def _apply_root(self, hidden: Tensor) -> Tensor:
        return hidden + self.root_adapter(self.root_norm(hidden))

    @staticmethod
    def _masked_mean(hidden: Tensor, mask: Tensor, dimension: int = 1) -> Tensor:
        weights = mask.to(hidden.dtype).unsqueeze(-1)
        return (hidden * weights).sum(dim=dimension) / weights.sum(dim=dimension).clamp_min(1.0)

    def _encode_committed(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        mode = self.mode_embedding.weight[self.MODE_CAUSAL].view(1, 1, -1)
        hidden = self.backbone(
            inputs_embeds=self.backbone.embed_tokens(input_ids) + mode,
            attention_mask=attention_mask,
            attention_mode="causal",
        )
        return self._apply_root(hidden)

    def causal_logits(
        self, input_ids: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Normal autoregressive path, independent of private diffusion."""

        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()
        hidden = self._encode_committed(input_ids, attention_mask)
        return self.lm_head(hidden)

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        *,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        do_sample: bool = False,
        eos_token_id: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """KV-cached autoregressive generator for the causal answer channel.

        The response cue (when configured) is appended internally and never
        appears in the returned sequence, which stays ``[prompt][answer]`` so
        callers can slice by prompt length as before.
        """

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()
        eos = self.config.eos_token_id if eos_token_id is None else eos_token_id
        prefill_embeds, prefill_mask = self._channel_prefill(
            input_ids, attention_mask, None
        )
        hidden, past = self.backbone(
            inputs_embeds=prefill_embeds,
            attention_mask=prefill_mask,
            attention_mode="causal",
            use_cache=True,
        )
        next_logits = self.lm_head(self._apply_root(hidden)[:, -1].to(self.lm_head.weight.dtype))
        mode = self.mode_embedding.weight[self.MODE_CAUSAL].view(1, 1, -1)
        generated = input_ids
        running_mask = prefill_mask
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        for _ in range(max_new_tokens):
            logits = next_logits / temperature
            if top_k is not None and 0 < top_k < logits.shape[-1]:
                threshold = logits.topk(top_k, dim=-1).values[:, -1, None]
                logits = logits.masked_fill(logits < threshold, -torch.inf)
            if do_sample:
                next_token = torch.multinomial(
                    logits.softmax(dim=-1), 1, generator=generator
                ).squeeze(1)
            else:
                next_token = logits.argmax(dim=-1)
            next_token = torch.where(
                finished, torch.full_like(next_token, eos), next_token
            )
            generated = torch.cat((generated, next_token[:, None]), dim=1)
            finished |= next_token == eos
            if bool(finished.all()):
                break
            running_mask = torch.cat(
                (running_mask, torch.ones_like(next_token[:, None])), dim=1
            )
            hidden, past = self.backbone(
                inputs_embeds=self.backbone.embed_tokens(next_token[:, None]) + mode,
                attention_mask=running_mask,
                attention_mode="causal",
                past_key_values=past,
                use_cache=True,
            )
            next_logits = self.lm_head(self._apply_root(hidden)[:, -1].to(self.lm_head.weight.dtype))
        return generated

    def _prepare_briefs(
        self,
        context_hidden: Tensor,
        attention_mask: Tensor,
        global_state: Tensor,
        *,
        lane_briefs: Optional[Tensor],
        lane_brief_ids: Optional[Tensor],
        lane_brief_attention_mask: Optional[Tensor],
    ) -> Tensor:
        batch = context_hidden.shape[0]
        lanes = self.config.num_lanes
        if lane_briefs is not None and lane_brief_ids is not None:
            raise ValueError("provide lane_briefs or lane_brief_ids, not both")
        if lane_briefs is not None:
            expected = (batch, lanes, self.config.hidden_size)
            if lane_briefs.shape != expected:
                raise ValueError("lane_briefs must have shape %r" % (expected,))
            return lane_briefs
        if lane_brief_ids is not None:
            if lane_brief_ids.ndim != 3 or lane_brief_ids.shape[:2] != (batch, lanes):
                raise ValueError("lane_brief_ids must have shape [batch, num_lanes, length]")
            brief_length = lane_brief_ids.shape[-1]
            flat_ids = lane_brief_ids.reshape(batch * lanes, brief_length)
            if lane_brief_attention_mask is None:
                flat_mask = (flat_ids != self.config.pad_token_id).long()
            else:
                flat_mask = lane_brief_attention_mask.reshape(batch * lanes, brief_length)
            mode = self.mode_embedding.weight[self.MODE_BRIEF].view(1, 1, -1)
            encoded = self.backbone(
                inputs_embeds=self.backbone.embed_tokens(flat_ids) + mode,
                attention_mask=flat_mask,
                attention_mode="causal",
            )
            encoded = self._apply_root(encoded)
            pooled = self._masked_mean(encoded, flat_mask).reshape(batch, lanes, -1)
            return pooled

        # Version 8.0 structural lane asymmetry: each lane summarizes its own
        # contiguous span of the context instead of sharing one pooled
        # summary.  Symmetric parallel branches under a shared loss receive
        # near-identical gradients and converge (Studies 5-8 measured lane
        # cosine 0.90-0.97 through seven configurations); different inputs
        # break the symmetry structurally rather than by penalty.
        position_chunks = torch.tensor_split(
            torch.arange(context_hidden.shape[1], device=context_hidden.device), lanes
        )
        lane_views = []
        for chunk in position_chunks:
            chunk_mask = torch.zeros_like(attention_mask)
            chunk_mask[:, chunk] = attention_mask[:, chunk]
            lane_views.append(self._masked_mean(context_hidden, chunk_mask))
        context = torch.stack(lane_views, dim=1)
        global_expanded = global_state[:, None, :].expand(-1, lanes, -1)
        lane_ids = torch.arange(lanes, device=context_hidden.device)
        learned = self.brief_queries[None, :, :] + self.lane_embedding(lane_ids)[None, :, :]
        learned = learned.expand(batch, -1, -1)
        return self.brief_projection(torch.cat((context, global_expanded, learned), dim=-1))

    @staticmethod
    def _normalize_private_embedding(hidden: Tensor) -> Tensor:
        """Keep recurrent diffusion embeddings on a stable unit sphere."""

        return F.normalize(hidden.float(), dim=-1).to(hidden.dtype)

    def _private_recurrence(
        self,
        context_hidden: Tensor,
        context_attention_mask: Tensor,
        global_state: Tensor,
        briefs: Tensor,
        noisy_lane_ids: Tensor,
        transition_timesteps: Tensor,
        lane_target_ids: Optional[Tensor],
        lane_target_attention_mask: Tensor,
        *,
        sample_reverse: bool,
        adaptive_halt: bool,
        advance_final_state: bool,
        generator: Optional[torch.Generator],
        route_override: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Apply the tied private transition for an explicit timestep schedule.

        Training can pass a one-column schedule for DiffusionBlocks-style local
        learning or a short schedule for joint tuning.  Inference passes T..1,
        which prevents refinement depth from silently truncating the reverse
        diffusion chain.
        """

        batch, lanes, canvas_length = noisy_lane_ids.shape
        if transition_timesteps.ndim != 2 or transition_timesteps.shape[0] != batch:
            raise ValueError("transition_timesteps must have shape [batch, updates]")
        if transition_timesteps.shape[1] <= 0:
            raise ValueError("the private recurrence needs at least one update")
        if bool(
            ((transition_timesteps < 1) | (transition_timesteps > self.diffusion.steps)).any()
        ):
            raise ValueError("private recurrence timesteps must be within 1..T")

        width = self.config.hidden_size
        flat_batch = batch * lanes
        lane_ids = torch.arange(lanes, device=noisy_lane_ids.device)
        lane_condition = self.lane_embedding(lane_ids)[None, :, :].expand(batch, -1, -1)
        global_lanes = global_state[:, None, :].expand(-1, lanes, -1)
        slow = self.slow_initialization(torch.cat((briefs, global_lanes), dim=-1))

        noisy_embedding = self._normalize_private_embedding(
            self.backbone.embed_tokens(noisy_lane_ids)
        )
        fast = self.fast_initialization(
            torch.cat(
                (
                    noisy_embedding,
                    briefs[:, :, None, :].expand(-1, -1, canvas_length, -1),
                    global_lanes[:, :, None, :].expand(-1, -1, canvas_length, -1),
                ),
                dim=-1,
            )
        )
        # The initial state is exactly discrete, so embedding current_ids is
        # equivalent to multiplying a 151,936-way one-hot tensor by the Qwen
        # table and avoids a large dense allocation/matmul on every local step.
        canvas_probabilities: Optional[Tensor] = None
        current_ids = noisy_lane_ids
        self_condition = torch.zeros_like(fast)
        active = torch.ones(batch, lanes, dtype=torch.bool, device=noisy_lane_ids.device)

        context_lanes = context_hidden[:, None, :, :].expand(-1, lanes, -1, -1)
        context_lanes = context_lanes.reshape(flat_batch, context_hidden.shape[1], width)
        context_mask = context_attention_mask[:, None, :].expand(-1, lanes, -1)
        context_mask = context_mask.reshape(flat_batch, context_hidden.shape[1])
        canvas_mask = lane_target_attention_mask.reshape(flat_batch, canvas_length)
        combined_mask = torch.cat((context_mask, canvas_mask), dim=1)
        flat_briefs = briefs.reshape(flat_batch, width)
        flat_global = global_lanes.reshape(flat_batch, width)

        forced_route: Optional[Tensor] = None
        if route_override is not None:
            if not 0 <= int(route_override) < self.config.num_experts:
                raise ValueError("route_override is outside the expert bank")
            forced_route = torch.full(
                (flat_batch,),
                int(route_override),
                dtype=torch.long,
                device=noisy_lane_ids.device,
            )

        denoise_logits: Optional[Tensor] = None
        last_private_hidden: Optional[Tensor] = None
        cached_route: Optional[Tensor] = None
        halt_history = []
        route_history = []
        route_path_history = []
        router_probability_history = []
        nll_history = []

        for update_index in range(transition_timesteps.shape[1]):
            timestep = transition_timesteps[:, update_index]
            flat_timestep = timestep[:, None].expand(-1, lanes).reshape(-1)
            time_condition = self.timestep_conditioner(flat_timestep)
            depth_values = torch.full_like(flat_timestep, update_index)
            depth_condition = self.depth_conditioner(depth_values)
            flat_slow = slow.reshape(flat_batch, width)
            route_context = self.route_context_projection(
                torch.cat((flat_slow, flat_briefs, flat_global), dim=-1)
            )

            if canvas_probabilities is None:
                expected_token_embedding = self.backbone.embed_tokens(
                    current_ids.reshape(flat_batch, canvas_length)
                )
            else:
                expected_token_embedding = torch.matmul(
                    canvas_probabilities.reshape(flat_batch, canvas_length, -1),
                    self.backbone.embed_tokens.weight,
                )
            expected_token_embedding = self._normalize_private_embedding(
                expected_token_embedding
            )
            private_condition = (
                flat_briefs
                + flat_slow
                + flat_global
                + lane_condition.reshape(flat_batch, width)
                + time_condition
                + depth_condition
            )
            private_embedding = (
                expected_token_embedding
                + fast.reshape(flat_batch, canvas_length, width)
                + self.self_condition_projection(
                    self_condition.reshape(flat_batch, canvas_length, width)
                )
                + private_condition[:, None, :]
                + self.mode_embedding.weight[self.MODE_PRIVATE].view(1, 1, -1)
            )
            combined = torch.cat((context_lanes, private_embedding), dim=1)
            transformed = self.backbone(
                inputs_embeds=combined,
                attention_mask=combined_mask,
                attention_mode="protected",
                context_length=context_hidden.shape[1],
            )
            transformed = self._apply_root(transformed)
            private_hidden = transformed[:, context_hidden.shape[1] :]
            refresh_route = update_index % self.config.route_window_steps == 0
            selected_override = None if refresh_route else cached_route
            if forced_route is not None:
                # A routing intervention pins every window to one named expert
                # so evaluation can test whether routing is causally live.
                selected_override = forced_route
            (
                private_hidden,
                route_indices,
                router_probabilities,
                route_path_masks,
            ) = self.expert_bank(
                private_hidden,
                route_context,
                selected_override=selected_override,
            )
            if refresh_route:
                cached_route = route_indices.detach()

            flat_fast = fast.reshape(flat_batch, canvas_length, width)
            fast_input = torch.cat((flat_fast, private_hidden), dim=-1)
            proposed_fast = torch.tanh(self.fast_candidate(fast_input))
            fast_gate = torch.sigmoid(self.fast_gate(fast_input))
            updated_fast = (1.0 - fast_gate) * flat_fast + fast_gate * proposed_fast
            active_tokens = active.reshape(flat_batch, 1, 1)
            updated_fast = torch.where(active_tokens, updated_fast, flat_fast)
            fast = updated_fast.reshape(batch, lanes, canvas_length, width)

            denoise_hidden = self.denoise_norm(private_hidden + updated_fast)
            # The 151,936-way Qwen vocabulary is too large for robust FP16
            # normalization on every CUDA kernel.  Keep its probabilities and
            # CE accumulation in FP32 even though the frozen projection is FP16.
            denoise_logits = self.lm_head(denoise_hidden).float().reshape(
                batch, lanes, canvas_length, self.config.vocab_size
            )
            pooled = self._masked_mean(private_hidden, canvas_mask).reshape(
                batch, lanes, width
            )
            halt_logits = self.halt_head(torch.cat((pooled, slow), dim=-1)).squeeze(-1)
            halt_history.append(halt_logits)
            route_history.append(route_indices.reshape(batch, lanes))
            route_path_history.append(
                route_path_masks.reshape(batch, lanes, self.config.num_experts)
            )
            router_probability_history.append(
                router_probabilities.reshape(batch, lanes, self.config.num_experts)
            )

            if lane_target_ids is not None:
                token_nll = _masked_token_cross_entropy(
                    denoise_logits, lane_target_ids, lane_target_attention_mask
                )
                mask = lane_target_attention_mask.to(token_nll.dtype)
                per_lane_nll = token_nll.sum(dim=-1) / mask.sum(
                    dim=-1
                ).clamp_min(1.0)
                nll_history.append(per_lane_nll)

            is_final_update = update_index + 1 == transition_timesteps.shape[1]
            if not is_final_update or advance_final_state:
                predicted_clean = denoise_logits.softmax(dim=-1, dtype=torch.float32)
                flat_current_ids = current_ids.reshape(flat_batch, canvas_length)
                flat_predicted = predicted_clean.reshape(
                    flat_batch, canvas_length, self.config.vocab_size
                )
                next_ids, next_probabilities = self.diffusion.reverse_step(
                    flat_current_ids,
                    flat_predicted,
                    flat_timestep,
                    sample=sample_reverse,
                    generator=generator,
                )
                next_ids = next_ids.reshape(batch, lanes, canvas_length)
                next_probabilities = next_probabilities.reshape(
                    batch, lanes, canvas_length, self.config.vocab_size
                )
                active_canvas = active[:, :, None]
                current_ids = torch.where(active_canvas, next_ids, current_ids)
                if canvas_probabilities is None:
                    canvas_probabilities = next_probabilities
                else:
                    canvas_probabilities = torch.where(
                        active_canvas[:, :, :, None],
                        next_probabilities,
                        canvas_probabilities,
                    )
                self_condition = self._normalize_private_embedding(
                    torch.matmul(
                        predicted_clean.detach(), self.backbone.embed_tokens.weight
                    )
                )

            if (update_index + 1) % self.config.slow_update_every == 0:
                slow_input = torch.cat((slow, pooled, briefs), dim=-1)
                proposed_slow = torch.tanh(self.slow_candidate(slow_input))
                slow_gate = torch.sigmoid(self.slow_gate(slow_input))
                updated_slow = (1.0 - slow_gate) * slow + slow_gate * proposed_slow
                slow = torch.where(active[:, :, None], updated_slow, slow)

            if adaptive_halt and update_index + 1 >= self.config.min_halt_steps:
                newly_halted = torch.sigmoid(halt_logits) >= self.config.halt_threshold
                active = active & ~newly_halted
            last_private_hidden = private_hidden.reshape(
                batch, lanes, canvas_length, width
            )
            if adaptive_halt and not bool(active.any()):
                transition_timesteps = transition_timesteps[:, : update_index + 1]
                break

        if denoise_logits is None or last_private_hidden is None:
            raise RuntimeError("private recurrence executed zero updates")
        verification_logits = self.verification_head(
            torch.cat((last_private_hidden, fast), dim=-1)
        ).squeeze(-1)
        token_mask = lane_target_attention_mask.to(last_private_hidden.dtype)
        pooled_private = (
            last_private_hidden * token_mask[:, :, :, None]
        ).sum(dim=2) / token_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        lane_summaries = self.summary_head(
            torch.cat((pooled_private, slow, briefs), dim=-1)
        )

        return {
            "denoise_logits": denoise_logits,
            "verification_logits": verification_logits,
            "halt_logits": torch.stack(halt_history, dim=-1),
            "route_indices": torch.stack(route_history, dim=-1),
            "router_probabilities": torch.stack(router_probability_history, dim=-2),
            "route_path_masks": torch.stack(route_path_history, dim=-2),
            "lane_summaries": lane_summaries,
            "fast_state": fast,
            "slow_state": slow,
            "private_hidden": last_private_hidden,
            "canvas_probabilities": canvas_probabilities,
            "nll_history": nll_history,
            "transition_timesteps": transition_timesteps,
        }

    def _router_balance_loss(self, private: Mapping[str, Tensor]) -> Tensor:
        probabilities = private["router_probabilities"]
        route_indices = private["route_indices"]
        mean_probability = probabilities.reshape(-1, self.config.num_experts).mean(dim=0)
        mean_assignment = F.one_hot(route_indices, self.config.num_experts).float()
        mean_assignment = mean_assignment.reshape(-1, self.config.num_experts).mean(dim=0)
        return self.config.num_experts * torch.sum(
            mean_probability * mean_assignment.detach()
        )

    def _router_marginal_entropy_loss(self, private: Mapping[str, Tensor]) -> Tensor:
        """Return KL(mean routing probabilities || uniform) in FP32.

        The Version 5.4 audit showed near-deterministic routing (98.4% one
        expert for seed 17).  The Switch-style balance term above only couples
        probabilities to realized assignments; this term directly penalizes a
        collapsed routing marginal and is exactly zero at uniform usage.
        """

        probabilities = private["router_probabilities"].float().reshape(
            -1, self.config.num_experts
        )
        marginal = probabilities.mean(dim=0).clamp_min(1.0e-9)
        return torch.sum(marginal * (marginal * self.config.num_experts).log())

    def forward_local_denoise(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        *,
        lane_target_ids: Tensor,
        lane_target_attention_mask: Optional[Tensor] = None,
        lane_briefs: Optional[Tensor] = None,
        lane_brief_ids: Optional[Tensor] = None,
        lane_brief_attention_mask: Optional[Tensor] = None,
        timesteps: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> HLWMLocalDenoiseOutput:
        """Train one sampled-noise tied transition without recurrent BPTT."""

        batch = input_ids.shape[0]
        lanes = self.config.num_lanes
        if lane_target_ids.ndim != 3 or lane_target_ids.shape[:2] != (batch, lanes):
            raise ValueError("lane_target_ids must have shape [batch, num_lanes, length]")
        canvas_length = lane_target_ids.shape[-1]
        device = input_ids.device
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()
        if lane_target_attention_mask is None:
            lane_target_attention_mask = (
                lane_target_ids != self.config.pad_token_id
            ).long()
        if lane_target_attention_mask.shape != lane_target_ids.shape:
            raise ValueError("lane_target_attention_mask must match lane_target_ids")

        context_hidden = self._encode_committed(input_ids, attention_mask)
        context_summary = self._masked_mean(context_hidden, attention_mask)
        briefs = self._prepare_briefs(
            context_hidden,
            attention_mask,
            context_summary,
            lane_briefs=lane_briefs,
            lane_brief_ids=lane_brief_ids,
            lane_brief_attention_mask=lane_brief_attention_mask,
        )
        if timesteps is None:
            timesteps = self.diffusion.sample_training_timesteps(batch, device, generator)
        else:
            timesteps = timesteps.to(device=device, dtype=torch.long)
        flat_targets = lane_target_ids.reshape(batch * lanes, canvas_length)
        flat_timesteps = timesteps[:, None].expand(-1, lanes).reshape(-1)
        corrupted_ids = self.diffusion.q_sample(
            flat_targets, flat_timesteps, generator
        ).reshape(batch, lanes, canvas_length)
        private = self._private_recurrence(
            context_hidden,
            attention_mask,
            context_summary,
            briefs,
            corrupted_ids,
            timesteps[:, None],
            lane_target_ids,
            lane_target_attention_mask,
            sample_reverse=False,
            adaptive_halt=False,
            advance_final_state=False,
            generator=generator,
        )
        token_loss = _masked_token_cross_entropy(
            private["denoise_logits"], lane_target_ids, lane_target_attention_mask
        )
        mask = lane_target_attention_mask.to(token_loss.dtype)
        denoise_loss = token_loss.sum() / mask.sum().clamp_min(1.0)
        brief_loss = (
            1.0
            - F.cosine_similarity(
                private["lane_summaries"].float(), briefs.float(), dim=-1
            )
        ).mean()
        router_loss = self._router_balance_loss(private)
        router_entropy_loss = self._router_marginal_entropy_loss(private)
        if self.config.expert_diversity_weight > 0:
            expert_diversity_loss = self.expert_bank.output_diversity(context_summary)
        else:
            expert_diversity_loss = denoise_loss.new_zeros(())
        if lanes > 1:
            normalized_lanes = F.normalize(private["lane_summaries"].float(), dim=-1)
            similarities = torch.matmul(
                normalized_lanes, normalized_lanes.transpose(1, 2)
            )
            off_diagonal = ~torch.eye(
                lanes, dtype=torch.bool, device=device
            )[None, :, :]
            lane_diversity_loss = similarities.masked_select(off_diagonal).square().mean()
        else:
            lane_diversity_loss = denoise_loss.new_zeros(())
        loss = (
            self.config.denoise_loss_weight * denoise_loss
            + self.config.brief_loss_weight * brief_loss
            + self.config.router_aux_weight * router_loss
            + self.config.router_entropy_weight * router_entropy_loss
            + self.config.expert_diversity_weight * expert_diversity_loss
            + self.config.lane_diversity_weight * lane_diversity_loss
        )
        return HLWMLocalDenoiseOutput(
            loss=loss,
            denoise_loss=denoise_loss,
            brief_loss=brief_loss,
            router_loss=router_loss,
            router_entropy_loss=router_entropy_loss,
            expert_diversity_loss=expert_diversity_loss,
            lane_diversity_loss=lane_diversity_loss,
            denoise_logits=private["denoise_logits"],
            corrupted_ids=corrupted_ids,
            timesteps=timesteps,
            route_indices=private["route_indices"],
            route_path_masks=private["route_path_masks"],
            transition_count=1,
        )

    def _build_workspace(
        self,
        private: Mapping[str, Tensor],
        global_state: Tensor,
        context_summary: Tensor,
        lane_attention_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Cross the synchronization barrier using summaries, never lane tokens."""

        verification_probabilities = torch.sigmoid(private["verification_logits"])
        verification_weights = lane_attention_mask.to(verification_probabilities.dtype)
        verification_error = (
            verification_probabilities * verification_weights
        ).sum(dim=-1) / verification_weights.sum(dim=-1).clamp_min(1.0)
        reliability = 1.0 - verification_error
        summary_weights = reliability / reliability.sum(dim=1, keepdim=True).clamp_min(
            1.0e-6
        )
        aggregate = torch.einsum(
            "bn,bnh->bh", summary_weights, private["lane_summaries"]
        )
        global_input = torch.cat((global_state, aggregate), dim=-1)
        proposed_global = torch.tanh(self.global_candidate(global_input))
        global_gate = torch.sigmoid(self.global_gate(global_input))
        global_state = (1.0 - global_gate) * global_state + global_gate * proposed_global

        expanded_global = global_state[:, None, :].expand(
            -1, self.config.num_lanes, -1
        )
        lane_score_input = torch.cat(
            (
                private["lane_summaries"],
                expanded_global,
                reliability.unsqueeze(-1),
            ),
            dim=-1,
        )
        lane_scores = self.synthesis_lane_score(lane_score_input).squeeze(-1)
        lane_scores = lane_scores + reliability.clamp_min(1.0e-6).log()
        synthesis_weights = lane_scores.softmax(dim=-1)
        synthesis_summary = torch.einsum(
            "bn,bnh->bh", synthesis_weights, private["lane_summaries"]
        )
        prefix = self.workspace_prefix_projection(
            torch.cat((context_summary, global_state, synthesis_summary), dim=-1)
        )
        prefix = prefix.reshape(
            global_state.shape[0],
            self.config.synthesis_prefix_tokens,
            self.config.hidden_size,
        )
        prefix = self.workspace_prefix_norm(prefix)
        prefix = prefix + self.mode_embedding.weight[self.MODE_SYNTHESIS].view(1, 1, -1)
        if self.config.workspace_memory_windows > 0:
            # Latent read-out memory: windowed masked means of the final
            # per-lane canvas hidden states plus one summary token per lane.
            # This widens the synchronization barrier's channel from three
            # pooled vectors to a structured latent memory while still never
            # exposing private token identities to the decoder.
            hidden_states = private["private_hidden"].float()
            token_weights = lane_attention_mask.to(hidden_states.dtype)[..., None]
            batch, lanes, _, width = hidden_states.shape
            window_chunks = torch.tensor_split(
                hidden_states * token_weights, self.config.workspace_memory_windows, dim=2
            )
            weight_chunks = torch.tensor_split(
                token_weights, self.config.workspace_memory_windows, dim=2
            )
            windows = torch.stack(
                [
                    chunk.sum(dim=2) / weight.sum(dim=2).clamp_min(1.0)
                    for chunk, weight in zip(window_chunks, weight_chunks)
                ],
                dim=2,
            )
            # Inputs are normalized before the zero-initialized projections so
            # every memory token starts near-null (a small positional marker)
            # and the channel opens only as gradients demand it; a trailing
            # norm would re-inflate the null tokens and defeat the gate.
            windows = self.workspace_memory_projection(
                self.workspace_memory_norm(windows.to(prefix.dtype))
            ) + self.workspace_memory_position[None, None, :, :]
            windows = windows + self.workspace_memory_lane[None, :lanes, None, :]
            summary_tokens = self.workspace_summary_projection(
                self.workspace_memory_norm(
                    private["lane_summaries"].to(prefix.dtype)
                )
            ) + self.workspace_memory_lane[None, :lanes, :]
            memory = torch.cat(
                (windows.reshape(batch, -1, width), summary_tokens), dim=1
            )
            memory = memory + self.mode_embedding.weight[self.MODE_PRIVATE].view(1, 1, -1)
            prefix = torch.cat((prefix, memory), dim=1)
        # Version 8.0: one learned scalar gate scales the whole prefix.  The
        # near-null start lives here (tanh of a small positive init), so the
        # projections above can carry live gradients from step 0.
        prefix = prefix * torch.tanh(self.prefix_gate).to(prefix.dtype)
        return global_state, prefix, verification_error

    def _build_gist_prefix(
        self,
        context_hidden: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Attribution-control prefix (Version 9.0): mean-pool projection only.

        Same token budget as the workspace prefix, same mode embeddings, no
        canvas, no diffusion, no refinement, no gate. Trained on alternating
        masked batches so the audit comparison is not against a straw control.
        """

        if self.config.gist_prefix_tokens <= 0:
            raise ValueError("gist prefix requested but gist_prefix_tokens is zero")
        pooled = self._masked_mean(context_hidden, attention_mask)
        prefix = self.gist_projection(self.gist_norm(pooled.unsqueeze(1)).squeeze(1))
        prefix = prefix.reshape(
            pooled.shape[0], self.config.gist_prefix_tokens, self.config.hidden_size
        )
        return prefix + self.mode_embedding.weight[self.MODE_SYNTHESIS].view(1, 1, -1)

    def _premise_probe_loss(
        self,
        workspace_prefix: Tensor,
        premise_ids: Tensor,
        premise_attention_mask: Tensor,
        premise_negative_ids: Tensor,
    ) -> Optional[Tuple[Tensor, Tensor]]:
        """Latent-only premise decoding loss and accuracy (Version 9.0).

        Position ``i`` of the prefix must identify premise token ``i`` among a
        per-row candidate set of the premise tokens plus in-prompt negatives.
        Scoring is restricted to that candidate set, so there is no
        vocabulary-size softmax. The probe reads ONLY the prefix, so the loss
        has no path that bypasses the latent.
        """

        mask = premise_attention_mask.bool()
        if not bool(mask.any()):
            return None
        length = premise_ids.shape[1]
        if workspace_prefix.shape[1] < length:
            raise ValueError(
                "premise supervision needs at least %d prefix positions, found %d"
                % (length, workspace_prefix.shape[1])
            )
        queries = self.premise_probe(workspace_prefix[:, :length, :].float())
        candidates = torch.cat(
            (premise_ids.clamp_min(0), premise_negative_ids), dim=1
        )
        candidate_embeddings = self.backbone.embed_tokens(candidates).float().detach()
        logits = torch.einsum("bph,bch->bpc", queries, candidate_embeddings)
        logits = logits / float(self.config.hidden_size) ** 0.5
        targets = (
            torch.arange(length, device=premise_ids.device)
            .unsqueeze(0)
            .expand(premise_ids.shape[0], -1)
        )
        loss = F.cross_entropy(logits[mask], targets[mask])
        predicted_ids = candidates.gather(1, logits.argmax(dim=-1))
        accuracy = (
            (predicted_ids == premise_ids.clamp_min(0))[mask].float().mean().detach()
        )
        return loss, accuracy

    def _channel_prefill(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        workspace_prefix: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """Embeddings and mask for ``[context][prefix?][cue]`` (Version 8.0).

        The two channels are a matched pair: the causal channel is exactly
        the workspace channel with ``workspace_prefix=None``.  The sequence
        starts and ends with hard tokens — position zero stays real text
        (attention-sink insurance on a no-BOS base model) and decoding always
        continues from the hard response cue, never from soft vectors or a
        document boundary.  Context and cue carry the causal register.
        """

        batch = input_ids.shape[0]
        mode = self.mode_embedding.weight[self.MODE_CAUSAL].view(1, 1, -1)
        parts = [self.backbone.embed_tokens(input_ids) + mode]
        masks = [attention_mask]
        if workspace_prefix is not None:
            parts.append(workspace_prefix.to(parts[0].dtype))
            masks.append(
                torch.ones(
                    batch,
                    workspace_prefix.shape[1],
                    dtype=attention_mask.dtype,
                    device=input_ids.device,
                )
            )
        if self.config.response_cue_ids is not None:
            cue = torch.tensor(
                self.config.response_cue_ids,
                dtype=torch.long,
                device=input_ids.device,
            ).unsqueeze(0).expand(batch, -1)
            parts.append(self.backbone.embed_tokens(cue) + mode)
            masks.append(
                torch.ones(
                    batch, cue.shape[1], dtype=attention_mask.dtype, device=input_ids.device
                )
            )
        return (
            torch.cat(parts, dim=1).to(self.backbone.embed_tokens.weight.dtype),
            torch.cat(masks, dim=1),
        )

    def _channel_teacher_force(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        workspace_prefix: Optional[Tensor],
        target_ids: Tensor,
        target_attention_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Teacher-force targets through a channel; no BOS is ever inserted.

        Returns logits aligned to ``target_ids`` (position ``i`` predicted
        from the token before it; the first target is predicted from the last
        prefill token) and the hidden states at the target positions.
        """

        prefill_embeds, prefill_mask = self._channel_prefill(
            input_ids, attention_mask, workspace_prefix
        )
        mode = self.mode_embedding.weight[self.MODE_CAUSAL].view(1, 1, -1)
        # The fp32 mode embedding promotes the sum; cast back before the
        # frozen reduced-precision backbone (session I-5: the audit's naked
        # causal logprob crashed on fp32 x fp16 in q_proj).
        target_embeds = (self.backbone.embed_tokens(target_ids) + mode).to(
            self.backbone.embed_tokens.weight.dtype
        )
        combined = torch.cat((prefill_embeds, target_embeds), dim=1)
        combined_mask = torch.cat((prefill_mask, target_attention_mask), dim=1)
        hidden = self.backbone(
            inputs_embeds=combined,
            attention_mask=combined_mask,
            attention_mode="causal",
        )
        hidden = self._apply_root(hidden)
        target_length = target_ids.shape[1]
        predictor_hidden = hidden[:, -target_length - 1 : -1]
        target_hidden = hidden[:, -target_length:]
        return self.lm_head(predictor_hidden), target_hidden

    # ------------------------------------------------------------------
    # Version 10.0 dense-supervision channel.

    def _grounded_embedding_std(self) -> Tensor:
        """Frozen embedding table's global std, computed once and buffered."""

        if float(self.latent_embed_std) <= 0.0:
            with torch.no_grad():
                self.latent_embed_std.copy_(
                    self.backbone.embed_tokens.weight.float().std()
                )
        return self.latent_embed_std

    def ground_latents(self, vectors: Tensor) -> Tensor:
        """Rescale latent vectors to the base embedding standard deviation.

        Scale-only grounding (per vector): the thought keeps its direction
        but lives at embedding magnitude, so the frozen decoder reads it as
        an in-distribution input rather than an off-manifold spike.
        """

        std = self._grounded_embedding_std().to(vectors.dtype)
        per_vector = vectors.float().std(dim=-1, keepdim=True).clamp_min(1.0e-6)
        return (vectors.float() / per_vector * std.float()).to(vectors.dtype)

    def produce_latent_thoughts(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Autoregressively produce K continuous thoughts after the prompt.

        Returns ``(thought_embeds, thought_states)``: the grounded input
        embeddings fed back into the model (K positions, fixed offsets after
        the padded prompt, matching the segment convention every other
        channel in this file uses) and the raw last-layer hidden states at
        those positions (the projector's input).  The loop is cache-free so
        it composes with gradient checkpointing; K is small and the
        sequences are short, and the cost is priced in the Study 10 ledger.
        The first thought reads the last VALID prompt position per row, not
        the padded tail (right-padding would otherwise hand it a zero
        state).
        """

        if self.config.latent_thoughts <= 0:
            raise ValueError("latent_thoughts is zero; the channel is disabled")
        batch = input_ids.shape[0]
        mode = self.mode_embedding.weight[self.MODE_SYNTHESIS].view(1, 1, -1)
        embed_dtype = self.backbone.embed_tokens.weight.dtype
        prompt_embeds = (
            self.backbone.embed_tokens(input_ids)
            + self.mode_embedding.weight[self.MODE_CAUSAL].view(1, 1, -1)
        ).to(embed_dtype)
        last_valid = attention_mask.long().sum(dim=1).clamp_min(1) - 1
        thought_embeds: List[Tensor] = []
        thought_states: List[Tensor] = []
        for index in range(self.config.latent_thoughts):
            if thought_embeds:
                appended = torch.cat(thought_embeds, dim=1)
                embeds = torch.cat((prompt_embeds, appended), dim=1)
                mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            batch,
                            appended.shape[1],
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ),
                    dim=1,
                )
            else:
                embeds, mask = prompt_embeds, attention_mask
            hidden = self.backbone(
                inputs_embeds=embeds, attention_mask=mask, attention_mode="causal"
            )
            if index == 0:
                state = hidden[torch.arange(batch, device=hidden.device), last_valid]
            else:
                state = hidden[:, -1]
            thought_states.append(state.unsqueeze(1))
            projected = self.latent_projection(state.float()).to(hidden.dtype)
            thought_embeds.append(
                (self.ground_latents(projected) + mode.squeeze(1)).unsqueeze(1).to(embed_dtype)
            )
        return torch.cat(thought_embeds, dim=1), torch.cat(thought_states, dim=1)

    def student_channel_teacher_force(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        thought_embeds: Tensor,
        thought_states: Optional[Tensor],
        target_ids: Tensor,
        target_attention_mask: Tensor,
        *,
        use_prefix_slots: bool = True,
        collect_hidden_states: bool = False,
        route_index: Optional[Tensor] = None,
        thought_attention_mask: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """Teacher-force answer targets through the Version 10.0 channel.

        The sequence is ``[prompt][K thoughts][cue][answer]``; the answer
        additionally reads the thought states through the gated per-layer
        K/V slots when ``use_prefix_slots`` is set (the audit's slot-ablation
        arm sets it False; the shuffled-prefix arm permutes
        ``thought_embeds``/``thought_states`` within family before calling).
        ``collect_hidden_states`` returns the per-layer hidden states at the
        pre-answer position, the CODI distillation read-out.
        """

        batch = input_ids.shape[0]
        causal_mode = self.mode_embedding.weight[self.MODE_CAUSAL].view(1, 1, -1)
        parts = [self.backbone.embed_tokens(input_ids) + causal_mode]
        masks = [attention_mask]
        parts.append(thought_embeds.to(parts[0].dtype))
        if thought_attention_mask is not None:
            # The teacher pass reuses this method with right-padded gold
            # trace embeddings in the thoughts slot; its mask must be real.
            masks.append(thought_attention_mask)
        else:
            masks.append(
                torch.ones(
                    batch,
                    thought_embeds.shape[1],
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
            )
        if self.config.response_cue_ids is not None:
            cue = torch.tensor(
                self.config.response_cue_ids,
                dtype=torch.long,
                device=input_ids.device,
            ).unsqueeze(0).expand(batch, -1)
            parts.append(self.backbone.embed_tokens(cue) + causal_mode)
            masks.append(
                torch.ones(
                    batch,
                    cue.shape[1],
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
            )
        parts.append(self.backbone.embed_tokens(target_ids) + causal_mode)
        masks.append(target_attention_mask)
        combined = torch.cat(
            [part.to(self.backbone.embed_tokens.weight.dtype) for part in parts], dim=1
        )
        combined_mask = torch.cat(masks, dim=1)
        prefix_kv: Optional[List[Tuple[Tensor, Tensor]]] = None
        if use_prefix_slots and self.config.kv_prefix_slots > 0:
            if thought_states is None:
                raise ValueError("prefix slots need thought states")
            prefix_kv = self.kv_prefix_projector(thought_states)
        outputs = self.backbone(
            inputs_embeds=combined,
            attention_mask=combined_mask,
            attention_mode="causal",
            prefix_kv=prefix_kv,
            collect_hidden_states=collect_hidden_states,
            route_index=route_index,
        )
        layer_hiddens: Optional[List[Tensor]] = None
        if collect_hidden_states:
            hidden, layer_hiddens = outputs  # type: ignore[misc]
        else:
            hidden = outputs
        hidden = self._apply_root(hidden)
        target_length = target_ids.shape[1]
        logits = self.lm_head(hidden[:, -target_length - 1 : -1].to(self.lm_head.weight.dtype))
        result: Dict[str, Any] = {
            "logits": logits,
            "target_hidden": hidden[:, -target_length:],
        }
        # Logits over the thought/trace segment: position ``prompt_len + i - 1``
        # predicts segment token ``i`` (the teacher pass reads its trace CE
        # here; the CoLaR pass reads its per-window CE here).
        prompt_length = input_ids.shape[1]
        segment_length = thought_embeds.shape[1]
        result["segment_logits"] = self.lm_head(
            hidden[:, prompt_length - 1 : prompt_length + segment_length - 1].to(
                self.lm_head.weight.dtype
            )
        )
        if layer_hiddens is not None:
            # The pre-answer read-out position: the last token before the
            # answer span (the final cue token when a cue is configured).
            result["pre_answer_layer_states"] = [
                states[:, -target_length - 1] for states in layer_hiddens
            ]
        return result

    def reconstruct_from_thoughts(
        self,
        thought_embeds: Tensor,
        target_ids: Tensor,
        target_attention_mask: Tensor,
    ) -> Tensor:
        """Reconstruction CE: decode withheld content from the thoughts alone.

        The sequence is ``[K thoughts][recon cue][targets]`` — no prompt, so
        the only source for the targets is the latent channel.  This is the
        dense objective every working prefix-content system in the
        literature has and Studies 1--10 lacked; it is the warm-phase loss
        and a continuing term in the main phase.
        """

        batch = thought_embeds.shape[0]
        cue = self.recon_cue_embedding.view(1, 1, -1).expand(batch, 1, -1)
        causal_mode = self.mode_embedding.weight[self.MODE_CAUSAL].view(1, 1, -1)
        target_embeds = self.backbone.embed_tokens(target_ids.clamp_min(0)) + causal_mode
        combined = torch.cat(
            (thought_embeds, cue.to(thought_embeds.dtype), target_embeds.to(thought_embeds.dtype)),
            dim=1,
        ).to(self.backbone.embed_tokens.weight.dtype)
        ones = torch.ones(
            batch,
            thought_embeds.shape[1] + 1,
            dtype=target_attention_mask.dtype,
            device=target_attention_mask.device,
        )
        mask = torch.cat((ones, target_attention_mask), dim=1)
        hidden = self.backbone(
            inputs_embeds=combined, attention_mask=mask, attention_mode="causal"
        )
        hidden = self._apply_root(hidden)
        target_length = target_ids.shape[1]
        logits = self.lm_head(hidden[:, -target_length - 1 : -1].to(self.lm_head.weight.dtype))
        losses = _masked_token_cross_entropy(
            logits, target_ids.clamp_min(0), target_attention_mask
        )
        denominator = target_attention_mask.to(losses.dtype).sum().clamp_min(1.0)
        return losses.sum() / denominator

    def latent_step_logits(self, thought_states: Tensor) -> Tensor:
        """Per-latent decode logits (SIM-CoT budget form) via the frozen head."""

        return self.lm_head(self.latent_step_head(thought_states.float()).to(
            self.lm_head.weight.dtype
        ))

    def _teacher_force_workspace(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        workspace_prefix: Tensor,
        target_ids: Tensor,
        target_attention_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        return self._channel_teacher_force(
            input_ids,
            attention_mask,
            workspace_prefix,
            target_ids,
            target_attention_mask,
        )

    def _teacher_force_causal(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        target_ids: Tensor,
        target_attention_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        return self._channel_teacher_force(
            input_ids, attention_mask, None, target_ids, target_attention_mask
        )

    def causal_answer_mean_logprob(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        answer_ids: Tensor,
        answer_attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Per-row mean causal-channel log-probability of an answer.

        This is both the fourth publish-combiner feature (as
        ``exp(mean logprob)``) and the max-logprob baseline's score, so the
        baseline is nested inside the calibrated rule by construction.
        """

        if answer_attention_mask is None:
            answer_attention_mask = torch.ones_like(answer_ids)
        logits, _ = self._teacher_force_causal(
            input_ids, attention_mask, answer_ids, answer_attention_mask
        )
        log_probabilities = F.log_softmax(logits.float(), dim=-1).gather(
            -1, answer_ids[:, :, None]
        ).squeeze(-1)
        weights = answer_attention_mask.to(log_probabilities.dtype)
        return (log_probabilities * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)

    def candidate_policy_features(
        self,
        candidate_hidden: Tensor,
        candidate_logits: Tensor,
        candidate_attention_mask: Tensor,
        global_state: Tensor,
        verification_error: Tensor,
    ) -> Tensor:
        """Return the exact commitment-head input for one frozen candidate.

        The on-policy policy-head phase caches these features for generated
        train-anchor emissions and their graded hard negatives, so head
        training and inference score candidates through one representation.
        """

        probabilities = candidate_logits.float().softmax(dim=-1)
        entropy = -(
            probabilities * probabilities.clamp_min(1.0e-9).log()
        ).sum(dim=-1)
        normalized_entropy = entropy / max(1.0, math.log(self.config.vocab_size))
        pooled = self._masked_mean(candidate_hidden, candidate_attention_mask)
        verifier_mean = verification_error.mean(dim=1)
        entropy_mean = (
            normalized_entropy
            * candidate_attention_mask.to(normalized_entropy.dtype)
        ).sum(dim=-1) / candidate_attention_mask.sum(dim=-1).clamp_min(1)
        return torch.cat(
            (
                pooled,
                global_state,
                verifier_mean[:, None],
                entropy_mean[:, None],
            ),
            dim=-1,
        )

    def _score_candidate(
        self,
        candidate_hidden: Tensor,
        candidate_logits: Tensor,
        candidate_attention_mask: Tensor,
        global_state: Tensor,
        verification_error: Tensor,
    ) -> Tensor:
        return self.commitment_head(
            self.candidate_policy_features(
                candidate_hidden,
                candidate_logits,
                candidate_attention_mask,
                global_state,
                verification_error,
            )
        )

    def _commit_mask(
        self,
        commitment_logits: Tensor,
        logprob_feature: Optional[Tensor] = None,
        agreement_feature: Optional[Tensor] = None,
    ) -> Tensor:
        """Authorize only candidates accepted by the calibrated publish rule.

        The third logit scores the frozen public candidate itself.  Earlier
        prototypes reused private-canvas token error as a hard publication
        gate, which rejected semantically correct generated answers whenever
        their private denoising trajectory happened to be difficult.
        """

        if commitment_logits.ndim != 2 or commitment_logits.shape[-1] != 3:
            raise ValueError("candidate policy logits must have shape [batch, 3]")
        weights = self.config.publish_weights
        required = 0 if weights is None else len(weights) - 3
        available = int(logprob_feature is not None) + int(agreement_feature is not None)
        if weights is not None and available >= required:
            return (
                self.publish_score(commitment_logits, logprob_feature, agreement_feature)
                >= self.config.publish_threshold
            )
        commit_probability = torch.sigmoid(commitment_logits[:, 0])
        risk_probability = torch.sigmoid(commitment_logits[:, 1])
        verifier_probability = torch.sigmoid(commitment_logits[:, 2])
        return (
            (commit_probability >= self.config.commitment_threshold)
            & (risk_probability <= self.config.risk_threshold)
            & (verifier_probability <= self.config.verifier_error_threshold)
        )

    def publish_score(
        self,
        commitment_logits: Tensor,
        logprob_feature: Optional[Tensor] = None,
        agreement_feature: Optional[Tensor] = None,
    ) -> Tensor:
        """One calibrated scalar per candidate: selection and publication rule.

        With fitted ``publish_weights`` this is a logistic combination of the
        head probabilities, plus (four weights, Version 8.0) the candidate's
        ``exp(causal mean logprob)`` so the max-logprob baseline is nested in
        the rule, plus (five weights, Version 9.0) the within-pool agreement
        fraction so plurality voting is nested as well.  Before calibration it
        falls back to a monotone blend so N-best selection is still well
        defined during collection.
        """

        if commitment_logits.ndim != 2 or commitment_logits.shape[-1] != 3:
            raise ValueError("candidate policy logits must have shape [batch, 3]")
        probabilities = torch.sigmoid(commitment_logits.float())
        weights = self.config.publish_weights
        if weights is not None:
            columns = [probabilities]
            if len(weights) >= 4:
                if logprob_feature is None:
                    raise ValueError(
                        "a %d-weight publish rule requires the logprob feature" % len(weights)
                    )
                columns.append(logprob_feature.float().view(-1, 1))
            if len(weights) == 5:
                if agreement_feature is None:
                    raise ValueError(
                        "a five-weight publish rule requires the agreement feature"
                    )
                columns.append(agreement_feature.float().view(-1, 1))
            features = torch.cat(columns, dim=-1)
            weight_tensor = features.new_tensor(weights)
            return torch.sigmoid(
                features @ weight_tensor + float(self.config.publish_bias)
            )
        return (
            probabilities[:, 0]
            * (1.0 - probabilities[:, 1])
            * (1.0 - probabilities[:, 2])
        )

    def forward_hlwm(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        *,
        target_ids: Optional[Tensor] = None,
        lane_target_ids: Optional[Tensor] = None,
        target_attention_mask: Optional[Tensor] = None,
        lane_target_attention_mask: Optional[Tensor] = None,
        negative_target_ids: Optional[Tensor] = None,
        negative_target_attention_mask: Optional[Tensor] = None,
        commitment_supervision_mask: Optional[Tensor] = None,
        noisy_lane_ids: Optional[Tensor] = None,
        timesteps: Optional[Tensor] = None,
        lane_briefs: Optional[Tensor] = None,
        lane_brief_ids: Optional[Tensor] = None,
        lane_brief_attention_mask: Optional[Tensor] = None,
        canvas_length: Optional[int] = None,
        sample_reverse: bool = False,
        adaptive_halt: Optional[bool] = None,
        full_reverse_schedule: Optional[bool] = None,
        route_override: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        answer_input_ids: Optional[Tensor] = None,
        answer_attention_mask: Optional[Tensor] = None,
        skip_synthesis_kl: bool = False,
        premise_ids: Optional[Tensor] = None,
        premise_attention_mask: Optional[Tensor] = None,
        premise_negative_ids: Optional[Tensor] = None,
        prefix_source: str = "workspace",
    ) -> HLWMOutput:
        """Run private refinement and create a prefix for Qwen answer decoding.

        Version 9.0 information asymmetry: the workspace always builds from
        ``input_ids`` (full context); when ``answer_input_ids`` is provided,
        every answer-channel operation (teacher forcing, negatives, the KL
        reference) conditions on it instead, so on masked rows the prefix is
        the only path from the withheld premise to the answer.
        ``prefix_source="gist"`` swaps the workspace prefix for the
        attribution-control gist prefix in the answer channel only.
        """

        if prefix_source not in ("workspace", "gist"):
            raise ValueError("prefix_source must be 'workspace' or 'gist'")
        batch = input_ids.shape[0]
        lanes = self.config.num_lanes
        device = input_ids.device
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()
        if answer_input_ids is not None:
            if answer_input_ids.shape[0] != batch:
                raise ValueError("answer_input_ids batch mismatch")
            if answer_attention_mask is None:
                answer_attention_mask = (
                    answer_input_ids != self.config.pad_token_id
                ).long()
        channel_input_ids = answer_input_ids if answer_input_ids is not None else input_ids
        channel_attention_mask = (
            answer_attention_mask if answer_input_ids is not None else attention_mask
        )
        if target_ids is not None and (target_ids.ndim != 2 or target_ids.shape[0] != batch):
            raise ValueError("target_ids must have shape [batch, answer_length]")
        if lane_target_ids is not None and (
            lane_target_ids.ndim != 3 or lane_target_ids.shape[:2] != (batch, lanes)
        ):
            raise ValueError("lane_target_ids must have shape [batch, num_lanes, length]")
        inferred_length = (
            lane_target_ids.shape[-1]
            if lane_target_ids is not None
            else noisy_lane_ids.shape[-1]
            if noisy_lane_ids is not None
            else target_ids.shape[-1]
            if target_ids is not None
            else canvas_length
        )
        if inferred_length is None or int(inferred_length) <= 0:
            raise ValueError("provide lane targets, noise, answer targets, or canvas_length")
        canvas_length = int(inferred_length)
        if lane_target_ids is None and target_ids is not None:
            if target_ids.shape[-1] != canvas_length:
                raise ValueError("target length cannot initialize the private canvas")
            lane_target_ids = target_ids[:, None, :].expand(-1, lanes, -1)
        if lane_target_ids is not None and lane_target_ids.shape[-1] != canvas_length:
            raise ValueError("lane target length mismatch")
        if target_ids is not None:
            if target_attention_mask is None:
                target_attention_mask = (target_ids != self.config.pad_token_id).long()
            if target_attention_mask.shape != target_ids.shape:
                raise ValueError("target_attention_mask must match target_ids")
        if negative_target_ids is not None:
            if target_ids is None or negative_target_ids.shape != target_ids.shape:
                raise ValueError("negative_target_ids must match target_ids")
            if negative_target_attention_mask is None:
                negative_target_attention_mask = (
                    negative_target_ids != self.config.pad_token_id
                ).long()
            if negative_target_attention_mask.shape != negative_target_ids.shape:
                raise ValueError(
                    "negative_target_attention_mask must match negative_target_ids"
                )
        if commitment_supervision_mask is not None:
            if commitment_supervision_mask.shape != (batch,):
                raise ValueError("commitment_supervision_mask must have shape [batch]")
            commitment_supervision_mask = commitment_supervision_mask.to(
                device=device, dtype=torch.bool
            )
        if lane_target_attention_mask is None:
            if lane_target_ids is not None:
                lane_target_attention_mask = (
                    lane_target_ids != self.config.pad_token_id
                ).long()
            else:
                lane_target_attention_mask = torch.ones(
                    batch,
                    lanes,
                    canvas_length,
                    dtype=torch.long,
                    device=device,
                )
        if lane_target_attention_mask.shape != (batch, lanes, canvas_length):
            raise ValueError("lane_target_attention_mask has the wrong shape")

        context_hidden = self._encode_committed(input_ids, attention_mask)
        context_summary = self._masked_mean(context_hidden, attention_mask)
        global_state = context_summary
        briefs = self._prepare_briefs(
            context_hidden,
            attention_mask,
            global_state,
            lane_briefs=lane_briefs,
            lane_brief_ids=lane_brief_ids,
            lane_brief_attention_mask=lane_brief_attention_mask,
        )

        supervised_private = lane_target_ids is not None
        if timesteps is None:
            timesteps = torch.full(
                (batch,), self.config.diffusion_steps, dtype=torch.long, device=device
            )
        else:
            timesteps = timesteps.to(device=device, dtype=torch.long)
        if timesteps.shape != (batch,):
            raise ValueError("timesteps must have shape [batch]")

        if noisy_lane_ids is None:
            if lane_target_ids is not None:
                flat_targets = lane_target_ids.reshape(batch * lanes, canvas_length)
                flat_timesteps = timesteps[:, None].expand(-1, lanes).reshape(-1)
                noisy_lane_ids = self.diffusion.q_sample(
                    flat_targets, flat_timesteps, generator
                ).reshape(batch, lanes, canvas_length)
            else:
                noisy_lane_ids = self.diffusion.sample_noise(
                    (batch, lanes, canvas_length), device, generator
                )
        if noisy_lane_ids.shape != (batch, lanes, canvas_length):
            raise ValueError("noisy_lane_ids has the wrong shape")
        noisy_lane_ids = noisy_lane_ids.to(device=device, dtype=torch.long)
        initial_corruption = noisy_lane_ids.clone()

        if full_reverse_schedule is None:
            full_reverse_schedule = not supervised_private
        if full_reverse_schedule:
            reverse_timesteps = torch.arange(
                self.config.diffusion_steps,
                0,
                -1,
                dtype=torch.long,
                device=device,
            )[None, :].expand(batch, -1)
            adaptive_halt = False
        else:
            offsets = torch.arange(
                self.config.max_refinement_steps, device=device, dtype=torch.long
            )
            reverse_timesteps = (timesteps[:, None] - offsets[None, :]).clamp_min(1)
            if adaptive_halt is None:
                adaptive_halt = not self.training
        private = self._private_recurrence(
            context_hidden,
            attention_mask,
            global_state,
            briefs,
            noisy_lane_ids,
            reverse_timesteps,
            lane_target_ids,
            lane_target_attention_mask,
            sample_reverse=sample_reverse,
            adaptive_halt=bool(adaptive_halt),
            advance_final_state=True,
            generator=generator,
            route_override=route_override,
        )

        global_state, workspace_prefix, verification_error = self._build_workspace(
            private, global_state, context_summary, lane_target_attention_mask
        )
        synthesis_logits: Optional[Tensor] = None
        negative_commitment_logits: Optional[Tensor] = None
        commitment_logits = global_state.new_zeros((batch, 3))
        commit_mask = torch.zeros(batch, dtype=torch.bool, device=device)
        committed_ids = torch.empty(batch, 0, dtype=torch.long, device=device)
        loss_components: Dict[str, Tensor] = {}

        if lane_target_ids is not None and private["nll_history"]:
            nll_tensor = torch.stack(private["nll_history"], dim=-1)
            loss_components["denoise"] = nll_tensor.mean()
            verification_target = (
                private["denoise_logits"].detach().argmax(dim=-1) != lane_target_ids
            ).to(private["verification_logits"].dtype)
            verification_token_loss = F.binary_cross_entropy_with_logits(
                private["verification_logits"], verification_target, reduction="none"
            )
            verification_mask = lane_target_attention_mask.to(
                verification_token_loss.dtype
            )
            loss_components["verification"] = (
                verification_token_loss * verification_mask
            ).sum() / verification_mask.sum().clamp_min(1.0)

            halt_logits = private["halt_logits"]
            if nll_tensor.shape[-1] > 1:
                next_gain = nll_tensor[..., :-1] - nll_tensor[..., 1:]
                halt_target = (
                    next_gain <= self.config.halt_compute_cost
                ).to(halt_logits.dtype)
                halt_target = torch.cat(
                    (halt_target, torch.ones_like(halt_target[..., :1])), dim=-1
                )
            else:
                halt_target = torch.ones_like(halt_logits)
            loss_components["halt"] = F.binary_cross_entropy_with_logits(
                halt_logits, halt_target.detach()
            )
            loss_components["router_balance"] = self._router_balance_loss(private)
            loss_components["router_entropy"] = self._router_marginal_entropy_loss(
                private
            )
            if self.config.expert_diversity_weight > 0:
                loss_components["expert_diversity"] = self.expert_bank.output_diversity(
                    context_summary
                )
            if lanes > 1:
                normalized_lanes = F.normalize(private["lane_summaries"].float(), dim=-1)
                similarities = torch.matmul(
                    normalized_lanes, normalized_lanes.transpose(1, 2)
                )
                off_diagonal = ~torch.eye(
                    lanes, dtype=torch.bool, device=device
                )[None, :, :]
                # Squared cosine targets genuinely different (near-orthogonal)
                # summaries instead of rewarding an equally degenerate -1
                # anti-correlation between the two lanes.
                loss_components["lane_diversity"] = similarities.masked_select(
                    off_diagonal
                ).square().mean()

        answer_prefix = workspace_prefix
        if prefix_source == "gist":
            answer_prefix = self._build_gist_prefix(context_hidden, attention_mask)
            if answer_prefix.shape[1] != workspace_prefix.shape[1]:
                raise ValueError(
                    "gist prefix budget %d must match the workspace prefix %d"
                    % (answer_prefix.shape[1], workspace_prefix.shape[1])
                )
        if (
            premise_ids is not None
            and premise_attention_mask is not None
            and premise_negative_ids is not None
            and self.config.premise_aux_weight > 0
            and prefix_source == "workspace"
        ):
            premise_result = self._premise_probe_loss(
                workspace_prefix, premise_ids, premise_attention_mask, premise_negative_ids
            )
            if premise_result is not None:
                loss_components["premise_aux"] = premise_result[0]
                loss_components["premise_aux_accuracy"] = premise_result[1]

        if target_ids is not None:
            assert target_attention_mask is not None
            synthesis_logits, synthesis_hidden = self._teacher_force_workspace(
                channel_input_ids,
                channel_attention_mask,
                answer_prefix,
                target_ids,
                target_attention_mask,
            )
            token_loss = _masked_token_cross_entropy(
                synthesis_logits, target_ids, target_attention_mask
            )
            target_mask_float = target_attention_mask.to(token_loss.dtype)
            loss_components["synthesis"] = token_loss.sum() / target_mask_float.sum().clamp_min(1.0)
            if self.config.synthesis_kl_weight > 0 and not skip_synthesis_kl:
                # KL-to-causal anchor (Version 8.0): bound the workspace
                # channel's register drift against the same-model causal
                # distribution on the same targets.  The causal reference is
                # detached — the anchor may move only the workspace side.
                # Version 9.0 skips it on masked rows, where the reference
                # cannot answer and its pull is exactly anti-mechanism.
                with torch.no_grad():
                    causal_reference_logits, _ = self._teacher_force_causal(
                        channel_input_ids,
                        channel_attention_mask,
                        target_ids,
                        target_attention_mask,
                    )
                kl_per_token = F.kl_div(
                    F.log_softmax(synthesis_logits.float(), dim=-1),
                    F.log_softmax(causal_reference_logits.float(), dim=-1),
                    log_target=True,
                    reduction="none",
                ).sum(dim=-1)
                loss_components["synthesis_kl"] = (
                    kl_per_token * target_mask_float
                ).sum() / target_mask_float.sum().clamp_min(1.0)
            commitment_logits = self._score_candidate(
                synthesis_hidden,
                synthesis_logits,
                target_attention_mask,
                global_state,
                verification_error,
            )
            if negative_target_ids is not None:
                assert negative_target_attention_mask is not None
                negative_logits, negative_hidden = self._teacher_force_workspace(
                    channel_input_ids,
                    channel_attention_mask,
                    answer_prefix,
                    negative_target_ids,
                    negative_target_attention_mask,
                )
                negative_commitment_logits = self._score_candidate(
                    negative_hidden,
                    negative_logits,
                    negative_target_attention_mask,
                    global_state,
                    verification_error,
                )
                clean_labels = torch.stack(
                    (
                        torch.ones(batch, device=device),
                        torch.zeros(batch, device=device),
                        torch.zeros(batch, device=device),
                    ),
                    dim=-1,
                ).to(commitment_logits.dtype)
                corrupt_labels = 1.0 - clean_labels
                paired_bce = F.binary_cross_entropy_with_logits(
                    commitment_logits, clean_labels, reduction="none"
                ).mean(dim=-1) + F.binary_cross_entropy_with_logits(
                    negative_commitment_logits, corrupt_labels, reduction="none"
                ).mean(dim=-1)
                rank_loss = F.relu(
                    1.0 - commitment_logits[:, 0] + negative_commitment_logits[:, 0]
                ) + F.relu(
                    1.0 - negative_commitment_logits[:, 1] + commitment_logits[:, 1]
                ) + F.relu(
                    1.0 - negative_commitment_logits[:, 2] + commitment_logits[:, 2]
                )
                paired_objective = paired_bce + 0.5 * rank_loss
                if commitment_supervision_mask is None:
                    loss_components["commitment"] = paired_objective.mean()
                else:
                    weights = commitment_supervision_mask.to(paired_objective.dtype)
                    loss_components["commitment"] = (
                        paired_objective * weights
                    ).sum() / weights.sum().clamp_min(1.0)
            commit_mask = self._commit_mask(commitment_logits)
            committed_ids = torch.where(
                commit_mask[:, None],
                target_ids,
                torch.full_like(target_ids, self.config.abstain_token_id),
            )

        total_loss: Optional[Tensor] = None
        if loss_components:
            total_loss = global_state.new_zeros(())
            weight_map = {
                "denoise": self.config.denoise_loss_weight,
                "synthesis": self.config.synthesis_loss_weight,
                "synthesis_kl": self.config.synthesis_kl_weight,
                "verification": self.config.verification_loss_weight,
                "commitment": self.config.commitment_loss_weight,
                "halt": self.config.halt_loss_weight,
                "router_balance": self.config.router_aux_weight,
                "router_entropy": self.config.router_entropy_weight,
                "expert_diversity": self.config.expert_diversity_weight,
                "lane_diversity": self.config.lane_diversity_weight,
                "premise_aux": self.config.premise_aux_weight,
                "premise_aux_accuracy": 0.0,
            }
            for name, value in loss_components.items():
                total_loss = total_loss + weight_map[name] * value

        return HLWMOutput(
            loss=total_loss,
            loss_components=loss_components,
            denoise_logits=private["denoise_logits"],
            synthesis_logits=synthesis_logits,
            verification_logits=private["verification_logits"],
            commitment_logits=commitment_logits,
            negative_commitment_logits=negative_commitment_logits,
            halt_logits=private["halt_logits"],
            route_indices=private["route_indices"],
            router_probabilities=private["router_probabilities"],
            route_path_masks=private["route_path_masks"],
            lane_summaries=private["lane_summaries"],
            fast_state=private["fast_state"],
            slow_state=private["slow_state"],
            global_state=global_state,
            final_canvas_probabilities=private["canvas_probabilities"],
            corrupted_ids=initial_corruption,
            timesteps=timesteps,
            reverse_timesteps=private["transition_timesteps"],
            commit_mask=commit_mask,
            committed_ids=committed_ids,
            workspace_prefix=workspace_prefix,
        )

    @torch.no_grad()
    def _decode_candidate(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        workspace_prefix: Optional[Tensor],
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Autoregressively decode one candidate through a channel (KV-cached).

        ``workspace_prefix=None`` decodes the plain causal channel; a prefix
        decodes the workspace channel.  Both continue from the hard response
        cue — no BOS token is ever inserted (the Version 6.0 mid-sequence
        BOS resolved to Qwen's end-of-document token and caused the scaffold
        emissions).  Temperature zero is greedy; positive samples.
        """

        prefill_embeds, prefill_mask = self._channel_prefill(
            input_ids, attention_mask, workspace_prefix
        )
        mode = self.mode_embedding.weight[self.MODE_CAUSAL].view(1, 1, -1)
        hidden, past = self.backbone(
            inputs_embeds=prefill_embeds,
            attention_mask=prefill_mask,
            attention_mode="causal",
            use_cache=True,
        )
        next_logits = self.lm_head(self._apply_root(hidden)[:, -1].to(self.lm_head.weight.dtype))
        running_mask = prefill_mask
        candidate_tokens = []
        sample = temperature > 0.0
        for _ in range(max_new_tokens):
            step_logits = next_logits
            if sample:
                step_logits = step_logits / temperature
            if top_k is not None and 0 < top_k < step_logits.shape[-1]:
                threshold = step_logits.topk(top_k, dim=-1).values[:, -1, None]
                step_logits = step_logits.masked_fill(step_logits < threshold, -torch.inf)
            if sample:
                next_token = torch.multinomial(
                    step_logits.softmax(dim=-1), 1, generator=generator
                )
            else:
                next_token = step_logits.argmax(dim=-1, keepdim=True)
            candidate_tokens.append(next_token)
            if int(next_token.item()) == self.config.eos_token_id:
                break
            running_mask = torch.cat(
                (running_mask, torch.ones_like(next_token)), dim=1
            )
            hidden, past = self.backbone(
                inputs_embeds=(self.backbone.embed_tokens(next_token) + mode).to(
                    self.backbone.embed_tokens.weight.dtype
                ),
                attention_mask=running_mask,
                attention_mode="causal",
                past_key_values=past,
                use_cache=True,
            )
            next_logits = self.lm_head(self._apply_root(hidden)[:, -1].to(self.lm_head.weight.dtype))
        return torch.cat(candidate_tokens, dim=1)

    @torch.no_grad()
    def decode_candidate_v10(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        thought_embeds: Tensor,
        thought_states: Optional[Tensor],
        *,
        use_prefix_slots: bool = True,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        route_index: Optional[Tensor] = None,
    ) -> Tensor:
        """KV-cached decode through the Version 10.0 channel.

        Prefill is ``[prompt][K thoughts][cue]`` — the same sequence the
        training path teacher-forces — and the per-layer K/V slots are
        supplied at prefill and at every cached step (slots are position-free
        constants, so passing them per step is exactly equivalent to the
        uncached full forward; the golden parity test asserts this).
        """

        batch = input_ids.shape[0]
        mode = self.mode_embedding.weight[self.MODE_CAUSAL].view(1, 1, -1)
        parts = [self.backbone.embed_tokens(input_ids) + mode]
        masks = [attention_mask]
        parts.append(thought_embeds.to(parts[0].dtype))
        masks.append(
            torch.ones(
                batch,
                thought_embeds.shape[1],
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
        )
        if self.config.response_cue_ids is not None:
            cue = torch.tensor(
                self.config.response_cue_ids, dtype=torch.long, device=input_ids.device
            ).unsqueeze(0).expand(batch, -1)
            parts.append(self.backbone.embed_tokens(cue) + mode)
            masks.append(
                torch.ones(
                    batch, cue.shape[1], dtype=attention_mask.dtype, device=attention_mask.device
                )
            )
        prefix_kv: Optional[List[Tuple[Tensor, Tensor]]] = None
        if use_prefix_slots and self.config.kv_prefix_slots > 0:
            if thought_states is None:
                raise ValueError("prefix slots need thought states")
            prefix_kv = self.kv_prefix_projector(thought_states)
        embed_dtype = self.backbone.embed_tokens.weight.dtype
        hidden, past = self.backbone(
            inputs_embeds=torch.cat([part.to(embed_dtype) for part in parts], dim=1),
            attention_mask=torch.cat(masks, dim=1),
            attention_mode="causal",
            use_cache=True,
            prefix_kv=prefix_kv,
            route_index=route_index,
        )
        next_logits = self.lm_head(self._apply_root(hidden)[:, -1].to(self.lm_head.weight.dtype))
        running_mask = torch.cat(masks, dim=1)
        candidate_tokens = []
        sample = temperature > 0.0
        for _ in range(max_new_tokens):
            step_logits = next_logits
            if sample:
                step_logits = step_logits / temperature
            if top_k is not None and 0 < top_k < step_logits.shape[-1]:
                threshold = step_logits.topk(top_k, dim=-1).values[:, -1, None]
                step_logits = step_logits.masked_fill(step_logits < threshold, -torch.inf)
            if sample:
                next_token = torch.multinomial(
                    step_logits.softmax(dim=-1), 1, generator=generator
                )
            else:
                next_token = step_logits.argmax(dim=-1, keepdim=True)
            candidate_tokens.append(next_token)
            if batch == 1 and int(next_token.item()) == self.config.eos_token_id:
                break
            running_mask = torch.cat((running_mask, torch.ones_like(next_token)), dim=1)
            hidden, past = self.backbone(
                inputs_embeds=(self.backbone.embed_tokens(next_token) + mode).to(embed_dtype),
                attention_mask=running_mask,
                attention_mode="causal",
                past_key_values=past,
                use_cache=True,
                prefix_kv=prefix_kv,
                route_index=route_index,
            )
            next_logits = self.lm_head(self._apply_root(hidden)[:, -1].to(self.lm_head.weight.dtype))
        return torch.cat(candidate_tokens, dim=1)

    def route_family_logits(self, thought_states: Tensor) -> Tensor:
        """Detached router probe over the prompt's last hidden state.

        ``thought_states[:, 0]`` is the hidden state at the last valid
        PROMPT position (a function of the prompt alone), so the probe
        predicts the family from exactly what a deployment router would see.
        The detach keeps StableMoE stage-1 discipline: the probe learns the
        deterministic route and never shapes it.
        """

        return self.family_router(thought_states[:, 0].detach().float())

    @torch.no_grad()
    def generate_hlwm_nbest(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        *,
        canvas_length: int = 64,
        max_new_tokens: int = 64,
        candidate_temperatures: Sequence[float] = (0.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8),
        top_k: Optional[int] = None,
        lane_briefs: Optional[Tensor] = None,
        lane_brief_ids: Optional[Tensor] = None,
        lane_brief_attention_mask: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
        disable_workspace_memory: bool = False,
        disable_workspace_prefix: bool = False,
        return_features: bool = False,
        candidate_channel: str = "causal",
        answer_input_ids: Optional[Tensor] = None,
        answer_attention_mask: Optional[Tensor] = None,
        prefix_source: str = "workspace",
    ) -> HLWMNBestGeneration:
        """Workspace-scored N-best: one workspace, N candidates, calibrated selection.

        Version 8.0 decouples the claims: candidates default to the *causal*
        channel — the same healthy pool self-consistency votes over — while
        the workspace heads score every candidate through the identical
        feature path used at head training.  ``candidate_channel="workspace"``
        decodes through the (gated) latent prefix instead, which is the
        generation-parity surface.  The highest calibrated publish score wins
        and is published only if it clears the fitted publication rule.
        ``disable_workspace_memory`` slices the prefix back to the narrow
        Version 5.x interface, giving the read-out channel its causal
        ablation inside the same run.
        """

        if input_ids.shape[0] != 1:
            raise ValueError("generate_hlwm_nbest currently supports batch size one")
        if max_new_tokens <= 0 or not len(candidate_temperatures):
            raise ValueError("token budget and candidate list must be nonempty")
        if candidate_channel not in ("causal", "workspace"):
            raise ValueError("candidate_channel must be 'causal' or 'workspace'")
        if prefix_source not in ("workspace", "gist"):
            raise ValueError("prefix_source must be 'workspace' or 'gist'")
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()
        if answer_input_ids is not None and answer_attention_mask is None:
            answer_attention_mask = (
                answer_input_ids != self.config.pad_token_id
            ).long()
        channel_input_ids = answer_input_ids if answer_input_ids is not None else input_ids
        channel_attention_mask = (
            answer_attention_mask if answer_input_ids is not None else attention_mask
        )
        workspace = self.forward_hlwm(
            input_ids,
            attention_mask,
            canvas_length=canvas_length,
            lane_briefs=lane_briefs,
            lane_brief_ids=lane_brief_ids,
            lane_brief_attention_mask=lane_brief_attention_mask,
            sample_reverse=False,
            adaptive_halt=False,
            full_reverse_schedule=True,
            generator=generator,
        )
        prefix: Optional[Tensor] = workspace.workspace_prefix
        if prefix_source == "gist":
            context_hidden = self._encode_committed(input_ids, attention_mask)
            prefix = self._build_gist_prefix(context_hidden, attention_mask)
        if disable_workspace_memory and prefix is not None:
            # Descriptive second arm: memory tokens only (the exact slice the
            # Version 8.0 audit shipped as its "ablation").
            prefix = prefix[:, : self.config.synthesis_prefix_tokens]
        if disable_workspace_prefix:
            # Version 9.0 full ablation: the causal arm IS the workspace arm
            # with the whole prefix removed.
            prefix = None
        decode_prefix = prefix if candidate_channel == "workspace" else None
        verification_error = torch.sigmoid(workspace.verification_logits).mean(dim=-1)
        candidate_ids_list: List[Tensor] = []
        features_list: List[Tensor] = []
        commitment_rows: List[Tensor] = []
        commit_probabilities: List[float] = []
        risk_probabilities: List[float] = []
        verifier_probabilities: List[float] = []
        publish_scores: List[float] = []
        candidate_mean_logprobs: List[float] = []
        for temperature in candidate_temperatures:
            candidate_ids = self._decode_candidate(
                channel_input_ids,
                channel_attention_mask,
                decode_prefix,
                max_new_tokens=max_new_tokens,
                temperature=float(temperature),
                top_k=top_k,
                generator=generator,
            )
            candidate_mask = torch.ones_like(candidate_ids)
            # Heads always score through the prefix channel, whichever channel
            # produced the candidate: the same feature path as head training.
            candidate_logits, candidate_hidden = self._teacher_force_workspace(
                channel_input_ids,
                channel_attention_mask,
                prefix,
                candidate_ids,
                candidate_mask,
            )
            features = self.candidate_policy_features(
                candidate_hidden,
                candidate_logits,
                candidate_mask,
                workspace.global_state,
                verification_error,
            )
            commitment_logits = self.commitment_head(features)
            probabilities = torch.sigmoid(commitment_logits[0].float()).cpu().tolist()
            candidate_ids_list.append(candidate_ids)
            if return_features:
                features_list.append(features[0].float().cpu())
            commitment_rows.append(commitment_logits)
            commit_probabilities.append(probabilities[0])
            risk_probabilities.append(probabilities[1])
            verifier_probabilities.append(probabilities[2])
            # Full-context causal mean logprob: the max-logprob baseline's
            # score and the publish rule's fourth feature, one and the same.
            mean_logprob = self.causal_answer_mean_logprob(
                input_ids, attention_mask, candidate_ids, candidate_mask
            )
            candidate_mean_logprobs.append(float(mean_logprob[0].cpu()))
        # Within-pool agreement (Version 9.0 fifth feature): fraction of the
        # pool sharing this candidate's exact answer token signature, so
        # plurality voting is nested inside the calibrated rule.
        eos_id = int(self.config.eos_token_id)

        def _signature(ids: Tensor) -> Tuple[int, ...]:
            values = ids[0].tolist()
            if eos_id in values:
                values = values[: values.index(eos_id)]
            return tuple(values)

        signatures = [_signature(ids) for ids in candidate_ids_list]
        pool_size = float(len(signatures))
        candidate_agreements = [
            sum(1.0 for other in signatures if other == signature) / pool_size
            for signature in signatures
        ]
        for index, commitment_row in enumerate(commitment_rows):
            publish_scores.append(
                float(
                    self.publish_score(
                        commitment_row,
                        commitment_row.new_tensor(
                            [math.exp(candidate_mean_logprobs[index])]
                        ),
                        commitment_row.new_tensor([candidate_agreements[index]]),
                    )[0].cpu()
                )
            )
        selected_index = max(range(len(publish_scores)), key=publish_scores.__getitem__)
        commitment_logits = commitment_rows[selected_index]
        selected_logprob_feature = commitment_logits.new_tensor(
            [math.exp(candidate_mean_logprobs[selected_index])]
        )
        selected_agreement_feature = commitment_logits.new_tensor(
            [candidate_agreements[selected_index]]
        )
        commit_mask = self._commit_mask(
            commitment_logits, selected_logprob_feature, selected_agreement_feature
        )
        committed = bool(commit_mask.item())
        output_ids = (
            candidate_ids_list[selected_index]
            if committed
            else torch.full(
                (1, 1),
                self.config.abstain_token_id,
                dtype=torch.long,
                device=input_ids.device,
            )
        )
        workspace = replace(
            workspace,
            commitment_logits=commitment_logits,
            commit_mask=commit_mask,
            committed_ids=output_ids,
        )
        return HLWMNBestGeneration(
            output_ids=output_ids,
            selected_index=selected_index,
            decision="publish" if committed else "abstain",
            candidate_ids=candidate_ids_list,
            candidate_features=features_list if return_features else None,
            commit_probabilities=commit_probabilities,
            risk_probabilities=risk_probabilities,
            verifier_error_probabilities=verifier_probabilities,
            publish_scores=publish_scores,
            candidate_mean_logprobs=candidate_mean_logprobs,
            temperatures=[float(value) for value in candidate_temperatures],
            private_verifier_error_probability=float(verification_error.mean().cpu()),
            macrocycles=int(workspace.reverse_timesteps.shape[1]),
            workspace=workspace,
            candidate_agreements=candidate_agreements,
        )

    @torch.no_grad()
    def generate_hlwm(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        *,
        canvas_length: int = 64,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        do_sample: bool = False,
        lane_briefs: Optional[Tensor] = None,
        lane_brief_ids: Optional[Tensor] = None,
        lane_brief_attention_mask: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
        answer_input_ids: Optional[Tensor] = None,
        answer_attention_mask: Optional[Tensor] = None,
        disable_workspace_prefix: bool = False,
        prefix_source: str = "workspace",
    ) -> HLWMGeneration:
        """Run full private denoising, then let Qwen autoregressively answer."""

        if input_ids.shape[0] != 1:
            raise ValueError("generate_hlwm currently supports batch size one")
        if max_new_tokens <= 0 or temperature <= 0:
            raise ValueError("generation token count and temperature must be positive")
        if prefix_source not in ("workspace", "gist"):
            raise ValueError("prefix_source must be 'workspace' or 'gist'")
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()
        if answer_input_ids is not None and answer_attention_mask is None:
            answer_attention_mask = (
                answer_input_ids != self.config.pad_token_id
            ).long()
        channel_input_ids = answer_input_ids if answer_input_ids is not None else input_ids
        channel_attention_mask = (
            answer_attention_mask if answer_input_ids is not None else attention_mask
        )
        workspace = self.forward_hlwm(
            input_ids,
            attention_mask,
            canvas_length=canvas_length,
            lane_briefs=lane_briefs,
            lane_brief_ids=lane_brief_ids,
            lane_brief_attention_mask=lane_brief_attention_mask,
            sample_reverse=False,
            adaptive_halt=False,
            full_reverse_schedule=True,
            generator=generator,
        )
        decode_prefix: Optional[Tensor] = workspace.workspace_prefix
        if prefix_source == "gist":
            context_hidden = self._encode_committed(input_ids, attention_mask)
            decode_prefix = self._build_gist_prefix(context_hidden, attention_mask)
        if disable_workspace_prefix:
            decode_prefix = None
        candidate_ids = self._decode_candidate(
            channel_input_ids,
            channel_attention_mask,
            decode_prefix,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 0.0,
            top_k=top_k,
            generator=generator,
        )
        candidate_mask = torch.ones_like(candidate_ids)
        candidate_logits, candidate_hidden = self._teacher_force_workspace(
            channel_input_ids,
            channel_attention_mask,
            decode_prefix,
            candidate_ids,
            candidate_mask,
        )
        verification_error = torch.sigmoid(workspace.verification_logits).mean(dim=-1)
        commitment_logits = self._score_candidate(
            candidate_hidden,
            candidate_logits,
            candidate_mask,
            workspace.global_state,
            verification_error,
        )
        logprob_feature = self.causal_answer_mean_logprob(
            input_ids, attention_mask, candidate_ids, candidate_mask
        ).exp()
        # A single decode is its own pool: agreement is one by definition.
        agreement_feature = commitment_logits.new_tensor([1.0])
        commit_mask = self._commit_mask(
            commitment_logits, logprob_feature, agreement_feature
        )
        committed = bool(commit_mask.item())
        output_ids = (
            candidate_ids
            if committed
            else torch.full(
                (1, 1),
                self.config.abstain_token_id,
                dtype=torch.long,
                device=input_ids.device,
            )
        )
        workspace = replace(
            workspace,
            synthesis_logits=candidate_logits,
            commitment_logits=commitment_logits,
            commit_mask=commit_mask,
            committed_ids=output_ids,
        )
        return HLWMGeneration(
            output_ids=output_ids,
            candidate_ids=candidate_ids,
            decision="publish" if committed else "abstain",
            commit_probability=float(torch.sigmoid(commitment_logits[0, 0]).cpu()),
            risk_probability=float(torch.sigmoid(commitment_logits[0, 1]).cpu()),
            verifier_error_probability=float(torch.sigmoid(commitment_logits[0, 2]).cpu()),
            private_verifier_error_probability=float(verification_error.mean().cpu()),
            macrocycles=int(workspace.reverse_timesteps.shape[1]),
            workspace=workspace,
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        *,
        mode: str = "hlwm",
        **kwargs: Any,
    ) -> Any:
        if mode == "causal":
            if kwargs:
                raise TypeError("causal mode received unsupported arguments: %s" % sorted(kwargs))
            return self.causal_logits(input_ids, attention_mask)
        if mode == "local_denoise":
            return self.forward_local_denoise(input_ids, attention_mask, **kwargs)
        if mode != "hlwm":
            raise ValueError("mode must be 'causal', 'local_denoise', or 'hlwm'")
        return self.forward_hlwm(input_ids, attention_mask, **kwargs)

    @staticmethod
    def _copy_parameter(target: Tensor, source: Tensor, name: str) -> None:
        if target.shape != source.shape:
            raise ValueError(
                "%s shape mismatch: target %r, source %r"
                % (name, tuple(target.shape), tuple(source.shape))
            )
        with torch.no_grad():
            target.copy_(source.detach().to(device=target.device, dtype=target.dtype))

    @classmethod
    def from_hf_model(
        cls,
        source_model: nn.Module,
        *,
        hlwm_overrides: Optional[Mapping[str, Any]] = None,
    ) -> "HLWMForConditionalGeneration":
        """Transplant compatible Qwen language weights from an in-memory model."""

        if not hasattr(source_model, "config"):
            raise TypeError("source model has no Hugging Face-style config")
        core = getattr(source_model, "model", None)
        if core is None or not all(hasattr(core, name) for name in ("embed_tokens", "layers", "norm")):
            raise TypeError("expected a Qwen-style model with embed_tokens, layers, and norm")
        if len(core.layers) == 0:
            raise ValueError("source model has no decoder layers")
        overrides = dict(hlwm_overrides or {})
        overrides.setdefault(
            "use_qk_norm", hasattr(core.layers[0].self_attn, "q_norm")
        )
        overrides.setdefault(
            "attention_bias", core.layers[0].self_attn.q_proj.bias is not None
        )
        overrides.setdefault(
            "mlp_bias", core.layers[0].mlp.gate_proj.bias is not None
        )
        config = HLWMConfig.from_hf_config(source_model.config, **overrides)
        model = cls(config)
        if len(core.layers) != len(model.backbone.layers):
            raise ValueError("source and target decoder layer counts differ")

        cls._copy_parameter(
            model.backbone.embed_tokens.weight,
            core.embed_tokens.weight,
            "embed_tokens.weight",
        )
        cls._copy_parameter(model.backbone.norm.weight, core.norm.weight, "norm.weight")
        for index, (target_layer, source_layer) in enumerate(
            zip(model.backbone.layers, core.layers)
        ):
            cls._copy_parameter(
                target_layer.input_layernorm.weight,
                source_layer.input_layernorm.weight,
                "layers.%d.input_layernorm.weight" % index,
            )
            cls._copy_parameter(
                target_layer.post_attention_layernorm.weight,
                source_layer.post_attention_layernorm.weight,
                "layers.%d.post_attention_layernorm.weight" % index,
            )
            for projection_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                target_projection = getattr(target_layer.self_attn, projection_name)
                source_projection = getattr(source_layer.self_attn, projection_name)
                cls._copy_parameter(
                    target_projection.weight,
                    source_projection.weight,
                    "layers.%d.self_attn.%s.weight" % (index, projection_name),
                )
                if target_projection.bias is not None:
                    cls._copy_parameter(
                        target_projection.bias,
                        source_projection.bias,
                        "layers.%d.self_attn.%s.bias" % (index, projection_name),
                    )
            if target_layer.self_attn.q_norm is not None:
                cls._copy_parameter(
                    target_layer.self_attn.q_norm.weight,
                    source_layer.self_attn.q_norm.weight,
                    "layers.%d.self_attn.q_norm.weight" % index,
                )
                cls._copy_parameter(
                    target_layer.self_attn.k_norm.weight,
                    source_layer.self_attn.k_norm.weight,
                    "layers.%d.self_attn.k_norm.weight" % index,
                )
            for projection_name in ("gate_proj", "up_proj", "down_proj"):
                target_projection = getattr(target_layer.mlp, projection_name)
                source_projection = getattr(source_layer.mlp, projection_name)
                cls._copy_parameter(
                    target_projection.weight,
                    source_projection.weight,
                    "layers.%d.mlp.%s.weight" % (index, projection_name),
                )
                if target_projection.bias is not None:
                    cls._copy_parameter(
                        target_projection.bias,
                        source_projection.bias,
                        "layers.%d.mlp.%s.bias" % (index, projection_name),
                    )

        output_embeddings = (
            source_model.get_output_embeddings()
            if hasattr(source_model, "get_output_embeddings")
            else getattr(source_model, "lm_head", None)
        )
        if output_embeddings is None:
            raise TypeError("source model exposes no output embeddings")
        cls._copy_parameter(
            model.lm_head.weight, output_embeddings.weight, "lm_head.weight"
        )
        return model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        hlwm_overrides: Optional[Mapping[str, Any]] = None,
        **huggingface_kwargs: Any,
    ) -> "HLWMForConditionalGeneration":
        """Load a Qwen checkpoint and transplant it into the HLWM backbone.

        ``transformers`` is imported lazily, so tiny offline tests do not need
        that package.  For the intended baseline use a pinned revision of
        ``Qwen/Qwen3-0.6B`` and record the resolved config with the experiment.
        """

        try:
            from transformers import AutoModelForCausalLM
        except ImportError as error:  # pragma: no cover - exercised without dependency in deployments.
            raise ImportError(
                "from_pretrained requires the optional 'transformers' package"
            ) from error
        source = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, **huggingface_kwargs
        )
        return cls.from_hf_model(source, hlwm_overrides=hlwm_overrides)


__all__ = [
    "CategoricalDiffusion",
    "HLWMConfig",
    "HLWMForConditionalGeneration",
    "HLWMGeneration",
    "HLWMLocalDenoiseOutput",
    "HLWMOutput",
    "QwenLikeBackbone",
    "RoutedAdapterBank",
]
