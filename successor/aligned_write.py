"""Aligned-write supervision for a latent channel into a frozen decoder.

UNTESTED. Written 2026-09-07 as the direct consequence of the J-space analysis
of the certified v10.0 run (see ../reports/hlwm-v10.0-jspace-results-2026-09-06.md):
the dense-supervision channel stored probe-readable premise content almost
entirely OUTSIDE the subspace the decoder's answer logits are sensitive to.
This module supervises the write inside that subspace and measures the split
during training, so a misaligned write is visible at step one instead of after
a post-mortem.

The decoder-sensitive subspace here is the span of the frozen unembedding rows
of the supervision targets: the logit-lens special case of the per-row Jacobian
construction in scripts/jspace_projection_audit.py (which remains the reference
measurement). Using unembedding rows keeps the loss cheap enough to run every
step; the Jacobian variant belongs in the audit, not the loss.
"""

from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class AlignedWriteLoss(nn.Module):
    """Supervise thought embeddings inside the decoder-sensitive subspace.

    Args:
        unembedding: the frozen ``lm_head`` weight, ``[vocab, width]``. Never
            trained here; registered as a buffer reference, not a copy.
        temperature: softmax temperature for the in-subspace decode loss.
    """

    def __init__(self, unembedding: Tensor, temperature: float = 1.0) -> None:
        super().__init__()
        if unembedding.dim() != 2:
            raise ValueError("unembedding must be [vocab, width]")
        self.register_buffer("unembedding", unembedding, persistent=False)
        self.temperature = float(temperature)

    def _target_basis(self, target_ids: Tensor) -> Tensor:
        """Orthonormal basis of the unembedding rows of one row's targets.

        target_ids: ``[k]`` distinct target token ids. Returns ``[r, width]``
        with r <= k (rank after QR).
        """
        rows = self.unembedding[target_ids].float()  # [k, width]
        # QR gives an orthonormal basis of the row span; rank deficiency is
        # fine (repeated or near-collinear target embeddings).
        q, r = torch.linalg.qr(rows.t(), mode="reduced")  # q: [width, k]
        keep = r.diagonal().abs() > 1.0e-6
        return q[:, keep].t()  # [rank, width]

    def split(self, thoughts: Tensor, target_ids: Tensor) -> Dict[str, Tensor]:
        """Decompose one row's thoughts against its target subspace.

        thoughts: ``[k_thoughts, width]``; target_ids: ``[k_targets]``.
        Returns the in-subspace component, the residual, and the energy split
        (the training-time analogue of the analysis' probe split).
        """
        basis = self._target_basis(target_ids)  # [r, width]
        flat = thoughts.float()
        coords = flat @ basis.t()  # [k_thoughts, r]
        inside = coords @ basis  # [k_thoughts, width]
        outside = flat - inside
        total = flat.pow(2).sum().clamp_min(1.0e-12)
        return {
            "inside": inside,
            "outside": outside,
            "inside_energy_fraction": inside.pow(2).sum() / total,
        }

    def forward(
        self,
        thoughts: Tensor,
        target_ids: Sequence[Tensor],
    ) -> Dict[str, Tensor]:
        """Loss over a batch.

        thoughts: ``[batch, k_thoughts, width]`` (the grounded thought embeds,
            requires_grad). target_ids: per-row 1-D tensors of the token ids
            the channel must carry (e.g. the withheld premise's tokens).

        The decode term asks the IN-SUBSPACE component alone to score the
        row's target tokens through the frozen unembedding: content that only
        exists in the orthogonal complement contributes nothing and is not
        rewarded. The alignment term reports (and mildly pressures) the
        energy split itself.
        """
        if thoughts.dim() != 3:
            raise ValueError("thoughts must be [batch, k, width]")
        decode_losses, fractions = [], []
        for row_thoughts, row_targets in zip(thoughts, target_ids):
            parts = self.split(row_thoughts, row_targets)
            pooled = parts["inside"].mean(dim=0)  # [width]
            logits = (self.unembedding.float() @ pooled) / self.temperature
            log_probs = F.log_softmax(logits, dim=-1)
            decode_losses.append(-log_probs[row_targets].mean())
            fractions.append(parts["inside_energy_fraction"])
        decode = torch.stack(decode_losses).mean()
        inside_fraction = torch.stack(fractions).mean()
        # Misalignment pressure: push write energy toward the sensitive
        # subspace. Kept separate so the two terms can be weighted, gated,
        # and — critically — logged independently.
        misalignment = 1.0 - inside_fraction
        return {
            "decode_loss": decode,
            "misalignment": misalignment,
            "inside_energy_fraction": inside_fraction.detach(),
        }
