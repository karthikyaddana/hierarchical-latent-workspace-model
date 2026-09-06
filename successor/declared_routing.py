"""Declared routing: in-band, token-visible route selection.

UNTESTED. Written 2026-09-07. Learned latent routing failed causal liveness on
every informative seed-run in this record (0 of 8, v5.5-v8.0). Declarative
Attention (arXiv:2609.02737) demonstrates the opposite design working zero-shot
at scale: the model DECLARES, in emitted tokens parsed like tool calls, where
its computation should go. This module is that design applied to expert
routing: the route is a literal token the model emits, the harness applies it,
and causal liveness becomes a property you can read off the token stream
instead of inferring from gate statistics.

Trainability: route declarations are ordinary tokens, so they are supervisable
by SFT on verified episodes (the 9,092-row corpus in the data factory carries
deterministic family labels for exactly this). A bounded learned mix survives
only WITHIN the declared expert, so the recorded collapse mode -- a router
silently concentrating on one expert with no token-level trace -- cannot recur
undetected: the declaration history IS the audit log.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor, nn

# One declaration token per route, emitted by the model inside the reasoning
# stream. Kept ASCII and regex-parseable; the harness strips them before any
# text reaches an evaluation surface.
ROUTE_PATTERN = re.compile(r"<route:([a-z0-9_-]{1,32})>")


def parse_declaration(text: str, routes: Sequence[str]) -> Optional[int]:
    """First declared route in ``text``, or None if absent/unknown.

    Unknown route names return None rather than raising: the caller decides
    whether an unparseable declaration falls back to the root path or aborts
    the row, and that decision belongs in a preregistered gate, not here.
    """
    match = ROUTE_PATTERN.search(text)
    if not match:
        return None
    name = match.group(1)
    try:
        return list(routes).index(name)
    except ValueError:
        return None


@dataclass
class DeclarationAudit:
    """Running tally of declarations: the liveness audit is just this object."""

    routes: Sequence[str]
    counts: Dict[str, int] = field(default_factory=dict)
    undeclared: int = 0
    unknown: int = 0

    def record(self, text: str) -> Optional[int]:
        match = ROUTE_PATTERN.search(text)
        if not match:
            self.undeclared += 1
            return None
        name = match.group(1)
        if name not in self.routes:
            self.unknown += 1
            return None
        self.counts[name] = self.counts.get(name, 0) + 1
        return list(self.routes).index(name)

    def summary(self) -> Dict[str, object]:
        total = sum(self.counts.values()) + self.undeclared + self.unknown
        return {
            "total": total,
            "declared_fraction": (sum(self.counts.values()) / total) if total else 0.0,
            "per_route": dict(self.counts),
            "undeclared": self.undeclared,
            "unknown": self.unknown,
        }


class DeclaredRouter(nn.Module):
    """Apply a declared route over a bank of expert deltas.

    The route index comes from the token stream (``parse_declaration``), never
    from a learned gate. The only learned quantity is ``within_mix``: a bounded
    per-expert scalar in [0, 1] blending the declared expert's delta with the
    always-on root path. It cannot reassign the route, only attenuate it, and
    it is logged per step.
    """

    def __init__(self, routes: Sequence[str], width: int) -> None:
        super().__init__()
        self.routes: List[str] = list(routes)
        self.expert_deltas = nn.ModuleList(
            nn.Linear(width, width, bias=False) for _ in self.routes
        )
        for delta in self.expert_deltas:
            nn.init.zeros_(delta.weight)  # experts open only as training demands
        self.within_mix = nn.Parameter(torch.zeros(len(self.routes)))

    def forward(self, hidden: Tensor, route_index: Optional[int]) -> Tensor:
        """hidden: ``[batch, seq, width]``. Root path when undeclared."""
        if route_index is None:
            return hidden
        mix = torch.sigmoid(self.within_mix[route_index])
        return hidden + mix * self.expert_deltas[route_index](hidden)
