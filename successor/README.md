# Successor prototype: aligned write, declared routing

**Status: UNTESTED.** No experiment has run this code. It exists because the
J-space analysis of the certified v10.0 run
(`reports/hlwm-v10.0-jspace-results-2026-09-06.md`) localized the
readable-but-not-usable failure to *write alignment*: the dense-supervision
channel stored the withheld premise in directions the decoder's answer logits
are insensitive to (probe 0.075/0.100 inside the decoder-sensitive subspace vs
0.400/0.725 in its complement, chance 0.129, both seeds). The two modules here
implement the two obligations that result imposes on any successor. Neither is
a claim; both require a preregistered gate battery before any number from them
means anything.

## `aligned_write.py`

`AlignedWriteLoss` supervises the latent thoughts *inside* the
decoder-sensitive subspace instead of hoping dense targets land there. The
sensitive subspace is taken from the frozen unembedding rows of the target
tokens (the logit-lens special case of the Jacobian construction used in the
analysis; the analysis script's SVD variant is the reference implementation).
The loss rewards target-token content specifically in the projection of the
thought embeds onto that subspace, and carries a diagnostic that reports the
in-subspace/out-of-subspace content split per batch — the quantity the v10.0
program never measured during training.

## `declared_routing.py`

`DeclaredRouter` replaces learned latent gating with in-band declarations:
the model emits a literal route token (parsed like a tool call, in the style
of Declarative Attention, arXiv:2609.02737) and the harness applies it. Routing
is therefore supervisable by ordinary SFT on verified episodes, and causal
liveness is auditable by reading the token stream — the property every learned
router in the v5.5–v10.0 record failed to demonstrate (0 of 8 informative
seed-runs). The hybrid keeps a learned component only *within* the declared
choice (a bounded per-expert mixing weight), so the failure mode that retired
the learned router — silent collapse with no token-level trace — cannot recur
undetected.

## What running this would require

1. A preregistered plan with gates frozen before launch (the v10.0 plan format).
2. The matched plain-LoRA control the program never ran.
3. A training budget in the published-success range (≥ 2×10⁵ row-presentations),
   not the 10³ class this record's aborts were measured at.
