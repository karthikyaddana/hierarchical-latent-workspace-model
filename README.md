# Hierarchical Latent Workspace Model — papers

Two preprints from a twenty-day, ten-experiment preregistered attempt to build a
latent-workspace language model: a frozen Qwen3-0.6B decoder wrapped in 73M–132M
trainable sidecars implementing a root-preserving expert graph, isolated parallel
reasoning lanes, and a private diffusion canvas behind a calibrated publication rule.

**All three proposed mechanisms failed their preregistered gates.** Both papers are
negative-results reports. Neither claims a capability win.

Karthik Yaddanapudi · Independent Researcher · <karthikyaddana@gmail.com>

## The papers

### 1. Readable but Not Usable — [`paper/postmortem-paper.pdf`](paper/postmortem-paper.pdf)

*A Preregistered Post-Mortem of a Latent-Workspace Language Model, Including Its Own
Novelty Audit* (28 pp)

The decisive experiment deleted the operative premise from the decoder's input token
ids, so the latent channel was the only route from evidence to answer. Generation
through the channel scored **0.000 on all 99 masked rows on both seeds**, while a
linear probe recovered roughly **0.30** of the withheld premise. The channel was
readable but not usable.

This constructively extends Lagged Coupling (Xun, 2026, arXiv:2609.01048), which
reports the same read-before-causal dissociation in pretrained models. Here the
channel was built deliberately, made structurally necessary, and every instrument
defect in the record was repaired — and the dissociation persisted.

The paper also contains a novelty audit of its own six architectural claims (every one
previously published, with three residual cells still unoccupied) and an
instrument-failure taxonomy with detectors covering ~30 documented defects in five
classes, including gates hardcoded to pass and an expert graph the trainer never called.

### 2. The Hierarchical Latent Workspace Model — [`paper/hlwm-paper.pdf`](paper/hlwm-paper.pdf)

*Root-Preserving Variable-Depth Expertise for Parallel Diffusion Reasoning* (52 pp)

The full technical report: the proposed architecture, and the ten preregistered
prototype experiments that falsified it at this scale. Routing passed its load gates on
one seed of two but failed causal liveness on **10 of 10 seed-runs**. Verified fan-in
lost to self-consistency by ~30 points, traced to a beginning-of-sequence id the pinned
tokenizer aliases to end-of-text. A dedicated repair experiment fixed that channel
(zero turn-scaffold emissions in 640 audited rows) and then measured the workspace
directly: ablating the read-out's 34 memory positions changed graded accuracy by 0.000
on one seed and *improved* it by 0.022 on the other. Parity was reached by irrelevance.

## What is explicitly not claimed

- The matched plain-LoRA control **never ran**, so no capability claim is made.
- The calibrated-abstention thread never beat a free mean-log-probability baseline at
  its own preregistered bar.
- The dense-supervision successor was terminated at 125 optimizer updates, so that
  question closes **undecidable, not falsified**.

## Building from source

Both papers are single self-contained LaTeX files — no external figures, no BibTeX.

```sh
tectonic -X compile paper/postmortem-paper.tex
tectonic -X compile paper/hlwm-paper.tex
```

`pdflatex` twice also works (the papers use `lastpage`, so a second pass is needed).

## License

Text and figures are released under [CC BY 4.0](LICENSE).
