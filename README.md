# What Makes Masked Autoencoders Learn Better Representations than Standard Autoencoders?

**Hanchen Zang, Yuan (Leo) Fang** — The University of Sydney
Spring 2026 DLCC submission (#55) · [Read the paper](paper/) · License: CC BY 4.0

A controlled study of *why* masking improves self-supervised visual representations — not just *that* it does. Everything except the reconstruction objective is held fixed: same tiny ViT, same dataset, same optimizer, same probing protocol.

---

## Headline result

Masking improves downstream representation quality while making pixel reconstruction dramatically worse.

| Model | Linear probe | k-NN (best) | MSE | PSNR |
|---|---|---|---|---|
| AE | 6.26 ± 0.11 | 4.97 | 0.000116 | 39.34 dB |
| MAE (75% mask) | **11.43 ± 0.46** | **7.34** | 0.039253 | 14.06 dB |

The AE reconstructs pixels roughly 25 dB better and transfers roughly 82% worse. Pixel fidelity is not a proxy for representation quality.

> **On the absolute numbers.** 6–11% top-1 on CIFAR-100 is low by design. The study uses a deliberately small ViT so that a full six-experiment mechanism analysis and a six-point mask-ratio sweep are tractable. Architecture, optimizer, schedule, and probe are identical across every comparison, so the matched *difference* is the measurement — not the absolute accuracy.

---

## The hypothesis, and how it survived contact with the data

**Central hypothesis:** masking acts as an inductive bias that stops the encoder from solving reconstruction through local pixel-copying, pushing it toward global, context-aware, semantically meaningful features.

Decomposed into three testable sub-claims:

| | Sub-claim | Verdict |
|---|---|---|
| **H1** | MAE features depend on longer-range context | **Not supported at inference; supported under masked reconstruction** |
| **H2** | MAE features are more shape-oriented, AE more texture-oriented | **Not supported** |
| **H3** | AEs exploit local pixel shortcuts; masking removes them | **Supported, but refined** |

Two of three sub-claims failed as originally stated. The revised mechanism below is narrower than the hypothesis we started with, and is what the evidence actually supports.

### Revised mechanism: patch-level contextual relations

Masking does not make the encoder uniformly more global, more shape-biased, or more robust. It makes local pixel copying unavailable, so the encoder learns *relations among visible patches* that are useful for recovering missing ones. Those relations improve linear separability — and they leave the model dependent on coherent patch organisation, which is exactly what the perturbation results show.

---

## Setup

| Component | Setting |
|---|---|
| Dataset | CIFAR-100 (60k images, 100 classes) |
| Image / patch size | 32×32 / 4×4 → 64 tokens |
| Embedding dim | 128 |
| Encoder / decoder depth | 4 / 2 Transformer layers, 4 heads |
| Pre-training | 100 epochs (Exp. 1–5), 20 epochs (Exp. 6) |
| Optimizer | AdamW, lr 1e-3, weight decay 0.05, batch 256 |
| Probe | Linear, 10 epochs, seeds 42/43/44 |
| Default mask ratio | 75% |

AE loss is full-image MSE; MAE loss is computed on masked patches only. The masking function is the sole difference between the two pipelines.

---

## Results by experiment

### 1 — Representation quality
MAE lifts linear-probe accuracy from 6.26% to 11.43% (+5.17 pp, +82.6% relative) and k-NN from 4.97% to 7.34%. The gain holds under both a trained classifier and a non-parametric neighbourhood measure, so it reflects feature-space structure rather than probe fitting.

### 2 — Reconstruction vs. representation
See the headline table. The AE wins decisively on MSE and PSNR and loses decisively on transfer. This is the paper's central counter-intuitive result and the direct evidence for the shortcut account.

### 3 — Attention distance and effective receptive field

Mean attention distance during full-image inference:

| Model | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| AE | 4.2700 | 4.1323 | 4.1293 | 4.1222 |
| MAE | 3.4100 | 2.1748 | 3.2084 | 3.1972 |

**AE attends *further* than MAE in all four layers.** ERF is near-identical (AE radius 13.05, entropy 0.976; MAE radius 12.95, entropy 0.961). H1 fails as stated.

It holds only in the condition that matters — while the MAE is actually solving masked reconstruction, masked-to-visible attention distance exceeds visible-to-visible in the deeper layers (L2: 3.2710 vs 2.5547; L3: 3.1752 vs 2.5093). Masking changes attention during the pretext task, not as a permanent property of the encoder.

MAE's layer-1 distance (2.1748) is a conspicuous dip against its own other layers, suggesting specialised intermediate processing under masked training.

### 4 — Shape vs. texture bias (Fourier cue-conflict)

| Model | Clean acc. | Shape bias | Texture bias |
|---|---|---|---|
| AE | 6.37 | 58.72 | 41.28 |
| MAE | 11.13 | **59.00** | 41.00 |

A 0.28 pp difference across a 5 pp accuracy gap. Per-seed shape bias is AE 58.90 / 58.06 / 59.20 and MAE 59.06 / 58.96 / 58.98 — the seed spread within each model exceeds the gap between them. This is a clean null result: whatever MAE gains, it is not a shift toward shape.

### 5 — Robustness under perturbation

| Model | Condition | Acc. | Rel. drop |
|---|---|---|---|
| AE | Clean | 6.26 | — |
| AE | High-pass | 1.18 | 81.17 |
| AE | Patch shuffle | 6.25 | 0.16 |
| AE | Patch drop | 4.41 | 29.49 |
| MAE | Clean | 11.43 | — |
| MAE | High-pass | 2.38 | 79.22 |
| MAE | Patch shuffle | 7.92 | **30.64** |
| MAE | Patch drop | 5.24 | **54.20** |

MAE is more accurate under every condition, yet degrades far more in relative terms under patch shuffling (30.64 vs 0.16) and patch dropping (54.20 vs 29.49). The AE is almost perfectly indifferent to patch shuffling — strong evidence it never encoded global layout at all.

This rules out the trivial "MAE is just more robust" reading and points at the refined mechanism: MAE's advantage is carried by information that lives in patch organisation.

### 6 — Mask-ratio ablation (causality)

| Mask ratio | 0% | 15% | 30% | 50% | 75% | 90% |
|---|---|---|---|---|---|---|
| Linear acc. (mean of 3 seeds) | 5.74 | 7.27 | 7.06 | **7.59** | 6.33 | 6.06 |

Every non-zero mask ratio beats the unmasked baseline, and the relationship is an inverted U peaking at 50% (+1.86 pp over 0%). Only the mask ratio varies, so the effect cannot be attributed to architecture or probe differences — this is the strongest causal evidence in the study.

The decline at 75% and 90% matters as much as the rise: masking is not monotonically good. Too much of it leaves the reconstruction target underdetermined for a small model on low-resolution inputs.

*Note: this ablation uses the 20-epoch schedule for tractability, so its 0% baseline is not directly comparable to the 100-epoch AE result in Experiment 1.*

---

## Limitations

- Tiny ViT on CIFAR-100 — absolute accuracies are far below large-scale MAE systems. Mitigated by matched conditions throughout and a 100-epoch verification run of the main comparison, but not eliminated.
- The cue-conflict test is a Fourier phase/amplitude proxy, not a full style-transfer benchmark.
- Linear separability is the primary downstream metric; fine-tuning, few-shot, and cross-dataset transfer are untested.
- Attention and ERF analyses are diagnostic, not causal interventions — they interpret behaviour rather than prove a specific attention pattern drives the gain.
- The mask-ratio ablation runs on a shorter schedule than the headline comparison.

---

## Repository

```
configs/          Shared YAML configs (one source of truth for hyperparameters)
src/
  models/         Encoder/decoder, AE, MAE
  pretrain/       Pre-training
  probing/        Linear probe + k-NN            (Exp. 1, 6)
  reconstruction/ MSE/PSNR analysis              (Exp. 2)
  attention/      Attention distance & ERF       (Exp. 3)
  shape_texture/  Cue-conflict generation & eval (Exp. 4)
  robustness/     Perturbation tests             (Exp. 5)
scripts/          Reproducibility entrypoints
results/          Logged metrics (CSV/JSON) and figures
paper/            LaTeX source and PDF
```

Every table above regenerates from the logged outputs under `results/`. Each experiment is a separate entrypoint under `scripts/`.

**Author contributions.** Leo: Experiments 2, 3, 5 (model-internals probing — intermediate features, attention maps, perturbation response). Hanchen: Experiments 1, 4, 6 (frozen-encoder evaluation harness). Shared: pre-training and checkpoint verification, result integration, discussion, writing.

---

## References

- He et al., *Masked Autoencoders Are Scalable Vision Learners*, CVPR 2022.
- Geirhos et al., *ImageNet-trained CNNs are biased towards texture*, ICLR 2019.
- Bao et al., *BEiT: BERT Pre-Training of Image Transformers*, ICLR 2022.
- Xie et al., *SimMIM: A Simple Framework for Masked Image Modeling*, CVPR 2022.
- Hendrycks & Dietterich, *Benchmarking Neural Network Robustness*, ICLR 2019.

*Generative AI tools were used during preparation for method inspiration, related-work discovery, and formatting. The authors designed the experiments, implemented the methods, analysed the results, and reviewed all generated content.*
