# Experiment 5: Robustness under Input Perturbations

## Purpose

Experiment 5 tests whether AE and MAE rely on different kinds of visual information.

The key idea is to selectively damage different cues at test time while keeping the frozen encoder and linear probe fixed. If AE relies more on local pixel statistics, it should be more affected by texture-removing perturbations. If MAE relies more on global structure, it may be more affected by perturbations that destroy spatial layout.

## Perturbations

This implementation uses three lightweight CIFAR-100 perturbations:

```text
high_pass      removes low-frequency image content and changes texture statistics
patch_shuffle  shuffles 4x4 patches, preserving local patches but destroying global layout
patch_drop     replaces 50% of 4x4 patches with a constant value
```

The model is trained only on clean CIFAR-100 images. Perturbations are applied only during evaluation.

## Run

```bash
python scripts/run_exp5_robustness_vit_cifar100.py
```

Default outputs:

```text
results/exp5_robustness/exp5_robustness_summary.csv
results/exp5_robustness/exp5_robustness_by_seed.csv
results/exp5_robustness/exp5_robustness_history.json
```

For a quick smoke test:

```bash
python scripts/run_exp5_robustness_vit_cifar100.py --epochs 1 --seeds 42 --max-train-samples 512 --max-test-samples 128 --num-workers 0
```

## Interpretation

The most important quantities are absolute and relative drops from clean accuracy.

A strong double dissociation would look like this:

```text
AE drops more under high_pass
MAE drops more under patch_shuffle
```

That would support the hypothesis that AE relies more on local pixel or texture cues, while MAE relies more on global spatial structure.

If MAE has higher accuracy under all perturbations, the result should be framed as general robustness rather than a clean double dissociation.

If both models collapse similarly, the experiment should be treated as inconclusive, especially because the current tiny ViT and CIFAR-100 setting has low absolute accuracy.

## Current Results

The formal run used 3 random seeds, 10 linear-probe epochs, and the full CIFAR-100 validation set.

Summary:

```text
AE  clean=5.74%
MAE clean=8.85%

AE  high_pass=1.26%, relative_drop=78.01%
MAE high_pass=1.27%, relative_drop=85.71%

AE  patch_shuffle=5.73%, relative_drop=0.12%
MAE patch_shuffle=6.84%, relative_drop=22.72%

AE  patch_drop=3.88%, relative_drop=32.46%
MAE patch_drop=4.09%, relative_drop=53.84%
```

The result is not a clean double dissociation. MAE keeps higher absolute accuracy under patch shuffle and patch drop, but its relative drop from its own clean accuracy is larger. This suggests that MAE learns stronger features overall while also relying more on intact patch-level spatial organization.

High-pass filtering is severe for both models and drives them to nearly the same low accuracy. This indicates that both linear probes still depend strongly on low-frequency or natural-image statistics in the current tiny ViT setting.

The most defensible conclusion is:

```text
MAE improves clean and perturbed absolute accuracy, but it is not simply invariant to all perturbations. Its larger relative degradation under patch shuffle and patch drop suggests that the extra information captured by MAE is partly tied to spatially coherent patch structure.
```
