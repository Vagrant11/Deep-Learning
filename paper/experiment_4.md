# Experiment 4: Shape vs Texture Bias

## Purpose

Experiment 4 asks whether the representation learned by MAE is more shape-oriented or texture-oriented than the AE baseline.

The original project plan proposed a full cue-conflict dataset in the style of Geirhos et al., where object shape and texture come from different classes. A full neural style-transfer dataset is expensive to build reliably, so this implementation uses a lightweight Fourier-domain proxy.

## Method

For each CIFAR-100 validation image, we create a conflict image from two examples with different labels:

```text
shape source:  phase from image A
texture source: amplitude from image B
```

Fourier phase preserves much of the spatial layout, while Fourier amplitude carries frequency-energy statistics that act as a proxy for texture. The resulting image therefore combines the shape/layout cue from one class with the texture/statistical cue from another.

After training a frozen-encoder linear probe on clean CIFAR-100 training images, we evaluate each conflict image by comparing two logits:

```text
shape logit   = classifier score for image A's class
texture logit = classifier score for image B's class
```

The main metric is `shape_bias_rate`, the fraction of conflict images where the shape logit is higher than the texture logit. We also report top-1 rates for shape, texture, and other classes.

## Run

```bash
python scripts/run_exp4_shape_texture_vit_cifar100.py
```

Default outputs:

```text
results/exp4_shape_texture/exp4_shape_texture_summary.csv
results/exp4_shape_texture/exp4_shape_texture_by_seed.csv
results/exp4_shape_texture/exp4_shape_texture_history.json
```

For a quick smoke test:

```bash
python scripts/run_exp4_shape_texture_vit_cifar100.py --epochs 1 --seeds 42 --max-train-samples 512 --max-test-samples 128 --num-workers 0
```

## Interpretation

If MAE has a higher shape-bias rate than AE, this supports the hypothesis that masking encourages more shape-oriented or semantically structured features.

If both models mostly predict other classes, the logit-comparison metric is still more informative than raw top-1 counts, because CIFAR-100 conflict images can be ambiguous and low resolution.

This experiment should be described as a Fourier cue-conflict proxy rather than a full shape-texture benchmark.

## Current Results

The formal run used 3 random seeds, 10 linear-probe epochs, the full CIFAR-100 training set, and 5000 Fourier conflict validation images.

Summary:

```text
AE  clean_acc=5.68%, shape_bias=58.39%, texture_bias=41.61%
MAE clean_acc=8.87%, shape_bias=60.57%, texture_bias=39.43%
```

Per-seed shape-bias rates were stable:

```text
AE:  58.08%, 58.12%, 58.96%
MAE: 60.40%, 60.60%, 60.70%
```

MAE therefore shows a small but consistent increase in shape preference under the Fourier cue-conflict proxy. The effect size is modest, so the safest conclusion is that MAE features are slightly more shape-aligned than AE features in this setting, not that MAE has a strong human-like shape bias.

Top-1 predictions mostly fall into neither the shape nor texture source class:

```text
AE  other_top1=96.25%
MAE other_top1=96.11%
```

This is expected for low-resolution CIFAR-100 conflict images and is why the main metric compares shape and texture logits directly.
