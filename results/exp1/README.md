# Experiment 1: Linear Probing and k-NN Classification

## Purpose

Experiment 1 tests whether the representations learned by MAE are more useful for downstream classification than those learned by a standard AE.

The evaluation freezes the pretrained encoder and measures representation quality using:

- **Linear probing**: train only a linear classifier on top of frozen features.
- **k-NN classification**: classify test features by nearest neighbors in frozen feature space.

## Current Prototype Setup

This run uses the current CIFAR-10 prototype implementation:

- Dataset: CIFAR-10
- Encoder: convolutional encoder from `src/models/ae.py`
- AE checkpoint: `checkpoints/ae_cifar10.pth`
- MAE checkpoint: `checkpoints/mae_cifar10.pth`
- Linear probe epochs: 10
- k-NN: `k=20`
- k-NN train feature limit: 10,000 images
- k-NN test set: full CIFAR-10 test set

Command:

```bash
python scripts/run_exp1.py --epochs 10 --knn-train-limit 10000 --output-dir results/exp1
```

## Results

| Model | Linear Probe Accuracy | k-NN Accuracy |
|---|---:|---:|
| AE | 0.4902 | 0.3874 |
| MAE | 0.5012 | 0.3449 |

The full linear-probe training history is saved in:

```text
results/exp1/exp1_linear_probe_history.json
```

The summary table is saved in:

```text
results/exp1/exp1_summary.csv
```

## Interpretation

The current prototype gives mixed evidence.

For linear probing, MAE slightly outperforms AE:

```text
MAE - AE = +1.10 percentage points
```

This suggests that the MAE encoder may learn features that are marginally more linearly separable than the AE encoder.

However, k-NN gives the opposite result:

```text
MAE - AE = -4.25 percentage points
```

This means the frozen MAE feature space is not consistently better under a non-parametric nearest-neighbor evaluation in the current prototype.

## Can This Be Used as the Final Experiment 1?

Not yet. This result is useful as a **preliminary prototype result**, but it should not be treated as the final Experiment 1 result described in the project README.

Reasons:

- The README defines the final experiment on **CIFAR-100**, while this run uses **CIFAR-10**.
- The README specifies a **ViT-style AE/MAE**, while this prototype uses a small convolutional AE/MAE.
- The k-NN result does not support the hypothesis that MAE features are consistently more semantically structured.
- Only one seed was used, so the small linear-probe gap may not be statistically stable.
- k-NN used 10,000 training features instead of the full training set.

## Next Steps

To make Experiment 1 stronger and closer to a final result:

1. Run at least 3 random seeds and report mean/std for linear probe accuracy.
2. Run k-NN with the full training set if runtime allows.
3. Add feature normalization and test several `k` values, such as `k={5, 10, 20, 50, 100}`.
4. Move from CIFAR-10 to CIFAR-100.
5. Replace the current convolutional prototype with the planned ViT-based AE/MAE.
6. Add a figure comparing AE vs MAE for both linear probe and k-NN.

For now, this experiment can be reported as:

> In the CIFAR-10 prototype, MAE slightly improves linear-probe accuracy over AE, but this improvement does not transfer to k-NN classification. Therefore, the prototype provides partial but not conclusive support for the representation-quality hypothesis.
