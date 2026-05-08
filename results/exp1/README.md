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
- Linear probe seeds: `42, 43, 44`
- k-NN: `k={5, 10, 20, 50, 100}`
- k-NN train feature limit: 10,000 images
- k-NN test set: full CIFAR-10 test set

Command:

```bash
python scripts/run_exp1.py \
  --epochs 10 \
  --seeds 42 43 44 \
  --knn-k 5 10 20 50 100 \
  --knn-train-limit 10000 \
  --output-dir results/exp1
```

## Results

| Model | Linear Probe Accuracy | k-NN Accuracy |
|---|---:|---:|
| AE | See `exp1_summary.csv` | See `exp1_knn_by_k.csv` |
| MAE | See `exp1_summary.csv` | See `exp1_knn_by_k.csv` |

The full linear-probe training history is saved in:

```text
results/exp1/exp1_linear_probe_history.json
```

Per-seed linear-probe results are saved in:

```text
results/exp1/exp1_linear_probe_by_seed.csv
```

k-NN results for each `k` value are saved in:

```text
results/exp1/exp1_knn_by_k.csv
```

The summary table is saved in:

```text
results/exp1/exp1_summary.csv
```

## Interpretation

The current prototype should be interpreted from the generated CSV files. The key quantities are:

- `linear_probe_mean`: average final linear-probe accuracy across seeds.
- `linear_probe_std`: standard deviation across seeds.
- `best_knn_k`: the best-performing `k` value in the k-NN sweep.
- `best_knn_acc`: the best k-NN accuracy found in the sweep.

## Can This Be Used as the Final Experiment 1?

Not yet. This result is useful as a **preliminary prototype result**, but it should not be treated as the final Experiment 1 result described in the project README.

Reasons:

- The README defines the final experiment on **CIFAR-100**, while this run uses **CIFAR-10**.
- The README specifies a **ViT-style AE/MAE**, while this prototype uses a small convolutional AE/MAE.
- The k-NN result may or may not support the hypothesis depending on the `k` sweep output.
- Multiple linear-probe seeds reduce randomness, but the pretrained checkpoints themselves are still single runs.
- k-NN used 10,000 training features instead of the full training set.

## Next Steps

To make Experiment 1 stronger and closer to a final result:

1. Run at least 3 random seeds and report mean/std for linear probe accuracy.
2. Run k-NN with the full training set if runtime allows.
3. Test several `k` values, such as `k={5, 10, 20, 50, 100}`.
4. Move from CIFAR-10 to CIFAR-100.
5. Replace the current convolutional prototype with the planned ViT-based AE/MAE.
6. Add a figure comparing AE vs MAE for both linear probe and k-NN.

For now, this experiment can be reported as:

> In the CIFAR-10 prototype, Experiment 1 evaluates frozen AE and MAE encoders using multi-seed linear probing and a k-NN sweep. These results are useful preliminary evidence, but the final claim should be based on the planned CIFAR-100 + ViT setting.
