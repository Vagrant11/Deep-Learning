# Experiment 3: Attention Distance and Effective Receptive Field

## Purpose

Experiment 3 tests whether the MAE encoder uses more global context than the AE encoder, and separates two evaluation settings:

1. Full-image inference, where both AE and MAE receive all patches.
2. Masked reconstruction, where MAE is evaluated under its training condition with 75% masked patches.

The primary metric is mean attention distance: for each ViT encoder layer, attention weights are used to average the spatial distance between query and key patches. Larger values indicate that the layer attends over longer-range patch relationships.

The masked-reconstruction setting also reports masked-to-visible attention distance. This measures whether masked query patches attend to distant visible patches, which is closer to the actual mechanism required by masked image modeling than a full-image average over all query-key pairs.

The script additionally reports far attention mass: the fraction of attention assigned to patch pairs whose grid distance is at least 4 patches.

The secondary metric is effective receptive field (ERF): gradients of the mean feature norm with respect to input pixels are summarized by radius, entropy, and concentration area. Larger radius or entropy indicates that the representation depends on a broader input region.

## Implementation

Run:

```bash
python scripts/run_exp3_attention_vit_cifar100.py
```

Default inputs:

```text
checkpoints/ae_tinyvit_cifar100.pth
checkpoints/mae_tinyvit_cifar100.pth
```

Default outputs:

```text
results/exp3_attention/exp3_attention_distance_by_layer.csv
results/exp3_attention/exp3_erf_summary.csv
```

The script evaluates up to 500 CIFAR-100 validation images by default. Use `--max-test-samples` for a smaller or larger sample, and `--skip-erf` if only attention distance is needed.

By default, the attention CSV contains:

```text
AE full_image
MAE full_image
MAE masked_reconstruction
```

## Interpretation

If MAE does not show larger attention distance during full-image inference, this should be reported as a limitation of the simple global-attention hypothesis rather than hidden.

The stronger mechanism-level test is the masked-reconstruction condition. If masked-to-visible attention distance or far attention mass is larger under this condition, the paper can argue that masking encourages non-local information use when the model is solving the masked reconstruction task, even if the same pattern is weaker during full-image inference.

If MAE also shows a larger or higher-entropy ERF, this provides additional evidence that its representation uses a broader spatial region of the input.

## Current Results

The 500-image CIFAR-100 validation run produced the following pattern.

During full-image inference, AE has slightly larger mean attention distance than MAE in all four layers:

```text
AE:  4.1301, 4.1474, 4.1362, 4.1423
MAE: 3.5623, 3.7915, 4.1254, 4.1004
```

The ERF metrics are also similar, with no evidence that MAE has a broader input dependency:

```text
AE:  radius=13.0597px, entropy=0.9790
MAE: radius=13.0221px, entropy=0.9705
```

Under masked reconstruction, MAE shows more long-range behavior in the deeper encoder layers. In layers 2 and 3, masked-to-visible attention distance is higher than the layer's overall mean attention distance:

```text
layer 2: overall=3.9869, masked_to_visible=4.0943
layer 3: overall=4.0224, masked_to_visible=4.1467
```

These results do not support a broad claim that MAE always attends more globally than AE. A more accurate conclusion is that, in this tiny ViT setting, masking changes the attention pattern most clearly during the masked reconstruction condition and primarily in deeper layers. The main performance gains observed in Experiments 1, 2, and 6 should therefore be attributed cautiously to masking-induced representation pressure rather than to a simple uniformly larger attention distance.
