# Experiment 6: Masking Ratio Ablation

## Goal

Experiment 6 tests whether masking itself is a causal factor behind the representation-quality improvement observed in masked autoencoders. Experiments 1 and 2 show that MAE learns better downstream representations than AE, even though AE achieves much better pixel-level reconstruction quality. Experiment 6 extends this argument by varying only the mask ratio during pre-training while keeping the model architecture, dataset, optimizer, probing protocol, and evaluation setup fixed.

If masking is an important control variable, representation quality should change systematically as the mask ratio changes. In particular, we expect an intermediate mask ratio to perform better than both the unmasked AE baseline and extremely high masking ratios. This would suggest that useful representation learning depends on a balance between preserving enough visible context and making the reconstruction task difficult enough to discourage local pixel-copying shortcuts.

## Method

We pre-train six tiny ViT reconstruction models on CIFAR-100 using the same architecture and training configuration. The only experimental variable is the pre-training mask ratio:

```text
0%, 15%, 30%, 50%, 75%, 90%
```

The 0% setting corresponds to a standard autoencoder objective, because no patches are masked. The remaining settings correspond to MAE-style masked reconstruction objectives of increasing difficulty.

After pre-training, we freeze the encoder and train a linear classifier on top of the frozen features. Linear probing is used as the main representation-quality metric. For each mask ratio, we run the linear probe with three random seeds: 42, 43, and 44. We report the mean and standard deviation across seeds.

The current implementation focuses on linear probing accuracy as the primary quantitative indicator of representation quality. The original project plan also proposed measuring shape bias and attention distance for each mask ratio. These mechanism-level measurements are left for the next phase rather than being reported as completed results in this experiment.

## Results

| Mask Ratio | Seed 42 | Seed 43 | Seed 44 | Mean Accuracy | Std. |
|---:|---:|---:|---:|---:|---:|
| 0% | 5.69% | 5.64% | 5.88% | 5.74% | 0.13% |
| 15% | 7.11% | 7.12% | 7.59% | 7.27% | 0.27% |
| 30% | 7.20% | 6.94% | 7.04% | 7.06% | 0.13% |
| 50% | 7.48% | 7.74% | 7.56% | 7.59% | 0.13% |
| 75% | 6.38% | 6.33% | 6.29% | 6.33% | 0.05% |
| 90% | 6.26% | 5.72% | 6.21% | 6.06% | 0.30% |

The best result is obtained at a 50% mask ratio, with a mean linear-probe accuracy of 7.59%. This is higher than the 0% AE baseline by 1.86 percentage points. In relative terms, the 50% mask ratio improves linear-probe accuracy by approximately 32.4% over the unmasked baseline.

All non-zero mask ratios outperform the 0% baseline. However, the improvement is not monotonic. Accuracy increases from 0% to 15%, remains similar at 30%, peaks at 50%, and then decreases at 75% and 90%.

## Discussion

Experiment 6 provides causal evidence that masking ratio directly affects downstream representation quality. Because all models share the same architecture and evaluation pipeline, the systematic change in linear-probe accuracy can be attributed to the change in the pre-training objective induced by different mask ratios.

The results follow an inverted-U pattern. Without masking, the model can reconstruct images using local pixel-level information, which appears to produce weaker features for downstream classification. Moderate masking improves the representation, likely because the encoder must infer missing content from broader context rather than simply copying nearby pixels. This interpretation is consistent with Experiment 2, where AE achieves much better reconstruction quality but worse representation quality.

At the same time, extremely high mask ratios reduce linear-probe accuracy. When 75% or 90% of patches are masked, the model receives too little visible information, making reconstruction overly difficult and reducing the quality of the learned representation. Thus, masking is beneficial, but only up to a point.

Overall, Experiment 6 strengthens the central argument of the project. The advantage of MAE over AE is not merely an architectural artifact or a consequence of training time. Instead, the masking ratio itself changes what the encoder must learn. A moderate level of masking appears to encourage more transferable representations, while no masking or excessive masking leads to weaker downstream performance.

## Limitation

This experiment should be interpreted as causal evidence for the effect of mask ratio on linear separability, not as a complete validation of every proposed mechanism. Shape bias, attention distance, and robustness under perturbations are mechanism-level analyses that should be evaluated separately in Experiments 3, 4, and 5.

## Files

The raw results are saved in:

```text
results/exp6_mask_ratio_vit_cifar100/exp6_mask_ratio_summary.csv
results/exp6_mask_ratio_vit_cifar100/exp6_mask_ratio_by_seed.csv
```

