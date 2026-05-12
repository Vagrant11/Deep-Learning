# Project Progress and Next Steps

## Current Status

The project currently has completed results for Experiments 1, 2, and 6 in the CIFAR-100 tiny ViT setting.

Experiment 1 shows that MAE learns more useful downstream representations than AE. MAE improves linear-probe accuracy from 5.74% to 8.85%, and also gives a small improvement in best k-NN accuracy.

Experiment 2 shows that reconstruction quality and representation quality are decoupled. AE achieves much better pixel-level reconstruction quality, but MAE achieves better downstream linear-probe accuracy. This supports the claim that low reconstruction error does not necessarily imply better representation learning.

Experiment 6 shows that mask ratio has a systematic effect on representation quality. The best linear-probe accuracy is obtained at a 50% mask ratio, while both the 0% AE baseline and very high mask ratios perform worse. This supports the interpretation that masking is an important causal factor.

## Recommended Next Step

The next experiment to implement should be Experiment 3: attention distance and effective receptive field.

This is the best next step because it directly tests the global-context hypothesis and does not require a new dataset. It can reuse the existing CIFAR-100 validation set and the trained AE/MAE checkpoints. If successful, it will add mechanism-level evidence explaining why masking improves representation quality.

The preferred order is:

1. Experiment 3: attention distance and effective receptive field.
2. Experiment 5: robustness under input perturbations.
3. Experiment 4: shape vs. texture bias, only if time allows.

## Experiment 3 Plan

Experiment 3 should compare AE and MAE encoder behavior using a small sample of CIFAR-100 validation images.

The main metric should be mean attention distance, computed as the average spatial distance between query and key patches weighted by attention strength. If MAE has larger attention distance than AE, this would support the hypothesis that masking encourages the encoder to use longer-range context.

An optional second metric is effective receptive field, computed from input gradients. This can show whether the representation depends on a wider spatial region of the input image.

Expected output:

```text
results/exp3_attention/
```

Suggested files:

```text
scripts/run_exp3_attention_vit_cifar100.py
src/attention/attention_distance.py
paper/experiment_3.md
```

## Experiment 5 Plan

Experiment 5 should evaluate whether AE and MAE rely on different kinds of visual information.

A practical version can use three perturbations:

```text
high-pass filtering
patch shuffling
patch dropping at inference
```

High-pass filtering weakens texture and low-frequency image structure. Patch shuffling destroys global spatial layout while preserving local patch content. Patch dropping tests whether the classifier remains stable when visible information is removed at inference time.

Expected output:

```text
results/exp5_robustness/
```

Suggested files:

```text
scripts/run_exp5_robustness_vit_cifar100.py
src/robustness/perturbations.py
paper/experiment_5.md
```

## Experiment 4 Plan

Experiment 4 should be treated as optional unless there is enough time to build a reliable cue-conflict evaluation set.

The full version requires shape-texture conflict images, similar to Geirhos et al. This is more difficult than Experiments 3 and 5 because it requires either neural style transfer or another controlled way to separate object shape from texture. A rushed implementation may produce unstable or hard-to-interpret results.

If time is limited, Experiment 4 should be described as future work rather than included as a completed result.

## Writing Strategy

The current paper should make the following claims strongly:

1. MAE learns better downstream representations than AE.
2. Better reconstruction quality does not imply better representation quality.
3. Mask ratio causally affects linear separability.

The paper should make the following claims more cautiously:

1. Masking may encourage more global context use.
2. Masking may reduce reliance on local pixel-level shortcuts.
3. Shape bias and texture bias require additional evaluation.

This framing keeps the paper aligned with the project title while avoiding overclaiming results that have not yet been measured.

