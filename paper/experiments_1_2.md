# Experiments 1 and 2: Representation Quality and Reconstruction Quality

## Experiment 1: Linear Probing and k-NN Classification

### Goal

The goal of Experiment 1 is to evaluate whether masked autoencoding produces more useful representations than standard autoencoding under a controlled setting. We compare an AE encoder and an MAE encoder with the same tiny ViT reconstruction architecture. Both models are pre-trained on CIFAR-100 with the same training setup. The only difference is the reconstruction objective: AE reconstructs the full image without masking, while MAE reconstructs images with a 75% random patch masking ratio.

### Method

After pre-training, we freeze each encoder and evaluate the learned representations using two downstream probing protocols.

First, we train a linear classifier on top of the frozen encoder features. Linear probing measures how linearly accessible class-level semantic information is in the representation.

Second, we perform k-NN classification directly in the learned feature space. k-NN evaluates whether samples from the same class form useful local neighborhoods without training an additional classifier.

For linear probing, we use three random seeds: 42, 43, and 44. For k-NN, we evaluate multiple values of k and report the best result for each model.

### Results

| Model | Linear Probe Accuracy | Std. | Best k-NN Accuracy | Best k |
|---|---:|---:|---:|---:|
| AE | 5.74% | 0.13% | 5.51% | 50 |
| MAE | 8.85% | 0.34% | 5.73% | 100 |

The AE encoder achieves linear-probe accuracies of 5.69%, 5.64%, and 5.88% across the three seeds, giving a mean accuracy of 5.74% with a standard deviation of 0.13%. In contrast, the MAE encoder achieves 8.92%, 8.49%, and 9.15%, giving a mean accuracy of 8.85% with a standard deviation of 0.34%.

Thus, MAE improves linear-probe accuracy by 3.12 percentage points over AE. In relative terms, this is an improvement of approximately 54.3%.

The k-NN improvement is smaller. AE achieves its best k-NN accuracy of 5.51% at k=50, while MAE achieves its best k-NN accuracy of 5.73% at k=100. Although the gain is modest, the direction is consistent with the linear probing result.

### Discussion

Experiment 1 supports the hypothesis that masking improves representation quality. The large linear-probe improvement suggests that MAE features contain more linearly accessible semantic information than AE features under the same architecture and dataset setting.

At the same time, the k-NN improvement is limited. This suggests that the MAE advantage in this setting is stronger for global linear separability than for local nearest-neighbor structure. One possible reason is that k-NN is more sensitive to feature scaling, feature normalization, and local cluster quality, while a linear classifier can learn a global decision boundary from the frozen features.

Overall, Experiment 1 confirms the basic phenomenon motivating this project: compared with a standard AE, MAE learns representations that are more useful for downstream classification.

## Experiment 2: Reconstruction Quality vs. Representation Quality

### Goal

Experiment 1 shows that MAE features perform better on downstream classification. Experiment 2 asks whether this improvement is explained by better image reconstruction, or whether reconstruction quality and representation quality are decoupled.

To test this, we compare pixel-level reconstruction quality with downstream representation quality for AE and MAE.

### Method

We evaluate both pre-trained models on the CIFAR-100 test set. For AE, reconstruction is evaluated without masking, matching its training objective. For MAE, reconstruction is evaluated with a 75% random patch masking ratio, matching its masked reconstruction objective.

We measure full-image mean squared error (MSE) and peak signal-to-noise ratio (PSNR) as reconstruction-quality metrics. We use the mean linear-probe accuracy from Experiment 1 as the representation-quality metric.

### Results

| Model | Eval Mask Ratio | Full Image MSE | PSNR | Linear Probe Accuracy |
|---|---:|---:|---:|---:|
| AE | 0.00 | 0.000195 | 37.09 | 5.74% |
| MAE | 0.75 | 0.059671 | 12.24 | 8.85% |

AE achieves much better pixel-level reconstruction quality, with a full-image MSE of 0.000195 and a PSNR of 37.09 dB. MAE has substantially worse reconstruction quality under masked evaluation, with a full-image MSE of 0.059671 and a PSNR of 12.24 dB.

Despite this, MAE achieves better representation quality. Its linear-probe accuracy is 8.85%, compared with 5.74% for AE.

### Discussion

Experiment 2 demonstrates a clear decoupling between reconstruction quality and representation quality. The model that reconstructs pixels more accurately is not the model that learns the better downstream representation.

This result suggests that AE may solve the reconstruction task largely through local pixel-level information. Such local reconstruction shortcuts can produce low reconstruction error without forcing the encoder to learn high-level semantic structure. MAE, by contrast, must infer missing patches from visible context. This makes the reconstruction objective harder, but encourages the encoder to capture more global and semantically useful information.

Together, Experiments 1 and 2 support the central claim of this project: masked reconstruction improves learned representations not because it improves pixel-level reconstruction, but because it changes the information that the encoder must capture. Better reconstruction quality does not necessarily imply better representation quality.

