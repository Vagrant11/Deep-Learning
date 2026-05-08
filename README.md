# What Makes Masked Autoencoders Learn Better Representations than Standard Autoencoders?

> **University of Sydney — Spring 2026 DLCC Submission (#55)**
> Authors: Hanchen Zang, Leo

---

## 📖 Overview

Masked Autoencoders (MAEs) have become a powerful self-supervised learning framework, achieving strong performance across image, video, and other representation learning tasks. While they consistently outperform standard Autoencoders (AEs) on downstream tasks, **the mechanism behind this advantage remains insufficiently understood**.

This project investigates the following central hypothesis:

> **Masking serves as an inductive bias that prevents the encoder from relying excessively on local pixel-level statistics, and instead encourages the learning of global, context-aware, and semantically meaningful representations.**

We design a series of controlled experiments to test this hypothesis, decomposing it into three sub-claims:

- **H1 (Globality)**: MAE features depend on longer-range context than AE features.
- **H2 (Semanticity)**: MAE features capture object-level / shape-level semantics, while AE features capture local texture statistics.
- **H3 (Anti-shortcut)**: AEs exploit local pixel-level shortcuts during reconstruction; masking eliminates these shortcuts.

---

## 🎯 Research Goals

1. **Confirm the phenomenon** — verify that MAE produces measurably better representations than AE under controlled conditions.
2. **Reveal the mechanism** — explain *why* this gap exists, not just *that* it exists.
3. **Establish causality** — show that masking itself (not architecture, training time, or augmentation) is the causal factor.

---

## 🧪 Experimental Plan

We design **6 experiments** organized into a layered argument: from **phenomenon → mechanism → causality**.

| # | Experiment | Layer | Sub-claim | Owner |
|---|-----------|-------|-----------|-------|
| 1 | Linear Probing & k-NN Classification | Phenomenon | — | Hanchen |
| 2 | Reconstruction Quality vs Representation Quality | Phenomenon | Counter-intuitive insight | Leo |
| 3 | Attention Distance & Effective Receptive Field | Mechanism | H1 (globality) | Leo |
| 4 | Shape vs Texture Bias | Mechanism | H2 (semanticity) | Hanchen |
| 5 | Robustness under Input Perturbations | Mechanism | H3 (anti-shortcut) | Leo |
| 6 | Masking Ratio Ablation | Causality | All | Hanchen |

### Experiment 1 — Linear Probing & k-NN Classification
**Goal**: Establish baseline evidence that MAE features are more linearly separable and semantically structured than AE features.
**Method**: Freeze the encoder of each pre-trained model. Train a linear classifier on top, and separately evaluate with non-parametric k-NN.
**Reveals**: The basic phenomenon — MAE produces more usable features for downstream classification.
**Dataset**: CIFAR-100 (train for probing, val for evaluation).

### Experiment 2 — Reconstruction Quality vs Representation Quality
**Goal**: Demonstrate the counter-intuitive decoupling between pixel-level reconstruction fidelity and downstream representation quality.
**Method**: Compare AE and MAE on (a) reconstruction metrics (MSE, PSNR, SSIM) and (b) linear probing accuracy. Plot the two against each other.
**Reveals**: AE achieves *higher* reconstruction quality but *lower* representation quality — direct evidence that pixel-level reconstruction is a shortcut, not a meaningful objective.
**Dataset**: CIFAR-100 validation set.

### Experiment 3 — Attention Distance & Effective Receptive Field (ERF)
**Goal**: Quantify whether MAE's encoder attends to more global context than AE's encoder.
**Method**:
- Compute mean **attention distance** per ViT layer (average query-key spatial distance).
- Compute **Effective Receptive Field** via input-gradient backpropagation.
**Reveals**: Mechanism for H1 — masking forces the encoder to build long-range dependencies even in early layers, because reconstructing masked patches requires non-local information.
**Dataset**: 100–500 sampled images from CIFAR-100 val (visualization-level, no full dataset needed).

### Experiment 4 — Shape vs Texture Bias
**Goal**: Determine whether MAE features encode object shape (semantic) or texture (pixel-statistical) information.
**Method**: Construct a cue-conflict dataset (Geirhos et al., 2019 style) by pairing the shape of one class with the texture of another via neural style transfer. Measure whether the linear classifier predicts the shape-class or the texture-class.
**Reveals**: Mechanism for H2 — directly tests the "global, semantically meaningful" vs "local pixel-level statistics" distinction stated in our hypothesis.
**Dataset**: CIFAR-100 (training), self-generated CIFAR-100 cue-conflict subset (evaluation).

### Experiment 5 — Robustness under Input Perturbations
**Goal**: Identify which type of information each model depends on, by selectively destroying different information types.
**Method**: Evaluate linear-probe accuracy under three perturbation regimes:
- **High-pass filtering** — destroys pixel-level texture.
- **Patch shuffling** — destroys global structure while preserving local patches.
- **Patch dropping at inference** — tests context-inference ability.
**Reveals**: Mechanism for H3 via *bidirectional dissociation* — AE should degrade more under high-pass filtering (texture-dependent), MAE should degrade more under patch shuffling (structure-dependent). Bidirectional evidence rules out the trivial "MAE is just more robust" explanation.
**Dataset**: CIFAR-100-C + custom perturbations.

### Experiment 6 — Masking Ratio Ablation
**Goal**: Establish a *causal* link between masking and representation quality.
**Method**: Pre-train MAE variants with masking ratio ∈ {0%, 15%, 30%, 50%, 75%, 90%} (where 0% degenerates to standard AE). Measure linear-probe accuracy, shape bias, and attention distance for each.
**Reveals**: Causality — if all "good" indicators rise monotonically (or follow an inverted-U) with mask ratio, masking itself is the controlling variable, not a confounding factor.
**Dataset**: CIFAR-100.

---

## 🗂️ Datasets Summary

| Dataset | Use | Notes |
|---------|-----|-------|
| **CIFAR-100** | Primary pre-training & evaluation | 60k images, 100 classes, manageable scale |
| **CIFAR-100-C** | Robustness (Exp. 5) | Available on Zenodo |
| **Custom cue-conflict subset** | Shape/texture bias (Exp. 4) | Generated via style transfer on CIFAR-100 |

---

## 🏗️ Architecture & Training Setup

To ensure all comparisons are controlled, both AE and MAE share:

- **Backbone**: ViT-S/16 (patch size 4 for CIFAR-100's 32×32 input)
- **Decoder**: lightweight ViT decoder (depth 4, dim 256)
- **Optimizer**: AdamW (lr=1.5e-4, weight decay=0.05)
- **Schedule**: 400 epochs, cosine decay, 40-epoch warmup
- **Batch size**: 512
- **Augmentations**: random crop + horizontal flip only (kept minimal to isolate the effect of masking)

The **only** difference between the AE baseline and the MAE is whether random patch masking is applied during the forward pass.

---

## 👥 Division of Work

### Shared (Phase 1)
Joint responsibility: train and verify the shared model checkpoints. Both authors must be familiar with the training pipeline.

**Deliverables**:
- `checkpoints/ae.pth`
- `checkpoints/mae_75.pth`
- `checkpoints/mae_ratio_{0,15,30,50,75,90}.pth`

### Leo — Feature Behavior Analysis Track
Experiments 2, 3, 5. These all involve hooking into model internals (intermediate features, attention maps, perturbation responses), and share a common probing-code structure.

### Hanchen — Downstream Evaluation Track
Experiments 1, 4, 6. These all involve frozen-encoder + linear-classifier pipelines, sharing a common evaluation harness.

### Joint (Phase 3)
Result integration, figures, Discussion section, paper writing.

---

## 📅 Timeline (5 weeks)

| Week | Joint Task | Leo | Hanchen |
|------|-----------|-----|---------|
| 1 | Code scaffold, data preparation | — | — |
| 2 | Pre-train AE, MAE, and 6 mask-ratio variants | (debug support) | (training lead) |
| 3 | — | Exp. 2, 3 | Exp. 1, 6 |
| 4 | — | Exp. 5 + write own Method/Results | Exp. 4 + write own Method/Results |
| 5 | Merge figures, write Discussion & Conclusion, proofread | | |

---

## 📁 Repository Structure

Current code is a runnable CIFAR-10 prototype. The full planned experiments will gradually replace the prototype convolutional AE/MAE with the controlled CIFAR-100 + ViT setup described above.

```
.
├── README.md
├── configs/                  # Shared YAML configs (do NOT edit independently)
│   ├── ae.yaml
│   ├── mae.yaml
│   ├── linear_probe.yaml
│   └── mask_ratio_sweep.yaml
├── checkpoints/              # Shared pre-trained models
├── src/
│   ├── models/               # Encoder/decoder, AE, MAE
│   ├── pretrain/             # Phase 1: reusable training code
│   ├── probing/              # Linear probe + k-NN (Hanchen, Exp. 1, 6)
│   ├── reconstruction/       # PSNR/SSIM analysis (Leo, Exp. 2)
│   ├── attention/            # Attention distance & ERF (Leo, Exp. 3)
│   ├── shape_texture/        # Cue-conflict generation & eval (Hanchen, Exp. 4)
│   └── robustness/           # Perturbation tests (Leo, Exp. 5)
├── scripts/                  # Reproducibility entrypoints
├── results/                  # Logged metrics, figures
└── paper/                    # LaTeX source
```

Prototype entrypoints:

```bash
python scripts/pretrain_ae_mae.py
python scripts/run_exp1.py
python scripts/run_linear_probe.py
```

---

## 🤝 Collaboration Conventions

1. **Single source of truth for hyperparameters** — all training uses files in `configs/`. No hard-coded values in scripts.
2. **Shared checkpoints** — only models trained from the agreed-upon configs are used for any downstream experiment. No private re-trains.
3. **Logging** — all runs logged to a shared Weights & Biases project; both authors can inspect each other's curves.
4. **Weekly sync** — 30-minute meeting every week to align on progress and catch integration issues early.
5. **Git workflow** — feature branches per experiment, PRs reviewed by the other author before merging to `main`.

---

## 📚 Key References

- He et al., *Masked Autoencoders Are Scalable Vision Learners*, CVPR 2022.
- Geirhos et al., *ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness*, ICLR 2019.
- Park & Kim, *How Do Vision Transformers Work?*, ICLR 2022.
- Hendrycks & Dietterich, *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations*, ICLR 2019.

---

## 📝 License

CC BY 4.0 (matching the OpenReview submission license).


# 是什么让掩码自编码器比标准自编码器学到更好的表征？

> **悉尼大学 — 2026 春季 DLCC 投稿（#55）**
> 作者：Hanchen Zang, Leo

---

## 📖 项目概述

掩码自编码器（Masked Autoencoders, MAE）已成为强大的自监督学习框架，在图像、视频等多种表征学习任务上表现优异。尽管 MAE 在下游任务上稳定地超越标准自编码器（Autoencoder, AE），**这种优势背后的机制目前仍缺乏充分的解释**。

本项目围绕以下核心假设展开研究：

> **掩码（masking）作为一种归纳偏置（inductive bias），阻止编码器过度依赖局部像素级统计信息，从而促使其学习全局的、上下文感知的、具有语义意义的表征。**

我们设计了一系列受控实验来验证这一假设，并将其拆解为三个可独立验证的子命题：

- **H1（全局性）**：MAE 学到的特征依赖于比 AE 更长距离的上下文。
- **H2（语义性）**：MAE 特征捕捉物体级 / 形状级语义信息，而 AE 特征捕捉局部纹理统计信息。
- **H3（反捷径）**：AE 在重建过程中利用了局部像素级捷径（shortcut）；掩码操作消除了这些捷径。

---

## 🎯 研究目标

1. **确认现象**——在受控条件下验证 MAE 确实比 AE 学到更好的表征。
2. **揭示机制**——解释这种差距**为什么**存在，而不仅仅是**是否**存在。
3. **建立因果**——证明 masking 这一操作本身（而非架构、训练时长或数据增强）是导致差异的因果变量。

---

## 🧪 实验方案

我们设计了 **6 个实验**，按"现象 → 机制 → 因果"的逻辑链层层递进。

| 编号 | 实验 | 论证层次 | 对应假设 | 负责人 |
|------|-----|---------|---------|-------|
| 1 | Linear Probing 与 k-NN 分类 | 现象层 | — | Hanchen |
| 2 | 重建质量 vs 表征质量解耦 | 现象层 | 反直觉核心洞察 | Leo |
| 3 | 注意力距离与有效感受野分析 | 机制层 | H1（全局性） | Leo |
| 4 | 形状偏好 vs 纹理偏好 | 机制层 | H2（语义性） | Hanchen |
| 5 | 输入扰动鲁棒性测试 | 机制层 | H3（反捷径） | Leo |
| 6 | Masking Ratio 消融实验 | 因果层 | 全部 | Hanchen |

### 实验 1 — Linear Probing 与 k-NN 分类
**目标**：建立基础证据，证明 MAE 特征比 AE 特征在线性可分性和语义结构上更优。
**方法**：冻结预训练编码器，在其顶部训练一个线性分类器；同时使用无参数 k-NN 进行评估作为对照。
**揭示规律**：基础现象——MAE 学到的特征对下游分类任务更有用。
**数据集**：CIFAR-100（训练集用于 probing，验证集用于评估）。

### 实验 2 — 重建质量 vs 表征质量解耦
**目标**：揭示一个反直觉现象——像素级重建质量与下游表征质量之间存在解耦。
**方法**：分别测量 AE 和 MAE 的（a）重建指标（MSE、PSNR、SSIM）和（b）线性 probe 准确率，并将两者绘制在同一图上对比。
**揭示规律**：AE 的重建质量*更高*，但表征质量*更差*——直接证明像素级重建是一个捷径，而非有意义的学习目标。
**数据集**：CIFAR-100 验证集。

### 实验 3 — 注意力距离与有效感受野（ERF）
**目标**：定量评估 MAE 编码器是否比 AE 编码器关注更全局的上下文。
**方法**：
- 计算每一层 ViT 的平均**注意力距离**（query 与 key 之间的平均空间距离）。
- 通过输入梯度反传计算**有效感受野**。
**揭示规律**：H1 的机制层证据——masking 强迫编码器即使在浅层也建立长距离依赖，因为重建被遮挡的 patch 必须利用非局部信息。
**数据集**：从 CIFAR-100 验证集中采样 100~500 张图像（可视化分析无需全数据集）。

### 实验 4 — 形状偏好 vs 纹理偏好
**目标**：判定 MAE 特征编码的是物体形状（语义）还是纹理（像素统计）。
**方法**：参考 Geirhos et al. (2019) 的方式，通过神经风格迁移构造 cue-conflict 数据集——将一个类别的形状与另一个类别的纹理结合。测量线性分类器倾向于预测形状对应的类还是纹理对应的类。
**揭示规律**：H2 的机制层证据——直接对应假设中"全局、语义"vs"局部像素统计"的核心区分。
**数据集**：CIFAR-100（训练）+ 自行生成的 CIFAR-100 cue-conflict 子集（评估）。

### 实验 5 — 输入扰动鲁棒性测试
**目标**：通过选择性破坏不同类型的信息，识别每种模型实际依赖的信息类别。
**方法**：在三种扰动设置下评估 linear probe 准确率：
- **高通滤波**——破坏像素级纹理。
- **Patch 打乱**——破坏全局结构，但保留每个 patch 的局部信息。
- **推理时随机遮挡 patch**——测试上下文推断能力。
**揭示规律**：H3 的*双向解离*证据——AE 在高通滤波下应当退化更严重（依赖纹理），MAE 在 patch 打乱下应当退化更严重（依赖全局结构）。这种双向证据排除了"MAE 只是更鲁棒"这种平凡解释。
**数据集**：CIFAR-100-C + 自定义扰动。

### 实验 6 — Masking Ratio 消融实验
**目标**：建立 masking 与表征质量之间的*因果*关系。
**方法**：在 mask ratio ∈ {0%, 15%, 30%, 50%, 75%, 90%} 下分别预训练 MAE（其中 0% 退化为标准 AE）。在每个 ratio 下测量线性 probe 准确率、形状偏好、注意力距离三项指标。
**揭示规律**：因果性证据——若所有"好"的指标随 mask ratio 单调上升（或呈倒 U 形），则证明 masking 本身是控制变量，而非混杂因素。
**数据集**：CIFAR-100。

---

## 🗂️ 数据集汇总

| 数据集 | 用途 | 备注 |
|-------|------|------|
| **CIFAR-100** | 主要预训练 + 评估 | 6 万张图，100 类，规模适中 |
| **CIFAR-100-C** | 鲁棒性测试（实验 5） | 可在 Zenodo 下载 |
| **自构造 cue-conflict 子集** | 形状/纹理偏好（实验 4） | 基于 CIFAR-100 通过风格迁移生成 |

---

## 🏗️ 架构与训练设置

为保证所有对比都是受控的，AE 与 MAE 共用以下设置：

- **骨干网络**：ViT-S/16（CIFAR-100 的 32×32 输入下 patch size 设为 4）
- **解码器**：轻量 ViT 解码器（depth=4, dim=256）
- **优化器**：AdamW（lr=1.5e-4，weight decay=0.05）
- **学习率调度**：400 epoch，余弦衰减，前 40 epoch warmup
- **Batch size**：512
- **数据增强**：仅 random crop + horizontal flip（保持最简，以隔离 masking 的效应）

AE 基线与 MAE 之间**唯一的差异**是前向传播中是否应用随机 patch masking。

---

## 👥 分工方案

### 共同任务（阶段 1）
两人共同负责：训练并验证共享模型 checkpoint。两位作者都需熟悉训练 pipeline。

**交付物**：
- `checkpoints/ae.pth`
- `checkpoints/mae_75.pth`
- `checkpoints/mae_ratio_{0,15,30,50,75,90}.pth`

### Leo —— 特征行为分析线
负责实验 2、3、5。这三个实验都涉及对模型内部行为的探测（中间特征、注意力图、扰动响应），代码风格相近，适合一人完成以保持一致性。

### Hanchen —— 下游任务评估线
负责实验 1、4、6。这三个实验都基于"冻结编码器 + 线性分类器"的 pipeline，共用同一套评估代码框架。

### 共同任务（阶段 3）
结果整合、图表绘制、Discussion 部分撰写、论文整体打磨。

---

## 📅 时间表（5 周）

| 周次 | 共同任务 | Leo | Hanchen |
|-----|---------|-----|---------|
| 第 1 周 | 搭建代码框架，准备数据 | — | — |
| 第 2 周 | 预训练 AE、MAE、6 个 mask ratio 变体 | （配合 debug） | （主导训练） |
| 第 3 周 | — | 实验 2、3 | 实验 1、6 |
| 第 4 周 | — | 实验 5 + 撰写自己的 Method/Results | 实验 4 + 撰写自己的 Method/Results |
| 第 5 周 | 整合图表，撰写 Discussion 与 Conclusion，整体校对 | | |

---

## 📁 仓库结构

当前代码是一个可运行的 CIFAR-10 prototype。后续完整实验会逐步把当前卷积 AE/MAE 替换为上文定义的 CIFAR-100 + ViT 受控实验设置。

```
.
├── README.md
├── configs/                  # 共享 YAML 配置（请勿独立修改）
│   ├── ae.yaml
│   ├── mae.yaml
│   ├── linear_probe.yaml
│   └── mask_ratio_sweep.yaml
├── checkpoints/              # 共享预训练模型
├── src/
│   ├── models/               # Encoder/decoder, AE, MAE
│   ├── pretrain/             # 阶段 1：可复用训练代码
│   ├── probing/              # Linear probe + k-NN（Hanchen，实验 1、6）
│   ├── reconstruction/       # PSNR/SSIM 分析（Leo，实验 2）
│   ├── attention/            # 注意力距离与 ERF（Leo，实验 3）
│   ├── shape_texture/        # Cue-conflict 生成与评估（Hanchen，实验 4）
│   └── robustness/           # 扰动鲁棒性测试（Leo，实验 5）
├── scripts/                  # 可复现性入口脚本
├── results/                  # 日志、指标、图表
└── paper/                    # LaTeX 源文件
```

Prototype 入口：

```bash
python scripts/pretrain_ae_mae.py
python scripts/run_exp1.py
python scripts/run_linear_probe.py
```

---

## 🤝 协作规范

1. **超参数单一来源**——所有训练均使用 `configs/` 下的配置文件，脚本中不允许硬编码超参。
2. **共享 checkpoint**——下游实验只能基于约定 config 训练出的模型，不得使用私自重训的版本。
3. **统一日志**——所有实验日志同步到共享的 Weights & Biases 项目，两人可互相查看曲线。
4. **每周同步**——每周 30 分钟例会，对齐进度，及早发现整合问题。
5. **Git 工作流**——按实验开 feature branch，PR 由另一作者 review 后合入 `main`。

---

## 📚 关键参考文献

- He et al., *Masked Autoencoders Are Scalable Vision Learners*, CVPR 2022.
- Geirhos et al., *ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness*, ICLR 2019.
- Park & Kim, *How Do Vision Transformers Work?*, ICLR 2022.
- Hendrycks & Dietterich, *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations*, ICLR 2019.

---

## 📝 许可证

CC BY 4.0（与 OpenReview 投稿许可一致）。
