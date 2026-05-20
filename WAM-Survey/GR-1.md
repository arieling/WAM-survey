---
title: "Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation"
method_name: "GR-1"
authors: [Hongtao Wu, Ya Jing, Chilam Cheang, Guangzeng Chen, Jiafeng Xu, Xinghang Li, Minghuan Liu, Hang Li, Tao Kong]
year: 2023
venue: ICLR 2024
tags: [video-generation, joint-autoregressive, gpt-style, causal-transformer, calvin, robot-policy, pretraining]
zotero_collection: 3-Robotics/WAM-Survey
image_source: online
arxiv_html: https://arxiv.org/abs/2312.13139
created: 2026-05-20
---

# Paper Note: Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation (GR-1)

## Metadata

| Item | Content |
|------|---------|
| Institution | ByteDance Research |
| Date | December 2023 |
| Project Page | N/A |
| Baselines | MCIL, RT-1, HULC, MT-R3M |
| Links | [arXiv](https://arxiv.org/abs/2312.13139) / Code: N/A |

---

## One-Sentence Summary

> GR-1 is a 195M GPT-style causal transformer that jointly predicts future video frames and robot actions in a single autoregressive forward pass — pretrained on Ego4D (800K clips, 8M frames) video prediction, then fine-tuned on CALVIN with combined $\mathcal{L}_{finetune} = \mathcal{L}_{arm} + \mathcal{L}_{gripper} + \mathcal{L}_{video}$ — achieving 4.21 CALVIN avg. task length and 85.4% zero-shot unseen scene generalization (vs. 53.3% prior SOTA).

---

## Core Contributions

1. **Unified Video + Action Joint Autoregressive Model**: First GPT-style transformer jointly predicting future video frames and robot actions in a single autoregressive sequence — video prediction provides spatial future context; action prediction is conditioned on this predicted future.
2. **Ego4D Video Pre-training at Scale**: Pretrained on 800K clips from Ego4D (8M frames from 3,500+ hours of egocentric video) for video frame prediction — large-scale pre-training dramatically improves downstream manipulation generalization.
3. **Masked [OBS] and [ACT] Token Strategy**: Causal attention with specially designed masked prediction tokens for future frames [OBS] and actions [ACT] — enables joint training without future information leakage.
4. **Zero-Shot Scene Generalization**: 85.4% zero-shot success on CALVIN unseen scenes (vs. 53.3% prior SOTA HULC) — video pre-training provides robust visual representations that generalize across environments.

---

## Problem Background

### Problem to Solve
Robot manipulation policies fail to generalize across scenes, object instances, and language instructions. They also do not leverage the vast amount of internet-scale egocentric video data. How can large-scale video generative pre-training improve generalization of language-conditioned visual robot manipulation?

### Limitations of Existing Methods
- HULC: hierarchical with latent plans; 3.06 CALVIN avg.; 53.3% unseen scenes.
- RT-1: language-conditioned BC; no video prediction; limited cross-scene generalization.
- MT-R3M: frozen visual encoder + GPT-style policy; no joint video+action training; 0.15–0.30 real robot.
- Standard VLAs: no video generation; limited temporal planning.

### Motivation
Video generative pre-training on internet-scale egocentric video teaches the model to predict physically plausible future states — this predictive capability transfers to manipulation by conditioning action generation on predicted future observations. The joint training scheme makes video prediction a continuous auxiliary signal during robot fine-tuning.

---

## Method Details

### Model Architecture

GR-1 is a **GPT-style causal transformer** with:
- 12 transformer layers, 12 attention heads, 384 hidden dimensions
- **Total Parameters**: 195M (46M trainable — encoders frozen)

**Input Encoders** (frozen):
- Language: CLIP text encoder
- Vision: ViT pretrained with MAE (CLS tokens + Perceiver Resampler for patch tokens)
- Robot State: Linear layers encoding 6D end-effector pose + binary gripper

**Output Decoders**:
- Video prediction: Transformer decoder (self-attention + MLPs) reconstructing image patches
- Action prediction: 3-layer MLP with separate heads for arm (continuous) and gripper (binary)

### Core Modules

#### Module 1: Token Sequence Design

**Design Motivation**: Causal autoregressive generation requires a carefully designed token sequence that enables: (1) conditioning on history, (2) predicting future observations, and (3) predicting actions at the current step.

**Pre-training token sequence**:
$(l, o_{t-h}, [\text{OBS}], \ldots, o_t, [\text{OBS}])$

**Fine-tuning token sequence**:
$(l, s_{t-h}, o_{t-h}, [\text{OBS}], [\text{ACT}], \ldots, s_t, o_t, [\text{OBS}], [\text{ACT}])$

- $l$: language tokens (CLIP)
- $o_{t-h}$: observation tokens (ViT, CLS + patches via Perceiver Resampler)
- $s_t$: robot state (linear layer)
- $[\text{OBS}]$: masked future frame prediction targets
- $[\text{ACT}]$: masked action prediction targets
- Learned relative timestep embeddings shared across modalities per timestep

#### Module 2: Causal Attention with Masked Predictions

**Design Motivation**: [OBS] and [ACT] tokens are prediction targets — causal masking prevents them from attending to their own ground-truth targets during training.

**Implementation**:
- Causal attention: each token attends only to preceding tokens
- Masked [OBS] and [ACT] tokens cannot attend to future ground-truth
- Video prediction and action prediction generated simultaneously in one forward pass

#### Module 3: Video Pre-training on Ego4D

**Implementation**:
- Dataset: Ego4D, 800K clips, 8M frames, 3,500+ hours of egocentric video
- Clip duration: 3 seconds; frame spacing: 1/3 second

The pre-training objective is to predict a future frame given an observation context. Formally:

$$\pi(l, o_{t-h:t}) \to o_{t+\Delta t}$$

The video U-Net is trained to reconstruct future frame $o_{t+\Delta t}$ from language instruction $l$ and observation history $o_{t-h:t}$, with MSE loss $\mathcal{L}_{video}$ applied between reconstructed and original image patches.

- Hyperparameters: batch 1,024; LR 3.6e-4; AdamW + cosine decay; 5 warmup epochs; 50 total epochs

#### Module 4: Robot Manipulation Fine-tuning

**Implementation**:
- Dataset: CALVIN (20K+ expert trajectories, 1% language-annotated)
- Input sequence: 10 frames; future prediction offset: $\Delta t=3$ steps
- Predicts both static camera and gripper camera frames simultaneously

During fine-tuning, GR-1 jointly predicts future frames and robot actions. The policy objective is:

$$\pi(l, o_{t-h:t}, s_{t-h:t}) \to o_{t+\Delta t}, a_t$$

where $s_{t-h:t}$ is the robot state history and $a_t$ is the action at the current step. The combined fine-tuning loss is:

$$\mathcal{L}_{finetune} = \mathcal{L}_{arm} + \mathcal{L}_{gripper} + \mathcal{L}_{video}$$

- $\mathcal{L}_{arm}$: Smooth-L1 for continuous arm actions
- $\mathcal{L}_{gripper}$: Binary Cross-Entropy for gripper open/close
- $\mathcal{L}_{video}$: MSE for future frame prediction

Video prediction remains a continuous auxiliary objective during robot fine-tuning, providing spatial future context that conditions action quality.

- Hyperparameters: batch 512; LR 1e-3; 1 warmup epoch; 20 epochs

---

## Experiments

### Datasets

| Dataset | Scale | Characteristics | Usage |
|---------|-------|----------------|-------|
| Ego4D | 800K clips, 8M frames, 3,500+ hours | Egocentric human video | Video pre-training |
| CALVIN | 20K+ trajectories | 4 environments, language-annotated | Fine-tuning/Evaluation |
| Real robot data | Custom | Object transport, drawer manipulation | Real-robot evaluation |

### CALVIN Benchmark

The table compares GR-1 against HULC on CALVIN, demonstrating the advantage of joint video+action pre-training for zero-shot scene generalization:

| Method | Avg. Task Length | Task 1 SR | Unseen Scene Avg. | Zero-shot SR |
|--------|-----------------|-----------|-------------------|-------------|
| HULC | 2.52 | 88.9% | 2.40 | 53.3% |
| **GR-1** | **4.21** | **94.9%** | **3.06** | **85.4%** |
| GR-1 (10% data) | — | 77.8% | — | — |

+1.69 avg. task length over HULC; +32.1% zero-shot unseen scene generalization.

### Real Robot Experiments

GR-1 substantially outperforms RT-1 and MT-R3M on real-robot tasks, including generalization to unseen object categories:

| Task | GR-1 | RT-1 | MT-R3M |
|------|------|------|--------|
| Object transport (seen) | **79%** | 0.27–0.35 | 0.15–0.30 |
| Object transport (unseen instances) | **73%** | — | — |
| Object transport (unseen categories) | **30%** | — | — |
| Drawer manipulation | **75%** | — | — |

### Ablation — Pre-training and Video Prediction

The ablation isolates the contributions of Ego4D pre-training and the video prediction auxiliary objective during fine-tuning:

| Configuration | Avg. Task Length |
|--------------|-----------------|
| **GR-1 full** | **4.21** |
| No pre-training + video pred. | 3.82 (−0.39) |
| No pre-training + no video pred. | 3.33 (−0.88) |

Pre-training contributes +0.39; video prediction during fine-tuning provides additional +0.49 — both critical components.

### Future Prediction Offset $\Delta t$ Ablation

The choice of future prediction offset $\Delta t$ affects how informative the predicted frame is. The ablation below shows $\Delta t=3$ is optimal — balancing frame similarity (too close loses predictive content) vs. horizon relevance (too far becomes unpredictable):

| $\Delta t$ | Avg. Task Length |
|-----------|-----------------|
| 1 | 3.61 |
| **3** | **3.82** |
| 5 | 3.67 |

### Implementation Details

- **Model**: 12 layers, 12 heads, 384 dim; 195M total (46M trainable)
- **Language Encoder**: CLIP (frozen)
- **Vision Encoder**: ViT + MAE (frozen) + Perceiver Resampler
- **Pre-training**: Batch 1,024; LR 3.6e-4; AdamW + cosine; 50 epochs; 5 warmup
- **Fine-tuning**: Batch 512; LR 1e-3; 20 epochs; 1 warmup
- **Real robot**: Batch 64; 30 epochs
- **Input horizon**: 10 frames
- **Future offset**: $\Delta t=3$

---

## Critical Analysis

### Strengths
1. First paper to demonstrate that joint video+action autoregressive training on Ego4D scale dramatically improves manipulation generalization.
2. Simple GPT-style architecture — straightforward to understand and reproduce.
3. Zero-shot unseen scene generalization: 85.4% (+32.1% over HULC) — practical for deployment.

### Limitations
1. **195M parameters**: Relatively small by current standards — limits capacity for complex tasks.
2. **Frozen encoders**: 46M trainable parameters; frozen CLIP and ViT may limit task-specific adaptation.
3. **Short video prediction horizon**: $\Delta t=3$ steps — limited long-horizon planning capability.
4. **MSE video loss**: Pixel-space MSE produces blurry predictions without perceptual guidance.

### Potential Improvements
1. Larger backbone (GR-2, GR-MG) for improved capacity.
2. Perceptual video loss (LPIPS) for sharper future frame predictions.
3. Longer prediction horizons with hierarchical planning.

### Reproducibility
- [ ] Code open-sourced
- [ ] Pretrained models available
- [x] Training details described (LR, batch, epochs, architecture)
- [x] CALVIN, Ego4D publicly available

---

## Related Notes

### Based On
- CLIP: Frozen language encoder
- ViT + MAE: Frozen vision encoder with Perceiver Resampler
- GPT: Causal autoregressive transformer architecture

### Compared Against
- HULC: Hierarchical manipulation baseline; outperformed 2.52→4.21 CALVIN avg.
- RT-1: Language-conditioned BC; outperformed on real robot

### Method Related
- Joint Autoregressive Video+Action: Unified prediction of future frames and robot actions
- Video Pre-training: Ego4D large-scale video representation learning
- Causal Transformer: Autoregressive sequence model for manipulation

### Hardware/Data Related
- CALVIN: Primary evaluation benchmark
- Ego4D: Large-scale egocentric video dataset for pre-training

---

## Quick Reference Card

> [!summary] GR-1 (ICLR 2024)
> - **Core**: 195M GPT-style causal transformer (12L 12H 384D, 46M trainable); joint future frame + action prediction; Ego4D (800K clips) video pre-train → CALVIN fine-tune; $\mathcal{L}_{finetune} = \mathcal{L}_{arm} + \mathcal{L}_{gripper} + \mathcal{L}_{video}$; $\Delta t=3$ future offset
> - **Method**: Pre-train: batch 1024 LR 3.6e-4 50ep; Fine-tune: batch 512 LR 1e-3 20ep; 10-frame input; CLIP+ViT+MAE frozen
> - **Results**: 4.21 CALVIN avg (vs. 2.52 HULC); 85.4% zero-shot unseen scene (vs. 53.3%); 79% real-world seen objects
> - **Code**: N/A

---

*Note created: 2026-05-20*
