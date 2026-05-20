---
title: "RoboEnvision: A Long-Horizon Video Generation Model for Multi-Task Robot Manipulation"
method_name: "RoboEnvision"
authors: [Liudi Yang, Yang Bai, George Eskandar, Fengyi Shen, Mohammad Altillawi, Dong Chen, Soumajit Majumder, Ziyuan Liu, Gitta Kutyniok, Abhinav Valada]
year: 2025
venue: arXiv 2025
tags: [video-generation, long-horizon-planning, keyframe-diffusion, robot-policy, cascaded-explicit, vlm, hierarchical]
zotero_collection: 3-Robotics/WAM-Survey
image_source: online
arxiv_html: https://arxiv.org/abs/2506.22007
created: 2026-05-20
---

# Paper Note: RoboEnvision: A Long-Horizon Video Generation Model for Multi-Task Robot Manipulation

## Metadata

| Item | Content |
|------|---------|
| Institution | University of Freiburg, LMU Munich, TU Munich, Huawei Munich Research Center |
| Date | June 2025 |
| Project Page | N/A |
| Baselines | [UniPi](UniPi.md), AVDC, OpenSora (Hierarchical/Autoregressive/Naive), RDT-1B |
| Links | [arXiv](https://arxiv.org/abs/2506.22007) / Code: N/A |

---

## One-Sentence Summary

> RoboEnvision avoids autoregressive error accumulation in long-horizon robot video generation by combining a VLM task decomposer, a keyframe diffusion model with semantics-preserving attention, and a frame-filling diffusion model — then regressing robot joint states from the generated video with a lightweight spatio-temporal transformer policy model.

---

## Core Contributions

1. **Non-Autoregressive Long-Horizon Video Pipeline**: A two-stage hierarchical approach (keyframe diffusion + filling diffusion) that generates long-horizon multi-task videos without chaining short-horizon segments, eliminating error accumulation.
2. **Semantics Preserving Attention Module**: Injects first-frame VAE features into the keyframe diffusion model's spatial attention to maintain object shape, position, and count consistency across widely-spaced keyframes.
3. **Keyframe-Instruction Cross-Attention**: A diagonal block masking strategy that explicitly aligns each generated keyframe with its corresponding atomic instruction, improving spatial reasoning and language grounding.
4. **Lightweight Spatio-Temporal Transformer Policy**: A compact policy model trained on long-horizon keyframes + interpolated frames that outperforms UniPi and RDT-1B on the new LHMM benchmark (67.4% vs. 34.1%).

---

## Problem Background

### Problem to Solve
Generating consistent long-horizon robot manipulation videos given a single high-level instruction (e.g., "clean the table", "classify fruits into boxes"). Existing video diffusion models struggle with: frame count limits, accumulated error in autoregressive chaining, spatial reasoning for small objects, and instruction-to-keyframe alignment.

### Limitations of Existing Methods
- **Autoregressive paradigm** (UniPi, VLP, AVDC): Future frames predicted from last generated frames — inconsistencies compound, observation space can shift dramatically between tasks.
- **Short-horizon-only models**: Generate 1-task videos and chain them, losing global context and introducing drift.
- **Standard temporal attention**: Insufficient for large motion between keyframes sampled at wide intervals; fails to maintain object count and shape.
- **Inverse dynamics models** (UniPi): Inaccurate when generated videos have inconsistencies.

### Motivation
Humans decompose long tasks into sub-tasks, imagine completion of each sub-task as a keyframe, then fill in the transitions. Mimicking this hierarchical structure — VLM decomposition → keyframe diffusion → filling diffusion → policy extraction — produces globally consistent long-horizon videos without autoregressive drift.

---

## Method Details

### Model Architecture

RoboEnvision is a **4-stage pipeline**:

1. **VLM Task Decomposer**: GPT-4o1 or DeepSeek decomposes high-level instruction $l_{HL}$ into $K$ atomic instructions $l^1, ..., l^K$
2. **Keyframe Diffusion Model** (Stage 1): DiT-based video diffusion model generating $K$ keyframes aligned to atomic instructions; built on OpenSora (~800M parameters)
3. **Filling Diffusion Model** (Stage 2): Fills frame gaps between consecutive keyframes with $F_i$ interpolated frames per segment; runs in parallel across segments post Stage 1
4. **Policy Model**: Lightweight spatio-temporal transformer regressing robot joint angles and gripper state from generated keyframes + some interpolated frames

### Core Modules

#### Module 1: Keyframe Diffusion with Keyframe-Instruction Cross-Attention

**Design Motivation**: Standard cross-attention attends to all instruction tokens from all keyframes simultaneously, causing semantic blurring. Each keyframe should be guided by exactly one atomic instruction.

**Implementation**:
- DiT architecture (spatial attention + temporal attention → 3D attention + cross-attention + feedforward)
- Text encoder embeds $K$ instructions into $K$ embeddings $\tau^i \in \mathbb R^{B \times 1 \times N_\tau \times D_\tau}$

The keyframe diffusion model is trained with a standard DDPM noise prediction objective. At training time, keyframe latent $z \in \mathbb R^{B \times K \times C \times H_z \times W_z}$ is noised via:

$$z_t = \sqrt{\alpha_t} z + \sqrt{1 - \alpha_t} \epsilon, \quad \epsilon \sim \mathcal N(0, \mathbf I)$$

and the denoiser is trained to predict the added noise given all $K$ instruction embeddings:

$$\min_\theta \mathbb E_{t, (z, \tau) \sim p_{data}, \epsilon \sim \mathcal N(0, \mathbf I)} \left\| \epsilon - \epsilon_\theta\!\left(z_t, t, \bigoplus_{i=1}^{K} \tau^i\right) \right\|^2_2$$

where $\bigoplus_{i=1}^{K} \tau^i$ denotes concatenation of all $K$ instruction embeddings.

To prevent semantic blurring — where features from one keyframe attend to the wrong instruction — a diagonal block cross-attention mask $\mathcal M \in \mathbb R^{B \times (KH_z W_z) \times (KN_\tau)}$ (0 on diagonal blocks, $-\infty$ off-diagonal) ensures each keyframe attends only to its paired instruction:

$$\mathcal A = \text{softmax}(QK^T + \mathcal M)V$$

Here $Q$ contains keyframe DiT feature queries and $K, V$ contain the instruction embedding keys and values.

#### Module 2: Semantics Preserving Attention (SPA)

**Design Motivation**: Robotic datasets have many small objects; keyframes sampled at large intervals show large object displacements outside training distribution. Standard image conditioning (channel concatenation) insufficient for fine-grained shape preservation.

**Implementation**:

In each transformer block's spatial attention layer, VAE features $z_0$ from the initial frame are reinjected. The standard spatial self-attention is augmented with cross-attention to first-frame VAE features:

$$\text{feature} = \text{Attention}(Q_S, K_S, V_S) + \text{Attention}(Q'_S, K_{z_0}, V_{z_0})$$

where $z_0$ is the VAE latent of the initial image, $K_{z_0}, V_{z_0}$ are its projected key and value features, and $Q_S, K_S, V_S$ are the standard spatial self-attention query, key, and value. Since VAE features share the same feature space as DiT features (unlike CLIP embeddings), this additive residual connection preserves fine-grained object shape and position across widely-spaced keyframes.

#### Module 3: 3D Full Attention (replacing temporal attention)

**Design Motivation**: Temporal attention operates on per-pixel sequences — insufficient for large inter-keyframe motions. 3D attention attends across all spatio-temporal tokens simultaneously.

**Implementation**:
- Replace temporal attention layers (attending on $(BH_z W_z, K, C)$) with 3D full attention (attending on $(B, KH_z W_z, C)$)
- More computationally expensive but essential for maintaining object count and position across widely-spaced keyframes

#### Module 4: Lightweight Policy Model

**Design Motivation**: Policy model must handle long-horizon variability; training on short-horizon frames misses the larger motion distribution of keyframe sequences.

**Architecture**:
- Input: generated keyframes + subset of interpolated frames
- Encoder: patch embedding → multiple transformer blocks (spatial attention + temporal attention + feedforward)
- Decoder: ResNet architecture downsampling to joint space dimensions (vs. MLP/global average pooling in prior work)
- Output: robot joint angles + gripper state per frame

---

## Experiments

### Datasets

| Dataset | Scale | Characteristics | Usage |
|---------|-------|----------------|-------|
| LanguageTable | 50k clips (concatenated) | Block manipulation tasks; long-horizon videos assembled via optical flow consistency | Training/Evaluation (124 videos) |
| LHMM (Long-Horizon Manipulation in MuJoCo) | 90k clips | New dataset; grocery/tool pick-place; 3–18 instructions; keyframe annotations from grasp detection | Training/Evaluation (100 videos, 45 long-horizon tasks) |

### Implementation Details

- **Base Model**: OpenSora (Diffusion Transformer architecture)
- **Video Diffusion Parameters**: ~800M
- **Training Resolution**: 360×640 (LanguageTable), 180×320 (LHMM)
- **Instruction Count**: 3–18 per long-horizon video
- **VLM for Task Decomposition**: GPT-4o1 (outperforms standard VLMs for spatial reasoning)
- **Policy Model Decoder**: ResNet (vs. MLP/global average pooling in prior work)
- **Inference**: Open-loop control; filling diffusion runs in parallel after keyframe generation

### Video Quality Evaluation

The comparison between prior works (UniPi, AVDC: short-horizon video chained autoregressively) and RoboEnvision (VLM decomposes instruction → Keyframe Diffusion → Filling Diffusion → Policy) is evaluated on video quality metrics. RoboEnvision achieves best performance across all 5 metrics on LHMM and 4/5 on LanguageTable:

| Method | LanguageTable LPIPS↓ | LT SSIM↑ | LT PSNR↑ | LT FVD↓ | LT CLIP↑ | LHMM LPIPS↓ | LH SSIM↑ | LH PSNR↑ | LH FVD↓ | LH CLIP↑ |
|--------|---------------------|----------|----------|---------|---------|------------|----------|----------|---------|---------|
| OpenSora (Hierarchical) | 0.1445 | 0.8269 | 22.82 | 147.37 | 24.57 | 0.1564 | 0.5257 | 16.61 | 231.02 | 23.51 |
| OpenSora (Autoregressive) | 0.1795 | 0.7839 | 21.77 | 176.61 | 24.15 | 0.1701 | 0.5232 | 16.46 | 241.35 | 23.58 |
| OpenSora (Naive) | 0.1723 | 0.8053 | 21.77 | 138.31 | **25.49** | 0.2086 | 0.4983 | 15.52 | 274.85 | 22.55 |
| AVDC | 0.1857 | 0.7687 | 21.32 | 189.64 | 23.32 | 0.2343 | 0.4729 | 15.33 | 267.93 | 21.37 |
| **RoboEnvision (Ours)** | **0.1324** | **0.8273** | **23.12** | **136.75** | 24.45 | **0.1282** | **0.5820** | **17.27** | **205.78** | **23.99** |

The only exception is CLIP score on LanguageTable (25.49 for Naive vs. 24.45 for Ours) — Naive concatenates all instructions as one long string which inflates CLIP similarity.

### Policy Model Success Rate on LHMM

Full RoboEnvision (67.4%) dramatically outperforms all baselines. Ablations confirm that both the non-autoregressive design and long-horizon training are critical:

| Method | Success Rate |
|--------|-------------|
| UniPi | 23.5% |
| RDT-1B | 34.1% |
| **Ours** | **67.4%** |
| Ours (short-horizon training) | 49.4% |
| Ours (Autoregressive inference) | 27.0% |

Training on short-horizon frames loses 18% (49.4%) — confirming long-horizon training is essential. Autoregressive inference collapses performance to UniPi level (27.0%), confirming the non-autoregressive design is critical.

### Ablation Study (LHMM, video quality)

Qualitative results confirm the role of each component: without both SPA and 3D Attention there is severe object disappearance; without SPA there is object shape distortion for small objects; without 3D Attention the wrong object count appears. Full RoboEnvision resolves all three issues.

| Configuration | LPIPS↓ | SSIM↑ | PSNR↑ | FVD↓ | CLIP↑ |
|--------------|--------|-------|-------|------|-------|
| Base (OpenSora hierarchical) | 0.1498 | 0.6924 | 20.07 | 184.71 | 24.10 |
| w/o Semantics Preserving Attention | 0.1415 | 0.7032 | 20.36 | 169.50 | 24.19 |
| w/o 3D Attention | 0.1430 | 0.7102 | 20.11 | 168.83 | 24.21 |
| **Full RoboEnvision** | **0.1305** | **0.7178** | **20.51** | **167.57** | **24.24** |

Both SPA and 3D Attention contribute to consistency; 3D attention primarily preserves object count/position, SPA preserves object shape for small objects. Together they reduce FVD from 184.71 to 167.57.

---

## Critical Analysis

### Strengths
1. Non-autoregressive design fundamentally avoids error accumulation — the most principled solution to long-horizon video consistency.
2. Semantics Preserving Attention is an elegant, lightweight fix: VAE feature injection adds minimal overhead while significantly improving small-object consistency.
3. Strong ablation study clearly isolates contributions of SPA (shape) and 3D attention (count/position).

### Limitations
1. **Open-loop execution only**: No replanning during execution; accumulated tracking error not corrected.
2. **No depth/3D reasoning**: Future work explicitly acknowledges depth integration as needed for physical alignment.
3. **VLM task decomposition dependency**: System requires accurate instruction decomposition from GPT-4o1; fails gracefully only if VLM produces feasible atomic steps.
4. **Simulation only**: Evaluated on MuJoCo and LanguageTable simulation; no real-robot deployment reported.
5. **Policy model trained on simulator ground truth**: Policy not trained end-to-end on generated videos, so quality gap between generated and real frames affects execution.

### Potential Improvements
1. Integrate depth/semantic maps as additional conditioning for physically-plausible video generation.
2. Closed-loop replanning by re-generating keyframes from current real observation mid-execution.
3. End-to-end joint training of video model and policy to reduce sim-to-generated-video gap.

### Reproducibility
- [ ] Code open-sourced
- [ ] Pretrained models available
- [x] Training details described (Section IV-A)
- [x] Dataset: LanguageTable (public), LHMM (new dataset, not yet released)

---

## Related Notes

### Based On
- OpenSora: Base DiT video diffusion model used as codebase
- Diffusion Transformer: Core DiT architecture for keyframe and filling diffusion
- Video Diffusion Model: Foundational generation method

### Compared Against
- [UniPi](UniPi.md): Short-horizon video + inverse dynamics baseline; outperformed 3x on success rate
- [AVDC](AVDC.md): Optical flow action extraction baseline; outperformed on all video quality metrics
- RDT-1B: Diffusion foundation model for bimanual manipulation; outperformed on LHMM tasks

### Method Related
- Hierarchical Planning: Two-stage coarse-to-fine video generation architecture
- Keyframe Generation: Core abstraction — keyframes as end-states of atomic sub-tasks
- Vision-Language Model: GPT-4o1 used for high-level instruction decomposition
- Attention Mechanism: Keyframe-Instruction Cross-Attention and Semantics Preserving Attention

### Hardware/Data Related
- MuJoCo: Physics simulator used for LHMM dataset and policy evaluation
- Language Table: Block manipulation benchmark used for training and evaluation

---

## Quick Reference Card

> [!summary] RoboEnvision (arXiv 2025)
> - **Core**: Non-autoregressive long-horizon video generation: VLM decomposition → keyframe diffusion (with SPA + 3D attention) → filling diffusion → lightweight policy model
> - **Method**: OpenSora DiT (~800M); diagonal block cross-attention masking for instruction-keyframe alignment; VAE feature injection for object consistency; spatio-temporal transformer + ResNet decoder policy
> - **Results**: 67.4% LHMM success rate (vs. 34.1% RDT-1B, 23.5% UniPi); SOTA on 5/5 LHMM video quality metrics
> - **Code**: N/A

---

*Note created: 2026-05-20*
