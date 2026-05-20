---
title: "RynnVLA-002: A Unified Vision-Language-Action and World Model"
method_name: "RynnVLA-002"
authors: [Jun Cen, Siteng Huang, Yuqian Yuan, Kehan Li, Hangjie Yuan, Chaohui Yu, Yuming Jiang, Jiayan Guo, Xin Li, Hao Luo, Fan Wang, Deli Zhao, Hao Chen]
year: 2025
venue: arXiv 2025
tags: [joint-autoregressive, world-model, chameleon, vq-gan, libero, attention-masking, continuous-action, lerobot]
zotero_collection: 3-Robotics/WAM-Survey
image_source: online
arxiv_html: https://arxiv.org/abs/2511.17502
created: 2026-05-20
---

# Paper Note: RynnVLA-002: A Unified Vision-Language-Action and World Model

## Metadata

| Item | Content |
|------|---------|
| Institution | Alibaba Group (DAMO Academy), Hupan Lab, Zhejiang University |
| Date | November 2025 |
| Project Page | N/A |
| Baselines | WorldVLA (discrete baseline), continuous ablation variants |
| Links | [arXiv](https://arxiv.org/abs/2511.17502) / Code: N/A |

---

## One-Sentence Summary

> RynnVLA-002 extends WorldVLA with a continuous action head (parallel decoding + bidirectional attention) on Chameleon (VQ-GAN codebook 8,192 + BPE + 256-bin discrete actions) — achieving 97.4% LIBERO with continuous actions (93.3% discrete), boosting real-world success from <30% to >80% with world model training, and achieving 7.75–48.20 Hz inference with the continuous head.

---

## Core Contributions

1. **Continuous Action Transformer Module**: Adds a continuous action head with parallel decoding and bidirectional attention to the Chameleon-based WorldVLA discrete framework — continuous actions (L1 regression) substantially outperform discrete bins in real-world deployment while maintaining comparable simulation accuracy.
2. **Dual-Loss Joint Training**: Combined loss $\mathcal L = \mathcal L_{dis} + \alpha \cdot \mathcal L_{conti}$ ($\alpha=10$) trains discrete VLA + world model (cross-entropy) alongside continuous action head (L1) — enables both training objectives jointly.
3. **World Model Real-World Boost**: World model training increases real-world success from <30% to >80% — demonstrates that world model provides critical generalization capability for sim-to-real transfer.
4. **Fast Inference**: Continuous action head with parallel decoding achieves 7.75–48.20 Hz (vs. 2.74–3.69 Hz for discrete+chunking) — practical for real-time robot control.

---

## Problem Background

### Problem to Solve
WorldVLA demonstrated mutual enhancement between world model and action prediction using discrete action tokens. However, discrete actions (256 bins per dimension) suffer from quantization errors that significantly hurt real-world performance. How can a continuous action head be integrated into the unified discrete token framework?

### Limitations of WorldVLA
- Discrete action tokens: 93.3% LIBERO but limited real-world performance due to quantization.
- Slow inference: discrete chunking at 2.74–3.69 Hz — too slow for precise real-world control.
- No continuous action space option.

### Motivation
Continuous actions preserve fine-grained motor precision critical for real-world manipulation (especially contact-rich tasks). Parallel decoding with bidirectional attention in the action head removes the sequential bottleneck of autoregressive discrete generation, enabling fast inference. Joint training with the world model provides generalization from dynamics priors.

---

## Method Details

### Model Architecture

RynnVLA-002 is built on **Chameleon** (same unified backbone as WorldVLA) with an additional continuous action head:

1. **Image Tokenizer**: VQ-GAN, compression ratio 16, codebook 8,192 (256 tokens/256×256, 1,024 tokens/512×512)
2. **Text Tokenizer**: BPE (from Chameleon), vocabulary 65,536
3. **Discrete Action/State Tokenizer**: 256 bins per dimension; same vocabulary as images/text
4. **Continuous Action Head**: Parallel decoding, bidirectional attention, L1 regression
5. **Unified Vocabulary**: 65,536 tokens total

### Core Modules

#### Module 1: Action Attention Masking (from WorldVLA)

**Implementation** (inherited):
- Current actions conditioned only on textual + visual input
- Prior action tokens masked to prevent error accumulation
- World model (future image tokens) uses standard causal masking
- Attention masking prevents success rate degradation with longer action chunks

The masking scheme ensures that each action chunk is predicted independently from observations, preventing the error accumulation that would otherwise occur with longer action chunks in autoregressive generation.

#### Module 2: Continuous Action Head

**Design Motivation**: Discrete action bins suffer from quantization error — particularly limiting for fine-grained contact tasks requiring sub-millimeter precision. Bidirectional attention enables the action head to model dependencies across all action dimensions simultaneously, unlike autoregressive generation.

**Implementation**:
- Parallel decoding: all action dimensions decoded simultaneously (no autoregressive dependency)
- Bidirectional attention: each action dimension attends to all others — captures cross-dimension correlations
- Loss: L1 regression on continuous action values
- Speed: 7.75–48.20 Hz depending on input configuration

#### Module 3: Joint Discrete + Continuous Training

**Training Configuration**:
- M=2 historical frames
- K=5–10 action chunk (K=10 for longer tasks, K=5 for shorter)
- N=1 world model prediction round per chunk

The combined training loss jointly optimizes the discrete world model prediction (cross-entropy on image and action tokens) and the continuous action regression (L1):

$$\mathcal L = \mathcal L_{dis} + \alpha \cdot \mathcal L_{conti}, \quad \alpha = 10$$

where $\mathcal L_{dis} = \mathcal L_{dis\_action} + \mathcal L_{img}$ is the cross-entropy loss on discrete action tokens and future image tokens (the world model objective), and $\mathcal L_{conti}$ is the L1 regression loss on continuous action values. The weight $\alpha=10$ gives substantially higher priority to continuous action quality than the world model loss weighting used in WorldVLA ($\alpha=0.04$ for the world model loss there).

**Dataset**:
- LIBERO (4 suites: Spatial, Object, Goal, Long)
- LeRobot SO100 (248 + 249 expert demonstrations, real-robot)
- Preprocessing: remove unsuccessful trajectories + filter no-operation actions

---

## Experiments

### Datasets

| Dataset | Scale | Characteristics | Usage |
|---------|-------|----------------|-------|
| LIBERO | 4 suites | Tabletop manipulation benchmark | Training/Evaluation |
| LeRobot SO100 | 248+249 demos | Real-robot SO100 arm | Real-robot fine-tuning |

### Implementation Details

- **Backbone**: Chameleon (mixed-modal early-fusion)
- **Image Tokenizer**: VQ-GAN (ratio 16, codebook 8,192)
- **Text Tokenizer**: BPE (vocabulary 65,536)
- **Discrete Action**: 256 bins/dimension; cross-entropy loss
- **Continuous Action Head**: Parallel decoding, bidirectional attention, L1 loss
- **Historical Frames**: M=2
- **Action Chunk**: K=5 (short tasks), K=10 (long tasks)
- **World Model**: N=1 future frame per chunk
- **Loss Weight**: $\alpha=10$ for continuous; $\mathcal L_{dis} = \mathcal L_{dis\_action} + \mathcal L_{img}$
- **Robot**: LeRobot SO100 (real-world); LIBERO sim (Franka)
- **Data Preprocessing**: Remove failed trajectories; filter no-op actions

### LIBERO Benchmark (Continuous Actions)

The continuous action head achieves near-perfect LIBERO performance across all four suites, substantially exceeding the discrete baseline:

| Suite | RynnVLA-002 |
|-------|------------|
| Spatial | 99.0% |
| Object | 99.8% |
| Goal | 96.4% |
| Long | 94.4% |
| **Average** | **97.4%** |

97.4% LIBERO with continuous actions (vs. 93.3% discrete, vs. WorldVLA 81.8% discrete) — continuous head dramatically improves performance.

### World Model Ablation

The world model consistently boosts both discrete and continuous action performance, confirming the mutual enhancement principle:

| Configuration | Discrete LIBERO | Continuous LIBERO |
|--------------|-----------------|-------------------|
| VLA alone (w/o world model) | 62.8% | 91.6% |
| **VLA + World Model** | **67.8%** | **94.6%** |

World model consistently boosts both discrete (+5%) and continuous (+3%).

### Real-World SO100 Results

World model training is the critical factor for real-world generalization — it brings real-world success from <30% to >80%:

| Task | Success Rate |
|------|-------------|
| Block placement (single-target) | 90.0% |
| Block placement (multi-target) | 90.0% |
| Block placement (with distractors) | 80.0% |
| Strawberry placement (single-target) | 80.0% |

### Inference Speed

Continuous parallel decoding achieves up to 48 Hz — suitable for real-time control at 30+ Hz, in stark contrast to the slow discrete autoregressive baseline:

| Configuration | Frequency (Hz) |
|--------------|----------------|
| Discrete + chunking | 2.74–3.69 |
| Continuous (wrist cam) | 7.75 |
| Continuous (no wrist cam) | 48.20 |

---

## Critical Analysis

### Strengths
1. Continuous action head provides dramatic real-world improvement (<30%→>80%) — practical for deployment.
2. Fast inference (up to 48 Hz) with parallel decoding — suitable for real-time control.
3. World model generalization benefit confirmed on real robot — not just simulation.

### Limitations
1. **No explicit learning rates/batch sizes reported**: Training hyperparameters not disclosed — limited reproducibility.
2. **Small real-robot dataset**: 248+249 demonstrations is limited; may not generalize to more diverse tasks.
3. **Chameleon-only**: No comparison against diffusion-based policies (ACT, Diffusion Policy) in real-world.

### Potential Improvements
1. Scale to larger real-robot datasets for broader task coverage.
2. Multi-round world model (N>1) for longer-horizon planning.
3. Distill discrete world model into continuous action head for shared representations.

### Reproducibility
- [ ] Code open-sourced
- [ ] Pretrained models available
- [ ] Learning rate, batch size, epochs not disclosed
- [x] Architecture described (Chameleon, VQ-GAN, continuous head design)
- [x] LIBERO publicly available

---

## Related Notes

### Based On
- [WorldVLA](WorldVLA.md): Predecessor discrete action framework; same Chameleon backbone
- Chameleon: Mixed-modal early-fusion backbone
- VQ-GAN: Discrete image tokenization

### Compared Against
- [WorldVLA](WorldVLA.md): Extends discrete-only WorldVLA (81.8% → 97.4% LIBERO continuous)

### Method Related
- Continuous Action Head: Parallel decoding with bidirectional attention for continuous robot actions
- Attention Masking: Preventing action error accumulation via selective masking
- Mutual Enhancement: World model and action model benefit each other during joint training
- Joint Autoregressive Video+Action: Unified discrete token prediction of images + actions

---

## Quick Reference Card

> [!summary] RynnVLA-002 (arXiv 2025)
> - **Core**: Chameleon backbone; VQ-GAN (ratio 16, codebook 8192) + BPE + 256-bin discrete + continuous head (parallel decode, bidirectional attn, L1); action masking; $\mathcal L = \mathcal L_{dis} + 10 \cdot \mathcal L_{conti}$; M=2 frames, K=5–10 chunk, N=1 world model
> - **Method**: LIBERO + LeRobot SO100 (248+249 demos); filter failed/no-op; dual discrete+continuous training
> - **Results**: 97.4% LIBERO continuous (93.3% discrete); real-world <30%→>80% with world model; 48.20 Hz continuous inference
> - **Code**: N/A

---

*Note created: 2026-05-20*
