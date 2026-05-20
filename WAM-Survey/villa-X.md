---
title: "villa-X: Enhancing Latent Action Modeling in Vision-Language-Action Models"
method_name: "villa-X"
authors: [Xiaoyu Chen, Hangxing Wei, Pushi Zhang, Chuheng Zhang, Kaixin Wang, Yanjiang Guo, Rushuai Yang, Yucen Wang, Xinquan Xiao, Li Zhao, Jianyu Chen, Jiang Bian]
year: 2025
venue: arXiv 2025
tags: [latent-action, vq-vae, cascaded-implicit, vla, cross-embodiment, flow-matching, simpler]
zotero_collection: 3-Robotics/WAM-Survey
image_source: online
arxiv_html: https://arxiv.org/abs/2507.23682
created: 2026-05-20
---

# Paper Note: villa-X: Enhancing Latent Action Modeling in Vision-Language-Action Models

## Metadata

| Item | Content |
|------|---------|
| Institution | Tsinghua University, Microsoft Research |
| Date | July 2025 |
| Project Page | N/A |
| Baselines | RT-1-X, Octo, OpenVLA, π₀, GR00T-N1.5, TraceVLA, Magma, MoTo, LAPA |
| Links | [arXiv](https://arxiv.org/abs/2507.23682) / Code: N/A |

---

## One-Sentence Summary

> villa-X is a Vision-Language-Latent-Action framework that trains a Spatial-Temporal Transformer LAM (IDM + VQ codebook size 32 + proprioceptive FDM) on 5.2M video clips, then jointly predicts latent actions + robot actions via dual PaliGemma-conditioned experts (18-layer transformers each) with stochastic attention masking — achieving 77.7% SIMPLER Google Robot (vs. 62% GR00T-N1.5) and zero-shot cross-embodiment transfer.

---

## Core Contributions

1. **Proprioceptive FDM for Latent Grounding**: Forward Dynamics Model predicting future robot states + actions from current proprioception + latent action token — aligns latent actions with physical robot dynamics, capturing subtle motions invisible in pixel-space videos.
2. **Joint Latent + Robot Action Modeling**: Dual transformer experts (ACT-latent + ACT-robot) jointly trained with flow matching; ACT-robot conditions on ACT-latent predictions via blockwise causal cross-attention.
3. **Stochastic Attention Masking**: 50% fully masked / 50% randomly masked attention from robot to latent actions during training — prevents over-reliance on latent tokens, improving robustness when latent predictions are uncertain.
4. **Embodiment Context Conditioning**: Learnable embeddings for dataset ID and control frequency disambiguate heterogeneous data from 1.6M robot trajectories across diverse platforms.

---

## Problem Background

### Problem to Solve
Latent action models (LAM) from prior work (LAPA) learn discrete latent actions from frame pairs but: (1) miss subtle physical motions invisible in pixel-space, and (2) don't fully integrate latent predictions with VLA policy learning. How can latent actions be better grounded in robot proprioception and more tightly integrated with VLA action generation?

### Limitations of Existing Methods
- LAPA: VQ-VAE IDM from pixels only; no proprioceptive grounding; 50.1% real-world.
- GR00T-N1.5: 62% SIMPLER Google Robot — villa-X achieves 77.7%.
- Standard VLAs (OpenVLA): no latent action pretraining; limited cross-embodiment generalization.

### Motivation
Proprioceptive grounding (joint angles, end-effector state) captures manipulation details that pixel-space video cannot represent precisely. Joint latent+robot action diffusion enables the policy to use semantic latent plans as intermediate representations while maintaining full proprioceptive control fidelity.

---

## Method Details

### Model Architecture

villa-X has **two main components**:

1. **Latent Action Model (LAM)**: IDM + VQ codebook + visual FDM + proprioceptive FDM
2. **Actor Module (ACT)**: PaliGemma VLM + dual transformer experts (latent + robot actions)

### Core Modules

#### Module 1: Latent Action Model (LAM)

**Design Motivation**: LAM discovers compact discrete action representations from video + proprioception, providing semantic latent tokens that summarize "what task step is happening" in a cross-embodiment compatible way.

**Sub-components**:

**IDM (Inverse Dynamics Model)**:
- Architecture: Spatial-Temporal Transformer, 12 blocks, 768 hidden dim, 32 attention heads
- Input: video frames $(8 \times 224 \times 224)$ with 14×14 patch embedding
- Outputs latent action $z_t$ from frame pair $(x_t, x_{t+H})$

**VQ Module**:
- Codebook size: 32 discrete codes
- Quantizes IDM output to discrete latent action tokens

**Visual FDM (Forward Dynamics Model)**:
- 12-layer Vision Transformer
- Predicts future video frames given current frame + latent action

**Proprioceptive FDM**:
- 2-layer MLP conditioned on embodiment context $c_e$ (dataset ID + control frequency embeddings)
- Predicts future robot states and actions from current proprioceptive state, latent action, and embodiment context:

$$(\hat q_{t+1}, \ldots, \hat q_{t+k}, \hat a_{t+1}, \ldots, \hat a_{t+k}) = \text{proprio-FDM}(q_t, z_t, c_e)$$

- This proprioceptive grounding grounds latent actions in physical robot dynamics, capturing subtle motions invisible in pixel-space video.

**LAM Training**:
- Batch size: 512; LR: 1.5×10⁻⁴ (linear warmup 2,000 steps)
- Hardware: 128× A100 GPUs, ~4 days
- Data: 1.6M robot trajectories (OpenX + AgiBot) + 3.6M human video clips

#### Module 2: Actor Module (ACT)

**Design Motivation**: Conditioning robot action generation on latent action predictions provides semantic task-level guidance; dual-expert architecture allows specialization of latent and robot action prediction while maintaining joint optimization.

**Architecture**:
- VLM Encoder: PaliGemma (3B parameters, 224×224, 128-token text)
- ACT-latent expert: 18-layer Transformer, 1,024 hidden dim, 8 heads, sequence N=6 latent tokens
- ACT-robot expert: 18-layer Transformer, same dims, sequence M=4 robot action tokens
- Conditioning: ACT-robot attends to ACT-latent via blockwise causal cross-attention

The joint policy factorizes into a latent action predictor (conditioned only on observation + language) and a robot action predictor (additionally conditioned on latent actions, proprioception, and embodiment context):

$$\pi(a_{t:t+M-1}, z_{t:t+(N-1)k} \mid o_t, l, q_t, c_e) = \pi_{\text{robot}}(\cdot \mid z_{t:t+(N-1)k}, o_t, l, q_t, c_e) \cdot \pi_{\text{latent}}(z_{t:t+(N-1)k} \mid o_t, l)$$

**Stochastic Attention Masking**:
- 50% of training steps: fully mask robot→latent attention
- 50% of training steps: randomly mask
- Prevents over-reliance on latent predictions; improves robustness

**ACT Training**:
- LR: 5×10⁻⁵ (200-step warmup); gradient clip max norm 1.0
- Hardware: 64× A100, ~4 days
- Loss: conditional flow matching

---

## Experiments

### Datasets

| Dataset | Scale | Characteristics | Usage |
|---------|-------|----------------|-------|
| OpenX + AgiBot (robot) | 1.6M trajectories, 223.5M frames | Multi-platform robot manipulation | LAM + ACT training |
| Ego4D + EPIC-KITCHENS + SSv2 + others | 3.6M clips | Human manipulation videos | LAM video training |

### Results

The following table compares villa-X against GR00T-N1.5 on the SIMPLER benchmark and shows the effect of removing the latent action component.

**Table 1: SIMPLER Benchmark**

| Method | Google Robot | WidowX |
|--------|-------------|--------|
| GR00T-N1.5 | 62.0% | — |
| **villa-X** | **77.7%** | **62.5%** |
| villa-X (w/o latent) | 36.5% | 49.0% |

villa-X outperforms GR00T-N1.5 by +15.7% on Google Robot; removing latent actions causes −41.2% collapse — latent actions are critical.

**Table 2: LIBERO Benchmark**

| Suite | Success Rate |
|-------|-------------|
| Spatial | 97.5% |
| Object | 97.0% |
| Goal | 91.5% |
| Long | 74.5% |
| **Average** | **90.1%** |

**Table 3: Real-World — Xhand Dexterous Hand**

| Task | Success Rate |
|------|-------------|
| Pick & Place (seen) | 84% |
| Cube Stack (seen) | 75% |

**Table 4: Real-World — Realman Robot (Zero-Shot)**

| Task | Success Rate |
|------|-------------|
| Pick out | 100% |
| Stack | 50% |
| Unstack | 100% |

Zero-shot transfer to unseen Realman robot with 100% on pick and unstack.

The following ablations isolate the contribution of the proprioceptive FDM and the stochastic attention masking.

**Table 5: Ablation — LAM Components (WidowX)**

| Configuration | Success Rate |
|--------------|-------------|
| Full villa-X (w/ proprio-FDM) | 40.8% |
| Without proprio-FDM | 32.3% (−8.5%) |
| Without LAM | 33.1% (−7.7%) |

**Table 6: Ablation — Policy Components (Google Robot)**

| Configuration | Success Rate |
|--------------|-------------|
| **Full villa-X** | **58.5%** |
| w/o attention masking | 53.2% (−5.3%) |
| w/o embodiment context | 49.1% (−9.4%) |

### Implementation Details

- **IDM Backbone**: Spatial-Temporal Transformer (12 blocks, 768 dim, 32 heads)
- **VQ Codebook**: Size 32
- **Visual FDM**: 12-layer ViT
- **Proprio-FDM**: 2-layer MLP with embodiment context $c_e$
- **VLM Encoder**: PaliGemma (3B parameters, 224×224, 128-token text)
- **ACT Experts**: 18 layers each, 1,024 dim, 8 heads; N=6 latent, M=4 robot tokens
- **LAM Training**: Batch 512, LR 1.5e-4, 128×A100, 4 days
- **ACT Training**: Batch unspecified, LR 5e-5, 64×A100, 4 days
- **Stochastic Masking**: 50% full mask / 50% random mask
- **Policy Objective**: Conditional flow matching

---

## Critical Analysis

### Strengths
1. Proprioceptive FDM provides unique approach to grounding latent actions in physical dynamics (+8.5% without it).
2. Embodiment context conditioning handles heterogeneous multi-platform data effectively (+9.4% contribution).
3. 77.7% SIMPLER Google Robot — state-of-the-art among latent action methods.

### Limitations
1. **Codebook size only 32**: Very small vocabulary may not capture full manipulation diversity.
2. **LAM training scale**: 128 A100-days for LAM + 64 A100-days for ACT — expensive two-stage training.
3. **Stochastic masking heuristic**: 50%/50% split requires tuning; no principled derivation.

### Potential Improvements
1. Larger VQ codebook for richer latent action vocabulary.
2. End-to-end joint LAM + ACT training to avoid two-stage pipeline.
3. Adaptive attention masking based on latent prediction confidence.

### Reproducibility
- [ ] Code open-sourced
- [ ] Pretrained models available
- [x] Training details described (LR, batch, hardware, duration)
- [x] OpenX, AgiBot datasets publicly available

---

## Related Notes

### Based On
- PaliGemma: 3B VLM encoder for vision-language conditioning
- VQ-VAE: Discrete latent action codebook
- Flow Matching: Action generation training paradigm

### Compared Against
- GR00T-N1.5: Nvidia robot foundation model; outperformed 62%→77.7% SIMPLER
- LAPA: Latent action pretraining; villa-X improves with proprio-FDM

### Method Related
- Latent Action Model: Discrete latent action discovery from video+proprioception
- Dual Expert Architecture: Specialized latent and robot action transformers
- Stochastic Attention Masking: Training regularization for action-latent coupling

---

## Quick Reference Card

> [!summary] villa-X (arXiv 2025)
> - **Core**: LAM (ST-Transformer IDM + VQ codebook 32 + visual FDM + proprio-FDM) on 5.2M clips; ACT (PaliGemma 3B + dual 18-layer experts, N=6 latent / M=4 robot, flow matching); stochastic attention masking 50%/50%; embodiment context $c_e$
> - **Method**: LAM: batch 512, LR 1.5e-4, 128×A100 4 days; ACT: LR 5e-5, 64×A100 4 days; 1.6M robot + 3.6M human clips
> - **Results**: 77.7% SIMPLER Google Robot (vs. 62% GR00T-N1.5); 90.1% LIBERO avg; 100% zero-shot Realman pick/unstack
> - **Code**: N/A

---

*Note created: 2026-05-20*
