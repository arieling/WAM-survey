---
title: "Learning to Act from Actionless Videos through Dense Correspondences"
method_name: "AVDC"
authors: [Po-Chen Ko, Jiayuan Mao, Yilun Du, Shao-Hua Sun, Joshua B. Tenenbaum]
year: 2023
venue: ICLR 2024
tags: [video-generation, optical-flow, dense-correspondences, robot-policy, action-free, cascaded-explicit]
zotero_collection: 3-Robotics/WAM-Survey
image_source: online
arxiv_html: https://arxiv.org/abs/2310.08576
created: 2026-05-18
---

# Paper Note: Learning to Act from Actionless Videos through Dense Correspondences

## Metadata

| Item | Content |
|------|---------|
| Institution | National Taiwan University, MIT |
| Date | October 2023 (ICLR 2024) |
| Project Page | https://flow-diffusion.github.io/ |
| Baselines | BC-Scratch, R3M+BC, S3D+BC, UniPi, SuSIE |
| Links | [arXiv](https://arxiv.org/abs/2310.08576) / [Code](https://github.com/flow-diffusion/AVDC) |

---

## One-Sentence Summary

> AVDC learns robot manipulation policies from actionless videos by generating future video plans with a text-conditioned diffusion model, extracting dense optical flow correspondences between frames, and converting SE(3) rigid body transforms directly into robot arm commands — requiring zero action labels during training.

---

## Core Contributions

1. **Action-Label-Free Policy Learning**: Demonstrates that robot policies can be learned from videos with no action annotations, using only paired text descriptions and video observations.
2. **Dense Flow-to-Action Pipeline**: Introduces a 3-stage pipeline: video diffusion → optical flow (GMFlow) → SE(3) transform estimation → robot execution, bridging the visual plan to motor command gap geometrically.
3. **Zero-Shot Cross-Environment Transfer**: A single video diffusion model trained on Meta-World simulation videos transfers to Visual Pusher and iTHOR environments without any environment-specific fine-tuning.
4. **Closed-Loop Replanning**: Integrates a replanning trigger (< 1mm movement over 15 steps) to handle execution failures and partially completed tasks, extending open-loop video plans to robust closed-loop control.

---

## Problem Background

### Problem to Solve
How to learn robot manipulation policies when action labels are unavailable — only videos of task execution (paired with text descriptions) are provided. This reflects the realistic constraint that large-scale video datasets (e.g., human demonstrations, internet videos) rarely contain robot joint angles or end-effector actions.

### Limitations of Existing Methods
- Standard behavioral cloning (BC-Scratch, R3M+BC) requires action annotations for every training frame.
- UniPi generates video plans but uses inverse dynamics models that still need some action-labeled data.
- SuSIE uses video synthesis for subgoal generation but relies on downstream BC policies trained with actions.
- Optical flow methods exist but had not been applied to bridge video plan execution to robot control without action labels.

### Motivation
Dense optical flow between synthesized future frames encodes rich geometric information about object and scene motion. By estimating the SE(3) rigid body transform that best explains the pixel-level flow (combined with depth), one can directly compute camera/end-effector displacement without ever observing action labels.

---

## Method Details

### Model Architecture

AVDC is a **3-stage pipeline** with no end-to-end training signal:

- **Stage 1 — Video Diffusion Model**: Text-conditioned image-to-video diffusion model generating $T=8$ future frames from the current observation
  - Architecture: Video U-Net with factorized spatio-temporal convolutions (spatial 2D conv + temporal 1D conv)
  - Text conditioning: CLIP-Text encoder
  - Noise schedule: linear $\beta$ schedule, $T_{noise}=1000$ steps
  - Parameters: 109M–201M (depending on resolution variant)
  - Training: 4× V100 GPUs, ~1 day, Adam optimizer, lr=5e-5
- **Stage 2 — Optical Flow Estimation**: [[GMFlow]] computes dense optical flow between consecutive synthesized frame pairs $(I_t, I_{t+1})$
  - Output: pixel-level displacement field $(u^i_t, v^i_t)$ for all pixels $i$
  - GMFlow: frozen pretrained model, no fine-tuning
- **Stage 3 — SE(3) Transform Estimation**: Recovers rigid body transform $T_t \in SE(3)$ from flow + depth
  - Depth: monocular depth estimator applied to synthesized frames (or ground-truth depth in sim)
  - Optimization: minimizes reprojection error of optical flow under rigid transform assumption
  - Output: end-effector displacement $(\Delta x, \Delta y, \Delta z, \Delta \text{rotation})$ per step
  - Execution: robot arm commanded with $T_t$ deltas, gripper open/close separately predicted

### Core Modules

#### Module 1: Text-Conditioned Video Diffusion (Stage 1)

**Design Motivation**: Synthesize physically plausible future frames conditioned on a text task description, serving as a "video plan" for the manipulation task.

**Implementation**:
- Input: current observation frame $I_0$ + text description $c$
- Factorized spatio-temporal U-Net: 2D spatial convolutions handle per-frame appearance, 1D temporal convolutions handle cross-frame motion consistency
- CLIP-Text encoder encodes $c$ into cross-attention conditioning vectors
- Generates $T=8$ frames at 64×64 resolution (with optional upsampling)

The video diffusion model is trained to denoise noisy video sequences conditioned on text. The training objective minimizes the mean-squared error between the predicted and true noise:

$$
\mathcal{L}_{MSE} = \|\epsilon - \epsilon_\theta(\sqrt{1-\beta_t}\, img_{1:T} + \sqrt{\beta_t}\,\epsilon,\; t \mid txt)\|^2
$$

where $img_{1:T}$ are the clean video frames, $\epsilon \sim \mathcal{N}(0, I)$ is the added Gaussian noise, $\beta_t$ is the noise schedule scalar, $\epsilon_\theta$ is the video U-Net parameterized by $\theta$, and $txt$ is the CLIP-encoded text description.

#### Module 2: Dense Optical Flow via GMFlow (Stage 2)

**Design Motivation**: Optical flow provides pixel-level geometric motion evidence that can be converted to SE(3) transforms without any action supervision.

**Implementation**:
- Apply frozen [[GMFlow]] between every consecutive pair $(I_t, I_{t+1})$ in generated video
- GMFlow: Transformer-based global matching flow estimator
- Output: flow field $\mathbf{F}_t = \{(u^i_t, v^i_t)\}$ for pixels $i$
- Flow computed on generated frames, not real observations — avoids domain gap issues

#### Module 3: SE(3) Rigid Transform Estimation (Stage 3)

**Design Motivation**: Under the rigid body assumption (robot arm dominates scene motion), the optical flow field encodes the 6-DoF end-effector transform.

**Implementation**:
- Lift flow correspondences to 3D using depth map $D_t$ and camera intrinsics $K$: $x_i = K^{-1}[u^i, v^i, 1]^T \cdot D_t(u^i, v^i)$

Given the lifted 3D points, the SE(3) rigid body transform $T_t$ is recovered by minimizing the reprojection error of the optical flow under the rigid transform assumption:

$$
\mathcal{L}_{Trans} = \sum_i \left\| u^i_t - \frac{(KT_t x_i)_1}{(KT_t x_i)_3} \right\|^2_2 + \left\| v^i_t - \frac{(KT_t x_i)_2}{(KT_t x_i)_3} \right\|^2_2
$$

Here $u^i_t, v^i_t$ are the optical flow displacements for pixel $i$ from GMFlow, $K$ is the camera intrinsic matrix, $T_t \in SE(3)$ is the rigid transform to recover, $x_i$ is the 3D point lifted from pixel $i$, and $(\cdot)_1, (\cdot)_2, (\cdot)_3$ denote the x, y, z components after projection. The recovered $T_t$ is directly converted to robot end-effector delta commands via known kinematics.

#### Module 4: Closed-Loop Replanning

**Design Motivation**: Open-loop video plans accumulate error; replanning from current observation corrects drift.

**Implementation**:
- Monitor end-effector position at each step
- Trigger replanning if movement < 1mm over 15 consecutive steps AND task not detected as complete
- Also replan after completing each video plan segment (receding horizon)
- New video generated conditioned on current real observation

---

## Experiments

### Datasets

| Dataset | Scale | Characteristics | Usage |
|---------|-------|----------------|-------|
| Meta-World (sim videos) | ~50k video clips | 50 robot manipulation tasks, no action labels | Training video diffusion model |
| Visual Pusher | ~500 eval episodes | Continuous pushing task | Zero-shot evaluation |
| iTHOR | ~600 eval episodes | Household navigation+manipulation | Zero-shot evaluation |

### Meta-World Task Success Rates

The table below compares AVDC against action-supervised baselines across difficulty levels, demonstrating that actionless training with video diffusion + dense correspondences outperforms conventional behavioral cloning:

| Method | Overall | Easy | Medium | Hard |
|--------|---------|------|--------|------|
| BC-Scratch | 16.2% | 34.0% | 12.1% | 2.5% |
| R3M + BC | 18.9% | 38.3% | 15.2% | 3.3% |
| S3D + BC | 14.1% | 28.7% | 11.3% | 2.3% |
| UniPi | 10.5% | 22.5% | 7.6% | 1.5% |
| **AVDC (Ours)** | **43.1%** | **72.5%** | **36.4%** | **21.0%** |

AVDC achieves 43.1% overall success vs. 18.9% for the best action-supervised baseline (R3M+BC).

### Zero-Shot Cross-Environment Transfer

A single AVDC model trained only on Meta-World simulation videos achieves strong zero-shot performance on two held-out environments, demonstrating that the geometry-based action extraction pipeline generalizes without environment-specific training:

| Method | Visual Pusher (Success) | iTHOR (Success) |
|--------|------------------------|-----------------|
| BC-Scratch | 0% | N/A |
| UniPi | 30% | 18.7% |
| SuSIE | 55% | 25.0% |
| **AVDC (Ours)** | **90%** | **31.3%** |

### Ablation Study

The following ablation isolates the contribution of each AVDC component:

| Configuration | Meta-World Overall |
|---------------|--------------------|
| AVDC (No Replanning) | 32.4% |
| AVDC (No Depth) | 28.7% |
| AVDC (Sparse Flow) | 36.2% |
| **AVDC (Full)** | **43.1%** |

Replanning (+10.7%), depth information (+14.4%), and dense flow (+6.9%) all contribute significantly to overall performance.

### Implementation Details

- **Video Diffusion Backbone**: Video U-Net, factorized spatio-temporal conv (2D spatial + 1D temporal)
- **Text Encoder**: CLIP-Text (frozen pretrained)
- **Parameters**: 109M–201M (resolution-dependent)
- **Optimizer**: Adam, lr=5e-5
- **Batch Size**: 16
- **Training Steps**: 200k
- **Hardware**: 4× NVIDIA V100 GPUs, ~1 day training
- **Inference Resolution**: 64×64 (video generation), upsampled for flow
- **Optical Flow Model**: GMFlow (frozen pretrained, no fine-tuning)
- **Depth Estimator**: Monocular depth estimation applied to synthesized frames
- **Replanning Threshold**: < 1mm movement over 15 steps

---

## Critical Analysis

### Strengths
1. Eliminates action label requirement entirely — learns manipulation from pure video+text, opening the door to internet-scale video training.
2. Geometric action extraction (SE(3) from flow+depth) is interpretable and domain-agnostic — no environment-specific modules needed.
3. Strong zero-shot transfer (90% on Visual Pusher) demonstrates that geometry-based pipeline generalizes across tasks and environments.

### Limitations
1. **Rigid body assumption**: SE(3) estimation breaks for deformable objects, soft materials, or multi-body scenes with independent motions.
2. **Depth quality dependency**: Monocular depth estimators are noisy; errors propagate directly to SE(3) estimation and robot commands.
3. **Short video horizon**: Only 8 frames per plan; complex multi-step tasks require many replanning cycles, increasing computation.
4. **Sim-to-real gap**: Video diffusion trained on simulation; real-robot deployment requires either real-data fine-tuning or domain adaptation.

### Potential Improvements
1. Replace monocular depth with stereo or RGB-D sensors for more accurate 3D lifting.
2. Extend to deformable object manipulation by relaxing rigid body assumption (e.g., per-segment rigid priors).
3. Scale video diffusion training to real internet videos for direct real-world deployment.
4. Combine with goal-conditioned policies (as in VLP) for more robust execution.

### Reproducibility
- [x] Code open-sourced (https://github.com/flow-diffusion/AVDC)
- [ ] Pretrained models available
- [x] Training details complete (Section 4 + Appendix)
- [x] Datasets accessible (Meta-World sim, iTHOR, Visual Pusher)

---

## Related Notes

### Based On
- [[Video Diffusion Model]]: Core generation backbone for synthesizing future frame sequences
- [[GMFlow]]: Optical flow estimator used for dense correspondence extraction
- [[SE(3) Transform]]: Rigid body geometry for converting flow to robot commands
- [[Classifier-Free Guidance]]: Text+frame conditioning during video generation

### Compared Against
- [[UniPi]]: Video-as-policy baseline; AVDC outperforms by ~4x without action labels
- [[SuSIE]]: Video subgoal synthesis baseline; outperformed on cross-environment transfer

### Method Related
- [[Optical Flow]]: Dense pixel correspondences between video frames
- [[Inverse Dynamics Model]]: Alternative action extraction approach (requires action labels)
- [[Receding Horizon Control]]: Replanning strategy shared with VLP
- [[Goal-Conditioned Policy]]: Alternative execution module

### Hardware/Data Related
- [[Meta-World]]: Simulation benchmark used for training and primary evaluation
- [[iTHOR]]: Household manipulation environment for zero-shot evaluation

---

## Quick Reference Card

> [!summary] AVDC (ICLR 2024)
> - **Core**: Learn manipulation from actionless videos: video diffusion → optical flow (GMFlow) → SE(3) transform → robot commands
> - **Method**: Text-conditioned video U-Net (109-201M, 4×V100, 1 day); GMFlow flow estimation; rigid body SE(3) optimization; closed-loop replanning
> - **Results**: 43.1% Meta-World overall (vs. 16.2% BC-Scratch); 90% Visual Pusher zero-shot; 31.3% iTHOR zero-shot
> - **Code**: https://github.com/flow-diffusion/AVDC

---

*Note created: 2026-05-18*
