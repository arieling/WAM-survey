---
title: "Vidar: Embodied Video Diffusion Model for Generalist Manipulation"
method_name: "Vidar"
authors: [Yao Feng, Hengkai Tan, Xinyi Mao, Chendong Xiang, Guodong Liu, Shuhe Huang, Hang Su, Jun Zhu]
year: 2025
venue: arXiv 2025
tags: [video-generation, bimanual, inverse-dynamics, cross-embodiment, robot-policy, cascaded-explicit, flow-matching]
zotero_collection: 3-Robotics/WAM-Survey
image_source: online
arxiv_html: https://arxiv.org/abs/2507.12898
created: 2026-05-20
---

# Paper Note: Vidar: Embodied Video Diffusion Model for Generalist Manipulation

## Metadata

| Item | Content |
|------|---------|
| Institution | Tsinghua University, Shengshu Tech |
| Date | July 2025 (v4: December 2025) |
| Project Page | N/A |
| Baselines | [UniPi](UniPi.md), VPP, Pi0.5 |
| Links | [arXiv](https://arxiv.org/abs/2507.12898) / Code: Supplemental materials |

---

## One-Sentence Summary

> Vidar enables few-shot cross-embodiment bimanual manipulation by combining internet-scale video diffusion (rectified flow) pre-trained on a unified multi-view observation space from 750K robot episodes with a Masked Inverse Dynamics Model (MIDM) that learns action-relevant pixel masks without dense supervision — requiring only ~20 minutes of demonstrations on an unseen robot platform.

---

## Core Contributions

1. **Three-Stage Training Recipe ("One Prior, Many Embodiments")**: Internet pre-training → embodied domain pre-training on 750K multi-view bimanual clips across 3 robot platforms → few-shot target domain fine-tuning (~232 episodes, 20 min).
2. **Unified Observation Space**: Encodes robot platform, camera configuration, task, and multi-view scene jointly as structured text instruction + aggregated multi-view image tensor — enabling cross-embodiment generalization without action labels in the video model.
3. **Masked Inverse Dynamics Model (MIDM)**: Lightweight U-Net mask predictor + ResNet action regressor trained end-to-end with Huber loss + L1 sparsity regularization — learns to focus on hands/tools/contact regions without pixel-level annotation.
4. **Test-Time Scaling (TTS)**: Generate $K=3$ candidate videos with different random seeds; rank by GPT-4o evaluating physical plausibility and instruction alignment; select best — reduces stochastic variance at inference.

---

## Problem Background

### Problem to Solve
Scaling bimanual manipulation to new robot embodiments: each platform typically requires large, homogeneous demonstrations; end-to-end pixel-to-action pipelines degrade under background/viewpoint shifts. How to efficiently adapt to a new platform with only ~20 minutes of demonstrations?

### Limitations of Existing Methods
- VLA models (RT-1, RT-2, Pi0): require 130K–1B labeled demonstrations; actions tightly coupled to model architecture; hard to transfer.
- UniPi: single-platform video diffusion + inverse dynamics; does not leverage heterogeneous cross-embodiment data.
- VPP: uses single denoising step features for action prediction — noisy and unstable in unseen environments.
- Standard inverse dynamics: suffers from background noise and texture biases; poor generalization to new backgrounds.

### Motivation
Video diffusion models trained on diverse, cross-embodiment data learn embodiment-agnostic interaction priors (affordances, contacts, motion continuity). Factorizing the policy through video space — $\pi = I \circ G$ — separates the representation burden (handled by large $G$ pretrained on abundant data) from the lightweight adaptation (handled by small $I$ trained with few demonstrations). MIDM addresses the action extraction gap by learning what to look at, not just what to see.

---

## Method Details

### Model Architecture

Vidar has **two main components**:

1. **Embodied Video Foundation Model** $G: \mathcal L \times \mathcal O \to \mathbb P(\mathcal V)$: Rectified flow video diffusion model; Wan2.2 (5B) for simulation, Vidu 2.0 for real-world; conditioned on unified observation space; generates 60-frame videos at 8fps
2. **Masked Inverse Dynamics Model** $I: \mathcal V \to \mathcal A$: U-Net mask predictor (92M) + ResNet-50 action regressor; trained exclusively on fine-tuning data; converts video windows to robot joint commands

### Core Modules

#### Module 1: Unified Observation Space

**Design Motivation**: Heterogeneous embodiments have different morphologies, camera configurations, and viewpoints. A unified space that jointly encodes all context allows a single video model to learn across platforms.

**Implementation**: The unified observation space pairs an aggregated multi-view image tensor with a structured language description covering robot platform, camera layout, and task:

$$\mathcal U = \{\langle \mathbf o, \mathbf l \rangle \mid \mathbf o = \text{aggregate}(\mathbf I^{(1)}, \ldots, \mathbf I^{(V)}),\; \mathbf l = \text{concatenate}(l_r, l_c, l_t)\}$$

where $\mathbf o = \bigoplus_{k=1}^V \phi_{r_k}(\mathbf I^{(k)})$ is the spatially resized and tiled aggregation of up to $V$ camera views; $l_r$ is the robot platform instruction (e.g., "The aloha robot is currently performing..."); $l_c$ is the camera configuration description; and $l_t$ is the task instruction. The video model conditions on $\mathbf l$ via text encoding and does not predict actions (action-free video generation).

#### Module 2: Rectified Flow Video Generation Model

**Design Motivation**: Rectified flow models produce high-quality videos with straight-line ODE trajectories and fewer sampling steps than DDPM.

**Implementation**:
- Parameterizes velocity field $v: \mathcal V \times \mathbb R \times \mathcal L \to \mathcal V$
- ODE: $dx_t/dt = v(x_t, t, c)$ from Gaussian $x_0$ to video $x_1$
- The rectified flow training objective trains the velocity field $v$ to predict the constant flow direction from Gaussian noise to the target video:

$$\mathcal L_G = \mathbb E_{c, t, x_0, x_1}\left[\left\|(x_1 - x_0) - v(tx_1 + (1-t)x_0, t, c)\right\|^2\right]$$

where $x_0 \sim \mathcal N(0, I)$ is Gaussian noise, $x_1$ is the target video frame sequence, $t \in [0, 1]$ is flow time, and $c$ is the structured conditioning from unified observation space $\mathcal U$.

- Pre-trained on internet-scale videos; continues pre-training on 750K robotic episodes; fine-tuned on target platform with SFT (full-parameter)
- Batch size: 128; Pre-training: 10K steps; Fine-tuning: 12K steps (Wan2.2) / 13K steps (Vidu 2.0)
- Hardware: 64× Ampere-series 80GB GPUs; ~64 hours for Vidu 2.0 full training
- Inference: 60 frames @ 8fps = 7.5 seconds; ~25 seconds per video on 8× A100-80GB

The overall policy factorization through video space is written as:

$$\pi = I \circ G, \quad G: \mathcal L \times \mathcal O \to \mathbb P(\mathcal V), \quad I: \mathcal V \to \mathcal A$$

This factorization separates the large pretrained video generator $G$ from the lightweight inverse dynamics model $I$, enabling few-shot adaptation to new embodiments by retraining only $I$.

#### Module 3: Masked Inverse Dynamics Model (MIDM)

**Design Motivation**: Standard inverse dynamics models (ResNet) memorize background textures in training but fail to generalize to new scenes. Learning a spatial mask that selectively attends to action-relevant regions (robot arms, tools, contact patches) provides background-robust action extraction.

**Implementation**:
- Mask predictor $U$: U-Net architecture, 5 down/up-sampling layers, 92M parameters
- Action regressor $R$: ResNet-50
- Forward pass: $m = U(x)$, $\hat a = R(\text{Round}(m) \odot x)$
- The MIDM training loss combines a Huber action regression term with an L1 sparsity regularizer that encourages minimal, task-critical masks without segmentation supervision:

$$\mathcal L_I = \mathbb E_{x, a}\left[l(\hat a - a) + \lambda \|m\|_1\right], \quad m = U(x), \quad \hat a = R(\text{Round}(m) \odot x)$$

where $m \in [0, 1]^{H \times W}$ is the spatial attention mask, $l(\cdot)$ is the Huber loss, $\lambda = 3 \times 10^{-3}$ is the sparsity weight, and $\odot$ is element-wise multiplication (mask application). Straight-through estimators allow training through the Round(·) operation.

- $\lambda = 3 \times 10^{-3}$ (optimal; ablated across 5 values)
- Training: 60K iterations, 8× Hopper-series 80GB GPUs, ~5 hours; AdamW lr=5e-4, warmup 6K steps

Sample MIDM visualizations show input images from unseen backgrounds (including reflective surfaces) paired with MIDM-predicted masked images. The masks correctly attend to robot arm regions while ignoring background clutter, validating background-robust action extraction.

#### Module 4: Test-Time Scaling (TTS)

**Design Motivation**: Video diffusion is stochastic — single samples may be physically implausible or task-irrelevant. Rejection sampling via a pretrained VLM evaluator selects the most reliable rollout.

**Implementation**:
- Generate $K=3$ candidate videos with different random seeds in parallel
- Rank candidates using GPT-4o: assess physical plausibility + instruction alignment from 5–7 sampled frames
- Select $\arg\max_i q_\eta(\tilde v^{(i)}_{1:T})$
- Cost: ~$0.003 per comparison; TTS accounts for ~25% of total inference latency

---

## Experiments

### Datasets

| Dataset | Scale | Characteristics | Usage |
|---------|-------|----------------|-------|
| Agibot-World | Part of 750K | 3-view bimanual, Genie-1 robot | Embodied pre-training |
| RoboMind | Part of 750K | Franka Panda + Aloha, varied tasks | Embodied pre-training |
| RDT | Part of 750K | ALOHA bimanual | Embodied pre-training |
| Egodex | Supplementary | Egocentric dexterous manipulation | Embodied pre-training (sim only) |
| RoboTwin 2.0 | 20/50 episodes per task | Aloha (agilex), 50 tasks, multi-task | Target fine-tuning + evaluation |
| Vidar Real | 232 episodes (~20 min) | Custom Aloha, 81 tasks, 3 views | Target fine-tuning + evaluation |

### Results

The following table benchmarks Vidar on RoboTwin 2.0 across low-data and standard regimes with clean and randomized scenes.

**Table 1: RoboTwin 2.0 Benchmark (50 tasks, 100 episodes, multi-task setting)**

| Data Regime | Scenario | Pi0* (single-task) | Pi0.5 | **Vidar** |
|-------------|---------|-------------------|-------|----------|
| Low | Clean | — | 25.0% | **60.0%** |
| Low | Randomized | — | 9.2% | **15.7%** |
| Standard | Clean | 46.42% | 44.8% | **65.8%** |
| Standard | Randomized | 16.34% | 14.2% | **17.5%** |

Vidar substantially outperforms Pi0.5 across all settings, especially on clean scenarios (60% vs 25% low-data, 65.8% vs 44.8% standard). Pi0* trains a separate model per task and is not directly comparable.

The following table reports success rates on the real-world Aloha bimanual platform trained with only ~20 minutes of demonstrations.

**Table 2: Real-World Success Rates (Aloha bimanual, ~20 min demos)**

| Method | Seen Tasks & Backgrounds | Unseen Tasks | Unseen Backgrounds |
|--------|-------------------------|--------------|-------------------|
| VPP | 4.5% | 13.3% | 0.0% |
| UniPi | 36.4% | 6.7% | 22.2% |
| **Vidar (Ours)** | **68.2%** | **66.7%** | **55.6%** |

Vidar achieves 58% improvement over VPP (4.5% → 68.2%) and 40% over UniPi (36.4% → 68.2%) on seen tasks. Crucially, Vidar generalizes to unseen tasks (66.7% vs. 6.7% UniPi) and unseen backgrounds including reflective surfaces (55.6% vs. 0% VPP).

The following table measures the effect of embodied pre-training on VBench video quality metrics.

**Table 3: VBench Video Quality (Embodied Pre-training Ablation)**

| Configuration | Subject Consistency | Background Consistency | Imaging Quality |
|--------------|--------------------|-----------------------|-----------------|
| Vidu 2.0 (base) | 0.565 | 0.800 | 0.345 |
| + Embodied Pre-training | **0.855** | **0.909** | **0.667** |

Embodied pre-training dramatically improves subject consistency (0.565→0.855) and imaging quality (0.345→0.667) for robot manipulation video generation.

The following table isolates the generalization benefit of MIDM over standard ResNet inverse dynamics.

**Table 4: MIDM vs. ResNet Inverse Dynamics**

| Model | Training Accuracy | Testing Accuracy | Testing $l_1$ Error |
|-------|------------------|-----------------|-------------------|
| ResNet | 99.9% | 24.3% | 0.0430 |
| **MIDM (Ours)** | **99.9%** | **49.0%** | **0.0308** |

Both models memorize the training set (99.9%). ResNet completely fails to generalize (24.3% test accuracy). MIDM achieves 49.0% — +24.7% — by learning to mask irrelevant background information.

**Table 5: Ablation Study**

| Configuration | Seen Tasks | Unseen Tasks | Unseen Backgrounds |
|--------------|-----------|--------------|-------------------|
| Vidar w/o TTS | 45.5% | 33.3% | 44.4% |
| Vidar w/o MIDM | 59.1% | 26.7% | 22.2% |
| **Vidar (Full)** | **68.2%** | **66.7%** | **55.6%** |

Both TTS (+22.7% on seen tasks) and MIDM (+40% on unseen tasks vs. w/o MIDM) contribute substantially; MIDM is especially critical for unseen background generalization.

**Table 6: Wan2.2 Real-World Results vs. Pi0.5**

| Scenario | Vidar (Wan2.2) | Pi0.5 |
|----------|---------------|-------|
| 7 Seen Tasks | **69.3%** | 34.3% |
| 7 Unseen Tasks | **67.1%** | 12.9% |

### Implementation Details

- **Video Models**: Wan2.2 (5B parameters, simulation), Vidu 2.0 (real-world), HunyuanVideo (13B, ablation)
- **Wan2.2 Hyperparameters**: Pre-train lr=2e-5, fine-tune lr=2e-5, warmup 200 steps, AdamW weight decay 0.1, pre-train 10K steps, fine-tune 12K steps
- **Vidu 2.0**: Batch size 128, 23K total iterations (10K pre-train + 13K fine-tune), full-parameter SFT, ~64 hours on 64× Ampere 80GB GPUs
- **Video Sampling Rate**: 8 fps, downsampled from original; 60 frames = 7.5 sec video
- **MIDM Parameters**: 92M; U-Net 5 up/down-sampling layers; ResNet-50; Huber loss; $\lambda = 3 \times 10^{-3}$; lr=5e-4; warmup 6K steps; AdamW $\beta=(0.9, 0.999)$, weight decay=$10^{-2}$; 60K iterations on 8× Hopper 80GB, ~5 hours
- **Inference Mode**: Open-loop control; videos generated once before execution
- **TTS**: $K=3$ candidates, GPT-4o ranking of 5–7 sampled frames; ~$0.003 per comparison
- **Robot Platform**: 14-DoF bimanual Aloha; 3 RGB cameras; 3.9 kg arm; 1mm repeatability

---

## Critical Analysis

### Strengths
1. "One prior, many embodiments" recipe is highly practical: scales to new platforms with only ~20 minutes of data.
2. MIDM elegantly solves background generalization without any dense annotation — just action supervision + sparsity regularization.
3. Strong empirical results across seen/unseen tasks and backgrounds validate the approach comprehensively.

### Limitations
1. **Open-loop control only**: Videos generated once; no mid-execution replanning. Sensitive to initial video quality.
2. **Video visibility assumption**: MIDM assumes arms are visible in video frames — breaks for highly occluded bimanual setups (noted in Appendix F).
3. **Inference latency**: ~25 seconds per video generation; TTS adds 25% overhead. Not suitable for reactive tasks.
4. **TTS cost**: GPT-4o API dependency for production deployment adds per-inference cost.

### Potential Improvements
1. Closed-loop replanning by regenerating video from current observation when task stalls.
2. Distillation/quantization for real-time inference speedup (explicitly noted as future work).
3. Multi-turn MIDM to handle arm occlusion by predicting from multiple camera views jointly.

### Reproducibility
- [x] Code open-sourced (supplemental materials; HunyuanVideo + MIDM)
- [ ] Full pretrained models available
- [x] Training details complete (Appendix A, B with full hyperparameter tables)
- [x] Datasets accessible (Agibot-World, RoboMind, RDT, Egodex all public)

---

## Related Notes

### Based On
- Rectified Flow: Core video generation model (Wan2.2, Vidu 2.0, HunyuanVideo)
- Flow Matching: Training objective for velocity field learning
- Inverse Dynamics Model: Action extraction approach

### Compared Against
- [UniPi](UniPi.md): Single-platform video + inverse dynamics baseline; outperformed +32% on seen tasks
- [VPP](VPP.md): Video prediction policy using single denoising step features; outperformed dramatically

### Method Related
- Test-Time Scaling: Compute-scaled inference via rejection sampling
- Unified Observation Space: Novel multi-view cross-embodiment conditioning design
- Masked Attention: MIDM's sparsity-guided attention to action-relevant regions

### Hardware/Data Related
- ALOHA Robot: Target bimanual platform (14-DoF, 3 cameras)
- RoboTwin: Simulation benchmark for bimanual manipulation evaluation
- Agibot-World: Large-scale embodied pre-training dataset

---

## Quick Reference Card

> [!summary] Vidar (arXiv 2025)
> - **Core**: Cross-embodiment video diffusion (rectified flow, 750K pre-training episodes, unified observation space) + MIDM (92M U-Net mask predictor + ResNet-50 action regressor) for few-shot bimanual manipulation
> - **Method**: Wan2.2/Vidu 2.0 video models; robot+camera+task unified text conditioning; MIDM with Huber+L1 loss ($\lambda=3\times10^{-3}$); TTS with K=3, GPT-4o ranking
> - **Results**: 68.2% seen / 66.7% unseen tasks / 55.6% unseen backgrounds (vs. 36.4% / 6.7% / 22.2% UniPi); 65.8% RoboTwin standard vs. 44.8% Pi0.5
> - **Code**: Supplemental (HunyuanVideo + MIDM)

---

*Note created: 2026-05-20*
