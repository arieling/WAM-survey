---
title: "TesserAct: Learning 4D Embodied World Models"
method_name: "TesserAct"
authors: [Haoyu Zhen, Qiao Sun, Hongxin Zhang, Junyan Li, Siyuan Zhou, Yilun Du, Chuang Gan]
year: 2025
venue: arXiv 2025
tags: [4d-world-model, video-generation, depth-normal, robot-policy, 3d-reconstruction, cascaded-explicit]
zotero_collection: 3-Robotics/WAM-Survey
image_source: online
arxiv_html: https://arxiv.org/abs/2504.20995
created: 2026-05-20
---

# Paper Note: TesserAct: Learning 4D Embodied World Models

## Metadata

| Item | Content |
|------|---------|
| Institution | UMass Amherst, MIT |
| Date | April 2025 |
| Project Page | https://tesseractworld.github.io/ |
| Baselines | [UniPi](UniPi.md), Image-BC, OpenSora, CogVideoX, 4D Point-E, Shape of Motion |
| Links | [arXiv](https://arxiv.org/abs/2504.20995) / Code: N/A |

---

## One-Sentence Summary

> TesserAct learns 4D embodied world models by fine-tuning CogVideoX to jointly generate RGB, Depth, and Normal (RGB-DN) videos, then reconstructing temporally-consistent 4D point clouds from these predictions via optical-flow-constrained depth optimization — enabling novel view synthesis and robot policies that significantly outperform 2D video baselines.

---

## Core Contributions

1. **RGB-DN Video Generation**: Extends video diffusion (CogVideoX) to jointly predict RGB, depth, and surface normal maps per frame through separate modality projectors, producing geometrically-grounded video predictions.
2. **4D Scene Reconstruction Algorithm**: Converts generated RGB-DN videos into full 4D scenes via normal-guided depth refinement + optical-flow-based temporal consistency constraints — 1 minute vs. 2 hours for optimization-based methods.
3. **Cross-Domain Training Dataset**: Compiles 285k annotated videos across synthetic (RLBench), real robot (RT-1, Bridge), and human hand (SomethingSomethingV2) with depth/normal estimated by off-the-shelf models.
4. **3D-Aware Robot Policy**: PointNet encoder on predicted point clouds + MLP inverse dynamics extracts 7-DoF actions, outperforming 2D video-based policies by ~10-15% on RLBench tasks.

---

## Problem Background

### Problem to Solve
Standard video-based world models predict future RGB frames but lack explicit 3D geometric understanding — they cannot reason about object shape, depth, or 3D configuration changes. This limits their use for tasks requiring spatial reasoning (grasping, placement, novel view synthesis).

### Limitations of Existing Methods
- 2D video models (UniPi, AVDC): predict pixel-level futures without geometric structure; no explicit depth/shape reasoning.
- 3D-VLA: predicts only goal states, trained on synthetic data only, no temporal dynamics.
- 4D generation (Shape of Motion, DynamicGaussians): high-quality but require hours of per-scene optimization; not applicable to real-time robot planning.
- Naive depth estimation post-processing: loses temporal consistency across frames.

### Motivation
Depth and surface normals are strongly geometrically coupled to RGB — a video diffusion model trained to jointly predict all three modalities learns physically-consistent 3D scene dynamics. The RGB-DN representation is cheap to predict (reuses video architecture) yet provides rich geometric information for 3D reasoning and novel view synthesis.

---

## Method Details

### Model Architecture

TesserAct has **three components**:

1. **RGB-DN Video Diffusion Model**: CogVideoX DiT fine-tuned to jointly denoise RGB + depth + normal latents
2. **4D Scene Reconstruction**: Normal-integration + optical-flow consistency optimization → temporally consistent depth maps → 4D point cloud
3. **Robot Policy**: PointNet on 4D point clouds + MLP inverse dynamics → 7-DoF actions

### Core Modules

#### Module 1: RGB-DN Video Diffusion

**Design Motivation**: Jointly predicting RGB, depth, and normal in a single pass ensures geometric consistency — depth and normals are geometrically coupled (surface orientation ↔ depth gradient), and training jointly propagates this constraint through the generative process.

**Implementation**:
- Base model: CogVideoX (DiT video diffusion)
- **Separate modality projectors**: Three distinct projection heads process RGB, depth, and normal latents before feeding into shared DiT
- **DNProj output module**: Additional Conv3D layer + DNProj head for depth/normal prediction alongside original RGB output path

The depth and normal predictions are produced from the DiT hidden state $h$ and a Conv3D-processed combination of the predicted RGB noise and the noisy/clean video latents:

$$\epsilon^*_{d,n} = \text{DNProj}(h, \text{Conv3D}(\epsilon^*_v, [z_t; z_0]))$$

where $h$ is the DiT hidden state, $\epsilon^*_v$ is the predicted RGB noise, and $z_t, z_0$ are the noisy and clean video latents. This allows depth/normal heads to leverage the RGB predictions for geometrically consistent output.

The full joint denoising loss trains the DiT denoiser $\epsilon_\theta$ to simultaneously predict RGB noise $\epsilon_v$, depth noise $\epsilon_d$, and normal noise $\epsilon_n$ from noisy concatenated latents $\mathbf{x}_t = [\mathbf{v}_t, \mathbf{d}_t, \mathbf{n}_t]$:

$$\mathcal{L} = \mathbb{E}\left[\left\|[\epsilon_v, \epsilon_d, \epsilon_n] - \epsilon_\theta(\mathbf{x}_t, t, \mathbf{x}_0, \mathcal{T})\right\|^2\right]$$

where $\mathbf{x}_0$ is the clean initial frame latent and $\mathcal{T}$ is the text instruction embedding.

- Training frames: 49 (general), 13 (RLBench fine-tune)
- Resolution: 512×512
- CFG scale: 7.5

#### Module 2: 4D Scene Reconstruction

**Design Motivation**: Raw predicted depth maps are per-frame independent; temporal consistency must be explicitly enforced to construct a coherent 4D scene.

**Three-component optimization** at each frame:

The 4D reconstruction algorithm minimizes a composite objective over the refined depth map $\tilde{\mathcal{D}}$ at each frame:

$$\arg\min_{\tilde{\mathcal{D}}} \; \mathcal{L}_s(\tilde{\mathcal{D}}, \mathcal{N}_i) + \mathcal{L}_c(\tilde{\mathcal{D}}, \hat{\mathcal{D}}_{i-1}, \mathcal{F}_i, \mathcal{F}_{i-1}) + \mathcal{L}_r(\tilde{\mathcal{D}}, \mathcal{D}_i)$$

- $\mathcal{L}_s$: **Spatial loss** — enforces that the depth gradient matches the predicted surface normals $\mathcal{N}_i$ via normal integration (perspective geometry constraint)
- $\mathcal{L}_c$: **Consistency loss** — enforces temporal coherence using optical flow $\mathcal{F}$ between consecutive frames, with separate weights $\lambda_{cd}$ for dynamic regions $\mathcal{M}_d$ and $\lambda_{cb}$ for background $\mathcal{M}_b$
- $\mathcal{L}_r$: **Regularization loss** — keeps the refined depth $\tilde{\mathcal{D}}$ aligned with the raw predicted depth $\mathcal{D}_i$ from DNProj

where $\hat{\mathcal{D}}_{i-1}$ is the refined depth from the previous frame and $\mathcal{F}_i, \mathcal{F}_{i-1}$ are optical flows between consecutive frames. This optimization runs for each frame in approximately 1 minute total (vs. ~2 hours for optimization-based baselines).

#### Module 3: 3D-Aware Robot Policy

**Design Motivation**: 3D point clouds from reconstructed scenes provide richer spatial features than 2D pixels for action prediction — PointNet naturally handles unordered point cloud inputs.

**Implementation**:
- Filter 4D point clouds (remove background, downsample)
- PointNet encoder processes filtered point clouds
- 4-layer MLP with instruction text embeddings → 7-DoF joint actions
- Training augmentation: 20% Gaussian noise on image + point cloud coordinates for robustness
- MLP dimensions: 1024

---

## Experiments

### Datasets

| Dataset | Domain | Embodiment | Count | Depth Source | Normal Source |
|---------|--------|-----------|-------|--------------|---------------|
| RLBench | Synthetic | Franka Panda | 80k | Simulator | Depth2Normal |
| RT-1 Fractal | Real | Google Robot | 80k | RollingDepth | Marigold-LCM |
| Bridge | Real | WidowX | 25k | RollingDepth | Marigold-LCM |
| SomethingSomethingV2 | Human Hand | — | 100k | Estimated | Estimated |

### Implementation Details

- **Base Model**: CogVideoX (DiT video diffusion)
- **Total Training Iterations**: 40,000
- **Batch Size**: 16 (global)
- **Learning Rate**: 1e-4 with 1,000 warmup steps
- **Optimizer**: Adam ($\epsilon = 1 \times 10^{-15}$)
- **Precision**: BF16
- **EMA Decay**: 0.99
- **Gradient Clipping**: 1.0
- **Output Frames**: 49 (general), 13 (RLBench fine-tune)
- **Resolution**: 512×512
- **Diffusion Steps**: 50 (DDPM scheduler)
- **CFG Scale**: 7.5
- **Conv3D Layers**: 3 (in DNProj head)
- **Policy MLP**: 4 layers, 1024 dimensions

### 4D Scene Reconstruction Quality

The full pipeline (Figure 1) is: text instruction + initial RGB frame → RGB-DN video diffusion generates multi-frame RGB + depth + normal predictions → 4D reconstruction algorithm converts to temporally consistent point cloud → PointNet policy extracts 7-DoF robot actions. The joint RGB-DN training produces far more geometrically accurate 4D reconstructions than baseline video models:

| Method | Chamfer L1 (Real) | Depth AbsRel (Real) | Normal 11.25° (Real) | Chamfer L1 (Syn) | Depth AbsRel (Syn) | Normal 11.25° (Syn) |
|--------|-----------------|-------------------|---------------------|----------------|-----------------|-------------------|
| OpenSora | 0.3013 | — | — | — | — | — |
| CogVideoX | 0.2191 | 26.17 | 22.70 | 0.2884 | 19.81 | 26.04 |
| **TesserAct (Ours)** | **0.2030** | **22.07** | **27.80** | **0.0811** | **16.02** | **36.85** |

TesserAct dramatically reduces Chamfer distance vs. CogVideoX (0.0811 vs. 0.2884 on synthetic), confirming that joint RGB-DN training produces much more geometrically accurate 4D reconstructions.

### Robot Policy Success Rate (RLBench, 100 episodes)

3D geometric understanding consistently improves manipulation policy quality:

| Task | Image-BC | UniPi* | **TesserAct** |
|------|----------|--------|--------------|
| Close box | 53% | 81% | **88%** |
| Open drawer | 4% | 67% | **80%** |
| Open jar | 0% | 38% | **44%** |
| *Average advantage* | — | ~10-15% below Ours | **Best** |

TesserAct consistently outperforms UniPi* (2D video + inverse dynamics) by ~10-15% across tasks, and dramatically outperforms Image-BC.

### Novel View Synthesis Quality

TesserAct's 4D reconstruction achieves better novel view synthesis quality AND is ~120x faster than optimization-based Shape of Motion:

| Method | PSNR↑ | SSIM↑ | Time |
|--------|-------|-------|------|
| Shape of Motion | 10.94 | 24.02 | ~2 hours |
| **TesserAct (Ours)** | **12.99** | **42.62** | **~1 min** |

### Loss Weighting Parameters

| Dataset | $\lambda_d$ | $\lambda_b$ | $\lambda_{g1}$ | $\lambda_{g2}$ |
|---------|-------------|-------------|----------------|----------------|
| RT-1/Bridge | 20 | 200 | 20 | 20 |
| RLBench | 20 | 200 | 2 | 2 |

---

## Critical Analysis

### Strengths
1. RGB-DN representation is an elegant compromise: richer than pure RGB, cheaper than full 3D reconstruction, compatible with existing video diffusion architectures.
2. 4D reconstruction runs in ~1 minute vs. hours for optimization-based methods — practical for real applications.
3. Cross-domain training (synthetic + real + human hands) improves generalization.

### Limitations
1. **Single-surface limitation**: RGB-DN representation captures only the visible surface; objects' back sides, internal structure, or occluded regions are not modeled.
2. **Off-the-shelf depth/normal quality**: Training data quality depends on RollingDepth and Marigold-LCM estimation quality; errors propagate to learned model.
3. **Real robot evaluation limited**: Primary evaluation on RLBench (synthetic); real-robot deployment with 3D policy not fully demonstrated.
4. **Depth ambiguity from monocular**: Single-camera depth prediction inherits monocular ambiguities (scale uncertainty, texture-less regions).

### Potential Improvements
1. Multi-view generation to construct complete 4D models (explicitly noted by authors as future work).
2. Closed-loop replanning using 3D scene state for online correction.
3. Integration with physical simulation for physically-plausible constraint satisfaction.

### Reproducibility
- [ ] Code open-sourced
- [ ] Pretrained models available
- [x] Training details complete (Section 4 + Appendix)
- [x] Datasets accessible (RLBench, RT-1, Bridge, SSv2 all public; depth/normal from public models)

---

## Related Notes

### Based On
- CogVideoX: Base DiT video diffusion model fine-tuned for RGB-DN prediction
- PointNet: Point cloud encoder for 3D-aware policy
- Optical Flow: Used for temporal consistency in 4D reconstruction
- Video Diffusion Model: Core generation framework

### Compared Against
- [UniPi](UniPi.md): 2D video + inverse dynamics baseline; TesserAct outperforms by ~10-15%
- Shape of Motion: 4D reconstruction baseline; outperformed 120x faster and better quality

### Method Related
- 4D Gaussian Splatting: Related 4D scene representation approach
- Depth Estimation: RollingDepth + Marigold-LCM used for training data annotation
- Normal Estimation: Surface normal prediction as auxiliary geometry signal
- Inverse Dynamics Model: Action extraction approach

### Hardware/Data Related
- RLBench: Primary robot policy evaluation benchmark
- Bridge Dataset: Real robot training data
- RT-1 Dataset: Real robot training data

---

## Quick Reference Card

> [!summary] TesserAct (arXiv 2025)
> - **Core**: 4D embodied world model via joint RGB+Depth+Normal video generation (CogVideoX DiT) + optical-flow-constrained 4D reconstruction + PointNet robot policy
> - **Method**: CogVideoX fine-tuned with DNProj/Conv3D output heads; 285k training videos with estimated depth/normal; 4D reconstruction in ~1 min; 4-layer MLP policy on point clouds
> - **Results**: +10-15% over UniPi* on RLBench; PSNR 12.99 vs 10.94 (Shape of Motion) at 120x speedup; Chamfer distance 0.08 vs 0.29 on synthetic
> - **Code**: https://tesseractworld.github.io/

---

*Note created: 2026-05-20*
