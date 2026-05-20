---
title: "Geometry-aware 4D Video Generation for Robot Manipulation"
method_name: "4DGen"
authors: [Zeyi Liu, Shuang Li, Eric Cousineau, Siyuan Feng, Benjamin Burchfiel, Shuran Song]
year: 2025
venue: arXiv 2025
tags: [4d-world-model, video-generation, multi-view, pointmap, robot-policy, cascaded-explicit-geometric, stable-video-diffusion]
zotero_collection: 3-Robotics/WAM-Survey
image_source: online
arxiv_html: https://arxiv.org/abs/2507.01099
created: 2026-05-20
---

# Paper Note: Geometry-aware 4D Video Generation for Robot Manipulation (4DGen)

## Metadata

| Item | Content |
|------|---------|
| Institution | Stanford University, Toyota Research Institute |
| Date | July 2025 |
| Project Page | N/A |
| Baselines | Dreamitate, Diffusion Policy, DP3 |
| Links | [arXiv](https://arxiv.org/abs/2507.01099) / Code: N/A |

---

## One-Sentence Summary

> 4DGen extends Stable Video Diffusion with cross-view pointmap alignment supervision to generate geometry-consistent multi-view RGB-D video predictions from single RGB-D input (no camera poses required), then extracts robot end-effector trajectories via FoundationPose 6-DoF tracking — achieving 64% average manipulation success vs. 12–25% for prior video-based and BC baselines.

---

## Core Contributions

1. **Cross-View Pointmap Alignment Supervision**: Enforces 3D geometric consistency across generated video views by supervising the model with cross-view pointmap alignment during training — ensuring multi-view predictions are spatially coherent without requiring camera poses as input.
2. **Joint RGB + Geometry Diffusion**: Single U-Net architecture generates both RGB videos and depth pointmaps jointly via combined diffusion + pointmap loss ($\lambda=1$) — temporal and geometric consistency optimized simultaneously.
3. **Novel-View 4D Generation from Single RGB-D**: Generates future multi-view RGB-D sequences from a single RGB-D observation without requiring camera pose specification — enabling flexible deployment with arbitrary camera configurations.
4. **FoundationPose-Based Action Extraction**: Off-the-shelf 6-DoF pose tracker (FoundationPose + SAM2) extracts robot end-effector trajectory directly from predicted multi-view RGB-D videos — no learned IDM needed.

---

## Problem Background

### Problem to Solve
Predicting future robot states in 3D-consistent multi-view RGB-D for manipulation planning. Single-view video generation lacks geometric grounding; existing multi-view methods require camera poses. How can a video diffusion model generate geometrically consistent multi-view future sequences from a single RGB-D observation without camera pose specification?

### Limitations of Existing Methods
- TesserAct/MVISTA-4D: require camera pose specification or calibration; complex cross-view attention.
- Dreamitate: single-view only; requires 3D-printed tools with known CAD for tracking; 12% success.
- Diffusion Policy: no video world model; requires task-specific teleoperation; 12% success.
- DP3: 3D point cloud BC; no video prediction; 25% success.

### Motivation
Generating multi-view RGB-D videos with 3D consistency provides rich geometric information for reliable pose tracking (FoundationPose) without requiring a learned inverse dynamics model. Cross-view pointmap supervision trains the model to implicitly understand 3D scene structure — enforcing consistency without explicit camera pose conditioning.

---

## Method Details

### Model Architecture

4DGen is a **latent video diffusion model**:

1. **Base**: Stable Video Diffusion U-Net (encoder-decoder) extended with cross-view cross-attention
2. **Joint Output**: RGB video frames + depth pointmaps for two camera views simultaneously
3. **Action Extraction**: FoundationPose 6-DoF tracking on generated multi-view RGB-D

**Total Trainable Parameters**: 2.4B

### Core Modules

#### Module 1: Multi-View Video Diffusion U-Net

**Design Motivation**: Extending SVD to multi-view enables joint training of temporal coherence (from SVD pretraining) and cross-view 3D consistency (from new cross-attention layers).

**Implementation**:
- Input to U-Net: 16-channel concatenation of two camera views ($4c \times 2$ views) in latent space
- Image VAE: 10 temporal frames, 4 latent channels per frame, spatial dims 32×40
- Cross-attention layers: enable "information transfer" between decoders for reference view and projected view
- Output: RGB video + depth pointmaps for both views

#### Module 2: Cross-View Pointmap Alignment Supervision

**Design Motivation**: Without explicit geometric supervision, multi-view video models generate view-inconsistent depth predictions. Supervising with cross-view pointmap alignment enforces 3D consistency as an auxiliary objective.

**Implementation**:
- Predicts pointmap pairs: one in native view coordinate system, one projected into reference camera frame
- Cross-attention enables information flow between native and projected view decoders
- Training ablation: removing cross-attention causes mIoU drop from 0.70 → 0.41

The geometric consistency of the generated views is measured by the intersection-over-union between native and reference-projected pointmaps:

$$\text{mIoU} = \frac{|\mathcal{P}_{\text{native}} \cap \mathcal{P}_{\text{projected}}|}{|\mathcal{P}_{\text{native}} \cup \mathcal{P}_{\text{projected}}|}$$

This metric quantifies how well both generated views describe the same underlying 3D scene.

#### Module 3: Joint Training Objective

To enforce both visual quality and 3D geometric consistency in a single pass, the model is trained with a combined RGB + pointmap loss. The two objectives are weighted equally:

$$\mathcal{L} = \mathcal{L}_{RGB} + \lambda \cdot \mathcal{L}_{PM}, \quad \lambda = 1$$

$\mathcal{L}_{RGB}$ is the diffusion reconstruction loss applied to both camera views; $\mathcal{L}_{PM}$ penalizes misalignment between native and projected pointmaps. Equal weighting ($\lambda=1$) enforces geometric consistency without sacrificing visual quality.

**Implementation**:
- Optimizer: AdamW, lr=1e-6
- Inference: EulerEDMSampler, 25 denoising steps, ~30s on RTX 4090

#### Module 4: Robot Action Extraction (FoundationPose)

**Design Motivation**: Off-the-shelf 6-DoF pose tracking on generated multi-view RGB-D is more reliable than a learned IDM — no additional training needed, and metric depth from multiple views constrains the pose estimate.

**Implementation**:
1. SAM2: segments gripper/end-effector in initial frame
2. FoundationPose: tracks 6-DoF end-effector pose across generated multi-view frames
3. Confidence scoring: filters low-confidence pose estimates
4. Gripper state: inferred from distance between finger centroids in predicted video
5. Global frame transformation: transforms predicted poses to robot base frame for execution

---

## Experiments

### Datasets

| Dataset | Scale | Characteristics | Usage |
|---------|-------|----------------|-------|
| Simulation | 3 tasks × 25 demos × 16 views = 1,200 videos | Multi-view simulated robot manipulation | Training/Evaluation |
| Real-world | 4 tasks × 20 demos, dual FRAMOS D415e cameras | Tabletop Franka manipulation | Training/Evaluation |

### 4D Generation Quality

The table below reports geometric consistency (mIoU) and video quality (FVD-nn) with and without cross-view attention, demonstrating the critical role of the alignment supervision:

| Task | mIoU (Ours) | mIoU (w/o cross-attn) | FVD-nn (Ours) | FVD-nn (w/o cross-attn) |
|------|------------|----------------------|---------------|------------------------|
| StoreCerealBoxUnderShelf | **0.70** | 0.41 | **411.20** | 497.43 |
| PutSpatulaOnTable | **0.69** | 0.44 | — | — |
| Multi-task Real World | **0.56** | 0.32 | — | — |

Cross-view attention is critical — removing it drops mIoU by 0.29 (41% relative reduction) while increasing FVD by 86 points.

### Robot Manipulation Success Rate

4DGen is evaluated against Dreamitate, Diffusion Policy, and DP3 on three real-world tabletop tasks:

| Task | 4DGen | Dreamitate | Diffusion Policy | DP3 |
|------|-------|------------|-----------------|-----|
| StoreCerealBoxUnderShelf | **73%** | 12% | 12% | 25% |
| PutSpatulaOnTable | **67%** | — | — | — |
| PlaceAppleFromBowlIntoBin | **53%** | — | — | — |
| **Average** | **64%** | 12% | 12% | 25% |

4DGen substantially outperforms all baselines — Dreamitate and Diffusion Policy both at 12%, DP3 at 25%. +39–52% over baselines on comparable tasks.

### Implementation Details

- **Base Model**: Stable Video Diffusion (SVD) U-Net
- **Trainable Parameters**: 2.4B
- **Image VAE**: 10 temporal frames, 4 channels, 32×40 spatial latent
- **Input to U-Net**: 16-channel (4c × 2 views)
- **Optimizer**: AdamW, lr=1×10⁻⁶
- **Batch Size**: 4
- **Training**: ~60 epochs per task
- **Hardware**: 4× NVIDIA RTX A6000 (48GB each)
- **Denoising Steps (Inference)**: 25 (EulerEDMSampler)
- **Inference Time**: ~30 seconds per 10-frame prediction on RTX 4090
- **Pose Tracker**: FoundationPose (6-DoF)
- **Segmentation**: SAM2
- **Cameras**: Dual FRAMOS D415e (real), 16 synthetic views (simulation)

---

## Critical Analysis

### Strengths
1. Cross-view pointmap alignment elegantly enforces 3D consistency without camera pose input — practical for real deployment.
2. FoundationPose extraction avoids learning a separate IDM — leverages pretrained 6-DoF tracker.
3. 64% average success vs. 12–25% baselines — substantial empirical improvement.

### Limitations
1. **30-second inference**: Slow for closed-loop control — limits to open-loop plan-then-execute.
2. **Small training datasets**: 1,200 simulation videos and 20 real demos per task — limited diversity.
3. **Depth quality dependency**: Depth estimation accuracy is critical for pointmap alignment and FoundationPose tracking.
4. **Two-view limitation**: Only two camera views — limited 3D coverage compared to methods with 16 views.

### Potential Improvements
1. Distillation for faster inference (fewer denoising steps).
2. Larger training datasets for better generalization.
3. Closed-loop replanning with online video regeneration from current observations.

### Reproducibility
- [ ] Code open-sourced
- [ ] Pretrained models available
- [x] Training details described (hardware, lr, batch size, epochs)
- [ ] Datasets not publicly available (custom robot demonstrations)

---

## Related Notes

### Based On
- [[Stable Video Diffusion]]: Base video diffusion model extended with multi-view capability
- [[FoundationPose]]: 6-DoF pose tracking for action extraction
- [[SAM2]]: Object segmentation for FoundationPose initialization

### Compared Against
- [[Dreamitate]]: Single-view video diffusion with tool tracking; outperformed 12%→64%
- [[Diffusion Policy]]: Teleoperation-based BC; outperformed 12%→64%
- [[DP3]]: 3D point cloud policy; outperformed 25%→64%

### Method Related
- [[Pointmap]]: 3D-aware scene representation for geometric consistency supervision
- [[Cross-View Attention]]: Information transfer between multiple camera views
- [[4D World Model]]: Spatiotemporally consistent scene prediction

---

## Quick Reference Card

> [!summary] 4DGen (arXiv 2025)
> - **Core**: SVD U-Net extended with cross-view cross-attention (2.4B params); joint RGB + pointmap generation for 2 views; cross-view alignment supervision ($\mathcal{L} = \mathcal{L}_{RGB} + \lambda \mathcal{L}_{PM}$, $\lambda=1$); FoundationPose 6-DoF tracking for action extraction
> - **Method**: 16-channel input (4c×2 views), 32×40 latent, 25-step EulerEDM; AdamW lr=1e-6 batch 4, ~60 epochs, 4×A6000; ~30s inference on RTX 4090
> - **Results**: 64% avg manipulation success (vs. 12% Dreamitate, 12% Diffusion Policy, 25% DP3); mIoU 0.70 vs 0.41 without cross-attention
> - **Code**: N/A

---

*Note created: 2026-05-20*
