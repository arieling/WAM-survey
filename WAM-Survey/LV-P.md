---
title: "Large Video Planner Enables Generalizable Robot Control"
method_name: "LV-P"
authors: [Boyuan Chen, Tianyuan Zhang, Haoran Geng, Caiyi Zhang, Peihao Li, Kiwhan Song, William T. Freeman, Jitendra Malik, Pieter Abbeel, Russ Tedrake, Vincent Sitzmann, Yilun Du]
year: 2025
venue: arXiv 2025
tags: [video-generation, hand-retargeting, diffusion-forcing, robot-policy, cascaded-explicit-geometric, cross-embodiment, dexterous]
zotero_collection: 3-Robotics/WAM-Survey
image_source: online
arxiv_html: https://arxiv.org/abs/2512.15840
created: 2026-05-20
---

# Paper Note: Large Video Planner Enables Generalizable Robot Control (LV-P)

## Metadata

| Item | Content |
|------|---------|
| Institution | MIT, UC Berkeley, Harvard |
| Date | December 2025 |
| Project Page | N/A |
| Baselines | π₀, OpenVLA, Wan2.1 I2V 14B, Cosmos-Predict2 14B, Hunyuan I2V 13B |
| Links | [arXiv](https://arxiv.org/abs/2512.15840) / Code: N/A |

---

## One-Sentence Summary

> LV-P is a 14B-parameter video foundation model (built on WAN 2.1 I2V, trained on LVP-1M with diffusion forcing) that generates human manipulation video plans from text + initial observation, then extracts dexterous robot actions via a three-stage pipeline — HaMeR hand pose estimation + MegaSAM 4D alignment + Dex-Retargeting — achieving zero-shot generalization across Franka parallel-gripper and G1 dexterous-hand platforms.

---

## Core Contributions

1. **Diffusion Forcing for Flexible Video Planning**: Training with independently randomized noise levels across temporal tokens enables flexible conditioning on 0–6 latent history frames and V2V extension for multi-stage planning — without architectural modification.
2. **LVP-1M Dataset**: 1.4M action-centric video clips curated from 8 sources (robotics + human activity), temporally aligned to human action speed (3 seconds at 16 FPS), with action-centric re-captioning via Gemini (4.1M captions total).
3. **History Guidance**: CFG variant combining conditional (with history frames) and unconditional scores — enforces temporal coherence between generated video frames and prior context.
4. **3-Stage Human-to-Robot Action Extraction**: HaMeR hand mesh estimation → MegaSAM 4D depth alignment (scale disambiguation) → Dex-Retargeting (or parallel gripper retargeting) → cuRobo IK → robot execution — enabling dexterous hand control from generated human manipulation videos.

---

## Problem Background

### Problem to Solve
VLA foundation models require large-scale robot demonstrations across diverse tasks and embodiments — expensive to collect. Human manipulation videos are abundant but require human-to-robot morphology transfer. Can large-scale video pretraining on human + robot video produce a video planner that generates physically grounded manipulation plans transferable to diverse robot embodiments with zero task-specific training?

### Limitations of Existing Methods
- π₀, OpenVLA: robot-data-dependent; struggle with novel objects and zero-shot generalization.
- Standard I2V models (Wan2.1, Cosmos-Predict2, Hunyuan): not specialized for manipulation — lack precise contact and object motion (Level 3–4 quality significantly lower than LV-P).
- UniPi/AVDC: limited model scale; action extraction via 2D flow (no 3D geometric recovery for dexterous control).

### Motivation
A large-scale video model trained on both human and robot manipulation videos learns physics, affordances, and task structure from internet-scale diverse data. By generating human manipulation videos as plans, the system can leverage abundant human video diversity, and structured action extraction (HaMeR + 4D depth + retargeting) transfers generated plans to real robot execution.

---

## Method Details

### Model Architecture

LV-P has **two major components**:

1. **Large Video Planner (14B)**: WAN 2.1 I2V base + diffusion forcing training → generates human manipulation video plans
2. **Action Extraction Pipeline**: HaMeR → MegaSAM → Dex-Retargeting → cuRobo IK → robot commands

### Core Modules

#### Module 1: Video Foundation Model (Diffusion Forcing)

**Design Motivation**: Standard I2V models can only condition on a fixed initial frame. Diffusion Forcing — training with independently randomized noise levels per temporal token — enables conditioning on arbitrary history length (0–6 frames), supporting V2V chaining for multi-stage planning.

**Implementation**:
- Base model: WAN 2.1 I2V, 14B parameters
- Temporal encoding: causally-temporal 3D VAE encoder (8×8×4 spatiotemporal patches → 16-channel embeddings)
- Training: flow matching with shifted noise schedule; independently randomized noise levels per temporal token (diffusion forcing)
- History conditioning: 0–6 latent frames (≈24 pixel frames) at inference
- Training stages:
  - Stage 1 (Pretraining): LVP-1M, 60K steps, batch 128, 200B tokens, 128 H100 SXM5 GPUs, ~14 days
  - Stage 2 (Low-camera-motion fine-tuning): 10K additional steps on curated subsets with low optical flow magnitude

Each temporal token $x_k$ at position $k$ receives independently sampled noise level $\sigma_k$ during training, enabling flexible conditioning on arbitrary subsets of history tokens at inference without any architectural modification.

#### Module 2: History Guidance

**Design Motivation**: Pure text-conditioned video generation ignores temporal context — generated frames may be incoherent with prior execution. History guidance combines guidance from both text and history frames for physically consistent plans.

**Implementation**:

History guidance interpolates between a history-conditioned score and a text-only score, weighting temporal coherence against pure language guidance. Concretely, the combined score is defined as:

$$\nabla \log \tilde p(x_k) = w_h \cdot \nabla \log p(x_k \mid x_{\text{hist}}, c_{\text{text}}, k) + (1-w_h) \cdot \nabla \log p(x_k \mid c_{\text{text}}, k)$$

- $w_h$: history guidance weight controlling the trade-off between temporal coherence (left term) and pure text conditioning (right term)
- $x_{\text{hist}}$: latent history frames from prior execution steps
- $c_{\text{text}}$: text instruction embedding

This CFG variant enables seamless V2V conditioning for iterative multi-stage planning.

#### Module 3: LVP-1M Dataset

**Design Motivation**: Robot manipulation datasets are diverse but small; human activity datasets are large but lack robot relevance. Combining both with action-centric filtering and captioning provides the scale and diversity needed for a generalizable video planner.

**Composition**:
| Source | Clips |
|--------|-------|
| AgiBot-World | 863K |
| DROID | 192K |
| Panda-70M (filtered) | 196K |
| Something-Something | 93K |
| Language-Tables | 71K |
| Ego4D | 39K |
| Bridge | 25K |
| Epic-Kitchens | 7K |
| **Total** | **1.4M** |

The dataset is processed by temporally aligning all clips to 3-second segments at 16 FPS — a speed chosen to match robot action execution — then filtering with optical flow to remove rapid camera motion, verifying object presence with an object detector, and re-captioning with Gemini (4.1M captions, 2–5 per clip).

#### Module 4: Human Hand Action Extraction

**Design Motivation**: Generated videos show human hands — recovering 3D hand trajectories requires: (1) per-frame pose estimation, (2) metric-scale 3D reconstruction (monocular depth is scale-ambiguous), (3) temporal smoothing, and (4) robot morphology transfer.

**Three-Stage Pipeline**:

**Stage 1 — Hand Pose Estimation**:
- HaMeR: estimates MANO hand vertices and wrist orientation (camera coordinates) per frame
- MegaSAM: 4D depth reconstruction — backprojects wrist joints into metric 3D space, enforcing temporal consistency
- Linear interpolation for missing frames; Savitzky-Golay causal filtering for smoothing

**Stage 2 — Robot Retargeting**:
- Dexterous hands (G1 + Inspire): Dex-Retargeting optimization (DexPilot-style objective) maps human keypoints to robot joint angles
- Parallel grippers (Franka): separate retargeting module for reduced-DOF grippers

**Stage 3 — Execution**:
- Camera-to-robot frame alignment uses a fixed rotation matrix $M \in SO(3)$ estimated offline
- Arm trajectory: cuRobo inverse kinematics solver
- Finger control synchronized with arm motion

---

## Experiments

### Datasets

| Dataset | Scale | Characteristics | Usage |
|---------|-------|----------------|-------|
| LVP-1M | 1.4M clips | 8 sources (robot + human), 3-sec clips at 16 FPS, 4.1M Gemini captions | Model training |
| In-the-wild eval | 100 tasks | Third-party videos, diverse tasks | Video quality evaluation |
| Real-world tasks | 10 trials/task | Franka + G1 platforms | Robot execution evaluation |

### Implementation Details

- **Base Model**: WAN 2.1 I2V, 14B parameters
- **VAE**: Temporally causal 3D, 8×8×4 patches → 16-channel embeddings
- **Training Objective**: Flow matching with shifted noise schedule
- **Training Stage 1**: 60K steps, batch 128, 200B tokens, 128 H100 SXM5 GPUs, 14 days total
- **Training Stage 2**: 10K steps, low-camera-motion subset
- **History Frames**: 0–6 latent frames (≈24 pixel frames)
- **Hand Pose Model**: HaMeR (MANO mesh)
- **Depth Model**: MegaSAM (monocular 4D reconstruction)
- **Smoothing**: Savitzky-Golay causal filter
- **Dexterous Retargeting**: Dex-Retargeting (DexPilot-style)
- **IK Solver**: cuRobo
- **Robot Platforms**: Franka Panda (parallel gripper), Unitree G1 + Inspire dexterous hand

### Video Motion Planning Quality (100 In-the-Wild Tasks)

LV-P is evaluated against standard I2V baselines on 100 in-the-wild manipulation tasks at four quality levels — with Level 3 (continuous object motion) and Level 4 (physically plausible full trajectory) being the hardest. Results show the following per-method average/best@4 success rates:

| Method | Level 1 (Avg/Best@4) | Level 2 (Avg/Best@4) | Level 3 (Avg/Best@4) | Level 4 (Avg/Best@4) |
|--------|---------------------|---------------------|---------------------|---------------------|
| **LV-P (Ours)** | **87.3% / 100%** | **63.2% / 85%** | **59.3% / 82%** | **44.0% / 71%** |
| Wan 2.1 I2V 14B | 83.9% / 99% | 47.0% / 80% | 39.3% / 76% | 20.5% / 53% |
| Cosmos-Predict2 14B | 45.3% / 81% | 11.9% / 35% | 7.5% / 24% | 2.5% / 9% |
| Hunyuan I2V 13B | 68.7% / 96% | 27.3% / 65% | 13.5% / 42% | 7.2% / 27% |

LV-P substantially outperforms all video baselines at all quality levels — especially Level 3–4 (physically plausible continuous motion): +20% over Wan2.1, +52% over Cosmos, +46% over Hunyuan at Level 3 Avg.

### Real-Robot Results — Franka + Parallel Gripper

LV-P is compared against π₀ and OpenVLA on four tabletop tasks (10 trials each) using the Franka Panda robot:

| Task | LV-P | π₀ | OpenVLA |
|------|------|-----|---------|
| Pick Objects | 5/10 | 3/10 | 0/10 |
| Pick A into B | 3/10 | 1/10 | 0/10 |
| Open Drawer | 2/10 | 1/10 | 0/10 |
| Press Button | 4/10 | 0/10 | 0/10 |

LV-P outperforms π₀ and OpenVLA on all tasks despite being zero-shot (no task-specific fine-tuning).

### Real-Robot Results — G1 + Inspire Dexterous Hand

The following results represent the first demonstration of zero-shot dexterous manipulation from generated human video plans on a humanoid dexterous hand platform:

| Task | LV-P |
|------|------|
| Wipe Table | 8/10 |
| Sweep Tennis Ball | 5/5 |
| Open Door | 6/10 |
| Scoop Coffee Beans | 3/5 |
| Pick Objects | 4/10 |
| Press Elevator Button | 4/5 |
| Tear Tape | 2/5 |

---

## Critical Analysis

### Strengths
1. Zero-shot dexterous manipulation from human video plans — first system to achieve this at scale.
2. Diffusion forcing enables flexible history conditioning and multi-stage planning without architectural changes.
3. LVP-1M's scale (1.4M clips) and diversity (robot + human) drives strong video quality improvements over baselines.

### Limitations
1. **Video generation latency**: Several minutes per sample — not real-time deployable.
2. **Hand pose extraction errors**: HaMeR + MegaSAM pipeline introduces per-step errors that compound over trajectories.
3. **Parallel gripper retargeting**: DOF mismatch between human fingers and 1-DOF parallel gripper causes imprecision.
4. **Open-loop execution**: No closed-loop replanning from current observations during execution.

### Potential Improvements
1. Distillation for faster video generation (fewer denoising steps).
2. Closed-loop video replanning at key waypoints during execution.
3. Tactile sensing integration to compensate for hand pose estimation errors.

### Reproducibility
- [ ] Code open-sourced
- [ ] Pretrained models available
- [x] Training details described (steps, batch size, GPU count, duration)
- [ ] LVP-1M dataset not publicly released

---

## Related Notes

### Based On
- WAN 2.1: Base 14B I2V video model
- Diffusion Forcing: Independent per-token noise schedule for flexible conditioning
- HaMeR: Human hand mesh recovery for action extraction
- Dex-Retargeting: Human-to-dexterous-robot hand motion transfer

### Compared Against
- π₀: VLA baseline; outperformed on all real-robot tasks
- Wan 2.1 I2V: Video generation baseline; outperformed by +20% on Level 3 video quality
- Cosmos-Predict2: Video generation baseline; outperformed by +52% on Level 3

### Method Related
- History Guidance: CFG variant for temporal coherence in video generation
- cuRobo: GPU-accelerated IK solver for robot arm control
- Cross-Embodiment Transfer: Zero-shot transfer from human video to robot execution

### Hardware/Data Related
- LVP-1M: Action-centric curated video dataset for manipulation video pretraining

---

## Quick Reference Card

> [!summary] LV-P (arXiv 2025)
> - **Core**: WAN 2.1 I2V 14B + diffusion forcing (independent per-token noise) + history guidance → human manipulation video; HaMeR + MegaSAM 4D depth + Dex-Retargeting + cuRobo IK → dexterous robot actions; zero-shot
> - **Method**: LVP-1M (1.4M clips, 8 sources, Gemini re-captioned); Stage 1: 60K steps batch 128 128×H100 14 days; Stage 2: 10K steps low-camera-motion
> - **Results**: Video L3: 59.3%/82% vs 39.3%/76% (Wan2.1); Franka pick 5/10 vs 3/10 (π₀); G1 dexterous wipe 8/10 (zero-shot)
> - **Code**: N/A

---

*Note created: 2026-05-20*
