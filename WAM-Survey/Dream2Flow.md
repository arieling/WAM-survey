---
title: "Dream2Flow: Bridging Video Generation and Open-World Manipulation with 3D Object Flow"
method_name: "Dream2Flow"
authors: [Karthik Dharmarajan, Wenlong Huang, Jiajun Wu, Li Fei-Fei, Ruohan Zhang]
year: 2025
venue: arXiv 2025
tags: [video-generation, 3d-flow, open-world, robot-policy, cascaded-explicit-geometric, multi-embodiment, point-tracking]
zotero_collection: 3-Robotics/WAM-Survey
image_source: online
arxiv_html: https://arxiv.org/abs/2512.24766
created: 2026-05-20
---

# Paper Note: Dream2Flow: Bridging Video Generation and Open-World Manipulation with 3D Object Flow

## Metadata

| Item | Content |
|------|---------|
| Institution | Stanford University (inferred: Fei-Fei Li, Jiajun Wu, Ruohan Zhang) |
| Date | December 2025 |
| Project Page | [dream2flow.github.io](https://dream2flow.github.io) |
| Baselines | Not explicitly specified |
| Links | [arXiv](https://arxiv.org/abs/2512.24766) / Code: N/A |

---

## One-Sentence Summary

> Dream2Flow generates manipulation videos from initial RGB-D observation + task instruction (via Kling 2.6, Veo 3.1, or Sora 2), reconstructs 3D object flow using vision foundation models (object masks + depth + point tracking), and converts flow to robot commands via trajectory optimization or RL — achieving 66.7% success (40/60 trials) across rigid, articulated, deformable, and granular object categories on Franka, Spot, and Fourier GR1 platforms.

---

## Core Contributions

1. **3D Object Flow as Universal Manipulation Interface**: Uses 3D object flow reconstructed from generated videos as an intermediate representation that decouples task-level state changes from embodiment-specific actuator commands — enabling transfer across Franka, Boston Dynamics Spot, and Fourier GR1 humanoid.
2. **Multi-Video-Model Compatibility**: System tested with three frontier video models (Kling 2.6, Veo 3.1, Sora 2) — demonstrates that the 3D flow extraction pipeline is model-agnostic.
3. **Broad Object Category Coverage**: Handles rigid, articulated, deformable, and granular materials within a single unified pipeline — broader than most prior flow-based methods.
4. **Zero-Shot Multi-Embodiment Transfer**: Successfully deployed on three robot platforms without platform-specific training — 3D flow is embodiment-agnostic.

---

## Problem Background

### Problem to Solve
Video generation models (Kling, Veo, Sora) can synthesize physically plausible manipulation sequences from text descriptions, but generating robot control commands from these videos requires bridging the "embodiment gap" — the robot's morphology differs from any appearance in the generated video. How can generated videos be converted to executable robot actions without task-specific training?

### Limitations of Existing Methods
- 2D flow methods (Im2Flow2Act): restricted to image-plane motion; miss depth-perpendicular trajectories.
- Prior video-to-action methods: require robot data for inverse dynamics or retargeting; not zero-shot.
- Single-platform methods: embodiment-specific; cannot transfer across robot morphologies.

### Motivation
3D object flow captures the physical motion of objects through space — the same flow pattern is achievable by any robot with suitable kinematics. Frontier video models implicitly understand task semantics and object physics; extracting 3D flow from their outputs provides a task-level geometric plan executable by diverse robot platforms.

---

## Method Details

### Model Architecture

Dream2Flow is a **three-stage pipeline**:

1. **Video Generation**: Image-to-video model (Kling 2.6 / Veo 3.1 / Sora 2) conditioned on initial RGB-D frame + task instruction
2. **3D Flow Reconstruction**: Vision foundation models extract object masks, depth, and point tracking from generated video → 3D object flow
3. **Robot Policy**: Trajectory optimization or RL converts 3D flow to executable robot commands

### Core Modules

#### Module 1: Video Generation

**Design Motivation**: Frontier video models encode rich physics priors and task semantics from internet-scale training — they generate plausible manipulation sequences given only a task description and initial scene image.

**Implementation**:
- Tested models: Kling 2.6, Veo 3.1, Sora 2
- Input: initial RGB-D observation + task instruction
- Output: manipulation video sequence conditioned on instruction semantics
- Model-agnostic design: same downstream pipeline works with any video generation model

#### Module 2: 3D Object Flow Reconstruction

**Design Motivation**: Converting generated video to 3D flow requires: (1) identifying which objects are relevant (object masks), (2) recovering metric depth (depth estimation), (3) tracking object keypoints across frames (point tracking).

The core 3D flow extraction operation lifts 2D tracked points into metric 3D space using calibrated depth:

$$\mathcal{F} = \text{Lift}(\text{Track}(V_{\text{gen}}), D_{\text{calib}})$$

where $V_{\text{gen}}$ is the generated video, $D_{\text{calib}}$ is the metric depth anchored to the initial RGB-D observation, $\text{Track}$ performs 2D point tracking across frames, and $\text{Lift}$ projects 2D tracked coordinates into 3D using the calibrated depth.

**Implementation**:
- Object masks: vision foundation models (e.g., Grounding-SAM2) segment target objects
- Video depth: monocular depth estimation anchored to initial RGB-D ground-truth for metric scale
- Point tracking: tracks object keypoints across generated video frames
- 3D object flow: reconstructed by lifting 2D tracked points into metric 3D using calibrated depth
- Output: 3D flow trajectory representing how target objects move through space

#### Module 3: Robot Policy (Trajectory Tracking)

**Design Motivation**: 3D flow provides the target object trajectory — the robot policy must find end-effector motion that realizes this trajectory given the robot's kinematics.

**Implementation**:
- Trajectory optimization: given 3D flow, finds robot joint/end-effector trajectory minimizing distance to predicted object trajectory
- Reinforcement learning variant: RL policy trained to track object trajectories specified by 3D flow
- Decouples task-level planning (3D flow) from low-level control (robot-specific IK/RL)

---

## Experiments

### Datasets

| Dataset | Scale | Characteristics | Usage |
|---------|-------|----------------|-------|
| Real-world evaluation | 60 trials | Diverse tasks (rigid, articulated, deformable, granular) | Evaluation |
| Generalization test | 5 trials × 6 variants | Instance/background/viewpoint variants | Robustness evaluation |

### Real-World Success Rate

The pipeline is evaluated end-to-end across 60 trials covering all four object categories. The table tracks each stage of the pipeline to identify failure modes:

| Stage | Count |
|-------|-------|
| Successful video generation | 48/60 |
| Successful flow extraction | 44/60 |
| Successful robot execution | **40/60 (66.7%)** |

66.7% end-to-end success; failure analysis: video hallucination (6 cases), object morphing during generation (6 cases), flow extraction failure (4 cases), execution failure (4 cases).

### Generalization Test

Across 5 trials per variant for 6 task variants, the system demonstrates robust generalization to novel object instances, backgrounds, and viewing angles — confirming that 3D flow is view-invariant after metric depth calibration.

### Object Category Coverage

The unified pipeline handles all four object categories, a broader scope than most prior 3D flow methods which struggle with deformable or granular materials:

| Category | Handled |
|----------|---------|
| Rigid | Yes |
| Articulated | Yes |
| Deformable | Yes |
| Granular | Yes |

### Implementation Details

- **Video Generators Tested**: Kling 2.6, Veo 3.1 (Google), Sora 2 (OpenAI)
- **Depth Source**: Initial RGB-D from robot sensor; monocular depth for generated video frames (anchored to initial depth)
- **Object Segmentation**: Vision foundation models (exact tool not specified)
- **Point Tracking**: Vision foundation models (exact tool not specified)
- **Downstream Policy**: Trajectory optimization or RL (task-dependent)
- **Robot Platforms**: Franka Panda, Boston Dynamics Spot, Fourier GR1

---

## Critical Analysis

### Strengths
1. Broadest object category coverage (rigid + articulated + deformable + granular) of any 3D flow method.
2. Multi-embodiment zero-shot transfer across three qualitatively different platforms (arm, quadruped, humanoid).
3. Video model agnosticism — same pipeline works with Kling, Veo, or Sora.

### Limitations
1. **Video hallucination**: Frontier models generate implausible motions in ~10% of trials — quality bottleneck.
2. **Proprietary model dependency**: Kling, Veo, Sora are all closed-source APIs — reproducibility limited.
3. **Limited technical disclosure**: Paper provides limited implementation details (depth model, tracking model, retargeting method not specified).
4. **No explicit baselines**: Comparative evaluation against AVDC, Im2Flow2Act, or other flow methods not reported.

### Potential Improvements
1. Open-source video models to improve reproducibility.
2. Better flow extraction for deformable/granular objects where rigid body assumptions break down.
3. Closed-loop replanning from current observations.

### Reproducibility
- [ ] Code open-sourced
- [ ] Pretrained models available (video models are proprietary)
- [ ] Training/implementation details not fully disclosed
- [x] Project page with demo videos available

---

## Related Notes

### Based On
- [[Kling]]: Frontier image-to-video model (option 1)
- [[Veo]]: Google frontier video model (option 2)
- [[Sora]]: OpenAI frontier video model (option 3)

### Method Related
- [[3D Object Flow]]: Task-level embodiment-agnostic action representation
- [[Point Tracking]]: Foundation model for keypoint tracking across video frames
- [[Zero-Shot Manipulation]]: Execution without task-specific training
- [[Multi-Embodiment Transfer]]: Cross-platform robot control from single plan

---

## Quick Reference Card

> [!summary] Dream2Flow (arXiv 2025)
> - **Core**: Frontier video model (Kling 2.6 / Veo 3.1 / Sora 2) → 3D object flow (mask + calibrated depth + point tracking) → trajectory optimization / RL → robot execution; zero-shot multi-embodiment
> - **Method**: RGB-D anchored depth calibration; vision foundation models for object grounding + tracking; model-agnostic pipeline; Franka + Spot + GR1
> - **Results**: 66.7% end-to-end success (40/60 trials); robust across object instances, backgrounds, viewpoints; all object categories (rigid/articulated/deformable/granular)
> - **Code**: N/A

---

*Note created: 2026-05-20*
