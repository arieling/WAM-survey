---
title: "3DFlowAction: Learning Cross-Embodiment Manipulation from 3D Flow World Model"
method_name: "3DFlowAction"
authors: [Hongyan Zhi, Peihao Chen, Siyuan Zhou, Yubo Dong, Quanxi Wu, Lei Han, Mingkui Tan]
year: 2025
venue: arXiv 2025
tags: [3d-flow, optical-flow, cross-embodiment, world-model, robot-policy, cascaded-explicit-geometric, animatediff]
zotero_collection: 3-Robotics/WAM-Survey
image_source: online
arxiv_html: https://arxiv.org/abs/2506.06199
created: 2026-05-20
---

# Paper Note: 3DFlowAction: Learning Cross-Embodiment Manipulation from 3D Flow World Model

## Metadata

| Item | Content |
|------|---------|
| Institution | South China University of Technology, Tencent Robotics X, HKUST, Pazhou Laboratory |
| Date | June 2025 |
| Project Page | [GitHub](https://github.com/Hoyyyaard/3DFlowAction/) |
| Baselines | AVDC, Rekep, Im2Flow2Act |
| Links | [arXiv](https://arxiv.org/abs/2506.06199) / Code: Promised |

---

## One-Sentence Summary

> 3DFlowAction uses a 3D optical flow world model (AnimateDiff + LoRA on SD, trained on ManiFlow-110k) to predict how objects move through 3D space as a cross-embodiment action representation, then extracts robot actions via SVD-based transformation estimation + AnyGrasp pose selection + closed-loop GPT-4o validation — achieving 70% success on real-world tasks vs. 20–25% for prior methods.

---

## Core Contributions

1. **3D Flow as Embodiment-Agnostic Action Representation**: Represents manipulation tasks as 3D point trajectory fields $\mathcal{F} \in \mathbb{R}^{T \times H \times W \times 4}$ (2D position + depth + visibility) — capturing object motion in 3D space without embodiment-specific information, enabling cross-robot transfer.
2. **ManiFlow-110k Dataset**: Large-scale 3D flow dataset synthesized from 110,000 instances across diverse sources (BridgeV2, RH20T, LIBERO, DROID, AGIBot, Robot Affordances) — enabling data-driven 3D flow world model training.
3. **Closed-Loop Planning with GPT-4o Validation**: SVD-based transformation matrix estimation from first/last point clouds generates robot trajectories; GPT-4o validates rendered final states before execution, enabling closed-loop replanning.
4. **Cross-Embodiment Zero-Shot Transfer**: 3D flow world model trained on mixed embodiment data generalizes to Franka (67.5%) and XTrainer (70%) without retraining — 3D flow is embodiment-agnostic.

---

## Problem Background

### Problem to Solve
Robot manipulation policies trained on robot-specific data fail to transfer across embodiments. 2D flow methods (Im2Flow2Act) miss depth-perpendicular movements and rotations. Full video generation (AVDC) is computationally expensive and includes background artifacts. How can manipulation be represented in a way that is: (1) embodiment-agnostic, (2) captures full 3D motion, and (3) scales with internet-scale video data?

### Limitations of Existing Methods
- AVDC: full video generation includes background; computationally expensive; 2D image-space only.
- Im2Flow2Act: 2D flow restricted to image plane — misses depth-perpendicular movement and 3D rotation.
- Rekep: VLM code-based; requires prompt engineering; cannot represent complex continuous trajectories.
- Standard VLAs (π₀): data-hungry; embdiment-specific; struggle with out-of-domain objects/backgrounds.

### Motivation
3D optical flow captures the physical motion of objects through space — the same physical trajectory must occur regardless of which robot executes it. Representing manipulation as 3D point trajectories creates a truly embodiment-agnostic interface that can be trained on diverse robot/human video datasets and then realized by any robot with suitable kinematics.

---

## Method Details

### Model Architecture

3DFlowAction has **four main components**:

1. **3D Flow World Model**: AnimateDiff + LoRA on SD — predicts $\mathcal{F}$ from RGB observations + task prompt
2. **3D Flow Extraction Pipeline**: Grounding-SAM2 + Co-tracker3 + DepthAnythingV2 for dataset construction
3. **Action Generation**: SVD transformation estimation + AnyGrasp pose selection + IK feasibility filtering
4. **Closed-Loop Planning**: GPT-4o validates rendered final states before execution

### Core Modules

#### Module 1: 3D Flow World Model

**Design Motivation**: Predicting 3D object flow from observations and task prompts provides a geometry-aware, embodiment-agnostic action plan that captures full 3D motion including depth-perpendicular trajectories and rotations.

**Implementation**:
- Backbone: Stable Diffusion U-Net with LoRA layers (preserves generative priors)
- Motion module: AnimateDiff, trained from scratch (temporal self-attention across frames)
- Conditioning: CLIP encoder processes RGB observation + task prompt; sinusoidal positional encoding for initial point features
- Output: $\mathcal{F} \in \mathbb{R}^{T \times H \times W \times 4}$ — channels: $(u, v)$ image coordinates, depth $d$, visibility $\nu \in \{0,1\}$
- Training: ManiFlow-110k dataset; AdamW (lr=1e-4, weight decay=0.01); batch 512; 500 epochs; 8×8 V100 GPUs, ~2 days

The 3D flow field is formally defined as:

$$\mathcal{F} \in \mathbb{R}^{T \times H \times W \times 4}$$

Each spatial location $(h,w)$ carries 4 channels: image-plane coordinates $(u,v)$, depth $d$, and binary visibility $\nu$ — encoding where each observed point moves across all $T$ timesteps in 3D space.

#### Module 2: 3D Flow Extraction Pipeline (Dataset Construction)

**Design Motivation**: Existing datasets lack 3D flow labels; automated extraction from RGB-D robot/human videos enables large-scale dataset construction.

**Implementation**:
- Grounding-SAM2: segments robot gripper from initial frame (excluded from flow to maintain embodiment-agnostic representation)
- Co-tracker3: identifies and tracks moving object keypoints across frames
- DepthAnythingV2: estimates per-pixel depth for 3D lifting
- 2D optical flow (from tracked points) projected into 3D space using depth estimates

#### Module 3: Action Generation

**Design Motivation**: 3D flow provides the target object trajectory; robot actions must be derived by finding the end-effector motion that realizes this trajectory given the robot's kinematics.

To synchronize the predicted flow with the current execution state, the action generation stage solves an alignment problem over keypoints. Specifically, it finds the predicted flow timestep $t$ that best matches the current keypoint configuration:

$$f^{(t)}(k_{\text{initial}}) = \min \sum_i \|k_{\text{initial}}^i - k_{\text{pred}}^i(t)\|_2^2$$

This objective aligns initial keypoint positions with predicted flow timestep $t$, anchoring trajectory interpolation to the robot's actual state.

**Implementation**:
1. Transformation matrix estimation: SVD between first and last frame point clouds → rigid body transform $\mathbf{T}$
2. Grasp pose selection: AnyGrasp generates candidates; $\mathbf{T}$ selects task-relevant pose by filtering IK-feasible candidates
3. End-effector trajectory: interpolated along extracted 3D flow keypoints via inverse kinematics

#### Module 4: Closed-Loop Planning with GPT-4o Validation

**Design Motivation**: 3D flow predictions can be rendered to generate predicted final states — GPT-4o can evaluate whether the rendered final state matches the task goal, enabling replanning before execution.

**Implementation**:
- Render predicted final object state from estimated transformation $\mathbf{T}$
- GPT-4o validates rendered image against task instruction
- If validation fails: replan (regenerate flow or modify transformation)
- Closed-loop validation prevents executing physically implausible plans

---

## Experiments

### Datasets

| Dataset | Scale | Characteristics | Usage |
|---------|-------|----------------|-------|
| ManiFlow-110k | 110,000 instances | Synthesized from BridgeV2, RH20T, LIBERO, DROID, AGIBot, Robot Affordances | 3D flow world model training |
| Real-world tasks | 10 trials/task | Pour tea, insert pen, hang cup, open drawer | Evaluation |

### Real-World Manipulation Results

The following table reports real-world task success rates across 10 trials per task, comparing 3DFlowAction against AVDC, Rekep, and Im2Flow2Act:

| Task | AVDC | Rekep | Im2Flow2Act | **3DFlowAction** |
|------|------|-------|-------------|-----------------|
| Pour tea | 10% | 20% | 20% | **60%** |
| Insert pen | 20% | 10% | 20% | **70%** |
| Hang cup | 0% | 30% | 0% | **50%** |
| Open drawer | 50% | 20% | 60% | **100%** |
| **Total** | 20% | 20% | 25% | **70%** |

3DFlowAction achieves 70% overall vs. 20–25% for all baselines — largest gains on insert pen (+50%) and hang cup (+20–50%).

### Cross-Embodiment Transfer

Because 3D flow is embodiment-agnostic by design, the same model transfers zero-shot to different robot platforms without retraining:

| Robot Platform | Success Rate |
|----------------|-------------|
| Franka | 67.5% |
| XTrainer | 70% |

### Object and Background Generalization

The 3D flow world model — pretrained on ManiFlow-110k — generalizes substantially better to unseen objects and backgrounds than AVDC or π₀:

| Generalization Type | 3DFlowAction | AVDC | π₀ |
|--------------------|-------------|------|-----|
| Novel objects | **55%** | 15% | 40% |
| Novel backgrounds | **50%** | 0% | 32.5% |

### Ablation Study

The ablation below isolates the contribution of large-scale pretraining and closed-loop GPT-4o validation:

| Configuration | Success Rate |
|--------------|-------------|
| Full 3DFlowAction | **70%** |
| Without closed-loop GPT-4o rendering | 50% (−20%) |
| Without large-scale pretraining (ManiFlow-110k) | 30% (−40%) |

Large-scale pretraining is the most critical component (−40% without it); closed-loop GPT-4o validation adds +20%.

### Implementation Details

- **Flow World Model Backbone**: Stable Diffusion U-Net + LoRA (frozen encoder/decoder, LoRA in attention)
- **Motion Module**: AnimateDiff (trained from scratch)
- **Conditioning**: CLIP encoder for RGB + text; sinusoidal positional encoding for points
- **Flow Format**: $\mathbb{R}^{T \times H \times W \times 4}$ (u, v, depth, visibility)
- **Dataset Construction Tools**: Grounding-SAM2, Co-tracker3, DepthAnythingV2
- **Optimizer**: AdamW, lr=1e-4, weight decay=0.01
- **Batch Size**: 512
- **Training**: 500 epochs, 8×8 V100 GPUs, ~2 days
- **Grasp Pose Estimation**: AnyGrasp (open-vocabulary)
- **Validation**: GPT-4o (rendered final state)
- **Action Type**: End-effector control via inverse kinematics

---

## Critical Analysis

### Strengths
1. 3D flow is truly embodiment-agnostic — same model works across Franka and XTrainer without retraining.
2. Large-scale ManiFlow-110k dataset provides strong generalization to novel objects and backgrounds.
3. GPT-4o closed-loop validation adds interpretability and planning quality (+20%).

### Limitations
1. **Flexible/deformable objects**: Non-rigid deformations cannot be represented as rigid body transformations — fundamental limitation.
2. **Occlusion sensitivity**: Gripper occlusion during manipulation disrupts 3D flow tracking.
3. **SVD rigid transform assumption**: Assumes rigid body motion — fails for tasks with non-rigid intermediate states.
4. **AnyGrasp dependency**: Grasp quality gates task success; complex grasps may not be in AnyGrasp's distribution.

### Potential Improvements
1. Deformable flow representation for non-rigid manipulation (cloth, dough).
2. Learned transformation estimation replacing SVD for non-rigid cases.
3. More sophisticated closed-loop replanning beyond single GPT-4o validation.

### Reproducibility
- [x] Code promised (GitHub: https://github.com/Hoyyyaard/3DFlowAction/)
- [ ] ManiFlow-110k dataset not yet released
- [x] Training details described (lr, batch size, epochs, hardware)
- [x] Baseline datasets accessible (LIBERO, BridgeV2, DROID public)

---

## Related Notes

### Based On
- [[AnimateDiff]]: Motion modules for temporal flow prediction
- [[Stable Diffusion]]: U-Net backbone with LoRA for flow world model
- [[AnyGrasp]]: Grasp pose estimation from predicted 3D flow
- [[Grounding-SAM2]]: Object segmentation for dataset construction

### Compared Against
- [[AVDC]]: 2D video world model; outperformed 20%→70%
- [[Im2Flow2Act]]: 2D object-centric flow; outperformed 25%→70%
- [[Rekep]]: VLM code-based planning; outperformed 20%→70%

### Method Related
- [[3D Optical Flow]]: Core embodiment-agnostic action representation
- [[SVD Transform Estimation]]: Rigid body transformation from point clouds
- [[Cross-Embodiment Transfer]]: Zero-shot generalization across robot platforms

### Hardware/Data Related
- [[ManiFlow-110k]]: Synthesized 3D flow dataset for world model pretraining

---

## Quick Reference Card

> [!summary] 3DFlowAction (arXiv 2025)
> - **Core**: 3D flow world model (AnimateDiff+LoRA on SD, ManiFlow-110k) → SVD transform estimation + AnyGrasp pose selection + GPT-4o validation + IK → robot execution; embodiment-agnostic by design
> - **Method**: $\mathcal{F} \in \mathbb{R}^{T \times H \times W \times 4}$; AdamW lr=1e-4, batch 512, 500 epochs, 8×8 V100 ~2 days; Grounding-SAM2 + Co-tracker3 + DepthAnythingV2 for labels
> - **Results**: 70% real-world (vs. 20–25% AVDC/Rekep/Im2Flow2Act); 67.5–70% zero-shot cross-embodiment; 55% novel objects, 50% novel backgrounds
> - **Code**: Promised at GitHub

---

*Note created: 2026-05-20*
