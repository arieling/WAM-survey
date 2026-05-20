---
title: "NovaFlow: Zero-Shot Manipulation via Actionable Flow from Generated Videos"
method_name: "NovaFlow"
authors: [Hongyu Li, George Konidaris, Lingfeng Sun, Jiahui Fu, Yafei Hu, Duy Ta, Jennifer Barry]
year: 2025
venue: arXiv 2025
tags: [video-generation, 3d-flow, zero-shot, robot-policy, cascaded-explicit-geometric, point-tracking, model-predictive-control]
zotero_collection: 3-Robotics/WAM-Survey
image_source: online
arxiv_html: https://arxiv.org/abs/2510.08568
created: 2026-05-20
---

# Paper Note: NovaFlow: Zero-Shot Manipulation via Actionable Flow from Generated Videos

## Metadata

| Item | Content |
|------|---------|
| Institution | Brown University, Robotics and AI Institute |
| Date | October 2025 |
| Project Page | N/A |
| Baselines | AVDC, VidBot, Diffusion Policy (10–30 demos), IDM (UniPi) |
| Links | [arXiv](https://arxiv.org/abs/2510.08568) / Code: N/A |

---

## One-Sentence Summary

> NovaFlow converts text task descriptions into robot actions by generating manipulation videos (Wan2.1 or Veo), extracting 3D object flow via monocular depth calibration + TAPIP3D point tracking, and executing via Kabsch-algorithm rigid-body control or MPC-based deformable dynamics — all zero-shot with no task-specific training, achieving competitive results against imitation learning with 10–30 demonstrations.

---

## Core Contributions

1. **Modular Zero-Shot Pipeline**: Fully modular architecture — video generator, depth estimator, 3D tracker, and executor are independently swappable — enabling continuous improvement as frontier models advance without retraining.
2. **Depth-Calibrated 3D Flow Extraction**: MegaSaM monocular depth estimation anchored to ground-truth initial depth map via scaling factors, then TAPIP3D tracks 32×32 query points in calibrated XYZ coordinates — producing metric-scale 3D object flow $\mathcal{F} \in \mathbb{R}^{T \times M \times 3}$.
3. **Unified Rigid + Deformable Execution**: Kabsch-algorithm rotation estimation for rigid objects; MPC with particle-based dynamics model for deformable objects — single pipeline handles both manipulation regimes.
4. **VLM Rejection Sampling**: Generates N video candidates; Google Gemini selects the most plausible flow visualization — improving video quality without fine-tuning.

---

## Problem Background

### Problem to Solve
Zero-shot robot manipulation without any task-specific demonstrations or fine-tuning. Can frontier video generation models provide sufficient action information to execute diverse manipulation tasks (rigid, articulated, deformable) on real robots without robot training data?

### Limitations of Existing Methods
- AVDC: video-to-action via 2D flow; no 3D geometric recovery; requires simulated or robot data.
- VidBot: video-conditioned; limited 3D understanding; zero-shot but lower precision.
- Diffusion Policy: requires 10–30+ task-specific demonstrations; not zero-shot.
- IDM (UniPi-style): requires robot data for inverse dynamics training; not truly zero-shot.

### Motivation
Frontier video models (Wan2.1, Veo) have consumed internet-scale video data and can generate physically plausible manipulation sequences from text descriptions. If 3D object flow can be reliably extracted from these generated videos, the flow provides a task-level geometric plan executable by any robot without task-specific training.

---

## Method Details

### Model Architecture

NovaFlow has **two main components**:

1. **Flow Generator**: Video generation → depth calibration → 3D point tracking → object grounding → actionable 3D flow
2. **Flow Executor**: Rigid (Kabsch + IK) or Deformable (MPC) action generation from 3D flow

### Core Modules

#### Module 1: Video Generation

**Design Motivation**: Frontier video models encode physics, object affordances, and task semantics from internet-scale training — generating plausible manipulation sequences without any robot data.

**Implementation**:
- Wan2.1: 41 frames at 1280×720, 16 FPS
- Veo: 8-second clips at 24 FPS (more robust without goal images)
- Optional goal image conditioning: improves precision tasks substantially (block insertion: 40%→80%)
- VLM rejection sampling: generate N candidates; Gemini selects most plausible flow visualization

#### Module 2: 3D Flow Extraction

**Design Motivation**: Monocular depth estimation suffers from scale ambiguity; anchoring to initial ground-truth depth provides metric-scale 3D coordinates for reliable trajectory extraction.

**Implementation**:
1. MegaSaM: monocular depth estimation on generated video frames
2. Depth calibration: scaling factors computed from ratio of initial ground-truth depth (from robot depth sensor) to estimated depth at first frame
3. TAPIP3D: tracks M=32×32 query points across T frames in calibrated XYZ space
4. Grounded-SAM2 (Grounding DINO + SAM2): isolates target object regions, filtering background point tracks

The resulting actionable 3D flow is a tensor $\mathcal{F} \in \mathbb{R}^{T \times M \times 3}$ representing M tracked keypoints in calibrated metric XYZ coordinates across T timesteps extracted from the generated manipulation video. This compact representation captures the task-level geometric plan in 3D space.

#### Module 3: Rigid Object Executor (Kabsch Algorithm)

**Design Motivation**: For rigid objects, the 3D flow represents a rigid body transformation at each timestep — Kabsch algorithm estimates the optimal rotation between point cloud snapshots.

**Implementation**:

Given M tracked keypoints per frame, the Kabsch algorithm finds the optimal rotation $R$ that aligns the initial keypoint configuration to the target configuration at time $t$:

$$\arg\min_R \sum_{i=1}^K \|R(f_i^1 - c^1) - (f_i^t - c^t)\|^2$$

where $c^t$ is the centroid of keypoints at time $t$ and $f_i^t$ is the $i$-th keypoint. The end-effector pose is then composed as $T_{ee}^t = T_{obj}^t \cdot T_{grasp}$, and the full trajectory is smoothed with a Levenberg-Marquardt IK solver.

#### Module 4: Deformable Object Executor (MPC)

**Design Motivation**: Deformable objects (rope, cloth) cannot be represented as rigid bodies — dense flow provides a per-particle objective for model predictive control.

**Implementation**:
- Particle-based dynamics model tracks object state $s^t$ (dense particle positions)
- The MPC objective minimizes the distance between current particle positions and the target flow trajectory:

$$\min \sum_{t=1}^{N_p} \|s_i^t - f_i^t\|^2$$

over a planning horizon $H$ with dynamics constraints. This drives the robot to bring each object particle to its corresponding target position from the generated video plan.

---

## Experiments

### Datasets

| Dataset | Scale | Characteristics | Usage |
|---------|-------|----------------|-------|
| Real-world Franka | 10 trials/task | Tabletop manipulation | Evaluation |
| Real-world Spot | Limited | Quadruped platform | Cross-platform test |

### Implementation Details

- **Video Generator 1**: Wan2.1 (41 frames, 1280×720, 16 FPS)
- **Video Generator 2**: Veo (8-second clips, 24 FPS, Google)
- **Depth Estimator**: MegaSaM (monocular)
- **3D Point Tracker**: TAPIP3D (32×32 query points per frame)
- **Object Grounding**: Grounded-SAM2 (Grounding DINO + SAM2)
- **Rejection Sampling**: Google Gemini (VLM video quality evaluator)
- **Rigid Executor**: Kabsch algorithm + Levenberg-Marquardt trajectory optimizer
- **Deformable Executor**: MPC with particle dynamics model, horizon $H$
- **Runtime**: ~2 minutes end-to-end per task (single H100 GPU, Veo model)
- **Robot Platforms**: Franka Panda (tabletop), Boston Dynamics Spot (quadruped)

### Real-World Task Success Rate (10 trials each)

The full pipeline (Figure 1) is: text description → (Wan2.1/Veo) video + optional goal image conditioning → MegaSaM depth → calibration → TAPIP3D 3D tracking → Grounded-SAM2 object masking → actionable 3D flow → [rigid: Kabsch+IK] or [deformable: MPC] → robot execution. N generated videos are first filtered by Gemini for physical plausibility. The resulting zero-shot success rates are:

| Task Type | Task | NovaFlow | AVDC | VidBot | Diffusion Policy (30 demos) |
|-----------|------|----------|------|--------|---------------------------|
| Rigid | Hanging mug | Best | Lower | Lower | Competitive |
| Rigid | Block insertion | 80% (w/goal img) | — | — | Competitive |
| Rigid | Cup placement | Outperforms | — | — | — |
| Rigid | Watering plant | Outperforms | — | — | — |
| Articulated | Drawer opening | Outperforms | — | — | — |
| Deformable | Rope straightening | Outperforms | — | — | — |

NovaFlow outperforms zero-shot baselines (AVDC, VidBot) across all task types and is competitive with imitation learning methods using 10–30 demonstrations — without any task-specific training.

### Ablation — Goal Image Conditioning

| Configuration | Block Insertion Success |
|--------------|------------------------|
| Wan2.1 without goal image | 40% |
| Wan2.1 with goal image | **80%** |
| Veo without goal image | 80% |

Goal image conditioning doubles success on precision tasks (block insertion: 40%→80%); Veo is more robust to the absence of goal images.

### Failure Analysis

| Failure Mode | Frequency |
|-------------|-----------|
| Grasp failure | Most frequent |
| Execution error | Frequent |
| 3D tracking inaccuracy | Less frequent |
| Video hallucination | Less frequent |

Physical interaction (grasping, execution) is the primary bottleneck — upstream flow estimation is relatively robust.

---

## Critical Analysis

### Strengths
1. Truly zero-shot — no task-specific demonstrations or fine-tuning needed.
2. Modular architecture enables drop-in replacement of components as better video/depth/tracking models emerge.
3. Handles rigid, articulated, and deformable objects in one unified pipeline.
4. Competitive with 10–30-demonstration imitation learning.

### Limitations
1. **~2-minute inference**: Not suitable for real-time or reactive manipulation.
2. **Grasp/execution bottleneck**: Physical execution fails even when 3D flow is accurate — requires more robust grasp planners.
3. **Video hallucination**: Frontier video models occasionally generate physically implausible manipulations.
4. **Proprietary dependencies**: Veo (Google closed API) and Gemini validation limit reproducibility.

### Potential Improvements
1. Faster 3D tracking and depth estimation for reduced latency.
2. Better grasp planning integrated with 3D flow contact information.
3. Closed-loop flow re-estimation from current robot camera during execution.

### Reproducibility
- [ ] Code open-sourced
- [ ] Pretrained models available (Veo is proprietary)
- [x] Architecture described (all modules named and detailed)
- [ ] Exact results tables not fully specified in extracted content

---

## Related Notes

### Based On
- Wan2.1: Primary open-source video generation model
- Veo: Google frontier video model (closed API)
- TAPIP3D: 3D point tracking in calibrated space
- Grounded-SAM2: Object grounding and segmentation
- MegaSaM: Monocular depth estimation

### Compared Against
- [AVDC](AVDC.md): 2D video flow method; outperformed on all tasks
- Diffusion Policy: 10–30-demo imitation learning; matched without demonstrations

### Method Related
- Kabsch Algorithm: Optimal rigid rotation estimation from point correspondences
- Model Predictive Control: Deformable object manipulation planning
- Zero-Shot Manipulation: Task execution without task-specific training
- Rejection Sampling: VLM-based video quality selection

---

## Quick Reference Card

> [!summary] NovaFlow (arXiv 2025)
> - **Core**: Wan2.1/Veo video generation → MegaSaM depth calibration → TAPIP3D 3D tracking (32×32 pts) → Grounded-SAM2 object masking → $\mathcal{F} \in \mathbb{R}^{T \times M \times 3}$; rigid: Kabsch+LM IK; deformable: MPC particle control; Gemini rejection sampling
> - **Method**: Fully modular, no fine-tuning; goal image conditioning optional; N candidate rejection sampling via Gemini; ~2 min/task on H100
> - **Results**: Outperforms AVDC/VidBot (zero-shot); competitive with Diffusion Policy (10–30 demos); block insertion 40%→80% with goal image conditioning
> - **Code**: N/A

---

*Note created: 2026-05-20*
