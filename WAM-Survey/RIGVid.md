---
title: "Robotic Manipulation by Imitating Generated Videos Without Physical Demonstrations"
method_name: "RIGVid"
authors: [Shivansh Patel, Shraddhaa Mohan, Hanlin Mai, Unnat Jain, Svetlana Lazebnik, Yunzhu Li]
year: 2025
venue: arXiv 2025
tags: [video-generation, 6dof-tracking, closed-loop, robot-policy, cascaded-explicit-geometric, zero-shot, kling]
zotero_collection: 3-Robotics/WAM-Survey
image_source: online
arxiv_html: https://arxiv.org/abs/2507.00990
created: 2026-05-20
---

# Paper Note: Robotic Manipulation by Imitating Generated Videos Without Physical Demonstrations (RIGVid)

## Metadata

| Item | Content |
|------|---------|
| Institution | UIUC, UC Irvine, Columbia University |
| Date | July 2025 |
| Project Page | N/A |
| Baselines | ReKep, Track2Act, AVDC, 4D-DPM, Gen2Act |
| Links | [arXiv](https://arxiv.org/abs/2507.00990) / Code: N/A |

---

## One-Sentence Summary

> RIGVid imitates AI-generated videos (Kling v1.6, GPT-4o filtered) without any physical demonstrations by extracting 6-DoF object pose trajectories via monocular depth calibration + FoundationPose tracking, retargeting to robot end-effector via rigid body composition, and enabling closed-loop execution with deviation detection and recovery — achieving 85% success vs. 7.5–67.5% for prior zero-shot and video-based methods.

---

## Core Contributions

1. **GPT-4o Video Filtering**: Automatic quality filtering of generated videos using GPT-4o examining four evenly-spaced frames — achieves 0.84 Pearson correlation with human judgment (vs. 0.34 for video-text consistency scores), drastically outperforming standard VLM similarity metrics.
2. **6-DoF Object Pose Trajectory Extraction**: Using FoundationPose on depth-calibrated generated video frames — structured 6-DoF pose representation outperforms point tracks, optical flow, and feature fields for manipulation tasks with occlusion and rotation.
3. **Closed-Loop Execution with Deviation Recovery**: Real-time FoundationPose tracking during deployment; deviations exceeding 3cm/20° trigger recovery (backtrack to last successful point and resume) — enabling robust execution despite open-loop trajectory planning.
4. **Zero-Shot Cross-Embodiment Transfer**: Same pipeline tested on xArm7 and ALOHA robot (80% pouring success) and bimanual configuration — no robot-specific training.

---

## Problem Background

### Problem to Solve
Robot manipulation typically requires physical demonstrations via teleoperation or motion capture — expensive and time-consuming. AI-generated videos from language instructions provide cheap, scalable demonstrations. How can generated videos be converted to robot execution without physical demonstrations or robot-specific training?

### Limitations of Existing Methods
- Gen2Act: generates human videos, sparse point tracking; no 6-DoF pose; 67.5% success.
- AVDC: 2D optical flow only; no 3D geometric recovery; 32.5% success.
- 4D-DPM: feature field based; 35% success.
- Track2Act: only 7.5% success — point tracking insufficient for complex 3D tasks.
- ReKep (VLM keypoints): lacks temporal trajectory detail; 50% success.

### Motivation
6-DoF object pose is a more informative and robust action representation than 2D flow, point tracks, or keypoints — it directly specifies spatial position and orientation, handles occlusion naturally via model-based tracking, and can be reliably retargeted to robot end-effector motion via rigid body composition.

---

## Method Details

### Model Architecture

RIGVid is a **modular zero-shot pipeline**:

1. **Video Generation + Filtering**: Kling v1.6 API + GPT-4o automatic filtering
2. **Depth Calibration**: Monocular depth anchored to initial real RGB-D
3. **6-DoF Pose Extraction**: Grounding DINO → SAM-2 → FoundationPose → smoothing
4. **Motion Retargeting**: Rigid body composition (object→gripper→end-effector)
5. **Closed-Loop Execution**: Real-time deviation detection + recovery

### Core Modules

#### Module 1: Video Generation and GPT-4o Filtering

**Design Motivation**: Not all generated videos depict physically plausible task execution — automatic filtering via VLM dramatically improves downstream success by removing hallucinated or failed generations.

**Implementation**:
- Generator: Kling v1.6 (best among Sora, Kling v1.5, v1.6 tested)
  - Sora: cinematic quality but changes viewpoint/objects; 0% filter pass rate
  - Kling v1.5: 27% avg. pass rate (physically implausible)
  - Kling v1.6: 67.5% avg. pass rate (most reliable)
- GPT-4o filtering: examines 4 evenly-spaced frames; validates task execution correctness
- Filtering accuracy: pouring 83%, lifting 66%, placing 55%, sweeping 45%
- Correlation with human judgment: 0.84 (vs. 0.34 for video-text consistency)

#### Module 2: Depth Calibration

**Design Motivation**: Monocular depth estimation suffers from scale/shift ambiguity. Anchoring to the initial real depth map (from robot RGB-D sensor) provides metric-scale depth for accurate 3D pose extraction.

**Implementation**:
- Monocular depth predictor (Ke et al.) applied to all generated video frames
- Affine transformation computed to align estimated depth scale/shift to initial real depth map
- Provides metric-scale 3D for FoundationPose initialization

#### Module 3: 6-DoF Object Pose Extraction

**Design Motivation**: 6-DoF pose provides structured, complete geometric description of object state — handles occlusion via model-based prior, captures rotations that 2D flow cannot represent.

**Implementation**:
1. GPT-4o: identifies active object category from task description
2. Grounding DINO: grounds category to bounding box in initial frame
3. SAM-2: refines segmentation to precise mask
4. BundleSDF: reconstructs 3D mesh from initial RGB-D observations
5. FoundationPose: tracks 6-DoF pose across all generated video frames using mesh prior
6. Smoothing: moving average filter reduces pose estimation jitter

#### Module 4: Motion Retargeting

**Design Motivation**: Robot end-effector can be related to the grasped object via a fixed rigid body transformation if the grasp is maintained throughout execution.

**Implementation**:

Assuming a fixed grasp transform $T_{\text{gripper} \to \text{object}}^{\text{grasp}}$ from grasp pose, the end-effector trajectory at each timestep $t$ is computed by composing the extracted object pose with the grasp-to-end-effector offset:

$$T_{ee}^t = T_{\text{gripper} \to \text{ee}} \circ T_{\text{object}}^t \circ (T_{\text{gripper} \to \text{object}}^{\text{grasp}})^{-1}$$

This retargets the entire extracted object trajectory to robot end-effector commands: the inverse of the grasp-time transform converts from world to object frame, $T_{\text{object}}^t$ provides the predicted pose at time $t$, and the constant gripper-to-end-effector offset completes the kinematic chain.

#### Module 5: Closed-Loop Execution with Recovery

**Design Motivation**: Open-loop trajectory replay fails when physical execution deviates from planned trajectory. Real-time deviation detection enables recovery without replanning.

**Implementation**:
- Real-time FoundationPose tracking on live robot RGB-D during execution

Recovery is triggered when the measured pose deviates beyond threshold from the planned trajectory:

$$\text{Deviate} = (\|\Delta p\|_2 > 3\text{cm}) \lor (\|\Delta \theta\|_2 > 20°)$$

When triggered, the system backtracks to the last successful trajectory point and resumes execution. Grasp planning uses AnyGrasp for generating and selecting feasible grasp candidates.

---

## Experiments

### Datasets

| Dataset | Scale | Characteristics | Usage |
|---------|-------|----------------|-------|
| Generated videos | Per-task (Kling) | Language-conditioned, GPT-4o filtered | Plan generation |
| Real-world eval | 10 trials/task | xArm7, Orbbec Femto Bolt | Evaluation |

### Implementation Details

- **Video Generator**: Kling v1.6 API
- **Video Filtering**: GPT-4o (4 evenly-spaced frames)
- **Depth Estimator**: Ke et al. monocular depth + affine calibration to initial depth
- **Object Grounding**: Grounding DINO
- **Segmentation**: SAM-2
- **Pose Tracker**: FoundationPose (requires BundleSDF mesh)
- **Mesh Reconstruction**: BundleSDF
- **Smoothing**: Moving average filter
- **Grasp Planner**: AnyGrasp
- **Deviation Thresholds**: 3cm position, 20° orientation
- **Robot Platform**: xArm7 + Orbbec Femto Bolt RGBD camera

### Video Model Comparison

The full pipeline is: language + initial scene → Kling v1.6 generation → GPT-4o filtering → monocular depth calibration → Grounding DINO + SAM-2 + FoundationPose 6-DoF tracking → rigid body retargeting → AnyGrasp → closed-loop xArm7 execution. The choice of video generator has a strong effect on downstream success due to GPT-4o filtering quality:

| Model | Avg. Filter Pass Rate | Usability |
|-------|----------------------|-----------|
| Sora | 0% | Not usable |
| Kling v1.5 | 27% | Limited |
| **Kling v1.6** | **67.5%** | **Best** |

Sora completely fails GPT-4o filtering (changes viewpoint/objects); Kling v1.6 provides best generation quality for manipulation.

### Task Success Rate (10 trials each)

RIGVid from generated videos (85%) closely approaches real-video performance (90%) — only a 5% gap despite using AI-generated demonstrations:

| Task | RIGVid (Generated) | Real Videos |
|------|-------------------|-------------|
| Pouring Water | **100%** | 100% |
| Lifting Lid | **80%** | 90% |
| Placing Spatula | **90%** | 90% |
| Sweeping Trash | **70%** | 80% |
| **Average** | **85%** | **90%** |

### Comparison Against Baselines

| Method | Success Rate |
|--------|-------------|
| **RIGVid (Ours)** | **85%** |
| Gen2Act | 67.5% |
| ReKep | 50% |
| 4D-DPM | 35% |
| AVDC | 32.5% |
| Track2Act | 7.5% |

RIGVid outperforms all baselines by substantial margins — +17.5% over Gen2Act (next best), +35% over ReKep, +77.5% over Track2Act.

### Cross-Embodiment Transfer

Rigid body retargeting transfers seamlessly to ALOHA without modification:

| Platform | Task | Success Rate |
|----------|------|-------------|
| ALOHA robot | Pouring | 80% |
| xArm7 (bimanual) | Shoe placement | Success demonstrated |

---

## Critical Analysis

### Strengths
1. GPT-4o filtering (0.84 correlation) dramatically improves downstream quality — practical and effective.
2. 6-DoF pose outperforms all lower-dimensional action representations (flow, point tracks, feature fields).
3. Closed-loop deviation recovery adds crucial robustness to open-loop planning.
4. 85% success from generated videos (only 5% below real-video baseline).

### Limitations
1. **BundleSDF mesh requirement**: Requires pre-computation of 3D object mesh from initial observations — adds setup time, may fail for complex/transparent objects.
2. **Rigid body assumption**: Motion retargeting assumes fixed grasp transform — fails for tasks requiring gripper repositioning during execution.
3. **Kling API dependency**: Closed-source commercial API limits reproducibility and offline deployment.
4. **Depth estimation bottleneck**: All execution failures (except one gripper slip) attributed to depth estimation errors.

### Potential Improvements
1. Open-source video model replacement for reproducibility.
2. Online mesh update during execution for deformable or repositioning tasks.
3. Better metric depth models (e.g., anchoring to multiple RGB-D frames rather than just initial frame).

### Reproducibility
- [ ] Code open-sourced
- [ ] Pretrained models (Kling is proprietary API)
- [x] Pipeline described (all components named)
- [ ] BundleSDF mesh preprocessing details not specified

---

## Related Notes

### Based On
- [[Kling]]: Frontier image-to-video model for generating manipulation demonstrations
- [[FoundationPose]]: 6-DoF object pose tracking for action extraction
- [[Grounding DINO]]: Open-vocabulary object detection for target identification
- [[SAM-2]]: Object segmentation for FoundationPose initialization
- [[BundleSDF]]: 3D object mesh reconstruction from RGB-D sequences

### Compared Against
- [[Gen2Act]]: Human video + sparse tracking; outperformed 67.5%→85%
- [[AVDC]]: 2D optical flow; outperformed 32.5%→85%
- [[Track2Act]]: Point tracking; outperformed 7.5%→85%

### Method Related
- [[6-DoF Pose Tracking]]: Structured pose representation for manipulation
- [[Motion Retargeting]]: Rigid body transform for embodiment transfer
- [[Closed-Loop Execution]]: Real-time deviation detection and recovery

---

## Quick Reference Card

> [!summary] RIGVid (arXiv 2025)
> - **Core**: Kling v1.6 generation → GPT-4o filtering (0.84 correlation) → monocular depth calibration → Grounding DINO + SAM-2 + FoundationPose 6-DoF tracking → rigid body retargeting → closed-loop execution (3cm/20° recovery); zero-shot, no physical demos
> - **Method**: BundleSDF mesh from initial obs; moving average smoothing; AnyGrasp; xArm7 + Orbbec Femto Bolt; ALOHA cross-embodiment
> - **Results**: 85% avg (vs. 67.5% Gen2Act, 50% ReKep, 32.5% AVDC, 7.5% Track2Act); 80% ALOHA zero-shot; 100% pouring
> - **Code**: N/A

---

*Note created: 2026-05-20*
