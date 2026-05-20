---
title: "Dreamitate: Real-World Visuomotor Policy Learning via Video Generation"
method_name: "Dreamitate"
authors: [Junbang Liang, Ruoshi Liu, Ege Ozguroglu, Sruthi Sudhakar, Achal Dave, Pavel Tokmakov, Shuran Song, Carl Vondrick]
year: 2024
venue: arXiv 2024
tags: [video-generation, human-demonstration, stereo-video, 3d-tracking, robot-policy, cascaded-explicit-geometric, stable-video-diffusion]
zotero_collection: 3-Robotics/WAM-Survey
image_source: online
arxiv_html: https://arxiv.org/abs/2406.16862
created: 2026-05-20
---

# Paper Note: Dreamitate: Real-World Visuomotor Policy Learning via Video Generation

## Metadata

| Item | Content |
|------|---------|
| Institution | Columbia University, Toyota Research Institute, Stanford University |
| Date | June 2024 |
| Project Page | N/A |
| Baselines | Diffusion Policy |
| Links | [arXiv](https://arxiv.org/abs/2406.16862) / Code: N/A |

---

## One-Sentence Summary

> Dreamitate fine-tunes Stable Video Diffusion on stereo human manipulation videos collected with 3D-printed trackable tools, then generates future execution videos from which MegaPose extracts 3D tool trajectories directly transferable to robot end-effector control — achieving 85–92.5% success across four tasks with strong generalization from one-third the training data.

---

## Core Contributions

1. **Trackable Tool as Embodiment Bridge**: Using common tools with known CAD models (3D-printed) eliminates embodiment gap — the same tool is used by the human demonstrator and the robot, enabling direct 3D trajectory transfer without imitation learning.
2. **Stereo Video Diffusion for 3D Trajectory Extraction**: Fine-tuning SVD to generate stereo video pairs (two cameras at 45° separation) provides the 3D geometric information needed for precise MegaPose-based tool tracking.
3. **Data-Efficient Generalization**: Dreamitate retains internet-scale priors from SVD pretraining, enabling strong generalization with one-third the demonstrations compared to baselines — data efficiency critical for real-world deployment.
4. **Human-Video-Only Training**: No teleoperation required — human demonstrations collected with trackable tools are sufficient for training without robot-specific data collection.

---

## Problem Background

### Problem to Solve
Robot manipulation training requires expensive teleoperation data. Human demonstration videos are abundant but suffer from embodiment mismatch — the robot must somehow transfer human hand/arm motions to its own morphology. How can human manipulation videos be used directly for robot learning without embodiment transfer?

### Limitations of Existing Methods
- Teleoperation-based BC (Diffusion Policy): requires expensive robot-specific demonstrations; 12.5–55% success on evaluated tasks; poor data efficiency.
- Standard video imitation: embodiment mismatch between human hands and robot grippers causes distribution shift.
- Prior flow-based methods: require robot data for policy training; limited 3D understanding.

### Motivation
Using physical tools with known 3D geometry as the manipulation interface creates a platform-agnostic task representation: the tool's 3D pose trajectory can be extracted from video via model-based pose estimation (MegaPose), then replayed by any robot that can hold the same tool. SVD's internet-scale visual priors enable generalization to novel objects with minimal fine-tuning.

---

## Method Details

### Model Architecture

Dreamitate has **three stages**:

1. **Data Collection**: Stereo video recording of human demonstrations with 3D-printed tools
2. **Video Generation**: SVD fine-tuned to predict future stereo execution video from initial frame
3. **Action Extraction**: MegaPose tracks tool 3D pose in generated video → robot end-effector trajectory via inverse kinematics

### Core Modules

#### Module 1: Human Demonstration with Trackable Tools

**Design Motivation**: Tools with known CAD models can be precisely tracked in 3D via model-based pose estimation. Using the same physical tool for both human demonstrations and robot execution eliminates embodiment gap entirely.

**Implementation**:
- 3D-printed tools with known CAD models (rotation handle, scoop, sweeper, push tool)
- Stereo camera setup: two cameras positioned 45° apart, resolution 768×448
- Human demonstrator uses tools naturally — no motion capture or teleoperation
- Dataset sizes: Rotation (31 objects, 371 frames), Scooping (17 bowls, 368 frames), Sweeping (6 particles, 356 frames), Push-Shape (26 letter-shaped objects, 727 frames)

#### Module 2: Stereo Video Diffusion (Fine-tuned SVD)

**Design Motivation**: Generating stereo video pairs provides 3D geometric consistency needed for accurate MegaPose tracking. Fine-tuning SVD (pretrained on internet video) with task-specific demonstrations retains internet-scale priors while specializing to manipulation dynamics.

**Implementation**:
- Base model: Stable Video Diffusion (SVD), pretrained on internet-scale video
- Fine-tuning: only spatial and temporal attention layers (encoder/decoder frozen)
- Stereo conditioning: per-frame embeddings conditioned on viewing angle to generate left + right camera views

The model is trained to reconstruct both camera views per frame. The training loss is a per-frame L2 reconstruction objective applied simultaneously to the left and right stereo streams:

$$\mathcal{L} = \|\hat{v}_t^1 - v_t^1\|_2 + \|\hat{v}_t^2 - v_t^2\|_2$$

where $v_t^1$ and $v_t^2$ are the ground-truth left and right camera video frames at timestep $t$, and $\hat{v}_t^1$, $\hat{v}_t^2$ are the corresponding generated frames. This loss enforces consistency between generated and real demonstration frames across both views simultaneously.

- Inference: 30 denoising steps, classifier-free guidance = 1.0
- Resolution: 768×448, learning rate: 1e-5, batch size: 3–4, training steps: 15,360–17,408

#### Module 3: 3D Tool Tracking and Trajectory Extraction

**Design Motivation**: Model-based 6-DoF pose estimation (MegaPose) on generated stereo frames extracts precise 3D tool trajectory without learning a separate inverse dynamics model.

**Implementation**:
- MegaPose estimates 6-DoF tool pose in each generated video frame using known CAD model

The robot action at each timestep is obtained by applying a composed transformation — MegaPose 6-DoF pose estimation followed by inverse kinematics — to the corresponding generated video frame:

$$a_t = T(\hat{v}_t)$$

where $\hat{v}_t$ is the generated stereo video frame at timestep $t$ and $T(\cdot)$ denotes the full chain from MegaPose pose estimation to inverse kinematics. No additional robot-specific learning is required beyond this deterministic mapping.

---

## Experiments

### Datasets

| Dataset | Scale | Characteristics | Usage |
|---------|-------|----------------|-------|
| Rotation task | 31 objects, 371 frames | Diverse rigid objects, rotation handle tool | Train/Eval |
| Scooping task | 17 bowls × 8 particle types, 368 frames | Granular material, scoop tool | Train/Eval |
| Sweeping task | 6 particle configurations, 356 frames | Soft particles, sweeper tool | Train/Eval |
| Push-Shape task | 26 letter-shaped objects, 727 frames | Letter recognition + pushing, push tool | Train/Eval |

### Task Success Rate

The following table reports success rates across 40 trials per task, comparing Dreamitate to the teleoperation-based Diffusion Policy baseline:

| Task | Dreamitate | Diffusion Policy |
|------|------------|-----------------|
| Rotation | **92.5%** | 55.0% |
| Scooping | **85.0%** | 55.0% |
| Sweeping | **92.5%** | 12.5% |
| Push-Shape (mIoU) | **0.731** | 0.550 |

Dreamitate substantially outperforms Diffusion Policy across all tasks — largest gap on Sweeping (+80%), demonstrating advantage on tasks requiring geometric precision beyond what teleoperation-trained policies achieve.

### Data Efficiency (Generalization Performance)

The data efficiency ablation compares both methods at reduced training set sizes. Dreamitate's internet-scale SVD priors allow it to maintain strong generalization with one-third the training data, while Diffusion Policy degrades substantially:

| Data Fraction | Dreamitate | Diffusion Policy |
|--------------|------------|-----------------|
| 100% | Strong | Baseline |
| 33% | Strong (maintained) | Significant drop |

### Implementation Details

- **Video Generator**: Stable Video Diffusion (SVD), pretrained on internet video
- **Fine-tuned Layers**: Spatial and temporal attention only (encoder/decoder frozen)
- **Resolution**: 768×448 pixels
- **Learning Rate**: 1e-5
- **Batch Size**: 3–4
- **Training Steps**: 15,360–17,408 per task
- **Denoising Steps (Inference)**: 30
- **Classifier-Free Guidance**: 1.0
- **3D Tracker**: MegaPose (model-based 6-DoF pose estimation)
- **Camera Setup**: Stereo, two cameras at 45° separation
- **Action Type**: 6-DoF end-effector control via inverse kinematics
- **Evaluation**: 40 trials per task

---

## Critical Analysis

### Strengths
1. No teleoperation required — human video collection is scalable and low-cost.
2. Direct 3D trajectory transfer via model-based tracking — no learned inverse dynamics, no embodiment-specific training.
3. Excellent data efficiency: one-third of demonstrations maintains strong generalization (SVD priors).
4. 85–92.5% success rates demonstrate practical real-world effectiveness.

### Limitations
1. **Trackable tool requirement**: Fails with heavy occlusion or tools without known CAD models — limits to specific tool-use tasks.
2. **Rigid tool constraint**: Fine-grained manipulation (soft tools, deformable objects) is difficult to represent as 6-DoF rigid body motion.
3. **Open-loop execution**: Generated video is fixed before execution — no replanning mid-task from current observations.
4. **Inference latency**: 30-step SVD denoising prevents real-time closed-loop control.

### Potential Improvements
1. Adaptive video replanning from current observations for closed-loop execution.
2. Extend to deformable tools via particle-based trajectory representation.
3. Faster inference (distillation) for real-time control.

### Reproducibility
- [ ] Code open-sourced
- [ ] Pretrained models available
- [x] Training details described (steps, learning rate, batch size, resolution)
- [ ] Demonstration videos not publicly released

---

## Related Notes

### Based On
- [[Stable Video Diffusion]]: Base model for stereo video generation
- [[MegaPose]]: Model-based 6-DoF pose estimation for 3D trajectory extraction

### Compared Against
- [[Diffusion Policy]]: Teleoperation-based BC baseline; outperformed by +30–80% across tasks

### Method Related
- [[Human Demonstration]]: Tool-based human video as robot training data
- [[Stereo Vision]]: Dual-camera setup for 3D geometric recovery
- [[Inverse Kinematics]]: Tool trajectory to robot end-effector mapping
- [[Classifier-Free Guidance]]: Video generation conditioning mechanism

---

## Quick Reference Card

> [!summary] Dreamitate (arXiv 2024)
> - **Core**: Fine-tune SVD on stereo human demo videos (3D-printed trackable tools); generate future stereo execution video; MegaPose extracts 6-DoF tool trajectory → robot end-effector via IK
> - **Method**: SVD fine-tune (spatial+temporal attention, 15K–17K steps, lr=1e-5, batch 3–4, 768×448); stereo L2 loss; 30-step DDPM inference; MegaPose pose estimation on generated frames
> - **Results**: 85–92.5% success (vs 12.5–55% Diffusion Policy); 0.731 mIoU Push-Shape; strong 1/3-data generalization
> - **Code**: N/A

---

*Note created: 2026-05-20*
