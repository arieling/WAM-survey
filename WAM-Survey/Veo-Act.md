---
title: "Veo-Act: How Far Can Frontier Video Models Advance Generalizable Robot Manipulation?"
method_name: "Veo-Act"
authors: [Zhongru Zhang, Chenghan Yang, Qingzhou Lu, Yanjiang Guo, Jianke Zhang, Yucheng Hu, Jianyu Chen]
year: 2026
venue: arXiv 2026
tags: [video-generation, inverse-dynamics, hierarchical-policy, robot-manipulation, vla, cascaded-explicit]
zotero_collection: 3-Robotics/WAM-Survey
image_source: online
arxiv_html: https://arxiv.org/abs/2604.04502
created: 2026-05-20
---

# Paper Note: Veo-Act: How Far Can Frontier Video Models Advance Generalizable Robot Manipulation?

## Metadata

| Item | Content |
|------|---------|
| Institution | Tsinghua University |
| Date | April 2026 |
| Project Page | N/A |
| Baselines | π₀.₅ |
| Links | [arXiv](https://arxiv.org/abs/2604.04502) / Code: N/A |

---

## One-Sentence Summary

> Veo-Act uses Veo-3 (frontier internet video model) as a high-level visual planner to generate task-level trajectory videos, then extracts actions via a multi-head inverse dynamics model (DINOv2 ViT-B/16 + dual MLP heads for action + interaction gate) and delegates contact-rich interactions to a VLA policy (π₀.₅) — achieving 3.2× improvement over π₀.₅ across semantic generalization tasks.

---

## Core Contributions

1. **Frontier Video Model as Visual Planner**: First systematic study of using Veo-3 (Google's state-of-the-art internet video generation model) as a robot manipulation planner — finding it provides correct task-level semantics but lacks precision for low-level control.
2. **Multi-Head Inverse Dynamics Model (IDM)**: DINOv2 ViT-B/16 encoder + two separate MLP heads jointly predicting action $a_t$ and interaction gate signal $G_t \in [0,1]$ — the gate signal identifies contact-rich phases requiring a more precise VLA policy.
3. **Hierarchical Planning + VLA Executor**: Veo-3 video plan provides instruction-following guidance; when $G_t > \tau$ (contact phase detected), control switches to fine-tuned π₀.₅ VLA — combining global semantic planning with precise local manipulation.
4. **Comprehensive Semantic Generalization Evaluation**: Four evaluation settings specifically targeting scenarios where language-only policies (π₀.₅) fail: wrist-camera invisible, similar-object distractors, pass-by interaction, and richer semantics.

---

## Problem Background

### Problem to Solve
How can frontier internet video models (Veo-3) advance robot manipulation? Specifically, can video generation provide semantic grounding that language-conditioned VLAs lack — particularly for tasks requiring instruction following in the presence of distractors or occlusions?

### Limitations of Existing Methods
- Standard VLAs (π₀.₅): achieve only 45–56% instruction-following in scenarios with similar-object distractors or occlusions; cannot leverage internet-scale visual knowledge.
- UniPi/AVDC: use weaker video models; inverse dynamics without gate signal cannot distinguish contact-rich phases.
- Pure IDM approaches: insufficient for precise contact interactions (grasping) — video generation lacks sub-centimeter accuracy.

### Motivation
Frontier video models (Veo-3) have consumed vast internet video data and understand task semantics, object appearances, and rough motion trajectories far better than robotics-specific models. However, they lack the precision for direct motor control. A hierarchical architecture that uses Veo-3 for "what to do and where to go" while delegating "how to grasp" to a fine-tuned VLA combines the strengths of both.

---

## Method Details

### Model Architecture

Veo-Act is a **hierarchical system**:

1. **Veo-3 Video Generator**: Generates future frame sequence $\{I_1^*, I_2^*, \ldots, I_n^*\}$ conditioned on initial observation $I_0$ and task instruction $l$
2. **Multi-Head IDM**: Extracts action $a_t$ and interaction gate $G_t$ from consecutive video frame pairs
3. **Action Smoothing**: Temporal filtering with keypoint preservation + window averaging + safety bounds
4. **VLA Switch (π₀.₅)**: Takes over execution when $G_t > \tau$ (contact-rich phase); resumes planned actions when interaction completes

The Veo-Act framework operates as follows: Veo-3 generates a task-level video trajectory from the scene image + instruction. The IDM extracts actions and the gate signal from consecutive frame pairs. When the gate exceeds threshold $\tau$, π₀.₅ VLA takes over for the contact phase and the system returns to the video plan after contact completes.

### Core Modules

#### Module 1: Veo-3 as Visual Planner

**Design Motivation**: Veo-3's internet-scale training gives it strong semantic understanding of object identities, task structure, and spatial relationships — exactly what's needed for instruction following in cluttered scenes.

**Implementation**:
- Input: initial scene image $I_0$ + language instruction $l$
- Output: video sequence $I_{0:n}^* = \{I_1^*, \ldots, I_n^*\}$
- Not fine-tuned on robot data — used zero-shot/few-shot
- Generates approximate task-level trajectory (correct semantics, approximate geometry)

#### Module 2: Multi-Head Inverse Dynamics Model

**Design Motivation**: Standard single-head IDM cannot distinguish smooth reaching (where video-derived actions work) from contact-rich phases (where fine-grained control is needed). Dual-head design explicitly models both.

**Architecture**:
- Visual encoder: DINOv2 ViT-B/16 (pretrained, fine-tuned at lr=5e-5)
- Input: consecutive frame pairs $(I_t^*, I_{t+1}^*)$ from generated video
- Head 1 (Action): MLP → robot joint action $a_t$
- Head 2 (Gate): MLP → scalar gate $G_t \in [0, 1]$ indicating contact/interaction probability
- The two heads are trained jointly with separate loss terms and weighting coefficients:

$$\mathcal{L} = \lambda_{act} \cdot \mathcal{L}_{act}(a_t, \hat{a}_t) + \lambda_{gate} \cdot \mathcal{L}_{gate}(G_t, \hat{g}_t)$$

where $a_t$ is the predicted robot action (joint angles), $\hat{a}_t$ is the ground-truth action, $G_t \in [0,1]$ is the predicted interaction gate (high = contact phase), and $\hat{g}_t$ is the ground-truth gate label.

#### Module 3: Action Smoothing

**Design Motivation**: Raw IDM predictions are noisy; robot execution requires smooth, safe trajectories.

**Implementation**:
- Temporal filtering: window averaging to smooth frame-to-frame variations
- Keypoint preservation: retains critical action waypoints (e.g., grasp poses)
- Safety bounds: enforces joint limits and workspace constraints on critical action dimensions

#### Module 4: Hierarchical Execution with VLA Switch

**Design Motivation**: Video generation produces correct high-level semantics but lacks sub-centimeter grasp precision. A VLA fine-tuned on robot data handles contact-rich phases better.

**Implementation**:
- Maintain action queue from video-derived IDM predictions
- Execute from queue while $G_t \leq \tau$ (reaching/positioning phase)
- Switch to π₀.₅ VLA policy when $G_t > \tau$ (interaction/contact detected)
- Resume queued actions when VLA signals interaction completion
- π₀.₅ fine-tuned: batch size 32, lr=2.5e-5, 40,000 iterations, Smooth L1 loss ($\beta=0.1$)

Two task-success metrics are used in evaluation. Instruction-following success requires the robot to reach within distance $\tau_{ins}$ of the target object:

$$\text{Succ}_{ins} := \mathbb{1}\left[\min_t d_{ins}^t \leq \tau_{ins}\right] = 1$$

Overall task success additionally requires the target object to be nearly stationary and at the goal location:

$$\text{Succ}_{task} := \mathbb{1}\left[\exists t: \|v_{obj}^t\|_2 \leq \tau_{static} \wedge d_{task}^t \leq \tau_{task}\right] = 1$$

---

## Experiments

### Datasets

| Dataset | Scale | Characteristics | Usage |
|---------|-------|----------------|-------|
| Simulation play data | 300k frame pairs | Random play + grasping trajectories | IDM training |
| Simulation random motion | 100k frame pairs | Additional motion diversity | IDM training |
| Real-world data | 150k samples | Cross-domain for IDM generalization | IDM training |
| **Total IDM** | **550k samples** | Cross-domain | IDM training |
| π₀.₅ fine-tuning data | Not specified | Task-specific robot demonstrations | VLA fine-tuning |

### Results

The four evaluation settings specifically target scenarios where language-only policies fail: (1) wrist-camera invisible — target occluded from wrist view; (2) similar-object distractors — multiple visually similar objects; (3) pass-by — obstacle along approach path; (4) richer semantics — compositional instructions.

**Table 1: Simulation Results (30 trials per setting)**

| Setting | π₀.₅ Instruct. | π₀.₅ Overall | **Veo-Act Instruct.** | **Veo-Act Overall** |
|---------|----------------|-------------|----------------------|-------------------|
| Similar-object (Experimental) | 0.47 | 0.40 | **0.93** | **0.93** |
| Pass-by (Experimental) | 0.03 | 0.00 | **0.50** | **0.47** |

Veo-Act dramatically improves over π₀.₅ on instruction following (0.47→0.93 similar objects; 0.03→0.50 pass-by) and overall task success. Pass-by is especially challenging — π₀.₅ fails at 0% overall, Veo-Act achieves 47%.

**Table 2: Real Robot Results**

| Setting | π₀.₅ Instruct. | Veo-Act Instruct. | π₀.₅ Overall | Veo-Act Overall |
|---------|----------------|------------------|-------------|----------------|
| Similar-object | 0.56 | **0.94** | — | — |
| Pass-by | 0.23 | **0.92** | — | — |
| Richer semantics | 0.21 | **0.95** | — | — |

The ~3.2× overall improvement across three settings on the real robot is driven by Veo-3's internet-scale visual knowledge, which enables robust generalization to semantic challenges that fine-tuned π₀.₅ fails on.

The following ablation isolates the contribution of each component in the wrist-camera-invisible setting.

**Table 3: Ablation Study (Wrist-camera invisible setting)**

| Variant | Instruct.-follow | Overall |
|---------|-----------------|---------|
| ResNet backbone (vs. DINOv2) | 0.73 | 0.57 |
| Without noise augmentation | 0.73 | 0.53 |
| Single-head IDM | 0.83 | 0.57 |
| **Full Veo-Act** | **0.83** | **0.67** |

DINOv2 outperforms ResNet backbone. Noise augmentation improves robustness (+0.14 overall). Dual-head IDM with gate signal improves overall task success (+0.10 vs. single-head).

### Implementation Details

- **Video Generator**: Veo-3 (Google; not fine-tuned on robot data — zero-shot/few-shot)
- **IDM Visual Encoder**: DINOv2 ViT-B/16 (fine-tuned at lr=5e-5)
- **IDM Optimizer**: AdamW ($\beta=(0.9, 0.999)$)
- **IDM Other Modules LR**: 5e-4
- **IDM LR Scheduler**: Cosine with 8,500 warmup steps
- **IDM Training**: 85,000 iterations, ~10 hours on 4× Ampere 80GB GPUs; batch size 32 (8/GPU)
- **VLA (π₀.₅) Fine-tuning**: 40,000 iterations, batch size 32, lr=2.5e-5, Smooth L1 loss ($\beta=0.1$)
- **Robot**: 7-DoF arm + 12-DoF dexterous hand; global RGB camera + wrist RGB camera; state dimension 21
- **Evaluation**: 30 trials per simulation setting; 13–19 trials per real-robot setting

---

## Critical Analysis

### Strengths
1. First systematic evaluation of a frontier internet video model (Veo-3) for robot manipulation — provides important empirical insights about the strengths/limitations of large video models.
2. Interaction gate signal is an elegant solution: IDM autonomously discovers contact phases without explicit labeling.
3. 3.2× improvement on semantic generalization tasks demonstrates clear value of internet video knowledge.

### Limitations
1. **Veo-3 generation fidelity**: Small distribution shifts in scene appearance cause substantial prediction differences — sensitivity to domain gap.
2. **Contact-rich task limitation**: Video models generate geometrically approximate trajectories; insufficient for precise grasping without VLA fallback.
3. **Proprietary dependency**: Relies on Veo-3 (Google closed-source API) — limits reproducibility and deployment flexibility.
4. **Latency**: Video generation at inference is slow (not quantified precisely).

### Potential Improvements
1. Use open-source frontier video models (Wan2.2, HunyuanVideo) as alternatives to Veo-3.
2. Fine-tune video model on robot-specific data for improved geometric precision.
3. Replace binary gate with continuous confidence score for smoother VLA handoff.

### Reproducibility
- [ ] Code open-sourced
- [ ] Pretrained models available (Veo-3 is proprietary)
- [x] Training details complete (IDM + VLA hyperparameters)
- [ ] Full dataset specification not provided

---

## Related Notes

### Based On
- Veo-3: Google's frontier video generation model used as visual planner
- DINOv2: Self-supervised ViT encoder for IDM feature extraction
- π₀: VLA model (π₀.₅) used as contact-phase executor

### Compared Against
- π₀: State-of-the-art VLA baseline; outperformed 3.2× on semantic generalization

### Method Related
- Inverse Dynamics Model: Action extraction from video frame pairs
- Hierarchical Policy: Two-level control (video planner + VLA executor)
- Interaction Gate: Novel binary signal distinguishing reaching vs. contact phases

---

## Quick Reference Card

> [!summary] Veo-Act (arXiv 2026)
> - **Core**: Veo-3 internet video model as semantic visual planner + DINOv2 multi-head IDM (action + gate signal) + π₀.₅ VLA for contact phases
> - **Method**: Zero-shot Veo-3 for task trajectories; DINOv2 ViT-B/16 IDM trained on 550k samples (85K iters, 4× A100); interaction gate $G_t$ triggers VLA switch
> - **Results**: 0.47→0.93 instruction-following (similar objects), 0.03→0.50 (pass-by); ~3.2× improvement over π₀.₅ on real robot
> - **Code**: N/A

---

*Note created: 2026-05-20*
