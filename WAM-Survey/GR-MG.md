---
title: "GR-MG: Leveraging Partially-Annotated Data via Multi-Modal Goal-Conditioned Policy"
method_name: "GR-MG"
authors: [Peiyan Li, Hongtao Wu, Yan Huang, Chilam Cheang, Liang Wang, Tao Kong]
year: 2024
venue: arXiv 2024
tags: [video-generation, goal-conditioned, joint-autoregressive, gpt-style, calvin, partially-annotated, multi-modal]
zotero_collection: 3-Robotics/WAM-Survey
image_source: online
arxiv_html: https://arxiv.org/abs/2408.14368
created: 2026-05-20
---

# Paper Note: GR-MG: Leveraging Partially-Annotated Data via Multi-Modal Goal-Conditioned Policy

## Metadata

| Item | Content |
|------|---------|
| Institution | Chinese Academy of Sciences (NLPR), ByteDance Research |
| Date | August 2024 |
| Project Page | N/A |
| Baselines | GR-1, HULC, MDT, SPIL, RoboFlamingo, SuSIE, 3D Diffusion Actor, OpenVLA, Octo |
| Links | [arXiv](https://arxiv.org/abs/2408.14368) / Code: N/A |

---

## One-Sentence Summary

> GR-MG leverages three types of partially-annotated data via a progress-guided goal image generator (InstructPix2Pix, 10-bin task progress) and a GPT-style multi-modal goal-conditioned policy (text + goal image + observations → cVAE actions), achieving 4.04 CALVIN avg. task length and 78.1% real-robot success on 58 tasks.

---

## Core Contributions

1. **Three-Type Partially-Annotated Data Utilization**: Trains on fully-annotated trajectories (text + action), text-only videos (no actions), and action-only trajectories (no text) — dramatically expands usable training data without requiring complete annotation.
2. **Progress-Guided Goal Image Generation**: InstructPix2Pix generates goal images conditioned on current observation + text instruction + discretized task progress (10 bins, 0–100%) — progress conditioning provides stronger geometric guidance for goal image quality.
3. **Multi-Modal Goal-Conditioned Policy**: GPT-style transformer conditioned on both text instruction and generated goal image — [PROG] + [OBS] + [ACT] token structure; cVAE generates action trajectories.
4. **Few-Shot Generalization**: 10-shot: 17.5% (vs. 2.5% GR-1); 30-shot: 37.5% (vs. 22.5%) — goal image conditioning provides spatial anchor for few-shot learning.

---

## Problem Background

### Problem to Solve
Obtaining robot trajectories fully annotated with both language instructions and action labels is expensive. Much existing data is only partially annotated: videos with text but no actions, or trajectories with actions but no text. How can partially-annotated data be used effectively for training language-conditioned robot policies?

### Limitations of Existing Methods
- GR-1: requires fully annotated data (language + action labels); limited few-shot generalization.
- SuSIE: diffusion-based goal generation without task progress conditioning.
- Language-only VLAs: cannot leverage action-only trajectory data.

### Motivation
Goal images provide an intermediate visual plan that: (1) can be generated from text-only videos (no actions needed), (2) conditions the policy with spatial/geometric information, (3) enables few-shot learning by providing a concrete target configuration. Progress conditioning improves goal image quality by specifying which point in the trajectory to generate.

---

## Method Details

### Model Architecture

GR-MG has **two main components**:

1. **Progress-Guided Goal Image Generation Model**
2. **Multi-Modal Goal-Conditioned Policy**

### Core Modules

#### Module 1: Progress-Guided Goal Image Generation

**Design Motivation**: Goal images at a fixed future offset may be inconsistent with trajectory length. Progress-based conditioning (0–100%) enables sampling goal images at any relative position in the trajectory, adaptable to both short and long tasks.

**Implementation**:
- Base model: InstructPix2Pix (diffusion-based image editing)
- Inputs: current observation + text instruction + task progress
- Text encoding: T5-Base

Task progress is discretized into 10 uniformly spaced bins:

$$\text{progress} \in \{0.0, 0.1, 0.2, \ldots, 1.0\}$$

This discrete progress value is injected into InstructPix2Pix alongside the text instruction to specify which point in the task trajectory to generate as the goal image — enabling the generator to produce geometrically appropriate goals for tasks of varying length.

- Output: goal image $g_t$ representing predicted future scene state
- Training: on text-annotated videos (with or without action labels)
- Goal image sampling: N steps ahead in trajectory

#### Module 2: Multi-Modal Goal-Conditioned Policy

**Design Motivation**: Conditioning on both text (semantic) and goal image (geometric/spatial) provides complementary information — text specifies what to do; goal image specifies where to end up.

The policy predicts robot actions from the full multi-modal context. The action at each timestep is:

$$a_t = \pi(l, g_t, o_{t-h:t}, s_{t-h:t})$$

where $l$ is the language instruction, $g_t$ is the goal image from the progress-guided generator, $o_{t-h:t}$ is the observation history, and $s_{t-h:t}$ is the robot state history. The goal image provides a concrete spatial target that complements the semantic information in the language instruction.

**Architecture** (GPT-style, same backbone as GR-1):
- Input: text instruction + current observation sequence + robot state + goal image
- Token types: [PROG] (task progress), [OBS] (visual observations), [ACT] (action targets)
- Goal image: encoded via MAE tokenization
- Action generation: cVAE generates action trajectory $a_{t:t+k}$

**Training Strategy**:
- Stage 1: Train goal generator on text-annotated videos
- Stage 2: Train policy on action-only trajectories (null string conditioning)
- Stage 3: Fine-tune on fully-annotated trajectories

**Initialization**: Policy weights initialized from Ego4D video pre-training (GR-1 backbone)

---

## Experiments

### Datasets

| Dataset | Type | Characteristics | Usage |
|---------|------|----------------|-------|
| CALVIN | Fully-annotated | ~18K trajectories, 3 envs | All 3 training stages |
| Something-Something V2 | Text+video, no actions | Internet manipulation | Goal generator training |
| RT-1 | Action-only, no text | Robot manipulation | Policy stage 2 training |
| Real-robot | Fully-annotated | 18K demos, 37 tasks | Real-robot fine-tuning |

### CALVIN ABC→D Benchmark

GR-MG achieves the highest 5-task completion rate among compared methods, showing that the goal image provides meaningful spatial guidance for sustained long-horizon execution:

| Method | 1-Task SR | 5-Task SR | Avg. Length |
|--------|-----------|-----------|-------------|
| GR-1 | 94.9% | — | 4.21 |
| 3D Diffusion Actor | 93.8% | 41.2% | 3.35 |
| **GR-MG** | **96.8%** | **64.4%** | **4.04** |
| GR-MG (10% data) | — | — | 3.11 |

4.04 avg. task length (vs. 3.35 prior SOTA); +23.2% on 5-task completion vs. 3D Diffusion Actor; 10% data still achieves 3.11 (vs. GR-1 10%: 1.41).

### Real-Robot Results (58 Tasks, Kinova Gen3)

On real robot evaluation, goal image conditioning provides substantial spatial guidance — particularly for generalization beyond the training distribution:

| Setting | GR-MG | GR-1 |
|---------|-------|------|
| Simple | **78.1%** | 68.7% |
| Generalization | **60.6%** | 44.4% |

+9.4% simple, +16.2% generalization over GR-1.

### Few-Shot Learning (Kinova Gen3)

The goal image acts as a spatial anchor that is especially critical when labeled demonstration data is scarce:

| Data | GR-MG | GR-1 |
|------|-------|------|
| 10-shot | **17.5%** | 2.5% |
| 30-shot | **37.5%** | 22.5% |

Goal image conditioning enables 7× improvement in 10-shot learning over GR-1.

### Multi-Modal Ablation

The ablation below confirms that combining text and goal image conditioning outperforms either modality alone, and that progress conditioning contributes measurably to goal image quality:

| Configuration | Performance |
|--------------|------------|
| Text only | Lower |
| Goal image only | Lower |
| **Text + Goal image (GR-MG)** | **Best** |
| Without progress | Lower SSIM/MSE |

### Implementation Details

- **Policy Backbone**: GPT-style transformer (GR-1 architecture), initialized from Ego4D weights
- **Goal Generator**: InstructPix2Pix + T5-Base; 10-bin progress discretization
- **Goal Image Encoding**: MAE tokenization
- **Action Head**: cVAE (conditional VAE)
- **Robot (Simulation)**: Franka Emika Panda, parallel-jaw gripper
- **Robot (Real)**: Kinova Gen-3 + Robotiq 2F-85; static + wrist cameras
- Training hyperparameters not specified in paper

---

## Critical Analysis

### Strengths
1. Three-type data utilization dramatically expands usable training data without requiring full annotation.
2. Progress-guided goal image generation provides flexible spatial planning across varying trajectory lengths.
3. 7× improvement in 10-shot learning demonstrates strong few-shot generalization from goal image conditioning.

### Limitations
1. **InstructPix2Pix quality ceiling**: Goal image quality directly limits policy performance — diffusion editing may produce geometrically inaccurate goals.
2. **Progress conditioning approximation**: 10-bin discretization is coarse — continuous progress may be more precise.
3. **Training hyperparameters not disclosed**: Reproducibility limited by missing LR, batch size, epoch information.

### Potential Improvements
1. Continuous progress conditioning for finer goal image control.
2. Closed-loop goal image regeneration mid-task for dynamic replanning.
3. Higher-quality goal image generation via larger diffusion models.

### Reproducibility
- [ ] Code open-sourced
- [ ] Pretrained models available
- [ ] Training hyperparameters not disclosed
- [x] CALVIN, RT-1 datasets publicly available

---

## Related Notes

### Based On
- [GR-1](GR-1.md): Policy backbone architecture and Ego4D pre-training
- InstructPix2Pix: Goal image generation via text-guided image editing
- T5: Text encoder for goal generator conditioning

### Compared Against
- [GR-1](GR-1.md): Prior version; outperformed 4.21→4.04 (slight drop but better 5-task SR)
- 3D Diffusion Actor: Prior CALVIN SOTA; outperformed 3.35→4.04
- SuSIE: Goal image generation baseline

### Method Related
- Goal-Conditioned Policy: Future state image as visual plan
- Partially-Annotated Data: Leveraging incomplete annotation data
- Few-Shot Robot Learning: Goal image as spatial anchor for minimal-data learning

---

## Quick Reference Card

> [!summary] GR-MG (arXiv 2024)
> - **Core**: Progress-guided goal generator (InstructPix2Pix + T5, 10-bin progress, text+action-free video training) → goal image $g_t$ → GPT-style transformer policy (text + goal image + obs → cVAE actions); 3-type partial annotation utilization
> - **Method**: GR-1 backbone + Ego4D init; 3-stage training (goal gen → action-only → fully-annotated); CALVIN 18K + SSv2 + RT-1; Kinova Gen3
> - **Results**: 4.04 CALVIN avg (10% data: 3.11 vs GR-1 10%: 1.41); 78.1% real-robot simple; 10-shot 17.5% (vs. 2.5% GR-1)
> - **Code**: N/A

---

*Note created: 2026-05-20*
