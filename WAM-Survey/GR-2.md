---
title: "GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge for Robot Manipulation"
method_name: "GR-2"
authors: [Chi-Lam Cheang, Guangzeng Chen, Ya Jing, Tao Kong, Hang Li, Yifeng Li, Yuxiao Liu, Hongtao Wu, Jiafeng Xu, Yichu Yang, Hanbo Zhang, Minzhao Zhu]
year: 2024
venue: arXiv 2024
tags: [video-generation, joint-autoregressive, gpt-style, vqgan, calvin, robot-policy, scaling, web-scale]
zotero_collection: 3-Robotics/WAM-Survey
image_source: online
arxiv_html: https://arxiv.org/abs/2410.06158
created: 2026-05-20
---

# Paper Note: GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge for Robot Manipulation

## Metadata

| Item | Content |
|------|---------|
| Institution | ByteDance Research |
| Date | October 2024 |
| Project Page | N/A |
| Baselines | GR-1, RT-1, MT-ACT, HULC, RoboFlamingo |
| Links | [arXiv](https://arxiv.org/abs/2410.06158) / Code: N/A |

---

## One-Sentence Summary

> GR-2 scales GR-1 to web-scale pretraining (38M video clips from 6 datasets, 50B+ tokens) with VQGAN discrete tokenization + GPT-style transformer (GR-2-B: 95M trainable params), achieving 97.7% success on 105 robot tasks, 4.64 CALVIN avg. task length, and strong generalization to unseen environments (71.7%→87% with data augmentation).

---

## Core Contributions

1. **Web-Scale Video Pre-training**: 38M text-video clips from Howto100M + Ego4D + Something-Something V2 + EPIC-KITCHENS + Kinetics-700 + RT-1 + Bridge — orders of magnitude larger than GR-1's Ego4D-only pre-training.
2. **VQGAN Discrete Image Tokenization**: Replaces ViT continuous embeddings with VQGAN discrete tokens — enables unified autoregressive prediction of image tokens and action tokens in the same vocabulary space.
3. **Scalable Architecture**: 4 model sizes (GR-2-S: 30M, GR-2-B: 95M, GR-2-L: 312M, GR-2-XL: 719M) — success rate scales consistently with model size.
4. **Multi-Task and Bin-Picking Generalization**: 97.7% on 105 manipulation tasks; 80% on unseen bin-picking objects (122 total) — demonstrates broad task generalization at scale.

---

## Problem Background

### Problem to Solve
GR-1 demonstrated that video pre-training improves manipulation, but used only Ego4D (800K clips). Can scaling pre-training to web-scale video (38M clips, 50B+ tokens) substantially improve manipulation capability across 100+ tasks and unseen generalization scenarios?

### Limitations of GR-1
- Ego4D only (800K clips) — limited coverage of physical dynamics.
- Continuous ViT embeddings — not unified with token prediction.
- 4.21 CALVIN avg. — GR-2 achieves 4.64.
- 79% → 33.3% on bin-picking unseen (vs. GR-2: 80%/77% seen/unseen cluttered).

### Motivation
Scaling video pre-training data to web scale provides far broader physical dynamics priors — more diverse object interactions, hand-object relationships, and task semantics. VQGAN tokenization creates a unified discrete token space for images and actions, enabling standard autoregressive sequence modeling.

---

## Method Details

### Model Architecture

GR-2 is a **GPT-style transformer** with discrete image tokenization:

1. **Language**: CLIP text encoder (frozen) → language tokens
2. **Images**: VQGAN → discrete image tokens (unified token space)
3. **Robot State**: Linear layers → state tokens
4. **GPT Transformer**: Autoregressive causal transformer
5. **Action**: cVAE → action trajectory prediction

**Model Sizes**:
| Variant | Trainable Parameters |
|---------|---------------------|
| GR-2-S | 30M |
| GR-2-B (default) | 95M |
| GR-2-L | 312M |
| GR-2-XL | 719M |

### Core Modules

#### Module 1: VQGAN Discrete Image Tokenization

**Design Motivation**: Discrete tokens enable unified autoregressive prediction of images and actions without separate image decoder architectures. VQGAN provides compact, semantically meaningful image codes.

**Implementation**:
- VQGAN quantizes RGB images to discrete codebook tokens
- Image tokens processed in same vocabulary as other sequence tokens
- Enables joint next-token prediction for both visual and action outputs

#### Module 2: Two-Stage Training

**Stage 1 — Web-Scale Video Pre-training**:
- Dataset: 38M clips from Howto100M, Ego4D, SSv2, EPIC-KITCHENS, Kinetics-700, RT-1, Bridge
- Data processing: MediaPipe hand filtering + diffusion model re-captioning
- Objective: Predict subsequent video frames given text description + initial frame
- Scale: 50+ billion tokens

**Stage 2 — Robot Data Fine-tuning**:
- Multi-task: ~40K trajectories across 105 tasks
- Bin-picking: ~94K pick-and-place trajectories (55 objects)
- Data augmentation: Diffusion model for object insertion + SAM for background removal
- Output: Future video frames + action trajectories simultaneously

#### Module 3: Policy and Action Generation

**Implementation**:

GR-2 jointly predicts the next observation and an action trajectory from the current context. The policy function is:

$$\pi(l, o_{t-h:t}, s_{t-h:t}) \to o_{t+1}, a_{t:t+k}$$

where $l$ is the language instruction, $o_{t-h:t}$ is the observation history, $s_{t-h:t}$ is the robot state history, $o_{t+1}$ is the predicted next observation, and $a_{t:t+k}$ is the action trajectory generated by the cVAE. Predicting $o_{t+1}$ simultaneously with actions means video prediction remains a continuous auxiliary objective during fine-tuning — consistent with GR-1's design principle.

---

## Experiments

### Datasets

| Dataset | Scale | Characteristics | Usage |
|---------|-------|----------------|-------|
| Howto100M + Ego4D + SSv2 + EPIC + Kinetics + RT-1 + Bridge | 38M clips, 50B+ tokens | Diverse human+robot video | Stage 1 pre-training |
| Multi-task robot demos | ~40K trajectories, 105 tasks | Kinova Gen3 manipulation | Stage 2 fine-tuning |
| Bin-picking | ~94K trajectories, 55 objects | Pick-and-place | Stage 2 fine-tuning |

### Multi-Task Learning (105 Tasks)

GR-2 is evaluated across five difficulty levels for multi-task manipulation on a Kinova Gen3 robot:

| Setting | GR-2 | GR-1 |
|---------|-------|------|
| Simple | **97.7%** | — |
| Distractor | **90.5%** | — |
| Unseen Backgrounds | **71.4%** | — |
| Unseen Environments | **71.7%** (87.0% with augmentation) | — |
| Low-data (50 traj/task) | **73.9%** | — |

### Bin-Picking (122 Objects)

GR-2 dramatically outperforms GR-1 on bin-picking, where web-scale pretraining provides far richer object diversity priors:

| Setting | GR-2 | GR-1 |
|---------|------|------|
| Seen | ~90% | ~33% |
| Unseen | ~80% | — |
| Cluttered Seen | ~78% | — |
| Cluttered Unseen | ~77% | — |

### CALVIN Benchmark

| Method | Avg. Task Length | 1-Task SR | 5-Task SR |
|--------|-----------------|-----------|-----------|
| GR-1 | 4.21 | 94.9% | — |
| **GR-2** | **4.64** | **98.6%** | **85.9%** |

+0.43 avg. task length over GR-1; 5-task SR improved substantially.

### Scaling Results

Success rate scales consistently with model size across all datasets — both validation loss and task success rate improve monotonically with scale:

| Model | Trainable Params | Trend |
|-------|-----------------|-------|
| GR-2-S | 30M | Lower |
| GR-2-B | 95M | Baseline |
| GR-2-L | 312M | Better |
| GR-2-XL | 719M | Best |

### Implementation Details

- **Base**: GPT-style transformer; VQGAN discrete tokenization; CLIP text encoder (frozen); cVAE action head
- **Model Sizes**: 30M / 95M / 312M / 719M trainable params
- **Stage 1**: 38M clips, 50B+ tokens; MediaPipe hand filtering + diffusion re-captioning
- **Stage 2**: ~134K trajectories; diffusion model data augmentation + SAM background removal
- **Robot**: Kinova Gen3 7-DoF + Robotiq 2F-85; dual cameras; WBC at 200 Hz

---

## Critical Analysis

### Strengths
1. Web-scale pre-training (38M clips) provides substantially richer physical dynamics priors than Ego4D alone.
2. Consistent model scaling — 4 sizes from 30M to 719M with monotonic improvement.
3. 97.7% on 105 tasks demonstrates broad multi-task capability.

### Limitations
1. **Training details not fully disclosed**: Hardware, learning rates, batch sizes not specified in paper.
2. **Proprietary robot platform**: Kinova Gen3 — different from common Franka Panda benchmarks.
3. **VQGAN tokenization loss**: Discrete tokenization may lose fine-grained spatial detail for precision tasks.

### Reproducibility
- [ ] Code open-sourced
- [ ] Pretrained models available
- [ ] Training hyperparameters not fully specified
- [x] Architecture described (model sizes, components)

---

## Related Notes

### Based On
- [[GR-1]]: Predecessor architecture; GR-2 scales and improves
- [[VQGAN]]: Discrete image tokenization backbone
- [[CLIP]]: Frozen language encoder

### Compared Against
- [[GR-1]]: Prior version; outperformed 4.21→4.64 CALVIN, 33%→90% bin-picking
- [[HULC]]: Hierarchical policy; outperformed on CALVIN

### Method Related
- [[Web-Scale Video Pre-training]]: 38M clip pretraining for physical dynamics
- [[Scaling Laws]]: Consistent performance gains with model size
- [[Data Augmentation]]: Diffusion model + SAM for unseen environment generalization

---

## Quick Reference Card

> [!summary] GR-2 (arXiv 2024)
> - **Core**: GPT-style transformer (GR-2-B: 95M trainable, 4 sizes 30M–719M); VQGAN discrete tokenization; CLIP language; cVAE action head; 38M clip web-scale pre-training; joint frame+action prediction
> - **Method**: Stage 1: 38M clips (Howto100M+Ego4D+SSv2+EPIC+Kinetics+RT-1+Bridge), 50B+ tokens; Stage 2: ~134K robot trajectories; diffusion augmentation; WBC 200Hz
> - **Results**: 97.7% 105-task multi-task; 4.64 CALVIN avg (vs. 4.21 GR-1); 80% unseen bin-picking; 87% unseen env (with augmentation)
> - **Code**: N/A

---

*Note created: 2026-05-20*
