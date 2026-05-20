---
title: "Video Generators are Robot Policies"
method_name: "VideoPolicy"
authors: [Junbang Liang, Pavel Tokmakov, Ruoshi Liu, Sruthi Sudhakar, Paarth Shah, Rares Ambrus, Carl Vondrick]
year: 2025
venue: arXiv 2025
tags: [video-generation, cascaded-implicit, stable-video-diffusion, robot-policy, two-stage, libero, robocasa]
zotero_collection: 3-Robotics/WAM-Survey
image_source: online
arxiv_html: https://arxiv.org/abs/2508.00795
created: 2026-05-20
---

# Paper Note: Video Generators are Robot Policies (VideoPolicy)

## Metadata

| Item | Content |
|------|---------|
| Institution | Columbia University, Toyota Research Institute |
| Date | August 2025 |
| Project Page | [videopolicy.cs.columbia.edu](https://videopolicy.cs.columbia.edu) |
| Baselines | Diffusion Policy, UVA, GR00T, π₀, OpenVLA, UniPi |
| Links | [arXiv](https://arxiv.org/abs/2508.00795) / Code: N/A |

---

## One-Sentence Summary

> VideoPolicy demonstrates that video generation serves as an effective proxy for robot policy learning via a two-stage approach — fine-tune SVD on robot demonstrations (video U-Net $\mu_\theta$), then train a 1D action U-Net ($\alpha_\theta$) conditioned on 5 intermediate video decoder features via stop-gradient — achieving 94% on LIBERO-10 and 63% on RoboCasa with only 50 demonstrations.

---

## Core Contributions

1. **Video Generators as Policy Backbone**: Demonstrates that fine-tuning SVD on robot demonstrations and extracting intermediate video U-Net features provides better policy representations than training directly on actions — 2-stage (63%) substantially outperforms joint end-to-end training (57%) and no video fine-tuning (9%).
2. **Two-Stage Decoupled Training**: Stage 1 fine-tunes SVD on robot video; Stage 2 trains 1D action U-Net on frozen video features with stop-gradient — prevents action loss from disrupting video generation quality.
3. **5-Layer Intermediate Feature Extraction**: Action U-Net conditioned on spatiotemporal embeddings from 5 SVD decoder layers (layers 9, 14, 17, 20, 23) via CNN adapters — captures multi-scale video generation features.
4. **Action-Free Pre-Training**: Model trained with video-only supervision on 12/24 tasks achieves 60%+ on unseen tasks — video generation pre-training provides generalizable environmental dynamics without action labels.

---

## Problem Background

### Problem to Solve
Robot policies require demonstrations with action labels; generalization is limited. Video generation models trained on internet-scale data encode rich environment dynamics. Can video generation serve as a proxy for policy learning — improving generalization and reducing action-label requirements?

### Limitations of Existing Methods
- Standard Diffusion Policy: action-only training; no video generation; limited to demonstration distribution.
- UniPi: generates video then applies IDM; two-stage error propagation; 9 seconds inference.
- π₀: VLA; 85% LIBERO-10 — VideoPolicy achieves 94%.
- OpenVLA: 7B; 54% LIBERO-10; requires large labeled datasets.
- Joint video+action training: 57% (vs. 2-stage 63%) — shared gradients hurt video quality.

### Motivation
The video generative model serves as the primary policy learner — it encodes not just what to do, but the rich environmental dynamics needed to generalize. The action decoder is a lightweight interface that reads these dynamics without corrupting them (stop-gradient). This separation of concerns enables video pre-training on action-free data to transfer to novel tasks.

---

## Method Details

### Model Architecture

VideoPolicy has **two U-Net components**:

1. **Video U-Net** $\mu_\theta$: SVD-based latent video diffusion — generates 25-frame multi-view video
2. **Action U-Net** $\alpha_\theta$: 1D CNN (Diffusion Policy style) — generates action sequence conditioned on frozen video features

### Core Modules

#### Module 1: Video U-Net (Stage 1 — Fine-tuning SVD)

**Design Motivation**: SVD encodes rich environment dynamics from internet-scale video pretraining. Fine-tuning on robot demonstrations adapts these priors to manipulation-specific visual dynamics.

**Implementation**:
- Base: Stable Video Diffusion (SVD), pretrained
- Input: initial frame $v_0$ + CLIP text conditioning $c$
- Output: 25-frame video (8 frames/camera × 3 cameras + 1 padding)
- Resolution: 256×256 (simulation), 448×320 (real-world)
- The video U-Net is trained with the following denoising loss — predicting noise $\epsilon$ from noisy latent $z_i$ given CLIP text embedding $\phi(c)$ and initial frame $z_{i,0}$:

$$\mathcal{L}_{\text{video}} = \mathbb{E}_{z_0, \epsilon, i}\left[\|\epsilon - \mu_\theta(z_i, i, \phi(c), z_{i,0})\|^2\right]$$

- Training: lr=1e-5, batch 32, 8×A100, ~2 weeks (RoboCasa: 368,866 steps)

#### Module 2: Feature Extraction (CNN Adapters)

**Design Motivation**: Different SVD decoder layers capture features at different abstraction levels — aggregating from multiple layers provides richer conditioning than a single layer.

**Implementation**:
- Extract spatiotemporal embeddings from 5 SVD decoder layers: {9, 14, 17, 20, 23}
- CNN adapter per layer: transforms spatiotemporal tensor → global conditioning vector $h_i$
- Stop-gradient: action loss does NOT backpropagate into video U-Net ($\mu_\theta$ frozen after Stage 1)

#### Module 3: Action U-Net (Stage 2 — Action Learning)

**Design Motivation**: 1D U-Net from Diffusion Policy efficiently generates action sequences; conditioning on frozen video features provides rich task and environment context.

**Implementation**:
- Architecture: 1D CNN adapted from Diffusion Policy
- Conditioned on: $h_i$ (aggregated video decoder features from all 5 layers)
- Input: noisy action sequence $a_i$ at denoising step $i$
- Output: clean action sequence $\{a_t\}$
- The action denoising objective predicts noise from the noisy action $a_i$ conditioned on the frozen video decoder features $h_i$:

$$\mathcal{L}_{\text{action}} = \mathbb{E}_{a_0, \epsilon, i}\left[\|\epsilon - \alpha_\theta(a_i, i, h_i)\|^2\right]$$

where gradient does not propagate to $\mu_\theta$.

The full video-conditioned action generation can be written as: video generation $f$ produces the frame sequence $\{v̂_t\} = f(v_0, c)$, and action generation $g$ operates on intermediate video features $\psi_i$ from generator layers as $\{a_t\} = g(\psi_0, \ldots, \psi_i)$ where $\psi_i = f_i(v_0, c)$ — the video generator itself is the policy backbone.

- Training: lr=5e-5, batch 32; video U-Net frozen (stop-gradient)

---

## Experiments

### Datasets

| Dataset | Scale | Characteristics | Usage |
|---------|-------|----------------|-------|
| RoboCasa | 50 demos/task, 24 tasks | Diverse kitchen simulation | Training/Evaluation |
| LIBERO-10 | 50 demos/task, 10 tasks | Tabletop simulation benchmark | Evaluation |
| Real-world | 200 demos/task | 3-camera Franka setup | Real-robot evaluation |

### Results

The key design choice — two-stage decoupled training vs. joint end-to-end training — is evaluated on RoboCasa. The following table shows that the stop-gradient separation is critical and that video fine-tuning is essential.

**Table 1: RoboCasa Benchmark (50 demonstrations, 24 tasks)**

| Method | Avg. Success Rate |
|--------|-----------------|
| **VideoPolicy (2-Stage)** | **63%** |
| VideoPolicy + MimicGen (300 demos) | 66% |
| VideoPolicy (joint training) | 57% |
| VideoPolicy (no video fine-tuning) | 9% |
| Diffusion Policy | Lower |
| UVA | Lower |
| GR00T (300 demos required) | Lower |

2-stage training (+6% over joint) is critical; video fine-tuning essential (+54% over no fine-tuning).

**Table 2: LIBERO-10 Benchmark (50 demonstrations)**

| Method | Avg. Success Rate |
|--------|-----------------|
| **VideoPolicy** | **94%** |
| π₀ | 85% |
| OpenVLA | 54% |
| Diffusion Policy | Lower |

94% on LIBERO-10 with only 50 demonstrations — outperforms π₀ (85%) and OpenVLA (54%).

**Table 3: Real-World Tasks (200 demonstrations)**

| Task | Success Rate |
|------|-------------|
| Open Drawer | 80–90% |
| Pick and Place | 80–100% |
| M&Ms to Cup (unseen objects) | 80–90% |
| M&Ms to Cup (unseen backgrounds) | 20% |

Strong generalization to unseen objects (80–90%) but limited robustness to background changes (20%) — suggests some background feature leakage.

The following table demonstrates that video generation pre-training on action-free data transfers to unseen tasks.

**Table 4: Action-Free Pre-Training (12/24 tasks video-only)**

| Configuration | Success on Unseen 12 Tasks |
|--------------|--------------------------|
| VideoPolicy (video pre-train on 12 tasks) | 60%+ |
| Baseline (no pre-training) | Near 0% |

Video generation pre-training on action-free data generalizes to 60%+ on unseen tasks — environmental dynamics learned from video transfer to robot control.

**Table 5: Video Horizon Ablation**

| Video Horizon | Success (distribution shift) |
|--------------|------------------------------|
| 0 steps | Lowest |
| 32 steps | Highest |

Longer video prediction horizons consistently improve success rate, especially under distribution shift — longer horizon encodes richer dynamics.

### Implementation Details

- **Video Model Base**: Stable Video Diffusion (SVD)
- **Video LR**: 1e-5; Action LR: 5e-5
- **Batch Size**: 32
- **Resolution**: 256×256 (sim), 448×320 (real)
- **Video Frames**: 25 (8/camera × 3 cameras + 1 padding)
- **Decoder Layers Used**: {9, 14, 17, 20, 23}
- **Denoising Steps**: 30 (inference, 9 seconds/video)
- **Hardware**: 8× A100 GPUs, ~2 weeks (RoboCasa)
- **Inference Time**: ~9 seconds per 25-frame video

---

## Critical Analysis

### Strengths
1. Stop-gradient design elegantly separates video learning from action learning — prevents mutual interference.
2. 94% LIBERO-10 and 63% RoboCasa with only 50 demonstrations — highly data efficient.
3. Action-free pre-training generalizes to unseen tasks — video dynamics transfer without action labels.

### Limitations
1. **9-second inference**: 30-step SVD denoising is far too slow for real-time control.
2. **Background sensitivity**: 20% success on unseen backgrounds in M&Ms task — not fully generalizable.
3. **Multi-camera requirement**: Needs 3 synchronized cameras (8 frames × 3 views) — complex setup.
4. **Video frame overhead**: 25-frame video generation even when only 1–2 frames are needed.

### Potential Improvements
1. Distillation to 1–2 denoising steps for real-time control.
2. Background augmentation during video fine-tuning.
3. Adaptive video horizon (shorter for simple tasks, longer for complex).

### Reproducibility
- [ ] Code open-sourced
- [x] Project page available (videopolicy.cs.columbia.edu)
- [x] Training details described (LR, batch size, hardware, steps)
- [x] Datasets accessible (LIBERO, RoboCasa public)

---

## Related Notes

### Based On
- [[Stable Video Diffusion]]: SVD fine-tuned as video U-Net policy backbone
- [[Diffusion Policy]]: 1D action U-Net architecture

### Compared Against
- [[π₀]]: VLA baseline; outperformed 85%→94% on LIBERO-10
- [[OpenVLA]]: 7B VLA; outperformed 54%→94% on LIBERO-10
- [[UniPi]]: Video+IDM; outperformed substantially

### Method Related
- [[Two-Stage Training]]: Decoupled video and action learning with stop-gradient
- [[Video Feature Extraction]]: Multi-layer decoder feature conditioning
- [[Action-Free Pre-Training]]: Video-only training as pre-training for unseen tasks

### Hardware/Data Related
- [[RoboCasa]]: Large-scale kitchen manipulation benchmark
- [[LIBERO]]: Tabletop manipulation benchmark

---

## Quick Reference Card

> [!summary] VideoPolicy (arXiv 2025)
> - **Core**: SVD fine-tuned as video policy (Stage 1: $\mathcal{L}_{video}$, lr=1e-5); 5 decoder layers → CNN adapters → $h_i$; 1D action U-Net ($\mathcal{L}_{action}$, stop-gradient, lr=5e-5) conditioned on $h_i$; decoupled 2-stage
> - **Method**: 25 frames (8/cam×3cams+pad), 256×256→448×320; batch 32; 8×A100 ~2 weeks; 30-step inference (9s); RoboCasa 368K steps
> - **Results**: 94% LIBERO-10 (vs. 85% π₀, 54% OpenVLA); 63% RoboCasa (50 demos); 60%+ unseen tasks (action-free pre-train)
> - **Code**: N/A

---

*Note created: 2026-05-20*
