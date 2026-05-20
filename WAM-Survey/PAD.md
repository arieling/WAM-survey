---
title: "PAD: Prediction with Action Diffuser"
method_name: "PAD"
authors: []
year: 2024
venue: arXiv 2024
tags: [joint-diffusion, unified-stream, dit, ddpm, metaworld, generalization, multi-modal, depth]
zotero_collection: 3-Robotics/WAM-Survey
image_source: online
arxiv_html: https://arxiv.org/abs/2411.18179
created: 2026-05-20
---

# Paper Note: PAD: Prediction with Action Diffuser

## Metadata

| Item | Content |
|------|---------|
| Institution | Not specified |
| Date | November 2024 |
| Project Page | N/A |
| Baselines | GR-1, RT-2, Diffusion Policy |
| Links | [arXiv](https://arxiv.org/abs/2411.18179) / Code: N/A |

---

## One-Sentence Summary

> PAD is a unified DiT (up to 661M, XL/2) that jointly denoises k future RGB frames, depth maps, and robot actions in a single diffusion process — pretrained on BridgeData-v2 (60K trajectories, no action labels) and fine-tuned for manipulation, achieving 72.5% MetaWorld 50-task (+26.3% over GR-1) with image prediction contributing +29% and a joint loss $\mathcal{L} = \lambda_I \mathcal{L}_{diff}^I + \lambda_A \mathcal{L}_{diff}^A + \lambda_E \mathcal{L}_{diff}^E$.

---

## Core Contributions

1. **Unified Joint Diffusion**: Single DiT jointly denoises RGB future frames, depth maps, and robot actions — no separate video/action modules; multi-modal token concatenation with attention masking for missing modalities.
2. **Action-Free Video Pretraining**: Pretrained on BridgeData-v2 (60K trajectories, no action labels) using image prediction loss only — enables large-scale robot video pretraining without action annotation.
3. **Depth Modality Integration**: Depth images encoded as 32×32×1 and patchified alongside RGB — PAD-Depth variant achieves 78% real-world (vs. 72% RGB-only) — depth provides complementary geometric information.
4. **Strong Multi-Task and OOD Generalization**: 72.5% on MetaWorld 50-task (+26.3% over GR-1); 28% improvement on unseen objects — joint image+action prediction forces learning of generalizable visual dynamics.

---

## Problem Background

### Problem to Solve
Standard action-only policies do not model future visual states, limiting their ability to learn generalizable world dynamics. Can jointly predicting future image frames alongside actions in a unified diffusion process improve generalization to new objects and environments?

### Limitations of Existing Methods
- GR-1: 57.4% MetaWorld; separate video/action modules; pixel-space MSE video loss.
- RT-2: 52.2% MetaWorld; no image prediction; language-only conditioning.
- Standard Diffusion Policy: action-only; no visual future prediction.

### Motivation
Joint diffusion over images and actions creates mutual conditioning — the model must predict physically consistent future frames while predicting actions, preventing shortcut learning. Action-free video pretraining exploits abundant robot video without requiring expensive action labels, providing visual dynamics priors at scale.

---

## Method Details

### Model Architecture

PAD uses a **Diffusion Transformer (DiT)** with configurable sizes:

| Variant | Parameters | Layers | Hidden | Heads |
|---------|------------|--------|--------|-------|
| B/2 | 128M | 12 | 768 | — |
| L/2 | 449M | 24 | 1,024 | — |
| **XL/2** | **661M** | **28** | **1,152** | **16** |

**Input Encoding**:
- RGB images: Frozen VAE → 32×32×4 latent, patchified
- Robot poses: MLP encoder + linear transformation → tokens
- Depth images: Downsampled 32×32×1, patchified (patch size 8)
- Text: Frozen CLIP encoder

**Attention**: Self-attention across all modalities; masked attention for missing modalities (padded + masked positions)

**Output**: Inverse encoders reconstruct RGB, depth, and action from predicted latents

### Core Modules

#### Module 1: Multi-Modal Joint Denoising

**Design Motivation**: A single diffusion process across all modalities encourages cross-modal consistency — future frame predictions are constrained by action predictions and vice versa, preventing independent shortcut learning.

**Implementation**:
- Inputs (RGB, depth, poses, text) → token sequences via modality-specific encoders
- Concatenated + self-attention: each token attends to all others (or masked for missing)
- Channel concatenation conditioning: conditional latent concatenated with noise latent in channel dimension
- Joint denoising: simultaneous reconstruction of k=3 future RGB frames, depth, and actions

The unified objective for jointly denoising future RGB frames ($I$), actions ($A$), and depth ($E$) is:

$$\mathcal{L}(\theta) = \lambda_I \mathcal{L}_{diff}^I + \lambda_A \mathcal{L}_{diff}^A + \lambda_E \mathcal{L}_{diff}^E$$

where each per-modality loss follows the standard DDPM noise prediction objective:

$$\mathcal{L}_{diff}^\delta(\theta) = \mathbb{E}\left[\|\epsilon^\delta - \epsilon_\theta^\delta(z_t^\delta, t, C, l)\|_2^2\right]$$

Here $\epsilon^\delta$ is the noise added to modality $\delta$, $C$ is the observation conditioning, and $l$ is the language embedding. During action-free pretraining, $\lambda_A = \lambda_E = 0$ and only the image prediction loss is active. During task adaptation, $\lambda_A$ and $\lambda_E$ are linearly increased from 0.0 to 2.0 over 100K steps.

#### Module 2: Action-Free Pretraining

**Implementation**:
- Dataset: BridgeData-v2 (60,000 RGB video trajectories, no action labels)
- Hardware: 4× A100, ~2 days
- Loss: image prediction only ($\lambda_A = \lambda_E = 0$, $\lambda_I = 1.0$)
- Prediction: k=3 future frames at interval i=4 steps

#### Module 3: Task-Specific Adaptation

**Implementation**:
- MetaWorld: 50 trajectories/task × 50 tasks
- Real-world Panda: 200 trajectories/task × 6 tasks
- LR: 1e-4; batch: 256; 100K steps; ~1 day (4 A100)
- Loss weight schedule: $\lambda_I = 1.0$ (constant); $\lambda_A, \lambda_E$ linearly increase from 0.0 → 2.0 over 100K steps
- Inference: DDIM sampling, 75 steps

---

## Experiments

### Datasets

| Dataset | Scale | Characteristics | Usage |
|---------|-------|----------------|-------|
| BridgeData-v2 | 60,000 trajectories | Action-free robot video | Pretraining |
| MetaWorld | 50 tasks × 50 traj | Simulation manipulation | Fine-tuning/Evaluation |
| Real-world Panda | 6 tasks × 200 traj | Real-robot manipulation | Fine-tuning/Evaluation |

### Implementation Details

- **Backbone**: DiT XL/2 (661M); input resolution 256×256
- **VAE**: Frozen; 32×32×4 latent
- **Text**: Frozen CLIP
- **Depth**: 32×32×1 → patchified (patch size 8)
- **Future frames**: k=3, frame interval i=4
- **Pretraining**: 200K steps; LR 1e-4; batch 256; 4×A100, ~2 days
- **Adaptation**: 100K steps; LR 1e-4; batch 256; 4×A100, ~1 day
- **Loss schedule**: $\lambda_A, \lambda_E$ linearly increase 0→2.0 over adaptation
- **Inference**: DDIM 75 steps; resolution 256×256

### MetaWorld 50-Task Benchmark

PAD is compared against GR-1 (separate video/action modules) and RT-2 (language-conditioned only) on the MetaWorld 50-task benchmark:

| Method | Average Success |
|--------|----------------|
| RT-2* | 52.2% |
| GR-1 | 57.4% |
| **PAD (XL/2)** | **72.5%** |

+26.3% relative improvement over GR-1; joint image prediction is the dominant contributor (+29% as shown by the ablation below).

### Real-World Panda Tasks

Depth integration provides complementary geometric information to RGB appearance:

| Variant | Average Success |
|---------|----------------|
| PAD (RGB) | 72% |
| PAD-Depth | 78% |

Depth integration provides +6% on real-world tasks.

### OOD Generalization

PAD's joint image-action prediction forces generalization to new objects and visual distributions:

| Difficulty | PAD vs. Baseline |
|------------|-----------------|
| Easy (1–4 disturbances) | +28% success rate |
| Medium (5–15 disturbances) | Substantial gain |
| Hard (unseen objects) | +28% vs. strongest baseline |

### Ablation — Image Prediction

The contribution of the image prediction objective is confirmed by ablation:

| Configuration | Average Success |
|--------------|----------------|
| **PAD full** | **72.5%** |
| PAD w/o image prediction | 43.6% (−29%) |
| PAD w/o co-training | 59.2% (−13.3%) |

Image prediction is the dominant contributor (+29%); pretraining co-training adds +13.3%.

### Scaling Analysis

PAD shows monotonic improvement with compute scale — scale matters for joint diffusion:

| Variant | GFLOPs | Success |
|---------|--------|---------|
| XL/8 | 7.7 | 48.2% |
| XL/4 | 29.5 | 64.5% |
| **XL/2** | **119.1** | **72.5%** |

---

## Critical Analysis

### Strengths
1. Action-free video pretraining exploits 60K unannotated robot videos — scalable pretraining approach.
2. +29% from image prediction demonstrates importance of explicit future modeling.
3. Depth modality integration is architecturally elegant (same DiT, just another token stream).

### Limitations
1. **DDIM 75 steps**: Slow inference — not suitable for real-time control without acceleration.
2. **Limited real-world tasks**: Only 6 task categories; broader evaluation needed.
3. **Architecture details**: Author list not provided; limited context for related work positioning.

### Potential Improvements
1. Flow-matching objective for faster inference (fewer steps).
2. More diverse pretraining data (Open X-Embodiment, Ego4D).
3. Real-time inference via consistency distillation.

### Reproducibility
- [ ] Code open-sourced
- [ ] Pretrained models available
- [x] Architecture and training details described (LR, batch, steps, hardware)
- [x] MetaWorld, BridgeData-v2 publicly available

---

## Related Notes

### Compared Against
- [[GR-1]]: 57.4% MetaWorld; outperformed +15.1%
- [[RT-2]]: 52.2% MetaWorld; outperformed +20.3%

### Method Related
- [[Joint Diffusion]]: Single diffusion process over all modalities
- [[Action-Free Pretraining]]: Video pretraining without action labels
- [[Diffusion Transformer]]: DiT backbone for unified multi-modal denoising
- [[Depth Integration]]: Multi-modal geometric conditioning

---

## Quick Reference Card

> [!summary] PAD (arXiv 2024)
> - **Core**: DiT XL/2 (661M, 28L 1152H 16heads); joint DDPM denoising over k=3 future RGB + depth + action; $\mathcal{L} = \lambda_I \mathcal{L}_{diff}^I + \lambda_A \mathcal{L}_{diff}^A + \lambda_E \mathcal{L}_{diff}^E$; masked attention for missing modalities; frozen VAE (32×32×4) + CLIP
> - **Method**: Pretrain: 200K steps BridgeData-v2 60K (action-free); Adapt: 100K steps LR=1e-4 batch=256 4×A100; $\lambda_A,\lambda_E$ 0→2.0 linear schedule; DDIM 75 steps inference
> - **Results**: 72.5% MetaWorld 50-task (vs. 57.4% GR-1); 78% real-world (PAD-Depth); +29% from image prediction; +28% OOD objects
> - **Code**: N/A

---

*Note created: 2026-05-20*
