---
title: "VideoVLA: Video Generators Can Be Generalizable Robot Manipulators"
method_name: "VideoVLA"
authors: []
year: 2024
venue: arXiv 2024
tags: [joint-diffusion, unified-stream, cogvideox, dit, simpler-env, generalization, video-diffusion, robot-policy]
zotero_collection: 3-Robotics/WAM-Survey
image_source: online
arxiv_html: https://arxiv.org/abs/2512.06963
created: 2026-05-20
---

# Paper Note: VideoVLA: Video Generators Can Be Generalizable Robot Manipulators

## Metadata

| Item | Content |
|------|---------|
| Institution | Not explicitly stated |
| Date | December 2024 |
| Project Page | N/A |
| Baselines | CogVideoX-5B (image-only), OpenSora-1.1 |
| Links | [arXiv](https://arxiv.org/abs/2512.06963) / Code: N/A |

---

## One-Sentence Summary

> VideoVLA adapts CogVideoX-5B as a multi-modal DiT that jointly generates future video (13 frames, 3D-causal VAE) and robot actions (6-step chunks) from T5 language conditioning — pretrained on Open X-Embodiment (22.5M frames, 22 embodiments) — achieving 92.3%/82.9%/66.2% on SIMPLER in-domain tasks and 65.2%/58% generalization to novel objects/skills, while bidirectional attention provides +4.9% over causal masking.

---

## Core Contributions

1. **CogVideoX-5B as Unified Policy Backbone**: Adapts CogVideoX-5B (5B multi-modal DiT with 3D-causal VAE) as a joint video+action diffusion model — no separate video and action modules; same DiT jointly denoises video latents and action tokens.
2. **Open X-Embodiment Pretraining at Scale**: Pretrained on 22.5M frames across 60 robot datasets and 22 embodiments — large-scale multi-robot pretraining provides broad manipulation priors.
3. **Bidirectional Attention**: Self-attention over all video and action tokens (no causal masking) — +4.9% over causal attention (80.4% vs. 75.5%); bidirectional conditioning provides better cross-modal alignment.
4. **Strong Cross-Task Generalization**: 65.2% on novel objects (SIMPLER), 58% on cross-embodiment skill transfer — video generation capability enables imaginative generalization.

---

## Problem Background

### Problem to Solve
Large-scale video generation models (like CogVideoX) have learned rich physical dynamics from internet-scale video. Can these video priors be directly transferred to robot manipulation policy learning through joint video+action diffusion fine-tuning?

### Limitations of Existing Methods
- Standard VLAs: no video generation; limited generalization to novel objects.
- Separate video + action models: decoupled optimization; no joint conditioning.
- OpenSora-1.1 backbone: 50.2% vs. CogVideoX-5B 80.4% — backbone quality matters critically.

### Motivation
Video generation models predict future visual states — the same computation underlying visual planning. By jointly generating actions with future video frames, the policy leverages the video model's rich world dynamics priors for action prediction. Cross-embodiment pretraining on 22 embodiments provides broad manipulation priors transferable to specific robot platforms.

---

## Method Details

### Model Architecture

VideoVLA uses **CogVideoX-5B** as a unified joint diffusion model:

1. **Text Encoder**: T5 — converts instructions to 226 fixed-length tokens
2. **Video Encoder**: 3D-causal VAE from CogVideoX — encodes video to latent space
3. **Backbone**: CogVideoX-5B DiT with self-attention blocks
4. **Action Representation**: 7-D vectors (3D rotation, 3D translation, 1 binary gripper state)
5. **Joint Prediction**: Same DiT denoises both video latents and action tokens

### Core Modules

#### Module 1: Joint Video + Action Denoising

**Design Motivation**: Single DiT jointly denoising video and action creates mutual conditioning — future video predictions inform action predictions and vice versa. No separate action head is needed.

**Implementation**:
- Video tokens: 3D-causal VAE encodes RGB frames → latent tokens
- Action tokens: embedded as sequence tokens in same DiT input
- Self-attention: bidirectional over all tokens (video + action + language)
- Output: 13 future frame latents (simulation) or 4 latents (real-world); 6 action steps (first 3 executed)
- Both video latents and action tokens are concatenated into a unified token sequence and denoised with the standard DDPM loss:

$$\mathcal{L}(\theta) = \mathbb{E}\left[\|\epsilon - \epsilon_\theta(z_t, t, c)\|^2\right]$$

where $z_t$ concatenates noisy video latents and action tokens and $c$ is the T5 language conditioning.

#### Module 2: Three-Stage Training

**Pretraining** (100K iterations):
- Dataset: Open X-Embodiment (22.5M frames, 60 datasets, 22 embodiments)
- Provides broad physical dynamics priors across robot types

**Fine-tuning** (15K iterations):
- Dataset: Task-specific demonstrations
- Optimizer: AdamW; LR 1e-5; weight decay 1e-4; batch 256
- Hardware: 32 AMD MI300X GPUs
- Inference: DDIM sampling, 50 denoising steps

#### Module 3: Future Frame Prediction Analysis

**Implementation**:
- Simulation: 13 future frame latents (49 video frames)
- Real-world: 4 future latents (13 frames)
- Motion similarity metric: SIFT keypoint tracking + SAM foreground segmentation
- Finding: higher video-action motion similarity → higher task success rate

---

## Experiments

### Datasets

| Dataset | Scale | Characteristics | Usage |
|---------|-------|----------------|-------|
| Open X-Embodiment | 22.5M frames, 60 datasets | 22 embodiments | Pretraining |
| Custom Realman | 5,824 samples | Pick, stack, place | Real-world evaluation |
| SIMPLER | — | Google Robot simulation | Evaluation |

### Results

The following table shows in-domain performance on the SIMPLER benchmark using the Google Robot setup.

**Table 1: SIMPLER In-Domain (Google Robot, Visual Matching)**

| Task | VideoVLA |
|------|----------|
| Pick Up | 92.3% |
| Move Near | 82.9% |
| Open/Close Drawer | 66.2% |

Strong in-domain performance on SIMPLER benchmark.

**Table 2: SIMPLER Generalization**

| Generalization Type | VideoVLA |
|--------------------|---------|
| Novel Objects (10 objects) | 65.2% |
| New Skills (7 skills) | 48.6% |

**Table 3: Real-World (Realman Robot)**

| Task | In-Domain | Novel Objects |
|------|-----------|--------------|
| Pick Up | 70.8% | — |
| Stack | 66.7% | — |
| Place | 56.3% | — |
| Overall | 64.6% | 50.6% |
| Cross-embodiment | 58.0% | — |

The following ablation table isolates the contribution of the video generation objective and the bidirectional attention design choice.

**Table 4: Architecture Ablation**

| Configuration | Average Success |
|--------------|----------------|
| OpenSora-1.1 | 50.2% |
| Action-only (no video) | 25.5% |
| Video loss removal | 27.0% |
| Causal attention | 75.5% |
| **CogVideoX-5B (full)** | **80.4%** |

Video generation loss is critical (removing it drops to 27%); bidirectional attention +4.9% over causal; CogVideoX-5B backbone quality is decisive.

**Table 5: Future Frame Count**

| Frames | Success |
|--------|---------|
| 13 frames | 75.2% |
| **49 frames** | **80.4%** |

More future frames → better performance; longer horizon provides richer temporal context.

### Implementation Details

- **Backbone**: CogVideoX-5B; 3D-causal VAE; T5 text encoder (226 tokens)
- **Action**: 7-D (3D rotation + 3D translation + binary gripper); 6-step chunk; execute first 3
- **Video frames**: 13 latents (49 frames) simulation; 4 latents (13 frames) real-world
- **Attention**: Bidirectional over all tokens (no causal masking)
- **Pretraining**: 100K iterations; OXE 22.5M frames
- **Fine-tuning**: 15K iterations; AdamW LR 1e-5, WD 1e-4; batch 256; 32× AMD MI300X
- **Inference**: DDIM 50 steps
- **Motion analysis**: SIFT + SAM for video-action consistency metric

---

## Critical Analysis

### Strengths
1. CogVideoX-5B as backbone provides rich video priors — demonstrated by large gap over OpenSora (80.4% vs. 50.2%).
2. Cross-embodiment pretraining (22 embodiments) enables broad generalization.
3. Strong video-action motion correlation finding provides interpretability.

### Limitations
1. **DDIM 50 steps**: Slow inference — not suitable for real-time closed-loop control.
2. **No code released**: Limited reproducibility; large MI300X GPU cluster needed.
3. **Limited benchmark diversity**: Primarily SIMPLER and custom real-world; no LIBERO, CALVIN.

### Potential Improvements
1. Consistency distillation for 1–4 step inference.
2. Broader benchmark evaluation (LIBERO, RLBench).
3. Conditioning on intermediate future frames for hierarchical planning.

### Reproducibility
- [ ] Code open-sourced
- [ ] Pretrained models available
- [x] Training details described (LR, batch, iterations, hardware)
- [x] Open X-Embodiment, SIMPLER publicly available

---

## Related Notes

### Based On
- CogVideoX: 5B video generation backbone; 3D-causal VAE; DiT architecture
- Open X-Embodiment: Large-scale multi-robot pretraining dataset
- T5: Text encoder for language conditioning

### Compared Against
- OpenSora-1.1: Alternative video backbone; substantially outperformed (80.4% vs. 50.2%)

### Method Related
- Joint Diffusion: Unified video+action DDPM denoising
- Bidirectional Attention: Full self-attention over video+action tokens
- Cross-Embodiment Pretraining: Multi-robot pretraining for generalization

---

## Quick Reference Card

> [!summary] VideoVLA (arXiv 2024)
> - **Core**: CogVideoX-5B DiT; 3D-causal VAE; T5 (226 tokens); bidirectional self-attention over video+action tokens; DDPM joint denoising; 7-D action (3pos+3rot+gripper), 6-step chunk (execute 3); 49 frames simulation / 13 real
> - **Method**: Pretrain: 100K iters OXE 22.5M frames 22 embodiments; Fine-tune: 15K iters LR=1e-5 WD=1e-4 batch=256 32×MI300X; DDIM 50 steps
> - **Results**: 92.3%/82.9%/66.2% SIMPLER in-domain; 65.2% novel objects; 58% cross-embodiment; video loss removal → 27.0%; bidirectional vs. causal: +4.9%
> - **Code**: N/A

---

*Note created: 2026-05-20*
