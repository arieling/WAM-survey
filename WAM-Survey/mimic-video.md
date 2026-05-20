---
title: "mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs"
method_name: "mimic-video"
authors: [Jonas Pai, Liam Achenbach, Victoriano Montesinos, Benedek Forrai, Oier Mees, Elvis Nava]
year: 2025
venue: arXiv 2025
tags: [video-generation, cascaded-implicit, cosmos-predict2, flow-matching, partial-denoising, robot-policy, data-efficiency]
zotero_collection: 3-Robotics/WAM-Survey
image_source: online
arxiv_html: https://arxiv.org/abs/2512.15692
created: 2026-05-20
---

# Paper Note: mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs

## Metadata

| Item | Content |
|------|---------|
| Institution | mimic robotics, Microsoft Zurich, ETH Zurich, ETH AI Center, UC Berkeley |
| Date | December 2025 |
| Project Page | N/A |
| Baselines | π₀.₅-style VLA, OpenVLA, Octo, Diffusion Policy, ThinkAct, DiT-Block |
| Links | [arXiv](https://arxiv.org/abs/2512.15692) / Code: N/A |

---

## One-Sentence Summary

> mimic-video (Video-Action Models) conditions a lightweight DiT action decoder on intermediate latent representations from partial denoising (stopping at flow-time $\tau_v$) of Cosmos-Predict2 (2B) — achieving 10× sample efficiency and 93.9% LIBERO/56.3% SIMPLER-Bridge without generating full pixel-space videos, while an oracle study confirms that perfect video latents yield near-perfect policy success.

---

## Core Contributions

1. **Partial Denoising for Feature Extraction**: Extracts intermediate latent representations $h^{\tau_v}$ by stopping Cosmos-Predict2 denoising at flow-time $\tau_v$ (optimal at $\tau_v \approx 1.0$, high noise) rather than full reconstruction — provides richer, more generalizable features than clean predictions while avoiding expensive full video generation.
2. **Video-Action Model (VAM) Framework**: Paired Cosmos-Predict2 video backbone + lightweight DiT action decoder (cross-attention to video latents + self-attention over actions + AdaLN MLP heads) — treats the action decoder as an implicit IDM conditioned on video dynamics features.
3. **10× Sample Efficiency**: Action decoder reaches VLA baseline performance with only 10% of training data; competitive at 2% (one episode per task) — video dynamics pretraining dramatically reduces robot-specific data requirements.
4. **Oracle Study**: Conditioning on ground-truth video latents yields near-perfect success — confirms that video prediction quality is the bottleneck, not action decoding capacity.

---

## Problem Background

### Problem to Solve
VLA architectures use vision-language pretraining for robot policies — but language pretraining is "blind to physical causality" (it captures semantic relationships, not physical dynamics). Can pretraining on internet-scale video provide better physical priors for robot manipulation policies?

### Limitations of Existing Methods
- π₀.₅-style VLA: language-visual pretraining; 35.4% SIMPLER-Bridge; lacks physical dynamics priors.
- VideoPolicy: requires full 30-step SVD denoising (9 seconds/inference) for feature extraction.
- OpenVLA: 7B parameters; large labeled dataset requirement.
- Diffusion Policy: no video pretraining; limited generalization.

### Motivation
Internet-scale video models (Cosmos-Predict2) understand physical causality — how objects move, interact, and respond to forces. Extracting intermediate denoising latents (without full video generation) provides features encoding predicted future dynamics: richer than static visual observations, cheaper than full video synthesis. The action decoder reads these "future-aware" features as an implicit IDM.

---

## Method Details

### Model Architecture

mimic-video has **two components** trained in two stages:

1. **Video Backbone**: Cosmos-Predict2 (2B DiT) with LoRA fine-tuning on robotics data
2. **Action Decoder**: Lightweight DiT (cross-attention + self-attention + AdaLN) trained from scratch on frozen video features

### Core Modules

#### Module 1: Cosmos-Predict2 Video Backbone (Stage 1 — LoRA Fine-tuning)

**Design Motivation**: Cosmos-Predict2's internet-scale video pretraining encodes physical dynamics priors. LoRA fine-tuning on robotics data adapts these priors to manipulation-specific visual dynamics without full fine-tuning cost.

**Implementation**:
- Base model: Cosmos-Predict2, 2B parameters, Diffusion Transformer on latent video patches
- Fine-tuning: Low-Rank Adapters (LoRA) on robotics datasets
- Training: lr=1.778e-4 (BridgeDataV2); batch 256; 27K–70K steps depending on dataset

The video generation backbone uses a flow matching formulation. Under optimal transport flow matching, the interpolation between clean data $x^0$ (at $\tau=0$) and noise $\epsilon$ (at $\tau=1$) follows:

$$x^\tau = (1-\tau)x^0 + \tau\epsilon, \quad \tau \in [0,1]$$

Training teaches the velocity field to reverse this flow; the key insight is that stopping at an intermediate $\tau_v > 0$ yields features that encode partial predictions rather than clean reconstructions.

#### Module 2: Partial Denoising Feature Extraction

**Design Motivation**: Full denoising is expensive (many steps) and the resulting clean video predictions suffer from domain mismatch. Stopping at intermediate $\tau_v$ extracts representations that encode predicted future state without full reconstruction overhead.

**Implementation**:
- Partial denoising: run Cosmos-Predict2 from noise to intermediate flow-time $\tau_v$
- The extracted latent is defined as:

$$h^{\tau_v} = \text{CosmosPredict2}_{\tau_v}(v_0, \text{instruction})$$

where $v_0$ is the initial observation and $h^{\tau_v}$ encodes intermediate video prediction features used to condition the action decoder.

- Optimal: $\tau_v \approx 1.0$ (high noise) — counterintuitive; attributed to richer intermediate representations and avoidance of distribution mismatch
- Per-task $\tau_v$ tuning improves SIMPLER-Bridge from 46.9% → 56.3%

#### Module 3: Lightweight DiT Action Decoder (Stage 2)

**Design Motivation**: A small, efficient action decoder reading video latent features as an implicit IDM — the video backbone provides "where and how to act" via predictive dynamics features; the decoder translates these to robot commands.

**Architecture**:
- Conditional flow matching: learns velocity field $v_\theta$ to transform noise to actions
- Cross-attention: action tokens attend to video latent $h^{\tau_v}$
- Self-attention: over action token sequence
- Two-layer MLPs with AdaLN (Adaptive LayerNorm) modulation for timestep injection
- The decoder is trained by minimizing the distance from the ground-truth conditional flow vector field:

$$\mathcal L = \mathbb E\left[\|v_\theta(a^\tau, \tau, h^{\tau_v}) - (a^0 - \epsilon)\|^2\right]$$

where $a^\tau$ is the noisy action at flow-time $\tau$, $a^0$ is the clean target action, and $\epsilon$ is noise. The decoder learns to predict the velocity $(a^0 - \epsilon)$ conditioned on the video latent $h^{\tau_v}$.

- Training: lr=1e-4, batch 128, frozen video backbone (stop-gradient)

---

## Experiments

### Datasets

| Dataset | Scale | Characteristics | Usage |
|---------|-------|----------------|-------|
| BridgeDataV2 | 60K trajectories | Kitchen manipulation | Video LoRA fine-tuning |
| LIBERO | Standard | 4 suites, tabletop simulation | Evaluation |
| SIMPLER-Bridge | Standard | Simulation manipulation benchmark | Evaluation |
| Real bimanual | Custom | Package handling, tape stowing | Real-world evaluation |

### Implementation Details

- **Video Backbone**: Cosmos-Predict2, 2B parameters
- **Fine-tuning**: LoRA (Low-Rank Adapters)
- **Video LR**: 1.778e-4 (BridgeDataV2); Batch 256
- **Action Decoder LR**: 1e-4; Batch 128
- **Training Steps**: 27K–70K (dataset dependent)
- **Partial Denoising**: optimal $\tau_v \approx 1.0$ (per-task tunable)
- **Action Decoder Architecture**: DiT with cross-attention + self-attention + AdaLN MLPs
- **Training Paradigm**: Conditional flow matching (optimal transport interpolation)

### SIMPLER-Bridge Benchmark

The oracle study in Figure 2 reveals that conditioning the action decoder on ground-truth video latents yields near-perfect success, confirming that video prediction quality (not decoder capacity) is the performance bottleneck. On the SIMPLER-Bridge benchmark, per-task $\tau_v$ tuning provides a substantial additional gain:

| Method | Avg. Success Rate |
|--------|-----------------|
| π₀.₅-style VLA | 35.4% |
| mimic-video (default) | 46.9% |
| **mimic-video (per-task τᵥ)** | **56.3%** |

This represents +20.9% over π₀.₅-style VLA baseline; per-task $\tau_v$ tuning adds +9.4%. The optimal flow-time peaks at high $\tau_v \approx 1.0$ (high noise level) rather than low $\tau_v$ (clean prediction) — intermediate noisy features provide richer, more generalizable representations.

### LIBERO Benchmark

| Method | Avg. Success Rate |
|--------|-----------------|
| OpenVLA | 54% (without fine-tuning) |
| OpenVLA-OFT (fine-tuned) | 96.9% |
| **mimic-video** | **93.9%** |

93.9% on LIBERO is competitive with fully fine-tuned OpenVLA-OFT.

### Real-World Bimanual Tasks

| Task | mimic-video | DiT-Block (multi-view) |
|------|------------|------------------------|
| Package handling | **72%** | 42.6% |
| Tape stowing | **93%** | 74.1% |

+29.4% on package handling, +18.9% on tape stowing — substantial real-world gains from video dynamics pretraining.

### Data Efficiency

| Training Data | Policy Performance |
|--------------|------------------|
| 100% data | Baseline VLA |
| 10% data | Matches VLA baseline |
| 2% data (1 episode/task) | Competitive |

The 10× sample efficiency improvement demonstrates that video dynamics pretraining dramatically reduces robot-specific data requirements.

---

## Critical Analysis

### Strengths
1. Partial denoising avoids expensive full video generation while preserving rich predictive features.
2. 10× sample efficiency is highly practical for real-world deployment with limited demonstrations.
3. Oracle study provides key insight: video quality is the bottleneck — motivates future video model improvements.

### Limitations
1. **Per-task $\tau_v$ tuning required**: Optimal flow-time varies by task — requires additional tuning overhead.
2. **Cosmos-Predict2 dependency**: Large 2B proprietary model as backbone.
3. **Video generation quality bottleneck**: Oracle gap indicates current video models still limit policy quality.
4. **Limited public details**: Exact implementation of partial denoising and adapter architecture not fully specified.

### Potential Improvements
1. Learned adaptive $\tau_v$ selection from task context.
2. Better video generation quality to close oracle gap.
3. Multi-task $\tau_v$ schedule for deployment without per-task tuning.

### Reproducibility
- [ ] Code open-sourced
- [ ] Pretrained models available
- [x] Training details described (LR, batch size, steps, LoRA)
- [x] LIBERO, SIMPLER publicly available

---

## Related Notes

### Based On
- Cosmos-Predict2: 2B video DiT backbone for physical dynamics pretraining
- Flow Matching: Action decoder training and video generation paradigm
- LoRA: Parameter-efficient video backbone fine-tuning

### Compared Against
- π₀: VLA baseline; outperformed 35.4%→56.3% SIMPLER-Bridge
- OpenVLA: 7B VLA; comparable LIBERO performance

### Method Related
- Partial Denoising: Stopping diffusion at intermediate steps for feature extraction
- Implicit IDM: Action decoder as implicit inverse dynamics on video latents
- Data Efficiency: Few-shot robot policy learning via video pretraining

---

## Quick Reference Card

> [!summary] mimic-video (arXiv 2025)
> - **Core**: Cosmos-Predict2 (2B, LoRA fine-tuned) partial denoising at $\tau_v \approx 1.0$ → intermediate latent $h^{\tau_v}$ → lightweight DiT action decoder (cross-attention + AdaLN, conditional flow matching); stop-gradient; no full video generation
> - **Method**: Video: lr=1.778e-4, batch 256, 27K–70K steps; action: lr=1e-4, batch 128; per-task $\tau_v$ tuning; BridgeDataV2 LoRA
> - **Results**: 56.3% SIMPLER-Bridge (vs. 35.4% VLA); 93.9% LIBERO; 72%/93% real bimanual; 10× sample efficiency; near-perfect oracle
> - **Code**: N/A

---

*Note created: 2026-05-20*
