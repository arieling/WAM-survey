---
title: "VILP: Imitation Learning with Latent Video Planning"
method_name: "VILP"
authors: [Zhengtong Xu, Qiang Qiu, Yu She]
year: 2025
venue: arXiv 2025
tags: [video-diffusion, latent-planning, cascaded-implicit, robot-policy, vqgan, multi-view, real-time]
zotero_collection: 3-Robotics/WAM-Survey
image_source: online
arxiv_html: https://arxiv.org/abs/2502.01784
created: 2026-05-20
---

# Paper Note: VILP: Imitation Learning with Latent Video Planning

## Metadata

| Item | Content |
|------|---------|
| Institution | Purdue University |
| Date | February 2025 |
| Project Page | N/A |
| Baselines | UniPi, Diffusion Policy, LSTM-GMM, IBC |
| Links | [arXiv](https://arxiv.org/abs/2502.01784) / Code: N/A |

---

## One-Sentence Summary

> VILP achieves real-time latent video planning (5–14 Hz) for robot imitation learning via a VQGAN-compressed 3D U-Net diffusion model with multi-view cross-attention conditioning, generating temporally-aligned multi-view video plans from which a goal-conditioned CNN+MLP policy extracts actions — achieving 56.7% max success on Nut-Assembly vs. 35.3% UniPi and 0% UniPi on real-robot Arrange-Blocks.

---

## Core Contributions

1. **Real-Time Latent Video Generation**: First work achieving real-time robot video planning (0.058s per frame, 5–14 Hz control) via VQGAN compression + 3D U-Net with DDIM (4–8 steps), using 6.2–10.0 GB GPU memory vs. 20.5–82.5 GB for UniPi.
2. **Multi-View Cross-Attention Conditioning**: Three-part conditioning: visual encoders (modified ResNet-18) → low-dimensional context vectors → cross-attention into 3D U-Net decoder layers — enables view-consistent multi-perspective video generation.
3. **Temporally-Aligned Multi-View Planning**: Separate diffusion models per camera view generate temporally-synchronized multi-view videos — providing richer geometric information than single-view planning.
4. **Receding Horizon Control**: Goal-conditioned CNN+MLP policy executes only first $N_e$ actions from generated video plan, re-plans at each step — mitigates error accumulation from open-loop execution.

---

## Problem Background

### Problem to Solve
Video world models for robot manipulation face a speed-memory trade-off: generating high-quality video plans (UniPi) requires 20.5–82.5 GB GPU memory and 1.4–2.5 seconds per frame — far too slow for real-time control. How can latent video diffusion enable fast, memory-efficient video planning at manipulation-capable control rates?

### Limitations of Existing Methods
- UniPi: 16–64 denoising steps; 1.4–2.5 seconds/frame; 82.5 GB GPU; pixel-space; 0% on real Arrange-Blocks.
- Diffusion Policy: no video world model; requires extensive action-labeled demonstrations; 28% Nut-Assembly.
- Standard VLAs: no explicit visual planning; limited generalization to novel scenes.

### Motivation
VQGAN compression reduces video representation to a compact latent space, enabling fast 3D U-Net diffusion with far fewer denoising steps (4–8 DDIM steps). Multi-view generation provides richer spatial information for the downstream policy without per-view communication overhead during inference.

---

## Method Details

### Model Architecture

VILP has **two main components**:

1. **Latent Video Diffusion Model**: VQGAN compression → 3D U-Net diffusion with multi-view cross-attention conditioning
2. **Goal-Conditioned Policy**: CNN encoders + MLP heads → action sequences via receding horizon control

### Core Modules

#### Module 1: VQGAN Compression Module

**Design Motivation**: Compressing RGB/depth observations to a compact latent space dramatically reduces memory requirements and enables fast diffusion model inference.

**Implementation**:
- VQGAN autoencoder: compresses RGB/depth images $(H̃ \times W̃ \times C̃)$ to latent $(H \times W \times C)$
- Resolution: 96×160 or 160×96 pixels
- Provides both memory efficiency (6.2–10 GB vs. 82.5 GB UniPi) and faster encoding/decoding

#### Module 2: 3D U-Net Diffusion Backbone

**Design Motivation**: 3D convolutions capture spatiotemporal structure within each camera view — enabling coherent multi-frame video generation as a joint distribution.

**Implementation**:
- Architecture: 3D U-Net with 3D convolutions for spatiotemporal features
- Input: noisy latent video of H frames per camera view
- The unconditional latent video diffusion training objective is the standard DDPM denoising score matching in latent space:

$$\mathcal{L}(\theta) = \mathbb{E}\left[(\epsilon^k - \epsilon_\theta(z_t^k, k))^2\right]$$

where the model predicts noise $\epsilon^k$ from noisy latent $z_t^k$ at diffusion step $k$. When conditioned on the current observation $o_t$ via cross-attention, this becomes:

$$\mathcal{L}(\theta) = \mathbb{E}\left[(\epsilon^k - \epsilon_\theta(o_t, z_t^k, k))^2\right]$$

- Inference: DDIM with 4, 8, or 16 steps (VILP-4, VILP-8, VILP-16)
- Video frames per generation: 5–8 frames

#### Module 3: Multi-View Cross-Attention Conditioning

**Design Motivation**: Robot manipulation requires spatial understanding from multiple camera viewpoints. Cross-attention conditioning (vs. concatenation-only) provides more expressive view-specific information routing.

**Three-part conditioning design**:
1. **Visual encoders**: modified ResNet-18 compresses multi-view observations to low-dimensional context vectors $c_t$
2. **Cross-attention layers**: embed $c_t$ into 3D U-Net intermediate decoder layers
3. **Separate diffusion models per view**: each camera view has its own diffusion model, generating temporally-aligned video plans

The full VILP architecture processes multi-view RGB/depth observations through VQGAN encoder → latent → 3D U-Net diffusion (DDIM 4–16 steps, cross-attention conditioned on ResNet-18 features) → latent video plan → VQGAN decoder → generated video frames → CNN+MLP policy → action sequence (receding horizon $N_e$ steps).

The multi-view conditioning design uses separate 3D U-Net per camera view; ResNet-18 encodes all views to low-dimensional vectors; cross-attention injects these across all views simultaneously — enabling temporally synchronized multi-view predictions.

**Ablation**: cross-attention + low-dimensional fusion: 88.0% vs. concatenation-only: 72.5% on Sim Push-T.

#### Module 4: Goal-Conditioned Policy with Receding Horizon Control

**Design Motivation**: Open-loop trajectory execution from generated videos accumulates errors over long horizons. Receding horizon control re-plans at each step using the most recent observation.

**Implementation**:
- Input: consecutive generated frame pairs (current + next predicted frame)
- CNN encoder: extracts features from generated frame pairs
- MLP head: predicts action sequence
- Execution: only first $N_e$ actions executed before replanning
- Optimal horizons (ablation): 6-frame video + 8-step action, or 12-frame video + 16-step action

---

## Experiments

### Datasets

| Dataset | Scale | Characteristics | Usage |
|---------|-------|----------------|-------|
| Nut-Assembly | 200 episodes | Robosuite simulation | Policy training/eval |
| Can-PickPlace | Not specified | Robosuite simulation | Policy eval |
| Arrange-Blocks | 610 sim / 220 real demos | Simulated + Franka real-world | Training/eval |
| Sim Push-T | Simulated | 2D pushing tasks | Conditioning ablation |
| Move-the-Stack, Push-T, Towers-of-Hanoi | 90/10 split | Video quality evaluation | TVP quality eval |

### Results

The following table compares video planning quality (FID, FVD) and inference speed between VILP and UniPi.

**Table 1: Video Planning Quality**

| Task | Method | FID↓ | FVD↓ | Speed |
|------|--------|------|------|-------|
| Push-T | VILP-4 | 17.85 | 454.11 | 0.058s/frame |
| Push-T | VILP-8 | 14.65 | 460.98 | faster than UniPi |
| Push-T | UniPi-64 | 16.72 | 675.85 | 0.14–2.5s/frame |
| Towers-of-Hanoi | VILP-8 | **16.20** | — | — |
| Towers-of-Hanoi | UniPi-64 | 23.63 | — | slower |

VILP-8 achieves lower FVD (460 vs. 675) at substantially higher inference speed; FID comparable or better with 8× fewer denoising steps.

**Table 2: GPU Memory Comparison**

| Method | GPU Memory |
|--------|-----------|
| **VILP** | **6.2–10.0 GB** |
| UniPi | 20.5–82.5 GB |

8×–13× more memory-efficient than UniPi — enables consumer GPU deployment.

**Table 3: Policy Performance — Nut-Assembly**

| Method | Epochs | Max Success | Mean Success |
|--------|--------|-------------|--------------|
| **VILP-4** | 100 | **56.7%** | **53.2%** |
| UniPi-16 | 100 | 35.3% | 27.2% |
| Diffusion Policy | 300 | 28.0% | 24.3% |

VILP-4 achieves 56.7% vs. 35.3% UniPi (+21.4%) and 28.0% Diffusion Policy (+28.7%) with fewer training epochs.

**Table 4: Real Robot Results — Arrange-Blocks (15 trials)**

| Method | Success | Inference Time |
|--------|---------|---------------|
| **VILP-16** | **7/15 (46.7%)** | 0.238s |
| UniPi-16 | 0/15 (0%) | 1.422s |

VILP achieves 46.7% on real robot while UniPi completely fails (0%); VILP is 6× faster per inference step.

The following ablation compares conditioning strategies on the Sim Push-T task.

**Table 5: Conditioning Ablation — Sim Push-T**

| Conditioning | Success |
|-------------|---------|
| Concatenation only | 72.5% |
| Cross-attention | 82.6% |
| Cross-attention + low-dim fusion | **88.0%** |

Cross-attention + low-dimensional context vector fusion provides +15.5% over concatenation-only.

### Implementation Details

- **Video Compression**: VQGAN autoencoder
- **Backbone**: 3D U-Net with 3D convolutions
- **Conditioning**: Modified ResNet-18 visual encoders + cross-attention
- **Denoising Steps**: 4, 8, or 16 (DDIM)
- **Video Frames**: 5–8 frames per prediction
- **Resolution**: 96×160 or 160×96 pixels
- **Batch Size**: 16
- **Hardware**: NVIDIA RTX A6000
- **Inference Rate**: 5–14 Hz (0.058–0.238s per step)
- **Robot**: Franka Panda
- **GPU Memory**: 6.2–10.0 GB (vs. 20.5–82.5 GB UniPi)

---

## Critical Analysis

### Strengths
1. First work achieving real-time robot video planning (5–14 Hz) — practical for manipulation control.
2. 8×–13× memory reduction over UniPi enables consumer GPU deployment.
3. 46.7% real-robot success vs. 0% UniPi demonstrates critical practical advantage.

### Limitations
1. **Limited task diversity**: Evaluated on relatively simple tabletop tasks (Nut-Assembly, Arrange-Blocks).
2. **Small dataset scale**: 200–610 episodes — may not generalize to diverse environments.
3. **Low resolution**: 96×160 pixels limits fine-grained manipulation precision.
4. **Per-view separate models**: Multiple diffusion models increase memory linearly with camera count.

### Potential Improvements
1. Higher-resolution generation for precision manipulation.
2. Unified multi-view model with cross-view attention.
3. Pre-training on internet-scale video data for broader generalization.

### Reproducibility
- [ ] Code open-sourced
- [ ] Pretrained models available
- [x] Training details described (batch size, hardware, DDIM steps, resolution)
- [x] Standard benchmarks (Robosuite, Push-T) publicly available

---

## Related Notes

### Based On
- [[VQGAN]]: Autoencoder for efficient video latent compression
- [[DDIM]]: Fast denoising inference for real-time video generation
- [[Diffusion Policy]]: Action prediction baseline

### Compared Against
- [[UniPi]]: Pixel-space video planning; outperformed 35.3%→56.7% sim, 0%→46.7% real
- [[Diffusion Policy]]: BC baseline; outperformed 28%→56.7%

### Method Related
- [[Latent Video Diffusion]]: Compact latent space for fast video generation
- [[Receding Horizon Control]]: Online replanning to reduce open-loop error accumulation
- [[Multi-View Video Generation]]: Synchronized multi-perspective planning

---

## Quick Reference Card

> [!summary] VILP (arXiv 2025)
> - **Core**: VQGAN compression → 3D U-Net DDIM diffusion (4–16 steps) with multi-view cross-attention conditioning (ResNet-18 + cross-attention) → temporally-aligned multi-view video plan → CNN+MLP goal-conditioned policy with receding horizon control
> - **Method**: 96×160 resolution; 5–8 frames; batch 16; NVIDIA RTX A6000; 6.2–10 GB GPU; 0.058–0.238s/step (5–14 Hz)
> - **Results**: 56.7% Nut-Assembly (vs. 35.3% UniPi); 46.7% real Arrange-Blocks (vs. 0% UniPi); 88% Sim Push-T (cross-attention + fusion)
> - **Code**: N/A

---

*Note created: 2026-05-20*
