---
title: "VAG: Dual-Stream Video-Action Generation for Embodied Data Synthesis"
method_name: "VAG"
authors: [Xiaolei Lang, Yang Wang, et al.]
year: 2026
venue: arXiv 2026
tags: [video-generation, dual-stream, flow-matching, robot-policy, data-synthesis, cascaded-explicit]
zotero_collection: 3-Robotics/WAM-Survey
image_source: online
arxiv_html: https://arxiv.org/abs/2604.09330
created: 2026-05-20
---

# Paper Note: VAG: Dual-Stream Video-Action Generation for Embodied Data Synthesis

## Metadata

| Item | Content |
|------|---------|
| Institution | GigaAI, Zhejiang University, Peking University, CAS |
| Date | April 2026 |
| Project Page | N/A |
| Baselines | SVD, Wan2.2, Cosmos-Predict2, ResNet50+MLPs, AnyPos, π₀.₅ |
| Links | [arXiv](https://arxiv.org/abs/2604.09330) / Code: N/A |

---

## One-Sentence Summary

> VAG jointly generates temporally aligned robot action sequences and high-quality 480P video via a dual-stream flow-matching architecture (Cosmos-Predict2 video branch + 1D U-Net action branch), where video features condition the action branch via adaptive 3D pooling, enabling embodied data synthesis that boosts downstream VLA pretraining by +20%.

---

## Core Contributions

1. **Dual-Stream Flow-Matching Architecture**: First method to jointly synthesize high-quality video (480P, 93 frames, 10 seconds) and action sequences (dim 7–16) via two coordinated flow-matching streams — Cosmos-Predict2 DiT for video and a modified 1D U-Net (from Diffusion Policy) for actions.
2. **Adaptive 3D Pooling for Cross-Modal Conditioning**: Non-learnable adaptive 3D average pooling compresses spatiotemporal video feature tensors into action-compatible token sequences, enabling efficient video-to-action conditioning without learnable cross-attention overhead.
3. **Embodied Data Synthesis Pipeline**: VAG generates synthetic robot trajectory-video pairs from existing robot data; these synthetic paired demonstrations substantially improve downstream VLA pretraining (+20% absolute improvement, 35%→55%).
4. **Temporal Alignment Between Video and Action**: Both streams share the same flow-matching noise schedule and are trained jointly, enforcing temporal consistency between generated video frames and action sequences.

---

## Problem Background

### Problem to Solve
Robot manipulation requires large-scale paired (video, action) demonstrations, which are expensive to collect. Can a generative model synthesize high-quality paired video-action data that is useful for downstream policy learning? Specifically, can joint generation maintain the temporal alignment between visual observations and robot commands necessary for effective data augmentation?

### Limitations of Existing Methods
- SVD, Wan2.2, Cosmos-Predict2: generate video-only; no joint action prediction; cannot produce paired demonstration data.
- ResNet50+MLPs, AnyPos: action-only predictors; no video generation; lack visual grounding for data synthesis.
- Standard video world models (UniPi, AVDC): generate video then extract actions via IDM — two-stage process with IDM errors; limited action quality.
- π₀.₅: VLA without explicit video world model; cannot synthesize video-action pairs.

### Motivation
Joint generation of video and action sequences enforces temporal consistency between the two modalities — the video provides visual ground truth while the action branch must align with it. The dual-stream architecture, with video features conditioning the action branch, naturally creates paired demonstrations suitable for pretraining downstream policies.

---

## Method Details

### Model Architecture

VAG has **two parallel flow-matching streams** sharing training:

1. **Video Stream**: Cosmos-Predict2 (2B parameters) DiT denoiser — generates T=93 frames at 432×768 (480P) from text instruction + initial frame
2. **Action Stream**: Modified 1D U-Net from Diffusion Policy — generates action sequence $A \in \mathbb{R}^{T \times D}$ conditioned on compressed video features
3. **Cross-Modal Bridge**: Adaptive 3D average pooling (non-learnable) compresses video DiT intermediate features from spatiotemporal tensor to action-compatible token sequence

The overall architecture is illustrated by the video and action streams running in parallel during training, with the adaptive 3D pooling bridge transferring spatiotemporal video features into action U-Net conditioning. Both streams share the same flow-matching training objective.

### Core Modules

#### Module 1: Video Stream (Cosmos-Predict2 DiT)

**Design Motivation**: Cosmos-Predict2 provides strong video generation priors from large-scale internet video pretraining; fine-tuning on robot data adapts it to manipulation-specific visual dynamics.

**Implementation**:
- Base model: Cosmos-Predict2, 2B parameters
- Output: T=93 video frames at 432×768 resolution (480P), ~10 seconds at 10Hz
- Conditioned on: text instruction + initial scene frame
- Training loss — the video DiT is trained with a flow-matching objective minimizing the distance between the predicted velocity and the target velocity computed from the clean latent:

$$\mathcal{L}(\theta_1) = \|\phi_1(D(z'; \theta_1)) - z\|^2$$

where $D(z'; \theta_1)$ is the Cosmos-Predict2 DiT denoiser output given noisy latent $z'$, $z$ is the clean video latent (ground truth), and $\phi_1$ is the flow-matching velocity target function.

- 35 denoising steps at inference

#### Module 2: Action Stream (1D U-Net)

**Design Motivation**: 1D U-Net from Diffusion Policy is proven effective for action sequence generation; modifying it to accept video-conditioned features enables cross-modal alignment.

**Implementation**:
- Architecture: 1D U-Net (from Diffusion Policy), modified for video conditioning
- Output: action sequence $A \in \mathbb{R}^{T \times D}$ where $D \in \{7, 14, 16\}$ depending on robot embodiment
- Action embedding dimension: $C'' = 132$
- Training loss — the action stream is trained with an analogous flow-matching objective:

$$\mathcal{L}(\theta_2) = \|\phi_2(U(A'; \theta_2)) - A\|^2$$

where $U(A'; \theta_2)$ is the 1D U-Net denoiser output given noisy action sequence $A'$, $A \in \mathbb{R}^{T \times D}$ is the clean action sequence (ground truth), and $\phi_2$ is the flow-matching velocity target function for actions.

#### Module 3: Adaptive 3D Pooling Bridge

**Design Motivation**: Video DiT features are spatiotemporal tensors with incompatible dimensionality for the 1D action U-Net. Adaptive 3D pooling provides a parameter-free, architecture-agnostic bridge that preserves the most salient spatiotemporal structure.

**Implementation**:
- Input: spatiotemporal feature tensor from Cosmos-Predict2 DiT intermediate layers
- Operation: non-learnable adaptive 3D average pooling
- Output: compressed token sequence compatible with 1D U-Net conditioning
- No learnable parameters in the bridge — purely geometric downsampling

The embodied data synthesis pipeline is depicted as follows: given existing robot demonstrations (text instruction + initial frame + action label), VAG generates paired synthetic video + action sequences that are then used to pretrain downstream VLA policies.

---

## Experiments

### Datasets

| Dataset | Scale | Characteristics | Usage |
|---------|-------|----------------|-------|
| AgiBot | 1,794 train / 200 test | Action dim 16, manipulation tasks | Training/Evaluation |
| LIBERO | 400 train / 50 test | Action dim 7, 4 suites | Training/Evaluation |
| Self-collected | 131 train / 20 test | Action dim 14 | Training/Evaluation |

### Results

The following table compares video generation quality on the AgiBot manipulation dataset across three standard metrics.

**Table 1: Video Generation Quality (AgiBot Dataset)**

| Method | FVD↓ | SSIM↑ | PSNR↑ |
|--------|------|-------|-------|
| SVD | — | — | — |
| Wan2.2 | — | — | — |
| Cosmos-Predict2 | — | — | — |
| **VAG (Ours)** | **Best** | **Best** | **Best** |

VAG achieves best video generation quality on AgiBot manipulation dataset, outperforming baseline video generation models fine-tuned on the same data.

**Table 2: LIBERO Benchmark Success Rate**

| Method | Success Rate |
|--------|-------------|
| ResNet50+MLPs | — |
| AnyPos | 66% |
| **VAG** | **79%** |

VAG achieves 79% LIBERO success vs. 66% AnyPos (+13%) and substantially outperforms ResNet50+MLPs.

The key demonstration of VAG's utility for embodied data synthesis is the downstream VLA pretraining experiment. Adding VAG-synthesized paired demonstrations to the π₀.₅ training data yields a substantial absolute improvement:

**Table 3: VLA Pretraining with Synthetic VAG Data**

| Configuration | Success Rate |
|--------------|-------------|
| π₀.₅ baseline | 35% |
| π₀.₅ + VAG synthetic data | **55%** |

Adding VAG-synthesized paired demonstrations to VLA pretraining improves downstream performance from 35% to 55% — a +20% absolute improvement, demonstrating the utility of VAG for embodied data synthesis.

**Table 4: Self-Collected Real-World Tasks**

| Setup | Success Rate |
|-------|-------------|
| VAG-generated trajectory replay | 62% |

VAG generates plausible video-action pairs for custom real-world manipulation tasks (self-collected dataset: 131 train / 20 test, action dim 14).

### Implementation Details

- **Video Model**: Cosmos-Predict2, 2B parameters
- **Video Resolution**: 432×768 (480P), T=93 frames (~10 seconds at 10Hz)
- **Action Embedding Dimension**: $C'' = 132$
- **Denoising Steps (Inference)**: 35 steps
- **Optimizer**: AdamW
- **Training Iterations**: 40K (AgiBot), 20K (LIBERO)
- **Hardware**: 8× NVIDIA H20 GPUs, batch size 1 per GPU
- **Action Dimensions**: 16 (AgiBot), 7 (LIBERO), 14 (self-collected)

---

## Critical Analysis

### Strengths
1. Joint video-action generation enforces temporal alignment — synthesized data maintains correspondence between visual observations and robot commands, critical for pretraining.
2. +20% improvement in downstream VLA pretraining demonstrates clear practical value for data synthesis.
3. Adaptive 3D pooling bridge is parameter-free — no additional learnable overhead for cross-modal conditioning.

### Limitations
1. **One-directional conditioning**: Video branch is not influenced by action branch — only video→action conditioning, not bidirectional. Future work targets bidirectional conditioning.
2. **High inference cost**: 35 denoising steps for 93-frame 480P video is computationally expensive for large-scale synthesis.
3. **Cosmos-Predict2 dependency**: Relies on a large (2B) proprietary pretrained model; limited deployment flexibility.
4. **Action fidelity ceiling**: Video quality gates action quality — poor video generation leads to misaligned actions.

### Potential Improvements
1. Bidirectional conditioning where action branch also influences video branch.
2. Distillation to fewer denoising steps for faster synthesis.
3. Multi-embodiment generalization via embodiment-conditioned action stream.

### Reproducibility
- [ ] Code open-sourced
- [ ] Pretrained models available
- [x] Training details described (iterations, batch size, hardware)
- [ ] AgiBot dataset not fully public

---

## Related Notes

### Based On
- Cosmos-Predict2: 2B video generation model used as video stream backbone
- Diffusion Policy: 1D U-Net action stream architecture
- Flow Matching: Shared training paradigm for both streams

### Compared Against
- π₀: VLA baseline; downstream pretraining improved by +20% with VAG data
- AnyPos: Action prediction baseline; outperformed 66%→79% on LIBERO

### Method Related
- Dual-Stream Architecture: Parallel video and action generation
- Data Synthesis: Using generative models to augment robot demonstration datasets
- Behavioral Cloning: Downstream policy learning from synthesized data

### Hardware/Data Related
- AgiBot: Primary evaluation dataset
- LIBERO: Secondary evaluation benchmark

---

## Quick Reference Card

> [!summary] VAG (arXiv 2026)
> - **Core**: Dual-stream flow-matching: Cosmos-Predict2 (2B) video DiT + 1D U-Net action branch; adaptive 3D pooling bridge (non-learnable) for video→action conditioning; joint training for temporal alignment
> - **Method**: 480P @ 10Hz, T=93 frames; action dim 7–16; 35 denoising steps; 8×H20, 20–40K iterations; losses $\mathcal{L}(\theta_1) = \|\phi_1(D(z';\theta_1))-z\|^2$, $\mathcal{L}(\theta_2) = \|\phi_2(U(A';\theta_2))-A\|^2$
> - **Results**: LIBERO 79% vs 66% AnyPos; VLA pretraining 35%→55% (+20%) with synthetic VAG data
> - **Code**: N/A

---

*Note created: 2026-05-20*
