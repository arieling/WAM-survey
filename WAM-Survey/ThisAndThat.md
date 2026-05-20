---
title: "This&That: Language-Gesture Controlled Video Generation for Robot Planning"
method_name: "This&That"
authors: [Boyang Wang, Nikhil Sridhar, Chao Feng, Mark Van der Merwe, Adam Fishman, Nima Fazeli, Jeong Joon Park]
year: 2024
venue: arXiv 2024
tags: [video-generation, gesture-conditioning, controlnet, robot-policy, behavioral-cloning, cascaded-explicit, multimodal]
zotero_collection: 3-Robotics/WAM-Survey
image_source: online
arxiv_html: https://arxiv.org/abs/2407.05530
created: 2026-05-20
---

# Paper Note: This&That: Language-Gesture Controlled Video Generation for Robot Planning

## Metadata

| Item | Content |
|------|---------|
| Institution | University of Michigan, University of Washington |
| Date | July 2024 |
| Project Page | N/A |
| Baselines | [AVDC](AVDC.md), SVD, StreamingT2V, DragAnything, ACT |
| Links | [arXiv](https://arxiv.org/abs/2407.05530) / Code: N/A |

---

## One-Sentence Summary

> This&That combines deictic language instructions ("pick this, put that") with pointing gestures to condition a two-stage Stable Video Diffusion model for unambiguous robot video plan generation, then extracts actions via a video-conditioned behavioral cloning architecture (DiVA) — achieving 91.7% human alignment and strong disambiguation of identical objects.

---

## Core Contributions

1. **Language-Gesture Conditioned Video Generation**: Extends SVD with (1) language conditioning via FiLM modulation and (2) a ControlNet-style gesture branch that takes 2D pointing coordinates as sparse spatial conditioning — enabling unambiguous specification of which object to manipulate.
2. **Deictic Communication Interface**: Demonstrates that simple deictic language ("this", "that") paired with gestures matches the performance of precise natural language descriptions (87.5% vs. 91.7%), significantly lowering the cognitive cost of task specification.
3. **DiVA (Diffusion Video to Action)**: A transformer-based behavioral cloning architecture conditioned on generated video frames as goal images, with TokenLearner compression and temporal noise augmentation for robust action extraction.
4. **Automatic Gesture Labeling Pipeline**: Uses YOLOv8 + TrackAnything to automatically annotate grasp/release gestures from Bridge dataset videos, enabling scalable training without manual annotation.

---

## Problem Background

### Problem to Solve
How to communicate robot manipulation tasks unambiguously when the scene contains multiple similar or identical objects. Pure language descriptions ("pick the red block") fail when objects are identical; specifying exact positions verbally ("pick the block 10cm from the edge") is unnatural. Gestures (pointing) combined with deictic language ("pick this") provide the most natural and unambiguous specification.

### Limitations of Existing Methods
- Language-only video generation (AVDC, UniPi): Fails to disambiguate between identical or similar objects; depends on accurate, detailed text descriptions.
- Trajectory-based conditioning (DragAnything): Requires complete motion trajectories as input, which are difficult to provide without expert knowledge.
- Vision-only BC policies: Cannot generalize to novel object arrangements without video goal conditioning.
- Standard ControlNet for gesture: Naive application produces over-specified or under-specified conditioning due to the spatial-temporal sparsity of 2D gesture points.

### Motivation
Human-robot interaction naturally involves pointing gestures combined with simple deictic language. The overall system models the distribution $p_\theta(I_{0:T} | I_0, C_{text}, C_{gest})$ — video generation conditioned on initial frame, language, and gesture — which enables disambiguation and generalization that neither modality achieves alone.

---

## Method Details

### Model Architecture

This&That uses a **two-stage video diffusion model** built on Stable Video Diffusion (SVD) + **DiVA** for action extraction:

- **Stage 1 — Language-Conditioned Finetuning**: SVD UNet with FiLM modulation on CLIP language embeddings; generates task-consistent video given text description
- **Stage 2 — Gesture-Conditioning Branch**: Frozen Stage 1 UNet + trainable ControlNet-style gesture branch; conditions on 2D pointing coordinates encoded as sparse spatial images
- **DiVA Policy**: ResNet-18 + TokenLearner + Transformer encoder/decoder; conditioned on $N=25$ goal frames from generated video; outputs action chunks of size $k=10$

### Core Modules

#### Module 1: Language-Conditioned Video Diffusion (Stage 1)

**Design Motivation**: SVD lacks language conditioning; standard CLIP text embedding injection requires careful normalization.

**Implementation**:
- CLIP text encoder: $x_{text} \in \mathbb{R}^{b \times 77 \times 1024}$
- SVD first-frame CLIP: $x_{I_0} \in \mathbb{R}^{b \times 1 \times 1024}$
- Concatenate: $x_{concat} \in \mathbb{R}^{b \times 78 \times 1024}$
- Apply LayerNorm before cross-attention integration (critical — without it, performance degrades significantly)
- FiLM modulation: scales/shifts UNet feature maps conditioned on concatenated embeddings
- Training: Gaussian noise injection into frames $I_{0:T}$, MSE denoising loss
- Data: 25,767 Bridge V1/V2 videos; 99K iterations, 8× L40S GPUs, lr=1e-5

The full video generation objective is to model the distribution of future video frames conditioned on the initial frame, language description, and gesture coordinates:

$$
p_\theta(I_0, \ldots, I_T \mid I_0, C_{text}, C_{gest})
$$

where $I_0$ is the initial observation frame, $C_{text}$ is the CLIP-encoded language description (e.g., "pick this, put that"), and $C_{gest}$ is the 2D gesture image with Gaussian-dilated point markers.

#### Module 2: Gesture-Conditioning Branch (Stage 2, ControlNet-style)

**Design Motivation**: Naive ControlNet applied to sparse 2D gesture coordinates fails because gestures are spatially sparse (small point markers) — the conditioning signal is too weak. Additionally, gesture points must be aligned across the temporal dimension.

**Key Innovation**: Concatenate first-frame latent + noisy video frames + gesture image into a dense input $(B \times T) \times 12 \times H \times W$ (instead of gesture image alone).

**Implementation**:
- Gesture representation: two 2D coordinate points → $10 \times 10$ pixel squares with 2D Gaussian dilation
- Input to gesture branch formed by concatenating the first-frame latent, noisy video latent, and gesture image latent channel-wise:

$$
\text{input} = [\mathcal{E}(I_0),\; \epsilon_t,\; \mathcal{E}(C_{gest})] \in \mathbb{R}^{(B \times T) \times 12 \times H \times W}
$$

where $\mathcal{E}(I_0)$ is the VAE latent of the first frame (4 channels), $\epsilon_t$ is the noisy video latent at denoising step $t$ (4 channels), and $\mathcal{E}(C_{gest})$ is the VAE latent of the sparse gesture image (4 channels), yielding a 12-channel dense input per frame that resolves the sparsity problem of naive gesture conditioning.

- Freeze Stage 1 UNet weights; train only gesture branch
- Zero convolution initialization for zero outputs at start of training
- Gesture points distributed across frames temporally (not repeated identically)
- Latent masking with probability $p$: randomly mask gesture conditioning to enable classifier-free guidance at inference
- Data: 14,735 Bridge videos (after filtering by tracking quality), 30K iterations, 4× L40S GPUs, lr=5e-6

The following figure illustrates the full VDM architecture: Stage 1 shows the SVD UNet with FiLM modulation taking concatenated language + image CLIP embeddings after LayerNorm; Stage 2 shows the frozen UNet + gesture branch receiving the 12-channel dense input, with zero-conv outputs added to the UNet decoder.

#### Module 3: Automatic Gesture Labeling

**Design Motivation**: Manual annotation of grasp/release gesture coordinates at scale is infeasible; Bridge dataset has only high-level task metadata.

**Pipeline**:
1. Train YOLOv8 on 450 manually annotated gripper images → gripper bounding box detection
2. Apply TrackAnything to track target objects across frames
3. Identify key grasp/release moments from Bridge metadata (grasp = gripper closes on object)
4. Filter by tracking accuracy threshold → 14,735 usable videos from 25,767 total

#### Module 4: DiVA (Diffusion Video to Action) Policy

**Design Motivation**: Video-conditioned BC needs to handle temporal misalignment between generated video timing and actual execution; standard goal-image BC is brittle to frame selection.

**Architecture**:
- ResNet-18 (ImageNet pretrained, fine-tuned) processes observations and goal frames at 256×384×3 → $\mathbb{R}^{8 \times 12 \times 512}$ → flattened to $\mathbb{R}^{96 \times 512}$
- TokenLearner: 96 spatial tokens → 16 learned summary tokens per image
- Pose encoding: MLP converts end-effector state $s_t$ to 1 token
- Transformer encoder: 4 layers, self-attention + cross-attention to goal tokens $G \in \mathbb{R}^{N \times 16 \times 512}$
- Transformer decoder: 7 layers with fixed 2D sinusoidal positional embeddings $\mathbb{R}^{k \times 512}$
- Per-token MLPs: decode $k=10$ output tokens into action sequences
- Loss: L1 (more stable than L2 for robot control)
- Temporal noise augmentation: randomly sample goal frames from $N=25$ groups to handle temporal misalignment

The DiVA policy mapping can be written as:

$$
\pi_\theta(a_{t:t+k} \mid o_t, s_t, \tau)
$$

where $a_{t:t+k}$ is the action chunk of size $k=10$, $o_t$ is the live image observation at time $t$, $s_t$ is the robot end-effector pose, and $\tau$ is a subset of $N=25$ goal frames sampled from the generated video. The action chunk formulation with temporal noise augmentation makes the policy robust to the stochastic timing of generated video frames.

---

## Experiments

### Datasets

| Dataset | Scale | Characteristics | Usage |
|---------|-------|----------------|-------|
| Bridge V1 + V2 | 25,767 videos (Stage 1); 14,735 videos (Stage 2) | Real robot teleoperation; varied manipulation tasks | VDM training/testing (646 test videos) |
| IsaacGym Simulation | Custom pick-place with 4 blocks | Hand-scripted policies; in-distribution + out-of-distribution (identical blocks) | Policy rollout evaluation |

### Results

The following table reports user alignment across task types, comparing vision-only, language-only, gesture-only, and combined conditioning modes.

**Table 1: User Alignment Study (Bridge V1/V2)**

| Task | Vision-only | AVDC | V.+Lang. (Regular / Deictic) | V.+Gesture | V.+Lang.+Gesture (Regular / Deictic) |
|------|-------------|------|------------------------------|-----------|---------------------------------------|
| Pick & Place | 0% / — | 8.3% / 8.3% | 37.5% / 4.2% | 58.3% / — | **95.8% / 91.6%** |
| Stacking | 6.6% / — | 0% / 0% | 26.7% / 6.6% | 66.7% / — | **80.0% / 66.7%** |
| Folding | 11.1% / — | 5.6% / 5.6% | 50.0% / 33.3% | 55.6% / — | **88.9% / 94.4%** |
| Open/Close | 60.0% / — | 40.0% / 40.0% | 100.0% / 66.7% | 100.0% / — | **100.0% / 93.3%** |
| **Average** | **16.7%** | **12.5%** | **51.4% / 25.0%** | **68.1%** | **91.7% / 87.5%** |

Full V.+Lang.+Gesture achieves 91.7% (regular text) and 87.5% (deictic "this/that") — far above gesture-only (68.1%) or language-only (51.4%). Deictic language with gesture is nearly as effective as precise regular language with gesture, validating the practical utility of the interface.

The following table benchmarks video generation quality on the Bridge test set across five standard metrics.

**Table 2: Video Generation Quality (Bridge Test Set)**

| Method | FID↓ | FVD↓ | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|------|------|-------|-------|--------|
| SVD (baseline) | 29.49 | 657.49 | 12.47 | 0.334 | 0.391 |
| StreamingT2V | 42.57 | 780.81 | 11.35 | 0.324 | 0.504 |
| DragAnything | 34.38 | 764.58 | 12.76 | 0.364 | 0.466 |
| AVDC | 163.93 | 1512.25 | 20.23 | 0.663 | 0.507 |
| **This&That (Ours)** | **17.28** | **84.58** | **21.71** | **0.787** | **0.112** |

This&That dramatically outperforms all baselines across all 5 video quality metrics. Notably it achieves FVD of 84.58 vs. 657.49 for plain SVD — the gesture and language conditioning together dramatically improve video quality for robot manipulation, not just task alignment.

The following table reports simulation rollout results from IsaacGym, with in-distribution and out-of-distribution identical-block pick-and-place.

**Table 3: Simulation Rollout (IsaacGym, Pick & Place)**

| Method | Pick In-dist | Pick Out-of-dist | Place In-dist | Place Out-of-dist |
|--------|-------------|-----------------|--------------|------------------|
| ACT (Vision-only) | 5 | 3 | 0 | 1 |
| ACT (V.+Lang.) | 3 | 3 | 0 | 0 |
| ACT (V.+Lang.+Gesture) | 57 | 56 | 35 | 35 |
| AVDC-retrain (V.+Lang.) | 67 | 40 | 46 | 14 |
| Video-based (V.+Lang.) | 93 | 60 | 82 | 26 |
| **Video-based (V.+Lang.+Gesture) Ours** | **95** | **87** | **93** | **80** |

Gesture conditioning is critical for out-of-distribution generalization (identical blocks): video+language+gesture achieves 87/80 vs. 60/26 for video+language-only. Gesture removes ambiguity that language alone cannot resolve.

### Implementation Details

- **Base Video Model**: Stable Video Diffusion (SVD), latent space with VAE encoder $\mathcal{E}$ / decoder $\mathcal{D}$
- **Language Encoder**: CLIP from Stable Diffusion 2.1 (1024-dim embeddings, 77 tokens)
- **Stage 1 Training**: 99K iterations, batch size 1/GPU, lr=1e-5, AdamW 8-bit, 8× L40S GPUs
- **Stage 2 Training**: 30K iterations (Bridge) / 15K (IsaacGym), lr=5e-6, 4× L40S GPUs
- **Motion Parameters**: Motion bucket ID=200, noise augmentation=0.1
- **Augmentation**: Horizontal flip (p=0.45, disabled for position-sensitive tasks)
- **Policy ResNet-18**: ImageNet pretrained, fine-tuned; input 256×384×3
- **TokenLearner**: 96 spatial tokens → 16 summary tokens per image
- **Transformer**: Encoder 4 layers, Decoder 7 layers; action chunk $k=10$
- **Optimal $N$**: 25 goal frames (ablated; performance plateaus at $N \geq 15$)

---

## Critical Analysis

### Strengths
1. Elegant deictic interface: "this" + gesture is natural for humans and requires no precise object description — directly applicable to household robotics.
2. Strong generalization to identical objects (out-of-distribution): gesture conditioning disambiguates what language cannot.
3. Dense input trick for ControlNet is a practical and effective fix for gesture sparsity — widely applicable to other sparse spatial conditioning problems.

### Limitations
1. **Object appearance artifacts**: Generated videos show objects changing shape/appearance — inherent to 2D video without 3D geometric constraints.
2. **3D depth ambiguity**: 2D gesture points don't specify depth; only partially resolved by language context.
3. **Short-horizon tasks only**: Not extended to multi-step long-horizon sequences.
4. **Real robot not tested**: Simulation results only (WidowX 250 unavailable); prior Bridge work suggests sim-to-real transfer would likely hold.
5. **Gesture annotation failure rate**: Tracking failures reduce usable data from 25,767 to 14,735 videos (~43% filtering).

### Potential Improvements
1. 3D gesture specification via depth sensors or stereo for full 6-DoF pointing.
2. Multi-step task chaining with gesture conditioning at each sub-task.
3. Real robot deployment on Bridge dataset platform.

### Reproducibility
- [ ] Code open-sourced
- [ ] Pretrained models available
- [x] Training details complete (paper + appendix)
- [x] Datasets accessible (Bridge V1/V2 public; IsaacGym sim described)

---

## Related Notes

### Based On
- Stable Video Diffusion: Base video generation model fine-tuned with language and gesture conditioning
- ControlNet: Gesture conditioning branch architecture
- FiLM: Feature-wise Linear Modulation for language conditioning
- TokenLearner: Token compression for efficient vision transformer processing

### Compared Against
- [AVDC](AVDC.md): Video-based policy baseline; outperformed on all video quality metrics and alignment
- DragAnything: Trajectory-controlled video generation; outperformed on all metrics

### Method Related
- Behavioral Cloning: DiVA policy training via imitation learning
- Action Chunking: Action prediction strategy ($k=10$ steps)
- Classifier-Free Guidance: Gesture conditioning with latent masking for CFG at inference

### Hardware/Data Related
- Bridge Dataset: Primary training and evaluation dataset
- IsaacGym: Simulation platform for rollout evaluation

---

## Quick Reference Card

> [!summary] This&That (arXiv 2024)
> - **Core**: Language + gesture conditioned SVD for robot video planning; deictic "this/that" + pointing gesture for unambiguous object specification
> - **Method**: Two-stage SVD fine-tuning (FiLM language + ControlNet gesture branch with dense 12-channel input); DiVA policy (ResNet-18 + TokenLearner + Transformer, N=25 goal frames, k=10 action chunk)
> - **Results**: 91.7% user alignment (V.+Lang.+Gesture) vs. 16.7% vision-only; FVD 84.58 vs. 657 for SVD baseline; 95/87 pick success (in/out-dist)
> - **Code**: N/A

---

*Note created: 2026-05-20*
