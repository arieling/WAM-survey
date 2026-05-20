---
title: "Say, Dream, and Act: Learning Video World Models for Instruction-Driven Robot Manipulation"
method_name: "SayDreamAct"
authors: [Songen Gu, Yunuo Cai, Tianyu Wang, Simo Wu, Yanwei Fu]
year: 2026
venue: arXiv 2026
tags: [video-generation, adversarial-distillation, robot-policy, libero, in-context-learning, cascaded-explicit]
zotero_collection: 3-Robotics/WAM-Survey
image_source: online
arxiv_html: https://arxiv.org/abs/2602.10717
created: 2026-05-20
---

# Paper Note: Say, Dream, and Act: Learning Video World Models for Instruction-Driven Robot Manipulation

## Metadata

| Item | Content |
|------|---------|
| Institution | Not specified (likely Chinese institution) |
| Date | February 2026 |
| Project Page | N/A |
| Baselines | OpenVLA, CoT-VLA, UniVLA, π₀, GR00T N1, DreamVLA |
| Links | [arXiv](https://arxiv.org/abs/2602.10717) / Code: N/A |

---

## One-Sentence Summary

> SayDreamAct (Dream4manip) is a three-stage framework that adapts Cosmos-Predict2 as a world model via adversarial distillation for fast video generation, uses length-agnostic keyframe sampling for arbitrary-horizon trajectories, and conditions an in-context action model (ACT backbone) on both generated future frames and real historical observations to achieve 98.2% success on the LIBERO benchmark.

---

## Core Contributions

1. **Cosmos-Predict2 as World Model Backbone**: Systematically benchmarks multiple video generation models and selects Cosmos-Predict2 (2B) for robot manipulation prediction quality; adapts via adversarial distillation for 10× inference speedup.
2. **Adversarial Distillation**: Combines latent adversarial loss with reconstruction loss to distill a fast 10-step inference model from the full diffusion model — trading minimal quality for substantial speed improvement.
3. **Length-Agnostic Imagination**: Uniform temporal sampling compresses arbitrary-length trajectories into $n=9$ fixed keyframes, enabling the world model to imagine both short and long-horizon futures with the same architecture.
4. **In-Context Action Model**: Transformer-based action predictor (ACT) conditioned on generated future frames + real historical observations — the generated video serves as a "visual plan" that corrects spatial errors in imagination.

---

## Problem Background

### Problem to Solve
Video world models for robot manipulation face three core challenges: (1) long temporal horizon — standard video models are limited in sequence length; (2) spatial inconsistency — generated frames have object position errors; (3) inference cost — diffusion models are slow for real-time control.

### Limitations of Existing Methods
- UniPi, AVDC, GR-1: limited temporal horizon, single-step video prediction.
- Standard video diffusion: too slow for robot deployment (86 seconds for 35 denoising steps).
- Goal-image conditioned BC: ignores intermediate trajectory information.
- OpenVLA, CoT-VLA: language-action models without explicit visual imagination; 76.5–81.1% on LIBERO.

### Motivation
"Say" (VLM plans task), "Dream" (world model imagines execution), "Act" (action model executes given imagination). Distillation makes dreaming fast enough for deployment. In-context conditioning on generated frames gives the action model geometric guidance beyond raw language instructions.

---

## Method Details

### Model Architecture

**Three-stage pipeline**:
1. **Say (Task Planning)**: VLM (Qwen2.5 with LoRA rank 64) parses language instruction into sub-task plan
2. **Dream (World Model)**: Cosmos-Predict2 fine-tuned + adversarially distilled → generates $n=9$ keyframe video of task execution
3. **Act (Action Model)**: ACT-based transformer conditioned on generated video + real observation history → robot joint commands

### Core Modules

#### Module 1: World Model Selection and Adaptation

**Design Motivation**: Different video generation architectures have different strengths for robot manipulation; systematic benchmarking identifies Cosmos-Predict2 as optimal for manipulation prediction fidelity.

**Implementation**:
- Benchmarks multiple video models: Cosmos-Predict2 selected as backbone (2B parameters)
- Fine-tunes on LIBERO: 10,000 iterations for LIBERO, 1,000 iterations for real-world
- Uses preconditioning factors: $c_{skip} = (\sigma_t + 1)^{-1}$, $c_{out} = -\sigma_t/(\sigma_t + 1)$, $c_{in} = c_{skip}$

These preconditioning scalars modulate skip connections and output scaling in the diffusion model architecture for stable training across different noise levels. The noise schedule for distillation is:

$$\sigma_t = \left(\sigma_{min}^{1/p} + \frac{t}{T_g}(\sigma_{max}^{1/p} - \sigma_{min}^{1/p})\right)^p, \quad T_g = 8 \text{ steps}$$

This defines the noise levels at which the distilled 10-step model operates, enabling fast inference with only $T_g=8$ target denoising steps.

#### Module 2: Adversarial Distillation for Fast Inference

**Design Motivation**: Full diffusion (35 steps, 86 seconds) is too slow for robot deployment. Adversarial distillation compresses to 10 steps (30 seconds) with minimal quality loss.

**Implementation**:

The discriminator is trained to distinguish ground-truth frames $x'$ from generated frames $\hat x'$ using a hinge loss:

$$\mathcal L_{adv}^D = \mathbb E[\text{ReLU}(1 - D(x')) + \text{ReLU}(1 + D(\hat x'))]$$

The generator is trained to fool the discriminator while staying close to the target reconstruction:

$$\mathcal L_{adv}^G = \mathbb E[-D(\hat x')]$$

$$\mathcal L_{rec} = \|\hat x_0 - x_0\|_2^2 \cdot \frac{(1 + \sigma_t)^2}{\sigma_t^2}$$

The noise-schedule-dependent weighting in $\mathcal L_{rec}$ downweights the reconstruction term at low noise levels (where the model should already be accurate) and upweights it at high noise levels. The combined generator objective is:

$$\mathcal L_G = \lambda \cdot \mathcal L_{adv}^G + \mathcal L_{rec}, \quad \lambda = 0.1$$

This distillation achieves SSIM 0.70 at 10 steps (30 seconds) vs. 0.72 at 35 steps (86 seconds) — 3× speedup with only 0.02 SSIM degradation.

#### Module 3: Length-Agnostic Imagination

**Design Motivation**: Robot trajectories have variable lengths; the world model needs a fixed-size representation for arbitrary horizons.

**Implementation**:

To handle trajectories of arbitrary length $T$, uniform temporal sampling extracts a fixed number of keyframes:

$$S(\tau) = \{x_{t_i}\}_{i=1}^n, \quad t_i = \left\lfloor \frac{i}{n} \cdot T \right\rfloor, \quad n = 9$$

This provides $n=9$ uniformly-spaced keyframes from any trajectory length $T$, giving the world model a fixed-size temporal representation regardless of task duration.

#### Module 4: In-Context Action Model

**Design Motivation**: Generated video frames contain spatial errors; the action model needs to be robust to these errors. Conditioning on both generated frames (goal guidance) and real observation history (current state) provides corrective capability.

**Architecture**:
- Transformer-based (ACT backbone)
- Consumes both generated future keyframes and real historical observations
- Cross-attention between real observations and generated frames
- Outputs robot joint command sequences

---

## Experiments

### Datasets

| Dataset | Scale | Characteristics | Usage |
|---------|-------|----------------|-------|
| LIBERO | ~1,850 training samples (Spatial-433, Object-456, Goal-436, Long-389) | 4 suites, 10 subtasks each, 500 test runs per suite | Training/Evaluation |
| Real-world | Not specified | Franka 7-DoF arm + Intel RealSense D435 | Real-robot evaluation |

### Implementation Details

- **World Model**: Cosmos-Predict2 (2B parameters); fine-tuned 10,000 iterations (LIBERO), 1,000 (real-world)
- **Adversarial Distillation**: $\lambda = 0.1$; 10 denoising steps at inference
- **VLM**: Qwen2.5 with LoRA (rank 64)
- **Action Model**: ACT-based transformer; trained 20,000 steps on 8 GPUs; batch size 128; lr=2e-4
- **World Model Optimizer**: FusedAdamW, lr=$2^{-14.5}$
- **Keyframes**: $n=9$ uniformly sampled
- **Robot**: Franka 7-DoF arm; Intel RealSense D435 camera
- **Test Protocol**: 10 subtasks × 50 executions = 500 runs per suite

### Video Generation Quality

The full pipeline (Figure 1) is: Say stage (Qwen2.5 VLM parses language instruction into sub-task plan) → Dream stage (Cosmos-Predict2 generates 9 keyframe video of execution) → Act stage (ACT transformer conditioned on generated video + real observations produces joint commands).

| Model | FVD↓ | SSIM↑ | PSNR↑ | LPIPS↓ |
|-------|------|-------|-------|--------|
| Cosmos-2B (base) | — | — | — | — |
| **Cosmos-2B-DA+Dis (Ours)** | **238.09** | **0.84** | **26.82** | **0.04** |

### LIBERO Benchmark Success Rate

SayDreamAct achieves state-of-the-art 98.2% total success on LIBERO, substantially outperforming UniVLA (95.2%), π₀ (94.2%), and DreamVLA (92.6%):

| Method | Params | Spatial | Object | Goal | Long | **Total** |
|--------|--------|---------|--------|------|------|-----------|
| OpenVLA | 7B | — | — | — | — | 76.5% |
| CoT-VLA | 7B | — | — | — | — | 81.1% |
| DreamVLA | 0.57B | — | — | — | — | 92.6% |
| π₀ | 3B | — | — | — | — | 94.2% |
| GR00T N1 | 2B | — | — | — | — | 93.9% |
| UniVLA | 7B | — | — | — | — | 95.2% |
| **Dream4manip (Ours)** | — | **99.4%** | **99.2%** | **98.6%** | **95.4%** | **98.2%** |

Near-perfect performance on Spatial (99.4%) and Object (99.2%) suites.

### Denoising Steps Ablation

The adversarial distillation speed-quality trade-off is quantified as follows:

| Denoising Steps | SSIM↑ | Generation Time |
|----------------|-------|-----------------|
| 10 steps | 0.70 | 29.80s |
| 35 steps | 0.72 | 86.20s |

10 denoising steps (with adversarial distillation) achieves SSIM 0.70 at 30 seconds — 3× faster than 35 steps with only 0.02 SSIM degradation.

---

## Critical Analysis

### Strengths
1. Near-SOTA 98.2% on LIBERO with efficient model size — competitive with 7B models (UniVLA) at lower parameter count.
2. Adversarial distillation provides 3× speedup with minimal quality loss — practical for deployment.
3. Length-agnostic keyframe sampling elegantly handles variable-length trajectory prediction.

### Limitations
1. **Insufficient inference speed**: Even with distillation, 30 seconds per video generation limits real-time control.
2. **Error accumulation in long-horizon**: Generated predictions in dynamic environments drift over long sequences.
3. **Video quality dependency**: Action model performance directly coupled to world model fidelity.
4. **Limited real-world evaluation**: Primarily LIBERO simulation; real-robot results on Franka not fully detailed.

### Potential Improvements
1. Further distillation to 1–2 step generation for real-time inference.
2. Closed-loop video regeneration based on current observation for drift correction.
3. Multi-modal conditioning (depth, proprioception) for more accurate world model predictions.

### Reproducibility
- [ ] Code open-sourced
- [ ] Pretrained models available
- [x] Training details described (iterations, learning rate, batch size)
- [x] Datasets accessible (LIBERO public)

---

## Related Notes

### Based On
- Cosmos-Predict2: Video generation model used as world model backbone
- ACT: Action Chunking with Transformers backbone for action prediction
- Adversarial Training: Distillation methodology for fast inference

### Compared Against
- UniVLA: 7B parameter video-language-action model; outperformed 95.2% → 98.2%
- DreamVLA: Video-conditioned VLA; outperformed 92.6% → 98.2%
- π₀: Flow-based VLA; outperformed 94.2% → 98.2%

### Method Related
- Temporal Sampling: Length-agnostic keyframe extraction
- Behavioral Cloning: Action model training paradigm
- Classifier-Free Guidance: Conditioning approach in Cosmos diffusion model

### Hardware/Data Related
- LIBERO: Primary evaluation benchmark (4 suites, 1850 training demos)

---

## Quick Reference Card

> [!summary] SayDreamAct (arXiv 2026)
> - **Core**: Three-stage Say-Dream-Act: Qwen2.5 VLM planning → Cosmos-Predict2 (adversarially distilled, 10 steps) video imagination with n=9 keyframes → ACT action model conditioned on generated video + real history
> - **Method**: Cosmos-Predict2 2B; adversarial distillation ($\lambda=0.1$); 10K world model iterations; 20K ACT iterations; batch 128
> - **Results**: 98.2% total LIBERO success (vs. 95.2% UniVLA, 94.2% π₀, 92.6% DreamVLA); Spatial 99.4%, Long 95.4%
> - **Code**: N/A

---

*Note created: 2026-05-20*
