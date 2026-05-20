---
title: "UWM: Unified World Models for Robotic Pretraining"
method_name: "UWM"
authors: []
year: 2025
venue: arXiv 2025
tags: [joint-diffusion, unified-stream, decoupled-timesteps, register-tokens, libero, real-robot, sdxl-vae, generalization]
zotero_collection: 3-Robotics/WAM-Survey
image_source: online
arxiv_html: https://arxiv.org/abs/2504.02792
created: 2026-05-20
---

# Paper Note: UWM: Unified World Models for Robotic Pretraining

## Metadata

| Item | Content |
|------|---------|
| Institution | Not specified |
| Date | April 2025 |
| Project Page | N/A |
| Baselines | Diffusion Policy, PAD, GR-1 |
| Links | [arXiv](https://arxiv.org/abs/2504.02792) / Code: N/A |

---

## One-Sentence Summary

> UWM uses modality-specific independent diffusion timesteps ($t_a \sim U(0,T)$, $t_{o'} \sim U(0,T)$) in a shared diffusion transformer (ResNet-18 + frozen SDXL VAE, register tokens for cross-modal exchange) — enabling flexible inference modes (policy, video prediction, forward/inverse dynamics) — pretrained on 2K DROID action-labeled + 2K action-free videos, achieving 0.79 LIBERO OOD and 0.92/0.84 Stack-Bowls in/OOD (vs. 0.48/0.36 DP).

---

## Core Contributions

1. **Decoupled Modality-Specific Timesteps**: Independent noise timesteps $t_a$ and $t_{o'}$ for action and observation diffusion — setting $t_{o'} = T$ yields pure policy inference; $t_a = T$ yields video prediction; $t_a = 0$ yields forward dynamics; $t_{o'} = 0$ yields inverse dynamics — one model supports all inference modes.
2. **Register Tokens for Cross-Modal Feature Exchange**: Learnable intermediary tokens attending to both modality streams — enable efficient feature sharing between action and observation diffusion without separate cross-attention modules.
3. **Action-Free Video Co-Training**: Joint training on action-labeled DROID trajectories + action-free DROID videos (masking via diffusion timesteps $t_a = T$ for action-free data) — enables learning from unannotated robot video.
4. **Strong Real-Robot OOD Generalization**: 21/30 vs. 12/30 (Stack-Bowls) across lighting/background/clutter variations — decoupled world model pretraining provides more robust visual dynamics representations.

---

## Problem Background

### Problem to Solve
Most robot learning frameworks treat video prediction and action prediction as separate problems with separate models. Can a unified diffusion model with modality-specific noise schedules simultaneously learn all four inference modes (policy, video prediction, forward/inverse dynamics) from a single training objective?

### Limitations of Existing Methods
- Diffusion Policy: action-only; 0.48/0.36 Stack-Bowls in/OOD.
- PAD: joint diffusion but single timestep across modalities; cannot flexibly switch inference modes.
- GR-1: separate video/action decoders; pixel-space MSE video loss.

### Motivation
Independent timestep scheduling creates a versatile model: $t_{o'} = T$ (pure noise observation) forces the model to generate future observations from scratch (world model); $t_a = T$ forces action generation from scratch (policy). This flexible conditioning enables a single model to serve as policy, world model, forward/inverse dynamics model — all from one training objective.

---

## Method Details

### Model Architecture

UWM uses a **shared diffusion transformer** with:
1. **ResNet-18 encoder**: Encodes current observations → n_embd dimensions
2. **Frozen SDXL VAE**: Encodes/decodes 224×224 RGB → 28×28×4 latents (spatial compression)
3. **Spatiotemporal Patchifier**: (4,4,2) patch size for image embeddings
4. **Register Tokens**: Learnable intermediary tokens for cross-modal feature exchange
5. **Sinusoidal Timestep Encoder**: Shared between action and observation timesteps
6. **AdaLN Conditioning**: Adaptive layer normalization for timestep + observation conditioning

### Core Modules

#### Module 1: Decoupled Diffusion Timesteps

**Design Motivation**: Independent noise levels per modality allow the model to be conditioned on any combination of clean/noisy observation + action, enabling flexible inference at any combination of "how much noise" per modality.

**Implementation**:
- Training: Sample $t_a \sim U(0,T)$ and $t_{o'} \sim U(0,T)$ independently
- Noisy samples:
  - $a_{t_a} = \sqrt{\bar{\alpha}_{t_a}} \cdot a + \sqrt{1-\bar{\alpha}_{t_a}} \cdot \epsilon_a$
  - $o'_{t_{o'}} = \sqrt{\bar{\alpha}_{t_{o'}}} \cdot o' + \sqrt{1-\bar{\alpha}_{t_{o'}}} \cdot \epsilon_{o'}$
- Inference modes: set $t_a$ or $t_{o'}$ to 0 (clean) or T (pure noise) to select desired mode

The joint training objective is a weighted sum of denoising losses for action and observation, with independently sampled timesteps:

$$\ell(\theta) = \mathbb{E}\left[w_a \|\epsilon_\theta^a - \epsilon_a\|_2^2 + w_{o'} \|\epsilon_\theta^{o'} - \epsilon_{o'}\|_2^2\right]$$

where $t_a, t_{o'} \sim U(0,T)$ independently, and $w_a$, $w_{o'}$ are weights that balance the two objectives.

This decoupled timestep design directly enables all four inference modes by controlling the noise level of each modality at inference time:

| Mode | $t_a$ | $t_{o'}$ | Output |
|------|--------|-----------|--------|
| Policy | T (noise) | 0 (clean) | $p(a \mid o)$ |
| Video prediction | 0 (clean) | T (noise) | $p(o' \mid o)$ |
| Forward dynamics | 0 (clean) | T (noise) | $p(o' \mid o, a)$ |
| Inverse dynamics | T (noise) | 0 (clean) | $p(a \mid o, o')$ |

During policy inference the observation is set to pure noise ($t_{o'} = T$) and actions are denoised iteratively via the reverse process:

$$a_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(a_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} s_\theta(o, a_t, o'_T, t, T)\right) + \sigma_t \delta_t$$

#### Module 2: Register Tokens

**Design Motivation**: Cross-modal feature sharing without explicit cross-attention — learnable tokens act as intermediaries, attending to both action and observation streams and providing a shared feature pool.

**Implementation**:
- Learnable tokens concatenated into both modality streams
- Attend to all tokens in both streams via self-attention
- Empirically demonstrated to be critical for cross-modal consistency

#### Module 3: Action-Free Co-Training

**Implementation**:
- Action-labeled: 2,000 DROID trajectories with $(o, a, o')$ tuples
- Action-free: 2,000 DROID trajectories with $(o, o')$ only — $t_a = T$ throughout (action is pure noise)
- Single joint training objective handles both data types via timestep conditioning

---

## Experiments

### Datasets

| Dataset | Scale | Characteristics | Usage |
|---------|-------|----------------|-------|
| DROID (action-labeled) | 2,000 trajectories | Franka manipulation | Pretraining |
| DROID (action-free) | 2,000 trajectories | Same platform, no actions | Co-training |
| Real robot | 5 tasks | Stack-Bowls, Block-Cabinet, Paper-Towel, Hang-Towel, Rice-Cooker | Evaluation |
| LIBERO-100 | 90 train + 10 eval | Tabletop manipulation | Evaluation |

### Results

The following table compares in-distribution and OOD success rates on four real-robot manipulation tasks, where UWM Co uses co-training with action-free video and UWM Pre is pretrained only.

**Table 1: Real Robot Tasks (In-Distribution / OOD)**

| Task | UWM Co | UWM Pre | DP | PAD | GR-1 |
|------|--------|---------|-----|-----|------|
| Stack-Bowls | 0.92/0.84 | 0.86/0.76 | 0.48/0.36 | 0.08/0.08 | 0.66/0.48 |
| Block-Cabinet | 0.84/0.72 | 0.76/0.60 | 0.60/0.26 | 0.00/0.00 | 0.66/0.44 |
| Paper-Towel | 0.86/0.84 | 0.78/0.78 | 0.52/0.48 | 0.42/0.34 | 0.60/0.60 |
| Hang-Towel | 0.86/0.76 | 0.82/0.64 | 0.64/0.28 | 0.52/0.30 | 0.66/0.48 |

UWM Co (co-trained with action-free video) substantially outperforms DP on all tasks; especially OOD — 0.84 vs. 0.36 on Stack-Bowls.

**Table 2: LIBERO OOD**

| Method | OOD Success ± std |
|--------|------------------|
| GR-1 | 0.58 ± 0.14 |
| PAD | 0.57 ± 0.19 |
| Diffusion Policy | 0.71 ± 0.12 |
| **UWM** | **0.79 ± 0.11** |

UWM achieves +8% over Diffusion Policy on LIBERO OOD with lower variance (0.11 vs. 0.19 PAD).

The following table isolates OOD robustness on Stack-Bowls with lighting, background, and clutter variations.

**Table 3: OOD Robustness (Stack-Bowls)**

| Method | Successes/30 trials |
|--------|---------------------|
| Diffusion Policy | 12/30 |
| **UWM Co** | **21/30** |

UWM is significantly more robust to visual distribution shifts.

### Implementation Details

- **Backbone**: Shared DiT with AdaLN timestep conditioning
- **Observation Encoder**: ResNet-18 → n_embd
- **Image Latents**: Frozen SDXL VAE; 224×224 → 28×28×4; patchifier (4,4,2)
- **Register Tokens**: Learnable, concatenated into both modality streams
- **Pretraining**: 100K steps; uniform sampling from mixed data
- **Fine-tuning**: 10K steps (LIBERO); varies per task (real robot)
- **Loss**: $w_a$, $w_{o'}$ weighted; independently sampled $t_a, t_{o'} \sim U(0,T)$
- **Inference**: Policy mode ($t_{o'} = T$); reverse diffusion for actions

---

## Critical Analysis

### Strengths
1. Decoupled timesteps are an elegant solution enabling one model to serve as policy, world model, and dynamics model.
2. Register tokens provide efficient cross-modal exchange without adding architectural complexity.
3. Strong OOD generalization (21/30 vs. 12/30 Stack-Bowls) demonstrates practical robustness benefit.

### Limitations
1. **Small pretraining dataset**: Only 4,000 DROID trajectories — much smaller than GR-2 (38M clips) or Cosmos Policy.
2. **Training hyperparameters not specified**: LR, batch size, optimizer not disclosed — limited reproducibility.
3. **Limited task evaluation**: 5 real-world task categories; broader evaluation needed.

### Potential Improvements
1. Scale pretraining to Open X-Embodiment for broader manipulation coverage.
2. Longer prediction horizon (currently single-step $o'$) for multi-step planning.
3. Apply flexible inference modes (forward dynamics, IDM) at deployment for planning.

### Reproducibility
- [ ] Code open-sourced
- [ ] Pretrained models available
- [ ] Learning rate, batch size, optimizer not specified
- [x] Architecture described (components, register tokens, decoupled timesteps)
- [x] LIBERO, DROID publicly available

---

## Related Notes

### Compared Against
- Diffusion Policy: 0.71 LIBERO OOD, 0.36 OOD Stack-Bowls; outperformed
- [PAD](PAD.md): 0.57 LIBERO OOD; outperformed +22%
- [GR-1](GR-1.md): 0.58 LIBERO OOD; outperformed +21%

### Method Related
- Decoupled Diffusion Timesteps: Independent noise per modality enabling flexible inference
- Register Tokens: Cross-modal feature exchange in unified diffusion framework
- Action-Free Pretraining: Video co-training without action labels via timestep masking
- Forward Dynamics Model: Single model supports multiple inference modes

---

## Quick Reference Card

> [!summary] UWM (arXiv 2025)
> - **Core**: Shared DiT; ResNet-18 obs encoder; frozen SDXL VAE (224→28×28×4, patchifier 4,4,2); register tokens; AdaLN; decoupled $t_a, t_{o'} \sim U(0,T)$; $\ell = w_a\|\epsilon_\theta^a-\epsilon_a\|^2 + w_{o'}\|\epsilon_\theta^{o'}-\epsilon_{o'}\|^2$
> - **Method**: Pretrain: 100K steps DROID 2K action-labeled + 2K action-free; fine-tune: 10K steps LIBERO; co-training: $t_a=T$ for action-free data; flexible inference: policy/video/forward/inverse dynamics by setting $t$ values
> - **Results**: 0.79 LIBERO OOD (vs. 0.71 DP); 0.92/0.84 Stack-Bowls in/OOD (vs. 0.48/0.36 DP); 21/30 vs. 12/30 OOD robustness
> - **Code**: N/A

---

*Note created: 2026-05-20*
