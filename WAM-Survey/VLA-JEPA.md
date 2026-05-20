---
title: "VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model"
method_name: "VLA-JEPA"
authors: [Jingwen Sun, Wenyao Zhang, Zekun Qi, Shaojie Ren, Zezhi Liu, Hanxin Zhu, Guangzhong Sun, Xin Jin, Zhibo Chen]
year: 2026
venue: arXiv 2026
tags: [joint-diffusion, jepa, latent-world-model, v-jepa2, qwen3-vl, flow-matching, libero, simpler-env, robustness]
zotero_collection: 3-Robotics/WAM-Survey
image_source: online
arxiv_html: https://arxiv.org/abs/2602.10098
created: 2026-05-20
---

# Paper Note: VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model

## Metadata

| Item | Content |
|------|---------|
| Institution | University of Science and Technology of China, EPFL (inferred from authors) |
| Date | February 2026 |
| Project Page | N/A |
| Baselines | OpenVLA-OFT, π₀.₅, villa-x, LAPA |
| Links | [arXiv](https://arxiv.org/abs/2602.10098) / Code: N/A |

---

## One-Sentence Summary

> VLA-JEPA applies JEPA-style leakage-free latent world model pretraining (V-JEPA2 frozen target encoder + 12-layer autoregressive latent world model) to a Qwen3-VL-2B backbone with DiT-B flow-matching action head — pretrained on 220K Something-Something-v2 + 76K DROID — achieving 97.2% LIBERO, 79.5% LIBERO-Plus (vs. 69.6% OpenVLA-OFT), and 65.2% SimplerEnv, with human videos specifically boosting robustness.

---

## Core Contributions

1. **JEPA-Style Leakage-Free Latent World Modeling**: Target encoder (frozen V-JEPA2) produces latent representations of future frames as prediction targets; student pathway sees only current observation — prevents information leakage while learning rich dynamics priors in latent space (not pixel space).
2. **Latent Space Prediction for Robustness**: World model predicts in V-JEPA2 latent space rather than pixel space — latent representations are inherently invariant to camera motion and background variations, providing OOD robustness.
3. **Human Video Pretraining for Robustness**: Something-Something-v2 (220K) + DROID (76K) pretraining — human videos primarily enhance stability and robustness (esp. repeated grasping), rather than introducing new manipulation skills.
4. **Strong LIBERO-Plus Performance**: 79.5% vs. 69.6% OpenVLA-OFT on perturbed LIBERO tasks — demonstrates superior robustness to layout/visual perturbations from latent world model pretraining.

---

## Problem Background

### Problem to Solve
VLAs lack explicit world model pretraining that teaches them to predict future states, limiting robustness to visual distribution shifts. Pixel-space world models (like those in GR-1, GR-2) encode all visual details including irrelevant texture/background. Can JEPA-style latent world model pretraining provide robust dynamics priors without pixel-space prediction overhead?

### Limitations of Existing Methods
- GR-1/GR-2: pixel-space video prediction — sensitive to texture/background; large computational overhead.
- OpenVLA-OFT: no world model; 69.6% LIBERO-Plus (limited robustness).
- villa-x: 44.9% SimplerEnv — latent action model but no dynamics pretraining.
- LAPA: dense visual encoding without spatial attention selectivity.

### Motivation
JEPA's leakage-free architecture (target encoder produces targets, student only sees current state) enables learning rich dynamics representations without exposure to future frames — preventing trivial shortcuts. Latent space prediction naturally discards appearance noise (camera motion, background), providing inherent robustness. Human manipulation videos teach physical dynamics without requiring robot labels.

---

## Method Details

### Model Architecture

VLA-JEPA has three main components:

1. **VLM Backbone**: Qwen3-VL-2B with SigLIP-2 vision encoder; 224×224 input for VLM, 256×256 for world model encoder
2. **Latent World Model**: Frozen V-JEPA2 target encoder + 12-layer autoregressive transformer with time-causal attention
3. **Action Head**: DiT-B (16 layers, flow-matching)

**Key tokens**:
- Learnable latent action tokens: $\langle latent_i \rangle$ (repeated $K = 24/T$ times)
- Embodied action token: $\langle action \rangle$ (repeated 32 times)

### Core Modules

#### Module 1: Leakage-Free Latent World Model

**Design Motivation**: JEPA's target encoder produces supervision signals from future frames, but the student pathway only sees current observations — no future information leakage. Latent prediction avoids pixel-space reconstruction, focusing on semantically meaningful dynamics.

**Implementation**:
- **Target encoder**: V-JEPA2 (frozen) — processes future frames to produce latent targets $s_{t+k}$
- **Student pathway**: 12-layer autoregressive transformer with time-causal attention
- **Future horizon**: T=8 frames
- **Latent action tokens**: $K = 24/T$ learnable tokens encode action context for world model conditioning
- The world model is trained with a latent prediction loss that minimizes the squared distance between student-predicted latent states and those produced by the frozen V-JEPA2 target encoder over T=8 future frames:

$$\mathcal L_{WM} = \sum_{k=1}^{T} \mathbb E\left[\|\hat s_{t_k} - s_{t_k}\|^2\right]$$

where $\hat s_{t_k}$ are the student-predicted future latents and $s_{t_k}$ are the corresponding frozen V-JEPA2 target encoder outputs.

#### Module 2: Qwen3-VL-2B VLM Backbone

**Implementation**:
- Qwen3-VL-2B: language + vision understanding
- SigLIP-2 vision encoder (256×256 for world model encoder input)
- 224×224 VLM input resolution
- Embodied action token $\langle action \rangle$ (32 repetitions) for action prediction

#### Module 3: DiT-B Flow-Matching Action Head

**Implementation**:
- DiT-B architecture: 16 layers
- The action head is trained with a flow-matching loss predicting velocity from the noisy action, conditioned on world model latents $z_a$:

$$\mathcal L_{FM} = \mathbb E\left[\|v_\theta(a_t, t \mid z_a) - (a_0^H - \epsilon)\|_2^2\right]$$

- Action dimension: 7 (delta position + delta axis-angle)
- Denoising timesteps: 4

The two objectives are combined into a single joint training loss weighted by $\beta$:

$$\mathcal L = \mathcal L_{FM} + \beta \cdot \mathcal L_{WM}$$

### Two-Stage Training

**Stage 1 — JEPA Pretraining** (50K steps):
- Data: Something-Something-v2 (220K videos) + DROID (76K robot demos)
- Joint pretraining on human videos + robot data
- World model learns latent dynamics

**Stage 2 — Action-Head Fine-tuning**:
- Simulation: 30K steps on LIBERO/SimplerEnv
- Real-world: 20K steps on 100 custom demonstrations
- LR (VLM/WM): 1e-5; LR (action head): 1e-4
- Cosine schedule with linear warmup
- Batch: 256 (32 per GPU × 8 GPUs)

---

## Experiments

### Datasets

| Dataset | Scale | Characteristics | Usage |
|---------|-------|----------------|-------|
| Something-Something-v2 | 220K videos | Human manipulation, diverse dynamics | Stage 1 JEPA pretraining |
| DROID | 76K demos | Robot demonstrations | Stage 1 JEPA pretraining |
| LIBERO | ~2K demos | Tabletop manipulation | Stage 2 fine-tuning/evaluation |
| LIBERO-Plus | — | Perturbed LIBERO | Robustness evaluation |
| SimplerEnv | Fractal + BridgeV2 | Distribution shift benchmark | Evaluation |
| Real-world | 100 demos | Custom manipulation tasks | Real-world fine-tuning |

### Results

The following table shows that VLA-JEPA achieves competitive performance on standard LIBERO; its main advantage over OpenVLA-OFT emerges in the robustness evaluation.

**Table 1: LIBERO Benchmark**

| Method | LIBERO Avg. |
|--------|------------|
| π₀.₅ | 96.9% |
| OpenVLA-OFT | 97.1% |
| **VLA-JEPA** | **97.2%** |

Competitive on standard LIBERO — VLA-JEPA's advantage lies in robustness (LIBERO-Plus).

**Table 2: LIBERO-Plus (Perturbed Tasks)**

| Method | LIBERO-Plus Avg. |
|--------|-----------------|
| OpenVLA-OFT | 69.6% |
| **VLA-JEPA** | **79.5%** |

+9.9% on perturbed tasks — latent world model pretraining provides strong robustness to layout perturbations.

**Table 3: SimplerEnv Google Robot**

| Method | SimplerEnv Avg. |
|--------|----------------|
| villa-x | 44.9% |
| **VLA-JEPA** | **65.2%** |

+20.3% over villa-x on SimplerEnv — latent dynamics learning generalizes well to distribution shift.

The following ablation measures the specific contribution of human video (Something-Something-v2) pretraining.

**Table 4: Human Video Ablation**

| Configuration | LIBERO | LIBERO-Plus | Real |
|--------------|--------|-------------|------|
| Without SSv2 | Similar | Lower | Lower |
| **With SSv2** | Similar | Higher | Higher |

Human videos primarily enhance robustness and stability (LIBERO-Plus, real-world), not standard benchmark accuracy — physical dynamics priors from internet video improve out-of-distribution performance.

**Table 5: Future Horizon Ablation**

| T (frames) | Performance |
|------------|------------|
| T=4 | Lower (long-horizon tasks) |
| **T=8** | **Best** |
| T=12 | Marginal gain |

T=8 is optimal — sufficient horizon for manipulation dynamics without excessive computational cost.

### Implementation Details

- **VLM**: Qwen3-VL-2B; SigLIP-2 vision encoder (256×256 world model, 224×224 VLM)
- **World Model**: V-JEPA2 (frozen target encoder) + 12-layer autoregressive transformer (time-causal attention)
- **Action Head**: DiT-B (16 layers, flow-matching)
- **Action Latent Tokens**: $K=24/T$ learnable tokens; embodied action token ×32
- **Future Horizon**: T=8 frames
- **Stage 1**: 50K steps (SSv2 220K + DROID 76K)
- **Stage 2 (sim)**: 30K steps; LR (VLM/WM) 1e-5, LR (action head) 1e-4; cosine schedule; batch 256 (32/GPU × 8 GPUs)
- **Stage 2 (real)**: 20K steps; 100 demos
- **Denoising steps**: 4
- **Action dim**: 7 (delta position + delta axis-angle)

---

## Critical Analysis

### Strengths
1. Leakage-free JEPA pretraining provides strong robustness gains (LIBERO-Plus: +9.9%) without pixel reconstruction.
2. Latent-space prediction inherently robust to appearance changes — no explicit OOD augmentation needed.
3. Human video ablation rigorously identifies their specific contribution (robustness, not new skills).

### Limitations
1. **No code released**: Limited reproducibility.
2. **LIBERO-Plus gains not fully explained**: What specific perturbations drive the 9.9% gap?
3. **T=8 horizon limited**: Long-horizon tasks (>50 steps) may require longer prediction horizons.

### Potential Improvements
1. Larger VLM backbone (7B+) for more complex task understanding.
2. Longer prediction horizons with hierarchical latent world model.
3. Online world model adaptation during deployment for novel environments.

### Reproducibility
- [ ] Code open-sourced
- [ ] Pretrained models available
- [x] Key hyperparameters described (LR, batch, steps, T, K)
- [x] LIBERO, SSv2, DROID publicly available

---

## Related Notes

### Based On
- V-JEPA2: Frozen target encoder for leakage-free latent world model
- Qwen3-VL: 2B VLM backbone
- JEPA: Joint-Embedding Predictive Architecture — target encoder for self-supervised prediction

### Compared Against
- OpenVLA-OFT: 69.6% LIBERO-Plus; outperformed +9.9%
- villa-x: 44.9% SimplerEnv; outperformed +20.3%
- π₀.₅: 96.9% LIBERO; comparable

### Method Related
- Latent World Model: JEPA-style prediction in latent rather than pixel space
- Leakage-Free Prediction: Target encoder produces future targets; student sees only current observations
- Flow Matching: DiT-B action prediction objective
- Human Video Pretraining: SSv2 teaches robustness and physical dynamics priors

---

## Quick Reference Card

> [!summary] VLA-JEPA (arXiv 2026)
> - **Core**: Qwen3-VL-2B + SigLIP-2; V-JEPA2 frozen target encoder + 12-layer time-causal transformer world model (T=8, latent space, leakage-free); DiT-B 16-layer flow-matching action head; $\mathcal L = \mathcal L_{FM} + \beta \cdot \mathcal L_{WM}$; K=24/T latent action tokens + ×32 embodied action token
> - **Method**: Stage1: 50K steps (SSv2 220K + DROID 76K); Stage2: 30K sim (LR 1e-5/1e-4, batch 256 32/GPU×8) + 20K real (100 demos); 4 denoising steps; 7-dim action
> - **Results**: 97.2% LIBERO; 79.5% LIBERO-Plus (vs. 69.6% OFT); 65.2% SimplerEnv (vs. 44.9% villa-x); human videos → robustness not new skills
> - **Code**: N/A

---

*Note created: 2026-05-20*
