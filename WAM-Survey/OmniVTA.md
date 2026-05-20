---
title: "OmniVTA: Visuo-Tactile World Modeling for Contact-Rich Robotic Manipulation"
method_name: "OmniVTA"
authors: [Yuhang Zheng, Songen Gu, Weize Li, Yupeng Zheng, Yujie Zang, Shuai Tian, Xiang Li, Ce Hao, Chen Gao, Si Liu, Haoran Li, Yilun Chen, Shuicheng Yan, Wenchao Ding]
year: 2026
venue: arXiv 2026
tags: [visuo-tactile, world-model, cascaded-implicit, contact-rich, diffusion-policy, slow-fast-control, tactile-sensor]
zotero_collection: 3-Robotics/WAM-Survey
image_source: online
arxiv_html: https://arxiv.org/abs/2603.19201
created: 2026-05-20
---

# Paper Note: OmniVTA: Visuo-Tactile World Modeling for Contact-Rich Robotic Manipulation

## Metadata

| Item | Content |
|------|---------|
| Institution | TARS Robotics, NUS, Fudan, CASIA, Tsinghua, Zhongguancun Academy, Beihang |
| Date | March 2026 |
| Project Page | [mrsecant.github.io/OmniVTA](https://mrsecant.github.io/OmniVTA) |
| Baselines | Diffusion Policy, KineDex, ForceMimic, RDP, UVA, exUMI |
| Links | [arXiv](https://arxiv.org/abs/2603.19201) / Code: Promised |

---

## One-Sentence Summary

> OmniVTA addresses contact-rich manipulation via a four-component system: TactileVAE (3D causal conv + INR decoder) for tactile encoding, a two-stream Visuo-Tactile World Model (VTWM) diffusion transformer for joint RGB+tactile prediction, an Adaptive Fusion Policy (AFP) with Latent Tactile Differential encoding for action chunking, and a 60Hz Reflexive Latent Tactile Controller (RLTC) for corrective actions — trained on OmniViTac (21,879 trajectories, 86 tasks, 4 tactile sensor types).

---

## Core Contributions

1. **OmniViTac Dataset**: 21,879 trajectories across 86 tasks, 126 objects, 6 interaction patterns (assembly, cutting, adjustment, peeling, wiping, grasping), 4 tactile sensor types — largest multi-modal visuo-tactile robot dataset.
2. **TactileVAE**: Causal 3D-conv VAE with Implicit Neural Representation (INR) decoder modeling continuous deformation field $d(x) = D_\theta(\gamma(x), \Phi(z_t, x))$ — substantially outperforms PCA and PointNet-AE on reconstruction.
3. **Two-Stream VTWM**: Parallel visual (SD-VAE) + tactile (TactileVAE) diffusion transformer with dynamic-aware weighted loss ($\mathcal L_{VTWM} = \mathcal L_{diff} + \lambda_1 \mathcal L_{dyn} + \lambda_2 \mathcal L_{amp}$) — predicts future visual and tactile jointly.
4. **Hierarchical Slow-Fast Control**: Slow Policy (action chunks from VTWM predictions) + Fast Policy (60Hz RLTC corrective actions from real-time tactile feedback) — RLTC adds +21% on Wipe task.

---

## Problem Background

### Problem to Solve
Contact-rich manipulation (cutting, peeling, assembly) requires precise force/torque control that vision alone cannot provide. How can visuo-tactile world models provide predictive tactile feedback for planning while maintaining high-frequency closed-loop control?

### Limitations of Existing Methods
- Vision-only world models (UniPi, VPP): cannot capture contact forces — fail on contact-rich tasks.
- Diffusion Policy: 12% Wipe; no tactile; no predictive planning.
- KineDex: 40% Wipe; limited tactile integration.
- ForceMimic: 33% Wipe; joint trajectory diffusion without world model.
- RDP (Reactive Diffusion Policy): 50% Wipe — OmniVTA achieves 80%.

### Motivation
Contact-rich tasks require two timescales: slow (~15Hz visual planning) and fast (~60Hz tactile correction). A world model predicting both visual and tactile future states enables informed action planning; a separate reflex controller handles real-time contact deviations.

---

## Method Details

### Model Architecture

OmniVTA has **four main components**:

1. **TactileVAE**: Tactile encoding → latent $z_t$ with INR decoder
2. **VTWM**: Two-stream DiT → joint future visual + tactile prediction
3. **AFP**: Adaptive fusion policy with LTD encoder → action chunks
4. **RLTC**: 60Hz MLP reflex controller from real-time tactile

### Core Modules

#### Module 1: TactileVAE

**Design Motivation**: Tactile data (marker displacement fields) requires spatiotemporal compression; INR decoder enables continuous deformation modeling at arbitrary resolution.

**Implementation**:
- Encoder: 3D causal convolutions, VAE structure, downsampling factor $s = 2^M$
- Input: $H \times W \times 3$ marker displacement tensor

The TactileVAE decoder uses an Implicit Neural Representation (INR) to model continuous deformation fields at arbitrary spatial resolution. Given a query point $x$, positional encoding $\gamma(x)$, and local features $\Phi(z_t, x)$ obtained by spatially interpolating the learned latent $z_t$, the decoded deformation is:

$$d(x) = D_\theta(\gamma(x), \Phi(z_t, x))$$

This formulation enables the decoder to generalize to arbitrary spatial coordinates rather than being tied to a fixed grid resolution.

- Loss: $\mathcal L_{TacVAE} = \|d(x) - \hat d(x)\|_2^2 + \lambda_{KL} \cdot \mathcal L_{KL}$, $\lambda_{KL} = 10^{-6}$
- Training: 50 epochs, 8× A100

#### Module 2: Visuo-Tactile World Model (VTWM)

**Design Motivation**: Joint prediction of visual and tactile futures provides complete scene dynamics for policy planning — separate modality streams allow independent fine-tuning.

**Implementation**:
- Architecture: Two-stream spatial-temporal diffusion transformer
  - Visual branch: SD-VAE encoding
  - Tactile branch: TactileVAE encoding
- Conditioning: vision + tactile embeddings + action projections (2D image-plane projections of 3D end-effector)

The VTWM is trained with a dynamic-aware weighted loss that emphasizes temporally changing and high-amplitude tactile signals. Let $w_{dyn}^i$ weight tokens by temporal difference and $w_{amp}^i$ weight tokens by signal amplitude; $m_i$ is a validity mask:

$$\mathcal L_{dyn} = \mathbb E\left[\sum w_{dyn}^i \odot (1-m_i) \odot \|\epsilon_i - \epsilon_\theta(z_{o,t})_i\|_2^2\right]$$

$$\mathcal L_{amp} = \mathbb E\left[\sum w_{amp}^i \odot (1-m_i) \odot \|\epsilon_i - \epsilon_\theta(z_{o,t})\|_2^2\right]$$

$$\mathcal L_{VTWM} = \mathcal L_{diff} + \lambda_1 \mathcal L_{dyn} + \lambda_2 \mathcal L_{amp}, \quad \lambda_1=\lambda_2=1.0$$

The dynamic-aware terms ensure that the world model focuses its capacity on predicting regions where tactile signals change rapidly and where contact forces are strong — the most informative parts for control.

- Training: AdamW lr=1e-4, batch 55/GPU, 100K steps; gradient clip 0.1 after 20K steps

#### Module 3: Adaptive Visuo-Tactile Fusion Policy (AFP)

**Design Motivation**: Contact probability varies across task phases — adaptive gating selects appropriate modality weights (visual vs. tactile) dynamically.

**Latent Tactile Differential (LTD) Encoder**:

The LTD encoder constructs a joint tactile representation by concatenating the current tactile features, the world-model-predicted future tactile features, and their difference:

$$f_t = \text{concat}(f_t^c, f_t^p, f_t^p - f_t^c)$$

where $f_t^c$ is the current tactile feature, $f_t^p$ is the predicted future tactile feature from VTWM, and the difference $f_t^p - f_t^c$ acts as an anticipatory contact dynamics signal that allows the policy to adapt before contact deviations become large.

**Adaptive Fusion**:
- Contact probability $p_{contact}$ from gating network → normalized weights $W_t + W_v = 1$
- Fused representation: $f_{vt} = \text{concat}(W_v \odot f_v, W_t \odot \tilde f_t)$

**Action Chunk Generation**:

The AFP generates action chunks using diffusion policy with FiLM modulation. The denoising update follows:

$$A_{c,t-1} = \alpha_t A_{c,t} - \gamma_k \epsilon_\theta(A_{c,t}, t, f_c) + \sigma_t \mathcal N(0,I)$$

- 6 actions per chunk, interpolated to 60Hz
- Training: 250K steps, $\mathcal L_{AFP} = \mathcal L_{act} + \lambda_{ct} \mathcal L_{bce}$, $\lambda_{ct}=0.2$

#### Module 4: Reflexive Latent Tactile Controller (RLTC)

**Design Motivation**: High-frequency (60Hz) corrective actions from real-time tactile feedback compensate for errors that slow planning (15Hz) cannot handle.

**Implementation**:
- Architecture: MLP
- Input: single-frame tactile repeated M times for temporal consistency
- Output: corrective actions in TCP coordinate frame
- Training: supervised on recovery segments from abnormal tactile states
- Loss: $\mathcal L_{RLTC} = \|a_r - \hat a_r\|_2^2$
- Final actions: weighted sum of AFP chunks + RLTC corrections

---

## Experiments

### Datasets

| Dataset | Scale | Characteristics | Usage |
|---------|-------|----------------|-------|
| OmniViTac | 21,879 trajectories, 86 tasks, 126 objects | 6 interaction patterns, 4 tactile sensor types | Full training |
| Tactile samples (TactileVAE) | 1.2M samples | 20% trajectories + additional sensor interactions | TactileVAE training |

### Implementation Details

- **TactileVAE**: 3D causal conv; 50 epochs; 8×A100; $\lambda_{KL}=10^{-6}$
- **VTWM**: SD-VAE (visual) + TactileVAE (tactile); AdamW lr=1e-4; batch 55/GPU; 100K steps; clip after 20K; $\lambda_1=\lambda_2=1.0$
- **AFP**: 250K steps; $\lambda_{ct}=0.2$; 6-action chunks at 60Hz; vision 15Hz, tactile 60Hz, proprio 60Hz
- **RLTC**: MLP; trained on recovery demonstrations
- **Inference**: Vision at 15Hz, RLTC at 60Hz
- **Robot**: xArm7 + Xense tactile (60Hz, 35×20 displacement) + RealSense D435 wrist (15Hz RGB)

### Overall Task Performance

OmniVTA substantially outperforms all baselines on deformable and force-sensitive tasks, while being comparable on simpler grasping tasks:

| Task | OmniVTA | RDP | KineDex | DP |
|------|---------|-----|---------|-----|
| Wipe | **0.80** | 0.50 | 0.40 | 0.12 |
| Peel | **0.55** | 0.48 | — | — |
| Cut | **0.85** | 0.65 | — | — |
| Assembly | 0.60 | 0.60 | — | — |
| Grasp | **0.90** | 0.88 | — | — |
| Adjustment | **0.65** | 0.50 | — | — |

OmniVTA substantially outperforms on deformable/force-sensitive tasks (Wipe +60%, Cut +31%, Adjustment +30%); comparable on grasp.

### TactileVAE Reconstruction Quality

The TactileVAE with INR decoder significantly outperforms simpler alternatives for tactile signal reconstruction:

| Method | L2↓ | Cosine Similarity↑ |
|--------|-----|-------------------|
| PCA | 0.091 | 0.810 |
| PointNet-AE | 0.059 | 0.910 |
| **TactileVAE** | **0.038** | **0.930** |

TactileVAE outperforms both baselines substantially — the INR decoder is critical for precise tactile reconstruction.

### RLTC Ablation (Wipe task)

The contribution of the 60Hz RLTC reflex controller is quantified by ablating it from the full system:

| Configuration | Success Rate |
|--------------|-------------|
| OmniVTA w/o RLTC | 0.66 |
| **OmniVTA full** | **0.80** |

60Hz RLTC adds +21% on Wipe — high-frequency correction is critical for deformable surface tasks.

### Generalization

| Generalization Type | Success Rate |
|--------------------|-------------|
| Unseen object positions | 58% avg |
| Unseen cutting tool | 83% |
| Object displacement perturbation | 60% |

---

## Critical Analysis

### Strengths
1. OmniViTac dataset (21,879 trajectories, 86 tasks) is a significant contribution — first large-scale multi-sensor visuo-tactile dataset.
2. Hierarchical slow-fast control elegantly separates long-horizon planning from high-frequency tactile correction.
3. INR decoder for tactile provides continuous deformation field modeling — principled and effective.

### Limitations
1. **Tactile sensor heterogeneity**: 4 different sensor types with different resolutions — training across sensors is complex.
2. **Wrist-mounted camera only**: Single 15Hz wrist camera for visual observations — limited scene context.
3. **Two-stage policy training**: AFP (250K steps) + RLTC separate training — pipeline complexity.

### Potential Improvements
1. Unified cross-sensor tactile representation.
2. Global camera view in addition to wrist camera.
3. End-to-end joint training of AFP + RLTC.

### Reproducibility
- [x] Code and data promised (project page)
- [x] Training details described (steps, LR, batch size, hardware)
- [x] Dataset will be released (OmniViTac)

---

## Related Notes

### Based On
- Stable Diffusion VAE: Visual encoding for VTWM visual branch
- Diffusion Policy: Action chunk generation backbone for AFP

### Compared Against
- Reactive Diffusion Policy (RDP): Best prior baseline; outperformed 50%→80% on Wipe
- KineDex: Tactile-enhanced policy; outperformed 40%→80%

### Method Related
- Implicit Neural Representation: Continuous deformation field modeling in TactileVAE
- Slow-Fast Control: Hierarchical 15Hz planning + 60Hz tactile correction
- Tactile Sensing: Contact detection and force estimation for manipulation

### Hardware/Data Related
- OmniViTac: New large-scale visuo-tactile manipulation dataset
- xArm7: Robot platform for evaluation

---

## Quick Reference Card

> [!summary] OmniVTA (arXiv 2026)
> - **Core**: TactileVAE (3D causal conv + INR decoder, $d(x)=D_\theta(\gamma(x),\Phi(z_t,x))$) → Two-stream VTWM (visual SD-VAE + tactile DiT, dynamic-aware loss) → AFP (LTD encoder + adaptive gating + diffusion chunks) + RLTC (60Hz MLP reflex); OmniViTac 21,879 traj 86 tasks
> - **Method**: TactileVAE: 50ep 8×A100; VTWM: lr=1e-4 batch 55/GPU 100K steps; AFP: 250K steps $\lambda_{ct}=0.2$; vision 15Hz + RLTC 60Hz
> - **Results**: 80% Wipe (vs. 50% RDP, 40% KineDex, 12% DP); 85% Cut, 90% Grasp; +21% from RLTC; 83% unseen tool
> - **Code**: Promised

---

*Note created: 2026-05-20*
