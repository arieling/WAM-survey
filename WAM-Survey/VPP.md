---
title: "Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations"
method_name: "VPP"
authors: [Yucheng Hu, Yanjiang Guo, Pengchao Wang, Xiaoyu Chen, Yen-Jen Wang, Jianke Zhang, Koushil Sreenath, Chaochao Lu, Jianyu Chen]
year: 2024
venue: ICML
tags: [video-diffusion, predictive-representation, cascaded-implicit, robot-policy, stable-video-diffusion, calvin, dexterous]
zotero_collection: 3-Robotics/WAM-Survey
image_source: online
arxiv_html: https://arxiv.org/abs/2412.14803
created: 2026-05-20
---

# Paper Note: Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations (VPP)

## Metadata

| Item | Content |
|------|---------|
| Institution | Peking University, UC Berkeley |
| Date | December 2024 |
| Project Page | N/A |
| Baselines | UniPi, MDT, SuSIE, GR-1, RoboUniview, RT-1, Diffusion Policy |
| Links | [arXiv](https://arxiv.org/abs/2412.14803) / Code: N/A |

---

## One-Sentence Summary

> VPP uses a Text-guided Video Prediction (TVP) model (SVD 1.5B fine-tuned on 375K trajectories including internet human+robot videos) as a vision encoder — extracting multi-layer upsampling features through a single forward pass — then conditions a Video Former + diffusion transformer action head on these predictive representations to achieve 4.33 CALVIN avg. task length (vs. 3.65 prior SOTA) and 68.2% MetaWorld success.

---

## Core Contributions

1. **VDM as Vision Encoder (not Generator)**: Rather than using VDMs for explicit video generation and pixel-space IDM, VPP treats the SVD U-Net as a vision encoder — extracting multi-scale predictive features from a single forward pass at a fixed noise level, avoiding expensive video sampling at inference.
2. **Text-guided Video Prediction (TVP) Model**: CLIP-conditioned SVD (1.5B) trained on 375K trajectories from internet human video, robot datasets, and self-collected data — multi-loss training weighted across dataset types.
3. **Video Former for Spatiotemporal Feature Aggregation**: Learnable tokens aggregate multi-layer upsampling features from TVP via spatial-temporal attention, producing fixed-length token sequences for action prediction.
4. **Diffusion Transformer Action Head**: Cross-attention conditioning on Video Former tokens for action chunk prediction — generates 10-step action chunks at 7–10 Hz inference on consumer RTX 4090.

---

## Problem Background

### Problem to Solve
Video world models require expensive pixel-space video generation at inference time — generating H frames then applying IDM is slow and introduces generation errors. Can the predictive features inside a video diffusion model be used directly as a vision encoder for action prediction, capturing future state information without explicit video generation?

### Limitations of Existing Methods
- UniPi, SuSIE: require full video generation at inference — expensive; generation errors propagate to actions.
- GR-1: autoregressive video prediction; slower; 57.4% MetaWorld.
- RoboUniview: 3D-aware; 3.65 CALVIN; does not leverage internet video scale.
- Standard vision encoders (ResNet, ViT): not trained on video prediction — lack predictive temporal information.

### Motivation
VDM U-Nets have learned to predict future visual states from noisy inputs — their intermediate activations naturally encode both current observations and predicted futures. By extracting these features without full denoising (single forward pass at fixed noise level), we get "predictive visual representations" that implicitly encode where things are going — providing richer action signals than standard observation encoders.

---

## Method Details

### Model Architecture

VPP has **three main components**:

1. **TVP Model** (SVD 1.5B): Vision encoder producing predictive multi-scale features
2. **Video Former**: Spatiotemporal aggregator → fixed-length token sequence
3. **Diffusion Transformer Action Head**: Action chunk generation conditioned on tokens

### Core Modules

#### Module 1: Text-Guided Video Prediction (TVP) Model

**Design Motivation**: SVD pretraining encodes physical world dynamics from internet-scale video; CLIP language conditioning enables task-relevant feature extraction; multi-dataset training covers both human and robot manipulation distributions.

**Implementation**:
- Base model: Stable Video Diffusion (SVD), 1.5B parameters
- Output resolution: 16×256×256 frames
- Language conditioning: CLIP embeddings injected via cross-attention
- Training loss: weighted combination across dataset types, where $D_H$ is internet human manipulation (Something-Something-v2, 191,642 clips), $D_R$ is internet robot data (RT-1, Bridge, BC-Z, 155,541 trajectories), and $D_C$ is self-collected (Panda arm + dexterous hand, 4,476 trajectories):

$$\mathcal L_{\text{video}} = \lambda_H \mathcal L_{D_H} + \lambda_R \mathcal L_{D_R} + \lambda_C \mathcal L_{D_C}$$

- Training: 8× A100 GPUs, 2–3 days (TVP)
- At inference: single forward pass at fixed noise level (not full denoising) → predictive features

#### Module 2: Multi-Scale Feature Extraction

**Design Motivation**: Different U-Net decoder layers encode features at different spatial scales — aggregating across layers provides richer representation than using only the final layer output.

**Implementation**:
- Extract features from multiple upsampling decoder layers of TVP U-Net
- Concatenate multi-layer features into spatiotemporal feature volume
- No video sampling at inference — single forward pass only

#### Module 3: Video Former

**Design Motivation**: Multi-scale spatiotemporal features from TVP are high-dimensional; Video Former compresses them into fixed-length token sequences that can efficiently condition the action head.

**Implementation**:
- Learnable tokens attend to spatiotemporal features via spatial-temporal attention
- Output: fixed-length token sequence $Q''$
- Ablation shows Video Former contributes +0.47 CALVIN avg. length vs. removing it

#### Module 4: Diffusion Transformer Action Head

**Design Motivation**: Diffusion-based action generation handles multimodal action distributions; cross-attention on Video Former tokens provides rich predictive conditioning.

**Implementation**:
- Architecture: Diffusion Transformer with cross-attention conditioning on $Q''$
- The diffusion transformer is trained to predict the clean action $a_0$ from the noisy action $a_k$, conditioned on language embedding $l_{\text{emb}}$ and Video Former tokens $Q''$:

$$\mathcal L_{\text{diff}}(\psi; A) = \mathbb E_{a_0, \epsilon, k} \|D_\psi(a_k, l_{\text{emb}}, Q'') - a_0\|^2$$

- Action chunk size: 10 steps
- Inference: ~140ms per frame on RTX 4090 (7–10 Hz control)
- Policy training: 4× A100, 6–12 hours, batch 64–128, lr 1×10⁻⁴–5×10⁻⁵

The standard DDPM forward process adds noise to clean frames as $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon_t$, which determines the noise level at which TVP features are extracted for the vision encoder role.

The full VPP pipeline is: observation frames → TVP (SVD U-Net, CLIP-conditioned) single forward pass → multi-layer decoder features → Video Former (learnable tokens + ST-attention) → Q'' tokens → Diffusion Transformer action head → 10-step action chunk.

---

## Experiments

### Datasets

| Dataset | Scale | Characteristics | Usage |
|---------|-------|----------------|-------|
| Something-Something-v2 | 191,642 clips | Internet human hand manipulation | TVP training ($D_H$) |
| RT-1 + Bridge + BC-Z + others | 155,541 trajectories | Internet robot manipulation | TVP training ($D_R$) |
| Self-collected | 4,476 trajectories | Franka Panda + dexterous hand | TVP training ($D_C$) |
| CALVIN ABC→D | Standard split | 4-task chaining benchmark | Evaluation |
| MetaWorld | 50 tasks | Simulation benchmark | Evaluation |

### Results

The following table benchmarks VPP on the CALVIN ABC→D long-horizon task chaining evaluation, which requires completing a sequence of four tasks in a row.

**Table 1: CALVIN ABC→D Benchmark**

| Method | Avg. Task Length |
|--------|-----------------|
| RoboUniview (prev. SOTA) | 3.65 |
| GR-1 | 3.06 |
| **VPP (Ours)** | **4.33** |
| VPP (10% data) | 3.25 |

4.33 avg. task length vs. 3.65 prior SOTA (+18.6% relative); VPP with only 10% data (3.25) exceeds full-data baselines.

**Table 2: MetaWorld (50 tasks)**

| Method | Success Rate |
|--------|-------------|
| GR-1 | 57.4% |
| **VPP (Ours)** | **68.2%** |

+10.8% over GR-1 on MetaWorld 50-task evaluation.

**Table 3: Real-World Manipulation**

| Platform | Task Type | Success Rate |
|----------|-----------|-------------|
| Franka Panda | Seen tasks | 85.6% |
| Franka Panda | Unseen tasks | 73.7% |
| Dexterous hand | Seen tasks | 74.9% |
| Dexterous hand | Unseen tasks | 60.5% |
| Tool-use tasks | General | 68% |

The following ablation shows that SVD pretraining is by far the most critical component of the system.

**Table 4: Ablation Study (CALVIN ABC→D)**

| Configuration | Avg. Task Length |
|--------------|-----------------|
| **Full VPP** | **4.33** |
| Standard ViT/ResNet encoder | 1.23–2.58 |
| Without internet data | 3.97 (−0.36) |
| Without SVD pretraining | 1.63 (−2.34) |
| Without Video Former | 3.86 (−0.47) |
| Without feature aggregation | 3.60 (−0.73) |

SVD pretraining is by far the most critical component (−2.34 without it); internet data provides +0.36 gain; standard vision encoders dramatically underperform (1.23–2.58 vs. 4.33).

### Implementation Details

- **TVP Base Model**: Stable Video Diffusion (SVD), 1.5B parameters
- **Output Resolution**: 16×256×256
- **Language Conditioning**: CLIP (cross-attention injection)
- **TVP Training**: Batch 4; LR 1e-4; 12–40 epochs; 2–3 days on 8×A100
- **Policy Training**: Batch 64–128; LR 1e-4–5e-5; 6–12 hours on 4×A100
- **Action Chunk**: 10 steps
- **Inference Latency**: ~140ms per frame (RTX 4090)
- **Control Frequency**: 7–10 Hz

---

## Critical Analysis

### Strengths
1. Single forward pass inference (not full denoising) makes TVP a practical vision encoder at inference — 7–10 Hz control on consumer hardware.
2. Internet-scale human + robot video pretraining provides broad generalization — 10% data version still outperforms full-data baselines.
3. SVD pretraining provides +2.34 avg. task length — strongest single contribution in ablation.

### Limitations
1. **Fixed noise level**: Using features from a fixed noise level may not capture the most task-relevant predictive information across all timesteps.
2. **SVD model size**: 1.5B TVP + Video Former + DiT action head — substantial compute for fine-tuning.
3. **No explicit video generation**: While efficient, the implicit predictive representations may be less interpretable than explicit video plans.

### Potential Improvements
1. Multi-noise-level feature fusion (ensemble features across multiple noise levels).
2. Distillation to smaller TVP for faster inference.
3. Online fine-tuning from robot deployment feedback.

### Reproducibility
- [ ] Code open-sourced
- [ ] Pretrained models available
- [x] Training details described (LR, batch size, epochs, hardware)
- [x] CALVIN, MetaWorld datasets publicly available

---

## Related Notes

### Based On
- Stable Video Diffusion: 1.5B VDM used as vision encoder (TVP model)
- CLIP: Language conditioning for text-guided video prediction

### Compared Against
- [GR-1](GR-1.md): Autoregressive video prediction policy; outperformed 57.4%→68.2% MetaWorld, 3.06→4.33 CALVIN
- RoboUniview: 3D-aware policy; outperformed 3.65→4.33 CALVIN
- [UniPi](UniPi.md): Pixel-space video planning; outperformed substantially

### Method Related
- Predictive Visual Representation: VDM features encode future state information
- Video Former: Spatiotemporal token compression for action conditioning
- Diffusion Transformer: Action chunk generation with cross-attention

### Hardware/Data Related
- CALVIN: Long-horizon manipulation benchmark (ABC→D)
- MetaWorld: 50-task simulation benchmark

---

## Quick Reference Card

> [!summary] VPP (ICML 2024)
> - **Core**: SVD 1.5B (CLIP-conditioned) as vision encoder — single forward pass predictive features → Video Former (ST-attention) → diffusion transformer action head (10-step chunks); 375K trajectories (191K human + 156K robot + 5K self-collected)
> - **Method**: $\mathcal L_{video} = \lambda_H \mathcal L_{D_H} + \lambda_R \mathcal L_{D_R} + \lambda_C \mathcal L_{D_C}$; TVP: batch 4, 8×A100; policy: batch 64–128, 4×A100; 140ms/frame RTX 4090
> - **Results**: 4.33 CALVIN avg. (vs. 3.65 RoboUniview); 68.2% MetaWorld (vs. 57.4% GR-1); SVD pretraining: +2.34 CALVIN
> - **Code**: N/A

---

*Note created: 2026-05-20*
