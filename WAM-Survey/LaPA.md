---
title: "Latent Action Pretraining From Videos"
method_name: "LAPA"
authors: [Seonghyeon Ye, Joel Jang, Byeongguk Jeon, Sejune Joo, Jianwei Yang, Baolin Peng, Ajay Mandlekar, Reuben Tan, Yu-Wei Chao, Yuchen Lin, Lars Liden, Kimin Lee, Jianfeng Gao, Luke Zettlemoyer, Dieter Fox, Minjoon Seo]
year: 2024
venue: arXiv 2024
tags: [latent-action, vq-vae, pretraining, cascaded-implicit, vla, action-free, cross-embodiment, open-x]
zotero_collection: 3-Robotics/WAM-Survey
image_source: online
arxiv_html: https://arxiv.org/abs/2410.11758
created: 2026-05-20
---

# Paper Note: Latent Action Pretraining From Videos (LAPA)

## Metadata

| Item | Content |
|------|---------|
| Institution | KAIST, University of Washington, Microsoft Research, NVIDIA, Allen Institute for AI |
| Date | October 2024 |
| Project Page | N/A |
| Baselines | OpenVLA, UniPi, VPT, ActionVLA, Scratch |
| Links | [arXiv](https://arxiv.org/abs/2410.11758) / Code: N/A |

---

## One-Sentence Summary

> LAPA discovers discrete latent action representations from unlabeled internet videos via a C-ViViT VQ-VAE inverse dynamics model, then pretrains a 7B VLM (LWM-Chat-1M) to predict these latent action tokens — enabling VLA pretraining without robot action labels, achieving 50.1% real-world manipulation (vs. 43.9% OpenVLA) at 30× lower pretraining compute (272 vs. 21,500 A100-hours).

---

## Core Contributions

1. **Action-Label-Free VLA Pretraining**: Pretrains a 7B VLM on internet videos without any robot action labels — discrete latent action tokens discovered by VQ-VAE serve as pseudo-action labels for VLM pretraining.
2. **C-ViViT VQ-VAE Inverse Dynamics**: Spatial+temporal transformer encoder with NSVQ codebook (vocabulary $8^4$) compresses frame pairs $(x_t, x_{t+H})$ into discrete latent action tokens — captures inter-frame transition semantics.
3. **30× Pretraining Efficiency**: 272 H100-hours vs. 21,500 A100-hours for OpenVLA — same-scale 7B VLM pretraining but without requiring action-labeled datasets.
4. **Cross-Embodiment Transfer from Human Video**: LAPA pretrained on Something-Something-V2 (human hand manipulation, 200K videos) outperforms OpenVLA pretrained on BridgeV2 robot data on real-world manipulation — latent action space transfers across embodiments.

---

## Problem Background

### Problem to Solve
VLA pretraining requires large-scale robot demonstrations with action labels — expensive to collect (OpenVLA: 970K action-labeled trajectories, 21,500 A100-hours). Internet-scale human manipulation videos are abundant but lack robot action labels. How can VLAs be pretrained on actionless videos by discovering latent action representations?

### Limitations of Existing Methods
- OpenVLA: requires 970K action-labeled robot trajectories; 21,500 A100-hours; not scalable to internet video.
- UniPi: full video diffusion; 22% Language Table (vs. LAPA 62%) — poor action prediction from pixel-space plans.
- VPT: requires ground-truth action labels for IDM training; not applicable to internet video.
- ActionVLA: same backbone as LAPA but uses ground-truth actions — LAPA approaches it with no labels.

### Motivation
Inter-frame visual transitions in videos implicitly encode "what happened" — the VQ-VAE discovers a discrete codebook that captures these transitions as latent actions. A VLM pretrained to predict these latent tokens learns to understand the relationship between language instructions, visual observations, and action semantics — enabling efficient transfer to robot-specific fine-tuning with few demonstrations.

---

## Method Details

### Model Architecture

LAPA has a **three-stage pipeline**:

1. **Stage 1 — Latent Action Quantization**: C-ViViT encoder + VQ-VAE → discrete latent action codebook
2. **Stage 2 — Latent Pretraining**: LWM-Chat-1M (7B VLM) predicts latent action tokens from videos
3. **Stage 3 — Action Fine-tuning**: Replace latent action head with robot action head; fine-tune on few labeled demonstrations

### Core Modules

#### Module 1: C-ViViT VQ-VAE (Stage 1)

**Design Motivation**: VQ-VAE learns a discrete codebook of visual transition patterns — each token compactly represents the "action" that transforms $x_t$ into $x_{t+H}$. Discrete tokens are compatible with VLM text-token prediction architectures.

**Implementation**:
- Encoder: C-ViViT variant — spatial transformer + temporal transformer processes frame pair $(x_t, x_{t+H})$
- Decoder: Spatial transformer reconstructs $x_{t+H}$ from $x_t$ and latent action tokens
- Codebook: NSVQ (Normalized Soft Vector Quantization) prevents gradient collapse; maximizes codebook utilization
- Vocabulary: $8^4$ = 4,096 discrete latent action codes (default)
- Window size $H$: frames-apart lookahead; robust across 0.6–2.4 second ranges

The quantization objective combines frame reconstruction with codebook alignment:

$$\mathcal{L}_{VQ} = \mathcal{L}_{recon} + \mathcal{L}_{codebook}$$

$\mathcal{L}_{recon}$ trains the VQ-VAE to reconstruct $x_{t+H}$ from $x_t$ and the quantized latent action $z_t$; $\mathcal{L}_{codebook}$ enforces codebook alignment via the NSVQ commitment and codebook losses, preventing gradient collapse and maximizing codebook utilization.

The discrete latent action token is produced by quantizing the C-ViViT encoder output from the frame pair:

$$z_t = \text{Quantize}(\text{C-ViViT}(x_t, x_{t+H}))$$

Each $z_t$ compactly represents the visual transition from frame $x_t$ to frame $x_{t+H}$ — serving as a pseudo-action label for VLM pretraining without requiring any ground-truth robot action annotations.

#### Module 2: VLM Latent Action Pretraining (Stage 2)

**Design Motivation**: A large VLM pretrained to predict latent action tokens learns the mapping from (observation, language) → action semantics — the same function needed for robot control, but without robot-specific action labels.

**Implementation**:
- Backbone: LWM-Chat-1M (Large World Model), 7B parameters
- Latent labeling: C-ViViT encoder from Stage 1 labels all video frames with latent action tokens
- VLM head: single MLP layer predicts latent action tokens from VLM hidden states
- Training: vision encoder frozen; language model parameters unfrozen
- Datasets: BridgeV2 (60K), Open-X (970K), Something-Something V2 (200K)
- Compute: 8× H100 GPUs, 34 hours (272 H100-hours total)

#### Module 3: Action Fine-tuning (Stage 3)

**Design Motivation**: Latent action tokens from Stage 2 cannot be directly executed — a small fine-tuning stage maps VLM representations to continuous robot-executable actions with minimal labeled data.

**Implementation**:
- Replace MLP latent action head with new action head for continuous end-effector deltas
- Action space: discretized continuous actions with equal-bin distribution per dimension
- Fine-tuning: LoRA (low-rank adaptation), batch size 32
- Minimal labeled data: 100–7K downstream trajectories (varies by task)
- Vision encoder frozen; language model unfrozen

---

## Experiments

### Datasets

| Dataset | Scale | Characteristics | Usage |
|---------|-------|----------------|-------|
| BridgeV2 | 60K trajectories | Real kitchen manipulation (no action labels for pretraining) | Latent pretraining |
| Open-X | 970K trajectories | Diverse robot manipulation (no labels for pretraining) | Latent pretraining |
| Something-Something V2 | 200K clips | Internet human hand manipulation | Latent pretraining ($D_H$) |
| Language Table | 181K sim / 1K–7K fine-tune | Simulated tabletop | Stage 2 pretraining + Stage 3 fine-tune |
| SIMPLER | 100 traj fine-tune | BridgeV2 sim benchmark | Fine-tuning eval |
| Real-world | 450 traj/task | Tabletop manipulation, Franka | Fine-tuning + eval |

### Real-World Tabletop Manipulation (54 rollouts)

The key result demonstrates that action-label-free pretraining surpasses fully-supervised OpenVLA at dramatically lower compute cost:

| Method | Pretraining Data | Compute | Success Rate |
|--------|-----------------|---------|-------------|
| **LAPA (Open-X)** | Open-X (no labels) | 272 H100-hrs | **50.1%** |
| OpenVLA | Open-X (970K labeled) | 21,500 A100-hrs | 43.9% |

LAPA achieves +6.22% over OpenVLA at 30× lower compute.

### Language Table Benchmark

On the Language Table benchmark, LAPA without action labels approaches ActionVLA (which uses ground-truth labels) on in-domain and cross-task settings, while showing a larger gap on cross-environment generalization:

| Method | In-Domain | Cross-Task | Cross-Env |
|--------|-----------|-----------|----------|
| **LAPA** | **62.0%** | **73.2%** | **33.6%** |
| ActionVLA (GT actions) | 77.0% | 77.0% | 64.8% |
| UniPi | 22.0% | 20.8% | 13.6% |

LAPA without action labels approaches ActionVLA (GT labels) on in-domain and cross-task; larger gap on cross-environment generalization — latent actions capture most task information. UniPi substantially outperformed (+40%).

### Human Video Pretraining (Something-Something V2)

LAPA pretrained on SSv2 (human hand manipulation, no robot data) outperforms OpenVLA pretrained on BridgeV2 robot data on real-world manipulation — confirming that latent actions transfer across the human-to-robot embodiment gap.

### Ablation — Scaling

Across multiple ablation axes, LAPA shows expected scaling behavior:
- Model size: continuous improvement with larger parameters
- Data scaling: linear benefits with more pretraining trajectories
- Vocabulary size: task-dependent optimal ($8^4$ default)
- Window size H: robust across 0.6–2.4 seconds; degrades at extremes

### Implementation Details

- **Stage 1**: C-ViViT VQ-VAE; NSVQ codebook; vocabulary $8^4$; window size H (0.6–2.4s range)
- **Stage 2**: LWM-Chat-1M, 7B parameters; MLP latent action head; batch 128; 8×H100, 34 hours (272 H100-hours)
- **Stage 3**: LoRA fine-tuning; batch 32; vision encoder frozen; new action head for continuous end-effector deltas
- **Evaluation**: 54 real-world rollouts (success rate metric)

---

## Critical Analysis

### Strengths
1. 30× pretraining efficiency gain over OpenVLA — action-label-free pretraining is highly practical.
2. Cross-embodiment transfer from human videos — extends usable pretraining data beyond robot datasets.
3. Discrete latent action tokens naturally compatible with VLM autoregressive prediction.

### Limitations
1. **Fine-grained skill gap**: Underperforms on precision grasping — latent actions may not capture sub-centimeter motion details.
2. **Cross-environment generalization**: Substantially below ActionVLA on cross-environment (33.6% vs. 64.8%) — latent action pretraining doesn't fully replace GT action labels for generalization.
3. **Real-time inference**: 7B VLM + C-ViViT decoder introduce latency challenges.
4. **Language dependency**: Requires text descriptions of video frames for pretraining conditioning.

### Potential Improvements
1. Hierarchical latent action codebooks for finer action granularity.
2. Self-supervised language caption generation from video frames.
3. Larger codebook vocabulary for more expressive latent action space.

### Reproducibility
- [ ] Code open-sourced
- [ ] Pretrained models available
- [x] Training details described (batch size, hardware, compute budget)
- [x] Open-X, BridgeV2, SSv2 publicly accessible

---

## Related Notes

### Based On
- [[LWM-Chat-1M]]: 7B Large World Model used as VLM backbone
- [[VQ-VAE]]: Discrete latent codebook for action quantization
- [[C-ViViT]]: Spatial+temporal transformer encoder for video-level feature extraction

### Compared Against
- [[OpenVLA]]: Supervised VLA baseline; outperformed +6.22% at 30× lower compute
- [[UniPi]]: Video diffusion policy; outperformed +40% on Language Table
- [[ActionVLA]]: Same backbone with GT actions; approaches on in-domain tasks

### Method Related
- [[Latent Action Quantization]]: VQ-VAE discrete action discovery from videos
- [[Inverse Dynamics Model]]: C-ViViT encodes frame transitions as latent actions
- [[Low-Rank Adaptation]]: LoRA fine-tuning for action head adaptation

### Hardware/Data Related
- [[Open-X]]: Large-scale robot manipulation dataset (no labels needed)
- [[Something-Something V2]]: Internet human manipulation video dataset

---

## Quick Reference Card

> [!summary] LAPA (arXiv 2024)
> - **Core**: C-ViViT VQ-VAE ($8^4$ codebook, NSVQ) discovers discrete latent actions from video pairs $(x_t, x_{t+H})$; LWM-Chat-1M 7B pretrained to predict latent tokens; fine-tuned with LoRA on few labeled robot demos
> - **Method**: 3-stage: Stage 1 VQ-VAE; Stage 2 7B LWM batch 128 8×H100 34hr (272 H100-hrs); Stage 3 LoRA batch 32; window H robust 0.6–2.4s
> - **Results**: 50.1% real-world (vs. 43.9% OpenVLA); 30× lower compute; human video SSv2 pretraining transfers to robot; 62% Language Table in-domain
> - **Code**: N/A

---

*Note created: 2026-05-20*
