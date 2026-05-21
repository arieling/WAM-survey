---
title: "DreamDojo: A Generalist Robot World Model from Large-Scale Human Videos"
method_name: "DreamDojo"
authors: [Shenyuan Gao, William Liang, Kaiyuan Zheng, Ayaan Malik, Seonghyeon Ye, Sihyun Yu, Wei-Cheng Tseng, Yuzhu Dong, Kaichun Mo, Chen-Hsuan Lin, Qianli Ma, Seungjun Nah, Loic Magne, Jiannan Xiang, Yuqi Xie, Ruijie Zheng, Dantong Niu, You Liang Tan, K.R. Zentner, George Kurian, Suneel Indupuru, Pooya Jannaty, Jinwei Gu, Jun Zhang, Jitendra Malik, Pieter Abbeel, Ming-Yu Liu, Yuke Zhu, Joel Jang, Linxi "Jim" Fan]
year: 2026
venue: arXiv
tags: [world-action-model, video-diffusion, foundation-model, human-video-pretraining, latent-action, model-based-planning, policy-evaluation]
zotero_collection: 3-Robotics/WAM-Survey
image_source: online
arxiv_html: https://arxiv.org/html/2602.06949
created: 2026-05-21
---

# Paper Note: DreamDojo

## Metadata

| Field | Content |
|-------|---------|
| Institution | NVIDIA, UC Berkeley, and others |
| Date | February 2026 |
| Project Page | [dreamdojo-world.github.io](https://dreamdojo-world.github.io/) |
| Model Variants | DreamDojo-2B, DreamDojo-14B |
| Robot Platforms | GR-1, G1, AgiBot, YAM |
| Links | [arXiv](https://arxiv.org/abs/2602.06949) |

---

## One-Line Summary

> DreamDojo pretrains a latent video diffusion world model on 44,711 hours of egocentric human video (the DreamDojo-HV dataset) using continuous latent actions as proxy labels for unlabeled data, then adapts it to robot embodiments via post-training — achieving near-parity with ground-truth robot action labels, 10.81 FPS through distillation, and strong downstream results in policy evaluation, model-based planning, and live teleoperation.

---

## Core Contributions

1. **DreamDojo-HV: Largest World Model Pretraining Dataset**: DreamDojo assembles 43,827 hours of egocentric human video — 15× longer duration, 96× more unique skills (6,015 vs ~87 in DROID/AgiBot), and 2,000× more unique scenes than the previous largest world model training corpus. This scale enables pretraining a foundation world model with strong generalization to diverse objects, environments, and manipulation skills without requiring any robot action labels.

2. **Continuous Latent Actions for Unlabeled Video Learning**: Human video has no associated robot action labels, yet the model needs action conditioning to be useful for robot control. DreamDojo trains a VAE (32-dimensional bottleneck) that takes consecutive frame pairs and infers continuous latent action embeddings via an information bottleneck — effectively a self-supervised inverse dynamics model. These latent actions serve as unified proxy labels across both human and robot embodiments during pretraining, closing the domain gap without manual annotation. Latent actions achieve near-parity with ideal ground-truth retargeted labels (PSNR 20.913 vs 20.960).

3. **Action Conditioning Design for Video Diffusion**: DreamDojo introduces two techniques that substantially improve action-conditioned generation accuracy: (a) **relative action transformation** — rebaselining joint poses at each latent frame boundary (every 4 timesteps) so the model sees incremental rather than absolute joint positions, reducing distribution shift; (b) **chunked action injection** — concatenating 4 consecutive actions and injecting them into the corresponding latent frame, preventing temporal causality confusion. Together these add +8.8% PSNR on in-lab evaluation.

4. **Real-Time Distillation**: A two-stage distillation pipeline converts the 35-step teacher diffusion model into a 4-step student at 10.81 FPS (4× speedup, 93% PSNR retention). Warmup stage regresses student outputs to teacher ODE solutions; distillation stage uses score-based KL distribution matching. The student model can sustain stable autoregressive rollouts exceeding 1 minute on a single GPU.

---

## Problem Background

### Problem Being Solved

Robot world models that predict future video frames conditioned on actions are powerful tools for policy evaluation, planning, and data augmentation — but training them requires large-scale robot-action-labeled video, which is scarce and expensive to collect. Human video exists in vastly greater quantity and diversity but lacks robot action annotations. DreamDojo asks: can a world model be pretrained at scale on unannotated human video, then efficiently adapted to specific robot embodiments?

### Limitations of Existing Methods

- **Robot-data-only world models** (e.g., Cosmos-Predict, [DreamZero](DreamZero.md)): Limited by the scale and diversity of available robot demonstration datasets. The largest robot datasets have ~87 unique skills and ~564 unique scenes — far less than what humans routinely do across everyday life. World models trained on these data generalize poorly to novel objects, environments, and manipulation strategies.
- **Human video pretrained models without action conditioning** (e.g., video generation foundations): General video generation models can synthesize plausible future frames but have no mechanism for action-conditioned control — they cannot simulate "what happens if the robot does X" because they have no action input pathway.
- **Retargeting / embodiment transfer approaches**: Converting human motion capture or video keypoints into robot joint trajectories is annotation-heavy, imprecise, and does not scale to diverse internet video.
- **Cosmos-Predict2.5 baseline**: Achieves 18.274 PSNR on the DreamDojo-HV evaluation set without human video pretraining; DreamDojo-14B reaches 18.924 (+3.6%) through large-scale human video pretraining.

### Motivation

The key insight is that an inverse dynamics model (IDM) can extract action-like embeddings from any video without manual annotation. If these latent actions are low-dimensional enough (information bottleneck design) to disentangle irrelevant visual variation, they serve as effective conditioning signals that transfer from human to robot video with minimal domain gap. Human video then becomes free supervision for learning physical dynamics, and only a small amount of robot-labeled data is needed to adapt the learned action pathway to actual joint commands.

---

## Method

### Architecture Overview

DreamDojo is built on **Cosmos-Predict2.5**, a latent video diffusion transformer, with the **WAN2.2** spatiotemporal VAE tokenizer. Two scales are released:

| Variant | Parameters | GPU (pretraining) | Use Case |
|---------|------------|-------------------|----------|
| DreamDojo-2B | ~2B | — | Fast inference, teleoperation |
| DreamDojo-14B | ~14B | — | High-fidelity planning, evaluation |

- **Input**: RGB video sequences at 640×480 resolution, 13-frame context window
- **Action conditioning**: Chunked relative robot joint actions or continuous latent actions (pretraining)
- **Output**: Future video frame predictions (latent, decoded by WAN2.2 VAE)
- **Generative framework**: Flow matching (continuous normalizing flows)

### Continuous Latent Actions

**Motivation**: Human videos lack robot joint angle labels. To enable action-conditioned generation during pretraining, DreamDojo learns latent action representations directly from adjacent frame pairs via a self-supervised VAE.

**Design**: The latent action VAE has:
- **Encoder** $q_\phi(\hat a \mid f^t, f^{t+1})$: spatiotemporal Transformer that takes two consecutive frames and produces a 32-dimensional embedding with an information bottleneck
- **Decoder** $p_\theta(f^{t+1} \mid \hat a, f^t)$: conditions on the latent action and current frame to predict the next frame

The training objective is:

$$
\mathcal L^\text{pred} = \mathbb E\left[\log p_\theta(f^{t+1} \mid \hat a, f^t)\right] - \beta \cdot D_\text{KL}\!\left(q_\phi(\hat a \mid f^{t:t+1}) \,\|\, p(\hat a)\right)
$$

The $\beta$-VAE bottleneck forces $\hat a$ to capture only motion-relevant information, disentangling body identity, lighting, and background — which are common across human and robot video. At pretraining time, $\hat a$ is inferred online for each consecutive frame pair and used as the action conditioning signal.

During post-training on robot data, the action encoder is discarded and replaced with the target embodiment's joint action vector, with only the first layer of the action MLP reinitialized. All other weights (including the video backbone) are fine-tuned.

### Action Conditioning Design

Two design choices substantially improve generation quality when conditioning on robot joint actions:

**Relative action transformation**: Instead of conditioning on absolute joint angles, DreamDojo recomputes joint poses relative to the pose at each latent frame boundary (every 4 timesteps). This normalizes away cumulative pose drift and reduces the input distribution shift between pretraining (human latent actions) and post-training (robot joint actions), since relative increments are more embodiment-agnostic.

**Chunked action injection**: 4 consecutive action vectors are concatenated into a single conditioning chunk and injected into the corresponding latent frame. Without chunking, injecting individual per-timestep actions into latent frames creates temporal causality confusion because the spatiotemporal latent operates at 4× temporal downsampling — a single latent frame corresponds to 4 raw video frames. Chunking aligns action granularity with latent temporal resolution.

**Ablation results (GR-1 validation set)**:

| Components | PSNR | LPIPS | Counterfactual PSNR |
|-----------|------|-------|---------------------|
| Baseline (no relative, no chunked) | 16.199 | 0.315 | 19.448 |
| + Relative transformation | 16.522 | 0.304 | 19.482 |
| + Chunked injection | 17.626 | 0.267 | 20.783 |
| + Temporal consistency loss | 17.630 | 0.266 | 20.980 |

### Training Pipeline

**Phase 1 — Pretraining** (140k steps, 256 H100 GPUs):

| Data source | Hours | Sampling ratio |
|-------------|-------|----------------|
| In-lab robot data | 55 | 1× |
| EgoDex (Apple Vision Pro) | 829 | 2× |
| DreamDojo-HV (egocentric human) | 43,827 | 10× |
| **Total** | **44,711** | |

- Resolution: 640×480, 13-frame sequences
- Learning rate: $1.6 \times 10^{-4}$, batch size: 1024
- Action conditioning: latent actions for DreamDojo-HV/EgoDex, real joint actions for in-lab data

**Phase 2 — Post-training** (50k steps, 128 H100 GPUs):

- Reinitializes first layer of action MLP to match target robot's action dimension
- Fine-tunes all weights on target-embodiment demonstration data
- Batch size: 512
- Enables adaptation to GR-1, G1, AgiBot, YAM with minimal robot-specific data

**Training loss**:

$$
\mathcal L_\text{final} = \mathcal L_\text{flow} + 0.1 \cdot \mathcal L_\text{temporal}
$$

The temporal consistency loss $\mathcal L_\text{temporal}$ penalizes frame-to-frame velocity inconsistencies in the latent space:

$$
\mathcal L_\text{temporal} = \mathbb E\left[\sum_i \left\| z^{i+1} - z^i - (v^{i+1} - v^i) \right\|^2 \right]
$$

where $z^i$ are latent frames and $v^i$ are the predicted flow velocities. This regularizes temporal smoothness without additional labeled data.

### Distillation for Real-Time Inference

The full teacher model runs at 2.72 FPS (35 denoising steps) — too slow for live teleoperation. DreamDojo applies a two-stage distillation:

**Warmup stage**: The student network is trained to regress toward teacher ODE trajectory endpoints:

$$
\mathcal L_\text{warmup} = \mathbb E\left\| G_\text{student}(x_t, t) - x_0 \right\|^2
$$

**Distillation stage**: Score-based KL distribution matching, using the teacher as a reference:

$$
\nabla \mathcal L_\text{distill} = -\mathbb E\left[(s_\text{real} - s_\text{fake}) \cdot \frac{\partial G_\text{student}}{\partial \theta}\right]
$$

The student is trained on variable-length sequences (13–49 frames) with random window selection, enabling autoregressive rollout without exposure bias from fixed-length training.

**Distillation results**:

| Metric | Teacher (35 steps) | Student (4 steps) |
|--------|--------------------|-------------------|
| PSNR | 14.086 | 13.146 (93%) |
| SSIM | 0.442 | 0.379 |
| LPIPS | 0.412 | 0.485 |
| FPS | 2.72 | 10.81 (4×) |

The student maintains 93% of teacher PSNR while running at 10.81 FPS — sufficient for real-time teleoperation on a single RTX 5090.

---

## Experiments

### Evaluation Protocol

DreamDojo is evaluated on six out-of-distribution test sets covering different domain gaps:

| Eval Set | Description | Size |
|----------|-------------|------|
| In-lab Eval | Manus glove interactions, unseen objects | 25 samples |
| EgoDex Eval | Apple Vision Pro dexterous items | — |
| DreamDojo-HV Eval | Diverse human egocentric scenarios | — |
| Counterfactual Eval | Actions absent from training (patting, missed reaches) | — |
| EgoDex-novel Eval | Background-edited EgoDex sequences | 25 samples |
| DreamDojo-HV-novel Eval | Background-edited human video scenarios | 25 samples |

Automatic metrics: PSNR, SSIM, LPIPS. Human preference: 12 volunteers rate side-by-side video pairs on physics correctness (object permanence, shape consistency, contact causality) and action following.

### Action Conditioning Comparison (Table 2, In-lab Eval)

| Method | PSNR | SSIM | LPIPS |
|--------|------|------|-------|
| Without pretraining | 20.576 | 0.774 | 0.222 |
| Action-free (no action input) | 20.797 | 0.773 | 0.222 |
| Latent actions (ours) | 20.913 | 0.776 | 0.219 |
| Retargeted ground-truth (ideal) | 20.960 | 0.773 | 0.219 |

Latent actions close 97% of the gap between action-free and ideal ground-truth labels.

### Human Preference Evaluation (Table 4, OOD Scenarios)

| Comparison | Physics Win Rate | Action Following Win Rate |
|-----------|-----------------|--------------------------|
| DreamDojo-2B vs Cosmos-Predict2.5 | 62.5% | 63.45% |
| DreamDojo-14B vs Cosmos-Predict2.5 | 73.5% | 72.55% |
| DreamDojo-14B vs DreamDojo-2B | 72.5% | 65.53% |

The 14B model substantially outperforms both the 2B model and the Cosmos-Predict2.5 baseline on OOD scenarios, demonstrating that scale amplifies the benefit of large-scale human video pretraining.

### Data Mixture Effects (Table 3, DreamDojo-HV Eval)

| Configuration | PSNR |
|---------------|------|
| Cosmos-Predict2.5 baseline | 18.274 |
| DreamDojo-14B (with HV pretraining) | 18.924 (+3.6%) |
| Counterfactual action test | 20.472 → 21.087 |

The counterfactual action test (actions not present in robot training data) shows the largest relative improvement, confirming that human video pretraining specifically transfers diverse action priors that robot data alone cannot provide.

### Downstream Applications

**Policy Evaluation** (AgiBot fruit packing task):

DreamDojo generates rollouts for policy checkpoints at different training stages and ranks them. Pearson correlation with real-world task success rate: **r = 0.995**, Mean Maximum Rank Violation: 0.003. This near-perfect ranking correlation makes DreamDojo a viable substitute for physical robot evaluation during policy development.

**Model-Based Planning**:

A 5-checkpoint ensemble of DreamDojo world models generates candidate future states for a given action; a value model scores each future. Best-of-N selection with N=5 candidates:
- +17% success on high-variance policy group
- 2× improvement vs. uniform action sampling for converged policies

**Live Teleoperation**:

DreamDojo-2B (distilled student) runs at real-time speed with PICO VR controller input, on a single RTX 5090 GPU. The operator sees live world model rollouts with ~1-frame latency.

---

## Critical Analysis

### Strengths

1. **Scale of pretraining data**: The 44k-hour DreamDojo-HV dataset is 15–96× larger than prior datasets across all relevant axes. This scale qualitatively changes what the world model learns — from limited skill coverage to broad physical understanding. The 73.5% physics win rate over Cosmos-Predict2.5 on OOD scenarios validates that scale directly translates to generalization.

2. **Elegant handling of the annotation problem**: The continuous latent action approach neatly sidesteps the need for any human video annotation. By learning IDM-style action proxies with an information bottleneck VAE, DreamDojo extracts motion-relevant signals from raw video that are informative enough for conditioning yet generic enough to transfer across embodiments. The 97% gap closure to ideal ground-truth labels is a strong validation of this design.

3. **Practical downstream utility**: Policy evaluation at r = 0.995 real-world correlation and +17% planning improvement are concrete, deployable capabilities — not just generative quality improvements. A world model that reliably ranks policies and improves test-time planning is immediately useful in a real robotics development pipeline.

4. **Two-scale deployment**: Releasing both 2B (fast, teleoperation) and 14B (accurate, evaluation/planning) variants covers the latency-accuracy trade-off practically. The 4× distillation speedup to 10.81 FPS enables real-time applications that would otherwise require separate fast models.

### Limitations

1. **Uncommon action failure modes**: DreamDojo struggles with low-frequency actions outside the human video distribution (e.g., slapping, fast waving). Despite the dataset's diversity, rare actions are still underrepresented, and the model does not generalize to them reliably — a fundamental data coverage limitation.

2. **Optimistic failure prediction**: The model tends to predict higher success rates than actual robot outcomes, indicating it underrepresents failure modes. This limits the reliability of value estimates in model-based planning for tasks where failure is frequent, and means policy evaluation correlation may degrade for poorly-trained policies.

3. **Single-viewpoint only**: No native multi-view support. Real robot setups typically use multiple cameras (wrist + third-person), and fusing multi-view predictions is not addressed. This may limit the accuracy of generated rollouts when critical scene information is only visible from non-egocentric perspectives.

4. **Compute requirements**: Pretraining at 256 H100 GPUs for 140k steps is inaccessible to most labs. Post-training requires 128 H100s. Even single-GPU inference at 10.81 FPS requires an RTX 5090, limiting deployment outside of well-resourced setups.

5. **Evaluation metrics vs. task performance**: PSNR/SSIM/LPIPS measure pixel-level reconstruction quality, which may not fully capture physically meaningful accuracy (e.g., predicting correct object positions after contact). Human preference evaluation is more relevant but limited to 25–50 samples per eval set.

---

## Related Notes

### Based On

- Cosmos-Predict2.5: DreamDojo builds directly on this latent video diffusion backbone, leveraging its pretrained spatiotemporal attention and WAN2.2 VAE tokenizer. The foundation model pretraining removes the need to learn basic video dynamics from scratch.
- Flow Matching: Used as the generative framework in place of traditional DDPM/DDIM diffusion, enabling faster sampling with fewer steps.

### Compared Against

- [DreamZero](DreamZero.md): Another video diffusion WAM that achieves a 38× speedup stack. DreamDojo focuses on scaling human video pretraining rather than architectural speedup; the two approaches are complementary.
- Cosmos-Predict2.5: The direct backbone baseline. DreamDojo outperforms it by 73.5% physics win rate on OOD scenarios through large-scale human video pretraining.

### Method Related

- Inverse Dynamics Model (IDM): The continuous latent action VAE is functionally an IDM — it infers "what action" connects two frames. The information bottleneck design is the key addition over standard IDMs, forcing action-relevant compression.
- [FLARE](FLARE.md): Also learns future latent representations to condition policies; DreamDojo instead generates full video frames from those representations.
- Model-Based Planning: The best-of-N planning application (value model selects among candidate action rollouts) is architecturally similar to the planning in [CosmosPolicy](CosmosPolicy.md), but DreamDojo is used as the world model rather than being the policy itself.

---

> [!summary] DreamDojo (Feb 2026)
> - **Core**: Foundation world model pretrained on 44k hours of human egocentric video using continuous latent actions as proxy labels
> - **Key technique**: Latent action VAE (32-dim information bottleneck) + relative action transform + chunked injection + temporal consistency loss
> - **Scale**: 15× more data, 96× more skills, 2,000× more scenes than prior SOTA dataset
> - **Results**: 73.5% human preference win vs Cosmos-Predict2.5; policy eval r=0.995; +17% planning improvement
> - **Deployment**: 10.81 FPS via 4-step distillation on single RTX 5090; 1+ min stable rollouts
> - **Code**: [dreamdojo-world.github.io](https://dreamdojo-world.github.io/)
