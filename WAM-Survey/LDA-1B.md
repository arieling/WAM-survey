---
title: "LDA-1B: Scaling Latent Dynamics Action Model via Universal Embodied Data Ingestion"
method_name: "LDA-1B"
authors: [Xiaoshen Han, Jiahan Li, Shuo Feng, Baining Liu, Guojian Wang, Junliang Guo, Hang Zhao, Xiaoxiao Long]
year: 2025
venue: arXiv
tags: [robot-foundation-model, world-model, latent-dynamics, diffusion-policy, heterogeneous-data, dexterous-manipulation]
zotero_collection: 3-Robotics/WAM-Survey
image_source: online
arxiv_html: https://arxiv.org/html/2602.12215v1
created: 2026-05-20
---

# Paper Note: LDA-1B: Scaling Latent Dynamics Action Model via Universal Embodied Data Ingestion

## Metadata

| Field | Content |
|-------|---------|
| Institution | Tsinghua University, Shanghai AI Laboratory |
| Date | February 2025 |
| Project Page | N/A |
| Baselines | [[pi0.5]], [[GR00T-N1.6]], [[UWM]] |
| Links | [arXiv](https://arxiv.org/abs/2602.12215) / Code: N/A |

---

## One-Line Summary

> LDA-1B is a 1-billion-parameter robot foundation model that achieves scalable pretraining by assigning heterogeneous embodied data—ranging from actionless human videos to low-quality robot trajectories—distinct supervisory roles within a unified latent dynamics framework operating in DINO feature space.

---

## Core Contributions

1. **Universal Embodied Data Ingestion Strategy**: Rather than filtering data down to high-quality expert demonstrations, LDA-1B assigns each tier of data a distinct role: actionless human videos supervise visual forecasting only; low-quality robot trajectories train forward and inverse dynamics but not policy; high-quality trajectories train all objectives. This enables 30,000+ hours of mixed-quality data to be exploited without degrading action quality—compared to baselines that degrade when mixed with low-quality data, LDA-1B gains 10% by including the noisy 30% of demonstrations.

2. **Latent Dynamics in DINO Feature Space**: Unlike prior world models (e.g., [[UWM]]) that predict future observations in pixel-space via [[Latent Diffusion Model|VAE]] latents, LDA-1B predicts in [[DINO]] feature space. DINOv2 features are semantically structured, suppressing background noise and encoding object-level spatial relations. This prevents representation entanglement and enables consistent scaling—UWM saturates quickly while LDA-1B continues improving from 0.1B to 1B parameters and as data grows to 30k hours.

3. **EI-30K Dataset**: A new large-scale embodied interaction dataset with over 30,000 hours of standardized human and robot trajectories (8k hours real robots, 8.6k simulated, 7.2k human with actions, 10k actionless human videos), all aligned to a unified hand-centric coordinate system and converted to LeRobot format with quality annotations.

---

## Problem Background

### Problem Being Solved

Existing [[robot-foundation-model|robot foundation models]] are trained primarily via large-scale [[Behavior Cloning|behavior cloning]] (BC) on high-quality expert demonstrations. This creates a fundamental bottleneck: the supply of expert-quality teleoperation data is expensive and scarce. Meanwhile, vast quantities of lower-quality robot trajectories, simulated data, and human activity videos exist but cannot be effectively exploited by BC-centric approaches—adding low-quality data to BC training often degrades performance.

The core question is: how to scale robot pretraining to 30,000+ hours of heterogeneous, mixed-quality data while maintaining or improving downstream manipulation performance?

### Limitations of Existing Methods

- **Behavior Cloning (e.g., [[pi0.5]], RDT, InternVLA)**: Train exclusively on high-quality teleoperation data. Fundamentally limited by data quantity ceiling (~10k hours). Discards all dynamics knowledge implicit in lower-quality data. Cannot leverage actionless human videos at all.

- **Hybrid alignment methods (e.g., Being-H0, UniVLA)**: Incorporate heterogeneous data via latent action modeling or visual foresight but achieve only ~6,000 hours of effective embodied data. The alignment between heterogeneous data modalities remains shallow.

- **Pixel-space world models (e.g., [[UWM]])**: Jointly optimize policy and video generation but operate in VAE pixel-space. VAE latents entangle appearance, geometry, and dynamics at a low-level granularity, causing rapid saturation as data and model size increase. The representation capacity is wasted on redundant appearance reconstruction.

- **Unified Video Action Models (DyWA, FLARE, WorldVLA)**: Show that co-training next-state prediction improves generalization, but do not explicitly model data quality roles, limiting heterogeneous data exploitation.

### Motivation

The key insight is that heterogeneous data of varying quality is not uniformly useful for all learning objectives. Specifically:
- **Action optimality** is only relevant to policy learning — low-quality actions should not contaminate the policy objective.
- **State transitions and dynamics** can be learned from any paired observation sequence, regardless of action quality.
- **Visual forecasting** (predicting plausible next visual states) can be supervised even from actionless video.

By decoupling these objectives and assigning data to only the objectives it is appropriate for, the model can exploit the full 30k-hour data mixture. Furthermore, operating in DINO latent space rather than pixel space means each predicted token carries semantic content, enabling the dynamics model to learn object-level causal relationships rather than appearance details.

---

## Method

### Architecture Overview

LDA-1B uses a **Multi-Modal Diffusion Transformer (MM-DiT)** with the following structure:
- **Input**: Current RGB observations, language instruction, task specification (one of four objectives)
- **Visual Encoder**: Frozen [[DINO|DINOv2]] encoder — encodes current observation into semantic latent tokens
- **Language/Vision Encoder**: Frozen [[VLM]] ([[Qwen-VL|Qwen3]]) — encodes observations and language to conditioning tokens
- **Core Module**: [[MMDiT|MM-DiT]] — jointly denoises noisy action chunks and noisy future DINO visual features
- **Conditioning**: [[自适应层归一化|AdaLN]] injects VLM tokens, diffusion timestep embeddings, and task embeddings into each Transformer block
- **Output**: Denoised action sequence (delta end-effector poses + finger configuration) and/or future DINO feature predictions
- **Total parameters**: ~1B (trainable MM-DiT + action encoder/decoder; frozen VLM + DINO)

![Figure 2 — Architecture of LDA](https://arxiv.org/html/2602.12215v1/x1.png)

**Caption**: LDA jointly denoises action chunks and future visual latents under multiple co-training objectives: policy learning, forward dynamics, inverse dynamics, and visual forecasting. Conditioned on VLM tokens, diffusion timestep embeddings, and task embeddings, the model uses a multi-modal diffusion transformer where action and visual experts are decoupled but interact through shared self-attention.

---

### Universal Data Ingestion via Multi-Task Co-training

**Motivation**: Standard co-training mixes data of all quality levels for all objectives. This is harmful: if a low-quality trajectory (where the robot fails or pauses) is used for policy supervision, the policy learns suboptimal behaviors. The goal is to exploit all data without cross-contaminating learning signals.

**Design**: Four distinct training objectives are defined, each with a corresponding task embedding:

1. **Policy Learning** — supervised by high-quality demonstrations: Given $(o_t, \ell)$ → predict $a_{t+1:t+k}$
2. **Forward Dynamics** — supervised by any trajectory: Given $(o_t, a_{t+1:t+k}, \ell)$ → predict $o_{t+1:t+k}$
3. **Inverse Dynamics** — supervised by any trajectory: Given $(o_{t:t+k}, \ell)$ → predict $a_{t+1:t+k}$
4. **Visual Forecasting** — supervised by any video, even actionless: Given $o_t$ → predict $o_{t+1:t+k}$

Each data sample in the training batch is routed to only the subset of objectives appropriate to its quality tier:

| Data Type | Policy | Forward Dynamics | Inverse Dynamics | Visual Forecasting |
|-----------|--------|-----------------|------------------|--------------------|
| High-quality robot/human | ✓ | ✓ | ✓ | ✓ |
| Low-quality robot/human | ✗ | ✓ | ✓ | ✓ |
| Actionless human videos | ✗ | ✗ | ✗ | ✓ |

**Implementation**: Four learnable task embeddings (one per objective) are added to the diffusion timestep embedding to specify the active objective. Two learnable **register tokens** serve as placeholder representations when the action or visual modality is absent (e.g., for visual-only forecasting, an action register token replaces the absent action chunk in the MM-DiT input). This unified architecture supports all four input-output configurations without separate networks.

---

### Representation of Predictive Targets

**Visual Representation — DINO Latent Space**:

**Motivation**: Pixel-space VAE reconstructions require the model to predict every appearance detail of the scene, including static background, lighting, and texture — all irrelevant to learning robot dynamics. This wastes model capacity and creates representation entanglement between action-induced state changes and task-irrelevant visual variation.

**Design**: The frozen DINOv2 encoder converts each observation frame into a grid of semantic patch tokens. These tokens preserve object identity, spatial layout, and inter-object relations while being invariant to low-level appearance variations. Predicting future DINO tokens therefore corresponds to predicting semantic object-level state transitions — directly relevant to dynamics learning.

**Action Representation — Hand-Centric Unified Coordinate Frame**:

All action representations are expressed as hand-centric motion in a shared coordinate system aligned to the wrist frame:
- **Parallel-jaw grippers**: 6-DOF delta end-effector pose + 1-DOF gripper width
- **Dexterous hands (low-DOF, e.g., BrainCo 10-DOF)**: 6-DOF wrist pose + 10 finger joint angles in wrist frame
- **Dexterous hands (high-DOF, e.g., Sharpa 22-DOF)**: 6-DOF wrist pose + 22 finger joint angles in wrist frame
- **Human demonstrations**: 6-DOF wrist pose + full MANO hand parameters + camera extrinsics

By expressing all actions in the wrist coordinate frame, the model can share representations across diverse embodiments with radically different DOF counts.

![Figure 3 — Aligned End-Effector Coordinate Systems](https://arxiv.org/html/2602.12215v1/figures/unified_eef.jpg)

**Caption**: All robot and human embodiments are manually aligned to a shared end-effector coordinate frame, ensuring cross-embodiment consistency in the action space.

**Temporal Organization**:
- Visual observations sampled at **3 Hz** — reduces redundancy from temporally correlated frames, reducing computation
- Actions sampled at **10 Hz** — preserves fine-grained action dynamics within each action chunk
- [[Action Chunking]] is used: the model predicts a chunk of $k$ future actions at once, improving temporal consistency

---

### MM-DiT Architecture

**Motivation**: Action prediction and future-state prediction are two coupled outputs — conditioning the action predictor on predicted future states and vice versa should improve consistency. A joint denoising architecture achieves this through shared multi-modal self-attention.

**Design**: The MM-DiT processes action tokens and future visual feature tokens jointly:

- **Action tokens**: The noisy action chunk $a_{t_a}$ is projected via a modality-specific linear encoder to action tokens
- **Visual tokens**: The noisy future DINO features $o'_{t_o}$ are projected via a visual-specific linear encoder
- **Shared self-attention**: Both sets of tokens are concatenated and processed through shared multi-modal self-attention, enabling action-visual interaction and cross-prediction
- **Modality-specific expert layers**: FFN layers are split per modality (action expert, visual expert) to preserve modality-specific processing
- **AdaLN conditioning**: Conditioning signals (VLM tokens $c$, timestep embedding, task embedding) are injected into each transformer block via [[自适应层归一化|Adaptive Layer Normalization (AdaLN)]]
- **Cross-attention for language**: High-level semantic language guidance is provided to the model via cross-attention to VLM token sequence

The model simultaneously predicts velocity fields for both modalities:

[[Flow Matching|Flow Matching Instantiation]]:

$$
(\epsilon_a^\theta, \epsilon_o^\theta) = s_\theta(o, a_{t_a}, o'_{t_o}, t_a, t_o')
$$

**Meaning**: The denoising network $s_\theta$ takes the current observation $o$, noisy action $a_{t_a}$ at diffusion time $t_a$, noisy future visual features $o'_{t_o}$ at diffusion time $t_o'$, and predicts the velocity fields for both modalities simultaneously.

**Symbols**:
- $\epsilon_a^\theta$: predicted velocity/noise for the action modality
- $\epsilon_o^\theta$: predicted velocity/noise for the visual modality
- $o$: current observation tokens (from frozen DINOv2 + VLM)
- $a_{t_a}$: noisy action chunk at diffusion timestep $t_a$
- $o'_{t_o}$: noisy future visual features at diffusion timestep $t_o'$
- $t_a, t_o'$: independent diffusion timesteps for action and visual streams

---

### Training Objective

LDA-1B uses a [[Flow Matching|flow-matching]] objective over both action and visual modalities. The action loss and observation loss activate selectively depending on which task objectives are active for a given data sample:

[[Flow Matching|Action Flow Matching Loss]]:

$$
l_{\mathrm{action}}^\theta = \mathbb{E}_{\substack{(o_{t:t+k}, a_{t+1:t+k}, \ell) \sim \mathcal{D} \\ \tau_a \sim \mathcal{U}(0, T_\tau) \\ \epsilon_a \sim \mathcal{N}(\mathbf{0}, \mathbf{I})}} \|v_a^\theta - (\epsilon_a - a_{t+1:t+k})\|_2^2
$$

**Meaning**: Minimizes the L2 error between the predicted action velocity field $v_a^\theta$ and the ground-truth flow direction $(\epsilon_a - a_{t+1:t+k})$. Only active for data samples with valid action supervision (high-quality data for policy; any trajectory for inverse dynamics).

**Symbols**:
- $v_a^\theta$: predicted action velocity field from the model
- $\epsilon_a$: sampled noise for the action modality
- $a_{t+1:t+k}$: ground-truth action chunk
- $\tau_a$: diffusion timestep sampled uniformly from $[0, T_\tau]$

[[Flow Matching|Observation Flow Matching Loss]]:

$$
l_{\mathrm{obs}}^\theta = \mathbb{E}_{\substack{(o_{t:t+k}, a_{t+1:t+k}, \ell) \sim \mathcal{D} \\ \tau_o \sim \mathcal{U}(0, T_\tau) \\ \epsilon_o \sim \mathcal{N}(\mathbf{0}, \mathbf{I})}} \|v_o^\theta - (\epsilon_o - o_{t+1:t+k})\|_2^2
$$

**Meaning**: Minimizes the L2 error between the predicted visual velocity field and the ground-truth flow direction in DINO feature space. Active for forward dynamics, inverse dynamics, and visual forecasting objectives.

[[Flow Matching|Combined Loss]]:

$$
l^\theta = l_{\mathrm{action}}^\theta + l_{\mathrm{obs}}^\theta
$$

**Meaning**: Both losses are summed with equal weighting. The selective activation of each term per data sample type ensures that low-quality data contributes only to the appropriate objectives.

---

### Inference

At deployment, LDA-1B runs in **policy mode** only: given the current observation $o_t$ and language instruction $\ell$, it denoises the action tokens from pure Gaussian noise to produce the next action chunk $a_{t+1:t+k}$. The visual forecasting head is disabled during inference — only the action prediction pathway is active. Standard DDIM/flow-matching sampling is used with a fixed number of denoising steps. The model uses [[Action Chunking]] with temporal ensemble to smooth action execution.

---

## EI-30K Dataset

### Dataset Composition

| Data Source | Hours | Quality | Action Availability |
|-------------|-------|---------|---------------------|
| Real-world robot data | 8,030 | Mixed (50-80% expert) | Full 6-DOF + end-effector |
| Simulated robot data | 8,600 | High (scripted/RL) | Full 6-DOF |
| Human demos with actions | 7,200 | Mixed | MANO wrist + hand |
| Human videos without actions | 10,000 | N/A | None |
| **Total** | **~33,830** | Mixed | Varies |

![Figure 4 — Statistics of EI-30K](https://arxiv.org/html/2602.12215v1/figures/dataset.jpg)

**Caption**: EI-30K dataset statistics. The dataset contains over 30k hours of diverse human and robot interaction data (right), spanning varying episode lengths (left) and a rich set of manipulation tasks (center).

### Data Standardization

All data is converted to **LeRobot format** — a unified representation of observations, actions, and language annotations. This reduces engineering overhead for downstream users.

**Coordinate alignment**: All action representations are expressed in a shared hand-centric wrist frame (see above). For human data, camera extrinsics are retained to convert between camera-frame MANO pose and world-frame wrist pose.

**Quality annotation**: Each trajectory is assigned a quality label based on:
- Action accuracy (does the trajectory complete the intended task?)
- Annotation completeness (does it have language, spatial metadata, action labels?)
- Motion content (does the hand meaningfully interact with objects?)

**Language normalization**: Language annotations across heterogeneous datasets are normalized via a VLM to ensure semantic consistency (e.g., "pick up the cup" and "grasp the mug" are unified under a canonical description).

**Cleaning**: Motion segments without meaningful hand-object interaction are removed. Trajectories with corrupted sensor readings or missing modalities are excluded.

---

## Experiments

### Robot Platforms

![Figure 5 — Real-World Manipulation Demonstrations](https://arxiv.org/html/2602.12215v1/x3.png)

**Caption**: Real-world manipulation demonstrations across multiple robotic platforms and end-effectors. Galbot G1 with a Sharpa 22-DOF dexterous hand (top-left), Unitree G1 with a BrainCo 10-DOF dexterous hand (middle and bottom-left), and Galbot G1 with a two-finger gripper (right).

### Datasets and Benchmarks

| Dataset/Benchmark | Scale | Characteristics | Usage |
|-------------------|-------|-----------------|-------|
| RoboCasa-GR1 | 24 tasks, 1k traj/task | GR-1 humanoid + Fourier dexterous hands, tabletop rearrangement + articulated objects | Simulation eval |
| Agibot World (held-out) | Large-scale | Diverse real-world manipulation, used for scaling analysis | Scaling eval |
| Galbot G1 real tasks | 8 tasks | Gripper manipulation: pick & place, contact-rich, fine, long-horizon | Real-world eval |
| Unitree G1 real tasks | 5 tasks | Dexterous: 3 BrainCo (low-DOF), 2 Sharpa (high-DOF) | Dexterous eval |

### Implementation Details

- **Architecture**: MM-DiT (Multi-Modal Diffusion Transformer)
- **Model sizes evaluated**: 0.1B, 0.5B, 1B parameters (trainable components)
- **Frozen components**: VLM (Qwen3), DINOv2 encoder
- **Optimizer**: AdamW
- **Training iterations**: 400,000
- **Hardware**: 48 NVIDIA H800 GPUs
- **Compute**: 4,608 GPU hours total
- **Visual sampling rate**: 3 Hz
- **Action sampling rate**: 10 Hz
- **Fine-tuning data**: 100 teleoperated trajectories per task (naturally mixed quality: ~50-80% expert-level)
- **Fine-tuning paradigm**: Identical data regime to pretraining (quality-aware routing preserved)

### Main Results — Simulation (RoboCasa-GR1)

LDA-1B achieves the best simulation performance at 55.4%, a **35.4 percentage point improvement** over the UWM baseline (20.0%) and a **4.1 point improvement** over the GR00T baseline (51.3%) — demonstrating that DINO-based latent dynamics substantially outperforms pixel-space world models and pure BC.

![Figure 2 (Table II) — RoboCasa-GR1 Results](https://arxiv.org/html/2602.12215v1/x2.png)

**Caption**: Results on RoboCasa-GR1 showing the impact of state representation (VAE vs. DINO), model size, and MM-DiT architecture on task success rates.

| Model | Parameters | Success Rate |
|-------|-----------|-------------|
| GR00T-N1.6 | 1B | 47.6% |
| GR00T-EI30k (pretrained on EI-30K) | 1B | 51.3% |
| UWM-0.1B (VAE latents) | 0.1B | 14.2% |
| UWM-1B (VAE latents) | 1B | 19.3% |
| UWM(MM-DiT) (VAE + MM-DiT, no DINO) | 1B | 20.0% |
| LDA(DiT) (DINO + plain DiT, no MM-DiT) | ~1B | 48.9% |
| LDA-0.5B (DINO + MM-DiT) | 0.5B | 50.7% |
| **LDA-1B** (DINO + MM-DiT, full) | **1B** | **55.4%** |

**Key finding**: The single most impactful change is switching from VAE to DINO latents (UWM(MM-DiT) 20.0% → LDA(DiT) 48.9%), a **+28.9 point gain** that dwarfs the contribution of architecture (MM-DiT vs. plain DiT: +6.5 points) or scale (0.5B → 1B: +4.7 points). This validates the core hypothesis: semantically structured DINO latents are the key enabler of scalable latent dynamics learning.

### Main Results — Real-World Gripper Manipulation

All models are few-shot fine-tuned with 100 robot demonstrations per task on Galbot G1 and evaluated on 8 tasks.

![Figure 6 — Gripper Manipulation Results](https://arxiv.org/html/2602.12215v1/x4.png)

**Caption**: Success rate comparison on real-world gripper manipulation tasks (8 tasks: Pick & Place, Contact-rich, Fine, Long-horizon). LDA-1B (dark blue) consistently outperforms GR00T-N1.6 and π0.5 baselines.

Key real-world improvements over [[pi0.5]]:
- **Contact-rich manipulation** (e.g., clean rubbish): **+21%** (e.g., LDA: 35%, π0.5: 0%)
- **Long-horizon manipulation**: **+23%**
- **Overall gripper tasks**: LDA-1B consistently leads across all 8 task categories

### Main Results — Real-World Dexterous Manipulation

![Figure 7 — Dexterous Manipulation Results](https://arxiv.org/html/2602.12215v1/x4.png)

**Caption**: Success rate comparison on 5 dexterous manipulation tasks (3 low-DOF BrainCo tasks, 2 high-DOF Sharpa tasks). LDA-1B's large-scale human pretraining enables precise finger coordination from limited robot fine-tuning data.

Key dexterous results:
- **Overall dexterous tasks**: **+48%** improvement over baselines
- **Pull Nail (low-DOF BrainCo)**: LDA: ~80%, π0.5: largely failing
- **Flip Bread (high-DOF Sharpa)**: LDA: ~90%, π0.5: ~10%

The magnitude of dexterous improvement (+48%) is larger than for gripper tasks (+21-23%), consistent with the hypothesis that large-scale human hand data pretraining (7.2k hours MANO + 10k actionless human videos) provides strong latent priors for dexterous finger coordination.

### Main Results — Generalization

| Perturbation Condition | LDA-1B | GR00T-N1.6 | π0.5 |
|------------------------|--------|------------|------|
| Novel objects | **60.0%** | 40.0% | 26.7% |
| Unseen backgrounds | **60.0%** | 40.0% | 20.0% |
| OOD positions | **40.0%** | 20.0% | 6.7% |

**Key finding**: LDA-1B's generalization advantage is largest in the hardest condition (OOD positions: 40% vs. 6.7% for π0.5), suggesting that latent dynamics pretraining on diverse embodied data teaches the model to focus on object affordances rather than memorized spatial mappings.

### Data-Efficient Mixed-Quality Fine-tuning

This experiment directly validates the universal data ingestion hypothesis: does adding low-quality data help LDA-1B but hurt baselines?

| Method | Expert-Only Fine-tuning | Expert + Low-Quality | Change |
|--------|------------------------|----------------------|--------|
| π0.5 | 50-60% | ~40% | **−10 to −20 pp** |
| LDA-1B | 50-70% | 60-80% | **+10 pp** |

**Key finding**: Adding naturally-collected (mixed-quality, ~50-80% expert) data to fine-tuning **degrades π0.5** by 10-20 percentage points but **improves LDA-1B** by 10 points. This directly demonstrates that LDA-1B's quality-aware data routing enables it to extract value from imperfect demonstrations that would otherwise be discarded.

### Ablation Study

The ablation in Table II systematically evaluates the contribution of each design choice:

| Configuration | Success Rate | Delta |
|---------------|-------------|-------|
| UWM-1B (VAE + standard DiT) | 19.3% | baseline |
| UWM(MM-DiT) (VAE + MM-DiT) | 20.0% | +0.7% |
| LDA(DiT) (DINO + standard DiT) | 48.9% | **+28.9%** |
| LDA-0.5B (DINO + MM-DiT, 0.5B) | 50.7% | +30.7% |
| **LDA-1B** (DINO + MM-DiT, 1B) | **55.4%** | **+35.4%** |

**Key finding**: The DINO representation is by far the most critical component. The MM-DiT architecture provides an additional +6.5 points, and scaling to 1B provides another +4.7 points, but neither architectural change is nearly as impactful as the shift to DINO latent space.

### Scaling Analysis

![Figure 10 — Scaling Analysis](https://arxiv.org/html/2602.12215v1/figures/scaling.jpg)

**Caption**: Scaling analysis of LDA evaluated by action prediction L1 error on the held-out Agibot World test set. Top: Action prediction error decreases to 6.6 with 30k hours of training data, demonstrating effective exploitation of diverse data sources. Bottom: LDA consistently outperforms UWM across model sizes (0.1B → 1B) with increasing training data, while UWM saturates rapidly.

Four training configurations are compared as data volume increases:
1. **Policy Only** (grey): Trains only BC on action-labeled data → unstable scaling, degrades with low-quality data
2. **Policy + Visual Forecasting** (green): Adds actionless video supervision → more stable but still limited
3. **Policy + Forward/Inverse Dynamics** (brown): Adds dynamics objectives → improves robustness but exhausts labeled data
4. **Full Co-training** (blue, LDA-1B): All four objectives with quality-aware routing → consistent improvement across all data quantities

Key scaling results:
- Action prediction error reaches **6.6** after using all 30k hours of data (full co-training)
- After action-labeled trajectories are exhausted, adding 10,000 actionless videos **continues to reduce prediction error** — demonstrating that visual forecasting supervision from actionless data transfers to action prediction
- LDA-1B consistently outperforms UWM at every model size (0.1B, 0.5B, 1B), while UWM performance saturates or degrades with more data and capacity

### Latent Dynamics Visualization

![Figure 9 — Latent Forward Dynamics](https://arxiv.org/html/2602.12215v1/figures/dino.jpg)

**Caption**: Visualization of latent forward dynamics. The model generates accurate future DINO visual representations (top row) aligned with ground truth (bottom row) across time steps, capturing semantic object structure and motion dynamics including object permanence, contact continuity, and motion consistency.

### Attention Heat Maps

![Figure 11 — Action-Conditioned Attention Heat Maps](https://arxiv.org/html/2602.12215v1/figures/attn_diff.jpg)

**Caption**: Attention difference visualization (attention under active command minus "No-Op" attention). "Push Right" (top) highlights the mug's leading edge and expected trajectory. "Push Close" (bottom) concentrates attention on the contact surface. Background clutter and non-interactive regions are suppressed, indicating genuine action-conditioned visual grounding.

The attention difference maps demonstrate that LDA-1B develops **action-conditioned spatial attention**: when given a specific action command, the model attends to the exact region of the scene that will be affected by that action — the leading edge of a pushed object, the contact surface of a closing gripper. This is a qualitative indicator that latent dynamics learning teaches meaningful action-visual grounding.

---

## Critical Analysis

### Strengths

1. **Principled data quality handling**: The quality-aware routing scheme is conceptually clean and practically effective — it solves a real problem (how to use low-quality data) with a minimal mechanism (task embeddings + register tokens). The +10% gain from low-quality data (while baselines lose 10-20%) is a compelling demonstration.

2. **DINO space insight**: Predicting in DINO feature space rather than pixel space is a well-motivated choice backed by ablation evidence. The +28.9 point gain from this single change (vs. VAE) is exceptionally large and suggests this is a widely applicable principle for world model design in robotics.

3. **Strong dexterous generalization**: +48% on dexterous tasks from limited fine-tuning data is the strongest result, and the explanation (large-scale human hand pretraining provides latent priors for finger coordination) is plausible and novel.

4. **Data scaling confirmed**: The paper provides a rare demonstration in robotics that scaling data from ~10k to 30k hours continues to improve performance — and specifically shows that actionless video (10k hours) continues to reduce action prediction error even after exhausting trajectory data.

### Limitations

1. **Fixed frozen DINO features**: The DINOv2 encoder is frozen. This means the visual representation cannot be adapted to the robot's specific visual domain or fine-tuned for task-specific features. In visually novel environments, the representation may not transfer as effectively.

2. **Egocentric camera assumption**: The dataset and model are predominantly designed for wrist/head-mounted egocentric cameras. Extending to third-person or fixed-camera setups (common in warehouse/industrial robotics) is unaddressed.

3. **Coordinate frame annotation overhead**: The hand-centric coordinate alignment requires manual annotation of camera extrinsics and kinematic parameters for each embodiment. This is labor-intensive and may be a barrier to incorporating new robot platforms.

4. **No real-time inference analysis**: The paper does not discuss inference latency or frequency, which matters for contact-rich tasks requiring >10 Hz control.

5. **Limited task diversity**: Real-world evaluation covers 8 gripper + 5 dexterous tasks. Long-horizon tasks with more than ~3 sub-goals are not evaluated.

### Potential Improvements

1. **Jointly learning visual representation**: Rather than freezing DINOv2, jointly learning a task-relevant visual encoder alongside the dynamics model could yield even better scaling and domain-specific representations.

2. **Richer sensory modalities**: Incorporating proprioception, tactile feedback, or force-torque signals would enable more capable contact-rich and dexterous policies — particularly relevant given the dexterous manipulation results.

3. **Automatic data quality estimation**: The current quality annotation is human-labeled. An automatic quality estimator (e.g., trained on success/failure labels) would reduce curation overhead and enable self-improving data pipelines.

4. **Multi-camera and third-person views**: Extending to non-egocentric cameras would substantially increase the usable human video data pool.

### Reproducibility

- [ ] Code open-sourced
- [ ] Pretrained models available
- [x] Training details complete (optimizer, steps, GPU count)
- [x] Datasets: EI-30K described, but not yet publicly released

---

## Related Notes

### Based On

- [[UWM]]: LDA-1B directly extends the Unified World Model (UWM) framework by replacing VAE latents with DINO features and adding quality-aware multi-task data routing
- [[MMDiT]]: The joint denoising architecture is based on the Multi-Modal Diffusion Transformer, adapted for action + visual joint denoising
- [[DINO]]: DINOv2 serves as the frozen visual encoder and the target representation space for forward dynamics prediction
- [[Flow Matching]]: The training objective uses flow matching rather than DDPM-style score matching

### Compared Against

- [[pi0.5]]: 3B-parameter BC-trained model, serves as the strongest real-world baseline; LDA-1B outperforms on all real-world categories with 7x fewer parameters
- [[GR00T-N1.6]]: NVIDIA's 1B heterogeneous-data model; LDA-1B exceeds on both simulation (55.4% vs 51.3%) and all real-world tasks
- [[UWM]]: The closest prior world model baseline; LDA-1B demonstrates that DINO latents unlock scaling that VAE latents cannot

### Method Related

- [[Diffusion Policy]]: The action prediction head uses the same diffusion/flow-matching policy learning paradigm
- [[Action Chunking]]: Used for temporal consistency in action prediction
- [[VLM]]: Qwen3 VLM serves as the language and observation encoder, providing conditioning tokens via cross-attention
- [[Qwen-VL]]: The specific VLM backbone used (frozen) for multimodal conditioning
- [[自适应层归一化|Adaptive Layer Normalization (AdaLN)]]: Used to inject all conditioning signals (timestep, task embedding, VLM tokens) into each Transformer block

### Hardware / Data

- [[EI-30K]]: The new dataset introduced by this paper — 30k+ hours of standardized human and robot interaction data
- [[RoboCasa]]: The simulation benchmark used for main evaluation (24 tasks, GR-1 humanoid)

---

> [!summary] LDA-1B (2025)
> - **Core**: Universal embodied data ingestion routes heterogeneous data to appropriate learning objectives, enabling 30k-hour scalable pretraining
> - **Method**: MM-DiT jointly denoises actions and future DINO latents; DINO space is critical (+28.9% over VAE baseline)
> - **Results**: 55.4% on RoboCasa-GR1; +21% contact-rich, +48% dexterous, +23% long-horizon over π0.5
> - **Code**: Not yet released
