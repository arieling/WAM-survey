---
title: "Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning"
method_name: "CosmosPolicy"
authors: [Moo Jin Kim, Yihuai Gao, Tsung-Yi Lin, Yen-Chen Lin, Yunhao Ge, Grace Lam, Percy Liang, Shuran Song, Ming-Yu Liu, Chelsea Finn, Jinwei Gu]
year: 2026
venue: arXiv
tags: [world-action-model, video-diffusion, unified-stream, robot-manipulation, model-based-planning]
zotero_collection: 3-Robotics/WAM-Survey
image_source: online
arxiv_html: https://arxiv.org/html/2601.16163v1
created: 2026-05-20
---

# Paper Note: Cosmos Policy

## Metadata

| Field | Content |
|-------|---------|
| Institution | NVIDIA, Stanford University |
| Date | January 2026 |
| Project Page | [research.nvidia.com/labs/dir/cosmos-policy/](https://research.nvidia.com/labs/dir/cosmos-policy/) |
| Baselines | pi0, pi0.5, OpenVLA-OFT, CogVLA, UniVLA, FLARE, GR00T-N1.5, Video Policy, Diffusion Policy |
| Links | [arXiv](https://arxiv.org/abs/2601.16163) / [Code](https://research.nvidia.com/labs/dir/cosmos-policy/) |

---

## One-Line Summary

> CosmosPolicy fine-tunes a pretrained 2B-parameter video [[Diffusion Policy|diffusion model]] into a robot policy by encoding actions, future states, and value estimates as latent frames in the video sequence — achieving state-of-the-art results on LIBERO (98.5%) and RoboCasa (67.1%) with no architectural changes.

---

## Core Contributions

1. **Latent Frame Injection for Multi-Modal Unification**: CosmosPolicy introduces a mechanism to encode non-image modalities (robot proprioception, action chunks, state values) as latent frames interleaved with image frames in the video diffusion sequence. This requires no modifications to the model architecture — the same [[Diffusion Transformer|DiT]] backbone processes all modalities uniformly, allowing the policy to leverage the rich spatio-temporal priors learned by the video foundation model without adding new modules or training stages.

2. **Joint Training with Auxiliary Supervision**: Rather than training a policy-only model, CosmosPolicy jointly trains three capabilities in a single model: (a) policy prediction $p(a, s', V(s') \mid s)$ from demonstration data, (b) world model prediction $p(s', V(s') \mid s, a)$ from rollout data, and (c) value function prediction $p(V(s') \mid s, a, s')$ from rollouts. This auxiliary supervision provides richer training signal — the model must simultaneously predict actions, future observations, and value estimates — reducing overfitting and improving generalization compared to single-output policy learning.

3. **Model-Based Planning via Best-of-N Sampling**: At inference time, CosmosPolicy supports model-based planning by using the same fine-tuned checkpoint (additionally trained on rollout data) as a world model. It samples multiple candidate action chunks, predicts future states and value estimates for each, and selects the highest-value trajectory. This plug-in planning module improves performance on tasks with high-precision requirements or ambiguous initial states by +12.5 points on real-world tasks, without retraining.

---

## Problem Background

### Problem Being Solved

Robot visuomotor policies must map raw visual observations to continuous action sequences — a task that requires reasoning about object geometry, contact dynamics, action multimodality, and long-horizon planning. Prior approaches either train policies from scratch (suffering from limited data efficiency) or adapt large pretrained models (but lose temporal and physical reasoning by using image-language models rather than video models). The core question is: how to adapt video generation models, which encode rich spatio-temporal priors from internet-scale video pretraining, into effective robot policies without discarding those priors through architectural redesign?

### Limitations of Existing Methods

- **[[VLA|Vision-Language-Action models]]** (e.g., [[pi0]], [[pi0.5]], [[OpenVLA-OFT]]): Fine-tuned from vision-language models that process static image-text pairs. These models lack the temporal dynamics and physical motion priors present in video models, making them weaker at reasoning about contact mechanics and motion continuity. They also typically require large-scale robot action pretraining datasets to compensate.
- **Video-based policy methods** (e.g., [[VideoPolicy]], [[UniVLA]], [[AVDC]]): Prior video-to-policy approaches require multiple training stages — first pretraining a video representation, then attaching a separate action prediction head. This multi-stage pipeline risks discarding the original video model's temporal priors and complicates training. Action decoding is often architecturally decoupled from the video generation process.
- **Traditional model-based RL** (e.g., Dyna, MBPO, TD-MPC, Dreamer): These methods maintain separate world models and policy networks with distinct learning objectives. Integration requires careful balancing of multiple loss terms and separate data pipelines, and the world models often struggle with high-dimensional visual observations.
- **Single-output diffusion policies**: [[Diffusion Policy]] fine-tuned from video models typically predict only actions, leaving future state prediction capacity unutilized — thus failing to leverage the model's natural video generation strength as auxiliary supervision.

### Motivation

The key insight is that a video generation model already "knows" how the world works — it has learned to predict plausible future frames conditioned on past frames. If robot actions, proprioceptive states, and value estimates can be encoded as latent frames in the same diffusion sequence, the model can learn to predict all of them jointly using its existing temporal attention mechanisms, with no architectural modifications. The video generation prior provides a strong initialization for future state prediction, which in turn provides auxiliary supervision signal for better action prediction and natural support for model-based planning.

---

## Method

### Architecture Overview

CosmosPolicy uses a **[[Latent Diffusion Model]]** backbone ([[Diffusion Transformer|DiT]]-based) with the following structure:

- **Input**: Multi-camera RGB images (e.g., wrist + third-person views), robot proprioception (joint positions/velocities), optional language instruction
- **Backbone**: [[Cosmos-Predict2-2B]] video diffusion transformer, Wan2.1 spatiotemporal VAE tokenizer
- **Core modules**: [[Latent Frame Injection]], joint training batch composition, [[Action Chunking|action chunk]] decoder, value head
- **Output**: Action chunks (position/velocity targets), future proprioception, future camera images, state value estimate $V(s')$
- **Total parameters**: ~2 billion (frozen VAE; trainable DiT denoiser)

![Figure 1 — Cosmos Policy Overview](https://arxiv.org/html/2601.16163v1/fig/cosmos_policy_figure1.jpeg)

**Caption**: Overview of CosmosPolicy. The system takes multi-camera observations and proprioception as input, encodes all modalities as latent frames in a unified diffusion sequence, and outputs action chunks, future states, and value estimates. The same architecture supports direct policy execution (left branch) and model-based planning (right branch) via best-of-N sampling.

### Latent Frame Injection

**Motivation**: The Cosmos-Predict2-2B backbone is a video diffusion model that operates on sequences of latent frames. To inject non-image modalities (proprioception vectors, action arrays, scalar values) without modifying the architecture, these quantities must be encoded into the same latent space as video frames. A naïve approach would add new network layers, but this would discard the model's pretrained inductive biases. Latent frame injection instead reuses the existing tokenizer and attention structure by reshaping low-dimensional vectors into latent-shaped tensors.

**Design**: The Wan2.1 spatiotemporal VAE compresses images as:

$$
(1 + T) \times H \times W \times 3 \;\longrightarrow\; \left(1 + \frac{T}{4}\right) \times \frac{H}{8} \times \frac{W}{8} \times 16
$$

producing latent volumes of shape $H' \times W' \times C'$. Non-image modalities are injected as follows:

1. **Normalization**: All scalar/vector quantities normalized to $[-1, +1]$ range based on per-channel statistics from the training dataset.
2. **Tiling**: The normalized values are duplicated (broadcast/tiled) to fill an $H' \times W' \times C'$ volume matching the image latent shape.
3. **Interleaving**: These synthetic latent frames are inserted at designated positions in the temporal sequence, interleaved with image latent frames.

For a 2-camera setup, the full latent sequence is:

| Position | Content |
|----------|---------|
| 1 | Blank placeholder (conditioning frame) |
| 2 | Robot proprioception (current) |
| 3 | Wrist camera image (current) |
| 4 | Third-person camera 1 (current) |
| 5 | Third-person camera 2 (current) |
| 6 | Action chunk $a$ |
| 7 | Future proprioception $s'$ |
| 8 | Future wrist image $s'_\text{wrist}$ |
| 9 | Future camera 1 image $s'_1$ |
| 10 | Future camera 2 image $s'_2$ |
| 11 | State value $V(s')$ |

![Figure 2 — Latent Frame Injection](https://arxiv.org/html/2601.16163v1/fig/cosmos_policy_diffusion_sequence_v2_main_version.001.jpeg)

**Caption**: Latent frame injection mechanism. Raw images pass through the VAE encoder to produce image latent frames. Non-image modalities (proprioception, actions, values) are normalized and tiled into the same latent shape, then interleaved in the temporal sequence. The [[Diffusion Transformer]] processes all frames uniformly with temporal attention, treating the full sequence as a video to denoise.

During training, the current observation frames (positions 1–5) are kept clean (noise $\sigma = 0$), while the output frames (positions 6–11) are corrupted with noise and the model learns to denoise them. This is identical to conditional video generation where the conditioning frames are the current robot observations.

![Figure 8 — Detailed Latent Sequence (Appendix)](https://arxiv.org/html/2601.16163v1/fig/cosmos_policy_diffusion_sequence_v2_detailed_for_appendix.001.jpeg)

**Caption**: Detailed view of the full latent diffusion sequence, showing exactly which positions correspond to which modalities, how blank placeholder frames are used, and how the denoising is applied to output frames only.

### Noise Schedule Modification

**Motivation**: The original Cosmos-Predict2 noise schedule uses a log-normal distribution optimized for natural video generation, where the signal-to-noise ratio at high noise levels must preserve semantic structure. However, action and value predictions benefit from a different noise distribution — particularly, exposure to very high noise levels during training helps the denoising network learn more robust features.

**Design**: CosmosPolicy uses a **hybrid noise distribution**:

- **70%** of training samples draw $\sigma$ from the original log-normal: $\log \sigma \sim \mathcal{N}(1.39, 1.2^2)$
- **30%** of samples draw $\sigma$ uniformly from $[1.0, 85.0]$

This extends coverage into the high-noise regime. Additionally, at inference time, $\sigma_\text{min}$ is changed from $0.002$ to $4.0$, eliminating the low signal-to-noise final denoising steps that add noise but provide minimal information for action/value prediction.

![Figure 9 — Noise Schedule](https://arxiv.org/html/2601.16163v1/fig/noise_schedules_figure.001.jpeg)

**Caption**: Comparison of base model noise distribution (log-normal, left) and Cosmos Policy's adjusted hybrid distribution (right). The hybrid distribution extends coverage into the $[1.0, 85.0]$ uniform range, improving the model's ability to handle action generation under very high noise levels.

### Joint Training Objective

**Motivation**: Training only on action prediction from demonstrations leaves value prediction and world model prediction capacity unused. Using the same model for all three tasks provides multi-task regularization and richer gradients, while also enabling model-based planning at inference time without a separate world model.

**Design**: The training batch is composed from two data sources in fixed proportions:

| Batch fraction | Data source | Target outputs | What is trained |
|----------------|-------------|----------------|-----------------|
| 50% | Demonstrations $(s, a, s')$ | $a, s', V(s')$ | Policy + world model + value |
| 25% | Rollouts $(s, a, s')$ | $s', V(s')$ | World model + value (given $a$) |
| 25% | Rollouts $(s, a, s')$ | $V(s')$ | Value function (given $s, a, s'$) |

The training loss is the standard [[Diffusion Policy|denoising score matching]] loss applied to the output frames, with each training sample masking a different subset of output positions:

[[Denoising Score Matching|Cosmos Policy Training Loss]]:

$$
\mathcal{L} = \mathbb{E}_{(s, a, s', V), \sigma, \epsilon}\left[\lambda(\sigma) \left\| D_\theta(x_\text{noisy}; \sigma, c) - x_\text{clean} \right\|_2^2 \right]
$$

**Meaning**: The model $D_\theta$ denoises a corrupted version of the output frames $x_\text{noisy} = x_\text{clean} + \sigma \cdot \epsilon$ back to the clean targets $x_\text{clean}$, conditioned on the clean observation frames $c$ (current images + proprioception). The loss weight $\lambda(\sigma)$ follows the Karras et al. EDM formulation. The "output frames" are masked per the batch composition above — e.g., for demonstration data, the target includes action, future state, and value frames.

**Symbols**:
- $s$: current observation (images + proprioception)
- $a$: action chunk
- $s'$: future state (images + proprioception)
- $V(s')$: scalar state value of the future state
- $\sigma$: noise level
- $\epsilon \sim \mathcal{N}(0, I)$: Gaussian noise
- $\lambda(\sigma)$: EDM loss weighting
- $D_\theta$: learned denoiser (DiT backbone)
- $c$: conditioning (clean observation frames)

The value targets $V(s')$ are computed from rollout data as discounted returns (Monte Carlo), providing a learned signal for whether the robot successfully completed the task from state $s'$.

### Inference

**Direct policy execution (fast path)**:

1. Encode current images via VAE encoder → image latent frames
2. Encode proprioception → proprioception latent frame
3. Initialize output frames (positions 6–11) with pure Gaussian noise
4. Run parallel denoising: all output frames denoised simultaneously in $K$ steps (LIBERO/RoboCasa: $K=5$; ALOHA: $K=10$)
5. Decode action chunk latent → continuous actions via VAE decoder
6. Execute first $H$ steps of the action chunk (action horizon)

**Model-based planning (slow path)**:

1. Sample $N=5$ candidate action sequences $\{a_i\}$ using the base checkpoint (direct policy; parallel decoding, $K=10$ steps)
2. For each $a_i$, use the planning checkpoint (additionally fine-tuned on rollouts) to predict $M=3$ future state samples $\{s'_{ij}\}$ via autoregressive decoding ($K=5$ steps)
3. For each $(a_i, s'_{ij})$, predict $L=5$ value samples $\{V_{ijl}\}$ ($K=5$ steps)
4. Aggregate: "majority mean" — threshold values at 50 to determine predicted success/failure majority, then average within the majority group to produce $\hat{V}(a_i)$
5. Select $a^* = \arg\max_i \hat{V}(a_i)$ and execute

Total planning time per action chunk: ~5 seconds, limiting applicability to quasi-static tasks.

---

## Experiments

### Datasets and Benchmarks

| Dataset | Scale | Characteristics | Usage |
|---------|-------|-----------------|-------|
| LIBERO | 500 demos/suite (4 suites, 10 tasks each) | Tabletop manipulation, 4 difficulty suites (Spatial, Object, Goal, Long) | Train + eval |
| RoboCasa | 50 human demos/task (subset of full 1000+ MimicGen) | Kitchen manipulation, diverse tasks | Train + eval (few-shot) |
| ALOHA (real robot) | 185 total demos (4 tasks) | Bimanual manipulation, high-precision, long-horizon | Train + eval |
| Planning rollouts | 648 rollouts | 505 from prior eval + 143 collected | Planning fine-tune |

### Implementation Details

- **Backbone**: Cosmos-Predict2-2B-Video2World (2B parameter latent diffusion transformer)
- **VAE**: Wan2.1 spatiotemporal (frozen during all fine-tuning)
- **Optimizer**: AdamW
- **Hardware (LIBERO)**: 64 H100 GPUs, 40,000 gradient steps, batch size 1920, 48 hours
- **Hardware (RoboCasa)**: 32 H100 GPUs, 45,000 gradient steps, batch size 800, 48 hours
- **Hardware (ALOHA)**: 8 H100 GPUs, 50,000 gradient steps, batch size 200, 48 hours
- **Action chunk size**: 16 timesteps (LIBERO), 32 timesteps / execute 16 (RoboCasa), 50 timesteps at 25 Hz / 2 seconds (ALOHA)
- **Language conditioning**: T5-XXL text embeddings via cross-attention
- **Final training losses (ALOHA)**: action L1=0.010, proprio=0.008, image=0.085–0.097, value=0.007

### Main Results

CosmosPolicy achieves state-of-the-art on both simulation benchmarks. On LIBERO, it reaches **98.5% average success rate** across all four suites, surpassing all prior methods including CogVLA (97.4%) and OpenVLA-OFT (97.1%). On RoboCasa, it achieves **67.1% with only 50 demonstrations**, outperforming FLARE (66.4%) and Video Policy (66.0%) which use 300 demonstrations — a 6× data efficiency advantage.

**LIBERO Results:**

| Method | Spatial | Object | Goal | Long | Average |
|--------|---------|--------|------|------|---------|
| **Cosmos Policy** | **98.1%** | **100.0%** | **98.2%** | **97.6%** | **98.5%** |
| CogVLA | 98.6% | 98.8% | 96.6% | 95.4% | 97.4% |
| OpenVLA-OFT | 97.6% | 98.4% | 97.9% | 94.5% | 97.1% |
| π0.5 | 98.8% | 98.2% | 98.0% | 92.4% | 96.9% |
| UniVLA | 96.5% | 96.8% | 95.6% | 92.0% | 95.2% |
| Video Policy | — | — | — | 94.0% | — |

**Key finding**: CosmosPolicy leads most clearly on the Long-horizon suite (97.6% vs 95.4% for CogVLA), where temporal reasoning and multi-step planning are most important — precisely the regimes where video model priors provide the most benefit.

**RoboCasa Results (50 demos vs 300 for baselines):**

| Method | Training Demos | Average SR |
|--------|----------------|------------|
| **Cosmos Policy** | **50** | **67.1%** |
| FLARE | 300 | 66.4% |
| Video Policy | 300 | 66.0% |
| GR00T-N1.5 | 300 | 64.1% |
| π0 | 300 | 62.5% |

**Key finding**: The 6× data efficiency advantage is attributable to the strong video generation prior — the model arrives at fine-tuning having already learned visual dynamics, making it far more sample-efficient than methods that train robot priors from scratch.

### Real-World ALOHA Results

The real-world evaluation uses four bimanual tasks on a physical ALOHA robot: "put X on plate" (pick-and-place under object variation), "fold shirt" (long-horizon deformable), "put candies in bowl" (multi-object with spatial uncertainty), and "put candy in ziploc bag" (high-precision, partial observability). Scoring uses a 100-point scale per task.

| Method | In-distribution | OOD | Full Average |
|--------|-----------------|-----|------|
| **Cosmos Policy** | **96.3** | **89.3** | **93.6** |
| π0.5 | 87.8 | 92.5 | 88.6 |
| π0 | 81.3 | 71.7 | 77.9 |
| OpenVLA-OFT+ | 68.3 | 51.7 | 62.0 |

**Key finding**: CosmosPolicy leads by a large margin (+5.0 points over π0.5 overall) and shows particularly strong robustness on OOD conditions — suggesting the video prior generalizes to unseen object configurations better than language-model-based VLAs.

![Figure 3 — ALOHA Robot Rollouts](https://arxiv.org/html/2601.16163v1/fig/cosmos_policy_aloha_rollouts.001.jpeg)

**Caption**: Cosmos Policy executing the four ALOHA real-world tasks. The model demonstrates precise bimanual coordination including shirt folding (deformable object, long horizon) and ziploc bag insertion (high-precision, partial occlusion). Shown are both in-distribution and OOD test conditions.

![Figure 4 — ALOHA Quantitative Results](https://arxiv.org/html/2601.16163v1/fig/aloha_task_performance_results_v2.001.jpeg)

**Caption**: Real-world ALOHA evaluation results comparing Cosmos Policy, π0.5, π0, and OpenVLA-OFT+ across four tasks and in-distribution / OOD splits. Cosmos Policy leads in 3 of 4 tasks and all aggregate metrics.

### Ablation Study

The ablation on LIBERO isolates two key components:

| Configuration | LIBERO Avg SR | Drop |
|---------------|---------------|------|
| **Cosmos Policy (full)** | **98.5%** | — |
| Without auxiliary losses (action only) | 97.0% | −1.5% |
| Trained from scratch (no video pretraining) | 94.6% | −3.9% |

**Key findings**:
- **Video pretraining matters most**: Removing the pretrained video model initialization causes a 3.9% absolute drop, confirming that the video generation prior — not just the architecture — is critical. The from-scratch baseline uses the same DiT architecture with random initialization.
- **Auxiliary supervision matters**: Removing future state and value prediction from training (predicting actions only) causes a 1.5% drop, validating that joint training with world model and value targets provides meaningful regularization and additional gradient signal beyond demonstration imitation alone.

### Model-Based Planning Analysis

Planning evaluation focuses on the two most challenging ALOHA tasks. The planner uses the learned world model (planning checkpoint) to select the best action from $N=5$ candidates.

| Configuration | Candies in bowl | Candy in ziploc |
|---------------|-----------------|-----------------|
| Cosmos Policy + planning (V(s')) | Best | Best |
| Cosmos Policy + planning (Q(s,a)) | Worse | Worse |
| Cosmos Policy (no planning) | Baseline | Baseline |

**Improvement from planning**: +12.5 points on both tasks. The $V(s')$ formulation (predict value of future state, requiring world model) significantly outperforms the $Q(s, a)$ model-free variant, which only predicts action value without simulating future states. This validates the utility of the jointly-learned world model for planning.

![Figure 6 — World Model Predictions](https://arxiv.org/html/2601.16163v1/fig/cosmos_policy_planning_rollouts.001.jpeg)

**Caption**: Comparison of world model predictions from the base Cosmos Policy checkpoint vs. the planning checkpoint (additionally fine-tuned on rollouts). The planning checkpoint produces more accurate future state predictions, enabling better value estimates for best-of-N selection.

![Figure 7 — Planning Results](https://arxiv.org/html/2601.16163v1/fig/planning_results.001.jpeg)

**Caption**: Model-based planning evaluation results on challenging initial states. The V(s') formulation (using world model to predict future state value) outperforms both the model-free Q(s,a) variant and the base policy without planning.

### Failure Mode Analysis

Analyzing failures of competing methods reveals CosmosPolicy's key advantages:

![Figure 5 — Competitor Failure Modes](https://arxiv.org/html/2601.16163v1/fig/pi05_openvlaoft_aloha_rollouts.001.jpeg)

**Caption**: Common failure modes of π0.5 and OpenVLA-OFT+ on challenging ALOHA tasks. π0.5 struggles with high-precision grasps (e.g., gripping the ziploc slider) where precise spatial localization is required. OpenVLA-OFT+ exhibits mode averaging behavior — reaching to a position between multiple candidate objects — revealing that L1-regression action heads cannot represent multimodal action distributions. CosmosPolicy's diffusion-based action generation naturally handles multi-modal distributions.

### Test Conditions and Scoring Detail

**ALOHA evaluation protocol:**

| Task | In-dist | OOD | Time Limit | Scoring |
|------|---------|-----|------------|---------|
| Put X on plate | 20 | 10 | 350 steps (14 s) | 50 pts touch + 50 pts on plate |
| Fold shirt | 12 | 8 | 1600 steps (64 s) | 10 pts/stage, −10 for incomplete |
| Put candies in bowl | 15 | 10 | 1100 steps (44 s) | 20 pts/candy placed |
| Put candy in ziploc | 20 | 6 | 1100 steps (44 s) | 20 pts/stage (5 stages) |

![Figure 10 — In-distribution Initial Conditions](https://arxiv.org/html/2601.16163v1/fig/iid_initial_positions.jpeg)

**Caption**: Sample in-distribution initial positions for the four ALOHA robot tasks, showing the range of object placements used during in-distribution testing.

![Figure 11 — OOD Initial Conditions](https://arxiv.org/html/2601.16163v1/fig/ood_initial_positions.jpeg)

**Caption**: Out-of-distribution initial conditions including unseen object types (pink shirt → unseen pink shirt variant, blue ziploc bag), different object placements, and distractor objects not seen during training.

---

## Critical Analysis

### Strengths

1. **Architectural elegance via reuse**: Latent frame injection is genuinely elegant — by treating all modalities as "video frames," CosmosPolicy requires zero architectural modification to the pretrained model. This preserves the full pretrained weight structure and ensures the temporal attention layers retain their learned video dynamics priors. Most prior work attaches external action heads (disrupting the learned representations) or requires multi-stage training pipelines.

2. **Exceptional data efficiency**: Matching or exceeding 300-demo baselines with only 50 demos on RoboCasa demonstrates strong sample efficiency from video pretraining. This matters practically — collecting robot demonstrations is expensive, so methods that leverage video priors to reduce demonstration requirements have clear real-world value.

3. **Unified policy + world model + value function**: Training a single model for all three roles enables plug-in planning without training a separate world model. This architectural unification reduces engineering complexity and ensures the world model and policy share representations, unlike modular MBRL approaches where misalignment between the two models is a known failure mode.

4. **Strong real-world performance with minimal data**: Achieving 93.6% average on real ALOHA tasks with only 185 total demonstrations across 4 tasks demonstrates robust generalization — a critical gap often poorly addressed by simulation-trained methods.

5. **Principled handling of action multimodality**: Diffusion-based action generation naturally represents multimodal distributions, avoiding the mode-averaging failure of L1-regression VLAs (observed explicitly in OpenVLA-OFT+ failures).

### Limitations

1. **Planning inference speed**: Model-based planning requires ~5 seconds per action chunk due to the large model size and multiple sampling rounds (5 actions × 3 future states × 5 values = 75 denoising passes). This restricts applicability to quasi-static tasks where the robot can pause to plan, ruling out fast dynamic manipulation or reactive responses to unexpected perturbations.

2. **Rollout data requirement for planning**: Effective planning requires substantial rollout data beyond demonstrations (648 rollouts used in experiments). Collecting rollouts requires a functional initial policy, creating a chicken-and-egg problem. The planning module provides its largest benefits late in training after significant rollout collection.

3. **Single-step planning horizon**: The current best-of-N approach evaluates one action chunk at a time. Multi-step tree search (planning several chunks ahead) is not explored, limiting the planner's foresight on tasks requiring coordinated long-horizon decisions across multiple action chunks.

4. **Partial observability and occlusion**: The ziploc bag task shows degraded performance under self-occlusion (the bag partially covers the candy being inserted). Future states are hard to predict accurately when key visual features are occluded, degrading value estimates and planning quality.

5. **Compute requirements**: Training requires 8–64 H100 GPUs for 48 hours depending on the benchmark — substantially higher than lightweight policy methods like ACT or small Diffusion Policy variants, limiting accessibility for researchers without large compute budgets.

### Potential Improvements

1. **Multi-step lookahead planning**: Extending best-of-N to tree search with rollout (MCTS-style) would allow the model to evaluate action sequences several chunks ahead, potentially enabling better performance on tasks requiring coordinated multi-step reasoning like "fold shirt."

2. **Faster inference via distillation**: Applying [[Consistency Model]] or [[Adversarial Diffusion Distillation（ADD）|ADD]]-style distillation to compress the policy to 1–2 denoising steps would enable planning-level action quality at direct-policy speeds, addressing the latency bottleneck.

3. **Online rollout collection**: Integrating online rollout collection during deployment (similar to online RL) would allow the planning checkpoint to continuously improve as the robot operates, potentially enabling self-supervised adaptation to new environments.

### Reproducibility

- [x] Code open-sourced (training scripts at project page)
- [x] Pretrained models available (checkpoints released)
- [x] Training details complete (all hyperparameters in appendix)
- [x] Datasets accessible (LIBERO and RoboCasa are public; ALOHA demos released)

---

## Related Notes

### Based On

- [[Cosmos-Predict2-2B]]: CosmosPolicy directly fine-tunes the Cosmos-Predict2-2B-Video2World video diffusion model, treating it as the backbone. The entire value proposition rests on the temporal and physical priors encoded in this model during video pretraining.
- [[Latent Diffusion Model]]: The architecture uses a latent diffusion framework where actions and observations are represented in the compressed latent space of the Wan2.1 VAE, keeping the denoising network operating in a low-dimensional space.
- [[Action Chunking]]: CosmosPolicy predicts action chunks (sequences of future actions) rather than single-step actions, reducing compounding error and enabling smooth multi-step execution.
- [[Diffusion Policy]]: The general principle of using diffusion models for visuomotor policy learning, extended here from standalone diffusion models to large pretrained video foundations.

### Compared Against

- [[pi0]]: π0 fine-tunes a vision-language model (PaliGemma) with a separate flow-matching action head. CosmosPolicy outperforms it significantly on ALOHA (93.6 vs 77.9), attributing the gap to better temporal priors from video pretraining.
- [[pi0.5]]: π0.5 extends π0 with a larger language model backbone and large-scale pretraining on diverse robot data. Despite this additional pretraining, CosmosPolicy matches or exceeds it while using no cross-embodiment robot action pretraining dataset.
- [[OpenVLA-OFT]]: OpenVLA with Optimal Fine-Tuning, which uses a VLM backbone with parallel decoding for action prediction. CosmosPolicy substantially outperforms it (93.6 vs 62.0 on ALOHA), with OpenVLA-OFT+ showing clear mode-averaging failure on multimodal tasks.
- [[VideoPolicy]]: A recent video-based policy requiring multi-stage training. CosmosPolicy achieves comparable performance with a simpler single-stage fine-tuning approach.

### Method Related

- [[Diffusion Transformer]]: The DiT architecture is the backbone of Cosmos-Predict2-2B, enabling scalable latent diffusion with transformer-based attention over the latent sequence.
- [[世界模型|World Model]]: CosmosPolicy is simultaneously a world model — it predicts future visual states given current observations and actions. This dual role enables model-based planning without a separate architecture.
- [[Action Chunking]]: Predicting $H$-step action chunks is critical to smooth execution. CosmosPolicy uses chunks of 16–50 steps depending on the task frequency.

### Hardware / Data

- [[LIBERO]]: Standard simulation benchmark for tabletop robot manipulation, providing 4 progressive difficulty suites to evaluate policy generalization.
- [[RoboCasa]]: Kitchen manipulation benchmark emphasizing task diversity and real-world realism. Used here in a few-shot (50-demo) setting.
- [[ALOHA]]: Bimanual robot platform for dexterous manipulation. Used for real-world evaluation across 4 high-precision tasks.

---

> [!summary] CosmosPolicy (2026)
> - **Core**: Fine-tune a 2B video diffusion model into a robot policy by encoding actions, future states, and values as latent frames — no architectural changes
> - **Method**: Latent frame injection + joint policy/world-model/value training + best-of-N model-based planning
> - **Results**: LIBERO 98.5%, RoboCasa 67.1% (6× fewer demos), ALOHA real-robot 93.6%
> - **Code**: https://research.nvidia.com/labs/dir/cosmos-policy/
