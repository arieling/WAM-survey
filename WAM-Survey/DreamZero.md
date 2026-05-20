---
title: "World Action Models are Zero-shot Policies"
method_name: "DreamZero"
authors: [Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, Chuning Zhu, Jiannan Xiang, Ayaan Malik, Kyungmin Lee, William Liang, Nadun Ranawaka, Jiasheng Gu, Yinzhen Xu, Guanzhi Wang, Fengyuan Hu, Avnish Narayan, Johan Bjorck, Jing Wang, Gwanghyun Kim, Dantong Niu, Ruijie Zheng, Yuqi Xie, Jimmy Wu, Qi Wang, Ryan Julian, Danfei Xu, Yilun Du, Yevgen Chebotar, Scott Reed, Jan Kautz, Yuke Zhu, Linxi Fan, Joel Jang]
year: 2026
venue: arXiv
tags: [world-action-model, video-diffusion, zero-shot, robot-manipulation, cross-embodiment, flow-matching, inverse-dynamics]
zotero_collection: 3-Robotics/WAM-Survey
image_source: online
arxiv_html: https://arxiv.org/html/2602.15922
created: 2026-05-20
---

# Paper Note: World Action Models are Zero-shot Policies

## Metadata

| Field | Content |
|-------|---------|
| Institution | NVIDIA |
| Date | February 2026 |
| Project Page | https://dreamzero0.github.io |
| Baselines | GR00T N1.6, π₀.₅, VLA from-scratch, VLA pretrained |
| Links | [arXiv](https://arxiv.org/abs/2602.15922) / [Code](https://github.com/dreamzero0/dreamzero) |

---

## One-Line Summary

> DreamZero is a 14B [[WAM|World Action Model]] built on video diffusion that jointly predicts future video frames and robot actions, achieving over 2× better zero-shot generalization than state-of-the-art [[VLA|Vision-Language-Action models]] by leveraging spatiotemporal priors from web-scale video pretraining.

---

## Core Contributions

1. **DreamZero Architecture**: A 14B [[WAM|World Action Model]] built on the Wan2.1-I2V-14B-480P [[视频生成模型|video diffusion]] backbone that jointly denoises video latents and action sequences via a single autoregressive [[DiT]] using [[Flow Matching|flow matching]]. This joint formulation enables the model to learn an implicit [[逆向动力学模型|inverse dynamics model]] (IDM) from every consecutive frame pair rather than relying solely on episode-level action labels.

2. **Zero-shot Generalization**: Over 2× improvement in generalization to unseen tasks and motions compared to state-of-the-art VLAs (39.5% vs. <1–16% task progress on AgiBot; 49% vs. 7–31% on DROID-Franka), while also outperforming on seen-task evaluation (62.2% vs. 27.4%).

3. **38× Inference Speedup via DreamZero-Flash**: A multi-level optimization stack — system-level CFG parallelism and DiT caching, implementation-level Torch Compile / CUDA Graphs / NVFP4 quantization, and a model-level decoupled noise schedule — that reduces per-chunk latency from 5.7 seconds to 150 ms, enabling real-time 7 Hz closed-loop control on 2× GB200 GPUs.

4. **Cross-Embodiment Transfer from Video-Only Data**: Using only 10–20 minutes of unlabeled video demonstrations from a different robot or a human, DreamZero achieves a 42%+ relative improvement on unseen tasks — no action labels required. A separate few-shot embodiment adaptation experiment shows that 30 minutes of "play data" suffices to transfer to an entirely new robot while retaining zero-shot generalization.

5. **Open-Source Release**: Model weights, inference code, and reproducibility scripts for RoboArena, PolaRiS, and Genie Sim 3.0 benchmarks at https://github.com/dreamzero0/dreamzero.

---

## Problem Background

### Problem Being Solved

State-of-the-art [[VLA|Vision-Language-Action (VLA) models]] achieve impressive semantic generalization — they can interpret natural language instructions and manipulate objects across different scenes — but fail when asked to execute novel *physical motions* or *manipulation skills* that are absent from their training data. A model trained on pick-and-place tasks can recognize the instruction "untie a shoelace" but cannot execute it because the spatiotemporal motor pattern was never demonstrated.

The root cause is architectural: VLAs initialize from large [[VLM|vision-language models]] pretrained on static image-text pairs. These models encode excellent representations of *what objects are* and *what task semantics mean*, but they lack the spatiotemporal priors needed to understand *how physical motions unfold over time*.

### Limitations of Existing Methods

**VLAs (e.g., π₀, GR00T N1.6, OpenVLA-OFT)**:
- Require large-scale task-specific and environment-specific action demonstrations — generalization requires collecting teleoperation data across hundreds of diverse environments for each target task.
- Even with pretraining on multi-task data, VLAs typically generalize at the object/semantic level but not at the level of novel physical motions or skills.
- Scaling model capacity alone does not resolve this: the authors show that both a 5B and a 14B VLA achieve near-zero success on diverse heterogeneous training data, because the fundamental bottleneck is the quality of the learning signal (static image pretraining), not model capacity.
- Cross-embodiment transfer requires action labels from the target embodiment, which are expensive to collect.

**Separate Video Prediction + IDM (pipeline approaches)**:
- Prior work (Du et al., Zhou et al.) separates video prediction from inverse dynamics into two independently trained models. This breaks the gradient signal between visual futures and motor commands, limiting the alignment between predicted videos and extracted actions.
- Bidirectional World Action Models require subsampling video to fixed sequence lengths, distorting the native frame rate and degrading video-action alignment with language captions.

**Latent-space World Models (Dreamer series, V-JEPA)**:
- Learn dynamics from scratch in compact latent spaces, without leveraging the rich spatiotemporal knowledge encoded in web-scale video pretraining.
- Typically require model-predictive control (MPC) at test time, which is computationally expensive for real-time robotics.

### Motivation

The key insight is that *video is a dense representation of how the physical world evolves*. Every consecutive frame pair implicitly encodes the robot's state transition and the associated motor command needed to produce it. If a model can predict realistic future video frames conditioned on a language instruction, it has implicitly learned the physics of how that task is performed — and an [[逆向动力学模型|inverse dynamics model]] can then extract the actions that produce those predicted futures.

This reframes robot learning as video generation: improving video generation quality *directly and automatically* improves policy quality, without requiring any change to the action labeling pipeline. Furthermore, video-only demonstrations (from humans or other robots) can be used as training signal without any action annotation, enabling extremely efficient cross-embodiment transfer.

---

## Method

### Architecture Overview

DreamZero uses an **autoregressive [[DiT|Diffusion Transformer]] with joint video-action denoising** built on top of the Wan2.1-I2V-14B-480P image-to-video model:

- **Input**: Visual observation history (image sequence), language instruction, proprioceptive state
- **Backbone**: Autoregressive DiT (14B parameters) using [[Flow Matching|flow matching]] with causal attention masking
- **Core modules**: Frozen VAE encoder, frozen text encoder, trainable state encoder, trainable action encoder, trainable DiT blocks, video decoder, trainable action decoder
- **Output**: Jointly denoised future video frames and corresponding action sequences
- **Total parameters**: 14 billion (backbone); encoder/decoder modules are lightweight additions

![Figure 4: Model Architecture of DreamZero](https://arxiv.org/html/2602.15922v1/x4.png)
*The model takes three inputs (visual context, language, proprioceptive state) and jointly predicts future video frames and actions via an autoregressive DiT. Training (left) denoises noisy chunks conditioned on clean context; inference (right) runs asynchronously with ground-truth observation feedback into the KV cache.*

The high-level design decomposes the joint policy as:

$$
\underbrace{\pi_{\theta}(\mathbf{o}_{l:l+H}, \mathbf{a}_{l:l+H} \mid \mathbf{o}_{0:l}, \mathbf{c}, \mathbf{q}_{l})}_{\text{DreamZero}} = \underbrace{\pi_{\theta}(\mathbf{o}_{l:l+H} \mid \mathbf{o}_{0:l}, \mathbf{c}, \mathbf{q}_{l})}_{\text{video prediction}} \cdot \underbrace{\pi_{\theta}(\mathbf{a}_{l:l+H} \mid \mathbf{o}_{0:l+H}, \mathbf{q}_{l})}_{\text{IDM}}
$$

where $\mathbf{o}_{l:l+H}$ are future video frames, $\mathbf{a}_{l:l+H}$ are future actions, $\mathbf{c}$ is the language instruction, $\mathbf{q}_{l}$ is proprioceptive state, $\mathbf{o}_{0:l}$ is the visual observation history, and $H$ is the prediction horizon.

**Crucially, DreamZero trains a single end-to-end model rather than two separate models.** The video prediction and [[逆向动力学模型|IDM]] components are jointly optimized via shared denoising timesteps, forcing deep integration between the two modalities. The predicted video serves as an implicit visual planner — the IDM head reads the denoised visual future and extracts the motor commands that produced it.

![Figure 1: Overview of DreamZero's four capabilities](https://arxiv.org/html/2602.15922v1/x1.png)
*By jointly predicting video and action, DreamZero inherits world physics priors that enable: (1) effective learning from diverse, non-repetitive data; (2) open-world generalization; (3) cross-embodiment learning from video-only data; (4) few-shot adaptation to new robots.*

### Autoregressive Chunk-wise Generation

**Motivation**: Bidirectional architectures process fixed-length sequences, requiring video subsampling that distorts native frame rates and breaks video-action alignment. An autoregressive architecture conditions each chunk on arbitrary-length history, preserving native frame rates and enabling KV-cache efficiency.

**Design**:
- The trajectory is divided into chunks of $K = 2$ latent frames each (empirically: $K = 1$ underperforms).
- Up to $M = 4$ chunks are processed per forward pass, giving a maximum context of $4 \times 2 = 8$ latent frames ≈ 33 raw frames ≈ 6.6 seconds.
- Video generation is autoregressive over chunks; action generation within each chunk is **non-autoregressive** to prevent error propagation from closed-loop action prediction.
- KV-cache is exploited across chunks: the key-value representations of all previous clean chunks are computed once and reused.

**Critical closed-loop design**: After the robot executes an action chunk, ground-truth observations are encoded via the VAE and injected *back into the KV cache*, replacing the previously predicted video latents. This eliminates the compounding error that plagues pure autoregressive video generation — the model always conditions on real observations, not hallucinated ones.

![Figure 13: Bidirectional vs. Autoregressive WAMs](https://arxiv.org/html/2602.15922v1/x15.png)
*When the sampling point falls mid-task ($T=20$), bidirectional WAMs must subsample video to align with the language caption, distorting native FPS and degrading video-action alignment. Autoregressive WAMs preserve both language-video correspondence and native frame rate.*

### Attention Masking Strategy

**Training phase**: The current noisy chunk (video latents $\mathbf{z}_{t_k}^k$ and action latents $\mathbf{a}_{t_k}^k$) attends via causal cross-attention to all clean previous chunks $\mathcal{C}_k = \{(\mathbf{z}_1^j, \mathbf{a}_1^j)\}_{j=1}^{k-1}$. Language instruction tokens attend to all positions. This is a teacher-forcing setup: the model sees clean context and must denoise the current noisy target.

**Inference phase**: KV-cache of the initial conditioning frames is computed once and concatenated for all subsequent predictions. After each action chunk executes, the KV cache is updated with ground-truth encoded observations.

![Figure 14: Attention strategy](https://arxiv.org/html/2602.15922v1/x16.png)
*(a) QKV self-attention mask during training: queries from the current noisy chunk (Z1, Z2, Z3, Y1, Y2, Y3) can attend to all conditioning frames (C0, C1, C2). (b) During inference, KV-cache of conditioning frames is concatenated to predict action and next frames.*

### Training Objective: Flow Matching with Coupled Noise

**Approach**: [[Flow Matching|Flow matching]] with a linear interpolation noise schedule (not DDPM-style denoising).

**Noise interpolation**: For each chunk $k$ with timestep $t_k \sim \mathcal{U}(0, 1)$:

$$
\mathbf{z}_{t_k}^k = t_k \mathbf{z}_1^k + (1 - t_k) \mathbf{z}_0^k, \qquad \mathbf{a}_{t_k}^k = t_k \mathbf{a}_1^k + (1 - t_k) \mathbf{a}_0^k
$$

where $\mathbf{z}_0^k \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ is Gaussian noise, $\mathbf{z}_1^k$ is the clean video latent, $\mathbf{a}_0^k \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ is action noise, and $\mathbf{a}_1^k$ is the clean normalized action. All frames within a chunk share the same timestep $t_k$; different chunks have independent timesteps.

**Key design choice**: Video and action share the same denoising timestep $t_k^{\text{video}} = t_k^{\text{action}} = t_k$. This coupled schedule accelerates training convergence compared to approaches that use separate schedules, forcing the model to develop integrated representations.

**Flow-matching loss**:

$$
\mathcal{L}(\theta) = \mathbb{E}_{\mathbf{z}, \mathbf{a}, \{t_k\}} \left[ \frac{1}{K} \sum_{k=1}^{K} w(t_k) \left\lVert \mathbf{u}_{\theta}\!\left([\mathbf{z}_{t_k}^k, \mathbf{a}_{t_k}^k]; \mathcal{C}_k, \mathbf{c}, \mathbf{q}_k, t_k\right) - \mathbf{v}^k \right\rVert^2 \right]
$$

where:
- $w(t_k) > 0$ is a predefined weighting function
- $\mathbf{u}_{\theta}$ is the joint video-action DiT that predicts the velocity field
- $\mathbf{v}^k \coloneqq [\mathbf{z}_1^k, \mathbf{a}_1^k] - [\mathbf{z}_0^k, \mathbf{a}_0^k]$ is the velocity target (clean minus noise)
- $\mathcal{C}_k = \{(\mathbf{z}_1^j, \mathbf{a}_1^j)\}_{j=1}^{k-1}$ is the clean context (teacher-forcing)

### Trainable vs. Frozen Components

| Component | Status | Rationale |
|-----------|--------|-----------|
| VAE | Frozen | Pretrained visual encoder is sufficient; fine-tuning risks catastrophic forgetting of video priors |
| Text encoder | Frozen | Pretrained language representations transferred as-is |
| State encoder | Trainable | Novel module mapping proprioceptive input to the DiT token space |
| Action encoder | Trainable | Novel module mapping normalized actions to latent space |
| DiT blocks (all) | Trainable | Core adaptation to robotic domain |
| Action decoder | Trainable | Novel module extracting actions from DiT output |

**Training configuration**:
- AgiBot: 100K steps, batch size 128
- DROID: 100K steps, batch size 128
- Actions represented as relative joint positions; idle actions filtered out

### DreamZero-Flash: Decoupled Noise Schedules for Single-Step Inference

**Problem**: Reducing denoising steps below 4 degrades action quality catastrophically (83% → 52% task progress). The root cause is a train-test mismatch: standard flow matching trains the model to predict velocity fields at all noise levels, but few-step inference requires predicting clean actions from *still-noisy video context*. The model has never seen this regime during training.

**Solution**: Decouple the noise schedules for video and action during training.

*Standard DreamZero*: $t_k^{\text{video}} = t_k^{\text{action}} = t_k$, where $t_k \sim \mathcal{U}(0, 1)$

*DreamZero-Flash*: 

$$
t_k^{\text{video}} = 1 - \eta, \quad \eta \sim \text{Beta}(\alpha, \beta), \qquad t_k^{\text{action}} \sim \mathcal{U}(0, 1)
$$

with $\alpha > \beta$ (specifically $\alpha = 7, \beta = 1$). The expected video timestep is $\mathbb{E}[t_k^{\text{video}}] = 1 - \mathbb{E}[\eta] = 1 - \frac{\alpha}{\alpha + \beta} = 1 - \frac{7}{8} = 0.125$.

**Effect**: Video timesteps are biased toward high-noise states (near pure noise), while action timesteps remain uniformly distributed. This trains the model explicitly to predict clean actions even when the visual context is heavily corrupted — precisely the regime encountered during single-step inference.

![Figure 5: Decoupled Noise Schedules](https://arxiv.org/html/2602.15922v1/x5.png)
*DreamZero (blue) uses coupled noise for video and action (both uniform $\mathcal{U}(0,1)$). DreamZero-Flash (red) biases video toward high-noise states via $\text{Beta}(7,1)$ while keeping action noise uniform — closing the train-test gap for single-step inference.*

**Result**: DreamZero-Flash at 1 denoising step achieves 74% task progress vs. 83% for standard DreamZero at 4 steps — recovering ~89% of 4-step performance at 2.33× the speed.

**Post-processing**: Action chunks are upsampled to 2× resolution, filtered with a Savitzky-Golay filter, then downsampled back to suppress high-frequency noise artifacts from single-step generation.

### Inference: Asynchronous Closed-Loop Execution

**The reactivity gap problem**: Naive inference requires ~5.7 seconds per action chunk due to 16 denoising steps, 14B parameters, and sequential execution. At 30 Hz action execution, each chunk covers 1.6 seconds — far shorter than 5.7 seconds, making synchronous inference impossible.

**Asynchronous redesign**: The motion controller continuously executes the most recently computed action chunk while inference runs concurrently on the latest observation. The latency constraint shifts from "inference must complete before motion starts" to "inference must complete before the current chunk expires." This converts a hard real-time constraint into a softer throughput requirement.

**Full closed-loop algorithm**:
1. **Prefill** ($t=0$): Encode initial image to video latent $\mathbf{z}_{\text{init}}$; initialize KV cache with clean conditioning frames.
2. **Autoregressive loop**: Sample Gaussian noise $\mathbf{x}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$; for each denoising step $i$ from 0 to $N-1$:
   - Check DiT cache: if cosine similarity between successive velocity predictions exceeds threshold $\epsilon$, reuse cached velocities (DiT Caching).
   - Otherwise run full velocity prediction: $\mathbf{v}_i \leftarrow \mathbf{u}_{\theta}(\mathbf{x}_{t_i}; \mathcal{C}, \mathbf{c}, \mathbf{q}, t_i)$
   - Solver step: $\mathbf{x}_{t_{i+1}} \leftarrow \mathbf{x}_{t_i} + \text{Step}(\mathbf{v}_i, t_i, t_{i+1})$
3. **Action extraction**: $\hat{\mathbf{a}} \leftarrow \text{SavitzkyGolayFilter}(\mathbf{x}_1^{\text{action}})$; execute asynchronously on robot.
4. **Cache update (critical)**: Receive ground-truth observation $\mathbf{o}_{\text{real}}$; encode to $\mathbf{z}_{\text{real}} \leftarrow \text{VAE}(\mathbf{o}_{\text{real}})$; replace predicted video latents in KV cache with $\mathbf{z}_{\text{real}}$.

The KV cache replacement with ground-truth observations prevents compounding hallucination errors — predicted frames that were slightly wrong are never propagated into future predictions.

---

## Experiments

### Datasets and Benchmarks

| Dataset | Scale | Characteristics | Usage |
|---------|-------|-----------------|-------|
| AgiBot G1 | ~500 hours, 7,200 episodes | 22 unique environments; ~42 subtasks/episode; 4.4 min avg duration; bimanual parallel gripper | Primary pretraining and evaluation |
| DROID | Public, heterogeneous | Multi-embodiment single-arm manipulation; diverse tasks | Validation / reproducibility |
| YAM adaptation | ~30 min play data, 55 trajectories | Bimanual parallel gripper; 11 unique tasks | Few-shot embodiment adaptation |
| Cross-embodiment video | 72 trajectories × 8 demos × 9 tasks | 20 min robot (YAM) or 12 min human egocentric | Video-only transfer |

The AgiBot dataset is specifically designed to prioritize **task diversity over repetition** — it avoids collecting many repeated demonstrations of the same task and instead covers a wide variety of environments and manipulation skills across homes, restaurants, supermarkets, coffee shops, and offices.

![Figure 6: AgiBot Dataset Statistics](https://arxiv.org/html/2602.15922v1/x6.png)
*Distribution statistics for the AgiBot pretraining corpus: (a) episode duration distribution (avg 4.4 min), (b) subtask count per episode (~42 subtasks), (c) skill distribution across 7.2K episodes (~500 hours).*

![Figure 7: AgiBot Evaluation Setup](https://arxiv.org/html/2602.15922v1/x9.png)
*Evaluation is conducted in unseen environments with unseen objects — DreamZero is evaluated as a zero-shot policy for all seen-task assessments.*

### Implementation Details

- **Backbone**: Wan2.1-I2V-14B-480P (image-to-video diffusion)
- **Parameters**: 14B (full backbone); ablation variant uses 5B (Wan2.1-I2V-5B-480P)
- **Hardware (inference)**: 2× NVIDIA GB200 GPUs for 7 Hz; 1× H100 for slower inference
- **Video sampling**: 5 FPS; action sampling: 30 Hz (AgiBot), 15 Hz (DROID)
- **Chunk size**: $K=2$ latent frames per chunk, 1.6 seconds per chunk
- **Training steps**: 100K steps, batch size 128 (both datasets)
- **Action representation**: Relative joint positions, idle actions filtered

### Main Results: Seen Task Generalization

DreamZero is evaluated zero-shot on tasks present in the pretraining distribution but in unseen environments with unseen objects.

![Figure 8: Seen Task Evaluation](https://arxiv.org/html/2602.15922v1/x10.png)
*DreamZero outperforms all VLA baselines across all task categories on the AgiBot G1 evaluation. VLAs trained from scratch achieve near-zero success; pretrained VLAs show modest performance from embodiment-specific pretraining.*

| Method | PnP-Easy | PnP-Hard | Contact-Rich | Overall |
|--------|----------|----------|--------------|---------|
| DreamZero | **~65%** | **~60%** | **~63%** | **62.2%** |
| GR00T N1.6 (Pretrained) | ~27% | — | — | 27.4% |
| π₀.₅ (Pretrained) | — | — | — | — |
| VLA (from scratch) | ~0% | ~0% | ~0% | ~0% |

**Key finding**: DreamZero achieves 62.2% average task progress — over **2× higher than the best pretrained VLA baseline (27.4%)**. VLAs trained from scratch on the same diverse data achieve near-zero success, while even pretrained VLAs struggle because their video-free pretraining cannot provide the physics priors needed for diverse manipulation.

### Main Results: Zero-Shot Generalization to Unseen Tasks

The most important evaluation: 10 tasks *entirely absent from pretraining* (ironing, painting, pulling carts, cube stacking, removing hat from mannequin, untying shoelaces, etc.).

![Figure 9: Zero-shot Generalization to Unseen Tasks](https://arxiv.org/html/2602.15922v1/x11.png)
*DreamZero achieves non-trivial task progress on 10 unseen tasks while VLAs struggle across both embodiments (AgiBot G1 and DROID-Franka).*

| Embodiment / Method | Task Progress | Success Rate |
|---------------------|---------------|--------------|
| **AgiBot G1** | | |
| DreamZero | **39.5%** | — |
| VLA (pretrained) | 16.3% | — |
| VLA (from scratch) | <1% | <1% |
| **DROID-Franka** | | |
| DreamZero | **49%** | **22.5%** |
| GR00T N1.6 | 31% | 12.5% |
| π₀.₅ | 33% | 7.5% |

**Selected per-task results**: "Remove Hat from Mannequin" — 85.7%; "Shake Hands" — 59.2%.

**Key observation**: VLAs "often reach toward objects and attempt grasping regardless of the instruction, suggesting they overfit to dominant training behaviors (e.g., pick-and-place) rather than understanding novel task semantics." DreamZero's joint video prediction forces it to model the actual motion required by each instruction.

DreamZero was also tested on 100+ additional free-form tasks including "Pop the balloon" and "Press elevator button."

![Figure 2: Joint Video and Action Prediction](https://arxiv.org/html/2602.15922v1/x2.png)
*DreamZero jointly generates video and action. Predicted actions closely align with the generated video on totally unseen tasks.*

![Figure 3: Free-form Evaluation](https://arxiv.org/html/2602.15922v1/x3.png)
*DreamZero performs a diverse range of tasks including object manipulation, tool use, and human-robot interaction.*

### Post-Training Results

Task-specific fine-tuning (50K steps each) on three downstream long-horizon tasks:

| Task | Data | DreamZero | GR00T N1.6 | VLA (from scratch) |
|------|------|-----------|------------|---------------------|
| Shirt Folding (5 stages) | 33 hours | ~85% | ~85% | ~0% |
| Fruit Packing (10 fruits) | 12 hours | ~90% | ~75% | ~0% |
| Table Bussing (10 items) | 40 hours | ~83% | ~70% | ~0% |

![Figure 10: Post-training Results](https://arxiv.org/html/2602.15922v1/x12.png)
*WAMs enable stronger post-training results across all three tasks, and environment generalization of DreamZero is retained after post-training.*

**Key finding**: DreamZero matches or outperforms VLA baselines on all three tasks and critically **retains environment generalization** after post-training (evaluated in a different geographic location). VLAs trained from scratch achieve near-zero success despite 12–40 hours of task-specific data.

### Cross-Embodiment Transfer from Video-Only Data

DreamZero leverages the insight that joint video-action training makes it possible to learn from demonstrations where only video is available — no action labels needed.

Two settings:
- **Robot-to-Robot (YAM → AgiBot)**: 72 trajectories, 8 demos × 9 unseen tasks, 20 minutes, both bimanual parallel grippers.
- **Human-to-Robot (Egocentric → AgiBot)**: 72 trajectories, 8 demos × 9 unseen tasks, 12 minutes of egocentric human video.

Training: co-train from DreamZero-AgiBot checkpoint on 1:1 mixture with pretraining data for 10K steps.

| Method | Task Progress | Relative Improvement |
|--------|---------------|----------------------|
| DreamZero (baseline) | 38.3% ± 7.6% | — |
| + Human2Robot Transfer | 54.3% ± 10.4% | +41.8% relative |
| + Robot2Robot Transfer | **55.4% ± 9.5%** | **+44.6% relative** |

![Figure 11: Cross-Embodiment Transfer](https://arxiv.org/html/2602.15922v1/x13.png)
*Robot-to-robot (YAM → AgiBot) and human-to-robot transfer improve task progress on unseen tasks by 42%+ relative using only 10–20 minutes of video-only data.*

**Key finding**: Robot-to-robot transfer yields the largest gain (38.3% → 55.4%) due to the narrower embodiment gap. Even human egocentric video (larger morphological gap, different viewpoint) provides 42% relative improvement. Unlike recent VLA transfer approaches, **no action labels are required**.

### Few-Shot Embodiment Adaptation

Post-training DreamZero-AgiBot on 30 minutes of "play data" (55 unscripted trajectories, 11 tasks) on the YAM bimanual robot:

![Figure 12: Few-shot Embodiment Adaptation](https://arxiv.org/html/2602.15922v1/x14.png)
*Post-trained on 30 minutes of YAM play data, DreamZero generalizes to novel objects (pumpkins, teddy bears, pens, cup noodles) and retains strong language following despite minimal training data.*

The policy generalizes to novel objects and retains zero-shot language following because it only needs to learn the implicit IDM mapping from visual futures to YAM motor commands — the video generation priors and language understanding are already captured.

### Inference Speed Ablation

| Optimization | H100 | GB200 |
|---|---|---|
| Baseline | 1× | 1.1× |
| + CFG Parallelism | 1.9× | 1.8× |
| + DiT Caching | 5.5× | 5.4× |
| + Torch Compile + CUDA Graphs | 8.9× | 10.9× |
| + Kernel & Scheduler Opts. | 9.6× | 14.8× |
| + Quantization (NVFP4) | — | 16.6× |
| + DreamZero-Flash | — | **38×** |

Total: 5.7 seconds → 150 ms latency; enables 7 Hz closed-loop control on 2× GB200.

**CFG Parallelism**: [[Classifier-Free Guidance|Classifier-free guidance]] requires conditional + unconditional forward passes. Distributing across two GPUs cuts per-step latency by 47%.

**DiT Caching**: Exploits directional consistency of velocity predictions during flow matching. When cosine similarity between successive velocity predictions exceeds a threshold $\epsilon$, cached velocities are reused rather than recomputing the full DiT forward pass. Reduces effective denoising steps from 16 to 4 with minimal quality loss.

**Torch Compile + CUDA Graphs** (mode `reduce-overhead`, `fullgraph=True`, `dynamic=False`): Eliminates CPU overhead and fuses operators. Static shapes cause recompilation only on the first trajectory.

**NVFP4 Quantization** (GB200 Blackwell architecture): Weights and activations in NVFP4; sensitive operations (QKV, LayerNorm, RoPE) in FP8/FP16; softmax in FP8; accumulation in FP16.

### Ablation Study: Data Diversity, Model Scale, and Architecture

| Architecture | Model Size | Data | Task Progress |
|---|---|---|---|
| **Q1. Data Diversity** | | | |
| DreamZero (AR) | 14B | Repetitive | 33% ± 4.2% |
| DreamZero (AR) | 14B | Diverse | **50% ± 6.3%** |
| **Q2. Model Scale** | | | |
| DreamZero (AR) | 5B | Diverse | 21% ± 4.2% |
| DreamZero (AR) | 14B | Diverse | **50% ± 6.3%** |
| VLA | 5B | Diverse | 50% ± 0.0%* |
| VLA | 14B | Diverse | 50% ± 0.0%* |
| **Q3. Architecture** | | | |
| DreamZero (BD) | 14B | Diverse | 50% ± 14.4% |
| DreamZero (AR) | 14B | Diverse | **50% ± 6.3%** |

*50% ± 0.0% means 0% task success with 50% partial progress by always reaching toward objects without grasping — VLAs fail to learn from diverse data regardless of scale.

**Q1 — Data Diversity**: Diverse data improves task progress by +17 pp (51% relative) over repetitive data of the same scale (500 hours each). WAMs benefit from diversity because the core challenge is learning a robust IDM — which requires varied state-action correspondences across many contexts.

**Q2 — Model Scale**: WAMs show clear scaling: 14B outperforms 5B by +29 pp (138% relative). The 5B model is prone to visual hallucinations that propagate to erroneous actions. By contrast, VLAs at 5B and 14B both fail to learn from diverse data — scaling model size alone cannot fix the weak learning signal from static-image pretraining.

**Q3 — Autoregressive vs. Bidirectional**: Both achieve similar mean task progress (50%), but AR has dramatically lower variance (6.3 vs. 14.4), produces substantially smoother motions, and is 3–4× faster at inference due to KV-caching.

### DreamZero-Flash Ablation

| Method | Denoising Steps | Task Progress | Inference Speed | Speed-up |
|--------|-----------------|---------------|-----------------|----------|
| DreamZero | 4 | 83% ± 6.1% | 350 ms | 1× |
| DreamZero (no Flash) | 1 | 52% ± 10.2% | 150 ms | 2.33× |
| DreamZero-Flash | 1 | **74% ± 10.1%** | 150 ms | 2.33× |

Reducing from 4 to 1 denoising step without Flash loses 31 pp. Flash recovers 22 pp of that loss, achieving ~89% of 4-step performance at 2.33× speed — demonstrating that the decoupled noise schedule effectively closes the train-test gap.

---

## Critical Analysis

### Strengths

1. **Fundamental insight is well-validated**: The claim that jointly predicting video and action enables better generalization than direct action prediction is supported across multiple ablations (diverse data, model scale, architecture choice) and multiple evaluation settings (seen tasks, unseen tasks, cross-embodiment transfer).

2. **Practical deployment credibility**: Unlike many world model papers that only demonstrate sim evaluations, DreamZero achieves 7 Hz real-time closed-loop control on physical robots via rigorous co-design of model, system, and algorithm optimizations.

3. **Minimal cross-embodiment data requirement**: 10–20 minutes of unlabeled video to achieve 42% relative improvement is a remarkably low bar, suggesting the approach can scale to large-scale human video in the future.

4. **Clear understanding of why things work**: The paper offers mechanistic explanations for each design choice (why AR > BD, why decoupled schedules help, why diversity matters for IDM learning), not just empirical results.

5. **Comprehensive open-source release**: Model weights, inference code, and benchmark evaluation scripts — rare for a paper at this scale.

### Limitations

1. **Computational cost**: 7 Hz on 2× GB200 GPUs is still expensive compared to VLAs running at 20+ Hz on consumer GPUs. Smaller video backbone models with equivalent generalization would be needed for edge deployment.

2. **Short-horizon context**: The 6.6-second KV cache window limits long-horizon task performance. Robust long-horizon execution requires either a System 2 planner or significantly extended context windows.

3. **High-precision manipulation**: The broad generalization pretraining strategy underrepresents the dense demonstrations needed for sub-centimeter precision tasks (key insertion, fine assembly). The trade-off between breadth and dexterity is acknowledged but unresolved.

4. **Cross-embodiment data scale**: The human-to-robot transfer results are limited to 12 minutes of in-lab egocentric video. The paper hypothesizes but has not demonstrated that large-scale in-the-wild human video (e.g., Ego4D) would yield further improvements.

5. **Scaling law evidence is preliminary**: The paper shows clear WAM scaling benefits (5B vs. 14B) but acknowledges lacking evidence for formal scaling laws — it is unclear how performance continues to scale with more parameters, more data, or longer training.

6. **AgiBot dataset not yet public**: The primary evaluation relies on a proprietary 500-hour dataset not yet released, limiting immediate reproducibility (though DROID evaluations are available).

### Reproducibility

- [x] Code open-sourced (https://github.com/dreamzero0/dreamzero)
- [x] Pretrained models available (model weights released)
- [x] Training details complete (steps, batch size, action representation, filtering)
- [ ] AgiBot dataset not yet public (planned for future release; DROID checkpoint available)

---

## Related Notes

### Based On

- [[WAM|World Action Model (WAM)]]: DreamZero is the primary large-scale instantiation of the WAM paradigm — jointly predicting video and actions rather than predicting actions directly.
- [[视频生成模型|Video Generation Model]]: Inherits spatiotemporal priors from Wan2.1-I2V-14B-480P, a web-scale image-to-video diffusion model.
- [[Flow Matching|Flow Matching]]: Training objective uses flow matching with linear interpolation noise schedule rather than DDPM-style denoising.
- [[DiT|Diffusion Transformer (DiT)]]: Backbone architecture for joint denoising of video and action tokens.
- [[逆向动力学模型|Inverse Dynamics Model (IDM)]]: The action prediction head acts as an implicit IDM trained end-to-end with the video predictor.
- [[Action Chunking]]: Actions are predicted and executed in 1.6-second chunks at 30 Hz.

### Compared Against

- [[VLA|VLA (Vision-Language-Action)]]: The primary comparison class — DreamZero consistently outperforms VLA baselines on generalization tasks, especially for novel physical motions absent from training.
- [[Classifier-Free Guidance|CFG]]: Used during DreamZero inference; parallelized across two GPUs for 47% latency reduction.
- GR00T N1.6 (NVIDIA pretrained VLA): Best pretrained VLA baseline; achieves 27.4% on seen AgiBot tasks vs. DreamZero's 62.2%.
- π₀.₅ (Physical Intelligence): Competitive pretrained VLA; outperformed by DreamZero on zero-shot unseen task evaluation.
- Bidirectional WAM (BD): Ablation showing that autoregressive architecture has lower variance, smoother motions, and 3–4× faster inference.

---

> [!summary] DreamZero (2026)
> - **Core**: A 14B World Action Model that jointly predicts video and robot actions, enabling zero-shot generalization to novel physical motions by leveraging spatiotemporal priors from web-scale video pretraining.
> - **Method**: Autoregressive DiT with flow matching; coupled video-action denoising; DreamZero-Flash decouples noise schedules for single-step inference; 38× speedup stack enables 7 Hz real-time control.
> - **Results**: 62.2% task progress on seen tasks (vs. 27.4% best VLA), 39.5–49% on unseen tasks (vs. <1–31% VLAs); 42%+ cross-embodiment transfer improvement from 10–20 min video-only data.
> - **Code**: https://github.com/dreamzero0/dreamzero
