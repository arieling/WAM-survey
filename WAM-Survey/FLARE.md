---
title: "FLARE: Robot Learning with Implicit World Modeling"
method_name: "FLARE"
authors: [Ruijie Zheng, Jing Wang, Scott Reed, Johan Bjorck, Yu Fang, Fengyuan Hu, Joel Jang, Kaushil Kundalia, Zongyu Lin, Loic Magne, Avnish Narayan, You Liang Tan, Guanzhi Wang, Qi Wang, Jiannan Xiang, Yinzhen Xu, Seonghyeon Ye, Jan Kautz, Furong Huang, Yuke Zhu, Linxi Fan]
year: 2025
venue: arXiv
tags: [world-action-model, implicit-world-model, diffusion-policy, robot-manipulation, vla, flow-matching, egocentric-learning]
zotero_collection: 3-Robotics/WAM-Survey
image_source: online
arxiv_html: https://arxiv.org/html/2505.15659v1
created: 2026-05-20
---

# Paper Note: FLARE

## Metadata

| Field | Content |
|-------|---------|
| Institution | NVIDIA; University of Maryland, College Park; Nanyang Technological University; University of Texas, Austin |
| Date | May 2025 |
| Project Page | N/A |
| Baselines | Diffusion Policy, UWM, GR00T N1 (Scratch), FLARE Policy Only |
| Links | [arXiv](https://arxiv.org/abs/2505.15659) / [Code](N/A) |

---

## One-Line Summary

> FLARE augments a [[Diffusion Transformer]]-based robot policy with lightweight future latent alignment tokens that implicitly model world dynamics, achieving up to 26% improvement over baselines with minimal architectural overhead.

---

## Core Contributions

1. **Implicit Latent World Modeling via Future Alignment**: Instead of predicting full future frames pixel-by-pixel, FLARE introduces M learnable future tokens into the [[Diffusion Transformer]] (DiT) input sequence. At an intermediate DiT layer, these tokens' hidden states are projected and aligned via cosine similarity loss to the frozen latent embedding of the next observed frame. This forces the action denoising network to develop an implicit, compressed internal representation of future world states without the expense of full generative reconstruction.

2. **Action-Aware Vision-Language Embedding Model**: The future target embeddings are not raw [[SigLIP-2]] features but are produced by a compact model specifically pretrained with an action flow-matching co-objective on ~2,990 hours of cross-embodiment robotic data. This co-training ensures the embedding captures task-relevant information (what matters for manipulation) rather than generic visual semantics, which ablations show contributes a net 11.1 percentage point gain over using raw SigLIP-2 features.

3. **Action-Free Human Video Co-Training**: By decoupling the alignment loss from the action prediction loss, FLARE enables mixing action-labeled robot demonstrations with action-free human egocentric videos during training. Human videos contribute only to the future latent alignment objective, effectively acting as additional world-model supervision without requiring any action annotation or pose estimation, and approximately doubling success rate on novel objects when only 10 teleoperated demonstrations are available.

---

## Problem Background

### Problem Being Solved

Robot manipulation policies trained purely via [[imitation learning]] suffer from limited generalization: they learn to map observations to actions without internalizing a model of how the world responds to actions. When encountering new objects, viewpoints, or task configurations, these policies fail because they have no internal mechanism for anticipating consequences. The fundamental question is: how do you cheaply inject world-dynamics awareness into a [[VLA]] policy without the enormous cost of training a full video-generative model?

### Limitations of Existing Methods

- **[[Diffusion Policy]]** learns a [[flow-matching]] action prior over observation-conditioned action chunks, but treats the environment as a black box. It contains no mechanism for predicting how the scene will evolve after actions are taken, limiting generalization.
- **[[UWM]]** (Unified World Model) jointly trains a video prediction and action prediction head with a shared diffusion objective. Although it reasons over future frames, the pixel-level reconstruction task competes with the policy task for model capacity and requires large numbers of gradient steps (the authors train UWM for 400k steps vs. 80k for FLARE). Its 29.5% success rate on the GR-1 benchmark—below even vanilla Diffusion Policy—demonstrates that joint pixel prediction does not always help.
- **[[GR-1]] / [[GR-2]]** families unify next-token prediction across video and actions, requiring large transformer capacity and extensive video data for the video prediction head to contribute meaningfully.
- **[[DINO-WM]]** aligns with DINO features at the current timestep (analogous to [[REPA]] for image generation), which provides representation regularization but does not explicitly model future dynamics.
- **Prior VLA-based approaches** ([[π0]], [[GR00T N1]]) use diffusion/flow-matching policy heads atop frozen VLMs but do not incorporate world model supervision, limiting data efficiency.

### Motivation

The key insight is that world modeling for policy improvement does not require generating pixel-perfect future frames — it requires the policy network to maintain an internal state that is *predictive* of how the world will look after the planned actions. Concretely, if the hidden states of a [[Diffusion Transformer]] at an intermediate layer can be made predictive of the latent embedding of the next observation, the network must implicitly represent future-relevant scene structure. This is analogous to the [[JEPA]] (Joint Embedding Predictive Architecture) philosophy: predict in abstract latent space rather than pixel space, which is more computationally efficient and avoids the difficulty of learning high-frequency visual details that are irrelevant to control. The alignment loss is a simple cosine similarity objective — no decoder, no reconstruction network — keeping the method almost free in terms of added parameters and compute.

---

## Method

### Architecture Overview

FLARE uses a **[[Diffusion Transformer]] (DiT) policy with future latent alignment** with the following structure:

- **Input**: Multi-camera RGB observations $o_t$, language instruction, proprioceptive state $q_t$, noised action chunk $A_t^\tau$, and $M$ learnable future token embeddings
- **Backbone**: [[Diffusion Transformer]] with alternating cross-attention (attending to vision-language embeddings) and self-attention layers; same architecture as [[GR00T N1]]
- **Vision-Language Encoder**: [[Eagle]] VLM for producing observation embeddings $\phi_t = \text{VL}(o_t)$
- **Core modules**: Future latent alignment module (Section 3.1), Action-aware vision-language embedding model (Section 3.2)
- **Output**: Denoised action chunk $A_t$ of length $H$

![Figure 1 — FLARE vs. Conventional Flow-Matching Policy](https://arxiv.org/html/2505.15659v1/x1.png)

**Caption**: Left: a standard flow-matching policy learns to predict actions conditioned on the current observation. Right: FLARE additionally injects future tokens into the action denoising process and aligns their hidden-state projections to future observation embeddings, embedding world-model awareness into the policy without a separate generation branch.

![Figure 2 — FLARE Architecture](https://arxiv.org/html/2505.15659v1/x2.png)

**Caption**: Full FLARE architecture. The DiT receives three token types: (1) state/action tokens from $q_t$ and $A_t^\tau$, (2) vision-language conditioning from $\phi_t$ via cross-attention, and (3) $M$ learnable future tokens. At layer $L$, future token activations are extracted, projected by an MLP $f_\theta$, and aligned to the frozen future observation embedding $g(\phi_{t+H})$ via cosine similarity loss. The action prediction head runs in parallel, unaffected by the alignment branch.

---

### Future Latent Alignment Module

**Motivation**: A policy that can predict how the scene will look after executing its planned actions must internally represent the causal consequences of those actions. By adding $M$ learnable future tokens to the DiT input and penalizing their intermediate-layer projections for diverging from actual future observation embeddings, FLARE creates a training signal that forces the shared self-attention mechanism to encode future-relevant information into all tokens, including the action-prediction tokens. The critical design choice is to use *future* embeddings (not current-frame embeddings as in [[REPA]]) — this is what makes the loss a world-model objective rather than a representation-regularization objective.

**Design**: The DiT input sequence is augmented with $M = 32$ learnable future token embeddings appended to the state and action tokens. All tokens attend to each other via self-attention and receive vision-language context via cross-attention. At internal layer $L = 6$ (out of typically 16 or more total layers), the hidden states corresponding to the $M$ future tokens are extracted and passed through a lightweight MLP $f_\theta: \mathbb{R}^{B \times M \times D_{\text{DiT}}} \to \mathbb{R}^{B \times M \times D_{\text{embed}}}$. These projected features are compared to the output of the frozen future embedding model $g(\phi_{t+H}) \in \mathbb{R}^{B \times M \times D_{\text{embed}}}$, where $\phi_{t+H}$ is the vision-language embedding of the observation $H$ steps ahead. The alignment is computed as cosine similarity, negated to form a minimization objective.

**Layer selection rationale**: Using too early a layer (e.g., layer 4) hurts performance because the representations have not yet integrated cross-attention context. Using too deep a layer forces the same neurons to serve both action prediction and future prediction, causing conflicts. Layer 6 is empirically optimal, providing a good trade-off between information richness and separation from the action output head.

[[Future Latent Alignment|Alignment Loss]]:

$$
\mathcal{L}_{\text{align}}(\theta) = -\mathbb{E}_\tau\left[\cos\!\left(f_\theta(\phi_t, A_t^\tau, q_t),\; g(\phi_{t+H})\right)\right]
$$

**Meaning**: Minimizing this loss (making cosine similarity approach 1) forces the MLP projection of future token activations to match the direction of the future observation embedding. The expectation over $\tau$ means this alignment is required to hold regardless of the noise level in the denoising trajectory, preventing the world-model signal from only being informative at specific noise scales.

**Symbols**:
- $f_\theta$: MLP projection of future token hidden states from DiT layer $L$, shape $B \times M \times D$
- $g$: frozen future observation embedding model (action-aware VL encoder), shape $B \times M \times D$
- $\phi_t$: vision-language embedding of current observation
- $\phi_{t+H}$: vision-language embedding of future observation $H$ steps ahead
- $A_t^\tau$: noised action chunk at flow-matching timestep $\tau$
- $q_t$: current proprioceptive state
- $\cos$: cosine similarity
- $B$: batch size, $M$: number of future tokens, $D$: embedding dimension

**Distinction from REPA**: [[REPA]] aligns DiT hidden states with the *current* image embedding in text-to-image diffusion models, acting as a representation learning regularizer. FLARE aligns with *future* embeddings, making it a predictive world-model objective. Additionally, FLARE uses separate learnable future tokens (not reusing image or noise tokens), so the flow-matching stream and alignment stream can interact via self-attention while optimizing partially independent objectives.

---

### Action-Aware Vision-Language Embedding Model

**Motivation**: The quality of the alignment target $g(\phi_{t+H})$ determines how useful the alignment loss is. A generic vision encoder (e.g., raw [[SigLIP-2]] patch tokens) would produce embeddings that capture low-level visual detail irrelevant to manipulation (background texture, lighting variation). What is needed is an embedding that captures *what matters for the task*: object positions, affordances, robot-object spatial relations. The model achieves this by co-training the embedding model with an action prediction head.

**Design**: The action-aware embedding model has the following components:

1. **Vision encoder**: [[SigLIP-2]]-large (patch size 16, image size 256×256) encodes each 256×256 RGB image into 256 patch tokens.
2. **Language encoder**: Encodes the padded language instruction into 32 tokens.
3. **Fusion**: Four layers of self-attention transformer blocks operate on the concatenated 288 tokens (256 vision + 32 language), allowing cross-modal integration.
4. **Compression ([[Q-Former]])**: A Q-former module with $M=32$ learnable query tokens runs interleaved self-attention (among queries) and cross-attention (queries to fused tokens), compressing the full representation into 32 fixed-size query token embeddings. This design generalizes across multi-camera setups by processing each camera independently and pooling.
5. **Action head (pretraining only)**: 8 additional DiT blocks are attached during pretraining to predict actions via flow-matching, ensuring the embedding captures task-relevant information.

The model is pretrained on approximately 2,990 hours of diverse cross-embodiment robotic data (see Table 3 in Appendix B), spanning real robot and simulation across multiple morphologies (GR-1 humanoid, DROID, RT-1, Bridge-v2, etc.).

![Figure 10 — Q-Former-Based Vision-Language Embedding Module](https://arxiv.org/html/2505.15659v1/x10.png)

**Caption**: Architecture of the action-aware embedding module. SigLIP-2 encodes images; language encodes instructions; four self-attention blocks fuse the tokens; Q-Former compresses to $M=32$ query tokens. During pretraining, 8 DiT blocks attach to enable action prediction, instilling task-awareness into the embeddings.

![Figure 3 — Pretraining Data Mixture](https://arxiv.org/html/2505.15659v1/x3.png)

**Caption**: Data mixture used to pretrain the action-aware embedding model — approximately 2,990 hours of robotic trajectories across 8 datasets, spanning real-robot manipulation and simulation, multiple embodiments (humanoid GR-1, arm, tabletop).

**EMA Posttraining Adaptation**: During downstream policy training, if the target embedding model is kept completely frozen, a distribution shift may arise between the pretraining domain and the downstream task (different lighting, object types, camera placements). To mitigate this, FLARE uses an exponential moving average (EMA) update of the target embedding model, where the slowly-evolving target tracks the policy's own vision-language encoder:

[[Exponential Moving Average|EMA Update Rule]]:

$$
\theta_{\text{target\_vl}} \leftarrow \rho\,\theta_{\text{target\_vl}} + (1-\rho)\,\theta_{\text{policy\_vl}}
$$

**Meaning**: The target embedding model drifts slowly towards the policy's VL encoder, allowing it to adapt to the downstream distribution without destabilizing training.

**Symbols**:
- $\theta_{\text{target\_vl}}$: parameters of the target (frozen-side) vision-language embedding model
- $\theta_{\text{policy\_vl}}$: parameters of the policy's own vision-language encoder, updated by gradient descent
- $\rho$: EMA decay rate; $\rho = 0.995$ empirically optimal (see ablation, Figure 9)

---

### Training Objective (Full)

FLARE training combines two objectives. The primary objective is the flow-matching loss from [[π0]] / [[GR00T N1]]:

[[Flow Matching|Action Flow-Matching Loss]]:

$$
\mathcal{L}_{\text{fm}}(\theta) = \mathbb{E}_\tau\left[\left\|V_\theta(\phi_t, A_t^\tau, q_t) - (\epsilon - A_t)\right\|^2\right]
$$

**Meaning**: The DiT velocity head $V_\theta$ is trained to predict the residual $(\epsilon - A_t)$ at each noise level $\tau$, so that integrating along the flow trajectory recovers the clean action chunk $A_t$.

**Symbols**:
- $V_\theta$: DiT velocity prediction head
- $A_t^\tau = \tau A_t + (1-\rho)\epsilon$: noised action at timestep $\tau$
- $\epsilon \sim \mathcal{N}(0, I)$: sampled noise
- $\tau \in [0,1]$: flow-matching timestep, sampled from $p(\tau) = \text{Beta}\!\left(\frac{s-\tau}{s}; 1.5, 1\right)$ with $s=0.999$

The combined training loss is:

[[FLARE Loss|Combined Objective]]:

$$
\mathcal{L} = \mathcal{L}_{\text{fm}} + \lambda\,\mathcal{L}_{\text{align}}
$$

**Meaning**: The alignment coefficient $\lambda = 0.2$ balances the world-model signal against the action prediction objective. Ablations show that the model is robust across $\lambda \in [0.2, 1.0]$, but $\lambda = 0.2$ is empirically optimal.

**Symbols**:
- $\lambda$: alignment loss weight; optimal $\lambda = 0.2$

**Human video co-training**: For action-free human egocentric videos, only $\mathcal{L}_{\text{align}}$ is applied (no action labels are required). Robot demonstrations receive both $\mathcal{L}_{\text{fm}}$ and $\mathcal{L}_{\text{align}}$.

---

### Inference

At inference time, action generation proceeds by iterative denoising over $K=4$ steps starting from pure noise $A_t^0 \sim \mathcal{N}(0, I)$:

[[Flow Matching|Euler Integration Step]]:

$$
A_t^{\tau + 1/K} = A_t^\tau + \frac{1}{K}\,V_\theta(\phi_t, A_t^\tau, q_t)
$$

The $M$ future tokens are present in the input sequence but their outputs are discarded — they exist only as auxiliary supervision during training. No future observations are needed at inference time. The total computational overhead from the future tokens is negligible: $M=32$ tokens add a small fraction to the self-attention cost of the DiT.

---

## Experiments

### Datasets and Benchmarks

| Dataset | Scale | Characteristics | Usage |
|---------|-------|-----------------|-------|
| RoboCasa (simulation) | 24 atomic tasks | Simulated kitchen; pick-and-place, door, faucet; 3 RGB cameras (left, right, wrist) | Multi-task benchmark |
| GR-1 Tabletop (simulation) | 24 tasks | GR-1 humanoid; 18 object rearrangement + 6 articulated object tasks; single egocentric head camera | Multi-task benchmark |
| Real GR1 humanoid | 4 tabletop tasks × 100 trajectories | Pick-and-place with real robot; 4 objects (apple, can, bottled water, cucumber) | Post-training evaluation |
| Human egocentric videos | 5 novel objects × 150 GoPro demos | Head-mounted GoPro; action-free; novel object geometries | Co-training w/o action labels |
| Cross-embodiment pretraining | ~169.5M frames, ~2,990 hr | 8 datasets: GR-1 in-house, DROID, RT-1, Bridge-v2, Language Table, MUTEX, Plex, RoboSet, GR-1 Simulation | Embedding model pretraining |

![Figure 4 — Simulation Benchmarks](https://arxiv.org/html/2505.15659v1/x4.png)

**Caption**: Left: RoboCasa 24-task kitchen simulation benchmark. Right: GR-1 24-task tabletop simulation benchmark. The tasks range from pick-and-place to articulated object manipulation (cabinet, microwave, drawer), testing both dexterous precision and spatial reasoning.

![Figure 5 — Real GR1 Tasks Setup](https://arxiv.org/html/2505.15659v1/x5.png)

**Caption**: Four real-world tabletop pick-and-place tasks for the GR-1 humanoid robot, evaluated over 8 reference initial configurations each, with four distinct objects (apple, can, bottled water, cucumber). Success requires precise grasp and placement.

### Implementation Details

- **Backbone**: [[Diffusion Transformer]] with alternating cross-attention + self-attention, same as GR00T N1; [[Eagle]] VLM for vision-language encoding
- **Embedding model backbone**: [[SigLIP-2]]-large (patch size 16, resolution 256×256)
- **Future tokens**: $M=32$ learnable tokens; alignment at DiT layer $L=6$
- **Optimizer**: AdamW ($\beta_1=0.95$, $\beta_2=0.999$, $\epsilon=10^{-8}$, weight decay $10^{-5}$)
- **Learning rate**: Cosine decay with 5% warmup
- **Batch size**: 1024 (policy training); 8192 (embedding pretraining)
- **Training steps**: 80,000 (policy training); 150,000 (embedding pretraining)
- **Hardware**: 32× NVIDIA H100 (policy training); 256× NVIDIA H100 (embedding pretraining)
- **Denoising steps at inference**: $K=4$
- **EMA rate**: $\rho=0.995$
- **Alignment coefficient**: $\lambda=0.2$

### Main Results

FLARE achieves state-of-the-art performance on both simulation benchmarks, outperforming all baselines by 8–26 percentage points depending on the benchmark.

| Method | RoboCasa (Pick & Place) | RoboCasa (Doors) | RoboCasa (Other) | **RoboCasa Avg** | GR-1 (Rearrange) | GR-1 (Articulated) | **GR-1 Avg** |
|--------|------------------------|------------------|------------------|-----------------|-------------------|---------------------|-------------|
| **FLARE** | **53.2%** | **88.8%** | **80.0%** | **70.1%** | **58.2%** | **51.3%** | **55.0%** |
| Policy Only (FLARE arch.) | 43.8% | 78.7% | 75.2% | 61.9% | 46.6% | 47.4% | 44.0% |
| GR00T N1 (Scratch) | 44.1% | 80.0% | 69.6% | 60.6% | 51.8% | 42.8% | 45.1% |
| UWM | 35.6% | 82.0% | 74.2% | 60.8% | 30.1% | 38.4% | 29.5% |
| Diffusion Policy | 29.2% | 78.7% | 61.3% | 51.7% | 40.4% | 50.1% | 40.9% |

All methods trained for 80,000 gradient steps (UWM extended to 400k to verify results). Evaluation: 50 episodes per task every 1000 steps; maximum over the last 5 checkpoints reported.

**Key finding**: FLARE's gain over Policy Only (the same architecture trained without alignment) demonstrates that the alignment loss itself — not the architectural additions — is responsible for improvement. The gap is largest on the GR-1 benchmark (+11%), where tasks require reasoning about complex articulated object interactions, suggesting implicit world modeling is most valuable when manipulation requires anticipating mechanical responses to contact.

Notably, UWM's poor GR-1 performance (29.5%) despite longer training reveals that explicit pixel-level video prediction can *hurt* policy learning when model capacity is insufficient to excel at both objectives simultaneously.

### Post-Training with Pretrained Embedding Model

![Figure 6 — Post-Training Results](https://arxiv.org/html/2505.15659v1/x6.png)

**Caption**: Data-efficiency comparison. Left: RoboCasa simulation with varying training trajectory count per task — FLARE with pretrained cross-embodiment embedding consistently outperforms policy-only at all data scales. Right: Real GR1 humanoid results — FLARE achieves 95.1% success, 14% above the policy-only baseline.

When using the cross-embodiment pretrained embedding model as the alignment target (instead of re-training a task-specific one from scratch):

- **RoboCasa simulation**: FLARE achieves **71.3%** vs. 70.2% with in-domain embedding — the pretrained cross-embodiment embedding generalizes to unseen tasks without fine-tuning.
- **Real GR1 humanoid (100 trajectories/task)**: FLARE achieves **95.1% success rate**, 14 percentage points above the policy-only baseline (81%).

The real robot qualitative analysis reveals a mechanistically interesting behavior: when an object (can, water bottle) is placed near the robot's hand in the scene, the policy-only baseline frequently knocks it over due to lack of spatial anticipation. FLARE consistently maneuvers around or over the obstacle, demonstrating that the implicit world model has learned to represent object permanence and causal consequences of arm trajectories.

### Human Egocentric Video Co-Training

![Figure 7 — Novel Object Generalization with Human Videos](https://arxiv.org/html/2505.15659v1/x7.png)

**Caption**: Left: five novel objects with distinctive geometries (blue tape, etc.) requiring novel grasping strategies not seen in GR-1 pretraining data. Right: success rate comparison — adding 150 action-free human GoPro demonstrations per object alongside 10 robot demos roughly doubles policy-only performance.

| Training Data | Success Rate |
|---------------|-------------|
| 1 robot demo / object | ~60% |
| 10 robot demos, policy only | ~40% |
| 10 robot demos + 150 human egocentric videos | **~80%** |

**Key finding**: Human egocentric videos, which contain no action labels and come from a different embodiment (human hand vs. robot gripper), nevertheless dramatically improve policy performance on novel objects when used as additional alignment-loss supervision. The method requires no pose estimation, no motion retargeting, and no action annotation — the alignment loss absorbs whatever predictive visual signal the human videos provide.

### Ablation Study

![Figure 8 — Ablation: DiT Layer and Loss Coefficient](https://arxiv.org/html/2505.15659v1/x8.png)

**Caption**: Left: success rate as a function of which DiT layer's activations are used for alignment. Layer 6 is optimal; layer 4 degrades performance (insufficient context); deeper layers plateau or decline (action-alignment conflict). Right: success rate vs. alignment coefficient $\lambda$. The method is robust for $\lambda \in [0.2, 1.0]$.

**Table: Target Embedding Model Ablation (GR-1 benchmark)**

| Configuration | Success Rate | Delta vs. No Alignment |
|---------------|-------------|----------------------|
| No FLARE alignment loss | 43.9% | — |
| SigLIP-2 (direct features) | 49.6% | +5.7% |
| SigLIP-2 (average pooled) | 50.9% | +7.0% |
| **Action-aware embedding (FLARE full)** | **55.0%** | **+11.1%** |

**Key finding**: Every form of future alignment helps, but action-aware pretraining is essential for the full gain. Raw SigLIP-2 features capture visual semantics but not task-relevant manipulation structure; the Q-Former compression and action co-training objective distill the embedding into a manipulation-relevant representation, providing an additional 5.4 percentage points over average-pooled SigLIP-2.

**DiT Layer Selection**: Layer 6 is optimal. Shallower layers (e.g., layer 4) have not yet integrated cross-attention context, making their representations less informative for future prediction. Deeper layers are close to the action output and optimizing them for future alignment creates conflicting gradient signals with the action prediction objective.

**FLARE Loss Coefficient $\lambda$**: Performance peaks at $\lambda = 0.2$ and remains strong through $\lambda = 1.0$. The model is insensitive to moderate overweighting of the alignment loss, suggesting the two objectives are largely complementary within a reasonable range.

![Figure 9 — EMA Coefficient Ablation](https://arxiv.org/html/2505.15659v1/x9.png)

**Caption**: Success rate on 24-task RoboCasa (300 trajectories/task) as a function of EMA decay rate $\rho$. All EMA variants exceed policy-only (43.9%). $\rho = 0.995$ is optimal; $\rho = 0.99$ degrades (too-frequent updates destabilize the alignment target); $\rho = 1.0$ (fully frozen) is suboptimal but still competitive.

| EMA Rate $\rho$ | Success Rate | Notes |
|-----------------|-------------|-------|
| 0.99 | ~50% | Too-fast updates destabilize target |
| **0.995** | **~55%** | **Optimal — slow, stable adaptation** |
| 0.999 | ~53% | Near-optimal; very slow adaptation |
| 1.0 (frozen) | ~51% | No adaptation to downstream domain |
| 0.0 (policy only, no FLARE) | 43.9% | Baseline |

---

## Critical Analysis

### Strengths

1. **Minimal architectural overhead with large gains**: FLARE adds only $M=32$ learnable future tokens and one MLP head to a standard DiT policy. At inference, these tokens are discarded. The 8–26% improvement comes almost entirely from the training signal, not from added model capacity.
2. **Principled connection to JEPA philosophy**: Aligning in latent space rather than pixel space avoids the hardest aspects of video prediction (high-frequency textures, lighting variation, irrelevant background motion) and focuses the world-model signal on manipulation-relevant structure. This is more computationally tractable and arguably more semantically aligned with what control requires.
3. **Action-free human video utilization without engineering**: The decoupled loss formulation trivially allows human videos without action labels to contribute to training. No pose estimation, motion retargeting, or keypoint tracking pipeline is needed, making data collection cheap and scalable.
4. **Strong real-robot results**: 95.1% on GR-1 humanoid with only 100 trajectories per task is a compelling practical result. The qualitative obstacle-avoidance behavior (maneuvering around objects rather than knocking them over) provides evidence that the implicit world model captures geometrically meaningful scene structure.
5. **Robust to hyperparameters**: The ablations show the method is insensitive to $\lambda$ over a broad range and to EMA rate over a 10× range, suggesting it is not tuned to a narrow operating point.

### Limitations

1. **Pick-and-place focus**: Real-robot evaluation is limited to pick-and-place tasks. More complex humanoid manipulation (tool use, bimanual manipulation, fine-grained dexterous tasks) remains unvalidated. It is unclear whether the implicit world model scales to longer-horizon, higher-precision tasks.
2. **Embedding model pretraining cost**: The action-aware embedding model requires 256 H100 GPUs × 150k steps, which is a substantial infrastructure investment. For research groups without access to this scale, using the pretrained embedding as-is limits customization.
3. **Controlled egocentric data**: Human videos were collected with head-mounted GoPro cameras in a controlled lab setting with the same camera height and viewpoint as the robot's egocentric camera. Extending to truly in-the-wild, diverse human video (different viewpoints, diverse environments) has not been demonstrated and may introduce embodiment gaps.
4. **No reinforcement learning integration**: The method relies entirely on imitation learning. For tasks where expert demonstrations are insufficient (sparse reward, exploration required), there is no clear path to incorporating RL without redesigning the world-model component.
5. **Limited scalability analysis**: It is unknown whether performance continues to improve with more data and model scale, or whether the implicit alignment objective saturates.

### Potential Improvements

1. **Hierarchical future alignment**: Align at multiple future horizons (e.g., $H$, $2H$, $4H$) with different token groups, enabling the policy to reason at multiple temporal scales simultaneously.
2. **Diverse in-the-wild video pretraining**: Train the embedding model on large-scale internet egocentric video (Ego4D, EPIC-Kitchens) to make the alignment target more general and enable richer action-free co-training.
3. **Reinforcement learning integration**: Use the implicit world model as a learned reward signal or for planning-based action selection at test time, bridging imitation and reinforcement learning.

### Reproducibility

- [ ] Code open-sourced
- [ ] Pretrained models available
- [x] Training details complete (Appendix C provides full hyperparameters)
- [x] Datasets accessible (Open X-Embodiment, RoboCasa, GR-1 simulation)

---

## Related Notes

### Based On

- [[π0]]: FLARE inherits the [[flow-matching]] policy head and DiT architecture from π0, extending it with future alignment tokens
- [[GR00T N1]]: FLARE uses the same DiT architecture and [[Eagle]] VLM backbone as GR00T N1; the "Policy Only" baseline is essentially GR00T N1 trained from scratch
- [[REPA]]: Representation alignment for diffusion models inspired FLARE's design; FLARE extends REPA from current-frame to future-frame alignment and applies it to robot control

### Compared Against

- [[UWM]]: Unified World Model — joint video + action diffusion objective; FLARE outperforms by 26% on GR-1 despite 5× fewer gradient steps
- [[Diffusion Policy]]: U-Net denoising policy; FLARE outperforms by 18% on RoboCasa, 14% on GR-1
- [[GR00T N1]]: Same architecture trained from scratch without future alignment; FLARE outperforms by 9.4% on RoboCasa

### Method Related

- [[JEPA]]: Joint Embedding Predictive Architecture — the conceptual foundation for predicting in latent space rather than pixel space
- [[Q-Former]]: Used for compressing vision-language tokens in the action-aware embedding model
- [[SigLIP-2]]: Vision-language encoder serving as backbone for the action-aware embedding model
- [[Flow Matching]]: Continuous generative framework used for both policy action generation and embedding model pretraining
- [[Diffusion Transformer]]: Core policy backbone; the future tokens interact with action tokens via self-attention within the DiT layers

### Hardware / Data

- [[GR-1 Humanoid]]: Target robot for real-world evaluation; 100 trajectories per task for post-training
- [[Open X-Embodiment]]: Source of cross-embodiment pretraining data for the action-aware embedding model
- [[DROID]]: Largest component of pretraining mixture (428 hours, 23.1M frames)

---

> [!summary] FLARE (2025)
> - **Core**: Implicit world modeling by aligning diffusion transformer hidden states with future observation latent embeddings via cosine similarity loss
> - **Method**: $M=32$ future tokens added to DiT input; aligned at layer 6 to action-aware Q-Former embeddings; combined loss $\mathcal{L} = \mathcal{L}_{\text{fm}} + 0.2\,\mathcal{L}_{\text{align}}$; enables action-free human video co-training
> - **Results**: 70.1% on RoboCasa (+8.2% over Policy Only), 55.0% on GR-1 simulation (+11.0%), 95.1% on real GR-1 humanoid (+14%); novel object generalization ~80% with human egocentric co-training
> - **Code**: N/A
