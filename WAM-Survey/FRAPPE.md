---
title: "FRAPPE: Infusing World Modeling into Generalist Policies via Multiple Future Representation Alignment"
method_name: "FRAPPE"
authors: [Han Zhao, Jingbo Wang, Wenxuan Song, Shuai Chen, Yang Liu, Yan Wang, Haoang Li, Donglin Wang]
year: 2026
venue: arXiv
tags: [world-action-model, diffusion-policy, future-representation, robot-manipulation]
zotero_collection: 3-Robotics/WAM-Survey
image_source: online
arxiv_html: https://arxiv.org/html/2602.17259
created: 2026-05-20
---

# Paper Note: FRAPPE

## Metadata

| Field | Content |
|-------|---------|
| Institution | Zhejiang University, Westlake University, HKUST (GZ), South China University of Technology, ShanghaiTech University, Tsinghua University |
| Date | February 2026 |
| Project Page | https://h-zhao1997.github.io/frappe |
| Baselines | RDT-1B, π₀, π₀.₅, DP, VPP |
| Links | [arXiv](https://arxiv.org/abs/2602.17259) / [Code](https://github.com/Jbo-Wang/frappe) / [Weights](https://huggingface.co/collections/hhhJB/frappe) |

---

## One-Line Summary

> FRAPPE is a two-stage fine-tuning scheme that infuses implicit world modeling into a diffusion-based VLA by aligning learnable future prefix tokens with multiple visual foundation model representations, avoiding pixel-level generation overhead while eliminating inference-time error accumulation.

---

## Core Contributions

1. **Multiple Future Representation Alignment**: Rather than predicting raw pixels or relying on a single visual encoder, FRAPPE supervises learnable future-prefix tokens to match latent representations from three independent visual foundation models (CLIP, DINOv2, ViT). This multi-encoder supervision avoids the inductive biases inherent in any single representation, enabling more robust world-modeling.

2. **Two-Stage Fine-Tuning (Mid-training + Post-training)**: A mid-training phase with full-parameter fine-tuning aligns the model to a distilled Theia teacher, establishing a stable world-modeling prior. A subsequent post-training phase using the MiPA (Mixture-of-Prefix-and-LoRA) architecture scales alignment to three separate VFMs in parallel with frozen backbone weights. This staged design ensures convergence stability and parameter efficiency.

3. **Action-Free Human Video Co-Training**: FRAPPE can leverage large-scale action-free egocentric human video data (TASTE-Rob: 100,856 sequences) during mid-training by omitting the action loss and using only the alignment loss. This dramatically reduces dependence on expensive teleoperation demonstrations and enables scaling robot learning with freely available video data.

---

## Problem Background

### Problem Being Solved

Generalist robotic [[VLA|Vision-Language-Action (VLA)]] models learn a reactive mapping from current observations and language instructions to action sequences. They lack explicit awareness of how the world will evolve after executing an action. This deficit hurts generalization: in novel or out-of-distribution scenarios, the model cannot reason about causal consequences and tends to make premature or physically inconsistent decisions. Existing attempts to add [[世界模型|world modeling]] to VLAs fall into two families, both with critical flaws.

### Limitations of Existing Methods

**Pixel-level world models** (e.g., [[VPP]], [[AVDC]]): These methods train the policy to explicitly generate or decode future video frames, then use the predicted frames as auxiliary input. The problem is twofold. First, generating full-resolution future images forces the model to spend capacity fitting visually redundant pixels (textures, backgrounds) rather than task-relevant object semantics. Second, at inference time the policy is conditioned on its own predictions, introducing compounding errors—hallucinated future frames propagate to distorted actions.

**Single-representation implicit alignment** (e.g., [[FLARE]], [[MWM]]): These methods align internal features with a single pre-trained visual encoder, bypassing pixel generation. However, any single encoder carries inductive biases tied to its training objective (e.g., CLIP favors language-grounded semantics; DINOv2 favors geometric structure). A policy aligned to just one encoder inherits those biases and may underperform on tasks that benefit from complementary representations.

**[[Diffusion Policy]]-based VLAs without world modeling** (e.g., [[RDT]], [[π0]]): These achieve strong generalization through scale but lack any future-state awareness, leading to failures on precise, contact-rich, or multi-step tasks where anticipating consequences is critical.

### Motivation

The key insight is that **implicit representation alignment can inject world-modeling capacity without generating pixels and without running world models at inference**. By training learnable token sequences ("future prefixes") to predict the latent representations that a well-trained visual encoder would assign to a future observation, the model internalizes predictive dynamics during training. At inference, no future observation is generated; the prefixes simply encode the learned predictive prior. Using *multiple* diverse visual encoders as simultaneous teachers prevents over-specialization and provides richer supervisory signal.

---

## Method

### Architecture Overview

FRAPPE uses a **two-stage fine-tuning strategy on top of a pretrained [[Diffusion Policy|diffusion-based VLA]]** with:
- **Input**: Language instruction $l$, current visual observation $\mathbf{o}_t$, noisy action chunk $\tilde{\mathbf{a}}_t$, denoising step $k$, learnable future-prefix tokens $\mathbf{p}_t$
- **Backbone**: [[RDT|Robotic Diffusion Transformer (RDT-1B)]] — a 1B-parameter [[DiT|Diffusion Transformer]] pretrained on diverse robot demonstrations
- **Core modules**: Future Prefix Tokens (n×d learnable tokens), Visual Foundation Model (VFM) teachers (CLIP 400M, DINOv2 142M, ViT 300M), MiPA (Mixture-of-Prefix-and-LoRA) router
- **Output**: Denoised action chunk $\mathbf{a}_t$ (predicted directly; no future frame generation at inference)
- **Total parameters (trainable at post-training)**: Prefix tokens + LoRA modules per expert; backbone frozen

![Figure 2 — FRAPPE Training and Inference Overview](https://arxiv.org/html/2602.17259v1/x5.png)

**Caption**: The left side shows the mid-training phase where a single future prefix stream is aligned to the distilled Theia encoder using full-parameter fine-tuning. The right side shows the post-training phase where three parallel MiPA expert streams (each with dedicated prefix tokens and LoRA modules) are aligned simultaneously to CLIP, DINOv2, and ViT, while the shared RDT backbone remains frozen. At inference, all three expert outputs are gated and aggregated by a router MLP.

---

### Future Prefix Tokens

**Motivation**: Standard [[VLA]] models consume the current observation $\mathbf{o}_t$ and produce an action. They have no explicit pathway to represent what the world should look like in the near future. Introducing a set of dedicated learnable tokens that are trained to encode future visual states gives the backbone implicit access to predictive dynamics at every denoising step, without modifying the backbone architecture or adding inference-time world-model rollouts.

**Design**: A set of learnable tokens $\mathbf{p} \in \mathbb{R}^{n \times d}$ (where $n$ is prefix length and $d$ is the model's hidden dimension) is concatenated with the backbone's input sequence at every denoising step. During training, these tokens receive gradient signal from an alignment loss that encourages them to encode the visual foundation model's representation of a future observation $\mathbf{o}_{t+h}$ (horizon $h=8$ steps ahead). Because the tokens are fully differentiable and concatenated with the action inputs, the backbone learns to use them for action generation. During inference, the tokens simply exist as learned parameters—no future observation needs to be generated or fed in.

The denoising function becomes:

$$
\mathbf{a}_{t}, \mathbf{p}_{t} = f_{\theta}(l, \mathbf{o}_{t}, \tilde{\mathbf{a}}_{t}, k)
$$

**Meaning**: The backbone $f_\theta$ now outputs both the denoised action chunk $\mathbf{a}_t$ and an updated prefix representation $\mathbf{p}_t$ that is supervised to match the future-state encoder. This joint output means the backbone learns to internally reason about future states as part of producing actions.

**Symbols**:
- $\mathbf{a}_t$: predicted action chunk at timestep $t$
- $\mathbf{p}_t$: updated future-prefix representation
- $l$: language instruction
- $\mathbf{o}_t$: current visual observation
- $\tilde{\mathbf{a}}_t$: noisy action chunk at denoising step $k$
- $k$: denoising step index

---

### Mid-Training Phase

**Motivation**: Directly applying multi-encoder alignment from scratch is unstable. The backbone has been pretrained for action generation; jumping immediately to multi-teacher supervision risks conflicting gradients. A preparatory mid-training phase bridges this gap by first aligning the model to a single, comprehensive teacher.

**Design**: In mid-training (15,000 steps), all backbone parameters are unfrozen (full-parameter fine-tuning). The teacher is a **distilled Theia encoder** (86M parameters), which is itself a distilled model trained to combine the representations of all three VFMs (CLIP, DINOv2, ViT). Using a unified distilled teacher provides a stable, internally consistent alignment target that avoids the directional conflicts that would arise from simultaneously aligning to three independent teachers before the model has adapted.

The alignment loss in mid-training uses cosine similarity between the predicted prefix and the (stop-gradient) teacher encoding:

[[世界模型|Alignment Loss]]:

$$
\mathcal{L}_{\Phi} = \cos(\mathbf{p}_t, \text{sg}(\mathbf{e}))
$$

**Meaning**: The cosine similarity between the prefix output $\mathbf{p}_t$ and the teacher embedding $\mathbf{e}$ (with stop-gradient so the teacher is not updated) is maximized. Using cosine rather than MSE makes the loss scale-invariant and avoids magnitude collapse.

**Symbols**:
- $\mathbf{p}_t$: prefix output from the backbone
- $\mathbf{e}$: teacher encoder embedding of future observation $\mathbf{o}_{t+h}$
- $\text{sg}(\cdot)$: stop-gradient operator

**Action-free co-training**: During mid-training, action-free human egocentric video data (TASTE-Rob) can be co-trained. For these samples, the action loss $\mathcal{L}_{\text{action}}$ is simply omitted and only $\mathcal{L}_\Phi$ is applied. This allows the model to learn future-state representations from web-scale video without requiring any robot demonstrations or action labels.

---

### Post-Training Phase: MiPA (Mixture-of-Prefix-and-LoRA)

**Motivation**: After mid-training the model can predict future representations from a single unified teacher. However, that teacher is a distillation—it may lose fine-grained specialization present in the individual VFMs. The post-training phase expands to multiple independent VFM teachers simultaneously, giving the model access to CLIP's language-semantic features, DINOv2's geometry-aware features, and ViT's general visual features in parallel.

**Design**: Three expert streams are added to the frozen backbone. Each expert $i \in \{1, 2, 3\}$ consists of:
- A dedicated set of learnable prefix tokens $\mathbf{p}^{(i)} \in \mathbb{R}^{n \times d}$
- A set of [[LoRA]] modules injected into the backbone layers for that expert (backbone weights frozen; only LoRA $A$, $B$ matrices trained)
- A corresponding frozen VFM teacher: expert 1 → CLIP (400M), expert 2 → DINOv2 (142M), expert 3 → ViT (300M)

A lightweight **router network** computes gating weights $\{w_i\}_{i=1}^{M}$ (where $M=3$, $\sum_i w_i = 1$). Each expert stream produces a latent action representation $z_i$, and the final action is obtained by weighted aggregation:

[[Action Chunking|Action Aggregation]]:

$$
\mathbf{a}_t = \text{MLP}\!\left(\sum_{i=1}^{M} w_i \cdot z_i\right)
$$

**Meaning**: The router softly selects how much each expert's prediction contributes. Because different tasks may benefit from different visual features (e.g., a task requiring precise grasp may benefit more from DINOv2's geometric features), the router can adapt the blend per input.

**Symbols**:
- $w_i$: gating weight for expert $i$ (computed by router from input features)
- $z_i$: latent action representation from expert $i$
- $M = 3$: number of experts

The multi-encoder alignment loss sums individual alignment losses:

$$
\mathcal{L}_{\text{align}} = \sum_{i=1}^{M} \mathcal{L}_{\Phi_i}
$$

To prevent mode collapse (where the router always assigns all weight to one expert), a **load-balancing loss** is applied:

[[Mixture-of-Transformers|Load Balance Loss]]:

$$
\mathcal{L}_{\text{balance}} = \frac{1}{B} \sum_{j=1}^{B} \left(\log \sum_{i=1}^{M} e^{\mathbf{g}_{i,j}}\right)^{2}
$$

**Meaning**: This penalizes over-concentration of router logits $\mathbf{g}_{i,j}$ for sample $j$ in batch $B$, encouraging all three experts to be utilized.

**Label smoothing** is additionally applied to router weights to prevent sharp overconfidence:

$$
w'_i = w_i \cdot (1 - \epsilon) + \frac{\epsilon}{M}
$$

where $\epsilon = 0.1$ is the smoothing factor.

![Figure 3 — Parameter Scale Experiments (RDT-130M vs RDT-1B)](https://arxiv.org/html/2602.17259v1/figure/allTolora.png)

**Caption**: Comparison of FRAPPE applied to RDT-130M and RDT-1B backbones. FRAPPE on the 130M model achieves performance competitive with the unmodified RDT-1B, demonstrating that the world-modeling enhancement provides an effective parameter-efficiency boost.

---

### Training Objective

The full training objective combines three losses with scalar hyperparameters:

[[Diffusion Policy|Action Loss]]:

$$
\mathcal{L}_{\text{action}} := \text{MSE}(\mathbf{a}_t, f_\theta(l, \mathbf{o}_t, \tilde{\mathbf{a}}_t, k))
$$

**Meaning**: Standard [[Diffusion Policy|diffusion denoising]] objective—mean squared error between predicted and ground-truth action chunks. This is the primary task signal.

**Symbols**:
- $\mathbf{a}_t$: ground-truth action chunk
- $f_\theta(\cdot)$: policy network (backbone + prefixes + LoRA)

Combined objective:

[[世界模型|FRAPPE Total Loss]]:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{action}} + \lambda_1 \mathcal{L}_{\text{align}} + \lambda_2 \mathcal{L}_{\text{balance}}
$$

**Meaning**: The action generation objective is regularized by the world-modeling alignment signal and the diversity-encouraging balance loss. The alignment term is kept small ($\lambda_1 = 0.05$) to prevent it from overwhelming the action objective.

**Symbols**:
- $\lambda_1 = 0.05$: alignment loss weight (ablated; optimal found at 0.05)
- $\lambda_2$: balance loss weight (tuned separately)

**Alignment depth ablation**: The layer at which the prefix representation is extracted for alignment was ablated across all 28 DiT layers. Layer 21 was found optimal (23.5% hard success rate), indicating that mid-to-late representations capture the most semantically useful future-state information.

**Future horizon ablation**: The number of steps ahead $h$ was ablated. $h = 8$ was optimal (35.3% hard success rate). Too-short horizons lack predictive informativeness; too-long horizons produce noisy or impossible future states.

---

### Inference

FRAPPE introduces **no additional computational overhead from world models at inference time**. The inference procedure is identical to the base RDT pipeline:

1. Receive language instruction $l$ and current observation $\mathbf{o}_t$ from robot sensors
2. Initialize noisy action $\tilde{\mathbf{a}}_t$ from Gaussian noise
3. Run $K$ denoising steps (default $K=5$; $K=3$ also tested for lower latency):
   - For each step, compute $\mathbf{a}_t, \mathbf{p}_t = f_\theta(l, \mathbf{o}_t, \tilde{\mathbf{a}}_t, k)$ across all three MiPA expert streams in parallel
   - Router computes $\{w_i\}$ and aggregates expert outputs via $\mathbf{a}_t = \text{MLP}(\sum_i w_i \cdot z_i)$
   - No alignment loss or VFM teachers run at inference
4. Execute the denoised action chunk on the robot

The prefixes act as learned latent codes that implicitly encode predictive dynamics learned during training. The additional latency over baseline RDT is only ~20ms (from 0.214s to 0.235s for $K=5$ steps), and memory usage is 8.0GB vs. 3.7GB baseline—a reasonable trade-off.

---

## Experiments

### Datasets and Benchmarks

| Dataset | Scale | Characteristics | Usage |
|---------|-------|-----------------|-------|
| RoboTwin 2.0 | 8 bimanual tasks × 100 eval trials | Simulation benchmark; Easy/Hard settings with varying object diversity | Evaluation |
| Robot teleoperation data | 50 task-specific trajectories | Expert demonstrations via bimanual AgileX manipulator | Training |
| Task-specific egocentric | 360+ trajectories/hour | Human manipulation video, no action labels | Co-training |
| TASTE-Rob | 100,856 sequences | Web-scale human egocentric video, no action labels | Co-training |

### Implementation Details

- **Backbone**: RDT-1B (primary); RDT-130M (scale validation)
- **Optimizer**: AdamW (standard configuration)
- **Batch size**: 32
- **Training steps**: 20,000 total (15,000 mid-training + 5,000 post-training)
- **Hardware**: 2× NVIDIA H100 GPUs (simulation experiments); 8× H100 GPUs (web-scale egocentric co-training)
- **Teacher encoders**: CLIP (400M), DINOv2 (142M), ViT (300M); Theia distilled (86M) for mid-training
- **Label smoothing**: $\epsilon = 0.1$
- **Alignment loss weight**: $\lambda_1 = 0.05$
- **Alignment depth**: Layer 21 of 28 DiT layers
- **Future horizon**: $h = 8$ steps
- **Real-world platform**: Bimanual AgileX manipulator with 6-DoF per arm; 40 trials per variation (basic tasks), 20 trials (long-horizon)

### Main Results

FRAPPE achieves state-of-the-art performance on RoboTwin 2.0 across 8 bimanual manipulation tasks in both Easy and Hard settings. The Hard setting (with out-of-distribution objects and configurations) is the critical test of generalization. FRAPPE substantially outperforms all baselines on Hard, achieving 25.5% vs. the next-best π₀ at 14.1%—nearly a 2× improvement.

| Method | Easy Avg. | Hard Avg. |
|--------|-----------|-----------|
| [[Diffusion Policy\|DP]] | 31.3% | 0.0% |
| [[VPP]] | 35.8% | 4.0% |
| [[RDT]] (baseline) | 47.4% | 15.1% |
| [[π0]] | 57.1% | 14.1% |
| [[pi0.5\|π₀.₅]] | 45.4% | 13.3% |
| **FRAPPE** | **57.5%** | **25.5%** |

**Per-task breakdown (Easy% / Hard%)**:

| Task | DP | VPP | RDT | π₀ | π₀.₅ | FRAPPE |
|------|----|-----|-----|----|-------|--------|
| Handover Mic | 53/0 | 91/7 | 90/31 | 98/13 | 98/13 | **98/45** |
| Pick Dual Bottles | 24/0 | 28/6 | 45/14 | 57/12 | 37/13 | **67/28** |
| Handover Block | 10/0 | 8/0 | 45/14 | 45/8 | 2/0 | **50/18** |
| Place Object Basket | 15/0 | 30/0 | 30/6 | 16/2 | 42/4 | **35/15** |
| Put Object Cabinet | 42/0 | 15/12 | 33/18 | 68/18 | 46/15 | **52/37** |
| Place Shoe | 23/0 | 40/4 | 35/7 | 28/6 | 28/12 | **41/18** |
| Stack Bowls Two | 61/0 | 59/3 | 74/27 | 91/41 | 85/44 | **80/36** |
| Put Bottle Dustbin | 22/0 | 15/0 | 27/4 | 54/13 | 25/5 | **37/7** |

**Key finding**: FRAPPE is particularly strong in the Hard setting, where generalization to novel object instances and configurations is required. The gap from 15.1% (RDT baseline) to 25.5% demonstrates that world-modeling enhances out-of-distribution robustness. FRAPPE matches π₀'s Easy performance (57.5% vs 57.1%) while exceeding it by 11.4 points on Hard, suggesting that alignment-based world modeling is more beneficial under distribution shift than scale alone.

![Figure 1 — Performance Overview](https://arxiv.org/html/2602.17259v1/x4.png)

**Caption**: FRAPPE significantly outperforms state-of-the-art models in both simulated (RoboTwin 2.0) and real-world complex scenarios (long-horizon bimanual tasks). The figure highlights the gap on Hard simulation tasks and real-world success rates.

### Ablation Study

The ablation systematically tests the contribution of each training stage and configuration. The key finding is that **both mid-training and post-training stages are essential**, and that LoRA modules in post-training are necessary (prefix-only post-training degrades performance).

| Configuration | Avg. Success Rate | Notes |
|---------------|-------------------|-------|
| RDT baseline (20k steps) | 39.8% | No world modeling |
| Mid-train full-ft (20k) | 45.3% | Single Theia teacher; good foundation |
| Mid-train prefix & LoRA (20k) | 28.3% | Prefix+LoRA without full-ft fails |
| Post-train prefix only (20k) | 14.5% | No mid-training; prefix alone insufficient |
| Post-train prefix & LoRA (20k) | 27.5% | No mid-training; degraded |
| Mid-train (15k) + post-train prefix (5k) | 44.8% | Mid-training helps but prefix-only post-train stagnates |
| **FRAPPE (15k mid + 5k post w/ LoRA)** | **52.3%** | Full two-stage with LoRA — best |

**Key finding**: Mid-training full-parameter fine-tuning is critical for establishing a stable world-modeling prior. Post-training prefix-only updates (without LoRA) stagnate or regress, because the frozen backbone cannot adapt its representations to incorporate multi-teacher supervision. The combination of mid-training + post-training LoRA achieves the best result, confirming that progressive capability expansion is the right design choice.

**Hyperparameter sensitivity** (tested on hard setting):

| Hyperparameter | Configuration | Hard Success |
|----------------|---------------|--------------|
| $\lambda_1$ (alignment weight) | 0.01 | 18.3% |
| $\lambda_1$ | **0.05** | **32.5%** |
| $\lambda_1$ | 0.1 | 21.0% |
| Alignment depth | Layer 14 | 19.2% |
| Alignment depth | **Layer 21** | **23.5%** |
| Alignment depth | Layer 28 | 20.1% |
| Future horizon $h$ | 4 steps | 28.7% |
| Future horizon $h$ | **8 steps** | **35.3%** |
| Future horizon $h$ | 16 steps | 29.1% |

### Additional Analysis: Inference Efficiency

FRAPPE's overhead is minimal compared to the performance gain:

| Variant | Memory | Latency ($K=5$) | Avg. Success |
|---------|--------|-----------------|--------------|
| RDT baseline | 3.7 GB | 0.214 s | 39.8% |
| Mid-train only | 3.7 GB | 0.228 s | 45.3% |
| FRAPPE ($K=5$) | 8.0 GB | 0.235 s | 52.3% |
| FRAPPE ($K=3$) | 8.0 GB | 0.173 s | 48.5% |

The 8.0 GB memory requirement fits within a single standard GPU. At $K=3$ denoising steps, FRAPPE achieves 48.5% success with only 0.173s latency—faster than the baseline at $K=5$—making it viable for real-time deployment.

### Additional Analysis: Real-World Generalization

Four basic tasks were evaluated on a bimanual AgileX manipulator under four types of distribution shift: lighting variation, table height variation, object pose variation, and novel object instances.

![Figure 4 — Real-World Results](https://arxiv.org/html/2602.17259v1/x6.png)

**Caption**: Real-world performance on 4 representative tasks under seen and unseen (generalization) conditions. FRAPPE consistently outperforms the RDT baseline across all generalization axes, particularly under novel object and lighting conditions.

### Additional Analysis: Long-Horizon Tasks

A complex multi-step task was tested requiring 3 dependent sub-tasks and 4 interactive objects (e.g., pick object → place in container → close lid).

![Figure 5 — Long-Horizon Performance](https://arxiv.org/html/2602.17259v1/x7.png)

**Caption**: Subtask completion rates for a 3-stage long-horizon task. The RDT baseline achieves 0% full-task success due to failures at critical contact-rich sub-tasks. FRAPPE achieves 20% full-task success by maintaining action continuity and anticipating fine-grained manipulation requirements.

**Key finding**: The RDT baseline completely fails (0%) on the long-horizon task. FRAPPE reaches 20% success, demonstrating that world-modeling is not just a marginal improvement but an enabling capability for sequential multi-step manipulation.

### Additional Analysis: Egocentric Data Pyramid

FRAPPE's compatibility with action-free human video is validated via a data pyramid experiment with a pick-and-place task under limited teleoperation data:

| Data Combination | Success Rate |
|-----------------|--------------|
| Robot teleoperation only (5 demos) | Baseline (low) |
| + Task-specific egocentric data | Significant improvement |
| + Web-scale egocentric (TASTE-Rob) | +10–15% over teleoperation-only |
| All three combined | Best performance |

![Figure 6 — Egocentric Data Integration](https://arxiv.org/html/2602.17259v1/x8.png)

**Caption**: The data pyramid framework integrating robot teleoperation (top), task-specific human egocentric video (middle), and web-scale action-free video (TASTE-Rob, bottom). Results show progressive improvements as more data tiers are added, with web-scale egocentric data providing an additional 10–15% improvement.

**Key finding**: Action-free video supervision via the alignment loss is an effective substitute for action annotations, enabling data collection at 100,856 sequences from web sources vs. 120 teleoperation trajectories per hour. This addresses a key practical bottleneck in robot learning.

---

## Critical Analysis

### Strengths

1. **No inference-time overhead from world models**: Unlike VPP or model-based methods that run generative models at test time, FRAPPE's world-modeling capacity is entirely encoded in the learned prefix parameters. This means zero inference-time world-model rollout cost and no error accumulation from recursive prediction.

2. **Multi-encoder supervision prevents representation bias**: Using three diverse VFM teachers (CLIP for semantic, DINOv2 for geometric, ViT for general visual) provides complementary supervisory signals. The ablation confirms this is better than a single encoder—the mid-training Theia distillation also serves as a useful intermediate step.

3. **Action-free video co-training is practically impactful**: The ability to train on TASTE-Rob (100,856 sequences) without action labels dramatically reduces data collection costs. The 10–15% improvement from web-scale video demonstrates genuine utility.

4. **Strong out-of-distribution generalization**: The 2× improvement on Hard tasks (25.5% vs 14.1% π₀) is the most compelling result—it shows the world-modeling prior helps exactly in the regime where standard reactive policies fail.

5. **Parameter efficiency**: FRAPPE on RDT-130M approaches baseline RDT-1B performance, meaning the world-modeling enhancement effectively compensates for 8× fewer parameters.

### Limitations

1. **Memory overhead at post-training inference**: Running three VFM expert streams in parallel requires 8.0 GB vs. 3.7 GB baseline. While this fits a single GPU, it doubles memory and may constrain deployment on edge devices or smaller systems.

2. **Hard task performance still modest (25.5%)**: While substantially better than baselines, 25.5% Hard success rate leaves significant room for improvement. The method still struggles on high-precision tasks like "Put Bottle Dustbin" (7% Hard) suggesting that world-modeling alone is insufficient for the most demanding manipulation tasks.

3. **Hyperparameter sensitivity**: The ablation shows meaningful performance differences across alignment layer depth, $\lambda_1$, and horizon $h$. Practitioners must tune these for new task distributions, adding engineering overhead.

4. **Router mechanism underspecified**: The paper does not fully detail the router network architecture (number of layers, input features, training dynamics), making it difficult to reproduce or adapt the gating mechanism.

5. **Benchmark scope**: All simulation experiments are on RoboTwin 2.0. Broader evaluation on LIBERO, MetaWorld, or real-world multi-task benchmarks would strengthen generalizability claims.

### Potential Improvements

1. **Hierarchical teacher alignment**: Distilling from more diverse teachers (e.g., adding R3M or MVP for contact-rich tasks) or using adaptive teacher weighting based on task type could further reduce inductive bias.

2. **Lightweight expert compression**: Post-training MiPA experts could potentially be merged via LoRA weight averaging for inference-time efficiency, reducing memory from 8.0 GB closer to baseline.

### Reproducibility

- [x] Code open-sourced (https://github.com/Jbo-Wang/frappe)
- [x] Pretrained models available (https://huggingface.co/collections/hhhJB/frappe)
- [x] Training details complete
- [x] Datasets accessible

---

## Related Notes

### Based On

- [[RDT]]: FRAPPE is built directly on top of RDT-1B (and RDT-130M), using its pretrained DiT backbone as the frozen base model in post-training
- [[Diffusion Policy]]: The action generation objective is a standard diffusion denoising MSE loss, inheriting the action chunking and denoising framework
- [[Theia]]: The mid-training phase uses a Theia-distilled 86M encoder as the unified teacher, compressing CLIP/DINOv2/ViT into a single alignment target

### Compared Against

- [[π0]]: Strong VLA baseline with 57.1% Easy / 14.1% Hard; FRAPPE matches Easy but nearly doubles Hard performance
- [[pi0.5]]: Smaller π₀ variant; FRAPPE outperforms on both settings despite using a different base model
- [[VPP]]: Explicit video prediction policy; FRAPPE achieves better generalization while avoiding inference-time video generation overhead
- [[RDT]]: Direct base model; FRAPPE improves it from 47.4/15.1 to 57.5/25.5 through world-modeling fine-tuning

### Method Related

- [[世界模型|World Model]]: FRAPPE implements implicit world modeling via representation alignment rather than explicit generative prediction
- [[LoRA]]: MiPA uses LoRA for parameter-efficient post-training fine-tuning of three expert streams
- [[Action Chunking]]: The action output is a chunk of $h$ future actions, consistent with standard VLA practice
- [[VLA]]: FRAPPE is a VLA enhancement method; the base architecture follows VLA conventions

### Hardware / Data

- [[TASTE-Rob]]: 100,856-sequence web-scale egocentric video dataset used for action-free co-training in mid-training phase

---

> [!summary] FRAPPE (2026)
> - **Core**: Implicit world modeling via learnable future prefix tokens aligned to multiple visual foundation models, avoiding pixel generation and inference-time error accumulation
> - **Method**: Two-stage fine-tuning (mid-training: full-ft with distilled Theia teacher; post-training: MiPA with frozen backbone + 3 parallel VFM experts via LoRA)
> - **Results**: 57.5% Easy / 25.5% Hard on RoboTwin 2.0 (vs. next-best 57.1% / 14.1% π₀); 20% success on long-horizon tasks where RDT baseline scores 0%
> - **Code**: https://github.com/Jbo-Wang/frappe
