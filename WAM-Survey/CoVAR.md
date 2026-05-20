---
title: "CoVAR: Co-generation of Video and Action for Robotic Manipulation via Multi-Modal Diffusion"
method_name: "CoVAR"
authors: [Liudi Yang, Yang Bai, George Eskandar, Fengyi Shen, Mohammad Altillawi, Dong Chen, Ziyuan Liu, Abhinav Valada]
year: 2025
venue: arXiv 2025
tags: [world-action-model, multi-stream, cross-attention, diffusion, robot-manipulation]
zotero_collection: 3-Robotics/WAM-Survey
image_source: online
arxiv_html: https://arxiv.org/html/2512.16023v1
created: 2026-05-20
---

# Paper Note: CoVAR

## Metadata

| Field | Content |
|-------|---------|
| Institution | University of Freiburg; Ludwig Maximilian University of Munich; Munich Center for Machine Learning (MCML); Technical University of Munich; Huawei Heisenberg Research Center (Munich) |
| Date | December 2024 |
| Project Page | N/A |
| Baselines | UVA, UWM, PAD, Unipi, RoboEnvision |
| Links | [arXiv](https://arxiv.org/abs/2512.16023) / Code: N/A |

---

## One-Line Summary

> CoVAR co-generates synchronized video predictions and robot action sequences via parallel [[Diffusion Policy|video]] and [[WAM|action]] [[DiT|diffusion transformers]] connected by a novel Bridge Attention mechanism, achieving state-of-the-art manipulation success rates on Calvin, Libero90, and real UR5 benchmarks.

---

## Core Contributions

1. **Parallel Multi-modal Diffusion**: Extends a pretrained [[视频生成模型|video diffusion]] backbone ([[DiT|Open-Sora-1.2]]) with a dedicated parallel action [[DiT]] branch, enabling joint [[整流流匹配|rectified flow]]-based denoising of video latents and robot actions without discarding pretrained visual knowledge.
2. **Bridge Attention**: A cross-modal interaction module that maintains separate per-modality query, key, and value projection matrices while computing joint attention over the concatenated video and action token sequences — enabling bidirectional information exchange without feature entanglement.
3. **Action Refinement Module**: A lightweight transformer that maps coarse actions (produced at low resolution) to precise robot control signals by cross-attending to initial visual observations and text instructions, critical for low-resolution training datasets.

---

## Problem Background

### Problem Being Solved

Robotic manipulation policies must predict sequences of robot joint states given text instructions and visual observations. Many large-scale robot video datasets lack action annotations, making it difficult to train action policies directly. Simultaneously, generating physically consistent future video frames provides a useful "mental simulation" that can guide action planning. The challenge is to jointly generate both modalities in a way that they remain synchronized and mutually consistent.

### Limitations of Existing Methods

- **Two-stage methods** (e.g., [[UniPi]], [[RoboEnvision]]): Generate video first, then extract actions via a separate inverse dynamics model. The two stages are decoupled, so video generation errors propagate to the action stage and the two objectives cannot co-optimize.
- **Joint model methods** (e.g., [[UWM]], [[PAD]]): Concatenate video and action tokens and denoise with a single shared attention module. This forces both modalities to share the same query/key/value projections, which can conflict since video features are spatially dense (many tokens, patch-based) while action features are temporally compact (joint-state vectors). Shared projections also overwrite pretrained visual representations.
- **Annotation scarcity**: Standard video diffusion pretraining on large web-scale data does not include robot action labels. Methods that require joint training from scratch discard this pretrained knowledge.

### Motivation

The key insight is that video and action modalities are fundamentally different in token structure and semantics, so they benefit from separate projection parameters while still needing to exchange information. Bridge Attention addresses this by maintaining modality-specific projections but computing attention jointly — a middle ground between fully shared (joint model) and fully decoupled (two-stage). Additionally, inheriting a pretrained [[视频生成模型|video diffusion]] backbone gives CoVAR a head start on visual quality that scratch-trained joint models cannot match.

---

## Method

### Architecture Overview

CoVAR builds on [[整流流匹配|rectified flow]]-based diffusion with a [[DiT|Diffusion Transformer]] backbone (Open-Sora-1.2, 1.1B parameters). A parallel action DiT branch (0.3B new parameters, total 1.4B) is added alongside the frozen/fine-tuned video DiT. The two branches interact at every transformer block through Bridge Attention. For low-resolution datasets, a separate Action Refinement Module refines the coarse action outputs.

- **Inputs**: Initial image observation, robot joint state, text instruction, noisy video latent $x_t^1$, noisy action sequence $x_t^2$
- **Backbone**: Open-Sora-1.2 video DiT (pretrained) + parallel action DiT (newly initialized)
- **Core modules**: Bridge Attention (cross-modal alignment), UNet-based action decoder (multi-scale features), Action Refinement Module (precision recovery at low resolution)
- **Output**: Denoised video frame sequence + action sequence (robot joint states at each timestep)

![Figure 1: Architectural comparison of two-stage, joint model, and CoVAR approaches](https://arxiv.org/html/2512.16023v1/x1.png)

*Figure 1: (a) Two-stage Model generates video then extracts actions. (b) Joint Model concatenates all tokens into a single shared DiT. (c) CoVAR uses parallel DiT branches connected by Bridge Attention.*

![Figure 2: CoVAR overview showing (A) parallel DiT streams, (B) Bridge Attention, (C) Action Refinement Module](https://arxiv.org/html/2512.16023v1/x2.png)

*Figure 2: (A) The video diffusion backbone processes video latents while the parallel action DiT processes action embeddings. (B) Bridge Attention connects the two streams at each transformer block. (C) For low-resolution datasets, coarse actions are further refined by the Action Refinement Module.*

### Multi-Modal Rectified Flow

CoVAR adopts [[整流流匹配|rectified flow]] to jointly model the video-action distribution. The combined data is denoted:

$$
X_0 = (x_0^1,\; x_0^2)
$$

where $x_0^1 \in \mathbb{R}^{d_1}$ is the video latent and $x_0^2 \in \mathbb{R}^{d_2}$ is the flattened robot action sequence. The corresponding noise sample is $X_1 = (x_1^1, x_1^2)$ drawn from independent Gaussians. Rectified flow constructs a straight-line interpolation path:

$$
X_t = (1-t)\,X_0 + t\,X_1, \quad t \in [0,1]
$$

The flow ODE specifies that the velocity field equals the straight-line direction:

$$
\frac{dX_t}{dt} = X_1 - X_0
$$

The network $v_\theta = (v_\theta^1,\, v_\theta^2)$ predicts this velocity for both modalities simultaneously. The **joint training loss** is:

$$
\mathcal{L} = \bigl\|x_1^1 - x_0^1 - v_\theta^1\bigr\|_2 \;+\; \bigl\|x_1^2 - x_0^2 - v_\theta^2\bigr\|_2
$$

where $v_\theta^1$ is predicted by the video DiT branch and $v_\theta^2$ by the action DiT branch. Both terms are trained end-to-end together, so improvements in video prediction directly improve action prediction and vice versa. This contrasts with two-stage methods where video and action losses are optimized independently.

### Action Representation

Robot actions are low-dimensional joint-state vectors at each timestep. Rather than using a [[变分自编码器|VAE]] (which would add computational overhead and a second training objective), CoVAR encodes actions with a **lightweight MLP encoder** that maps raw joint states into a higher-dimensional embedding space compatible with the DiT's channel dimension. For decoding, a **UNet-based action decoder** is used instead of a simple linear layer. The UNet's skip connections provide multi-scale feature aggregation from intermediate DiT layers, which empirically improves action precision (see ablation, Table V).

### Bridge Attention Module

**Motivation**: In a standard joint model, video tokens $f_v \in \mathbb{R}^{B \times N_v \times C}$ and action tokens $f_a \in \mathbb{R}^{B \times N_a \times C}$ are concatenated and fed through a shared self-attention layer with single sets of Q/K/V matrices. This forces the same projection to handle both semantically different token types. For video tokens (spatial patches from a pretrained model), the learned projections encode visual texture and motion; for action tokens (joint-state embeddings), the projections must encode temporal dynamics. Sharing them creates a conflict that degrades both modalities.

**Design**: Bridge Attention maintains **separate** Q, K, V projections for each modality ($q_1, k_1, v_1$ for video; $q_2, k_2, v_2$ for actions) but computes attention **jointly** over the concatenated token sequence. This allows modality-specific feature extraction while still enabling information exchange:

$$
\begin{bmatrix} f_v \\ f_a \end{bmatrix} = \mathrm{Attention}\!\left(
\begin{bmatrix} q_1 f_v \\ q_2 f_a \end{bmatrix},\;
\begin{bmatrix} k_1 f_v \\ k_2 f_a \end{bmatrix},\;
\begin{bmatrix} v_1 f_v \\ v_2 f_a \end{bmatrix}
\right)
$$

The attention scores are computed across all $N_v + N_a$ tokens, so each video token can attend to action tokens and vice versa. However, the queries, keys, and values that generate and receive these attention signals are produced by modality-appropriate projections. The interaction is bidirectional: visual context guides action generation, and action state guides which visual regions are attended to.

The ablation compares Bridge Attention against two degenerations:
- **Standard self-attention on concatenated tokens** (shared Q/K/V): success rate drops from 0.68 to 0.32, PSNR from 17.67 to 16.83
- **Standard cross-attention** (action attends to video only, unidirectional): success rate drops to 0.20, PSNR to 16.56

This confirms that both the modality-specific projections (Bridge Attention vs. shared self-attention) and the bidirectionality (Bridge Attention vs. cross-attention) are critical.

### Action Refinement Module

**Motivation**: Calvin and Libero90 datasets have low spatial resolutions (200×200 and 128×128 respectively). At these resolutions, the video DiT generates coarse latents that map to coarse action signals. The absolute positional errors in coarse actions may exceed the tolerance required for successful grasping and manipulation. A refinement stage can recover high-precision control signals by re-attending to the original full-resolution observation.

**Design**: The Action Refinement Module is a lightweight transformer that:
1. Embeds the coarse action tokens and the initial image observation tokens via separate embedding networks
2. Concatenates the action and image tokens and processes through **self-attention** layers to aggregate spatial-visual information across the full sequence
3. Applies **cross-attention** conditioned on the text instruction embedding, allowing the refinement to be task-aware
4. Decodes the refined action token sequence to output-dimension joint states via a linear projection

The module is fine-tuned separately from the main CoVAR model using 450 video-action demonstration pairs on Libero90. Its lightweight design (transformer with few layers, not a full DiT) allows rapid fine-tuning without risk of forgetting the main model's learned representations.

On Libero90, adding the refinement module improves success rate from 0.592 → 0.873 (Pick), 0.520 → 0.860 (Open & Close), 0.422 → 0.711 (Combination) — substantial gains of 28–47% absolute across all task categories.

### Inference

Inference uses 30 steps of rectified flow sampling (ODE integration from $t=1$ noise to $t=0$ data). Each 35-frame sequence takes approximately 4 seconds to generate. The output action sequence is at the video frame rate; since the real robot operates at 100 Hz, **temporal interpolation** is applied between action keyframes to generate dense control signals at the required frequency.

---

## Experiments

### Datasets and Benchmarks

| Dataset | Scale | Resolution | Characteristics | Usage |
|---------|-------|------------|-----------------|-------|
| Calvin | ~20K teleoperated demos | 200×200 | Text-paired; 5 task categories (Drawer, Cabinet, Light, Pick, Push) | Training + evaluation |
| Libero90 | 90 tasks × 50 expert demos | 128×128 | 3 task categories (Pick, Open & Close, Combination); densely annotated | Training + evaluation |
| Real-world (UR5) | 1K self-collected demos | 180×320 | Bowl stacking, nut/screw/dowel picking; no public baseline training data | Real-world evaluation |

### Implementation Details

- **Backbone**: Open-Sora-1.2 (1.1B video DiT parameters)
- **New modules**: 0.3B parameters (parallel action DiT, Bridge Attention projections, UNet decoder, Action Refinement Module), total 1.4B
- **Sequence length**: 35 frames per training sample
- **Optimizer**: AdamW (details not fully specified in the paper)
- **Hardware**: 4 GPUs, ~1 day of training
- **Action Refinement fine-tuning**: 450 video-action pairs on Libero90
- **Inference**: 30 rectified flow steps, ~4 s per 35-frame sequence

### Main Results — Video Quality

Video quality is evaluated with PSNR (pixel fidelity), SSIM (structural similarity), LPIPS (perceptual similarity, lower is better), and FVD (Fréchet Video Distance, lower is better).

**Calvin dataset:**

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FVD ↓ |
|--------|--------|--------|---------|-------|
| UVA | 19.01 | 0.758 | 0.180 | 97.90 |
| PAD | 18.72 | 0.734 | 0.174 | 83.40 |
| UWM | 18.04 | 0.730 | 0.181 | 85.85 |
| OpenSora-1.2 (baseline, no actions) | 19.60 | 0.768 | 0.171 | 61.00 |
| **CoVAR** | **19.95** | **0.766** | **0.156** | 72.42 |

**Libero90 dataset:**

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FVD ↓ |
|--------|--------|--------|---------|-------|
| UVA | 19.57 | 0.716 | 0.154 | 86.21 |
| PAD | 19.65 | 0.781 | 0.218 | 98.39 |
| UWM | 19.87 | 0.735 | 0.212 | 87.83 |
| OpenSora-1.2 (baseline, no actions) | 20.18 | 0.817 | 0.156 | 63.33 |
| **CoVAR** | **20.09** | **0.826** | **0.143** | **70.64** |

**Key finding**: CoVAR achieves the best perceptual quality (LPIPS and SSIM) on both benchmarks among joint models. It does not always beat Open-Sora-1.2 on PSNR/FVD (adding action constraints slightly perturbs video generation), but this is an acceptable tradeoff since the goal is joint generation. Compared to other joint methods (UVA, PAD, UWM), CoVAR consistently leads on all metrics.

![Figure 4: Qualitative video comparison showing CoVAR's reduced artifacts](https://arxiv.org/html/2512.16023v1/x4.png)

*Figure 4: CoVAR generates robotic arm and manipulated objects with fewer visual artifacts and more physically plausible motion compared to UVA, PAD, and UWM baselines.*

### Main Results — Action Success Rate

**Calvin dataset (5 task categories):**

| Method | Drawer | Cabinet | Light | Pick | Push |
|--------|--------|---------|-------|------|------|
| UVA | 0.875 | 0.667 | 0.711 | 0.758 | 0.785 |
| UWM | 0.813 | 0.733 | 0.644 | 0.576 | 0.714 |
| PAD | 0.781 | 0.467 | 0.489 | 0.485 | 0.642 |
| Unipi | 0.469 | 0.267 | 0.289 | 0.182 | 0.452 |
| **CoVAR** | **1.000** | **0.800** | **0.867** | **0.909** | **0.929** |

**Key finding**: CoVAR achieves 100% on Drawer and significantly outperforms all baselines across all 5 task categories. The gain over the second-best method (UVA) ranges from 13–15% absolute on most tasks.

**Libero90 dataset (3 task categories):**

| Method | Pick | Open & Close | Combination |
|--------|------|-------------|-------------|
| UVA | 0.676 | 0.640 | 0.489 |
| UWM | 0.606 | 0.600 | 0.400 |
| PAD | 0.625 | 0.480 | 0.355 |
| CoVAR (w/o refinement) | 0.592 | 0.520 | 0.422 |
| **CoVAR** | **0.873** | **0.860** | **0.711** |

**Key finding**: Without the Action Refinement Module, CoVAR is actually below UVA on Pick tasks due to the low resolution of Libero90 (128×128). The refinement module is essential, boosting CoVAR from underperforming to the best method by a wide margin (+19.7% over UVA on Pick, +22% on Open & Close, +22.2% on Combination).

**Real-world UR5 experiment:**

| Method | Nut picking | Screw picking | Dowel picking |
|--------|-------------|---------------|---------------|
| Unipi | 0.00 | 0.06 | 0.02 |
| RoboEnvision | 0.04 | 0.10 | 0.12 |
| **CoVAR** | **0.64** | **0.74** | **0.70** |

**Key finding**: Real-world gap is dramatic — Unipi and RoboEnvision both nearly fail (0–12%), while CoVAR achieves 64–74% across all three tasks. This validates that the video-action co-generation approach transfers well to physical robots with real sensor noise.

![Figure 3: Generated video-action pair alignment (red = ground truth, blue = CoVAR predicted)](https://arxiv.org/html/2512.16023v1/x3.png)

*Figure 3: Blue action trajectories (CoVAR predictions) closely track the red ground truth trajectories, demonstrating tight video-action synchronization.*

![Figure 5: Rollout comparison with and without action refinement module](https://arxiv.org/html/2512.16023v1/x5.png)

*Figure 5: Without the Action Refinement Module, coarse actions fail to grasp objects precisely. With refinement, the robot successfully completes manipulation tasks.*

![Figure 6: Real-world consecutive pick-and-place demonstrations on UR5](https://arxiv.org/html/2512.16023v1/x6.png)

*Figure 6: High-precision pick-and-place on the real UR5 robot. The generated video correctly predicts the future scene, and the extracted actions successfully execute the manipulation.*

### Ablation Study

The ablation is conducted on the real-world UR5 dataset to isolate the contribution of each component:

| Variant | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FVD ↓ | Success Rate ↑ |
|---------|--------|--------|---------|-------|----------------|
| w/o Bridge Attention (standard self-attention on concat.) | 16.83 | 0.693 | 0.255 | 137.66 | 0.32 |
| w/o Bridge Attention (standard cross-attention) | 16.56 | 0.645 | 0.263 | 145.26 | 0.20 |
| w/o UNet decoder (linear decoder) | 16.85 | 0.690 | 0.255 | 141.62 | 0.24 |
| w/o video DiT (action-only generation) | — | — | — | — | 0.08 |
| **Full CoVAR** | **17.67** | **0.736** | **0.238** | **133.89** | **0.68** |

**Key findings**:
- **Bridge Attention is the most critical component**: Replacing it with either shared self-attention or cross-attention both dramatically degrade action success (0.32 and 0.20 vs. 0.68) and video quality. Cross-attention (unidirectional) is actually worse than shared self-attention, indicating that bidirectionality matters more than modality-specific projections alone.
- **UNet decoder matters**: Removing multi-scale skip connections and using a simple linear decoder drops success from 0.68 to 0.24, confirming that rich cross-scale feature aggregation is needed for precise action decoding.
- **Video pretraining is essential**: Removing the video DiT entirely (action-only generation without visual grounding) reduces success to 0.08, near chance level. This validates the central hypothesis that video generation provides critical pretrained knowledge that guides action generation.

![Figure 7: Qualitative ablation trajectories comparing Bridge Attention variants](https://arxiv.org/html/2512.16023v1/x7.png)

*Figure 7: Qualitative rollouts comparing the full CoVAR model against Bridge Attention ablations. The full model produces smooth, accurate trajectories; ablated variants show significant deviations.*

---

## Critical Analysis

### Strengths

1. **Elegant modular design**: Bridge Attention solves the modality conflict problem with a minimal change — just separate Q/K/V projections — without requiring a fundamentally different architecture. This makes CoVAR easy to build on top of existing video diffusion models.
2. **Leverages pretrained video diffusion**: By inheriting Open-Sora-1.2's weights for the video branch, CoVAR avoids training from scratch and achieves high visual quality even with limited robot demonstration data (~20K trajectories).
3. **Strong real-world performance**: The 64–74% real-world success rate versus near-zero for baselines is a compelling demonstration that the method transfers to physical deployment, which many video-based policies struggle with.
4. **Thorough ablation**: The ablation systematically ablates each major component (Bridge Attention type, decoder type, video modality) on the real-world dataset — the hardest and most practically relevant setting.

### Limitations

1. **Monocular video only**: CoVAR operates on single-camera 2D video, limiting its ability to reason about 3D scene geometry. Grasping tasks that require precise 3D localization (e.g., bin picking with occlusions, deformable objects) may require stereo or depth inputs.
2. **Low-resolution training data**: The Action Refinement Module is a workaround for a dataset resolution problem. As robot datasets improve in resolution, it becomes less necessary, but for now it adds an extra fine-tuning stage that requires paired video-action data.
3. **Slow inference**: 4 seconds per 35-frame sequence at 30 diffusion steps is too slow for reactive control at 100 Hz. Temporal interpolation bridges this gap but cannot generate reactive responses to unexpected scene changes mid-execution.
4. **No public code or pretrained models**: Reproducibility is limited; practitioners must implement and train from scratch.
5. **Real-world dataset is self-collected and small (1K demos)**: The real-world evaluation cannot be fully reproduced by other researchers, making it harder to verify the claimed performance.

### Reproducibility

- [ ] Code open-sourced
- [ ] Pretrained models available
- [x] Training details complete (batch size partially unspecified, but GPU count, days, and hyperparameter classes are given)
- [x] Datasets accessible (Calvin and Libero90 are public; real-world dataset is not released)

---

## Related Notes

### Based On

- [[视频生成模型|Open-Sora-1.2]]: Video DiT backbone; CoVAR inherits its pretrained weights and extends it with a parallel action branch
- [[整流流匹配|Rectified Flow]]: The flow-matching ODE framework underlying both video and action denoising
- [[DiT]]: Diffusion Transformer architecture used for both the video and action branches

### Compared Against

- [[UniPi]]: Two-stage baseline (generate video → extract actions via inverse dynamics); CoVAR outperforms on all benchmarks by removing the decoupled pipeline bottleneck
- [[PAD]]: Joint diffusion baseline that concatenates all tokens with shared Q/K/V; CoVAR outperforms by using separate modality-specific projections via Bridge Attention
- [[UWM]]: Unified world model baseline using a single shared transformer; CoVAR outperforms across all metrics
- [[RoboEnvision]]: Two-stage method with a lightweight policy model; CoVAR achieves 6–17× higher real-world success rates

---

> [!summary] CoVAR (2025)
> - **Core**: Co-generate video predictions and robot actions via parallel [[DiT]] branches connected by [[WAM|Bridge Attention]] with modality-specific Q/K/V projections
> - **Method**: Extend pretrained Open-Sora-1.2 video DiT with a parallel action DiT; Bridge Attention computes joint attention over concatenated tokens but uses separate per-modality projections; Action Refinement Module for low-resolution datasets
> - **Results**: Best action success on Calvin (1.000 Drawer, 0.929 Push), Libero90 (0.873 Pick), and real UR5 (0.64–0.74); best perceptual video quality (LPIPS) among joint generation methods
> - **Code**: N/A
