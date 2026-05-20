---
title: "ARDuP: Active Region Video Diffusion for Universal Policies"
method_name: "ARDuP"
authors: [Shuaiyi Huang, Mara Levy, Abhinav Shrivastava, Zhenyu Jiang, Yuke Zhu, Anima Anandkumar, Linxi Fan, De-An Huang]
year: 2024
venue: arXiv 2024
tags: [video-generation, latent-idm, active-region, cascaded-implicit, robot-policy, stable-diffusion, bridge]
zotero_collection: 3-Robotics/WAM-Survey
image_source: online
arxiv_html: https://arxiv.org/abs/2406.13301
created: 2026-05-20
---

# Paper Note: ARDuP: Active Region Video Diffusion for Universal Policies

## Metadata

| Item | Content |
|------|---------|
| Institution | University of Maryland, UT Austin, Caltech, NVIDIA |
| Date | June 2024 |
| Project Page | N/A |
| Baselines | UniPi, Diffuser, Transformer BC variants |
| Links | [arXiv](https://arxiv.org/abs/2406.13301) / Code: N/A |

---

## One-Sentence Summary

> ARDuP decomposes video generation into two stages — active region generation (Co-Tracker + SAM → pseudo-mask latents) followed by full video diffusion conditioned on active regions — then extracts 7-DoF actions directly from adjacent latent pairs via a convolutional latent inverse dynamics model (no pixel decoding required), achieving up to +21.3% improvement over UniPi on CLIPort tasks.

---

## Core Contributions

1. **Active Region Decomposition**: Separates "where to act" (active region generation, task-relevant interaction areas) from "what happens" (video generation) — active region conditioning provides strong task grounding that guides video generation toward task-relevant object motions.
2. **Latent Inverse Dynamics Model**: Predicts 7-DoF actions directly from adjacent latent frame pairs (128×128×4) without decoding to pixel space — computationally more efficient than prior methods requiring full RGB reconstruction.
3. **Unsupervised Pseudo-Active Region Extraction**: Co-Tracker (dense point tracking, grid M=60) + movement threshold $\tau=2$ + SAM segmentation generates active region pseudo-labels automatically from videos — no manual annotation needed.
4. **Efficient Latent Pipeline**: Entire action extraction pipeline operates in SD VAE latent space — avoids expensive pixel-space decoding while preserving spatial information needed for action prediction.

---

## Problem Background

### Problem to Solve
Video-conditioned robot policies must generate full video sequences even for simple tasks where only a small spatial region undergoes task-relevant change. How can video generation be focused on task-relevant regions to improve efficiency and policy quality simultaneously?

### Limitations of Existing Methods
- UniPi: generates full video frames; no task-relevant focus; action extraction requires pixel decoding; 65.4% on Place Bowl.
- Standard video diffusion for robotics: expensive pixel-space generation; poor spatial grounding for manipulation.
- Diffuser/BC baselines: no video world model; limited generalization.

### Motivation
Manipulation tasks primarily involve small spatial regions (the robot hand + target object) while the background remains static. Generating accurate task-relevant active regions first, then conditioning video generation on them, focuses the model's capacity on what matters — improving both generation quality and downstream action prediction.

---

## Method Details

### Model Architecture

ARDuP has **three learned components**:

1. **Active Region Generator** $\psi$: Latent diffusion model → task-relevant region masks
2. **Video Planner** $\phi$: Latent video diffusion + temporal modules → H=6 frame latent sequence
3. **Latent IDM** $\pi$: Convolutional layers + linear projection → 7-DoF action vector

**Encoder**: Stable Diffusion VAE (512×512 RGB → 128×128×4 latent; 4× spatial downsampling × 4 channels)

### Core Modules

#### Module 1: Pseudo-Active Region Extraction (Dataset Construction)

**Design Motivation**: Active region labels (which image regions are task-relevant) are not directly available in demonstration datasets. Automated extraction from videos provides scalable pseudo-supervision.

**Implementation**:
- Co-Tracker: dense point tracking on video frames with grid size $M=60$
- Point movement per timestep: $\Delta \mathbf{p}_h = \|\mathbf{p}_h - \mathbf{p}_{h-1}\|_2$

Active points are selected using their average per-step movement across the sequence. A point is deemed "active" (task-relevant) when its average displacement exceeds a pixel threshold:

$$\Delta \bar{\mathbf{p}} = \frac{1}{H}\sum_{h=1}^{H} \|\mathbf{p}_h - \mathbf{p}_{h-1}\|_2 > \tau, \quad \tau = 2$$

Points satisfying this condition are then passed to SAM to obtain coherent region masks $\mathbf{M}$.

The resulting active region frame composites the original initial frame $x_0$ within the detected mask against a white background $x_b$, producing a clean spatial conditioning signal:

$$o = x_0 \circ \mathbf{M} + x_b \circ (1 - \mathbf{M})$$

This representation shows only task-relevant objects while suppressing irrelevant background regions.

#### Module 2: Active Region Generator $\psi$

**Design Motivation**: Active regions depend on both the task semantics (which object to interact with) and the current scene — a dedicated latent diffusion model learns this conditional distribution.

**Implementation**:
- Backbone: modified UNet (Stable Diffusion based)
- Text conditioning: T5-XXL text encoder
- Input: initial frame latent + task text embedding
- Output: active region frame latent $\hat{o}$ (masked image showing task-relevant regions on white background)

#### Module 3: Video Planner $\phi$

**Design Motivation**: Conditioning video generation on active regions guides the model to produce task-relevant object motions in the correct spatial locations.

**Implementation**:
- Backbone: latent video diffusion model with temporal modules
- Input: initial frame latent + active region latent (concatenated with frame latents) + task text
- Output: H=6 frame latent sequence $\{z_1, \ldots, z_H\}$
- Operates entirely in VAE latent space (no pixel decoding during generation)

#### Module 4: Latent Inverse Dynamics Module $\pi$

**Design Motivation**: Actions can be predicted from latent frame pairs without full pixel decoding — latent representations preserve sufficient spatial information for action discrimination.

**Implementation**:
- Input: two adjacent frame latents $(z_t, z_{t+1})$, each $128 \times 128 \times 4$
- Architecture: convolutional layers with skip connections + linear projection
- Output: 7-dimensional action vector (6-DoF end-effector + gripper)
- Training: supervised with ground-truth actions from demonstrations

---

## Experiments

### Datasets

| Dataset | Scale | Characteristics | Usage |
|---------|-------|----------------|-------|
| CLIPort | 110K demonstrations, 11 tasks | Simulated tabletop manipulation | Training/Evaluation |
| BridgeData v2 | 60,096 trajectories (95%/5% split) | Real-world kitchen manipulation | Training/Evaluation |

### CLIPort Success Rate

The following table compares ARDuP against UniPi on three representative CLIPort tasks, demonstrating consistent improvements from active region conditioning:

| Task | UniPi | **ARDuP** | Improvement |
|------|-------|-----------|-------------|
| Place Bowl | 65.4% | **86.7%** | +21.3% |
| Pack Object | 51.8% | **69.0%** | +17.2% |
| Pack Pair | 30.9% | **46.6%** | +15.7% |

ARDuP substantially outperforms UniPi across all CLIPort tasks — +17–21% absolute improvement.

### Ablation: Active Region Quality

The ablation below shows that active region quality is the primary performance bottleneck — the gap between unsupervised pseudo-regions (86.7%) and ground-truth regions (100%) represents the ceiling available from better region estimation:

| Region Type | Place Bowl Success |
|------------|-------------------|
| No active region conditioning | 83.4% |
| Unsupervised pseudo-regions | 86.7% (+3.3%) |
| Supervised active regions | 93.3% (+9.9%) |
| Ground truth regions | 100.0% (+16.6%) |

Even unsupervised pseudo-regions provide +3.3% gain; supervised regions provide +9.9%. The ceiling (GT regions) suggests substantial room for improvement via better region generation.

### Implementation Details

- **VAE**: Stable Diffusion (512×512 → 128×128×4 latent)
- **Text Encoder**: T5-XXL
- **Co-Tracker Grid**: M=60 points
- **Movement Threshold**: $\tau=2$ pixels
- **Video Horizon**: H=6 frames
- **Diffusion Models LR**: 3×10⁻⁴, batch size 96
- **Latent IDM LR**: 5×10⁻⁴, batch size 3,072
- **Hardware**: 8× A100 GPUs
- **Action Dimension**: 7 (6-DoF end-effector + gripper)
- **Execution**: Open-loop

---

## Critical Analysis

### Strengths
1. Active region decomposition elegantly focuses generation capacity on task-relevant spatial areas.
2. Latent IDM avoids expensive pixel decoding — more efficient than pixel-space action extraction.
3. Unsupervised pseudo-label extraction (Co-Tracker + SAM) provides scalable training supervision.

### Limitations
1. **Ablation ceiling**: GT active regions reach 100% but unsupervised pseudo-regions only 86.7% — active region generator quality is the bottleneck.
2. **Open-loop execution**: No closed-loop replanning — sensitive to prediction errors.
3. **H=6 horizon**: Short video horizon (6 frames) limits long-horizon task planning.
4. **Two-stage generation**: Active region generator + video planner both require inference — doubles generation cost.

### Potential Improvements
1. Improved active region generation via cross-attention visualization or explicit object detectors.
2. Closed-loop action execution with mid-task video replanning.
3. Longer video horizons with hierarchical planning.

### Reproducibility
- [ ] Code open-sourced
- [ ] Pretrained models available
- [x] Training details described (LR, batch size, hardware, thresholds)
- [x] Datasets accessible (CLIPort, BridgeData v2 public)

---

## Related Notes

### Based On
- [[Stable Diffusion]]: VAE and UNet backbone for latent video generation
- [[Co-Tracker]]: Dense point tracking for pseudo-active region extraction
- [[SAM]]: Segmentation for active region mask generation

### Compared Against
- [[UniPi]]: Pixel-space video planning; outperformed by +17–21% on CLIPort
- [[Diffuser]]: Trajectory diffusion; outperformed

### Method Related
- [[Latent Inverse Dynamics Model]]: Action extraction from latent frame pairs without pixel decoding
- [[Active Region]]: Task-relevant spatial focus for video generation
- [[Video Diffusion]]: Latent temporal generation backbone

### Hardware/Data Related
- [[CLIPort]]: Simulated manipulation benchmark
- [[BridgeData v2]]: Real-world kitchen manipulation dataset

---

## Quick Reference Card

> [!summary] ARDuP (arXiv 2024)
> - **Core**: Two-stage video diffusion: Active Region Generator (SD UNet + T5-XXL, Co-Tracker+SAM pseudo-labels) → Video Planner (conditioned on active regions) → Latent IDM (conv+linear, 7-DoF) from adjacent latent pairs; no pixel decoding
> - **Method**: SD VAE (512→128×128×4); Co-Tracker M=60, τ=2; H=6 frames; diffusion LR=3e-4 batch 96; IDM LR=5e-4 batch 3072; 8×A100; CLIPort 110K + BridgeV2 60K
> - **Results**: 86.7% vs 65.4% (UniPi) on Place Bowl; +17–21% across CLIPort tasks; unsupervised regions +3.3%, GT regions +16.6%
> - **Code**: N/A

---

*Note created: 2026-05-20*
