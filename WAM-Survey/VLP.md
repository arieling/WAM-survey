---
title: "Video Language Planning"
method_name: "VLP"
authors: [Yilun Du, Mengjiao Yang, Pete Florence, Fei Xia, Ayzaan Wahid, Brian Ichter, Pierre Sermanet, Tianhe Yu, Pieter Abbeel, Joshua B. Tenenbaum, Leslie Kaelbling, Andy Zeng, Jonathan Tompson]
year: 2023
venue: arXiv 2023
tags: [video-generation, long-horizon-planning, tree-search, robot-policy, cascaded-explicit, vlm]
zotero_collection: 3-Robotics/WAM-Survey
image_source: online
arxiv_html: https://arxiv.org/abs/2310.10625
created: 2026-05-20
---

# Paper Note: Video Language Planning

## Metadata

| Item | Content |
|------|---------|
| Institution | Google DeepMind, MIT, UC Berkeley |
| Date | October 2023 |
| Project Page | https://video-language-planning.github.io/ |
| Baselines | [UniPi](UniPi.md), PaLM-E, RT-2, LAVA |
| Links | [arXiv](https://arxiv.org/abs/2310.10625) / Code: N/A |

---

## One-Sentence Summary

> VLP synergizes Vision-Language Model (VLM) and Text-to-Video Model via forward tree search: the VLM proposes next-step text actions and scores candidate futures, while the video model simulates their outcomes — enabling long-horizon video plans that scale with compute budget across diverse real robot platforms.

---

## Core Contributions

1. **VLP Algorithm**: Tree search combining VLM as policy + value function with text-to-video model as dynamics model, producing long-horizon (hundreds of frames) multimodal (video + language) plans.
2. **Compute-Scalable Planning**: Plan quality monotonically improves with branching factor/beam width — a key advantage over single-pass generation (UniPi).
3. **Goal-Conditioned Action Extraction**: Uses short-horizon goal-conditioned policies conditioned on each synthesized intermediate frame (rather than inverse dynamics), reducing execution burden.
4. **Multi-Platform Validation**: Demonstrated on Language Table robot, 7DoF mobile manipulator, and 14DoF bi-manual ALOHA platform.

---

## Problem Background

### Problem to Solve
Long-horizon robot manipulation tasks require both high-level semantic reasoning (what to do next) and low-level visual dynamics (how the world changes). Neither LLMs/VLMs alone nor video models alone suffice.

### Limitations of Existing Methods
- LLMs/VLMs work with text abstractions but cannot reason about physical dynamics or visual constraints.
- UniPi trains video models directly on long-horizon goals — this degrades quickly for complex, multi-step tasks (only 2-4% completion on make-line tasks).
- Direct end-to-end models (RT-2, LAVA) trained on long-horizon task-action pairs fail to generalize.

### Motivation
Forward tree search enables discovering long-horizon plans by evaluating and pruning many short-horizon video branches, leveraging the complementary strengths of VLMs (semantic reasoning) and video models (visual dynamics).

---

## Method Details

### Model Architecture

VLP is a **planning algorithm** combining three learned modules:

- **VLM Policy** $\pi_{VLM}(x, g) \to a$: Given current image $x$ and goal $g$, generates candidate text actions; implemented via fine-tuned PaLM-E (12B)
- **Video Dynamics Model** $f_{VM}(x, a) \to x_{1:S}$: Given image $x$ and text action $a$, synthesizes short-horizon video; implemented as text-to-video diffusion model (same architecture as UniPi)
- **VLM Heuristic Function** $H_{VLM}(x, g)$: Given image $x$ and goal $g$, predicts steps-to-completion scalar; implemented via fine-tuned PaLM-E
- **Goal-Conditioned Policy** $\pi_{control}(x, x_g) \to u$: Low-level controller tracking synthesized subgoals; LAVA architecture with ResNet image encoder replacing CLIP text encoder

### Core Modules

#### Module 1: Parallel Hill-Climbing Tree Search (Algorithm 1)

**Design Motivation**: Directly applying VLM greedily fails at long horizons; tree search allows exploring and evaluating multiple future branches.

**Implementation**:
- Initialize $B$ parallel video plan beams
- At each step: for each beam, sample $A$ text actions from $\pi_{VLM}$, synthesize $D$ videos per action via $f_{VM}$ → evaluate $A \times D$ candidates with $H_{VLM}$
- Keep video with highest heuristic, extend beam
- Every 5 steps: replace lowest-value beam with copy of highest-value beam
- Final plan: beam with highest $H_{VLM}$ value

The long-horizon video plan optimization objective is to find the video sequence that maximizes the VLM's heuristic estimate of proximity to the goal at the final frame:

$$
x^*_{1:H} = \arg\max_{x_{1:H} \sim f_{VM}, \pi_{VLM}} H_{VLM}(x_H, g)
$$

where $x_{1:H}$ is the sequence of $H$ image frames constituting the video plan, $f_{VM}$ is the video dynamics model (text-to-video), $\pi_{VLM}$ is the VLM policy generating text action proposals, and $H_{VLM}(x_H, g)$ is the heuristic value — the negated predicted steps-to-completion.

#### Module 2: Exploitative Dynamics Prevention

**Design Motivation**: Greedy optimization of $H_{VLM}$ can exploit video model artifacts (teleporting objects, occluded bad states).

**Implementation**: Discard generated videos where $H_{VLM}$ increases by more than a fixed threshold in a single step.

#### Module 3: Receding Horizon Replanning

**Design Motivation**: Fixed planning horizon can't complete very long tasks; accumulated execution error degrades tracking.

**Implementation**: Generate fixed-horizon video plan, execute for fixed steps, then replan from current observation — rolling horizon execution.

#### Module 4: Goal-Conditioned Action Extraction

The goal-conditioned policy is trained to predict action $u_t$ given current image $x_t$ and a future subgoal image $x_{t+h}$ sampled from demonstration trajectories:

$$
\pi_{control}(x_t, x_{t+h}) \to u_t
$$

where $x_t$ is the current observation image, $x_{t+h}$ is the future goal image (randomly sampled horizon $h$), and $u_t$ is the low-level control action to execute. Conditioning on every intermediate frame from the generated video rather than just the final frame provides the best performance (92% vs. 80% Group-by-Color completion).

---

## Experiments

### Datasets

| Dataset | Scale | Characteristics | Usage |
|---------|-------|----------------|-------|
| Language Table (sim+real) | ~20k trajectories, 400k text labels | Block manipulation tasks | Training/Testing |
| RT-1 Dataset | Large-scale | 7DoF mobile manipulation | Training VLM/Video model |
| ALOHA demonstrations | ~1200 teleoperated | Kitchen stacking, bi-manual | Training 14DoF experiments |

### Results

The following table shows how video plan accuracy scales with the compute budget (number of beams and branches).

**Figure 3: Planning Budget vs. Performance**

| Beams | Language Branch | Video Branch | Line Performance |
|-------|----------------|--------------|-----------------|
| 1 | 1 | 1 | 4% |
| 1 | 1 | 4 | 10% |
| 1 | 4 | 4 | 22% |
| 2 | 4 | 4 | **56%** |

Performance scales monotonically with compute budget. Full VLP (Beam 2, Branch 16) dramatically outperforms no-planning (Beam 1, Branch 1).

The following table compares video plan synthesis quality between UniPi and VLP variants; the VLM heuristic is critical — removing it drops performance dramatically.

**Table 1: Video Plan Accuracy**

| Model | Sim Move Area | Sim Group Color | Sim Make Line | Real Move Area | Real Group Color | Real Make Line |
|-------|--------------|-----------------|---------------|---------------|-----------------|----------------|
| UniPi | 2% | 4% | 2% | 4% | 12% | 4% |
| VLP (No Value) | 10% | 42% | 8% | 20% | 64% | 4% |
| **VLP (Ours)** | **58%** | **98%** | **66%** | **78%** | **100%** | **56%** |

VLP outperforms UniPi by 10-29x on video plan synthesis.

**Table 2: Execution Performance on Long Horizon Tasks**

| Method | Move Area Reward | Move Area Completion | Group Color Reward | Group Color Completion | Make Line Reward | Make Line Completion |
|--------|-----------------|---------------------|-------------------|----------------------|-----------------|----------------------|
| UniPi | 30.8 | 0% | 44.0 | 4% | 44.0 | 4% |
| LAVA | 59.8 | 22% | 50.0 | 2% | 33.5 | 0% |
| RT-2 | 18.5 | 0% | 46.0 | 26% | 36.5 | 2% |
| PaLM-E | 36.5 | 0% | 43.5 | 2% | 26.2 | 0% |
| **VLP** | **87.3** | **64%** | **95.8** | **92%** | **65.0** | **16%** |

VLP substantially outperforms all baselines on actual robot execution. Group by color reaches 92% completion vs. 26% for the next best (RT-2).

The following table compares action extraction strategies, showing that conditioning the goal policy on every intermediate frame from the video plan is optimal.

**Table 4: Action Extraction Comparison**

| Action Inference | Group Color Score | Group Color Completion |
|-----------------|-------------------|----------------------|
| Inverse Dynamics | 89.7 | 80% |
| Goal Policy (Last frame) | 85.0 | 66% |
| Goal Policy (Every frame) | **95.8** | **92%** |

Conditioning goal policy on every intermediate frame outperforms both sparse conditioning and inverse dynamics.

### Implementation Details

- **Video Model**: Same as UniPi — Video U-Net with DDIM sampler (64 base + 4 super-res steps), CFG scale 5; 24×40 base → 48×80 → 192×320 resolution
- **VLM**: 12B PaLM-E, fine-tuned jointly for policy + heuristic prediction
- **Goal-Conditioned Policy**: LAVA architecture (CLIP text encoder replaced with ResNet image encoder for goal)
- **Hardware**: Video model — 64 TPUv3 pods, 3 days; VLM — 64 TPUv3 pods, 1 day; Goal policy — 16 TPUv3 pods, 1 day
- **Planning Time**: ~30 min per video plan for Language Table (horizon 16)
- **Beam Config**: B=2, language branch A=4, video branch D=4

---

## Critical Analysis

### Strengths
1. Elegant combination: VLM handles long-horizon semantics, video model handles visual dynamics — each plays to its strength.
2. Compute-scalable: more planning → better results, enabling quality-time trade-off at inference.
3. Demonstrated across three real hardware platforms with different DoF.

### Limitations
1. Image-only world state representation: no 3D, no physics, no mass/force modeling.
2. Video model dynamics imperfect: objects sometimes teleport or disappear — exploitative planning is a known failure mode.
3. Separate per-domain training: video models and VLMs trained separately per domain, no unified cross-domain model.
4. Inference cost: ~30 min per plan, ~1 hour per environment execution.

### Potential Improvements
1. Multi-view video plans for better 3D reasoning (partially explored with ALOHA).
2. Reinforcement learning feedback for physics-aware video dynamics.
3. Larger, unified video+VLM models pretrained on internet data for better generalization.

### Reproducibility
- [ ] Code open-sourced
- [ ] Pretrained models available
- [x] Training details complete (Appendix A)
- [x] Datasets accessible (Language Table, RT-1 dataset, ALOHA)

---

## Related Notes

### Based On
- [UniPi](UniPi.md): Direct predecessor; VLP replaces single-pass video generation with tree search
- PaLM-E: VLM backbone for both policy and heuristic function
- Video Diffusion Model: Core dynamics model for short-horizon rollouts

### Compared Against
- [UniPi](UniPi.md): Direct comparison; VLP outperforms by ~10-29x on video plan quality
- RT-2: Strong VLA baseline; VLP outperforms significantly on long-horizon tasks
- LAVA: Language-conditioned BC baseline

### Method Related
- Tree Search: Core planning algorithm (parallel hill climbing)
- Receding Horizon Control: Replanning strategy for long-horizon execution
- Goal-Conditioned Policy: Action execution module

---

## Quick Reference Card

> [!summary] VLP (2023)
> - **Core**: Tree search over video+language space using VLM (policy+value) + text-to-video model (dynamics)
> - **Method**: Parallel beam search, B=2 beams, A=4 language branches, D=4 video branches; goal-conditioned policy for execution
> - **Results**: 92% completion on Group-by-Color vs. 26% next best (RT-2); demonstrated on 3 real robot platforms
> - **Code**: https://video-language-planning.github.io/

---

*Note created: 2026-05-20*
