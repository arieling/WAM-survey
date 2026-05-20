---
title: "Zero-Shot Robotic Manipulation with Pretrained Image-Editing Diffusion Models"
method_name: "SuSIE"
authors: [Kevin Black, Mitsuhiko Nakamoto, Pranav Atreya, Homer Walke, Chelsea Finn, Aviral Kumar, Sergey Levine]
year: 2024
venue: ICLR 2024
tags: [robot-manipulation, diffusion-models, subgoal-generation, hierarchical-policy, zero-shot-generalization, image-editing, goal-conditioned-rl]
zotero_collection: 3-Robotics/WAM-Survey
image_source: online
arxiv_html: https://arxiv.org/html/2310.10639
created: 2026-05-20
---

# Paper Note: Zero-Shot Robotic Manipulation with Pretrained Image-Editing Diffusion Models

## Metadata

| Field | Content |
|-------|---------|
| Institution | UC Berkeley (RAIL Lab) |
| Date | October 2023 |
| Project Page | https://rail-berkeley.github.io/susie |
| Baselines | RT-2-X, Oracle GCBC, language-conditioned BC |
| Links | [arXiv](https://arxiv.org/abs/2310.10639) / [Code](https://github.com/kvablack/susie) |

---

## One-Line Summary

> SuSIE finetunes InstructPix2Pix on robot and human video data to synthesize visual subgoals that guide a language-agnostic low-level policy, achieving zero-shot manipulation via hierarchical image-based planning.

---

## Core Contributions

1. **Hierarchical image-editing planner**: SuSIE repurposes InstructPix2Pix, a pretrained image-editing diffusion model, as a high-level robotic planner. By finetuning on video data (robot trajectories and human manipulation videos), the model learns to "hallucinate" a plausible next visual state given the current observation and a language instruction, generating a concrete subgoal image rather than an abstract language command. This bridges the gap between internet-scale visual knowledge and robot control.

2. **Decoupled high-level and low-level control**: The system explicitly separates semantic planning (what state to move toward) from motor execution (how to get there). The high-level model operates in visual observation space and is entirely independent of robot action spaces, allowing it to be trained on non-robot video data. The low-level goal-conditioned behavioral cloning (GCBC) policy operates in action space and is never exposed to language at test time, only to goal images — making it more precise and robust than language-conditioned policies.

3. **State-of-the-art zero-shot generalization**: By leveraging internet-scale pretraining in the diffusion model and robot-specific action knowledge in the low-level policy, SuSIE achieves superior generalization to novel objects and scenes without any additional finetuning. It outperforms RT-2-X (a 55-billion parameter vision-language-action model trained on 20× more data) on real-world manipulation tasks, and more than quadruples the previous best result on the CALVIN zero-shot benchmark (0.26 vs. 0.05 for chaining five instructions).

---

## Problem Background

### Problem Being Solved

Robotic manipulation systems trained on limited robot demonstration data struggle to generalize to new environments, new object appearances, or novel object configurations at test time. The core challenge is that robot training datasets are small compared to the scale of internet image and video data; a robot trained to pick a specific red apple in a lab environment may fail entirely when presented with a yellow bell pepper in a different scene. The goal of zero-shot generalization — successfully executing manipulation instructions in environments never seen during training — has remained elusive for most methods.

The CALVIN benchmark formalizes this: a policy is trained on three tabletop environments (A, B, C) and must generalize to an unseen fourth environment (D) with different object placements, textures, and distractors. Chaining five consecutive language-conditioned instructions in this setting was a hard open problem before SuSIE.

### Limitations of Existing Methods

- **Language-conditioned policies (e.g., HULC, MCIL)**: These methods directly map language instructions to actions via a shared embedding. They suffer because language is ambiguous and underspecified for precise motor control — the word "grasp" does not tell the policy the required wrist angle, grasp height, or finger pressure. They also cannot leverage internet-scale non-robot data because they require action labels paired with language.

- **Vision-language-action models (e.g., RT-2)**: Models like RT-2-X scale language-conditioned policies to billions of parameters and train on diverse robot and internet data. However, they are expensive to train (require Google-scale infrastructure), expensive at inference time, and still rely on language as the sole semantic interface — which limits precision. RT-2-X's 55B parameters trained on 20× more data than SuSIE are outperformed on real-world zero-shot tasks.

- **Goal-conditioned policies trained directly on robot data**: A goal-conditioned behavioral cloning (GCBC) policy given goal images can be precise, but at test time a goal image for a novel configuration must be provided by an oracle. Without a way to generate appropriate goal images automatically from language instructions, this approach is not applicable in practice. The "Oracle GCBC" in SuSIE's evaluation represents the theoretical upper bound for a goal-conditioned policy with perfect goal images — and SuSIE actually exceeds this oracle, showing that its hierarchical decomposition confers precision benefits beyond simply supplying better goals.

- **Video prediction models for planning**: Prior work on video prediction for robot planning (e.g., SV2P, RSSM-based models) trained generative models on robot data alone. This limits the diversity and quality of predictions because robot datasets are small. Crucially, these models must predict actions jointly with observations or use complex planning procedures.

### Motivation

The key insight is that **image editing and goal-conditioned control are complementary**: an image-editing model trained on internet-scale video knows what a plausible "next frame" looks like after an action (semantic and visual understanding), while a goal-conditioned policy trained on robot data knows precisely how to move to reach a given visual target (motor precision). By separating these concerns, each component can be trained on the data it is best suited for, and their combination exceeds what either could achieve alone.

A second important insight is that **the diffusion model does not need to output actions** — it only needs to output images. This means it can be trained on any visual data where consecutive frames represent meaningful state transitions: human hand manipulation videos, timelapse videos of object rearrangements, or robot rollouts. This dramatically increases the volume and diversity of data available for training the planning component.

---

## Method

### Architecture Overview

SuSIE uses a **two-level hierarchical architecture** with:
- **Input**: Current observation image $o_t$ (RGB, 256×256) and language instruction $l$
- **High-level backbone**: InstructPix2Pix (Stable Diffusion v1-5 UNet, finetuned on video data)
- **Low-level backbone**: DDPM-based goal-conditioned behavioral cloning policy (`gc_ddpm_bc`)
- **Core modules**: Subgoal diffusion model, goal-conditioned low-level policy
- **Output**: Joint actions $a_t$ executed on the robot
- **Total parameters**: ~860M (diffusion model, SD v1-5 scale) + low-level policy (architecture not specified, trained on BridgeData V2)

![Figure 1 — SuSIE Overview](https://rail-berkeley.github.io/susie/teaser.svg)

**Caption**: SuSIE alternates between two phases. In the high-level phase, the finetuned InstructPix2Pix model takes the current observation and language instruction and generates a subgoal image representing the desired visual state after completing the next subgoal. In the low-level phase, a goal-conditioned policy executes actions to drive the robot toward the subgoal image. Once the subgoal is approximately achieved (or after a fixed number of steps), a new subgoal is generated. The low-level policy is entirely language-agnostic — it receives only image observations and the goal image as input.

### High-Level Subgoal Generator

**Motivation**: A language instruction like "pick up the pepper" is semantically underspecified for a low-level policy — it does not convey the required approach direction, grasp height, or finger configuration. However, an image of the pepper already grasped and lifted provides direct visual supervision that constrains precisely what the robot's end state should look like. The high-level model's job is to translate ambiguous language instructions into concrete visual targets, leveraging broad world knowledge from internet pretraining to fill in visual details that language omits.

**Design**: The high-level model is InstructPix2Pix — a conditional image diffusion model that takes an image and a text instruction and outputs an edited version of the image reflecting the instruction. The base model is Stable Diffusion v1-5, extended with additional conditioning on the input image by concatenating the noised image latent with the VAE-encoded input image across channels in the UNet.

For robotic use, this model is **finetuned on video data** where consecutive frames form (input image, output image) pairs and the corresponding language annotation serves as the instruction. Two data sources are used:

1. **BridgeData V2**: Robot manipulation demonstrations with language annotations. These provide robot-specific visual transitions: how objects move when grasped, how the scene changes after a manipulation action. This ensures the model generates subgoals that are achievable by the robot and visually consistent with the robot's embodiment.

2. **Something-Something (SS-v2)**: A large-scale dataset of 220,847 human hand manipulation video clips annotated with action templates (e.g., "picking [object] up from [surface]", "pushing [object] to the left"). These clips provide diverse object interactions across many object categories and environments not seen in robot data. Training on SS-v2 is critical for zero-shot generalization to novel objects.

The finetuning uses the standard denoising diffusion probabilistic model (DDPM) objective on video frame pairs:

Diffusion Finetuning Loss:

$$
\mathcal{L}_{\text{diff}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\!\left(\mathbf{x}_t, t, c_I, c_L\right) \right\|^2 \right]
$$

**Meaning**: The UNet $\boldsymbol{\epsilon}_\theta$ is trained to predict the noise $\boldsymbol{\epsilon}$ added to the latent $\mathbf{x}_t$ at diffusion timestep $t$, conditioned on the input image conditioning $c_I$ (current observation encoded by VAE) and the language conditioning $c_L$ (instruction text encoded by CLIP text encoder).

**Symbols**:
- $\mathbf{x}_t$: Noised latent of the target subgoal image at diffusion timestep $t$
- $\boldsymbol{\epsilon}$: Gaussian noise added during forward process
- $\boldsymbol{\epsilon}_\theta$: UNet noise prediction network (SD v1-5 architecture)
- $c_I$: VAE encoding of the current observation $o_t$ (concatenated with $\mathbf{x}_t$ in channel dimension)
- $c_L$: CLIP text encoding of the language instruction $l$

The best-performing model checkpoint is trained for **40,000 gradient steps** on combined BridgeData + Something-Something data.

At inference, the model uses classifier-free guidance (CFG) to sharpen the generated subgoal toward the language instruction, sampling with a standard DDIM schedule.

### Low-Level Goal-Conditioned Policy

**Motivation**: Once a subgoal image is generated, a precise motor controller is needed to actually move the robot to achieve that visual target. Language-conditioned policies are imprecise because they cannot exploit the rich spatial information encoded in an image — the exact position of the end effector relative to the object, approach angle, etc. A goal-conditioned policy operating directly on image observations can leverage this spatial information for precise grasps, especially on challenging objects (smooth, lightweight, or unusually shaped).

**Design**: The low-level policy is the `gc_ddpm_bc` agent from the BridgeData V2 repository — a diffusion policy-style goal-conditioned behavioral cloning model. It takes as input:
- The current observation image $o_t$
- The goal image $g$ (subgoal generated by the high-level model)
- Outputs: predicted robot actions $a_t$ over a prediction horizon of **4 steps**

The policy uses **delta-goal relabeling** (`delta_goals` strategy): during training, goal images are sampled as observations from a fixed number of steps ahead in the same trajectory rather than the final frame. This encourages the policy to learn fine-grained, step-wise goal reaching rather than large jumps, which matches the subgoal generation paradigm at test time.

The policy is trained exclusively on BridgeData V2 robot demonstrations and has **no access to language instructions** during training or inference. This is a key design choice — language is entirely handled by the high-level model, and the low-level policy operates purely in the visual space.

### Hierarchical Inference Loop

**Design**: At test time, SuSIE alternates between subgoal generation and subgoal execution in a closed loop:

1. **Subgoal generation**: Given the current observation $o_t$ and language instruction $l$, sample a subgoal image $g \sim p_\theta(g \mid o_t, l)$ using the finetuned diffusion model with DDIM sampling.
2. **Subgoal execution**: Execute the low-level policy $\pi_\phi(a \mid o, g)$ for a fixed number of steps (or until the subgoal appears reached), producing a sequence of robot actions.
3. **Replanning**: After the execution window, observe the new state $o_{t'}$ and repeat from step 1 with the same instruction $l$ (or the next instruction in a chain).

This iterative replanning provides robustness to execution errors: if the robot fails to fully achieve a subgoal, the diffusion model can regenerate an adjusted subgoal from the new current state.

The separation of concerns also means the high-level model generates subgoals at a **coarser temporal resolution** (every $k$ steps) rather than every timestep, which is computationally efficient given that diffusion sampling is expensive.

### Training Objective

The overall system is trained in two independent stages — there is **no joint end-to-end training**:

**Stage 1 — Subgoal model finetuning**: Finetune InstructPix2Pix on (observation, next-frame, language) triples from BridgeData V2 and Something-Something using the DDPM denoising loss described above. The VAE and CLIP text encoder are kept frozen; only the UNet weights are updated.

**Stage 2 — Low-level policy training**: Train `gc_ddpm_bc` on BridgeData V2 robot demonstrations using standard behavioral cloning with delta-goal image relabeling. This stage is entirely independent of the subgoal model.

Low-Level Policy Loss:

$$
\mathcal{L}_{\text{BC}} = \mathbb{E}_{(o_t, g, a_t) \sim \mathcal{D}} \left[ \left\| a_t - \pi_\phi(o_t, g) \right\|^2 \right]
$$

**Meaning**: The policy $\pi_\phi$ is trained to predict demonstrated actions $a_t$ given current observation and goal image, under mean-squared-error loss. In the diffusion policy variant, this loss is applied to the denoised action prediction rather than a direct regression.

**Symbols**:
- $\mathcal{D}$: BridgeData V2 robot demonstration dataset
- $o_t$: Current observation image
- $g$: Goal image sampled by delta-goal relabeling from the same trajectory
- $a_t$: Demonstrated robot action (end-effector delta or joint velocities)
- $\pi_\phi$: Goal-conditioned policy network parameterized by $\phi$

---

## Experiments

### Datasets and Benchmarks

| Dataset | Scale | Characteristics | Usage |
|---------|-------|-----------------|-------|
| BridgeData V2 | ~60,000 demonstrations | Multi-task, multi-scene tabletop robot manipulation with language annotations | Train subgoal model (high-level) + train low-level policy |
| Something-Something v2 | 220,847 video clips | Human hand manipulation of diverse objects, fine-grained action templates | Finetuning subgoal model only (no robot actions) |
| CALVIN (ABC→D) | 4 simulated environments | Tabletop manipulation with 34 tasks; zero-shot eval on env D unseen during training | Simulated evaluation (subgoal model + low-level policy) |
| Real-world BridgeData scenes | 3 novel scenes (A, B, C) | Different tabletop setups with novel objects never seen in training | Zero-shot real-world evaluation |

### Implementation Details

- **High-level backbone**: InstructPix2Pix (Stable Diffusion v1-5 UNet, `FlaxUNet2DConditionModel`); VAE and CLIP text encoder frozen
- **Low-level policy**: `gc_ddpm_bc` from BridgeData V2 repository; action prediction horizon of 4 steps; delta-goal relabeling
- **Input resolution**: 256×256 pixels
- **Diffusion training steps**: 40,000 steps on BridgeData + Something-Something
- **Framework**: JAX (specific version pinned in requirements.txt)
- **Hardware**: Not reported in detail; JAX-based training implies TPU/GPU cluster
- **Inference**: DDIM sampling for subgoal generation; standard forward pass for low-level policy

### Main Results

#### CALVIN Benchmark (Simulated, Zero-Shot ABC→D)

SuSIE's most striking result is on the CALVIN ABC→D zero-shot benchmark, where the policy is trained on environments A, B, and C and evaluated on a completely unseen environment D. The metric is the number of tasks successfully completed in a chain of 5 consecutive instructions; longer chains require the policy to maintain task progress across multiple steps.

| Method | Avg. Chain Length (max 5) | Notes |
|--------|--------------------------|-------|
| HULC (prior SOTA) | ~0.5 | Language-conditioned, trained on ABCD |
| Previous best (zero-shot setting) | 0.05 | Best prior zero-shot method |
| **SuSIE (ours)** | **0.26** | **5.2× improvement over prior best** |

**Key finding**: SuSIE achieves a chain length of 0.26 in the zero-shot setting, more than five times the previous best of 0.05. This demonstrates that leveraging internet-scale pretrained image editing for subgoal generation enables qualitatively better generalization than prior language-conditioned approaches trained on robot data alone.

#### Real-World Evaluation (Novel Scenes with Novel Objects)

Real-world experiments test SuSIE on three novel tabletop scenes (A, B, C) with objects not present in training data, including challenging items like a yellow bell pepper (lightweight, smooth, wide), which require precise grasping strategies.

| Method | Scene A | Scene B | Scene C | Average |
|--------|---------|---------|---------|---------|
| Language-conditioned BC | — | — | — | Low |
| RT-2-X (55B params, 20× more data) | lower | lower | lower | lower |
| Oracle GCBC (ground-truth goals) | — | — | — | lower than SuSIE |
| **SuSIE** | **0.87** | **0.50** | **0.88** | **0.75** |

**Key finding**: SuSIE achieves the best success rate across all three novel scenes, with an average of approximately 0.75. Notably, it outperforms RT-2-X despite using a model that is orders of magnitude smaller and trained on far less data. Most surprisingly, SuSIE also outperforms Oracle GCBC — a baseline that receives ground-truth goal images from the dataset rather than generated ones. This suggests that the hierarchical decomposition itself (not just better goal images) provides precision benefits, possibly because the diffusion-generated subgoals are better calibrated to the low-level policy's training distribution than held-out dataset frames.

![Figure — Real-World Scene Generalization](https://rail-berkeley.github.io/susie/scenes_combined.svg)

**Caption**: SuSIE performance across three novel real-world manipulation scenes (A, B, C) with previously unseen objects. The system consistently achieves the highest success rates compared to all baselines including RT-2-X.

### Ablation Study

While the paper does not report a traditional ablation table with isolated component removals, several design choices are directly validated by the experimental comparisons:

| Configuration | Observation | Design Insight |
|---------------|-------------|----------------|
| SuSIE (BridgeData + SS-v2) | Best overall | SS-v2 human video data improves zero-shot generalization |
| SuSIE (BridgeData only) | Worse on novel objects | Without human video diversity, the diffusion model generalizes less to unseen objects |
| Oracle GCBC (ground-truth goals) | Lower than SuSIE | Hierarchical decomposition helps beyond just semantic goal specification |
| RT-2-X (55B, language-conditioned) | Lower than SuSIE | Scale alone does not compensate for the precision advantage of image-based goals |
| Language-conditioned BC | Much lower | Language is an imprecise interface for motor control |

**Key finding**: The combination of pretrained image editing + robot finetuning + human video data (Something-Something) is essential. Removing any component degrades performance. The fact that SuSIE beats Oracle GCBC validates that hierarchical decomposition — not just goal image quality — is a key factor.

### Additional Analysis

#### Precision on Challenging Objects

SuSIE's most notable qualitative result is its ability to grasp a yellow bell pepper with 0.5 success rate, while RT-2-X achieves 0.0–0.1. Bell peppers are challenging because they are lightweight (easy to knock over), have a smooth surface (prone to slipping), and have a wide, flat shape requiring precise grasp height calibration. The image-based subgoal provides the low-level policy with a concrete visual target showing the pepper already grasped, which specifies the precise required configuration implicitly.

#### Error Recovery

The iterative replanning loop provides an automatic error recovery mechanism. If the robot partially misexecutes a subgoal (e.g., nudges the object instead of grasping), the diffusion model re-observes the new state and generates an adjusted subgoal from the current position. This closed-loop correction is not possible in open-loop language-conditioned policies.

---

## Critical Analysis

### Strengths

1. **Elegant decoupling of semantic understanding and motor precision**: By separating the "what" (image editing) from the "how" (goal-conditioned control), SuSIE lets each component use its ideal data modality. The diffusion model benefits from internet-scale pretraining; the low-level policy benefits from precise action supervision. This decomposition avoids forcing a single model to do both semantic reasoning and precise control simultaneously.

2. **Data efficiency via heterogeneous training sources**: The high-level model can train on any video data with language annotations — human manipulation videos, robot rollouts, or even narrated YouTube content — without requiring action labels. This dramatically expands the effective training distribution and is directly responsible for the improvement in zero-shot object generalization.

3. **Compatibility with existing robot learning infrastructure**: The low-level policy (`gc_ddpm_bc`) is a standard, unmodified component from the BridgeData V2 ecosystem. This means SuSIE can plug into existing robot learning setups with minimal changes, and improvements to low-level goal-conditioned policies automatically transfer to SuSIE.

4. **Practical outperformance of much larger models**: SuSIE uses ~860M parameters and trains on a fraction of RT-2-X's data, yet outperforms it in zero-shot real-world tasks. This demonstrates that architectural design choices (hierarchical decomposition, image-based goal specification) can be more important than raw scale.

### Limitations

1. **Fixed-interval subgoal generation**: SuSIE replans at regular fixed intervals regardless of task progress. If the robot is mid-grasp, regenerating a subgoal may produce an inconsistent or contradictory goal image. Adaptive replanning based on task progress monitoring would improve reliability on longer-horizon tasks (see TaKSIE, which improves on this).

2. **Subgoal quality is not guaranteed**: The diffusion model may generate physically implausible or ambiguous subgoals, particularly for tasks involving interactions not well-represented in training data. There is no explicit consistency check between the generated subgoal and the current physical state.

3. **Two-stage training is not end-to-end**: The subgoal model and low-level policy are trained independently. There may be a distribution mismatch between the diffusion-generated subgoal images and the goal images the low-level policy was trained on (real dataset frames). The fact that SuSIE outperforms Oracle GCBC suggests this mismatch is manageable, but it could become a bottleneck in harder settings.

4. **Inference latency**: Diffusion model sampling (even with DDIM) is computationally expensive compared to a single forward pass of a language-conditioned policy. The system requires replanning at every $k$ steps, which may limit real-time performance on fast-moving tasks.

5. **Real-world evaluation scope**: The real-world experiments are conducted on a single robot platform (BridgeData-style tabletop setup) with a limited set of manipulation tasks. Generalization to dexterous manipulation, mobile manipulation, or contact-rich tasks is not evaluated.

### Potential Improvements

1. **Adaptive replanning**: Replace fixed-interval subgoal generation with a learned or heuristic trigger based on task progress estimation (distance to current subgoal, trajectory confidence).
2. **Joint finetuning of subgoal model and low-level policy**: A distillation or co-training procedure could reduce the distribution mismatch between generated subgoals and the policy's training distribution.
3. **Video-based subgoal sequences**: Instead of generating a single next subgoal, generate a short video of the full task execution and extract keyframes as a planned subgoal sequence.

### Reproducibility

- [x] Code open-sourced (https://github.com/kvablack/susie)
- [x] Pretrained models available (HuggingFace: kvablack/susie, trained for 40k steps)
- [x] Training details complete (dataset, steps, architecture documented in README)
- [x] Datasets accessible (BridgeData V2 and Something-Something v2 are public)

---

## Related Notes

### Based On

- InstructPix2Pix: The pretrained image-editing model that SuSIE finetunes as its high-level planner. SuSIE's key contribution is discovering that this model can be adapted for robotic subgoal generation by finetuning on robot and human video data.
- BridgeData V2: The robot manipulation dataset used to train both the subgoal model finetuning and the low-level policy. SuSIE uses the `gc_ddpm_bc` policy implementation directly from the BridgeData V2 codebase.
- Stable Diffusion: The underlying generative model architecture (SD v1-5 UNet, VAE, CLIP text encoder) on which InstructPix2Pix and therefore SuSIE's high-level model is built.

### Compared Against

- RT-2: RT-2-X is the primary strong baseline — a 55B-parameter VLA model trained on 20× more data. SuSIE outperforms it on zero-shot real-world tasks, demonstrating that hierarchical image-based planning is more data-efficient than scaling a single VLA.
- GCBC: Oracle GCBC represents the idealized upper bound for goal-conditioned control with perfect goal images. SuSIE surpassing this baseline validates that the hierarchical loop itself (not just goal quality) confers benefits.
- HULC: A strong language-conditioned manipulation policy for CALVIN; the prior SOTA on CALVIN before SuSIE.

### Method Related

- Hierarchical Reinforcement Learning: SuSIE implements a temporal abstraction hierarchy where the high-level model sets subgoals and the low-level policy achieves them, analogous to options frameworks in HRL.
- Diffusion Policy: The `gc_ddpm_bc` low-level policy is a diffusion-based behavioral cloning method; the same DDPM denoising framework underlies both the subgoal generator and the action predictor.
- Goal-Conditioned Behavioral Cloning: The foundational framework for the low-level policy, where demonstrations are relabeled with reached states as goals (HER-style).
- Classifier-Free Guidance: Used during inference in the high-level diffusion model to sharpen subgoal generation toward the language instruction.

### Hardware / Data

- BridgeData V2: Primary robot dataset — tabletop manipulation demonstrations with WidowX robot arm, language annotated
- Something-Something v2: Human hand manipulation video dataset, 220,847 clips; critical for zero-shot object generalization

---

> [!summary] Zero-Shot Robotic Manipulation with Pretrained Image-Editing Diffusion Models (2024)
> - **Core**: Finetune InstructPix2Pix on robot and human video to generate visual subgoals that guide a language-agnostic goal-conditioned policy in a hierarchical loop
> - **Method**: Two-stage: (1) finetuned InstructPix2Pix generates subgoal images from current observation + language; (2) gc_ddpm_bc policy executes to reach subgoal
> - **Results**: 0.26 avg. chain length on CALVIN zero-shot (5× prior best 0.05); outperforms RT-2-X (55B params) on real-world novel-object tasks with avg ~0.75 success rate
> - **Code**: https://github.com/kvablack/susie
