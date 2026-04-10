# World models and V-JEPA: LeCun's bet on non-generative intelligence

**V-JEPA represents a fundamentally different approach to building intelligent systems — one that predicts in abstract representation space rather than generating pixels, and has now demonstrated state-of-the-art video understanding, intuitive physics reasoning, and zero-shot robot control.** Rooted in Yann LeCun's 2022 vision for autonomous machine intelligence, the JEPA (Joint Embedding Predictive Architecture) framework sidesteps what LeCun calls the "generative model trap" by learning world models that discard irrelevant perceptual details and focus on semantically meaningful dynamics. With V-JEPA 2 scaling to **1.2 billion parameters** trained on over **1 million hours of video**, and its action-conditioned variant achieving 65–80% success on zero-shot pick-and-place tasks using just 62 hours of unlabeled robot data, the approach has moved from theoretical proposal to empirical validation in under three years. The implications extend beyond video understanding — JEPA challenges the dominant paradigm that next-token prediction and pixel generation are the path to general intelligence.

> **Note (March 2026):** V-JEPA through V-JEPA 2 were developed at Meta FAIR under LeCun's leadership. In November 2025, LeCun departed Meta after clashing with new Chief AI Officer Alexandr Wang (formerly of Scale AI) over the company's all-in pivot to LLMs. LeCun launched **AMI Labs** to continue the JEPA/world model research program, announcing a $1B funding round on March 15, 2026. V-JEPA 2.1 (March 16, 2026) still shipped under the [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) GitHub org — likely already in the pipeline before the departure. Future JEPA releases are expected from AMI Labs. Meta's AI efforts are now led by Wang and co-head Nat Friedman under the Meta Superintelligence Labs banner.

## LeCun's cognitive architecture rejects the generative paradigm

The intellectual foundation for V-JEPA lies in LeCun's June 2022 position paper, *"A Path Towards Autonomous Machine Intelligence,"* which proposes a complete cognitive architecture for autonomous agents built around six differentiable modules: a **perception module**, a **world model**, an **actor** (with reactive and deliberative modes mirroring Kahneman's System 1/System 2), a **cost module** (split into immutable intrinsic costs and a trainable critic), **short-term memory**, and a **configurator** that serves as executive control.

The world model sits at the center of this architecture. Its job is to estimate missing information about the current world state, predict plausible future states given proposed actions, and represent uncertainty through latent variables. Critically, LeCun argues these predictions must occur in **abstract representation space**, not pixel space. A generative model tasked with predicting the next video frame must account for every blade of grass, every water ripple, every carpet texture — details that are both computationally expensive and fundamentally unpredictable over longer horizons. A world model should instead represent a car approaching a fork in terms of position, velocity, and orientation, with a latent variable encoding whether it turns left or right.

This insight is formalized through **energy-based models (EBMs)**. Rather than computing P(y|x), an energy function F(x, y) assigns low energy to compatible (x, y) pairs and high energy to incompatible ones. The system doesn't need to *predict* y from x — it only needs to evaluate whether a proposed y is *compatible* with x. For handling multiple valid futures, a latent variable z parameterizes the space of possibilities: F_w(x, y) = min_z E_w(x, y, z). This framework naturally handles multimodality (multiple valid predictions) without requiring explicit probability distributions over high-dimensional output spaces.

LeCun identifies a critical failure mode: **representation collapse**, where the energy function becomes flat and assigns low energy everywhere. He categorizes training methods into contrastive approaches (push energy up for negative samples) and regularized/non-contrastive approaches (constrain the volume of low-energy regions). He argues strongly for regularized methods because contrastive approaches require negative samples that grow **exponentially with representation dimensionality** — a fundamental scalability barrier.

## JEPA predicts representations, not pixels

The **Joint Embedding Predictive Architecture** operationalizes this vision. Its components are an x-encoder that maps input x to representation s_x, a y-encoder that maps target y to representation s_y, and a predictor that maps (s_x, z) to a predicted representation s̃_y. The energy is simply the prediction error: E(x, y, z) = D(s_y, Pred(s_x, z)).

Three design principles distinguish JEPA from competing self-supervised paradigms:

- **Versus generative approaches** (MAE, diffusion models, autoregressive models): Generative methods predict raw pixels or tokens, forcing the model to reconstruct irrelevant low-level details. JEPA's encoders can learn invariances that discard unpredictable information, producing representations that are both more semantically meaningful and **1.5–6× more computationally efficient** to train. Masked Autoencoders (MAE) achieve 71.5% ImageNet accuracy with >10,000 GPU-hours; I-JEPA reaches 73.3% with ~2,500 GPU-hours.

- **Versus contrastive learning** (SimCLR, CLIP, MoCo): Contrastive methods learn aligned representations by pushing negative pairs apart, but require careful negative sampling and suffer from the dimensionality curse. JEPA uses non-contrastive training — specifically **VICReg** (Variance-Invariance-Covariance Regularization) — which maintains representation quality by ensuring variance in each embedding dimension, minimizing prediction error, and decorrelating embedding components. No negative samples needed.

- **Versus joint embedding without prediction** (CLIP-style): CLIP learns to match representations across modalities but is not predictive. JEPA explicitly predicts one representation from another, enabling the model to learn temporal dynamics and causal structure.

Collapse is prevented through an **exponential moving average (EMA) target encoder** with stop-gradient, inspired by BYOL. The target encoder's weights are a slowly-moving average of the main encoder, ensuring stable prediction targets. Four criteria govern training: maximize information content of both encoder outputs, minimize prediction error, and minimize information in the latent variable z (preventing z from simply copying the target).

## V-JEPA 1 architecture and training mechanics

V-JEPA (published February 2024, accepted at ICLR 2025 with Featured Certification) extends JEPA to video. The architecture uses a **Vision Transformer (ViT)** backbone operating on 3D spatiotemporal patches.

**Tokenization** converts video into tubelets of size **2×16×16** (2 frames × 16 pixels × 16 pixels). For a 16-frame clip at 224×224 resolution, this produces a sequence of tokens processed by the transformer. The flagship models are **ViT-L/16** (~300M parameters) and **ViT-H/16** (~630M parameters).

The **masking strategy** is arguably V-JEPA's most important design choice. Rather than randomly dropping individual patches (which would be trivially solvable via spatiotemporal interpolation in redundant video), V-JEPA uses **multi-block spatiotemporal masking**: spatially contiguous blocks with random aspect ratios (0.75–1.5) are sampled and **repeated across the entire temporal dimension**. Short-range masks cover ~15% of each frame (8 blocks); long-range masks cover ~70% (2 blocks). The overall masking ratio averages **~90%**. This extreme masking forces the model to develop genuine scene understanding rather than exploiting local redundancy.

The **predictor** is a narrow 12-layer transformer with embedding dimension **384** (~22M parameters, roughly ViT-S scale). It receives the encoder's output for visible tokens concatenated with learnable mask tokens carrying positional embeddings for the masked positions. Its output is the predicted representation for each masked position.

The **training loss** is L1 regression between predicted and target representations:

> minimize ‖P_ϕ(Δ_y, E_θ(x)) − sg(Ē_θ(y))‖₁

where sg denotes stop-gradient and Ē is the EMA target encoder. L1 was chosen over L2 for training stability — the optimal L1 predictor computes the conditional median, which is more robust to outliers. Training used **VideoMix2M** (~2 million videos from HowTo100M, Kinetics-400/600/700, and Something-Something v2) with batch size 3072 for 90K iterations using AdamW.

Key V-JEPA 1 results on frozen evaluation (attentive probing, no encoder fine-tuning): **81.9% on Kinetics-400**, **72.2% on Something-Something v2**, and **77.9% on ImageNet-1K** — surpassing prior video self-supervised methods by 4–10 points.

## V-JEPA 2 scales the recipe with four ingredients

V-JEPA 2 (June 2025) preserves the core JEPA framework but applies systematic scaling across four axes:

**Data**: VideoMix2M → **VideoMix22M** (~22 million samples comprising >1 million hours of video). The expanded dataset adds curated YT-Temporal-1B (~19M videos) and ImageNet images treated as 16-frame pseudo-videos. Data curation — filtering YT-Temporal-1B for quality — proved important for downstream performance.

**Model**: The flagship encoder is **ViT-g/16** (~1 billion parameters), with the full model (encoder + predictor) totaling **1.2 billion parameters**. A key architectural change replaces absolute sinusoidal position embeddings with **3D Rotary Position Embeddings (3D-RoPE)**, which partition the feature dimension into three segments for temporal, height, and width axes. This stabilizes training at large scale and enables flexible resolution/duration scaling without retraining.

**Training duration**: 90K → **252K iterations**, using a simplified warmup → constant → cooldown learning rate schedule (peak LR ~5.25×10⁻⁴, minimum ~1×10⁻⁶, weight decay 0.04).

**Resolution**: Progressive training starts at 16 frames / 256×256 during warmup and constant phases, then scales to **64 frames / 384×384** during cooldown. This yields an **8.4× speedup** compared to training at full resolution throughout.

The cumulative impact of these scaling ingredients, starting from V-JEPA 1's ViT-L baseline at 84.2% average across six tasks: data scaling adds +1.0 point, model scaling +1.5, longer training +0.8, with higher resolution providing additional gains — reaching **88.2% average** across six classification benchmarks. On Something-Something v2, V-JEPA 2 achieves **77.3%** (frozen evaluation), substantially outperforming InternVideo's 69.7%.

## The action-conditioned world model enables zero-shot robotics

V-JEPA 2-AC transforms the passive video model into an active world model through a two-stage approach. After pretraining, the V-JEPA 2 encoder is **frozen**, and a new **300M-parameter action-conditioned predictor** is trained on just **62 hours** of unlabeled robot video from the DROID dataset (23K trajectories of Franka Emika Panda arms). This predictor has 24 transformer layers, 16 attention heads, width 1024, and uses **block-causal attention** — each patch feature at time t can attend to all patches, actions, and end-effector states from current and previous timesteps.

The model ingests three input streams per timestep: **visual features** (16×16×1408 per frame from the frozen encoder), **7D actions** (delta end-effector position, orientation, and gripper state), and **7D proprioception** (absolute end-effector state). Training combines a **teacher-forcing loss** (one-step prediction applied simultaneously at T=15 positions) and a **multi-step rollout loss** (T=2 steps using the model's own predictions as input) to combat autoregressive error accumulation.

Planning uses **Model-Predictive Control with Cross-Entropy Method (CEM)**. Given a current observation and goal image, the energy function is the L1 distance in representation space between predicted future state and goal: E = ‖Ē(predicted) − Ē(goal)‖₁. CEM samples 800 candidate action sequences per step on a single NVIDIA RTX 4090, optimizing to minimize this energy. The energy landscape is empirically shown to be **smooth and locally convex**, enabling efficient optimization in **~16 seconds per action** — compared to ~4 minutes for pixel-generation approaches like Cosmos.

Deployed **zero-shot** on Franka arms with RobotiQ grippers in two labs neither present in training data, using uncalibrated monocular RGB cameras, V-JEPA 2-AC achieves **65–80% success on pick-and-place**, **65% on grasping**, and **100% on reaching** — dramatically outperforming Octo (a vision-language-action model pretrained on 1M+ trajectories, achieving only 15% on grasping) and Cosmos (0–30% on manipulation). No task-specific training, rewards, or environment-specific data collection was required.

## The growing JEPA ecosystem spans audio, language, and theory

The JEPA family has expanded rapidly since 2023. **I-JEPA** (CVPR 2023) established the image-domain foundation, achieving competitive ImageNet performance without data augmentations. **MC-JEPA** (July 2023) jointly learned optical flow and content features. **VL-JEPA** (December 2024) extended JEPA to vision-language, predicting continuous text embeddings rather than discrete tokens — achieving 50% parameter reduction and 2.85× faster decoding versus autoregressive VLMs while outperforming GPT-4o and Gemini-2.0 on world-modeling benchmarks with only **1.6B parameters**.

A theoretically significant development is **LeJEPA** (Balestriero & LeCun, November 2025), which provides the first rigorous mathematical foundation for JEPAs by proving that the optimal embedding distribution is an isotropic Gaussian. Its SIGReg (Sketched Isotropic Gaussian Regularization) method eliminates the need for stop-gradient, EMA teachers, and other heuristics — achieving 79% ImageNet top-1 with ViT-H/14 in just 100 epochs (versus I-JEPA's 300). **C-JEPA** (NeurIPS 2024) integrated contrastive and JEPA objectives, improving convergence.

The community has extended JEPA to diverse domains: **A-JEPA** for audio (achieving SOTA on AudioSet), **S-JEPA** for EEG signals, **Brain-JEPA** for neuroscience, **3D-JEPA** and **Point-JEPA** for 3D understanding, **T-JEPA** for tabular data, **ACT-JEPA** for robotics policy learning, and **UI-JEPA** for user interface understanding. The most recent update, **V-JEPA 2.1** (released March 16, 2026), introduces dense predictive loss (all tokens contribute, not just masked ones), deep self-supervision at multiple intermediate encoder layers, and separate tokenizers for images and videos.

## Benchmarks reveal strengths in temporal reasoning with gaps in physical interaction

V-JEPA 2's benchmark profile reveals a model that excels at temporal and motion understanding while remaining competitive on appearance tasks:

| Benchmark | Task type | V-JEPA 2 (ViT-g/384) | Context |
|---|---|---|---|
| Something-Something v2 | Motion classification | **77.3%** | Outperforms InternVideo (69.7%), VideoMAEv2 |
| Epic-Kitchens-100 | Action anticipation | **39.7 R@5** | 44% relative improvement over prior SOTA |
| Diving-48 | Fine-grained motion | **90.2%** | Frozen backbone evaluation |
| PerceptionTest | Video QA | **84.0** | SOTA at 8B scale (with LLM alignment) |
| TempCompass | Temporal QA | **76.9** | SOTA at 8B scale |
| ImageNet-1K | Image classification | **84.6%** | Competitive with DINOv2, SigLIP2 |

On **intuitive physics** (a separate Meta FAIR evaluation), V-JEPA demonstrates statistically significant understanding of object permanence (72.1% vs. 52.5% baseline), continuity, shape constancy, support, and inertia. Notably, both pixel-prediction models and multimodal LLMs perform **near chance** on these tasks, validating the latent-prediction approach. However, V-JEPA struggles with **object-to-object interactions** — collisions, solidity, and gravity — which may require the hierarchical representations LeCun has proposed but not yet implemented.

## Limitations point toward the next generation of research

Several fundamental challenges remain. **Short temporal context** (3–4 second clips of 16 frames) limits understanding of events requiring longer causal reasoning. **Long-horizon planning** degrades as autoregressive predictions accumulate error; current robotics demonstrations are limited to ≤16 seconds, and multi-step tasks require manually specified visual sub-goals. **Camera position sensitivity** in V-JEPA 2-AC means that when the robot base isn't visible or camera angle changes significantly, the implicitly learned action coordinate system becomes ambiguous.

The gap between V-JEPA and human performance remains substantial on physical reasoning benchmarks. On **IntPhys 2** (detecting physics violations), current video models perform at or near chance while humans achieve near-perfect accuracy. On **CausalVQA**, models reach ~60% versus ~95% human accuracy, answering "what happened" reasonably well but struggling with counterfactual and predictive questions. Goal specification in robotics remains limited to **image-based goals** — no natural language interface exists yet for the planning system.

The broader open problems map directly onto LeCun's original proposal. **Hierarchical JEPA** (H-JEPA) — predictions at multiple time scales and abstraction levels — remains theoretical. **Multi-modal integration** (vision + audio + tactile + proprioception) is nascent. The transition from **passive observation to active agency** has been demonstrated only for tabletop manipulation. And the question of whether non-contrastive, non-generative methods can ultimately match or exceed the capabilities of the scaled generative paradigm (GPT-4, Sora, Gemini) remains the central open bet in the field.

## Conclusion

V-JEPA's trajectory from LeCun's 2022 position paper to V-JEPA 2.1's dense self-supervision in early 2026 demonstrates that predicting in abstract representation space is not merely a theoretical preference but a practical advantage — yielding superior sample efficiency, temporal reasoning, and transfer to physical control. The action-conditioned variant's ability to achieve zero-shot robot manipulation from just 62 hours of unlabeled video, outperforming models trained on orders of magnitude more robot data, is perhaps the strongest empirical evidence yet that world models need not generate pixels to understand physics. The key insight that separates JEPA from the generative mainstream is architectural: by placing learned encoders between raw perception and the prediction objective, the system gains the freedom to represent only what matters. Whether this freedom scales to the complexity of real-world autonomous intelligence — hierarchical reasoning, multi-modal integration, open-ended planning — defines the research frontier that V-JEPA has opened but not yet conquered.