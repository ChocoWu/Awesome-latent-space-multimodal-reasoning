# Awesome Latent-Space Multimodal Reasoning [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> A curated list of papers on **reasoning in latent / continuous / discrete-concept space**, with a strong focus on the **multimodal** setting (vision, video, audio, action). The line that connects everything here: *reasoning is not done by emitting more text tokens, but by iterating, refining, or expanding **internal representations** — hidden states, continuous thoughts, discrete concept codes, latent visual tokens, learned actions, or world-model rollouts.*

Scope: 2024–2026, with selected seminal precursors. Entry format: **Title** [arXiv:ID] — one-sentence summary. PRs welcome.

> ⚠️ **Note on arXiv IDs.** IDs were collected via live web search where possible; a small number of older entries are recalled from training data and marked `[arXiv:?]` when not independently verifiable. Please open an issue if you spot any ID that is wrong.

---

## Table of Contents

- [0. Surveys & Position Papers](#0-surveys--position-papers)
- [1. Continuous-Thought CoT (the COCONUT line)](#1-continuous-thought-cot-the-coconut-line)
- [2. Implicit CoT, Thinking / Pause / Filler Tokens](#2-implicit-cot-thinking--pause--filler-tokens)
- [3. Looped & Recurrent-Depth Transformers](#3-looped--recurrent-depth-transformers)
- [4. Discrete Latent / Concept-Level Language Models](#4-discrete-latent--concept-level-language-models)
- [5. Discrete-Diffusion & Masked LM Reasoning](#5-discrete-diffusion--masked-lm-reasoning)
- [6. Multimodal Latent CoT (Reasoning in Visual Latents)](#6-multimodal-latent-cot-reasoning-in-visual-latents)
- [7. Visual Chain-of-Thought (Classics & Datasets)](#7-visual-chain-of-thought-classics--datasets)
- [8. Thinking *with* Images — Sketchpad, Whiteboard, Zoom](#8-thinking-with-images--sketchpad-whiteboard-zoom)
- [9. Long-CoT / o1-style Multimodal Reasoners](#9-long-cot--o1-style-multimodal-reasoners)
- [10. RL & Process Rewards for Multimodal Reasoning](#10-rl--process-rewards-for-multimodal-reasoning)
- [11. Image / Video Generation as Reasoning](#11-image--video-generation-as-reasoning)
- [12. Unified Multimodal Models (Shared Latent for Understand + Generate)](#12-unified-multimodal-models-shared-latent-for-understand--generate)
- [13. World Models, JEPA & Latent-Action Reasoning](#13-world-models-jepa--latent-action-reasoning)
- [14. Audio / Speech Latent Reasoning](#14-audio--speech-latent-reasoning)
- [15. Theory & Mechanistic Analyses](#15-theory--mechanistic-analyses)

---

## 0. Surveys & Position Papers

- **A Survey on Latent Reasoning** [[arXiv:2507.06203](https://arxiv.org/abs/2507.06203)] — Comprehensive map of latent CoT methods (continuous thought, recurrent depth, hidden-state propagation, infinite-depth masked diffusion).
- **Reasoning Beyond Language: A Comprehensive Survey on Latent Chain-of-Thought Reasoning** [[arXiv:2505.16782](https://arxiv.org/abs/2505.16782)] — Taxonomy of intrinsic vs. auxiliary latent CoT; covers thinking tokens, looped transformers, continuous embeddings.
- **Thinking with Images for Multimodal Reasoning: Foundations, Methods, and Future Frontiers** [[arXiv:2506.23918](https://arxiv.org/abs/2506.23918)] — Surveys the "think *with* images" paradigm — three-stage evolution from external tools → programmatic ops → intrinsic visual imagination.
- **From Perception to Cognition: A Survey of Vision-Language Interactive Reasoning in MLLMs** [[arXiv:2509.25373](https://arxiv.org/abs/2509.25373)] — Decomposes VL reasoning into perception and cognition, with emphasis on iterative latent re-examination.
- **Multimodal Chain-of-Thought Reasoning: A Comprehensive Survey** [[arXiv:2503.12605](https://arxiv.org/abs/2503.12605)] — End-to-end survey of MM-CoT methods, datasets, and benchmarks.
- **Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models** [[arXiv:2503.16419](https://arxiv.org/abs/2503.16419)] — Includes latent / compressed CoT as an efficiency lever.
- **Efficient Reasoning Models: A Survey** [[arXiv:2504.10903](https://arxiv.org/abs/2504.10903)] — Catalogs continuous-thought, looped, and compressed-CoT models for inference efficiency.
- **Beyond Chain-of-Thought: A Survey of Chain-of-X Paradigms for LLMs** [[arXiv:2404.15676](https://arxiv.org/abs/2404.15676)] — Broader CoT survey with an implicit / latent CoT branch.

---

## 1. Continuous-Thought CoT (the COCONUT line)

- **Training Large Language Models to Reason in a Continuous Latent Space (COCONUT)** [[arXiv:2412.06769](https://arxiv.org/abs/2412.06769)] — Seminal: feeds the last hidden state back as the next input embedding, performing CoT directly in continuous space; enables BFS-like superposition.
- **CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation** [[arXiv:2502.21074](https://arxiv.org/abs/2502.21074)] — Self-distills explicit CoT into compact continuous thoughts by aligning hidden states of teacher (with CoT) and student (without).
- **SoftCoT: Soft Chain-of-Thought for Efficient Reasoning with LLMs** [[arXiv:2502.12134](https://arxiv.org/abs/2502.12134)] — A frozen LLM is augmented with soft thought tokens produced by a lightweight assistant, giving parameter-efficient latent reasoning.
- **SoftCoT++: Test-Time Scaling with Soft Chain-of-Thought Reasoning** [[arXiv:2505.11484](https://arxiv.org/abs/2505.11484)] — Extends SoftCoT with diverse soft-thought sampling + contrastive learning for test-time scaling.
- **CCoT: Compressed Chain-of-Thought via Dense Representations** [[arXiv:2412.13171](https://arxiv.org/abs/2412.13171)] — Generates variable-length "contemplation" tokens as dense compressed CoT before the answer.
- **Soft Thinking: Unlocking the Reasoning Potential of LLMs in Continuous Concept Space** [[arXiv:2505.15778](https://arxiv.org/abs/2505.15778)] — Replaces discrete sampling with probability-weighted concept tokens, letting the model reason over superpositions in latent concept space.
- **LightThinker: Thinking Step-by-Step Compression** [[arXiv:2502.15589](https://arxiv.org/abs/2502.15589)] — Dynamically compresses verbose reasoning into compact gist tokens during generation.
- **Token Assorted: Mixing Latent and Text Tokens for Improved Language Model Reasoning** [[arXiv:2502.03275](https://arxiv.org/abs/2502.03275)] — Replaces parts of CoT with VQ-VAE latent tokens; mixes discrete latent + text tokens for shorter reasoning traces.
- **Think Silently, Think Fast: Dynamic Latent Compression of LLM Reasoning Chains** [[arXiv:2505.16552](https://arxiv.org/abs/2505.16552)] — Learns when and how to compress reasoning into a few latent slots on the fly.
- **Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought** [[arXiv:2505.12514](https://arxiv.org/abs/2505.12514)] — Proves continuous thought vectors can encode superposed search frontiers, performing BFS in latent space.
- **Continuous Chain of Thought Enables Parallel Exploration and Reasoning** [[arXiv:2505.23648](https://arxiv.org/abs/2505.23648)] — Shows continuous thought tokens implicitly encode multiple reasoning paths, enabling parallel exploration.
- **Latent Thinking Optimization: Your Latent Reasoning LM Secretly Encodes Reward Signals in its Latent Thoughts** [[arXiv:2509.26314](https://arxiv.org/abs/2509.26314)] — ICLR'26 paper: optimizes latent thoughts using reward signals already encoded in them.
- **ImgCoT: Compressing Long Chain of Thought into Compact Visual Tokens for Efficient Reasoning** [[arXiv:2601.22730](https://arxiv.org/abs/2601.22730)] — Stores long textual CoT as compact visual latent tokens for memory-efficient reasoning.

---

## 2. Implicit CoT, Thinking / Pause / Filler Tokens

- **Implicit Chain of Thought Reasoning via Knowledge Distillation** [[arXiv:2311.01460](https://arxiv.org/abs/2311.01460)] — Distills explicit CoT into the model's *vertical* (hidden-state) computation so it answers without emitting reasoning text.
- **From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step** [[arXiv:2405.14838](https://arxiv.org/abs/2405.14838)] — Gradually removes CoT tokens during training so the model internalises the entire chain into hidden states.
- **Think Before You Speak: Training Language Models with Pause Tokens** [[arXiv:2310.02226](https://arxiv.org/abs/2310.02226)] — Learnable pause tokens give the transformer extra latent forward passes before producing output.
- **Let's Think Dot by Dot: Hidden Computation in Transformer Language Models** [[arXiv:2404.15758](https://arxiv.org/abs/2404.15758)] — Filler/dot tokens act as hidden computation buffers, expanding latent reasoning capacity.
- **Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking** [[arXiv:2403.09629](https://arxiv.org/abs/2403.09629)] — Trains the LM to generate internal rationales per token via REINFORCE, embedded between visible tokens.
- **Fast Quiet-STaR: Thinking Without Thought Tokens** [[arXiv:2505.17746](https://arxiv.org/abs/2505.17746)] — Removes explicit thought tokens and instead shifts the reasoning entirely into latent compute.
- **Guiding Language Model Reasoning with Planning Tokens** [[arXiv:2310.05707](https://arxiv.org/abs/2310.05707)] — Learned planning tokens act as latent control codes structuring the reasoning trajectory.
- **Disentangling Memory and Reasoning Ability in LLMs** [[arXiv:2411.13504](https://arxiv.org/abs/2411.13504)] — Explicit memory/reasoning latent control tokens route the hidden-state computation across steps.
- **Distilling System 2 into System 1** [[arXiv:2407.06023](https://arxiv.org/abs/2407.06023)] — Distills slow explicit CoT (System 2) into a fast forward-pass (System 1) inside the same model's hidden states.
- **System-1.x: Learning to Balance Fast and Slow Planning with Language Models** [[arXiv:2407.14414](https://arxiv.org/abs/2407.14414)] — Trains a controller to route between explicit (slow) and implicit-latent (fast) reasoning modes.

---

## 3. Looped & Recurrent-Depth Transformers

- **Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach (Huginn)** [[arXiv:2502.05171](https://arxiv.org/abs/2502.05171)] — Geiping et al.: a 3.5B recurrent-depth transformer iterates a latent block at test time, scaling compute without emitting tokens.
- **Looped Transformers are Better at Learning Learning Algorithms** [[arXiv:2311.12424](https://arxiv.org/abs/2311.12424)] — Loop-unrolled transformers learn iterative algorithms (latent reasoning) more efficiently than feed-forward depth.
- **Looped Transformers for Length Generalization** [[arXiv:2409.15647](https://arxiv.org/abs/2409.15647)] — Loop unrolling enables length generalization on algorithmic tasks by iterating latent computation.
- **Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer** [[arXiv:2507.02199](https://arxiv.org/abs/2507.02199)] — Probes Huginn-style models to test whether they truly perform latent CoT.
- **AlgoFormer: An Efficient Transformer Framework with Algorithmic Structures** [[arXiv:2402.13572](https://arxiv.org/abs/2402.13572)] — Loop sub-blocks implement algorithm-style latent reasoning inside one transformer.
- **CoTFormer: A Chain-of-Thought Driven Architecture with Budget-Adaptive Computation Cost** [[arXiv:2310.10845](https://arxiv.org/abs/2310.10845)] — Interleaves intermediate hidden states as implicit CoT-like recurrence.
- **Universal Transformers** [[arXiv:1807.03819](https://arxiv.org/abs/1807.03819)] — Foundational depth-recurrent transformer with adaptive computation time.

---

## 4. Discrete Latent / Concept-Level Language Models

- **Large Concept Models: Language Modeling in a Sentence Representation Space** [[arXiv:2412.08821](https://arxiv.org/abs/2412.08821)] — Meta LCM: autoregressively predicts next *SONAR sentence embedding* rather than next token, decoupling reasoning from surface form.
- **Next Concept Prediction in Discrete Latent Space Leads to Stronger Language Models** [[arXiv:2602.08984](https://arxiv.org/abs/2602.08984)] — ConceptLM: quantises hidden states with VQ to form a concept vocabulary; jointly trains NCP + NTP, implicitly planning before generation.
- **Dynamic Large Concept Models: Latent Reasoning in an Adaptive Semantic Space** [[arXiv:2512.24617](https://arxiv.org/abs/2512.24617)] — LCM with adaptive concept granularity, scaling latent reasoning to longer horizons.
- **Byte Latent Transformer: Patches Scale Better Than Tokens (BLT)** [[arXiv:2412.09871](https://arxiv.org/abs/2412.09871)] — Meta replaces tokenization with dynamically entropy-grouped byte patches reasoned over by a latent transformer.
- **CoCoMix: Continuous Concept Mixing for Language Model Pre-training** [[arXiv:2502.08524](https://arxiv.org/abs/2502.08524)] — Interleaves predicted SAE-extracted concepts with tokens, supervising the latent reasoning substrate during pretraining.
- **SONAR: Sentence-Level Multimodal and Language-Agnostic Representations** [[arXiv:2308.11466](https://arxiv.org/abs/2308.11466)] — The fixed-size multilingual sentence embedding space underlying LCM-style concept reasoning.
- **Hierarchical Autoregressive Transformers for Concept-Level Language Modeling** [[arXiv:2406.08850](https://arxiv.org/abs/2406.08850)] — Encoder–LM–decoder hierarchy predicting sentence/segment embeddings; precursor to LCM.
- **PLANNER: Concept-Level Latent Planning for Long Text Generation** [[arXiv:2306.02531](https://arxiv.org/abs/2306.02531)] — Latent-variable model plans paragraph-level concept embeddings before decoding tokens.
- **MegaByte: Predicting Million-byte Sequences with Multiscale Transformers** [[arXiv:2305.07185](https://arxiv.org/abs/2305.07185)] — Hierarchical patch-of-bytes model, conceptual precursor to byte-level latent reasoning.
- **SpaceByte: Towards Deleting Tokenization from Large Language Modeling** [[arXiv:2404.14408](https://arxiv.org/abs/2404.14408)] — Byte-level LM with dynamic patching framed as tokenizer-free reasoning substrate.
- **MambaByte: Token-free Selective State Space Model** [[arXiv:2401.13660](https://arxiv.org/abs/2401.13660)] — Byte-level Mamba positioned as substrate for tokenizer-free latent reasoning.
- **Codebook Features: Sparse and Discrete Interpretability for Neural Networks** [[arXiv:2310.17230](https://arxiv.org/abs/2310.17230)] — Replaces dense activations with VQ codebook entries — an interpretable discrete reasoning substrate.

---

## 5. Discrete-Diffusion & Masked LM Reasoning

- **LLaDA: Large Language Diffusion with Masking** [[arXiv:2502.09992](https://arxiv.org/abs/2502.09992)] — 8B masked discrete-diffusion LM whose iterative unmasking is reinterpreted as latent CoT.
- **Reinforcing the Diffusion Chain of Lateral Thought with Diffusion Language Models (DCoLT)** [[arXiv:2505.10446](https://arxiv.org/abs/2505.10446)] — Treats each reverse-diffusion step as a latent "thinking" action and trains via outcome RL.
- **SEDD: Score Entropy Discrete Diffusion** [[arXiv:2310.16834](https://arxiv.org/abs/2310.16834)] — Discrete-diffusion LM whose denoising sequence doubles as latent reasoning.
- **MDLM: Simple and Effective Masked Diffusion Language Models** [[arXiv:2406.07524](https://arxiv.org/abs/2406.07524)] — Cleaner masked-diffusion formulation reused as the reasoning substrate in many discrete-latent CoT works.
- **DiffuLLaMA: Scaling Diffusion Language Models via Adaptation from Autoregressive Models** [[arXiv:2410.17891](https://arxiv.org/abs/2410.17891)] — Adapts LLaMA into a discrete-diffusion LM enabling parallel latent reasoning.
- **Diffusion-of-Thought (DoT)** [[arXiv:2402.07754](https://arxiv.org/abs/2402.07754)] — CoT via discrete diffusion over reasoning tokens, allowing self-correcting multi-pass inference.

---

## 6. Multimodal Latent CoT (Reasoning in Visual Latents)

- **Mirage / Machine Mental Imagery: Empower Multimodal Reasoning with Latent Visual Tokens** [[arXiv:2506.17218](https://arxiv.org/abs/2506.17218)] — VLM augments decoding with latent visual tokens — when it "thinks visually", hidden states become next tokens, no pixels emitted.
- **Heima: Hidden Multimodal Reasoning via Latent Chain-of-Thought** [[arXiv:2501.19201](https://arxiv.org/abs/2501.19201)] — Compresses verbose multimodal CoT into a few latent "thinking tokens" for efficient reasoning.
- **Multimodal Chain of Continuous Thought for Latent-Space Reasoning in VLMs** [[arXiv:2508.12587](https://arxiv.org/abs/2508.12587)] — Extends Coconut's continuous-thought paradigm to vision-language models.
- **Latent Visual Reasoning (LVR)** [[arXiv:2509.24251](https://arxiv.org/abs/2509.24251)] — Autoregressive reasoning directly in visual embedding space — LM emits latent states that reconstruct key visual tokens.
- **Interleaved Latent Visual Reasoning with Selective Perceptual Modeling** [[arXiv:2512.05665](https://arxiv.org/abs/2512.05665)] — Interleaves textual generation with latent visual representations acting as evolving cues.
- **Reasoning in the Dark: Interleaved Vision-Text Reasoning in Latent Space (IVT-LR)** [[arXiv:2510.12603](https://arxiv.org/abs/2510.12603)] — Injects both visual and textual latents into the reasoning chain.
- **LatentUM: Unleashing the Potential of Interleaved Cross-Modal Reasoning via a Latent-Space Unified Model** [[arXiv:2604.02097](https://arxiv.org/abs/2604.02097)] — Unified latent-space model for interleaved cross-modal reasoning.
- **MILR: Improving Multimodal Image Generation via Test-Time Latent Reasoning** [[arXiv:2509.22761](https://arxiv.org/abs/2509.22761)] — Test-time search over latent vectors of discrete image+text tokens for better generation.
- **Multimodal Reasoning via Latent Refocusing (LaRe)** [[arXiv:2511.02360](https://arxiv.org/abs/2511.02360)] — Combines visual refocusing with rich latent representations for iterative reasoning in latent space.
- **Reasoning Within the Mind: Dynamic Multimodal Interleaving in Latent Space (DMLR)** [[arXiv:2512.12623](https://arxiv.org/abs/2512.12623)] — Confidence-guided latent policy gradient + dynamic visual injection into latent think tokens.
- **Latent Sketchpad: Sketching Visual Thoughts to Elicit Multimodal Reasoning in MLLMs** [[arXiv:2510.24514](https://arxiv.org/abs/2510.24514)] — Generates continuous visual latents within the reasoning trajectory — never decoded to images.

---

## 7. Visual Chain-of-Thought (Classics & Datasets)

- **Multimodal Chain-of-Thought Reasoning in Language Models (MM-CoT)** [[arXiv:2302.00923](https://arxiv.org/abs/2302.00923)] — Seminal two-stage rationale-then-answer framework fusing vision and language for CoT on ScienceQA.
- **Visual CoT: A Comprehensive Dataset and Benchmark for Chain-of-Thought Reasoning** [[arXiv:2403.16999](https://arxiv.org/abs/2403.16999)] — 438k-sample dataset with bounding-box rationales for region-grounded multimodal CoT.
- **Compositional Chain-of-Thought Prompting for Large Multimodal Models (CCoT)** [[arXiv:2311.17076](https://arxiv.org/abs/2311.17076)] — Prompts MLLMs to first generate scene graphs as compositional reasoning steps.
- **DDCoT: Duty-Distinct Chain-of-Thought for Multimodal Reasoning** [[arXiv:2310.16436](https://arxiv.org/abs/2310.16436)] — Splits reasoning into critical-thinking + acting steps for LM and VLM modules.
- **T-SciQ: Teaching Multimodal Chain-of-Thought via LLM Signals** [[arXiv:2305.03453](https://arxiv.org/abs/2305.03453)] — Distills GPT-generated planning + CoT signals into smaller models for scientific multimodal QA.

---

## 8. Thinking *with* Images — Sketchpad, Whiteboard, Zoom

- **Visual Sketchpad: Sketching as a Visual Chain of Thought for Multimodal LMs** [[arXiv:2406.09403](https://arxiv.org/abs/2406.09403)] — Equips GPT-4V with drawing tools so it sketches intermediate visual artifacts while reasoning.
- **Interactive Sketchpad: A Multimodal Tutoring System** [[arXiv:2503.16434](https://arxiv.org/abs/2503.16434)] — Adds a human-in-the-loop layer to visual sketching for collaborative problem solving.
- **Whiteboard-of-Thought: Thinking Step-by-Step Across Modalities** [[arXiv:2406.14562](https://arxiv.org/abs/2406.14562)] — LLM draws Matplotlib/Turtle whiteboards as visual scratchpads to solve visuospatial problems.
- **Multimodal Visualization-of-Thought (MVoT)** [[arXiv:2501.07542](https://arxiv.org/abs/2501.07542)] — Trains a unified MLLM to natively generate interleaved image-text visualizations of its reasoning.
- **V\*: Guided Visual Search as a Core Mechanism in MLLMs (SEAL/VStar)** [[arXiv:2312.14135](https://arxiv.org/abs/2312.14135)] — Iterative visual-search loop letting the VLM zoom into salient regions during reasoning.
- **CogCoM: Chain of Manipulations for Vision-Language Models** [[arXiv:2402.04236](https://arxiv.org/abs/2402.04236)] — VLM performs explicit manipulation chains (grounding, zoom, OCR) on the image as reasoning steps.
- **ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts** [[arXiv:2312.00784](https://arxiv.org/abs/2312.00784)] — Overlaid arrows/circles act as intermediate visual cues for reasoning.
- **Image-of-Thought (IoT) Prompting** [[arXiv:2405.13872](https://arxiv.org/abs/2405.13872)] — Decomposes images into stepwise visual rationales (crop, highlight, mask) for zero-shot MLLM reasoning.
- **Chain-of-Spot: Interactive Reasoning Improves Large Vision-Language Models** [[arXiv:2403.12966](https://arxiv.org/abs/2403.12966)] — Iteratively identifies and re-attends to regions-of-interest as a visual CoT.
- **Scaffolding Coordinates to Promote Vision-Language Coordination in MLLMs** [[arXiv:2402.12058](https://arxiv.org/abs/2402.12058)] — Overlays coordinate scaffolds on images for grounded intermediate spatial reasoning.
- **Zoom Eye: Controlling the Field of View of Large Vision-Language Models** [[arXiv:2411.16044](https://arxiv.org/abs/2411.16044)] — Tree-search-based zooming gives the VLM explicit intermediate visual states.
- **DualFocus: Integrating Macro and Micro Perspectives in MLLMs** [[arXiv:2402.14767](https://arxiv.org/abs/2402.14767)] — Alternates global + local crops as a coarse-to-fine reasoning chain.
- **DC²: Divide, Conquer and Combine for High-Resolution VLM Reasoning** [[arXiv:2408.15556](https://arxiv.org/abs/2408.15556)] — Recursively crops the image as intermediate reasoning steps for high-res scenes.
- **v¹: Learning to Point Visual Tokens for Multimodal Mathematical Grounded Reasoning** [[arXiv:2505.18842](https://arxiv.org/abs/2505.18842)] — Lets the VLM "point" at visual tokens as an intermediate grounded reasoning state.
- **ReFocus: Visual Editing as a Chain of Thought for Structured Image Understanding** [[arXiv:2501.05452](https://arxiv.org/abs/2501.05452)] — Edits the image (mask, highlight) as the unit of each CoT step.

---

## 9. Long-CoT / o1-style Multimodal Reasoners

- **LLaVA-CoT: Let Vision Language Models Reason Step-by-Step (LLaVA-o1)** [[arXiv:2411.10440](https://arxiv.org/abs/2411.10440)] — Structured 4-stage (summary / caption / reasoning / conclusion) generation with stage-level beam search.
- **Mulberry: Empowering MLLM with o1-like Reasoning and Reflection via Collective MCTS** [[arXiv:2412.18319](https://arxiv.org/abs/2412.18319)] — Builds long-CoT MLLM training data via collective Monte Carlo Tree Search.
- **Insight-V: Exploring Long-Chain Visual Reasoning with MLLMs** [[arXiv:2411.14432](https://arxiv.org/abs/2411.14432)] — Two-agent (reasoner + summarizer) pipeline plus iterative DPO for long visual CoT.
- **Marco-o1: Towards Open Reasoning Models for Open-Ended Solutions** [[arXiv:2411.14405](https://arxiv.org/abs/2411.14405)] — MCTS + reflection extended to open-ended (incl. multimodal) reasoning.
- **Virgo: A Preliminary Exploration of Reproducing o1-like MLLM** [[arXiv:2501.01904](https://arxiv.org/abs/2501.01904)] — Distills textual long-CoT into a VLM; demonstrates text CoT transfers to visual reasoning.
- **LlamaV-o1: Rethinking Step-by-step Visual Reasoning in LLMs** [[arXiv:2501.06186](https://arxiv.org/abs/2501.06186)] — Curriculum + beam search produces interpretable stepwise visual reasoning.
- **URSA: Understanding and Verifying CoT Reasoning in Multimodal Mathematics** [[arXiv:2501.04686](https://arxiv.org/abs/2501.04686)] — Process-reward-modeled multimodal math reasoner with verifier-guided search.
- **MAmmoTH-VL: Eliciting Multimodal Reasoning with Instruction Tuning at Scale** [[arXiv:2412.05237](https://arxiv.org/abs/2412.05237)] — 12M rationale-rich multimodal instructions for CoT-style training.
- **Kimi k1.5: Scaling RL with LLMs** [[arXiv:2501.12599](https://arxiv.org/abs/2501.12599)] — Scales RL with long-context and visual inputs for multimodal o1-class reasoning.
- **Kimi-VL Technical Report** [[arXiv:2504.07491](https://arxiv.org/abs/2504.07491)] — MoE VLM activating only 2.8B parameters, with long-context multimodal reasoning and agentic capabilities.
- **RedStar: Does Scaling Long-CoT Data Unlock Better Slow-Reasoning Systems?** [[arXiv:2501.11284](https://arxiv.org/abs/2501.11284)] — Scales long-CoT SFT to multimodal regime; studies data efficiency.
- **R1-Onevision: Advancing Generalized Multimodal Reasoning through Cross-Modal Formalization** [[arXiv:2503.10615](https://arxiv.org/abs/2503.10615)] — Cross-modal formalization to bridge vision and language for general reasoning.

---

## 10. RL & Process Rewards for Multimodal Reasoning

- **Vision-R1: Incentivizing Reasoning Capability in MLLMs via RL** [[arXiv:2503.06749](https://arxiv.org/abs/2503.06749)] — Cold-start SFT + RL pipeline that produces long visual reasoning chains; 78.2% on MathVista (32B).
- **VisualThinker-R1-Zero / R1-Zero's "Aha Moment" in Visual Reasoning** [[arXiv:2503.05132](https://arxiv.org/abs/2503.05132)] — First R1-zero replication on a base VLM (Qwen2-VL-2B), showing emergent visual reflection.
- **MM-Eureka: Exploring Visual Aha Moments with Rule-based Large-scale RL** [[arXiv:2503.07365](https://arxiv.org/abs/2503.07365)] — Reproduces R1-style aha behaviors in InternVL backbones on multimodal math.
- **VLM-R1: A Stable and Generalizable R1-style Large Vision-Language Model** [[arXiv:2504.07615](https://arxiv.org/abs/2504.07615)] — Open recipe applying R1 RL to VLMs for grounding-aware reasoning.
- **OpenVLThinker: Complex Vision-Language Reasoning via Iterative SFT-RL Cycles** [[arXiv:2503.17352](https://arxiv.org/abs/2503.17352)] — Iterates SFT and RL stages to scale long visual CoT.
- **Perception-R1: Advancing Multimodal Reasoning Capabilities of MLLMs via Visual Perception Reward** [[arXiv:2506.07218](https://arxiv.org/abs/2506.07218)] — Adds explicit perception reward to R1-style RL.
- **Improved Visual-Spatial Reasoning via R1-Zero-Like Training** [[arXiv:2504.00883](https://arxiv.org/abs/2504.00883)] — Targets visuo-spatial benchmarks with rule-based RL.
- **EchoInk-R1: Audio-Visual Reasoning in Multimodal LLMs via RL** [[arXiv:2505.04623](https://arxiv.org/abs/2505.04623)] — Extends R1-style RL to audio-visual reasoning.
- **MM-RLHF: The Next Step Forward in Multimodal LLM Alignment** [[arXiv:2502.10391](https://arxiv.org/abs/2502.10391)] — Large-scale multimodal preference dataset + reward model for reasoning alignment.
- **RLHF-V: Trustworthy MLLMs via Behavior Alignment from Fine-grained Correctional Human Feedback** [[arXiv:2312.00849](https://arxiv.org/abs/2312.00849)] — Token-level RLHF on hallucination corrections.

---

## 11. Image / Video Generation as Reasoning

- **Think in Strokes, Not Pixels: Process-Driven Image Generation via Interleaved Reasoning** [[arXiv:2604.04746](https://arxiv.org/abs/2604.04746)] — Decomposes synthesis into iterative `plan → draft → reflect → refine` cycles; lifts BAGEL-7B by +5% GenEval / +6% WISE.
- **GoT: Unleashing Reasoning Capability of MLLM for Visual Generation and Editing** [[arXiv:2503.10639](https://arxiv.org/abs/2503.10639)] — MLLM first generates a semantic+spatial reasoning chain (9M-sample dataset), then synthesizes via a diffusion model with a Semantic-Spatial Guidance module.
- **T2I-R1: Reinforcing Image Generation with Collaborative Semantic-level and Token-level CoT** [[arXiv:2505.00703](https://arxiv.org/abs/2505.00703)] — Bi-level CoT (semantic + token) trained with BiCoT-GRPO; +13% T2I-CompBench, +19% WISE on Janus-Pro.
- **ReasonGen-R1: CoT for Autoregressive Image Generation through SFT and RL** [[arXiv:2505.24875](https://arxiv.org/abs/2505.24875)] — SFT on written rationales + GRPO RL for AR image generators with explicit "thinking".
- **ImageGen-CoT: Enhancing T2I In-Context Learning with Chain-of-Thought** [[arXiv:2503.19312](https://arxiv.org/abs/2503.19312)] — Inserts textual CoT steps before each generated image to improve in-context T2I.
- **Can We Generate Images with CoT? (PARM / PARM++)** [[arXiv:2501.13926](https://arxiv.org/abs/2501.13926)] — Process reward models score each generation step; +24% GenEval on Show-o, surpassing SD3 by +15%.
- **Autoregressive Image Generation Guided by Chains of Thought** [[arXiv:2502.16965](https://arxiv.org/abs/2502.16965)] — Aligns AR image generation with explicit CoT planning at every token block.
- **Improving Chain-of-Thought Efficiency for Autoregressive Image Generation (ShortCoTI)** [[arXiv:2510.05593](https://arxiv.org/abs/2510.05593)] — +54% reasoning efficiency on T2I-CompBench while improving accuracy.
- **Chain-of-Image Generation: Toward Monitorable and Controllable Image Generation** [[arXiv:2512.08645](https://arxiv.org/abs/2512.08645)] — LLM decomposes a complex prompt into stepwise instructions executed by a progressive image generator.
- **Visual-CoG: Stage-Aware Reinforcement Learning with Chain-of-Guidance** [[arXiv:2508.18032](https://arxiv.org/abs/2508.18032)] — Stage-aware RL for image generation that mirrors a CoG (chain-of-guidance) trajectory.
- **CoT-lized Diffusion: Let's Reinforce T2I Generation Step-by-Step** [[arXiv:2507.04451](https://arxiv.org/abs/2507.04451)] — Each denoising step is treated as a reasoning step refined by CoT.
- **Interleaving Reasoning for Better Text-to-Image Generation** [[arXiv:2509.06945](https://arxiv.org/abs/2509.06945)] — Interleaves textual reasoning and partial image generation across rounds.
- **RPG: Mastering Text-to-Image via Multimodal Recaptioning, Planning, and Generation** [[arXiv:2401.11708](https://arxiv.org/abs/2401.11708)] — Plans region-wise sub-prompts as a reasoning chain before diffusion synthesis.
- **T2I-Eval-R1: RL-Driven Reasoning for Interpretable Text-to-Image Evaluation** [[arXiv:2505.17897](https://arxiv.org/abs/2505.17897)] — RL-trained reasoning evaluator for T2I quality.
- **T2I-ReasonBench: Benchmarking Reasoning-Informed Text-to-Image Generation** [[arXiv:2508.17472](https://arxiv.org/abs/2508.17472)] — Benchmark targeting generation tasks that *require* reasoning, not just rendering.
- **What, Whether and How? Unveiling Process Reward Models for Thinking with Images Reasoning** [[arXiv:2602.08346](https://arxiv.org/abs/2602.08346)] — Process reward model analysis for image-thinking reasoning.

---

## 12. Unified Multimodal Models (Shared Latent for Understand + Generate)

> Foundational models that put understanding + generation into one latent — many recent reasoning works (Mirage, GoT, Think in Strokes, T2I-R1, etc.) are built on these.

- **Chameleon: Mixed-Modal Early-Fusion Foundation Models** [[arXiv:2405.09818](https://arxiv.org/abs/2405.09818)] — Meta's early-fusion token-based model mixing image and text tokens in one AR stream.
- **Emu3: Next-Token Prediction is All You Need** [[arXiv:2409.18869](https://arxiv.org/abs/2409.18869)] — Unified next-token model over discretized vision/video/text; pure AR suffices for understanding and generation.
- **Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation** [[arXiv:2410.13848](https://arxiv.org/abs/2410.13848)] — DeepSeek's framework that decouples visual encoders for understanding vs. generation while sharing one AR backbone.
- **JanusFlow: Harmonizing Autoregression and Rectified Flow** [[arXiv:2411.07975](https://arxiv.org/abs/2411.07975)] — Integrates rectified-flow generation with AR understanding inside one minimalist transformer.
- **Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling** [[arXiv:2501.17811](https://arxiv.org/abs/2501.17811)] — Scaled-up Janus pushing unified gen/understand to new SOTA; backbone for T2I-R1.
- **Show-o: One Single Transformer to Unify Multimodal Understanding and Generation** [[arXiv:2408.12528](https://arxiv.org/abs/2408.12528)] — Unified AR (text) + discrete-diffusion (image) sharing one latent space.
- **Show-o2: Improved Native Unified Multimodal Models** [[arXiv:2506.15564](https://arxiv.org/abs/2506.15564)] — Strengthens shared-latent reasoning across text, image, and video.
- **Liquid: Language Models are Scalable and Unified Multi-modal Generators** [[arXiv:2412.04332](https://arxiv.org/abs/2412.04332)] — Native LLM handling of VQ image tokens; reveals scaling laws for unified visual reasoning.
- **VILA-U: Unified Foundation Model Integrating Visual Understanding and Generation** [[arXiv:2409.04429](https://arxiv.org/abs/2409.04429)] — Single unified vision tokenizer aligned with text.
- **AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling** [[arXiv:2402.12226](https://arxiv.org/abs/2402.12226)] — Treats image, speech, music, text as discrete tokens within one AR LLM.
- **Unified-IO 2: Scaling Autoregressive Multimodal Models with Vision, Language, Audio, and Action** [[arXiv:2312.17172](https://arxiv.org/abs/2312.17172)] — Single AR encoder-decoder over discrete tokens for image, text, audio, embodied action.
- **Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model** [[arXiv:2408.11039](https://arxiv.org/abs/2408.11039)] — One transformer trained jointly with AR (text) + diffusion (continuous image latents).
- **MonoFormer: One Transformer for Both Diffusion and Autoregression** [[arXiv:2409.16280](https://arxiv.org/abs/2409.16280)] — A shared transformer serves AR text and diffusion image generation simultaneously.
- **BAGEL: Emerging Properties in Unified Multimodal Pretraining** [[arXiv:2505.14683](https://arxiv.org/abs/2505.14683)] — ByteDance MoE unified model showing emergent multimodal reasoning (free-form editing, future-frame prediction, 3D, navigation).
- **MetaMorph: Multimodal Understanding and Generation via Instruction Tuning** [[arXiv:2412.14164](https://arxiv.org/abs/2412.14164)] — Instruction-tunes an LLM to predict continuous visual tokens.
- **ANOLE: Open, Autoregressive, Native LMM for Interleaved Image-Text Generation** [[arXiv:2407.06135](https://arxiv.org/abs/2407.06135)] — Open-source interleaved generator on Chameleon backbone.
- **MIO: A Foundation Model on Multimodal Tokens** [[arXiv:2409.17692](https://arxiv.org/abs/2409.17692)] — Any-to-any AR model over speech, image, video, and text tokens.
- **MMaDA: Multimodal Large Diffusion Language Models** [[arXiv:2505.15809](https://arxiv.org/abs/2505.15809)] — Discrete-diffusion backbone for unified multimodal generation and reasoning.
- **Unified Autoregressive Visual Generation and Understanding with Continuous Tokens** [[arXiv:2503.13436](https://arxiv.org/abs/2503.13436)] — Continuous-latent unified model bridging Chameleon and Transfusion.
- **Lumina-mGPT: Illuminate Flexible Photorealistic Text-to-Image Generation** [[arXiv:2408.02657](https://arxiv.org/abs/2408.02657)] — Decoder-only LM reasoning over discrete image tokens.

---

## 13. World Models, JEPA & Latent-Action Reasoning

### 13.1 JEPA family (predictive latent reasoning)

- **I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture** [[arXiv:2301.08243](https://arxiv.org/abs/2301.08243)] — Foundational JEPA: predicts representations of masked regions in latent space.
- **V-JEPA: Revisiting Feature Prediction for Learning Visual Representations from Video** [[arXiv:2404.08471](https://arxiv.org/abs/2404.08471)] — Video JEPA learning predictive embeddings without pixel-level reconstruction.
- **V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning** [[arXiv:2506.09985](https://arxiv.org/abs/2506.09985)] — 1M+ hours of video; post-trained into an action-conditioned latent world model for zero-shot robotic planning.
- **A-JEPA: Joint-Embedding Predictive Architecture Can Listen** [[arXiv:2311.15830](https://arxiv.org/abs/2311.15830)] — Extends JEPA to audio.
- **MC-JEPA: Joint-Embedding Predictive Architecture for Motion and Content** [[arXiv:2307.12698](https://arxiv.org/abs/2307.12698)] — Jointly predicts motion and content in latent space.
- **DINO-WM: World Models on Pre-trained Visual Features Enable Zero-Shot Planning** [[arXiv:2411.04983](https://arxiv.org/abs/2411.04983)] — Latent dynamics on frozen DINO features → planning via latent prediction.

### 13.2 World models with latent planning

- **DreamerV3: Mastering Diverse Domains through World Models** [[arXiv:2301.04104](https://arxiv.org/abs/2301.04104)] — Latent recurrent world model planning via imagined latent rollouts across 150+ tasks.
- **Genie: Generative Interactive Environments** [[arXiv:2402.15391](https://arxiv.org/abs/2402.15391)] — DeepMind's latent-action world model turning videos into playable latent environments.
- **LWM: World Model on Million-Length Video and Language with Blockwise RingAttention** [[arXiv:2402.08268](https://arxiv.org/abs/2402.08268)] — Joint long-context language + video world model.
- **GAIA-1: A Generative World Model for Autonomous Driving** [[arXiv:2309.17080](https://arxiv.org/abs/2309.17080)] — Latent video world model conditioned on text/actions.
- **iVideoGPT: Interactive VideoGPTs are Scalable World Models** [[arXiv:2405.15223](https://arxiv.org/abs/2405.15223)] — Tokenized interactive world model trained on large video corpora.
- **Pandora: Towards General World Model with Natural Language Actions and Video States** [[arXiv:2406.09455](https://arxiv.org/abs/2406.09455)] — Combines an LLM with video latent dynamics for text-conditioned simulation.
- **Cosmos World Foundation Model Platform** [[arXiv:2501.03575](https://arxiv.org/abs/2501.03575)] — NVIDIA's family of video latent world models for physical-AI reasoning.
- **WorldDreamer: Towards General World Models via Predicting Masked Tokens** [[arXiv:2401.09985](https://arxiv.org/abs/2401.09985)] — Masked-latent-video-token prediction as general world-model objective.
- **Diffusion World Model (DWM)** [[arXiv:2402.03570](https://arxiv.org/abs/2402.03570)] — Replaces recurrent latent dynamics with diffusion for long-horizon latent prediction.

### 13.3 Latent action / VLA / embodied reasoning

- **LAPA: Latent Action Pretraining from Videos** [[arXiv:2410.11758](https://arxiv.org/abs/2410.11758)] — Unsupervised VLA pretraining via VQ-VAE latent actions from action-unlabeled internet videos (CoRL'24 Best Paper).
- **Latent Action Pretraining Through World Modeling** [[arXiv:2509.18428](https://arxiv.org/abs/2509.18428)] — Combines latent action learning with a world-model objective.
- **VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model** [[arXiv:2602.10098](https://arxiv.org/abs/2602.10098)] — Integrates JEPA-style latent world modeling into a VLA stack.
- **GR-1: Large-Scale Video Generative Pre-training for Visual Robot Manipulation** [[arXiv:2312.13139](https://arxiv.org/abs/2312.13139)] — Video generative pretraining yields a latent that transfers to manipulation.
- **GR-2: Generative Video-Language-Action Model with Web-Scale Knowledge** [[arXiv:2410.06158](https://arxiv.org/abs/2410.06158)] — Predicts future video latents and actions jointly.
- **UniPi: Learning Universal Policies via Text-Guided Video Generation** [[arXiv:2302.00111](https://arxiv.org/abs/2302.00111)] — Plans in video-latent space by generating goal-conditioned futures.
- **VPP: Video Prediction Policy** [[arXiv:2412.14803](https://arxiv.org/abs/2412.14803)] — Video diffusion latents as predictive features for robot policy learning.
- **Seer: Predictive Inverse Dynamics Models for VLA** [[arXiv:2412.15109](https://arxiv.org/abs/2412.15109)] — Latent future prediction + inverse dynamics for manipulation.

---

## 14. Audio / Speech Latent Reasoning

- **Audio-CoT: Exploring Chain-of-Thought Reasoning in Large Audio Language Models** [[arXiv:2501.07246](https://arxiv.org/abs/2501.07246)] — First systematic study of CoT in audio LLMs.
- **Audio-Reasoner: Improving Reasoning Capability in Large Audio Language Models** [[arXiv:2503.02318](https://arxiv.org/abs/2503.02318)] — Structured CoT SFT + planning → SOTA audio reasoning.
- **R1-AQA: RL for Audio Question Answering** [[arXiv:2503.11197](https://arxiv.org/abs/2503.11197)] — Applies R1-style RL to audio LLMs for stepwise auditory reasoning.
- **EchoInk-R1: Audio-Visual Reasoning in Multimodal LLMs via RL** [[arXiv:2505.04623](https://arxiv.org/abs/2505.04623)] — R1-style RL on joint audio-visual reasoning.

---

## 15. Theory & Mechanistic Analyses

- **The Expressive Power of Transformers with Chain of Thought** [[arXiv:2310.07923](https://arxiv.org/abs/2310.07923)] — Formal results on how CoT length expands transformer computational class.
- **Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought** [[arXiv:2505.12514](https://arxiv.org/abs/2505.12514)] — Proves continuous thoughts can encode superposed search frontiers (BFS in latent space).
- **Do Large Language Models Latently Perform Multi-Hop Reasoning?** [[arXiv:2402.16837](https://arxiv.org/abs/2402.16837)] — Probing study showing multi-hop reasoning occurs partly in hidden states without explicit CoT.
- **Iteration Head: A Mechanistic Study of Chain-of-Thought** [[arXiv:2406.02128](https://arxiv.org/abs/2406.02128)] — Mechanistic analysis of how transformers implement iterative latent computation underlying CoT.
- **Do Latent Tokens Think? A Causal and Adversarial Analysis of Chain-of-Continuous-Thought** [[arXiv:2512.21711](https://arxiv.org/abs/2512.21711)] — Causal/adversarial probe of what latent thought tokens actually compute.

---

## How to contribute

PRs adding new papers are welcome. Please keep the format consistent:

```
- **<Title>** [[arXiv:YYMM.NNNNN](https://arxiv.org/abs/YYMM.NNNNN)] — <one-sentence summary>
```

Suggested guidelines:
- Stick to papers where *reasoning happens in some non-text latent space* (continuous hidden states, discrete concept codes, latent visual tokens, latent actions, world-model rollouts, …) — not pure text CoT prompting.
- Bias toward 2024 onward; older work only if it's a clear precursor cited by recent latent-reasoning papers.
- Place the paper in the section that best matches its *method*, not its application; if in doubt, file under §6 (Multimodal Latent CoT) or §11 (Generation-as-Reasoning).

## Related Awesome Lists

- [LatentCoT-Horizon](https://github.com/multimodal-art-projection/LatentCoT-Horizon) — Curated latent-CoT papers.
- [Awesome_Think_With_Images](https://github.com/zhaochen0110/Awesome_Think_With_Images) — Companion list for the *Thinking with Images* survey.
- [Awesome-Latent-Space](https://github.com/YU-deep/Awesome-Latent-Space) — Broader latent-space paper list.
