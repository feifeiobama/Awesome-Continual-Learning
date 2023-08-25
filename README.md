# Awesome Continual Learning
This repository contains a curated list of continual learning papers and [BibTeX entries](./CL.bib) (mostly until 2022).

* [Survey](#Survey)
* [Replay-based](#Replay-based)
  * [Memory replay](#Memory-replay)
  * [Generative replay](#Generative-replay)
* [Regularization-based](#Regularization-based)
  * [Bayesian-based](#Bayesian-based)
  * [Subspace-based](#Subspace-based)
  * [Distillation-based](#Distillation-based)
* [Architecture-based](#Architecture-based)
  * [Expansion](#Expansion)
  * [Mask](#Mask)
  * [Decompose](#Decompose)
* [Application](#Application)
  * [Object detection](#Object-detection)
  * [Semantic segmentation](#Semantic-segmentation)
  * [Image generation](#Image-generation)
  * [Person re-identification](#Person-re-identification)
  * [Vision-language learning](#Vision-language-learning)
  * [Reinforcement learning](#Reinforcement-learning)
  * [Others](#Others)



## Survey

* (Book 18) Lifelong Machine Learning
* (PhD dissertation 19) Continual Learning with Deep Architectures
* (PhD dissertation 19) Continual Learning in Neural Networks
* (Trends in Cognitive Science 20) Embracing Change: Continual Learning in Deep Neural Networks
* (TPAMI 21) A Continual Learning Survey: Defying Forgetting in Classification Tasks
* (JAIR 22) Towards Continual Reinforcement Learning: A Review and Perspectives
* (Neurocomputing 22) Online Continual Learning in Image Classification: An Empirical Survey
* (arXiv 22) An Introduction to Lifelong Supervised Learning
* (Trends in Neurosciences 23) Continual Task Learning in Natural and Artificial Agents
* (Neural Networks 23) A Wholistic View of Continual Learning with Deep Neural Networks: Forgotten Lessons and the Bridge to Active and Open World Learning
* (arXiv 23) A Comprehensive Survey of Continual Learning: Theory, Method and Application
* (arXiv 23) Deep Class-Incremental Learning: A Survey
* (arXiv 23) A Comprehensive Survey of Forgetting in Deep Learning Beyond Continual Learning



## Replay-based

### Memory replay

* (CVPR 17) iCaRL: Incremental Classifier and Representation Learning
* (NeurIPS 17) Gradient Episodic Memory for Continual Learning
* (ECCV 18) End-to-End Incremental Learning
* (ICLR 19) Learning to Learn without Forgetting by Maximizing Transfer and Minimizing Interference
* (ICLR 19) Efficient Lifelong Learning with A-GEM
* (CVPR 19) Large Scale Incremental Learning
* (CVPR 19) Learning a Unified Classifier Incrementally via Rebalancing
* (NeurIPS 19) Experience Replay for Continual Learning
* (NeurIPS 19) Gradient based Sample Selection for Online Continual Learning
* (NeurIPS 19) Online Continual Learning with Maximal Interfered Retrieval
* (ICMLW 19) On Tiny Episodic Memories in Continual Learning
* (ECCV 20) GDumb: A Simple Approach that Questions Our Progress in Continual Learning
* (NeurIPS 20) Coresets via Bilevel Optimization for Continual Learning and Streaming
* (NeurIPS 20) Dark Experience for General Continual Learning: A Strong, Simple Baseline
* (ICPR 20) Rethinking Experience Replay: A Bag of Tricks for Continual Learning
* (AAAI 21) Using Hindsight to Anchor Past Knowledge in Continual Learning
* (ICLR 21) Dataset Condensation with Gradient Matching
* (CVPR 21) Rainbow Memory: Continual Learning with a Memory of Diverse Samples
* (ICML 21) Grad-Match: Gradient Matching Based Data Subset Selection for Efficient Deep Model Training
* (ICCV 21) Rehearsal Revealed: The Limits and Merits of Revisiting Samples in Continual Learning
* (NeurIPS 21) Gradient-based Editing of Memory Examples for Online Task-free Continual Learning
* (NeurIPS 21) RMM: Reinforced Memory Management for Class-Incremental Learning
* (NeurIPSW 21) Gradient-Matching Coresets for Continual Learning
* (ICLR 22) Online Coreset Selection for Rehearsal-based Continual Learning
* (ICLR 22) New Insights on Reducing Abrupt Representation Change in Online Continual Learning
* (ICLR 22) Memory Replay with Data Compression for Continual Learning
* (CVPR 22) GCR: Gradient Coreset based Replay Buffer Selection for Continual Learning
* (ICML 22) Improving Task-free Continual Learning by Distributionally Robust Memory Evolution
* (ICML 22) Online Continual Learning through Mutual Information Maximization
* (NeurIPS 22) Exploring Example Influence in Continual Learning
* (NeurIPS 22) Retrospective Adversarial Replay for Continual Learning
* (NeurIPS 22) Repeated Augmented Rehearsal: A Simple but Strong Baseline for Online Continual Learning
* (TPAMI 22) Class-Incremental Continual Learning into the eXtended DER-verse
* (ICLR 23) Error Sensitivity Modulation based Experience Replay: Mitigating Abrupt Representation Drift in Continual Learning
* (CVPR 23) Computationally Budgeted Continual Learning: What Does Matter?
* (CVPR 23) Class-Incremental Exemplar Compression for Class-Incremental Learning
* (CVPR 23) PCR: Proxy-based Contrastive Replay for Online Class-Incremental Continual Learning
* (CVPR 23) Rebalancing Batch Normalization for Exemplar-based Class-Incremental Learning
* (CVPR 23) Regularizing Second-Order Influences for Continual Learning
* (ICML 23) BiRT: Bio-inspired Replay in Vision Transformers for Continual Learning
* (ICML 23) DualHSIC: HSIC-Bottleneck and Alignment for Continual Learning
* (IJCV 23) Trust-Region Adaptive Frequency for Online Continual Learning

### Generative replay

* (NeurIPS 17) Continual Learning with Deep Generative Replay
* (NeurIPS 18) Memory Replay GANs: Learning to Generate New Categories without Forgetting
* (NeurIPS 20) GAN Memory with No Forgetting
* (Nature Communications 20) Brain-Inspired Replay for Continual Learning with Artificial Neural Networks
* (Neurocomputing 20) Lifelong Generative Modeling
* (ICCV 21) Always Be Dreaming: A New Approach for Data-Free Class-Incremental Learning
* (CVPR 22) Learning to Imagine: Diversify Memory for Incremental Learning using Unlabeled Data
* (ICLR 23) Incremental Learning of Structured Memory via Closed-Loop Transcription
* (ICML 23) DDGR: Continual Learning with Deep Diffusion-based Generative Replay



## Regularization-based

* (ICML 17) Continual Learning through Synaptic Intelligence
* (ECCV 18) Memory Aware Synapses: Learning What (Not) to Forget
* (ECCV 18) Riemannian Walk for Incremental Learning: Understanding Forgetting and Intransigence
* (NeurIPS 20) Understanding the Role of Training Regimes in Continual Learning
* (CVPR 23) Achieving a Better Stability-Plasticity Trade-off via Auxiliary Networks in Continual Learning

### Bayesian-based

* (PNAS 17) Overcoming Catastrophic Forgetting in Neural Networks
* (NeurIPS 17) Overcoming Catastrophic Forgetting by Incremental Moment Matching
* (NeurIPS 18) Online Structured Laplace Approximations for Overcoming Catastrophic Forgetting
* (ICLR 18) Variational Continual Learning
* (NeurIPS 19) Uncertainty-based Continual Learning with Adaptive Regularization
* (ICLR 20) Uncertainty-guided Continual Learning with Bayesian Neural Networks
* (ICLR 20) Continual Learning with Adaptive Weights (CLAW)
* (NeurIPS 21) Natural Continual Learning: Success Is a Journey, Not (Just) A Destination
* (NeurIPS 21) AFEC: Active Forgetting of Negative Transfer in Continual Learning
* (AISTATS 22) Provable Continual Learning via Sketched Jacobian Approximations

### Subspace-based

* (Nature Machine Intelligence 19) Continual Learning of Context-Dependent Processing in Neural Networks
* (AISTATS 20) Orthogonal Gradient Descent for Continual Learning
* (NeurIPS 20) Continual Learning in Low-rank Orthogonal Subspaces
* (ICLR 21) Gradient Projection Memory for Continual Learning
* (ICLR 21) Linear Mode Connectivity in Multitask and Continual Learning
* (CVPR 21) Training Networks in Null Space of Feature Covariance for Continual Learning
* (CVPR 21) Layerwise Optimization by Gradient Decomposition for Continual Learning
* (NeurIPS 21) Flattening Sharpness for Dynamic Gradient Projection Memory Benefits Continual Learning
* (ICLR 22) TRGP: Trust Region Gradient Projection for Continual Learning
* (ICLR 22) Continual Learning with Recursive Gradient Optimization
* (ICML 22) Continual Learning with Guarantees via Weight Interval Constraints
* (CVPR 22) Towards Better Plasticity-Stability Trade-off in Incremental Learning: A Simple Linear Connector
* (AAAI 23) Continual Learning with Scaled Gradient Projection
* (ICLR 23) Building a Subspace of Policies for Scalable Continual Learning
* (CVPR 23) Adaptive Plasticity Improvement for Continual Learning
* (CVPR 23) Decoupling Learning and Remembering: a Bilevel Memory Framework with Knowledge Projection for Task-Incremental Learning
* (ICML 23) Optimizing Mode Connectivity for Class Incremental Learning

### Distillation-based

* (TPAMI 17) Learning without Forgetting
* (CVPR 19) Learning without Memorizing
* (CVPR 20) Few-Shot Class-Incremental Learning
* (ECCV 20) Topology-Preserving Class-Incremental Learning
* (ICCV 21) Co^2^L: Contrastive Continual Learning
* (TIP 22) CKDF: Cascaded Knowledge Distillation Framework for Robust Incremental Learning
* (TPAMI 23) Variational Data-Free Knowledge Distillation for Continual Learning
* (ICML 23) Prototype-Sample Relation Distillation: Towards Replay-Free Continual Learning



## Architecture-based

* (ICLR 20) Continual Learning with Hypernetworks
* (NeurIPS 21) DualNet: Continual Learning, Fast and Slow
* (ICLR 22) Learning Fast, Learning Slow: A General Continual Learning Method based on Complementary Learning System
* (ICLR 22) Model Zoo: A Growing Brain That Learns Continually
* (ECCV 22) CoSCL: Cooperation of Small Continual Learners is Stronger Than a Big One
* (CVPR 23) Bilateral Memory Consolidation for Continual Learning

### Expansion

* (arXiv 16) Progressive Neural Networks
* (CVPR 17) Expert Gate: Lifelong Learning with a Network of Experts
* (ICML 17) AdaNet: Adaptive Structural Learning of Artificial Neural Networks
* (ICLR 18) Lifelong Learning with Dynamically Expandable Networks
* (ICML 19) Learn to Grow: A Continual Structure Learning Framework for Overcoming Catastrophic Forgetting
* (ICLR 20) A Neural Dirichlet Process Mixture Model for Task-Free Continual Learning
* (ECCV 20) Side-Tuning: A Baseline for Network Adaptation via Additive Side Networks
* (CVPR 21) DER: Dynamically Expandable Representation for Class Incremental Learning
* (CVPR 21) Adaptive Aggregation Networks for Class-Incremental Learning
* (NeurIPS 21) BNS: Building Network Structures Dynamically for Continual Learning
* (CVPR 22) DyTox: Transformers for Continual Learning with Dynamic Token Expansion
* (CVPR 22) Learning to Prompt for Continual Learning
* (ECCV 22) DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning
* (NeurIPS 22) S-Prompts Learning with Pre-trained Transformers: An Occam's Razor for Domain Incremental Learning
* (ICLR 23) Progressive Prompts: Continual Learning for Language Models
* (ICLR 23) Continual Transformers: Redundancy-Free Attention for Online Inference
* (CVPR 23) Task Difficulty Aware Parameter Allocation & Regularization for Lifelong Learning
* (CVPR 23) Dense Network Expansion for Class Incremental Learning
* (CVPR 23) DKT: Diverse Knowledge Transfer Transformer for Class Incremental Learning
* (CVPR 23) CODA-Prompt: COntinual Decomposed Attention-based Prompting for Rehearsal-Free Continual Learning
* (ICML 23) Learning Expressive Priors for Generalization and Uncertainty Estimation in Neural Networks
* (ICML 23) Lifelong Language Pretraining with Distribution-Specialized Experts
* (ICCV 23) A Unified Continual Learning Framework with General Parameter-Efficient Tuning
* (ICCV 23) CLR: Channel-wise Lightweight Reprogramming for Continual Learning
* (ICCV 23) Exemplar-Free Continual Transformer with Convolutions

### Mask

* (CVPR 18) PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning
* (ICML 18) Overcoming Catastrophic Forgetting with Hard Attention to the Task
* (NeurIPSW 19) Continual Learning via Neural Pruning
* (NeurIPS 20) Supermasks in Superposition
* (ICLR 21) Long Live the Lottery: The Existence of Winning Tickets in Lifelong Learning
* (CVPR 21) Continual Learning via Bit-Level Information Preserving
* (CVPR 22) Meta-Attention for ViT-backed Continual Learning
* (ICML 22) Forget-free Continual Learning with Winning Subnetworks
* (ICML 22) NISPA: Neuro-Inspired Stability-Plasticity Adaptation for Continual Learning in Sparse Networks
* (ICML 23) Discrete Key-Value Bottleneck
* (ICML 23) Parameter-Level Soft-Masking for Continual Learning

### Decompose

* (ICLR 20) Scalable and Order-robust Continual Learning with Additive Parameter Decomposition
* (ICLR 20) BatchEnsemble: an Alternative Approach to Efficient Ensemble and Lifelong Learning
* (CVPR 21) Efficient Feature Transformations for Discriminative and Generative Continual Learning
* (NeurIPS 21) Mitigating Forgetting in Online Continual Learning with Neuron Calibration
* (NeurIPS 21) Achieving Forgetting Prevention and Knowledge Transfer in Continual Learning
* (ICLR 22) TRGP: Trust Region Gradient Projection for Continual Learning



## Application

### Object detection

* (ICCV 17) Incremental Learning of Object Detectors without Catastrophic Forgetting
* (WACV 20) Class-incremental Learning via Deep Model Consolidation
* (CVPR 20) Incremental Few-Shot Object Detection
* (CVPR 21) Towards Open World Object Detection
* (ICCV 21) Wanderlust: Online Continual Object Detection in the Real World
* (NeurIPS 21) Bridging Non Co-occurrence with Unlabeled In-the-wild Data for Incremental Object Detection
* (TPAMI 21) Incremental Object Detection via Meta-Learning
* (CVPR 22) Overcoming Catastrophic Forgetting in Incremental Object Detection via Elastic Response Distillation
* (CVPR 22) Continual Object Detection via Prototypical Task Correlation Guided Gating Mechanism
* (CVPR 23) Continual Detection Transformer for Incremental Object Detection
* (ICCV 23) Augmented Box Replay: Overcoming Foreground Shift for Incremental Object Detection

### Semantic segmentation

* (ICCVW 19) Incremental Learning Techniques for Semantic Segmentation
* (CVPR 20) Modeling the Background for Incremental Learning in Semantic Segmentation
* (AAAI 21) A Continual Learning Framework for Uncertainty-Aware Interactive Image Segmentation
* (AAAI 21) Unsupervised Model Adaptation for Continual Semantic Segmentation
* (CVPR 21) Continual Semantic Segmentation via Repulsion-Attraction of Sparse and Disentangled Latent Representations
* (CVPR 21) PLOP: Learning without Forgetting for Continual Semantic Segmentation
* (CVPR 21) Incremental Few-Shot Instance Segmentation
* (ICCV 21) RECALL: Replay-based Continual Learning in Semantic Segmentation
* (NeurIPS 21) SSUL: Semantic Segmentation with Unknown Label for Exemplar-based Class-Incremental Learning
* (WACV 22) Multi-Domain Incremental Learning for Semantic Segmentation
* (CVPR 22) Representation Compensation Networks for Continual Semantic Segmentation
* (CVPR 22) Class Similarity Weighted Knowledge Distillation for Continual Semantic Segmentation
* (CVPR 22) Incremental Learning in Semantic Segmentation from Image Labels
* (CVPR 22) Learning Multiple Dense Prediction Tasks from Partially Annotated Data
* (ECCV 22) RBC: Rectifying the Biased Context in Continual Semantic Segmentation
* (ECCV 22) Continual Semantic Segmentation via Structure Preserving and Projected Feature Alignment
* (NeurIPS 22) Decomposed Knowledge Distillation for Class-Incremental Semantic Segmentation
* (NeurIPS 22) ALIFE: Adaptive Logit Regularizer and Feature Replay for Incremental Semantic Segmentation
* (NeurIPS 22) Mining Unseen Classes via Regional Objectness: A Simple Baseline for Incremental Segmentation
* (TPAMI 22) Modeling the Background for Incremental and Weakly-Supervised Semantic Segmentation
* (TPAMI 22) Uncertainty-aware Contrastive Distillation for Incremental Semantic Segmentation
* (TNNLS 22) Self-Training for Class-Incremental Semantic Segmentation
* (CVPR 23) Foundation Model Drives Weakly Incremental Learning for Semantic Segmentation
* (CVPR 23) CoMFormer: Continual Learning in Semantic and Panoptic Segmentation
* (CVPR 23) Continual Semantic Segmentation with Automatic Memory Sample Selection
* (CVPR 23) Principles of Forgetting in Domain-Incremental Semantic Segmentation in Adverse Weather Conditions
* (CVPR 23) Unsupervised Continual Semantic Adaptation through Neural Rendering
* (CVPR 23) Federated Incremental Semantic Segmentation
* (CVPR 23) Incrementer: Transformer for Class-Incremental Semantic Segmentation with Knowledge Distillation Focusing on Old Class
* (CVPR 23) Endpoints Weight Fusion for Class Incremental Semantic Segmentation

### Image generation

* (NeurIPS 18) Memory Replay GANs: Learning to Generate New Categories without Forgetting
* (ICCV 19) Lifelong GAN: Continual Learning for Conditional Image Generation
* (ECCV 20) Piggyback GAN: Efficient Lifelong Learning for Image Conditioned Generation
* (CVPR 21) Hyper-LifelongGAN: Scalable Lifelong Learning for Image Conditioned Generation
* (CVPR 21) Efficient Feature Transformations for Discriminative and Generative Continual Learning
* (ICCV 23) LFS-GAN: Lifelong Few-Shot Image Generation

### Person re-identification

* (AVSS 19) Continuous Learning without Forgetting for Person Re-Identification
* (AAAI 21) Generalising without Forgetting for Lifelong Person Re-Identification
* (WACV 21) Continual Representation Learning for Biometric Identification
* (CVPR 21) Lifelong Person Re-Identification via Adaptive Knowledge Accumulation
* (AAAI 22) Lifelong Person Re-identification by Pseudo Task Knowledge Preservation
* (CVPR 22) Lifelong Unsupervised Domain Adaptive Person Re-Identification with Coordinated Anti-forgetting and Adaptation
* (ACM MM 22) Meta Reconciliation Normalization for Lifelong Person Re-Identification
* (ACM MM 22) Patch-based Knowledge Distillation for Lifelong Person Re-Identification
* (BMVC 22) Positive Pair Distillation Considered Harmful: Continual Meta Metric Learning for Lifelong Object Re-Identification

### Vision-language learning

* (ACL 19) Psycholinguistics Meets Continual Learning: Measuring Catastrophic Forgetting in Visual Question Answering
* (EMNLP 20) Visually Grounded Continual Learning of Compositional Phrases
* (NeurIPS 20) RATT: Recurrent Attention to Transient Tasks for Continual Image Captioning
* (ECCV 22) Generative Negative Text Replay for Continual Vision-Language Pretraining
* (NeurIPS 22) BMU-MoCo: Bidirectional Momentum Update for Continual Video-Language Modeling
* (AAAI 23) Symbolic Replay: Scene Graph as Prompt for Continual Learning on VQA Task
* (CVPR 23) ConStruct-VL: Data-Free Continual Structured VL Concepts Learning
* (CVPR 23) VQACL: A Novel Visual Question Answering Continual Learning Setting
* (ICML 23) Open-VCLIP: Transforming CLIP to an Open-vocabulary Video Model via Interpolated Weight Optimization
* (ICML 23) Continual Vision-Language Representation Learning with Off-Diagonal Information
* (ICCV 23) Preventing Zero-Shot Transfer Degradation in Continual Learning of Vision-Language Models
* (ICCV 23) CTP: Towards Vision-Language Continual Pretraining via Compatible Momentum Contrast and Topology Preservation

### Reinforcement learning

* (AAAI 18) Selective Experience Replay for Lifelong Learning
* (ICLR 18) Continuous Adaptation via Meta-Learning in Nonstationary and Competitive Environments
* (ICLR 18) Progressive Reinforcement Learning with Distillation for Multi-Skilled Motion Control
* (ICML 18) State Abstractions for Lifelong Reinforcement Learning
* (ICML 18) Policy and Value Transfer in Lifelong Reinforcement Learning
* (ICML 18) Continual Reinforcement Learning with Complex Synapses
* (NeurIPS 18) Lifelong Inverse Reinforcement Learning
* (ICLR 19) Deep Online Learning via Meta-Learning: Continual Adaptation for Model-Based RL
* (ICLR 19) Learning to Learn without Forgetting by Maximizing Transfer and Minimizing Interference
* (ICML 19) Policy Consolidation for Continual Reinforcement Learning
* (NeurIPS 19) Experience Replay for Continual Learning
* (NeurIPS 20) Deep Reinforcement and InfoMax Learning
* (NeurIPS 20) Continual Learning of Control Primitives: Skill Discovery via Reset-Games
* (NeurIPS 20) Task-Agnostic Online Reinforcement Learning with an Infinite Mixture of Gaussian Processes
* (NeurIPS 20) Lifelong Policy Gradient Learning of Factored Policies for Faster Training Without Forgetting
* (ICLR 21) CoMPS: Continual Meta Policy Search
* (ICLR 21) Reset-Free Lifelong Learning with Skill-Space Planning
* (Neurocomputing 21) Pseudo-Rehearsal: Achieving Deep Reinforcement Learning without Catastrophic Forgetting
* (AAAI 22) Same State, Different Task: Continual Reinforcement Learning without Interference
* (ICLR 22) Modular Lifelong Reinforcement Learning via Neural Composition
* (ICLR 22) Generalisation in Lifelong Reinforcement Learning through Logical Composition
* (NeurIPS 22) Model-based Lifelong Reinforcement Learning with Bayesian Exploration
* (NeurIPS 22) Continual Learning In Environments With Polynomial Mixing Times
* (NeurIPS 22) Disentangling Transfer in Continual Reinforcement Learning
* (ICLR 23) Building a Subspace of Policies for Scalable Continual Learning
* (ICML 23) Continual Task Allocation in Meta-Policy Network via Sparse Prompting

#### Non-stationary MDP

* (NeurIPS 19) A Meta-MDP Approach to Exploration for Lifelong Reinforcement Learning
* (NeurIPS 19) Non-Stationary Markov Decision Processes a Worst-Case Approach using Model-Based Reinforcement Learning
* (AAAI 20) Lifelong Learning with a Changing Action Set
* (ICML 20) Optimizing for the Future in Non-Stationary MDPs
* (NeurIPS 20) Dynamic Regret of Policy Optimization in Non-stationary Environments
* (NeurIPS 20) Towards Safe Policy Improvement for Non-Stationary MDPs
* (AAAI 21) Lipschitz Lifelong Reinforcement Learning
* (ICML 21) Deep Reinforcement Learning amidst Continual Structured Non-Stationarity
* (ICLR 22) Reinforcement Learning in Presence of Discrete Markovian Context Evolution
* (NeurIPS 22) Off-Policy Evaluation for Action-Dependent Non-stationary Environments
* (NeurIPS 22) Factored Adaptation for Non-Stationary Reinforcement Learning

### Others

* (SIGGRAPH 19) Unsupervised Incremental Learning for Hand Shape and Pose Estimation
* (AAAI 20) Generative Continual Concept Learning
* (CVPR 20) Online Depth Learning against Forgetting in Monocular Videos
* (ECCV 20) SPARK: Spatial-aware Online Incremental Attack Against Visual Tracking
* (IROS 20) Latent Replay for Real-Time Continual Learning
* (CVPRW 20) Continual Learning for Anomaly Detection in Surveillance Videos
* (CVPRW 20) Continual Learning of Object Instances
* (ICMLW 20) Continual Learning in Human Activity Recognition: an Empirical Analysis of Regularization
* (arXiv 20) Learning Causal Models Online
* (AAAI 21) Continual Learning for Named Entity Recognition
* (CVPR 21) Image De-raining via Continual Learning
* (CVPRW 21) Selective Replay Enhances Learning in Online Continual Analogical Reasoning
* (ICCV 21) Class-Incremental Learning for Action Recognition in Videos
* (ICCV 21) Else-Net: Elastic Semantic Network for Continual Action Recognition from Skeleton Data
* (ICCV 21) Detection and Continual Learning of Novel Face Presentation Attacks
* (ICCV 21) Continual Learning for Image-Based Camera Localization
* (ICCV 21) Continual Neural Mapping: Learning An Implicit Scene Representation from Sequential Observations
* (BMVC 21) Incremental Learning for Animal Pose Estimation using RBF k-DPP
* (WWW 22) Multimodal Continual Graph Learning with Neural Architecture Search
* (CVPR 22) Lifelong Graph Learning
* (CVPR 22) Continual Test-Time Domain Adaptation
* (ICML 22) Efficient Test-Time Model Adaptation without Forgetting
* (ECCV 22) Novel Class Discovery without Forgetting
* (ECCV 22) incDFM: Incremental Deep Feature Modeling for Continual Novelty Detection
* (AISTATS 22) Online Continual Adaptation with Active Self-Training
* (ACL 22) Continual Prompt Tuning for Dialog State Tracking
* (Findings of ACL 22) Consistent Representation Learning for Continual Relation Extraction
* (COLING 22) Continuous Detection, Rapidly React: Unseen Rumors Detection based on Continual Prompt-Tuning
* (RA-L 22) Improving Pedestrian Prediction Models with Self-Supervised Continual Learning
* (ICLR 23) Towards Open Temporal Graph Neural Networks
* (WWW 23) Dynamically Expandable Graph Convolution for Streaming Recommendation
* (CVPR 23) Cloud-Device Collaborative Adaptation to Continual Changing Environments in the Real-world
* (CVPR 23) PIVOT: Prompting for Video Continual Learning
* (CVPR 23) Geometry and Uncertainty-Aware 3D Point Cloud Class-Incremental Semantic Segmentation
