# Awesome Continual Learning
This repository contains a curated list of continual learning papers (mostly until 2022). Please feel free to pull request.

* [Survey](#Survey)
* [Replay-based](##Replay-based)
  * [Memory replay](###Memory-replay)
  * [Generative replay](###Generative-replay)
* [Regularization-based](##Regularization-based)
  * [Bayesian-based](###Bayesian-based)
  * [Subspace-based](###Subspace-based)
  * [Distillation-based](###Distillation-based)
* [Architecture-based](##Architecture-based)
  * [Expansion](###Expansion)
  * [Mask](###Mask)
  * [Decompose](###Decompose)
* [Application](#Application)
  * [Object detection](##Object-detection)
  * [Semantic segmentation](##Semantic-segmentation)
  * [Image generation](##Image-generation)
  * [Person reID](##Person-reID)
  * [Vision-language learning](##Vision-language-learning)
  * [Reinforcement learning](##Reinforcement-learning)
  * [Others](##Others)



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
* (CVPR 23) Class-Incremental Exemplar Compression for Class-Incremental Learning
* (CVPR 23) Regularizing Second-Order Influences for Continual Learning

### Generative replay

* (NeurIPS 17) Continual Learning with Deep Generative Replay
* (NeurIPS 18) Memory Replay GANs: Learning to Generate New Categories without Forgetting
* (NeurIPS 20) GAN Memory with No Forgetting
* (Nature Communications 20) Brain-Inspired Replay for Continual Learning with Artificial Neural Networks
* (Neurocomputing 20) Lifelong Generative Modeling
* (ICCV 21) Always Be Dreaming: A New Approach for Data-Free Class-Incremental Learning
* (CVPR 22) Learning to Imagine: Diversify Memory for Incremental Learning using Unlabeled Data
* (ICLR 23) Incremental Learning of Structured Memory via Closed-Loop Transcription



## Regularization-based

* (ICML 17) Continual Learning through Synaptic Intelligence
* (ECCV 18) Memory Aware Synapses: Learning What (Not) to Forget
* (NeurIPS 20) Understanding the Role of Training Regimes in Continual Learning

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
* (ICML 22) Continual Learning with Guarantees via Weight Interval Constraints
* (CVPR 22) Towards Better Plasticity-Stability Trade-off in Incremental Learning: A Simple Linear Connector
* (AAAI 23) Continual Learning with Scaled Gradient Projection
* (ICLR 23) Building a Subspace of Policies for Scalable Continual Learning

### Distillation-based

* (TPAMI 17) Learning without Forgetting
* (CVPR 19) Learning without Memorizing
* (CVPR 20) Few-Shot Class-Incremental Learning
* (ECCV 20) Topology-Preserving Class-Incremental Learning
* (ICCV 21) Co^2^L: Contrastive Continual Learning



## Architecture-based

* (ICLR 20) Continual Learning with Hypernetworks
* (NeurIPS 21) DualNet: Continual Learning, Fast and Slow
* (ICLR 22) Learning Fast, Learning Slow: A General Continual Learning Method based on Complementary Learning System
* (ICLR 22) Model Zoo: A Growing Brain That Learns Continually
* (ECCV 22) CoSCL: Cooperation of Small Continual Learners is Stronger Than a Big One

### Expansion

* (CVPR 17) Expert Gate: Lifelong Learning with a Network of Experts
* (ICLR 18) Lifelong Learning with Dynamically Expandable Networks
* (ICML 19) Learn to Grow: A Continual Structure Learning Framework for Overcoming Catastrophic Forgetting
* (ICLR 20) A Neural Dirichlet Process Mixture Model for Task-Free Continual Learning
* (CVPR 21) DER: Dynamically Expandable Representation for Class Incremental Learning
* (CVPR 21) Adaptive Aggregation Networks for Class-Incremental Learning
* (NeurIPS 21) BNS: Building Network Structures Dynamically for Continual Learning
* (CVPR 22) DyTox: Transformers for Continual Learning with Dynamic Token Expansion
* (CVPR 22) Learning to Prompt for Continual Learning
* (ECCV 22) DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning
* (NeurIPS 22) S-Prompts Learning with Pre-trained Transformers: An Occam's Razor for Domain Incremental Learning
* (ICLR 23) Progressive Prompts: Continual Learning for Language Models
* (ICLR 23) Continual Transformers: Redundancy-Free Attention for Online Inference 

### Mask

* (arXiv 16) Progressive Neural Networks
* (ICML 17) AdaNet: Adaptive Structural Learning of Artificial Neural Networks
* (CVPR 18) PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning
* (ICML 18) Overcoming Catastrophic Forgetting with Hard Attention to the Task
* (NeurIPS 20) Supermasks in Superposition
* (ICLR 21) Long Live the Lottery: The Existence of Winning Tickets in Lifelong Learning
* (CVPR 21) Continual Learning via Bit-Level Information Preserving
* (CVPR 22) Meta-Attention for ViT-backed Continual Learning
* (ICML 22) Forget-free Continual Learning with Winning Subnetworks

### Decompose

* (ICLR 20) Scalable and Order-robust Continual Learning with Additive Parameter Decomposition
* (ICLR 20) BatchEnsemble: an Alternative Approach to Efficient Ensemble and Lifelong Learning
* (CVPR 21) Efficient Feature Transformations for Discriminative and Generative Continual Learning
* (NeurIPS 21) Mitigating Forgetting in Online Continual Learning with Neuron Calibration
* (ICLR 22) TRGP: Trust Region Gradient Projection for Continual Learning



## Application

### Object detection



### Semantic segmentation



### Image generation

* (NeurIPS 18) Memory Replay GANs: Learning to Generate New Categories without Forgetting
* (ICCV 19) Lifelong GAN: Continual Learning for Conditional Image Generation
* (ECCV 20) Piggyback GAN: Efficient Lifelong Learning for Image Conditioned Generation
* (CVPR 21) Hyper-LifelongGAN: Scalable Lifelong Learning for Image Conditioned Generation
* (CVPR 21) Efficient Feature Transformations for Discriminative and Generative Continual Learning



### Person reID

* (AVSS 19) Continuous Learning without Forgetting for Person Re-Identification
* (AAAI 21) Generalising without Forgetting for Lifelong Person Re-Identification
* (WACV 21) Continual Representation Learning for Biometric Identification
* (CVPR 21) Lifelong Person Re-Identification via Adaptive Knowledge Accumulation
* (AAAI 22) Lifelong Person Re-identification by Pseudo Task Knowledge Preservation
* (CVPR 22) Lifelong Unsupervised Domain Adaptive Person Re-Identification with Coordinated Anti-forgetting and Adaptation
* (ACM MM 22) Meta Reconciliation Normalization for Lifelong Person Re-Identification
* (ACM MM 22) Patch-based Knowledge Distillation for Lifelong Person Re-Identification



### Vision-language learning

* (ACL 19) Psycholinguistics Meets Continual Learning: Measuring Catastrophic Forgetting in Visual Question Answering
* (EMNLP 20) Visually Grounded Continual Learning of Compositional Phrases
* (NeurIPS 20) RATT: Recurrent Attention to Transient Tasks for Continual Image Captioning
* (ECCV 22) Generative Negative Text Replay for Continual Vision-Language Pretraining
* (NeurIPS 22) BMU-MoCo: Bidirectional Momentum Update for Continual Video-Language Modeling
* (AAAI 23) Symbolic Replay: Scene Graph as Prompt for Continual Learning on VQA Task



### Reinforcement learning

* (ICLR 23) Building a Subspace of Policies for Scalable Continual Learning



### Others

* (SIGGRAPH 19) Unsupervised Incremental Learning for Hand Shape and Pose Estimation
* (AAAI 20) Generative Continual Concept Learning
* (CVPR 20) Online Depth Learning against Forgetting in Monocular Videos
* (CVPRW 20) Continual Learning of Object Instances
* (ICMLW 20) Continual Learning in Human Activity Recognition: an Empirical Analysis of Regularization
* (arXiv 20) Learning Causal Models Online
* (AAAI 21) Continual Learning for Named Entity Recognition
* (CVPR 21) Image De-raining via Continual Learning
* (CVPRW 21) Selective Replay Enhances Learning in Online Continual Analogical Reasoning
* (ICCV 21) Class-Incremental Learning for Action Recognition in Videos
* (ICCV 21) Else-Net: Elastic Semantic Network for Continual Action Recognition from Skeleton Data
* (ICCV 21) Continual Learning for Image-Based Camera Localization
* (ICCV 21) Continual Neural Mapping: Learning An Implicit Scene Representation from Sequential Observations
* (BMVC 21) Incremental Learning for Animal Pose Estimation using RBF k-DPP
* (CVPR 22) Lifelong Graph Learning
* (CVPR 22) Continual Test-Time Domain Adaptation
* (ICML 22) Efficient Test-Time Model Adaptation without Forgetting
* (ECCV 22) Novel Class Discovery without Forgetting
* (ECCV 22) incDFM: Incremental Deep Feature Modeling for Continual Novelty Detection
* (ACL 22) Continual Prompt Tuning for Dialog State Tracking
* (Findings of ACL 22) Consistent Representation Learning for Continual Relation Extraction
* (COLING 22) Continuous Detection, Rapidly React: Unseen Rumors Detection based on Continual Prompt-Tuning
* (RA-L 22) Improving Pedestrian Prediction Models with Self-Supervised Continual Learning