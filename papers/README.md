# Evolution Timeline of Flow Matching & Riemannian Flow Matching

This repository folder catalogs **25 landmark papers** documenting the evolution of **Flow Matching (FM)**, **Riemannian Flow Matching (RFM)**, **Guided Inversion**, and their applications to **Human Pose and Motion**, culminating in **PoseD-Flow (CVPR 2026)**.

---

## 1. Evolution Timeline Overview

```
                                  [2022-10] Flow Matching for Generative Modeling (Lipman et al.)
                                                               │
                     ┌─────────────────────────────────────────┴─────────────────────────────────────────┐
                     ▼                                                                                     ▼
    [2023-02] Flow Matching on General Geometries (RFM)                                  [2023-05] FlowGrad (Liu et al.)
                 (Chen & Lipman, ICLR 2024)                                                 Trajectory Guidance via Gradients
                     │                                                                                     │
         ┌───────────┴───────────────────────────────┐                                                     ▼
         ▼                                           ▼                                     [2024-02] D-Flow (Ben-Hamu et al.)
[Biomolecules & Chemistry]                   [Robotics & 3D Objects]                         Differentiating through Generative Flows
 • [2023-10] FoldFlow (Bose et al.)           • [2024-10] EquiGraspFlow (Lim et al.)                       │
 • [2023-10] FrameFlow (Yim et al.)           • [2025-09] RFMPose (Ouyang et al.)                          ├────────────────────────┐
 • [2024-06] FlowMM (Miller et al.)                          │                                             ▼                        ▼
 • [2024-10] MOFFlow (Kim et al.)                            │                                [2024-10] OC-Flow        [2025-01] TFG-Flow
 • [2024-12] FlowDock (Morehead et al.)                      │                                Optimal Control Guidance   SO(3)-Invariant Control
 • [2025-10] MATCHA (Frolova et al.)                         │                                             │                        │
                                                             │                                             ▼                        ▼
[Human Motion (Euclidean/Latent)]                            │                                [2025-08] Source-Guided   [2025-10] ESS-Flow
 • [2023-12] Motion Flow Matching (Hu et al.)                │                                 Flow Matching Inversion   Source-Space Guidance
 • [2025-04] FlowMotion (Canales Cuba et al.)                │                                             │                        │
 • [2026-03] OmniPoser (Liu & Tang)                          │                                             └───────────┬────────────┘
         │                                                   │                                                         │
         └───────────────────────────────────┬───────────────┴─────────────────────────────────────────────────────────┘
                                             ▼
                          [2026-03] PoseD-Flow (Nadar, Foti, Birdal - CVPR 2026)
                           • PoseRFM: First RFM model on articulated human pose SO(3)^K
                           • Riemannian D-Flow: First training-free Riemannian ODE inversion
```

---

## 2. Chronological Catalog of Downloaded Papers

All papers are stored locally in this directory with chronological prefix indices:

| # | Date | Title | Authors & Venue | Key Focus | Local PDF | Online Source |
|---|---|---|---|---|---|---|
| **01** | 2022-10 | *Flow Matching for Generative Modeling* | Lipman et al. (ICLR 2023) | **Foundational FM**: Vector field regression for continuous normalizing flows | [PDF](01_2022-10_Lipman_Flow_Matching_for_Generative_Modeling.pdf) | [arXiv:2210.02747](https://arxiv.org/abs/2210.02747) |
| **02** | 2023-02 | *Flow Matching on General Geometries* | Chen & Lipman (ICLR 2024) | **First Riemannian FM (RFM)**: Extends FM to Riemannian manifolds & compact Lie groups | [PDF](02_2023-02_Chen_Flow_Matching_on_General_Geometries_RFM.pdf) | [arXiv:2302.03660](https://arxiv.org/abs/2302.03660) |
| **03** | 2023-05 | *FlowGrad: Controlling the Output of Generative ODEs with Gradients* | Liu et al. (CVPR 2023) | **Early Flow Guidance**: Trajectory control via backpropagation along integration steps | [PDF](03_2023-05_Liu_FlowGrad_Controlling_Generative_ODEs.pdf) | [CVF OpenAccess](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_FlowGrad_Controlling_the_Output_of_Generative_ODEs_With_Gradients_CVPR_2023_paper.pdf) |
| **04** | 2023-10 | *SE(3)-Stochastic Flow Matching for Protein Backbone Generation (FoldFlow)* | Bose et al. (ICLR 2024) | **RFM in Biology**: Protein backbone generation on $\mathrm{SE}(3)$ | [PDF](04_2023-10_Bose_FoldFlow_SE3_Stochastic_Flow_Matching.pdf) | [arXiv:2310.02391](https://arxiv.org/abs/2310.02391) |
| **05** | 2023-10 | *Fast Protein Backbone Generation with SE(3) Flow Matching (FrameFlow)* | Yim et al. (2023) | **RFM in Biology**: Rigid-body frames for protein design using $\mathrm{SO}(3) \times \mathbb{R}^3$ | [PDF](05_2023-10_Yim_FrameFlow_Fast_Protein_Backbone_Generation.pdf) | [arXiv:2310.05297](https://arxiv.org/abs/2310.05297) |
| **06** | 2023-12 | *Motion Flow Matching for Human Motion Synthesis and Editing* | Hu et al. (2023) | **First Human Motion FM**: Text-to-motion generation in Euclidean coordinate/latent space | [PDF](06_2023-12_Hu_Motion_Flow_Matching.pdf) | [arXiv:2312.08895](https://arxiv.org/abs/2312.08895) |
| **07** | 2024-02 | *D-Flow: Differentiating Through Flows for Controlled Generation* | Ben-Hamu et al. (ICML 2024) | **Source Optimization Guidance**: Inverts frozen Euclidean flows by optimizing noise source | [PDF](07_2024-02_Ben-Hamu_D-Flow_Controlled_Generation.pdf) | [arXiv:2402.14017](https://arxiv.org/abs/2402.14017) |
| **08** | 2024-05 | *Flow Priors for Linear Inverse Problems via Iterative Corrupted Trajectory Matching* | Zhang et al. (NeurIPS 2024) | **Inverse Problems**: Flow matching priors for linear inverse problems | [PDF](08_2024-05_Zhang_Flow_Priors_Linear_Inverse_Problems.pdf) | [arXiv:2405.18816](https://arxiv.org/abs/2405.18816) |
| **09** | 2024-06 | *FlowMM: Generating Materials with Riemannian Flow Matching* | Miller et al. (ICML 2024) | **RFM in Materials**: Periodic crystal structures and material discovery | [PDF](09_2024-06_Miller_FlowMM_Generating_Materials_with_RFM.pdf) | [arXiv:2406.04713](https://arxiv.org/abs/2406.04713) |
| **10** | 2024-10 | *MOFFlow: Flow Matching for Structure Prediction of Metal-Organic Frameworks* | Kim et al. (2024) | **RFM in Chemistry**: Structure prediction for metal-organic frameworks | [PDF](10_2024-10_Kim_MOFFlow_Structure_Prediction.pdf) | [arXiv:2410.17270](https://arxiv.org/abs/2410.17270) |
| **11** | 2024-10 | *Training Free Guided Flow Matching with Optimal Control (OC-Flow)* | Wang et al. (ICLR 2025) | **Optimal Control Guidance**: Guided flow matching formulated as optimal control with $\mathrm{SO}(3)$ test | [PDF](11_2024-10_Wang_OC-Flow_Guided_Flow_Matching_Optimal_Control.pdf) | [arXiv:2410.18070](https://arxiv.org/abs/2410.18070) |
| **12** | 2024-10 | *EquiGraspFlow: SE(3)-Equivariant 6-DoF Grasp Pose Generative Flows* | Lim et al. (CoRL 2024) | **Robotics RFM**: Equivariant grasp pose generation on $\mathrm{SE}(3)$ | [PDF](12_2024-10_Lim_EquiGraspFlow_SE3_Grasp_Pose.pdf) | [PMLR 270](https://proceedings.mlr.press/v270/lim25a.html) |
| **13** | 2024-12 | *FlowDock: Geometric Flow Matching for Generative Protein-Ligand Docking* | Morehead & Cheng (ISMB 2025) | **RFM in Molecular Docking**: Geometric flow matching for ligand poses | [PDF](13_2024-12_Morehead_FlowDock_Geometric_Flow_Matching.pdf) | [arXiv:2412.10966](https://arxiv.org/abs/2412.10966) |
| **14** | 2025-01 | *TFG-Flow: Training-free Guidance in Multimodal Generative Flow* | Lin et al. (ICLR 2025) | **Training-Free Guidance**: Multimodal flow guidance with $\mathrm{SO}(3)$-invariant control | [PDF](14_2025-01_Lin_TFG-Flow_Training_Free_Guidance_Multimodal.pdf) | [arXiv:2501.14216](https://arxiv.org/abs/2501.14216) |
| **15** | 2025-02 | *On the Guidance of Flow Matching* | Feng et al. (2025) | **Theoretical Guidance Analysis**: Analysis of guidance mechanics in flow models | [PDF](15_2025-02_Feng_On_the_Guidance_of_Flow_Matching.pdf) | [arXiv:2502.02150](https://arxiv.org/abs/2502.02150) |
| **16** | 2025-02 | *Training-Free Guidance Beyond Differentiability (Tree Search)* | Guo et al. (2025) | **Non-differentiable Guidance**: Path steering in diffusion and flow models using tree search | [PDF](16_2025-02_Guo_Training_Free_Guidance_Beyond_Differentiability.pdf) | [arXiv:2502.11420](https://arxiv.org/abs/2502.11420) |
| **17** | 2025-04 | *FlowMotion: Target-Predictive Conditional Flow Matching for Human Motion* | Canales Cuba et al. (C&G 2025) | **Human Motion FM**: Target-predictive conditional FM for text-driven motion | [PDF](17_2025-04_Canales_Cuba_FlowMotion_Human_Motion_Generation.pdf) | [arXiv:2504.01338](https://arxiv.org/abs/2504.01338) |
| **18** | 2025-06 | *GGBall: Graph Generative Model on Poincaré Ball* | Bu et al. (2025) | **Hyperbolic RFM**: Flow matching on the Poincaré ball manifold | [PDF](18_2025-06_Bu_GGBall_Graph_Generative_Model_Poincare.pdf) | [arXiv:2506.07198](https://arxiv.org/abs/2506.07198) |
| **19** | 2025-08 | *Source-Guided Flow Matching* | Wang et al. (2025) | **Source-Space Guidance**: Optimizing initial distribution/source space for flow inversion | [PDF](19_2025-08_Wang_Source_Guided_Flow_Matching.pdf) | [arXiv:2508.14807](https://arxiv.org/abs/2508.14807) |
| **20** | 2025-09 | *RFMPose: Generative Category-Level Object Pose Estimation via RFM* | Ouyang et al. (NeurIPS 2025) | **Object Pose RFM**: 6D object pose generation with geodesic Riemannian flows | [PDF](20_2025-09_Ouyang_RFMPose_Category_Level_Object_Pose.pdf) | [NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/618c0a236e7aace748d09bc8fbed3e4d-Abstract-Conference.html) |
| **21** | 2025-09 | *BrainFlow: A Holistic Pathway of Dynamic Neural System on Manifold* | Zhou et al. (NeurIPS 2025) | **Brain Connectivity RFM**: SPD and Cholesky manifold flow matching | [PDF](21_2025-09_Zhou_BrainFlow_Dynamic_Neural_System_Manifold.pdf) | [NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2e098001625a8b8f22bf6c1aef09c4e5-Abstract-Conference.html) |
| **22** | 2025-10 | *ESS-Flow: Training-free Guidance as Inference in Source Space* | Kalaivanan et al. (2025) | **Distributional Source Guidance**: Extended source space optimization via sequential Monte Carlo | [PDF](22_2025-10_Kalaivanan_ESS-Flow_Source_Space_Guidance.pdf) | [arXiv:2510.05849](https://arxiv.org/abs/2510.05849) |
| **23** | 2025-10 | *MATCHA: Multi-Stage Riemannian Flow Matching for Molecular Docking* | Frolova et al. (2025) | **Multi-stage RFM**: Docking on rigid Lie groups $\mathrm{SE}(3)$ and internal torsions | [PDF](23_2025-10_Frolova_MATCHA_Multi_Stage_RFM_Docking.pdf) | [arXiv:2510.14586](https://arxiv.org/abs/2510.14586) |
| **24** | 2026-03 | *OmniPoser: Flexible Human Motion Recovery with Masked Flow Matching* | Liu & Tang (ECCV 2026) | **Sparse Motion Recovery**: High-speed masked conditional flow matching for heterogeneous sensors | [PDF](24_2026-03_Liu_OmniPoser_Flexible_Motion_Recovery.pdf) | [ECCV Proceedings](https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/3252.pdf) |
| **25** | 2026-03 | *PoseD-Flow: Versatile and Guided Flow Matching Model of Human Pose* | Nadar, Foti, Birdal (CVPR 2026) | **First Articulated Human Pose RFM + Riemannian D-Flow**: Full pose prior on $\mathrm{SO}(3)^{21}$ with manifold adjoint inversion | [PDF](25_2026-03_Nadar_PoseD-Flow_Versatile_Guided_Pose_Prior.pdf) | [CVPR 2026](file:///home/dat/HPE/Nadar_PoseD-Flow_Versatile_and_Guided_Flow_Matching_Model_of_Human_Pose_CVPR_2026_paper.pdf) |
| **26** | 2025-10 | *Riemannian Consistency Model* | Cheng et al. (NeurIPS 2025) | **Inference implementation reference — few-step manifold distillation**: motivates our [four-step distillation pilot](../scripts/run_distillation_pilot.py) and [transported-response loss](../src/acceleration/distillation.py). Our pilot is RCM-inspired endpoint compression; [reading and experiment note](../protocol/riemannian_consistency_distillation.md) documents differences from the paper's RCM objective and sampler. | [PDF](26_2025-10_Cheng_Riemannian_Consistency_Model.pdf) | [NeurIPS PDF](https://papers.neurips.cc/paper_files/paper/2025/file/7422317f84e8c83e4c1ad2ff87e1e88e-Paper-Conference.pdf) |
