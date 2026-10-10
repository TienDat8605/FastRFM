# Protocol Specification & Manifest: PoseD-Flow Reproduction

This document defines the frozen reproduction specification, establishes provenance for every component (Published, Baseline-Derived, Reconstructed), and resolves discrepancies identified across the main paper and supplement.

---

## 1. Discrepancy Resolutions

| Setting / Issue | Published Evidence | Frozen Resolution | Classification |
| :--- | :--- | :--- | :--- |
| **Tangent Output Dimension** | Main text §3 mentions $189$ flat input and tangent vectors; Supplement §D (iii) explicitly states: $\dim(u_t) = K \times 3 = 21 \times 3 = 63$. | Model input is $190$ ($21 \times 9 = 189$ rotation matrix entries $+ 1$ time $t$). Model output is $63$ tangent coordinates, reshaped to $(B, 21, 3)$ Lie algebra vectors $\omega$ and mapped to $\mathfrak{so}(3)$ skew-symmetric matrices via the hat operator $\hat{\omega}$. | **Published (Supplement §D)** |
| **Rotation Blending & Projection** | Supplement Eq. (D.1) uses linear blending of matrices: $x_0 = \sqrt{\alpha} x_0^{\text{rand}} + \sqrt{1-\alpha} x_0^{\text{obs}}$. Matrices do not remain on $\mathrm{SO}(3)$. | Compute the blended matrix, then project each $3 \times 3$ block onto $\mathrm{SO}(3)$ via SVD / Polar decomposition: $R = U \operatorname{diag}(1, 1, \det(UV^T)) V^T$ where $M = U \Sigma V^T$. | **Reconstructed (Documented)** |
| **Distance Metric in Completion** | Main paper Eq. (20) writes geodesic distance; Supplement Definition 7 specifies $L_1$ product metric $d(x, x') = \sum_{k=1}^{21} d_{\mathrm{SO}(3)}(R_k, R_k')$. | Completion objective: $\mathcal{L}_{\text{data}} = \sum_{k \in \text{observed}} \theta_k$, where $\theta_k = \arccos\left(\frac{\text{tr}(R_k^{\text{obs}\top} R_k) - 1}{2}\right)$. For training RFM: squared tangent norm $\|\hat{u}_t - \dot{x}_t\|^2$. | **Published (Supplement Def 7)** |
| **Solver & Retraction Policy** | Main paper mentions Exp-map Euler; Supplement Table E.1 & E.2 compare Euler, Midpoint, per-step retraction, Exp map, and last-step retraction. | **Generation**: Midpoint solver with 100 steps ($N=100$, 200 function evaluations), last-step projection onto $\mathrm{SO}(3)$.<br>**Inverse Tasks** (Completion, Denoising, HMR): Euler solver with 100 steps, last-step projection onto $\mathrm{SO}(3)$ during forward pass. | **Published (Supplement §E)** |
| **Trajectory Regularizer** | Main paper §4 Eq. (11): $\mathcal{L}_{\text{traj}} = \sum_{i, k} (3 - \text{tr}(R_k^{(i)}))$. | Penalizes deviation of intermediate rotation matrices from identity across ODE steps and joints. Weight $\lambda_{\text{traj}}$ specified per task in Supplement Table D.1. | **Published** |
| **PCK@50 Evaluation Metric** | Main paper §5: "counts a joint as correct if its distance to the ground truth is within 50 mm". | 3D Procrustes-aligned percentage of joints with Euclidean error $\le 50\text{ mm}$ on EHF 100 images, standard in SMPLify/EHF evaluation. | **Baseline-Derived** |

---

## 2. Settings Provenance Matrix

### A. Model Architectures & Training

| Parameter / Choice | Value | Status | Notes |
| :--- | :--- | :--- | :--- |
| **Architecture** | 4 hidden layers, 512 units, Swish (SiLU) activation | Published | Main paper §3 |
| **Input Representation** | 21 joints $\times$ 9 elements ($3 \times 3$ rotation matrices) + 1 scalar time $t \in [0, 1]$ = 190 | Published | Main paper §3 |
| **Output Representation** | 21 joints $\times$ 3 tangent components = 63 | Published | Supplement §D |
| **Training Steps** | 50,000 steps | Published | Main paper §3 |
| **Batch Size** | 4,096 | Published | Main paper §3 |
| **Optimizer** | AdamW ($\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$, $\beta_1=0.9, \beta_2=0.999$) | Published | Main paper §3 |
| **Scheduler** | ReduceLROnPlateau (factor 0.5, patience 5 validation cycles) | Published | Main paper §3 |
| **EMA** | Exponential Moving Average decay 0.99 | Published | Main paper §3 |
| **RFM Path** | Geodesic path: $\dot{x}_t(x \mid x_1) = -\text{Log}_{x_t}(x_1) / (1 - t)$ | Published | Main paper Eq. (8) |
| **Source Distribution** | Uniform Haar distribution on $\mathrm{SO}(3)^{21}$ | Reconstructed | Standard isotropic random rotation |
| **PoseFM Baseline** | Euclidean FM on flattened matrices, matching 4x512 MLP | Reconstructed | Main paper §5 |
| **PoseFM + Geometry** | Euclidean FM with manifold-aware Riemannian D-Flow guidance | Reconstructed | Main paper §5 |

### B. Guidance & Inverse Tasks Hyperparameters (Supplement Table D.1)

| Task | Solver | NFE | Iters | Learning Rate | Blend $\alpha$ | Loss Weights |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pose Completion** | Euler | 100 | 200 (legs, torso, left leg)<br>300 (arms) | 0.1 | 0.25 | $\lambda_{\text{data}} = 1$, $\lambda_{\text{traj}} = 10^{-3}$ |
| **Motion Denoising** | Euler | 100 | 300 (HumanEva 40mm)<br>400 (HumanEva 100mm, HPS) | 0.25 | 0.75 | $\lambda_{\text{data}} = 1$, $\lambda_{\text{traj}} = 10^{-5}$,<br>$\lambda_{\text{temp}} = 10^{-2}$ (40mm), $10^{-1}$ (100mm) |
| **Human Mesh Recovery** | Euler | 100 | 400 | 0.1 | 0.75 | $\lambda_{\text{data}} = 1$, $\lambda_{\alpha} = \lambda_{\beta} = \lambda_{\text{traj}} = 50$ |

---

## 3. Dataset Preprocessing & Splits

| Dataset | Split Role | Filtering & Preprocessing | Segment / Frame Counts |
| :--- | :--- | :--- | :--- |
| **AMASS** | Train: CMU, MPI_Limits, TotalCapture, Eyes_Japan, KIT, BioMotionLab_NTroje, BMLhandball, BMLmovi, EKUT, TCD_handMocap, ACCAD | Trim first/last 10% of sequences, resample to 30 Hz. Local body rotations $K=21$. Root translation/orientation zeroed. | $\approx 18$ million poses |
| **AMASS Test** | Test (Completion): Transitions_mocap, SSM_synced | Subsample every 10th frame. 10 hypotheses generated per pose. | Test split |
| **HumanEva** | Evaluation (Denoising only): Excluded from training/tuning | 30 Hz, 60-frame chunks (2 seconds). | 190 segments |
| **HPS** | Evaluation (Denoising): Held-out generalisation | 30 Hz, 60-frame chunks. 50 segments per subject using seed 42. | 350 segments |
| **EHF** | Evaluation (Mesh Recovery): 100 images with GT meshes | ViTPose-H 2D keypoints; CLIFF detections. | 100 images |
| **3DPW** | Qualitative comparison | In-the-wild test sequences. | Benchmark sequences |
