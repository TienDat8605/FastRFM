# Target Results Sheet: PoseD-Flow (CVPR 2026) Tables 1–6

This sheet records all published target metrics, units, uncertainty ranges, and baseline benchmarks directly from the main paper and official supplementary material.

---

## Table 1: Unconditional Pose Generation

- **Protocol**: 20 independent sets of 500 poses sampled unconditionally.
- **Evaluation**: Mean $\pm$ 95% confidence interval across the 20 sets.
- **Metrics**: 
  - FID: Fréchet Inception Distance computed on 3D SMPL joint positions (lower is better, $\downarrow$).
  - APD (cm): Average Pairwise Distance between 3D joint positions (higher indicates diversity, $\uparrow$).
  - $d_{\text{NN}}$ (rad): Mean nearest-neighbor geodesic/angular distance to training poses (lower is better, $\downarrow$).

| Method | FID $\downarrow$ | APD (cm) $\uparrow$ | $d_{\text{NN}}$ (rad) $\downarrow$ | Source / Status |
| :--- | :--- | :--- | :--- | :--- |
| GMM [7] | $0.435 \pm 0.017$ | $21.944 \pm 0.102$ | $0.159 \pm 0.001$ | Published baseline |
| VPoser [53] | $0.048 \pm 0.002$ | $14.684 \pm 0.138$ | $0.074 \pm 0.000$ | Baseline control |
| GAN-S [19] | $0.201 \pm 0.030$ | $10.914 \pm 0.396$ | $0.098 \pm 0.001$ | Published baseline |
| Pose-NDF [60] | $3.920 \pm 0.034$ | $\mathbf{37.813} \pm 0.085$ | $0.838 \pm 0.001$ | Published baseline |
| GFPose-A [16, 26] | $1.246 \pm 0.005$ | $13.876 \pm 0.116$ | — | Published baseline |
| GFPose-Q [16, 26] | $1.624 \pm 0.002$ | $6.773 \pm 0.112$ | $0.159 \pm 0.000$ | Published baseline |
| FM-Dis [26] | $0.346 \pm 0.007$ | $6.849 \pm 0.199$ | $0.086 \pm 0.001$ | Published baseline |
| NRDF [26] | $0.636 \pm 0.007$ | $23.116 \pm 0.105$ | $0.177 \pm 0.001$ | Baseline control |
| Lie-HMR [32] | $0.825 \pm 0.067$ | $23.999 \pm 0.602$ | — | Published baseline |
| DPoser [46] | $0.019 \pm 0.001$ | $14.992 \pm 0.123$ | $0.073 \pm 0.001$ | Baseline control |
| **PoseFM** (Euclidean baseline) | $0.016 \pm 0.001$ | $14.762 \pm 0.176$ | $0.079 \pm 0.000$ | Re-trained control |
| **PoseRFM** ($N=100$) | $0.014 \pm 0.001$ | $15.481 \pm 0.124$ | $\mathbf{0.069} \pm 0.001$ | **Target reproduction** |
| **PoseRFM** ($N=1000$) | $\mathbf{0.013} \pm 0.001$ | $15.544 \pm 0.167$ | $0.070 \pm 0.001$ | **Target reproduction** |

---

## Table 2: Pose Completion

- **Protocol**: AMASS test split subsampled by 10. 10 hypotheses generated per test pose. Visible joints replaced with observations for primary benchmark.
- **Metrics**: 
  - MPVPE (mm): Mean Per Vertex Position Error on occluded body parts (mean $\pm$ std across 10 hypotheses, $\downarrow$).
  - APD (cm): Average Pairwise Distance across the 10 hypotheses (diversity, $\uparrow$).

| Method | Left Leg MPVPE $\downarrow$ | Left Leg APD $\uparrow$ | Legs MPVPE $\downarrow$ | Legs APD $\uparrow$ | Arms MPVPE $\downarrow$ | Arms APD $\uparrow$ | Torso MPVPE $\downarrow$ | Torso APD $\uparrow$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| VPoser [53] | $200.51 \pm 12.20$ | 2.41 | $221.34 \pm 16.40$ | 5.44 | $206.72 \pm 12.91$ | 4.08 | $58.71 \pm 7.38$ | 1.56 |
| Pose-NDF [60] | $168.45 \pm 8.66$ | 1.95 | $169.86 \pm 6.12$ | 1.97 | $260.94 \pm 4.81$ | 1.20 | $114.97 \pm 5.43$ | 0.93 |
| CVPoser [46] | $128.04 \pm 10.36$ | 1.91 | $134.35 \pm 10.17$ | 2.43 | $162.82 \pm 5.58$ | 1.08 | $51.23 \pm 4.32$ | 0.57 |
| DPoser [46] | $\mathbf{78.31} \pm 27.13$ | 6.53 | $102.46 \pm 25.39$ | 7.75 | $\mathbf{104.94} \pm 26.44$ | 5.69 | $44.60 \pm 14.65$ | 2.21 |
| **PoseFM** | $102.60 \pm 34.82$ | $\mathbf{8.77}$ | $129.04 \pm 30.93$ | $\mathbf{9.94}$ | $140.88 \pm 35.63$ | $\mathbf{7.85}$ | $51.86 \pm 17.25$ | $\mathbf{2.70}$ |
| **PoseRFM** (ours) | $83.81 \pm 29.07$ | 6.02 | $\mathbf{95.00} \pm 26.08$ | 7.23 | $107.36 \pm 30.41$ | 5.72 | $\mathbf{39.75} \pm 9.12$ | 1.21 |

---

## Table 3: Motion Denoising

- **Protocol**: 30 Hz sequences, 60-frame (2-second) segments. Joint coordinates perturbed with Gaussian noise $\sigma \in \{40, 100\}\text{ mm}$.
- **Datasets**: 190 HumanEva segments (AMASS subset), 350 HPS segments (50 per subject, seed 42).
- **Metric**: MPJPE (mm) between recovered 3D joints and ground truth ($\downarrow$).

| Method | AMASS (HumanEva) 40 mm $\downarrow$ | AMASS (HumanEva) 100 mm $\downarrow$ | HPS 40 mm $\downarrow$ | HPS 100 mm $\downarrow$ |
| :--- | :--- | :--- | :--- | :--- |
| w/o prior | 24.19 | 51.48 | 23.67 | 50.87 |
| VPoser [53] | 23.42 | 49.10 | 22.78 | 46.69 |
| Pose-NDF [60] | 22.13 | 46.10 | 21.60 | 47.50 |
| MVAE [39] | 26.80 | — | — | — |
| HuMoR [54] | 22.69 | — | — | — |
| DPoser [46] | 19.87 | $\mathbf{33.18}$ | 20.54 | 35.32 |
| **PoseRFM** (ours) | $\mathbf{18.88}$ | 34.68 | $\mathbf{19.79}$ | $\mathbf{32.95}$ |

---

## Tables 4 & 5: Human Mesh Recovery (EHF Dataset)

- **Protocol**: All 100 EHF images. ViTPose-H 2D detections. SMPLify optimization.
- **Initializations**: From scratch (mean pose) vs. CLIFF (`hr48-PA53.7_MJE91.4_MVE110.0_agora_val.pt`).
- **Metrics**: 
  - PA-MPJPE (mm): Procrustes-aligned Mean Per Joint Position Error ($\downarrow$).
  - PA-MPVPE (mm): Procrustes-aligned Mean Per Vertex Position Error ($\downarrow$).
  - PCK@50 (%): Percentage of Correct Keypoints within 50 mm ($\uparrow$).

### Table 4: PA-MPJPE Overview

| Method | From Scratch $\downarrow$ | CLIFF Init $\downarrow$ |
| :--- | :--- | :--- |
| w/o fitting | 108.57 | 56.62 |
| GMM [7] | 58.32 | 51.02 |
| VPoser [53] | 58.08 | 49.39 |
| GAN-S [19] | 57.26 | 49.58 |
| Pose-NDF [60] | 57.87 | 49.50 |
| NRDF [26] | 57.38 | 49.27 |
| DPoser [46] | 56.05 | 49.05 |
| **PoseRFM** (ours) | $\mathbf{54.85}$ | $\mathbf{47.24}$ |

### Table 5: Detailed Multi-Metric Benchmark

| Initialization | Method | PA-MPJPE (mm) $\downarrow$ | PA-MPVPE (mm) $\downarrow$ | PCK@50 (%) $\uparrow$ |
| :--- | :--- | :--- | :--- | :--- |
| **From Scratch** | DPoser [46] | 56.05 | 53.67 | 61.60 |
| | **PoseRFM** (ours) | $\mathbf{54.85}$ | $\mathbf{53.16}$ | $\mathbf{65.29}$ |
| **CLIFF Init** | DPoser [46] | 49.05 | 52.92 | 66.62 |
| | **PoseRFM** (ours) | $\mathbf{47.24}$ | $\mathbf{48.67}$ | $\mathbf{71.40}$ |

---

## Table 6: Geometric Ablation Study (Pose Completion)

- **Protocol**: AMASS test split, left-leg and legs occluded.
- **Ablated factors**:
  - Model formulation: PoseFM (Euclidean FM), PoseFM + geometry, PoseRFM (Riemannian FM).
  - Geodesic data loss: off ($\times$) vs. on ($\checkmark$).
  - Trajectory loss $\mathcal{L}_{\text{traj}}$: off ($\times$) vs. on ($\checkmark$).

| Model Variant | Geodesic Loss | Trajectory Loss | Left Leg MPVPE (mm) $\downarrow$ | Left Leg APD (cm) $\uparrow$ | Legs MPVPE (mm) $\downarrow$ | Legs APD (cm) $\uparrow$ |
| :--- | :---: | :---: | :--- | :--- | :--- | :--- |
| **PoseFM** | $\times$ | $\times$ | 102.60 | 8.77 | 129.04 | 9.94 |
| | $\checkmark$ | $\times$ | 215.25 | 20.11 | 273.96 | 25.94 |
| | $\times$ | $\checkmark$ | 141.66 | 5.92 | 148.77 | 6.29 |
| | $\checkmark$ | $\checkmark$ | 180.51 | 16.81 | 212.90 | 20.52 |
| **PoseFM + Geometry** | $\times$ | $\times$ | 99.79 | 7.45 | 119.92 | 8.56 |
| | $\checkmark$ | $\times$ | 91.00 | 6.34 | 115.58 | 7.45 |
| | $\times$ | $\checkmark$ | 129.55 | 4.86 | 138.05 | 4.78 |
| | $\checkmark$ | $\checkmark$ | 90.46 | 5.79 | 115.67 | 6.53 |
| **PoseRFM** | $\times$ | $\times$ | 107.32 | 9.56 | 120.26 | 11.26 |
| | $\checkmark$ | $\times$ | 91.80 | 8.07 | 107.28 | 10.02 |
| | $\times$ | $\checkmark$ | 109.35 | 1.81 | 129.44 | 1.50 |
| | $\checkmark$ | $\checkmark$ | $\mathbf{83.81}$ | 6.02 | $\mathbf{95.00}$ | 7.23 |
