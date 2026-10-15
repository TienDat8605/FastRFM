# FastRFM

Research workspace for accelerating Riemannian flow matching and guided human
pose inference, including the PoseD-Flow reproduction framework, experimental
code, proposals, hypotheses, manuscript drafts, reference papers, and reports.

## Research documents

- [Reproduction plan](PLAN.md) and [preparation checklist](TODO.md)
- [Acceleration proposals and hypotheses](protocol/acceleration_blueprint.md)
- [Acceleration pilot](protocol/acceleration_pilot.md)
- [Guidance-aware manifold distillation hypothesis](protocol/riemannian_consistency_distillation.md)
- [Exact-pullback four-step screening protocol](protocol/pullback_screen.md)
- [Colab inference study](protocol/inference_colab.md) and [3DPW study](protocol/study_3dpw.md)
- [Manuscript draft](papers/fast_riemannian_dflow_draft.md),
  [LaTeX source](papers/fast_riemannian_dflow.tex), and
  [PDF](papers/fast_riemannian_dflow.pdf)
- [Reference paper catalog](papers/README.md)
- Experiment reports and measurements under `results/`

Proposals and projected speedups are research hypotheses; consult the experiment
reports for measured outcomes and limitations. Dataset downloads, licensed body
models, checkpoints, and runtime archives remain local. Preparation scripts and
protocols document how to obtain inputs and reproduce runs.

## PoseD-Flow reproduction workspace

Active follow-up: [exact-pullback four-step screening](protocol/pullback_screen.md),
via `posed_flow_colab_poc.py --mode pullback_screen`, with frozen sequence-disjoint
confirmation and a four-hour T4 budget.

Current work: [Colab SSH inference acceleration](protocol/inference_colab.md),
using `posed_flow_colab_poc.py --mode inference` to compare two/four-step
midpoint shortcut students with full-graph and fixed-anchor discrete adjoints.
The original 3DPW runtime was stopped at the user's request; partial outputs
are archived locally. This small-data GPU PoC is distinct from full-paper
reproduction. Results are downloaded under `results/inference_colab/`.

This repository implements the complete reproduction framework for **PoseD-Flow** (CVPR 2026): *Versatile and Guided Flow Matching Model of Human Pose* (Jebastin Nadar, Simone Foti, Tolga Birdal).

---

## 1. Structure Overview

```text
.
├── protocol/
│   ├── targets.md              # Target results sheet (Tables 1–6 metrics, units, baselines)
│   └── specification.md        # Protocol manifest & discrepancy resolutions
├── configs/
│   ├── train_poserfm.yaml      # 50k steps, AdamW, EMA 0.99, batch 4096
│   ├── train_posefm.yaml       # Matched Euclidean baseline
│   ├── eval_generation.yaml    # Table 1: FID, APD, d_NN (Midpoint solver)
│   ├── eval_completion.yaml    # Table 2: 4 occlusion masks, 10 hypotheses
│   ├── eval_denoising.yaml     # Table 3: HumanEva & HPS (40mm & 100mm)
│   ├── eval_hmr.yaml           # Tables 4 & 5: EHF benchmark (ViTPose + CLIFF)
│   └── eval_ablations.yaml     # Table 6: 12 geometric ablation conditions
├── src/
│   ├── geometry/
│   │   ├── so3.py              # SO(3)^K Exp/Log maps, hat/vee, Haar sampling, SVD projection
│   │   ├── radam.py            # Riemannian Adam optimizer (Algorithm 3)
│   │   └── smpl_utils.py       # Differentiable kinematics & SMPL joint tree
│   ├── models/
│   │   ├── poserfm.py          # PoseRFM (190 -> 63) Riemannian flow matching
│   │   └── posefm.py           # PoseFM (190 -> 189) Euclidean flow matching
│   ├── solvers/
│   │   ├── ode_solvers.py      # Euler and Midpoint ODE integration
│   │   └── dflow.py            # Riemannian D-Flow guidance (Algorithm 2)
│   ├── data/
│   │   ├── amass_dataset.py    # AMASS splits, 30 Hz resampling, 10% trimming
│   │   └── masks.py            # Left leg, legs, arms, torso occlusion masks
│   └── evaluation/
│       ├── metrics.py          # FID, APD, d_NN, MPJPE, MPVPE, PA-MPJPE, PCK@50
│       ├── eval_generation.py  # Generation pipeline
│       ├── eval_completion.py  # Completion pipeline
│       ├── eval_denoising.py   # Denoising pipeline
│       ├── eval_hmr.py         # Human mesh recovery pipeline
│       └── eval_ablations.py   # 12-configuration ablation matrix runner
├── scripts/
│   ├── train.py                # Model training runner
│   ├── prepare_amass.py        # AMASS dataset auditor & manifest generator
│   ├── prepare_denoising_data.py # HumanEva (190) and HPS (350) segment spec
│   ├── prepare_completion_masks.py # Occlusion masks precomputation
│   ├── prepare_ehf.py          # EHF specification auditor
│   └── run_table_summary.py    # Report generator & tolerance verifier
├── tests/
│   ├── test_geometry.py        # Numerical tests for SO(3) and kinematics
│   └── test_models.py          # Forward, loss, backward, and metrics tests
├── requirements.txt            # Pinned Python package dependencies
├── pyproject.toml              # Build specification
├── PLAN.md                     # Full reproduction experiment plan
└── TODO.md                     # Step-by-step preparation checklist
```

---

## 2. Environment Setup (Remote GPU)

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install -e .

# 2. Run unit tests
python3 -m unittest discover tests
```

---

## 3. Reproduction Workflow

### Step 1: Data Preparation
```bash
python scripts/prepare_amass.py --amass_dir /path/to/AMASS --output_dir data/amass
python scripts/prepare_denoising_data.py --output_dir data/denoising
python scripts/prepare_completion_masks.py
python scripts/prepare_ehf.py
```

### Step 2: Training Models (Seeds 0, 1, 2)
```bash
python scripts/train.py --config configs/train_poserfm.yaml --seed 0
python scripts/train.py --config configs/train_posefm.yaml --seed 0
```

### Step 3: Benchmarking & Reporting
```bash
# Aggregates results against targets.md and verifies tolerance bounds:
python scripts/run_table_summary.py --results_dir results --output_md results/reproduction_summary.md
```
