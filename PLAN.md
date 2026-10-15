# PoseD-Flow reproduction experiment plan

Active follow-up: [exact-pullback four-step screening](protocol/pullback_screen.md),
via `posed_flow_colab_poc.py --mode pullback_screen`, with frozen sequence-disjoint
confirmation and a four-hour T4 budget.

**Current work:** inference acceleration on a fresh Colab T4 via SSH, defined in
[protocol/inference_colab.md](protocol/inference_colab.md). The previous 3DPW
runtime was terminated at the user's request; partial outputs were recovered
under `results/3dpw_study/20261014T014832Z-all/recovered/`. The earlier transfer
protocol remains in [protocol/study_3dpw.md](protocol/study_3dpw.md). The full-paper
reproduction below remains a separate longer-term objective.

## 1. Objective and evidence

Reproduce **Tables 1–6**: train PoseRFM from scratch, evaluate generation, completion, denoising, and human mesh recovery, and reproduce the geometric ablations. Use one remote GPU with at least 24 GB VRAM.

Sources:

- [Local main paper](Nadar_PoseD-Flow_Versatile_and_Guided_Flow_Matching_Model_of_Human_Pose_CVPR_2026_paper.pdf).
- [Official supplement](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Nadar_PoseD-Flow_Versatile_and_CVPR_2026_supplemental.pdf).
- [DPoser-X implementation](https://github.com/moonbow721/DPoser-X).
- [NRDF implementation](https://github.com/hynann/NRDF).

The advertised PoseDFlow repository returned 404 during inspection on 2026-10-10. Distinguish **author-protocol reproduction** from **documented reconstruction** wherever implementation details remain unavailable.

Material passport: academic-research-suite / experiment-agent; mode `plan`; date 2026-10-10; version `posedflow_reproduction_v1`; experimental results **UNVERIFIED**.

Preparation checklist: [TODO.md](TODO.md).

## 2. Preparation and implementation

### Freeze the reproduction specification

Before training, create a protocol manifest covering dataset versions, splits, body model, joint ordering, preprocessing, masks, metric definitions, checkpoints, and every experimental configuration. Mark each setting as publication-specified, inherited from baseline code, or reconstructed.

Resolve these discrepancies explicitly:

| Issue | Planned treatment |
|---|---|
| Main text suggests a 189-dimensional tangent output; supplement states 21 × 3 | Predict 63 tangent coordinates and convert them into joint-wise tangent vectors. |
| Main text uses an L2 product distance; supplement states L1 | Use the published sum of per-joint rotation angles for completion; retain the squared tangent norm for flow training. |
| Guidance description emphasizes manifold integration; supplementary results favor final-step projection | Use Euler with final-step rotation projection for primary inverse-task runs; validate an exponential-map variant on a pilot. |
| Initialization blends rotation matrices outside SO(3) | Apply the supplement's square-root-weighted blend, then project each matrix onto SO(3); label projection placement reconstructed. |
| PCK@50 is described as 2D accuracy with a millimetre threshold | Verify the baseline evaluator's coordinate space and alignment. If unresolved, report explicitly named 3D PCK@50mm as a reconstructed metric. |

For unavailable settings, adopt the reconstruction defaults below, freeze them before test evaluation, and report their effect separately. Do not claim exact reproduction for affected cells.

### Data preparation

- Obtain AMASS, HPS, EHF, 3DPW, and compatible SMPL assets.
- Use the NRDF/VPoser AMASS split: training contains CMU, MPI_Limits, TotalCapture, Eyes_Japan_Dataset, KIT, BioMotionLab_NTroje, BMLhandball, BMLmovi, EKUT, TCD_handMocap, and ACCAD; test contains Transitions_mocap and SSM_synced; validation contains HumanEva, MPI_HDM05, SFU, and MPI_mosh.
- Trim the first and last 10% of training sequences and resample to 30 Hz. Check the resulting count against the supplement's approximately **18 million poses**.
- Store 21 local body rotations; manage root orientation, translation, and shape separately.
- Fix shape to zero for generation, completion, and denoising. Optimize shape during mesh recovery.
- Reserve HumanEva exclusively for denoising evaluation when choosing hyperparameters; use the other validation collections for tuning.
- Record source sequence and frame identifiers so train, validation, and test overlap can be checked.

### Models and guidance

Implement PoseRFM with four 512-unit hidden layers, Swish activation, flattened rotation matrices plus scalar time as input, and a tangent velocity output. Train using geodesic interpolation and the conditional Riemannian flow-matching target.

Use the reported settings:

- 50,000 optimizer steps; effective batch size 4,096.
- AdamW: learning rate 1e-3, weight decay 1e-4.
- EMA decay 0.99.
- ReduceLROnPlateau: factor 0.5, patience five validation evaluations.

Reconstruction defaults: independent Haar-uniform source rotations; FP32; Adam betas `(0.9, 0.999)`; validation every 500 steps using a fixed validation sample; evaluate the EMA checkpoint with the lowest validation loss. Save optimizer, scheduler, EMA, and random-number states.

Train an architecture-matched Euclidean PoseFM control. If its original parameterization remains unavailable, use flattened rotation matrices with Euclidean interpolation and velocity regression, projecting generated endpoints to valid rotations. Label this baseline reconstructed.

For inverse tasks, freeze the prior and optimize the source through the complete differentiable ODE. Implement Riemannian Adam with moment transport. Use the paper's trajectory penalty:

```text
L_traj = sum over ODE steps and joints of (3 - trace(R)).
```

Maintain a configuration-driven interface for preparation, training, task evaluation, and table aggregation. Every evaluation must identify its checkpoint, data manifest, seed, solver, projection policy, and metric version.

## 3. Experiment matrix and targets

The targets below come from the main paper. Complete target rows, including baseline values and uncertainty labels, should be transcribed into the protocol manifest.

| Experiment | Protocol | PoseRFM target |
|---|---|---|
| **Table 1: generation** | 20 independent sets of 500 poses; midpoint, 100 and 1,000 integration steps | At 100: FID **0.014**, APD **15.481 cm**, d_NN **0.069 rad**. At 1,000: **0.013**, **15.544 cm**, **0.070 rad**. |
| **Table 2: completion** | AMASS test, every tenth frame; left leg, legs, arms, torso masked; 10 hypotheses per pose | MPVPE: **83.81, 95.00, 107.36, 39.75 mm**. APD: **6.02, 7.23, 5.72, 1.21 cm**. |
| **Table 3: denoising** | HumanEva and HPS; 30 Hz; 60-frame segments; Gaussian joint noise of 40 and 100 mm | HumanEva MPJPE: **18.88 / 34.68 mm**. HPS: **19.79 / 32.95 mm**. |
| **Tables 4–5: mesh recovery** | All 100 EHF images; mean-pose and CLIFF initialization | PA-MPJPE: **54.85 / 47.24 mm**; PA-MPVPE: **53.16 / 48.67 mm**; PCK@50: **65.29 / 71.40**. |
| **Table 6: ablation** | Three model variants × geodesic loss on/off × trajectory loss on/off; left-leg and legs completion | Reproduce all 12 configurations, including the full PoseRFM targets **83.81 / 95.00 mm**. |

### Generation

Compute pose FID from flattened SMPL joint positions, using fixed reference statistics. Reuse the NRDF evaluator for APD and nearest-neighbour distance, preserving its joint aggregation and quaternion-angle convention. Verify any approximate nearest-neighbour search against exact search on a small subset.

Record integration steps and actual network evaluations separately: midpoint generally requires two evaluations per step.

### Completion

Use baseline body-part masks and associated vertex subsets. Optimize with Euler, 100 steps, learning rate 0.1, blending coefficient 0.25, data weight 1, and trajectory weight 1e-3. Run 300 optimization iterations for arms and 200 otherwise.

Replace visible rotations with observations for the published evaluation. Report occluded-region MPVPE and APD, averaging across all ten hypotheses. Also report visible-region and full-body errors **before replacement**, following the supplementary diagnostic. Do not substitute best-of-ten error.

### Denoising

Verify the supplement's counts: **190 HumanEva segments** and **350 HPS segments**, selected as 50 per subject using seed 42.

Use Euler, 100 steps, learning rate 0.25, blending coefficient 0.75, and trajectory weight 1e-5. Use:

- HumanEva, 40 mm: 300 iterations.
- HumanEva, 100 mm: 400 iterations.
- HPS, either noise level: 400 iterations.
- Temporal weight 1e-1 at 100 mm and 1e-2 at 40 mm.

Reuse the baseline's initial joint-to-pose fitting procedure for backward-flow initialization; label it reconstructed if author behavior cannot be established. Share identical noisy observations across methods.

### Mesh recovery

Use ViTPose-H detections and the CLIFF checkpoint named in the supplement: `hr48-PA53.7_MJE91.4_MVE110.0_agora_val.pt`.

Follow SMPLify camera fitting, then optimize source pose and shape using confidence-weighted robust reprojection, joint-angle, shape, and trajectory losses. Use Euler, 100 steps, 400 iterations, learning rate 0.1, blending coefficient 0.75, and the supplement's regularizer weights of 50.

Reuse the baseline's camera conventions, robustifier, joint mapping, and angle penalty. Treat EHF as the quantitative benchmark and 3DPW as the qualitative comparison; any new quantitative 3DPW result is an extension.

### Baselines and ablations

Rerun DPoser-body and VPoser as principal controls using compatible released checkpoints. Keep all other published baseline rows in the comparison, visibly distinguishing copied results from rerun results.

For Table 6, reproduce PoseFM, PoseFM with geometry, and PoseRFM. If the middle variant's implementation remains unavailable, define it as the Euclidean-trained model with manifold-aware sampling and source optimization, and label it reconstructed. Use identical observations and hypothesis seeds across ablation conditions.

## 4. Validation, analysis, and acceptance

Before full runs, require:

- Rotation conversion, Exp/Log, tangent projection, and optimizer transport checks, including small-angle and near-pi cases.
- Finite-difference verification of gradients through a short ODE.
- Deterministic dataset manifests, disjoint splits, correct segment counts, masks, and units.
- Metric checks: identical poses produce zero reconstruction error; rigid transforms disappear under Procrustes alignment; APD behaves correctly for identical hypotheses.
- Agreement with imported baseline evaluators on fixed synthetic and real examples.

Run one small pilot per task before full evaluation. Capture peak VRAM, runtime, loss curves, rotation validity, and failure counts. Reduce evaluation batch size if necessary; use gradient accumulation to preserve the training batch size.

Train PoseRFM and PoseFM with seeds **0, 1, and 2**. For each checkpoint, retain the paper's sampling protocol. Separate uncertainty across generated batches, hypotheses, and training seeds. For paired method comparisons, bootstrap by source sequence for pose tasks and by image for EHF.

Predeclare these practical acceptance thresholds:

- FID: absolute difference at most **0.002**.
- Reconstruction errors, APD, and nearest-neighbour distance: relative difference at most **5%**.
- PCK: absolute difference at most **2 percentage points**.

These are reproduction tolerances, not equivalence proofs. Report every cell's signed deviation and protocol fidelity independently. Assess whether geometric components improve the intended conditions; do not require PoseRFM to outperform DPoser in every cell.

Keep test data out of hyperparameter selection. Report failures and exclusions; never silently remove difficult examples.

## 5. Execution order, resources, and deliverables

1. **Protocol and data gate:** freeze sources, defaults, splits, assets, and metric conventions.
2. **Implementation gate:** pass numerical checks and task pilots.
3. **Training:** finish one PoseRFM/PoseFM seed, then the remaining seeds.
4. **Evaluation:** generation → completion → denoising → EHF → geometric ablations → qualitative comparisons.
5. **Reporting:** regenerate Tables 1–6 and document matched, mismatched, reconstructed, and blocked cells.

The supplement reports **eight hours per model on one A30**. Six comparable training runs therefore suggest approximately **48 GPU-hours for training alone**. Treat this as a reference estimate; benchmark inference pilots before budgeting the substantially slower guided evaluations. Do not assume published A100 timings transfer directly to the chosen GPU.

Deliver a compact reproduction workspace containing the frozen protocol, configuration files, dataset manifests, checkpoints, per-example predictions, metric records, reconstructed tables, selected visualizations, and a deviation report. Separate reusable code/configuration from run outputs, with one recorded source for each setting.

Completion means every target cell has an evaluated result or a specific blocker, every reported number can be regenerated from saved artifacts, and all reconstruction choices remain visible. Supplementary climate experiments and retraining every historical baseline are outside this agreed main-results reproduction.
