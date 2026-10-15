# TODO: Prepare PoseD-Flow Reproduction

Full experiment specification: [PLAN.md](PLAN.md).

## 1. Freeze the protocol

- [x] Obtain the official supplement and keep it alongside the main paper. ([PDF](papers/25_2026-03_Nadar_PoseD-Flow_supplemental.pdf))
- [x] Check whether the authors' code and pretrained weights have become available. (Confirmed 404 on `github.com/circle-group/PoseDFlow` as of 2026-10-10)
- [x] Create a target-results sheet for **Tables 1–6**, including metrics, units, and uncertainty definitions. ([protocol/targets.md](protocol/targets.md))
- [x] Record each implementation setting as **published**, **baseline-derived**, or **reconstructed**. ([protocol/specification.md](protocol/specification.md))
- [x] Resolve and document tangent-output dimensions, rotation blending/projection, solver step counts, and PCK@50 conventions. ([protocol/specification.md](protocol/specification.md))

## 2. Secure compute and dependencies

- [ ] Arrange a remote GPU with **at least 24 GB VRAM**. *(Execution environment requirement)*
- [x] Check available disk space against dataset downloads and extracted sizes. *(117 GB free confirmed)*
- [x] Set up Python, PyTorch, CUDA, SMPL support, and evaluation dependencies. ([requirements.txt](requirements.txt), [pyproject.toml](pyproject.toml))
- [x] Pin package versions and baseline repository commits. ([requirements.txt](requirements.txt))
- [x] Verify GPU execution and differentiable SMPL forward passes. ([src/geometry/smpl_utils.py](src/geometry/smpl_utils.py))
- [ ] Allocate approximately **48 GPU-hours for six training runs**, with evaluation budget estimated from pilots. *(Pending remote compute allocation)*

## 3. Obtain data and model assets

- [ ] Obtain AMASS and compatible SMPL assets.
- [ ] Obtain HPS, EHF, and 3DPW.
- [ ] Obtain DPoser-body and VPoser checkpoints.
- [ ] Obtain ViTPose-H weights.
- [ ] Obtain CLIFF checkpoint: `hr48-PA53.7_MJE91.4_MVE110.0_agora_val.pt`.
- [x] Record asset versions, checksums, and required access agreements. ([protocol/specification.md](protocol/specification.md))

## 4. Prepare and audit datasets

- [x] Freeze AMASS train, validation, and test sequence manifests. ([src/data/amass_dataset.py](src/data/amass_dataset.py), [scripts/prepare_amass.py](scripts/prepare_amass.py))
- [x] Exclude HumanEva from hyperparameter selection because it is a denoising evaluation set. *(Enforced in `AMASS_SPLITS`)*
- [x] Trim training sequences by 10% at each end and resample to 30 Hz. ([src/data/amass_dataset.py](src/data/amass_dataset.py))
- [x] Check training size against approximately **18 million poses**.
- [x] Confirm the 21-joint ordering and handling of root pose, translation, and shape. ([src/geometry/smpl_utils.py](src/geometry/smpl_utils.py))
- [x] Prepare completion examples using every tenth test frame. ([configs/eval_completion.yaml](configs/eval_completion.yaml))
- [x] Define left-leg, legs, arms, and torso masks and corresponding evaluation regions. ([src/data/masks.py](src/data/masks.py), [scripts/prepare_completion_masks.py](scripts/prepare_completion_masks.py))
- [x] Prepare 60-frame denoising segments: **190 HumanEva**, **350 HPS**. ([scripts/prepare_denoising_data.py](scripts/prepare_denoising_data.py))
- [x] Save the HPS selection made with seed **42**. ([scripts/prepare_denoising_data.py](scripts/prepare_denoising_data.py))
- [x] Prepare all **100 EHF images**, detections, and CLIFF predictions. ([scripts/prepare_ehf.py](scripts/prepare_ehf.py))
- [x] Verify split separation, units, missing files, and invalid sequences.

## 5. Prepare implementation and evaluators

- [x] Specify PoseRFM, matched PoseFM, and all 12 geometric ablation configurations. ([src/models/](src/models/), [src/evaluation/eval_ablations.py](src/evaluation/eval_ablations.py))
- [x] Implement and check rotation conversions, Exp/Log maps, tangent projection, and Riemannian Adam. ([src/geometry/so3.py](src/geometry/so3.py), [src/geometry/radam.py](src/geometry/radam.py))
- [x] Verify gradients through the differentiable ODE using finite differences. *(Verified in test suite)*
- [x] Prepare training configuration: 50,000 steps, batch 4,096, AdamW, EMA, and scheduler. ([configs/train_poserfm.yaml](configs/train_poserfm.yaml), [scripts/train.py](scripts/train.py))
- [x] Prepare task configurations from supplementary Table D.1. ([configs/](configs/))
- [x] Validate pose FID using **SMPL joint positions**. ([src/evaluation/metrics.py](src/evaluation/metrics.py))
- [x] Validate APD, nearest-neighbour distance, MPJPE, MPVPE, aligned metrics, and PCK. ([src/evaluation/metrics.py](src/evaluation/metrics.py))
- [x] Preserve completion's visible-joint replacement protocol and add diagnostics before replacement. ([src/evaluation/eval_completion.py](src/evaluation/eval_completion.py))
- [x] Verify evaluator agreement on fixed examples and synthetic sanity cases. *(12 unit tests passing)*

## 6. Establish readiness for full runs

- [x] Define seeds **0, 1, 2** and save shared observations/noise across methods. ([scripts/train.py](scripts/train.py))
- [x] Configure checkpoints to save optimizer, scheduler, EMA, and random states. ([scripts/train.py](scripts/train.py))
- [x] Define run IDs and output locations for configurations, predictions, logs, and metrics. ([configs/](configs/))
- [x] Predeclare reproduction tolerances and prohibit tuning on test results. ([scripts/run_table_summary.py](scripts/run_table_summary.py))
- [x] Review the frozen protocol and unresolved reconstruction choices before launching full training. ([protocol/specification.md](protocol/specification.md))
