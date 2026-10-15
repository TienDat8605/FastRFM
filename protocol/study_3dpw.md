# PoseD-style recovery with a 3DPW prior

Status: implementation; empirical conclusions require complete controlled runs.
This is a transfer study, not reproduction of published Tables 1–6.

## Hypotheses and scope

- H1: frozen-prior Riemannian D-Flow improves completion/denoising over initialization and no-prior fitting.
- H2: backward initialization and trajectory regularization contribute measurable gains.
- H3: limited clean-pose fine-tuning improves recovery under unchanged guidance.

The original HMR epoch-38 residual/time-embedding prior is retained. Fine-tuning
uses 3DPW train only; all AMASS collections remain reserved for future transfer
evaluation. Earlier HMR experiments exposed test data, so this study does not
claim a pristine previously unseen test set. The eight environment-pilot poses
are excluded from final cohorts.

## Inputs and fidelity

`configs/study_3dpw.yaml` is the canonical experiment configuration. Counts:
development 48 poses/12 segments; final 240/60; smoke 2/1. All segments are
60 consecutive frames at 30 Hz. Pose sampling is balanced by sequence.

Official neutral SMPL produces both ground truth and predictions at zero shape,
root orientation, and translation. Denoising applies 40/100 mm independent
coordinate noise to canonical joints reconstructed from 3DPW rotations; it is
not image-based HMR or the original HumanEva protocol.

Completion uses DPoser-style masks, including torso `[2,5,8,15,16]` rather than
the earlier HPE four-spine-joint mask. Primary error is MPVPE on vertices whose
dominant SMPL skinning joint belongs to the mask. APD uses masked joints.
Both region choices are explicit reconstruction conventions. Visible rotations
are replaced only for primary metrics; pre-replacement consistency is recorded.

Published inference hyperparameters are retained. Extrinsic Euler with final
SO(3) projection is primary; intrinsic Euler is an optional diagnostic. Polar
projection uses its actual off-manifold differential, not an identity-singular-
value tangent approximation. Forward trajectory penalty sums post-step states
(initial state excluded); its weight is applied after averaging over examples.

The optimizer uses the bi-invariant metric with body-coordinate Euclidean norm,
one scalar second moment per SO(3) joint factor, and analytic parallel transport
of momentum. Factorization is a reconstruction choice. Denoising loss averages
squared joint residual sums over frames plus mean consecutive-pose geodesic sums.
Temporal acceleration error uses 30 Hz and has units mm/s².

No-prior completion fitting has no signal for missing rotations and can equal
mean filling; report this fact rather than implying it is a strong learned
conditional control. Denoising direct fitting uses the same deterministic
joint-to-pose initialization as the guided methods and matched main iterations.

## Runs and artifacts

```bash
# Numerical/body-model audit and small real-setting validation
bash scripts/run_3dpw_colab.sh
# Representative-batch throughput and resource forecast
STUDY_COMMAND=benchmark bash scripts/run_3dpw_colab.sh
# Frozen-prior validation matrix
STUDY_COMMAND=evaluate bash scripts/run_3dpw_colab.sh
# Validation-gated adaptation only
STUDY_COMMAND=finetune bash scripts/run_3dpw_colab.sh
# Complete pipeline within remaining GPU budget
STUDY_COMMAND=all bash scripts/run_3dpw_colab.sh
```

The Bash runner restores expired sessions, synchronizes code separately from
data, logs timestamps, waits for subprocess exit without reading remote logs,
and retrieves checksummed artifacts. No CLI execution deadline options are used.
Remote output lives in `/content/posedflow/study/output/3dpw_study_v4`; downloads
live under `results/3dpw_study/<run-id>/study_results`.

Immutable config/manifests and implementation hashes gate resume. After changing
study code, set `STUDY_OUTPUT=3dpw_study_v5` to preserve prior artifacts and start
a new version. Per-batch JSON and predictions support
partial resumption. `audit.json` identifies source/checkpoint/body hashes.
`state.json` records completion, failure, or budget exhaustion. Budget covers
study-process elapsed time, including idle evaluation overhead; upload/setup
cost is separate. Budget checks happen between cells/training steps, allowing
the current cell to finish. Eight hours is a cap, not a runtime guarantee.
The complete pipeline benchmarks real batch sizes before launching the full
matrix and exits with `budget_insufficient` if its original-prior forecast does
not fit; fine-tuning costs can require additional budget even when that forecast
fits. It never silently changes cohort sizes, hypotheses, seeds, or ODE settings.

Fine-tuning requires complete successful development controls. A relative gain
below 5% over the best simple/direct control for either task triggers up to
4,000 updates for three seeds. Validation flow loss selects EMA checkpoints;
the initial model remains a candidate. No test outcome selects settings.

Reports use sequence means, paired case/seed comparisons, and sequence bootstrap
intervals. These intervals quantify evaluation variability, not uncertainty
over independent pretraining runs. Failed cells remain visible. A null finding
is acceptable, and a partial budget-limited matrix is never a completed study.

Evaluation batches up to 128 completion poses or 32 denoising segments on T4.
Random sources and noise remain keyed by example, not batch. Temporal losses
never bridge segment boundaries. Source gradients undo the example average
before Adam so its epsilon does not change scale with batch size. Denoising
uses the official SMPL rigid transform for its first 22 skeletal joints;
forward values and tangent gradients are audited against the full body model.
