# 3DPW study contract

Inputs: `configs/study_3dpw.yaml`, packed 3DPW train/validation/test arrays,
the original HMR PoseRFM implementation/checkpoint, and official neutral SMPL.

Process: audit geometry and provenance; freeze cohorts; evaluate frozen guidance;
use validation outcomes to trigger limited fine-tuning; evaluate selected priors
on the frozen test cohort and aggregate paired sequence-level results.

Outputs: immutable manifests, per-case metrics and predictions, audit records,
fine-tuning checkpoints, timings, failures, and a regenerated report under
`results/3dpw_study/` (remote equivalent `/content/posedflow/study/output/3dpw_study`).

Human check: assess fidelity and reconstruction choices, controls, uncertainty,
and incomplete cells before interpreting any method gain. The eight-pose pilot
is environment evidence only. AMASS is reserved for later transfer evaluation.
