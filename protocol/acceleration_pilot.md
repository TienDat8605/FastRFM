# PoseD-style inference acceleration pilot

## Material Passport

Mode: run; date: 2026-10-14; stage: exploratory screening; empirical claims
refer only to saved pilot artifacts. Full-paper reproduction remains separate.

## Inputs and outputs

Inputs: staged 3DPW train/validation arrays, frozen HMR epoch-38 checkpoint,
official neutral SMPL, and the study geometry implementation. AMASS and 3DPW
test poses are not used to select acceleration settings.

Run `python scripts/run_acceleration_pilot.py --device cpu` in the ML-study
environment. Results, predictions, immutable run settings, and the summary are
written under `results/acceleration_pilot/`. `scripts/audit_pose_arrays.py`
records the input inventory and exports the original real 3DPW training poses
to `data/pose_arrays/3dpw_train_smpl72.npz`, with sequence/person/frame provenance.
The training folder contains one ten-frame mock sequence; this pilot excludes
it from mean-pose estimation and exports. The previously frozen full-study
bundle includes it; that distinction must remain visible in interpretation.

## Screening design

Eight sequence-balanced validation poses, two hypotheses per pose, seeds 0 and 1,
legs mask, neutral zero-shape geometry, fixed checkpoint, fixed backward source
initialization, and 200 optimization iterations form the baseline. This is a
small completion pilot, not evidence for motion denoising or general performance.

Candidates:

- Euler with 20 or four steps, retaining 200 source optimization iterations.
  Because the original trajectory cost sums over steps, set
  `lambda_N=lambda_100*100/N`. This approximates matching integrated penalty
  strength; it does not make the different discrete objectives identical.
- Ten-step activation-checkpoint blocks with the original 100-step discrete
  solver. Check endpoint, trajectory cost, and source gradient parity in tests.
  This trades recomputation for fewer saved activations; it need not be faster.
- Ten source optimization iterations with the original 100-step solver.
- One 20-step intrinsic rollout with local endpoint-surrogate guidance at
  strengths 0, 0.1, 1, and 10. Each local graph is discarded. This heuristic
  changes both solver and guidance and is not an implementation of OC-Flow.

Record solver and end-to-end time (including backward initialization and final
body metrics), masked MPVPE/MPJPE, pre-replacement visible-joint error, APD,
failures, per-case predictions, and saved tensor payload. CPU timings do not
predict GPU speedups. Saved payload counts repeated saves and is not peak VRAM.
CUDA execution additionally records synchronized timing; peak memory needs
separate reporting before any GPU memory claim. Method order is reversed for
the second seed to reduce startup/order bias; timings have no repeated-run CI.

Before promotion: rerun promising candidates on larger development cohorts,
both completion masks and 40/100 mm denoising, with synchronized repeated GPU
timing, peak allocated/reserved memory, failure rates, and unchanged controls.
Use validation for selection; evaluate test once after the settings are frozen.
No pretrained learned-prior plausibility metric is available in this pilot.

## Source audit and longer-term ideas

The PoseD-Flow main paper specifies four hidden layers of width 512, 189 rotation
coordinates plus scalar time, and 50,000 steps at batch 4096. Its supplement
reports approximately 18 million AMASS poses and eight hours on one A30.
Training on a small subset or synthetic teacher samples is an alternative
prior-training study; it does not reproduce the original data distribution.
Retain HumanEva for evaluation rather than training if its denoising protocol
will later be reported. Raw AMASS here is SMPL-H-156, not SMPL-72: PoseRFM uses
coordinates 3:66. The first 72 SMPL-H coordinates are not a valid SMPL-72 layout.

[Riemannian Consistency Model](https://arxiv.org/abs/2510.00983) demonstrates
few-step generation on SO(3). A useful follow-up is a two/four-step student
trained from this frozen teacher. Evaluate teacher endpoint error and source
Jacobian-vector agreement as well as recovery error: matching endpoints alone
does not preserve teacher guidance derivatives. PoseD-Flow Theorem 4 is an
infinitesimal source-gradient chain-rule statement; a different map has a
different Jacobian, and Adam with trajectory cost is not that exact update.

[OC-Flow](https://arxiv.org/abs/2410.18070) already addresses SO(3). Research
novelty should concern articulated SO(3)^21 objectives or efficient discrete
adjoints, not the first SO(3) extension. Continuous adjoints save activation
storage but do not guarantee faster execution or identical discrete gradients.
Running trajectory costs require adjoint forcing. Stopping the backward solve
at t=0.3 saves roughly 30% of backward steps, and leaves the source gradient
uncomputed without further approximation.

Amortized source initialization is another follow-up: train on training-only
observation/source pairs, compare against the current backward-flow initializer,
include data-generation/training cost, and benchmark zero/five/ten refinement
iterations. A deterministic initializer can collapse completion diversity;
condition it on a hypothesis seed and retain APD and recovery metrics.

Human check: inspect the speed/error tradeoff and failed cells before selecting
a candidate for the larger validation experiment. Unmeasured speedup factors
from the earlier blueprint are hypotheses, not results.
