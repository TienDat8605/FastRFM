# Colab SSH inference acceleration

## Material Passport

Date: 2026-10-14; mode: run; stage: exploratory PoC. Prior CPU screening is in
`protocol/acceleration_pilot.md` and `protocol/riemannian_consistency_distillation.md`.
Neither measured results nor algorithmic novelty are assumed before completion.

## Runtime and inputs

Session: `posedflow-inference`, fresh Tesla T4 (15 GiB). Execute through SSH
using the dedicated Colab key and named-host config. User requested deletion
of the previous `posedflow-study` session; its partial output archive is at
`results/3dpw_study/20261014T014832Z-all/recovered/partial_study.tar.gz`.
Training uses 23,543 real 3DPW SMPL-72 frames, slicing root-inclusive parameters
to body coordinates 3:66. Validation uses 10,858 body-63 frames from disjoint
sequences. The training mock sequence is excluded. AMASS test collections and
3DPW test poses are not used. Original neutral SMPL evaluates canonical zero
shape/root/translation recovery; this is not image-based HMR.

## Executable path

`posed_flow_colab_poc.py --mode inference` calls
`src/acceleration/colab_benchmark.py`. Run on the GPU host:

```bash
cd /content/posedflow-inference
python posed_flow_colab_poc.py --mode inference \
  --data_path data/3dpw_train_smpl72.npz \
  --validation_path data/validation.npz --model_root . \
  --num_joints 21 --steps 1000 --distill_steps 500 \
  --opt_iters 20 --repeats 2 --output_dir jobs/RUN_ID/results
```

`scripts/run_inference_colab_job.sh` installs SMPL dependencies, runs 21 targeted
tests, and executes this command as a logged subprocess. `exit_code` and
`results/state.json` distinguish failure, progress, and completed screening.
Local run IDs are indexed by `results/inference_colab/latest_run_id`.

## Experiment matrix

Train one four-hidden-layer 512-unit PoseRFM with 190 inputs and 63 tangent
outputs for 1,000 updates (batch 256, EMA). This is a small-data short-training
prior, not a reproduction of 50k updates/batch 4096/AMASS training.

Generate 640 paired teacher source/endpoints, reserving 128 sources for student
checkpoint selection. Source preparation uses training poses only. Distill
separate two/four-step intrinsic midpoint students for 500 updates; use identical
data/minibatch RNG and select by teacher endpoint geodesic loss. The initial
student remains a candidate. These are endpoint-map shortcuts, not RCM's
continuous consistency loss or its re-noised multistep sampler. Each midpoint
step uses two network evaluations. Endpoint matching alone does not preserve
teacher source derivatives or covariance guarantees.

Compare eight methods: full-graph Euler 50/100; discrete adjoint Euler 50/100;
untrained midpoint 2/4; distilled midpoint 2/4. Eight balanced validation poses,
two hypotheses, legs/torso masks, two inference seeds, 20 optimization iterations,
and two synchronized timing repeats produce 32 paired cells and 64 measured
optimization batches. Shared Haar source hypotheses isolate algorithm changes;
no fast-only warm start is allowed. The trajectory penalty is summed over
post-step states, with weight `.001*100/N` for approximate quadrature parity.
Different solvers still have different discrete objectives.

## Adjoint mechanism and acceptance

`src/acceleration/adjoint.py` stores at most five fixed anchor rotations, reconstructs
the preceding state by fixed-point inversion, and computes one step's local
VJP. Include running-cost forcing at every reverse step. Reject failed inverse
residuals and replay the corresponding prefix without retaining a trajectory.
Stored graph/state memory is bounded independently of solver depth for a fixed
model and batch, but reconstruction/replay increases computation. This is a
tolerance-limited discrete adjoint, not OC-Flow or exact continuous dynamics.

The initial unanchored GPU audit exceeded the 1% gradient-error gate at N=100
(1.24% relative error). Its artifacts remain preserved. The revised implementation
resets inverse reconstruction at four fixed intervals and reruns the original
gate without relaxing it. Compatible prior/student training artifacts are reused
through `--reuse_training_dir`; input/training settings must match their manifest.

Before recovery, compare gradients to the same full discrete solver at 25, 50,
100, and 200 steps. Require relative gradient error <=1% and cosine >=0.999.
Record local inversion attempts, fallbacks, source reconstruction residual,
synchronized wall time, peak allocated/reserved GPU memory, and allocated peak
increment over resident tensors. Tests include running-cost gradients and forced
replay. An adjoint storage claim needs the GPU memory scaling audit; it does not
imply faster inference.

## Outputs and interpretation

Remote: `/content/posedflow-inference/jobs/<run-id>/results/`. Download to
`results/inference_colab/<run-id>/` via SSH/SCP. Retain input/code fingerprints,
manifest, prior and student checkpoints, paired teacher targets, per-case
predictions, timing repeats, MPVPE/MPJPE/APD, visible fit before clamping, inverse
counters, gradient/memory records, and report. Solver time excludes body metrics;
batch throughput must not be called single-pose latency.

Human check: inspect completed/failed cells and measured speed/error/memory
tradeoffs. One training seed, eight poses, short prior training, missing learned
plausibility metrics, and selection-cohort endpoint errors limit conclusions.
Expand validation, training seeds, optimizer budgets, and denoising before making
publication-level claims. A null or slower adjoint result remains valid evidence.

## Completed GPU screening

Run `20261014T024059Z` completed all 32 cells (64 timed optimization batches)
with exit code zero. Artifact checksums and executed source hashes were verified
locally. Twenty-one targeted tests passed, as did all 12 additional task-gradient
checks on paired Haar source hypotheses. The latter's maximum relative gradient
error was 0.00000196. These checks sample two validation poses and two hypotheses
per pose, both masks/seeds, and 50/100/200 solver steps.

Pooled results: Euler 100 required 9.067 seconds per optimization batch and had
102.08 mm hidden-region MPVPE. Euler 50 required 4.554 seconds with 101.93 mm.
Four-step midpoint required 0.832 seconds (10.90x faster) with 110.34 mm.
Two-step distilled midpoint required 0.501 seconds (18.11x faster) with 130.76 mm.
The four-step student selected its initial weights; its distillation did not
improve selection-cohort endpoint loss, and it equals the midpoint-four control.
Endpoint-only compression has not shown a quality-preserving two/four-step gain.

Anchored adjoint 100 matched the teacher's recovery but required 32.260 seconds
(3.56x slower). Additional peak allocated memory during the 16-hypothesis
optimization batch decreased from 21.26 MiB to 0.42 MiB. Resident tensors reduce
the total allocated-memory benefit: peak 98.1 MiB versus 77.3 MiB. In warmed
two-pose backward probes, full-graph additional memory grew from 1.48 to 5.88 MiB
over 50 to 200 steps; adjoint additional memory stayed about 0.059 MiB. The first
25-step readings include one-time allocations and are not steady-state estimates.
Source restoration is exact at an anchor by construction; pre-reset
`max_anchor_drift` is the informative reconstruction diagnostic. Prefix fallback
replays from the nearest earlier saved anchor.

Current screening takeaway: Euler 50 offers a practical twofold speedup with
similar pilot accuracy. Midpoint four offers a larger speed/error tradeoff.
The adjoint achieves bounded trajectory storage at increased time. Treat these
as small-data completion results, not denoising or full-paper claims.

Local outputs:

- [Overall GPU report](../results/inference_colab/20261014T024059Z/results/report.md)
- [Per-mask report](../results/inference_colab/20261014T024059Z/results/report_by_mask.md)
- [Speed/quality plot](../results/inference_colab/20261014T024059Z/results/speed_quality.pdf)
- [Memory scaling plot](../results/inference_colab/20261014T024059Z/results/memory_scaling.pdf)
- [Per-mask CSV](../results/inference_colab/20261014T024059Z/results/summary_by_mask.csv)
