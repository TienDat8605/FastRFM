# Colab GPU inference improvement PoC

## Material Passport

Mode: run; date: 2026-10-14; stage: exploratory; outputs: this report, manifest, checkpoints, predictions, timing and gradient/memory JSON records.

GPU: Tesla T4; real train=23543, validation=10858; prior trained for 1000 updates. Eight validation poses, two hypotheses, two masks, two inference seeds, 20 optimization iterations, 2 timing repetitions. No test data used.

| Method | Solver s/batch | Speedup | Peak allocated MiB | Hidden MPVPE mm | Visible rad | APD cm |
|---|---:|---:|---:|---:|---:|---:|
| euler100 | 9.067 | 1.00x | 98.1 | 102.08 | 0.198 | 3.65 |
| euler50 | 4.554 | 1.99x | 87.6 | 101.93 | 0.198 | 3.60 |
| adjoint100 | 32.260 | 0.28x | 77.3 | 102.08 | 0.198 | 3.65 |
| adjoint50 | 19.470 | 0.47x | 77.3 | 101.93 | 0.198 | 3.60 |
| midpoint2 | 0.518 | 17.52x | 77.9 | 127.16 | 0.211 | 9.40 |
| midpoint4 | 0.832 | 10.90x | 78.8 | 110.34 | 0.200 | 5.27 |
| student2 | 0.501 | 18.11x | 77.9 | 130.76 | 0.212 | 10.40 |
| student4 | 0.841 | 10.79x | 78.8 | 110.34 | 0.200 | 5.27 |

## Memory and gradient audit

| Steps | Backend | Backward-probe s | Incremental peak MiB | Gradient relative error |
|---|---|---:|---:|---:|
| 25 | full_graph | 0.131 | 9.87 | 0.000000 |
| 25 | adjoint | 0.914 | 1.06 | 0.000014 |
| 50 | full_graph | 0.212 | 1.48 | 0.000000 |
| 50 | adjoint | 0.957 | 0.06 | 0.000027 |
| 100 | full_graph | 0.420 | 2.95 | 0.000000 |
| 100 | adjoint | 1.467 | 0.06 | 0.000046 |
| 200 | full_graph | 0.842 | 5.88 | 0.000000 |
| 200 | adjoint | 2.680 | 0.06 | 0.000042 |

The adjoint stores at most five fixed anchors and one local graph; O(1) is with respect to ODE depth for fixed model and batch. It reconstructs prior states to a tolerance and uses prefix replay on failed inverse residual checks. Inverse/replay work can make it slower; see reconstruction counters. It includes the running-cost adjoint forcing. It is not OC-Flow or an exact continuous adjoint.

Student midpoint steps require two network evaluations each. Endpoint distillation does not preserve teacher guidance derivatives or covariance guarantees. Short prior training, small validation samples, changing trajectory discretization, and absence of a learned plausibility metric limit interpretation. Solver times exclude body metrics; peak memory includes resident models and the official body. Training/generation costs are recorded in training.json. The distillation selection cohort is not independent generalization evidence.
