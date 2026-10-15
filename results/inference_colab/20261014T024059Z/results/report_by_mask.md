# Results by completion mask

Eight validation poses, two hypotheses, two inference seeds; two timing repeats. Changes are paired against the 100-step full-graph teacher within each mask.

| Mask | Method | Solver s/batch | Speedup | MPVPE mm | Change mm | Visible rad | Incremental peak MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| legs | euler100 | 9.084 | 1.00x | 126.29 | +0.00 | 0.204 | 21.26 |
| legs | euler50 | 4.552 | 2.00x | 126.24 | -0.05 | 0.204 | 10.73 |
| legs | adjoint100 | 32.300 | 0.28x | 126.29 | -0.00 | 0.204 | 0.42 |
| legs | adjoint50 | 19.612 | 0.46x | 126.24 | -0.05 | 0.204 | 0.42 |
| legs | midpoint2 | 0.525 | 17.31x | 150.72 | +24.43 | 0.215 | 1.05 |
| legs | midpoint4 | 0.831 | 10.93x | 134.05 | +7.76 | 0.206 | 1.90 |
| legs | student2 | 0.501 | 18.13x | 151.36 | +25.07 | 0.217 | 1.05 |
| legs | student4 | 0.838 | 10.84x | 134.05 | +7.76 | 0.206 | 1.90 |
| torso | euler100 | 9.051 | 1.00x | 77.87 | +0.00 | 0.192 | 21.26 |
| torso | euler50 | 4.555 | 1.99x | 77.63 | -0.24 | 0.192 | 10.73 |
| torso | adjoint100 | 32.220 | 0.28x | 77.87 | -0.00 | 0.192 | 0.42 |
| torso | adjoint50 | 19.327 | 0.47x | 77.63 | -0.24 | 0.192 | 0.42 |
| torso | midpoint2 | 0.510 | 17.74x | 103.61 | +25.74 | 0.207 | 1.05 |
| torso | midpoint4 | 0.832 | 10.87x | 86.63 | +8.76 | 0.194 | 1.90 |
| torso | student2 | 0.500 | 18.09x | 110.17 | +32.30 | 0.208 | 1.05 |
| torso | student4 | 0.843 | 10.74x | 86.63 | +8.76 | 0.194 | 1.90 |

The four-step student selected update zero; it is identical to the undistilled four-step midpoint control. Distillation did not improve its selection loss.

Actual Haar-source task-gradient audit: 12/12 checks passed; maximum relative gradient error 0.00000196. These checks use the first two validation poses, two hypotheses each, both masks/seeds, and 50/100/200 steps.

Adjoint gradient/memory probes use two validation poses as initial states. The first 25-step incremental memory readings include one-time allocations; use warmed 50/100/200-step measurements to assess solver-depth scaling. Restoring the source anchor makes its final residual zero; max_anchor_drift records pre-reset drift. Timing and quality are exploratory, not population confidence intervals.
