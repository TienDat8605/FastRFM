# Guidance-aware manifold distillation

## Material Passport

Date: 2026-10-14; workflow: academic-research-suite / experiment-agent;
mode: run; status: exploratory implementation and benchmark. Novelty is
unestablished. Author decisions and interpretation remain with the researcher.

## What the RCM paper establishes

[Cheng et al., Riemannian Consistency Model, NeurIPS 2025](https://arxiv.org/abs/2510.00983)
learn manifold-respecting endpoint predictors. With a linear time schedule,
their parameterization on SO(3) can be written

\[
f_\theta(R_t,t)=R_t\operatorname{Exp}((1-t)v_\theta(R_t,t)),
\qquad f_\theta(R,1)=R.
\]

The consistency objective compares predictions from neighboring points of the
same teacher flow using geodesic distance. Its continuous formulation accounts
for both exponential-map differentials and covariant derivatives. For body
coordinates on SO(3), the connection includes the term
\(\tfrac12 u\times v\). This geometry is essential when tangent vectors belong
to different rotations. Their simplified loss makes an approximation to the
exponential-map differentials; it is not a universal exact identity on curved
manifolds. The paper also uses derivative warmup and tangent clipping.

The experiments include two-step generation on synthetic SO(3) distributions
and higher-dimensional tori. These support the feasibility of manifold
consistency modeling; they do not establish pose completion quality or a
particular T4 speedup. Appendix C.2's multistep sampler predicts an endpoint
and reintroduces noise through geodesic interpolation. It is not simply a
four-step Euler solver. Fresh sampling noise must be handled deliberately when
the map is differentiated with respect to its original source.

Local source: `papers/26_2025-10_Cheng_Riemannian_Consistency_Model.pdf`.
The PDF was read with PyMuPDF4LLM; equations lost in extracted Markdown were
cross-checked against the [arXiv HTML](https://arxiv.org/html/2510.00983v2).
Extraction: `results/acceleration_sources/riemannian_consistency_model.md`.

## Selected candidate

We choose a deterministic **four-step manifold endpoint-map student**. It uses
intrinsic Euler and is trained against the frozen 100-step PoseD-style teacher.
This is RCM-inspired endpoint compression, not a reproduction of the RCM loss
or sampler. The additional research hypothesis is that preserving derivatives
along task-relevant source directions will make the fast map more useful for
source optimization than endpoint matching alone.

The tested twist combines:

1. Source perturbations on SO(3)^21, rather than perturbing flattened matrices.
2. Levi-Civita transport between teacher/student endpoint tangent spaces.
3. A mixture of random source directions and directions obtained from the
   teacher's source gradient for observed-joint completion objectives.

Let \(Y_T=\Psi_T(X)\), \(Y_S=\Psi_S(X)\), and
\(X_\pm=X\operatorname{Exp}(\pm\epsilon U)\). Define per-joint responses

\[
b_T=\frac{\operatorname{Log}(Y_T^\top\Psi_T(X_+))-
             \operatorname{Log}(Y_T^\top\Psi_T(X_-))}{2\epsilon},
\]

with the analogous \(b_S\). These are finite directional approximations to
endpoint body velocities. With \(A=\operatorname{Log}(Y_S^\top Y_T)\),
parallel transport from teacher to student under the standard bi-invariant
metric is \(\operatorname{Ad}_{\operatorname{Exp}(A/2)}b_T\). In SO(3) vector
coordinates, the adjoint action is multiplication by that rotation matrix.

\[
\mathcal L=\mathbb E\frac1K\sum_j
 \|\operatorname{Log}(Y_{S,j}^\top Y_{T,j})\|^2
 +\lambda\mathbb E\frac1K\sum_j
 \|b_{S,j}-\operatorname{Exp}(A_j/2)b_{T,j}\|^2.
\]

For half the source directions, take the teacher gradient of the squared
observed-joint geodesic residual and normalize it per joint. Other directions
are random unit vectors. Teacher targets are detached. Student endpoint and
perturbed rollouts retain gradients. The response coefficient warms up over
100 updates. Cut-locus and finite-difference instability remain limitations;
the pilot checks teacher response differences at epsilon 0.01, 0.02, and 0.04.

## Why novelty remains a question

[Knowledge Transfer with Jacobian Matching](https://proceedings.mlr.press/v80/srinivas18a.html)
already develops derivative matching for distillation.
[Restoring Initial Noise Sensitivity in Text-to-Image Distillation via Geometric Alignment](https://arxiv.org/abs/2606.01651)
specifically addresses loss of source sensitivity after generative distillation.
Our research question concerns articulated manifold geometry, transported
responses, and task-directed sampling. We cannot claim to invent Jacobian
matching, sensitivity-preserving distillation, or SO(3) guidance.

Neither teacher endpoint matching nor this finite directional loss proves
preservation of the teacher Jacobian, its covariance interpretation, or
PoseD-Flow Theorem 4. The response is sampled locally in limited directions.
Moreover, the downstream optimizer is Adam with trajectory regularization,
not the theorem's infinitesimal unregularized gradient step.

## Benchmark contract

Run in the ML-study environment:

```bash
python scripts/run_distillation_pilot.py --device cpu \
  --output results/distillation_pilot_v2
```

Inputs: staged training poses, the frozen HMR checkpoint, official neutral SMPL,
and the previously selected eight real validation poses from the acceleration
pilot. Train/test roles remain separate; no AMASS or test poses enter selection.
The existing prior's earlier test exposure prevents a pristine-test claim.

Generate 512 training and 128 heldout sources from disjoint training pose
indices. Half the sources are Haar and half use backward initialization of
partial training observations. Across eight-row blocks, balance legs/torso
masks, source type, and random/task gradient directions independently.
Retain teacher pairs and provenance. Exclude the training mock sequence.

Compare the original 100-step teacher, untrained intrinsic four-step map, and
two identical four-step students: endpoint-only and transported-response.
Both students start from teacher weights, use identical minibatches and 600
updates, and select checkpoints using heldout endpoint error, including the
initial model as a candidate. The response student costs more training compute;
this is an equal-update comparison, not an equal-training-time comparison.

Evaluate heldout teacher-map endpoint and response errors. Then evaluate legs
completion on eight validation poses, two hypotheses and two inference seeds,
with the same teacher-based source initialization and 200 optimization iterations.
Report hidden-region MPVPE, APD, pre-clamping visible error, and actual autograd
task-gradient cosine relative to the teacher at those sources. Time solver and
end-to-end execution with synchronization; exclude but separately report data
generation and distillation training. CUDA runs report peak allocated/reserved
memory. CPU runs make no T4 latency or VRAM claim.

Use the original trajectory weight for the teacher and multiply it by 25 for
four-step students to approximate integrated penalty strength. The student
uses intrinsic intermediate states, while the teacher's intermediate matrices
are off-manifold; this remains an objective difference rather than exact
regularizer equivalence.

Outputs: `results/distillation_pilot_v2/manifest.json`, teacher pairs, selected
student checkpoints, `training.json`, per-case metrics/predictions, `results.json`,
and `report.md`. `results/distillation_pilot/` is an interrupted preliminary
setup and must not be used for conclusions. The baseline's trajectory-cost
configuration and independent source/mask balancing were corrected in v2.

Human check: inspect results and finite-difference stability before selecting
any follow-up. A credible contribution needs larger validation cohorts,
denoising, random-versus-task direction and transported-versus-untransported
ablations, additional training seeds, compute-matched comparisons, and repeated
T4 timing/peak-memory measurement. A null or adverse result remains valid.

## Observed screening outcome

The v2 run completed all eight evaluation cells and retained both selected
student checkpoints. On CPU, the teacher required 26.70 seconds per measured
16-hypothesis batch and had 106.08 mm hidden-region MPVPE. The endpoint-only
student required 1.96 seconds (13.62x) with 117.40 mm MPVPE. The transported
student required 1.86 seconds (14.39x) with 117.75 mm MPVPE. Timing variation
does not establish a speed difference between those identical four-step
architectures. Source task-gradient cosine was 0.899 for endpoint-only and
0.654 for transported-response, compared with 0.790 for the untrained four-step
map. The transported student's pre-clamping visible error was also worse.

The selected transported model reduced selection-cohort response MSE from
6.9513 to 4.3705 (37%); this is not independent heldout generalization evidence,
because the same cohort selected checkpoints by endpoint error. At epsilon
0.01 and 0.04, the teacher's finite response changed by about 4.7% and 9.9%
relative to epsilon 0.02 on a 32-source diagnostic subset. This check does not
cover all training or evaluation sources.

Conclusion within pilot scope: response matching improved its surrogate
criterion but did not improve downstream source guidance or completion.
Do not promote it or claim a positive contribution. The separate 20-step
solver screening remains more promising: 4.56x speedup and approximately
unchanged hidden-region MPVPE on these same eight poses. That screening still
requires larger validation, visible-fit checks, denoising, and T4 measurement.
