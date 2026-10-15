# Task-directed transported-response distillation pilot

## Material Passport

Mode: run; date: 2026-10-14; status: exploratory executed experiment; scope: four-step deterministic endpoint-map compression, validation legs completion.

Device cpu; one distillation seed; 512 training and 128 heldout training-only sources; eight validation poses, two hypotheses, two inference seeds. CPU results do not establish T4 speed or VRAM savings.

| Method | End-to-end s | Speedup | Hidden MPVPE mm | Visible rad | APD cm | Source gradient cosine |
|---|---:|---:|---:|---:|---:|---:|
| teacher100 | 26.70 | 1.00x | 106.08 | 0.043 | 0.58 | 1.000 |
| raw4 | 1.98 | 13.51x | 116.71 | 0.061 | 1.29 | 0.790 |
| endpoint4 | 1.96 | 13.62x | 117.40 | 0.055 | 0.94 | 0.899 |
| transport4 | 1.86 | 14.39x | 117.75 | 0.104 | 1.60 | 0.654 |

## Teacher-map fidelity on heldout training sources

| Student | Mean endpoint rad | Transported response MSE | Selected update | Training s |
|---|---:|---:|---:|---:|
| raw4 | 0.2048 | 6.951333 | 0 | 0.00 |
| endpoint4 | 0.1682 | 5.682239 | 500 | 12.22 |
| transport4 | 0.1935 | 4.370491 | 300 | 31.32 |

Teacher-pair generation: 11.89 seconds. Training and generation are excluded from inference timing and must be amortized over use.

The transported-response candidate is an extension inspired by RCM, established Jacobian matching, and noise-sensitivity distillation. Novelty is unestablished. It does not inherit the teacher's covariance interpretation. Finite-difference stability is in training.json. Endpoint-only and response-aware students share data, initialization, sampling, updates, and endpoint-based checkpoint selection; the response-aware model has higher training compute. Four-step students use intrinsic Euler and a quadrature-rescaled trajectory weight; their trajectory objectives differ from the teacher. Tests evaluate transport and finite derivatives. No test cohort or AMASS poses are used here. Expand validation and evaluate denoising and GPU memory before promoting any candidate.

The fidelity cohort also selected checkpoints by endpoint loss and is not an independent generalization cohort. Source gradient cosine measures task gradients only. The response-matching variant did not improve downstream guidance in this pilot.
