# Acceleration pilot

## Material Passport

Mode: run; date: 2026-10-14; evidence: exploratory executed experiment; source: run_acceleration_pilot.py; scope: validation legs completion.

Device: cpu; 8 poses, two hypotheses, 2 seeds. Times include shared 100-step initialization and body metrics. CPU measurements do not establish GPU speed or memory savings.

| Method | End-to-end s | Speedup | MPVPE mm | Change mm | Visible rad | APD cm | Saved payload MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 27.40 | 1.00x | 106.08 | +0.00 | 0.043 | 0.58 | 479.9 |
| checkpoint | 47.12 | 0.58x | 106.16 | +0.08 | 0.042 | 0.57 | 0.3 |
| steps20 | 6.01 | 4.56x | 105.65 | -0.43 | 0.055 | 0.73 | 96.1 |
| steps4 | 1.80 | 15.21x | 118.60 | +12.52 | 0.051 | 4.61 | 19.4 |
| iterations10 | 1.52 | 18.06x | 124.60 | +18.52 | 0.066 | 1.94 | 479.9 |
| local0.0 | 0.15 | 185.63x | 135.66 | +29.58 | 0.199 | 3.81 | nan |
| local0.1 | 0.15 | 186.47x | 133.29 | +27.21 | 0.164 | 3.53 | nan |
| local1.0 | 0.15 | 185.69x | 124.17 | +18.09 | 0.024 | 2.39 | nan |
| local10.0 | 0.15 | 186.56x | 130.21 | +24.13 | 0.244 | 3.48 | nan |

Saved payload counts tensor saves (including repeated references); it is not peak memory. Checkpoint correctness is checked separately by endpoint, penalty, and gradient parity. Few-step penalty weights preserve approximate time-integrated strength, not an identical objective. Local guidance uses an endpoint surrogate and changes the solver/guidance algorithm. These examples support screening only; validate chosen settings on larger development cohorts and denoising before test evaluation. See manifest.json and results.json for provenance and failures.
