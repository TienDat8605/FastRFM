# Exact-pullback screening evidence

Run state: failed. Confirmation complete: True.

No completed promotion decision is available.

Development summary rows use the selected checkpoint only; measurements.json retains all checkpoint evaluations in chronological order.

Derivative fidelity is audited independently on 1,024 cached source points; each audit includes two random and two task covectors. These sampled checks do not prove covariance preservation.

| Student | Normalized VJP MSE | Mean VJP cosine |
|---|---:|---:|
| endpoint_0_seed0 | 0.00782326 | 0.997876 |
| random_0.001_seed0 | 0.00779707 | 0.99788 |
| random_0.01_seed0 | 0.00759326 | 0.997909 |
| random_0.1_seed0 | 0.00634766 | 0.998087 |
| task_0.001_seed0 | 0.00779591 | 0.99788 |
| task_0.001_seed1 | 0.00806953 | 0.997905 |
| task_0.01_seed0 | 0.00756937 | 0.997913 |
| task_0.1_seed0 | 0.00659121 | 0.998055 |

Compare `development` with `zero_trajectory` rows in summary.csv to assess running-cost sensitivity. Checkpoints were selected solely using development hidden MPVPE; confirmation outcomes must not be used to retune coefficients.

Timing is per 64-hypothesis optimization batch, including all configured outer iterations, rather than isolated network latency. CUDA allocated and reserved memory are reported separately. The frozen prior has only the original pilot training, and this validation study cannot establish the published AMASS protocol or broader generalization.
