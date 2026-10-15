# Inference research contract

Inputs: the PoC entry point, frozen or pilot-trained pose priors, training-only
pose arrays, sequence-disjoint validation arrays, and official neutral SMPL.
Read `protocol/riemannian_consistency_distillation.md` for previous CPU screening;
the Colab GPU implementation is `colab_benchmark.py`.

Process: train/compress endpoint maps; validate adjoint gradients against the
same discrete solver including running costs; measure paired recovery, GPU
time, and live allocated/reserved memory. Keep solver and guidance conventions
explicit. Novelty and continuous-adjoint equivalence are not assumed.

Outputs: `results/distillation_pilot_v2/` for prior CPU work and
`results/inference_colab/<run-id>/` for downloaded GPU runs. Remote GPU artifacts
live at `/content/posedflow-inference/jobs/<run-id>/results/`.

Human check: assess small-cohort/short-training limits, gradient gates, inverse
replay counters, speed/error tradeoffs, and missing task/method ablations before
promoting a candidate. AMASS test collections remain reserved.

Next GPU screen: `pullback_screen.py` and `protocol/pullback_screen.md` implement
exact transported VJP distillation with sequence-disjoint locked confirmation.
Its outputs live in `results/pullback_screen/`; existing pilot modes are preserved.
