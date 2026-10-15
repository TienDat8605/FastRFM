# Exact-pullback four-step screen

Entrypoint: `posed_flow_colab_poc.py --mode pullback_screen`. Implementation:
`src/acceleration/pullback_screen.py`; Colab launcher:
`scripts/run_pullback_screen_colab.sh`. Use SSH to transfer and execute.

Freeze the recovered pilot teacher. Shuffle twelve validation sequences with
seed 42: eight development, four confirmation; select 32 balanced poses each,
frame identifiers divisible by ten. Exclude original pilot ids from confirmation.
AMASS and test splits remain reserved.

Cache 4096 Haar training sources and 1024 independent audit sources. Match
endpoints and exact source-chart VJPs for two Gaussian or two normalized
training-pose task covectors. Teacher-to-student transport uses Exp(Log(YSᵀYT)/2).
The endpoint base branch is independent of source perturbation; retain its
parameter derivative. The metric is half Frobenius / Euclidean body coordinates.

Screen endpoint-only and random/task VJP coefficients .001, .01, .1, each 1500
AdamW updates, batch 32, learning rate 2e-5, weight decay 1e-4, clip 1, coefficient
warmup 100. Checkpoints every 250, including zero; select by worst per-mask
relative development MPVPE. Diagnose zero versus step-rescaled trajectory cost.
Repeat selected derivative recipe with a second training seed.

Locked confirmation: teacher100, teacher50, raw midpoint4 and both selected
students; legs/torso, two hypotheses, inference seeds 0/1, 200 optimizer iterations,
one warmup iteration, two synchronized timing repetitions. Promotion requires
both student seeds to achieve ≥8× speedup, hidden and visible errors ≤1.05×
teacher, and APD ≥0.90× teacher for each mask. VJP audits are separate evidence.

Execution budget is four hours. Forecast remaining phases from measured work;
stop between work units, preserve checkpoints/cache shards and label incomplete
confirmation. A completed negative experiment is a valid result. Allocated and
reserved VRAM are separate; no claimed covariance guarantee or novel full RCM.

For an already running study, start
`python scripts/collect_pullback_screen.py --run_id RUN_ID --destination results/pullback_screen/RUN_ID`.
The collector makes five-minute snapshots, then retrieves the complete artifacts
and writes `summary.csv` and `mechanism_report.md` after the remote job exits.
It retries transient SSH failures and preserves the running job. Development
summaries use each selected checkpoint; raw measurements retain all checkpoints.
