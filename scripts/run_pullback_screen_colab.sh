#!/usr/bin/env bash
set -euo pipefail
cd /content/posedflow-inference
run_id=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
job=jobs/$run_id
mkdir -p "$job"
exec >> "$job/job.log" 2>&1
trap 'echo $? > "$job/exit_code"' EXIT
python -m unittest discover -s tests -p test_exact_pullback.py
python posed_flow_colab_poc.py --mode pullback_screen \
 --data_path data/3dpw_train_smpl72.npz --validation_path data/validation.npz \
 --model_root /content/posedflow-inference --teacher_run teacher_reference \
 --output_dir "$job/results" --budget_hours 4
