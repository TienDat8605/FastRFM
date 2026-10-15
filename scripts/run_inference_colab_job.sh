#!/usr/bin/env bash
set -Eeuo pipefail
cd /content/posedflow-inference
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
JOB_DIR="/content/posedflow-inference/jobs/$RUN_ID"
mkdir -p "$JOB_DIR"
exec > >(tee -a "$JOB_DIR/job.log") 2>&1
trap 'rc=$?; echo "$rc" > "$JOB_DIR/exit_code"; exit "$rc"' EXIT
python -m pip install --no-build-isolation smplx chumpy
python - <<'PY'
import unittest
patterns = ['test_adjoint.py', 'test_distillation.py', 'test_study_acceleration.py',
            'test_study_geometry.py', 'test_study_data_metrics.py']
suite = unittest.TestSuite(unittest.defaultTestLoader.discover('tests', pattern=p) for p in patterns)
result = unittest.TextTestRunner(verbosity=1).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
PY
REUSE_ARGS=()
if [ -n "${REUSE_TRAINING_DIR:-}" ]; then
  REUSE_ARGS=(--reuse_training_dir "$REUSE_TRAINING_DIR")
fi
python -u posed_flow_colab_poc.py --mode inference \
  --data_path data/3dpw_train_smpl72.npz \
  --validation_path data/validation.npz \
  --model_root /content/posedflow-inference \
  --num_joints 21 --steps 1000 --distill_steps 500 --opt_iters 20 --repeats 2 \
  --output_dir "$JOB_DIR/results" "${REUSE_ARGS[@]}"
