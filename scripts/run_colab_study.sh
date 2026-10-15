#!/usr/bin/env bash
# Upload pose data/assets, validate official models, and run a small GPU pilot.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="${SESSION_NAME:-posedflow-study}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
RUN_DIR="$ROOT/results/colab"
mkdir -p "$RUN_DIR"
LOG_FILE="$RUN_DIR/orchestration.log"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'rc=$?; echo "[$(date -Is)] FAILED line=$LINENO exit=$rc"; exit "$rc"' ERR
step() { echo "[$(date -Is)] $*"; }
cd "$ROOT"
step '1/7 Check and reuse Colab session'
status_ok=0
for attempt in 1 2 3; do
    if SESSION_STATUS="$(colab status -s "$SESSION_NAME" 2>&1)"; then
        echo "$SESSION_STATUS"
        status_ok=1
        break
    fi
    echo "$SESSION_STATUS"
    step "Session API attempt $attempt failed; retry connection without replacing the VM"
    sleep 5
done
[[ "$status_ok" == 1 ]]
if ! rg -q 'Hardware:.*Status:' <<< "$SESSION_STATUS"; then
    COLAB_PYTHON="$(dirname "$(readlink -f "$(command -v colab)")")/python"
    "$COLAB_PYTHON" scripts/recover_colab_session.py --session "$SESSION_NAME" --snapshot "$RUN_DIR/allocation_before.json"
    if ! colab new -s "$SESSION_NAME" --gpu T4; then
        step 'Allocation response failed; recover a sole unnamed T4 if creation succeeded server-side'
        "$COLAB_PYTHON" scripts/recover_colab_session.py --session "$SESSION_NAME" --before "$RUN_DIR/allocation_before.json"
    fi
fi
step '2/7 Package local code, pose-only datasets, models, and checkpoint'
if [[ ! -f "$RUN_DIR/posedflow_inputs.tar.gz" || "${REFRESH_BUNDLE:-0}" == 1 ]]; then
    python3 scripts/prepare_colab_bundle.py
else
    step 'Reuse prepared local bundle (REFRESH_BUNDLE=1 rebuilds it)'
fi
DIGEST="$(cat "$RUN_DIR/bundle.sha256")"
step '3/7 Check remote bundle fingerprint'
cat > "$RUN_DIR/check_remote.py" <<PY
from pathlib import Path
p=Path('/content/posedflow/bundle.sha256')
print('BUNDLE_MATCH' if p.exists() and p.read_text().strip() == '$DIGEST' else 'BUNDLE_MISSING')
PY
colab exec -s "$SESSION_NAME" -f "$RUN_DIR/check_remote.py" > "$RUN_DIR/remote_check.log" 2>&1
cat "$RUN_DIR/remote_check.log"
if ! rg -q '^BUNDLE_MATCH' "$RUN_DIR/remote_check.log"; then
    step '4/7 Upload bundle'
    CHUNK_DIR="$RUN_DIR/chunks-$DIGEST"
    mkdir -p "$CHUNK_DIR"
    if [[ ! -f "$CHUNK_DIR/complete" ]]; then
        split -b 16M -d -a 4 "$RUN_DIR/posedflow_inputs.tar.gz" "$CHUNK_DIR/part-"
        touch "$CHUNK_DIR/complete"
    fi
    cat > "$RUN_DIR/check_chunks.py" <<PY
from pathlib import Path
for p in Path('/content').glob('posedflow-$DIGEST-part-*'):
    print('CHUNK_EXISTS '+p.name+' '+str(p.stat().st_size))
PY
    colab exec -s "$SESSION_NAME" -f "$RUN_DIR/check_chunks.py" > "$RUN_DIR/chunk_inventory.log" 2>&1
    # Each chunk is a separate request, below the runtime proxy size limit.
    for chunk in "$CHUNK_DIR"/part-*; do
        remote_name="posedflow-$DIGEST-$(basename "$chunk")"
        size="$(stat -c %s "$chunk")"
        if rg -q "^CHUNK_EXISTS $remote_name $size$" "$RUN_DIR/chunk_inventory.log"; then
            step "Reuse $remote_name"
        else
            step "Upload $(basename "$chunk") ($size bytes)"
            uploaded=0
            for attempt in 1 2 3; do
                if colab upload -s "$SESSION_NAME" "$chunk" "/content/$remote_name"; then
                    uploaded=1
                    break
                fi
                step "Upload attempt $attempt failed for $(basename "$chunk"); retain prior chunks"
                sleep 5
            done
            [[ "$uploaded" == 1 ]]
        fi
    done
else
    step '4/7 Remote bundle already present; reuse'
fi
step '5/7 Launch remote setup and pilot as a logged subprocess'
cat > "$RUN_DIR/launch_remote.py" <<PY
from pathlib import Path
import subprocess, sys
root=Path('/content/posedflow')
root.mkdir(exist_ok=True)
runner=root/'run_job.py'
runner.write_text('''from pathlib import Path
import hashlib, json, subprocess, sys, tarfile, traceback
root=Path('/content/posedflow')
try:
    fingerprint=root/'bundle.sha256'
    if not fingerprint.exists() or fingerprint.read_text().strip() != "$DIGEST":
        archive=Path('/content/posedflow_inputs.tar.gz')
        parts=sorted(Path('/content').glob('posedflow-$DIGEST-part-*'))
        if parts:
            with archive.open('wb') as destination:
                for part in parts:
                    with part.open('rb') as source:
                        import shutil
                        shutil.copyfileobj(source,destination)
        with archive.open('rb') as stream:
            assert hashlib.file_digest(stream,'sha256').hexdigest() == "$DIGEST", 'Upload checksum mismatch'
        with tarfile.open(archive) as bundle:
            bundle.extractall(root, filter='data')
        fingerprint.write_text("$DIGEST")
    subprocess.run([sys.executable,'-m','pip','install','smplx','chumpy','scipy'],check=True)
    if "$PREPARE_ONLY" != "1":
        subprocess.run([sys.executable,'-u',str(root/'study/colab_pose_pilot.py')],check=True)
    (root/'exit_code').write_text('0')
except BaseException:
    traceback.print_exc()
    (root/'exit_code').write_text('1')
''')
status=root/'exit_code'
if status.exists(): status.unlink()
with (root/'remote_job.log').open('w') as log:
    proc=subprocess.Popen([sys.executable,'-u',str(runner)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'pid').write_text(str(proc.pid))
print('Remote job launched:',proc.pid)
PY
colab exec -s "$SESSION_NAME" -f "$RUN_DIR/launch_remote.py"
step '6/7 Wait for remote process completion (no log monitoring)'
cat > "$RUN_DIR/job_status.py" <<'PY'
from pathlib import Path
import os
root=Path('/content/posedflow')
status=root/'exit_code'
if status.exists():
    print('JOB_EXIT='+status.read_text().strip())
else:
    try:
        pid=int((root/'pid').read_text())
        os.kill(pid,0)
        state=Path(f'/proc/{pid}/stat').read_text().split()[2]
        print('JOB_EXIT=1' if state == 'Z' else 'JOB_RUNNING')
    except (OSError, ValueError):
        print('JOB_EXIT=1')
PY
while true; do
    colab exec -s "$SESSION_NAME" -f "$RUN_DIR/job_status.py" > "$RUN_DIR/job_status.log" 2>&1
    if rg -q '^JOB_EXIT=' "$RUN_DIR/job_status.log"; then break; fi
    sleep 15
done
step '7/7 Retrieve logs and outputs after process exit'
colab download -s "$SESSION_NAME" /content/posedflow/remote_job.log "$RUN_DIR/remote_job.log"
if rg -q '^JOB_EXIT=0' "$RUN_DIR/job_status.log"; then
    if [[ "$PREPARE_ONLY" != 1 ]]; then
        colab download -s "$SESSION_NAME" /content/posedflow/study/output/pilot_report.json "$RUN_DIR/pilot_report.json"
        colab download -s "$SESSION_NAME" /content/posedflow/study/output/predictions.pt "$RUN_DIR/predictions.pt"
    fi
    step 'SUCCESS. Session retained with uploaded data for subsequent study runs.'
else
    step "Remote job failed. See $RUN_DIR/remote_job.log"
    exit 1
fi
