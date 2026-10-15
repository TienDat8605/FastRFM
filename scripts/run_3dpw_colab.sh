#!/usr/bin/env bash
# STUDY_COMMAND=smoke (default), evaluate, finetune, audit, report, or all.
# COHORT=development (default) or final for evaluation; no CLI execution deadline.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="${SESSION_NAME:-posedflow-study}"
STUDY_COMMAND="${STUDY_COMMAND:-smoke}"
COHORT="${COHORT:-development}"
STUDY_OUTPUT="${STUDY_OUTPUT:-3dpw_study_v4}"
[[ "$STUDY_OUTPUT" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid STUDY_OUTPUT'; exit 2; }
case "$STUDY_COMMAND" in smoke|benchmark|evaluate|finetune|audit|report|all) ;; *) echo 'Invalid STUDY_COMMAND'; exit 2;; esac
case "$COHORT" in development|final) ;; *) echo 'Invalid COHORT'; exit 2;; esac
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$STUDY_COMMAND"
RUN_DIR="$ROOT/results/3dpw_study/$RUN_ID"
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/orchestration.log") 2>&1
trap 'rc=$?; echo "[$(date -Is)] FAILED line=$LINENO exit=$rc"; exit "$rc"' ERR
step() { echo "[$(date -Is)] $*"; }
cd "$ROOT"
step '1/5 Restore or reuse Colab data and environment'
PREPARE_ONLY=1 SESSION_NAME="$SESSION_NAME" bash scripts/run_colab_study.sh
step '2/5 Package and upload current implementation only'
tar --exclude='__pycache__' --exclude='*.pyc' -czf "$RUN_DIR/study_code.tar.gz" \
    src configs tests scripts/run_3dpw_study.py
colab upload -s "$SESSION_NAME" "$RUN_DIR/study_code.tar.gz" /content/study_code.tar.gz
cat > "$RUN_DIR/launch.py" <<PY
from pathlib import Path
import json, os, subprocess, sys, tarfile
root=Path('/content/posedflow/study')
active=Path('/content/posedflow/study_active.json')
if active.exists():
    previous=json.loads(active.read_text())
    try:
        os.kill(previous['pid'], 0)
        state=Path('/proc/'+str(previous['pid'])+'/stat').read_text().split()[2]
    except (OSError, ValueError):
        state='gone'
    if state not in ('gone','Z'):
        raise RuntimeError('An existing study job is running: '+str(previous))
with tarfile.open('/content/study_code.tar.gz') as bundle:
    bundle.extractall(root, filter='data')
job=Path('/content/posedflow/jobs/$RUN_ID')
job.mkdir(parents=True, exist_ok=True)
runner=job/'runner.py'
runner.write_text('''from pathlib import Path
import json, subprocess, sys, tarfile, traceback
job=Path("/content/posedflow/jobs/$RUN_ID")
root=Path("/content/posedflow/study")
code=1
try:
    subprocess.run([sys.executable,"-m","pip","install","pyyaml"],check=True)
    code=subprocess.call([sys.executable,"-u",str(root/"scripts/run_3dpw_study.py"),
                          "$STUDY_COMMAND","--root",str(root),"--cohort","$COHORT",
                          "--output",str(root/"output/$STUDY_OUTPUT")])
except BaseException:
    traceback.print_exc()
finally:
    output=root/"output/$STUDY_OUTPUT"
    archive=job/"artifacts.tar.gz"
    if output.exists():
        with tarfile.open(archive,"w:gz") as bundle:
            bundle.add(output,arcname="study_results")
        import hashlib
        parts=[]
        with archive.open("rb") as stream:
            index=0
            while data:=stream.read(16*1024*1024):
                name="artifacts.part-"+str(index).zfill(4)
                (job/name).write_bytes(data)
                parts.append(name)
                index+=1
        with archive.open("rb") as stream:
            digest=hashlib.file_digest(stream,"sha256").hexdigest()
        (job/"downloads.json").write_text(json.dumps({"parts":parts,"sha256":digest}))
    (job/"exit_code").write_text(str(code))
''')
with (job/'remote.log').open('w') as log:
    proc=subprocess.Popen([sys.executable,'-u',str(runner)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(job/'pid').write_text(str(proc.pid))
active.write_text(json.dumps({'pid':proc.pid,'job':'$RUN_ID'}))
print('REMOTE JOB:', '$RUN_ID', 'PID:', proc.pid)
PY
step "3/5 Launch $STUDY_COMMAND"
colab exec -s "$SESSION_NAME" -f "$RUN_DIR/launch.py"
cat > "$RUN_DIR/status.py" <<PY
from pathlib import Path
import os
job=Path('/content/posedflow/jobs/$RUN_ID')
if (job/'exit_code').exists():
    print('JOB_EXIT='+(job/'exit_code').read_text().strip())
else:
    try:
        pid=int((job/'pid').read_text())
        os.kill(pid,0)
        print('JOB_EXIT=1' if Path('/proc/'+str(pid)+'/stat').read_text().split()[2] == 'Z' else 'JOB_RUNNING')
    except (OSError, ValueError):
        print('JOB_EXIT=1')
PY
step '4/5 Wait for process exit; logs are not inspected during execution'
while true; do
    colab exec -s "$SESSION_NAME" -f "$RUN_DIR/status.py" > "$RUN_DIR/status.log" 2>&1
    if rg -q '^JOB_EXIT=' "$RUN_DIR/status.log"; then break; fi
    sleep 15
done
step '5/5 Download logs and study artifacts after exit'
colab download -s "$SESSION_NAME" "/content/posedflow/jobs/$RUN_ID/remote.log" "$RUN_DIR/remote.log"
if colab download -s "$SESSION_NAME" "/content/posedflow/jobs/$RUN_ID/downloads.json" "$RUN_DIR/downloads.json"; then
    while read -r part; do
        colab download -s "$SESSION_NAME" "/content/posedflow/jobs/$RUN_ID/$part" "$RUN_DIR/$part"
    done < <(python3 -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["parts"]))' "$RUN_DIR/downloads.json")
    python3 - "$RUN_DIR" <<'PY'
from pathlib import Path
import hashlib, json, sys, tarfile
root=Path(sys.argv[1]); manifest=json.loads((root/'downloads.json').read_text())
archive=root/'artifacts.tar.gz'
with archive.open('wb') as destination:
    for part in manifest['parts']:
        destination.write((root/part).read_bytes())
with archive.open('rb') as stream:
    assert hashlib.file_digest(stream,'sha256').hexdigest() == manifest['sha256']
with tarfile.open(archive) as bundle:
    bundle.extractall(root,filter='data')
PY
fi
if ! rg -q '^JOB_EXIT=0' "$RUN_DIR/status.log"; then
    step "Remote process failed; inspect $RUN_DIR/remote.log"
    exit 1
fi
step "Process exited successfully; artifacts: $RUN_DIR/study_results"
