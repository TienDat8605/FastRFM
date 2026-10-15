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
job=Path('/content/posedflow/jobs/20261013T141900Z-benchmark')
job.mkdir(parents=True, exist_ok=True)
runner=job/'runner.py'
runner.write_text('''from pathlib import Path
import json, subprocess, sys, tarfile, traceback
job=Path("/content/posedflow/jobs/20261013T141900Z-benchmark")
root=Path("/content/posedflow/study")
code=1
try:
    subprocess.run([sys.executable,"-m","pip","install","pyyaml"],check=True)
    code=subprocess.call([sys.executable,"-u",str(root/"scripts/run_3dpw_study.py"),
                          "benchmark","--root",str(root),"--cohort","development",
                          "--output",str(root/"output/3dpw_study_v3")])
except BaseException:
    traceback.print_exc()
finally:
    output=root/"output/3dpw_study_v3"
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
active.write_text(json.dumps({'pid':proc.pid,'job':'20261013T141900Z-benchmark'}))
print('REMOTE JOB:', '20261013T141900Z-benchmark', 'PID:', proc.pid)
