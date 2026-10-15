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
    if not fingerprint.exists() or fingerprint.read_text().strip() != "1d837501640d036c19a8de1e124cd43e1ff493996f5124399859214cd720b8d5":
        archive=Path('/content/posedflow_inputs.tar.gz')
        parts=sorted(Path('/content').glob('posedflow-1d837501640d036c19a8de1e124cd43e1ff493996f5124399859214cd720b8d5-part-*'))
        if parts:
            with archive.open('wb') as destination:
                for part in parts:
                    with part.open('rb') as source:
                        import shutil
                        shutil.copyfileobj(source,destination)
        with archive.open('rb') as stream:
            assert hashlib.file_digest(stream,'sha256').hexdigest() == "1d837501640d036c19a8de1e124cd43e1ff493996f5124399859214cd720b8d5", 'Upload checksum mismatch'
        with tarfile.open(archive) as bundle:
            bundle.extractall(root, filter='data')
        fingerprint.write_text("1d837501640d036c19a8de1e124cd43e1ff493996f5124399859214cd720b8d5")
    subprocess.run([sys.executable,'-m','pip','install','smplx','chumpy','scipy'],check=True)
    if "1" != "1":
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
