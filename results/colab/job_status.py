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
