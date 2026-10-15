from pathlib import Path
import os
job=Path('/content/posedflow/jobs/20261013T133548Z-smoke')
if (job/'exit_code').exists():
    print('JOB_EXIT='+(job/'exit_code').read_text().strip())
else:
    try:
        pid=int((job/'pid').read_text())
        os.kill(pid,0)
        print('JOB_EXIT=1' if Path('/proc/'+str(pid)+'/stat').read_text().split()[2] == 'Z' else 'JOB_RUNNING')
    except (OSError, ValueError):
        print('JOB_EXIT=1')
