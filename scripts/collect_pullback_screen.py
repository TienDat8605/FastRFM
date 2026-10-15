"""Collect an existing SSH study; never restart, terminate, or retune it."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1]


def command(argv):
    result=subprocess.run(argv,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=55)
    if result.returncode:raise RuntimeError(result.stdout[-1500:])
    return result.stdout.strip()


def collect(host,run_id,destination):
    destination.mkdir(parents=True,exist_ok=True)
    remote=f'/content/posedflow-inference/jobs/{run_id}'
    started=time.monotonic();last_backup=0
    while time.monotonic()-started<4.5*3600:
        try:
            marker=command(['ssh','-o','ConnectTimeout=20',host,f'if test -f {shlex.quote(remote+"/exit_code")}; then cat {shlex.quote(remote+"/exit_code")}; else echo RUNNING; fi'])
            finished=marker!='RUNNING'
            if finished or time.monotonic()-last_backup>300:
                archive=remote+'/collected_results.tar.gz'
                excludes=[] if finished else ['--exclude=cache_shards','--exclude=targets.pt','--exclude=*update*.pt']
                command(['ssh','-o','ConnectTimeout=20',host,' '.join(shlex.quote(v) for v in ['tar','-czf',archive,'-C',remote+'/results',*excludes,'.'])+'; archive_status=$?; test "$archive_status" -le 1'])
                local_archive=destination/('artifacts.tar.gz' if finished else 'snapshot.tar.gz')
                command(['scp',f'{host}:{archive}',str(local_archive)])
                with tarfile.open(local_archive) as bundle:bundle.extractall(destination,filter='data')
                command([sys.executable,'scripts/summarize_pullback_screen.py',str(destination)])
                last_backup=time.monotonic()
                print(json.dumps({'backup':str(local_archive),'finished':finished}),flush=True)
            if finished:
                (destination/'exit_code').write_text(marker+'\n')
                (destination/'collection_status.json').write_text(json.dumps({'state':'collected','remote_exit_code':marker},indent=2))
                return
        except (RuntimeError,subprocess.TimeoutExpired,FileNotFoundError,tarfile.TarError) as error:
            print(f'Retrying collection: {error}',flush=True)
        time.sleep(45)
    (destination/'collection_status.json').write_text(json.dumps({'state':'collector_timeout','remote_job_preserved':True},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--host',default='posedflow-inference');parser.add_argument('--run_id',required=True);parser.add_argument('--destination',required=True)
    args=parser.parse_args();collect(args.host,args.run_id,Path(args.destination).resolve())
