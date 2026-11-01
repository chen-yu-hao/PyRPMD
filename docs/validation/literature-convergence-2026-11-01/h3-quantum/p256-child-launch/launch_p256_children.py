"""Launch the separately adopted complete P256 child cohort exactly once."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import resource
import subprocess
import time

BASE=Path('/workspace/scratch/a5e4820daedd')
REPO=BASE/'PyRPMD'
OUT=BASE/'literature-convergence-quantum/p256-fixed-eight-chain-v1/child-execution'
DRIVER=REPO/'benchmarks/h3/p256_child_campaign.py'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authorization-sha256',required=True)
    parser.add_argument('--coordinator-sha256',required=True)
    args=parser.parse_args()
    auth=OUT/'authorization.json'
    if sha(auth)!=args.authorization_sha256 or sha(DRIVER)!=args.coordinator_sha256:
        raise ValueError('Reviewed launch identities changed')
    authorization=json.loads(auth.read_text())
    if authorization['workers']!=2 or authorization['cpu_affinity']!=[6,7]:
        raise ValueError('Concrete two-worker CPU allocation differs')
    if authorization['whole_campaign_wall_seconds']!=36000 or authorization['finalization_reserve_seconds']!=120:
        raise ValueError('Reviewed bounded child allocation differs')
    command=[str(REPO/'.venv/bin/python'),str(DRIVER),'run','--authorization',str(auth)]
    selected={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}
    selected.update(PYTHONPATH=str(REPO/'src')+':'+str(REPO),PYTHONDONTWRITEBYTECODE='1')
    identities={str(p):sha(p) for p in [DRIVER,auth,Path(__file__)]}
    launch=dict(schema='pyrpmd.p256-child.root-launch.v1',
        before_subprocess_call_utc=datetime.now(timezone.utc).isoformat(),command=command,
        working_directory=str(REPO),selected_environment=selected,source_sha256=identities,
        workers=2,CPU_affinity=[6,7],whole_campaign_wall_seconds=36000,
        finalization_reserve_seconds=120,all_eight_seeds_required=True,automatic_analysis=False)
    with (OUT/'launch.json').open('x') as stream:
        stream.write(json.dumps(launch,indent=2)+'\n')
    started=time.monotonic();usage=resource.getrusage(resource.RUSAGE_CHILDREN)
    with (OUT/'campaign.log').open('x') as stream:
        completed=subprocess.run(command,cwd=REPO,env=dict(os.environ,**selected),stdout=stream,stderr=subprocess.STDOUT)
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    receipt=dict(launch,finished_utc=datetime.now(timezone.utc).isoformat(),
        exit_code=completed.returncode,wall_seconds=time.monotonic()-started,
        child_tree_user_CPU_seconds=after.ru_utime-usage.ru_utime,
        child_tree_system_CPU_seconds=after.ru_stime-usage.ru_stime,
        sources_unchanged=identities=={p:sha(Path(p)) for p in identities},log_sha256=sha(OUT/'campaign.log'))
    with (OUT/'completion.json').open('x') as stream:
        stream.write(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ['exit_code','wall_seconds','sources_unchanged']}),flush=True)
    raise SystemExit(completed.returncode)

if __name__=='__main__':
    main()
