"""Run the fixed new-case selections sequentially using existing interpreters."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import time

BASE=Path('/workspace/scratch/a5e4820daedd')
REPO=BASE/'pyrpmd-convergence-integration'
OUT=REPO/'docs/validation/literature-convergence-2026-11-01/integration-4698c12/four-pair-p256-compatibility'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    selection=json.loads((OUT/'selection.json').read_text())
    records=[]
    for label,row in selection['per_interpreter'].items():
        work=BASE/'distribution-check/four-pair-p256-compatibility'/label
        variables={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','OPENMM_CPU_THREADS')}
        variables.update(PYTHONPATH='src:.',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(work),MPLCONFIGDIR=str(work/'matplotlib'))
        command=['nice','-n','10','taskset','-c','8',row['python'],str(OUT/'run_checks.py'),'--label',label]
        begun=datetime.now(timezone.utc).isoformat();started=time.monotonic()
        with (OUT/label/'pytest.log').open('x') as stream:
            completed=subprocess.run(command,cwd=REPO,env=dict(os.environ,**variables),stdout=stream,stderr=subprocess.STDOUT,timeout=120)
        report=json.loads((OUT/label/'compatibility.json').read_text())
        item=dict(label=label,command=command,started_utc=begun,finished_utc=datetime.now(timezone.utc).isoformat(),
            exit_code=completed.returncode,wall_seconds=time.monotonic()-started,
            declared_timeout_seconds=120,selected_environment=variables,
            compatibility_sha256=sha(OUT/label/'compatibility.json'),log_sha256=sha(OUT/label/'pytest.log'),
            passed=report['passed'])
        with (OUT/label/'launch-receipt.json').open('x') as stream:
            stream.write(json.dumps(item,indent=2)+'\n')
        records.append(item)
        print(json.dumps({k:item[k] for k in ['label','exit_code','wall_seconds','passed']}),flush=True)
        if completed.returncode or not report['passed']:
            raise SystemExit(completed.returncode or 1)
    with (OUT/'execution.json').open('x') as stream:
        stream.write(json.dumps(dict(schema='pyrpmd.three-interpreter-bounded-gate-execution.v1',
            source_sha256=sha(Path(__file__)),selection_sha256=sha(OUT/'selection.json'),
            sequential=True,records=records),indent=2)+'\n')

if __name__=='__main__':
    main()
