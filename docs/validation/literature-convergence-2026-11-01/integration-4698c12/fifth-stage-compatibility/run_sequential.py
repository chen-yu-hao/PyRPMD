"""Run the fixed new method selection on three existing interpreters."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import signal
import subprocess
import time

ROOT = Path('/workspace/scratch/a5e4820daedd')
REPO = ROOT / 'pyrpmd-convergence-integration'
OUT = REPO / 'docs/validation/literature-convergence-2026-11-01/integration-4698c12/fifth-stage-compatibility'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def main():
    selection = json.loads((OUT / 'selection.json').read_text())
    records = []
    failed = False
    for label, row in selection['per_interpreter'].items():
        work = ROOT / 'distribution-check/fifth-stage-compatibility' / label
        work.mkdir(parents=True, exist_ok=False)
        variables = {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OPENMM_CPU_THREADS')}
        variables.update(PYTHONPATH='src:.', PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(work), MPLCONFIGDIR=str(work / 'matplotlib'))
        command = ['nice', '-n', '10', 'taskset', '-c', '8', row['python'], str(OUT / 'run_checks.py'), '--label', label]
        begun = datetime.now(timezone.utc).isoformat()
        started = time.monotonic()
        timed_out = False
        with (OUT / label / 'pytest.log').open('x') as stream:
            process = subprocess.Popen(command, cwd=REPO, env=dict(os.environ, **variables), stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=180)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    code = process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    code = process.wait(timeout=5)
        report_path = OUT / label / 'compatibility.json'
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        item = {'label': label, 'command': command, 'started_utc': begun,
                'finished_utc': datetime.now(timezone.utc).isoformat(), 'exit_code': code,
                'wall_seconds': time.monotonic() - started, 'declared_wall_timeout_seconds': 180,
                'timed_out': timed_out, 'owned_test_group_reaped': process.poll() is not None,
                'selected_environment': variables, 'log_sha256': sha(OUT / label / 'pytest.log'),
                'compatibility_sha256': sha(report_path) if report_path.exists() else None,
                'passed': code == 0 and not timed_out and report.get('passed') is True}
        write_new(OUT / label / 'launch-receipt.json', item)
        records.append(item)
        print(json.dumps({k: item[k] for k in ('label', 'exit_code', 'wall_seconds', 'passed')}), flush=True)
        if not item['passed']:
            failed = True
            break
    write_new(OUT / 'execution.json', {
        'schema': 'pyrpmd.three-interpreter-bounded-gate-execution.v2',
        'driver_sha256': sha(Path(__file__)), 'selection_sha256': sha(OUT / 'selection.json'),
        'sequential': True, 'all_three_complete': len(records) == 3,
        'passed': len(records) == 3 and not failed, 'records': records,
    })
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
