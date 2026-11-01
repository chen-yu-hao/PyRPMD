"""Analyze both registered 200 fs pair batches only after the full extra cohort finishes."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import resource
import subprocess
import time

ROOT = Path('/workspace/scratch/a5e4820daedd/PyRPMD')
DATA = ROOT.parent / 'literature-convergence-quantum/recrossing-primary'
OUTPUT = ROOT.parent / 'literature-convergence-quantum/recrossing-analysis-200-four-pairs'
ANALYZER = ROOT / 'benchmarks/h3/quantum_recrossing_analyze.py'
PROTOCOL = DATA / 'protocol.json'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def ready(seeds):
    for seed in seeds:
        for step in ['0.1', '0.05']:
            path = DATA / 'extra-pair-batch-1' / f'seed-{seed}/dt-{step}/horizon-200/report.json'
            if not path.exists():
                return False
            report = json.loads(path.read_text())
            if not report['complete'] or not report['source_unchanged']:
                return False
    return True


def main():
    assert not OUTPUT.exists(), 'Refuse to replace an earlier full-cohort analysis'
    protocol = json.loads(PROTOCOL.read_text())
    seeds = protocol['seeds']
    assert seeds == [721301, 721303, 721307, 721309, 721313, 721317, 721321, 721323]
    assert sha(ANALYZER) == '4d1925461219e87d85d4e1f4ac98263c4bb58ff0d7713075782145ba909649c1'
    identities = {str(p): sha(p) for p in [ANALYZER, PROTOCOL, Path(__file__)]}
    wait_start = time.monotonic()
    while not (DATA / 'schedule-200-pairs-1-completed.json').exists() or not ready(seeds):
        if time.monotonic() - wait_start > 7200:
            raise TimeoutError('The complete fixed cohort did not finish within the waiting budget')
        time.sleep(10)
    assert identities == {p: sha(Path(p)) for p in identities}
    command = [str(ROOT / '.venv/bin/python'), str(ANALYZER), '--data', str(DATA),
               '--output', str(OUTPUT), '--horizon', '200', '--include-extra-pairs']
    environment = dict(os.environ, PYTHONPATH=str(ROOT / 'src') + ':' + str(ROOT),
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    clock = time.monotonic()
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = subprocess.run(command, cwd=ROOT, env=environment)
    final_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    unchanged = identities == {p: sha(Path(p)) for p in identities}
    receipt = {'schema': 'pyrpmd.h3.complete-quantum-analysis-execution.v1',
        'started_utc': started, 'command': command, 'independent_seeds': seeds,
        'horizon_fs': 200, 'pair_batches': [0, 1], 'complete_cohort_required': True,
        'available_only_used': False, 'source_sha256': identities,
        'sources_unchanged': unchanged, 'elapsed_seconds': time.monotonic() - clock,
        'CPU_seconds': final_usage.ru_utime + final_usage.ru_stime - usage.ru_utime - usage.ru_stime,
        'returncode': result.returncode, 'no_continuation_launched': True, 'extra_coordinator_pool_shutdown_completion_required': True}
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / 'execution.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if not unchanged:
        raise RuntimeError('Analysis source identity changed')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
