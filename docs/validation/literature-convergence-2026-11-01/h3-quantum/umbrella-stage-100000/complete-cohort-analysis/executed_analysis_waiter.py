"""Analyze the registered 100000-sweep cohort only after confirmed pool drain."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import resource
import subprocess
import time


ROOT = Path('/workspace/scratch/a5e4820daedd/PyRPMD')
CAMPAIGN = ROOT.parent / 'literature-convergence-quantum/campaign-run'
OUTPUT = ROOT.parent / 'literature-convergence-quantum/full-cohort-stage-100000'
PROTOCOL = ROOT / 'docs/validation/literature-convergence-2026-11-01/h3-quantum/protocol.json'
ANALYZER = ROOT / 'benchmarks/h3/quantum_sampling_analyze.py'
SCHEDULER = ROOT / 'benchmarks/h3/quantum_sampling_scheduler.py'
ATTEMPT = CAMPAIGN / 'scheduler-attempts/stage-100000-abdba56327464ed99bbcc630d0843bd2'
SEEDS = [211017, 211019, 211021, 211023, 211027, 211031, 211033, 211039]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError('Refuse to replace an earlier complete-cohort analysis')
    expected = {
        str(PROTOCOL): '5834471583ae8e513050dc1c22e065459e1cfa6f139cc5490fcb4eded9fe711a',
        str(ANALYZER): '4a3476d0ccf12cea1bd9791c4a8ccd7ebcf482b8801e61c87738cd3970e72ea4',
        str(SCHEDULER): '1fa4d02c5a9ed5884e81fd0de1203cec47570d9a514a512d636524583e59ed98',
        str(Path(__file__).resolve()): sha(Path(__file__)),
    }
    if expected != {p: sha(Path(p)) for p in expected}:
        raise ValueError('Frozen source identity differs before waiting')
    protocol = json.loads(PROTOCOL.read_text())
    if protocol['independent_ladder_seeds'] != SEEDS:
        raise ValueError('Registered cohort differs')
    completion_path = CAMPAIGN / 'stage-100000.json'
    lock = CAMPAIGN / '.quantum-scheduler-lock'
    wait_start = time.monotonic()
    while not completion_path.exists() or lock.exists():
        if (ATTEMPT / 'failure.json').exists():
            raise RuntimeError('Scheduler wrote a drained failure; no analysis or resumption is authorized by this waiter')
        if time.monotonic() - wait_start > 43200:
            raise TimeoutError('The complete fixed cohort did not finish within the 12-hour waiting budget')
        time.sleep(10)
    completion = json.loads(completion_path.read_text())
    if not (completion['schema'] == 'pyrpmd.h3.quantum-umbrella-scheduler-completion.v1'
            and completion['production'] == 100000
            and completion['requested_seeds'] == completion['completed_seeds'] == SEEDS
            and completion['all_submitted_workers_drained'] is True
            and completion['prior_prefix_and_state_files_unchanged'] is True
            and completion['protocol_sha256'] == expected[str(PROTOCOL)]
            and completion['scheduler_source_sha256'] == expected[str(SCHEDULER)]
            and completion['attempt_directory'] == str(ATTEMPT)):
        raise ValueError('Global completion does not bind the expected drained full cohort')
    for seed in SEEDS:
        stage = CAMPAIGN / f'seed-{seed}/stages/100000/stage.json'
        row = json.loads(stage.read_text())
        if row['seed'] != seed or row['production_sweeps'] != 100000 or row['protocol_sha256'] != expected[str(PROTOCOL)]:
            raise ValueError('Per-seed stage identity differs')
    if expected != {p: sha(Path(p)) for p in expected}:
        raise ValueError('Frozen source identity changed while waiting')
    completion_hash = sha(completion_path)
    command = [str(ROOT / '.venv/bin/python'), str(ANALYZER),
               '--campaign', str(CAMPAIGN), '--protocol', str(PROTOCOL),
               '--output', str(OUTPUT), '--stage', '100000']
    environment = dict(os.environ, PYTHONPATH=str(ROOT / 'src') + ':' + str(ROOT),
                       PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    clock = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = subprocess.run(command, cwd=ROOT, env=environment)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    unchanged = expected == {p: sha(Path(p)) for p in expected}
    completion_unchanged = sha(completion_path) == completion_hash
    receipt = {
        'schema': 'pyrpmd.h3.complete-umbrella-analysis-execution.v1',
        'started_utc': started, 'command': command, 'independent_seeds': SEEDS,
        'stage': 100000, 'complete_cohort_required': True, 'available_only_used': False,
        'source_sha256': expected, 'sources_unchanged': unchanged,
        'completion_manifest': str(completion_path), 'completion_sha256': completion_hash,
        'completion_unchanged': completion_unchanged,
        'all_submitted_workers_drained_and_lock_released_required': True,
        'elapsed_seconds': time.monotonic() - clock,
        'CPU_seconds': after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
        'returncode': result.returncode, 'no_refinement_launched': True,
        'worker_control_unchanged_by_this_waiter': True,
    }
    OUTPUT.mkdir(exist_ok=True)
    with (OUTPUT / 'execution.json').open('x') as stream:
        stream.write(json.dumps(receipt, indent=2) + '\n')
    if not unchanged or not completion_unchanged:
        raise RuntimeError('Source or completion identity changed during analysis')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
