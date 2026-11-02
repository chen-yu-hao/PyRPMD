"""Increase admission only after the registered extra-child pool has exited."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os

ROOT = Path('/workspace/scratch/a5e4820daedd')
DATA = ROOT / 'literature-convergence-quantum'
SEEDS = [721301, 721303, 721307, 721309, 721313, 721317, 721321, 721323]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    completion = DATA / 'recrossing-primary/schedule-200-pairs-1-completed.json'
    drained = DATA / 'extra-pairs-stage-completion-exec.json'
    scheduler = ROOT / 'PyRPMD/benchmarks/h3/quantum_sampling_scheduler.py'
    control = DATA / 'umbrella-worker-control.json'
    record = DATA / 'umbrella-worker-admission-2-to-4.json'
    if record.exists():
        raise FileExistsError('The admission change already has a receipt')
    done = json.loads(completion.read_text())
    exited = json.loads(drained.read_text())
    assert done['inventory']['seeds'] == SEEDS
    assert [row['seed'] for row in done['completed']] == SEEDS
    assert all(row['complete'] for row in done['completed'])
    assert exited['exec_session_id'] == 74559 and exited['exit_code'] == 0
    assert exited['completion_sha256'] == sha(completion)
    assert sha(scheduler) == '1fa4d02c5a9ed5884e81fd0de1203cec47570d9a514a512d636524583e59ed98'
    assert sha(control) == 'ec6d3935ab657a76836dd4b92e3b258b7ac93e7485f964cb9e15a0894f58f03e'
    assert json.loads(control.read_text()) == {'active_workers': 2}
    assert (DATA / 'campaign-run/.quantum-scheduler-lock').is_dir()
    attempt = DATA / 'campaign-run/scheduler-attempts/stage-100000-abdba56327464ed99bbcc630d0843bd2'
    assert attempt.is_dir() and not (attempt / 'failure.json').exists()
    old = control.read_bytes()
    new = (json.dumps({'active_workers': 4}, indent=2) + '\n').encode()
    decision = {
        'schema': 'pyrpmd.umbrella.worker-admission-change.v1',
        'decided_utc': datetime.now(timezone.utc).isoformat(),
        'reason': 'The complete registered two-worker extra-child cohort exited normally; its two CPU slots are reassigned to unchanged umbrella seed workers.',
        'previous_active_worker_ceiling': 2, 'new_active_worker_ceiling': 4,
        'previous_control_sha256': hashlib.sha256(old).hexdigest(),
        'new_control_sha256': hashlib.sha256(new).hexdigest(),
        'extra_cohort_completion': {'path': str(completion), 'sha256': sha(completion)},
        'exec_exit_receipt': {'path': str(drained), 'sha256': sha(drained)},
        'scheduler_sha256': sha(scheduler), 'driver_sha256': sha(Path(__file__)),
        'campaign_attempt': str(attempt), 'production_sweeps': 100000,
        'no_sampler_protocol_RNG_or_scientific_gate_changes': True,
        'new_workers_may_be_admitted_at_next_coordinator_poll': True,
    }
    # Record the concrete decision before changing the only mutable control.
    with record.open('x') as stream:
        json.dump(decision, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary = control.with_name('umbrella-worker-control.2-to-4.tmp')
    with temporary.open('xb') as stream:
        stream.write(new)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, control)
    assert sha(control) == decision['new_control_sha256']
    print(json.dumps({'record': str(record), 'record_sha256': sha(record),
                      'control_sha256': sha(control), 'active_workers': 4}))


if __name__ == '__main__':
    main()
