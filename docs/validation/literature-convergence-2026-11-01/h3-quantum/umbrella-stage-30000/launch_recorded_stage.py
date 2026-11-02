"""Record the reviewed precision continuation, then exec its frozen coordinator."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil

ROOT = Path('/workspace/scratch/a5e4820daedd')
REPO = ROOT / 'PyRPMD'
BASE = Path('docs/validation/literature-convergence-2026-11-01')


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    plan_path = REPO / BASE / 'h3-quantum/umbrella-scheduler/launch-plan.json'
    plan = json.loads(plan_path.read_text())
    source = REPO / 'benchmarks/h3/quantum_sampling_scheduler.py'
    assert sha(source) == plan['scheduler_source_sha256'] == '1fa4d02c5a9ed5884e81fd0de1203cec47570d9a514a512d636524583e59ed98'
    review_path = REPO / BASE / 'review/pr68-software-review/umbrella-scheduler-review/review-receipt.json'
    assert sha(review_path) == '464cd4baa050fd9d8b660e672c2cb6b1e7116f6461b886eff4174c3e1b41dbf6'
    review = json.loads(review_path.read_text())
    assert review['remaining_blockers'] == []
    assert review['reviewed_source']['sha256'] == sha(source)
    archive = ROOT / 'pyrpmd-h3-quantum-umbrella-stage-30000-2026-11-01.tar.gz'
    assert sha(archive) == '24c1390e11791c9ed2092f98eeba30776bbbb5d46bb8f9a7801b09d62a027c07'
    command = plan['command']
    assert command[command.index('--stage') + 1] == '100000'
    assert command[command.index('--workers') + 1] == '2'
    assert len(plan['complete_registered_seeds']) == 8
    control = Path(plan['initial_control_file']['path'])
    assert not control.exists(), 'Refusing to replace an existing worker control'
    receipt_path = ROOT / 'literature-convergence-quantum/umbrella-stage100000-launch.json'
    assert not receipt_path.exists(), 'Refusing to overwrite a launch receipt'
    tmp = control.with_suffix('.root-launch-tmp')
    with tmp.open('x') as stream:
        json.dump({'active_workers': 2}, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.link(tmp, control)
    tmp.unlink()
    receipt = {
        'schema': 'pyrpmd.h3.umbrella-stage100000-root-launch.v1',
        'recorded_immediately_before_exec_utc': datetime.now(timezone.utc).isoformat(),
        'timestamp_scope': 'Recorded just before the exec handoff; coordinator events record subsequent worker admissions.',
        'cwd': plan['cwd'], 'command': command, 'environment_overrides': plan['environment'],
        'source_sha256': sha(source), 'launcher_sha256': sha(Path(__file__)),
        'launch_plan_sha256': sha(plan_path), 'independent_review_sha256': sha(review_path),
        'verified_prior_stage_archive': {'name': archive.name, 'bytes': archive.stat().st_size,
                                         'sha256': sha(archive)},
        'initial_worker_control': {'path': str(control), 'sha256': sha(control), 'active_workers': 2},
        'complete_registered_seeds': plan['complete_registered_seeds'],
        'precision_trigger_only': True,
        'legacy_driver_state': 'Root-owned original 30000-sweep driver completed; no concurrent legacy driver or direct one_seed invocation is authorized or launched.',
        'resource_budget': {'umbrella_workers': 2, 'H3_extra_pair_workers': 2, 'water_workers': 4, 'CPU_ceiling': 8},
        'later_admission_change': 'Only after the H3 extra-pair workers finish, root may atomically change active_workers to 4; no sampling parameter, RNG state, seed order, or acceptance criterion changes.',
        'log_path': str(ROOT / 'literature-convergence-quantum/umbrella-stage100000.log'),
    }
    with receipt_path.open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    for name in ['PyRPMD', 'pyrpmd-convergence-integration']:
        target = ROOT / name / BASE / 'h3-quantum/umbrella-stage-30000/continuation-launch.json'
        assert not target.exists()
        shutil.copyfile(receipt_path, target)
    os.chdir(plan['cwd'])
    environment = dict(os.environ)
    environment.update(plan['environment'])
    os.execvpe(command[0], command, environment)


if __name__ == '__main__':
    main()
