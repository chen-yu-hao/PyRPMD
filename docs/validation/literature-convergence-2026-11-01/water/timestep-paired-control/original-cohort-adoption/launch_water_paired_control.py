"""Execute only the adopted fixed eight-pair timestep control, once."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import time

ROOT = Path('/workspace/scratch/a5e4820daedd')
REPO = ROOT / 'PyRPMD'
ADOPTION = REPO / 'docs/validation/literature-convergence-2026-11-01/water/timestep-paired-control/original-cohort-adoption/root-adoption.json'
EXECUTION = ROOT / 'water-paired-timestep-execution-2026-11-01'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def write(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + '\n')

def main():
    adoption = json.loads(ADOPTION.read_text())
    assert adoption['decision'] == 'adopted_fixed_numerical_control'
    assert adoption['launcher_sha256'] == sha(Path(__file__))
    guards = adoption['input_and_source_guards']
    def check_guards():
        changed = [name for name, expected in guards.items() if sha(name) != expected]
        if changed:
            raise ValueError('Bound inputs changed: ' + repr(changed))
    check_guards()
    draft = json.loads(Path(adoption['pending_adoption']['path']).read_text())
    output = Path(adoption['control_output'])
    assert not output.exists()
    EXECUTION.mkdir(exist_ok=False)
    write(EXECUTION / 'RUN_ONCE.json', dict(adoption_sha256=sha(ADOPTION),
        started_utc=datetime.now(timezone.utc).isoformat(), driver_sha256=sha(Path(__file__))))
    env = dict(os.environ, **adoption['resources']['environment'])
    records = []
    began = time.monotonic()
    def execute(label, command, timeout=None):
        check_guards()
        actual = ['nice', '-n', '5', 'taskset', '-c', '4', *command]
        start = time.monotonic()
        with (EXECUTION / (label + '.stdout.log')).open('x') as stdout, (EXECUTION / (label + '.stderr.log')).open('x') as stderr:
            result = subprocess.run(actual, cwd=REPO, env=env, stdout=stdout,
                                    stderr=stderr, timeout=timeout)
        check_guards()
        row = dict(label=label, command=actual, exit_code=result.returncode,
                   elapsed_seconds=time.monotonic()-start,
                   stdout_sha256=sha(EXECUTION / (label + '.stdout.log')),
                   stderr_sha256=sha(EXECUTION / (label + '.stderr.log')),
                   completed_utc=datetime.now(timezone.utc).isoformat())
        write(EXECUTION / (label + '.json'), row)
        records.append(row)
        if result.returncode:
            raise RuntimeError('Fixed control command failed: ' + label)
        return (EXECUTION / (label + '.stdout.log')).read_text()
    try:
        execute('registration', draft['commands']['register_after_adoption'], timeout=120)
        registration = output / 'registration.json'
        registration_sha = sha(registration)
        registered = json.loads(registration.read_text())
        selected = json.loads(Path(draft['selected_inputs']['path']).read_text())['selected']
        assert registered['fixed_seeds'] == draft['fixed_seeds']
        for row in selected:
            actual = registered['selection'][str(row['seed'])]
            assert actual['manifest_sha256'] == row['manifest_sha256']
            assert actual['checkpoint'] == row['checkpoint']
        complete = False
        for index in range(1, adoption['resources']['maximum_invocations'] + 1):
            if sha(registration) != registration_sha:
                raise ValueError('Registration changed')
            if time.monotonic()-began >= adoption['resources']['total_admission_wall_seconds']:
                raise TimeoutError('No further control admission within fixed total budget')
            result = json.loads(execute(f'run-{index}', draft['commands']['bounded_run_after_registration']))
            if not isinstance(result.get('complete'), bool):
                raise ValueError('Malformed fixed-run completion result')
            print(json.dumps(dict(invocation=index, complete=result['complete'],
                                  elapsed_seconds=time.monotonic()-began)), flush=True)
            if result['complete']:
                complete = True
                break
        if not complete:
            raise TimeoutError('Fixed invocation budget exhausted with retained checkpoints')
        execute('complete-analysis', draft['commands']['analyze_after_every_pair_complete'], timeout=300)
        write(EXECUTION / 'completion.json', dict(
            schema='pyrpmd.water-adopted-timestep-execution.v1', exit_code=0,
            fixed_control_complete=True, registration_sha256=registration_sha,
            source_inputs_unchanged=True, all_commands_reaped=True,
            completed_utc=datetime.now(timezone.utc).isoformat(),
            wall_seconds=time.monotonic()-began, records=records,
            adopted_physical_steps=384000, adopted_NVE_ps=64,
            literature_reproduction_claimed=False, preparation_extension_started=False))
    except BaseException as error:
        write(EXECUTION / 'failure.json', dict(
            schema='pyrpmd.water-adopted-timestep-execution-failure.v1',
            error_type=type(error).__name__, error=str(error), records=records,
            completed_utc=datetime.now(timezone.utc).isoformat(),
            all_committed_data_retained=True, automatic_retry=False))
        raise
    print(json.dumps(dict(fixed_control_complete=True, wall_seconds=time.monotonic()-began)), flush=True)

if __name__ == '__main__':
    main()
