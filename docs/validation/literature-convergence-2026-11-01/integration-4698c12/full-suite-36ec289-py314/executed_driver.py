"""Run the ordinary whole-project pytest selection at the published head.

This is a separate current-head CI check, not an addition to the method-case
union. It uses the existing Python 3.14 environment and builds/installs nothing.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from xml.etree import ElementTree

ROOT = Path('/workspace/scratch/a5e4820daedd')
REPO = ROOT / 'pyrpmd-convergence-integration'
OUT = REPO / 'docs/validation/literature-convergence-2026-11-01/integration-4698c12/full-suite-36ec289-py314'
PRIOR = OUT.parent / 'auxiliary-proof-compatibility-v2'
PYTHON = ROOT / 'distribution-check/unit-py314/bin/python'
HEAD = '36ec289981639bbeb8feecf3b30400b4cbda9afe'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + '\n')

def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO, text=True).strip()

def load_auditor():
    path = PRIOR / 'run_checks.py'
    spec = importlib.util.spec_from_file_location('pyrpmd_current_head_test_auditor', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def worker():
    assert Path(sys.executable) == PYTHON
    assert sys.version_info[:2] == (3, 14)
    assert Path.cwd() == REPO
    assert sorted(os.sched_getaffinity(0)) == [6, 8]
    assert git('rev-parse', 'HEAD') == HEAD
    assert not git('diff', '--name-only') and not git('diff', '--cached', '--name-only')
    auditor = load_auditor()
    selection = json.loads((PRIOR / 'selection.json').read_text())
    before = json.loads((PRIOR / 'guard-before.json').read_text())
    assert auditor.guards(before)['all_unchanged']
    initial = auditor.runtime('py314', selection)
    write(OUT / 'runtime-before.json', initial)
    import pytest
    started = time.monotonic()
    args = ['-q', '-p', 'no:cacheprovider',
            '--basetemp=' + str(ROOT / 'distribution-check/full-suite-36ec289-py314/pytest-tmp'),
            '--junitxml=' + str(OUT / 'pytest.xml')]
    code = int(pytest.main(args, plugins=[auditor.RuntimeAudit('py314', selection, OUT)]))
    tree = ElementTree.parse(OUT / 'pytest.xml')
    counts = {key: sum(int(s.attrib.get(key, '0')) for s in tree.iter('testsuite'))
              for key in ['tests', 'failures', 'errors', 'skipped']}
    ids = [c.attrib.get('classname', '') + '::' + c.attrib['name'] for c in tree.iter('testcase')]
    after = auditor.runtime('py314', selection)
    write(OUT / 'runtime-after.json', after)
    guard_after = auditor.guards(before)
    clean = not git('diff', '--name-only') and not git('diff', '--cached', '--name-only')
    unchanged_head = git('rev-parse', 'HEAD') == HEAD
    env_same = initial['environment']['distributions'] == after['environment']['distributions']
    report = dict(schema='pyrpmd.current-head-full-pytest.v1',
        completed_utc=datetime.now(timezone.utc).isoformat(), head=HEAD,
        pytest_arguments=args, pytest_exit_code=code, junit_counts=counts,
        elapsed_seconds=time.monotonic()-started,
        distinct_junit_case_ids=len(set(ids)), junit_case_records=len(ids),
        case_ids=ids, source_head_unchanged=unchanged_head, tracked_tree_unchanged=clean,
        guard_after=guard_after, dependencies_unchanged=env_same,
        source_driver_sha256=sha(Path(__file__)), auditor_sha256=sha(PRIOR / 'run_checks.py'),
        junit_sha256=sha(OUT / 'pytest.xml'), builds=0, installs=0,
        note='The unfiltered ordinary project pytest selection. Optional-dependency or platform skips are retained and are not passing tests. This run is not added to earlier overlapping case-union counts.',
        passed=code == 0 and counts['failures'] == 0 and counts['errors'] == 0
            and guard_after['all_unchanged'] and clean and unchanged_head and env_same)
    write(OUT / 'report.json', report)
    print(json.dumps({key: report[key] for key in ['pytest_exit_code', 'junit_counts', 'elapsed_seconds', 'passed']}), flush=True)
    return 0 if report['passed'] else 1

def main():
    if '--worker' in sys.argv:
        return worker()
    assert git('rev-parse', 'HEAD') == HEAD
    OUT.mkdir(exist_ok=False)
    work = ROOT / 'distribution-check/full-suite-36ec289-py314'
    work.mkdir(exist_ok=False)
    variables = {name: '1' for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OPENMM_CPU_THREADS']}
    variables.update(PYTHONPATH='src:.', PYTHONDONTWRITEBYTECODE='1',
                     TMPDIR=str(work), MPLCONFIGDIR=str(work / 'matplotlib'))
    command = ['nice', '-n', '10', 'taskset', '-c', '6,8', str(PYTHON),
               str(Path(__file__).resolve()), '--worker']
    write(OUT / 'admission.json', dict(schema='pyrpmd.current-head-full-suite-admission.v1',
        created_before_execution_utc=datetime.now(timezone.utc).isoformat(),
        source_head=HEAD, source_tree=git('rev-parse', 'HEAD^{tree}'),
        command=command, environment=variables, affinity=[6, 8],
        whole_process_group_wall_limit_seconds=1800,
        reason='The current-head GitHub ordinary PyRPMD tests gate is still queued. Execute its unfiltered local selection on existing Python 3.14 to check interactions beyond the bounded benchmark-method union.',
        active_long_scientific_workers=4, new_long_scientific_workers=0,
        dependency_install=False, native_rebuild=False))
    start = time.monotonic()
    with (OUT / 'pytest.log').open('x') as stream:
        process = subprocess.Popen(command, cwd=REPO, env=dict(os.environ, **variables),
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            code = process.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait(timeout=5)
    write(OUT / 'execution.json', dict(schema='pyrpmd.current-head-full-suite-execution.v1',
        completed_utc=datetime.now(timezone.utc).isoformat(), exit_code=code,
        timed_out=timed_out, wall_seconds=time.monotonic()-start,
        owned_test_process_reaped=process.poll() is not None,
        log_sha256=sha(OUT / 'pytest.log'), driver_sha256=sha(Path(__file__))))
    print(json.dumps(dict(exit_code=code, timed_out=timed_out, wall_seconds=time.monotonic()-start)), flush=True)
    return code if not timed_out else 124

if __name__ == '__main__':
    raise SystemExit(main())
