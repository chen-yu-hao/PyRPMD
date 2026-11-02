"""Run only the newly frozen method tests using existing native ABI products.

This runner installs/builds nothing, writes no source, and leaves the six
previously documented ABI discovery links untouched. It records actual imports
in the test process and a separately spawned interpreter.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig
import time
from xml.etree import ElementTree

EVIDENCE = Path(__file__).resolve().parent
REPO = EVIDENCE.parents[4]
ORIGINAL = Path('/workspace/scratch/a5e4820daedd/PyRPMD')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def environment():
    return dict(python=sys.version, executable=sys.executable, prefix=sys.prefix,
                platform=platform.platform(), abi=sysconfig.get_config_var('SOABI'),
                distributions={d.metadata['Name']: d.version for d in metadata.distributions()},
                affinity=sorted(os.sched_getaffinity(0)),
                variables={name: os.environ.get(name) for name in [
                    'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE', 'OMP_NUM_THREADS',
                    'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                    'OPENMM_CPU_THREADS', 'TMPDIR', 'MPLCONFIGDIR']})


def guards(before):
    result = {}
    for label, root in [('integrated', REPO), ('original', ORIGINAL)]:
        changed = [name for name, expected in before[label].items()
                   if not (root/name).is_file() or sha(root/name) != expected]
        result[label] = dict(files_checked=len(before[label]), changed=changed)
    changed = [name for name, expected in before['external_models'].items()
               if not Path(name).is_file() or sha(name) != expected]
    result['external_files'] = dict(files_checked=len(before['external_models']), changed=changed)
    link_changes = []
    for row in before['abi_links']:
        path = Path(row['path'])
        if (not path.is_symlink() or str(path.readlink()) != row['target']
                or str(path.resolve()) != row['resolved'] or sha(path) != row['sha256']):
            link_changes.append(str(path))
    result['owned_ABI_links'] = dict(files_checked=6, changed=link_changes)
    result['all_unchanged'] = all(not row['changed'] for row in result.values())
    return result


def runtime(label, selection):
    import pyrpmd
    import openmm
    assert Path(pyrpmd.__file__).resolve() == REPO/'src/pyrpmd/__init__.py'
    assert metadata.version('OpenMM') == '8.6.1'
    native = {}
    for name, expected in selection['native_products'][label].items():
        module = importlib.import_module('pyrpmd.' + name)
        path = Path(module.__file__)
        assert path.resolve() == Path(expected['path']).resolve()
        assert sha(path) == expected['sha256']
        native[name] = dict(path=str(path), resolved=str(path.resolve()), sha256=sha(path))
    return dict(environment=environment(), native=native,
                pyrpmd_import=str(Path(pyrpmd.__file__).resolve()),
                openmm=dict(version=metadata.version('OpenMM'), path=openmm.__file__,
                            platforms=[openmm.Platform.getPlatform(i).getName()
                                       for i in range(openmm.Platform.getNumPlatforms())]))


class RuntimeAudit:
    def __init__(self, label, selection, output):
        self.label, self.selection, self.output = label, selection, output

    def pytest_sessionstart(self, session):
        write(self.output/'pytest-runtime-start.json', runtime(self.label, self.selection))

    def pytest_sessionfinish(self, session, exitstatus):
        snapshot = runtime(self.label, self.selection)
        snapshot['pytest_exitstatus'] = int(exitstatus)
        imported = {}
        for name, module in sorted(sys.modules.items()):
            if not name.startswith(('pyrpmd', 'benchmarks.h3', 'benchmarks.water', 'benchmarks.materials_reference', 'benchmarks.hch4')):
                continue
            filename = getattr(module, '__file__', None)
            if filename and Path(filename).is_file():
                path = Path(filename).resolve()
                imported[name] = dict(path=str(path), sha256=sha(path))
                if path.suffix == '.py':
                    assert path.is_relative_to(REPO), (name, path)
        collected = {}
        for filename in self.selection['per_interpreter'][self.label]['files']:
            expected_test = REPO/filename
            modules = {id(item.module): item.module for item in session.items
                       if Path(item.path).resolve() == expected_test}
            assert len(modules) == 1, filename
            module = next(iter(modules.values()))
            assert Path(module.__file__).resolve() == expected_test
            collected[filename] = dict(path=str(expected_test), sha256=sha(expected_test))
        snapshot['actual_collected_test_modules'] = collected
        snapshot['actual_loaded_project_modules'] = imported
        write(self.output/'pytest-runtime-finish.json', snapshot)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True, choices=['py312', 'py313', 'py314'])
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    selection = json.loads((EVIDENCE/'selection.json').read_text())
    if args.probe:
        print(json.dumps(runtime(args.label, selection), sort_keys=True))
        return 0
    output = EVIDENCE/args.label
    before = json.loads((EVIDENCE/'guard-before.json').read_text())
    requested = selection['per_interpreter'][args.label]
    assert Path.cwd() == REPO and os.environ['PYTHONPATH'] == 'src:.'
    assert sorted(os.sched_getaffinity(0)) == selection['required_cpu_affinity']
    assert sys.version_info[:2] == (3, int(args.label[-2:]))
    assert Path(sys.executable) == Path(requested['python'])
    assert guards(before)['all_unchanged']
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == selection['source_head']
    prior_environment = environment()
    write(output/'environment-before.json', prior_environment)
    started = time.perf_counter()
    report = dict(schema='pyrpmd.ci-fixture-repair-method-gate.v1',
                  started_utc=datetime.now(timezone.utc).isoformat(), label=args.label,
                  source_head=selection['source_head'], source_state_note=selection['source_state_note'],
                  driver_sha256=sha(__file__), selection_sha256=sha(EVIDENCE/'selection.json'),
                  guard_sha256=sha(EVIDENCE/'guard-before.json'),
                  builds=0, dependency_changes=0, ABI_links_modified=False,
                  requested_files=requested['files'], expected_distinct_cases=requested['expected'])
    exit_code = 1
    try:
        command = [sys.executable, str(Path(__file__).resolve()), '--label', args.label, '--probe']
        child = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
        (output/'child-import.log').write_text(child.stderr)
        assert child.returncode == 0, child.stderr
        write(output/'child-import.json', json.loads(child.stdout))
        report['child_import_command'] = command
        import pytest
        work = Path('/workspace/scratch/a5e4820daedd/distribution-check/ci-fixture-repair-0c5fec-v2')/args.label
        pytest_args = ['-q', '-p', 'no:cacheprovider', *requested['files'],
                       '--basetemp='+str(work/'pytest-tmp'), '--junitxml='+str(output/'pytest.xml')]
        report['pytest_arguments'] = pytest_args
        exit_code = int(pytest.main(pytest_args, plugins=[RuntimeAudit(args.label, selection, output)]))
        report['pytest_exit_code'] = exit_code
        xml = ElementTree.parse(output/'pytest.xml')
        report['junit_counts'] = {key: sum(int(s.attrib.get(key, '0')) for s in xml.iter('testsuite'))
                                  for key in ['tests', 'failures', 'errors', 'skipped']}
        case_ids = [case.attrib.get('classname', '')+'::'+case.attrib['name'] for case in xml.iter('testcase')]
        report['case_ids'] = case_ids
        report['distinct_case_ids'] = len(set(case_ids))
        report['selection_count_matches'] = len(case_ids) == len(set(case_ids)) == requested['expected']
        report['junit_sha256'] = sha(output/'pytest.xml')
    except BaseException as error:
        report['driver_error'] = repr(error)
        raise
    finally:
        after = environment()
        write(output/'environment-after.json', after)
        report['environment_distributions_unchanged'] = after['distributions'] == prior_environment['distributions']
        report['guard_after'] = guards(before)
        report['elapsed_seconds'] = time.perf_counter()-started
        report['completed_utc'] = datetime.now(timezone.utc).isoformat()
        report['passed'] = (exit_code == 0 and report.get('selection_count_matches', False)
                            and report['guard_after']['all_unchanged']
                            and report['environment_distributions_unchanged']
                            and report.get('junit_counts', {}).get('skipped', 1) == 0)
        write(output/'compatibility.json', report)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
