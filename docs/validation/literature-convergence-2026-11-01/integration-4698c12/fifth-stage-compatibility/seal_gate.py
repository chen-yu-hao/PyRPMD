"""Seal the completed135-case delta without recounting overlapping tests."""
from pathlib import Path
from datetime import datetime, timezone
from xml.etree import ElementTree
import hashlib
import json
import shutil

ROOT = Path('/workspace/scratch/a5e4820daedd')
REPO = ROOT / 'pyrpmd-convergence-integration'
BASE = REPO / 'docs/validation/literature-convergence-2026-11-01/integration-4698c12'
OUT = BASE / 'fifth-stage-compatibility'
PRIOR = BASE / 'auxiliary-proof-compatibility-v2'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def main():
    selection = json.loads((OUT / 'selection.json').read_text())
    execution = json.loads((OUT / 'execution.json').read_text())
    assert execution['passed'] and execution['all_three_complete']
    prior = json.loads((PRIOR / 'case-coverage.json').read_text())
    assert sha(PRIOR / 'case-coverage.json') == selection['prior_coverage']['sha256']
    reports, unions = {}, {}
    for label in ('py312', 'py313', 'py314'):
        folder = OUT / label
        report = json.loads((folder / 'compatibility.json').read_text())
        launch = json.loads((folder / 'launch-receipt.json').read_text())
        assert report['passed'] and report['guard_after']['all_unchanged']
        assert launch['passed'] and launch['owned_test_group_reaped']
        assert not launch['timed_out'] and launch['exit_code'] == 0
        tree = ElementTree.parse(folder / 'pytest.xml')
        counts = {k: sum(int(s.attrib.get(k, '0')) for s in tree.iter('testsuite')) for k in ('tests', 'failures', 'errors', 'skipped')}
        ids = [c.attrib.get('classname', '') + '::' + c.attrib['name'] for c in tree.iter('testcase')]
        assert counts == dict(tests=135, failures=0, errors=0, skipped=0)
        assert len(ids) == len(set(ids)) == 135
        old = prior['case_ids'][label]
        assert len(old) == len(set(old)) == 596 and not set(old).intersection(ids)
        unions[label] = sorted(set(old).union(ids))
        reports[label] = {
            'new_distinct_cases': 135, 'prior_distinct_case_receipts_reused': 596,
            'combined_distinct_cases': len(unions[label]), 'junit_counts': counts,
            'pytest_seconds': sum(float(s.attrib['time']) for s in tree.iter('testsuite')),
            'supervised_wall_seconds': launch['wall_seconds'],
            'receipt_sha256': sha(folder / 'compatibility.json'),
            'junit_sha256': sha(folder / 'pytest.xml'),
            'guard_after': report['guard_after'],
            'native_imports_checked_in_actual_test_process_and_fresh_child': True,
            'dependencies_unchanged': report['environment_distributions_unchanged'],
        }
    assert unions['py312'] == unions['py313'] == unions['py314']
    write_new(OUT / 'case-coverage.json', {
        'schema': 'pyrpmd.method-case-coverage-union.v1',
        'note': 'Exactly135 newly selected identities unioned with596 prior identities on each interpreter; repeated focused or full-suite executions do not increase this union.',
        'prior_coverage_path': selection['prior_coverage']['path'],
        'prior_coverage_sha256': selection['prior_coverage']['sha256'],
        'case_ids': unions,
    })
    summary = {
        'schema': 'pyrpmd.fifth-stage-compatibility-summary.v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_head': selection['source_head'],
        'source_state_note': selection['source_state_note'],
        'scope': selection['test_scope'],
        'distinct_new_cases': 135, 'new_case_executions': 405,
        'combined_distinct_cases_per_interpreter': 731,
        'same_distinct_cases_all_interpreters': True,
        'interpreters': reports,
        'selection_sha256': sha(OUT / 'selection.json'),
        'driver_sha256': sha(OUT / 'run_checks.py'),
        'guard_before_sha256': sha(OUT / 'guard-before.json'),
        'case_coverage_sha256': sha(OUT / 'case-coverage.json'),
        'previous_596_cases_rerun': False,
        'prior_HCH4_wall_fixture_change': {
            'supersession_sha256': '42065fe6c3f8833f0c57929603e40c475317b07e5c27dce745db2f482ba1fead',
            'cross_python_inventory_sha256': 'ee0f53335877c9493a9bab111bc79c4ebdb0d0d65d97f845e6c2118295acdbdc',
            'added_distinct_cases': 0,
        },
        'builds': 0, 'installs': 0, 'ABI_links_modified': False,
        'active_original_sources_modified': False,
        'full_repository_suite_run_by_this_gate': False,
        'physical_PES_evaluations_or_dynamics_steps_by_this_gate': 0,
        'source_bundles': selection['frozen_inventories'],
    }
    write_new(OUT / 'summary.json', summary)
    for source, name in [(ROOT / 'prepare_fifth_stage_checks.py', 'prepare_gate.py'),
                         (ROOT / 'run_fifth_stage_checks.py', 'run_sequential.py'),
                         (Path(__file__), 'seal_gate.py')]:
        assert not (OUT / name).exists()
        shutil.copy2(source, OUT / name)
    (OUT / 'README.md').write_text('''# Additive method contracts on Python3.12–3.14

All **135 newly selected distinct cases pass on each of Python 3.12.14,
3.13.18 and 3.14.10**, with zero failures, errors or skips. They comprise
43 complete-P256-analysis admission cases,85 HCH4 proposal-efficiency cases,
and7 analytic VACF–MSD identity cases. The case-level union with earlier
unchanged evidence is **731 distinct passing identities per interpreter**.
The earlier596 cases were not rerun by this gate. The previously recorded
wall-fixture correction was separately exercised on all three interpreters
and adds no distinct identity.

These are synthetic mathematical/lifecycle checks and inert process tests.
They make no physical PES call, launch no actual reaction trajectory and
perform no current campaign analysis. Separate actual physical executions
and their failed scientific predicates remain separate evidence.

Actual test-process and fresh-child imports resolve to this integration
checkout and its existing verified native ABI products. All selected
integrated/original sources, external models and six ABI links remain
unchanged. No installation or native rebuild occurred. The newly copied
immutable bundles and the sole previously validated test-fixture delta are
explicit in the selection and copy receipts. The full-project Python3.14
gate had already completed before these new source files were copied.

The original water cohort's approved `curves.npz` is a compact derived
correlation product. It contains no raw coordinate/momentum/checkpoint
trajectory. Full raw archives remain separate, with extraction/replay
proofs and hashes recorded in the corresponding evidence.

See [summary](summary.json), [exact case identities](case-coverage.json),
[selection](selection.json), and the [copy receipt](copy-receipt.json).
'''.replace('Python3.12', 'Python 3.12').replace('cases,85', 'cases, 85').replace('and7 ', 'and 7 ').replace('earlier596', 'earlier 596').replace('Python3.14', 'Python 3.14'))
    files = [{'path': str(p.relative_to(REPO)), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(OUT.rglob('*')) if p.is_file()]
    write_new(OUT / 'publication-inventory.json', {
        'schema': 'pyrpmd.compatibility.publication-inventory.v1',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'files': files,
        'inventory_excludes_itself': True,
    })
    print(json.dumps({'inventory_sha256': sha(OUT / 'publication-inventory.json'), 'files': len(files), 'total_bytes': sum(r['bytes'] for r in files), 'new_cases_each': 135, 'case_union_each': 731}))


if __name__ == '__main__':
    main()
