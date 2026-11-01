"""Seal the completed65-case delta without recounting overlapping tests."""
from pathlib import Path
from datetime import datetime, timezone
from xml.etree import ElementTree
import hashlib
import json
import shutil

ROOT = Path('/workspace/scratch/a5e4820daedd')
REPO = ROOT / 'pyrpmd-convergence-integration'
BASE = REPO / 'docs/validation/literature-convergence-2026-11-01/integration-4698c12'
OUT = BASE / 'archive-method-compatibility'
PRIOR = BASE / 'fifth-stage-compatibility'


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
        assert counts == dict(tests=65, failures=0, errors=0, skipped=0)
        assert len(ids) == len(set(ids)) == 65
        old = prior['case_ids'][label]
        assert len(old) == len(set(old)) == 731 and not set(old).intersection(ids)
        unions[label] = sorted(set(old).union(ids))
        reports[label] = {
            'new_distinct_cases': 65, 'prior_distinct_case_receipts_reused': 731,
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
        'note': 'Exactly65 newly selected identities unioned with731 prior identities on each interpreter; repeated focused or full-suite executions do not increase this union.',
        'prior_coverage_path': selection['prior_coverage']['path'],
        'prior_coverage_sha256': selection['prior_coverage']['sha256'],
        'case_ids': unions,
    })
    summary = {
        'schema': 'pyrpmd.archive-method-compatibility-summary.v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_head': selection['source_head'],
        'source_state_note': selection['source_state_note'],
        'scope': selection['test_scope'],
        'distinct_new_cases': 65, 'new_case_executions': 195,
        'combined_distinct_cases_per_interpreter': 796,
        'same_distinct_cases_all_interpreters': True,
        'interpreters': reports,
        'selection_sha256': sha(OUT / 'selection.json'),
        'driver_sha256': sha(OUT / 'run_checks.py'),
        'guard_before_sha256': sha(OUT / 'guard-before.json'),
        'case_coverage_sha256': sha(OUT / 'case-coverage.json'),
        'previous_731_cases_rerun': False,
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
    for source, name in [(ROOT / 'prepare_archive_method_checks.py', 'prepare_gate.py'),
                         (ROOT / 'run_archive_method_checks.py', 'run_sequential.py'),
                         (Path(__file__), 'seal_gate.py')]:
        assert not (OUT / name).exists()
        shutil.copy2(source, OUT / name)
    (OUT / 'README.md').write_text('# Archive contracts on Python 3.12–3.14\n\nAll **65 newly selected distinct cases pass on each of Python 3.12.14,\n3.13.18 and 3.14.10**, with zero failures, errors or skips. These are the\n28 original archive lifecycle/exact-replay cases and 37 additive explicit\nread-only lease-anomaly admission cases. Their case-level union with earlier\nunchanged evidence is **796 distinct passing identities per interpreter**.\nThe earlier 731 cases were not rerun. Repeated focused and full-suite runs\ndo not add identities to this union.\n\nThese are inert fixtures: no actual campaign checkpoint, live archive,\nscientific analyzer or physical PES was executed. The separate actual\n100,000-sweep archive and replay retain their independent execution records.\n\nActual test-process and fresh-child imports resolve to this integration\ncheckout and its verified existing ABI products. All recorded sources,\nexternal prerequisites, environments and six ABI links are unchanged.\nNo installation or native build occurred. The prior full-project Python 3.14\ngate predates these additive archive modules; this receipt does not extend\nthat earlier full-suite scope.\n\nSee [summary](summary.json), [case identities](case-coverage.json),\n[selection](selection.json), and [immutable copy receipt](copy-receipt.json).\n')
    files = [{'path': str(p.relative_to(REPO)), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(OUT.rglob('*')) if p.is_file()]
    write_new(OUT / 'publication-inventory.json', {
        'schema': 'pyrpmd.compatibility.publication-inventory.v1',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'files': files,
        'inventory_excludes_itself': True,
    })
    print(json.dumps({'inventory_sha256': sha(OUT / 'publication-inventory.json'), 'files': len(files), 'total_bytes': sum(r['bytes'] for r in files), 'new_cases_each': 65, 'case_union_each': 796}))


if __name__ == '__main__':
    main()
