"""Freeze the actual 45-case additions and their union with earlier receipts."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
from xml.etree import ElementTree

BASE = Path('/workspace/scratch/a5e4820daedd')
REPO = BASE / 'pyrpmd-convergence-integration'
PARENT = REPO / 'docs/validation/literature-convergence-2026-11-01/integration-4698c12/new-stage-compatibility'
OUT = PARENT.parent / 'materials-preparation-compatibility'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + '\n')


def main():
    old = json.loads((PARENT / 'case-coverage.json').read_text())
    prior_summary = json.loads((PARENT / 'summary.json').read_text())
    selection = json.loads((OUT / 'selection.json').read_text())
    summaries, unions = {}, {}
    for label in ['py312', 'py313', 'py314']:
        folder = OUT / label
        report = json.loads((folder / 'compatibility.json').read_text())
        if not report['passed'] or not report['guard_after']['all_unchanged']:
            raise ValueError('A new gate did not pass')
        tree = ElementTree.parse(folder / 'pytest.xml')
        suites = list(tree.iter('testsuite'))
        counts = {key: sum(int(s.attrib.get(key, '0')) for s in suites)
                  for key in ['tests', 'failures', 'errors', 'skipped']}
        ids = [c.attrib.get('classname', '') + '::' + c.attrib['name'] for c in tree.iter('testcase')]
        if counts != {'tests': 45, 'failures': 0, 'errors': 0, 'skipped': 0} or len(set(ids)) != 45:
            raise ValueError('Actual XML differs from the fixed case selection')
        if set(ids) & set(old['case_ids'][label]) or len(old['case_ids'][label]) != 270:
            raise ValueError('Old/new case sets overlap or prior coverage differs')
        union = sorted(set(ids) | set(old['case_ids'][label]))
        if len(union) != 315:
            raise ValueError('Combined case count differs')
        unions[label] = union
        families = {prefix: sum(c.startswith(prefix) for c in ids) for prefix in
                    ['tests.test_materials_reference_eam4::', 'tests.test_water_preparation_coordinator::']}
        summaries[label] = {
            'new_distinct_cases': 45, 'prior_distinct_cases_reused': 270,
            'combined_distinct_passed_cases': 315, 'new_case_families': families,
            'junit_counts': counts, 'pytest_seconds': sum(float(s.attrib['time']) for s in suites),
            'receipt_elapsed_seconds': report['elapsed_seconds'],
            'receipt_sha256': sha(folder / 'compatibility.json'),
            'junit_sha256': sha(folder / 'pytest.xml'),
            'guard_after': report['guard_after'],
            'actual_native_imports_checked_in_test_and_child': True,
            'environment_distributions_unchanged': report['environment_distributions_unchanged'],
            'previous_270_claim_source_sha256': sha(PARENT / 'summary.json'),
            'previous_270_source_state': prior_summary['source_state_note'],
        }
    if not unions['py312'] == unions['py313'] == unions['py314']:
        raise ValueError('The three interpreter case unions differ')
    write(OUT / 'case-coverage.json', {
        'schema': 'pyrpmd.method-case-coverage-union.v1',
        'note': 'Union of exactly 45 newly executed cases per interpreter and 270 unchanged earlier distinct passing case identities. Historical failures/retries remain in their original receipts.',
        'prior_coverage_path': str((PARENT / 'case-coverage.json').relative_to(REPO)),
        'prior_coverage_sha256': sha(PARENT / 'case-coverage.json'), 'case_ids': unions})
    summary = {
        'schema': 'pyrpmd.materials-preparation-compatibility-summary.v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_head': selection['source_head'], 'source_state_note': selection['source_state_note'],
        'scope': selection['test_scope'], 'selection_sha256': sha(OUT / 'selection.json'),
        'driver_sha256': sha(OUT / 'run_checks.py'), 'source_guard_sha256': sha(OUT / 'guard-before.json'),
        'case_coverage_sha256': sha(OUT / 'case-coverage.json'),
        'distinct_new_cases': 45, 'new_case_executions': 135,
        'combined_distinct_cases_per_interpreter': 315,
        'same_distinct_cases_all_interpreters': True, 'interpreters': summaries,
        'previous_270_cases_rerun': False, 'dependency_changes': False, 'builds': 0,
        'ABI_links_modified': False, 'active_original_sources_modified': False,
        'full_repository_suite_run': False, 'long_dynamics_run': False,
        'source_bundles': selection['frozen_inventories'],
        'external_EAM4_library': selection['exact_EAM4_library'],
    }
    write(OUT / 'summary.json', summary)
    for original, name in [(BASE / 'prepare_materials_preparation_gate.py', 'prepare_gate.py'),
                           (Path(__file__), 'summarize_gate.py')]:
        destination = OUT / name
        if destination.exists():
            raise FileExistsError(destination)
        shutil.copyfile(original, destination)
    print(json.dumps({label: {k:v for k,v in row.items() if k in ['junit_counts','pytest_seconds','combined_distinct_passed_cases']} for label,row in summaries.items()}))


if __name__ == '__main__':
    main()
