"""Seal observed CI repairs and an independently checked source-only audit."""
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree
import hashlib
import json
import shutil

BASE = Path('/workspace/scratch/a5e4820daedd')
REPO = BASE / 'pyrpmd-convergence-integration'
ORIGINAL = BASE / 'PyRPMD'
REL = Path('docs/validation/literature-convergence-2026-11-01')
CI = REPO / REL / 'review/pr68-software-review/ci-0c5fec-20261101T2214Z'
FIRST = REPO / REL / 'integration-4698c12/ci-fixture-repair-0c5fec'
GATE = FIRST.with_name(FIRST.name + '-v2')
SOURCE = REPO / REL / 'review/hele2015-source-index'
AUDIT = REPO / REL / 'integration-4698c12/sixth-stage-adoptions'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')

def identity(path):
    return {'path': str(path.relative_to(REPO)), 'sha256': sha(path), 'bytes': path.stat().st_size}

def inventory(folder, *, extra=(), note=''):
    paths = sorted([p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts] + list(extra))
    assert not any(p.name == 'publication-inventory.json' for p in paths)
    value = {'schema': 'pyrpmd.compact-sixth-publication-inventory.v1',
             'created_utc': datetime.now(timezone.utc).isoformat(), 'scope': note,
             'files': [identity(p) for p in paths], 'listed_bytes': sum(p.stat().st_size for p in paths)}
    write(folder / 'publication-inventory.json', value)
    return identity(folder / 'publication-inventory.json')

AUDIT.mkdir(exist_ok=False)
execution = json.loads((GATE / 'execution.json').read_text())
assert execution['passed'] is True and execution['all_three_complete'] is True
ids, records = {}, {}
for label in ('py312', 'py313', 'py314'):
    report = json.loads((GATE / label / 'compatibility.json').read_text())
    launch = json.loads((GATE / label / 'launch-receipt.json').read_text())
    assert report['passed'] and report['guard_after']['all_unchanged'] and report['environment_distributions_unchanged']
    assert launch['passed'] and launch['exit_code'] == 0 and launch['owned_test_group_reaped'] and not launch['timed_out']
    tree = ElementTree.parse(GATE / label / 'pytest.xml')
    counts = {k: sum(int(s.attrib.get(k, '0')) for s in tree.iter('testsuite')) for k in ('tests', 'failures', 'errors', 'skipped')}
    assert counts == {'tests': 101, 'failures': 0, 'errors': 0, 'skipped': 0}
    ids[label] = sorted(c.attrib.get('classname', '') + '::' + c.attrib['name'] for c in tree.iter('testcase'))
    assert len(ids[label]) == len(set(ids[label])) == 101
    records[label] = {'counts': counts, 'report': identity(GATE / label / 'compatibility.json'),
                      'launch': identity(GATE / label / 'launch-receipt.json')}
assert ids['py312'] == ids['py313'] == ids['py314']
write(GATE / 'root-exec-observation.json', {
    'schema': 'pyrpmd.root-actual-exec-observation.v1', 'exec_session_id': 4174,
    'recorded_after_observation_utc': datetime.now(timezone.utc).isoformat(),
    'tool_result': {'chunk_id': '392cd5', 'wall_time_seconds': 0.000041946, 'exit_code': 0,
                    'output': '{"label": "py313", "exit_code": 0, "wall_seconds": 11.098864708008477, "passed": true}\n{"label": "py314", "exit_code": 0, "wall_seconds": 10.343366978995618, "passed": true}\n'},
    'prior_yield_in_same_session': {'chunk_id': 'aa90a5', 'output': '{"label": "py312", "exit_code": 0, "wall_seconds": 10.995878993999213, "passed": true}\n'}})
write(GATE / 'summary.json', {'schema': 'pyrpmd.ci-fixture-repair-summary.v1',
    'all_three_interpreters_passed': True, 'distinct_cases_per_interpreter': 101,
    'new_distinct_test_identities': 0, 'same_case_membership': True, 'records': records,
    'case_ids': ids, 'first_gate_stopped_before_collection': identity(FIRST / 'preflight-guard-failure.json'),
    'first_gate_cause': 'Inherited fourth-head campaign aggregate hash; the second preparation verified current aggregate bytes against published commit0c5fec before recording that already-published hash.',
    'full_suite_rerun': False, 'physical_PES_evaluations': 0, 'real_dynamics_steps': 0,
    'scientific_runtime_or_acceptance_criteria_modified': False})
for source, name in ((BASE / 'prepare_ci_fixture_gate_v2.py', 'prepare_gate.py'),
                     (BASE / 'run_ci_fixture_gate_v2.py', 'run_sequential.py')):
    shutil.copy2(source, GATE / name)
write(CI / 'fixture-supersession.json', {
    'schema': 'pyrpmd.ci-test-fixture-only-supersession.v1',
    'created_utc': datetime.now(timezone.utc).isoformat(), 'published_failed_head': '0c5fec22ce590634c45b8c46619cebf3e665eb85',
    'changes': [{
        'path': 'tests/' + filename, 'before': identity(CI / ('before-' + filename)),
        'after': identity(REPO / 'tests' / filename),
        'original_frozen_tree_test_sha256': sha(ORIGINAL / 'tests' / filename), 'reason': reason}
        for filename, reason in (
            ('test_hch4_efficiency_pilot.py', 'Make inert project and output siblings inside the fixture; production output containment rule remains unchanged.'),
            ('test_h3_p256_native_calibration.py', 'Use exact synthetic zero clock origin and a local time object; avoid addition/subtraction roundoff at the115s boundary and global time monkeypatching.'))],
    'deterministic_before_fix_reproduction': identity(CI / 'before-fix-reproduction.json'),
    'three_interpreter_full_module_gate': identity(GATE / 'summary.json'),
    'historical_inventories_remain_original': True, 'scientific_source_changes': [],
    'new_test_ids': 0, 'retained_full_failed_job_logs': [identity(p) for p in sorted(CI.glob('job-*.log'))]})
shutil.copy2(BASE / 'reproduce_ci_fixture_failures.py', CI / 'reproduce_before_fix.py')
(CI / 'README.md').write_text('''# CI failures at commit 0c5fec and fixture repairs

Five complete failed job logs are retained: four Python CPU jobs and one fused
native propagation job. Every job exposed the inert HCH4 fixture's assumption
that pytest temporary directories live under the repository workspace; the
Python3.14 job additionally exposed clock-origin floating-point cancellation.
Both defects were deterministically reproduced without any PES or dynamics.

The repair changes only the two test fixtures. The live output-containment
check and the real calibration resource deadlines are unchanged. All101 cases
from these two modules pass on each of Python3.12,3.13 and3.14, with zero failures,
errors or skips; this is focused validation, not a new whole-suite claim.
The first local gate stopped before collecting tests because it inherited an
older campaign-summary hash. That failure is retained separately; its successor
verified the summary against the already-published commit before testing.

Historical inventories keep their original test hashes. The supersession
receipt links exact old bytes, repaired bytes and the new passing gate; it does
not rewrite a prior scientific run's provenance or acceptance criteria.
''')
gate_inv = inventory(GATE, note='Three complete focused101-case runs; no new distinct identities and no physical execution.')
first_inv = inventory(FIRST, note='Retained initial local preflight failure; no pytest cases were collected.')
ci_inv = inventory(CI, note='Complete five failed CI job logs, exact original tests, deterministic reproduction and test-only supersession; generated logs preserved verbatim.')

rel = REL / 'review/hele2015-source-review'
inv_source = ORIGINAL / rel / 'publication-inventory.json'
assert sha(inv_source) == '6844344e4ef02504c34982e316173fb684f850f1921aaf0f07ae810733113101'
review_inventory = json.loads(inv_source.read_text())
for row in review_inventory['files'] + [{'path': str(rel / 'publication-inventory.json'), 'sha256': sha(inv_source), 'bytes': inv_source.stat().st_size}]:
    source, target = ORIGINAL / row['path'], REPO / row['path']
    assert source.stat().st_size == row['bytes'] and sha(source) == row['sha256']
    assert not target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    assert sha(target) == row['sha256']
review = json.loads((REPO / rel / 'independent-review.json').read_text())
assert review['approved_for_source_audit_only'] and not review['physical_comparison_eligible']
for row in review['root_bundle_inputs']:
    path = Path(row['path'])
    assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
misplaced = SOURCE / 'hch4-normalization-copy-observation.json'
expected = sha(misplaced)
misplaced.rename(AUDIT / misplaced.name)
assert sha(AUDIT / misplaced.name) == expected
write(SOURCE / 'independent-check-binding.json', {
    'schema': 'pyrpmd.hele2015-source-audit-independent-binding.v1',
    'review': identity(REPO / rel / 'independent-review.json'),
    'inventory': identity(REPO / rel / 'publication-inventory.json'),
    'review_scope': 'F+H2 Fig7 only: independent PDFMiner/pdfplumber coordinates and stdlib log calibration; other three graphs were not independently redigitized.',
    'maximum_independent_reader_relative_difference': review['agreement']['maximum_relative_difference_between_independent_PDF_readers'],
    'graph_coordinate_sensitivity_is_statistical_CI': False,
    'physical_comparison_eligible': False, 'literature_reproduced': False})
source_inv = inventory(SOURCE, extra=(REPO / 'benchmarks/literature/hele2015_digitization.py',),
    note='Offline original-vector source digitization and conditions, with independent F+H2 extraction; no copyrighted source PDF/TeX, no new molecular result, no changed registered reference.')
write(AUDIT / 'root-sealing-receipt.json', {'schema': 'pyrpmd.sixth-new-evidence-sealing.v1',
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'inventories': [gate_inv, first_inv, ci_inv, source_inv, identity(REPO / rel / 'publication-inventory.json')],
    'copied_independent_review_files': len(review_inventory['files']),
    'existing_original_bundles_overwritten': 0, 'scientific_sources_mutated': False})
print(json.dumps({'sealed_inventories': 5, 'focused_cases_each': 101, 'three_interpreters_passed': True,
                  'source_graph_physical_comparison_eligible': False, 'root_receipt': identity(AUDIT / 'root-sealing-receipt.json')}))
