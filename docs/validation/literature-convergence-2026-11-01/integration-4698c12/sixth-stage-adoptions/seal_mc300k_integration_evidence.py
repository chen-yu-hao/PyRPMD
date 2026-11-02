from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree
import hashlib
import json
import shutil

BASE = Path('/workspace/scratch/a5e4820daedd')
ORIGINAL = BASE / 'PyRPMD'
REPO = BASE / 'pyrpmd-convergence-integration'
REL = Path('docs/validation/literature-convergence-2026-11-01')
FIRST = REPO / REL / 'integration-4698c12/mc300k-owner-compatibility'
OUT = FIRST.with_name(FIRST.name + '-v2')
ADOPTION = REPO / REL / 'h3-quantum/umbrella-isolated300k-execution-v1'
TEST = 'tests/test_h3_quantum_isolated_continuation.py'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def identity(path):
    return {'path': str(path.relative_to(REPO)), 'bytes': path.stat().st_size, 'sha256': sha(path)}

def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')

def inventory(folder, scope):
    paths = sorted(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    assert not any(p.name == 'publication-inventory.json' for p in paths)
    write(folder / 'publication-inventory.json', {'schema': 'pyrpmd.mc300k-compact-publication-inventory.v1',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'scope': scope,
        'files': [identity(p) for p in paths], 'listed_bytes': sum(p.stat().st_size for p in paths)})
    return identity(folder / 'publication-inventory.json')

review_rel = REL / 'review/pr68-software-review/mc300k-pidfd-capability-review.json'
assert sha(ORIGINAL / review_rel) == '615d3024ead3319ea833428d876c4134ef21b151e9e9615b75152a9ec2d85260'
assert not (REPO / review_rel).exists()
shutil.copy2(ORIGINAL / review_rel, REPO / review_rel)
selection = json.loads((OUT / 'selection.json').read_text())
execution = json.loads((OUT / 'execution.json').read_text())
assert execution['passed'] and execution['all_three_complete']
reports, passing, unavailable = {}, {}, {}
for label in ('py312', 'py313', 'py314'):
    report = json.loads((OUT / label / 'compatibility.json').read_text())
    launch = json.loads((OUT / label / 'launch-receipt.json').read_text())
    expected = selection['per_interpreter'][label]
    assert report['passed'] and launch['passed'] and launch['owned_test_group_reaped']
    assert launch['exit_code'] == 0 and report['guard_after']['all_unchanged']
    assert report['environment_distributions_unchanged'] and report['capability_coverage_matches_fixed_selection']
    assert report['actual_skipped_case_ids'] == expected['expected_skipped_case_ids']
    assert report['available_passed_cases'] == expected['expected_passed']
    assert report['full_method_coverage'] == expected['full_method_coverage_expected']
    counts = report['junit_counts']
    assert counts == {'tests': 76, 'failures': 0, 'errors': 0, 'skipped': len(expected['expected_skipped_case_ids'])}
    unavailable[label] = report['actual_skipped_case_ids']
    passing[label] = sorted(set(report['case_ids']) - set(unavailable[label]))
    reports[label] = {'counts': counts, 'passing_cases': len(passing[label]),
        'full_method_coverage': report['full_method_coverage'],
        'report': identity(OUT / label / 'compatibility.json'),
        'launch': identity(OUT / label / 'launch-receipt.json')}
prior_path = REPO / REL / 'integration-4698c12/archive-method-compatibility/case-coverage.json'
prior = json.loads(prior_path.read_text())
unions = {}
for label in passing:
    old = prior['case_ids'][label]
    assert len(old) == len(set(old)) == 796 and not set(old).intersection(passing[label])
    unions[label] = sorted(set(old) | set(passing[label]))
assert {k: len(v) for k, v in unions.items()} == {'py312': 872, 'py313': 865, 'py314': 865}
write(OUT / 'case-coverage.json', {'schema': 'pyrpmd.method-case-coverage-union.v2',
    'prior_coverage': identity(prior_path), 'case_ids': unions,
    'new_passed_case_ids': passing, 'unavailable_case_ids': unavailable,
    'skips_are_not_passes': True, 'repaired_existing101_tests_do_not_add_identities': True})
write(OUT / 'root-exec-observation.json', {'schema': 'pyrpmd.root-actual-exec-observation.v1',
    'exec_session_id': 52956, 'recorded_after_observation_utc': datetime.now(timezone.utc).isoformat(),
    'tool_result': {'chunk_id': 'd35510', 'exit_code': 0, 'wall_time_seconds': 0.000026039,
        'output': '{"label": "py312", "exit_code": 0, "wall_seconds": 5.249456135003129, "passed": true}\n{"label": "py313", "exit_code": 0, "wall_seconds": 2.679044825010351, "passed": true}\n{"label": "py314", "exit_code": 0, "wall_seconds": 2.9834116010024445, "passed": true}\n'},
    'passed_field_scope': 'Every available test passes and exact predeclared capability skips match; only3.12 has complete76-case ownership coverage.'})
write(OUT / 'fixture-supersession.json', {'schema': 'pyrpmd.mc300k-pidfd-test-capability-supersession.v1',
    'created_utc': datetime.now(timezone.utc).isoformat(), 'current_path': TEST,
    'before': identity(FIRST / 'before-test_h3_quantum_isolated_continuation.py'),
    'after': identity(REPO / TEST), 'original_active_test_sha256': sha(ORIGINAL / TEST),
    'independent_review': identity(REPO / review_rel),
    'reason': 'Only seven real ownership tests require two optional Linux interfaces absent from the observed local conda3.13/3.14 builds. All function bodies and remaining69 tests are unchanged; the pre-Popen guard variant still runs.',
    'retained_initial_failure': identity(FIRST / 'py313/pytest.log'),
    'initial_false_positive_cases_not_counted_as_valid_coverage': [
        'test_stop_or_event_error_never_admits_remaining_workers[signal]',
        'test_stop_or_event_error_never_admits_remaining_workers[journal]'],
    'scientific_source_or_runtime_changed': False, 'new_distinct_test_identities': 0,
    'required_original_physical_runtime_has_full_coverage': True})
write(OUT / 'summary.json', {'schema': 'pyrpmd.mc300k-capability-aware-summary.v1',
    'records': reports, 'all_available_cases_pass': True,
    'full76_case_coverage_on_every_interpreter': False,
    'new_complete_ownership_coverage': ['py312'], 'known_unavailable_ownership_cases_each_conda_interpreter': 7,
    'combined_distinct_passing_identities': {k: len(v) for k, v in unions.items()},
    'original_scientific_sources_models_acceptance_criteria_unchanged': True,
    'real_PES_evaluations': 0, 'real_sampling_or_dynamics_steps': 0,
    'whole_repository_suite_rerun': False, 'GitHub_official_runtime_pidfd_coverage_pending': True})
(OUT / 'README.md').write_text('''# MC continuation ownership checks across Python runtimes

Python3.12 executes all76 cases successfully, including real owned-process
capture, kernel resource caps, signal-safe cleanup and complete-cohort checks.
The observed local conda-forge3.13 and3.14 builds expose neither `os.pidfd_open`
nor `signal.pidfd_send_signal`: each passes69 applicable cases and explicitly
skips the same7 ownership cases. These skips are unavailable coverage, not
passes. Full ownership compatibility on those interpreters is not claimed.

The initial unmarked3.13 run is retained:71 nominal passes and5 failures.
Two error-path cases had also passed for the wrong reason before their injected
event was reached. They are among the seven now capability-marked cases.
The source-guard case, which stops before any child starts, remains runnable.
All test bodies and the active3.12 scientific source/test bytes are unchanged.

The separate101-case repair gate passed without skips on all three runtimes.
Combining nonoverlapping new passing identities with the prior796 gives872
on3.12 and865 on each local conda interpreter. Repeated executions do not
inflate these counts. The new original scientific run uses the fully tested
3.12 environment; no physical work occurred in these integration checks.

Python documents the platform-specific interfaces at
[os.pidfd_open](https://docs.python.org/3.13/library/os.html#os.pidfd_open) and
[signal.pidfd_send_signal](https://docs.python.org/3.13/library/signal.html#signal.pidfd_send_signal).
The local attribute observations establish this build's availability; they
do not imply the same absence on GitHub's official Python builds.
''')
for source, folder, name in (
    (BASE / 'prepare_mc300k_integration_gate.py', FIRST, 'prepare_gate.py'),
    (BASE / 'run_mc300k_integration_gate.py', FIRST, 'run_sequential.py'),
    (BASE / 'prepare_mc300k_capability_gate.py', OUT, 'prepare_gate.py'),
    (BASE / 'run_mc300k_capability_gate.py', OUT, 'run_sequential.py')):
    shutil.copy2(source, folder / name)
(ADOPTION / 'README.md').write_text('''# Adopted fixed300k MC stage: started, result pending

Root adopted the exact independently reviewed source, final76-case inert gate,
unchanged scientific proposal and current joint CPU/memory/storage reservation.
The exec handoff is session15483. The startup observation records the first four
owned workers and their positive resource heartbeats; it reads no intermediate
MC estimate and makes no completion or convergence claim.

The only active copy is a fresh, verified restore of the complete100k archive.
Every registered seed continues to the original fixed300000 production endpoint.
Original raw data, prior verified extractions, the anomalous original lease,
RNG prefixes and all failed precision results remain preserved. No intermediate
rate can stop, extend or select this stage. Analysis requires a later actual
normal terminal observation, complete eight-chain verification and confirmed
owned-worker drain. This snapshot contains adoption/startup evidence only.
''')
first_inv = inventory(FIRST, 'Retained initial local full3.12 pass and unsupported-interface3.13 failures; no physical work.')
out_inv = inventory(OUT, 'Fixed0/7/7 capability-aware gate, exact skipped identities, independent test-only supersession and honest coverage union.')
adoption_inv = inventory(ADOPTION, 'Actual root resource/adoption/exec/startup metadata only; the running MC result is pending.')
print(json.dumps({'inventories': [first_inv, out_inv, adoption_inv],
    'distinct_passing_counts': {k: len(v) for k, v in unions.items()},
    'source_code_changed': False, 'full_three_interpreter_ownership_coverage': False}))
