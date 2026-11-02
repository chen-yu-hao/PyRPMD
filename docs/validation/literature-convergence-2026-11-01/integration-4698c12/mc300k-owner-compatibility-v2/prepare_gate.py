from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

BASE = Path('/workspace/scratch/a5e4820daedd')
REPO = BASE / 'pyrpmd-convergence-integration'
PRIOR = REPO / 'docs/validation/literature-convergence-2026-11-01/integration-4698c12/mc300k-owner-compatibility'
OUT = PRIOR.with_name(PRIOR.name + '-v2')
TEST = 'tests/test_h3_quantum_isolated_continuation.py'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')

assert sha(REPO / TEST) == 'da0b38029eb88ee273dfba2de708f759ca079b36ef981620f1238e6796a1d980'
assert sha(BASE / 'PyRPMD' / TEST) == '69df0a97c06992db5c24e074abee9866474ae4f02a90b8e9f6fe73d1d073ef05'
names = {
    'test_full_inert_cohort_preserves_order_once_and_two_worker_ceiling',
    'test_alarm_between_Popen_and_active_capture_drains_registered_child',
    'test_stop_or_event_error_never_admits_remaining_workers[signal]',
    'test_stop_or_event_error_never_admits_remaining_workers[journal]',
    'test_unconfirmed_drain_retains_exclusion_and_attempt',
    'test_alarm_during_frozen_exceptional_drain_is_independently_reaped',
    'test_real_capped_worker_self_heartbeat_is_accepted_before_next_admission',
}
prior_report = json.loads((PRIOR / 'py312/compatibility.json').read_text())
required_ids = sorted(x for x in prior_report['case_ids'] if x.rsplit('::', 1)[-1] in names)
assert len(required_ids) == 7
availability = json.loads((PRIOR / 'pidfd-availability.json').read_text())
OUT.mkdir(exist_ok=False)
selection = json.loads((PRIOR / 'selection.json').read_text())
selection.update(schema='pyrpmd.mc300k-capability-aware-selection.v1',
    created_before_execution_utc=datetime.now(timezone.utc).isoformat(),
    source_state_note='Published head plus exact frozen MC owner and integration-only capability marks on seven real-pidfd test identities; original active source/tests remain unchanged.',
    availability_receipt={'path': str((PRIOR / 'pidfd-availability.json').relative_to(REPO)),
                          'sha256': sha(PRIOR / 'pidfd-availability.json')},
    missing_capability_skips_are_passed_tests=False,
    missing_capability_coverage_is_unavailable=True)
for label, observed in zip(('py312', 'py313', 'py314'), availability['interpreters']):
    required = observed['os_pidfd_open'] and observed['signal_pidfd_send_signal']
    selection['per_interpreter'][label]['expected_skipped_case_ids'] = [] if required else required_ids
    selection['per_interpreter'][label]['expected_passed'] = 76 if required else 69
    selection['per_interpreter'][label]['full_method_coverage_expected'] = required
write(OUT / 'selection.json', selection)
guard = json.loads((PRIOR / 'guard-before.json').read_text())
assert guard['original'][TEST] == guard['integrated'][TEST] == '69df0a97c06992db5c24e074abee9866474ae4f02a90b8e9f6fe73d1d073ef05'
guard['integrated'][TEST] = sha(REPO / TEST)
write(OUT / 'guard-before.json', guard)
driver = (PRIOR / 'run_checks.py').read_text().replace('distribution-check/mc300k-owner-compatibility', 'distribution-check/mc300k-owner-compatibility-v2')
driver = driver.replace('pyrpmd.mc300k-owner-method-gate.v1', 'pyrpmd.mc300k-capability-aware-method-gate.v1')
anchor = "        report['junit_sha256'] = sha(output/'pytest.xml')"
replacement = '''        skipped = [case for case in xml.iter('testcase') if case.find('skipped') is not None]
        skipped_ids = sorted(case.attrib.get('classname', '')+'::'+case.attrib['name'] for case in skipped)
        skipped_reasons = [case.find('skipped').attrib.get('message', '') for case in skipped]
        report['actual_skipped_case_ids'] = skipped_ids
        report['skipped_reasons'] = skipped_reasons
        report['capability_coverage_matches_fixed_selection'] = (
            skipped_ids == requested['expected_skipped_case_ids']
            and all('os.pidfd_open and signal.pidfd_send_signal' in reason for reason in skipped_reasons))
        report['full_method_coverage'] = not skipped_ids
        report['available_passed_cases'] = report['junit_counts']['tests'] - len(skipped_ids) - report['junit_counts']['failures'] - report['junit_counts']['errors']
        report['skipped_cases_are_not_passes'] = True
''' + anchor
assert anchor in driver
driver = driver.replace(anchor, replacement)
driver = driver.replace("and report.get('junit_counts', {}).get('skipped', 1) == 0)",
                        "and report.get('capability_coverage_matches_fixed_selection', False))")
(OUT / 'run_checks.py').write_text(driver)
write(OUT / 'driver-provenance.json', {'parent_sha256': sha(PRIOR / 'run_checks.py'),
    'current_sha256': sha(OUT / 'run_checks.py'),
    'changes': ['Separate immutable attempt.', 'Require exact predeclared skipped identities/reasons from observed API availability.', 'Report69passed/7unavailable separately from full76case coverage; zero unexpected failures/errors allowed.']})
for label in selection['per_interpreter']:
    (OUT / label).mkdir()
launcher = (BASE / 'run_mc300k_integration_gate.py').read_text().replace('mc300k-owner-compatibility', 'mc300k-owner-compatibility-v2')
with (BASE / 'run_mc300k_capability_gate.py').open('x') as stream:
    stream.write(launcher)
print(json.dumps({'gate': str(OUT), 'expected_passed': {'py312': 76, 'py313': 69, 'py314': 69},
                  'expected_unavailable': {'py312': 0, 'py313': 7, 'py314': 7}}))
