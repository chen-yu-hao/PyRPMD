"""Adopt the independently reviewed fixed MC endpoint with current resources."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import shutil

BASE = Path('/workspace/scratch/a5e4820daedd')
REPO = BASE / 'PyRPMD'
REL = Path('docs/validation/literature-convergence-2026-11-01')
TOOL = REPO / REL / 'h3-quantum/umbrella-isolated300k-tool'
OUT = REPO / REL / 'h3-quantum/umbrella-isolated300k-execution-v1'
POLICY = REPO / REL / 'h3-quantum/umbrella-stage-300000-proposal/implementation-policy-v2.json'
SOURCE = REPO / 'benchmarks/h3/quantum_isolated_continuation.py'
TESTS = REPO / 'tests/test_h3_quantum_isolated_continuation.py'
TARGET = BASE / 'literature-convergence-quantum/umbrella-stage-300000-isolated-v1'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def identity(path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}

def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')

expected = {
    SOURCE: 'a7ec140dbe953a3e068cb491b24f0769e395cd4d2c449e9d2daf20aa45b4a2b7',
    TESTS: '69df0a97c06992db5c24e074abee9866474ae4f02a90b8e9f6fe73d1d073ef05',
    POLICY: 'a8cbef299f4bda18fbdab495fdabcae26982ea08aca25aca257f83866366c5da',
    TOOL / 'independent-review-final.json': '776f9520a6db6212790cd5c7b35f559321b9aebca1754cfb6a6fe15c06a789c0',
    TOOL / 'inert-gate-final/gate.json': '7d3359bbce96e1988e7a7b943485fc5d530f7f3f5836d8792c2b9569ed81bbbb',
    TOOL / 'publication-inventory.json': '7fcb7706bc43b26721daae0c6e99f3f47458a982b3348bd6eab2b9712c9585c3',
}
for path, digest in expected.items():
    assert sha(path) == digest, path
inventory = json.loads((TOOL / 'publication-inventory.json').read_text())
for row in inventory['files']:
    path = REPO / row['path']
    assert path.is_file() and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], path
review = json.loads((TOOL / 'independent-review-final.json').read_text())
gate = json.loads((TOOL / 'inert-gate-final/gate.json').read_text())
assert review['approved_for_fixed_isolated300000'] and gate['passed'] == 76
assert gate['failures'] == gate['errors'] == gate['skipped'] == gate['returncode'] == 0
assert not TARGET.exists() and not TARGET.with_name('.' + TARGET.name + '-continuation-lock').exists()
assert {0, 1, 2, 3, 4, 5, 6, 7}.issubset(os.sched_getaffinity(0))
policy = json.loads(POLICY.read_text())
GiB = 1024 ** 3
disk_rows = [
    {'name': 'water', 'remaining_bytes': 5750000000 + 5800000000 + 64 * 1024 ** 2,
     'basis': 'Revision2 prospective full400-branch raw cap5.75GB, one archive cap5.8GB and64MiB compact/streaming-replay metadata. No third physical full extraction. This reserves space; it does not authorize the still-unadopted water implementation.'},
    {'name': 'P256_children', 'remaining_bytes': 3500000000,
     'basis': 'Active registered800-chunk campaign: conservative remaining raw growth plus one future complete archive and extraction/analysis. Two completed worker receipts show resource use only; no partial physical estimates read.'}]
required = 5000000000 + sum(x['remaining_bytes'] for x in disk_rows) + 4 * GiB
free = shutil.disk_usage(BASE).free
assert free >= required, (free, required)
memory_rows = [
    {'name': 'water', 'bytes': 5 * GiB // 4, 'hard_limit': False,
     'basis': 'Prospective two512MiB worker AS caps plus256MiB stdlib owner cap.512MiB excluded import/context/layout probe passed; first actual PME force gate remains unexecuted. This is a reservation, not measured production RSS or runtime adoption.'},
    {'name': 'P256_children', 'bytes': GiB // 2, 'hard_limit': False,
     'basis': 'Planning reserve for two active workers and owner; exact completed-worker peak RSS88728/88820KiB is observed resource evidence, not an imposed cap or proof of future peak.'},
    {'name': 'other_diagnostics_and_runtime', 'bytes': 5 * GiB // 4, 'hard_limit': False,
     'basis': 'Reserved1.25GiB for bounded sequential diagnostics, source operations and runtime/system margin. No ninth sustained compute worker is admitted.'}]
assert 5 * GiB + sum(x['bytes'] for x in memory_rows) == 8 * GiB
water_probe = REPO / REL / 'water/timestep-momentum-expansion-proposal/revision-2/memory-probe/cap-512/result.json'
attempt = BASE / 'literature-convergence-quantum/p256-fixed-eight-chain-v1/recrossing/coordinator-attempts/abc9e44ad7c24313b23f64741046d2c1'
p256_resources = []
for seed in (825601, 825603):
    path = attempt / f'seed-{seed}-execution.json'
    value = json.loads(path.read_text())
    p256_resources.append({'receipt': identity(path), 'seed': seed,
        'resource_fields_only': {k: value[k] for k in ('CPU_affinity', 'peak_RSS_KiB',
            'whole_process_system_CPU_seconds', 'whole_process_user_CPU_seconds', 'whole_worker_call_wall_seconds')}})
OUT.mkdir(exist_ok=False)
resource_receipt = {
    'schema': 'pyrpmd.root-shared-resource-admission.v1',
    'observed_before_adoption_utc': datetime.now(timezone.utc).isoformat(),
    'visible_CPU_affinity': sorted(os.sched_getaffinity(0)), 'sustained_CPU_ceiling': 8,
    'fixed_assignment': {'MC300k': [0, 1, 2, 3], 'prospective_water': [4, 5], 'active_P256': [6, 7]},
    'CPU8_scope': 'Only bounded sequential diagnostics, never a ninth sustained scientific worker.',
    'filesystem_free_bytes': free, 'joint_required_incremental_bytes': required,
    'excess_over_joint_reservation_bytes': free - required,
    'MC_own_incremental_budget_bytes': 5000000000, 'external_disk_reservations': disk_rows,
    'global_free_reserve_bytes_counted_once': 4 * GiB,
    'old_raw_archives_and_verified_extractions_deleted': False,
    'storage_scope': 'Remaining growth only, from current free space. Later campaign admissions must subtract already-created owned bytes from their remaining allocations and recheck the same joint ledger; they cannot reuse the whole free-space figure independently.',
    'memory_budget_bytes': 8 * GiB, 'MC_per_process_RLIMIT_AS_bytes': GiB,
    'MC_four_workers_plus_owner_AS_sum_bytes': 5 * GiB,
    'external_memory_reservations': memory_rows, 'joint_memory_reservation_bytes': 8 * GiB,
    'sum_AS_caps_is_not_measured_RSS': True, 'unobservable_external_current_RSS': None,
    'no_proc_or_unrelated_process_access': True,
    'water_excluded_context_layout_probe': identity(water_probe),
    'MC_excluded_layout_probe': identity(TOOL / 'synthetic-AS-layout-probe.json'),
    'P256_completed_worker_resource_evidence': p256_resources,
    'new_physical_water_adoption_supplied': False,
    'scientific_gates_models_protocols_or_seeds_changed': False,
}
write(OUT / 'resource-admission.json', resource_receipt)
adoption = json.loads((TOOL / 'root-adoption-template.json').read_text())
adoption.update(adopted=True, template_is_non_executable=False,
    created_utc=datetime.now(timezone.utc).isoformat(),
    no_original_or_other_writer_running_or_pending=True,
    original_and_verified_evidence_remain_read_only=True,
    only_new_registered_working_copy_may_change=True,
    actual_resource_receipt=identity(OUT / 'resource-admission.json'),
    independent_review=identity(TOOL / 'independent-review-final.json'),
    test_gate=identity(TOOL / 'inert-gate-final/gate.json'),
    remaining_external_reservations=disk_rows,
    remaining_other_reserved_bytes=sum(x['remaining_bytes'] for x in disk_rows),
    other_reservation_basis='Current root resource-admission.json jointly reserves water, activeP256, MC and one global4GiB floor; archived evidence stays untouched.',
    external_memory_reservations=memory_rows,
    owner_exclusion_basis='All previously root-owned umbrella work completed with actual terminal observations. No other original/new umbrella writer is launched or authorized in this session. The anomalous original lease is preserved and never retired again; no unrelated process is probed.',
    implementation_inventory=identity(TOOL / 'publication-inventory.json'),
    root_adopter='/root', fixed_endpoint_rationale='Original registered300000 endpoint after failed100000 log-SE gate; never contingent on point agreement with paper.',
    physical_analysis_authorized_by_this_adoption=False)
assert adoption['workers'] == 4 and adoption['CPU_affinity'] == [0, 1, 2, 3]
assert adoption['sampling_envelope'] == policy['sampling_envelope']
assert adoption['source_sha256'] == sha(SOURCE)
write(OUT / 'root-adoption.json', adoption)
shutil.copy2(Path(__file__), OUT / 'adopt_resources.py')
print(json.dumps({'adoption': identity(OUT / 'root-adoption.json'),
    'resource_receipt': identity(OUT / 'resource-admission.json'),
    'free_bytes': free, 'required_bytes': required, 'excess_bytes': free - required,
    'workers': 4, 'physical_sampling_started': False}))
