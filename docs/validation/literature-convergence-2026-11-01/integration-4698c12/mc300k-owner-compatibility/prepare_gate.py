from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil

BASE = Path('/workspace/scratch/a5e4820daedd')
ORIGINAL = BASE / 'PyRPMD'
REPO = BASE / 'pyrpmd-convergence-integration'
REL = Path('docs/validation/literature-convergence-2026-11-01')
TOOL = REL / 'h3-quantum/umbrella-isolated300k-tool'
LIVE = ORIGINAL / REL / 'h3-quantum/umbrella-isolated300k-execution-v1'
PRIOR = REPO / REL / 'integration-4698c12/ci-fixture-repair-0c5fec-v2'
OUT = PRIOR.with_name('mc300k-owner-compatibility')
WORK = BASE / 'literature-convergence-quantum/umbrella-stage-300000-isolated-v1'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')

def identity(path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}

assert sha(ORIGINAL / TOOL / 'publication-inventory.json') == '7fcb7706bc43b26721daae0c6e99f3f47458a982b3348bd6eab2b9712c9585c3'
inv = json.loads((ORIGINAL / TOOL / 'publication-inventory.json').read_text())
selected = {row['path']: row for row in inv['files']}
p = TOOL / 'publication-inventory.json'
selected[str(p)] = dict(identity(ORIGINAL / p), path=str(p))
copied = []
for name, row in selected.items():
    source, target = ORIGINAL / name, REPO / name
    assert source.stat().st_size == row['bytes'] and sha(source) == row['sha256']
    if target.exists():
        assert sha(target) == row['sha256'], name
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(name)
    assert sha(target) == row['sha256']
resources = []
for seed in (211017, 211019, 211021, 211023):
    path = WORK / 'attempt' / f'seed-{seed}-resource.json'
    value = json.loads(path.read_bytes())
    assert value['seed'] == seed and value['RLIMIT_AS'] == [1073741824, 1073741824]
    assert value['peak_RSS_bytes'] > 0
    resources.append({'snapshot_of_mutable_resource_metadata': value,
                      'scope': 'Ownership/resource observation only; no MC state, acceptance rate, free energy or partial numerical estimate.'})
write(LIVE / 'startup-observation.json', {
    'schema': 'pyrpmd.h3.isolated300k-root-startup-observation.v1',
    'observed_utc': datetime.now(timezone.utc).isoformat(), 'exec_session_id': 15483,
    'actual_tool_start': {'chunk_id': 'ff3e85', 'session_id': 15483, 'wall_time_seconds': 1.000505541,
                        'status': 'running; no terminal return yet'},
    'source': identity(ORIGINAL / 'benchmarks/h3/quantum_isolated_continuation.py'),
    'registration': identity(WORK / 'registration.json'), 'plan': identity(WORK / 'attempt/plan.json'),
    'once_claim': identity(WORK / 'RUN_ONCE.json'), 'resources': resources,
    'owned_workers_observed': 4, 'fixed_total_seeds': 8, 'fixed_endpoint': 300000,
    'completion_or_scientific_analysis_claimed': False,
    'raw_checkpoints_or_partial_results_read_by_observer': False})
for path in (LIVE / 'root-adoption.json', LIVE / 'resource-admission.json', LIVE / 'root-launch.json',
             LIVE / 'adopt_resources.py', LIVE / 'launch_once.py', LIVE / 'startup-observation.json'):
    target = REPO / path.relative_to(ORIGINAL)
    assert not target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    assert sha(target) == sha(path)
OUT.mkdir(exist_ok=False)
write(OUT / 'copy-receipt.json', {'schema': 'pyrpmd.mc300k-integration-immutable-copy.v1',
    'created_utc': datetime.now(timezone.utc).isoformat(), 'inventory_sha256': sha(ORIGINAL / TOOL / 'publication-inventory.json'),
    'selected_files': selected, 'copied_new_paths': copied, 'different_existing_files_overwritten': 0,
    'original_active_scientific_sources_modified': False})
selection = json.loads((PRIOR / 'selection.json').read_text())
selection.update(schema='pyrpmd.mc300k-owner-compatibility-selection.v1',
    created_before_execution_utc=datetime.now(timezone.utc).isoformat(),
    source_state_note='Published head0c5fec plus the separately validated two fixture repairs and exact frozen new MC300k scheduling evidence; active scientific sources/models/protocols unchanged.',
    per_interpreter={label: {'python': row['python'], 'files': ['tests/test_h3_quantum_isolated_continuation.py'], 'expected': 76}
                     for label, row in selection['per_interpreter'].items()},
    required_cpu_affinity=[8], affinity_reason='One reserved CPU for short inert method tests; no physical restore or sampler invocation.',
    test_fixture_changes={})
write(OUT / 'selection.json', selection)
guard = json.loads((PRIOR / 'guard-before.json').read_text())
for label in ('original', 'integrated'):
    for name, row in selected.items():
        if name in guard[label]:
            assert guard[label][name] == row['sha256']
        guard[label][name] = row['sha256']
write(OUT / 'guard-before.json', guard)
driver = (PRIOR / 'run_checks.py').read_text().replace('distribution-check/ci-fixture-repair-0c5fec-v2', 'distribution-check/mc300k-owner-compatibility')
driver = driver.replace('pyrpmd.ci-fixture-repair-method-gate.v1', 'pyrpmd.mc300k-owner-method-gate.v1')
(OUT / 'run_checks.py').write_text(driver)
for label in selection['per_interpreter']:
    (OUT / label).mkdir()
launcher = (BASE / 'run_ci_fixture_gate_v2.py').read_text().replace('ci-fixture-repair-0c5fec-v2', 'mc300k-owner-compatibility')
launcher = launcher.replace("'-c', '0,6'", "'-c', '8'")
with (BASE / 'run_mc300k_integration_gate.py').open('x') as stream:
    stream.write(launcher)
print(json.dumps({'copied_new': len(copied), 'verified_selected': len(selected),
    'expected_cases_each': 76, 'active_original_sources_touched': False, 'gate': str(OUT)}))
