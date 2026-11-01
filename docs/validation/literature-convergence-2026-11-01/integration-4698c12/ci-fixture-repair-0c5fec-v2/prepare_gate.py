from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import subprocess

BASE = Path('/workspace/scratch/a5e4820daedd')
REPO = BASE / 'pyrpmd-convergence-integration'
PARENT = REPO / 'docs/validation/literature-convergence-2026-11-01/integration-4698c12/fifth-stage-compatibility'
OUT = PARENT.parent / 'ci-fixture-repair-0c5fec-v2'
FILES = ['tests/test_hch4_efficiency_pilot.py', 'tests/test_h3_p256_native_calibration.py']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')

OUT.mkdir(exist_ok=False)
prior = json.loads((PARENT / 'selection.json').read_text())
guard = json.loads((PARENT / 'guard-before.json').read_text())
changes = {}
for name in FILES:
    changes[name] = {'before': guard['integrated'][name], 'after': sha(REPO / name),
                     'original_scientific_tree_test_unchanged': sha(BASE / 'PyRPMD' / name) == guard['original'][name]}
    assert changes[name]['original_scientific_tree_test_unchanged']
    guard['integrated'][name] = sha(REPO / name)
campaign = 'benchmarks/literature/campaign.json'
published = subprocess.check_output(['git', 'show', '0c5fec22ce590634c45b8c46619cebf3e665eb85:' + campaign], cwd=REPO)
assert (REPO / campaign).read_bytes() == published
assert guard['integrated'][campaign] != sha(REPO / campaign)
changes[campaign] = {'before': guard['integrated'][campaign], 'after': sha(REPO / campaign),
                    'reason': 'Already published fifth-commit aggregate; first gate retained a fourth-head campaign hash and stopped before collecting tests.',
                    'current_bytes_equal_published_0c5fec': True}
guard['integrated'][campaign] = sha(REPO / campaign)
write(OUT / 'guard-before.json', guard)
selection = {'schema': 'pyrpmd.ci-fixture-repair-selection.v1',
             'created_before_execution_utc': datetime.now(timezone.utc).isoformat(),
             'source_head': '0c5fec22ce590634c45b8c46619cebf3e665eb85',
             'source_state_note': 'Published head plus exactly two test-fixture repairs, documented source-only literature evidence and no changes to scientific producers, models, runtime or acceptance criteria.',
             'native_products': prior['native_products'],
             'per_interpreter': {label: {'python': row['python'], 'files': FILES, 'expected': 101} for label, row in prior['per_interpreter'].items()},
             'test_fixture_changes': changes, 'required_cpu_affinity': [0, 6],
             'affinity_reason': 'Single-threaded short fixture checks; CPU6 remains visible for the existing isolated calibration resource-contract test.',
             'builds': 0, 'installs': 0, 'real_PES_evaluations': 0,
             'real_dynamics_steps': 0, 'old_CI_failures_retained': True}
write(OUT / 'selection.json', selection)
driver = (PARENT / 'run_checks.py').read_text()
driver = driver.replace('assert len(os.sched_getaffinity(0)) == 1',
                        'assert sorted(os.sched_getaffinity(0)) == selection[\'required_cpu_affinity\']')
driver = driver.replace('pyrpmd.fifth-stage-python-method-gate.v1', 'pyrpmd.ci-fixture-repair-method-gate.v1')
driver = driver.replace('distribution-check/fifth-stage-compatibility', 'distribution-check/ci-fixture-repair-0c5fec-v2')
(OUT / 'run_checks.py').write_text(driver)
write(OUT / 'driver-provenance.json', {'parent_path': str((PARENT / 'run_checks.py').relative_to(REPO)),
       'parent_sha256': sha(PARENT / 'run_checks.py'), 'new_sha256': sha(OUT / 'run_checks.py'),
       'changes': ['Separate evidence and temporary directories.', 'Expose CPU6 to the existing isolated resource contract test; retain one scientific-library thread.', 'Select only the two repaired test modules with all cases and retain exact runtime/source/native-product guards.']})
for label in selection['per_interpreter']:
    (OUT / label).mkdir()
launcher = (BASE / 'run_fifth_stage_checks.py').read_text().replace('fifth-stage-compatibility', 'ci-fixture-repair-0c5fec-v2')
launcher = launcher.replace("'-c', '8'", "'-c', '0,6'")
path = BASE / 'run_ci_fixture_gate_v2.py'
with path.open('x') as stream:
    stream.write(launcher)
print(json.dumps({'prepared': str(OUT), 'selection_sha256': sha(OUT / 'selection.json'), 'test_changes': changes}))
