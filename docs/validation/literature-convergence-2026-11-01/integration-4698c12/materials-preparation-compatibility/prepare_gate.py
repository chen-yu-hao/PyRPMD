"""Prepare a bounded integration gate without changing any frozen runtime."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess

BASE = Path('/workspace/scratch/a5e4820daedd')
REPO = BASE / 'pyrpmd-convergence-integration'
ORIGINAL = BASE / 'PyRPMD'
PARENT = REPO / 'docs/validation/literature-convergence-2026-11-01/integration-4698c12/new-stage-compatibility'
OUT = PARENT.parent / 'materials-preparation-compatibility'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + '\n')


def main():
    OUT.mkdir(exist_ok=False)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    if head != '67d55e9ce745a2e18906d7aff9327c7b8646a2cb':
        raise ValueError('Unexpected integration parent')
    old = json.loads((PARENT / 'guard-before.json').read_text())
    guard = {key: old[key] for key in ['integrated', 'original', 'external_models', 'abi_links']}
    differences = []
    for label, root in [('integrated', REPO), ('original', ORIGINAL)]:
        for name, expected in list(guard[label].items()):
            current = sha(root / name)
            if current != expected:
                differences.append({'checkout': label, 'path': name, 'previous_sha256': expected, 'current_sha256': current})
                if (label, name) != ('integrated', 'benchmarks/literature/campaign.json'):
                    raise ValueError(f'Unexplained earlier source-guard change: {label}/{name}')
                guard[label][name] = current
    bundles = [
        ('docs/validation/literature-convergence-2026-11-01/other-reactions/materials-source-index/publishable-files.json',
         'cf670ffafe6cb2dec4ab2199dd63aa2ea6cb3659252c8ef1428e96174d1debab'),
        ('docs/validation/literature-convergence-2026-11-01/water/preparation-coordinator-gate/publication-inventory.json',
         '09871c5277f29e681cba26d2c385740fe0e686c8f8a811ee1952f294b9631d06'),
    ]
    for name, expected in bundles:
        inventory = json.loads((REPO / name).read_text())
        for label, root in [('integrated', REPO), ('original', ORIGINAL)]:
            if sha(root / name) != expected:
                raise ValueError('Frozen bundle identity differs')
            guard[label][name] = expected
            for item in inventory['files']:
                if sha(root / item['path']) != item['sha256']:
                    raise ValueError('Frozen bundle member differs')
                guard[label][item['path']] = item['sha256']
    for name, expected in list(guard['external_models'].items()):
        if sha(Path(name)) != expected:
            raise ValueError('Earlier external model or ABI changed')
    eam_root = BASE / 'materials-source-index-assets/eam4-build-v1'
    for name in ['libeam4_static.so', 'build.json', 'nimetpot.original.f90', 'eam4.original-suffix.f90', 'eam4_bridge.f90']:
        path = eam_root / name
        guard['external_models'][str(path)] = sha(path)
    guard.update(schema='pyrpmd.materials-preparation-integration-guards.v1',
                 created_utc=datetime.now(timezone.utc).isoformat(),
                 parent_guard_sha256=sha(PARENT / 'guard-before.json'),
                 earlier_guard_differences=differences,
                 differences_reason='Root updated the Ni literature inventory after freezing the new author-cited static evidence; no scientific runtime changes.')
    source = (PARENT / 'run_checks.py').read_text()
    source = source.replace("'benchmarks.h3', 'benchmarks.water'", "'benchmarks.h3', 'benchmarks.water', 'benchmarks.materials_reference'")
    source = source.replace("distribution-check/new-stage-compatibility", "distribution-check/materials-preparation-compatibility")
    source = source.replace("pyrpmd.new-stage-python-method-gate.v1", "pyrpmd.materials-preparation-python-method-gate.v1")
    with (OUT / 'run_checks.py').open('x') as stream:
        stream.write(source)
    previous = json.loads((PARENT / 'selection.json').read_text())
    selection = dict(
        schema='pyrpmd.materials-preparation-integration-selection.v1',
        created_before_execution_utc=datetime.now(timezone.utc).isoformat(),
        source_head=head, source_state_note='Two independently frozen new bundles copied byte-for-byte; a documentation-only Ni campaign inventory update is explicit in guard deltas.',
        native_products=previous['native_products'],
        previous_270_case_coverage={'path':str(PARENT/'case-coverage.json'),'sha256':sha(PARENT/'case-coverage.json')},
        driver_parent={'path':str(PARENT/'run_checks.py'),'sha256':sha(PARENT/'run_checks.py')},
        driver_changes=['Include actually loaded materials_reference modules in audit', 'Separate evidence/temp directory', 'New receipt schema'],
        frozen_inventories=[{'path':p,'sha256':s} for p,s in bundles],
        exact_EAM4_library={'path':str(eam_root/'libeam4_static.so'),'sha256':sha(eam_root/'libeam4_static.so')},
        per_interpreter={label:{'python':row['python'], 'expected':45,
            'files':['tests/test_materials_reference_eam4.py','tests/test_water_preparation_coordinator.py']}
            for label,row in previous['per_interpreter'].items()},
        earlier_270_cases_rerun=False, builds=0, installs=0,
        test_scope='10 real single-bead EAM4 static cases and 35 orchestration fixtures, including bounded Python stub subprocess signals. No liquid or reaction dynamics are launched.',
    )
    write(OUT / 'selection.json', selection)
    write(OUT / 'guard-before.json', guard)
    for label in selection['per_interpreter']:
        (OUT / label).mkdir()
        (BASE / 'distribution-check/materials-preparation-compatibility' / label).mkdir(parents=True)
    print(json.dumps({'output':str(OUT),'guards':{k:len(guard[k]) for k in ['integrated','original','external_models']},'expected_new_cases_each':45,'previous_cases_each':270,'guard_differences':differences}))


if __name__ == '__main__':
    main()
