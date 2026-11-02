"""Prepare only the42 new audit/calibration/diagnostic contracts."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess

BASE=Path('/workspace/scratch/a5e4820daedd')
REPO=BASE/'pyrpmd-convergence-integration'
ORIGINAL=BASE/'PyRPMD'
PREFIX=Path('docs/validation/literature-convergence-2026-11-01')
PARENT=REPO/PREFIX/'integration-4698c12/four-pair-p256-compatibility'
OUT=PARENT.parent/'auxiliary-proof-compatibility'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def write(path,value):
    with path.open('x') as stream:stream.write(json.dumps(value,indent=2,sort_keys=True)+'\n')

def main():
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    if head!='67d55e9ce745a2e18906d7aff9327c7b8646a2cb':raise ValueError('Unexpected integration head')
    prior=json.loads((PARENT/'selection.json').read_text())
    previous_guard=json.loads((PARENT/'guard-before.json').read_text())
    guard={k:previous_guard[k] for k in ['integrated','original','external_models','abi_links']}
    for label,root in [('integrated',REPO),('original',ORIGINAL)]:
        for name,expected in guard[label].items():
            if sha(root/name)!=expected:raise ValueError(f'Earlier frozen source changed: {label}/{name}')
    bundles=[
        (PREFIX/'h3-quantum/p256-parent-stage/publication-inventory.json','24ba0b8efe7fe873923da7a0d06d055de2016930c1fc11ab22f8486b62afe61a'),
        (PREFIX/'h3-quantum/p256-native-calibration/publication-inventory.json','f7dd402d577e062eae14d00d50b1506dfd86cc08ceea2b31af2094699ab9e3a8'),
        (PREFIX/'water/production-stability-helper/publication-inventory.json','ac2c4283de357a1c06845f333702aa4bc870a474b52f235c5b44f9248796bc10')]
    for name,expected in bundles:
        inventory=json.loads((REPO/name).read_text())
        for label,root in [('integrated',REPO),('original',ORIGINAL)]:
            if sha(root/name)!=expected:raise ValueError('New frozen inventory differs')
            guard[label][str(name)]=expected
            for row in inventory['files']:
                path=root/row['path']
                if sha(path)!=row['sha256'] or path.stat().st_size!=row['bytes']:raise ValueError('Frozen member differs')
                guard[label][row['path']]=row['sha256']
    for name,expected in guard['external_models'].items():
        if sha(Path(name))!=expected:raise ValueError('External model/native identity changed')
    guard.update(schema='pyrpmd.auxiliary-proof-integration-guards.v1',created_utc=datetime.now(timezone.utc).isoformat(),
        parent_guard_sha256=sha(PARENT/'guard-before.json'),earlier_guard_differences=[])
    source=(PARENT/'run_checks.py').read_text().replace('distribution-check/four-pair-p256-compatibility','distribution-check/auxiliary-proof-compatibility').replace('pyrpmd.four-pair-p256-python-method-gate.v1','pyrpmd.auxiliary-proof-python-method-gate.v1')
    old="        snapshot['actual_loaded_project_modules'] = imported"
    new="""        archive_test = sys.modules.get('tests.test_h3_p256_parent_archive')
        assert archive_test is not None
        archive_path = Path(archive_test.archive.__file__).resolve()
        assert archive_path == REPO/'docs/validation/literature-convergence-2026-11-01/h3-quantum/p256-parent-stage/archive_parents.py'
        snapshot['actual_document_auditor_import'] = dict(path=str(archive_path), sha256=sha(archive_path))
        snapshot['actual_loaded_project_modules'] = imported"""
    if source.count(old)!=1:raise ValueError('Runtime audit anchor differs')
    source=source.replace(old,new)
    files=['tests/test_h3_p256_parent_archive.py','tests/test_h3_p256_native_calibration.py','tests/test_water_production_stability.py']
    selection=dict(schema='pyrpmd.auxiliary-proof-integration-selection.v1',created_before_execution_utc=datetime.now(timezone.utc).isoformat(),
        source_head=head,source_state_note='Three immutable auxiliary bundles copied byte-for-byte; previous guarded sources unchanged.',
        native_products=prior['native_products'],previous_554_case_coverage=dict(path=str(PARENT/'case-coverage.json'),sha256=sha(PARENT/'case-coverage.json')),
        driver_parent=dict(path=str(PARENT/'run_checks.py'),sha256=sha(PARENT/'run_checks.py')),
        driver_changes=['Separate evidence/temp/schema names','Explicitly record the document-backed archive auditor module imported by its test'],
        frozen_inventories=[dict(path=str(p),sha256=h) for p,h in bundles],
        per_interpreter={label:dict(python=row['python'],expected=42,files=files) for label,row in prior['per_interpreter'].items()},
        new_case_families=dict(parent_archive=20,native_calibration_contracts=16,water_production_stability=6),
        previous_554_cases_rerun=False,builds=0,installs=0,
        test_scope='Synthetic lifecycle, finite-array/restart-compare and paired-statistic/whole-cohort plotting contracts. No external PES call, real parent numerical read or physical propagation occurs in this gate.')
    OUT.mkdir(exist_ok=False)
    with (OUT/'run_checks.py').open('x') as stream:stream.write(source)
    write(OUT/'selection.json',selection);write(OUT/'guard-before.json',guard)
    for label in selection['per_interpreter']:
        (OUT/label).mkdir();(BASE/'distribution-check/auxiliary-proof-compatibility'/label).mkdir(parents=True)
    print(json.dumps(dict(expected_new_cases_each=42,prior_distinct_cases_each=554,output=str(OUT),
        guards={k:len(guard[k]) for k in ['integrated','original','external_models']})))

if __name__=='__main__':main()
