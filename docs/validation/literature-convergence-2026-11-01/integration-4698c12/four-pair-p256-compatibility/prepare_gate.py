"""Prepare the bounded cross-version gate for newly frozen benchmark tools."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import subprocess

BASE = Path('/workspace/scratch/a5e4820daedd')
REPO = BASE/'pyrpmd-convergence-integration'
ORIGINAL = BASE/'PyRPMD'
PREFIX = Path('docs/validation/literature-convergence-2026-11-01')
PARENT = REPO/PREFIX/'integration-4698c12/materials-preparation-compatibility'
OUT = PARENT.parent/'four-pair-p256-compatibility'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def write(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True)+'\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coordinator-inventory',required=True)
    parser.add_argument('--coordinator-sha256',required=True)
    parser.add_argument('--coordinator-cases',type=int,required=True)
    args=parser.parse_args()
    if args.coordinator_cases <= 0:
        raise ValueError('A frozen collected coordinator case count is required')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    if head!='67d55e9ce745a2e18906d7aff9327c7b8646a2cb':
        raise ValueError('Unexpected integration head')
    old=json.loads((PARENT/'guard-before.json').read_text())
    guard={key:old[key] for key in ['integrated','original','external_models','abi_links']}
    differences=[]
    for label,root in [('integrated',REPO),('original',ORIGINAL)]:
        for name,expected in list(guard[label].items()):
            actual=sha(root/name)
            if actual!=expected:
                differences.append(dict(checkout=label,path=name,previous_sha256=expected,current_sha256=actual))
                if (label,name)!=('integrated','benchmarks/literature/campaign.json'):
                    raise ValueError(f'Unexplained prior guard change: {label}/{name}')
                guard[label][name]=actual
    bundles=[
        (PREFIX/'water/timestep-paired-control/publication-inventory.json','fef19a7571f5ac5605c665d6a2e9d0964c208047df0d34497f4673596649072e'),
        (PREFIX/'water/preparation-plot-helper/publication-inventory.json','fd2a9e421a3c274c869563b30e07c5759954944964077e5681d2783dd2b173fe'),
        (PREFIX/'review/hch4-reference-pilot/publication-inventory.json','10ce01b004b9870e283a79e746781543edcafa5d7598e6539a21c2bfd008528e'),
        (PREFIX/'h3-quantum/p256-parent-tool/publishable-files-v3.json','0dd49f90e9dd4b038c9a665f63f1ae067bb73a56b4750180148f5008e5bd02cb'),
        (PREFIX/'h3-quantum/p256-child-tool/publication-inventory.json','1a769e61c4578d1fba0d1fd3dc7cc529e82d3b899c3cbf22d7df6d5dfa5fb256'),
        (Path(args.coordinator_inventory),args.coordinator_sha256),
    ]
    for name,expected in bundles:
        if name.is_absolute() or '..' in name.parts:
            raise ValueError('Nonrepository inventory')
        inventory=json.loads((REPO/name).read_text())
        for label,root in [('integrated',REPO),('original',ORIGINAL)]:
            if sha(root/name)!=expected:
                raise ValueError(f'Frozen inventory differs: {label}/{name}')
            guard[label][str(name)]=expected
            for row in inventory['files']:
                path=root/row['path']
                if sha(path)!=row['sha256'] or path.stat().st_size!=row['bytes']:
                    raise ValueError(f'Frozen member changed: {path}')
                guard[label][row['path']]=row['sha256']
    for name,expected in guard['external_models'].items():
        if sha(Path(name))!=expected:
            raise ValueError('External model or native product changed')
    source=(PARENT/'run_checks.py').read_text()
    source=source.replace("'benchmarks.materials_reference'", "'benchmarks.materials_reference', 'benchmarks.hch4'")
    source=source.replace('distribution-check/materials-preparation-compatibility','distribution-check/four-pair-p256-compatibility')
    source=source.replace('pyrpmd.materials-preparation-python-method-gate.v1','pyrpmd.four-pair-p256-python-method-gate.v1')
    previous=json.loads((PARENT/'selection.json').read_text())
    files=['tests/test_water_paired_timestep.py','tests/test_water_preparation_plot.py',
           'tests/test_hch4_reference_pilot.py','tests/test_h3_p256_parent_campaign.py',
           'tests/test_h3_p256_surface_recrossing.py','tests/test_h3_p256_recrossing_analysis.py',
           'tests/test_h3_p256_child_campaign.py']
    expected=166+args.coordinator_cases
    guard.update(schema='pyrpmd.four-pair-p256-integration-guards.v1',
        created_utc=datetime.now(timezone.utc).isoformat(),parent_guard_sha256=sha(PARENT/'guard-before.json'),
        earlier_guard_differences=differences,
        differences_reason='Root added completed fixed-P H3, methane pilot and OH source-consistency evidence to the separate integrated literature catalog. Frozen simulation sources and registry values are unchanged.')
    selection=dict(schema='pyrpmd.four-pair-p256-integration-selection.v1',
        created_before_execution_utc=datetime.now(timezone.utc).isoformat(),source_head=head,
        source_state_note='Six frozen new method/coordinator bundles copied byte-for-byte. The separate literature catalog update is explicit in the guard deltas.',
        native_products=previous['native_products'],
        previous_315_case_coverage=dict(path=str(PARENT/'case-coverage.json'),sha256=sha(PARENT/'case-coverage.json')),
        driver_parent=dict(path=str(PARENT/'run_checks.py'),sha256=sha(PARENT/'run_checks.py')),
        driver_changes=['Include actually loaded hch4 modules in audit','Separate evidence/temp directory','New receipt schema'],
        frozen_inventories=[dict(path=str(p),sha256=h) for p,h in bundles],
        per_interpreter={label:dict(python=row['python'],expected=expected,files=files) for label,row in previous['per_interpreter'].items()},
        new_case_families=dict(water_paired_timestep=13,water_preparation_plot=4,hch4_reference_pilot=30,
            p256_parent_campaign=61,p256_parent_binding=30,p256_statistical_analysis=28,p256_child_coordinator=args.coordinator_cases),
        previous_315_cases_rerun=False,builds=0,installs=0,
        test_scope='New contracts, analytic/synthetic statistics, small OpenMM restart/timestep fixtures and bounded inert subprocess lifecycle tests. No external reaction PES or long scientific production is launched.')
    OUT.mkdir(exist_ok=False)
    with (OUT/'run_checks.py').open('x') as stream:
        stream.write(source)
    write(OUT/'selection.json',selection)
    write(OUT/'guard-before.json',guard)
    for label in selection['per_interpreter']:
        (OUT/label).mkdir()
        (BASE/'distribution-check/four-pair-p256-compatibility'/label).mkdir(parents=True)
    print(json.dumps(dict(output=str(OUT),guards={k:len(guard[k]) for k in ['integrated','original','external_models']},expected_new_cases_each=expected,previous_cases_each=315,guard_differences=differences)))

if __name__=='__main__':
    main()
