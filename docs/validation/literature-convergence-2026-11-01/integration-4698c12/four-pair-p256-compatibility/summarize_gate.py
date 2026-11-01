"""Derive exact distinct case unions from the newly completed JUnit files."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
from xml.etree import ElementTree

BASE=Path('/workspace/scratch/a5e4820daedd')
REPO=BASE/'pyrpmd-convergence-integration'
PREFIX=Path('docs/validation/literature-convergence-2026-11-01/integration-4698c12')
PARENT=REPO/PREFIX/'materials-preparation-compatibility'
OUT=REPO/PREFIX/'four-pair-p256-compatibility'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def write(path,value):
    with path.open('x') as stream:
        stream.write(json.dumps(value,indent=2,sort_keys=True)+'\n')

def main():
    prior=json.loads((PARENT/'case-coverage.json').read_text())
    selection=json.loads((OUT/'selection.json').read_text())
    reports,unions={},{}
    for label in ['py312','py313','py314']:
        folder=OUT/label;report=json.loads((folder/'compatibility.json').read_text())
        expected=selection['per_interpreter'][label]['expected']
        if not report['passed'] or not report['guard_after']['all_unchanged']:
            raise ValueError('New compatibility gate did not pass')
        tree=ElementTree.parse(folder/'pytest.xml');suites=list(tree.iter('testsuite'))
        counts={key:sum(int(s.attrib.get(key,'0')) for s in suites) for key in ['tests','failures','errors','skipped']}
        ids=[c.attrib.get('classname','')+'::'+c.attrib['name'] for c in tree.iter('testcase')]
        if counts!={'tests':expected,'failures':0,'errors':0,'skipped':0} or len(set(ids))!=expected:
            raise ValueError('Actual distinct collection differs from frozen selection')
        old=prior['case_ids'][label]
        if len(old)!=315 or set(old)&set(ids):
            raise ValueError('Unexpected prior overlap/count')
        union=sorted(set(old)|set(ids));unions[label]=union
        families={name:sum(c.startswith('tests.'+Path(name).stem+'::') for c in ids) for name in selection['per_interpreter'][label]['files']}
        reports[label]=dict(new_distinct_cases=expected,prior_distinct_cases_reused=315,
            combined_distinct_passed_cases=len(union),new_case_families=families,junit_counts=counts,
            pytest_seconds=sum(float(s.attrib['time']) for s in suites),receipt_elapsed_seconds=report['elapsed_seconds'],
            receipt_sha256=sha(folder/'compatibility.json'),junit_sha256=sha(folder/'pytest.xml'),
            guard_after=report['guard_after'],actual_native_imports_checked_in_test_and_child=True,
            environment_distributions_unchanged=report['environment_distributions_unchanged'])
    if not unions['py312']==unions['py313']==unions['py314']:
        raise ValueError('Cross-interpreter case unions differ')
    new_count=reports['py312']['new_distinct_cases'];total=len(unions['py312'])
    write(OUT/'case-coverage.json',dict(schema='pyrpmd.method-case-coverage-union.v1',
        note='Exactly the fixed new case selection on each interpreter, unioned with unchanged earlier315 passing identities. Repeated focused reviews do not increase counts.',
        prior_coverage_path=str((PARENT/'case-coverage.json').relative_to(REPO)),prior_coverage_sha256=sha(PARENT/'case-coverage.json'),case_ids=unions))
    summary=dict(schema='pyrpmd.four-pair-p256-compatibility-summary.v1',created_utc=datetime.now(timezone.utc).isoformat(),
        source_head=selection['source_head'],source_state_note=selection['source_state_note'],scope=selection['test_scope'],
        selection_sha256=sha(OUT/'selection.json'),driver_sha256=sha(OUT/'run_checks.py'),source_guard_sha256=sha(OUT/'guard-before.json'),
        case_coverage_sha256=sha(OUT/'case-coverage.json'),distinct_new_cases=new_count,new_case_executions=3*new_count,
        combined_distinct_cases_per_interpreter=total,same_distinct_cases_all_interpreters=True,interpreters=reports,
        previous_315_cases_rerun=False,dependency_changes=False,builds=0,ABI_links_modified=False,
        active_original_sources_modified=False,full_repository_suite_run=False,long_dynamics_run=False,
        source_bundles=selection['frozen_inventories'])
    write(OUT/'summary.json',summary)
    for source,name in [(BASE/'prepare_four_pair_p256_gate.py','prepare_gate.py'),
                        (BASE/'run_four_pair_p256_gate.py','run_sequential.py'),(Path(__file__),'summarize_gate.py')]:
        with (OUT/name).open('xb') as stream:
            stream.write(source.read_bytes())
    readme=f'''# New P256, methane pilot and water contracts on three Python versions

All **{new_count} newly selected distinct cases pass on each of Python
3.12.14, 3.13.18 and 3.14.10**, with no failures, errors or skips. Their
case-level union with the unchanged earlier 315-case receipts is **{total}
distinct passing cases per interpreter**. The earlier cases were not rerun.

The new scope comprises water paired-timestep/preparation plots, the bounded
methane pilot, P256 parent generation, complete parent binding, covariance
and Welch analysis, and the single-attempt child coordinator. Synthetic
statistics, analytic potentials, small OpenMM states and inert subprocesses
test those contracts; these tests do not create a long physical trajectory
or a new reaction rate. Initial failures and earlier candidates remain in
their respective source bundles; repeated reviews are not new cases.

The actual test process and a fresh interpreter both load this separate
integration checkout and its existing native ABI products. All selected
integrated/original/model hashes and six retained ABI links are unchanged
across each gate. No dependency install, rebuild, active-runtime update or
full-project-suite execution is implied. The guard delta explicitly records
the integration-only literature catalog update since the earlier315 gate.

See [selection](selection.json), [summary](summary.json),
[exact case identities](case-coverage.json), and the three
[3.12](py312/compatibility.json), [3.13](py313/compatibility.json),
[3.14](py314/compatibility.json) source/runtime receipts. The frozen
publication inventory binds all commands, logs, JUnit files and checksums.
'''
    with (OUT/'README.md').open('x') as stream:
        stream.write(readme)
    files=[dict(path=str(p.relative_to(REPO)),sha256=sha(p),bytes=p.stat().st_size)
        for p in sorted(OUT.rglob('*')) if p.is_file()]
    write(OUT/'publication-inventory.json',dict(schema='pyrpmd.compatibility.publication-inventory.v1',
        created_utc=datetime.now(timezone.utc).isoformat(),files=files))
    print(json.dumps(dict(new_distinct_cases=new_count,total_distinct_cases=total,
        interpreters={label:{k:v for k,v in row.items() if k in ['junit_counts','pytest_seconds','combined_distinct_passed_cases']} for label,row in reports.items()},
        inventory_sha256=sha(OUT/'publication-inventory.json'),files=len(files),bytes=sum(r['bytes'] for r in files))))

if __name__=='__main__':
    main()
