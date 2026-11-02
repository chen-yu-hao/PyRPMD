"""Copy sealed additive evidence and prepare a non-overlapping method gate."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

ROOT = Path('/workspace/scratch/a5e4820daedd')
ORIGINAL = ROOT / 'PyRPMD'
REPO = ROOT / 'pyrpmd-convergence-integration'
BASE = Path('docs/validation/literature-convergence-2026-11-01')
EVIDENCE = REPO / BASE / 'integration-4698c12/fifth-stage-compatibility'
PRIOR = EVIDENCE.parent / 'auxiliary-proof-compatibility-v2'
INVENTORIES = {
    'water/original-fixed-cohort/publication-inventory.json': 'a547ecba7890116439f7bfd8f908805751d1ce72c68b1716cc67ac2d16d693b2',
    'water/original-cohort-archive-replay/publication-inventory.json': 'ed81dc97c108549006bda497f9832c64102e2309d798897f010e533610e43ac6',
    'water/vacf-msd-identity/publication-inventory.json': 'ef17405ea42c3068196c5f6d630a2819bcbef51ef862c6128cea2ed3fed799d9',
    'h3-quantum/p256-completion-tool/publication-inventory.json': 'c16c2d38582639169cb707b85a956686e89cdf1c604a33ad2d3be32c50328cb5',
    'review/hch4-reference-efficiency-implementation/publication-inventory.json': '83d73d906ff8e345ee39761029ad36b889426c7c2c01b7b2ae7781718040511f',
    'review/hch4-efficiency-execution-guide/publication-inventory.json': 'fcfb67a0b5db02a1ce50aa3ebf57e888b1182d5c7a3815cc6c55a2c612680272',
    'review/pr68-software-review/ci-36ec289-20261101T211518Z/publication-inventory.json': 'cffb2e44d4a5f5dd874246d3b04bf1907ef64c2a05a4c279332230f5a2d10d46',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def main():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == '36ec289981639bbeb8feecf3b30400b4cbda9afe'
    full = EVIDENCE.parent / 'full-suite-36ec289-py314-v2'
    assert sha(full / 'publication-inventory.json') == 'ae52792cf2c3f3c436d1762415c2fc8bd69b7d9151f79efe451a7f8986d4e35f'
    assert json.loads((full / 'root-exec-observation.json').read_text())['tool_result']['exit_code'] == 0
    assert json.loads((full / 'report.json').read_text())['passed'] is True
    assert not EVIDENCE.exists()
    selected = {}
    for name, expected in INVENTORIES.items():
        path = BASE / name
        assert sha(ORIGINAL / path) == expected, name
        inv = json.loads((ORIGINAL / path).read_text())
        for row in inv['files']:
            rel = Path(row['path'])
            assert not rel.is_absolute() and '..' not in rel.parts
            assert rel.parts[0] in {'docs', 'benchmarks', 'tests'}
            assert rel.suffix not in {'.so', '.o', '.dll', '.a', '.whl', '.gz'}
            if rel.suffix == '.npz':
                assert rel == BASE / 'water/original-fixed-cohort/analysis/curves.npz'
            raw = (ORIGINAL / rel).read_bytes()
            assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'], rel
            if str(rel) in selected:
                assert selected[str(rel)] == row['sha256']
            selected[str(rel)] = row['sha256']
        selected[str(path)] = expected
    copied = []
    for name, expected in sorted(selected.items()):
        source, target = ORIGINAL / name, REPO / name
        if target.exists():
            assert sha(target) == expected, f'Never overwrite an existing different file: {name}'
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            assert sha(target) == expected
            copied.append(name)
    EVIDENCE.mkdir()
    write_new(EVIDENCE / 'copy-receipt.json', {
        'schema': 'pyrpmd.fifth-stage-immutable-copy.v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_root': str(ORIGINAL), 'destination_root': str(REPO),
        'inventories': {str(BASE / k): v for k, v in INVENTORIES.items()},
        'verified_files': selected, 'copied_new_paths': copied,
        'existing_files_overwritten': 0,
        'single_approved_derived_NPZ': str(BASE / 'water/original-fixed-cohort/analysis/curves.npz'),
        'derived_NPZ_scope': 'Saved aggregate correlation curves only; no raw coordinate/momentum/checkpoint trajectory.',
        'full_suite_finished_before_any_copy': True,
    })
    old_selection = json.loads((PRIOR / 'selection.json').read_text())
    files = ['tests/test_h3_p256_complete_analysis.py', 'tests/test_hch4_efficiency_pilot.py', 'tests/test_water_vacf_msd_identity.py']
    selection = {
        'schema': 'pyrpmd.fifth-stage-integration-selection.v1',
        'created_before_execution_utc': datetime.now(timezone.utc).isoformat(),
        'source_head': '36ec289981639bbeb8feecf3b30400b4cbda9afe',
        'source_state_note': 'Published head plus the separately validated synthetic wall-fixture correction and the exact additive frozen files in copy-receipt.json. Original active scientific sources remain unchanged.',
        'native_products': old_selection['native_products'],
        'per_interpreter': {label: {'python': old['python'], 'files': files, 'expected': 135} for label, old in old_selection['per_interpreter'].items()},
        'new_case_families': {'P256_complete_analysis_admission': 43, 'HCH4_efficiency': 85, 'water_VACF_MSD_identity': 7},
        'test_scope': 'Synthetic identities, lifecycle, statistical integrals and analytic VACF/MSD oracles; no actual PES evaluation, real dynamics or live campaign analysis.',
        'required_cpu_affinity': [8], 'builds': 0, 'installs': 0,
        'previous_596_cases_rerun': False,
        'prior_coverage': {'path': str(PRIOR.relative_to(REPO) / 'case-coverage.json'), 'sha256': sha(PRIOR / 'case-coverage.json')},
        'frozen_inventories': [{'path': str(BASE / k), 'sha256': v} for k, v in INVENTORIES.items()],
    }
    write_new(EVIDENCE / 'selection.json', selection)
    guard = json.loads((PRIOR / 'guard-before.json').read_text())
    fixed = 'tests/test_hch4_reference_pilot.py'
    assert sha(REPO / fixed) == 'b19e275b7b6837584afeea8f17f8038ecf162be85f2bfa97447113ee1bd267e1'
    guard['integrated'][fixed] = sha(REPO / fixed)
    for name, digest in selected.items():
        for label in ('original', 'integrated'):
            if name in guard[label]:
                assert guard[label][name] == digest, (label, name)
            guard[label][name] = digest
    write_new(EVIDENCE / 'guard-before.json', guard)
    driver = (PRIOR / 'run_checks.py').read_text()
    begin = driver.index("        expected_test = REPO/'tests/test_h3_p256_parent_archive.py'")
    end = driver.index("        snapshot['actual_loaded_project_modules'] = imported", begin)
    driver = driver[:begin] + '''        collected = {}
        for filename in self.selection['per_interpreter'][self.label]['files']:
            expected_test = REPO/filename
            modules = {id(item.module): item.module for item in session.items
                       if Path(item.path).resolve() == expected_test}
            assert len(modules) == 1, filename
            module = next(iter(modules.values()))
            assert Path(module.__file__).resolve() == expected_test
            collected[filename] = dict(path=str(expected_test), sha256=sha(expected_test))
        snapshot['actual_collected_test_modules'] = collected
''' + driver[end:]
    driver = driver.replace('pyrpmd.auxiliary-proof-python-method-gate.v2', 'pyrpmd.fifth-stage-python-method-gate.v1')
    driver = driver.replace('distribution-check/auxiliary-proof-compatibility-v2', 'distribution-check/fifth-stage-compatibility')
    (EVIDENCE / 'run_checks.py').write_text(driver)
    write_new(EVIDENCE / 'driver-provenance.json', {
        'parent_path': str((PRIOR / 'run_checks.py').relative_to(REPO)),
        'parent_sha256': sha(PRIOR / 'run_checks.py'), 'new_sha256': sha(EVIDENCE / 'run_checks.py'),
        'changes': ['Audit exact three newly collected test modules instead of the previously selected parent-archive test.', 'Use separate gate/evidence/temp identities; preserve the same runtime/native/dependency/source guards.'],
    })
    for label in selection['per_interpreter']:
        (EVIDENCE / label).mkdir()
    print(json.dumps({'copied': len(copied), 'verified_paths': len(selected), 'expected_distinct_cases_each': 135, 'gate': str(EVIDENCE)}))


if __name__ == '__main__':
    main()
