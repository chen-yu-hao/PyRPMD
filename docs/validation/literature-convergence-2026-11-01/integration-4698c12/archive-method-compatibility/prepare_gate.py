"""Copy exact archive tools and completed results; admit only new inert cases."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

WORK = Path('/workspace/scratch/a5e4820daedd')
ORIGINAL = WORK / 'PyRPMD'
REPO = WORK / 'pyrpmd-convergence-integration'
BASE = Path('docs/validation/literature-convergence-2026-11-01')
PRIOR = REPO / BASE / 'integration-4698c12/fifth-stage-compatibility'
OUT = PRIOR.parent / 'archive-method-compatibility'
INVENTORIES = {
    'h3-quantum/umbrella-stage-100000/archive-preparation/publication-inventory.json': '6fff89937fa5e11ef58c4fb70ed3e676a03b6fffd9fce784a000970c20e0f79c',
    'h3-quantum/umbrella-stage-100000/archive-preparation-v2/publication-inventory.json': 'b69ac4725a8ada021d33b51f9e82e58b0fe59bdcd61d5cb5aee103d207b667a1',
    'h3-quantum/umbrella-stage-100000/complete-cohort-analysis/publication-inventory.json': 'e4e7da7f08e53f7e6f1e70e4439a80eb84cc67a7b25c839ddef19884d5d43e24',
    'h3-quantum/umbrella-stage-100000/stale-lease-review/publication-inventory.json': '0dc728a61b04fcc6d72098a7dab562d5eb1df305000f49ba420f962a710cb334',
    'h3-quantum/umbrella-stage-100000/stale-lease-postcondition-review/publication-inventory.json': '606c225a96c35f5e2c5f8259770a46317e5718f8ef75a257d0aee3c3985d264e',
    'review/hch4-reference-efficiency-replay/publication-inventory.json': '0662473b33f03494d58cde0c8a4a8b2dd66d600bf66b89bfd8647d5449490f73',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def main():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == '36ec289981639bbeb8feecf3b30400b4cbda9afe'
    assert not OUT.exists()
    selected = {}
    for name, expected in INVENTORIES.items():
        path = BASE / name
        assert sha(ORIGINAL / path) == expected
        inventory = json.loads((ORIGINAL / path).read_text())
        for row in inventory['files']:
            rel = Path(row['path'])
            assert not rel.is_absolute() and '..' not in rel.parts
            assert rel.parts[0] in {'docs', 'benchmarks', 'tests'}
            assert rel.suffix not in {'.so', '.o', '.dll', '.a', '.whl', '.gz', '.npz'}
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
            assert sha(target) == expected, name
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            assert sha(target) == expected
            copied.append(name)
    OUT.mkdir()
    write_new(OUT / 'copy-receipt.json', {
        'schema': 'pyrpmd.archive-method-immutable-copy.v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'inventories': {str(BASE / k): v for k, v in INVENTORIES.items()},
        'verified_files': selected, 'copied_new_paths': copied,
        'existing_files_overwritten': 0, 'raw_data_or_native_model_copied': False,
    })
    previous = json.loads((PRIOR / 'selection.json').read_text())
    files = ['tests/test_h3_quantum_umbrella_archive.py', 'tests/test_h3_quantum_umbrella_archive_v2.py']
    selection = {
        'schema': 'pyrpmd.archive-method-integration-selection.v1',
        'created_before_execution_utc': datetime.now(timezone.utc).isoformat(),
        'source_head': '36ec289981639bbeb8feecf3b30400b4cbda9afe',
        'source_state_note': 'Published head plus exact frozen additive sources and the previously validated HCH4 wall-fixture correction. Full-project gate and previous135-case gate completed before this copy.',
        'native_products': previous['native_products'],
        'per_interpreter': {label: {'python': row['python'], 'files': files, 'expected': 65} for label, row in previous['per_interpreter'].items()},
        'new_case_families': {'archive_v1_lifecycle_and_exact_replay': 28, 'archive_v2_specific_readonly_lease_admission': 37},
        'test_scope': '65 inert lifecycle, exact replay, checkpoint and explicit anomaly-admission fixtures; no actual campaign archive, PES or analysis execution.',
        'required_cpu_affinity': [8], 'builds': 0, 'installs': 0,
        'previous_731_cases_rerun': False,
        'prior_coverage': {'path': str(PRIOR.relative_to(REPO) / 'case-coverage.json'), 'sha256': sha(PRIOR / 'case-coverage.json')},
        'frozen_inventories': [{'path': str(BASE / k), 'sha256': v} for k, v in INVENTORIES.items()],
    }
    write_new(OUT / 'selection.json', selection)
    guard = json.loads((PRIOR / 'guard-before.json').read_text())
    for name, digest in selected.items():
        for label in ('original', 'integrated'):
            if name in guard[label]:
                assert guard[label][name] == digest, (label, name)
            guard[label][name] = digest
    write_new(OUT / 'guard-before.json', guard)
    driver = (PRIOR / 'run_checks.py').read_text()
    driver = driver.replace('pyrpmd.fifth-stage-python-method-gate.v1', 'pyrpmd.archive-method-python-gate.v1').replace('fifth-stage-compatibility', 'archive-method-compatibility')
    with (OUT / 'run_checks.py').open('x') as stream:
        stream.write(driver)
    write_new(OUT / 'driver-provenance.json', {
        'parent_path': str((PRIOR / 'run_checks.py').relative_to(REPO)),
        'parent_sha256': sha(PRIOR / 'run_checks.py'), 'new_sha256': sha(OUT / 'run_checks.py'),
        'changes': ['Separate evidence/temp/schema identities; exact selection comes from new selection.json. Import, native, source and dependency guards unchanged.'],
    })
    for label in selection['per_interpreter']:
        (OUT / label).mkdir()
    launcher = (WORK / 'run_fifth_stage_checks.py').read_text().replace('fifth-stage-compatibility', 'archive-method-compatibility')
    with (WORK / 'run_archive_method_checks.py').open('x') as stream:
        stream.write(launcher)
    print(json.dumps({'copied': len(copied), 'verified_paths': len(selected), 'expected_distinct_cases_each': 65, 'gate': str(OUT)}))


if __name__ == '__main__':
    main()
