"""Prepare the corrected short audit, preserving all first-attempt evidence."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess

BASE = Path('/workspace/scratch/a5e4820daedd')
REPO = BASE / 'pyrpmd-convergence-integration'
ORIGINAL = BASE / 'PyRPMD'
PREFIX = Path('docs/validation/literature-convergence-2026-11-01')
OLD = REPO / PREFIX / 'integration-4698c12/auxiliary-proof-compatibility'
OUT = OLD.with_name('auxiliary-proof-compatibility-v2')
PORTABLE = PREFIX / 'water/production-stability-portability'
PORTABLE_INVENTORY_SHA = 'cb62d954a7fe314ea29df0af2319d7ff1a2e7bed704a48ca0b151c8cbd9c8a44'
SUPERSESSION_SHA = 'fcef77c2a0813451e53ba8b890b01e2e434c8a137519b427f2e360e67892b82e'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + '\n')

def main():
    prior = json.loads((OLD / 'selection.json').read_text())
    guard = json.loads((OLD / 'guard-before.json').read_text())
    supersession_path = REPO / PORTABLE / 'supersession.json'
    assert sha(supersession_path) == SUPERSESSION_SHA
    supersession = json.loads(supersession_path.read_text())
    assert len(supersession['supersessions']) == 1
    change = supersession['supersessions'][0]
    name = change['current_integration_path']
    assert name == 'tests/test_water_production_stability.py'
    historical = REPO / change['preserved_snapshot_path']
    assert sha(historical) == change['original_sha256'] == guard['integrated'][name]
    expected = historical.read_text().replace(
        'def test_render_roundtrip_and_sidecar(tmp_path):\n',
        'def test_render_roundtrip_and_sidecar(tmp_path):\n    pytest.importorskip("matplotlib")\n')
    # The patch only adds one optional-dependency check inside the renderer test.
    if expected == historical.read_text():
        original_lines = historical.read_text().splitlines(keepends=True)
        current_lines = (REPO / name).read_text().splitlines(keepends=True)
        added = [line for line in current_lines if line not in original_lines]
        assert added == ['    pytest.importorskip("matplotlib")\n']
        assert ''.join(line for line in current_lines if line != added[0]) == historical.read_text()
    else:
        assert (REPO / name).read_text() == expected
    assert sha(REPO / name) == change['current_integration_sha256']
    guard['integrated'][name] = change['current_integration_sha256']
    for label, root in [('integrated', REPO), ('original', ORIGINAL)]:
        for path, expected_sha in guard[label].items():
            if sha(root / path) != expected_sha:
                raise ValueError(f'Protected source changed: {label}/{path}')
        inventory_path = root / PORTABLE / 'publication-inventory.json'
        assert sha(inventory_path) == PORTABLE_INVENTORY_SHA
        guard[label][str(PORTABLE / 'publication-inventory.json')] = PORTABLE_INVENTORY_SHA
        for entry in json.loads(inventory_path.read_text())['files']:
            p = root / entry['path']
            assert sha(p) == entry['sha256'] and p.stat().st_size == entry['bytes']
            guard[label][entry['path']] = entry['sha256']
    for path, expected_sha in guard['external_models'].items():
        assert sha(Path(path)) == expected_sha
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    assert head == prior['source_head']
    guard.update(schema='pyrpmd.auxiliary-proof-integration-guards.v2',
                 created_utc=datetime.now(timezone.utc).isoformat(),
                 first_attempt_guard_sha256=sha(OLD / 'guard-before.json'),
                 integration_only_supersession=dict(path=str(PORTABLE / 'supersession.json'),
                                                    sha256=SUPERSESSION_SHA, change=change))
    source = (OLD / 'run_checks.py').read_text().replace(
        'distribution-check/auxiliary-proof-compatibility',
        'distribution-check/auxiliary-proof-compatibility-v2').replace(
        'pyrpmd.auxiliary-proof-python-method-gate.v1',
        'pyrpmd.auxiliary-proof-python-method-gate.v2')
    old = "        archive_test = sys.modules.get('tests.test_h3_p256_parent_archive')\n        assert archive_test is not None\n"
    new = """        expected_test = REPO/'tests/test_h3_p256_parent_archive.py'
        collected = {id(item.module): item.module for item in session.items
                     if Path(item.path).resolve() == expected_test}
        assert len(collected) == 1
        archive_test = next(iter(collected.values()))
        assert Path(archive_test.__file__).resolve() == expected_test
"""
    assert source.count(old) == 1
    source = source.replace(old, new)
    selection = dict(prior)
    selection.update(schema='pyrpmd.auxiliary-proof-integration-selection.v2',
        created_before_execution_utc=datetime.now(timezone.utc).isoformat(),
        source_state_note='Original running sources remain unchanged. One integration-only rendering test now skips explicitly if its optional matplotlib dependency is absent; all three selected environments provide that dependency.',
        driver_changes=['Use collected item.module/path for the actual archive-auditor import',
                        'Pin this short synthetic gate to CPU 6 so the fixed resource-cap fixture executes',
                        'Use separate v2 evidence/temp paths; retain the entire failed initial attempt'],
        prior_failed_attempt=dict(path=str(OLD.relative_to(REPO)),
            inventory_sha256=sha(OLD / 'publication-inventory.json')),
        integration_only_supersession=dict(path=str(PORTABLE / 'supersession.json'),
            sha256=SUPERSESSION_SHA), required_cpu_affinity=[6])
    selection['frozen_inventories'] = [*prior['frozen_inventories'],
        dict(path=str(PORTABLE / 'publication-inventory.json'), sha256=PORTABLE_INVENTORY_SHA)]
    OUT.mkdir(exist_ok=False)
    with (OUT / 'run_checks.py').open('x') as stream:
        stream.write(source)
    write(OUT / 'selection.json', selection)
    write(OUT / 'guard-before.json', guard)
    for label in selection['per_interpreter']:
        (OUT / label).mkdir()
        (BASE / 'distribution-check/auxiliary-proof-compatibility-v2' / label).mkdir(parents=True)
    print(json.dumps(dict(output=str(OUT), expected_new_cases_each=42, cpu_affinity=[6],
                         integration_only_change=name, all_original_sources_unchanged=True)))

if __name__ == '__main__':
    main()
