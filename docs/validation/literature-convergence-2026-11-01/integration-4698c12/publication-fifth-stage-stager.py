"""Stage only verified frozen inventories and explicitly named integration edits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path('/workspace/scratch/a5e4820daedd/pyrpmd-convergence-integration')
BASE = Path('docs/validation/literature-convergence-2026-11-01')
SUPERSESSION = BASE / 'water/production-stability-portability/supersession.json'
SUPERSESSION_SHA = 'fcef77c2a0813451e53ba8b890b01e2e434c8a137519b427f2e360e67892b82e'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == '36ec289981639bbeb8feecf3b30400b4cbda9afe'
    selected = {}
    if digest(ROOT / SUPERSESSION) != SUPERSESSION_SHA:
        raise ValueError('Unexpected test portability supersession')
    supersession = json.loads((ROOT / SUPERSESSION).read_text())
    if len(supersession['supersessions']) != 1:
        raise ValueError('Only the documented one-test supersession is allowed')
    replacement = supersession['supersessions'][0]
    if (replacement['current_integration_path'] != 'tests/test_water_production_stability.py'
            or digest(ROOT / replacement['preserved_snapshot_path']) != replacement['original_sha256']
            or digest(ROOT / replacement['current_integration_path']) != replacement['current_integration_sha256']
            or digest(ROOT / supersession['old_inventory']['path']) != supersession['old_inventory']['sha256']
            or digest(ROOT / replacement['patch_path']) != replacement['patch_sha256']):
        raise ValueError('Test portability identities differ')
    inventories = [
        BASE / 'surface-partition/controls/publishable-files.json',
        BASE / 'water/stable-publication-manifest.json',
        BASE / 'other-reactions/publish-files.json',
        BASE / 'integration-4698c12/hch4/evidence-manifest.json',
        BASE / 'review/software-and-source-review-files.json',
        BASE / 'review/pr68-software-review/quantum-method-review-files.json',
    ]
    additional_inventories = [
        BASE / 'water/scheduling-v2/publication-inventory.json',
        BASE / 'surface-partition/raw-archive/publishable-files.json',
        BASE / 'water/transport-plot-helper/publication-inventory.json',
        BASE / 'water/preparation-continuation-gate/publication-inventory.json',
        BASE / 'h3-quantum/recrossing-stage-200-pairs-2/publication-inventory.json',
        BASE / 'h3-quantum/recrossing-stage-200-pairs-2/raw-archive/publishable-files.json',
        BASE / 'h3-quantum/umbrella-stage-30000/publication-inventory.json',
        BASE / 'h3-quantum/umbrella-stage-30000/raw-archive/publishable-files.json',
        BASE / 'h3-quantum/umbrella-scheduler/publishable-files.json',
        BASE / 'review/pr68-software-review/complete-200fs-review/publication-inventory.json',
        BASE / 'review/pr68-software-review/umbrella-scheduler-review/publication-inventory.json',
        BASE / 'integration-4698c12/python-313-314/publication-inventory.json',
        BASE / 'integration-4698c12/new-stage-compatibility/publication-inventory.json',
        BASE / 'other-reactions/materials-source-index/publishable-files.json',
        BASE / 'water/preparation-coordinator-gate/publication-inventory.json',
        BASE / 'integration-4698c12/materials-preparation-compatibility/publication-inventory.json',
        BASE / 'other-reactions/fh2-author-static/publication-inventory.json',
        BASE / 'other-reactions/oh-source-consistency/publication-inventory.json',
        BASE / 'review/hch4-reference-feasibility/publication-inventory.json',
        BASE / 'review/hch4-reference-feasibility/pilot-design/publication-inventory.json',
        BASE / 'h3-quantum/p256-dynamics-readiness/publishable-files.json',
        BASE / 'h3-quantum/p256-fixed-eight-chain-design/publishable-files.json',
        BASE / 'h3-quantum/recrossing-stage-200-pairs-4/archive-plan/publishable-files.json',
        BASE / 'water/timestep-paired-control/publication-inventory.json',
        BASE / 'h3-quantum/recrossing-stage-200-pairs-4/publication-inventory.json',
        BASE / 'h3-quantum/recrossing-stage-200-pairs-4/raw-archive/publication-inventory.json',
        BASE / 'h3-quantum/recrossing-stage-200-pairs-4/independent-review/publication-inventory.json',
        BASE / 'review/hch4-reference-pilot/publication-inventory.json',
        BASE / 'water/preparation-plot-helper/publication-inventory.json',
        BASE / 'h3-quantum/p256-parent-tool/publishable-files-v3.json',
        BASE / 'review/p256-parent-tool-independent-review/publication-inventory.json',
        BASE / 'h3-quantum/p256-child-tool/publication-inventory.json',
        BASE / 'h3-quantum/execution-resources-1938-1949/publication-inventory.json',
        BASE / 'h3-quantum/p256-child-coordinator/publication-inventory.json',
        BASE / 'h3-quantum/p256-parent-stage/publication-inventory.json',
        BASE / 'integration-4698c12/four-pair-p256-compatibility/publication-inventory.json',
        BASE / 'h3-quantum/p256-native-calibration/publication-inventory.json',
        BASE / 'h3-quantum/p256-child-launch/publication-inventory.json',
        BASE / 'water/production-stability-helper/publication-inventory.json',
        BASE / 'water/production-stability-portability/publication-inventory.json',
        BASE / 'integration-4698c12/auxiliary-proof-compatibility/publication-inventory.json',
        BASE / 'integration-4698c12/auxiliary-proof-compatibility-v2/publication-inventory.json',
        BASE / 'review/hch4-reference-efficiency-design/publication-inventory.json',
    ]
    new_inventory_hashes = {'water/original-fixed-cohort/publication-inventory.json': 'a547ecba7890116439f7bfd8f908805751d1ce72c68b1716cc67ac2d16d693b2', 'water/original-cohort-archive-replay/publication-inventory.json': 'ed81dc97c108549006bda497f9832c64102e2309d798897f010e533610e43ac6', 'water/vacf-msd-identity/publication-inventory.json': 'ef17405ea42c3068196c5f6d630a2819bcbef51ef862c6128cea2ed3fed799d9', 'h3-quantum/p256-completion-tool/publication-inventory.json': 'c16c2d38582639169cb707b85a956686e89cdf1c604a33ad2d3be32c50328cb5', 'review/hch4-reference-efficiency-implementation/publication-inventory.json': '83d73d906ff8e345ee39761029ad36b889426c7c2c01b7b2ae7781718040511f', 'review/hch4-efficiency-execution-guide/publication-inventory.json': 'fcfb67a0b5db02a1ce50aa3ebf57e888b1182d5c7a3815cc6c55a2c612680272', 'review/pr68-software-review/ci-36ec289-20261101T211518Z/publication-inventory.json': 'cffb2e44d4a5f5dd874246d3b04bf1907ef64c2a05a4c279332230f5a2d10d46', 'h3-quantum/umbrella-stage-100000/archive-preparation/publication-inventory.json': '6fff89937fa5e11ef58c4fb70ed3e676a03b6fffd9fce784a000970c20e0f79c', 'h3-quantum/umbrella-stage-100000/archive-preparation-v2/publication-inventory.json': 'b69ac4725a8ada021d33b51f9e82e58b0fe59bdcd61d5cb5aee103d207b667a1', 'h3-quantum/umbrella-stage-100000/complete-cohort-analysis/publication-inventory.json': 'e4e7da7f08e53f7e6f1e70e4439a80eb84cc67a7b25c839ddef19884d5d43e24', 'h3-quantum/umbrella-stage-100000/stale-lease-review/publication-inventory.json': '0dc728a61b04fcc6d72098a7dab562d5eb1df305000f49ba420f962a710cb334', 'h3-quantum/umbrella-stage-100000/stale-lease-postcondition-review/publication-inventory.json': '606c225a96c35f5e2c5f8259770a46317e5718f8ef75a257d0aee3c3985d264e', 'review/hch4-reference-efficiency-replay/publication-inventory.json': '0662473b33f03494d58cde0c8a4a8b2dd66d600bf66b89bfd8647d5449490f73', 'water/timestep-paired-control/completed-original-cohort/publication-inventory.json': 'ed974cc16cb5220101d2bb274fa9b5ea7137401699c071dd91e252467a03b6e6', 'water/post-cohort-next-stage-proposals/publication-inventory.json': 'bfafe068355b68f36594ba4148452ee4753ea7c6d44fa09ef5204340ccecc5b4', 'water/post-cohort-next-stage-timestep-outcome/publication-inventory.json': '2f60e915183c6924355b34fd57c4da679f0af67dc7c3d9fc662fcf51aab7dc40', 'h3-quantum/umbrella-stage-100000/stale-lease-retirement/publication-inventory.json': 'ee93e0b97db5b00f1b6766f18c9094981ed3d8b1f935f50c8dcbe84bc35da949', 'h3-quantum/p256-completion-tool/root-adopted-waiter/publication-inventory.json': '59950b0596b0afc06a02fd539ba83904b7bf4e87f581389870d32bb2f0de16be', 'h3-quantum/umbrella-stage-100000/raw-archive-v2/publication-inventory.json': '647f691b27a3126a07d2e57bb7fe4bbe3774b99a9a7cb684e0127e3f98267735', 'water/coarse-history-correspondence/publication-inventory.json': '28f547ed41f758e760a78ac561ff3a31d9a284d20c256a1a9b3ac9ea59881356', 'integration-4698c12/full-suite-36ec289-py314/publication-inventory.json': '90b510624794d3b87529250639d2f6cbec5bbf8ad3f9cc665859780989ec2179', 'integration-4698c12/full-suite-36ec289-py314-v2/publication-inventory.json': 'ae52792cf2c3f3c436d1762415c2fc8bd69b7d9151f79efe451a7f8986d4e35f', 'integration-4698c12/hch4-wall-fixture-supersession/publication-inventory.json': '86b1bde83db82173657684555385a8a11c8bec8da431d4d8c69c4b09a2ea6f6c', 'integration-4698c12/hch4-wall-fixture-cross-python/publication-inventory.json': 'ee0f53335877c9493a9bab111bc79c4ebdb0d0d65d97f845e6c2118295acdbdc', 'integration-4698c12/fifth-stage-compatibility/publication-inventory.json': '8b24c3671ebd3886a02a49e4a363be8ce8162490ab9c26acc460808217befdf9', 'integration-4698c12/archive-method-compatibility/publication-inventory.json': '00fe00dfd6633921acf239e567e908fa2a7eea17ed49a2b94dd5ef09ba486210'}
    new_inventory_hashes['h3-quantum/umbrella-stage-100000/completed-archive-review/publication-inventory.json'] = '01bc9fe274f47531fc68e358f02a18e7a7cacf6c725fe2c55ba78b891a1f43ef'
    new_inventory_hashes.update({'water/timestep-paired-control/original-cohort-adoption/publication-inventory.json': '6f4bffaf17b7e7041970c2d5de2d5948bb66080d6e508a24399218203460db56', 'water/timestep-paired-control/original-cohort-adoption-draft/publication-inventory.json': '3dc6fd9739e4d6b22b1a3beb01e6b3aa2d24fc0cf0079aef43fe9060b0407564'})
    for name, expected in new_inventory_hashes.items():
        relative = BASE / name
        if digest(ROOT / relative) != expected:
            raise ValueError(f'New inventory identity changed: {relative}')
        additional_inventories.append(relative)
    old_audit = json.loads((ROOT / BASE / 'integration-4698c12/publication-four-pair-stage-audit-v2.json').read_text())
    for name, expected in old_audit['inventories_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Historical inventory changed: {name}')
    hch4_folder = BASE / 'integration-4698c12/hch4-wall-fixture-supersession'
    hch4_receipt_path = hch4_folder / 'supersession.json'
    hch4_receipt_sha = '42065fe6c3f8833f0c57929603e40c475317b07e5c27dce745db2f482ba1fead'
    if digest(ROOT / hch4_receipt_path) != hch4_receipt_sha:
        raise ValueError('HCH4 test-only supersession changed')
    hch4_receipt = json.loads((ROOT / hch4_receipt_path).read_text())
    hch4_old_inventory = BASE / 'review/hch4-reference-pilot/publication-inventory.json'
    if digest(ROOT / hch4_old_inventory) != '10ce01b004b9870e283a79e746781543edcafa5d7598e6539a21c2bfd008528e':
        raise ValueError('Historical pilot inventory changed')
    hch4_old = hch4_folder / 'test_hch4_reference_pilot.before.py'
    hch4_current = Path('tests/test_hch4_reference_pilot.py')
    if digest(ROOT / hch4_old) != hch4_receipt['before']['sha256'] or digest(ROOT / hch4_current) != hch4_receipt['after']['sha256']:
        raise ValueError('HCH4 test supersession bytes differ')
    if digest(ROOT / hch4_folder / 'fixture-only.patch') != hch4_receipt['patch']['sha256']:
        raise ValueError('HCH4 fixture patch differs')
    # A missing intended bundle is an error, not a silent omission.
    inventories.extend(additional_inventories)
    for relative in inventories:
        inventory = json.loads((ROOT / relative).read_text())
        for entry in inventory['files']:
            path = Path(entry['path'])
            if (str(relative) == supersession['old_inventory']['path']
                    and str(path) == replacement['original_inventory_entry']):
                # Preserve and verify the historical source exactly. Publish the
                # explicitly superseding one-line optional-dependency fix.
                raw = (ROOT / replacement['preserved_snapshot_path']).read_bytes()
                if hashlib.sha256(raw).hexdigest() != entry['sha256'] or len(raw) != entry['bytes']:
                    raise ValueError('Historical test source does not match its frozen inventory')
                selected[str(path)] = replacement['current_integration_sha256']
                continue
            if relative == hch4_old_inventory and path == hch4_current:
                raw = (ROOT / hch4_old).read_bytes()
                if hashlib.sha256(raw).hexdigest() != entry['sha256'] or len(raw) != entry['bytes']:
                    raise ValueError('HCH4 historical snapshot no longer matches the frozen original inventory')
                selected[str(path)] = hch4_receipt['after']['sha256']
                continue
            raw = (ROOT / path).read_bytes()
            if hashlib.sha256(raw).hexdigest() != entry['sha256'] or len(raw) != entry['bytes']:
                raise ValueError(f'Frozen inventory mismatch: {path}')
            selected[str(path)] = entry['sha256']
        selected[str(relative)] = digest(ROOT / relative)

    freeze = json.loads((ROOT / BASE / 'water/code-freeze.json').read_text())
    compatibility = json.loads((ROOT / BASE / 'integration-4698c12/water-complete-check.json').read_text())
    for name, expected in freeze['source_and_test_sha256'].items():
        actual = digest(ROOT / name)
        if name == 'tests/test_water_native_schedule.py':
            if actual != compatibility['final_source_and_test_sha256'][name]:
                raise ValueError('Documented portable scheduler test changed')
        elif actual != expected:
            raise ValueError(f'Frozen water code changed: {name}')
        selected[name] = actual

    explicit = [
        Path('docs/literature-validation.md'),
        Path('benchmarks/literature/README.md'),
        Path('benchmarks/literature/campaign.json'),
        BASE / 'other-reactions/later-evidence.md',
        BASE / 'integration-4698c12/post-gate-literature-inventory-change.json',
        BASE / 'README.md',
        BASE / 'h3-quantum/README.md',
        Path('benchmarks/h3/quantum_convergence_plot.py'),
        BASE / 'water/executed-source/test_water_native_schedule_original.py.txt',
        BASE / 'review/integration-publication-audit-20261101.json',
        BASE / 'other-reactions/fh2-corrected-reference-check-2026-11-01.json',
        BASE / 'review/pr68-software-review/ci-c2c5ae2-20261101T175657Z.json',
        BASE / 'review/pr68-software-review/ci-c2-followup.json',
        BASE / 'review/pr68-software-review/ci-67d55-four-pair-prepublication.json',
    ]
    integration = ROOT / BASE / 'integration-4698c12'
    for pattern in ['README.md', 'water-*.json', 'water-*.xml', 'water-*.log',
                    'documentation-build.log', 'documentation-check.json',
                    'documentation-four-pair-stage*', 'publication-four-pair-stage*',
                    'documentation-fifth-stage*', 'publication-fifth-stage*']:
        explicit.extend(path.relative_to(ROOT) for path in integration.glob(pattern))
    for path in explicit:
        selected[str(path)] = digest(ROOT / path)

    derived_npz = {
        str(BASE / 'water/velocity-cadence/curves.npz'),
        str(BASE / 'water/original-fixed-cohort/analysis/curves.npz'),
        str(BASE / 'water/timestep-paired-control/completed-original-cohort/analysis/curves.npz'),
    }
    for name in selected:
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or path.parts[0] not in {'docs', 'tests', 'benchmarks'}:
            raise ValueError(f'Unexpected publication path: {path}')
        if path.suffix in {'.so', '.o', '.a', '.dll', '.whl'}:
            raise ValueError(f'Compiled or external artifact: {path}')
        if path.suffix == '.npz' and name not in derived_npz:
            raise ValueError(f'Unapproved raw checkpoint: {path}')
    # --force applies only to the individually verified whitelist, including
    # completed log files and the one approved derived-correlation NPZ.
    subprocess.run(['git', 'add', '--force', '--', *sorted(selected)], cwd=ROOT, check=True)
    # Byte-exact process logs and Matplotlib SVG retain producer whitespace;
    # archived unified patches require the leading space on context lines.
    # Their hashes are verified above. Do not normalize executed evidence.
    preserved_generated = ['*.log', '*.svg', '*.patch', '*.xml', '*.csv',
        str(BASE / 'integration-4698c12/water-method-tests.xml'),
        str(BASE / 'surface-partition/controls/comparison-plot-data.csv'),
        str(BASE / 'h3-quantum/recrossing-stage-200-pairs-2/*.csv'),
        str(BASE / 'h3-quantum/recrossing-stage-200-pairs-2/per-seed/*/*/*.csv'),
        str(BASE / 'h3-quantum/recrossing-stage-200-pairs-4/*.csv'),
        str(BASE / 'h3-quantum/recrossing-stage-200-pairs-4/extra-pair-batch-1/*/*/*.csv'),
        str(BASE / 'water/preparation-continuation-gate/*.csv'),
        str(BASE / 'water/rdf-noise-audit/bin-noise.csv'),
        str(BASE / 'integration-4698c12/python-313-314/compiler.txt')]
    preserved_generated.extend(str(BASE / 'review/pr68-software-review/ci-36ec289-20261101T211518Z' / name) for name in ['job-124973258623-failure.txt', 'job-124973261632-failure.txt', 'job-124973261515-failure.txt', 'job-124973261538-failure.txt', 'job-124973261478-failure.txt'])
    # Frozen source snapshots may retain exact benign EOF blanks. Inspect
    # every remaining diagnostic; no trailing whitespace or other defect is waived.
    probe = subprocess.run(['git', 'diff', '--cached', '--check', '--', '.',
        *[':!' + name for name in preserved_generated]], cwd=ROOT, text=True, capture_output=True)
    if probe.stderr or probe.returncode not in (0, 2):
        raise ValueError('Unexpected whitespace-check failure')
    eof_observations = []
    frozen_eof_blanks = []
    for line in probe.stdout.splitlines():
        name, number, diagnostic = line.split(':', 2)
        if name not in selected or not name.endswith('.py') or not number.isdigit() or diagnostic != ' new blank line at EOF.':
            raise ValueError(f'Unapproved whitespace diagnostic: {line}')
        frozen_eof_blanks.append(name)
        eof_observations.append({'path': name, 'sha256': selected[name], 'diagnostic': line})
    preserved_generated.extend(frozen_eof_blanks)
    subprocess.run(['git', 'diff', '--cached', '--check', '--', '.',
        *[':!' + name for name in preserved_generated]], cwd=ROOT, check=True)
    staged = subprocess.check_output(['git', 'diff', '--cached', '--name-only', '-z'], cwd=ROOT)
    changed = [s.decode() for s in staged.split(b'\0') if s]
    if not set(changed).issubset(selected):
        raise ValueError('Index contains an unreviewed path')
    for name in changed:
        raw = subprocess.check_output(['git', 'show', ':' + name], cwd=ROOT)
        if hashlib.sha256(raw).hexdigest() != selected[name]:
            raise ValueError(f'Index differs from verified bytes: {name}')
    tree = subprocess.check_output(['git', 'write-tree'], cwd=ROOT, text=True).strip()
    receipt = {'schema': 'pyrpmd.convergence.fifth-whitelist-staging.v1',
               'inventories': [str(p) for p in inventories],
               'verified_paths': len(selected), 'changed_paths': len(changed),
               'expected_tree': tree, 'paths': changed,
               'sha256': {p: selected[p] for p in changed},
               'whitespace_check_exclusions': preserved_generated,
               'exclusion_reason': 'Byte-exact generated evidence: logs, SVG, compiler version stdout, unified diff context, pytest failure traceback and standard CSV CRLF. All immutable hashes still checked.',
               'excluded_raw_trajectories_and_external_models': True}
    receipt['explicit_test_only_supersession'] = {
        'receipt_path': str(SUPERSESSION), 'receipt_sha256': SUPERSESSION_SHA,
        'historical_snapshot_verified_against_original_inventory': True,
        'replacement': replacement}
    receipt['additional_test_only_supersession'] = {
        'receipt_path': str(hch4_receipt_path), 'receipt_sha256': hch4_receipt_sha,
        'old_inventory_path': str(hch4_old_inventory),
        'old_inventory_sha256': digest(ROOT / hch4_old_inventory),
        'historical_snapshot_path': str(hch4_old), 'current_path': str(hch4_current),
        'historical_sha256': hch4_receipt['before']['sha256'],
        'current_sha256': hch4_receipt['after']['sha256'],
    }
    receipt['allowed_derived_npz'] = sorted(derived_npz)
    receipt['frozen_source_EOF_blank_lines_preserved'] = eof_observations
    out = ROOT.parent / 'convergence-fifth-publication-staging.json'
    out.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k not in {'paths', 'sha256'}}, indent=2))


if __name__ == '__main__':
    prepare()
