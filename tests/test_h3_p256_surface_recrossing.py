"""P256 parent binding contracts with synthetic data; no PES is evaluated."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.h3 import p256_surface_recrossing as child


def put_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def file_row(path):
    raw = path.read_bytes()
    return {'path': str(path.resolve()), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


@pytest.fixture
def cohort(tmp_path):
    specification = json.loads(child.DESIGN.read_text())
    parent_root = tmp_path / 'parents'
    parent_root.mkdir()
    registration = {'schema': 'pyrpmd.h3.p256-eight-parent-registration.v1',
                    'design_sha256': child.DESIGN_SHA, 'fixed_seeds': list(child.SEEDS),
                    'output': str(parent_root),
                    'driver_sha256': child.sha(child.ROOT / 'benchmarks/h3/p256_parent_campaign.py')}
    put_json(parent_root / 'registration.json', registration)
    rows, geometries = [], {}
    for expected in specification['parent_generation']['cohorts']:
        seed = expected['seed']
        directory = parent_root / expected['relative_directory']
        settings = expected['expected_settings']
        put_json(directory / 'settings.json', settings)
        put_json(directory / 'reference.json', {'synthetic': True})
        put_json(directory / 'reference-domain.json', {'synthetic': True, 'domain': [.5, 8.]})
        q = np.broadcast_to(np.array([[-1.757, 0., 0.], [0., 0., 0.], [1.757, 0., 0.]]),
                            (200, 256, 3, 3)).copy()
        child.atomic_savez(directory / 'checkpoint.npz', kind='h3.surface-partition',
            fingerprint=expected['expected_settings_fingerprint'],
            domain_sha256=child.sha(directory / 'reference-domain.json'),
            completed=22000, endpoint_q=q, endpoint_completed_sweeps=np.arange(2100, 22001, 100))
        report = {'complete': True, 'source_unchanged': True, 'phase': 'production',
                  'surface': 'ts', 'seed': seed, 'beads': 256,
                  'checkpoint_sha256': child.sha(directory / 'checkpoint.npz'),
                  'analysis': {'log_quantum_classical_ratio': -2.2}}
        put_json(directory / 'report.json', report)
        replay = None if seed in child.SEEDS[4:] else {
            'bitwise_verified_array_count': 16,
            'bitwise_verified_arrays': expected['exact_original_payloads'],
            'reference_bytes_equal': True, 'scientific_report_equal': True}
        row = {'seed': seed, 'cohort_role': expected['cohort_role'], 'complete': True,
               'identity_and_geometry_verified': True, 'sampling_flags_passed': True,
               'parents': 200, 'beads': 256, 'completed_sweeps': 22000,
               'settings_fingerprint': expected['expected_settings_fingerprint'],
               'files': {name: file_row(directory / name) for name in child.FILES},
               'checkpoint_sha256': child.sha(directory / 'checkpoint.npz'),
               'historical_replay': replay, 'minimum_BAR_overlap': .95,
               'minimum_energy_ESS': 1000., 'half_log_partition_difference': .01,
               'BAR_log_TS_quantum_classical_ratio': -2.2}
        rows.append(row)
        geometries[seed] = q
    manifest = {'schema': 'pyrpmd.h3.p256-eight-parent-completion.v1',
                'eligible_for_separately_authorized_children': True,
                'all_eight_counts_complete_and_verified': True,
                'all_owned_workers_drained': True, 'four_chain_interim_selection': False,
                'independent_parent_chains': 8, 'errors': [], 'unstarted_seeds': [],
                'seed_order': list(child.SEEDS), 'parents': rows,
                'design_sha256': child.DESIGN_SHA,
                'registration_sha256': child.sha(parent_root / 'registration.json'),
                'sampling_flags': specification['parent_generation']['sampling_flags']}
    put_json(parent_root / 'parents-complete.json', manifest)
    protocol = {'design_sha256': child.DESIGN_SHA, 'parent_root': str(parent_root),
                'parent_manifest_sha256': child.sha(parent_root / 'parents-complete.json'),
                'parent_registration_sha256': manifest['registration_sha256'],
                'beta': child.BETA, 'mass_au': child.MASS_AU,
                'model': specification['parent_generation']['cohorts'][0]['expected_settings']['model']}
    return parent_root, specification, manifest, protocol, geometries


def save_manifest(cohort):
    root, _, manifest, protocol, _ = cohort
    put_json(root / 'parents-complete.json', manifest)
    protocol['parent_manifest_sha256'] = child.sha(root / 'parents-complete.json')


def update_parent(cohort, seed, *, q=None, sweeps=None, report_changes=None):
    root, specification, manifest, _, geometries = cohort
    expected = next(row for row in specification['parent_generation']['cohorts'] if row['seed'] == seed)
    directory = root / expected['relative_directory']
    row = next(row for row in manifest['parents'] if row['seed'] == seed)
    if q is not None or sweeps is not None:
        child.atomic_savez(directory / 'checkpoint.npz', kind='h3.surface-partition',
            fingerprint=expected['expected_settings_fingerprint'],
            domain_sha256=child.sha(directory / 'reference-domain.json'), completed=22000,
            endpoint_q=geometries[seed] if q is None else q,
            endpoint_completed_sweeps=np.arange(2100, 22001, 100) if sweeps is None else sweeps)
    report = json.loads((directory / 'report.json').read_text())
    report['checkpoint_sha256'] = child.sha(directory / 'checkpoint.npz')
    report.update(report_changes or {})
    put_json(directory / 'report.json', report)
    row['checkpoint_sha256'] = report['checkpoint_sha256']
    row['files'] = {name: file_row(directory / name) for name in child.FILES}
    save_manifest(cohort)
    return directory


def test_full_eight_parent_binding_and_nested_source_routes(cohort):
    root, specification, manifest, protocol, _ = cohort
    assert child.verify_manifest(root, specification) == manifest
    for expected in specification['parent_generation']['cohorts']:
        q, sweeps, identity = child.load_parents(root / expected['relative_directory'], protocol)
        assert q.shape == (200, 256, 3, 3)
        assert len(sweeps) == 200 and sweeps[0] == 2100 and sweeps[-1] == 22000
        assert identity['seed'] == expected['seed']
        assert identity['parent_manifest_sha256'] == protocol['parent_manifest_sha256']


@pytest.mark.parametrize('change', ['omit_chain', 'reorder', 'undrained', 'failed_sampling',
                                   'forged_replay', 'partial_count', 'nonfinite', 'omitted_replay_payload'])
def test_ineligible_or_incomplete_cohort_is_rejected(cohort, change):
    root, specification, manifest, _, _ = cohort
    if change == 'omit_chain': manifest['parents'].pop()
    elif change == 'reorder': manifest['parents'].reverse()
    elif change == 'undrained': manifest['all_owned_workers_drained'] = False
    elif change == 'failed_sampling': manifest['parents'][3]['sampling_flags_passed'] = False
    elif change == 'forged_replay': manifest['parents'][5]['historical_replay'] = {'fake': True}
    elif change == 'partial_count': manifest['parents'][1]['parents'] = 199
    elif change == 'nonfinite': manifest['parents'][2]['minimum_energy_ESS'] = float('nan')
    else: manifest['parents'][0]['historical_replay']['bitwise_verified_arrays'] = ['omitted'] * 16
    save_manifest(cohort)
    with pytest.raises(ValueError): child.verify_manifest(root, specification)


@pytest.mark.parametrize('name', child.FILES)
def test_manifest_rejects_tampered_file_bytes(cohort, name):
    root, specification, manifest, _, _ = cohort
    path = Path(manifest['parents'][0]['files'][name]['path'])
    path.write_bytes(path.read_bytes() + b'corruption')
    with pytest.raises(ValueError, match='Bound parent file changed'):
        child.verify_manifest(root, specification)


@pytest.mark.parametrize('change', ['subsequence', 'wrong_beads', 'off_surface', 'COM', 'domain', 'reordered_sweeps', 'NaN'])
def test_parent_payload_geometry_and_sequence_rejected_even_with_updated_hashes(cohort, change):
    root, specification, manifest, protocol, geometries = cohort
    seed = child.SEEDS[0]
    q, sweeps = geometries[seed].copy(), np.arange(2100, 22001, 100)
    if change == 'subsequence': q = q[:199]
    elif change == 'wrong_beads': q = q[:, :128]
    elif change == 'off_surface': q[:, :, 0, 0] -= .1
    elif change == 'COM': q += .1
    elif change == 'domain': q *= .1
    elif change == 'reordered_sweeps': sweeps = sweeps[::-1]
    else: q[0, 0, 0, 0] = np.nan
    directory = update_parent(cohort, seed, q=q, sweeps=sweeps)
    with pytest.raises(ValueError): child.load_parents(directory, protocol)


def test_parent_calibration_and_subset_bypasses_are_rejected():
    for kwargs in ({'calibration': True}, {'count': 1}):
        with pytest.raises(ValueError, match='no calibration/subset bypass'):
            child.load_parents(Path('unused'), {}, **kwargs)


def test_stale_checkpoint_summary_and_report_cannot_bypass_actual_file_identity(cohort):
    root, specification, manifest, protocol, _ = cohort
    row = manifest['parents'][0]
    stale_hash = '0' * 64
    path = Path(row['files']['report.json']['path'])
    report = json.loads(path.read_text())
    report['checkpoint_sha256'] = stale_hash
    put_json(path, report)
    row['checkpoint_sha256'] = stale_hash
    row['files']['report.json'] = file_row(path)
    save_manifest(cohort)
    with pytest.raises(ValueError, match='checkpoint summary and file inventory differ'):
        child.verify_manifest(root, specification)
    with pytest.raises(ValueError, match='checkpoint summary and file inventory differ'):
        child.load_parents(path.parent, protocol)


def test_child_routes_registered_nested_cohort_before_any_dynamics(cohort, monkeypatch, tmp_path):
    root, specification, _, protocol, _ = cohort
    seen = []
    monkeypatch.setattr(child, 'register', lambda output, library: protocol)
    def stopped(directory, protocol):
        seen.append(directory)
        raise RuntimeError('stop-before-PES')
    monkeypatch.setattr(child, 'load_parents', stopped)
    for seed in (child.SEEDS[0], child.SEEDS[-1]):
        with pytest.raises(RuntimeError, match='stop-before-PES'):
            child.run_seed(tmp_path/'children', Path('unused'), seed, .05)
        expected = next(row for row in specification['parent_generation']['cohorts'] if row['seed'] == seed)
        assert seen[-1] == root / expected['relative_directory']


def test_registered_children_reject_changed_analysis_identity(cohort,monkeypatch,tmp_path):
    root,specification,_,_,_=cohort
    monkeypatch.setattr(child,'PARENTS',root)
    baseline=tmp_path/'baseline.json'
    put_json(baseline,{'completed_seeds':list(child.P128_SEEDS),'expected_seeds':list(child.P128_SEEDS),
        'criteria':{'complete_registered_cohort':True},'pair_batches':[0,1],'horizon_fs':200})
    monkeypatch.setattr(child,'BASELINE',baseline)
    monkeypatch.setattr(child,'BASELINE_SHA',child.sha(baseline))
    identity=dict(specification['child_experiment_definition']['PES_and_all_scientific_dependencies'])
    identity['benchmarks/h3/p256_recrossing_analyze.py']='1'*64
    monkeypatch.setattr(child,'source_identity',lambda:dict(identity))
    class MetadataOnlyModel:
        def __init__(self,*args):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        @property
        def metadata(self):return specification['parent_generation']['cohorts'][0]['expected_settings']['model']
    monkeypatch.setattr(child,'BKMP2Potential',MetadataOnlyModel)
    output=tmp_path/'children'
    child.register(output,Path('unused'))
    before=(output/'protocol.json').read_bytes()
    identity['benchmarks/h3/p256_recrossing_analyze.py']='2'*64
    with pytest.raises(ValueError,match='Registered P256 children or sources changed'):
        child.register(output,Path('unused'))
    assert (output/'protocol.json').read_bytes()==before


@pytest.mark.parametrize('kwargs', [{'seed': 721301}, {'horizon': 400}, {'pair_batch': 2},
                                   {'timestep': .025}, {'stop_after_chunks': 0}])
def test_unregistered_child_request_rejected_before_registration(tmp_path, monkeypatch, kwargs):
    def forbidden(*args, **kw):
        raise AssertionError('registration must not run')
    monkeypatch.setattr(child, 'register', forbidden)
    request = dict(seed=child.SEEDS[0], timestep=.05)
    request.update(kwargs)
    with pytest.raises(ValueError): child.run_seed(tmp_path, Path('unused'), **request)
