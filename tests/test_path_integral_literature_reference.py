"""Validate the benchmark's independent reference before trusting comparisons."""
import importlib.util
import itertools
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import logsumexp


spec = importlib.util.spec_from_file_location('pi_literature',
    Path(__file__).resolve().parents[1] / 'benchmarks/tier2_literature/path_integral.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def permutation_gaussian_energy(n, p, beta, dimension=2):
    """Integrate every permutation's full Gaussian independently (N! cost)."""
    logs, energies = [], []
    for permutation in itertools.permutations(range(n)):
        precision = np.eye(n*p)*(beta/p)
        for atom in range(n):
            for bead in range(p):
                a = atom*p+bead
                b = atom*p+bead+1 if bead+1 < p else permutation[atom]*p
                delta = np.zeros(n*p)
                delta[a] += 1
                delta[b] -= 1
                precision += (p/beta)*np.outer(delta, delta)
        sign, logdet = np.linalg.slogdet(precision)
        assert sign == 1
        logs.append(-dimension/2*logdet)
        energies.append(dimension*np.trace(np.linalg.inv(precision))/p)
    return np.dot(np.exp(np.array(logs)-logsumexp(logs)), energies)


@pytest.mark.parametrize('n,p,beta', [(1, 1, 2.), (2, 3, 2.), (3, 4, 6.), (4, 5, 6.)])
def test_finite_bead_boson_partition_matches_full_permutation_integrals(n, p, beta):
    assert reference.boson_energy(n, beta, p) == pytest.approx(
        permutation_gaussian_energy(n, p, beta), rel=3e-13)


def test_canonical_bosons_have_correct_one_particle_and_low_temperature_limits():
    for beta in (.5, 2., 6.):
        assert reference.boson_energy(1, beta) == pytest.approx(1/np.tanh(beta/2))
    assert reference.boson_energy(64, 40.) == pytest.approx(64., rel=2e-12)


def test_pigs_projection_depends_on_trial_but_ground_limit_does_not():
    broad, narrow = reference.pigs_variance(2., 129, .25), reference.pigs_variance(2., 129, 2.)
    assert broad[1] > .5 > narrow[1]
    for alpha in (.25, 1., 2.):
        finite, projected = reference.pigs_variance(12., 129, alpha)
        assert abs(projected-.5) < 4e-6
        assert abs(finite-projected) < 6e-4


def test_dimensional_noise_excludes_inactive_coordinates_before_forces():
    rng = reference.DimensionalNoise(np.random.default_rng(81), 2)
    saved = rng.bit_generator.state
    first = rng.normal(size=(36, 4, 3))
    assert np.count_nonzero(first[..., 2]) == 0
    rng.bit_generator.state = saved
    np.testing.assert_array_equal(first, rng.normal(size=(36, 4, 3)))


@pytest.mark.parametrize('drift', ['source', 'case', 'manifest'])
def test_completed_sampling_preflight_preserves_manifest_on_mismatch(tmp_path, monkeypatch, drift):
    definitions = reference.cases(40)[:1]
    case, seed = definitions[0], reference.SEEDS[0]
    provenance = reference.source_provenance()
    manifest = dict(cases=definitions, seeds=list(reference.SEEDS),
                    provenance=copy.deepcopy(provenance), original_marker='must survive')
    completed = dict(case=copy.deepcopy(case), seed=seed,
                     provenance=copy.deepcopy(provenance), mean=1.23)
    path = tmp_path/'jobs'/f'{reference.job_key(case, seed)}.json'
    reference.write_json(path, completed)
    assert reference.check_completed(path, case, seed, provenance)['mean'] == 1.23
    if drift == 'source':
        completed['provenance']['source_sha256']['driver'] = 'different source'
    elif drift == 'case':
        completed['case']['gamma_centroid'] = 2.
    else:
        manifest['provenance']['source_sha256']['driver'] = 'different source'
    reference.write_json(path, completed)
    manifest_path = tmp_path/'manifest.json'
    reference.write_json(manifest_path, manifest)
    before_manifest, before_result = manifest_path.read_bytes(), path.read_bytes()
    monkeypatch.setattr(sys, 'argv', ['path_integral.py', '--mode', 'all',
        '--steps', '40', '--case-limit', '1', '--output', str(tmp_path)])
    monkeypatch.setattr(reference, 'ProcessPoolExecutor',
                        lambda **kwargs: pytest.fail('workers started before cache preflight'))
    with pytest.raises(ValueError, match='(?i)(source provenance|parameters differ)'):
        reference.main()
    assert manifest_path.read_bytes() == before_manifest
    assert path.read_bytes() == before_result
    assert not (tmp_path/'suzuki_chin_convergence.csv').exists()


@pytest.mark.parametrize('drift', ['source', 'case', 'legacy'])
def test_partial_sampling_checkpoint_rejects_identity_before_restoring_state(tmp_path, monkeypatch, drift):
    case, seed = reference.cases(40)[0], reference.SEEDS[0]
    provenance = reference.source_provenance()
    path = tmp_path/'jobs'/f'{reference.job_key(case, seed)}.npz'
    path.parent.mkdir()
    metadata = dict(case_json=json.dumps(case), seed=seed,
                    provenance_json=json.dumps(provenance))
    np.savez_compressed(path, **metadata)
    reference.check_checkpoint(path, case, seed, provenance)
    if drift == 'source':
        changed = copy.deepcopy(provenance)
        changed['source_sha256']['driver'] = 'different source'
        metadata['provenance_json'] = json.dumps(changed)
    elif drift == 'case':
        metadata['case_json'] = json.dumps(dict(case, dimension=1, burn=0))
    else:
        metadata = dict(q=np.zeros((36, 2, 3)))
    np.savez_compressed(path, **metadata)
    before = path.read_bytes()
    monkeypatch.setattr(reference, 'Dynamics',
                        lambda *args, **kwargs: pytest.fail('engine created before cache preflight'))
    with pytest.raises(ValueError, match='Checkpoint'):
        reference.run_job(case, seed, tmp_path)
    assert path.read_bytes() == before
    assert not path.with_suffix('.json').exists()
