"""Saved output keeps diabatic amplitudes and adiabatic surface counts distinct."""

import json

import numpy as np
import pytest

from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.dynamics import State
from pyrpmd.nonadiabatic import DiabaticResult, NRPMDDynamics
from pyrpmd.nonadiabatic_ensemble import run_nonadiabatic_ensemble
from pyrpmd.surface_hopping import RPSHDynamics, RPSHState
from pyrpmd.workflows.nonadiabatic import _write_results


class ConstantMatrix:
    def evaluate(self, q):
        matrix = np.broadcast_to(np.array([[0., .3], [.3, 0.]]), (len(q), 2, 2)).copy()
        return DiabaticResult(matrix, np.zeros((*matrix.shape, *q.shape[1:])))


def result(method, count, independent=True):
    model = ConstantMatrix()
    nuclear = State(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)))
    if method == "rpsh":
        amplitudes = ([1., 0.], [0., 1.], [1/np.sqrt(2), 1/np.sqrt(2)])
        active = (0, 0, 1)
        parents = [RPSHState(nuclear.q, nuclear.p, coefficients=amplitudes[i],
                             active_surface=active[i]) for i in range(count)]
        def factory(index, rng):
            return RPSHDynamics(model, [1.], 3., 1, .1, 2, native=False, seed=rng)
        options = {}
    else:
        parents = [nuclear.copy() for _ in range(count)]
        def factory(index, rng):
            return NRPMDDynamics(model, [1.], 3., 1, .1, 2, native=False)
        options = {"initial_state": 0}
    return run_nonadiabatic_ensemble(parents, factory, steps=0,
                                     independent=independent, **options)


def write_and_load(tmp_path, ensemble):
    centers = np.zeros((ensemble.ntrajectories, len(ensemble.time), 1, 3))
    _write_results(tmp_path, ensemble, {}, "identity", centers, centers)
    with load_checkpoint(tmp_path/"results/nonadiabatic.npz", require_integrity=True) as archive:
        saved = {name: archive[name] for name in archive.files}
    return saved, json.loads(saved["metadata"].item())


def test_rpsh_output_records_separate_bases_counts_amplitude_means_and_their_sems(tmp_path):
    saved, metadata = write_and_load(tmp_path, result("rpsh", 3))
    assert metadata["population_basis"] == metadata["density_basis"] == "fixed_diabatic"
    assert metadata["amplitude_population_basis"] == "centroid_adiabatic"
    assert metadata["active_surface_basis"] == "centroid_adiabatic"
    assert metadata["population_estimator"] == "amplitude_density"
    assert metadata["statistics_axis"] == "trajectory"
    assert metadata["independent_trajectories"] is True
    assert metadata["sem_unavailable_reason"] is None
    np.testing.assert_allclose(saved["population_mean"], [[.5, .5]], atol=1e-15)
    np.testing.assert_array_equal(saved["active_surface_counts"], [[2, 1]])
    assert saved["active_surface_counts"].dtype.kind in "iu"
    np.testing.assert_allclose(saved["active_surface_fractions"], [[2/3, 1/3]], atol=1e-15)
    np.testing.assert_allclose(saved["active_surface_sem"], [[1/3, 1/3]], atol=1e-15)
    np.testing.assert_allclose(saved["adiabatic_amplitude_population_mean"], [[1/3, 2/3]], atol=1e-15)
    np.testing.assert_allclose(saved["adiabatic_amplitude_population_sem"], [[1/6, 1/6]], atol=1e-15)
    assert saved["adiabatic_amplitude_populations"].shape == (3, 1, 2)
    assert saved["active_surfaces"].shape == (3, 1)


@pytest.mark.parametrize("method", ["nrpmd", "rpsh"])
def test_single_trajectory_output_omits_all_unestimable_sems_but_keeps_means(tmp_path, method):
    saved, metadata = write_and_load(tmp_path, result(method, 1))
    assert metadata["sem_unavailable_reason"] == "fewer_than_two_trajectories"
    for name in ("population_sem", "density_sem_real", "density_sem_imag",
                 "active_surface_sem", "adiabatic_amplitude_population_sem"):
        assert name not in saved
    assert "population_mean" in saved and "density_mean" in saved
    if method == "rpsh":
        np.testing.assert_array_equal(saved["active_surface_counts"], [[1, 0]])
        assert "adiabatic_amplitude_population_mean" in saved
    else:
        assert metadata["population_estimator"] == "signed_mmst"
        assert "active_surface_basis" not in metadata
        assert "amplitude_population_basis" not in metadata
        assert not any(name.startswith(("active_surface", "adiabatic_amplitude")) for name in saved)


def test_output_does_not_assert_independence_or_publish_sems_when_not_requested(tmp_path):
    saved, metadata = write_and_load(tmp_path, result("rpsh", 3, independent=False))
    assert metadata["independent_trajectories"] is False
    assert metadata["sem_unavailable_reason"] == "trajectory_independence_not_asserted"
    assert "population_sem" not in saved
    assert "active_surface_sem" not in saved
    assert "adiabatic_amplitude_population_sem" not in saved
    np.testing.assert_array_equal(saved["active_surface_counts"], [[2, 1]])
    np.testing.assert_allclose(saved["adiabatic_amplitude_population_mean"], [[1/3, 2/3]], atol=1e-15)
