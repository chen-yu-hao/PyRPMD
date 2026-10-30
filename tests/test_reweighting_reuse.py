"""Reuse MBAR weights across binning choices only for the identical problem."""

import copy
import pickle

import numpy as np
import pytest

from pyrpmd.analysis.fes import reconstruct_fes, solve_mbar, UmbrellaReweighting
from pyrpmd.analysis.surface import surface_rate_pmf
from pyrpmd.workflows.fes import jsonable


def problem():
    rng = np.random.default_rng(518)
    centers = np.array([[-0.6], [0.0], [0.6]])
    samples = [rng.normal(center / 2, 0.6, size=(700, 1)) for center in centers[:, 0]]
    return samples, centers, np.full((3, 1, 1), 2.0), 2.0


def surface_options():
    return dict(basin=(-0.8, -0.1), dividing_surface=0.5, mass_metric=0.2,
                bins=13, ranges=(-1.8, 1.8), min_count=2, bootstrap=6, seed=342)


def test_reuse_keeps_surface_histogram_and_seeded_draws_while_avoiding_base_solve(monkeypatch):
    import pyrpmd.analysis.surface as module

    samples, centers, stiffness, beta = problem()
    # Deliberately use different FES bins and support threshold. Only weights
    # may be reused; copying the original histogram would change the rate.
    source = reconstruct_fes(samples, centers, stiffness, beta,
                             bins=7, min_count=1, bootstrap=0, return_reweighting=True)
    solve = module.solve_mbar
    calls = []

    def counted(*args, **kwargs):
        calls.append(True)
        return solve(*args, **kwargs)

    monkeypatch.setattr(module, "solve_mbar", counted)
    expected = surface_rate_pmf(samples, centers, stiffness, beta, **surface_options())
    baseline_calls = len(calls)
    calls.clear()
    # The result is also passed through process/MPI serialization in workflows.
    reusable = pickle.loads(pickle.dumps(source["_reweighting"]))
    actual = surface_rate_pmf(samples, centers, stiffness, beta,
                              reweighting=reusable, **surface_options())
    assert actual["ready"]
    assert actual["bootstrap"]["valid"] == 6
    assert baseline_calls == len(calls) + 1 == 7
    assert jsonable(actual) == jsonable(expected)


@pytest.mark.parametrize("changed", ["samples", "counts", "centers", "stiffness", "beta", "overlap"])
def test_reuse_rejects_changed_sampling_problem(changed):
    samples, centers, stiffness, beta = problem()
    source = reconstruct_fes(samples, centers, stiffness, beta, bootstrap=0,
                             return_reweighting=True)
    samples = copy.deepcopy(samples)
    options = surface_options()
    if changed == "samples":
        samples[0][0, 0] += 0.01
    elif changed == "counts":
        # Keep the concatenated coordinates identical but alter stream counts.
        samples[1] = np.concatenate((samples[0][-1:], samples[1]))
        samples[0] = samples[0][:-1]
    elif changed == "centers":
        centers[0, 0] += 0.01
    elif changed == "stiffness":
        stiffness[0, 0, 0] += 0.01
    elif changed == "beta":
        beta += 0.01
    else:
        options["overlap_threshold"] = 0.001
    with pytest.raises(ValueError, match="different samples, umbrellas or overlap"):
        surface_rate_pmf(samples, centers, stiffness, beta,
                         reweighting=source["_reweighting"], **options)


def test_reweighting_owns_immutable_array_snapshots_through_copy_and_pickle():
    reduced = np.array([[0.0, 0.1, 0.3, 0.4], [0.3, 0.4, 0.0, 0.1]])
    counts = np.array([2, 2])
    fit = solve_mbar(reduced, counts)
    fit.diagnostics["nested"] = {"value": [1]}
    context = UmbrellaReweighting.from_result(fit, reduced, counts, 1e-6)
    arrays = ("free_energies", "weights", "overlap", "components")
    expected = {name: getattr(context, name).copy() for name in arrays}
    for name in arrays:
        getattr(fit, name)[:] = 7
    fit.diagnostics["nested"]["value"][0] = 2
    restored = [context, copy.deepcopy(context)]
    restored += [pickle.loads(pickle.dumps(context, protocol=protocol)) for protocol in (4, 5)]
    for snapshot in restored:
        snapshot.validate_problem(reduced, counts, 1e-6)
        assert snapshot.diagnostics["nested"] == {"value": [1]}
        for name in arrays:
            value = getattr(snapshot, name)
            np.testing.assert_array_equal(value, expected[name])
            assert not value.flags.writeable
            with pytest.raises(ValueError):
                value.setflags(write=True)


def test_ordinary_fes_does_not_keep_a_reweighting_snapshot():
    samples, centers, stiffness, beta = problem()
    assert "_reweighting" not in reconstruct_fes(samples, centers, stiffness, beta, bootstrap=0)
