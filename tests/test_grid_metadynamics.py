"""Independent grid-bias accuracy, conservation, event and restart checks."""

import copy
import json

import numpy as np
import pytest

from pyrpmd.analysis.metadynamics import bias_fes
from pyrpmd.bias import GridMetadynamicsBias, MetadynamicsBias
from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.cv import CVStack, Cartesian, Distance
from pyrpmd.input import InputError, JobSpec
from pyrpmd.system import System
from pyrpmd.workflows.metadynamics import (
    _sample_replica, reanalyze_metadynamics, resolve_metadynamics, run_metadynamics,
)


def _cv(dimension=1, periodic=False):
    axes = [Cartesian(0, j) for j in range(dimension)]
    if periodic:
        axes[0].periods = (4.0,)
    return CVStack(axes)


def _bias(dimension=1, periodic=False, bins=80, hills=None, **kwargs):
    return GridMetadynamicsBias(_cv(dimension, periodic), [0.4] * dimension,
        0.1, 2.0, grid_ranges=[[-2.0, 2.0]] * dimension, grid_bins=bins,
        mode="standard", hills=hills, **kwargs)


@pytest.mark.parametrize("dimension", [1, 2])
@pytest.mark.parametrize("periodic", [False, True])
def test_grid_energy_gradient_agree_with_direct_hills_and_refine(dimension, periodic):
    direct = MetadynamicsBias(_cv(dimension, periodic), [0.4] * dimension,
                             0.1, 2.0, mode="standard")
    rng = np.random.default_rng(42)
    for i, center in enumerate(rng.uniform(-1.9, 1.9, (15, dimension))):
        direct.deposit(center, step=i + 1, time=(i + 1) * 0.01)
    points = rng.uniform(-2, 2, (250, dimension))
    expected = direct.physical_energy_gradient(points)
    errors = []
    for bins in (40, 80):
        grid = _bias(dimension, periodic, bins=bins, hills=direct.hills)
        actual = grid.physical_energy_gradient(points)
        errors.append([np.max(np.abs(a-b)) for a, b in zip(actual, expected)])
    assert errors[1][0] < 3e-6
    assert errors[1][1] < 2e-4
    assert errors[1][0] < errors[0][0] / 9
    assert errors[1][1] < errors[0][1] / 5


@pytest.mark.parametrize("dimension", [1, 2])
def test_hermite_force_is_derivative_of_applied_energy_even_on_coarse_grid(dimension):
    grid = _bias(dimension, bins=9)
    grid.deposit([0.19] * dimension, step=1, time=0.1)
    points = np.random.default_rng(91).uniform(-1.7, 1.7, (30, dimension))
    energy, gradient = grid.physical_energy_gradient(points)
    for axis in range(dimension):
        plus, minus = points.copy(), points.copy()
        plus[:, axis] += 1e-6
        minus[:, axis] -= 1e-6
        numerical = (grid.physical_energy(plus) - grid.physical_energy(minus)) / 2e-6
        np.testing.assert_allclose(gradient[:, axis], numerical, atol=4e-11, rtol=3e-6)
    assert np.isfinite(energy).all()


@pytest.mark.parametrize("sigma", [0.2, 1.4, 5.0])
def test_grid_periodic_seam_has_continuous_energy_force_and_exact_images(sigma):
    grid = GridMetadynamicsBias(_cv(2, True), [sigma, 0.4], 0.2, 1.0,
        grid_ranges=[[-2, 2], [-2, 2]], grid_bins=[120, 80], mode="standard")
    grid.deposit([1.91, -0.12], step=1, time=0.1)
    points = np.array([[-2.0, 0.1], [2.0, 0.1], [6.0, 0.1]])
    energies, gradients = grid.physical_energy_gradient(points)
    np.testing.assert_array_equal(energies, np.repeat(energies[:1], 3))
    np.testing.assert_array_equal(gradients, np.repeat(gradients[:1], 3, axis=0))
    epsilon = 1e-8
    seam = np.array([[2 - epsilon, 0.1], [-2 + epsilon, 0.1]])
    e, g = grid.physical_energy_gradient(seam)
    np.testing.assert_allclose(e[0], e[1], atol=2e-8, rtol=0)
    np.testing.assert_allclose(g[0], g[1], atol=2e-7, rtol=0)


@pytest.mark.parametrize("estimator", ["centroid", "bead_average"])
def test_grid_full_ring_jacobian_and_physical_bead_scaling(estimator):
    cv = CVStack([Distance([0, 1], estimator=estimator), Cartesian(1, 1, estimator=estimator)])
    grid = GridMetadynamicsBias(cv, [0.3, 0.4], 0.1, 2.0,
        grid_ranges=[[0.3, 2.4], [-2, 2]], grid_bins=[70, 60], bias_factor=5)
    grid.deposit([1.1, -0.1], step=1, time=0.1)
    q = np.tile([[-0.4, 0.1, 0.2], [0.8, -0.2, 0.3]], (4, 1, 1))
    q += np.random.default_rng(42).normal(scale=0.03, size=q.shape)
    energy, force = grid.energy_force(q)
    assert energy == pytest.approx(4 * grid.physical_energy(cv.value(q)))
    for index in np.ndindex(q.shape):
        plus, minus = q.copy(), q.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        numerical = -(grid.energy_force(plus)[0] - grid.energy_force(minus)[0]) / 2e-6
        assert force[index] == pytest.approx(numerical, abs=5e-10)


def test_well_tempered_heights_use_the_applied_grid_bias():
    grid = GridMetadynamicsBias(_cv(), [0.4], 0.3, 2.0,
        grid_ranges=[[-2, 2]], grid_bins=10, bias_factor=4)
    grid.deposit([0.13], step=1, time=0.1)
    before = float(grid.physical_energy([0.17]))
    hill = grid.deposit([0.17], step=2, time=0.2)
    assert hill["bias_before"] == before
    assert hill["height"] == pytest.approx(0.3 * np.exp(-2 * before / 3))


def test_grid_lookup_and_deposition_do_not_resum_history(monkeypatch):
    grid = _bias()
    grid.deposit([0.0], step=1, time=0.1)
    class NoHistoryIteration(list):
        def __iter__(self):
            raise AssertionError("lookup/deposition iterated old hills")
    grid._hills = NoHistoryIteration(grid._hills)
    def no_history_snapshot(self):
        raise AssertionError("lookup/deposition copied public hill history")
    with monkeypatch.context() as patch:
        patch.setattr(MetadynamicsBias, "hills", property(no_history_snapshot))
        grid.physical_energy_gradient([[0.0], [0.3]])
        grid.physical_energy([[0.0], [0.3]])
        grid.deposit([0.1], step=2, time=0.2)
        assert grid.hill_count == 2


def test_grid_domain_failure_and_failed_append_leave_history_and_grid_unchanged():
    grid = _bias()
    grid.deposit([0.0], step=1, time=0.1)
    before = grid.grid_state()["values"].copy()
    for point in ([-2.01], [2.01], [np.nan]):
        with pytest.raises(ValueError):
            grid.deposit(point, step=2, time=0.2)
        np.testing.assert_array_equal(grid.grid_state()["values"], before)
        assert len(grid.hills) == 1
    with pytest.raises(ValueError, match="strictly increase"):
        grid.deposit([0.2], step=1, time=0.2)
    np.testing.assert_array_equal(grid.grid_state()["values"], before)
    grid.physical_energy([[-2], [2]])


@pytest.mark.parametrize("rollback", ["truncate_history", "truncate_hills", "rollback_hills"])
def test_cache_restore_replay_and_rollback_are_bitwise_identical(rollback):
    grid = _bias(2, True)
    for i in range(9):
        grid.deposit([np.sin(i), np.cos(i)], step=i + 1, time=(i + 1) * 0.1)
    cache = grid.grid_state()
    exact = cache["values"].copy()
    restored = _bias(2, True, hills=grid.hills, grid_state=cache)
    replay = _bias(2, True, hills=grid.hills)
    np.testing.assert_array_equal(restored.grid_state()["values"], exact)
    np.testing.assert_array_equal(replay.grid_state()["values"], exact)
    grid.deposit([0.33, 0.12], step=10, time=1.0)
    getattr(grid, rollback)(9)
    np.testing.assert_array_equal(grid.grid_state()["values"], exact)
    bad = copy.deepcopy(cache)
    bad["metadata"]["history_sha256"] = "changed"
    with pytest.raises(ValueError, match="cache disagrees"):
        _bias(2, True, hills=grid.hills, grid_state=bad)


@pytest.mark.parametrize("ranges,bins,periodic", [
    ([[-2, 2]], True, False), ([[-2, 2]], 2.1, False),
    ([[-2, 2]], 1, False), ([[-2, 2]], 1_000_001, False),
    ([[-2, 1]], 10, True), ([[2, -2]], 10, False),
    ([[-2, np.inf]], 10, False), ([[-2, 2], [-2, 2]], 10, False),
])
def test_invalid_grid_geometry_is_rejected(ranges, bins, periodic):
    with pytest.raises(ValueError):
        GridMetadynamicsBias(_cv(1, periodic), [0.4], 0.1, 2,
                              grid_ranges=ranges, grid_bins=bins)


def _configuration(grid=True):
    config = {
        "task": "metadynamics", "beta": 2.0, "beads": 3, "dt_au": 0.02,
        "seed": 321, "native": False,
        "potential": {"provider": "periodic_cosine", "params": {"height": 0.2, "confinement": 0.1}},
        "sampling": {"equilibration_steps": 3, "production_steps": 24,
                     "sample_interval": 2, "andersen_interval": 4, "replicas": 1},
        "cvs": [{"name": "x", "kind": "surface_lateral", "atoms": [1], "reference_atoms": [0]}],
        "metadynamics": {"sigma": [0.3], "height": 0.05, "pace": 3, "bias_factor": 5.0},
        "analysis": {"bins": 6, "bootstrap": 0},
        "output": {"checkpoint_steps": 5},
    }
    if grid:
        config["metadynamics"]["grid"] = {"ranges": [[-2, 2]], "bins": 70}
    system = System(("Cu", "H"), [10.0, 2.0], [[0.2, 0.1, 0.0], [0.3, -0.05, 0.2]],
                    cell=np.diag([4., 4., 8.]), pbc=[True, True, False], fixed_atoms=[0])
    return config, system


@pytest.mark.parametrize("grid_backend", [False, True])
def test_external_thread_executor_retains_all_metadynamics_replicas(tmp_path, grid_backend):
    from concurrent.futures import ThreadPoolExecutor

    config, system = _configuration(grid_backend)
    config["sampling"]["replicas"] = 2
    expected = run_metadynamics(config, system, tmp_path / "serial", return_data=True)
    with ThreadPoolExecutor(max_workers=2) as executor:
        actual = run_metadynamics(config, system, tmp_path / "threads", executor=executor,
                                  return_data=True)
    assert actual["summary"]["samples"] == expected["summary"]["samples"] == 24
    assert actual["summary"]["replicas"] == len(actual["streams"]) == 2
    assert {s["replica"] for s in actual["streams"]} == {0, 1}
    for key in ("free_energy", "replica_spread", "mask", "raw_count"):
        np.testing.assert_array_equal(actual["data"][key], expected["data"][key])
    for a, b in zip(actual["streams"], expected["streams"]):
        for key in ("samples", "sample_steps", "sample_bias", "sample_energies"):
            np.testing.assert_array_equal(a[key], b[key])
        for key in ("q", "p"):
            np.testing.assert_array_equal(getattr(a["state"], key), getattr(b["state"], key))
        assert a["hills"] == b["hills"]
        assert a["rng"] == b["rng"]
        if grid_backend:
            np.testing.assert_array_equal(a["grid_state"]["values"], b["grid_state"]["values"])


@pytest.mark.parametrize("grid_backend", [False, True])
@pytest.mark.parametrize("integrator", ["cayley-bcb", "normal-mode-bab", "velocity-verlet"])
def test_fixed_periodic_metadynamics_failure_resume_and_reanalysis(tmp_path, monkeypatch, grid_backend, integrator):
    config, system = _configuration(grid_backend)
    config["dynamics"] = {"integrator": integrator}
    baseline = run_metadynamics(config, system, tmp_path / "full", return_data=True)
    bias_type = GridMetadynamicsBias if grid_backend else MetadynamicsBias
    original = bias_type.deposit
    def interrupted(self, values, **kwargs):
        result = original(self, values, **kwargs)
        if kwargs["step"] == 12:
            raise RuntimeError("after deposition")
        return result
    task = {"config": config, "system": system, "output": str(tmp_path / "resume"), "replica": 0}
    with monkeypatch.context() as patch:
        patch.setattr(bias_type, "deposit", interrupted)
        with pytest.raises(RuntimeError, match="after deposition"):
            _sample_replica(task)
    resumed = run_metadynamics(config, system, tmp_path / "resume", return_data=True)
    a, b = baseline["streams"][0], resumed["streams"][0]
    for field in ("q", "p"):
        np.testing.assert_array_equal(getattr(a["state"], field), getattr(b["state"], field))
    np.testing.assert_array_equal(b["state"].q[:, 0], np.broadcast_to(system.geometry[0], (3, 3)))
    np.testing.assert_array_equal(b["state"].p[:, 0], 0)
    for field in ("samples", "sample_steps", "sample_bias", "sample_energies"):
        np.testing.assert_array_equal(a[field], b[field])
    assert a["hills"] == b["hills"]
    assert a["rng"] == b["rng"]
    assert a["propagation"] == b["propagation"]
    if grid_backend:
        np.testing.assert_array_equal(a["grid_state"]["values"], b["grid_state"]["values"])
    again = reanalyze_metadynamics(config, system, tmp_path / "resume", return_data=True)
    np.testing.assert_array_equal(resumed["data"]["free_energy"], again["data"]["free_energy"])
    assert resumed["summary"]["bias_backend"] == ("grid" if grid_backend else "direct")


@pytest.mark.parametrize("corruption", ["q", "p", "shape", "time"])
def test_completed_metadynamics_checkpoint_validates_fixed_state(tmp_path, corruption):
    config, system = _configuration()
    task = {"config": config, "system": system, "output": str(tmp_path), "replica": 0}
    _sample_replica(task)
    path = tmp_path / "checkpoints/metadynamics_0000.npz"
    with load_checkpoint(path) as cp:
        data = {k: cp[k] for k in cp.files}
    if corruption in ("q", "p"):
        data[corruption][0, 0, 0] += 0.01
    elif corruption == "shape":
        data["q"], data["p"] = data["q"][:1], data["p"][:1]
    else:
        data["time"] = float(data["time"]) + 1
    atomic_savez(path, **data)
    with pytest.raises(ValueError, match="fixed atoms|state shape|time disagrees"):
        _sample_replica(task)


def test_grid_frontend_explicit_and_declared_units_and_defaults():
    job = {"task": "metadynamics", "temperature": "300 K", "beads": 2,
           "system": {"symbols": ["H"], "geometry": [[0, 0, 0]]},
           "potential": {"provider": "harmonic"},
           "sampling": {"production_steps": 20, "interval_steps": 2},
           "cvs": [{"name": "x", "kind": "cartesian", "atom": 0, "axis": "x", "unit": "angstrom"}],
           "metadynamics": {"sigma": 0.2, "height": "0.01 eV", "pace": 5,
                            "grid": {"ranges": [[-1, "1 angstrom"]]}}}
    cfg, system = JobSpec.from_mapping(job).resolve()
    _, _, options = resolve_metadynamics(cfg, system)
    np.testing.assert_allclose(options["grid"]["ranges"], [[-1.8897261246, 1.8897261246]], rtol=1e-9)
    assert options["grid"]["bins"] == [50]
    job["metadynamics"]["grid"]["unknown"] = 1
    with pytest.raises(InputError, match="grid"):
        JobSpec.from_mapping(job).resolve()


def test_grid_offline_fes_uses_the_applied_interpolant_and_domain():
    grid = _bias(bins=10)
    grid.deposit([0.3], step=1, time=0.1)
    samples = np.linspace(-1.9, 1.9, 100)[:, None]
    result = bias_fes([grid], [samples], bins=8)
    centers = (result["edges"][0][1:] + result["edges"][0][:-1]) / 2
    np.testing.assert_array_equal(result["bias_per_replica"][0], grid.physical_energy(centers[:, None]))
    config, system = _configuration()
    config["analysis"]["ranges"] = [[-3, 2]]
    with pytest.raises(ValueError, match="inside the bias grid"):
        resolve_metadynamics(config, system)
