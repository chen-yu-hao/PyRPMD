"""Scientific invariants for periodic MetaD, curved constraints and projection."""
import copy
import json
import numpy as np
import pytest

from pyrpmd.bias import WallBias
from pyrpmd.cv import Cartesian, Distance, CVStack, PeriodicDistance
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.potentials import FreePotential
from pyrpmd.system import System
from pyrpmd.thermostats import NoseHooverChain
from pyrpmd.workflows.surface import DistanceSurfaceCoordinate
from pyrpmd.workflows.rate import project_positions, project_momenta, constrained_step
from pyrpmd.workflows.metadynamics import _sample_replica, run_metadynamics, reanalyze_metadynamics
from pyrpmd.analysis.projection import marginalize_fes
from pyrpmd.analysis.surface import surface_normalization, normalize_surface_rates


@pytest.mark.parametrize("beads", [1, 4])
@pytest.mark.parametrize("estimator", ["centroid", "bead_average"])
def test_wall_energy_and_force_follow_plumed_convention(beads, estimator):
    wall = WallBias(Distance([0, 1], estimator=estimator), 1.7, lower=1.2, upper=2.0)
    q = np.tile([[0., 0., 0.], [2.3, 0.2, 0.1]], (beads, 1, 1))
    q += np.random.default_rng(4).normal(size=q.shape) * 0.04
    e, f = wall.energy_force(q)
    assert e == pytest.approx(beads * 1.7 * (wall.cv.value(q)[0] - 2.)**2)
    numerical = np.zeros_like(q)
    for index in np.ndindex(q.shape):
        plus, minus = q.copy(), q.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        numerical[index] = -(wall.energy_force(plus)[0] - wall.energy_force(minus)[0]) / 2e-6
    np.testing.assert_allclose(f, numerical, atol=3e-9)
    for r, expected in [(0.8, 1.7 * .4**2), (1.5, 0), (2.5, 1.7 * .5**2)]:
        x = np.array([[[0., 0., 0.], [r, 0., 0.]]])
        assert wall.energy_force(x)[0] == pytest.approx(expected)


def slab():
    return System(("H", "Cu"), np.array([2., 30.]), np.array([[1., .2, .7], [0., 0., 0.]]),
                  cell=np.diag([8., 8., 20.]), pbc=[True, True, False], fixed_atoms=[1])


def meta_config(thermostat="nose-hoover-chain", beads=1):
    return {"task": "metadynamics", "beta": 2., "beads": beads, "dt_au": .02, "seed": 78,
            "potential": {"provider": "free"}, "native": False,
            "cvs": [{"kind": "cartesian", "name": a, "atom": 0, "axis": a} for a in "xyz"],
            "metadynamics": {"sigma": [.2]*3, "height": .01, "pace": 3, "bias_factor": 5,
                              "walls_during_equilibration": False,
                              "walls": [{"cv": "x", "lower": .9, "upper": 1.1, "kappa": 2.}]},
            "sampling": {"equilibration_steps": 4, "production_steps": 30, "sample_interval": 2,
                         "thermostat": thermostat, "thermostat_params": {"tdamp": .4, "reset_after_equilibration": True},
                         "replicas": 1, "andersen_interval": 4, "gamma_centroid": .1},
            "output": {"checkpoint_steps": 4}, "analysis": {"bins": [4, 5, 6], "bootstrap": 0}}


@pytest.mark.parametrize("thermostat", ["nose-hoover-chain", "andersen", "pile-l"])
@pytest.mark.parametrize("beads", [1, 4])
@pytest.mark.parametrize("failed_step", [5, 16])
def test_periodic_three_cv_restart_preserves_bath_rng_hills_and_frozen_atoms(tmp_path, monkeypatch, thermostat, beads, failed_step):
    cfg, system = meta_config(thermostat, beads), slab()
    ref = run_metadynamics(cfg, system, tmp_path / "full", return_data=True)
    assert ref["data"]["free_energy"].shape == (4, 5, 6)
    a = ref["streams"][0]
    method = "sample_step" if thermostat == "pile-l" else "step"
    original = getattr(Dynamics, method)
    def fail(self, state, *args, **kwargs):
        value = original(self, state, *args, **kwargs)
        if state.step == failed_step:
            raise RuntimeError("after physical map at deposition boundary")
        return value
    task = {"config": cfg, "system": system, "output": str(tmp_path / "resume"), "replica": 0}
    with monkeypatch.context() as patch:
        patch.setattr(Dynamics, method, fail)
        with pytest.raises(RuntimeError, match="deposition boundary"):
            _sample_replica(task)
    b = _sample_replica(task)
    for key in ("hills", "rng", "thermostat_state", "propagation"):
        assert a[key] == b[key]
    for key in ("samples", "sample_steps", "sample_bias", "sample_energies"):
        np.testing.assert_array_equal(a[key], b[key])
    np.testing.assert_array_equal(a["state"].q, b["state"].q)
    np.testing.assert_array_equal(a["state"].p, b["state"].p)
    np.testing.assert_array_equal(b["state"].q[:, 1], 0.)
    np.testing.assert_array_equal(b["state"].p[:, 1], 0.)
    again = reanalyze_metadynamics(cfg, system, tmp_path / "resume", return_data=True)
    np.testing.assert_array_equal(ref["data"]["free_energy"], again["data"]["free_energy"])


@pytest.mark.parametrize("fixed", [[], [1]])
@pytest.mark.parametrize("beads", [1, 4])
@pytest.mark.parametrize("periodic", [False, True])
def test_curved_distance_rattle_is_reversible_and_has_constant_mobile_metric(fixed, beads, periodic):
    boundary = {"cell": [[8., 0., 0.], [2., 8., 0.], [0., 0., 20.]], "pbc": [True, True, False]} if periodic else {}
    system = System(("C", "O"), np.array([12., 16.]), np.array([[1., 1., .2], [0., 0., 0.]]), fixed_atoms=fixed, **boundary)
    cv = PeriodicDistance([0, 1], system.cell, system.pbc) if periodic else Distance([0, 1])
    coordinate = DistanceSurfaceCoordinate(cv, system, "lower")
    assert coordinate.mass_metric == pytest.approx(1 / 12 + (0 if fixed else 1 / 16))
    surface = coordinate.dividing_surface(1.4)
    dyn = Dynamics(FreePotential(), system.masses, 2., beads, .01, native=False,
                   fixed_atoms=fixed, fixed_positions=system.geometry, seed=12)
    q = project_positions(np.repeat(system.geometry[None], beads, axis=0), surface, system.masses)
    if periodic:
        unshifted = surface.value_and_jacobian(q)
        q[:, 0] += system.cell[0] - system.cell[1]
        shifted = surface.value_and_jacobian(q)
        np.testing.assert_allclose(shifted.value, unshifted.value, atol=2e-14)
        np.testing.assert_allclose(shifted.jacobian, unshifted.jacobian, atol=2e-14)
    state = State(q, dyn.sample_momenta())
    project_momenta(state.p, q, surface, system.masses)
    initial = state.copy()
    for _ in range(30):
        constrained_step(dyn, state, surface, tolerance=1e-13)
        value, grad = surface.centroid_value_gradient(state.q.mean(axis=0))
        assert abs(value) < 1e-12
        assert np.sum(grad**2 / system.masses[:, None]) == pytest.approx(coordinate.mass_metric)
        assert abs(np.sum(grad * state.p.mean(axis=0) / system.masses[:, None])) < 1e-12
    state.p *= -1
    for _ in range(30):
        constrained_step(dyn, state, surface, tolerance=1e-13)
    np.testing.assert_allclose(state.q, initial.q, atol=3e-12)
    np.testing.assert_allclose(state.p, -initial.p, atol=3e-11)
    if fixed:
        np.testing.assert_array_equal(state.q[:, 1], 0)
        np.testing.assert_array_equal(state.p[:, 1], 0)


def test_projection_integrates_density_with_bin_widths_and_conditions_channels():
    edges = [np.array([0., 1., 2.]), np.array([0., .5, 2., 5.]), np.array([0., 2., 3.])]
    # Correlated density, nonuniform bins; summing F or minimizing F gives a different result.
    density = np.arange(1., 13.).reshape(2, 3, 2)
    free = -np.log(density) / 3.
    selected = np.ones(free.shape, bool)
    selected[1, 1:, :] = False
    result = marginalize_fes(free + 1e4, edges, 3., axes=0, selection=selected)
    exact = np.sum(np.where(selected, density, 0) * np.diff(edges[1])[None, :, None] * np.diff(edges[2])[None, None, :], axis=(1, 2))
    expected = -np.log(exact) / 3.
    expected -= expected.min()
    np.testing.assert_allclose(result["free_energy"], expected, atol=3e-12)
    mask = np.zeros_like(selected)
    mask[0, 0, 0] = True
    partial = marginalize_fes(free, edges, 3., axes=0, mask=mask, selection=selected)
    assert partial["mask"].tolist() == [True, False]
    assert np.isnan(partial["free_energy"][0])
    assert 0 < partial["coverage"][0] < 1
    reordered = marginalize_fes(free, edges, 3., axes=[2, 0])
    assert reordered["free_energy"].shape == (2, 2)
    np.testing.assert_allclose(reordered["free_energy"], marginalize_fes(free, edges, 3., axes=[0, 2])["free_energy"].T)


def test_second_order_area_and_coverage_conversion_preserves_all_uncertainties():
    options = surface_normalization({"reaction_order": 2, "reference_adsorbates": 1,
                                     "coverage": .04, "site_count": 9}, slab())
    result = normalize_surface_rates({"rate_s_inv": None, "rate_estimate_s_inv": 100.,
                                      "rate_standard_error_s_inv": 2., "rate_percentile_95_ci_s_inv": [96., 104.]}, options)
    assert result["at_coverage"]["rate_estimate"] == pytest.approx(36.)
    assert result["at_coverage"]["rate_standard_error"] == pytest.approx(.72)
    assert result["at_coverage"]["rate"] is None
    assert result["second_order"]["rate_estimate"] == pytest.approx(100 * options["area_cm2"])
    np.testing.assert_allclose(result["at_coverage"]["rate_percentile_95_ci"], [34.56, 37.44])
    with pytest.raises(ValueError, match="reference_adsorbates"):
        surface_normalization({"reaction_order": 2}, slab())
