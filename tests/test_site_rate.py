"""Numerical basin normalization, signed flux and restart contracts."""

import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from scipy.special import ndtr

from pyrpmd.analysis.fes import reconstruct_fes, statistical_inefficiency
from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.constants import ANGSTROM_TO_AU, ATOMIC_TIME_S, KB_AU
from pyrpmd.dynamics import Dynamics
from pyrpmd.system import System
from pyrpmd.workflows import site_rate as site


def configuration():
    return {
        "task": "site_rate", "beta": 3.0, "beads": 4, "dt_au": .02,
        "seed": 13, "native": False, "potential": {"provider": "harmonic", "params": {"k": .2}},
        "dynamics": {"loop_backend": "python"},
        "sampling": {"equilibration_steps": 10, "production_steps": 100,
                     "sample_interval": 1, "andersen_interval": 5, "replicas": 1},
        "cvs": [{"name": "x", "kind": "cartesian", "atom": 0, "axis": 0}],
        "umbrella": {"cvs": ["x"], "centers": [-1., -.5, 0., .5, 1.], "stiffness": 2.},
        "analysis": {"bins": 12, "ranges": [[-1.5, 1.5]], "bootstrap": 0},
        "site_rate": {"cv": "x", "basin": [-.6, .6], "dividing_surface": .6, "direction": 1},
        "recrossing": {"parent_replicas": 2, "parent_samples": 5,
                      "equilibration_steps": 2, "parent_interval_steps": 2,
                      "child_pairs": 2, "child_steps": 40, "plateau_fraction": .5,
                      "bootstrap": 8, "constraint_tolerance": 1e-11},
        "output": {"checkpoint_steps": 3},
    }


def hydrogen():
    return System(("H",), [2.0], [[0., .1, -.1]])


def harmonic_histogram(bins=260):
    beta, stiffness = 3., 2.
    sigma = 1 / np.sqrt(beta * stiffness)
    edges = np.linspace(-1.3, 1.3, bins + 1)
    mass = np.diff(ndtr(edges / sigma))
    return {"edges": [edges], "density": mass / np.diff(edges), "probability_mass": mass,
            "mask": np.zeros(bins, dtype=bool), "connected": True, "measure": "cv_density"}


def read_payload(path):
    with load_checkpoint(path) as saved:
        return {key: np.array(saved[key], copy=True) for key in saved.files}


def test_harmonic_basin_normalized_qtst_and_finite_bin_convergence():
    beta, mass, stiffness = 3., 4., 2.
    basin = [-.8, .6]
    sigma = 1 / np.sqrt(beta * stiffness)
    density = np.exp(-.5 * (basin[1] / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
    probability = ndtr(basin[1] / sigma) - ndtr(basin[0] / sigma)
    expected = np.sqrt(1 / (2 * np.pi * beta * mass)) * density / probability
    coarse = site.estimate_site_qtst(harmonic_histogram(26), beta, 1 / mass, basin, basin[1])
    fine = site.estimate_site_qtst(harmonic_histogram(), beta, 1 / mass, basin, basin[1])
    assert fine["ready"]
    assert fine["k_qtst_au"] == pytest.approx(expected, rel=2e-4)
    assert abs(fine["k_qtst_au"] - expected) < abs(coarse["k_qtst_au"] - expected) / 50
    assert fine["k_qtst_s"] == fine["k_qtst_au"] / ATOMIC_TIME_S
    assert "finite_bin" in fine["estimator"]
    assert fine["basin_probability"] == pytest.approx(probability, abs=1e-14)


def test_qtst_is_invariant_under_cv_rescaling_and_density_normalization():
    data = harmonic_histogram()
    baseline = site.estimate_site_qtst(data, 3., .25, [-.8, .6], .6)
    scale = 7.3
    transformed = copy.deepcopy(data)
    transformed["edges"][0] *= scale
    transformed["density"] /= scale
    actual = site.estimate_site_qtst(transformed, 3., .25 * scale ** 2, np.array([-.8, .6]) * scale, .6 * scale)
    assert actual["k_qtst_au"] == pytest.approx(baseline["k_qtst_au"], rel=2e-14)
    transformed["density"] *= .37
    transformed["probability_mass"] *= .37
    assert site.estimate_site_qtst(transformed, 3., .25 * scale ** 2, np.array([-.8, .6]) * scale, .6 * scale)["k_qtst_au"] == pytest.approx(baseline["k_qtst_au"], rel=2e-14)


@pytest.mark.parametrize("failure", ["disconnected", "interior_gap", "surface_gap", "basin_outside", "surface_unbracketed"])
def test_qtst_refuses_unsupported_basin_and_surface(failure):
    data = harmonic_histogram()
    basin, surface = [-.8, .6], .6
    if failure == "disconnected":
        data["connected"] = False
    elif failure == "interior_gap":
        data["mask"][130] = True
    elif failure == "surface_gap":
        data["mask"][189:192] = True
    elif failure == "basin_outside":
        basin = [-1.4, .6]
    else:
        surface = 1.3
    result = site.estimate_site_qtst(data, 3., .25, basin, surface)
    assert not result["ready"]
    assert result["k_qtst_s"] is None
    assert result["reasons"]


def test_missing_support_outside_local_interval_does_not_invalidate_rate():
    data = harmonic_histogram()
    data["mask"][:10] = True
    data["mask"][-10:] = True
    assert site.estimate_site_qtst(data, 3., .25, [-.8, .6], .6)["ready"]
    data["probability_mass"] *= 2
    with pytest.raises(ValueError, match="normalization"):
        site.estimate_site_qtst(data, 3., .25, [-.8, .6], .6)


def test_validation_units_direction_and_fractional_mass_metric():
    config, system = configuration(), hydrogen()
    config["cvs"][0]["unit"] = "angstrom"
    prepared = site.validate_site_rate(config, system)
    assert prepared["basin"] == pytest.approx(np.array([-.6, .6]) * ANGSTROM_TO_AU)
    assert prepared["mass_metric"] == .5
    # Explicit quantity strings carry their own units, independent of CV input units.
    config["site_rate"].update(basin=["-0.6 bohr", "0.6 bohr"], dividing_surface="-0.6 bohr", direction=-1)
    assert site.validate_site_rate(config, system)["dividing_surface"] == -.6
    config["site_rate"]["direction"] = 1
    with pytest.raises(ValueError, match="upper basin"):
        site.validate_site_rate(config, system)
    cell = np.array([[4., 0., 0.], [1., 5., 0.], [.3, .5, 8.]])
    system = System(("H",), [2.], [[0., 0., 0.]], cell=cell, pbc=[True, True, True])
    config = configuration()
    config["cvs"] = [{"kind": "fractional", "name": "x", "atom": 0, "axis": 0}]
    prepared = site.validate_site_rate(config, system)
    assert prepared["unit"] == "dimensionless"
    assert prepared["mass_metric"] == pytest.approx(np.sum(np.linalg.inv(cell)[:, 0] ** 2) / 2)
    json.dumps(prepared, allow_nan=False)


@pytest.mark.parametrize("change", ["nonlinear", "bead_average", "wrong_cv", "gas", "rate_flux", "native", "boolean_direction", "wrapped_bias"])
def test_initial_scope_is_explicitly_rejected(change):
    config, system = configuration(), hydrogen()
    if change == "nonlinear":
        config["cvs"][0] = {"name": "x", "kind": "distance", "atoms": [0, 1]}
        system = System(("H", "H"), [2., 2.], [[0., 0., 0.], [1., 0., 0.]])
    elif change == "bead_average":
        config["cvs"][0]["estimator"] = "bead_average"
    elif change == "wrong_cv":
        config["site_rate"]["cv"] = "another"
    elif change == "gas":
        config["reaction"] = {"electronic_weight": 2.}
    elif change == "rate_flux":
        config["analysis"]["measure"] = "rate_flux"
    elif change == "native":
        config["dynamics"]["loop_backend"] = "native"
    elif change == "wrapped_bias":
        config["cvs"][0]["periods"] = [1.]
    else:
        config["site_rate"]["direction"] = True
    with pytest.raises(ValueError):
        site.validate_site_rate(config, system)


@pytest.mark.parametrize("method", ["centroid_value_gradient", "_affine_gradient"])
def test_site_rate_rejects_overridden_constant_gradient_methods(method):
    config, system = configuration(), hydrogen()
    original = site.resolve_umbrellas

    def modified(*args):
        resolved = original(*args)
        setattr(resolved[0].cvs[0], method, lambda *unused: None)
        return resolved

    with patch.object(site, "resolve_umbrellas", modified):
        with pytest.raises(ValueError, match="overridden CV methods"):
            site.validate_site_rate(config, system)


@pytest.mark.parametrize("direction", [1, -1])
def test_unthermostatted_free_child_pair_has_unit_initial_flux(direction, tmp_path):
    config, system = configuration(), hydrogen()
    config["potential"] = {"provider": "free"}
    config["site_rate"].update(direction=direction, dividing_surface=.6 * direction)
    q = np.repeat(system.geometry[None], config["beads"], axis=0)
    q[:, 0, 0] = .6 * direction
    task = dict(config=config, system=system, q=q, chain=0, parent=0, pair=0, output=str(tmp_path))
    result = site._child_task(task)
    assert result["denominator"] > 0
    np.testing.assert_allclose(result["numerator"] / result["denominator"], 1., atol=1e-13)
    assert result["endpoint_energy_error_kbt"] < 1e-12
    again = site._child_task(task)
    np.testing.assert_array_equal(result["numerator"], again["numerator"])
    assert again["propagation"]["python_steps"] == 2 * config["recrossing"]["child_steps"]


def test_parent_failed_thermostat_step_restores_rng_and_complete_accepted_prefix(tmp_path):
    config, system = configuration(), hydrogen()
    q = np.repeat(system.geometry[None], config["beads"], axis=0)
    task = dict(config=config, system=system, initial_q=q, chain=0, output=str(tmp_path / "reference"))
    expected = site._parent_task(task)
    interrupted = dict(task, output=str(tmp_path / "resumed"))
    original, calls = site.constrained_step, 0

    def fail_after_candidate_step(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        if calls == 6:  # elapsed=5 is also an Andersen refresh boundary.
            raise RuntimeError("injected parent interruption")
        return result

    with patch.object(site, "constrained_step", fail_after_candidate_step):
        with pytest.raises(RuntimeError, match="injected"):
            site._parent_task(interrupted)
    path = tmp_path / "resumed" / "checkpoints" / "site_parent_0000.npz"
    prefix = read_payload(path)
    assert int(prefix["elapsed"]) == 5
    assert len(prefix["snapshots"]) == 1
    actual = site._parent_task(interrupted)
    np.testing.assert_array_equal(actual["snapshots"], expected["snapshots"])
    baseline = read_payload(expected["checkpoint"])
    resumed = read_payload(actual["checkpoint"])
    for key in ("q", "p", "rng", "step", "snapshots", "propagation"):
        np.testing.assert_array_equal(resumed[key], baseline[key])
    assert np.max(abs(actual["snapshots"].mean(axis=1)[:, 0, 0] - .6)) < 1e-11
    assert actual["propagation"]["python_steps"] == 12


def test_child_second_branch_failure_replays_without_duplicate_flux(tmp_path):
    config, system = configuration(), hydrogen()
    q = np.repeat(system.geometry[None], config["beads"], axis=0)
    q[:, 0, 0] = .6
    task = dict(config=config, system=system, q=q, chain=2, parent=3, pair=1, output=str(tmp_path / "reference"))
    expected = site._child_task(task)
    interrupted = dict(task, output=str(tmp_path / "resumed"))
    original, calls = Dynamics.step, 0

    def fail_after_candidate_step(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(self, *args, **kwargs)
        if calls == 44:
            raise RuntimeError("injected child interruption")
        return result

    with patch.object(Dynamics, "step", fail_after_candidate_step):
        with pytest.raises(RuntimeError, match="injected"):
            site._child_task(interrupted)
    path = tmp_path / "resumed" / "checkpoints" / "site_child_0002_00003_0001.npz"
    prefix = read_payload(path)
    np.testing.assert_array_equal(prefix["steps"], [40, 3])
    assert np.count_nonzero(prefix["numerator"][1, 3:]) == 0
    actual = site._child_task(interrupted)
    np.testing.assert_array_equal(actual["numerator"], expected["numerator"])
    assert actual["denominator"] == expected["denominator"]
    baseline, resumed = read_payload(expected["checkpoint"]), read_payload(actual["checkpoint"])
    for key in ("q", "p", "rng", "steps", "numerator", "final_energy", "propagation"):
        np.testing.assert_array_equal(resumed[key], baseline[key])


def test_aggregation_preserves_negative_finite_time_transmission(tmp_path):
    config = configuration()
    data = harmonic_histogram()
    pmf = site.estimate_site_qtst(data, config["beta"], .5, [-.6, .6], .6)
    pmf.update(preparation={"direction": 1}, warnings=[], _bootstrap_qtst_s=[],
               pmf_uncertainty={"available": False, "reasons": ["bootstrap_disabled"]})
    children = [
        {"chain": chain, "parent": 0, "pair": 0, "surface": .6, "direction": 1,
         "numerator": np.full(40, -.4), "denominator": 1.,
         "endpoint_energy_error_kbt": 0.}
        for chain in (0, 1)]
    result = site._aggregate(children, pmf, config, tmp_path)
    assert result["kappa"] == pytest.approx(-.4)
    assert result["rate_estimate_s"] == pytest.approx(-.4 * pmf["k_qtst_s"])
    assert result["rate_s"] is None
    assert result["rate_percentile_95_ci_s"] is None
    assert not result["converged"]
    assert not result["rate_uncertainty"]["available"]
    with pytest.raises(ValueError, match="duplicate"):
        site._aggregate(children + children[:1], pmf, config, tmp_path)


def test_complete_periodic_workflow_and_restart(tmp_path):
    from pyrpmd.runner import run, resume_run

    config = configuration()
    config.update(beta=10., dt_au=.05)
    config["potential"] = {"provider": "builtin", "name": "periodic_cosine",
                           "params": {"height": [.1, 0., 0.], "confinement": .1}}
    config["sampling"].update(equilibration_steps=200, production_steps=1000,
                              sample_interval=2, andersen_interval=20)
    config["cvs"] = [{"kind": "fractional", "name": "x", "atom": 0, "axis": 0}]
    config["umbrella"]["centers"] = [-.6, -.3, 0., .3, .6]
    config["analysis"].update(bins=10, ranges=[[-.8, .8]])
    config["site_rate"].update(basin=[-.5, .5], dividing_surface=.5)
    config["recrossing"].update(parent_samples=3, equilibration_steps=20, parent_interval_steps=5)
    config["output"]["checkpoint_steps"] = 250
    system = System(("H",), [2.], [[0., 0., 0.]], cell=np.diag([4., 8., 8.]), pbc=[True, False, False])
    # Exercise public parsing, output manifests, dispatch and resume as well as
    # the numerical workflow. Numeric input quantities use atomic units.
    config["system"] = system
    config["temperature"] = 1 / (config.pop("beta") * KB_AU)
    config["dynamics"]["timestep"] = config.pop("dt_au")
    result = run(config, output=tmp_path)
    assert result["pmf"]["ready"]
    assert result["k_qtst_s"] > 0
    assert np.isfinite(result["rate_estimate_s"])
    assert result["child_pairs"] == 12
    assert result["parent_configurations"] == 6
    assert result["rate_unit"] == "s^-1"
    assert result["rate_standard_error_s"] is None
    assert not result["converged"]
    again = resume_run(tmp_path)
    assert again["rate_estimate_s"] == result["rate_estimate_s"]
    assert again["kappa"] == result["kappa"]
    assert len(list((tmp_path / "checkpoints").glob("site_child_*.npz"))) == 12
    json.loads((tmp_path / "summary.json").read_text())


def test_unsupported_pmf_stops_before_constrained_trajectories(tmp_path):
    config, system = configuration(), hydrogen()
    data = harmonic_histogram()
    data["mask"][130] = True
    data["statistical_inefficiency"] = np.ones(1)
    sampled = {"data": data, "summary": {"warnings": []},
               "streams": [{"samples": np.zeros((10, 1))}]}
    with patch.object(site, "run_fes", return_value=sampled), patch.object(site, "_parent_task") as parent:
        result = site.run_site_rate(config, system, tmp_path)
    assert result["status"] == "insufficient_support"
    assert result["rate_estimate_s"] is None
    parent.assert_not_called()


def sampled_harmonic_pmf(config, system):
    rng = np.random.default_rng(781)
    beta, physical_k, umbrella_k = config["beta"], .2, 2.
    streams = []
    for window, center in enumerate(config["umbrella"]["centers"]):
        samples = rng.normal(umbrella_k * center / (physical_k + umbrella_k),
                             np.sqrt(1 / (beta * (physical_k + umbrella_k))), (1500, 1))
        streams.append({"window": window, "replica": 0, "samples": samples,
                        "center": np.array([center]), "stiffness": np.array([[umbrella_k]]),
                        "last_q": np.repeat(system.geometry[None], config["beads"], axis=0)})
    data = reconstruct_fes([stream["samples"] for stream in streams],
                           [stream["center"] for stream in streams],
                           [stream["stiffness"] for stream in streams], beta,
                           bins=12, ranges=[[-1.5, 1.5]], bootstrap=0)
    return {"data": data, "streams": streams, "summary": {"warnings": []}}


def test_correlated_pmf_bootstrap_and_chain_cluster_rate_uncertainty(tmp_path):
    config, system = configuration(), hydrogen()
    config["analysis"]["bootstrap"] = 12
    sampled = sampled_harmonic_pmf(config, system)
    data = sampled["data"]
    # This caching/reindexing regression compares against historical draws,
    # so hold their original coordinate-only correlation policy fixed. The
    # production amplitude-sensitive policy is tested in test_fes_correlation.
    data["statistical_inefficiency"] = np.asarray([
        statistical_inefficiency(stream["samples"]) for stream in sampled["streams"]
    ])
    with patch.object(site, "umbrella_reduced_potentials", wraps=site.umbrella_reduced_potentials) as bias_matrix:
        pmf, tasks = site._estimate_pmf(sampled, config, system, tmp_path)
    assert bias_matrix.call_count == 1
    assert pmf["ready"]
    assert pmf["pmf_uncertainty"]["available"]
    assert pmf["pmf_uncertainty"]["valid"] == 12
    assert pmf["pmf_uncertainty"]["attempted"] == 12
    assert not pmf["pmf_uncertainty"]["failures"]
    # Original af90f20d draws: caching/reindexing the bias matrix must retain
    # each circular-block draw and its density/population covariance.
    np.testing.assert_allclose(pmf["_bootstrap_qtst_s"], [
        5318747484353080.0, 5205230174098718.0, 4537989576419254.0,
        5120295485005186.0, 5178448838812025.0, 5190898006599793.0,
        5177997520763552.0, 5449767309564981.0, 5209963077366592.0,
        4815956230820733.0, 5271839157421368.0, 5222931139830863.0,
    ], rtol=3e-13)
    assert pmf["pmf_uncertainty"]["standard_error_s"] > 0
    assert len(tasks) == 2
    children = [
        {"chain": chain, "parent": parent, "pair": pair, "surface": .6, "direction": 1,
         "numerator": np.full(40, .6 + .03 * (chain + parent)), "denominator": 1.,
         "endpoint_energy_error_kbt": 0.}
        for chain in range(2) for parent in range(4) for pair in range(2)]
    result = site._aggregate(children, pmf, config, tmp_path)
    assert result["rate_uncertainty"]["available"]
    assert result["rate_standard_error_s"] > 0
    assert result["rate_percentile_95_ci_s"][1] > result["rate_percentile_95_ci_s"][0]
    assert result["parent_configurations"] == 8  # Child pairs remain clustered.
    # Entire correlated streams provide no within-stream resampling evidence.
    data["statistical_inefficiency"][:] = 1500
    incomplete, _ = site._estimate_pmf(sampled, config, system, tmp_path / "correlated")
    assert not incomplete["pmf_uncertainty"]["available"]
    assert incomplete["pmf_uncertainty"]["standard_error_s"] is None
    assert "fewer_than_two_correlation_blocks_in_a_pmf_stream" in incomplete["pmf_uncertainty"]["reasons"]


def test_site_pmf_records_each_rejected_draw_without_reporting_partial_uncertainty(tmp_path):
    config, system = configuration(), hydrogen()
    config["analysis"]["bootstrap"] = 4
    sampled = sampled_harmonic_pmf(config, system)
    original, calls = site.solve_mbar, 0

    def incomplete_draw(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        if calls == 1:
            result.converged = False
        elif calls == 2:
            result.components = np.arange(len(result.components))
        return result

    with patch.object(site, "solve_mbar", incomplete_draw):
        pmf, _ = site._estimate_pmf(sampled, config, system, tmp_path)
    info = pmf["pmf_uncertainty"]
    assert info["requested"] == info["attempted"] == 4
    assert info["valid"] == 2
    assert len(info["draws"]) == 4
    assert [draw["draw"] for draw in info["failures"]] == [0, 1]
    assert info["failure_counts"] == {"mbar_not_converged": 1, "disconnected_overlap": 1}
    assert not info["available"]
    assert info["standard_error_s"] is None
    assert info["percentile_95_ci_s"] is None
    assert np.all(np.asarray(pmf["_bootstrap_qtst_s"]) > 0)


def test_site_pmf_solver_exception_keeps_its_existing_failure_semantics(tmp_path):
    config, system = configuration(), hydrogen()
    config["analysis"]["bootstrap"] = 2
    sampled = sampled_harmonic_pmf(config, system)
    with patch.object(site, "solve_mbar", side_effect=FloatingPointError("injected solver failure")):
        with pytest.raises(FloatingPointError, match="injected solver failure"):
            site._estimate_pmf(sampled, config, system, tmp_path)


@pytest.mark.parametrize("kind", ["parent", "child"])
def test_future_checkpoint_payload_is_not_silently_reinterpreted(kind, tmp_path):
    config, system = configuration(), hydrogen()
    q = np.repeat(system.geometry[None], config["beads"], axis=0)
    q[:, 0, 0] = .6
    if kind == "parent":
        task = dict(config=config, system=system, initial_q=q, chain=0, output=str(tmp_path))
        operation = site._parent_task
    else:
        task = dict(config=config, system=system, q=q, chain=0, parent=0, pair=0, output=str(tmp_path))
        operation = site._child_task
    record = operation(task)
    data = read_payload(record["checkpoint"])
    data["payload_version"] = 2
    atomic_savez(record["checkpoint"], kind="site_rate_" + kind, **data)
    with pytest.raises(ValueError, match="payload version"):
        operation(task)


def test_public_site_example_prepares_without_gas_defaults():
    from pyrpmd.input import JobSpec
    from pyrpmd.runner import prepare

    example = Path(__file__).resolve().parents[1] / "examples" / "materials" / "site_rate.inp"
    config, system = JobSpec.from_file(example).resolve()
    assert not {"reaction", "collision", "outcomes", "rate"} & set(config)
    assert site.validate_site_rate(config, system)["ready"]
    report = prepare(example)
    assert report["ready"]
    assert report["site_rate_preparation"]["normalization"].startswith("density")


def test_constrained_finite_p_gaussian_covariance_converges_quadratically():
    """Stationary parent covariance vs an independent conditional Gaussian.

    With an Andersen refresh each step, harmonic constrained dynamics is a
    linear Gaussian map. Its exact discrete stationary covariance is obtained
    without sampling noise; the canonical target is a Schur complement of
    the analytic ring-polymer covariance, including unequal real masses.
    """
    from scipy.linalg import null_space, solve_discrete_lyapunov
    from pyrpmd.dynamics import State
    from pyrpmd.potentials.builtin import HarmonicPotential
    from pyrpmd.cv_periodic import LinearProjection
    from pyrpmd.workflows.rate import constrained_step, project_momenta

    beads, beta, spring = 4, 2.0, 0.4
    masses = np.array([2.0, 20.0])
    normal = np.tile([1 / beads, -1 / beads], beads)
    basis = null_space(normal[None, :])
    laplacian = 2 * np.eye(beads) - np.roll(np.eye(beads), 1, axis=0) - np.roll(np.eye(beads), -1, axis=0)
    stiffness = spring * np.eye(2 * beads) + (beads / beta) ** 2 * np.kron(laplacian, np.diag(masses))
    covariance = beads / beta * np.linalg.inv(stiffness)
    constrained = covariance - np.outer(covariance @ normal, covariance @ normal) / (normal @ covariance @ normal)
    reference = basis.T @ constrained @ basis
    surface = site.SiteSurface(LinearProjection(0, [1, 0, 0], masses, reference_atoms=[1]), 0)
    errors = []
    for dt in (.1, .05):
        dyn = Dynamics(HarmonicPotential(k=spring), masses, beta, beads, dt, native=False, loop_backend="python")
        drift = np.zeros((2 * beads - 1, 2 * beads - 1))
        noise = np.zeros((2 * beads - 1, 2 * beads))
        for j in range(2 * beads - 1):
            q = np.zeros((beads, 2, 3))
            q[:, :, 0] = basis[:, j].reshape(beads, 2)
            state = State(q, np.zeros_like(q))
            constrained_step(dyn, state, surface, tolerance=1e-13)
            drift[:, j] = basis.T @ state.q[:, :, 0].ravel()
        for j in range(2 * beads):
            p = np.zeros((beads, 2, 3))
            p[j // 2, j % 2, 0] = np.sqrt(masses[j % 2] * beads / beta)
            q = np.zeros_like(p)
            project_momenta(p, q, surface, masses)
            state = State(q, p)
            constrained_step(dyn, state, surface, tolerance=1e-13)
            noise[:, j] = basis.T @ state.q[:, :, 0].ravel()
        stationary = solve_discrete_lyapunov(drift, noise @ noise.T)
        errors.append(np.linalg.norm(stationary - reference) / np.linalg.norm(reference))
        full = basis @ stationary @ basis.T
        assert abs(normal @ full @ normal) < 1e-12
    assert errors[1] < 3e-5
    assert .22 < errors[1] / errors[0] < .28
