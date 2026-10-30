"""Integration contracts for basin-normalized surface rate workflows."""

import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from pyrpmd.constants import KB_AU
from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.cv import Cartesian, SurfaceHeight
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.input import InputError, JobSpec
from pyrpmd.runner import prepare, run, resume_run
from pyrpmd.system import System
from pyrpmd.workflows.drpmd import run_drpmd
from pyrpmd.workflows.fes import _atomic_npz
from pyrpmd.workflows.rate import (
    _child_task, _parent_task, _recrossing_options, project_momenta,
    project_positions, constrained_step, run_rate,
)
from pyrpmd.workflows.surface import surface_definition


def cluster_job():
    """A free coordinate makes paired reactive flux analytically unity."""
    return {
        "task": "surface_rate", "temperature": 1 / (2 * KB_AU),
        "beads": 4, "seed": 318, "native": False,
        "system": {
            "symbols": ["Cu", "H"], "masses": [20.0, 2.0],
            "geometry": [[0.0, 0.0, 0.2], [0.1, 0.0, 0.2]],
            "fixed_atoms": [0],
        },
        "potential": {"provider": "free"},
        "dynamics": {"timestep": 0.05},
        "sampling": {"equilibration_steps": 50, "production_steps": 500,
                     "interval_steps": 5, "andersen_interval": 10, "replicas": 1},
        "cvs": [{"name": "height", "kind": "surface_height", "atoms": [1],
                 "reference_atoms": [0]}],
        "umbrella": {"centers": [-1.0, -0.5, 0.0, 0.5], "stiffness": 2.0},
        "surface": {"reactant_basin": [-0.8, -0.2], "dividing_surface": 0.0},
        "recrossing": {"parent_replicas": 2, "parent_samples": 3,
                       "equilibration_steps": 7, "parent_interval_steps": 5,
                       "child_pairs": 2, "child_steps": 24, "plateau_fraction": 0.5,
                       "bootstrap": 4},
        "analysis": {"bins": 8, "ranges": [[-1.2, 0.7]], "min_count": 1,
                     "bootstrap": 4},
        "output": {"checkpoint_steps": 5},
    }


def resolve_job(job=None):
    return JobSpec.from_mapping(cluster_job() if job is None else job).resolve()


@pytest.mark.parametrize("change", [
    {"surface": {"reactant_basin": [-0.8, -0.2]}},
    {"surface": {"reactant_basin": [-0.8, 0.2], "dividing_surface": 0.0}},
    {"surface": {"reactant_basin": [-0.2, -0.8], "dividing_surface": 0.0}},
    {"surface": {"reactant_basin": [-0.8, -0.2], "dividing_surface": True}},
    {"surface": {"reactant_basin": [-0.8, -0.2], "dividing_surface": 0.0, "product_side": "both"}},
    {"cvs": [{"kind": "distance", "atoms": [0, 1]}]},
    {"cvs": [{"kind": "surface_height", "atoms": [1], "estimator": "bead_average"}]},
    {"cvs": [{"kind": "cartesian", "atom": 0}]},
    {"umbrella": {"centers": [-0.5, 0.0], "stiffness": 2.0}},
    {"analysis": {"measure": "rate_flux"}},
    {"analysis": {"method": "umbrella_integration"}},
    {"recrossing": {"dividing_surface": 0.0}},
    {"reaction": {"electronic_weight": 2.0}},
    {"rate": {"recrossing": {"child_steps": 2}}},
])
def test_surface_input_rejects_ambiguous_or_unsupported_science(change):
    job = cluster_job()
    job.update(change)
    with pytest.raises(InputError):
        resolve_job(job)


def test_surface_definition_metric_uses_only_mobile_atoms_and_reverses_products():
    system = System(
        ("Cu", "H", "H"), np.array([20.0, 2.0, 3.0]),
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0], [0.0, 0.0, 4.0]]),
        fixed_atoms=[0],
    )
    config = cluster_job()
    config["cvs"] = [{"kind": "surface_height", "atoms": [1, 2], "reference_atoms": [0]}]
    config["umbrella"]["centers"] = [0.0, 1.0, 2.0, 3.0]
    config["surface"] = {"reactant_basin": [0.0, 1.0], "dividing_surface": 2.0}
    high = surface_definition(config, system)
    coordinate = high["coordinate"]
    assert high["mass_metric_au"] == pytest.approx(1 / 5)
    np.testing.assert_array_equal(coordinate.gradient[0], 0)
    raw = SurfaceHeight([1, 2], system.masses, reference_atoms=[0])
    q = np.repeat(system.geometry[None], 4, axis=0)
    q[:, 1:] += np.random.default_rng(38).normal(size=(4, 2, 3)) * 0.1
    np.testing.assert_allclose(coordinate.value(q), raw.value(q), atol=1e-14)
    high_surface = coordinate.dividing_surface(2.0)
    config["surface"].update(reactant_basin=[2.0, 3.0], product_side="lower")
    low_surface = surface_definition(config, system)["coordinate"].dividing_surface(2.0)
    np.testing.assert_allclose(high_surface.value(q), -low_surface.value(q))
    np.testing.assert_allclose(high_surface.value_and_jacobian(q).jacobian,
                               -low_surface.value_and_jacobian(q).jacobian)


def test_nonlinear_method_override_cannot_impersonate_affine_rate_coordinate():
    config, system = resolve_job()
    cv = Cartesian(1, "z")

    def nonlinear(q):
        gradient = np.zeros_like(q)
        gradient[1, 2] = 2 * q[1, 2]
        return q[1, 2] ** 2, gradient

    cv.centroid_value_gradient = nonlinear
    config["cvs"] = [cv]
    with pytest.raises(ValueError, match="overrid|linear|affine"):
        surface_definition(config, system)


@pytest.mark.parametrize("integrator", ["cayley-bcb", "normal-mode-bab", "velocity-verlet"])
@pytest.mark.parametrize("product_side", ["higher", "lower"])
def test_linear_parent_constraints_keep_fixed_beads_and_do_not_constrain_child_flux(tmp_path, product_side, integrator):
    job = cluster_job()
    job["dynamics"]["integrator"] = integrator
    if product_side == "lower":
        job["surface"].update(reactant_basin=[0.2, 0.8], product_side="lower")
        job["umbrella"]["centers"] = [-0.5, 0.0, 0.5, 1.0]
    config, system = resolve_job(job)
    surface = surface_definition(config, system)["coordinate"].dividing_surface(0.0)
    from pyrpmd.potentials import FreePotential

    dyn = Dynamics(FreePotential(), system.masses, config["beta"], config["beads"],
                   config["dt_au"], seed=733, native=False,
                   fixed_atoms=system.fixed_atoms, fixed_positions=system.geometry,
                   integrator=integrator)
    q = np.repeat(system.geometry[None], config["beads"], axis=0)
    q[:, 1] += np.random.default_rng(63).normal(scale=0.02, size=(config["beads"], 3))
    q = project_positions(q, surface, system.masses)
    state = State(q, dyn.sample_momenta())
    project_momenta(state.p, state.q, surface, system.masses)
    for _ in range(12):
        constrained_step(dyn, state, surface)
        assert abs(surface.value(state.q)[0]) < 1e-10
        dyn._check_fixed(state.q, state.p)
        assert abs(state.p[:, 1, 2].mean()) < 1e-12
    child = _child_task({
        "config": config, "system": system, "chain": 0, "parent": 0, "pair": 0,
        "xi_star": 0.0, "q": state.q, "child_steps": 24, "output": str(tmp_path),
    })
    assert child["denominator"] > 0
    np.testing.assert_allclose(child["numerator"], child["denominator"], atol=1e-12)
    assert child["pmf_method"] == "mbar_surface_density"
    assert child["propagation"]["python_steps"] == 48
    assert child["propagation"]["fallback_reasons"]["fixed atoms require the Python propagation loop"] == 48


def test_completed_parent_checkpoint_still_validates_fixed_state(tmp_path):
    config, system = resolve_job()
    options = _recrossing_options(config)
    task = {"config": config, "system": system, "options": options, "chain": 0,
            "xi_star": 0.0, "initial_q": np.repeat(system.geometry[None], 4, axis=0),
            "output": str(tmp_path)}
    parent = _parent_task(task)
    checkpoint = Path(parent["checkpoint"])
    with load_checkpoint(checkpoint) as saved:
        original = {key: saved[key].copy() for key in saved.files}
    for key in ("q", "p", "snapshots"):
        changed = copy.deepcopy(original)
        if key == "snapshots":
            changed[key][0, :, 0, 0] += 0.1
        else:
            changed[key][:, 0, 0] += 0.1
        _atomic_npz(checkpoint, **changed)
        with pytest.raises(ValueError, match="fixed"):
            _parent_task(task)
    _atomic_npz(checkpoint, **original)


def test_interrupted_constrained_parent_continues_the_identical_chain(tmp_path):
    config, system = resolve_job()
    options = _recrossing_options(config)
    task = {"config": config, "system": system, "options": options, "chain": 0,
            "xi_star": 0.0, "initial_q": np.repeat(system.geometry[None], 4, axis=0),
            "output": str(tmp_path / "full")}
    full = _parent_task(task)
    task["output"] = str(tmp_path / "restart")
    original_block = Dynamics.propagate_block

    def interrupt(dynamics, state, steps, **kwargs):
        if state.step >= 10:
            raise RuntimeError("deliberate constrained-parent interruption")
        return original_block(dynamics, state, steps, **kwargs)

    with patch.object(Dynamics, "propagate_block", interrupt):
        with pytest.raises(RuntimeError, match="constrained-parent interruption"):
            _parent_task(task)
    resumed = _parent_task(task)
    np.testing.assert_array_equal(resumed["snapshots"], full["snapshots"])
    assert resumed["propagation"] == full["propagation"]
    assert resumed["propagation_this_call"]["python_steps"] == 12
    with np.load(full["checkpoint"]) as a, np.load(resumed["checkpoint"]) as b:
        for key in ("q", "p", "step", "time", "rng", "snapshots", "elapsed"):
            np.testing.assert_array_equal(a[key], b[key])


@pytest.mark.parametrize("kind", ["rate", "drpmd"])
@pytest.mark.parametrize("boundary", ["fixed", "periodic"])
def test_gas_workflows_reject_surface_normalization_in_input_and_direct_api(tmp_path, kind, boundary):
    job = cluster_job()
    job["task"] = kind
    job.pop("surface")
    job["system"]["fragments"] = [[0], [1]]
    if boundary == "periodic":
        job["system"].pop("fixed_atoms")
        job["system"].update(cell=np.diag([4.0, 4.0, 10.0]).tolist(), pbc=[True, True, False])
    with pytest.raises(InputError, match="Gas-phase|normalization"):
        resolve_job(job)
    from pyrpmd.system import system_from_config

    system = system_from_config(job["system"])
    with pytest.raises(ValueError, match="Gas-phase"):
        (run_rate if kind == "rate" else run_drpmd)({}, system, tmp_path)


@pytest.mark.parametrize("boundary", ["fixed", "periodic"])
def test_network_input_cannot_drop_surface_boundary_metadata(boundary):
    source = Path(__file__).resolve().parents[1] / "examples/network/two_entrances.py"
    spec = JobSpec.from_file(source)
    if boundary == "fixed":
        spec.data["system"]["fixed_atoms"] = [0]
    else:
        spec.data["system"].update(cell=np.diag([10.0, 10.0, 20.0]).tolist(), pbc=[True, True, False])
    with pytest.raises(InputError, match="[Nn]etwork|[Gg]as-phase"):
        spec.resolve()


@pytest.mark.parametrize("boundary", ["fixed", "periodic"])
def test_metadynamics_input_rejects_unvalidated_surface_combination(boundary):
    job = cluster_job()
    job["task"] = "metadynamics"
    for key in ("surface", "umbrella", "recrossing"):
        job.pop(key)
    job["metadynamics"] = {"mode": "standard", "sigma": [0.2], "height": 0.01, "pace": 5}
    job["analysis"] = {"method": "bias_estimate", "bootstrap": 0}
    if boundary == "periodic":
        job["system"].pop("fixed_atoms")
        job["system"].update(cell=np.diag([4.0, 4.0, 10.0]).tolist(), pbc=[True, True, False])
    with pytest.raises(InputError, match="[Mm]etadynamics.*periodic|[Mm]etadynamics.*fixed"):
        resolve_job(job)


def test_adaptive_surface_fes_is_accepted_but_surface_rate_is_not():
    job = periodic_job()
    job["adaptive_umbrella"] = {
        "enabled": True, "pilot_steps": 100, "pilot_equilibration_steps": 10,
        "max_rounds": 2, "max_windows": 20, "target_overlap": 0.001,
        "min_effective_samples": 1.0,
    }
    with pytest.raises(InputError, match="[Aa]daptive"):
        resolve_job(job)
    job["task"] = "fes"
    job.pop("surface")
    job.pop("recrossing")
    config, system = resolve_job(job)
    assert config["adaptive_umbrella"]["enabled"] is True
    assert system.pbc == (True, True, False)
    assert system.fixed_atoms == (1,)


def periodic_job():
    """Small analytic periodic slab that exercises the complete public runner."""
    job = cluster_job()
    job.update(beads=2, seed=211)
    job["system"] = {
        "symbols": ["H", "Cu"], "masses": [2.0, 20.0],
        "geometry": [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        "cell": [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 12.0]],
        "pbc": [True, True, False], "fixed_atoms": [1],
    }
    factory = Path(__file__).resolve().parents[1] / "examples" / "surface" / "toy_surface.py"
    job["potential"] = {"provider": "python", "factory": f"{factory}:build",
                        "params": {"barrier": 0.2, "transverse_barrier": 0.2,
                                   "height": 1.0, "vertical_k": 1.0}}
    job["sampling"].update(equilibration_steps=100, production_steps=1200)
    job["cvs"] = [{"name": "migration", "kind": "surface_lateral", "atoms": [0],
                   "reference_atoms": [1], "direction": [1.0, 0.0, 0.0]}]
    job["umbrella"] = {"centers": np.linspace(-2.5, 2.5, 11).tolist(), "stiffness": 2.0}
    job["surface"] = {"reactant_basin": [-2.0, 2.0], "dividing_surface": 2.0,
                      "product_side": "higher"}
    job["analysis"] = {"bins": 12, "ranges": [[-3.0, 3.0]], "min_count": 1,
                       "bootstrap": 3}
    job["output"]["checkpoint_steps"] = 100
    return job


def test_periodic_surface_rate_public_run_and_completed_resume(tmp_path):
    job = periodic_job()
    prepared = prepare(job, numerical=True)
    assert prepared["surface_preparation"]["rate_unit"] == "s^-1"
    result = run(job, output=tmp_path / "rate")
    assert result["task"] == "surface_rate"
    assert result["pmf"]["ready"]
    assert result["pmf"]["measure"] == "cv_density"
    assert result["rate_estimate_s_inv"] > 0
    assert result["parent_configurations"] == 6
    assert result["child_pairs"] == 12
    assert result["rate_unit"] == "s^-1"
    assert not any("cm3" in key for key in result)
    assert result["converged"] is False
    with patch.object(Dynamics, "step", side_effect=AssertionError("completed resume propagated")):
        resumed = resume_run(tmp_path / "rate")
    for key in ("rate_estimate_s_inv", "rate_standard_error_s_inv", "kappa", "status"):
        assert resumed[key] == result[key]
    manifest = json.loads((tmp_path / "rate" / "input" / "system.json").read_text())
    assert manifest["pbc"] == [True, True, False]
    assert manifest["fixed_atoms"] == [1]
    with np.load(tmp_path / "rate" / "checkpoints" / "parent_0000.npz") as parent:
        np.testing.assert_array_equal(parent["q"][:, 1], 0.0)
        np.testing.assert_array_equal(parent["p"][:, 1], 0.0)
