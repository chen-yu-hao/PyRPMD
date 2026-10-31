"""FES PILE-L wiring, Gaussian invariant, and transactional restart."""
import json
from pathlib import Path
import tomllib

import numpy as np
import pytest
from scipy.linalg import solve_discrete_lyapunov

from pyrpmd.bias import UmbrellaBias
from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.cv import Cartesian, CVStack
from pyrpmd.dynamics import Dynamics, State, native_capabilities
from pyrpmd.input import InputError, JobSpec
from pyrpmd.potentials import HarmonicPotential
from pyrpmd.system import System
from pyrpmd.workflows.fes import run_fes, reanalyze_fes
from pyrpmd.workflows.sampling import fes_sampler


def config(scheme="cayley-bcb", native=False):
    return {"task": "fes", "beta": 1.7, "beads": 4, "dt_au": .03, "seed": 903,
            "native": native, "dynamics": {"integrator": scheme},
            "potential": {"provider": "builtin", "name": "harmonic", "params": {"k": .4}},
            "sampling": {"thermostat": "pile-l", "gamma_centroid": .5,
                         "equilibration_steps": 8, "production_steps": 40,
                         "sample_interval": 2, "andersen_interval": 7},
            "cvs": [{"name": a, "kind": "cartesian", "atom": 0, "axis": a} for a in ("x", "y")],
            "umbrella": {"centers": [[-.2, -.2], [.2, .2]], "stiffness": [[1., .2], [.2, .8]]},
            "analysis": {"bins": [4, 5], "bootstrap": 0}, "output": {"checkpoint_steps": 11}}


def system():
    return System(("H",), np.array([2.3]), np.zeros((1, 3)))


def payload(path):
    with load_checkpoint(path, require_integrity=True) as data:
        return {k: data[k] for k in data.files}


@pytest.mark.parametrize("scheme", ["cayley-bcb", "normal-mode-bab"])
@pytest.mark.parametrize("native", [False, True])
def test_actual_fes_calls_matching_pile_sampler_without_andersen(tmp_path, monkeypatch, scheme, native):
    cfg = config(scheme, native)
    original_step, original_momenta = Dynamics.sample_step, Dynamics.sample_momenta
    steps, momentum_draws = [], []

    def step(dynamics, state, bias=None, **kwargs):
        steps.append((kwargs["method"], kwargs["gamma_centroid"]))
        return original_step(dynamics, state, bias, **kwargs)

    def momenta(dynamics):
        momentum_draws.append(1)
        return original_momenta(dynamics)

    monkeypatch.setattr(Dynamics, "sample_step", step)
    monkeypatch.setattr(Dynamics, "sample_momenta", momenta)
    result = run_fes(cfg, system(), tmp_path, return_data=True)
    assert len(steps) == 2 * 48
    assert set(steps) == {("bcocb" if scheme == "cayley-bcb" else "baoab", .5)}
    assert len(momentum_draws) == 2
    assert result["summary"]["samples"] == 40
    assert result["data"]["free_energy"].shape == (4, 5)
    again = reanalyze_fes(cfg, system(), tmp_path, return_data=True)
    np.testing.assert_array_equal(again["data"]["free_energy"], result["data"]["free_energy"])


@pytest.mark.parametrize("scheme", ["cayley-bcb", "normal-mode-bab"])
def test_pile_observation_failure_restores_q_p_rng_and_samples(tmp_path, monkeypatch, scheme):
    cfg = config(scheme)
    run_fes(cfg, system(), tmp_path / "full")
    energy = Dynamics.energy
    tripped = False

    def failing(dyn, state, bias=None):
        nonlocal tripped
        if state.step == 14 and not tripped:
            tripped = True
            raise RuntimeError("injected sample failure")
        return energy(dyn, state, bias)

    with monkeypatch.context() as patch:
        patch.setattr(Dynamics, "energy", failing)
        with pytest.raises(RuntimeError, match="injected sample failure"):
            run_fes(cfg, system(), tmp_path / "resumed")
    assert int(payload(tmp_path / "resumed/checkpoints/window_00000_replica_0000.npz")["elapsed"]) == 13
    run_fes(cfg, system(), tmp_path / "resumed")
    for full in (tmp_path / "full/checkpoints").glob("*.npz"):
        a, b = payload(full), payload(tmp_path / "resumed/checkpoints" / full.name)
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)


def test_pile_is_available_in_both_adaptive_stages(tmp_path):
    cfg = config()
    cfg["adaptive_umbrella"] = {"enabled": True, "pilot_steps": 60, "pilot_equilibration_steps": 10,
                                "max_rounds": 2, "max_windows": 8, "target_overlap": 1e-5,
                                "min_effective_samples": 1.0}
    result = run_fes(cfg, system(), tmp_path, return_data=True)
    plan = json.loads((tmp_path / "adaptive-plan.json").read_text())
    assert plan["status"] == "frozen"
    assert result["summary"]["adaptive_umbrella"]["pilot_samples_used_in_production"] == 0
    assert len(list((tmp_path / "adaptation").rglob("window_*.npz"))) >= 2


@pytest.mark.parametrize("change", [
    {"dynamics": {"integrator": "velocity-verlet"}}, {"dynamics": {"loop_backend": "native"}},
    {"sampling": {"integrator": "baoab"}}, {"sampling": {"gamma_centroid": 0.0}},
    {"sampling": {"gamma_centroid": float("nan")}}, {"sampling": {"thermostat": "unknown"}},
    {"task": "rate"},
])
def test_invalid_sampler_rejected_before_output(tmp_path, change):
    cfg = config()
    for key, value in change.items():
        if isinstance(value, dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    with pytest.raises(ValueError):
        run_fes(cfg, system(), tmp_path / "absent")
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("scheme,expected", [("cayley-bcb", "bcocb"), ("normal-mode-bab", "baoab")])
def test_cli_pile_sampler_resolution(scheme, expected):
    root = Path(__file__).resolve().parents[1]
    data = tomllib.loads((root / "examples/fes/harmonic2d.inp").read_text())
    data["dynamics"]["integrator"] = scheme
    data["sampling"].update(thermostat="pile-l", gamma_centroid="0.01 fs^-1")
    resolved, _ = JobSpec.from_mapping(data).resolve()
    assert resolved["sampling"]["integrator"] == expected
    assert fes_sampler(resolved).gamma_centroid == resolved["sampling"]["gamma_centroid"]
    data["dynamics"]["integrator"] = "velocity-verlet"
    with pytest.raises(InputError, match="PILE-L requires"):
        JobSpec.from_mapping(data).resolve()


class PrescribedNoise:
    def __init__(self, beads):
        self.bit_generator = np.random.default_rng(1).bit_generator
        self.noise = np.zeros((beads, 1, 3))

    def normal(self, *, size):
        assert size == self.noise.shape
        return self.noise.copy()


@pytest.mark.parametrize("beads", [1, 4, 16])
@pytest.mark.parametrize("scheme", ["cayley-bcb", "normal-mode-bab"])
@pytest.mark.parametrize("native", [False, True])
def test_independent_2d_lyapunov_invariant_has_correct_temperature(beads, scheme, native):
    if native and not native_capabilities()["available"]:
        pytest.skip("native extension unavailable")
    beta, mass, physical_k = 1.7, 2.3, .7
    stiffness = np.array([[1.1, .3], [.3, .8]])
    dyn = Dynamics(HarmonicPotential(physical_k), [mass], beta, beads, .04,
                   native=native, integrator=scheme)
    noise = PrescribedNoise(beads)
    dyn.rng = noise
    cv = CVStack([Cartesian(0, "x"), Cartesian(0, "y")])
    bias = UmbrellaBias(cv, [0., 0.], stiffness)
    sampler = fes_sampler(config(scheme, native))
    dim = 2 * beads

    def step(vector, normal):
        state = State(np.zeros((beads, 1, 3)), np.zeros((beads, 1, 3)))
        state.q[:, 0, :2] = vector[:dim].reshape(beads, 2)
        state.p[:, 0, :2] = vector[dim:].reshape(beads, 2)
        noise.noise[:, 0, :2] = normal.reshape(beads, 2)
        sampler.step(dyn, state, bias)
        return np.r_[state.q[:, 0, :2].ravel(), state.p[:, 0, :2].ravel()]

    a = np.column_stack([step(e, np.zeros(dim)) for e in np.eye(2 * dim)])
    b = np.column_stack([step(np.zeros(2 * dim), e) for e in np.eye(dim)])
    stationary = solve_discrete_lyapunov(a, b @ b.T)
    centroid = np.zeros((2, 2 * dim))
    centroid[0, :dim:2] = 1 / beads
    centroid[1, 1:dim:2] = 1 / beads
    reference = np.linalg.inv(beta * (physical_k * np.eye(2) + stiffness))
    np.testing.assert_allclose(centroid @ stationary @ centroid.T, reference, rtol=2e-10, atol=2e-12)
