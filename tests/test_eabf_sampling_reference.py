"""Independent equilibrium reference for the actual eABF propagation map.

The harmonic joint covariance follows from a Gaussian integral. Extracting
the linear step/noise maps and solving a discrete Lyapunov equation checks the
sampled measure without a noisy stochastic campaign or injected force means.
"""

import json
import sys

import numpy as np
import pytest
from scipy.linalg import solve_discrete_lyapunov

import pyrpmd
from pyrpmd.abf import ExtendedABF
from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.cv import Cartesian
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.input import JobSpec
from pyrpmd.potentials.builtin import HarmonicPotential
from pyrpmd.workflows.abf import run_abf
from pyrpmd.workflows.sampling import FESSampler


class _SpecifiedNormal:
    """Supply independent unit impulses when identifying the linear noise map."""

    def __init__(self, values):
        self.values = iter(values)

    def normal(self, size):
        return np.full(size, next(self.values))


def _stationary_covariance(beads, dt):
    # Coordinates are (centroid x, mean bead momentum, lambda, p_lambda).
    # Internal ring modes stay zero; the auxiliary OU bath thermalizes the
    # coupled physical centroid, so its covariance is independent of P.
    dynamics = Dynamics(HarmonicPotential(.7), [2.], 2., beads, dt, native=False)

    def step(vector, normals):
        extended = ExtendedABF(
            Cartesian(0), [[-100., 0., 100.]], 2., [vector[2]],
            momentum=[vector[3]], coupling=3., mass=5., friction=.4,
        )
        q = np.zeros((beads, 1, 3))
        p = np.zeros_like(q)
        q[:, 0, 0], p[:, 0, 0] = vector[0], vector[1]
        state = State(q, p)
        dynamics.rng = _SpecifiedNormal(normals)
        extended.advance(dynamics, state, lambda md, s, bias: md.step(s, bias))
        return np.array([state.q[:, 0, 0].mean(), state.p[:, 0, 0].mean(),
                         extended.position[0], extended.momentum[0]])

    transition = np.stack([step(vector, [0., 0.]) for vector in np.eye(4)], axis=1)
    noise = np.stack([step(np.zeros(4), normals) for normals in np.eye(2)], axis=1)
    assert np.max(np.abs(np.linalg.eigvals(transition))) < 1
    return solve_discrete_lyapunov(transition, noise @ noise.T)


@pytest.mark.parametrize("beads", [1, 4])
def test_eabf_actual_step_has_canonical_covariance_and_second_order_error(beads):
    # beta=2, V(x)=0.7*x^2/2, U_c=3*(x-lambda)^2/2.
    # Integrating this coupled quadratic Hamiltonian gives covariance K^-1/beta;
    # physical and auxiliary momenta have variances m/beta and mu/beta.
    exact = np.zeros((4, 4))
    exact[np.ix_([0, 2], [0, 2])] = np.linalg.inv([[3.7, -3.], [-3., 3.]]) / 2
    exact[1, 1], exact[3, 3] = 1., 2.5
    errors = []
    for dt, tolerance in [(.02, 2e-4), (.01, 5e-5)]:
        actual = _stationary_covariance(beads, dt)
        np.testing.assert_allclose(actual, exact, rtol=0., atol=tolerance)
        assert np.linalg.eigvalsh(actual).min() > 0
        errors.append(np.max(np.abs(actual - exact)))
    assert errors[0] / errors[1] == pytest.approx(4., rel=5e-4)


@pytest.fixture(params=["python", "native"])
def nhc_backend(request, monkeypatch):
    if request.param == "python":
        monkeypatch.setitem(sys.modules, "pyrpmd._thermal", None)
        monkeypatch.delattr(pyrpmd, "_thermal", raising=False)
        return "pyrpmd.python-nhc-bath.v1"
    pytest.importorskip("pyrpmd._thermal")
    return "pyrpmd.native-ensemble.v1"


def _checkpoint_job(method="abf", thermostat="nhc"):
    table = {"ranges_au": [[-1., 1.]], "bins": 9, "pace": 3, "full_samples": 3}
    if method == "eabf":
        table.update(coupling_au=2., extended_mass_au=3., friction_au=.1)
    raw = {
        "task": "fes", "temperature": 300, "beads": 4, "seed": 19, "native": False,
        "system": {"symbols": ["H"], "masses": [1.], "geometry": [[.1, 0., 0.]]},
        "potential": {"provider": "harmonic", "params": {"k": .1}},
        "dynamics": {"timestep": .02}, "output": {"checkpoint_steps": 5},
        "fes": {
            "method": method,
            "cvs": [{"name": "x", "kind": "cartesian", "atom": 0, "axis": "x"}],
            method: table,
            "sampling": {"thermostat": thermostat,
                         "equilibration_steps": 2, "production_steps": 10, "interval_steps": 2},
            "analysis": {"min_count": 1},
        },
    }
    if thermostat == "nhc":
        raw["fes"]["sampling"]["thermostat_params"] = {"tau": .1}
    return raw


def _make_checkpoint(config, system, output, monkeypatch, *, complete, interrupt_at=7):
    if complete:
        run_abf(config, system, output)
    else:
        original = FESSampler.step

        def interrupt(self, dynamics, state, bias):
            if state.step == interrupt_at:
                raise RuntimeError("test interruption")
            return original(self, dynamics, state, bias)

        with monkeypatch.context() as patch:
            patch.setattr(FESSampler, "step", interrupt)
            with pytest.raises(RuntimeError, match="test interruption"):
                run_abf(config, system, output)


def _saved_operators(path):
    with load_checkpoint(path, require_integrity=True) as saved:
        arrays = {name: saved[name] for name in saved.files}
    return arrays, json.loads(str(arrays["operators"]))


def _reject_before_pes(config, system, output, path, arrays, operators, monkeypatch, match):
    arrays["operators"] = json.dumps(operators)
    atomic_savez(path, kind="abf_stream", **arrays)
    invalid_checkpoint = path.read_bytes()

    def unexpected_provider(*args, **kwargs):
        pytest.fail("invalid thermostat state must be rejected before PES construction")

    monkeypatch.setattr("pyrpmd.workflows.abf.make_task_potential", unexpected_provider)
    with pytest.raises(ValueError, match=match):
        run_abf(config, system, output)
    assert path.read_bytes() == invalid_checkpoint


@pytest.mark.parametrize("method", ["abf", "eabf"])
@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize("deletion", ["sampling", "baths"])
def test_abf_resume_rejects_missing_nhc_bath_even_for_complete_checkpoints(
        tmp_path, monkeypatch, nhc_backend, method, complete, deletion):
    config, system = JobSpec.from_mapping(_checkpoint_job(method)).resolve()
    _make_checkpoint(config, system, tmp_path, monkeypatch, complete=complete)
    path = tmp_path / "checkpoints" / f"{method}_0000.npz"
    arrays, operators = _saved_operators(path)
    assert int(arrays["step"]) == (12 if complete else 7)
    helper, = operators["sampling"].values()
    bath, = helper["baths"].values()
    assert bath["format"] == nhc_backend
    assert helper["native"] is None
    if deletion == "sampling":
        operators["sampling"] = {}
    else:
        # Keeping the valid sampling entry used to bypass the presence check,
        # silently creating a fresh NHC on the next step.
        helper["baths"] = {}
    _reject_before_pes(config, system, tmp_path, path, arrays, operators, monkeypatch,
                       "missing thermostat auxiliary state")


@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize("corruption", ["sampler_key", "duration_key", "auxiliary"])
def test_abf_resume_validates_the_current_sampler_and_nested_state(
        tmp_path, monkeypatch, nhc_backend, complete, corruption):
    config, system = JobSpec.from_mapping(_checkpoint_job()).resolve()
    _make_checkpoint(config, system, tmp_path, monkeypatch, complete=complete)
    path = tmp_path / "checkpoints" / "abf_0000.npz"
    arrays, operators = _saved_operators(path)
    key, = operators["sampling"]
    helper = operators["sampling"][key]
    duration, = helper["baths"]
    bath = helper["baths"][duration]
    assert bath["format"] == nhc_backend
    if corruption == "sampler_key":
        spec = json.loads(key)
        spec["tau"] *= 2
        operators["sampling"] = {json.dumps(spec, sort_keys=True): helper}
        match = "sampling operator key differs"
    elif corruption == "duration_key":
        helper["baths"] = {(float.fromhex(duration) * 2).hex(): bath}
        match = "missing thermostat auxiliary state"
    else:
        if nhc_backend == "pyrpmd.python-nhc-bath.v1":
            del bath["chain"]["momentum"]
        else:
            del bath["aux"]
        match = "checkpoint|chain variables"
    _reject_before_pes(config, system, tmp_path, path, arrays, operators, monkeypatch, match)


def _assert_resumes_exactly(config, system, output, monkeypatch, interrupt_at):
    full, split = output / "full", output / "split"
    run_abf(config, system, full)
    _make_checkpoint(config, system, split, monkeypatch, complete=False, interrupt_at=interrupt_at)
    method = config["fes"]["method"]
    path = split / "checkpoints" / f"{method}_0000.npz"
    _, saved = _saved_operators(path)
    run_abf(config, system, split)
    with load_checkpoint(full / "checkpoints" / f"{method}_0000.npz") as reference, load_checkpoint(path) as resumed:
        for name in ("q", "p", "time", "step", "operators", "bias_state", "samples", "sample_steps"):
            np.testing.assert_array_equal(resumed[name], reference[name], err_msg=name)
    return saved


@pytest.mark.parametrize("method", ["abf", "eabf"])
@pytest.mark.parametrize("interrupt_at", [0, 7])
def test_nhc_checkpoint_restores_its_saved_backend_and_all_auxiliaries(
        tmp_path, monkeypatch, nhc_backend, method, interrupt_at):
    config, system = JobSpec.from_mapping(_checkpoint_job(method)).resolve()
    operators = _assert_resumes_exactly(config, system, tmp_path, monkeypatch, interrupt_at)
    if interrupt_at == 0:
        assert operators["sampling"] == {}
    else:
        helper, = operators["sampling"].values()
        bath, = helper["baths"].values()
        assert bath["format"] == nhc_backend


@pytest.mark.parametrize("integrator", ["bcocb", "baoab"])
def test_composed_pile_restores_with_matching_or_cross_selected_ring_map(tmp_path, monkeypatch, integrator):
    pytest.importorskip("pyrpmd._thermal")
    raw = _checkpoint_job(thermostat="pile-l")
    raw["fes"]["sampling"].update(
        integrator=integrator, thermostat_params={"gamma_centroid": .1, "pile_scale": 1.})
    config, system = JobSpec.from_mapping(raw).resolve()
    operators = _assert_resumes_exactly(config, system, tmp_path, monkeypatch, 7)
    key, = operators["sampling"]
    spec = json.loads(key)
    assert ("sampling_integrator" in spec) == (integrator == "baoab")
