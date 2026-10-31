"""Independent OU/FDT, nuclear-limit and restart controls for EF-RPMD."""
import copy
import json

import numpy as np
import pytest
from scipy.linalg import expm

from pyrpmd.dynamics import Dynamics, State
from pyrpmd.electronic_friction import (
    ConstantElectronicFriction, ElectronicFrictionDynamics, FrictionResult,
    make_electronic_friction, validate_friction_result,
)
from pyrpmd.potentials.base import PotentialError
from pyrpmd.potentials.builtin import HarmonicPotential
from pyrpmd.system import System
from pyrpmd.units import parse_quantity


def dynamics(beads=4, masses=(2.,), beta=3., dt=.08, gamma=.7, **kwargs):
    return ElectronicFrictionDynamics(
        HarmonicPotential(k=.3), masses, beta, beads, dt,
        ConstantElectronicFriction.from_rates(gamma, masses), native=False, **kwargs)


@pytest.mark.parametrize("scheme", ["cayley-bcb", "normal-mode-bab", "velocity-verlet"])
@pytest.mark.parametrize("beads", [1, 4, 7])
def test_zero_friction_is_exact_ordinary_rpmd_and_consumes_no_rng(scheme, beads):
    dyn = dynamics(beads=beads, gamma=0, integrator=scheme)
    reference = Dynamics(dyn.potential, dyn.masses, dyn.beta, beads, dyn.dt,
                         integrator=scheme, native=False)
    rng = np.random.default_rng(12)
    state = State(rng.normal(size=(beads, 1, 3)), rng.normal(size=(beads, 1, 3)))
    other, rng_state = state.copy(), copy.deepcopy(dyn.rng.bit_generator.state)
    for _ in range(9):
        dyn.step(state)
        reference.step(other)
    np.testing.assert_array_equal(state.q, other.q)
    np.testing.assert_array_equal(state.p, other.p)
    assert (state.step, state.time) == (other.step, other.time)
    assert dyn.rng.bit_generator.state == rng_state


class FixedNoise:
    def __init__(self, value):
        self.value = np.asarray(value)

    def normal(self, size):
        assert size == self.value.shape
        return self.value.copy()


def test_correlated_unequal_mass_ou_matches_independent_matrix_exponential():
    masses = np.array([2., 11.])
    raw = np.random.default_rng(17).normal(size=(6, 6))
    eta = raw @ raw.T
    beta, duration = 2.7, .19
    dyn = ElectronicFrictionDynamics(HarmonicPotential(), masses, beta, 5, .04,
                                    ConstantElectronicFriction(eta), native=False)
    q = np.zeros((5, 2, 3))
    initial = np.arange(6) / 5
    m = np.repeat(masses, 3)
    drift = expm(-duration * eta @ np.diag(1 / m))
    expected_mean = drift @ initial
    expected_covariance = np.diag(m) / beta - drift @ np.diag(m / beta) @ drift.T
    # Gaussian cubature integrates first/second moments exactly; the reference
    # drift is computed in physical momenta, not by the implementation's eigensystem.
    outputs = []
    for noise in np.concatenate((np.sqrt(6) * np.eye(6), -np.sqrt(6) * np.eye(6))):
        dyn.rng = FixedNoise(noise)
        p = np.broadcast_to(initial.reshape(2, 3), q.shape).copy()
        dyn._centroid_ou(q, p, duration)
        outputs.append(p.mean(axis=0).ravel())
    outputs = np.asarray(outputs)
    np.testing.assert_allclose(outputs.mean(axis=0), expected_mean, atol=2e-14)
    covariance = (outputs - expected_mean).T @ (outputs - expected_mean) / len(outputs)
    np.testing.assert_allclose(covariance, expected_covariance, atol=3e-14)


@pytest.mark.parametrize("beads", [1, 4, 9])
def test_fdt_centroid_equilibrium_variance_is_bead_independent(beads):
    dyn = dynamics(beads=beads, gamma=.6, beta=1.7)
    # Six independent Gaussian coordinates: three for initial pbar, three for O.
    points = np.concatenate((np.sqrt(6) * np.eye(6), -np.sqrt(6) * np.eye(6)))
    results = []
    for point in points:
        pbar = np.sqrt(2 / dyn.beta) * point[:3]
        p = np.broadcast_to(pbar, (beads, 1, 3)).copy()
        dyn.rng = FixedNoise(point[3:])
        dyn._centroid_ou(np.zeros_like(p), p, .8)
        results.append(p.mean(axis=0).ravel())
    values = np.array(results)
    np.testing.assert_allclose(values.mean(axis=0), 0, atol=1e-15)
    np.testing.assert_allclose(values.T @ values / len(values), np.eye(3) * 2 / dyn.beta, atol=1e-14)


def test_position_dependent_tensor_is_averaged_over_beads_and_leaves_internal_modes():
    class Spatial:
        def evaluate(self, q):
            return FrictionResult(q[:, 0, 0, None, None] ** 2 * np.eye(3))
    dyn = ElectronicFrictionDynamics(HarmonicPotential(), [2.], 3., 2, .1,
                                    Spatial(), native=False)
    dyn.rng = FixedNoise(np.zeros(3))
    q = np.array([[[1., 0., 0.]], [[3., 0., 0.]]])
    p = np.array([[[2., 3., 4.]], [[4., 1., 2.]]])
    initial_mean, internal = p.mean(axis=0), p - p.mean(axis=0)
    dyn._centroid_ou(q, p, .7)
    # eta averaged at q=1,3 is 5, while eta(q_centroid)=4 would be wrong.
    np.testing.assert_allclose(p.mean(axis=0), initial_mean * np.exp(-5 * .7 / 2))
    np.testing.assert_allclose(p - p.mean(axis=0), internal, atol=1e-15)


@pytest.mark.parametrize("bad", [
    np.diag([-1., 1., 1.])[None], np.ones((2, 3, 3)),
    np.array([[[1., .2, 0], [0, 1, 0], [0, 0, 1]]]),
    np.full((1, 3, 3), np.nan), np.eye(3, dtype=complex)[None],
    FrictionResult(np.eye(3)[None], status=np.array([1])),
    FrictionResult(np.eye(3)[None], messages=(1,)),
])
def test_invalid_friction_fails_explicitly(bad):
    with pytest.raises((ValueError, PotentialError)):
        validate_friction_result(bad, (1, 1, 3))


def test_second_friction_failure_rolls_back_state_and_rng():
    class Broken:
        count = 0
        def evaluate(self, q):
            self.count += 1
            if self.count == 2:
                raise RuntimeError("electronic calculation failed")
            return np.broadcast_to(np.eye(3), (len(q), 3, 3))
    dyn = ElectronicFrictionDynamics(HarmonicPotential(), [2.], 3., 4, .1,
                                    Broken(), native=False, seed=7)
    state = State(np.ones((4, 1, 3)), np.ones((4, 1, 3)))
    before, rng = state.copy(), copy.deepcopy(dyn.rng.bit_generator.state)
    with pytest.raises(RuntimeError, match="electronic calculation failed"):
        dyn.step(state)
    np.testing.assert_array_equal(state.q, before.q)
    np.testing.assert_array_equal(state.p, before.p)
    assert (state.step, state.time) == (before.step, before.time)
    assert dyn.rng.bit_generator.state == rng


def test_failed_block_observation_restores_rng_and_state():
    class BrokenCV:
        dimension = 1
        def value(self, q):
            raise RuntimeError("broken observation")
    dyn = dynamics(seed=22)
    state = State(np.ones((4, 1, 3)), np.ones((4, 1, 3)))
    old = state.copy()
    rng, stats = copy.deepcopy(dyn.rng.bit_generator.state), copy.deepcopy(dyn.propagation_stats)
    with pytest.raises(RuntimeError, match="broken observation"):
        dyn.propagate(state, 3, observe=BrokenCV(), backend="python")
    np.testing.assert_array_equal(state.q, old.q)
    np.testing.assert_array_equal(state.p, old.p)
    assert state.step == old.step
    assert dyn.rng.bit_generator.state == rng
    assert dyn.propagation_stats == stats


def test_fixed_atoms_remain_exact_with_coupled_tensor():
    reference = np.array([[.1, .2, .3], [1., 2., 3.]])
    eta = np.eye(6) + .3 * np.ones((6, 6))
    dyn = ElectronicFrictionDynamics(HarmonicPotential(), [2., 10.], 3., 4, .1,
                                    ConstantElectronicFriction(eta), native=False,
                                    fixed_atoms=(1,), fixed_positions=reference)
    state = State(np.broadcast_to(reference, (4, 2, 3)), dyn.sample_momenta())
    for _ in range(30):
        dyn.step(state)
    np.testing.assert_array_equal(state.q[:, 1], np.broadcast_to(reference[1], (4, 3)))
    np.testing.assert_array_equal(state.p[:, 1], 0)


def test_rates_units_and_physical_tensor_conversion():
    provider = ConstantElectronicFriction.from_rates(".01 fs^-1", [2., 5.])
    eta = provider.evaluate(np.zeros((3, 2, 3))).tensor
    expected = parse_quantity(".01 fs^-1", "frequency") * np.repeat([2., 5.], 3)
    np.testing.assert_array_equal(np.diagonal(eta, axis1=1, axis2=2), np.broadcast_to(expected, (3, 6)))
    with pytest.raises(ValueError):
        ConstantElectronicFriction.from_rates(".01 fs", [2.])


def test_periodic_factory_requires_explicit_cell_contract_and_closes_on_error(tmp_path):
    plugin = tmp_path / "friction.py"
    plugin.write_text("class Provider:\n    def evaluate(self, q): pass\n    def close(self): self.closed=True\nobj=Provider()\ndef make(): return obj\n")
    system = System(("H",), np.array([2.]), np.zeros((1, 3)), cell=np.eye(3) * 10, pbc=(True, True, True))
    config = {"task": "diffusion", "electronic_friction": {"provider": "python", "factory": f"{plugin}:make"}}
    with pytest.raises(ValueError, match="periodic friction"):
        make_electronic_friction(config, system)
    from pyrpmd.potentials.python import import_object
    assert import_object(f"{plugin}:obj").closed


def test_fixed_surface_friction_stream_restart_and_analysis(tmp_path, monkeypatch):
    from pyrpmd.workflows.materials import _sample_replica, run_materials
    system = System(("H", "Ni"), np.array([2., 20.]), np.array([[.1, 0, 0], [1., 0, 0]]), fixed_atoms=(1,))
    config = {
        "task": "diffusion", "beta": 2., "beads": 3, "dt_au": .03,
        "seed": 412, "native": False,
        "potential": {"provider": "builtin", "name": "harmonic", "params": {"k": .4}},
        "sampling": {"equilibration_steps": 4, "production_steps": 30,
                     "sample_interval": 2, "replicas": 1, "gamma_centroid": .6, "thermostat": "pile-l"},
        "dynamics": {"loop_backend": "python"},
        "electronic_friction": {"gamma": .7},
        "diffusion": {"atom_indices": [0], "axes": [0], "bootstrap": 0},
        "output": {"checkpoint_steps": 4},
    }
    def task(where):
        return {"config": config, "system": system, "output": where, "replica": 0}
    reference = _sample_replica(task(tmp_path / "reference"))
    original = ElectronicFrictionDynamics.step
    def fail_after_step(self, state, *args, **kwargs):
        result = original(self, state, *args, **kwargs)
        if state.step == 11:
            raise RuntimeError("interrupt physical bath")
        return result
    with monkeypatch.context() as patch:
        patch.setattr(ElectronicFrictionDynamics, "step", fail_after_step)
        with pytest.raises(RuntimeError, match="interrupt physical bath"):
            _sample_replica(task(tmp_path / "resumed"))
    resumed = _sample_replica(task(tmp_path / "resumed"))
    for key in ("positions", "velocities", "observables", "sample_steps"):
        np.testing.assert_array_equal(reference[key], resumed[key])
    assert reference["rng"] == resumed["rng"]
    assert reference["propagation"] == resumed["propagation"]
    np.testing.assert_array_equal(resumed["state"].p[:, 1], 0)
    summary = run_materials(config, system, tmp_path / "resumed")
    assert summary["phase"] == "centroid_ef_rpmd_production"
    assert summary["nve_energy_diagnostics"] is None
    assert not summary["physical_converged"]
    json.dumps(summary, allow_nan=False)


def test_mobile_quantum_estimator_counts_no_fixed_kinetic_degrees():
    from pyrpmd.workflows.materials import quantum_observables
    q = np.array([[[.2, .3, .4], [1., 0, 0]]])
    value, _ = quantum_observables(State(q, np.zeros_like(q)), HarmonicPotential(), [2., 9.], 3., mobile_mask=[True, False])
    assert value[1] == value[2] == pytest.approx(3 / (2 * 3))
