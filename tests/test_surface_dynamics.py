"""Fixed-substrate dynamics and periodic FES must preserve physical state."""

import copy
from unittest.mock import patch

import numpy as np
import pytest

from pyrpmd.cv import Cartesian, CVStack
from pyrpmd.dynamics import Dynamics, State, INTEGRATORS, native_capabilities
from pyrpmd.potentials import FreePotential, HarmonicPotential, PESResult, PotentialError
from pyrpmd.system import System
from pyrpmd.workflows.fes import run_fes, steer_to_center, _fingerprint


REFERENCE = np.array([[1.25, -0.5, 0.25], [1.4, 0.1, 1.5]])


class CoupledSpring:
    """The frozen atom exerts a real force on the moving atom."""

    def evaluate(self, q):
        np.testing.assert_array_equal(q[:, 0], np.broadcast_to(REFERENCE[0], q[:, 0].shape))
        delta = q[:, 1] - q[:, 0]
        forces = np.empty_like(q)
        forces[:, 0] = 0.7 * delta
        forces[:, 1] = -0.7 * delta
        return PESResult(0.35 * np.sum(delta * delta, axis=1), forces)


def dynamics(potential=None, *, native=False, **kwargs):
    return Dynamics(
        CoupledSpring() if potential is None else potential,
        [12.0, 2.0], 3.0, 4, 0.02, seed=123, native=native,
        fixed_atoms=[0], fixed_positions=REFERENCE, **kwargs,
    )


def initial(dyn):
    q = np.repeat(REFERENCE[None], dyn.nbeads, axis=0)
    q[:, 1] += np.random.default_rng(22).normal(scale=0.04, size=(dyn.nbeads, 3))
    return State(q, dyn.sample_momenta())


def assert_fixed(state):
    np.testing.assert_array_equal(state.q[:, 0], np.broadcast_to(REFERENCE[0], state.q[:, 0].shape))
    np.testing.assert_array_equal(state.p[:, 0], 0.0)


@pytest.mark.parametrize("scheme", INTEGRATORS)
@pytest.mark.parametrize("native", [False] + ([True] if native_capabilities()["available"] else []))
def test_coupled_frozen_substrate_matches_reduced_hamiltonian_and_is_reversible(scheme, native):
    dyn = dynamics(integrator=scheme, native=native)
    state = initial(dyn)
    original = state.copy()
    reduced = Dynamics(HarmonicPotential(0.7, REFERENCE[0]), [2.0], 3.0, 4, 0.02,
                       native=native, integrator=scheme)
    moving = State(state.q[:, 1:2], state.p[:, 1:2])
    e0 = dyn.energy(state)
    for _ in range(100):
        dyn.step(state)
        reduced.step(moving)
        assert_fixed(state)
    np.testing.assert_allclose(state.q[:, 1:2], moving.q, atol=2e-14, rtol=0)
    np.testing.assert_allclose(state.p[:, 1:2], moving.p, atol=2e-14, rtol=0)
    # Verlet approximates the spring motion too; its bounded energy error is
    # larger than the Cayley/exact free-ring maps at this same timestep.
    assert dyn.energy(state) == pytest.approx(reduced.energy(moving), abs=5e-14)
    assert abs(dyn.energy(state) - e0) / abs(e0) < 1e-3
    state.p *= -1
    for _ in range(100):
        dyn.step(state)
    np.testing.assert_allclose(state.q, original.q, atol=2e-13, rtol=0)
    np.testing.assert_allclose(state.p, -original.p, atol=2e-13, rtol=0)


@pytest.mark.parametrize("scheme", INTEGRATORS)
def test_thermal_momenta_and_supported_thermostats_leave_fixed_atoms_exact(scheme):
    dyn = dynamics(integrator=scheme)
    momenta = np.array([dyn.sample_momenta() for _ in range(6000)])
    np.testing.assert_array_equal(momenta[:, :, 0], 0.0)
    assert np.isclose(momenta[:, :, 1].var(), 2.0 * 4 / 3, rtol=0.025)
    a, b = dynamics(integrator=scheme), dynamics(integrator=scheme)
    state, split = initial(a), initial(b)
    a.thermalize(state, 29, interval=7)
    b.thermalize(split, 11, interval=7)
    b.thermalize(split, 18, interval=7, resample=False)
    np.testing.assert_array_equal(state.q, split.q)
    np.testing.assert_array_equal(state.p, split.p)
    assert_fixed(state)
    assert a.propagation_stats["fallback_reasons"]["fixed atoms require the Python propagation loop"] == 29
    previous_mobile = state.q[:, 1].copy()
    for _ in range(20):
        if scheme == "velocity-verlet":
            a.thermalize(state, 1, interval=7, resample=False)
        else:
            a.sample_step(state, gamma_centroid=0.7)
        assert_fixed(state)
    assert np.linalg.norm(previous_mobile - state.q[:, 1]) > 1e-3


@pytest.mark.parametrize("scheme,method", [("cayley-bcb", "bcocb"), ("normal-mode-bab", "baoab")])
def test_surface_pile_l_failure_rolls_back_fixed_state_and_rng_then_retries_identically(scheme, method):
    class FailSecondForce(CoupledSpring):
        calls = 0

        def evaluate(self, q):
            self.calls += 1
            if self.calls == 2:
                raise PotentialError("second surface force failed")
            return super().evaluate(q)

    failing = dynamics(FailSecondForce(), integrator=scheme)
    reference = dynamics(integrator=scheme)
    state, expected = initial(failing), initial(reference)
    old = state.copy()
    rng_state = copy.deepcopy(failing.rng.bit_generator.state)
    with pytest.raises(PotentialError, match="second surface force failed"):
        failing.sample_step(state, method=method, gamma_centroid=0.7)
    np.testing.assert_array_equal(state.q, old.q)
    np.testing.assert_array_equal(state.p, old.p)
    assert (state.step, state.time) == (old.step, old.time)
    assert failing.rng.bit_generator.state == rng_state
    assert_fixed(state)
    failing.potential = CoupledSpring()
    failing.sample_step(state, method=method, gamma_centroid=0.7)
    reference.sample_step(expected, method=method, gamma_centroid=0.7)
    np.testing.assert_array_equal(state.q, expected.q)
    np.testing.assert_array_equal(state.p, expected.p)
    assert failing.rng.bit_generator.state == reference.rng.bit_generator.state
    assert_fixed(state)


@pytest.mark.parametrize("atoms,positions", [
    ([0, 0], REFERENCE), ([True], REFERENCE), ([2], REFERENCE),
    ([-1], REFERENCE), ([0.5], REFERENCE), ([0], None),
    ([0], [[0, 0, 0]]), ([0], REFERENCE + 1j), ([0], REFERENCE * np.nan),
])
def test_invalid_fixed_atom_configuration_is_rejected(atoms, positions):
    with pytest.raises(ValueError):
        Dynamics(FreePotential(), [1.0, 2.0], 3.0, 4, 0.01,
                 native=False, fixed_atoms=atoms, fixed_positions=positions)


def test_inconsistent_initial_state_and_overlapping_fragment_constraints_are_rejected():
    dyn = dynamics()
    for attribute in ("q", "p"):
        state = initial(dyn)
        getattr(state, attribute)[0, 0, 0] += 0.1
        original = state.copy()
        with pytest.raises(ValueError, match="fixed atoms"):
            dyn.step(state)
        np.testing.assert_array_equal(state.q, original.q)
        np.testing.assert_array_equal(state.p, original.p)
    with pytest.raises(ValueError, match="fragment COM constraints"):
        dyn.thermalize(initial(dyn), 2, fragments=[[0, 1]])


@pytest.mark.parametrize("scheme", INTEGRATORS)
def test_native_requirement_is_explicit_and_auto_blocks_report_fallback(scheme):
    dyn = dynamics(integrator=scheme)
    state = initial(dyn)
    a = dyn.propagate(state, 3)
    b = dyn.propagate_block(state, 2, record_states=True)
    assert a.backend == b.backend == "python"
    assert "fixed atoms" in a.fallback_reason and "fixed atoms" in b.fallback_reason
    assert_fixed(state)
    assert dyn.propagation_stats["python_steps"] == 5
    assert dyn.propagation_stats["fallback_reasons"][a.fallback_reason] == 5
    before = state.copy()
    for operation in (dyn.propagate, dyn.propagate_block):
        with pytest.raises(RuntimeError, match="fixed atoms"):
            operation(state, 1, backend="native")
    np.testing.assert_array_equal(state.q, before.q)
    np.testing.assert_array_equal(state.p, before.p)


class FailAtPlane(FreePotential):
    def evaluate(self, q):
        if np.mean(q[:, 1, 0]) > 0.005:
            raise PotentialError("test PES domain boundary")
        return super().evaluate(q)


@pytest.mark.parametrize("scheme", INTEGRATORS)
def test_fixed_atoms_keep_whole_block_rollback_and_valid_prefix_failure_contracts(scheme):
    dyn = dynamics(FailAtPlane(), integrator=scheme)
    q = np.repeat(REFERENCE[None], 4, axis=0)
    q[:, 1] = 0.0
    p = np.zeros_like(q)
    p[:, 1, 0] = 0.1  # 0.001 bohr per step, m=2 and dt=0.02
    state = State(q, p)
    original = state.copy()
    with pytest.raises(PotentialError, match="domain boundary"):
        dyn.propagate(state, 10)
    np.testing.assert_array_equal(state.q, original.q)
    np.testing.assert_array_equal(state.p, original.p)
    assert state.step == 0
    result = dyn.propagate_block(state, 10, record_states=True, raise_on_error=False)
    assert 1 <= result.completed < 10
    assert state.step == result.completed
    assert "domain boundary" in result.error
    assert_fixed(state)
    np.testing.assert_allclose(state.q[:, 1, 0], result.completed * 0.001, atol=1e-15)


def test_steering_uses_only_mobile_degrees_of_freedom():
    from pyrpmd.cv import Distance

    q = np.repeat(REFERENCE[None], 4, axis=0)
    cv = CVStack([Distance([0, 1])])
    steered = steer_to_center(q, cv, [2.5], [12.0, 2.0], fixed_atoms=[0])
    np.testing.assert_array_equal(steered[:, 0], q[:, 0])
    np.testing.assert_allclose(cv.value(steered), [2.5], atol=1e-7)
    with pytest.raises(ValueError, match="fixed atoms"):
        steer_to_center(q, CVStack([Cartesian(0, "z")]), [5.0], [12.0, 2.0], fixed_atoms=[0])


class PeriodicHeightPotential:
    """Analytic z-only PES: translationally invariant in the periodic plane."""

    metadata = {"supports_pbc": True}

    def __repr__(self):
        return "PeriodicHeightPotential(k=0.7,z0=1.5)"

    def bind_system(self, system):
        assert tuple(system.pbc) == (True, True, False)

    def evaluate(self, q):
        np.testing.assert_array_equal(q[:, 0], np.broadcast_to(REFERENCE[0], q[:, 0].shape))
        delta = q[:, 1, 2] - 1.5
        force = np.zeros_like(q)
        force[:, 1, 2] = -0.7 * delta
        return PESResult(0.35 * delta * delta, force)


def surface_fes_config():
    return {
        "task": "fes", "beta": 3.0, "beads": 4, "dt_au": 0.02,
        "seed": 41, "native": False, "potential": PeriodicHeightPotential(),
        "sampling": {"equilibration_steps": 8, "production_steps": 48,
                     "sample_interval": 2, "andersen_interval": 7, "replicas": 1},
        "cvs": [{"kind": "cartesian", "atom": 1, "axis": "z"}],
        "umbrella": {"centers": [1.2, 1.5, 1.8], "stiffness": 3.0},
        "analysis": {"bins": 6, "bootstrap": 0},
        "output": {"checkpoint_steps": 5},
    }


def test_periodic_fixed_substrate_fes_restarts_exactly(tmp_path):
    system = System(("C", "H"), np.array([12.0, 2.0]), REFERENCE,
                    cell=np.diag([8.0, 9.0, 16.0]), pbc=[True, True, False], fixed_atoms=[0])
    config = surface_fes_config()
    full = run_fes(config, system, tmp_path / "full", return_data=True)
    original_step = Dynamics.step

    def interrupt(dynamics, state, bias=None):
        if state.step == 23:
            raise RuntimeError("deliberate interruption after a saved checkpoint")
        return original_step(dynamics, state, bias)

    with patch.object(Dynamics, "step", interrupt):
        with pytest.raises(RuntimeError, match="deliberate interruption"):
            run_fes(config, system, tmp_path / "restart", return_data=True)
    resumed = run_fes(config, system, tmp_path / "restart", return_data=True)
    repeated = run_fes(config, system, tmp_path / "restart", return_data=True)
    for a, b, c in zip(full["streams"], resumed["streams"], repeated["streams"]):
        for key in ("samples", "energies", "last_q", "last_p"):
            np.testing.assert_array_equal(a[key], b[key])
            np.testing.assert_array_equal(b[key], c[key])
        assert_fixed(State(b["last_q"], b["last_p"]))
        assert b["propagation"]["fallback_reasons"]["fixed atoms require the Python propagation loop"] == 56
    changed = System.from_dict(system.to_dict())
    changed.cell[0, 0] += 1.0
    assert _fingerprint(config, changed, {}) != _fingerprint(config, system, {})
    with pytest.raises(ValueError, match="checkpoint physics differs"):
        run_fes(config, changed, tmp_path / "restart")
