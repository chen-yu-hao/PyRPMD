"""All-bead exchange BZP: independent derivatives, invariants and state safety.

The finite-box quantum NPT distribution is checked separately in
test_exchange_periodic_reference.py against reciprocal-space partition sums.
"""

import copy
import math

import numpy as np
import pytest

from pyrpmd.barostat import BZPBarostat
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.exchange_barostat import ExchangeBZPBarostat, validate_exchange_bzp_snapshot
from pyrpmd.pimd_methods import BosonicExchangePotential
from pyrpmd.potentials.base import PESResult, PotentialError
from pyrpmd.system import System


class PeriodicWell:
    """A fixed fractional cosine plus A/V with independently known virial."""

    def __init__(self, cell, height=0.1, volume_energy=0.4):
        self.set_cell(cell)
        self.height, self.volume_energy = height, volume_energy
        self.calls, self.fail_at = 0, None

    def set_cell(self, cell):
        self.cell = np.array(cell, copy=True)

    def evaluate(self, q):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("injected exchange force failure")
        inverse = np.linalg.inv(self.cell)
        phase = 2 * np.pi * q @ inverse
        offset = self.volume_energy / np.linalg.det(self.cell)
        energy = self.height * np.sum(1 - np.cos(phase), axis=(1, 2)) + offset
        force = -2 * np.pi * self.height * np.sin(phase) @ inverse.T
        virial = np.repeat((offset * np.eye(3))[None], len(q), axis=0)
        return PESResult(energy, force, virial=virial)


def make(*, beads=3, dt=0.01, friction=0.0, seed=714,
         periodic_mode="full_winding", statistics="bosonic"):
    cell = np.diag([4.2, 4.8, 5.0])
    masses, beta = np.array([1.4, 1.4]), 2.7
    system = System(("H", "H"), masses, [[0.3, 0.2, -0.1], [0.9, -0.4, 0.6]], cell=cell)
    physical = PeriodicWell(cell)
    action = BosonicExchangePotential(physical, masses, beta, beads, cell=cell,
                                     periodic_mode=periodic_mode, statistics=statistics)
    dynamics = Dynamics(action, masses, beta, beads, dt, integrator="velocity-verlet",
                        seed=seed, loop_backend="python", native=False,
                        cell=cell, pbc=system.pbc)
    barostat = ExchangeBZPBarostat(action, system, beta, beads, 0.02, 1.4,
                                 friction_au=friction)
    q = system.geometry[None] + np.random.default_rng(713).normal(0, 0.12, (beads, 2, 3))
    state = State(q, dynamics.sample_momenta())
    barostat.momentum = 0.2
    return barostat, dynamics, state, system, physical


@pytest.mark.parametrize("periodic_mode", ["minimum_image", "full_winding"])
def test_full_action_pressure_matches_affine_cell_energy_derivative(periodic_mode):
    barostat, dynamics, state, _, _ = make(periodic_mode=periodic_mode)
    original = barostat.snapshot()
    epsilon = 2e-6
    energy = []
    for direction in (-1, 1):
        scale = math.exp(direction * epsilon)
        barostat.restore({**original, "cell": (np.array(original["cell"]) * scale).tolist()})
        energy.append(dynamics.potential.evaluate_sampling(state.q * scale).energy.sum())
    barostat.restore(original)
    twice_kinetic = np.sum(state.p**2 / dynamics.masses[None, :, None])
    expected = (twice_kinetic - (energy[1] - energy[0]) / (2 * epsilon))
    expected /= 3 * dynamics.nbeads * barostat.volume
    assert barostat.pressure(state) == pytest.approx(expected, rel=2e-8, abs=2e-10)


def test_dilation_scales_every_bead_and_preserves_fractional_coordinates():
    barostat, _, state, _, _ = make()
    before, cell = state.copy(), barostat.cell
    factor = math.exp(0.2 * barostat.momentum / barostat.mass)
    barostat._dilate(state, 0.2)
    np.testing.assert_allclose(state.q, factor * before.q, rtol=0, atol=1e-14)
    np.testing.assert_allclose(state.p, before.p / factor, rtol=0, atol=1e-14)
    np.testing.assert_allclose(state.q @ np.linalg.inv(barostat.cell), before.q @ np.linalg.inv(cell), atol=1e-14)
    np.testing.assert_allclose(state.q - state.q.mean(axis=0),
                               factor * (before.q - before.q.mean(axis=0)), atol=1e-14)
    assert barostat.mass == pytest.approx(3 * 2 * 3 * 1.4**2 / 2.7)


def test_combined_drift_has_regular_zero_piston_limit():
    barostat, dynamics, state, _, _ = make()
    barostat.momentum = 0
    before, cell = state.copy(), barostat.cell
    barostat._drift_dilate(state, 0.03)
    np.testing.assert_allclose(state.q, before.q + 0.03 * before.p / dynamics.masses[None, :, None], atol=1e-14)
    np.testing.assert_array_equal(state.p, before.p)
    np.testing.assert_array_equal(barostat.cell, cell)


def test_single_bead_image_change_preserves_pressure_and_entire_step():
    left, dynamics_l, state_l, _, _ = make(friction=0.3)
    right, dynamics_r, state_r, _, _ = make(friction=0.3)
    image = np.array([2, -1, 1])
    state_r.q[0, 0] += image @ right.cell
    assert left.pressure(state_l) == pytest.approx(right.pressure(state_r), abs=2e-13)
    for _ in range(7):
        left.step(state_l, dynamics_l, andersen_interval=3)
        right.step(state_r, dynamics_r, andersen_interval=3)
    expected = state_l.q.copy()
    expected[0, 0] += image @ left.cell
    np.testing.assert_allclose(state_r.q, expected, atol=2e-12, rtol=0)
    np.testing.assert_allclose(state_r.p, state_l.p, atol=2e-12, rtol=0)
    np.testing.assert_allclose(right.cell, left.cell, atol=2e-12, rtol=0)
    assert right.momentum == pytest.approx(left.momentum, abs=2e-12)


@pytest.mark.parametrize("periodic_mode", ["minimum_image", "full_winding"])
def test_deterministic_map_is_time_reversible(periodic_mode):
    barostat, dynamics, state, _, _ = make(dt=0.005, periodic_mode=periodic_mode)
    original, piston = state.copy(), barostat.snapshot()
    for _ in range(50):
        barostat.step(state, dynamics, thermostat=None)
    state.p *= -1
    barostat.momentum *= -1
    for _ in range(50):
        barostat.step(state, dynamics, thermostat=None)
    np.testing.assert_allclose(state.q, original.q, atol=3e-12, rtol=0)
    np.testing.assert_allclose(state.p, -original.p, atol=3e-12, rtol=0)
    np.testing.assert_allclose(barostat.cell, piston["cell"], atol=3e-12, rtol=0)
    assert barostat.momentum == pytest.approx(-piston["momentum"], abs=3e-12)


def test_extended_energy_error_has_second_order_timestep_scaling():
    excursions = []
    for dt in (0.02, 0.01, 0.005):
        barostat, dynamics, state, _, _ = make(dt=dt)
        initial = barostat.extended_energy(state, dynamics)
        error = 0
        for _ in range(round(0.6 / dt)):
            barostat.step(state, dynamics, thermostat=None)
            error = max(error, abs(barostat.extended_energy(state, dynamics) - initial))
        excursions.append(error)
    assert 3.7 < excursions[0] / excursions[1] < 4.3
    assert 3.7 < excursions[1] / excursions[2] < 4.3


def test_force_piston_kick_includes_exact_quadratic_and_cubic_terms():
    barostat, dynamics, state, _, _ = make()
    initial_p, initial_pi = state.p.copy(), barostat.momentum
    result = barostat.potential.evaluate_sampling(state.q)
    duration = 0.07
    # Independent Gaussian quadrature integrates the quadratic K(p+tF).
    roots, weights = np.polynomial.legendre.leggauss(3)
    times = (roots + 1) * duration / 2
    force = []
    for time in times:
        momentum = initial_p + time * result.forces
        force.append(np.sum(momentum**2 / dynamics.masses[None, :, None])
                     + np.trace(result.virial, axis1=1, axis2=2).sum()
                     - 3 * dynamics.nbeads * barostat.pressure_au * barostat.volume
                     + 3 * dynamics.nbeads / dynamics.beta)
    expected = initial_pi + duration / 2 * np.dot(weights, force)
    barostat._force_piston_kick(state, result, duration)
    assert barostat.momentum == pytest.approx(expected, abs=1e-14)
    np.testing.assert_allclose(state.p, initial_p + duration * result.forces, atol=1e-14)


def test_deterministic_map_jacobian_has_the_required_volume_compressibility():
    barostat, dynamics, state, _, _ = make(beads=1, dt=0.004)
    original = barostat.snapshot()
    size = state.q.size
    volume = barostat.volume
    packed = np.r_[state.q.ravel(), state.p.ravel(), volume, barostat.momentum]

    def mapped(values):
        cell = np.asarray(original["cell"]) * (values[-2] / volume) ** (1 / 3)
        barostat.restore({**original, "cell": cell.tolist(), "momentum": values[-1]})
        candidate = State(values[:size].reshape(state.q.shape),
                          values[size:2 * size].reshape(state.p.shape))
        barostat.step(candidate, dynamics, thermostat=None)
        return np.r_[candidate.q.ravel(), candidate.p.ravel(), barostat.volume, barostat.momentum]

    transformed = mapped(packed)
    jacobian = np.empty((len(packed), len(packed)))
    for index in range(len(packed)):
        epsilon = 2e-6 * max(1, abs(packed[index]))
        plus, minus = packed.copy(), packed.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        jacobian[:, index] = (mapped(plus) - mapped(minus)) / (2 * epsilon)
    # In Cartesian q,p,V,pi coordinates the q/p dilation determinants cancel.
    # A d(log V) measure would instead remove this positive V'/V Jacobian.
    assert np.linalg.det(jacobian) == pytest.approx(transformed[-2] / volume, rel=2e-8)


def test_step_uses_two_physical_evaluations_and_records_one_step():
    barostat, dynamics, state, _, physical = make(friction=0.3)
    calls = physical.calls
    operators = dynamics.operator_state_dict()
    barostat.step(state, dynamics, andersen_interval=1)
    assert physical.calls - calls == 2
    assert state.step == 1
    assert state.time == dynamics.dt
    assert dynamics.propagation_stats["python_steps"] == 1
    assert dynamics.propagation_stats["native_steps"] == 0
    # Only RNG changes in the operator serialization; no hidden bath advances.
    current = dynamics.operator_state_dict()
    assert current["composition"] == operators["composition"]
    assert current["sampling"] == operators["sampling"]


def test_failed_step_restores_rng_cells_piston_clock_and_retry_exactly():
    barostat, dynamics, state, _, physical = make(friction=0.3)
    other_b, other_d, other, _, _ = make(friction=0.3)
    barostat.step(state, dynamics, andersen_interval=3)
    other_b.step(other, other_d, andersen_interval=3)
    saved, piston = state.copy(), barostat.snapshot()
    rng = copy.deepcopy(dynamics.rng.bit_generator.state)
    stats = copy.deepcopy(dynamics.propagation_stats)
    operators = dynamics.operator_state_dict()
    physical.fail_at = physical.calls + 2
    with pytest.raises(RuntimeError, match="injected exchange force failure"):
        barostat.step(state, dynamics, andersen_interval=1)
    np.testing.assert_array_equal(state.q, saved.q)
    np.testing.assert_array_equal(state.p, saved.p)
    assert (state.step, state.time) == (saved.step, saved.time)
    assert barostat.snapshot() == piston
    assert dynamics.rng.bit_generator.state == rng
    assert dynamics.propagation_stats == stats
    assert dynamics.operator_state_dict() == operators
    np.testing.assert_array_equal(physical.cell, piston["cell"])
    np.testing.assert_array_equal(dynamics.potential.cell, piston["cell"])
    physical.fail_at = None
    barostat.step(state, dynamics, andersen_interval=1)
    other_b.step(other, other_d, andersen_interval=1)
    np.testing.assert_array_equal(state.q, other.q)
    np.testing.assert_array_equal(state.p, other.p)
    assert barostat.snapshot() == other_b.snapshot()
    assert dynamics.rng.bit_generator.state == other_d.rng.bit_generator.state


def test_checkpoint_resume_preserves_global_collision_clock_and_both_rng_draws():
    barostat, dynamics, state, _, _ = make(friction=0.4)
    for _ in range(8):
        barostat.step(state, dynamics, andersen_interval=3)
    midpoint, piston = state.copy(), barostat.snapshot()
    rng = copy.deepcopy(dynamics.rng.bit_generator.state)
    for _ in range(7):
        barostat.step(state, dynamics, andersen_interval=3)
    resumed, other_d, _, _, _ = make(friction=0.4)
    resumed.restore(piston)
    other_d.rng.bit_generator.state = rng
    for _ in range(7):
        resumed.step(midpoint, other_d, andersen_interval=3)
    np.testing.assert_array_equal(midpoint.q, state.q)
    np.testing.assert_array_equal(midpoint.p, state.p)
    assert resumed.snapshot() == barostat.snapshot()
    assert other_d.rng.bit_generator.state == dynamics.rng.bit_generator.state


def test_fermionic_reweighting_has_identical_positive_sampling_trajectory():
    bosonic, d_b, state_b, _, _ = make(statistics="bosonic", friction=0.2)
    fermionic, d_f, state_f, _, _ = make(statistics="fermionic", friction=0.2)
    for _ in range(5):
        bosonic.step(state_b, d_b, andersen_interval=2)
        fermionic.step(state_f, d_f, andersen_interval=2)
    np.testing.assert_array_equal(state_b.q, state_f.q)
    np.testing.assert_array_equal(state_b.p, state_f.p)
    assert bosonic.snapshot() == fermionic.snapshot()


def test_centroid_checkpoint_and_wrong_coupling_are_rejected():
    barostat, dynamics, _, system, _ = make()
    original = barostat.snapshot()
    assert validate_exchange_bzp_snapshot(original, system.cell) == original
    wrong = [{key: value for key, value in original.items() if key != "coupling"},
             {**original, "coupling": "centroids"}, {**original, "coupling": True},
             {**original, "extra": 1}, {**original, "momentum": np.nan},
             {**original, "cell": np.diag([4.3, 4.8, 5.0]).tolist()}]
    for snapshot in wrong:
        with pytest.raises(ValueError):
            barostat.restore(snapshot)
        assert barostat.snapshot() == original
    # The old centroid class must remain an explicit error for this action.
    with pytest.raises(PotentialError, match="all_beads|centroid"):
        BZPBarostat(dynamics.potential, system, dynamics.beta, dynamics.nbeads, 0.02, 1.4)


@pytest.mark.parametrize("options", [{"thermostat": "pile-l"}, {"method": "bcocb"},
                                    {"gamma_centroid": 0.2}, {"andersen_interval": True},
                                    {"andersen_interval": 0}])
def test_unsupported_step_options_fail_before_state_or_noise_changes(options):
    barostat, dynamics, state, _, physical = make(friction=0.3)
    initial, piston = state.copy(), barostat.snapshot()
    rng = copy.deepcopy(dynamics.rng.bit_generator.state)
    with pytest.raises(ValueError):
        barostat.step(state, dynamics, **options)
    np.testing.assert_array_equal(state.q, initial.q)
    np.testing.assert_array_equal(state.p, initial.p)
    assert barostat.snapshot() == piston
    assert dynamics.rng.bit_generator.state == rng
    assert physical.calls == 0


@pytest.mark.parametrize("changes", [{"integrator": "cayley-bcb"},
                                     {"thermostat": {"mode": "andersen", "interval": 3}},
                                     {"loop_backend": "native"}])
def test_other_dynamics_operators_are_not_silently_bypassed(changes):
    barostat, original, state, _, _ = make()
    options = {"integrator": "velocity-verlet", "native": False, "loop_backend": "python"}
    options.update(changes)
    dynamics = Dynamics(original.potential, original.masses, original.beta,
                        original.nbeads, original.dt, **options)
    with pytest.raises(ValueError):
        barostat.step(state, dynamics)


def test_legacy_composed_engine_cannot_bind_an_all_bead_piston():
    barostat, original, _, _, _ = make()
    with pytest.raises(TypeError, match="barostat"):
        Dynamics(original.potential, original.masses, original.beta,
                 original.nbeads, original.dt, integrator="velocity-verlet",
                 barostat=barostat, native=False, loop_backend="python")


def test_invalid_clock_and_mutated_parameters_fail_explicitly():
    barostat, dynamics, state, _, _ = make()
    state.step = True
    with pytest.raises(ValueError, match="clock"):
        barostat.step(state, dynamics)
    state.step = 0
    dynamics.dt *= 2
    with pytest.raises(ValueError, match="immutable"):
        barostat.step(state, dynamics)
