"""Independent analytical and numerical checks of thermal instanton routines."""

import numpy as np
import pytest

from pyrpmd.constants import ATOMIC_TIME_S, BOHR_M, KB_AU
from pyrpmd.instanton import (
    InstantonError, RingPolymerAction, crossover_temperature, instanton_rate_1d,
    instanton_seed, optimize_instanton,
)
from pyrpmd.potentials.base import PESResult, Potential
from pyrpmd.potentials.builtin import HarmonicPotential


class Eckart(Potential):
    """V=V0 sech^2(a x); a is inverse width in bohr^-1."""

    def __init__(self, height=0.01, inverse_width=0.5, offset=0.0, asymmetry=0.0):
        self.height = height
        self.a = inverse_width
        self.offset = offset
        self.asymmetry = asymmetry

    def evaluate(self, q):
        x = q[:, 0, 0]
        t = np.tanh(self.a * x)
        v = self.height * (1 - t**2) + self.offset + self.asymmetry * t
        force = np.zeros_like(q)
        force[:, 0, 0] = self.a * (1 - t**2) * (2 * self.height * t - self.asymmetry)
        return PESResult(v, force)


MASK = np.array([[True, False, False]])
MASS = np.array([1822.0])
TS = np.zeros((1, 3))


def eckart_setup(nbeads=64, ratio=0.5, potential=None):
    potential = Eckart() if potential is None else potential
    tc = potential.a * np.sqrt(2 * potential.height / MASS[0]) / (2 * np.pi * KB_AU)
    temperature = ratio * tc
    path = instanton_seed(TS, [[1.0, 0.0, 0.0]], nbeads, 2.5)
    return potential, temperature, path


def exact_eckart_instanton(potential, temperature):
    """Continuum WKB stationary orbit: W(E)=2pi sqrt(2m)/a*(sqrt(V0)-sqrt(E))."""
    beta = 1 / (KB_AU * temperature)
    b = 2 * np.pi**2 * MASS[0] / potential.a**2
    return {
        "action": 2 * np.pi / potential.a * np.sqrt(2 * MASS[0] * potential.height) - b / beta,
        "energy": b / beta**2,
        "d_energy_d_beta": -2 * b / beta**3,
    }


def test_action_gradient_hessian_and_closed_ring_symmetry():
    rng = np.random.default_rng(6)
    q = rng.normal(scale=0.2, size=(8, 2, 3))
    mass = [1500.0, 2300.0]
    pes = HarmonicPotential(k=[0.03, 0.01, 0.02])
    action = RingPolymerAction(pes, mass, 250.0)
    eps = 1e-5
    gradient = action.evaluate(q)["gradient"]
    hessian = action.hessian(q)
    for index in [(0, 0, 0), (7, 1, 2), (3, 0, 1)]:
        plus, minus = q.copy(), q.copy()
        plus[index] += eps
        minus[index] -= eps
        ep, em = action.evaluate(plus), action.evaluate(minus)
        assert gradient[index] == pytest.approx((ep["action_hbar"] - em["action_hbar"]) / (2 * eps), abs=1e-8)
        column = np.ravel_multi_index(index, q.shape)
        np.testing.assert_allclose(hessian[:, column], (ep["gradient"] - em["gradient"]).ravel() / (2 * eps), atol=1e-8)
    assert action.evaluate(q)["action_hbar"] == pytest.approx(action.evaluate(np.roll(q, 3, axis=0))["action_hbar"])
    np.testing.assert_allclose(hessian, hessian.T, atol=1e-12)


def test_action_beta_derivative_is_imaginary_time_energy():
    pes, temperature, q = eckart_setup()
    beta = 1 / (KB_AU * temperature)
    step = 1e-5
    middle = RingPolymerAction(pes, MASS, temperature, active_mask=MASK).evaluate(q)
    plus = RingPolymerAction(pes, MASS, temperature / (1 + step), active_mask=MASK).evaluate(q)
    minus = RingPolymerAction(pes, MASS, temperature / (1 - step), active_mask=MASK).evaluate(q)
    assert middle["energy_hartree"] == pytest.approx((plus["action_hbar"] - minus["action_hbar"]) / (2 * step * beta), rel=1e-8)


def test_physical_hessian_handles_reused_force_buffer():
    class ReusingEckart(Eckart):
        def evaluate(self, q):
            value = super().evaluate(q)
            if not hasattr(self, "buffer") or self.buffer.shape != q.shape:
                self.buffer = np.empty_like(q)
            self.buffer[:] = value.forces
            return PESResult(value.energy, self.buffer)
    pes, temperature, q = eckart_setup()
    expected = RingPolymerAction(pes, MASS, temperature, active_mask=MASK).hessian(q)
    actual = RingPolymerAction(ReusingEckart(), MASS, temperature, active_mask=MASK).hessian(q)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_crossover_and_stationary_ts_guards():
    pes, temperature, _ = eckart_setup()
    assert crossover_temperature(pes, TS, MASS, active_mask=MASK) == pytest.approx(2 * temperature, rel=1e-8)
    assert crossover_temperature(pes, TS, 2 * MASS, active_mask=MASK) == pytest.approx(np.sqrt(2) * temperature, rel=1e-8)
    with pytest.raises(InstantonError, match="stationary"):
        crossover_temperature(pes, [[0.1, 0.0, 0.0]], MASS, active_mask=MASK)
    with pytest.raises(InstantonError, match="negative"):
        crossover_temperature(HarmonicPotential(k=1), TS, MASS, active_mask=MASK)


def test_eckart_bead_convergence_and_stationary_modes():
    errors = []
    phase_residuals = []
    for nbeads in (32, 64, 128):
        pes, temperature, q = eckart_setup(nbeads)
        result = optimize_instanton(pes, MASS, temperature, q, active_mask=MASK, transition_state=TS)
        analytic = exact_eckart_instanton(pes, temperature)
        errors.append(abs(result.action_hbar - analytic["action"]))
        phase_residuals.append(result.phase_mode_relative_residual)
        assert result.converged
        assert result.gradient_max < 1e-8
        assert result.negative_modes == 1
        assert result.energy_hartree == pytest.approx(analytic["energy"], rel=0.008)
        assert result.spread_bohr > 4.0
        assert result.to_dict()["physical_converged"] is False
    assert errors[0] / errors[1] > 3.9
    assert errors[1] / errors[2] > 3.9
    assert errors[2] < 0.0051
    assert phase_residuals[2] < phase_residuals[1] < phase_residuals[0]


def test_asymmetric_orbit_is_not_centroid_pinned():
    pes = Eckart(asymmetry=0.003)
    ts = np.array([[np.arctanh(0.15) / pes.a, 0.0, 0.0]])
    tc = crossover_temperature(pes, ts, MASS, active_mask=MASK)
    q = instanton_seed(ts, [[1.0, 0.0, 0.0]], 64, 2.0)
    result = optimize_instanton(pes, MASS, 0.65 * tc, q, active_mask=MASK, transition_state=ts)
    assert result.converged
    assert abs(np.mean(result.path_bohr[:, 0, 0]) - ts[0, 0]) > 0.1
    np.testing.assert_allclose(result.path_bohr, result.path_bohr[::-1], atol=1e-12)


def test_multidimensional_path_preserves_separable_action_but_is_not_a_rate():
    class EckartWithBath(Eckart):
        def evaluate(self, q):
            value = super().evaluate(q)
            value.energy += 0.005 * q[:, 0, 1]**2
            value.forces[:, 0, 1] = -0.01 * q[:, 0, 1]
            return value
    _, temperature, q = eckart_setup(32)
    active = np.array([[True, True, False]])
    result = optimize_instanton(EckartWithBath(), MASS, temperature, q, active_mask=active)
    expected = optimize_instanton(Eckart(), MASS, temperature, q, active_mask=MASK)
    assert result.converged
    assert result.action_hbar == pytest.approx(expected.action_hbar, abs=1e-10)
    assert result.to_dict()["scope"] == "stationary_action_path_not_a_multidimensional_rate"
    assert "rate" not in result.to_dict()


def test_one_dimensional_rate_matches_analytic_instanton_prefactor():
    pes, temperature, q = eckart_setup(128)
    beta = 1 / (KB_AU * temperature)
    partition = np.sqrt(MASS[0] / (2 * np.pi * beta))
    result = instanton_rate_1d(pes, MASS, temperature, q, active_mask=MASK,
                              reactant_partition=partition, normalization="per_bohr")
    analytic = exact_eckart_instanton(pes, temperature)
    prefactor = np.sqrt(-analytic["d_energy_d_beta"] / (2 * np.pi))
    expected = prefactor * np.exp(-analytic["action"]) / partition * BOHR_M / ATOMIC_TIME_S
    assert result["rate_unit"] == "m/s"
    assert result["rate"] == pytest.approx(expected, rel=0.006)
    assert result["d_energy_d_beta_hartree2"] == pytest.approx(analytic["d_energy_d_beta"], rel=0.001)
    assert result["beta_derivative_relative_difference"] < 0.0002
    assert result["physical_converged"] is False


def test_energy_reference_is_applied_consistently():
    pes, temperature, q = eckart_setup(32)
    reference = RingPolymerAction(pes, MASS, temperature, active_mask=MASK).evaluate(q)
    shifted = RingPolymerAction(Eckart(offset=0.004), MASS, temperature, active_mask=MASK,
                                energy_reference_hartree=0.004).evaluate(q)
    assert shifted["action_hbar"] == pytest.approx(reference["action_hbar"])
    assert shifted["energy_hartree"] == pytest.approx(reference["energy_hartree"])
    np.testing.assert_allclose(shifted["gradient"], reference["gradient"])


def test_failure_and_dimension_guards():
    pes, temperature, q = eckart_setup()
    with pytest.raises(InstantonError, match="below crossover"):
        optimize_instanton(pes, MASS, 3 * temperature, q, active_mask=MASK, transition_state=TS)
    with pytest.raises(InstantonError, match="collapsed"):
        optimize_instanton(pes, MASS, temperature, np.zeros_like(q), active_mask=MASK)
    with pytest.raises(ValueError, match="one active"):
        instanton_rate_1d(pes, MASS, temperature, q, active_mask=np.ones((1, 3), dtype=bool), reactant_partition=1.0)
    with pytest.raises(ValueError, match="positive"):
        instanton_rate_1d(pes, MASS, temperature, q, active_mask=MASK, reactant_partition=0.0)
    with pytest.raises(ValueError, match="normalization"):
        instanton_rate_1d(pes, MASS, temperature, q, active_mask=MASK, reactant_partition=1.0, normalization="cm3/s")
    malformed = q.copy()
    malformed[2, 0, 0] += 0.1
    with pytest.raises(ValueError, match="folded"):
        optimize_instanton(pes, MASS, temperature, malformed, active_mask=MASK)
    with pytest.raises(ValueError, match="even"):
        instanton_seed(TS, [[1, 0, 0]], 15, 1.0)
    with pytest.raises(ValueError, match="nonempty boolean"):
        RingPolymerAction(pes, MASS, temperature, active_mask=np.zeros((1, 3), dtype=bool)).evaluate(q)
    with pytest.raises(ValueError, match="identical"):
        malformed = q.copy()
        malformed[0, 0, 1] += 0.1
        RingPolymerAction(pes, MASS, temperature, active_mask=MASK).evaluate(malformed)
    with pytest.raises(ValueError, match="inactive coordinates"):
        optimize_instanton(pes, MASS, temperature, q, active_mask=MASK,
                          transition_state=[[0., 1., 0.]])


def test_unconverged_root_cannot_silently_become_a_rate():
    pes, temperature, q = eckart_setup()
    result = optimize_instanton(pes, MASS, temperature, q, active_mask=MASK,
                               max_iterations=1, raise_on_failure=False)
    assert not result.converged
    assert "residual" in result.message
    with pytest.raises(InstantonError, match="residual"):
        instanton_rate_1d(pes, MASS, temperature, q, active_mask=MASK,
                         reactant_partition=1.0, max_iterations=1)
