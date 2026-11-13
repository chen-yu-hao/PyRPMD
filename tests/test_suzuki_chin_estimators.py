"""SC physical estimators checked against independent actions and Gaussian integrals."""

import json

import numpy as np
import pytest

from pyrpmd.pimd_methods import SuzukiChinPotential
from pyrpmd.potentials.base import PESResult, PotentialError
from pyrpmd.potentials.builtin import FreePotential, HarmonicPotential
from pyrpmd.suzuki_chin import suzuki_chin_thermodynamics


class AnharmonicCellPotential:
    """Quartic Cartesian forces plus an explicit, position-dependent cell term."""

    def __init__(self, cell):
        self.set_cell(cell)

    def set_cell(self, cell):
        self.cell = np.array(cell, dtype=float, copy=True)

    def evaluate(self, q):
        volume = np.linalg.det(self.cell)
        r2 = np.sum(q**2, axis=(1, 2))
        a = 0.3 + 0.7 / volume
        energy = 0.5 * a * r2 + 0.04 * np.sum(q**4, axis=(1, 2)) + 0.9 / volume
        force = -a * q - 0.16 * q**3
        virial = np.einsum("bna,bnc->bac", force, q)
        virial += ((0.35 * r2 + 0.9) / volume)[:, None, None] * np.eye(3)
        return PESResult(energy, force, virial=virial)


@pytest.mark.parametrize("alpha", [0.0, 0.37, 1.0])
def test_sc_virial_is_full_affine_cell_derivative(alpha):
    rng = np.random.default_rng(71)
    cell = np.array([[2.3, 0.1, 0.0], [0.2, 2.7, 0.15], [0.1, 0.0, 3.1]])
    potential = SuzukiChinPotential(AnharmonicCellPotential(cell), [1.7, 2.3],
                                    3.2, 4, alpha=alpha, fd_step=1e-5)
    q = rng.normal(size=(4, 2, 3))
    initial = potential.evaluate(q)
    h = 1e-5
    expected = np.empty((4, 3, 3))
    for a in range(3):
        for b in range(3):
            deformation = np.eye(3)
            deformation[a, b] += h
            potential.set_cell(cell @ deformation.T)
            plus = potential.evaluate(q @ deformation.T).energy.copy()
            deformation[a, b] -= 2 * h
            potential.set_cell(cell @ deformation.T)
            minus = potential.evaluate(q @ deformation.T).energy.copy()
            expected[:, a, b] = -(plus - minus) / (2 * h)
    potential.set_cell(cell)
    np.testing.assert_allclose(initial.virial, expected, rtol=1e-7, atol=1e-8)
    # This provider has genuine cell dependence: Cartesian q tensor F is wrong.
    assert not np.allclose(initial.virial, np.einsum("bna,bnc->bac", initial.forces, q))


def test_pressure_matches_centroid_only_volume_derivative():
    cell = np.array([[3.0, 0.1, 0], [0, 3.3, 0.1], [0.05, 0, 3.7]])
    q = np.random.default_rng(12).normal(size=(6, 2, 3))
    potential = SuzukiChinPotential(AnharmonicCellPotential(cell), [1.2, 2.2], 2.4, 6,
                                    alpha=0.28, fd_step=2e-5)
    volume = np.linalg.det(cell)
    result = potential.estimators(q, volume=volume)
    center = q.mean(axis=0, keepdims=True)
    offsets = q - center
    h = 1e-5
    values = []
    for scale in (np.exp(h / 3), np.exp(-h / 3)):
        potential.set_cell(cell * scale)
        values.append(potential.evaluate(center * scale + offsets).energy.sum() / 6)
    potential.set_cell(cell)
    expected = 2 / (potential.beta * volume) - (values[0] - values[1]) / (2 * h * volume)
    assert result["pressure_au"] == pytest.approx(expected, rel=2e-8, abs=2e-10)


def _harmonic_precision(beta, p, mass, stiffness, alpha):
    """Direct Gaussian exponent, without using SC implementation parameters."""
    lap = 2 * np.eye(p) - np.roll(np.eye(p), 1, axis=0) - np.roll(np.eye(p), -1, axis=0)
    weighted = np.array([2 / 3 if j % 2 == 0 else 4 / 3 for j in range(p)])
    commutator = np.array([alpha / 9 if j % 2 == 0 else (1 - alpha) / 9 for j in range(p)])
    return (mass * p / beta * lap + np.diag(beta / p * weighted * stiffness
            + 2 * beta**3 / p**3 * commutator * stiffness**2 / mass))


def _harmonic_logz(beta, p, mass, stiffness, alpha):
    precision = _harmonic_precision(beta, p, mass, stiffness, alpha)
    return 1.5 * (p * np.log(mass * p / beta) - np.linalg.slogdet(precision)[1])


def _quadratic_gaussian_moments(potential, covariance):
    """Exactly integrate quadratic estimators using independent Gaussian moments.

    Reconstruct their quadratic forms in whitened coordinates; Isserlis gives
    Var(z.T A z)=2*Tr(A*A), with no stochastic/Monte Carlo tolerance.
    """
    p = potential.nbeads
    dimension = 3 * p
    pairs = [(i, j) for i in range(dimension) for j in range(i + 1, dimension)]
    z = np.zeros((1 + dimension + len(pairs), dimension))
    z[1:1 + dimension] = np.eye(dimension)
    for row, (i, j) in enumerate(pairs, 1 + dimension):
        z[row, i] = z[row, j] = 1
    q = (z @ np.linalg.cholesky(np.kron(covariance, np.eye(3))).T).reshape(-1, p, 1, 3)
    data = potential.estimators_batch(q)
    e, second = data["internal_energy_hartree"], data["scaled_action_second_beta_derivative"]
    a = np.diag(e[1:1 + dimension] - e[0])
    for row, (i, j) in enumerate(pairs, 1 + dimension):
        a[i, j] = a[j, i] = 0.5 * (e[row] - e[0] - a[i, i] - a[j, j])
    mean_e = e[0] + np.trace(a)
    var_e = 2 * np.sum(a * a)
    mean_second = second[0] + np.sum(second[1:1 + dimension] - second[0])
    return mean_e, 1.5 + potential.beta**2 * (var_e - mean_second), data


@pytest.mark.parametrize("alpha", [0.0, 0.43, 1.0])
def test_energy_and_heat_capacity_match_exact_finite_p_gaussian_partition(alpha):
    p, beta, mass, stiffness = 8, 4.1, 1.8, 0.9
    model = SuzukiChinPotential(HarmonicPotential(stiffness), [mass], beta, p, alpha=alpha)
    covariance = np.linalg.inv(_harmonic_precision(beta, p, mass, stiffness, alpha))
    mean_e, heat, _ = _quadratic_gaussian_moments(model, covariance)
    h = beta * 2e-4
    values = [_harmonic_logz(beta + delta * h, p, mass, stiffness, alpha) for delta in (-2, -1, 0, 1, 2)]
    reference_e = -(values[0] - 8 * values[1] + 8 * values[3] - values[4]) / (12 * h)
    reference_heat = beta**2 * (-values[0] + 16 * values[1] - 30 * values[2]
                              + 16 * values[3] - values[4]) / (12 * h**2)
    assert mean_e == pytest.approx(reference_e, rel=1e-8, abs=1e-9)
    assert heat == pytest.approx(reference_heat, rel=3e-6, abs=3e-6)


def test_energy_derivative_includes_explicit_beta_squared_force_correction():
    q = np.random.default_rng(32).normal(size=(4, 2, 3))
    cell = np.eye(3) * 4
    physical = AnharmonicCellPotential(cell)
    model = SuzukiChinPotential(physical, [1.2, 2.1], 3.4, 4, alpha=0.6, fd_step=1e-5)
    result = model.estimators(q)
    center = q.mean(axis=0)
    h = 3e-5 * model.beta
    values = []
    for beta in (model.beta - h, model.beta + h):
        other = SuzukiChinPotential(physical, model.masses, beta, 4, alpha=model.alpha)
        scaled = center + np.sqrt(beta / model.beta) * (q - center)
        values.append(beta / 4 * other.evaluate(scaled).energy.sum())
    expected = 3 / model.beta + (values[1] - values[0]) / (2 * h)
    assert result["internal_energy_hartree"] == pytest.approx(expected, rel=2e-8)
    assert result["internal_energy_hartree"] != pytest.approx(model.evaluate(q).energy.mean())
    assert result["potential_energy_hartree"] + result["kinetic_energy_hartree"] == pytest.approx(expected, rel=2e-8)
    assert "operator_internal_energy_hartree" not in result


def test_batch_scalar_consistency_and_heat_capacity_error_report():
    beta, p = 2.3, 4
    potential = SuzukiChinPotential(HarmonicPotential(0.7), [1.3], beta, p)
    covariance = np.linalg.inv(_harmonic_precision(beta, p, 1.3, 0.7, 0))
    paths = np.random.default_rng(8).normal(size=(800, p, 3))
    paths = np.einsum("ij,sjc->sic", np.linalg.cholesky(covariance), paths)[:, :, None]
    columns = potential.estimators_batch(paths)
    first = potential.estimators(paths[0])
    for key in first:
        assert columns[key][0] == pytest.approx(first[key], abs=1e-10)
    report = suzuki_chin_thermodynamics(columns, beta, block_size=40)
    expected = 1.5 + beta**2 * (columns["internal_energy_hartree"].var()
                               - columns["scaled_action_second_beta_derivative"].mean())
    assert report["heat_capacity_over_kB"] == pytest.approx(expected)
    assert report["heat_capacity_standard_error_over_kB"] > 0
    assert report["heat_capacity_kind"] == "Cv"
    json.dumps(report, allow_nan=False)
    small = suzuki_chin_thermodynamics([potential.estimators(q) for q in paths[:3]], beta, block_size=1)
    assert small["heat_capacity_standard_error_over_kB"] is None


def test_npt_heat_capacity_uses_external_pressure_and_enthalpy():
    beta, n = 2.0, 100
    model = SuzukiChinPotential(FreePotential(), [1.0], beta, 4)
    values = model.estimators_batch(np.zeros((n, 4, 1, 3)), volumes=8.0)
    np.testing.assert_allclose(values["pressure_au"], 1 / (beta * 8))
    volumes = np.linspace(7.0, 9.0, n)
    report = suzuki_chin_thermodynamics(values, beta, volumes=volumes, pressure_au=0.2,
                                      block_size=10)
    assert report["heat_capacity_kind"] == "Cp"
    assert report["heat_capacity_over_kB"] == pytest.approx(1.5 + beta**2 * (0.2 * volumes).var())
    assert report["enthalpy_hartree"] == pytest.approx(1.5 / beta + 0.2 * volumes.mean())
    with pytest.raises(ValueError, match="volumes"):
        suzuki_chin_thermodynamics(values, beta, pressure_au=0.2)
    with pytest.raises(ValueError, match="external pressure"):
        suzuki_chin_thermodynamics(values, beta, volumes=volumes)
    for pressure in (True, 1 + 0j, "0.2", np.nan, [0.2]):
        with pytest.raises(ValueError, match="pressure_au"):
            suzuki_chin_thermodynamics(values, beta, volumes=volumes, pressure_au=pressure)


def test_estimator_input_validation_and_absent_virial():
    potential = SuzukiChinPotential(HarmonicPotential(), [1.0], 2.0, 4)
    q = np.ones((4, 1, 3))
    with pytest.raises(PotentialError, match="virial"):
        potential.estimators(q, volume=5)
    optional = potential.estimators(q, volume=5, require_pressure=False)
    assert "pressure_au" not in optional
    assert optional["internal_energy_hartree"] == potential.estimators(q)["internal_energy_hartree"]
    for value in (1, "false", None, np.array([True])):
        with pytest.raises(ValueError, match="require_pressure"):
            potential.estimators(q, require_pressure=value)
    with pytest.raises(PotentialError, match="set_cell"):
        potential.set_cell(np.eye(3))
    for step in (0, -1, 1, np.nan, 1e-30):
        with pytest.raises(ValueError, match="finite_difference_step"):
            potential.estimators(q, finite_difference_step=step)
    for paths in (q, q[None, :-1], np.full((2, 4, 1, 3), np.nan)):
        with pytest.raises(ValueError, match="paths"):
            potential.estimators_batch(paths)


def test_physical_provider_reused_buffers_do_not_corrupt_sc_virial():
    class Reused(AnharmonicCellPotential):
        def evaluate(self, q):
            new = super().evaluate(q)
            if not hasattr(self, "last") or self.last.forces.shape != new.forces.shape:
                self.last = new
            else:
                self.last.energy[:], self.last.forces[:], self.last.virial[:] = new.energy, new.forces, new.virial
            return self.last
    q = np.random.default_rng(82).normal(size=(4, 2, 3))
    args = ([1.3, 2.4], 2.8, 4)
    actual = SuzukiChinPotential(Reused(np.eye(3) * 4), *args, alpha=0.5).evaluate(q)
    expected = SuzukiChinPotential(AnharmonicCellPotential(np.eye(3) * 4), *args, alpha=0.5).evaluate(q)
    np.testing.assert_allclose(actual.energy, expected.energy)
    np.testing.assert_allclose(actual.forces, expected.forces)
    np.testing.assert_allclose(actual.virial, expected.virial)
