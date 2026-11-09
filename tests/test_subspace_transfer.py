"""Free-energy environment projection and harmonic bead factor (subspace transfer theory)."""

import numpy as np
import pytest

from pyrpmd.analysis import harmonic_bead_factor, project_linear_subspace, project_path_subspace
from pyrpmd.analysis.subspace_correction import SubspaceCorrection, apply_subspace_correction

MASS = 1060.0


def _omega(s):
    return 0.01 * (1.0 - 0.3 / np.cosh(s / 0.66) ** 2)


def potential(x):
    s, y = x
    return 0.0156 / np.cosh(s / 0.66) ** 2 + 0.5 * MASS * _omega(s) ** 2 * y**2


def gradient(x):
    s, y = x
    sech2 = 1.0 / np.cosh(s / 0.66) ** 2
    dsech2 = -2.0 / 0.66 * np.tanh(s / 0.66) * sech2
    w = _omega(s)
    dw = -0.01 * 0.3 * dsech2
    return np.array([0.0156 * dsech2 + MASS * w * dw * y**2, MASS * w**2 * y])


def _free_energy(omega, beta):
    return np.log(2.0 * np.sinh(0.5 * beta * omega)) / beta


@pytest.mark.parametrize("beta", [400.0, 2000.0])
def test_linear_free_energy_projection_matches_harmonic_formula(beta):
    axis = np.sqrt(MASS) * np.linspace(-3.0, 3.0, 13)
    kwargs = dict(dividing_surface=0.0, reactant_reference=float(axis[1]))
    fe = project_linear_subspace(potential, gradient, [0.0, 0.0], [MASS, MASS], [1.0, 0.0], [axis],
                                 environment="free_energy", beta=beta, **kwargs)
    zpe = project_linear_subspace(potential, gradient, [0.0, 0.0], [MASS, MASS], [1.0, 0.0], [axis],
                                  environment="adiabatic", **kwargs)
    s = axis / np.sqrt(MASS)
    bare = 0.0156 / np.cosh(s / 0.66) ** 2
    assert np.allclose(fe.values, bare + _free_energy(_omega(s), beta), atol=2e-8)
    assert np.allclose(zpe.values, bare + 0.5 * _omega(s), atol=2e-8)
    assert fe.provenance["beta"] == beta
    # beta * omega = 4..20: the thermal term lowers the potential most where omega is smallest.
    assert np.all(fe.values <= zpe.values + 1e-12)


def test_free_energy_reduces_to_zero_point_energy_at_low_temperature():
    axis = np.sqrt(MASS) * np.linspace(-2.0, 2.0, 9)
    path = np.column_stack([np.linspace(-2.0, 2.0, 21), np.zeros(21)])
    arc = np.linspace(0.0, 4.0 * np.sqrt(MASS), 9)
    fe = project_path_subspace(potential, gradient, path, [MASS, MASS], arc, environment="free_energy", beta=1e5)
    zpe = project_path_subspace(potential, gradient, path, [MASS, MASS], arc, environment="adiabatic")
    assert np.allclose(fe.values, zpe.values, atol=1e-10)
    with pytest.raises(ValueError, match="beta"):
        project_linear_subspace(potential, gradient, [0.0, 0.0], [MASS, MASS], [1.0, 0.0], [axis],
                                environment="free_energy")


def _ring_polymer_partition(u, nbeads):
    n = np.arange(nbeads)
    return np.prod(((u / nbeads) ** 2 + 4.0 * np.sin(np.pi * n / nbeads) ** 2) ** -0.5)


@pytest.mark.parametrize("nbeads", [1, 4, 32])
def test_harmonic_bead_factor_matches_normal_mode_product(nbeads):
    beta, ts, reactant = 1500.0, [0.009, 0.004], [0.019]
    exact = 1.0
    for omega in ts:
        u = beta * omega
        exact *= _ring_polymer_partition(u, nbeads) * 2.0 * np.sinh(0.5 * u)
    for omega in reactant:
        u = beta * omega
        exact /= _ring_polymer_partition(u, nbeads) * 2.0 * np.sinh(0.5 * u)
    result = harmonic_bead_factor(ts, reactant, beta, nbeads)
    assert result["factor"] == pytest.approx(exact, rel=1e-10)
    if nbeads == 1:  # classical ring polymer: ratio of classical to quantum partition functions
        classical = np.prod([1 / (beta * w) * 2 * np.sinh(0.5 * beta * w) for w in ts])
        classical /= np.prod([1 / (beta * w) * 2 * np.sinh(0.5 * beta * w) for w in reactant])
        assert result["factor"] == pytest.approx(classical, rel=1e-10)


def test_harmonic_bead_factor_limits_and_application():
    assert harmonic_bead_factor([0.01], [0.02], 300.0, 100000)["factor"] == pytest.approx(1.0, abs=1e-6)
    assert harmonic_bead_factor([0.01], [0.01], 300.0, 8)["factor"] == pytest.approx(1.0, abs=1e-12)
    # A stiffer reactant mode carries a larger finite-P error, so RPMD(P) underestimates.
    assert harmonic_bead_factor([0.01], [0.02], 1500.0, 16)["factor"] < 1.0
    with pytest.raises(ValueError):
        harmonic_bead_factor([-0.01], [0.02], 300.0, 8)
    correction = SubspaceCorrection(300.0, 1, 2.0, 0.05, 2.0, 1.0, 8, {}, None, None, {})
    applied = apply_subspace_correction(1.0, correction, bead_factor=0.5, standard_error=0.1)
    assert applied["corrected_rate"] == pytest.approx(4.0)
    assert applied["bead_factor"] == 0.5
