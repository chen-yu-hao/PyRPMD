"""Diabatic bridge checks independent of the golden-rule path optimizer."""
import numpy as np
import pytest

from pyrpmd.golden_rule_adapters import diabatic_surfaces
from pyrpmd.nonadiabatic import DiabaticResult
from pyrpmd.potentials.base import PotentialError, check_potential
from pyrpmd.potentials.diabatic import DiabaticPythonPotential
from pyrpmd.potentials.nonadiabatic_models import DiscreteSpinBoson


def model(coupling=2e-5):
    return DiscreteSpinBoson([0.01, 0.02], [0.02, -0.04],
                             masses=[2.0, 3.0], bias=0.001, tunnelling=coupling)


def test_existing_matrix_contract_to_scalar_forces_and_identity():
    matrix = model()
    surfaces = diabatic_surfaces(matrix, coupling_hartree=2e-5)
    q = np.arange(18, dtype=float).reshape(3, 2, 3) / 10
    direct = matrix.evaluate(q)
    for state, surface in enumerate(surfaces):
        result = surface.evaluate(q)
        np.testing.assert_allclose(result.energy, direct.matrix[:, state, state])
        np.testing.assert_allclose(result.forces, -direct.gradients[:, state, state])
        assert check_potential(surface, q)["max_absolute_force_error"] < 1e-10
        assert surface.metadata["matrix_provider"]["metadata"]["model_identity"] == matrix.metadata["model_identity"]
        assert surface.metadata["matrix_provider"]["source_sha256"]["evaluate"]
        assert surface.diabatic_state == state
        assert surface.coupling_hartree == 2e-5
        assert not hasattr(surface, "hessian")
        with pytest.raises(AttributeError):
            surface.coupling_hartree = 9e-5


@pytest.mark.parametrize("options", [
    {"coupling_hartree": 0}, {"coupling_hartree": True},
    {"coupling_hartree": 1j}, {"coupling_hartree": np.inf},
    {"coupling_hartree": 2e-5, "coupling_rtol": 1},
    {"coupling_hartree": 2e-5, "coupling_atol_hartree": 2e-5},
    {"coupling_hartree": 2e-5, "coupling_rtol": -1},
    {"coupling_hartree": 2e-5, "coupling_gradient_tolerance": -1},
])
def test_invalid_condon_declaration_precedes_provider_work(options):
    class Uncalled:
        def evaluate(self, q):
            pytest.fail("invalid declaration must not evaluate the provider")
    with pytest.raises(ValueError):
        diabatic_surfaces(Uncalled(), **options)


@pytest.mark.parametrize("coupling", [0, 1e-5, 2e-5j])
def test_actual_coupling_must_match_declared_real_value(coupling):
    surface, _ = diabatic_surfaces(model(coupling), coupling_hartree=2e-5)
    with pytest.raises(PotentialError, match="constant Condon"):
        surface.evaluate(np.zeros((2, 2, 3)))


def test_noncondon_gradient_is_rejected_even_when_point_value_matches():
    class VariableCoupling:
        def evaluate(self, q):
            n = len(q)
            h = np.zeros((n, 2, 2))
            g = np.zeros((n, 2, 2, 1, 3))
            h[:, 0, 1] = h[:, 1, 0] = 2e-5 + 1e-4*q[:, 0, 0]
            g[:, 0, 1, 0, 0] = g[:, 1, 0, 0, 0] = 1e-4
            return DiabaticResult(h, g)
    surface, _ = diabatic_surfaces(VariableCoupling(), coupling_hartree=2e-5)
    with pytest.raises(PotentialError, match="zero coupling gradient"):
        surface.evaluate(np.zeros((2, 1, 3)))


@pytest.mark.parametrize("vary_gradient", [False, True])
def test_hermitian_projection_cannot_hide_stricter_condon_violations(vary_gradient):
    class NearlyHermitian:
        def evaluate(self, q):
            h = np.zeros((len(q), 2, 2))
            g = np.zeros((len(q), 2, 2, 1, 3))
            h[:, 0, 1] = h[:, 1, 0] = 2e-5
            if vary_gradient:
                g[:, 0, 1, 0, 0] = 4e-13
                g[:, 1, 0, 0, 0] = -4e-13
            else:
                h[:, 0, 1] += 4e-13
                h[:, 1, 0] -= 4e-13
            return (h, g) if vary_gradient else DiabaticResult(h, g)
    surface, _ = diabatic_surfaces(NearlyHermitian(), coupling_hartree=2e-5)
    message = "zero coupling gradient" if vary_gradient else "constant Condon"
    with pytest.raises(PotentialError, match=message):
        surface.evaluate(np.zeros((2, 1, 3)))


def test_borrowed_provider_and_callback_metadata():
    class Reusable:
        def __init__(self):
            self.closed = False
            self.h = np.zeros((2, 2, 2))
            self.g = np.zeros((2, 2, 2, 1, 3))
        def evaluate(self, q):
            self.h[:, 0, 0] = q[:, 0, 0]**2
            self.h[:, 1, 1] = (q[:, 0, 0]-1)**2
            self.h[:, 0, 1] = self.h[:, 1, 0] = 2e-5
            self.g[:, 0, 0, 0, 0] = 2*q[:, 0, 0]
            self.g[:, 1, 1, 0, 0] = 2*(q[:, 0, 0]-1)
            return self.h, self.g
        def close(self):
            self.closed = True
    owner = Reusable()
    provider = DiabaticPythonPotential(owner)
    surface, other = diabatic_surfaces(provider, coupling_hartree=2e-5)
    q = np.zeros((2, 1, 3))
    saved = surface.evaluate(q)
    other.evaluate(q+3)
    np.testing.assert_array_equal(saved.energy, [0, 0])
    np.testing.assert_array_equal(saved.forces, np.zeros_like(q))
    assert surface.metadata["matrix_provider"]["source_sha256"]["fn"]
    surface.close()
    other.close()
    assert not owner.closed


def test_exact_two_state_contract_and_provider_failure_are_preserved():
    class ThreeState:
        def evaluate(self, q):
            return np.zeros((len(q), 3, 3)), np.zeros((len(q), 3, 3, *q.shape[1:]))
    surface, _ = diabatic_surfaces(ThreeState(), coupling_hartree=2e-5)
    with pytest.raises(PotentialError, match="nstates mismatch"):
        surface.evaluate(np.zeros((1, 1, 3)))
    class Failed:
        def evaluate(self, q):
            return DiabaticResult(np.zeros((1, 2, 2)), np.zeros((1, 2, 2, 1, 3)),
                                  status=np.array([1]), messages=("provider failed",))
    surface, _ = diabatic_surfaces(Failed(), coupling_hartree=2e-5)
    with pytest.raises(PotentialError, match="frames failed"):
        surface.evaluate(np.zeros((1, 1, 3)))
