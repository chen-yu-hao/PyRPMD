"""Published molecular PES invariants, source identity and external-unit bridge."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pyrpmd.constants import BOHR_ANGSTROM, EV_TO_AU, AMU_TO_AU
from pyrpmd.dynamics import State
from pyrpmd.nonadiabatic import NRPMDState, NRPMDDynamics, validate_diabatic_result
from pyrpmd.potentials.base import PotentialError, BackendUnavailableError
from pyrpmd.potentials.diabatic import check_diabatic_potential
from pyrpmd.potentials.multistate_external import (
    NaH2LEPS5G, molecular_potential_from_dict, ChemPotPyDPEM,
)
from pyrpmd.surface_hopping import RPSHDynamics


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT/"benchmarks/molecular_nonadiabatic/vendor/chempotpy"


def geometries():
    return np.array([
        [[4.0, .5, .1], [-.7, 0., 0.], [.7, 0., 0.]],
        [[2.4, .3, .2], [-1., 0., 0.], [1., 0., 0.]],
        [[1.3, 3.6, -.2], [-.9, .1, .3], [.9, -.1, -.3]],
        [[7.1, -.5, 2.0], [-1.4, .1, 0.], [1.4, -.1, 0.]],
    ])


def test_matrix_contract_gradient_finite_difference_and_translation_force_sum():
    potential = NaH2LEPS5G()
    q = geometries()
    original = q.copy()
    result = validate_diabatic_result(potential.evaluate(q), q.shape, 2)
    assert result.matrix.shape == (4, 2, 2)
    assert result.gradients.shape == (4, 2, 2, 3, 3)
    np.testing.assert_array_equal(q, original)
    np.testing.assert_array_equal(result.matrix, result.matrix.swapaxes(1, 2))
    np.testing.assert_array_equal(result.gradients, result.gradients.swapaxes(1, 2))
    np.testing.assert_allclose(result.gradients.sum(axis=-2), 0, rtol=0, atol=2e-15)
    torque = np.cross(q[:, None, None], result.gradients).sum(axis=-2)
    np.testing.assert_allclose(torque, 0, rtol=0, atol=5e-15)
    error = check_diabatic_potential(potential, q, 2, epsilon=1e-5)
    assert error["max_absolute_gradient_error"] < 2e-8


def test_published_asymptote_and_original_zero_shift_are_preserved():
    q = np.array([[[1e4, 0., 0.], [-1.402721684/2, 0., 0.], [1.402721684/2, 0., 0.]]])
    result = NaH2LEPS5G().evaluate(q)
    assert abs(result.matrix[0, 0, 0]) < 5e-12
    assert result.matrix[0, 0, 1] == 0.
    assert result.matrix[0, 1, 1]-result.matrix[0, 0, 0] == pytest.approx(
        2.1037/27.2113961, rel=2e-10, abs=1e-12)


def test_all_atom_permutations_restore_gradients_and_hydrogen_symmetry():
    import itertools
    q = geometries()
    reference = NaH2LEPS5G().evaluate(q)
    symbols = ("Na", "H", "H")
    for order in itertools.permutations(range(3)):
        potential = NaH2LEPS5G(symbols=tuple(symbols[i] for i in order))
        result = potential.evaluate(q[:, order])
        np.testing.assert_allclose(result.matrix, reference.matrix, rtol=0, atol=2e-15)
        np.testing.assert_allclose(result.gradients, reference.gradients[..., order, :], rtol=0, atol=3e-14)


def test_rigid_rotation_and_translation_transform_covariantly():
    rotation, _ = np.linalg.qr(np.array([[1., 2., .3], [-.2, .5, 1.], [.7, -.8, .6]]))
    q = geometries()
    potential = NaH2LEPS5G()
    result = potential.evaluate(q)
    moved = potential.evaluate(q@rotation+np.array([9.2, -8.3, 5.1]))
    np.testing.assert_allclose(moved.matrix, result.matrix, rtol=0, atol=3e-14)
    np.testing.assert_allclose(moved.gradients, result.gradients@rotation, rtol=0, atol=3e-13)


def test_batching_is_frame_independent_and_metadata_has_real_symbols():
    potential = NaH2LEPS5G()
    q = geometries()
    result = potential.evaluate(q)
    separate = [potential.evaluate(frame[None]) for frame in q]
    np.testing.assert_array_equal(result.matrix, np.concatenate([r.matrix for r in separate]))
    np.testing.assert_array_equal(result.gradients, np.concatenate([r.gradients for r in separate]))
    assert potential.metadata["trajectory_batch_safe"] is True
    assert potential.metadata["thread_safety"] == "reentrant"
    assert potential.metadata["electronic_basis"] == "fixed_diabatic"
    assert potential.metadata["charge"] == 0
    assert potential.metadata["multiplicity"] == 2
    assert potential.symbols == ("Na", "H", "H")
    assert potential.coordinates == tuple(range(9))


def test_pinned_original_fortran_and_distributed_license_match_exact_bytes():
    source = NaH2LEPS5G().to_dict()["source"]
    data = (VENDOR/"NaH2_LEPS_5G_1992_DPEM.f").read_bytes()
    assert len(data) == 15875
    assert hashlib.sha256(data).hexdigest() == source["sha256"]
    # Git blob SHA pins the same bytes independently of the SHA-256 manifest.
    assert hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest() == "fc9a0c3cf6af0707303508fabc11b7dbb02c3877"
    license_bytes = (VENDOR/"LICENSE").read_bytes()
    assert hashlib.sha256(license_bytes).hexdigest() == "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
    assert (ROOT/"src/pyrpmd/potentials/_licenses/chempotpy-APACHE-2.0.txt").read_bytes() == license_bytes


def test_identity_roundtrip_and_parameter_immutability():
    potential = NaH2LEPS5G(symbols=["H", "Na", "H"])
    encoded = json.loads(json.dumps(potential.to_dict(), allow_nan=False))
    rebuilt = molecular_potential_from_dict(encoded)
    assert rebuilt.to_dict() == encoded
    q = geometries()[:, [1, 0, 2]]
    np.testing.assert_array_equal(rebuilt.evaluate(q).matrix, potential.evaluate(q).matrix)
    encoded["parameters"]["symbols"][0] = "O"
    metadata = potential.metadata
    metadata["model_identity"]["source"]["sha256"] = "changed"
    assert potential.symbols == ("H", "Na", "H")
    assert potential.to_dict()["source"]["sha256"] != "changed"
    with pytest.raises(AttributeError):
        potential.symbols = ("Na", "H", "H")


@pytest.mark.parametrize("mutation", [
    lambda x: x.update(extra=True), lambda x: x.pop("source"),
    lambda x: x.update(version=True), lambda x: x.update(version=2),
    lambda x: x.update(schema="arbitrary"), lambda x: x.update(model="custom"),
    lambda x: x["parameters"].update(scale=2), lambda x: x["parameters"].update(symbols=["Na", "H", "O"]),
    lambda x: x["source"].update(sha256="0"*64), lambda x: x["source"].pop("translation"),
])
def test_identity_restoration_rejects_changed_physics_or_provenance(mutation):
    value = NaH2LEPS5G().to_dict()
    mutation(value)
    with pytest.raises(ValueError):
        molecular_potential_from_dict(value)


@pytest.mark.parametrize("symbols", ["NaHH", ["Na", "H"], ["H", "H", "H"], ["Na", "Na", "H"], ["Na", "H", True]])
def test_invalid_atom_composition_fails(symbols):
    with pytest.raises(ValueError):
        NaH2LEPS5G(symbols=symbols)


@pytest.mark.parametrize("q", [
    np.zeros((1, 3, 3)), np.ones((1, 2, 3)), np.ones((3, 3)),
    np.ones((1, 3, 3), complex), np.full((1, 3, 3), np.nan),
    np.array([[[0., 0., 0.], [-.7, 0., 0.], [.7, 0., 0.]]]),
])
def test_bad_or_singular_geometries_raise_python_errors_instead_of_fortran_stop(q):
    with pytest.raises(PotentialError):
        NaH2LEPS5G().evaluate(q)


@pytest.mark.parametrize("method", ["nrpmd", "rpsh"])
def test_real_molecular_matrix_propagates_with_both_existing_methods(method):
    potential = NaH2LEPS5G()
    masses = np.array([22.98976928, 1.00782503223, 1.00782503223])*AMU_TO_AU
    q = np.repeat(geometries()[0:1], 3, axis=0)
    nuclear = State(q, np.zeros_like(q))
    if method == "nrpmd":
        state = NRPMDState.from_nuclear(nuclear, 2, 1, np.random.default_rng(71))
        dynamics = NRPMDDynamics(potential, masses, 1000., 3, .1, 2, native=False)
    else:
        dynamics = RPSHDynamics(potential, masses, 1000., 3, .1, 2, seed=71, native=False)
        state = dynamics.initialize(nuclear, active_surface=1)
    initial_energy = dynamics.energy(state)
    dynamics.step(state)
    dynamics.step(state)
    assert state.step == 2
    assert np.any(state.q != q)
    assert abs(dynamics.energy(state)-initial_energy) < 2e-8


def _mock_chempotpy(tmp_path, monkeypatch):
    """Synthetic polynomial exercises adapter units, not a molecular reference."""
    package = tmp_path/"chempotpy"
    (package/"NaH2").mkdir(parents=True)
    wrapper = package/"__init__.py"
    wrapper.write_text("# unit-conversion fixture, not a physical PES\n")
    (package/"NaH2/NaH2_LEPS_5G_1992_DPEM.f").write_bytes((VENDOR/"NaH2_LEPS_5G_1992_DPEM.f").read_bytes())
    seen = []

    def u(system, surface, geom):
        assert system == "NaH2" and surface == "NaH2_LEPS_5G_1992_DPEM"
        seen.append(deepcopy(geom))
        x = np.array([row[1:] for row in geom])
        common = 2*np.sum(x*x)
        coupling = .3*x[0, 0]+.2j*x[2, 1]
        return np.array([[common, coupling], [coupling.conjugate(), 1+common]])

    api = SimpleNamespace(__file__=str(wrapper), u=u)
    monkeypatch.setitem(sys.modules, "chempotpy", api)
    return api, seen, package


def test_optional_adapter_unit_chain_rule_complex_coherences_and_atom_order(tmp_path, monkeypatch):
    api, seen, _ = _mock_chempotpy(tmp_path, monkeypatch)
    symbols = ["H", "Na", "H"]
    adapter = ChemPotPyDPEM(symbols=symbols, difference_step_bohr=2e-4)
    q = geometries()[0:1, [1, 0, 2]]
    result = adapter.evaluate(q)
    assert len(seen) == 37
    assert [row[0] for row in seen[0]] == symbols
    np.testing.assert_allclose([row[1:] for row in seen[0]], q[0]*BOHR_ANGSTROM, rtol=0, atol=0)
    expected = 4*q[0]*BOHR_ANGSTROM**2*EV_TO_AU
    np.testing.assert_allclose(result.gradients[0, 0, 0], expected, rtol=0, atol=2e-11)
    assert result.gradients[0, 0, 1, 0, 0] == pytest.approx(.3*BOHR_ANGSTROM*EV_TO_AU, abs=2e-12)
    assert result.gradients[0, 0, 1, 2, 1] == pytest.approx(.2j*BOHR_ANGSTROM*EV_TO_AU, abs=2e-12)
    assert adapter.metadata["source_identity"]["difference_step_bohr"] == 2e-4
    assert adapter.metadata["electronic_basis"] == "fixed_diabatic"


def test_optional_adapter_rejects_wrong_source_and_bad_backend_matrix(tmp_path, monkeypatch):
    api, _, package = _mock_chempotpy(tmp_path, monkeypatch)
    adapter = ChemPotPyDPEM()
    api.u = lambda *args: np.array([0., 1.])  # adiabatic energies are not a DPEM.
    with pytest.raises(PotentialError, match="complete finite 2x2"):
        adapter.evaluate(geometries()[0:1])
    (package/"NaH2/NaH2_LEPS_5G_1992_DPEM.f").write_text("unverified source")
    with pytest.raises(BackendUnavailableError, match="pinned hash"):
        ChemPotPyDPEM()


@pytest.mark.parametrize("step", [True, 0., -1., np.inf, 1j, "1e-4"])
def test_optional_adapter_difference_step_must_be_explicit_positive_real(step):
    with pytest.raises(ValueError):
        ChemPotPyDPEM(difference_step_bohr=step)
