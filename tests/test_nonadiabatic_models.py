"""Independent model identities, derivatives, coordinate and restart contracts."""

import copy
import json

import numpy as np
import pytest

from pyrpmd.nonadiabatic import DiabaticResult, validate_diabatic_result
from pyrpmd.potentials.base import PotentialError
from pyrpmd.potentials.nonadiabatic_models import (
    DiscreteSpinBoson,
    LinearVibronicCoupling,
    TullyModelOne,
    TullyModelThree,
    TullyModelTwo,
    model_from_dict,
)


def finite_difference(model, q, epsilon=2e-6):
    result = model.evaluate(q)
    numerical = np.zeros_like(result.gradients)
    for atom in range(model.natoms):
        for axis in range(3):
            plus, minus = q.copy(), q.copy()
            plus[:, atom, axis] += epsilon
            minus[:, atom, axis] -= epsilon
            numerical[..., atom, axis] = (
                model.evaluate(plus).matrix - model.evaluate(minus).matrix) / (2 * epsilon)
    return numerical


@pytest.mark.parametrize("factory,center,derivative", [
    (TullyModelOne, [[0., .005], [.005, 0.]], [[.016, 0.], [0., -.016]]),
    (TullyModelTwo, [[0., .015], [.015, -.05]], [[0., 0.], [0., 0.]]),
    (TullyModelThree, [[.0006, .10], [.10, -.0006]], [[0., .09], [.09, 0.]]),
])
def test_tully_published_values_at_crossing(factory, center, derivative):
    # Tully (1990), Eqs. 21/23/24; the second model has V11=-A+E0.
    model = factory()
    result = model.evaluate(np.zeros((1, 1, 3)))
    assert isinstance(result, DiabaticResult)
    np.testing.assert_allclose(result.matrix[0], center, rtol=0, atol=1e-17)
    np.testing.assert_allclose(result.gradients[0, :, :, 0, 0], derivative, rtol=0, atol=2e-17)
    np.testing.assert_array_equal(result.gradients[..., 1:], 0.)


@pytest.mark.parametrize("factory", [TullyModelOne, TullyModelTwo, TullyModelThree])
def test_tully_all_elements_all_cartesian_derivatives(factory):
    # Include each side of the piecewise boundary and its continuous first
    # derivative at zero. The second derivative is not continuous there.
    model = factory(natoms=2, coordinate=4)
    q = np.random.default_rng(41).normal(size=(5, 2, 3))
    q[:, 1, 1] = [-1.7, -.3, 0., .4, 2.1]
    result = validate_diabatic_result(model.evaluate(q), q.shape, 2)
    np.testing.assert_allclose(result.gradients, finite_difference(model, q),
                               rtol=3e-6, atol=3e-11)
    shifted = q + 10.
    shifted[:, 1, 1] = q[:, 1, 1]
    np.testing.assert_array_equal(model.evaluate(shifted).matrix, result.matrix)
    inactive = result.gradients.reshape(5, 2, 2, -1)[..., [0, 1, 2, 3, 5]]
    np.testing.assert_array_equal(inactive, 0.)


def test_tully_asymptotes_and_extended_nonvanishing_coupling():
    q = np.zeros((2, 1, 3))
    q[:, 0, 0] = [-1000., 1000.]
    first, second, third = [cls().evaluate(q) for cls in
                            (TullyModelOne, TullyModelTwo, TullyModelThree)]
    np.testing.assert_allclose(first.matrix[:, 0, 0], [-.01, .01], atol=0, rtol=0)
    np.testing.assert_array_equal(first.matrix[:, 0, 1], 0.)
    np.testing.assert_allclose(second.matrix[:, 1, 1], [.05, .05], atol=0, rtol=0)
    np.testing.assert_array_equal(second.matrix[:, 0, 1], 0.)
    np.testing.assert_allclose(third.matrix[:, 0, 1], [0., .2], atol=0, rtol=0)
    for result in (first, second, third):
        np.testing.assert_array_equal(result.gradients, 0.)


def test_tully_parameters_inferred_atoms_and_decoupled_limit():
    model = TullyModelTwo(a=.4, b=.5, c=0., d=.9, e0=-.2, coordinate=7, natoms=None)
    assert model.natoms == 3
    q = np.zeros((1, 3, 3))
    q[0, 2, 1] = 2.
    result = model.evaluate(q)
    assert result.matrix[0, 1, 1] == pytest.approx(-.2 - .4 * np.exp(-2))
    assert result.gradients[0, 1, 1, 2, 1] == pytest.approx(.8 * np.exp(-2))
    np.testing.assert_array_equal(result.matrix[:, 0, 1], 0.)
    np.testing.assert_array_equal(result.gradients[:, 0, 1], 0.)


def complex_lvc():
    # Three states and three nonadjacent coordinates exercise complex coupling
    # derivatives independently of the two-state avoided-crossing formulas.
    rng = np.random.default_rng(117)
    h = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    h = (h + h.conj().T) / 2
    g = rng.normal(size=(3, 3, 3)) + 1j * rng.normal(size=(3, 3, 3))
    g = (g + g.swapaxes(1, 2).conj()) / 2
    return LinearVibronicCoupling(h, g, harmonic=[.3, .8, 1.4], reference=[.2, -.5, 1.],
                                 coordinates=[5, 0, 3])


def test_lvc_full_hamiltonian_and_all_complex_gradients():
    model = complex_lvc()
    assert model.nstates == 3 and model.natoms == 2
    q = np.random.default_rng(11).normal(size=(4, 2, 3))
    result = validate_diabatic_result(model.evaluate(q), q.shape, model.nstates)
    np.testing.assert_allclose(result.gradients, finite_difference(model, q),
                               rtol=2e-9, atol=8e-10)
    expected = np.repeat(model.h0[None], len(q), axis=0)
    for mode, index in enumerate(model.coordinates):
        atom, axis = divmod(index, 3)
        displacement = q[:, atom, axis] - model.reference[mode]
        expected += displacement[:, None, None] * model.linear[mode]
        for state in range(model.nstates):
            expected[:, state, state] += .5 * model.harmonic[mode] * displacement**2
    np.testing.assert_allclose(result.matrix, expected, atol=2e-15, rtol=3e-16)
    np.testing.assert_array_equal(result.matrix, result.matrix.swapaxes(1, 2).conj())
    np.testing.assert_array_equal(result.gradients, result.gradients.swapaxes(1, 2).conj())
    assert np.max(np.abs(result.gradients.imag)) > .1


def test_lvc_constant_unitary_covariance():
    model = complex_lvc()
    raw = np.random.default_rng(142).normal(size=(3, 3)) + 1j * np.eye(3)
    unitary, _ = np.linalg.qr(raw)
    rotated = LinearVibronicCoupling(
        unitary.conj().T @ model.h0 @ unitary,
        np.array([unitary.conj().T @ g @ unitary for g in model.linear]),
        harmonic=model.harmonic, reference=model.reference, coordinates=model.coordinates)
    q = np.random.default_rng(7).normal(size=(2, 2, 3))
    original, transformed = model.evaluate(q), rotated.evaluate(q)
    np.testing.assert_allclose(transformed.matrix, unitary.conj().T @ original.matrix @ unitary,
                               atol=4e-15, rtol=3e-15)
    for atom in range(model.natoms):
        for axis in range(3):
            expected = unitary.conj().T @ original.gradients[..., atom, axis] @ unitary
            np.testing.assert_allclose(transformed.gradients[..., atom, axis], expected,
                                       atol=4e-15, rtol=3e-15)


def test_lvc_one_state_scalar_limit_and_reference_translation():
    model = LinearVibronicCoupling([[.7]], [[[.4]], [[-.8]]],
                                 harmonic=[2., 3.], reference=[1., -2.])
    assert model.natoms == 1 and model.coordinates == (0, 1)
    q = np.array([[[1.5, -.2, 99.]], [[.2, -3., -42.]]])
    displacement = q[:, 0, :2] - [1., -2.]
    result = model.evaluate(q)
    expected = .7 + displacement @ [.4, -.8] + np.sum(displacement**2 * [1., 1.5], axis=1)
    np.testing.assert_allclose(result.matrix[:, 0, 0], expected)
    np.testing.assert_allclose(result.gradients[:, 0, 0, 0, :2],
                               displacement * [2., 3.] + [.4, -.8])
    translated = LinearVibronicCoupling(model.h0, model.linear,
                                      harmonic=model.harmonic, reference=[6., 3.])
    q[:, 0, :2] += 5.
    np.testing.assert_allclose(translated.evaluate(q).matrix, result.matrix)


def test_spin_boson_physical_mass_convention_and_complex_tunnelling():
    model = DiscreteSpinBoson([.3, .8], [.07, -.2], masses=[2., 5.],
                             bias=.11, tunnelling=.04 + .02j, reference=[.1, -.3])
    assert model.coordinates == (0, 3) and model.natoms == 2
    np.testing.assert_array_equal(model.nuclear_masses, [2., 5.])
    q = np.array([[[.2, 2., 3.], [.4, 5., 6.]], [[-.7, 0., 0.], [1.2, 0., 0.]]])
    r = model.evaluate(q)
    displacement = q[:, :, 0] - [.1, -.3]
    common = .5 * (2. * .3**2 * displacement[:, 0]**2 + 5. * .8**2 * displacement[:, 1]**2)
    gap_half = .11 + .07 * displacement[:, 0] - .2 * displacement[:, 1]
    np.testing.assert_allclose(r.matrix[:, 0, 0], common + gap_half)
    np.testing.assert_allclose(r.matrix[:, 1, 1], common - gap_half)
    np.testing.assert_array_equal(r.matrix[:, 0, 1], .04 + .02j)
    np.testing.assert_array_equal(r.matrix[:, 1, 0], .04 - .02j)
    np.testing.assert_allclose(r.gradients, finite_difference(model, q), rtol=2e-9, atol=8e-10)
    np.testing.assert_array_equal(r.gradients[..., 1:], 0.)


def test_spin_boson_diabatic_minima_without_counterterm_and_uncoupled_bath():
    model = DiscreteSpinBoson([.4, .7], [.2, -.1], masses=[3., 2.], bias=.05, tunnelling=0.)
    shift = model.couplings / model.harmonic
    q = np.zeros((2, 2, 3))
    q[0, :, 0], q[1, :, 0] = -shift, shift
    r = model.evaluate(q)
    reorganization_offset = np.sum(model.couplings**2 / (2 * model.harmonic))
    assert r.matrix[0, 0, 0] == pytest.approx(model.bias - reorganization_offset)
    assert r.matrix[1, 1, 1] == pytest.approx(-model.bias - reorganization_offset)
    np.testing.assert_allclose(r.gradients[0, 0, 0], 0., atol=2e-17)
    np.testing.assert_allclose(r.gradients[1, 1, 1], 0., atol=2e-17)
    no_bath_coupling = DiscreteSpinBoson([.4, .7], [0., 0.], masses=[3., 2.], tunnelling=.3)
    r = no_bath_coupling.evaluate(q)
    np.testing.assert_array_equal(r.gradients[:, 0, 0], r.gradients[:, 1, 1])
    np.testing.assert_array_equal(r.gradients[:, 0, 1], 0.)
    np.testing.assert_allclose(r.matrix[:, 0, 0] - r.matrix[:, 1, 1], 0., atol=0)


def test_spin_boson_explicit_cartesian_layout_and_same_atom_mass_guard():
    model = DiscreteSpinBoson([.2, .3], [.1, .1], masses=[4., 4.],
                             coordinates=[1, 2], natoms=3)
    np.testing.assert_array_equal(model.nuclear_masses, [4., 1., 1.])
    r = model.evaluate(np.ones((2, 3, 3)))
    np.testing.assert_array_equal(r.gradients[..., 1:, :], 0.)
    with pytest.raises(ValueError, match="same mass"):
        DiscreteSpinBoson([.2, .3], [.1, .1], masses=[4., 5.], coordinates=[1, 2])


def test_model_owns_immutable_parameters_and_detached_metadata():
    h0 = np.eye(2)
    linear = np.zeros((1, 2, 2))
    harmonic = np.array([.2])
    model = LinearVibronicCoupling(h0, linear, harmonic=harmonic)
    h0[:] = 10.
    linear[:] = 20.
    harmonic[:] = 30.
    np.testing.assert_array_equal(model.h0, np.eye(2))
    np.testing.assert_array_equal(model.linear, 0.)
    np.testing.assert_array_equal(model.harmonic, [.2])
    for name in ("h0", "linear", "harmonic", "reference"):
        with pytest.raises(ValueError):
            getattr(model, name).setflags(write=True)
    with pytest.raises(AttributeError, match="immutable"):
        model.harmonic = np.array([2.])
    with pytest.raises(AttributeError, match="immutable"):
        TullyModelOne().a = .2
    metadata = model.metadata
    metadata["model_identity"]["parameters"]["harmonic"][0] = 90.
    assert model.to_dict()["parameters"]["harmonic"] == [.2]


@pytest.mark.parametrize("model", [
    TullyModelOne(), TullyModelTwo(c=-.04, natoms=2, coordinate=5), TullyModelThree(),
    complex_lvc(), DiscreteSpinBoson([.2, .4], [.1, .3], masses=[2., 3.], tunnelling=.02j),
])
def test_json_model_roundtrip(model):
    identity = json.loads(json.dumps(model.to_dict(), allow_nan=False))
    restored = model_from_dict(identity)
    assert restored.to_dict() == identity
    q = np.random.default_rng(902).normal(size=(3, model.natoms, 3))
    np.testing.assert_array_equal(restored.evaluate(q).matrix, model.evaluate(q).matrix)
    np.testing.assert_array_equal(restored.evaluate(q).gradients, model.evaluate(q).gradients)
    json.dumps(model.metadata, allow_nan=False)


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(extra=True),
    lambda d: d.update(schema="another.schema"),
    lambda d: d.update(version=2),
    lambda d: d.update(version=True),
    lambda d: d.update(model="module:factory"),
    lambda d: d.update(model=[]),
    lambda d: d["parameters"].update(unexpected=3),
    lambda d: d["parameters"].pop("a"),
])
def test_model_identity_rejects_unknown_versions_models_and_keys(mutation):
    identity = copy.deepcopy(TullyModelOne().to_dict())
    mutation(identity)
    with pytest.raises(ValueError):
        model_from_dict(identity)


def test_complex_model_identity_rejects_invalid_payloads():
    identity = complex_lvc().to_dict()
    identity["parameters"]["h0"]["imag"] = [1.]
    with pytest.raises(ValueError, match="same-shape"):
        model_from_dict(identity)
    identity = complex_lvc().to_dict()
    identity["parameters"]["linear"]["extra"] = 0
    with pytest.raises(ValueError, match="exactly"):
        model_from_dict(identity)


@pytest.mark.parametrize("constructor", [
    lambda: TullyModelOne(a=True),
    lambda: TullyModelOne(b=-1),
    lambda: TullyModelTwo(d=np.nan),
    lambda: TullyModelThree(c=-1),
    lambda: TullyModelThree(a=1j),
    lambda: TullyModelOne(coordinate=True),
    lambda: TullyModelOne(coordinate=-1),
    lambda: TullyModelOne(coordinate=3),
    lambda: TullyModelOne(natoms=0),
    lambda: LinearVibronicCoupling([[0., 1.], [0., 0.]], np.zeros((1, 2, 2))),
    lambda: LinearVibronicCoupling([[1j]], [[[0.]]]),
    lambda: LinearVibronicCoupling([[0.]], [[[1j]]]),
    lambda: LinearVibronicCoupling([[0.]], np.zeros((1, 2, 2))),
    lambda: LinearVibronicCoupling([[0.]], np.zeros((0, 1, 1))),
    lambda: LinearVibronicCoupling([[0.]], [[[0.]]], harmonic=[-1.]),
    lambda: LinearVibronicCoupling([[0.]], [[[0.]]], reference=[np.inf]),
    lambda: LinearVibronicCoupling([[0.]], [[[0.]], [[0.]]], coordinates=[0, 0]),
    lambda: LinearVibronicCoupling([[0.]], [[[0.]]], coordinates=[.5]),
    lambda: LinearVibronicCoupling([[0.]], [[[0.]]], coordinates=[True]),
    lambda: LinearVibronicCoupling([[0.]], [[[0.]]], coordinates=[3], natoms=1),
    lambda: DiscreteSpinBoson([], []),
    lambda: DiscreteSpinBoson([0.], [.1]),
    lambda: DiscreteSpinBoson([-.1], [.1]),
    lambda: DiscreteSpinBoson([.1], [.1j]),
    lambda: DiscreteSpinBoson([.1], [np.nan]),
    lambda: DiscreteSpinBoson([.1], [.1], masses=[0.]),
    lambda: DiscreteSpinBoson([.1], [.1], masses=[True]),
    lambda: DiscreteSpinBoson([.1], [.1], tunnelling=True),
    lambda: DiscreteSpinBoson([.1], [.1], tunnelling=np.inf),
    lambda: DiscreteSpinBoson([1e300], [.1]),
])
def test_invalid_model_parameters_fail(constructor):
    with pytest.raises(ValueError):
        constructor()


@pytest.mark.parametrize("bad_q", [
    np.zeros((1, 2, 3)), np.zeros((1, 3)), np.zeros((0, 1, 3)),
    np.full((1, 1, 3), np.nan), np.zeros((1, 1, 3), dtype=complex),
    np.zeros((1, 1, 3), dtype=bool),
])
def test_coordinate_contract_is_checked(bad_q):
    for model in (TullyModelOne(), LinearVibronicCoupling([[0.]], [[[0.]]]),
                  DiscreteSpinBoson([.2], [.1])):
        with pytest.raises(PotentialError):
            model.evaluate(bad_q)


def test_model_does_not_mutate_coordinates_and_rejects_overflow():
    q = np.ones((2, 1, 3), dtype=np.float32)
    q.flags.writeable = False
    original = q.copy()
    result = TullyModelOne().evaluate(q)
    assert result.matrix.dtype == np.float64
    np.testing.assert_array_equal(q, original)
    model = LinearVibronicCoupling([[0.]], [[[0.]]], harmonic=[1e300])
    with pytest.raises(PotentialError, match="overflow"):
        model.evaluate(np.full((1, 1, 3), 1e300))
