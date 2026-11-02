"""Direct physical C ABI checks for the analytic built-in PES snapshots."""

import copy
import ctypes as C
import gc
import pickle
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from pyrpmd.potentials import (
    DoubleWellPotential,
    FreePotential,
    HarmonicPotential,
    MorsePairPotential,
    PeriodicCosinePotential,
    PotentialError,
    validate_result,
)
from pyrpmd.potentials.cabi import DP, Error, I32, Input, Output
from pyrpmd.potentials.native import NativePESBinding
from pyrpmd.propagation import NativeUnavailableError


native = pytest.importorskip("pyrpmd._native")
MODELS = ("free", "harmonic", "morse_pair", "double_well", "periodic_cosine")
EVALUATE = C.CFUNCTYPE(C.c_int32, C.c_void_p, C.POINTER(Input),
                      C.POINTER(Output), C.POINTER(Error))


def _potential(name):
    if name == "free":
        return FreePotential()
    if name == "harmonic":
        return HarmonicPotential(k=[[.4], [.6], [.8]], center=[.2, -.1, .3])
    if name == "morse_pair":
        return MorsePairPotential(atoms=(2, 0), depth=.3, alpha=.7, equilibrium=1.2)
    if name == "double_well":
        return DoubleWellPotential(height=.2, width=1.3, atom=1, axis=2, confinement=.04)
    if name == "periodic_cosine":
        return PeriodicCosinePotential(
            [[3., .2, .1], [.4, 4., .3], [.2, .5, 5.]],
            pbc=[True, False, True], height=[.2, .4, .3],
            repeats=(2, 3, 1), atoms=(2, 0), confinement=.1,
        )
    raise AssertionError(name)


def _coordinates():
    return np.random.default_rng(201).normal(scale=.7, size=(3, 3, 3))


def _call(binding, q, *, input_updates=None, output_updates=None,
          message_stride=128, omit_messages=False, error_capacity=256):
    """Call the compiled address without a Python PES adapter or integrator."""
    q = np.ascontiguousarray(q, dtype=np.float64)
    energy = np.full(len(q), -12345.)
    forces = np.full(q.shape, -12345.)
    status = np.full(len(q), -99, dtype=np.int32)
    messages = None if omit_messages else C.create_string_buffer(len(q) * message_stride)
    error_buffer = C.create_string_buffer(error_capacity)
    error = Error(C.addressof(error_buffer), len(error_buffer), 0)
    inp = Input(1, C.sizeof(Input), 1, 1, len(q), q.shape[1],
                q.ctypes.data_as(DP), q.size)
    out = Output(1, C.sizeof(Output), energy.ctypes.data_as(DP), energy.size,
                 forces.ctypes.data_as(DP), forces.size,
                 status.ctypes.data_as(C.POINTER(I32)), len(status),
                 None if messages is None else C.addressof(messages),
                 0 if messages is None else message_stride,
                 0 if messages is None else len(messages))
    for key, value in (input_updates or {}).items():
        setattr(inp, key, value)
    for key, value in (output_updates or {}).items():
        setattr(out, key, value)
    with binding.acquire() as (kind, address, context, natoms):
        assert kind == "cabi-v1"
        rc = EVALUATE(address)(context, C.byref(inp), C.byref(out), C.byref(error))
    decoded = () if messages is None else tuple(
        messages.raw[i * message_stride:(i + 1) * message_stride].split(b"\0", 1)[0].decode()
        for i in range(len(q))
    )
    return SimpleNamespace(rc=rc, energy=energy, forces=forces, status=status,
                           messages=decoded, error=error_buffer.value.decode(),
                           required_error_capacity=error.required_capacity)


@pytest.mark.parametrize("model", MODELS)
def test_compiled_binding_matches_python_energy_forces_and_finite_difference(model):
    q = _coordinates()
    potential = _potential(model)
    expected = potential.evaluate(q)
    # An ordinary evaluate call supplies dimensions for the no-argument hook.
    binding = potential.native_binding()
    assert isinstance(binding, NativePESBinding)
    assert binding.natoms == q.shape[1]
    actual = _call(binding, q)
    assert actual.rc == 0 and actual.error == ""
    np.testing.assert_array_equal(actual.status, expected.status)
    assert actual.messages == expected.messages
    np.testing.assert_allclose(actual.energy, expected.energy, atol=3e-14, rtol=3e-14)
    np.testing.assert_allclose(actual.forces, expected.forces, atol=3e-14, rtol=3e-14)

    # Independently differentiate Python energies, rather than comparing only
    # two implementations of the same analytic force expression.
    epsilon = 2e-6
    numerical = np.empty_like(q)
    for atom in range(q.shape[1]):
        for axis in range(3):
            plus, minus = q.copy(), q.copy()
            plus[:, atom, axis] += epsilon
            minus[:, atom, axis] -= epsilon
            numerical[:, atom, axis] = -(
                potential.evaluate(plus).energy - potential.evaluate(minus).energy
            ) / (2 * epsilon)
    np.testing.assert_allclose(actual.forces, numerical, atol=2e-9, rtol=2e-8)
    if model == "free":
        np.testing.assert_array_equal(expected.virial, np.zeros((len(q), 3, 3)))
    else:
        assert expected.virial is None


@pytest.mark.parametrize("k, center", [
    (.7, None),
    ([.2, .5, .8], [-.1, .4, .3]),
    ([[.2], [.5], [.8]], [[.1, .2, .3]]),
    (np.arange(9).reshape(3, 3) / 10, np.arange(9).reshape(3, 3) / 20),
])
def test_harmonic_native_parameters_preserve_atom_and_cartesian_broadcasting(k, center):
    potential = HarmonicPotential(k=k, center=center)
    q = _coordinates()
    binding = potential.native_binding(q.shape[1])
    actual, expected = _call(binding, q), potential.evaluate(q)
    assert actual.rc == 0
    np.testing.assert_allclose(actual.energy, expected.energy, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(actual.forces, expected.forces, atol=2e-14, rtol=2e-14)


@pytest.mark.parametrize("parameter", ["k", "center"])
def test_harmonic_frame_dependent_parameters_explicitly_decline_binding(parameter):
    potential = HarmonicPotential(**{parameter: np.ones((3, 1, 1))})
    potential.evaluate(_coordinates())
    with pytest.raises(NativeUnavailableError, match="frame|batch|broadcast"):
        potential.native_binding(3)


@pytest.mark.parametrize("pbc", [[True, True, True], [True, False, True], [False, True, False]])
@pytest.mark.parametrize("atoms", [None, (2, 0)])
def test_periodic_native_row_cell_translations_and_selected_atoms(pbc, atoms):
    cell = np.array([[3., .3, .1], [.4, 4., .2], [.2, .5, 5.]])
    potential = PeriodicCosinePotential(
        cell, pbc=pbc, height=[.2, .3, .4], repeats=(2, 1, 3),
        atoms=atoms, confinement=.2,
    )
    q = _coordinates()
    binding = potential.native_binding(3)
    expected = potential.evaluate(q)
    original = _call(binding, q)
    lattice_shift = np.array([2, -3, 4]) * np.asarray(pbc)
    translated = _call(binding, q + lattice_shift @ cell)
    np.testing.assert_allclose(original.energy, expected.energy, atol=3e-14, rtol=3e-14)
    np.testing.assert_allclose(original.forces, expected.forces, atol=3e-14, rtol=3e-14)
    np.testing.assert_allclose(translated.energy, original.energy, atol=5e-13, rtol=5e-13)
    np.testing.assert_allclose(translated.forces, original.forces, atol=5e-13, rtol=5e-13)
    if atoms is not None:
        np.testing.assert_array_equal(original.forces[:, 1], 0.)


def test_morse_failed_frames_keep_successful_neighbors_and_nan_outputs():
    potential = MorsePairPotential(depth=.3, alpha=.7, equilibrium=1.2)
    q = np.zeros((3, 2, 3))
    q[0, 1, 0] = 1.4
    q[2, 1, 0] = np.inf
    result = _call(potential.native_binding(2), q)
    assert result.rc == 0
    np.testing.assert_array_equal(result.status, [0, 2, 3])
    np.testing.assert_allclose(result.energy[:1], potential.evaluate(q[:1]).energy)
    np.testing.assert_allclose(result.forces[:1], potential.evaluate(q[:1]).forces)
    assert np.isnan(result.energy[1:]).all() and np.isnan(result.forces[1:]).all()
    assert result.messages[0] == ""
    assert "coincident" in result.messages[1]
    assert "NaN/Inf" in result.messages[2]


@pytest.mark.parametrize("model", MODELS)
def test_all_builtin_callbacks_reject_nonfinite_coordinates_per_frame(model):
    q = _coordinates()
    q[1, 2, 1] = np.nan
    result = _call(_potential(model).native_binding(3), q)
    assert result.rc == 0
    np.testing.assert_array_equal(result.status, [0, 3, 0])
    assert np.isnan(result.energy[1]) and np.isnan(result.forces[1]).all()
    assert np.isfinite(result.energy[[0, 2]]).all()
    assert "NaN/Inf" in result.messages[1]


def test_harmonic_overflow_returns_nonfinite_frame_status_and_truncated_message():
    q = np.full((1, 1, 3), 1e308)
    binding = HarmonicPotential().native_binding(1)
    result = _call(binding, q, message_stride=8)
    assert result.rc == 0 and result.status[0] == 3
    assert np.isnan(result.energy).all() and np.isnan(result.forces).all()
    assert result.messages == ("nonfini",)
    without_messages = _call(binding, q, omit_messages=True)
    assert without_messages.rc == 0 and without_messages.status[0] == 3


@pytest.mark.parametrize("model", ["morse_pair", "double_well"])
def test_scaled_square_overflow_matches_python_failure(model):
    # Python squares before multiplying by the energy scale. A tiny scale
    # must not turn an overflowing reference calculation into native success.
    if model == "morse_pair":
        potential = MorsePairPotential(depth=1e-300, alpha=1., equilibrium=357.)
        q = np.array([[[0., 0., 0.], [1., 0., 0.]]])
    else:
        potential = DoubleWellPotential(height=1e-300, confinement=0.)
        q = np.array([[[1e80, 0., 0.]]])
    with np.errstate(over="ignore", invalid="ignore"):
        reference = potential.evaluate(q)
    assert np.isinf(reference.energy).all()
    assert np.isfinite(reference.forces).all()
    with pytest.raises(PotentialError, match="nonfinite"):
        validate_result(reference, q.shape)

    result = _call(potential.native_binding(q.shape[1]), q)
    assert result.rc == 0
    np.testing.assert_array_equal(result.status, [3])
    assert np.isnan(result.energy).all() and np.isnan(result.forces).all()
    assert "nonfinite" in result.messages[0]


@pytest.mark.parametrize("target, field, value, expected_rc", [
    ("input", "abi_version", 2, 2),
    ("input", "struct_size", 0, 2),
    ("output", "abi_version", 0, 2),
    ("output", "struct_size", 0, 2),
    ("input", "layout", 2, 1),
    ("input", "scalar_type", 0, 1),
    ("input", "nstructures", -1, 1),
    ("input", "nstructures", 2**63 - 1, 1),
    ("input", "natoms", 4, 1),
    ("input", "positions_count", 0, 1),
    ("input", "positions", None, 1),
    ("output", "energies_count", 2, 1),
    ("output", "forces_count", 0, 1),
    ("output", "structure_status_count", 0, 1),
    ("output", "structure_message_stride", 0, 1),
    ("output", "structure_messages_capacity", 1, 1),
    ("output", "structure_messages", None, 1),
])
def test_builtin_cabi_rejects_invalid_headers_and_buffers_before_writing(target, field, value, expected_rc):
    binding = FreePotential().native_binding(3)
    result = _call(binding, _coordinates(), **{target + "_updates": {field: value}},
                   error_capacity=12)
    assert result.rc == expected_rc
    assert result.error and result.required_error_capacity > 12
    assert len(result.error) <= 11
    np.testing.assert_array_equal(result.energy, -12345.)
    np.testing.assert_array_equal(result.forces, -12345.)
    np.testing.assert_array_equal(result.status, -99)


def test_factory_capsule_owns_context_after_python_parameters_are_released():
    parameters = {"k": [.2] * 9, "center": [.1] * 9}
    owner, address, context = native.builtin_pes("harmonic", 3, parameters)
    function = EVALUATE(address)
    binding = NativePESBinding("cabi-v1", function, 3, owner, threading.RLock(), context)
    parameters["k"][:] = [99.] * 9
    del parameters, owner, function
    gc.collect()
    q = _coordinates()
    result = _call(binding, q)
    expected = HarmonicPotential(.2, .1).evaluate(q)
    assert result.rc == 0
    np.testing.assert_allclose(result.energy, expected.energy)
    np.testing.assert_allclose(result.forces, expected.forces)


def test_binding_cache_is_stable_and_parameter_mutation_preserves_old_snapshot():
    potential = HarmonicPotential(k=.2, center=.1)
    first = potential.native_binding(3)
    again = potential.native_binding(3)
    with first.acquire() as initial, again.acquire() as repeated:
        assert initial == repeated
    q = _coordinates()
    before = _call(first, q)
    potential.k[...] = .7
    second = potential.native_binding(3)
    with first.acquire() as old, second.acquire() as new:
        assert old[2] != new[2]
    after = _call(second, q)
    expected = potential.evaluate(q)
    np.testing.assert_allclose(after.energy, expected.energy)
    np.testing.assert_allclose(after.forces, expected.forces)
    np.testing.assert_array_equal(_call(first, q).energy, before.energy)
    np.testing.assert_array_equal(_call(first, q).forces, before.forces)
    del potential
    gc.collect()
    np.testing.assert_array_equal(_call(first, q).energy, before.energy)
    np.testing.assert_array_equal(_call(second, q).energy, after.energy)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("override", ["subclass", "instance", "class"])
def test_builtin_binding_refuses_overridden_python_semantics(model, override, monkeypatch):
    potential = _potential(model)
    original = type(potential).evaluate
    if override == "subclass":
        potential.__class__ = type("CustomPotential", (type(potential),), {})
    elif override == "instance":
        monkeypatch.setattr(potential, "evaluate", lambda q: original(potential, q))
    else:
        monkeypatch.setattr(type(potential), "evaluate", lambda self, q: original(self, q))
    with pytest.raises(NativeUnavailableError):
        potential.native_binding(3)


@pytest.mark.parametrize("model", MODELS)
def test_native_cache_does_not_break_pickle_or_deepcopy(model):
    potential = _potential(model)
    q = _coordinates()
    expected = potential.evaluate(q)
    original_binding = potential.native_binding(3)
    for restored in (copy.deepcopy(potential), pickle.loads(pickle.dumps(potential))):
        result = _call(restored.native_binding(3), q)
        assert result.rc == 0
        np.testing.assert_allclose(result.energy, expected.energy, atol=3e-14, rtol=3e-14)
        np.testing.assert_allclose(result.forces, expected.forces, atol=3e-14, rtol=3e-14)
        with original_binding.acquire() as old, restored.native_binding(3).acquire() as new:
            assert old[2] != new[2]


@pytest.mark.parametrize("kind, natoms, parameters", [
    ("unknown", 3, {}),
    ("free", 0, {}),
    ("free", True, {}),
    ("free", 3.0, {}),
    ("harmonic", 3, {"k": [.2], "center": [0.] * 9}),
    ("harmonic", 3, {"k": [-.2] * 9, "center": [0.] * 9}),
    ("harmonic", 3, {"k": [.2] * 9, "center": [np.nan] * 9}),
    ("morse_pair", 3, {"atoms": [0, 3], "depth": .2, "alpha": 1., "equilibrium": 1.4}),
    ("morse_pair", 3, {"atoms": [0, 0], "depth": .2, "alpha": 1., "equilibrium": 1.4}),
    ("double_well", 3, {"atom": 0, "axis": 3, "height": .2, "width": 1., "confinement": 0.}),
])
def test_native_factory_rejects_invalid_dimensions_parameters_and_indices(kind, natoms, parameters):
    with pytest.raises((ValueError, TypeError, OverflowError)):
        native.builtin_pes(kind, natoms, parameters)
