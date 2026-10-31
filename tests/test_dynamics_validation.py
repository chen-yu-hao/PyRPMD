"""Public nuclear data contracts and independent free-ring map checks."""

import copy

import numpy as np
import pytest
from scipy.linalg import expm, sqrtm

from pyrpmd.batched import BatchedState
from pyrpmd.dynamics import CayleyIntegrator, Dynamics, State, native_capabilities
from pyrpmd.isomorphic import IsomorphicEhrenfestState
from pyrpmd.native_ensemble import NativeEnsemble, _array
from pyrpmd.potentials import FreePotential


@pytest.mark.parametrize("dtype", [complex, bool, str, object])
@pytest.mark.parametrize("field", ["q", "p"])
@pytest.mark.parametrize("batched", [False, True])
def test_nuclear_states_reject_coercible_nonreal_data(dtype, field, batched):
    shape = (2, 3, 1, 3) if batched else (3, 1, 3)
    values = {name: np.ones(shape) for name in ("q", "p")}
    values[field] = values[field].astype(dtype)
    if dtype is complex:
        values[field] += 2j
    constructor = BatchedState if batched else State
    with pytest.raises(ValueError, match="real numerical"):
        constructor(**values)


@pytest.mark.parametrize("dtype", [complex, bool, str, object])
def test_ring_masses_reject_coercible_nonreal_data(dtype):
    mass = np.array([1], dtype=dtype)
    if dtype is complex:
        mass += 2j
    with pytest.raises(ValueError, match="real numerical"):
        CayleyIntegrator(mass, 2., 3, .01, native=False)


@pytest.mark.parametrize("field", ["beta", "dt"])
@pytest.mark.parametrize("value", [True, np.bool_(True), np.complex128(1 + 2j), "1", np.array(1.)])
def test_ring_scalar_parameters_require_real_scalars(field, value):
    options = dict(beta=2., dt=.01)
    options[field] = value
    with pytest.raises(ValueError, match="real numerical scalars"):
        CayleyIntegrator([1], nbeads=3, native=False, **options)


@pytest.mark.parametrize("value", [True, np.bool_(True), 1 + 2j, "1", np.array(1.),
                                  np.array([1.]), np.inf, np.nan, 10**400])
def test_state_time_requires_a_finite_real_scalar(value):
    q = np.zeros((1, 1, 3))
    with pytest.raises(ValueError, match="state clock"):
        State(q, q, time=value)


def test_state_normalizes_numpy_clock_scalars_without_wrapping():
    q = np.zeros((1, 1, 3))
    largest = np.iinfo(np.int64).max
    state = State(q, q, step=np.int64(largest), time=np.float64(-2.))
    assert type(state.step) is int
    assert type(state.time) is float
    Dynamics(FreePotential(), [1], 2., 1, 0., native=False).step(state)
    assert state.step == largest + 1
    assert state.time == -2.
    np.testing.assert_array_equal(state.q, q)


def test_nuclear_validation_preserves_complex_electronic_amplitudes():
    q = np.zeros((1, 1, 3))
    amplitudes = np.array([1., 1j]) / np.sqrt(2.)
    state = IsomorphicEhrenfestState(q, q, amplitudes=amplitudes)
    np.testing.assert_array_equal(state.amplitudes, amplitudes)


class CountedFree(FreePotential):
    def __init__(self):
        self.calls = 0

    def evaluate(self, q):
        self.calls += 1
        return super().evaluate(q)


NATIVE_CHOICES = [False] + (["required"] if native_capabilities()["available"] else [])


@pytest.mark.parametrize("native", NATIVE_CHOICES)
@pytest.mark.parametrize("api", ["step", "sample_step", "thermalize"])
@pytest.mark.parametrize("field,value", [("time", 1 + 2j), ("time", np.inf),
                                         ("step", -1), ("step", True)])
def test_invalid_live_clock_fails_before_force_or_rng_changes(native, api, field, value):
    potential = CountedFree()
    dynamics = Dynamics(potential, [1.], 2., 1, .01, seed=9, native=native)
    state = State(np.zeros((1, 1, 3)), np.ones((1, 1, 3)))
    old_q, old_p = state.q.copy(), state.p.copy()
    rng = copy.deepcopy(dynamics.rng.bit_generator.state)
    setattr(state, field, value)
    args = (state, 1) if api == "thermalize" else (state,)
    with pytest.raises(ValueError, match="state clock"):
        getattr(dynamics, api)(*args)
    assert potential.calls == 0
    assert dynamics.rng.bit_generator.state == rng
    np.testing.assert_array_equal(state.q, old_q)
    np.testing.assert_array_equal(state.p, old_p)


@pytest.mark.parametrize("api", ["step", "propagate", "propagate_block", "sample_step", "thermalize"])
def test_nonfinite_next_clock_does_not_commit_a_step(api):
    potential = CountedFree()
    dynamics = Dynamics(potential, [1.], 2., 1, 1e308, native=False)
    state = State(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)), time=1e308)
    before = state.copy()
    rng = copy.deepcopy(dynamics.rng.bit_generator.state)
    args = (state, 1) if api in ("propagate", "propagate_block", "thermalize") else (state,)
    with pytest.raises((FloatingPointError, RuntimeError), match="integration clock"):
        getattr(dynamics, api)(*args)
    np.testing.assert_array_equal(state.q, before.q)
    np.testing.assert_array_equal(state.p, before.p)
    assert (state.step, state.time) == (before.step, before.time)
    assert dynamics.rng.bit_generator.state == rng


def test_unrepresentable_thermalization_length_fails_before_resampling():
    dynamics = Dynamics(FreePotential(), [1.], 2., 1, .1, native=False)
    state = State(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)))
    rng = copy.deepcopy(dynamics.rng.bit_generator.state)
    with pytest.raises(FloatingPointError, match="integration clock"):
        dynamics.thermalize(state, 10**400)
    assert dynamics.rng.bit_generator.state == rng
    assert (state.step, state.time) == (0, 0.)
    np.testing.assert_array_equal(state.p, 0.)


@pytest.mark.parametrize("field", ["beta", "dt"])
def test_unrepresentable_ring_scalar_is_a_parameter_error(field):
    options = dict(beta=2., dt=.1)
    options[field] = 10**400
    with pytest.raises(ValueError, match="invalid beta/dt"):
        CayleyIntegrator([1.], nbeads=1, native=False, **options)


@pytest.mark.parametrize("field", ["gamma_centroid", "friction_scale"])
@pytest.mark.parametrize("value", [True, np.bool_(False), np.complex128(1 + 2j), "1", np.array(1.)])
def test_sampling_friction_requires_real_scalars_without_rng_changes(field, value):
    dynamics = Dynamics(FreePotential(), [1.], 2., 2, .01, native=False)
    state = State(np.zeros((2, 1, 3)), np.zeros((2, 1, 3)))
    rng = copy.deepcopy(dynamics.rng.bit_generator.state)
    options = {field: value}
    if field == "friction_scale":
        options["thermostat"] = "trpmd"
    with pytest.raises(ValueError, match=field):
        dynamics.sample_step(state, **options)
    assert dynamics.rng.bit_generator.state == rng
    assert (state.step, state.time) == (0, 0.)
    np.testing.assert_array_equal(state.q, 0.)
    np.testing.assert_array_equal(state.p, 0.)


@pytest.mark.parametrize("dtype", [complex, bool, str, object])
def test_native_array_conversion_rejects_nonphysical_types(dtype):
    value = np.array([1.], dtype=dtype)
    with pytest.raises(ValueError, match="real numerical"):
        _array(value)


def test_native_boolean_boundary_and_cell_masks_remain_supported():
    pytest.importorskip("pyrpmd._thermal")
    engine = NativeEnsemble(
        np.zeros((1, 1, 3)), [1.], 2., .001, potential={"kind": "free"},
        cell=np.eye(3), pbc=[True, True, True],
        barostat={"mode": "mttk", "pressure": 1., "cell_mask": np.eye(3, dtype=bool)},
        seed=20,
    )
    assert engine.run(1)["step"] == 1
    assert np.isfinite(engine.state()["q"]).all()


def dense_ring_map(beads, mass, beta, dt, scheme, half):
    eye = np.eye(beads)
    laplacian = 2 * eye - np.roll(eye, 1, axis=0) - np.roll(eye, -1, axis=0)
    generator = np.block([
        [np.zeros_like(eye), eye / mass],
        [-mass * (beads / beta)**2 * laplacian, np.zeros_like(eye)],
    ])
    if scheme == "normal-mode-bab":
        return expm((.5 if half else 1.) * dt * generator)
    identity = np.eye(2 * beads)
    full = np.linalg.solve(identity - .5 * dt * generator, identity + .5 * dt * generator)
    return np.real_if_close(sqrtm(full)) if half else full


@pytest.mark.parametrize("beads", [1, 5, 8])
@pytest.mark.parametrize("dt", [0., -.03, .03])
@pytest.mark.parametrize("scheme", ["cayley-bcb", "normal-mode-bab"])
@pytest.mark.parametrize("half", [False, True])
def test_reused_full_and_half_maps_match_independent_dense_reference(beads, dt, scheme, half):
    masses, beta = [1.3, 2.4], 3.7
    rng = np.random.default_rng(25)
    q = rng.normal(size=(beads, 2, 3))
    p = rng.normal(size=q.shape)
    expected = [np.concatenate((q[:, atom], p[:, atom])) for atom in range(2)]
    references = [dense_ring_map(beads, m, beta, dt, scheme, half) for m in masses]
    integrator = CayleyIntegrator(masses, beta, beads, dt, scheme=scheme, native=False)
    advance = integrator.half_step if half else integrator.free_step
    for _ in range(4):
        advance(q, p)
        expected = [matrix @ state for matrix, state in zip(references, expected)]
    for atom, state in enumerate(expected):
        np.testing.assert_allclose(q[:, atom], state[:beads], atol=3e-12, rtol=3e-12)
        np.testing.assert_allclose(p[:, atom], state[beads:], atol=3e-12, rtol=3e-12)


def test_numpy_direct_integrator_does_not_reuse_a_different_timestep_map():
    rng = np.random.default_rng(72)
    q, p = rng.normal(size=(2, 5, 1, 3))
    integrator = CayleyIntegrator([1.7], 3., 5, .02, native=False)
    integrator.free_step(q, p)
    integrator.dt = -.01
    expected = dense_ring_map(5, 1.7, 3., -.01, "cayley-bcb", False) @ np.concatenate((q[:, 0], p[:, 0]))
    integrator.free_step(q, p)
    np.testing.assert_allclose(q[:, 0], expected[:5], atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(p[:, 0], expected[5:], atol=1e-13, rtol=1e-13)
