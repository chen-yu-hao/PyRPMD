"""Independent real-mode sampling, stochastic rollback and large-ring checks.

The half maps below use 2x2 matrix exponentials/principal matrix square roots,
not the production half-map coefficients or FFT implementation. Noise follows
the documented real orthogonal mode order, including centroid and Nyquist.
"""

import copy
from pathlib import Path
import shutil
import subprocess
from types import MethodType

import numpy as np
import pytest
from scipy.linalg import expm, sqrtm

from pyrpmd.dynamics import CayleyIntegrator, Dynamics, State
from pyrpmd.potentials import CABIPotential, FreePotential, HarmonicPotential, PESResult, PotentialError


native = pytest.importorskip("pyrpmd._native")
pytestmark = pytest.mark.skipif(
    getattr(native, "SAMPLING_FREE_API", 0) < 1,
    reason="native free/OU/free API build required",
)
BEADS = [1, 2, 3, 5, 8, 16, 31, 32, 64]
SCHEMES = ["cayley-bcb", "normal-mode-bab"]
MASSES = np.array([1.1, 3.7, 12.0])
BETA, DT = 3.4, 0.017
FIXED_POSITIONS = np.array([[0.4, -0.3, 0.1], [0.6, 0.2, -0.1], [-0.2, 0.5, 0.3]])


def real_modes(beads, beta):
    j = np.arange(beads)
    rows, frequencies = [np.ones(beads) / np.sqrt(beads)], [0.]
    for frequency in range(1, (beads + 1) // 2):
        angle = 2 * np.pi * frequency * j / beads
        rows.extend([np.sqrt(2 / beads) * np.cos(angle), np.sqrt(2 / beads) * np.sin(angle)])
        frequencies.extend([2 * beads / beta * np.sin(np.pi * frequency / beads)] * 2)
    if beads % 2 == 0:
        rows.append((-1.) ** j / np.sqrt(beads))
        frequencies.append(2 * beads / beta)
    return np.asarray(rows), np.asarray(frequencies)


def independent_half_maps(masses, frequencies, dt, scheme):
    result = np.empty((len(frequencies), len(masses), 2, 2))
    eye = np.eye(2)
    for mode, frequency in enumerate(frequencies):
        for atom, mass in enumerate(masses):
            generator = np.array([[0., 1 / mass], [-mass * frequency**2, 0.]])
            if scheme == "normal-mode-bab":
                half = expm(0.5 * dt * generator)
            else:
                full = np.linalg.solve(eye - 0.5 * dt * generator, eye + 0.5 * dt * generator)
                half = np.real_if_close(sqrtm(full), tol=1000)
                assert not np.iscomplexobj(half)
            result[mode, atom] = half
    return result


def dense_half_step(q, p, modes, half_maps):
    shape = q.shape
    qk = (modes @ q.reshape(len(q), -1)).reshape(shape)
    pk = (modes @ p.reshape(len(p), -1)).reshape(shape)
    new_q = half_maps[:, :, 0, 0, None] * qk + half_maps[:, :, 0, 1, None] * pk
    new_p = half_maps[:, :, 1, 0, None] * qk + half_maps[:, :, 1, 1, None] * pk
    q[:] = (modes.T @ new_q.reshape(len(q), -1)).reshape(shape)
    p[:] = (modes.T @ new_p.reshape(len(p), -1)).reshape(shape)


def coupled_force(q):
    force = -0.4 * q
    difference = q[:, 1] - q[:, 0]
    force[:, 0] += 0.7 * difference
    force[:, 1] -= 0.7 * difference
    return force


class CoupledPotential:
    def __init__(self, fixed=False, fail_at=None):
        self.fixed, self.fail_at, self.calls = fixed, fail_at, 0

    def evaluate(self, q):
        self.calls += 1
        if self.calls == self.fail_at:
            raise PotentialError("injected force failure")
        if self.fixed:
            np.testing.assert_array_equal(q[:, 0], np.broadcast_to(FIXED_POSITIONS[0], q[:, 0].shape))
        energy = 0.2 * np.sum(q * q, axis=(1, 2)) + 0.35 * np.sum((q[:, 1] - q[:, 0])**2, axis=1)
        return PESResult(energy, coupled_force(q))


def initial(beads, fixed=False):
    rng = np.random.default_rng(913 + beads)
    q = FIXED_POSITIONS[None] + rng.normal(0, 0.1, (beads, len(MASSES), 3))
    p = rng.normal(size=q.shape)
    if fixed:
        q[:, 0] = FIXED_POSITIONS[0]
        p[:, 0] = 0.
    return State(q, p, step=3, time=0.125)


def dynamics(beads, scheme, fixed=False, **kwargs):
    return Dynamics(
        kwargs.pop("potential", CoupledPotential(fixed)), MASSES, BETA, beads, DT,
        integrator=scheme, native="required", seed=517,
        fixed_atoms=[0] if fixed else [], fixed_positions=FIXED_POSITIONS, **kwargs,
    )


def pin(q, p, fixed):
    if fixed:
        q[:, 0] = FIXED_POSITIONS[0]
        p[:, 0] = 0.


def dense_sampling_step(state, rng, modes, frequencies, half_maps, gamma, fixed):
    q, p = state.q.copy(), state.p.copy()
    force = coupled_force(q)
    if fixed:
        force[:, 0] = 0.
    p += 0.5 * DT * force
    dense_half_step(q, p, modes, half_maps)
    pin(q, p, fixed)
    damping = 2 * frequencies.copy()
    damping[0] = gamma
    decay = np.exp(-damping * DT)
    noise = rng.normal(size=p.shape) * np.sqrt(MASSES[None, :, None] * len(p) / BETA)
    if fixed:
        noise[:, 0] = 0.
    pk = (modes @ p.reshape(len(p), -1)).reshape(p.shape)
    pk = decay[:, None, None] * pk + np.sqrt(1 - decay**2)[:, None, None] * noise
    p[:] = (modes.T @ pk.reshape(len(p), -1)).reshape(p.shape)
    pin(q, p, fixed)
    dense_half_step(q, p, modes, half_maps)
    pin(q, p, fixed)
    force = coupled_force(q)
    if fixed:
        force[:, 0] = 0.
    p += 0.5 * DT * force
    pin(q, p, fixed)
    return State(q, p, state.step + 1, state.time + DT)


@pytest.mark.parametrize("beads", BEADS)
@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("fixed", [False, True])
def test_auto_sampling_matches_independent_dense_modes_and_noise_order(monkeypatch, beads, scheme, fixed):
    dyn = dynamics(beads, scheme, fixed)
    actual, expected = initial(beads, fixed), initial(beads, fixed)
    rng = np.random.default_rng(517)
    modes, frequencies = real_modes(beads, BETA)
    half_maps = independent_half_maps(MASSES, frequencies, DT, scheme)
    kernel, calls = native.sample_free, []

    def counted(*args):
        calls.append(True)
        return kernel(*args)

    monkeypatch.setattr(native, "sample_free", counted)
    # Repeated and changed gamma values exercise coefficient reuse/invalidation.
    for gamma in (0.2, 2.7, 2.7, 0.01):
        expected = dense_sampling_step(expected, rng, modes, frequencies, half_maps, gamma, fixed)
        dyn.sample_step(actual, gamma_centroid=gamma)
        np.testing.assert_allclose(actual.q, expected.q, rtol=2e-12, atol=8e-13)
        np.testing.assert_allclose(actual.p, expected.p, rtol=2e-12, atol=8e-12)
        assert (actual.step, actual.time) == (expected.step, expected.time)
        assert dyn.rng.bit_generator.state == rng.bit_generator.state
        if fixed:
            np.testing.assert_array_equal(actual.q[:, 0], np.broadcast_to(FIXED_POSITIONS[0], actual.q[:, 0].shape))
            np.testing.assert_array_equal(actual.p[:, 0], 0.)
    eligible = beads <= 8 or beads & (beads - 1) == 0
    assert len(calls) == (4 if eligible else 0)


@pytest.mark.parametrize("beads", BEADS)
@pytest.mark.parametrize("scheme", SCHEMES)
def test_direct_sampler_respects_each_real_mode_and_has_correct_psd_covariance(beads, scheme):
    mass = 2.3
    integrator = CayleyIntegrator([mass], BETA, beads, DT, native="required", scheme=scheme)
    modes, frequencies = real_modes(beads, BETA)
    half_maps = independent_half_maps([mass], frequencies, DT, scheme)
    # Distinct sine/cosine damping checks that no accepted input coefficient
    # is ignored by the low-level API. PILE-L itself uses equal paired values.
    decay = np.exp(-np.linspace(0.2, 4.1, beads) * DT)
    amplitude = np.sqrt(1 - decay**2)
    mask = np.zeros(1, dtype=np.uint8)
    columns = np.empty((2 * beads, beads))
    for mode in range(beads):
        q, p = np.zeros((beads, 1, 3)), np.zeros((beads, 1, 3))
        noise = np.zeros_like(q)
        noise[mode, 0, 0] = np.sqrt(mass * beads / BETA)
        native.sample_free(integrator._context, q, p, noise, decay, amplitude, mask)
        columns[:, mode] = np.concatenate((q[:, 0, 0], p[:, 0, 0]))
        np.testing.assert_array_equal(q[:, :, 1:], 0.)
        np.testing.assert_array_equal(p[:, :, 1:], 0.)
    response = np.vstack((modes.T * half_maps[:, 0, 0, 1], modes.T * half_maps[:, 0, 1, 1]))
    expected_columns = response * (amplitude * np.sqrt(mass * beads / BETA))
    np.testing.assert_allclose(columns, expected_columns, rtol=2e-12, atol=3e-13)
    covariance = columns @ columns.T
    expected_covariance = (response * (mass * beads / BETA * amplitude**2)) @ response.T
    np.testing.assert_allclose(covariance, expected_covariance, rtol=3e-12, atol=2e-12)
    assert np.linalg.eigvalsh(covariance).min() >= -2e-12 * max(1., np.linalg.norm(covariance))


@pytest.mark.parametrize("beads", [3, 31, 32])
@pytest.mark.parametrize("scheme", SCHEMES)
def test_direct_free_ou_free_with_unequal_masses_and_nonzero_initial_modes(beads, scheme):
    integrator = CayleyIntegrator(MASSES, BETA, beads, DT, native="required", scheme=scheme)
    actual, expected = initial(beads, True), initial(beads, True)
    modes, frequencies = real_modes(beads, BETA)
    half_maps = independent_half_maps(MASSES, frequencies, DT, scheme)
    rng = np.random.default_rng(917)
    noise = rng.normal(size=actual.p.shape) * np.sqrt(MASSES[None, :, None] * beads / BETA)
    noise[:, 0] = 0.
    decay = np.exp(-np.linspace(0.2, 4.1, beads) * DT)
    amplitude = np.sqrt(1 - decay**2)
    dense_half_step(expected.q, expected.p, modes, half_maps)
    pin(expected.q, expected.p, True)
    pk = (modes @ expected.p.reshape(beads, -1)).reshape(expected.p.shape)
    pk = decay[:, None, None] * pk + amplitude[:, None, None] * noise
    expected.p[:] = (modes.T @ pk.reshape(beads, -1)).reshape(expected.p.shape)
    pin(expected.q, expected.p, True)
    dense_half_step(expected.q, expected.p, modes, half_maps)
    pin(expected.q, expected.p, True)
    native.sample_free(integrator._context, actual.q, actual.p, noise, decay, amplitude, np.array([1, 0, 0], dtype=np.uint8))
    np.testing.assert_allclose(actual.q, expected.q, rtol=2e-12, atol=8e-13)
    np.testing.assert_allclose(actual.p, expected.p, rtol=2e-12, atol=8e-12)
    np.testing.assert_array_equal(actual.q[:, 0], initial(beads, True).q[:, 0])
    np.testing.assert_array_equal(actual.p[:, 0], 0.)


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("fail_at", [1, 2])
def test_auto_sampler_force_failure_preserves_state_rng_and_exact_retry(scheme, fail_at):
    provider = CoupledPotential(True, fail_at)
    failing = dynamics(32, scheme, True, potential=provider)
    reference = dynamics(32, scheme, True)
    actual, expected = initial(32, True), initial(32, True)
    original, rng_state = actual.copy(), copy.deepcopy(failing.rng.bit_generator.state)
    with pytest.raises(PotentialError, match="injected force failure"):
        failing.sample_step(actual, gamma_centroid=0.7)
    np.testing.assert_array_equal(actual.q, original.q)
    np.testing.assert_array_equal(actual.p, original.p)
    assert (actual.step, actual.time) == (original.step, original.time)
    assert failing.rng.bit_generator.state == rng_state
    assert provider.calls == fail_at
    failing.potential = CoupledPotential(True)
    failing.sample_step(actual, gamma_centroid=0.7)
    reference.sample_step(expected, gamma_centroid=0.7)
    np.testing.assert_array_equal(actual.q, expected.q)
    np.testing.assert_array_equal(actual.p, expected.p)
    assert failing.rng.bit_generator.state == reference.rng.bit_generator.state


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("hook,count", [("half_step", 2), ("_fourier_step", 2), ("_check", 3), ("_project_fixed", 4), ("_check_fixed", 1)])
@pytest.mark.parametrize("mode", ["instance", "class", "subclass"])
def test_sampling_overrides_retain_python_dispatch(monkeypatch, scheme, hook, count, mode):
    dyn = dynamics(8, scheme, True)
    state = initial(8, True)
    reference = dynamics(8, scheme, True, loop_backend="python")
    expected = initial(8, True)
    reference.sample_step(expected)
    target = dyn if hook in ("_project_fixed", "_check_fixed") else dyn.integrator
    original = getattr(type(target), hook)
    calls = []

    def changed(self, *args, **kwargs):
        calls.append(True)
        return original(self, *args, **kwargs)

    if mode == "instance":
        monkeypatch.setattr(target, hook, MethodType(changed, target))
    elif mode == "class":
        monkeypatch.setattr(type(target), hook, changed)
    else:
        target.__class__ = type("CustomizedSampling", (type(target),), {hook: changed})

    def forbidden(*args):
        raise AssertionError("native sampler bypassed a Python extension point")

    monkeypatch.setattr(native, "sample_free", forbidden)
    dyn.sample_step(state)
    assert len(calls) == count
    np.testing.assert_array_equal(state.q, expected.q)
    np.testing.assert_array_equal(state.p, expected.p)
    assert dyn.rng.bit_generator.state == reference.rng.bit_generator.state


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("hook", ["half_step", "_fourier_step", "_check", "_project_fixed", "_check_fixed"])
@pytest.mark.parametrize("mode", ["instance", "class"])
def test_original_method_rebound_to_another_object_still_uses_python(monkeypatch, scheme, hook, mode):
    dyn = dynamics(8, scheme, True)
    reference = dynamics(8, scheme, True, loop_backend="python")
    other = dynamics(8, scheme, True)
    if hook in ("half_step", "_fourier_step", "_check"):
        # The __func__ is unchanged, but this half map has a different dt.
        # Function identity alone is therefore not a sufficient dispatch gate.
        other.integrator = CayleyIntegrator(MASSES, BETA, 8, 2 * DT, native="required", scheme=scheme)
        target, reference_target, owner = dyn.integrator, reference.integrator, other.integrator
    else:
        target, reference_target, owner = dyn, reference, other
    borrowed = getattr(owner, hook)
    setattr(reference_target, hook, borrowed)
    expected = initial(8, True)
    reference.sample_step(expected)
    if mode == "instance":
        monkeypatch.setattr(target, hook, borrowed)
    else:
        monkeypatch.setattr(type(target), hook, borrowed)

    def forbidden(*args):
        raise AssertionError("native sampler bypassed a method bound to a different object")

    monkeypatch.setattr(native, "sample_free", forbidden)
    actual = initial(8, True)
    dyn.sample_step(actual)
    np.testing.assert_array_equal(actual.q, expected.q)
    np.testing.assert_array_equal(actual.p, expected.p)
    assert dyn.rng.bit_generator.state == reference.rng.bit_generator.state


@pytest.mark.parametrize("backend,native_enabled", [("python", True), ("auto", False)])
def test_explicit_python_sampling_avoids_native_kernel(monkeypatch, backend, native_enabled):
    dyn = Dynamics(FreePotential(), MASSES, BETA, 32, DT, loop_backend=backend, native=native_enabled)

    def forbidden(*args):
        raise AssertionError("Python sampling policy entered the native kernel")

    monkeypatch.setattr(native, "sample_free", forbidden)
    dyn.sample_step(initial(32))


def free_energy(state, masses, beta):
    delta = state.q - np.roll(state.q, 1, axis=0)
    return np.sum(state.p**2 / (2 * masses[None, :, None])) + 0.5 * (len(state.q) / beta)**2 * np.sum(masses[None, :, None] * delta**2)


@pytest.mark.parametrize("beads", [32, 64, 128, 256])
@pytest.mark.parametrize("dt", [0.003, 0.173])
def test_large_normal_mode_free_map_matches_fft_preserves_energy_and_reverses(beads, dt):
    compiled = CayleyIntegrator(MASSES, BETA, beads, dt, native="required", scheme="normal-mode-bab")
    fft = CayleyIntegrator(MASSES, BETA, beads, dt, native=False, scheme="normal-mode-bab")
    original, actual, expected = initial(beads), initial(beads), initial(beads)
    energy = free_energy(original, MASSES, BETA)
    for _ in range(40):
        compiled.free_step(actual.q, actual.p)
        fft.free_step(expected.q, expected.p)
    np.testing.assert_allclose(actual.q, expected.q, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(actual.p, expected.p, rtol=3e-11, atol=2e-9)
    assert abs(free_energy(actual, MASSES, BETA) / energy - 1) < 2e-12
    actual.p *= -1
    for _ in range(40):
        compiled.free_step(actual.q, actual.p)
    np.testing.assert_allclose(actual.q, original.q, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(actual.p, -original.p, rtol=3e-11, atol=2e-9)


@pytest.mark.skipif(not native.has_openmp(), reason="OpenMP build required")
@pytest.mark.parametrize("scheme", SCHEMES)
def test_native_sampling_thread_count_does_not_change_draws_or_arithmetic(scheme):
    single = dynamics(64, scheme, True, threads=1)
    threaded = dynamics(64, scheme, True, threads=3)
    a, b = initial(64, True), initial(64, True)
    for _ in range(4):
        single.sample_step(a)
        threaded.sample_step(b)
    np.testing.assert_array_equal(a.q, b.q)
    np.testing.assert_array_equal(a.p, b.p)
    assert single.rng.bit_generator.state == threaded.rng.bit_generator.state


@pytest.fixture(scope="module")
def harmonic_cabi(tmp_path_factory):
    compiler = shutil.which("cc") or shutil.which("gcc")
    if compiler is None:
        pytest.skip("C compiler required for native block fixture")
    root = Path(__file__).resolve().parents[1]
    library = tmp_path_factory.mktemp("sampling-native-pes") / "harmonic.so"
    subprocess.run([
        compiler, "-shared", "-fPIC", "-I", str(root / "native/include"),
        str(root / "native/examples/harmonic_pes.c"), "-o", str(library),
    ], check=True)
    provider = CABIPotential(library)
    yield provider
    provider.close()


@pytest.mark.parametrize("beads", [32, 64, 256])
@pytest.mark.parametrize("api", ["propagate", "propagate_block"])
def test_large_normal_mode_native_blocks_match_fft_and_are_split_invariant(harmonic_cabi, beads, api):
    dyn = Dynamics(harmonic_cabi, [1.7], 8., beads, 0.01, integrator="normal-mode-bab", native="required")
    reference = Dynamics(HarmonicPotential(), [1.7], 8., beads, 0.01, integrator="normal-mode-bab", native=False)
    rng = np.random.default_rng(675)
    original = State(rng.normal(0, 0.3, (beads, 1, 3)), rng.normal(size=(beads, 1, 3)))
    whole, split, expected = original.copy(), original.copy(), original.copy()
    initial_energy = dyn.energy(original)
    result = getattr(dyn, api)(whole, 17, backend="native")
    pieces = [getattr(dyn, api)(split, count, backend="native") for count in (1, 5, 3, 8)]
    for _ in range(17):
        reference.step(expected)
    assert result.backend == "native"
    np.testing.assert_array_equal(whole.q, split.q)
    np.testing.assert_array_equal(whole.p, split.p)
    np.testing.assert_array_equal(result.energies, np.concatenate([part.energies for part in pieces]))
    assert (whole.step, whole.time) == (split.step, split.time)
    np.testing.assert_allclose(whole.q, expected.q, rtol=2e-12, atol=3e-12)
    np.testing.assert_allclose(whole.p, expected.p, rtol=2e-12, atol=3e-11)
    assert abs(dyn.energy(whole) / initial_energy - 1) < 2e-4
    whole.p *= -1
    getattr(dyn, api)(whole, 17, backend="native")
    np.testing.assert_allclose(whole.q, original.q, rtol=2e-12, atol=3e-12)
    np.testing.assert_allclose(whole.p, -original.p, rtol=2e-12, atol=3e-11)
