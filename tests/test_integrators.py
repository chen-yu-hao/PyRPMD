"""Independent mechanical checks for each selectable RPMD integrator.

The references are a dense Hamiltonian matrix exponential and direct forces
from the cyclic graph Laplacian, not the production FFT/Cayley code.
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import copy
import shutil
import subprocess

import numpy as np
import pytest
from scipy.linalg import expm

from pyrpmd.dynamics import (
    CayleyIntegrator, Dynamics, State, INTEGRATORS, normalize_integrator,
    native_capabilities,
)
from pyrpmd.cv import Cartesian, CVStack
from pyrpmd.bias import UmbrellaBias
from pyrpmd.potentials import FreePotential, HarmonicPotential, CABIPotential
from pyrpmd.propagation import loop_options, scientific_config


def laplacian(beads):
    eye = np.eye(beads)
    return 2 * eye - np.roll(eye, 1, axis=0) - np.roll(eye, -1, axis=0)


def exact_harmonic(state, mass, beta, duration, stiffness=0.0):
    beads = len(state.q)
    eye = np.eye(beads)
    matrix = np.block([
        [np.zeros_like(eye), eye / mass],
        [-mass * (beads / beta) ** 2 * laplacian(beads) - stiffness * eye,
         np.zeros_like(eye)],
    ])
    result = expm(duration * matrix) @ np.concatenate((state.q[:, 0], state.p[:, 0]))
    return result[:beads, None, :], result[beads:, None, :]


def initial(beads=5):
    rng = np.random.default_rng(1846)
    return State(rng.normal(size=(beads, 1, 3)), rng.normal(size=(beads, 1, 3)))


native_choices = [False] + ([True] if native_capabilities()["available"] else [])


@pytest.mark.parametrize("beads", [1, 2, 3, 8, 17, 32])
@pytest.mark.parametrize("native", native_choices)
def test_exact_normal_modes_match_dense_matrix_exponential(beads, native):
    state = initial(beads)
    q, p = exact_harmonic(state, 1.7, 2.8, 0.31)
    dynamics = Dynamics(FreePotential(), [1.7], 2.8, beads, 0.31,
                        integrator="normal-mode-bab", native=native)
    dynamics.step(state)
    np.testing.assert_allclose(state.q, q, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(state.p, p, atol=5e-11, rtol=2e-12)


@pytest.mark.parametrize("scheme", INTEGRATORS)
@pytest.mark.parametrize("native", native_choices)
def test_all_schemes_have_second_order_harmonic_convergence(scheme, native):
    state = initial()
    qr, pr = exact_harmonic(state, 1.7, 2.8, 0.4, stiffness=0.7)
    errors = []
    for dt in (0.04, 0.02, 0.01):
        dynamics = Dynamics(HarmonicPotential(0.7), [1.7], 2.8, 5, dt,
                            integrator=scheme, native=native)
        trial = state.copy()
        for _ in range(round(0.4 / dt)):
            dynamics.step(trial)
        errors.append(np.linalg.norm(trial.q-qr) + np.linalg.norm(trial.p-pr))
    assert 3.7 < errors[0] / errors[1] < 4.3
    assert 3.7 < errors[1] / errors[2] < 4.3


@pytest.mark.parametrize("scheme", INTEGRATORS)
@pytest.mark.parametrize("native", native_choices)
def test_time_reversal_and_real_mass_hamiltonian(scheme, native):
    dynamics = Dynamics(HarmonicPotential(0.7), [1.7], 2.8, 5, 0.01,
                        integrator=scheme, native=native)
    state = initial()
    old = state.copy()
    energy = dynamics.energy(state)
    for _ in range(60):
        dynamics.step(state)
    assert abs(dynamics.energy(state) / energy - 1) < 1e-3
    state.p *= -1
    for _ in range(60):
        dynamics.step(state)
    np.testing.assert_allclose(state.q, old.q, atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(state.p, -old.p, atol=8e-11, rtol=2e-11)


@pytest.mark.parametrize("beads", [1, 2, 7])
@pytest.mark.parametrize("native", native_choices)
def test_verlet_uses_total_spring_and_physical_forces(beads, native):
    mass, beta, dt, stiffness = 1.7, 2.8, 0.03, 0.7
    state = initial(beads)
    matrix = mass * (beads/beta)**2 * laplacian(beads) + stiffness * np.eye(beads)
    half_p = state.p[:, 0] - dt/2 * matrix @ state.q[:, 0]
    q = state.q[:, 0] + dt * half_p/mass
    p = half_p - dt/2 * matrix @ q
    dynamics = Dynamics(HarmonicPotential(stiffness), [mass], beta, beads, dt,
                        integrator="velocity-verlet", native=native)
    dynamics.step(state)
    np.testing.assert_allclose(state.q[:, 0], q, atol=3e-14)
    np.testing.assert_allclose(state.p[:, 0], p, atol=3e-14)


@pytest.mark.parametrize("native", native_choices)
def test_p1_is_identical_classical_velocity_verlet(native):
    outputs = []
    for scheme in INTEGRATORS:
        dynamics = Dynamics(HarmonicPotential(0.7), [1.7], 2.8, 1, 0.03,
                            integrator=scheme, native=native)
        state = initial(1)
        dynamics.step(state)
        outputs.append(state)
    for state in outputs[1:]:
        np.testing.assert_allclose(state.q, outputs[0].q, atol=2e-15)
        np.testing.assert_allclose(state.p, outputs[0].p, atol=2e-15)


@pytest.mark.parametrize("native", native_choices)
def test_spring_stability_bound_and_exact_free_ring_energy(native):
    # P=8, beta=2 has omega_max=8: VV must be strictly below dt=0.25.
    for dt in (0.25, 0.3, -0.25):
        with pytest.raises(ValueError, match="ring stability"):
            Dynamics(FreePotential(), [1.7], 2.0, 8, dt,
                     integrator="velocity-verlet", native=native)
    for scheme in ("cayley-bcb", "normal-mode-bab"):
        dynamics = Dynamics(FreePotential(), [1.7], 2.0, 8, 0.7,
                            integrator=scheme, native=native)
        state = initial(8)
        energy = dynamics.energy(state)
        for _ in range(100):
            dynamics.step(state)
        assert abs(dynamics.energy(state) / energy - 1) < 3e-10


@pytest.fixture(scope="module")
def cabi_pes(tmp_path_factory):
    if not native_capabilities()["available"] or shutil.which("cc") is None:
        pytest.skip("compiled extension and C compiler required")
    root = Path(__file__).resolve().parents[1]
    library = tmp_path_factory.mktemp("integrator-pes") / "harmonic.so"
    subprocess.run([
        "cc", "-shared", "-fPIC", "-I", str(root / "native/include"),
        str(root / "native/examples/harmonic_pes.c"), "-o", str(library),
    ], check=True)
    potential = CABIPotential(library)
    yield potential
    potential.close()


@pytest.mark.parametrize("api", ["propagate", "propagate_block"])
@pytest.mark.parametrize("scheme", INTEGRATORS)
def test_fused_native_schemes_and_observations_match_reference(cabi_pes, api, scheme):
    dynamics = Dynamics(cabi_pes, [1.7], 2.8, 5, 0.01, integrator=scheme)
    cv = CVStack([Cartesian(0, "x"), Cartesian(0, "y")])
    bias = UmbrellaBias(cv, [0.2, -0.1], [[0.03, 0.004], [0.004, 0.07]])
    a, b = initial(), initial()
    options = {"observe": cv} if api == "propagate" else {"cv": cv, "record_states": True}
    actual = getattr(dynamics, api)(a, 23, bias=bias, backend="native", **options)
    reference = getattr(dynamics, api)(b, 23, bias=bias, backend="python", **options)
    assert actual.backend == "native"
    np.testing.assert_allclose(a.q, b.q, atol=3e-12, rtol=2e-12)
    np.testing.assert_allclose(a.p, b.p, atol=8e-12, rtol=2e-12)
    np.testing.assert_allclose(actual.values, reference.values, atol=2e-12)
    np.testing.assert_allclose(actual.energies, reference.energies, atol=2e-11)


@pytest.mark.parametrize("scheme", INTEGRATORS)
def test_native_chunk_boundaries_and_andersen_rng_are_exact(cabi_pes, scheme):
    a, b = initial(), initial()
    dynamics = Dynamics(cabi_pes, [1.7], 2.8, 5, 0.01, integrator=scheme)
    dynamics.propagate_block(a, 31, backend="native")
    for count in (2, 11, 1, 17):
        dynamics.propagate_block(b, count, backend="native")
    np.testing.assert_array_equal(a.q, b.q)
    np.testing.assert_array_equal(a.p, b.p)
    assert (a.step, a.time) == (b.step, b.time)
    d1 = Dynamics(cabi_pes, [1.7], 2.8, 5, 0.01, integrator=scheme, seed=77)
    d2 = Dynamics(cabi_pes, [1.7], 2.8, 5, 0.01, integrator=scheme, seed=77)
    a, b = initial(), initial()
    d1.thermalize(a, 31, interval=6)
    d2.thermalize(b, 11, interval=6)
    d2.thermalize(b, 20, interval=6, resample=False)
    np.testing.assert_array_equal(a.q, b.q)
    np.testing.assert_array_equal(a.p, b.p)
    assert d1.rng.bit_generator.state == d2.rng.bit_generator.state


@pytest.mark.parametrize("scheme", INTEGRATORS)
def test_native_centroid_rattle_with_each_selected_map(cabi_pes, scheme):
    from pyrpmd.workflows.rate import project_positions, project_momenta

    dynamics = Dynamics(cabi_pes, [1.7], 2.8, 5, 0.01, integrator=scheme)
    surface = Cartesian(0, "x")
    a = initial()
    a.q[:] = project_positions(a.q, surface, dynamics.masses)
    project_momenta(a.p, a.q, surface, dynamics.masses)
    b = a.copy()
    actual = dynamics.propagate_block(a, 19, constraint=surface, cv=surface,
                                     record_states=True, backend="native")
    reference = dynamics.propagate_block(b, 19, constraint=surface, cv=surface,
                                        record_states=True, backend="python")
    assert actual.backend == "native"
    assert np.max(np.abs(actual.values)) < 1e-10
    assert np.max(np.abs(actual.centroid_momenta[:, 0, 0])) < 1e-12
    np.testing.assert_allclose(actual.q, reference.q, atol=3e-12)
    np.testing.assert_allclose(actual.p, reference.p, atol=8e-12)


def test_names_workflow_selection_and_checkpoint_identity():
    assert normalize_integrator("normal_mode") == "normal-mode-bab"
    assert normalize_integrator("velocity_verlet") == "velocity-verlet"
    assert normalize_integrator("cayley") == "cayley-bcb"
    for bad in (None, 1, True, "BAOAB", "unknown"):
        with pytest.raises(ValueError, match="integrator must"):
            normalize_integrator(bad)
    configs = [{"dynamics": {"integrator": scheme, "block_steps": 7,
                            "loop_backend": "python"}} for scheme in INTEGRATORS]
    for config, scheme in zip(configs, INTEGRATORS):
        dynamics = Dynamics(FreePotential(), [1.0], 2.0, 2, 0.01, **loop_options(config))
        assert dynamics.integrator_name == scheme
        assert scientific_config(config) == {"dynamics": {"integrator": scheme}}


def test_stale_native_extension_cannot_silently_select_cayley():
    old = SimpleNamespace(has_openmp=lambda: False,
                          create=lambda *args: pytest.fail("old Cayley factory was called"))
    with patch("pyrpmd.dynamics.importlib.import_module", return_value=old):
        dynamics = Dynamics(FreePotential(), [1.0], 2.0, 2, 0.01,
                            integrator="normal-mode-bab")
        assert not dynamics.native
        assert "does not support normal-mode-bab" in dynamics.integrator.native_fallback_reason
        with pytest.raises(RuntimeError, match="does not support normal-mode-bab"):
            Dynamics(FreePotential(), [1.0], 2.0, 2, 0.01,
                     integrator="normal-mode-bab", native="required")
    with pytest.raises(RuntimeError, match="does not support normal-mode-bab"):
        dynamics.propagate(initial(2), 2, backend="native")


@pytest.mark.parametrize("scheme, method", [("cayley-bcb", "bcocb"), ("normal-mode-bab", "baoab")])
def test_stochastic_sampler_restores_rng_after_failed_second_force(scheme, method):
    class Fails(HarmonicPotential):
        calls = 0

        def evaluate(self, q):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("failed second force")
            return super().evaluate(q)

    dynamics = Dynamics(Fails(0.7), [1.7], 2.8, 5, 0.01,
                        integrator=scheme, seed=11, native=False)
    state, old = initial(), initial()
    rng = copy.deepcopy(dynamics.rng.bit_generator.state)
    with pytest.raises(RuntimeError, match="failed second force"):
        dynamics.sample_step(state, method=method)
    np.testing.assert_array_equal(state.q, old.q)
    np.testing.assert_array_equal(state.p, old.p)
    assert state.step == 0 and state.time == 0
    assert dynamics.rng.bit_generator.state == rng


def test_baoab_p1_matches_independent_scalar_langevin_step():
    mass, beta, dt, stiffness, gamma = 1.7, 2.8, 0.12, 0.7, 0.9
    dynamics = Dynamics(HarmonicPotential(stiffness), [mass], beta, 1, dt,
                        integrator="normal-mode-bab", seed=123, native=False)
    state = initial(1)
    p = state.p - dt/2 * stiffness * state.q
    q = state.q + dt/2 * p/mass
    decay = np.exp(-gamma*dt)
    p = decay*p + np.sqrt((1-decay**2)*mass/beta) * np.random.default_rng(123).normal(size=p.shape)
    q += dt/2 * p/mass
    p -= dt/2 * stiffness*q
    dynamics.sample_step(state, method="baoab", gamma_centroid=gamma)
    np.testing.assert_allclose(state.q, q, atol=2e-15)
    np.testing.assert_allclose(state.p, p, atol=2e-15)


def test_sampling_selection_is_explicit_and_strict_native_never_falls_back():
    dynamics = Dynamics(FreePotential(), [1.0], 2.0, 2, 0.01)
    with pytest.raises(ValueError, match="selected integrator"):
        dynamics.sample_step(initial(2), method="baoab")
    dynamics.loop_backend = "native"
    with pytest.raises(RuntimeError, match="Python/NumPy"):
        dynamics.sample_step(initial(2))
    verlet = Dynamics(FreePotential(), [1.0], 2.0, 2, 0.01, integrator="velocity-verlet")
    with pytest.raises(ValueError, match="Andersen"):
        verlet.sample_step(initial(2))
