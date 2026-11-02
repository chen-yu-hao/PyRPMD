"""Independent Hamiltonian, tensor, and response checks for spectroscopy."""

import copy
from typing import Dict

import numpy as np
import pytest
from scipy.linalg import expm

from pyrpmd.cmd import (PACMDDynamics, centroid_force_training_data,
                        fit_quadratic_centroid_pmf, ibi_update)
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.potentials import FreePotential, HarmonicPotential
from pyrpmd.potentials.base import PotentialError, check_potential
from pyrpmd.properties import (CallablePropertyModel, PropertyResult,
                               TorchScriptPropertyModel, evaluate_property_trajectory,
                               validate_property_result)
from pyrpmd.analysis.spectroscopy import (bead_averaged_polarizabilities,
                                         relative_raman_spectrum,
                                         relative_sfg_spectrum)


@pytest.mark.parametrize("beads", [1, 2, 7, 8])
def test_pacmd_free_step_matches_dense_hamiltonian_flow(beads):
    beta, mass, omega, dt = 3.2, 2.1, 13.7, .07
    eye = np.eye(beads)
    laplacian = 2 * eye - np.roll(eye, 1, axis=0) - np.roll(eye, -1, axis=0)
    p0 = np.ones((beads, beads)) / beads
    spring = mass * (beads / beta) ** 2 * laplacian
    kinetic_mass = mass * p0 + spring / omega ** 2
    generator = np.block([[np.zeros_like(eye), np.linalg.inv(kinetic_mass)],
                          [-spring, np.zeros_like(eye)]])
    rng = np.random.default_rng(213)
    q, p = rng.normal(size=(2, beads, 1, 3))
    expected = expm(dt * generator) @ np.concatenate([q[:, 0], p[:, 0]], axis=0)
    state = State(q, p)
    dynamics = PACMDDynamics(FreePotential(), [mass], beta, beads, dt,
                             target_frequency=omega, seed=21)
    initial_energy = dynamics.energy(state)
    dynamics.hamiltonian_step(state)
    np.testing.assert_allclose(state.q[:, 0], expected[:beads], rtol=1e-12, atol=3e-12)
    np.testing.assert_allclose(state.p[:, 0], expected[beads:], rtol=1e-12, atol=3e-12)
    assert dynamics.energy(state) == pytest.approx(initial_energy, rel=2e-13, abs=2e-13)


@pytest.mark.parametrize("omega,friction", [(8., .2), (30., 1.3)])
def test_pacmd_harmonic_centroid_is_physical_and_independent_of_adiabaticity(omega, friction):
    rng = np.random.default_rng(42)
    q, p = rng.normal(size=(2, 8, 1, 3))
    actual, reference = State(q, p), State(q, p)
    cmd = PACMDDynamics(HarmonicPotential(.7), [1.5], 3.7, 8, .01,
                        target_frequency=omega, friction_scale=friction, seed=88)
    rpmd = Dynamics(HarmonicPotential(.7), [1.5], 3.7, 8, .01,
                    integrator="normal-mode-bab", native=False)
    for _ in range(200):
        cmd.step(actual)
        rpmd.step(reference)
    np.testing.assert_allclose(actual.q.mean(0), reference.q.mean(0), atol=3e-13)
    np.testing.assert_allclose(actual.p.mean(0), reference.p.mean(0), atol=3e-13)
    np.testing.assert_allclose(cmd.velocities(actual.p).mean(0), actual.p.mean(0) / 1.5, atol=1e-13)


def test_pacmd_preserves_free_ring_canonical_covariance():
    beads, atoms, beta, mass, target = 8, 3500, 3.1, 2.3, 17.
    eye = np.eye(beads)
    laplacian = 2 * eye - np.roll(eye, 1, axis=0) - np.roll(eye, -1, axis=0)
    eigenvalues, basis = np.linalg.eigh(laplacian)
    spring = mass * (beads / beta) ** 2 * eigenvalues[1:]
    mu = spring / target ** 2
    rng = np.random.default_rng(115)
    qk, pk = np.zeros((beads, atoms, 3)), np.zeros((beads, atoms, 3))
    qk[1:] = rng.normal(size=qk[1:].shape) * np.sqrt(beads / (beta * spring[:, None, None]))
    pk[1:] = rng.normal(size=pk[1:].shape) * np.sqrt(beads * mu[:, None, None] / beta)
    state = State(np.einsum("bk,kac->bac", basis, qk), np.einsum("bk,kac->bac", basis, pk))
    dynamics = PACMDDynamics(FreePotential(), np.full(atoms, mass), beta, beads, .12,
                             target_frequency=target, seed=274)
    for _ in range(3):
        dynamics.step(state)
    qk = np.einsum("bk,bac->kac", basis, state.q)
    pk = np.einsum("bk,bac->kac", basis, state.p)
    np.testing.assert_allclose(np.mean(qk[1:] ** 2, axis=(1, 2)), beads / (beta * spring), rtol=.045)
    np.testing.assert_allclose(np.mean(pk[1:] ** 2, axis=(1, 2)), beads * mu / beta, rtol=.045)
    np.testing.assert_allclose(pk[0], 0, atol=4e-14)
    normalized_cross = np.mean(qk[1:] * pk[1:], axis=(1, 2)) * beta * target / beads
    assert np.max(np.abs(normalized_cross)) < .04


def test_pacmd_classical_limit_and_rng():
    state = State(np.ones((1, 1, 3)), np.full((1, 1, 3), .6))
    ref = state.copy()
    cmd = PACMDDynamics(HarmonicPotential(.4), [2.], 3., 1, .04, target_frequency=14., seed=5)
    ordinary = Dynamics(HarmonicPotential(.4), [2.], 3., 1, .04,
                        integrator="normal-mode-bab", native=False)
    rng_before = copy.deepcopy(cmd.rng.bit_generator.state)
    for _ in range(50):
        cmd.step(state)
        ordinary.step(ref)
    np.testing.assert_allclose(state.q, ref.q, atol=3e-15)
    np.testing.assert_allclose(state.p, ref.p, atol=3e-15)
    assert cmd.rng.bit_generator.state == rng_before


def test_pacmd_failure_restores_state_clock_and_rng():
    class FailSecond(HarmonicPotential):
        def __init__(self):
            super().__init__(.7)
            self.calls = 0

        def evaluate(self, q):
            self.calls += 1
            if self.calls == 2:
                raise PotentialError("injected model failure")
            return super().evaluate(q)

    cmd = PACMDDynamics(FailSecond(), [1.], 3., 8, .01, target_frequency=12., seed=19)
    state = State(np.ones((8, 1, 3)), np.zeros((8, 1, 3)))
    before = state.copy()
    rng_before = copy.deepcopy(cmd.rng.bit_generator.state)
    with pytest.raises(PotentialError, match="injected"):
        cmd.step(state)
    np.testing.assert_array_equal(state.q, before.q)
    np.testing.assert_array_equal(state.p, before.p)
    assert (state.step, state.time) == (before.step, before.time)
    assert cmd.rng.bit_generator.state == rng_before
    cmd.potential = HarmonicPotential(.7)
    other = PACMDDynamics(cmd.potential, [1.], 3., 8, .01, target_frequency=12., seed=19)
    cmd.step(state)
    other.step(before)
    np.testing.assert_array_equal(state.q, before.q)
    np.testing.assert_array_equal(state.p, before.p)


@pytest.mark.parametrize("options", [
    {"target_frequency": 0}, {"target_frequency": True}, {"target_frequency": np.inf},
    {"friction_scale": -1}, {"friction_scale": True}, {"dt": -1}, {"nbeads": True},
])
def test_pacmd_rejects_invalid_scales(options):
    args = dict(potential=FreePotential(), masses=[1.], beta=3., nbeads=8, dt=.01, target_frequency=10.)
    args.update(options)
    with pytest.raises(ValueError):
        PACMDDynamics(**args)


def test_force_matching_recovers_coupled_conservative_centroid_pmf():
    rng = np.random.default_rng(731)
    q = rng.normal(size=(100, 8, 2, 3))
    matrix = rng.normal(size=(6, 6))
    hessian = matrix.T @ matrix + np.eye(6)
    shift = rng.normal(size=6)
    force = -(q.reshape(100, 8, 6) @ hessian + shift).reshape(q.shape)
    fit = fit_quadratic_centroid_pmf(q, force, beta=2.3)
    held_out = rng.normal(size=(12, 2, 3))
    expected = -(held_out.reshape(12, 6) @ hessian + shift).reshape(held_out.shape)
    np.testing.assert_allclose(fit.potential.evaluate(held_out).forces, expected, atol=5e-14)
    assert fit.force_rmse < 2e-14
    assert fit.rank == fit.parameters
    assert check_potential(fit.potential, held_out[:1])["max_absolute_force_error"] < 2e-8
    data = centroid_force_training_data(q, force)
    np.testing.assert_allclose(data["mean_forces"], force.mean(axis=1))


def test_force_matching_rejects_unidentified_basis():
    q = np.ones((4, 8, 1, 3))
    with pytest.raises(ValueError, match="do not identify"):
        fit_quadratic_centroid_pmf(q, -q, beta=2.)


def test_ibi_update_has_correct_boltzmann_sign_and_fixed_point():
    x = np.linspace(-2, 2, 80)
    beta = 2.1
    target = np.exp(-beta * .5 * 1.7 * x ** 2)
    sampled = np.exp(-beta * .5 * .8 * x ** 2)
    update = ibi_update(np.zeros_like(x), target, sampled, beta=beta)
    np.testing.assert_allclose(update, .5 * (1.7 - .8) * x ** 2, atol=3e-16)
    np.testing.assert_array_equal(ibi_update(update, target, target, beta=beta), update)
    target[0] = 0
    with pytest.raises(ValueError, match="zero-density"):
        ibi_update(update, target, sampled, beta=beta)


def test_property_contract_owns_outputs_and_checks_tensor_symmetry():
    dipole = np.ones((3, 3))
    alpha = np.broadcast_to(np.eye(3), (3, 3, 3)).copy()
    value = validate_property_result({"dipole": dipole, "polarizability": alpha}, 3)
    dipole[:] = 0
    alpha[:] = 0
    np.testing.assert_array_equal(value.dipole, 1)
    np.testing.assert_array_equal(value.polarizability[0], np.eye(3))
    with pytest.raises(ValueError, match="required"):
        validate_property_result({"dipole": dipole}, 3, require=("polarizability",))
    alpha[0, 0, 1] = 1
    with pytest.raises(ValueError, match="symmetric"):
        validate_property_result({"polarizability": alpha}, 3)


def test_property_trajectory_preserves_bead_nonlinearity_and_chunks():
    batches = []
    def evaluate(q):
        batches.append(len(q))
        return {"dipole": (q ** 2).sum(axis=1)}
    q = np.random.default_rng(17).normal(size=(13, 5, 2, 3))
    out = evaluate_property_trajectory(CallablePropertyModel(evaluate), q, chunk_frames=11, require=("dipole",))
    assert max(batches) == 11
    assert sum(batches) == 65
    np.testing.assert_allclose(out["dipole"], (q ** 2).sum(axis=2))
    assert not np.allclose(out["dipole"].mean(axis=1), (q.mean(axis=1) ** 2).sum(axis=1))


def _tensor_series():
    t = np.arange(512) * .2
    s = np.sin(2 * np.pi * 32 * np.arange(512) / 512)
    alpha = np.zeros((len(t), 2, 3, 3))
    alpha[:, :, 0, 0] = (2 * s)[:, None]
    alpha[:, :, 1, 1] = s[:, None]
    alpha[:, :, 2, 2] = -s[:, None]
    alpha[:, :, 0, 1] = alpha[:, :, 1, 0] = (.3 * s)[:, None]
    return t, s, alpha


def test_raman_rotational_invariance_and_polarization_invariants():
    _, _, alpha = _tensor_series()
    rotation, _ = np.linalg.qr(np.random.default_rng(53).normal(size=(3, 3)))
    rotated = np.einsum("ij,tpjk,lk->tpil", rotation, alpha, rotation)
    a = relative_raman_spectrum(alpha, .2, window="boxcar", normalize=False)
    b = relative_raman_spectrum(rotated, .2, window="boxcar", normalize=False)
    for key in ("isotropic", "anisotropic", "vv", "vh", "relative_raman"):
        np.testing.assert_allclose(a[key], b[key], atol=2e-12)
    np.testing.assert_allclose(a["vv"] + a["vh"], a["relative_raman"])
    assert np.argmax(a["relative_raman"]) == 32
    assert 0 < a["depolarization_ratio"][32] < .75


def test_raman_pure_isotropic_and_anisotropic_limits():
    _, s, _ = _tensor_series()
    isotropic = s[:, None, None, None] * np.eye(3)[None, None, :, :]
    a = relative_raman_spectrum(isotropic, .2, window="boxcar", normalize=False)
    assert np.max(np.abs(a["vh"])) < 1e-25
    traceless = s[:, None, None, None] * np.diag([1., -1., 0.])[None, None, :, :]
    b = relative_raman_spectrum(traceless, .2, window="boxcar", normalize=False)
    assert b["depolarization_ratio"][32] == pytest.approx(.75)
    assert b["isotropic"][32] == 0
    stokes = relative_raman_spectrum(traceless, .2, beta=4.2, window="boxcar", normalize=False)
    x = 4.2 * b["angular_frequency_au"][32]
    assert stokes["relative_raman"][32] / b["relative_raman"][32] == pytest.approx(x / -np.expm1(-x))


def test_sfg_causal_response_matches_analytic_harmonic_correlation():
    # Uniform independent phases make the ensemble C(t)=cos(omega*t)/2,
    # including at finite available-origin lags.
    dt, frames, cycles, replicas = .02, 1200, 6, 12
    omega = 2 * np.pi * cycles / (frames * dt)
    t = np.arange(frames) * dt
    phases = 2 * np.pi * np.arange(replicas) / replicas
    observable = np.cos(omega * t[None, :] + phases[:, None])
    mu = np.zeros((replicas, frames, 1, 3))
    alpha = np.zeros((replicas, frames, 1, 3, 3))
    mu[:, :, 0, 2] = observable
    alpha[:, :, 0, 0, 0] = observable
    result = relative_sfg_spectrum(mu, alpha, dt, beta=3., max_lag=300,
                                  window="boxcar", nfft=1024, demean=False)
    tau = result["lag_time_au"]
    np.testing.assert_allclose(result["correlation"], .5 * np.cos(omega * tau), atol=1e-14)
    response = 1.5 * omega * np.sin(omega * tau)
    response[[0, -1]] *= .5
    expected = dt * np.exp(1j * np.outer(result["angular_frequency_au"], tau)) @ response
    np.testing.assert_allclose(result["susceptibility"], expected, atol=.0015)


def test_sfg_orientation_sign_and_inversion_cancellation():
    t, s, alpha = _tensor_series()
    mu = np.zeros((len(t), 2, 3))
    mu[:, :, 2] = s[:, None]
    a = relative_sfg_spectrum(mu, alpha, .2, beta=2., max_lag=90)
    b = relative_sfg_spectrum(-mu, alpha, .2, beta=2., max_lag=90)
    np.testing.assert_allclose(a["susceptibility"], -b["susceptibility"], atol=1e-15)
    np.testing.assert_allclose(a["relative_sfg_intensity"], b["relative_sfg_intensity"], atol=1e-15)
    pair = relative_sfg_spectrum(np.stack([mu, -mu]), np.stack([alpha, alpha]), .2,
                                 beta=2., max_lag=90)
    np.testing.assert_allclose(pair["susceptibility"], 0, atol=1e-15)
    np.testing.assert_allclose(pair["relative_sfg_intensity"], 0, atol=1e-30)
    assert pair["imaginary_standard_error"] is not None


@pytest.mark.parametrize("function,args,kwargs", [
    (relative_raman_spectrum, (np.zeros((5, 1, 3, 3)), 1.), {"normalize": "yes"}),
    (relative_raman_spectrum, (np.zeros((5, 1, 3, 3)), 1.), {"laser_wavenumber_cm1": 1000}),
    (relative_sfg_spectrum, (np.zeros((5, 1, 3)), np.zeros((5, 1, 3, 3)), 1.), {"beta": 2., "max_lag": 1}),
    (relative_sfg_spectrum, (np.zeros((5, 1, 3)), np.zeros((5, 1, 3, 3)), 1.), {"beta": 2., "component": "bad"}),
])
def test_spectroscopy_validation(function, args, kwargs):
    with pytest.raises(ValueError):
        function(*args, **kwargs)


def test_torchscript_property_contract_runs_real_optional_torch(tmp_path):
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            x = inputs["positions"]
            mu = x.sum(dim=1)
            alpha = torch.einsum("bi,bj->bij", mu, mu)
            return {"dipole": mu, "polarizability": alpha}

    path = tmp_path / "properties.pt"
    torch.jit.script(Model()).save(str(path))
    model = TorchScriptPropertyModel(path, [1, 8], dtype="float64")
    q = np.random.default_rng(42).normal(size=(4, 2, 3))
    result = model.evaluate(q)
    np.testing.assert_allclose(result.dipole, q.sum(axis=1), atol=1e-15)
    np.testing.assert_allclose(result.polarizability, np.einsum("bi,bj->bij", q.sum(1), q.sum(1)), atol=1e-15)
