"""Independent observable references, published covariances and failure boundaries.

Large statistical experiments live in benchmarks/torch_observable_validation.py;
these deterministic/short seeded regressions are suitable for the Torch CI job.
"""
from functools import lru_cache
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.special import gamma

torch = pytest.importorskip("torch")
from pyrpmd.torch_backend import TorchPES, TorchDynamics, TorchRPCTerm, TorchSplitPotential

_path = Path(__file__).resolve().parents[1] / "benchmarks" / "torch_observable_validation.py"
_spec = importlib.util.spec_from_file_location("torch_observable_validation", _path)
v = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v)


@pytest.fixture(autouse=True, scope="module")
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@lru_cache(None)
def reference(P, points=256):
    return v.transfer_quartic(P, points=points)


@pytest.mark.parametrize("P", [1, 3, 8, 16, 54])
def test_transfer_operator_primitive_virial_identity(P):
    # Independent finite-P integration by parts: both kinetic estimators agree.
    r = reference(P)
    assert r["kinetic_primitive"] == pytest.approx(r["kinetic_virial"], abs=2e-9)
    # Global scaling identity for a homogeneous quartic potential, even at finite P.
    assert r["kinetic_primitive"] == pytest.approx(2*r["potential"], abs=2e-9)


def test_classical_quartic_gamma_integrals():
    r = reference(1)
    assert r["q2"] == pytest.approx(2*gamma(.75)/gamma(.25), abs=2e-11)
    assert r["potential"] == pytest.approx(.25, abs=2e-11)
    assert r["kinetic_primitive"] == pytest.approx(.5, abs=2e-11)


@pytest.mark.parametrize("P", [1, 8, 16, 54])
def test_transfer_grid_convergence(P):
    a, b = reference(P, 256), reference(P, 512)
    for key in ("q2", "potential", "kinetic_virial", "kinetic_primitive"):
        assert a[key] == pytest.approx(b[key], abs=2e-9)


def test_quantum_and_finite_bead_references_converge_independently():
    coarse, fine = v.quantum_quartic(points=2048), v.quantum_quartic(points=4096)
    for key in ("q2", "potential", "kinetic"):
        assert abs(coarse[key]-fine[key]) < 6e-6
    assert abs(fine["kinetic"]-2*fine["potential"]) < 3e-6
    assert fine["highest_level_weight"] < 1e-100
    e8 = abs(reference(8)["kinetic_virial"]-fine["kinetic"])
    e16 = abs(reference(16)["kinetic_virial"]-fine["kinetic"])
    assert 3.8 < e8/e16 < 4.1  # primitive path-integral O(P^-2) convergence
    assert abs(reference(54)["kinetic_virial"]-fine["kinetic"]) < 6e-5


@pytest.mark.parametrize("P", [1, 8, 16])
def test_exact_quartic_rejection_sampler_against_transfer_operator(P):
    q, p, acceptance = v.quartic_iid(8192, P, 1., np.random.default_rng(440+P))
    assert 0 < acceptance < 1
    for key, values in v.observables(q, 1.).items():
        stat = v.summary(values)
        assert abs(stat["mean"]-reference(P)[key]) <= 6*stat["se"]+1e-10
    assert abs(np.mean(p*p)-P) < .08*P


@pytest.mark.parametrize("scheme", v.SCHEMES)
@pytest.mark.parametrize("mass,stiffness", [(1., .5), (1., 2.), (7., 2.)])
@pytest.mark.parametrize("omega", [0., .7, 3., 17.])
@pytest.mark.parametrize("dt", [.025, .17])
def test_2020_eq20_eq22_vs_independent_lyapunov(scheme, mass, stiffness, omega, dt):
    actual, L, Q = v.mode_covariance(omega, dt, mass, stiffness, 16, 3., scheme)
    paper = v.paper_mode_covariance(omega, dt, mass, stiffness, 16, 3., scheme)
    np.testing.assert_allclose(actual, paper, atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(paper-L@paper@L.T, Q, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("scheme", v.SCHEMES)
@pytest.mark.parametrize("P,k", [(1, 0), (5, 1), (8, 4), (16, 7)])
@pytest.mark.parametrize("mass", [1., 7.])
def test_production_map_vs_independent_matrix(scheme, P, k, mass):
    dt, beta, stiffness = .07, 2., .5
    basis = np.cos(2*np.pi*k*np.arange(P)/P); basis /= np.linalg.norm(basis)
    q = np.stack([basis, np.zeros(P)])
    p = np.stack([np.zeros(P), basis])
    s = v.state_1d(q, p)
    d = TorchDynamics(TorchPES(v.Oscillator(stiffness=stiffness, quartic=0.)),
                      [mass], beta, P, dt, integrator=scheme)
    d.step(s)
    matrix = np.stack([s.q[:, :, 0, 0].numpy()@basis, s.p[:, :, 0, 0].numpy()@basis])
    expected = v.mode_map(v.frequencies(P, beta)[k], dt, mass, stiffness, scheme)
    np.testing.assert_allclose(matrix, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("scheme", v.SCHEMES)
def test_production_sampling_against_finite_time_covariance(scheme):
    P, beta, dt, steps, n = 5, 2., .04, 20, 8192
    finite = []
    for w in v.frequencies(P, beta):
        C, L, _ = v.mode_covariance(w, dt, 1., 1., P, beta, scheme)
        Ln = np.linalg.matrix_power(L, steps)
        finite.append(C-Ln@C@Ln.T)
    q = np.zeros((n, P)); s = v.state_1d(q, q)
    d = TorchDynamics(TorchPES(v.Oscillator(stiffness=1., quartic=0.)), [1.], beta, P, dt,
                      integrator=scheme, thermostat="pile-l", seed=431)
    d.propagate(s, steps)
    for a, axis in [(s.q[:, :, 0, 0].numpy(), 0), (s.p[:, :, 0, 0].numpy(), 1)]:
        values = (a*a).mean(1)
        stat = v.summary(values)
        expected = sum(c[axis, axis] for c in finite)/P
        assert abs(stat["mean"]-expected) < 6*stat["se"]


@pytest.mark.parametrize("P", [8, 32, 128, 256])
def test_obcb_o_bias_is_not_dimension_free(P):
    # This is an EXPECTED algorithmic limitation, not an assertion of exact sampling.
    beta, dt = 8., .5
    w = v.frequencies(P, beta)
    exact = P/beta/(1+w*w)
    numeric = np.array([v.paper_mode_covariance(x, dt, 1., 1., P, beta, "cayley-bcb")[0, 0]
                        for x in w])
    np.testing.assert_allclose(numeric/exact, 16/15, rtol=1e-13)
    primitive = P/(2*beta)-(w*w*numeric).sum()/(2*P)
    target = P/(2*beta)-(w*w*exact).sum()/(2*P)
    assert primitive < target
    if P >= 128:
        assert primitive < 0  # warns against interpreting stability as correct statistics


def test_mts_resonance_is_not_removed_by_the_cayley_free_ring():
    P, beta, dt, inner, fast, slow = 1, 1., 3.0, 32, 1., .1
    model = TorchSplitPotential([TorchRPCTerm(TorchPES(v.Oscillator(stiffness=fast, quartic=0.)), P, P)],
                               [TorchRPCTerm(TorchPES(v.Oscillator(stiffness=slow, quartic=0.)), P, P)])
    d = TorchDynamics(model, [1.], beta, P, dt, inner_steps=inner, integrator="cayley-bcb")
    s = v.state_1d(np.array([[1.], [0.]]), np.array([[0.], [1.]]))
    d.step(s)
    actual = np.stack([s.q[:, 0, 0, 0].numpy(), s.p[:, 0, 0, 0].numpy()])
    K = np.array([[1., 0.], [-dt*slow/2, 1.]])
    expected = K@np.linalg.matrix_power(v.mode_map(0., dt/inner, 1., fast, "cayley-bcb"), inner)@K
    np.testing.assert_allclose(actual, expected, atol=2e-13)
    # Select a point inside the first MTS resonance band, NOT beyond the inner-step limit.
    assert max(abs(np.linalg.eigvals(expected))) > 1.1
    assert dt/inner*math.sqrt(fast) < 2
