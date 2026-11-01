"""Independent matrix/statistical regressions; the paper percentage is not a golden value."""
from pathlib import Path
import importlib.util
import math
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from pyrpmd.torch_backend import TorchPES, TorchRPCTerm, TorchSplitPotential, TorchDynamics, TorchState

spec = importlib.util.spec_from_file_location(
    "paper_validation", Path(__file__).parents[1] / "benchmarks" / "torch_paper_validation.py"
)
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("P,C", [(p, c) for p in (1,2,3,4,5,8,16,31,32) for c in range(1,p+1)])
def test_rpc_against_independent_real_basis(P, C):
    term = TorchRPCTerm(TorchPES(ref.IsotropicHarmonic()), P, C)
    matrix = term.matrix.numpy()
    np.testing.assert_allclose(matrix, ref.reference_contraction(P,C), atol=7e-15, rtol=0)
    np.testing.assert_allclose(matrix @ np.ones(P), np.ones(C), atol=7e-15, rtol=0)
    # Adjoint scaling is part of the Hamiltonian, not just interpolation.
    np.testing.assert_allclose((P/C) * matrix.T @ np.ones(C), np.ones(P), atol=7e-15, rtol=0)
    if P == C:
        np.testing.assert_array_equal(matrix, np.eye(P))


@pytest.mark.parametrize("scheme", ref.SCHEMES)
def test_harmonic_long_time_against_matrix_exponential_or_cayley_solve(scheme):
    P, beta, dt, nsteps = 16, 1., .1, 1000
    x, p = ref.harmonic_sample(12, P, beta, np.random.default_rng(71))
    xp = np.stack([np.fft.rfft(x,norm="ortho"),np.fft.rfft(p,norm="ortho")],axis=-1)
    maps = np.array([np.linalg.matrix_power(ref.harmonic_matrix(w,dt,scheme),nsteps)
                    for w in ref.frequencies(P,beta)])
    expected = np.einsum("kij,bkj->bki",maps,xp)
    d = TorchDynamics(TorchPES(ref.Polynomial()), [1.], beta, P, dt, integrator=scheme)
    s = ref.state_1d(x,p); d.propagate(s,nsteps)
    np.testing.assert_allclose(s.q[:,:,0,0],np.fft.irfft(expected[:,:,0],n=P,norm="ortho"),atol=1e-9,rtol=1e-10)
    np.testing.assert_allclose(s.p[:,:,0,0],np.fft.irfft(expected[:,:,1],n=P,norm="ortho"),atol=1e-9,rtol=1e-10)


@pytest.mark.parametrize("scheme", ref.SCHEMES)
@pytest.mark.parametrize("P", [1,5,8])
def test_coupled_unequal_mass_mts_is_symplectic_and_reversible(P, scheme):
    class Coupled(torch.nn.Module):
        metadata = {"batch_independent": True}
        def __init__(self, scale):
            super().__init__()
            K = torch.eye(6,dtype=torch.float64)
            K[0,3] = K[3,0] = -.3
            self.register_buffer("K", K*scale)
        def forward(self,q):
            x = q.reshape(len(q),6); g = x @ self.K
            return .5*(x*g).sum(1), -g.reshape_as(q)
    D = P*6; eye=np.eye(D); phase_eye=np.eye(2*D)
    q=phase_eye[:,:D].reshape(2*D,P,2,3); p=phase_eye[:,D:].reshape(2*D,P,2,3)
    split=TorchSplitPotential([TorchRPCTerm(TorchPES(Coupled(.9)),P)],
                              [TorchRPCTerm(TorchPES(Coupled(.1)),P,min(P,3))])
    d=TorchDynamics(split,[1.,5.],4.,P,.07,inner_steps=3,integrator=scheme)
    s=TorchState.from_arrays(q,p);d.propagate(s,5)
    M=np.concatenate([s.q.numpy().reshape(2*D,D),s.p.numpy().reshape(2*D,D)],axis=1).T
    J=np.block([[np.zeros_like(eye),eye],[-eye,np.zeros_like(eye)]])
    R=np.diag(np.r_[np.ones(D),-np.ones(D)])
    np.testing.assert_allclose(M.T@J@M,J,atol=4e-14,rtol=0)
    np.testing.assert_allclose(R@M@R@M,np.eye(2*D),atol=5e-14,rtol=0)


@pytest.mark.parametrize("P,dt", [(1,.1),(8,.1),(16,.02),(32,.02)])
@pytest.mark.parametrize("scheme", ref.SCHEMES)
def test_cold_start_pile_matches_independent_finite_time_covariance(P, dt, scheme):
    row=ref.sampling_case(P,dt,scheme,n=512,time_total=16.,seed=311)
    assert abs(row["z_vs_finite_time"]) < 5.0
    assert row["discrete_stationary_q2"] > row["finite_P_q2"]
    # With this OBCBO splitting the harmonic finite-step bias is not zero.
    if scheme == "cayley-bcb":
        assert row["discrete_stationary_q2"] == pytest.approx(
            row["finite_P_q2"]/(1-dt*dt/4),rel=2e-12)


@pytest.mark.parametrize("scheme", ref.SCHEMES)
def test_resonant_free_mode_cold_start(scheme):
    P,beta,n=8,8.,2048
    d=TorchDynamics(TorchPES(ref.Polynomial(quadratic=0)),[1.],beta,P,math.pi/2,
                    integrator=scheme,thermostat="trpmd",seed=1729)
    zero=torch.zeros((n,P,1,3),dtype=torch.float64);s=TorchState(zero,zero)
    d.propagate(s,100)
    # Nyquist mode has omega=2; exact free evolution has sin(omega*dt)=0.
    qk=torch.fft.rfft(s.q,dim=1,norm="ortho")[:,-1,0,:].real
    q2=qk.square().mean(1)
    if scheme == "normal-mode-bab":
        assert q2.mean().item() < 1e-24
    else:
        se=q2.std(unbiased=True).item()/math.sqrt(n)
        assert abs(q2.mean().item()-.25) < 5*se
    assert s.q.mean(1).abs().max().item() < 2e-12  # TRPMD leaves centroid alone.


def test_mts_accuracy_and_equal_inner_timestep_cost():
    result=ref.mts_sweep()
    for case in result["convergence"]:
        assert all(3.8 < ratio < 4.2 for ratio in case["error_ratios"])
        for row in case["rows"]:
            assert row["force_evaluations"] == {
                "fast":row["steps"]*row["inner_steps"]+1,"slow":row["steps"]+1}
    full,mts,rpc=result["equal_inner_step_cost"]
    assert [row["slow_calls"] for row in (full,mts,rpc)] == [401,101,101]
    assert [row["slow_frames"] for row in (full,mts,rpc)] == [12832,3232,808]
    assert [row["fast_calls"] for row in (full,mts,rpc)] == [401]*3


@pytest.mark.parametrize("C", [1,3,8])
@pytest.mark.parametrize("scheme", ref.SCHEMES)
def test_rpc_mts_pile_joint_cold_start(C, scheme):
    row=ref.rpc_sampling_case(C,scheme,n=512,duration=16.)
    assert abs(row["z_vs_finite_time"]) < 5.0
    assert row["discrete_stationary_q2"] > 0
