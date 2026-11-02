"""BCOCB-RESPA maps, total OU duration, contraction, cost, rollback and sampling."""
from pathlib import Path
import importlib.util
import copy
import math
import numpy as np
import pytest
from scipy.linalg import expm, solve_discrete_lyapunov

torch = pytest.importorskip('torch')
from pyrpmd.torch_backend import TorchDynamics, TorchPES, TorchRPCTerm, TorchSplitPotential, TorchState

spec = importlib.util.spec_from_file_location('mixed_reference', Path(__file__).parents[1] / 'benchmarks/mixed_sampling_validation.py')
ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


class Harmonic(torch.nn.Module):
    metadata = {'batch_independent': True}
    def __init__(self, stiffness):
        super().__init__(); self.k = stiffness; self.calls = 0; self.fail_at = None
    def forward(self, q):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError('inner force failure')
        return self.k * q.square().sum((1, 2)) / 2, -self.k * q


def engine(P=5, M=3, C=3, dt=.14, method='cayley-bcocb', thermostat='pile-l', layout='rfft', device='cpu'):
    f, s = Harmonic(.9), Harmonic(.1)
    potential = TorchSplitPotential([TorchRPCTerm(TorchPES(f, device=device), P)],
                                   [TorchRPCTerm(TorchPES(s, device=device), P, C)])
    d = TorchDynamics(potential, [1.7], 3., P, dt, inner_steps=M, seed=21,
                      integrator=method, thermostat=thermostat, normal_mode_layout=layout)
    return d, f, s


def state_1d(q, p, device='cpu'):
    x = np.zeros((*q.shape, 1, 3)); y = np.zeros_like(x)
    x[:, :, 0, 0] = q; y[:, :, 0, 0] = p
    return TorchState.from_arrays(x, y, device=device)


@pytest.mark.parametrize('P,C', [(1,1),(4,1),(5,3),(8,4),(8,8)])
@pytest.mark.parametrize('M', [1,2,3,5])
@pytest.mark.parametrize('layout', ['rfft','rpmdrate'])
@pytest.mark.parametrize('thermostat', ['pile-l','trpmd'])
def test_injected_full_mean_and_accumulated_noise_maps(P, C, M, layout, thermostat, monkeypatch):
    d, _, _ = engine(P, M, C, layout=layout, thermostat=thermostat)
    phase = np.eye(2 * P)
    state = state_1d(phase[:, :P], phase[:, P:])
    with monkeypatch.context() as patch:
        patch.setattr(torch, 'randn', lambda shape, **kw: torch.zeros(shape, dtype=kw['dtype'], device=kw['device']))
        d.step(state)
    actual = np.column_stack([state.q[:, :, 0, 0], state.p[:, :, 0, 0]]).T
    L, Q, H = ref.mts_gaussian_map(P, 3., 1.7, .14, M, contracted=C, thermostat=thermostat)
    np.testing.assert_allclose(actual, L, atol=3e-13, rtol=3e-13)
    state = state_1d(np.zeros((M * P, P)), np.zeros((M * P, P)))
    calls = 0
    def injected(shape, **kw):
        nonlocal calls
        z = torch.zeros(shape, dtype=kw['dtype'], device=kw['device'])
        z[calls*P:(calls+1)*P, :, 0, 0] = torch.eye(P, dtype=z.dtype, device=z.device)
        calls += 1
        return z
    with monkeypatch.context() as patch:
        patch.setattr(torch, 'randn', injected); d.step(state)
    noise = np.column_stack([state.q[:, :, 0, 0], state.p[:, :, 0, 0]]).T
    assert calls == M
    np.testing.assert_allclose(noise, H, atol=3e-13, rtol=3e-12)
    np.testing.assert_allclose(noise @ noise.T, Q, atol=3e-13, rtol=3e-12)


@pytest.mark.parametrize('M', [2,3,4])
def test_inner_ou_duration_and_no_extra_expensive_calls(M):
    d, f, s = engine(M=M)
    state = state_1d(np.ones((4, 5)), np.zeros((4, 5)))
    d.propagate(state, 9)
    assert d.force_evaluations == {'fast': 9*M+1, 'slow': 10}
    assert (f.calls, s.calls) == (9*M+1, 10)
    assert d._decay[0, 0, 0, 0].item() ** M == pytest.approx(math.exp(-d.dt), abs=3e-16)
    assert state.step == 9 and state.time == pytest.approx(9*d.dt)


@pytest.mark.parametrize('layout', ['rfft','rpmdrate'])
@pytest.mark.parametrize('M', [2,4])
def test_failed_inner_or_slow_force_rolls_back_entire_block(layout, M):
    d, fast, slow = engine(M=M, layout=layout)
    initial = state_1d(np.ones((3,5)), np.full((3,5), .3))
    for model, failure in [(fast, 4), (slow, 3)]:
        state = initial.copy() if hasattr(initial, 'copy') else TorchState(initial.q, initial.p)
        before = d.generator.get_state().clone(); model.fail_at = model.calls + failure
        with pytest.raises(RuntimeError, match='inner force failure'): d.propagate(state, 5)
        assert torch.equal(state.q, initial.q) and torch.equal(state.p, initial.p)
        assert state.step == 0 and state.time == 0
        assert torch.equal(d.generator.get_state(), before)
        model.fail_at = None
    reference, _, _ = engine(M=M, layout=layout)
    a, b = TorchState(initial.q, initial.p), TorchState(initial.q, initial.p)
    d.propagate(a,5); reference.propagate(b,5)
    assert torch.equal(a.q,b.q) and torch.equal(a.p,b.p)


@pytest.mark.parametrize('C', [1,3,5])
@pytest.mark.parametrize('M', [2,4])
def test_deterministic_convergence_reversibility_and_symplecticity(C, M):
    P, beta, mass, T = 5, 3., 1.7, 1.12
    R = np.diag(np.r_[np.ones(P), -np.ones(P)])
    J = np.block([[np.zeros((P,P)), np.eye(P)], [-np.eye(P), np.zeros((P,P))]])
    x = np.random.default_rng(32).normal(size=(3,2*P))
    eye = np.eye(P); lap = 2*eye - np.roll(eye,1,1) - np.roll(eye,-1,1)
    transform = ref.contraction(P,C)
    K = mass*(P/beta)**2 * lap + .9*eye + .1*P/C*transform.T@transform
    A = np.block([[0*eye, eye/mass],[-K,0*eye]])
    target = x @ expm(T*A).T
    errors = []
    for dt in [.14,.07,.035]:
        d, _, _ = engine(P, M, C, dt=dt, thermostat=None)
        state = state_1d(x[:,:P],x[:,P:]); d.propagate(state, round(T/dt))
        actual = np.column_stack([state.q[:,:,0,0],state.p[:,:,0,0]])
        errors.append(np.max(abs(actual-target)))
        L,_,_ = ref.mts_gaussian_map(P,beta,mass,dt,M,contracted=C,thermostat=None)
        np.testing.assert_allclose(L.T@J@L,J,atol=1e-13,rtol=0)
        np.testing.assert_allclose(R@L@R@L,np.eye(2*P),atol=1e-13,rtol=0)
    assert all(3.7 < a/b < 4.3 for a,b in zip(errors,errors[1:]))


@pytest.mark.parametrize('C,M', [(1,2),(3,4),(5,3)])
def test_cold_start_matches_discrete_covariance_not_assumed_gibbs(C, M):
    n, steps = 4096, 400
    d,_,_ = engine(M=M,C=C,dt=.1)
    state = state_1d(np.zeros((n,5)),np.zeros((n,5))); d.propagate(state,steps)
    L,Q,_ = ref.mts_gaussian_map(5,3.,1.7,.1,M,contracted=C)
    stationary = solve_discrete_lyapunov(L,Q); Ln = np.linalg.matrix_power(L,steps)
    cov = stationary - Ln@stationary@Ln.T
    for values,target in [(state.q[:,:,0,0].square().mean(1), np.trace(cov[:5,:5])/5),
                          (state.p[:,:,0,0].square().mean(1), np.trace(cov[5:,5:])/5)]:
        se=values.std(unbiased=True).item()/math.sqrt(n)
        assert abs(values.mean().item()-target)<5*se


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA hardware/runtime unavailable')
def test_cuda_mts_layout_and_residency():
    d,_,_=engine(M=3,layout='rpmdrate',device='cuda')
    state=state_1d(np.ones((4,5)),np.zeros((4,5)),device='cuda')
    d.propagate(state,5)
    assert state.q.is_cuda and state.p.is_cuda
    assert d.force_evaluations=={'fast':16,'slow':6}
