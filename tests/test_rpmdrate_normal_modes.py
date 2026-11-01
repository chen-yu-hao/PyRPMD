"""BCOCB/normal-mode interoperability, independent of RPMDrate installation."""
import ctypes.util
import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from pyrpmd.torch_normal_modes import RPMDrateNormalModes
from pyrpmd.torch_backend import TorchDynamics, TorchPES, TorchState, TorchRPCTerm, TorchSplitPotential

_spec = importlib.util.spec_from_file_location(
    "rpmdrate_reference", Path(__file__).parents[1] / "benchmarks" / "rpmdrate_normal_modes.py")
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)

P_VALUES = (1, 2, 3, 4, 5, 8, 16, 31, 32)


@pytest.fixture(autouse=True, scope="module")
def one_thread():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


class Coupled(torch.nn.Module):
    metadata = {"batch_independent": True}
    def __init__(self, scale=1.):
        super().__init__()
        self.register_buffer("K", torch.tensor([[1., -.2], [-.2, 1.3]], dtype=torch.float64) * scale)
    def forward(self, q):
        g = torch.einsum("nm,fmc->fnc", self.K, q)
        return .5 * (q * g).sum((1, 2)), -g


def engine(P, layout="rpmdrate", scheme="cayley-bcocb", thermostat="pile-l", **options):
    return TorchDynamics(TorchPES(Coupled()), [1., 5.], 3., P, .13,
                         normal_mode_layout=layout, integrator=scheme, thermostat=thermostat, **options)


def arrays(P, n=3):
    rng = np.random.default_rng(99)
    return [rng.normal(size=(n, P, 2, 3)) for _ in range(3)]


@pytest.mark.parametrize("P", P_VALUES)
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_transform_matches_explicit_source_convention(P, axis):
    modes = RPMDrateNormalModes(P)
    q = torch.tensor(np.moveaxis(arrays(P)[0], 1, axis))
    saved = q.clone()
    actual = modes.forward(q, bead_axis=axis)
    F, weights, _ = ref.real_transform(P)
    expected = np.moveaxis(q.numpy(), axis, -1) @ F.T
    np.testing.assert_allclose(np.moveaxis(actual.numpy(), axis, -1), expected, atol=3e-14, rtol=2e-13)
    torch.testing.assert_close(modes.inverse(actual, bead_axis=axis), q, atol=3e-14, rtol=2e-13)
    last = np.moveaxis(actual.numpy(), axis, -1)
    np.testing.assert_allclose(np.sum(last**2 * weights, -1), q.square().sum(axis), atol=1e-12)
    np.testing.assert_array_equal(modes.metric, weights)
    assert torch.equal(q, saved)
    # Return-value ownership prevents callers mutating cached conventions.
    modes.metric.zero_(); assert modes.metric[0].item() == 1.


@pytest.mark.parametrize("P", P_VALUES)
def test_actual_fftw_r2hc_hc2r(P):
    if not ctypes.util.find_library("fftw3"):
        pytest.skip("optional FFTW3 library unavailable")
    modes = RPMDrateNormalModes(P)
    x = np.moveaxis(arrays(P)[0], 1, -1)
    actual = modes.forward(torch.tensor(x))
    np.testing.assert_allclose(actual, ref.fftw_transform(x), atol=2e-14, rtol=2e-13)
    np.testing.assert_allclose(modes.inverse(actual), ref.fftw_transform(actual.numpy(), inverse=True), atol=2e-14, rtol=2e-13)


@pytest.mark.parametrize("P", P_VALUES)
def test_free_rotation_matches_rpmdrate_with_real_masses(P):
    if not ctypes.util.find_library("fftw3"):
        pytest.skip("optional FFTW3 library unavailable")
    q, p, _ = arrays(P)
    d = engine(P, scheme="normal-mode-bab", thermostat=None)
    actual = d._free(torch.tensor(q), torch.tensor(p))
    expected = ref.exact_free(q, p, [1., 5.], 3., .13)
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(a, e, atol=2e-12, rtol=2e-13)


@pytest.mark.parametrize("P", P_VALUES)
@pytest.mark.parametrize("thermostat", ["pile-l", "trpmd"])
def test_bcocb_full_step_matches_independent_matrix_reference(monkeypatch, P, thermostat):
    q, p, noise = arrays(P)
    d = engine(P, thermostat=thermostat)
    expected = ref.bcocb_reference(q, p, noise, [1., 5.], 3., .13,
                                   d.potential.fast[0].potential.model.K.numpy(),
                                   gamma_centroid=1. if thermostat == "pile-l" else 0.)
    monkeypatch.setattr(torch, "randn", lambda shape, **kw: torch.tensor(noise, dtype=kw["dtype"], device=kw["device"]))
    state = TorchState.from_arrays(q, p)
    d.step(state)
    for a, e in zip((state.q, state.p), expected):
        np.testing.assert_allclose(a, e, atol=5e-11, rtol=2e-12)


@pytest.mark.parametrize("P", [1, 5, 8, 31])
@pytest.mark.parametrize("scheme", ["normal-mode-bab", "cayley-bcb", "cayley-bcocb"])
@pytest.mark.parametrize("thermostat", [None, "pile-l", "trpmd"])
def test_same_bead_noise_gives_same_dynamics_across_layouts(P, scheme, thermostat):
    q, p, _ = arrays(P)
    a, b = TorchState.from_arrays(q, p), TorchState.from_arrays(q, p)
    old = engine(P, "rfft", scheme, thermostat, seed=81)
    new = engine(P, "rpmdrate", scheme, thermostat, seed=81)
    old.propagate(a, 15); new.propagate(b, 15)
    torch.testing.assert_close(a.q, b.q, atol=3e-12, rtol=2e-12)
    torch.testing.assert_close(a.p, b.p, atol=3e-11, rtol=2e-12)
    assert torch.equal(old.generator.get_state(), new.generator.get_state())
    assert old.force_evaluations == new.force_evaluations


@pytest.mark.parametrize("P", [1, 5, 8])
@pytest.mark.parametrize("native,loop", [(False, "python"), (True, "python"), (True, "auto")])
def test_bcocb_matches_existing_cpu_sampling_path(monkeypatch, P, native, loop):
    from pyrpmd.dynamics import Dynamics, State, native_capabilities
    from pyrpmd.potentials.base import PESResult
    if native and not native_capabilities()["available"]:
        pytest.skip("native extension unavailable")
    q, p, noise = [x[0] for x in arrays(P)]
    # CPU sampling uses real ORTHONORMAL centroid, cos/sin pairs, Nyquist.
    j = np.arange(P)
    rows = [np.ones(P) / np.sqrt(P)]
    for k in range(1, (P + 1) // 2):
        rows += [np.sqrt(2/P)*np.cos(2*np.pi*k*j/P), np.sqrt(2/P)*np.sin(2*np.pi*k*j/P)]
    if P % 2 == 0:
        rows.append((-1.)**j / np.sqrt(P))
    mode_noise = np.einsum("kb,bnc->knc", np.array(rows), noise)
    class Noise(np.random.Generator):
        def normal(self, size=None):
            assert tuple(size) == mode_noise.shape
            return mode_noise.copy()
    class PES:
        def evaluate(self, x):
            g = np.einsum("nm,pmc->pnc", np.array([[1.,-.2],[-.2,1.3]]), x)
            return PESResult(.5*(x*g).sum((1,2)), -g)
    cpu = Dynamics(PES(), [1.,5.], 3., P, .13, seed=Noise(np.random.PCG64(1)), native=native,
                   loop_backend=loop, integrator="cayley-bcb")
    cs = State(q, p)
    td = engine(P); ts = TorchState.from_arrays(q[None], p[None])
    monkeypatch.setattr(torch, "randn", lambda shape, **kw: torch.tensor(noise[None], dtype=kw["dtype"], device=kw["device"]))
    cpu.sample_step(cs, method="bcocb"); td.step(ts)
    np.testing.assert_allclose(ts.q[0], cs.q, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(ts.p[0], cs.p, atol=1e-11, rtol=2e-12)


def test_bcocb_sampling_to_normal_mode_production():
    from pyrpmd.dynamics import Dynamics, State
    from pyrpmd.potentials.base import PESResult
    class PES:
        def evaluate(self, x):
            g = np.einsum("nm,pmc->pnc", np.array([[1.,-.2],[-.2,1.3]]), x)
            return PESResult(.5*(x*g).sum((1,2)), -g)
    P = 8; q, p, _ = arrays(P, n=1)
    sampled = TorchState.from_arrays(q, p); sampler = engine(P, seed=5)
    sampler.propagate(sampled, 40)
    production = engine(P, scheme="normal-mode-bab", thermostat=None, seed=23)
    launch = TorchState(sampled.q, production.sample_momenta())
    cpu_state = State(launch.q[0].numpy(), launch.p[0].numpy())
    cpu = Dynamics(PES(), [1.,5.], 3., P, .13, native=False, integrator="normal-mode-bab")
    rng = production.generator.get_state().clone()
    production.propagate(launch, 50)
    for _ in range(50): cpu.step(cpu_state)
    np.testing.assert_allclose(launch.q[0], cpu_state.q, atol=3e-12, rtol=2e-12)
    np.testing.assert_allclose(launch.p[0], cpu_state.p, atol=3e-11, rtol=2e-12)
    assert torch.equal(rng, production.generator.get_state())


def test_packed_gaussian_covariance_including_sine_sign():
    P = 8; d = RPMDrateNormalModes(P)
    # Deterministic Gaussian covariance probe: independent basis noise.
    F = d.forward(torch.eye(P, dtype=torch.float64)).numpy().T
    expected, weights, _ = ref.real_transform(P)
    np.testing.assert_allclose(F, expected, atol=1e-15)
    np.testing.assert_allclose(F@F.T, np.diag(1/weights), atol=1e-15)
    assert F[-1, 1] < 0  # Im(z1) is negative sine, not positive sine.


def test_mode_fusion_and_host_transfer_budget(monkeypatch):
    d = engine(8); q, p, _ = arrays(8); state = TorchState.from_arrays(q, p)
    counters = dict(rfft=0, irfft=0, item=0)
    for name in ("rfft", "irfft"):
        original = getattr(torch.fft, name)
        def counted(*a, _name=name, _original=original, **kw):
            counters[_name] += 1; return _original(*a, **kw)
        monkeypatch.setattr(torch.fft, name, counted)
    original = torch.Tensor.item
    def item(x): counters["item"] += 1; return original(x)
    monkeypatch.setattr(torch.Tensor, "item", item)
    def forbidden(*a, **kw): raise AssertionError("implicit host transfer")
    for name in ("cpu", "numpy", "tolist"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    d.propagate(state, 7)
    assert counters == {"rfft": 21, "irfft": 14, "item": 1}
    assert d.force_evaluations == {"fast": 8, "slow": 0}


@pytest.mark.parametrize("P,C", [(5,1), (8,3)])
def test_rpc_bcocb_and_mts_other_integrators_are_layout_independent(P,C):
    for scheme,M in (("cayley-bcocb",1),("cayley-bcb",3),("normal-mode-bab",3)):
        q,p,_ = arrays(P); states=[]; engines=[]
        for layout in ("rfft", "rpmdrate"):
            terms = TorchSplitPotential([TorchRPCTerm(TorchPES(Coupled(.9)),P)],
                                        [TorchRPCTerm(TorchPES(Coupled(.1)),P,C)])
            d=TorchDynamics(terms,[1.,5.],3.,P,.05,inner_steps=M,integrator=scheme,
                            thermostat="pile-l",normal_mode_layout=layout,seed=27)
            s=TorchState.from_arrays(q,p);d.propagate(s,8);states.append(s);engines.append(d)
        torch.testing.assert_close(states[0].q,states[1].q,atol=1e-11,rtol=2e-12)
        torch.testing.assert_close(states[0].p,states[1].p,atol=1e-10,rtol=2e-12)
        assert engines[1].force_evaluations == {"fast":8*M+1,"slow":9}


@pytest.mark.parametrize("bad", [True, 0, -1, 2.5])
def test_bad_bead_count(bad):
    with pytest.raises(ValueError): RPMDrateNormalModes(bad)


@pytest.mark.parametrize("axis", [True, 4, -5, 1.5])
def test_bad_axis(axis):
    m=RPMDrateNormalModes(3)
    with pytest.raises(ValueError):m.forward(torch.ones((2,3),dtype=torch.float64),bead_axis=axis)


def test_bad_input_and_engine_layout_mutation():
    m=RPMDrateNormalModes(3)
    for value in (np.ones(3),torch.ones(3),torch.ones(2,dtype=torch.float64),
                  torch.ones((0,3),dtype=torch.float64)):
        with pytest.raises(ValueError):m.forward(value)
    for beta in (True,0.,-1.,float("nan")):
        with pytest.raises(ValueError):m.frequencies(beta)
    with pytest.raises(ValueError):engine(3,layout="normal-mode-bab")
    d=engine(3); d.normal_mode_layout="rfft"
    q,p,_=arrays(3)
    with pytest.raises(ValueError,match="immutable"):d.step(TorchState.from_arrays(q,p))


@pytest.mark.parametrize("failure", ["raise","nan"])
def test_packed_atomic_rng_rollback(failure):
    class Fail(Coupled):
        def __init__(self):super().__init__();self.calls=0
        def forward(self,q):
            self.calls+=1
            if self.calls==3:
                if failure=="raise":raise RuntimeError("injected failure")
                return q.sum((1,2))*float("nan"),q*float("nan")
            return super().forward(q)
    d=TorchDynamics(TorchPES(Fail()),[1.,5.],3.,8,.1,normal_mode_layout="rpmdrate",
                    integrator="cayley-bcocb",thermostat="pile-l",seed=9)
    q,p,_=arrays(8);s=TorchState.from_arrays(q,p);g=d.generator.get_state().clone()
    with pytest.raises((RuntimeError,FloatingPointError)):d.propagate(s,4)
    np.testing.assert_array_equal(s.q,q);np.testing.assert_array_equal(s.p,p)
    assert s.step==0 and s.time==0. and torch.equal(g,d.generator.get_state())


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA hardware/runtime unavailable")
def test_cuda_packed_layout_residency_and_parity():
    q,p,_=arrays(8);cpu=engine(8,thermostat=None)
    gpu=TorchDynamics(TorchPES(Coupled(),device="cuda"),[1.,5.],3.,8,.13,
                      normal_mode_layout="rpmdrate",integrator="cayley-bcocb")
    a=TorchState.from_arrays(q,p);b=TorchState.from_arrays(q,p,device="cuda")
    cpu.propagate(a,20);gpu.propagate(b,20)
    torch.testing.assert_close(a.q,b.q.cpu(),atol=1e-11,rtol=2e-12)
    torch.testing.assert_close(a.p,b.p.cpu(),atol=1e-10,rtol=2e-12)
    assert b.q.device.type==gpu._normal_modes.device.type=="cuda"
