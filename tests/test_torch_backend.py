"""Device-backend correctness tests; CUDA tests skip without a CUDA runtime."""
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from pyrpmd.torch_backend import (
    TorchPES, TorchRPCTerm, TorchSplitPotential, TorchState, TorchDynamics,
)


class Harmonic(torch.nn.Module):
    metadata = {"batch_independent": True}

    def __init__(self, k=0.2, direct=False):
        super().__init__()
        self.k, self.direct = k, direct
        self.calls = []

    def forward(self, q):
        self.calls.append(tuple(q.shape))
        energy = 0.5 * self.k * q.square().sum(dim=(1, 2))
        return (energy.detach(), -self.k * q) if self.direct else energy


class Pair(torch.nn.Module):
    metadata = {"batch_independent": True}

    def forward(self, q):
        return 0.3 * (q[:, 0] - q[:, 1]).square().sum(1) + 0.02 * q.square().sum((1, 2))


def tensors(beads=5, atoms=2, trajectories=3, seed=13):
    rng = np.random.default_rng(seed)
    shape = (trajectories, beads, atoms, 3)
    return tuple(torch.tensor(rng.normal(size=shape), dtype=torch.float64) for _ in range(2))


def engine(beads=5, atoms=2, **kwargs):
    return TorchDynamics(TorchPES(Harmonic()), np.arange(atoms) + 1.0,
                         beta=7.0, nbeads=beads, dt=0.03, **kwargs)


@pytest.fixture(autouse=True, scope="module")
def small_fft_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("direct", [False, True])
def test_pes_units_gradients_and_single_call(direct):
    model = Harmonic(0.7, direct)
    pes = TorchPES(model, length_factor=2.0, energy_factor=3.0)
    q = tensors()[0].reshape(15, 2, 3)
    with torch.no_grad():
        result = pes.evaluate_tensor(q)
    torch.testing.assert_close(result.energy, 0.5 * 0.7 * 3 * (q / 2).square().sum((1, 2)))
    torch.testing.assert_close(result.forces, -0.7 * q * 3 / 4)
    assert model.calls == [(15, 2, 3)]
    assert not result.forces.requires_grad and not result.energy.requires_grad


def test_model_parameters_do_not_accumulate_gradients():
    class Parametric(Harmonic):
        def __init__(self):
            super().__init__()
            self.k = torch.nn.Parameter(torch.tensor(0.4))
    model = Parametric()
    TorchPES(model).evaluate_tensor(tensors()[0].reshape(15, 2, 3))
    assert model.k.grad is None and model.k.requires_grad


@pytest.mark.parametrize("metadata", [{}, {"batch_independent": False},
    {"batch_independent": True, "stateful": True},
    {"batch_independent": True, "ring_polymer_coupled": True}])
def test_batch_independence_is_not_inferred(metadata):
    model = Harmonic()
    model.metadata = metadata
    with pytest.raises(ValueError, match="batch-independent"):
        TorchPES(model)


def test_explicit_independence_and_legacy_provider_bridge():
    model = Harmonic()
    model.metadata = {}
    TorchPES(model, batch_independent=True)
    provider = SimpleNamespace(metadata={"provider": "torch", "batch_independent": True},
        delegate=None, closed=False, model=model, device="cpu", length_factor=2.0,
        energy_factor=3.0, cell=None, pbc=[False] * 3, periodic=False, virial_mode="none")
    adapted = TorchPES.from_provider(provider)
    assert adapted.length_factor == 2 and adapted.energy_factor == 3
    provider.delegate = object()
    with pytest.raises(ValueError, match="delegate"):
        TorchPES.from_provider(provider)


@pytest.mark.parametrize("mode", ["detached", "shape", "dtype", "force_shape", "disconnected"])
def test_bad_model_results_fail(mode):
    class Bad(Harmonic):
        def forward(self, q):
            e = q.square().sum((1, 2))
            if mode == "detached":
                return e.detach()
            if mode == "shape":
                return e.sum()
            if mode == "dtype":
                return e.float()
            if mode == "force_shape":
                return e, -q[:, :, :1]
            return torch.ones(q.shape[0], dtype=q.dtype, requires_grad=True)
    with pytest.raises(ValueError):
        TorchPES(Bad()).evaluate_tensor(tensors()[0].reshape(15, 2, 3))


def test_periodic_autograd_virial_and_stress_units():
    class Periodic(Harmonic):
        def forward(self, q, cell, pbc):
            assert cell.shape == (len(q), 3, 3) and pbc.dtype == torch.bool
            return 0.5 * q.square().sum((1, 2))
    q = tensors()[0].reshape(15, 2, 3)
    pes = TorchPES(Periodic(), periodic=True, cell=np.eye(3) * 4, pbc=[True] * 3,
                   length_factor=2, energy_factor=3, virial="autograd")
    result = pes.evaluate_tensor(q)
    torch.testing.assert_close(result.forces, -q * 3 / 4)
    torch.testing.assert_close(result.virial, -torch.einsum("fna,fnb->fab", q, q) * 3 / 4)

    class Stress(Periodic):
        def forward(self, q, cell, pbc):
            return {"energy": q.square().sum((1, 2)), "forces": -2 * q,
                    "stress": torch.eye(3, dtype=q.dtype, device=q.device).expand(len(q), 3, 3)}
    pes = TorchPES(Stress(), periodic=True, cell=np.eye(3) * 4, pbc=[True] * 3,
                   length_factor=2, energy_factor=3, virial="provided")
    torch.testing.assert_close(pes.evaluate_tensor(q).virial, -24 * torch.eye(3, dtype=q.dtype).expand(15, 3, 3))


@pytest.mark.parametrize("beads", [1, 2, 5, 8])
@pytest.mark.parametrize("scheme", ["normal-mode-bab", "cayley-bcb"])
def test_normal_modes_match_numpy_reference(beads, scheme):
    q, p = tensors(beads)
    dyn = engine(beads, integrator=scheme)
    q1, p1 = dyn._free(q, p)
    qk, pk = np.fft.rfft(q.numpy(), axis=1, norm="ortho"), np.fft.rfft(p.numpy(), axis=1, norm="ortho")
    w = 2 * beads / 7 * np.sin(np.pi * np.arange(beads // 2 + 1) / beads)
    dt = 0.03
    if scheme == "normal-mode-bab":
        a, b, c = np.cos(dt * w), dt * np.sinc(dt * w / np.pi), w * np.sin(dt * w)
    else:
        z = dt * w / 2
        a, b, c = (1-z*z)/(1+z*z), dt/(1+z*z), dt*w*w/(1+z*z)
    a, b, c = [v[None, :, None, None] for v in (a, b, c)]
    mass = np.array([1., 2.])[None, None, :, None]
    expected_q = np.fft.irfft(a*qk+b*pk/mass, n=beads, axis=1, norm="ortho")
    expected_p = np.fft.irfft(a*pk-c*mass*qk, n=beads, axis=1, norm="ortho")
    np.testing.assert_allclose(q1.numpy(), expected_q, atol=3e-14, rtol=3e-14)
    np.testing.assert_allclose(p1.numpy(), expected_p, atol=3e-14, rtol=3e-14)


@pytest.mark.parametrize("beads", [1, 5, 8])
@pytest.mark.parametrize("scheme", ["normal-mode-bab", "cayley-bcb"])
def test_reversibility_and_finite_energy(beads, scheme):
    q, p = tensors(beads)
    state, dyn = TorchState(q, p), engine(beads, integrator=scheme)
    energy0 = dyn.energy(state)
    dyn.propagate(state, 20)
    assert torch.max(torch.abs((dyn.energy(state) - energy0) / energy0)).item() < 5e-5
    state.p.neg_()
    dyn.propagate(state, 20)
    torch.testing.assert_close(state.q, q, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(state.p, -p, atol=2e-12, rtol=2e-12)


def test_classical_limit_is_velocity_verlet():
    q, p = tensors(1)
    dyn, state = engine(1), TorchState(q, p)
    mass = torch.tensor([1., 2.], dtype=q.dtype).reshape(1, 1, 2, 1)
    ph = p - 0.5 * dyn.dt * 0.2 * q
    qe = q + dyn.dt * ph / mass
    pe = ph - 0.5 * dyn.dt * 0.2 * qe
    dyn.step(state)
    torch.testing.assert_close(state.q, qe)
    torch.testing.assert_close(state.p, pe)


@pytest.mark.parametrize("beads,contracted", [(1, 1), (5, 1), (5, 2), (5, 3), (8, 4), (8, 8)])
def test_rpc_energy_force_chain_rule_and_centroid(beads, contracted):
    q = tensors(beads, trajectories=2)[0]
    term = TorchRPCTerm(TorchPES(Pair()), beads, contracted)
    result = term.evaluate_tensor(q)
    qc = torch.einsum("cj,bjna->bcna", term.matrix, q)
    torch.testing.assert_close(qc.mean(1), q.mean(1), atol=2e-14, rtol=2e-14)
    expected_e = (beads/contracted) * Pair()(qc.reshape(2*contracted, 2, 3)).reshape(2, contracted).sum(1)
    torch.testing.assert_close(result.energy.sum(1), expected_e)
    eps = 1e-6
    for index in [(0, 0, 0, 1), (1, beads-1, 1, 2)]:
        plus, minus = q.clone(), q.clone()
        plus[index] += eps
        minus[index] -= eps
        fd = -(term.evaluate_tensor(plus).energy[index[0]].sum() - term.evaluate_tensor(minus).energy[index[0]].sum()) / (2*eps)
        torch.testing.assert_close(result.forces[index], fd, atol=5e-8, rtol=5e-8)


def test_regional_components_keep_environment_coupling_explicit():
    q = tensors(8, atoms=3, trajectories=2)[0]
    hydrogen, metal, coupling = Harmonic(0.5), Harmonic(0.1), Harmonic(0.03)
    terms = [TorchRPCTerm(TorchPES(hydrogen), 8, 8, atoms=[0]),
             TorchRPCTerm(TorchPES(metal), 8, 2, atoms=[1, 2]),
             TorchRPCTerm(TorchPES(coupling), 8, 8)]
    pes = TorchSplitPotential(terms)
    result = pes.evaluate_tensor(q)
    assert hydrogen.calls == [(16, 1, 3)]
    assert metal.calls == [(4, 2, 3)]
    assert coupling.calls == [(16, 3, 3)]
    # The metal component must not apply any force to the excluded H atom.
    assert torch.count_nonzero(terms[1].evaluate_tensor(q).forces[:, :, 0]) == 0
    eps, index = 1e-6, (0, 3, 2, 0)
    qp, qm = q.clone(), q.clone()
    qp[index] += eps
    qm[index] -= eps
    fd = -(pes.evaluate_tensor(qp).energy.sum() - pes.evaluate_tensor(qm).energy.sum())/(2*eps)
    torch.testing.assert_close(result.forces[index], fd, atol=5e-8, rtol=5e-8)


def test_batch_equivalence_and_one_call_per_endpoint():
    q, p = tensors(5, trajectories=4)
    model = Harmonic()
    dyn = TorchDynamics(TorchPES(model), [1, 2], 7, 5, 0.03)
    batch = TorchState(q, p)
    dyn.propagate(batch, 6)
    assert model.calls == [(20, 2, 3)] * 7
    for i in range(4):
        one = TorchState(q[i:i+1], p[i:i+1])
        engine(5).propagate(one, 6)
        torch.testing.assert_close(batch.q[i:i+1], one.q, atol=1e-13, rtol=1e-13)
        torch.testing.assert_close(batch.p[i:i+1], one.p, atol=1e-13, rtol=1e-13)


def test_mts_endpoint_counts_and_full_resolution_limit():
    q, p = tensors(8, trajectories=2)
    fast, slow = Harmonic(0.17), Harmonic(0.03)
    split = TorchSplitPotential([TorchRPCTerm(TorchPES(fast), 8)],
                                [TorchRPCTerm(TorchPES(slow), 8, 2)])
    dyn = TorchDynamics(split, [1, 2], 7, 8, 0.03, inner_steps=4)
    dyn.propagate(TorchState(q, p), 5)
    assert dyn.force_evaluations == {"fast": 21, "slow": 6}
    assert fast.calls == [(16, 2, 3)] * 21 and slow.calls == [(4, 2, 3)] * 6
    split = TorchSplitPotential([TorchRPCTerm(TorchPES(Harmonic(0.17)), 8)],
                                [TorchRPCTerm(TorchPES(Harmonic(0.03)), 8)])
    d1 = TorchDynamics(split, [1, 2], 7, 8, 0.03)
    d2 = engine(8)
    s1, s2 = TorchState(q, p), TorchState(q, p)
    d1.propagate(s1, 8)
    d2.propagate(s2, 8)
    torch.testing.assert_close(s1.q, s2.q, atol=3e-13, rtol=3e-13)
    torch.testing.assert_close(s1.p, s2.p, atol=3e-13, rtol=3e-13)


@pytest.mark.parametrize("beads", [1, 7, 8])
@pytest.mark.parametrize("thermostat", ["pile-l", "trpmd"])
def test_thermostat_stationary_variance_and_centroid(beads, thermostat):
    dyn = TorchDynamics(TorchPES(Harmonic()), [2], 2, beads, 0.8,
                        thermostat=thermostat, seed=190)
    p = dyn.sample_momenta(12000)
    out = dyn._thermostat(p)
    expected = 2 * beads / 2
    variances = out.permute(1, 0, 2, 3).reshape(beads, -1).var(1)
    torch.testing.assert_close(variances, torch.full_like(variances, expected), rtol=0.04, atol=0)
    if thermostat == "trpmd":
        torch.testing.assert_close(out.mean(1), p.mean(1), atol=5e-14, rtol=5e-14)
    if beads > 1:
        sample = out[:, :, 0, 0].numpy()
        covariance = np.cov(sample, rowvar=False) / expected
        np.testing.assert_allclose(covariance, np.eye(beads), atol=0.05, rtol=0.05)


def test_thermostat_reproducible_per_engine_and_block_boundaries():
    q, p = tensors()
    a, b = engine(thermostat="pile-l", seed=7), engine(thermostat="pile-l", seed=7)
    s1, s2 = TorchState(q, p), TorchState(q, p)
    a.propagate(s1, 6)
    b.propagate(s2, 2)
    b.propagate(s2, 4)
    torch.testing.assert_close(s1.q, s2.q, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(s1.p, s2.p, atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize("failure", ["exception", "nan_energy"])
def test_failed_block_rolls_back_coordinates_clock_and_rng(failure):
    class Fails(Harmonic):
        def forward(self, q):
            e = super().forward(q)
            if len(self.calls) == 3:
                if failure == "exception":
                    raise RuntimeError("test failure")
                return e.detach() * float("nan"), -self.k * q
            return e
    q, p = tensors()
    state = TorchState(q, p, step=4, time=0.12)
    dyn = TorchDynamics(TorchPES(Fails()), [1, 2], 7, 5, 0.03, thermostat="pile-l", seed=2)
    rng = dyn.generator.get_state().clone()
    with pytest.raises((RuntimeError, FloatingPointError)):
        dyn.propagate(state, 5)
    assert torch.equal(state.q, q) and torch.equal(state.p, p)
    assert state.step == 4 and state.time == 0.12
    assert torch.equal(rng, dyn.generator.get_state())


def test_no_hot_loop_host_copies_and_one_scalar_read(monkeypatch):
    q, p = tensors()
    dyn, state = engine(thermostat="pile-l"), TorchState(q, p)
    original, reads = torch.Tensor.item, []
    def forbidden(*args, **kwargs):
        raise AssertionError("implicit tensor-to-host transfer")
    def counted(value, *args, **kwargs):
        reads.append(value.numel())
        return original(value, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    monkeypatch.setattr(torch.Tensor, "numpy", forbidden)
    monkeypatch.setattr(torch.Tensor, "item", counted)
    dyn.propagate(state, 7)
    assert reads == [1]


def test_state_upload_snapshot_and_no_shared_storage():
    q, p = tensors()
    state = TorchState.from_arrays(q[0].numpy(), p[0].numpy())
    assert state.q.shape == (1, 5, 2, 3)
    snapshot = state.snapshot()
    snapshot["q"][:] = 0
    torch.testing.assert_close(state.q, q[:1])
    state.q.zero_()
    assert torch.count_nonzero(q).item() > 0


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_invalid_bead_counts(bad):
    with pytest.raises(ValueError):
        TorchRPCTerm(TorchPES(Harmonic()), bad)


def test_invalid_configuration_and_mutation_rejected():
    pes = TorchPES(Harmonic())
    with pytest.raises(ValueError):
        TorchRPCTerm(pes, 4, 5)
    with pytest.raises(ValueError):
        TorchRPCTerm(pes, 4, atoms=[1, 1])
    with pytest.raises(ValueError):
        TorchDynamics(pes, [1], 7, 4, 0.1, inner_steps=2)
    with pytest.raises(ValueError):
        TorchDynamics(pes, [-1], 7, 4, 0.1)
    dyn, state = engine(), TorchState(*tensors())
    dyn.dt = 0.2
    with pytest.raises(ValueError, match="immutable"):
        dyn.step(state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware/runtime unavailable")
@pytest.mark.parametrize("scheme", ["normal-mode-bab", "cayley-bcb"])
def test_cuda_parity_and_device_residency(scheme):
    q, p = tensors(7)
    cpu = engine(7, integrator=scheme)
    gpu = TorchDynamics(TorchPES(Harmonic(), device="cuda"), [1, 2], 7, 7, 0.03, integrator=scheme)
    sc, sg = TorchState(q, p), TorchState(q.cuda(), p.cuda())
    cpu.propagate(sc, 10)
    gpu.propagate(sg, 10)
    assert sg.q.is_cuda and sg.p.is_cuda
    assert all(v.is_cuda for v in gpu._abc)
    torch.testing.assert_close(sc.q, sg.q.cpu(), atol=3e-12, rtol=3e-12)
    torch.testing.assert_close(sc.p, sg.p.cpu(), atol=3e-12, rtol=3e-12)


@pytest.mark.parametrize("scheme", ["normal-mode-bab", "cayley-bcb"])
def test_energy_error_is_second_order_in_timestep(scheme):
    q, p = tensors(8)
    errors = []
    for dt, steps in [(0.03, 20), (0.015, 40)]:
        dyn = TorchDynamics(TorchPES(Harmonic()), [1, 2], 7, 8, dt, integrator=scheme)
        state = TorchState(q, p)
        before = dyn.energy(state)
        dyn.propagate(state, steps)
        errors.append((dyn.energy(state) - before).abs().max().item())
    assert 3.8 < errors[0] / errors[1] < 4.2


@pytest.mark.parametrize("value", [True, 1+2j])
def test_lossy_input_conversions_rejected(value):
    array = np.full((1, 2, 1, 3), value)
    with pytest.raises(ValueError, match="real numerical"):
        TorchState.from_arrays(array, array)
    with pytest.raises(ValueError, match="real numerical"):
        TorchDynamics(TorchPES(Harmonic()), [value], 7, 2, 0.03)


def test_external_aliases_are_rejected_without_modifying_state():
    dyn, state = engine(), TorchState(*tensors())
    state.p = state.q
    q = state.q.clone()
    with pytest.raises(ValueError, match="share storage"):
        dyn.step(state)
    assert torch.equal(state.q, q)


def test_malformed_capabilities_and_conflicting_provider_metadata():
    model = Harmonic()
    model.metadata = {"batch_independent": True, "capabilities": "not-a-mapping"}
    with pytest.raises(ValueError, match="capabilities"):
        TorchPES(model)
    provider = SimpleNamespace(metadata={"provider": "torch", "batch_independent": True,
                                        "coupled_frames": True}, delegate=None, closed=False)
    with pytest.raises(ValueError, match="batch-independent"):
        TorchPES.from_provider(provider)


def test_no_stale_endpoint_force_after_external_coordinate_edit():
    q, p = tensors()
    dyn, state = engine(), TorchState(q, p)
    dyn.propagate(state, 2)
    state.q.add_(0.25)
    reference = TorchState(state.q, state.p, state.step, state.time)
    engine().step(reference)
    dyn.step(state)
    torch.testing.assert_close(state.q, reference.q)
    torch.testing.assert_close(state.p, reference.p)


@pytest.mark.parametrize("beads", [3, 4])
def test_pure_spring_energy_exactly_preserved(beads):
    q, p = tensors(beads)
    for scheme in ("normal-mode-bab", "cayley-bcb"):
        dyn = TorchDynamics(TorchPES(Harmonic(0)), [1, 2], 7, beads, 0.1, integrator=scheme)
        state = TorchState(q, p)
        before = dyn.energy(state)
        dyn.propagate(state, 50)
        torch.testing.assert_close(dyn.energy(state), before, atol=1e-11, rtol=1e-11)


def test_zero_steps_do_not_call_model_or_consume_rng():
    q, p = tensors()
    model = Harmonic()
    dyn = TorchDynamics(TorchPES(model), [1, 2], 7, 5, 0.03, thermostat="pile-l")
    state = TorchState(q, p)
    rng = dyn.generator.get_state().clone()
    dyn.propagate(state, 0)
    assert model.calls == [] and state.step == 0
    assert torch.equal(rng, dyn.generator.get_state())


@pytest.mark.parametrize("scheme", ["normal-mode-bab", "cayley-bcb"])
def test_contracted_mts_time_reversal(scheme):
    q, p = tensors(8)
    split = TorchSplitPotential([TorchRPCTerm(TorchPES(Harmonic(.17)), 8)],
                                [TorchRPCTerm(TorchPES(Harmonic(.03)), 8, 3)])
    dyn = TorchDynamics(split, [1, 2], 7, 8, .03, inner_steps=3, integrator=scheme)
    state = TorchState(q, p)
    dyn.propagate(state, 12)
    state.p.neg_()
    dyn.propagate(state, 12)
    torch.testing.assert_close(state.q, q, atol=3e-12, rtol=3e-12)
    torch.testing.assert_close(state.p, -p, atol=3e-12, rtol=3e-12)


def test_python_float_inputs_do_not_round_through_float32():
    value = 1.234567890123456
    state = TorchState.from_arrays([[[[value, 0., 0.]]]], [[[[0., 0., 0.]]]])
    assert state.q[0, 0, 0, 0].item() == value
    dyn = TorchDynamics(TorchPES(Harmonic()), [value], 7, 1, .03)
    assert dyn._masses[0].item() == value


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware/runtime unavailable")
def test_cuda_rpc_mts_thermostat_has_no_array_host_copies(monkeypatch):
    q, p = tensors(8)
    fast = TorchRPCTerm(TorchPES(Harmonic(.17), device="cuda"), 8)
    slow = TorchRPCTerm(TorchPES(Harmonic(.03), device="cuda"), 8, 2)
    dyn = TorchDynamics(TorchSplitPotential([fast], [slow]), [1, 2], 7, 8, .03,
                        inner_steps=4, thermostat="pile-l")
    state = TorchState(q.cuda(), p.cuda())
    def forbidden(*args, **kwargs):
        raise AssertionError("array host transfer inside CUDA propagation")
    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    monkeypatch.setattr(torch.Tensor, "numpy", forbidden)
    dyn.propagate(state, 5)
    assert state.q.is_cuda and state.p.is_cuda
    assert slow.matrix.is_cuda


def test_full_checkout_provider_and_kernel_parity():
    """Runs in the repository's Torch CI job; needs the complete checkout."""
    optional = pytest.importorskip("pyrpmd.potentials.optional", reason="full upstream checkout required")
    from pyrpmd.dynamics import CayleyIntegrator
    from pyrpmd.potentials.contraction import RingPolymerContraction
    q, p = tensors(8)
    legacy = optional.TorchPotential(model=Harmonic(), device="cpu", virial="none")
    try:
        adapted = TorchPES.from_provider(legacy)
        frames = q.reshape(-1, 2, 3)
        expected = legacy.evaluate(frames.numpy())
        actual = adapted.evaluate_tensor(frames)
        np.testing.assert_allclose(actual.energy.numpy(), expected.energy, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(actual.forces.numpy(), expected.forces, rtol=1e-13, atol=1e-13)
        for count in (1, 2, 3, 4, 8):
            term = TorchRPCTerm(adapted, 8, count)
            reference = RingPolymerContraction(8, count)
            np.testing.assert_allclose(term.matrix.numpy(), reference.matrix, rtol=1e-13, atol=1e-13)
        for scheme in ("normal-mode-bab", "cayley-bcb"):
            ref = CayleyIntegrator([1., 2.], 7., 8, .03, native=False, scheme=scheme)
            qr, pr = q[0].numpy().copy(), p[0].numpy().copy()
            ref.free_step(qr, pr)
            actual_q, actual_p = engine(8, integrator=scheme)._free(q[:1], p[:1])
            np.testing.assert_allclose(actual_q[0].numpy(), qr, rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(actual_p[0].numpy(), pr, rtol=1e-13, atol=1e-13)
    finally:
        legacy.close()


@pytest.mark.parametrize("field", ["q", "p"])
def test_external_overlapping_view_rejected_before_atomic_commit(field):
    dyn, state = engine(thermostat="pile-l"), TorchState(*tensors())
    value = getattr(state, field)
    setattr(state, field, value[:, :1].expand_as(value))
    before_q, before_p = state.q.clone(), state.p.clone()
    rng = dyn.generator.get_state().clone()
    with pytest.raises(ValueError, match="contiguous"):
        dyn.propagate(state, 2)
    assert torch.equal(state.q, before_q) and torch.equal(state.p, before_p)
    assert state.step == 0 and state.time == 0.0
    assert torch.equal(rng, dyn.generator.get_state())
    assert dyn.force_evaluations == {"fast": 0, "slow": 0}


def test_constructor_owns_contiguous_storage_for_transposed_inputs():
    q, p = tensors(beads=3, atoms=5)
    q, p = q.transpose(1, 2), p.transpose(1, 2)
    state = TorchState(q, p)
    assert state.q.is_contiguous() and state.p.is_contiguous()
    torch.testing.assert_close(state.q, q)
    torch.testing.assert_close(state.p, p)
    TorchDynamics(TorchPES(Harmonic()), [1, 2, 3], 7, 5, 0.03).step(state)
    assert state.step == 1


@pytest.mark.parametrize("beads", [3, 4])
def test_pile_sampling_preserves_harmonic_configurational_variance(beads):
    """Check the finite-P Gibbs distribution, not just momentum variance."""
    beta, mass, stiffness = 7., 2., 0.2
    generator = torch.Generator().manual_seed(191)
    shape = (6000, beads, 1, 3)
    omega = 2 * beads / beta * torch.sin(
        torch.pi * torch.arange(beads // 2 + 1, dtype=torch.float64) / beads
    )
    expected = (beads / beta) / (stiffness + mass * omega.square())
    white = torch.randn(shape, dtype=torch.float64, generator=generator)
    q = torch.fft.irfft(
        torch.fft.rfft(white, dim=1, norm="ortho") * expected.sqrt()[None, :, None, None],
        n=beads, dim=1, norm="ortho",
    )
    dyn = TorchDynamics(TorchPES(Harmonic(stiffness)), [mass], beta, beads, 0.03,
                        thermostat="pile-l", seed=192)
    state = TorchState(q, dyn.sample_momenta(shape[0]))
    dyn.propagate(state, 30)
    power = torch.fft.rfft(state.q, dim=1, norm="ortho").abs().square().mean((0, 2, 3))
    torch.testing.assert_close(power, expected, atol=0., rtol=0.04)
    centroid_variance = state.q.mean(dim=1).square().mean()
    torch.testing.assert_close(centroid_variance, torch.tensor(1 / (beta * stiffness),
                               dtype=torch.float64), atol=0., rtol=0.04)
