"""Tensor workflow bindings: independent AU references and strict boundaries."""
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pyrpmd.potentials.base import BackendUnavailableError
from pyrpmd.potentials.builtin import (
    DoubleWellPotential, FreePotential, HarmonicPotential,
    MorsePairPotential, PeriodicCosinePotential,
)
from pyrpmd.potentials.optional import TorchPotential
from pyrpmd.potentials.torch_workflow import make_workflow_torch_pes
from pyrpmd.torch_backend import TensorPESResult, TorchDynamics, TorchPES, TorchState


def positions(frames=7, atoms=4):
    return np.random.default_rng(18).normal(size=(frames, atoms, 3))


def tensor(q, device="cpu"):
    return torch.tensor(q, dtype=torch.float64, device=device)


def compare(provider, q):
    expected = provider.evaluate(q)
    bound = make_workflow_torch_pes(provider, device="cpu")
    result = bound.evaluate_tensor(tensor(q))
    np.testing.assert_allclose(result.energy.numpy(), expected.energy, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(result.forces.numpy(), expected.forces, rtol=3e-13, atol=3e-13)
    if expected.virial is None:
        assert result.virial is None
    else:
        np.testing.assert_allclose(result.virial.numpy(), expected.virial, rtol=3e-13, atol=3e-13)
    assert result.energy.dtype == result.forces.dtype == torch.float64
    assert not result.energy.requires_grad and not result.forces.requires_grad
    return bound


def finite_difference(bound, q):
    force = bound.evaluate_tensor(tensor(q)).forces.numpy()
    delta = 2e-6
    for atom in range(q.shape[1]):
        for axis in range(3):
            plus, minus = q.copy(), q.copy()
            plus[:, atom, axis] += delta
            minus[:, atom, axis] -= delta
            derivative = -(bound.evaluate_tensor(tensor(plus)).energy.numpy()
                           - bound.evaluate_tensor(tensor(minus)).energy.numpy()) / (2 * delta)
            np.testing.assert_allclose(force[:, atom, axis], derivative, rtol=3e-8, atol=3e-9)


@pytest.mark.parametrize("frames,atoms", [(1, 1), (5, 4), (32, 2)])
def test_free_energy_forces_and_virial(frames, atoms):
    bound = compare(FreePotential(), positions(frames, atoms))
    assert bound.metadata["supports_pbc"] is True


_SHAPES = [(), (1,), (3,), (1, 1), (1, 3), (4, 1), (4, 3)]


@pytest.mark.parametrize("k_shape", _SHAPES)
@pytest.mark.parametrize("center_shape", _SHAPES)
def test_harmonic_preserves_numpy_atom_axis_broadcast(k_shape, center_shape):
    rng = np.random.default_rng(1)
    provider = HarmonicPotential(k=rng.uniform(0, 1, size=k_shape),
                                 center=rng.normal(size=center_shape))
    bound = compare(provider, positions())
    assert bound.metadata["supports_pbc"] is False


def test_harmonic_force_is_negative_energy_derivative():
    provider = HarmonicPotential(k=np.arange(12).reshape(4, 3) / 8,
                                 center=np.linspace(-0.4, 0.3, 12).reshape(4, 3))
    bound = compare(provider, positions(3))
    finite_difference(bound, positions(3))


@pytest.mark.parametrize("name", ["k", "center"])
def test_harmonic_frame_parameters_are_explicitly_rejected(name):
    provider = HarmonicPotential(**{name: np.ones((7, 4, 3))})
    with pytest.raises(ValueError, match="frame axis"):
        make_workflow_torch_pes(provider)


def test_harmonic_incompatible_atom_broadcast_rejects():
    bound = make_workflow_torch_pes(HarmonicPotential(k=np.ones((5, 3))))
    with pytest.raises(ValueError, match=r"broadcast to \[atoms,3\]"):
        bound.evaluate_tensor(tensor(positions()))


@pytest.mark.parametrize("pbc", [[True, True, True], [True, False, True], [False, True, False]])
@pytest.mark.parametrize("atoms", [None, [3, 1]])
def test_fixed_cosine_triclinic_partial_pbc_selection_and_confinement(pbc, atoms):
    cell = np.array([[4.0, .3, -.2], [.4, 5., .6], [.2, -.3, 6.]])
    provider = PeriodicCosinePotential(cell, pbc=pbc, height=[.2, .3, .4],
                                       repeats=[2, 3, 1], atoms=atoms, confinement=.07)
    q = positions(3)
    bound = compare(provider, q)
    finite_difference(bound, q)
    shifted = q + np.array([2, -3, 1])[np.asarray(pbc)] @ cell[np.asarray(pbc)]
    original = bound.evaluate_tensor(tensor(q))
    moved = bound.evaluate_tensor(tensor(shifted))
    torch.testing.assert_close(moved.energy, original.energy, rtol=3e-13, atol=3e-13)
    torch.testing.assert_close(moved.forces, original.forces, rtol=3e-13, atol=3e-13)
    assert moved.virial is None
    if atoms is not None:
        assert torch.count_nonzero(moved.forces[:, [0, 2]]) == 0


def test_cosine_out_of_range_atom_is_not_dropped():
    bound = make_workflow_torch_pes(PeriodicCosinePotential(np.eye(3), [True] * 3, atoms=[4]))
    with pytest.raises(ValueError, match="atom index out of range"):
        bound.evaluate_tensor(tensor(positions()))


@pytest.mark.parametrize("provider", [HarmonicPotential(k=[.2, .3, .4], center=[1., 2., 3.]),
    PeriodicCosinePotential(np.eye(3) * 5, [True] * 3, atoms=[1, 3], confinement=.1)])
def test_builtin_binding_owns_immutable_parameter_snapshot(provider):
    q = tensor(positions())
    bound = make_workflow_torch_pes(provider)
    before = bound.evaluate_tensor(q)
    if isinstance(provider, HarmonicPotential):
        provider.k[:] = 20
        provider.center[:] = -3
    else:
        provider.height[:] = 11
        provider.repeats[:] = 3
        provider.inverse[:] = 0
        provider.perpendicular[:] = 5
        provider.atoms[:] = [0, 2]
    after = bound.evaluate_tensor(q)
    assert torch.equal(before.energy, after.energy)
    assert torch.equal(before.forces, after.forces)


@pytest.mark.parametrize("cls", [FreePotential, HarmonicPotential])
def test_builtin_subclasses_and_evaluator_overrides_are_rejected(cls, monkeypatch):
    class Custom(cls):
        pass
    with pytest.raises(ValueError, match="no supported tensor"):
        make_workflow_torch_pes(Custom())
    provider = cls()
    provider.evaluate = lambda q: None
    with pytest.raises(ValueError, match="overridden"):
        make_workflow_torch_pes(provider)
    provider = cls()
    monkeypatch.setattr(cls, "evaluate", lambda self, q: None)
    with pytest.raises(ValueError, match="overridden"):
        make_workflow_torch_pes(provider)


@pytest.mark.parametrize("provider", [MorsePairPotential(), DoubleWellPotential(), object()])
def test_unsupported_models_have_no_numpy_fallback(provider):
    with pytest.raises(ValueError, match="no supported tensor"):
        make_workflow_torch_pes(provider)


@pytest.mark.parametrize("metadata", [{"stateful": True}, {"batch_independent": False},
    {"trajectory_batch_safe": False}, {"coupled_frames": True},
    {"ring_polymer_coupled": True}, {"capabilities": {"stateful": True}}])
def test_builtin_rejects_conflicting_metadata(metadata):
    provider = HarmonicPotential()
    provider.metadata = {**provider.metadata, **metadata}
    with pytest.raises(ValueError, match="stateless"):
        make_workflow_torch_pes(provider)


class Module(torch.nn.Module):
    metadata = {"batch_independent": True}

    def __init__(self, direct=False):
        super().__init__()
        self.register_buffer("k", torch.tensor(.4, dtype=torch.float64))
        self.direct = direct
        self.calls = 0

    def forward(self, q):
        self.calls += 1
        energy = .5 * self.k * q.square().sum((1, 2))
        return (energy, -self.k * q) if self.direct else energy


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("units", [None, {"positions": "angstrom", "energy": "eV"}])
def test_module_provider_units_one_call_and_borrowed_lifetime(direct, units):
    provider = TorchPotential(Module(direct), device="cpu", units=units, virial="none")
    q = positions()
    bound = compare(provider, q)
    before = provider.model.calls
    bound.evaluate_tensor(tensor(q))
    assert provider.model.calls == before + 1
    assert not provider.closed
    finite_difference(bound, q)
    provider.close()
    with pytest.raises(ValueError, match="closed"):
        bound.evaluate_tensor(tensor(q))


def test_module_periodic_autograd_virial_matches_numpy_and_affine_derivative():
    class Periodic(Module):
        def forward(self, q, cell, pbc):
            return .5 * self.k * q.square().sum((1, 2))
    provider = TorchPotential(Periodic(), device="cpu", cell=np.eye(3) * 5,
                              pbc=[True] * 3, periodic=True, virial="autograd",
                              units={"positions": "angstrom", "energy": "eV"})
    q = positions(2)
    bound = compare(provider, q)
    result = bound.evaluate_tensor(tensor(q))
    eps = 1e-6
    for a in range(3):
        for b in range(3):
            strain = np.eye(3)
            strain[a, b] += eps
            plus = bound.evaluate_tensor(tensor(q @ strain.T)).energy.numpy()
            strain[a, b] -= 2 * eps
            minus = bound.evaluate_tensor(tensor(q @ strain.T)).energy.numpy()
            np.testing.assert_allclose(result.virial[:, a, b].numpy(), -(plus-minus)/(2*eps),
                                       atol=3e-11, rtol=1e-8)


@pytest.mark.parametrize("kind", ["metadata", "float32", "closed", "delegate", "override"])
def test_module_provider_invalid_binding_does_not_change_precision(kind):
    provider = TorchPotential(Module(), device="cpu")
    if kind == "metadata":
        provider.metadata.pop("batch_independent")
    elif kind == "float32":
        provider.model.float()
    elif kind == "closed":
        provider.close()
    elif kind == "delegate":
        provider.delegate = object()
    else:
        provider.evaluate = lambda q: None
    with pytest.raises(ValueError):
        make_workflow_torch_pes(provider)
    if kind == "float32":
        assert provider.model.k.dtype == torch.float32


class Bridge:
    """T04's explicit as_torch_pes protocol, independent of its optional models."""
    def __init__(self):
        self.device = torch.device("cpu")
        self.metadata = {"batch_independent": True, "execution_device": "cpu", "dtype": "float64"}
        self.closed = False
        self.close_calls = self.bridge_calls = 0
        self.adapter = TorchPES(Module())

    def as_torch_pes(self):
        self.bridge_calls += 1
        return self.adapter

    def evaluate(self, q):
        raise AssertionError("NumPy evaluation must not run")

    def close(self):
        self.close_calls += 1
        self.closed = True


def test_explicit_ml_bridge_preserves_device_batch_and_ownership():
    source = Bridge()
    bound = make_workflow_torch_pes(source, device="cpu:0")
    assert isinstance(bound, TorchPES)
    assert source.bridge_calls == 1 and source.close_calls == 0
    q = tensor(positions())
    result = bound.evaluate_tensor(q)
    torch.testing.assert_close(result.energy, .2 * q.square().sum((1, 2)))
    torch.testing.assert_close(result.forces, -.4 * q)
    assert source.close_calls == 0
    source.close()
    with pytest.raises(ValueError, match="closed"):
        bound.evaluate_tensor(q)


@pytest.mark.parametrize("bad", [None, {}, {"batch_independent": False},
    {"batch_independent": True, "dtype": "float32"},
    {"batch_independent": True, "capabilities": "stateful"}])
def test_explicit_bridge_requires_source_declarations_before_calling(bad):
    source = Bridge()
    source.metadata = bad
    with pytest.raises(ValueError):
        make_workflow_torch_pes(source)
    assert source.bridge_calls == 0


@pytest.mark.parametrize("change", ["type", "independence", "precision", "device", "declaration"])
def test_explicit_bridge_return_contract_is_checked(change):
    source = Bridge()
    if change == "type":
        source.adapter = SimpleNamespace(device="cpu")
    elif change == "independence":
        source.adapter.metadata.pop("batch_independent")
    elif change == "precision":
        source.adapter.metadata["dtype"] = "float32"
    elif change == "device":
        source.adapter.device = torch.device("meta")
    else:
        source.adapter.metadata["execution_device"] = "meta"
    with pytest.raises(ValueError):
        make_workflow_torch_pes(source)


def test_bridge_errors_are_not_retried_or_swallowed():
    source = Bridge()
    failure = RuntimeError("backend failure including out of memory")
    def fail():
        source.bridge_calls += 1
        raise failure
    source.as_torch_pes = fail
    with pytest.raises(RuntimeError) as observed:
        make_workflow_torch_pes(source)
    assert observed.value is failure and source.bridge_calls == 1


@pytest.mark.parametrize("bad", ["object", "dtype", "shape", "virial"])
def test_bridge_evaluation_cannot_change_result_contract(bad):
    source = Bridge()
    bound = make_workflow_torch_pes(source)
    def invalid(q):
        energy, force, virial = q.new_zeros(len(q)), torch.zeros_like(q), None
        if bad == "object":
            return (energy, force)
        if bad == "dtype":
            force = force.float()
        elif bad == "shape":
            energy = energy[:, None]
        else:
            virial = q.new_zeros((len(q), 3))
        return TensorPESResult(energy, force, virial)
    source.adapter.evaluate_tensor = invalid
    with pytest.raises(ValueError):
        bound.evaluate_tensor(tensor(positions()))


def test_bound_source_device_mutation_rejects():
    source = Bridge()
    bound = make_workflow_torch_pes(source)
    source.device = torch.device("cuda:0")
    with pytest.raises(ValueError, match="device changed"):
        bound.evaluate_tensor(tensor(positions()))


@pytest.mark.parametrize("requested", ["meta", "mps", "cpu:3", False, "invalid"])
def test_invalid_devices_do_not_select_cpu(requested):
    with pytest.raises(ValueError, match="device must be"):
        make_workflow_torch_pes(FreePotential(), device=requested)


def test_unavailable_cuda_has_no_device_fallback(monkeypatch):
    original = torch.empty
    def unavailable(*args, **kwargs):
        if str(kwargs.get("device", "")).startswith("cuda"):
            raise RuntimeError("CUDA unavailable")
        return original(*args, **kwargs)
    monkeypatch.setattr(torch, "empty", unavailable)
    with pytest.raises(BackendUnavailableError, match="no device fallback"):
        make_workflow_torch_pes(FreePotential(), device="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware/runtime unavailable")
def test_cuda_builtin_parity_and_existing_cpu_provider_mismatch():
    provider = HarmonicPotential(k=[.2, .3, .4], center=[1, 2, 3])
    q = positions()
    expected = provider.evaluate(q)
    bound = make_workflow_torch_pes(provider, device="cuda")
    result = bound.evaluate_tensor(tensor(q, "cuda"))
    np.testing.assert_allclose(result.energy.cpu().numpy(), expected.energy, rtol=1e-13)
    np.testing.assert_allclose(result.forces.cpu().numpy(), expected.forces, rtol=1e-13)
    with pytest.raises(ValueError, match="differs from the configured"):
        make_workflow_torch_pes(Bridge(), device="cuda")


@pytest.mark.parametrize("provider", [FreePotential(), HarmonicPotential(),
    PeriodicCosinePotential(np.eye(3) * 5, [True] * 3), Bridge()])
def test_bound_propagation_has_no_per_step_host_transfer(provider, monkeypatch):
    bound = make_workflow_torch_pes(provider, device="cpu")
    dyn = TorchDynamics(bound, [1.] * 4, 5., 7, .03, thermostat="pile-l", integrator="cayley-bcocb")
    q = tensor(positions()).unsqueeze(0)
    state = TorchState(q, torch.zeros_like(q))
    item, calls = torch.Tensor.item, []
    def counted(value, *args, **kwargs):
        calls.append(1)
        return item(value, *args, **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError("implicit host transfer")
    monkeypatch.setattr(torch.Tensor, "item", counted)
    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    monkeypatch.setattr(torch.Tensor, "numpy", forbidden)
    monkeypatch.setattr(torch.Tensor, "tolist", forbidden)
    dyn.propagate(state, 5)
    assert len(calls) == 1  # The dynamics block's finite-status check only.
