"""Real exported metatomic models, without downloading pretrained weights."""

from itertools import product
from typing import Dict, List, Optional

import numpy as np
import pytest

torch = pytest.importorskip("torch")
mta = pytest.importorskip("metatomic.torch")
pytest.importorskip("vesin.metatomic")
pytest.importorskip("vesin_torch")
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelOutput, NeighborListOptions, System

from pyrpmd.potentials.batched_ml import MetatomicBatchedPotential
from pyrpmd.potentials.base import PotentialError
from pyrpmd.potentials.python import BOHR_ANGSTROM, EV_HARTREE


class RandomPairModel(torch.nn.Module):
    """Trainable radial polynomial with random parameters and exact derivatives."""

    def __init__(self, dtype=torch.float64, electronic=False, reverse=False):
        super().__init__()
        generator = torch.Generator().manual_seed(82617)
        self.coefficients = torch.nn.Parameter(
            0.1 + 0.2 * torch.rand(3, generator=generator, dtype=dtype)
        )
        self.cutoff = 2.4
        self.neighbors = NeighborListOptions(self.cutoff, full_list=False, strict=True)
        self.electronic = electronic
        self.reverse = reverse

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [self.neighbors]

    def requested_inputs(self) -> Dict[str, ModelOutput]:
        if self.electronic:
            return {
                "charge": ModelOutput(unit="e", sample_kind="system"),
                "spin_multiplicity": ModelOutput(unit="", sample_kind="system"),
            }
        return {}

    def forward(self, systems: List[System], outputs: Dict[str, ModelOutput],
                selected_atoms: Optional[Labels] = None) -> Dict[str, TensorMap]:
        energies = torch.jit.annotate(List[torch.Tensor], [])
        for system in systems:
            neighbors = system.get_neighbor_list(self.neighbors)
            vectors = neighbors.values[:, :, 0]
            pairs = neighbors.samples.values[:, :2].to(dtype=torch.long)
            radial = 1.0 - (vectors * vectors).sum(dim=1) / self.cutoff**2
            species = (system.types[pairs[:, 0]] + system.types[pairs[:, 1]]).to(
                dtype=system.positions.dtype
            )
            pair_energy = species * (
                self.coefficients[0] * radial**4
                + self.coefficients[1] * radial**5
                + self.coefficients[2] * radial**6
            )
            energy = pair_energy.sum()
            if self.electronic:
                energy = (energy + 0.123 * system.get_data("charge").block().values.sum()
                          + 0.321 * (
                              system.get_data("spin_multiplicity").block().values.sum() - 1
                          ))
            energies.append(energy)
        values = torch.stack(energies).reshape(-1, 1)
        indices = torch.arange(len(systems), device=values.device).reshape(-1, 1)
        if self.reverse:
            values = values.flip(0)
            indices = indices.flip(0)
        block = TensorBlock(
            values=values,
            samples=Labels(["system"], indices),
            components=torch.jit.annotate(List[Labels], []),
            properties=Labels(["energy"], torch.zeros((1, 1), device=values.device,
                                                     dtype=torch.int32)),
        )
        return {"energy": TensorMap(
            Labels(["_"], torch.zeros((1, 1), device=values.device, dtype=torch.int32)),
            [block],
        )}


class AnisotropicModel(torch.nn.Module):
    """Noncentral forces expose transposed virials hidden by radial models."""

    def __init__(self):
        super().__init__()
        self.register_buffer("matrix", torch.tensor(
            [[1.0, 0.3, 0.2], [0.3, 2.0, -0.4], [0.2, -0.4, 4.0]],
            dtype=torch.float64,
        ))

    def forward(self, systems: List[System], outputs: Dict[str, ModelOutput],
                selected_atoms: Optional[Labels] = None) -> Dict[str, TensorMap]:
        energies = torch.jit.annotate(List[torch.Tensor], [])
        for system in systems:
            delta = system.positions[1] - system.positions[0]
            energies.append(0.5 * (delta * (self.matrix @ delta)).sum())
        values = torch.stack(energies).reshape(-1, 1)
        zero = torch.zeros((1, 1), dtype=torch.int32, device=values.device)
        block = TensorBlock(
            values=values,
            samples=Labels(["system"], torch.arange(
                len(systems), dtype=torch.int32, device=values.device,
            ).reshape(-1, 1)),
            components=torch.jit.annotate(List[Labels], []),
            properties=Labels(["energy"], zero),
        )
        return {"energy": TensorMap(Labels(["_"], zero), [block])}


def _model(dtype=torch.float64, electronic=False, reverse=False):
    module = RandomPairModel(dtype=dtype, electronic=electronic, reverse=reverse).eval()
    return mta.AtomisticModel(
        module,
        mta.ModelMetadata(name="PyRPMD deterministic random radial test model"),
        mta.ModelCapabilities(
            outputs={"energy": ModelOutput(unit="eV", sample_kind="system")},
            atomic_types=[1, 8], interaction_range=module.cutoff, length_unit="angstrom",
            supported_devices=["cpu", "cuda"], dtype=str(dtype).removeprefix("torch."),
        ),
    )


def _coordinates(frames):
    generator = np.random.default_rng(7261)
    reference = np.array([[0.2, 0.3, 0.5], [2.1, 0.5, 0.2], [7.6, 0.2, 0.4]])
    return reference + generator.normal(scale=0.04, size=(frames, 3, 3))


def _reference(x, coefficients, cell=None, pbc=(False, False, False), charge=0, mult=1):
    """Independent NumPy image enumeration and analytic radial derivatives."""
    positions = x * BOHR_ANGSTROM
    lattice = np.zeros((3, 3)) if cell is None else cell * BOHR_ANGSTROM
    cutoff = 2.4
    energy = np.zeros(len(x))
    force = np.zeros_like(x)
    virial = np.zeros((len(x), 3, 3))
    species = [8, 1, 1]
    shifts = list(product(*[(-1, 0, 1) if flag else (0,) for flag in pbc]))
    for frame, q in enumerate(positions):
        for i in range(3):
            for j in range(i, 3):
                for shift in shifts:
                    if i == j and shift <= (0, 0, 0):
                        continue
                    vector = q[j] - q[i] + np.asarray(shift) @ lattice
                    radius2 = vector @ vector
                    if radius2 >= cutoff**2:
                        continue
                    u = 1.0 - radius2 / cutoff**2
                    weight = species[i] + species[j]
                    value = weight * sum(c * u**p for c, p in zip(coefficients, (4, 5, 6)))
                    derivative = (-2 * weight / cutoff**2 * vector
                                  * sum(p * c * u**(p - 1)
                                        for c, p in zip(coefficients, (4, 5, 6))))
                    energy[frame] += value
                    force[frame, i] += derivative
                    force[frame, j] -= derivative
                    virial[frame] -= np.outer(vector, derivative)
        energy[frame] += 0.123 * charge + 0.321 * (mult - 1)
    return energy * EV_HARTREE, force * EV_HARTREE * BOHR_ANGSTROM, virial * EV_HARTREE


@pytest.mark.parametrize("frames", [1, 2, 5])
@pytest.mark.parametrize("pbc", [(False, False, False), (True, True, True), (True, False, False)])
def test_exported_batch_matches_frames_and_analytic_reference(tmp_path, frames, pbc):
    model = _model()
    coefficients = model.module.coefficients.detach().numpy().copy()
    checkpoint = tmp_path / "random-metatomic.pt"
    model.save(str(checkpoint))
    cell = np.array([[8.0, 0.3, 0.1], [0.0, 8.2, 0.2], [0.0, 0.0, 8.1]])
    potential = MetatomicBatchedPotential(["O", "D", "T"], checkpoint, cell=cell,
                                         pbc=pbc, virial=True)
    q = _coordinates(frames)
    actual = potential.evaluate(q)
    singles = [potential.evaluate(frame[None]) for frame in q]
    expected = _reference(q, coefficients, cell=cell, pbc=pbc)
    for name, reference in zip(("energy", "forces", "virial"), expected):
        np.testing.assert_allclose(getattr(actual, name), reference, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(
            getattr(actual, name), np.concatenate([getattr(result, name) for result in singles]),
            rtol=1e-10, atol=1e-12,
        )
    assert potential.metadata["native_batch"] is True
    assert potential.metadata["batch_independent"] is True
    assert potential.metadata["trajectory_batch_safe"] is True


def test_one_model_call_for_all_frames_and_tensor_output_stays_on_device():
    potential = MetatomicBatchedPotential(["O", "H", "H"], _model(), virial=True)
    calls = []

    def record_call(module, args, kwargs):
        calls.append(len(kwargs["systems"]))

    hook = potential.model.register_forward_pre_hook(record_call, with_kwargs=True)
    q = torch.tensor(_coordinates(4), dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        result = potential.evaluate_tensor(q)
    hook.remove()
    assert calls == [4]
    assert q.grad is None
    for value in (result.energy, result.forces, result.virial):
        assert value.device == q.device
        assert value.dtype == q.dtype
        assert not value.requires_grad
    assert all(parameter.grad is None for parameter in potential.model.parameters())


def test_reverse_samples_are_reordered_without_mixing_frames():
    q = _coordinates(4)
    expected = MetatomicBatchedPotential(["O", "H", "H"], _model()).evaluate(q)
    actual = MetatomicBatchedPotential(["O", "H", "H"], _model(reverse=True)).evaluate(q)
    np.testing.assert_allclose(actual.energy, expected.energy, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(actual.forces, expected.forces, rtol=1e-12, atol=1e-14)


def test_charge_and_spin_are_passed_as_requested_system_inputs():
    model = _model(electronic=True)
    coefficients = model.module.coefficients.detach().numpy().copy()
    potential = MetatomicBatchedPotential(["O", "D", "T"], model, charge=-1,
                                         multiplicity=2, virial=True)
    q = _coordinates(3)
    actual = potential.evaluate(q)
    expected = _reference(q, coefficients, charge=-1, mult=2)
    for name, reference in zip(("energy", "forces", "virial"), expected):
        np.testing.assert_allclose(getattr(actual, name), reference, rtol=1e-10, atol=1e-12)


def test_explicit_float32_precision_and_no_implicit_tensor_cast():
    model = _model(dtype=torch.float32)
    coefficients = model.module.coefficients.detach().numpy().copy()
    potential = MetatomicBatchedPotential(["O", "H", "H"], model, dtype="float32")
    q = _coordinates(3)
    result = potential.evaluate_tensor(torch.tensor(q, dtype=torch.float32))
    expected = _reference(q.astype(np.float32), coefficients)
    assert result.energy.dtype == torch.float32
    np.testing.assert_allclose(result.energy.numpy(), expected[0], rtol=3e-6, atol=3e-8)
    np.testing.assert_allclose(result.forces.numpy(), expected[1], rtol=3e-5, atol=3e-8)
    with pytest.raises((ValueError, PotentialError), match="dtype|float32"):
        potential.evaluate_tensor(torch.tensor(q, dtype=torch.float64))
    with pytest.raises(ValueError, match="dtype"):
        MetatomicBatchedPotential(["O", "H", "H"], _model(dtype=torch.float32))


@pytest.mark.parametrize("kwargs, message", [
    ({"charge": 1}, "charge"),
    ({"multiplicity": 2}, "multiplicity"),
    ({"check_consistency": "yes"}, "check_consistency"),
])
def test_unsupported_electronic_state_and_options_fail_explicitly(kwargs, message):
    with pytest.raises(ValueError, match=message):
        MetatomicBatchedPotential(["O", "H", "H"], _model(), **kwargs)


def test_bind_system_updates_neighbors_and_checks_composition():
    model = _model()
    coefficients = model.module.coefficients.detach().numpy().copy()
    potential = MetatomicBatchedPotential(["O", "H", "H"], model, virial=True)
    q = _coordinates(2)
    potential.evaluate(q)
    cell = np.eye(3) * 8
    potential.bind_system({"symbols": ["O", "D", "T"], "cell": cell, "pbc": [True]*3})
    actual = potential.evaluate(q)
    expected = _reference(q, coefficients, cell=cell, pbc=[True]*3)
    for name, reference in zip(("energy", "forces", "virial"), expected):
        np.testing.assert_allclose(getattr(actual, name), reference, rtol=1e-10, atol=1e-12)
    with pytest.raises(PotentialError, match="composition"):
        potential.bind_system({"symbols": ["O", "O", "H"]})
    potential.close()
    with pytest.raises(PotentialError, match="closed"):
        potential.evaluate(q)


def test_forces_match_finite_differences():
    potential = MetatomicBatchedPotential(["O", "H", "H"], _model())
    q = _coordinates(2)
    result = potential.evaluate(q)
    displacement = 1e-5
    for atom, axis in [(0, 0), (1, 1), (2, 2)]:
        plus, minus = q.copy(), q.copy()
        plus[:, atom, axis] += displacement
        minus[:, atom, axis] -= displacement
        force = -(potential.evaluate(plus).energy - potential.evaluate(minus).energy) / (2*displacement)
        np.testing.assert_allclose(result.forces[:, atom, axis], force, rtol=2e-8, atol=2e-10)


def test_virial_axis_order_with_analytic_anisotropic_force():
    module = AnisotropicModel().eval()
    matrix = module.matrix.numpy().copy()
    model = mta.AtomisticModel(
        module, mta.ModelMetadata(), mta.ModelCapabilities(
            outputs={"energy": ModelOutput(unit="eV", sample_kind="system")},
            atomic_types=[1, 8], interaction_range=0.0, length_unit="angstrom",
            supported_devices=["cpu"], dtype="float64",
        ),
    )
    potential = MetatomicBatchedPotential(["O", "H", "H"], model, virial=True)
    q = _coordinates(2)
    result = potential.evaluate(q)
    delta = (q[:, 1] - q[:, 0]) * BOHR_ANGSTROM
    gradient = delta @ matrix.T
    force = np.zeros_like(q)
    force[:, 0] = gradient * EV_HARTREE * BOHR_ANGSTROM
    force[:, 1] = -force[:, 0]
    virial = -np.einsum("bi,bj->bij", gradient, delta) * EV_HARTREE
    assert not np.allclose(virial, virial.transpose(0, 2, 1))
    np.testing.assert_allclose(result.forces, force, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(result.virial, virial, rtol=1e-12, atol=1e-14)
    potential.close()
    potential.close()
    assert potential.model is None
    assert not potential._neighbors


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_tensors_and_neighbors_remain_on_cuda():
    potential = MetatomicBatchedPotential(["O", "H", "H"], _model(), device="cuda",
                                         cell=np.eye(3)*8, pbc=[True]*3, virial=True)
    q = torch.tensor(_coordinates(2), dtype=torch.float64, device="cuda")
    result = potential.evaluate_tensor(q)
    reference = MetatomicBatchedPotential(["O", "H", "H"], _model(),
                                         cell=np.eye(3)*8, pbc=[True]*3, virial=True).evaluate(q.cpu().numpy())
    for name in ("energy", "forces", "virial"):
        value = getattr(result, name)
        assert value.device == q.device
        np.testing.assert_allclose(value.cpu().numpy(), getattr(reference, name), rtol=1e-10, atol=1e-12)
