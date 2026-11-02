"""Real, locally initialized NequIP models: batching and force references."""

import copy
from contextlib import contextmanager

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("nequip", minversion="0.19.1")
ase = pytest.importorskip("ase")

from nequip.model import NequIPGNNModel
from nequip.utils.global_state import set_global_state
from pyrpmd.potentials._batched_nequip import NequIPBatchedPotential
from pyrpmd.potentials.base import PotentialError
from pyrpmd.potentials.python import BOHR_ANGSTROM, EV_HARTREE


def small_nequip(dtype="float64"):
    """One local interaction block, random weights, no downloaded model."""
    set_global_state(allow_tf32=False)
    return NequIPGNNModel(
        seed=173, model_dtype=dtype, type_names=["H", "O"], r_max=3.0,
        num_layers=1, l_max=1, parity=False, num_features=4,
        radial_mlp_depth=1, radial_mlp_width=8, avg_num_neighbors=4.0,
        per_type_energy_scales=1.0, per_type_energy_shifts=0.0,
    ).eval()


@pytest.fixture(scope="module")
def model64():
    return small_nequip()


def configurations(batch):
    rng = np.random.default_rng(43)
    return np.array([[0.3, 0.2, 0.1], [2.1, 0.2, 0.1], [-0.2, 1.9, 0.1]]) + rng.normal(
        scale=0.04, size=(batch, 3, 3),
    )


@pytest.mark.parametrize("batch", [1, 2, 5])
@pytest.mark.parametrize("pbc", [[False] * 3, [True] * 3, [True, False, True]])
def test_nequip_batch_matches_ase_and_isotope_aliases(model64, batch, pbc):
    from nequip.integrations.ase import NequIPCalculator
    from nequip.integrations.utils import basic_transforms

    cell = np.array([[7.0, 0.0, 0.0], [0.3, 8.0, 0.0], [0.2, 0.4, 9.0]])
    provider = NequIPBatchedPotential(["O", "D", "T"], model64,
                                     cell=cell, pbc=pbc, virial=True)
    calls = []
    hook = model64.register_forward_hook(lambda _m, _args, _out: calls.append(1))
    q = configurations(batch)
    result = provider.evaluate(q)
    hook.remove()
    assert len(calls) == 1
    calculator = NequIPCalculator(
        model64, device="cpu", transforms=basic_transforms(
            model64.metadata, 3.0, ["H", "O"], {"H": "H", "O": "O"},
        ),
    )
    energy, forces, virial = [], [], []
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        for frame in q:
            atoms = ase.Atoms("OHH", positions=frame * BOHR_ANGSTROM,
                              cell=cell * BOHR_ANGSTROM, pbc=pbc)
            atoms.calc = calculator
            energy.append(atoms.get_potential_energy(force_consistent=True) * EV_HARTREE)
            forces.append(atoms.get_forces() * EV_HARTREE * BOHR_ANGSTROM)
            virial.append(-atoms.get_stress(voigt=False) * atoms.get_volume() * EV_HARTREE)
    finally:
        torch.set_default_dtype(old_dtype)
    np.testing.assert_allclose(result.energy, energy, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(result.forces, forces, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(result.virial, virial, rtol=1e-10, atol=1e-12)
    assert provider.metadata["native_batch"] is True
    assert provider.metadata["trajectory_batch_safe"] is True


@pytest.mark.parametrize("periodic", [False, True])
def test_nequip_forces_and_virial_match_energy_finite_differences(model64, periodic):
    cell = np.eye(3) * 7.0
    provider = NequIPBatchedPotential(["O", "H", "H"], model64, cell=cell,
                                     pbc=[periodic] * 3, virial=True)
    q = configurations(2)
    result = provider.evaluate(q)
    step = 2e-5
    for atom, component in [(0, 0), (1, 1), (2, 2)]:
        plus, minus = q.copy(), q.copy()
        plus[:, atom, component] += step
        minus[:, atom, component] -= step
        derivative = (provider.evaluate(plus).energy - provider.evaluate(minus).energy) / (2 * step)
        np.testing.assert_allclose(result.forces[:, atom, component], -derivative,
                                   rtol=2e-7, atol=2e-9)
    if periodic:
        values = []
        for sign in (1, -1):
            strain = np.eye(3)
            strain[0, 0] += sign * step
            provider.set_cell(cell @ strain)
            values.append(provider.evaluate(q @ strain).energy)
        np.testing.assert_allclose(result.virial[:, 0, 0], -(values[0] - values[1]) / (2 * step),
                                   rtol=2e-7, atol=2e-9)


def test_nequip_tensor_batch_changes_and_permutation(model64):
    provider = NequIPBatchedPotential(["O", "D", "T"], model64)
    q = torch.tensor(configurations(5), dtype=torch.float64)
    with torch.no_grad():
        batch = provider.evaluate_tensor(q)
        singles = [provider.evaluate_tensor(frame[None]) for frame in q]
        again = provider.evaluate_tensor(q[[4, 2, 0]])
    torch.testing.assert_close(batch.energy, torch.cat([r.energy for r in singles]), rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(batch.forces, torch.cat([r.forces for r in singles]), rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(again.forces, batch.forces[[4, 2, 0]], rtol=1e-10, atol=1e-12)
    assert not batch.energy.requires_grad
    assert batch.energy.dtype == q.dtype and batch.energy.device == q.device
    with pytest.raises(PotentialError, match="dtype"):
        provider.evaluate_tensor(q.float())


def test_nequip_float32_has_explicit_tolerance():
    provider = NequIPBatchedPotential(["O", "D", "T"], small_nequip("float32"), dtype="float32")
    q = torch.tensor(configurations(3), dtype=torch.float32)
    result = provider.evaluate_tensor(q)
    singles = [provider.evaluate_tensor(frame[None]) for frame in q]
    torch.testing.assert_close(result.energy, torch.cat([r.energy for r in singles]), rtol=3e-5, atol=3e-7)
    torch.testing.assert_close(result.forces, torch.cat([r.forces for r in singles]), rtol=3e-5, atol=3e-7)
    assert result.energy.dtype == torch.float64
    assert result.forces.dtype == torch.float32
    assert provider.metadata["energy_dtype"] == "float64"
    with pytest.raises(ValueError, match="precision"):
        NequIPBatchedPotential(["O", "H", "H"], small_nequip("float32"))


def test_nequip_compiled_torchscript_local_round_trip(model64, tmp_path):
    from nequip.model.inference_models.torchscript import save_torchscript_model

    path = tmp_path / "random.nequip.pth"
    metadata = dict(model64.metadata, allow_tf32="0")
    save_torchscript_model(copy.deepcopy(model64), metadata, str(path), "cpu")
    provider = NequIPBatchedPotential(["O", "D", "T"], path, virial=True)
    expected = NequIPBatchedPotential(["O", "D", "T"], model64, virial=True).evaluate(configurations(3))
    actual = provider.evaluate(configurations(3))
    np.testing.assert_allclose(actual.energy, expected.energy, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(actual.forces, expected.forces, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("options", [{"charge": 1}, {"multiplicity": 2}])
def test_nequip_rejects_unsupported_electronic_state(model64, options):
    with pytest.raises(ValueError, match="charge/spin"):
        NequIPBatchedPotential(["O", "H", "H"], model64, **options)


def test_nequip_failure_is_not_retried_by_frame(model64, monkeypatch):
    provider = NequIPBatchedPotential(["O", "H", "H"], model64)
    calls = []
    def fail(_data):
        calls.append(1)
        raise RuntimeError("test batch failed")
    monkeypatch.setattr(model64, "forward", fail)
    with pytest.raises(PotentialError, match="test batch failed"):
        provider.evaluate(configurations(5))
    assert calls == [1]
    provider.close()
    with pytest.raises(PotentialError, match="closed"):
        provider.evaluate(configurations(1))


def test_nequip_rejects_energy_only_model(model64):
    model = copy.deepcopy(model64)
    model.model.do_derivatives = False
    with pytest.raises(ValueError, match="energy-gradient forces"):
        NequIPBatchedPotential(["O", "H", "H"], model)


def fake_aoti_metadata():
    from nequip.scripts._compile_utils import BATCH_INPUTS, ASE_OUTPUTS
    from nequip.utils.aoti_metadata import (
        NEQUIP_AOTI_INPUTS_KEY, NEQUIP_AOTI_OUTPUTS_KEY, serialize_aoti_keys,
    )

    return {
        "r_max": 3.0, "type_names": ["H", "O"], "model_dtype": "float64",
        "allow_tf32": "0", "AOTI_DEVICE_KEY": "cpu",
        NEQUIP_AOTI_INPUTS_KEY: serialize_aoti_keys(BATCH_INPUTS),
        NEQUIP_AOTI_OUTPUTS_KEY: serialize_aoti_keys(ASE_OUTPUTS),
    }


@pytest.mark.parametrize("key,value,match", [
    ("nequip_aoti_inputs", None, "recompile with --target batch"),
    ("nequip_aoti_outputs", None, "recompile with --target batch"),
    ("nequip_aoti_inputs", "pos edge_index atom_types cell edge_cell_shift", "batch input/output"),
    ("nequip_aoti_outputs", "total_energy forces", "batch input/output"),
    ("AOTI_DEVICE_KEY", None, "compile device"),
    ("AOTI_DEVICE_KEY", "cuda:0", "compile device"),
    ("model_dtype", None, "precision metadata"),
    ("allow_tf32", None, "precision metadata"),
    ("model_dtype", "float32", "precision differs"),
    ("allow_tf32", "1", "TF32"),
])
def test_nequip_aoti_requires_explicit_batch_and_precision_metadata(tmp_path, monkeypatch,
                                                                 key, value, match):
    import nequip.model.inference_models as loaders

    metadata = fake_aoti_metadata()
    if value is None:
        metadata.pop(key)
    else:
        metadata[key] = value
    path = tmp_path / "fake.nequip.pt2"
    path.touch()
    monkeypatch.setattr(loaders, "load_aotinductor_model",
                        lambda *_args: (torch.nn.Module(), metadata))
    with pytest.raises(ValueError, match=match):
        NequIPBatchedPotential(["O", "D", "T"], path)


def test_nequip_aoti_receives_contiguous_native_device_batch(tmp_path, monkeypatch):
    import nequip.model.inference_models as loaders
    from nequip.data import AtomicDataDict
    from nequip.scripts._compile_utils import BATCH_INPUTS, ASE_OUTPUTS

    observed = []
    class FakeCompiled(torch.nn.Module):
        def forward(self, data):
            assert all(value.is_contiguous() for value in data.values())
            assert all(value.device == torch.device("cpu") for value in data.values())
            assert data["pos"].dtype == data["cell"].dtype == torch.float64
            frames = data[AtomicDataDict.NUM_NODES_KEY].shape[0]
            observed.append(frames)
            return {"total_energy": torch.zeros(frames, dtype=torch.float64),
                    "forces": torch.zeros_like(data["pos"])}

    def fake_load(path, device, inputs, outputs):
        assert device == torch.device("cpu")
        assert inputs == list(BATCH_INPUTS) and outputs == list(ASE_OUTPUTS)
        return FakeCompiled(), fake_aoti_metadata()

    path = tmp_path / "fake.nequip.pt2"
    path.touch()
    monkeypatch.setattr(loaders, "load_aotinductor_model", fake_load)
    provider = NequIPBatchedPotential(["O", "D", "T"], path)
    q = torch.tensor(configurations(2), dtype=torch.float64).transpose(1, 2)
    assert not q.is_contiguous()
    result = provider.evaluate_tensor(q)
    assert observed == [2]
    assert result.forces.shape == q.shape and result.forces.dtype == q.dtype


@pytest.mark.parametrize("ordinal", [0, 1])
def test_nequip_aoti_type_only_cuda_metadata_uses_requested_device(tmp_path, monkeypatch, ordinal):
    import nequip.model.inference_models as loaders

    # Exercise device selection/metadata on CPU without pretending to execute
    # either a CUDA kernel or an AOTI artifact.
    provider = object.__new__(NequIPBatchedPotential)
    provider.torch = torch
    provider.device = torch.device(f"cuda:{ordinal}")
    path = tmp_path / "fake.nequip.pt2"
    path.touch()
    observed = []

    @contextmanager
    def device_context(selected):
        observed.append(("enter", selected))
        try:
            yield
        finally:
            observed.append(("exit", selected))

    def fake_load(_path, device, _inputs, _outputs):
        assert device == torch.device("cuda")
        assert observed == [("enter", provider.device)]
        metadata = fake_aoti_metadata()
        metadata["AOTI_DEVICE_KEY"] = "cuda"
        return torch.nn.Module(), metadata

    monkeypatch.setattr(torch.cuda, "device", device_context)
    monkeypatch.setattr(loaders, "load_aotinductor_model", fake_load)
    provider._load_model(path)
    assert observed == [("enter", provider.device), ("exit", provider.device)]
    wrong_device = fake_aoti_metadata()
    wrong_device["AOTI_DEVICE_KEY"] = f"cuda:{1 - ordinal}"
    with pytest.raises(ValueError, match="compile device differs"):
        provider._check_aoti_metadata(wrong_device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_nequip_cuda_batch_matches_cpu_without_coordinate_host_transfer(model64, monkeypatch):
    pytest.importorskip("nvalchemiops")
    q = torch.tensor(configurations(3), dtype=torch.float64)
    expected = NequIPBatchedPotential(["O", "H", "H"], model64).evaluate_tensor(q)
    provider = NequIPBatchedPotential(["O", "D", "T"], copy.deepcopy(model64), device="cuda")
    q_cuda = q.to("cuda")
    def no_cpu(_self, *args, **kwargs):
        raise AssertionError("evaluate_tensor must not move coordinates to CPU")
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", no_cpu)
        result = provider.evaluate_tensor(q_cuda)
    torch.testing.assert_close(result.energy.cpu(), expected.energy, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(result.forces.cpu(), expected.forces, rtol=1e-10, atol=1e-12)
