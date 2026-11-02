"""FAIR-Chem CPU integration with randomly initialized local checkpoints."""

import os
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fairchem.core", minversion="2.25.0")
ase = pytest.importorskip("ase")

from fairchem.core import FAIRChemCalculator
from fairchem.core.models.base import HydraModel
from fairchem.core.models.uma.escn_md import eSCNMDBackbone  # register model
from fairchem.core.units.mlip_unit.api.inference import MLIPInferenceCheckpoint
from pyrpmd.potentials._batched_fairchem import FairchemBatchedPotential
from pyrpmd.potentials.base import PotentialError
from pyrpmd.potentials.python import BOHR_ANGSTROM, EV_HARTREE


def random_checkpoint(path, dtype="float64", task_name="omol"):
    """Export a tiny untrained eSCN model using FAIR-Chem's native format."""
    config = {
        "_target_": "fairchem.core.models.base.HydraModel",
        "backbone": {
            "model": "escnmd_backbone", "max_num_elements": 100,
            "sphere_channels": 4, "lmax": 1, "mmax": 1,
            "otf_graph": True, "max_neighbors": 24, "cutoff": 3.0,
            "edge_channels": 4, "num_distance_basis": 8,
            "direct_forces": False, "regress_forces": True,
            "regress_stress": True, "num_layers": 1, "hidden_channels": 8,
            "act_type": "gate", "ff_type": "spectral",
            "use_dataset_embedding": True,
            "dataset_mapping": {task_name: task_name}, "always_use_pbc": False,
        },
        "heads": {
            "energy_force_stress": {
                "module": "fairchem.core.models.uma.escn_md.MLP_EFS_Head",
                "prefix": task_name,
            },
        },
        "pass_through_head_outputs": True, "supports_single_atoms": True,
    }
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(getattr(torch, dtype))
    try:
        with torch.random.fork_rng():
            torch.manual_seed(29)
            model = HydraModel(**{k: v for k, v in config.items() if k != "_target_"}).eval()
        tasks = []
        for prop, level, dim in (("energy", "system", [1]),
                                 ("forces", "atom", [3]),
                                 ("stress", "system", [1, 9])):
            tasks.append({
                "_target_": "fairchem.core.units.mlip_unit.mlip_unit.Task",
                "name": f"{task_name}_{prop}", "level": level, "property": prop,
                "out_spec": {"dim": dim, "dtype": dtype},
                "datasets": [task_name],
                "normalizer": {
                    "_target_": "fairchem.core.modules.normalization.normalizer.Normalizer",
                    "mean": 0.0, "rmsd": 1.0,
                },
            })
        checkpoint = MLIPInferenceCheckpoint(
            model_config=config, model_state_dict=model.state_dict(),
            ema_state_dict=torch.optim.swa_utils.AveragedModel(model).state_dict(),
            tasks_config=tasks,
        )
        torch.save(checkpoint, path)
    finally:
        torch.set_default_dtype(old_dtype)
    return path


@pytest.fixture(scope="module")
def checkpoint64(tmp_path_factory):
    return random_checkpoint(tmp_path_factory.mktemp("fairchem_random") / "model.pt")


def configurations(batch):
    rng = np.random.default_rng(19)
    return np.array([[0.3, 0.2, 0.1], [2.1, 0.2, 0.1], [-0.2, 1.9, 0.1]]) + rng.normal(
        scale=0.04, size=(batch, 3, 3),
    )


@pytest.mark.parametrize("batch", [1, 2, 5])
@pytest.mark.parametrize("periodic", [False, True])
def test_fairchem_batch_matches_ase_and_isotopes(checkpoint64, batch, periodic):
    cell = np.array([[7.0, 0.0, 0.0], [0.2, 8.0, 0.0], [0.3, 0.4, 9.0]])
    provider = FairchemBatchedPotential(["O", "D", "T"], checkpoint64,
                                       task_name="omol", cell=cell,
                                       pbc=[periodic] * 3, virial=True)
    q = configurations(batch)
    calls = []
    hook = provider.predictor.model.register_forward_hook(lambda _m, _args, _out: calls.append(1))
    actual = provider.evaluate(q)
    hook.remove()
    assert len(calls) == 1
    calculator = FAIRChemCalculator(provider.predictor, task_name="omol")
    energy, forces, virial = [], [], []
    for frame in q:
        atoms = ase.Atoms("OHH", positions=frame * BOHR_ANGSTROM,
                          cell=cell * BOHR_ANGSTROM, pbc=[periodic] * 3)
        atoms.info.update(charge=0, spin=1)
        atoms.calc = calculator
        energy.append(atoms.get_potential_energy(force_consistent=True) * EV_HARTREE)
        forces.append(atoms.get_forces() * EV_HARTREE * BOHR_ANGSTROM)
        virial.append(-atoms.get_stress(voigt=False) * atoms.get_volume() * EV_HARTREE)
    np.testing.assert_allclose(actual.energy, energy, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(actual.forces, forces, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(actual.virial, virial, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("periodic", [False, True])
def test_fairchem_finite_difference_forces_and_virial(checkpoint64, periodic):
    cell = np.eye(3) * 7.0
    provider = FairchemBatchedPotential(["O", "H", "H"], checkpoint64,
                                       task_name="omol", cell=cell,
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
                                   rtol=3e-6, atol=2e-9)
    if periodic:
        values = []
        for sign in (1, -1):
            strain = np.eye(3)
            strain[0, 0] += sign * step
            provider.set_cell(cell @ strain)
            values.append(provider.evaluate(q @ strain).energy)
        np.testing.assert_allclose(result.virial[:, 0, 0], -(values[0] - values[1]) / (2 * step),
                                   rtol=3e-6, atol=2e-9)


def test_fairchem_tensor_charge_spin_and_batch_permutation(checkpoint64):
    provider = FairchemBatchedPotential(["O", "D", "T"], checkpoint64,
                                       task_name="omol", charge=1, multiplicity=2)
    q = torch.tensor(configurations(3), dtype=torch.float64)
    seen = []
    hook = provider.predictor.model.register_forward_pre_hook(
        lambda _m, args: seen.append((args[0].charge.clone(), args[0].spin.clone())),
    )
    with torch.no_grad():
        result = provider.evaluate_tensor(q)
    hook.remove()
    assert len(seen) == 1
    assert seen[0][0].tolist() == [1, 1, 1]
    assert seen[0][1].tolist() == [2, 2, 2]
    reordered = provider.evaluate_tensor(q[[2, 0]])
    torch.testing.assert_close(reordered.energy, result.energy[[2, 0]], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(reordered.forces, result.forces[[2, 0]], rtol=1e-10, atol=1e-12)
    assert result.energy.dtype == q.dtype and result.energy.device == q.device
    with pytest.raises(PotentialError, match="dtype"):
        provider.evaluate_tensor(q.float())


def test_fairchem_float32_tolerance_and_precision_rejection(tmp_path):
    path = random_checkpoint(tmp_path / "float32.pt", "float32")
    provider = FairchemBatchedPotential(["O", "D", "T"], path,
                                       task_name="omol", dtype="float32")
    q = torch.tensor(configurations(3), dtype=torch.float32)
    batch = provider.evaluate_tensor(q)
    assert batch.energy.dtype == torch.float64
    assert batch.forces.dtype == torch.float32
    singles = [provider.evaluate_tensor(frame[None]) for frame in q]
    torch.testing.assert_close(batch.energy, torch.cat([r.energy for r in singles]), rtol=3e-5, atol=3e-7)
    torch.testing.assert_close(batch.forces, torch.cat([r.forces for r in singles]), rtol=3e-5, atol=3e-7)
    with pytest.raises(ValueError, match="dtype"):
        FairchemBatchedPotential(["O", "H", "H"], path, task_name="omol")


def test_fairchem_unsupported_state_and_inconsistent_normalization(checkpoint64):
    with pytest.raises(ValueError, match="other than omol"):
        FairchemBatchedPotential(["O", "H", "H"], checkpoint64, task_name="omat", charge=1)
    provider = FairchemBatchedPotential(["O", "H", "H"], checkpoint64, task_name="omol")
    predictor = provider.predictor
    predictor.model.module.backbone.regress_config.direct_forces = True
    with pytest.raises(ValueError, match="conservative"):
        FairchemBatchedPotential(["O", "H", "H"], predictor, task_name="omol")
    predictor.model.module.backbone.regress_config.direct_forces = False
    force_task = next(t for t in predictor.dataset_to_tasks["omol"] if t.property == "forces")
    force_task.normalizer.rmsd.fill_(2.0)
    with pytest.raises(ValueError, match="normalization"):
        FairchemBatchedPotential(["O", "H", "H"], predictor, task_name="omol")


def test_fairchem_batch_failure_is_not_retried(checkpoint64, monkeypatch):
    provider = FairchemBatchedPotential(["O", "H", "H"], checkpoint64, task_name="omol")
    calls = []
    def fail(_data):
        calls.append(1)
        raise RuntimeError("test native batch failure")
    monkeypatch.setattr(provider.predictor, "predict", fail)
    with pytest.raises(PotentialError, match="test native batch failure"):
        provider.evaluate(configurations(5))
    assert calls == [1]
    provider.close()
    with pytest.raises(PotentialError, match="closed"):
        provider.evaluate(configurations(1))


def test_fairchem_rebind_checks_stress_before_changing_boundary(checkpoint64):
    provider = FairchemBatchedPotential(["O", "H", "H"], checkpoint64,
                                       task_name="omol", virial=True)
    stress_task = next(t for t in provider.predictor.dataset_to_tasks["omol"]
                       if t.property == "stress")
    stress_task.normalizer.rmsd.fill_(2.0)
    with pytest.raises(ValueError, match="stress normalization"):
        provider.bind_system({"cell": np.eye(3) * 7.0, "pbc": [True] * 3})
    assert provider.cell is None and not provider.pbc.any()


def test_fairchem_construction_preserves_rng_and_environment(checkpoint64, monkeypatch):
    from fairchem.core.common.distutils import CURRENT_DEVICE_TYPE_STR

    monkeypatch.setenv(CURRENT_DEVICE_TYPE_STR, "cpu")
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    FairchemBatchedPotential(["O", "H", "H"], checkpoint64, task_name="omol")
    assert random.getstate() == python_state
    restored = np.random.get_state()
    assert restored[0] == numpy_state[0] and restored[2:] == numpy_state[2:]
    np.testing.assert_array_equal(restored[1], numpy_state[1])
    torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)
    assert os.environ[CURRENT_DEVICE_TYPE_STR] == "cpu"


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices are required")
def test_fairchem_honors_explicit_cuda_index(checkpoint64):
    original_device = torch.cuda.current_device()
    provider = FairchemBatchedPotential(["O", "D", "T"], checkpoint64,
                                       task_name="omol", device="cuda:1")
    assert torch.cuda.current_device() == original_device
    q = torch.tensor(configurations(2), dtype=torch.float64, device="cuda:1")
    result = provider.evaluate_tensor(q)
    assert result.energy.device == result.forces.device == torch.device("cuda:1")
