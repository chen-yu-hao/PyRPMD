"""Real, untrained MACE: independent ASE and energy-derivative references."""

from copy import deepcopy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mace.modules")
pytest.importorskip("vesin_torch")
pytest.importorskip("ase")

from mace.calculators import MACECalculator

from examples.potentials._random_mace import make_random_mace
from pyrpmd.potentials.adapters import ModelASEPotential
from pyrpmd.potentials.batched_ml import MACEBatchedPotential
from pyrpmd.potentials.python import BOHR_ANGSTROM


@pytest.fixture(scope="module", params=["float64", "float32"])
def mace_model(request):
    return request.param, make_random_mace(dtype=request.param)


def _geometry(batch):
    rng = np.random.default_rng(2034)
    base_angstrom = np.array([
        [-0.2, 0.1, 0.2], [0.8, 0.2, 0.0],
        [2.8, 1.7, -0.1], [1.9, 2.1, 0.3],
    ])
    cell_angstrom = np.array([[3.4, 0.2, 0.1], [0.0, 3.8, 0.15], [0.0, 0.1, 4.2]])
    q = (base_angstrom + rng.normal(scale=0.035, size=(batch, 4, 3))) / BOHR_ANGSTROM
    return q, cell_angstrom / BOHR_ANGSTROM


def _ase_reference(model, dtype, symbols, cell, pbc):
    calculator = MACECalculator(
        models=deepcopy(model), device="cpu", default_dtype=dtype,
        compute_stress=True,
    )
    return ModelASEPotential(
        symbols, calculator, provider="mace", cell=cell, pbc=pbc,
    )


@pytest.mark.parametrize("batch", [1, 2, 5])
@pytest.mark.parametrize("pbc", [[False, False, False], [True, True, True], [True, False, True]])
def test_mace_real_batch_matches_independent_ase(mace_model, batch, pbc):
    dtype, model = mace_model
    q, cell = _geometry(batch)
    symbols = ["D", "T", "H", "O"]
    tolerance = {"rtol": 1e-10, "atol": 2e-14} if dtype == "float64" else {
        "rtol": 3e-5, "atol": 3e-9,
    }
    calls = []
    batched_model = deepcopy(model)
    hook = batched_model.register_forward_pre_hook(lambda _, args: calls.append(len(args[0]["ptr"]) - 1))
    try:
        with MACEBatchedPotential(symbols, batched_model, dtype=dtype, cell=cell,
                                  pbc=pbc, virial=True) as provider:
            result = provider.evaluate(q)
            assert calls == [batch], "one actual model forward must handle the entire batch"
            assert provider.metadata["native_batch"] is True
            assert provider.metadata["batch_independent"] is True
            assert provider.metadata["trajectory_batch_safe"] is True
            with _ase_reference(model, dtype, symbols, cell, pbc) as scalar:
                expected = scalar.evaluate(q)
            np.testing.assert_allclose(result.energy, expected.energy, **tolerance)
            np.testing.assert_allclose(result.forces, expected.forces, **tolerance)
            # Upstream MACE intentionally masks molecular ASE stress to zero.
            # Its configurational virial is checked by affine derivatives below.
            if any(pbc):
                np.testing.assert_allclose(result.virial, expected.virial, **tolerance)

            tensor = provider.evaluate_tensor(torch.as_tensor(q, dtype=getattr(torch, dtype)))
            assert tensor.energy.dtype == getattr(torch, dtype)
            assert tensor.forces.device == torch.device("cpu")
            assert not tensor.energy.requires_grad
            np.testing.assert_allclose(tensor.energy.numpy(), result.energy, **tolerance)
            np.testing.assert_allclose(tensor.forces.numpy(), result.forces, **tolerance)
            np.testing.assert_allclose(tensor.virial.numpy(), result.virial, **tolerance)

            # Permuting frames and interleaving calls of other sizes cannot
            # couple a frame to its neighbours or retain previous-frame state.
            provider.evaluate(q[:1] + 0.03)
            reordered = provider.evaluate(q[::-1].copy())
            np.testing.assert_allclose(reordered.energy, result.energy[::-1], **tolerance)
            np.testing.assert_allclose(reordered.forces, result.forces[::-1], **tolerance)
    finally:
        hook.remove()


@pytest.mark.parametrize("periodic", [False, True])
def test_mace_force_and_virial_are_energy_derivatives(periodic):
    model = make_random_mace()
    q, cell = _geometry(2)
    symbols = ["D", "T", "H", "O"]
    pbc = [periodic] * 3
    provider = MACEBatchedPotential(symbols, model, cell=cell, pbc=pbc, virial=True)
    try:
        reference = provider.evaluate(q)
        epsilon = 2e-5
        plus, minus = q.copy(), q.copy()
        plus[:, 1, 2] += epsilon
        minus[:, 1, 2] -= epsilon
        numerical_force = -(provider.evaluate(plus).energy - provider.evaluate(minus).energy) / (2 * epsilon)
        np.testing.assert_allclose(reference.forces[:, 1, 2], numerical_force, rtol=2e-6, atol=2e-11)

        transform = np.eye(3)
        transform[0, 1] += epsilon
        provider.bind_system({"symbols": symbols, "cell": cell @ transform.T, "pbc": pbc})
        energy_plus = provider.evaluate(q @ transform.T).energy
        transform[0, 1] -= 2 * epsilon
        provider.bind_system({"symbols": symbols, "cell": cell @ transform.T, "pbc": pbc})
        energy_minus = provider.evaluate(q @ transform.T).energy
        numerical_virial = -(energy_plus - energy_minus) / (2 * epsilon)
        np.testing.assert_allclose(reference.virial[:, 0, 1], numerical_virial, rtol=2e-6, atol=2e-11)
    finally:
        provider.close()


def test_mace_no_cell_and_no_edges_remain_independent():
    model = make_random_mace()
    q = np.array([[[0., 0., 0.]], [[1., 2., 3.]], [[-2., 0., 1.]]])
    with MACEBatchedPotential(["D"], model) as provider:
        result = provider.evaluate(q)
        one = provider.evaluate(q[:1])
    np.testing.assert_allclose(result.energy, np.repeat(one.energy, 3), rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(result.forces, 0., atol=1e-14)


def test_mace_trusted_local_model_load_roundtrip(tmp_path):
    model = make_random_mace()
    path = tmp_path / "random-untrained.model"
    torch.save(model, path)
    q, cell = _geometry(2)
    symbols = ["H", "D", "T", "O"]
    with MACEBatchedPotential(symbols, model, cell=cell) as live:
        expected = live.evaluate(q)
    with MACEBatchedPotential(symbols, path, cell=cell) as loaded:
        result = loaded.evaluate(q)
    np.testing.assert_allclose(result.energy, expected.energy, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(result.forces, expected.forces, rtol=1e-10, atol=1e-14)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_mace_cuda_tensor_matches_cpu_without_changing_device():
    model = make_random_mace()
    q, cell = _geometry(5)
    symbols = ["D", "T", "H", "O"]
    with MACEBatchedPotential(symbols, deepcopy(model), cell=cell, virial=True) as cpu:
        expected = cpu.evaluate(q)
    with MACEBatchedPotential(symbols, model, device="cuda:0", cell=cell, virial=True) as gpu:
        tensor = torch.as_tensor(q, dtype=torch.float64, device="cuda:0")
        result = gpu.evaluate_tensor(tensor)
    for value in (result.energy, result.forces, result.virial):
        assert value.device == torch.device("cuda:0")
        assert value.dtype == torch.float64
    np.testing.assert_allclose(result.energy.cpu().numpy(), expected.energy, rtol=1e-10, atol=2e-14)
    np.testing.assert_allclose(result.forces.cpu().numpy(), expected.forces, rtol=1e-10, atol=2e-14)
    np.testing.assert_allclose(result.virial.cpu().numpy(), expected.virial, rtol=1e-10, atol=2e-14)
