"""Native random DeePMD models; no training data or pretrained downloads."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
models = pytest.importorskip("deepmd.pt.model.model")
pytest.importorskip("vesin_torch")

from pyrpmd.potentials.batched_ml import DeepMDBatchedPotential
from pyrpmd.potentials.base import PotentialError
from pyrpmd.potentials.python import BOHR_ANGSTROM, EV_HARTREE


def _model(*, precision="float64", sel=(4, 8), cutoff=3.0,
           fitting_type="ener", frame_parameters=0):
    return models.get_model({
        "type_map": ["O", "H"],
        "descriptor": {
            "type": "se_e2_a", "rcut": cutoff, "rcut_smth": cutoff - 0.5,
            "sel": list(sel), "neuron": [4, 8], "axis_neuron": 2,
            "precision": precision, "seed": 782,
        },
        "fitting_net": {
            "type": fitting_type, "neuron": [8, 8], "precision": precision,
            "seed": 783, "numb_fparam": frame_parameters,
        },
    }).eval()


def _coordinates(frames, atoms=3):
    generator = np.random.default_rng(4812)
    q = generator.uniform(-1, 1, size=(frames, atoms, 3))
    q[:, 0] += [0.2, 0.4, 0.1]
    q[:, 1] += [2.3, -0.1, 0.4]
    q[:, 2] += [7.5, 0.4, -0.1]
    return q


def _native_reference(model, q, cell=None, symbols=None, dtype=torch.float64):
    species = ["O", "H", "H"] if symbols is None else symbols
    type_map = model.get_type_map()
    types = torch.tensor([type_map.index(s) for s in species], dtype=torch.long).expand(len(q), -1)
    coordinates = torch.tensor(q, dtype=dtype) * BOHR_ANGSTROM
    box = None if cell is None else torch.tensor(
        cell * BOHR_ANGSTROM, dtype=dtype,
    ).reshape(1, 9).expand(len(q), 9)
    result = model(coordinates, types, box)
    return {
        "energy": result["energy"].detach().numpy().reshape(-1) * EV_HARTREE,
        "forces": result["force"].detach().numpy() * EV_HARTREE * BOHR_ANGSTROM,
        "virial": result["virial"].detach().numpy().reshape(-1, 3, 3) * EV_HARTREE,
    }


@pytest.mark.parametrize("frames", [1, 3, 5])
@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_sparse_batch_matches_native_forward_and_individual_frames(frames, periodic, precision):
    model = _model(precision=precision)
    dtype = getattr(torch, precision)
    tolerance = {"rtol": 1e-10, "atol": 1e-12} if precision == "float64" else {
        "rtol": 3e-5, "atol": 3e-8,
    }
    cell = np.array([[8.0, 0.3, 0.0], [0.0, 8.3, 0.2], [0.0, 0.0, 8.1]]) if periodic else None
    provider = DeepMDBatchedPotential(["O", "D", "T"], model, cell=cell,
                                     pbc=[periodic]*3, virial=True, dtype=precision)
    q = _coordinates(frames)
    expected = _native_reference(model, q, cell=cell, dtype=dtype)
    actual = provider.evaluate(q)
    singles = [provider.evaluate(frame[None]) for frame in q]
    for name in ("energy", "forces", "virial"):
        np.testing.assert_allclose(getattr(actual, name), expected[name], **tolerance)
        np.testing.assert_allclose(
            getattr(actual, name), np.concatenate([getattr(one, name) for one in singles]),
            **tolerance,
        )
    reordered = provider.evaluate(q[::-1].copy())
    np.testing.assert_allclose(reordered.forces, actual.forces[::-1], **tolerance)
    assert provider.metadata["native_batch"] is True
    assert provider.metadata["trajectory_batch_safe"] is True
    assert provider.metadata["batch_independent"] is True
    assert provider.metadata["dtype"] == precision
    assert provider.metadata["geometry_dtype"] == "float64"
    assert provider.metadata["energy_dtype"] == "float64"
    assert provider.metadata["virial_dtype"] == "float64"


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_torchscript_export_load_and_native_model_identity(tmp_path, precision):
    model = _model(precision=precision)
    path = tmp_path / "random-deepmd.pth"
    torch.jit.save(torch.jit.script(model), str(path))
    provider = DeepMDBatchedPotential(["O", "H", "H"], path, virial=True, dtype=precision)
    q = _coordinates(2)
    expected = _native_reference(model, q, dtype=getattr(torch, precision))
    actual = provider.evaluate(q)
    tolerance = {"rtol": 1e-10, "atol": 1e-12} if precision == "float64" else {
        "rtol": 3e-5, "atol": 3e-8,
    }
    for name in ("energy", "forces", "virial"):
        np.testing.assert_allclose(getattr(actual, name), expected[name], **tolerance)


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_one_atomic_model_call_and_tensor_evaluation_has_no_host_transfers(monkeypatch, precision):
    model = _model(precision=precision)
    provider = DeepMDBatchedPotential(["O", "H", "H"], model, virial=True, dtype=precision,
                                     cell=np.eye(3)*8, pbc=[True]*3)
    calls = []
    original = model.atomic_model.forward_common_atomic

    def count(extended, types, nlist, **kwargs):
        calls.append((len(extended), tuple(nlist.shape)))
        return original(extended, types, nlist, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("evaluate_tensor must not transfer tensors to NumPy/CPU")

    monkeypatch.setattr(model.atomic_model, "forward_common_atomic", count)
    q = torch.tensor(_coordinates(4), dtype=getattr(torch, precision), requires_grad=True)
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", forbidden)
        patch.setattr(torch.Tensor, "numpy", forbidden)
        with torch.no_grad():
            result = provider.evaluate_tensor(q)
    assert calls == [(4, (4, 3, 12))]
    assert q.grad is None
    for value, dtype in ((result.energy, torch.float64), (result.forces, q.dtype),
                         (result.virial, torch.float64)):
        assert value.dtype == dtype and value.device == q.device
        assert not value.requires_grad
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_forces_and_virial_match_independent_finite_differences(periodic, precision):
    cell = np.eye(3) * 8 if periodic else None
    provider = DeepMDBatchedPotential(["O", "H", "H"], _model(precision=precision), cell=cell,
                                     pbc=[periodic]*3, virial=True, dtype=precision)
    q = _coordinates(2)
    result = provider.evaluate(q)
    # Float32 finite differences need a larger step than frame/batch parity:
    # quantization in the energy network dominates at the float64 step size.
    step = 1e-5 if precision == "float64" else 3e-3
    tolerance = {"rtol": 3e-7, "atol": 3e-10} if precision == "float64" else {
        "rtol": 3e-3, "atol": 2e-6,
    }
    for atom, axis in [(0, 0), (1, 2), (2, 1)]:
        plus, minus = q.copy(), q.copy()
        plus[:, atom, axis] += step
        minus[:, atom, axis] -= step
        finite = -(provider.evaluate(plus).energy - provider.evaluate(minus).energy) / (2*step)
        np.testing.assert_allclose(result.forces[:, atom, axis], finite, **tolerance)
    for first, second in [(0, 0), (0, 1), (2, 1)]:
        transform = np.eye(3)
        transform[first, second] += step
        if periodic:
            provider.set_cell(cell @ transform.T)
        plus = provider.evaluate(q @ transform.T).energy
        transform[first, second] -= 2*step
        if periodic:
            provider.set_cell(cell @ transform.T)
        minus = provider.evaluate(q @ transform.T).energy
        finite = -(plus-minus)/(2*step)
        np.testing.assert_allclose(result.virial[:, first, second], finite, **tolerance)
        if periodic:
            provider.set_cell(cell)


@pytest.mark.parametrize("factor", [1-1e-6, 1.0, 1+1e-6])
@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_cutoff_boundary_matches_native_model(factor, precision):
    model = _model(cutoff=2.0, precision=precision)
    provider = DeepMDBatchedPotential(["O", "H", "H"], model, virial=True, dtype=precision)
    q = np.array([[[0., 0., 0.], [2.0*factor/BOHR_ANGSTROM, 0., 0.],
                   [-9., 1., 2.]]])
    expected = _native_reference(model, q, dtype=getattr(torch, precision))
    actual = provider.evaluate(q)
    tolerance = {"rtol": 1e-10, "atol": 1e-12} if precision == "float64" else {
        "rtol": 3e-5, "atol": 3e-8,
    }
    for name in ("energy", "forces", "virial"):
        np.testing.assert_allclose(getattr(actual, name), expected[name], **tolerance)


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_trained_neighbor_selection_overflow_matches_native_distance_policy(precision):
    model = _model(sel=(1, 2), cutoff=5.0, precision=precision)
    symbols = ["O", "H", "H", "O", "H", "H", "H"]
    generator = np.random.default_rng(984)
    q = generator.uniform(-2, 2, size=(3, len(symbols), 3))
    provider = DeepMDBatchedPotential(symbols, model, virial=True, dtype=precision)
    expected = _native_reference(model, q, symbols=symbols, dtype=getattr(torch, precision))
    actual = provider.evaluate(q)
    tolerance = {"rtol": 1e-10, "atol": 1e-12} if precision == "float64" else {
        "rtol": 3e-5, "atol": 3e-8,
    }
    for name in ("energy", "forces", "virial"):
        np.testing.assert_allclose(getattr(actual, name), expected[name], **tolerance)


def test_ambiguous_tie_at_trained_selection_boundary_is_explicit():
    provider = DeepMDBatchedPotential(["O", "H", "H", "H"], _model(sel=(0, 1)))
    q = np.array([[[0., 0., 0.], [2., 0., 0.], [-2., 0., 0.], [0., 2., 0.]]])
    with pytest.raises(PotentialError, match="equidistant"):
        provider.evaluate(q)


@pytest.mark.parametrize("kwargs, message", [
    ({"dtype": "float32"}, "precision"),
    ({"charge": 1}, "neutral"),
    ({"multiplicity": 2}, "multiplicity"),
    ({"cell": np.eye(3)*8, "pbc": [True, False, False]}, "partial PBC"),
])
def test_unsupported_modes_fail_explicitly(kwargs, message):
    with pytest.raises((ValueError, PotentialError), match=message):
        DeepMDBatchedPotential(["O", "H", "H"], _model(), **kwargs)


@pytest.mark.parametrize("original, requested", [("float32", "float64"), ("float64", "float32")])
def test_model_precision_metadata_cannot_be_overridden_by_casting_parameters(original, requested):
    model = _model(precision=original).to(dtype=getattr(torch, requested))
    with pytest.raises(ValueError, match="precision"):
        DeepMDBatchedPotential(["O", "H", "H"], model, dtype=requested)


@pytest.mark.parametrize("attribute", ["global_pt_float_precision", "global_pt_ener_float_precision"])
def test_unsupported_global_precision_is_not_rewritten(attribute):
    model = _model(precision="float32")
    setattr(model, attribute, torch.float32)
    with pytest.raises(ValueError, match="global geometry and energy precision"):
        DeepMDBatchedPotential(["O", "H", "H"], model, dtype="float32")
    assert getattr(model, attribute) == torch.float32


def test_float32_weights_and_native_double_biases_are_preserved():
    model = _model(precision="float32")
    model.atomic_model.out_bias.copy_(torch.tensor([[[0.123456789123], [-0.234567891234]]],
                                                  dtype=torch.float64))
    state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    provider = DeepMDBatchedPotential(["O", "D", "T"], model, dtype="float32", virial=True)
    q = torch.tensor(_coordinates(3), dtype=torch.float32)
    result = provider.evaluate_tensor(q)
    expected = _native_reference(model, q.numpy(), dtype=torch.float32)
    for name in ("energy", "forces", "virial"):
        np.testing.assert_allclose(getattr(result, name).numpy(), expected[name], rtol=3e-5, atol=3e-8)
    for name, tensor in model.state_dict().items():
        assert tensor.dtype == state[name].dtype
        assert torch.equal(tensor, state[name])
    assert {parameter.dtype for parameter in model.parameters()} == {torch.float32}
    assert model.atomic_model.out_bias.dtype == torch.float64
    with pytest.raises(ValueError, match="float64"):
        provider.as_torch_pes()


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_large_periodic_images_match_native_wrapping_without_mutating_inputs(precision):
    dtype = getattr(torch, precision)
    cell = torch.tensor([[8.0, 0.3, 0.0], [0.0, 8.3, 0.2], [0.0, 0.0, 8.1]], dtype=dtype)
    local = torch.tensor([[[0.2, 0.4, 0.1], [2.3, -0.1, 0.4], [0.3, 2.1, -0.2]]],
                         dtype=dtype).repeat(3, 1, 1)
    images = torch.tensor([[[4000., -3000., 2000.], [-2000., 1000., 5000.],
                             [3000., 3000., -4000.]]], dtype=dtype)
    q = local + images @ cell
    original = q.clone()
    model = _model(precision=precision)
    # Identity units isolate wrapping from float32 rounding during conversion.
    provider = DeepMDBatchedPotential(
        ["O", "D", "T"], model, dtype=precision, cell=cell.numpy(), pbc=[True]*3,
        virial=True, units={"positions": "bohr", "energy": "hartree"},
    )
    actual = provider.evaluate_tensor(q)
    native = model(q, torch.tensor([[0, 1, 1]]).expand(len(q), 3),
                   cell.reshape(1, 9).expand(len(q), 9))
    tolerance = {"rtol": 1e-10, "atol": 1e-12} if precision == "float64" else {
        "rtol": 3e-5, "atol": 3e-8,
    }
    for name, key in (("energy", "energy"), ("forces", "force"), ("virial", "virial")):
        value = getattr(actual, name)
        torch.testing.assert_close(value, native[key].reshape(value.shape), **tolerance)
    assert torch.equal(q, original)


@pytest.mark.parametrize("kwargs, message", [
    ({"fitting_type": "direct_force_ener"}, "conservative"),
    ({"frame_parameters": 1}, "frame or atomic"),
])
def test_model_requirements_cannot_be_silently_ignored(kwargs, message):
    with pytest.raises(ValueError, match=message):
        DeepMDBatchedPotential(["O", "H", "H"], _model(**kwargs))


def test_exported_data_modifier_is_rejected_instead_of_dropping_its_energy(tmp_path):
    path = tmp_path / "with-data-modifier.pth"
    torch.jit.save(torch.jit.script(_model()), str(path),
                   _extra_files={"data_modifier.pth": b"modifier-present"})
    with pytest.raises(ValueError, match="data modifiers"):
        DeepMDBatchedPotential(["O", "H", "H"], path)


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_rebind_and_close_release_model(precision):
    provider = DeepMDBatchedPotential(["O", "D", "T"], _model(precision=precision),
                                     dtype=precision, virial=True)
    q = _coordinates(2)
    provider.evaluate(q)
    previous_cell = provider._cell_tensor()
    provider.bind_system({"symbols": ["O", "H", "H"], "cell": np.eye(3)*8, "pbc": [True]*3})
    expected = _native_reference(provider.model, q, cell=np.eye(3)*8, dtype=getattr(torch, precision))
    actual = provider.evaluate(q)
    tolerance = {"rtol": 1e-10, "atol": 1e-12} if precision == "float64" else {
        "rtol": 3e-5, "atol": 3e-8,
    }
    for name in ("energy", "forces", "virial"):
        np.testing.assert_allclose(getattr(actual, name), expected[name], **tolerance)
    assert provider._cell_tensor() is not previous_cell
    assert provider._cell_tensor().dtype == torch.float64
    provider.close()
    provider.close()
    assert provider.model is None
    assert provider._neighbors is None
    with pytest.raises(PotentialError, match="closed"):
        provider.evaluate(q)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_cuda_batch_retains_device(precision):
    provider = DeepMDBatchedPotential(["O", "H", "H"], _model(precision=precision), device="cuda",
                                     cell=np.eye(3)*8, pbc=[True]*3, virial=True, dtype=precision)
    q = torch.tensor(_coordinates(2), dtype=getattr(torch, precision), device="cuda")
    result = provider.evaluate_tensor(q)
    reference = _native_reference(_model(precision=precision), q.cpu().numpy(),
                                  cell=np.eye(3)*8, dtype=getattr(torch, precision))
    tolerance = {"rtol": 1e-10, "atol": 1e-12} if precision == "float64" else {
        "rtol": 3e-5, "atol": 3e-8,
    }
    for name in ("energy", "forces", "virial"):
        value = getattr(result, name)
        assert value.device == q.device
        assert value.dtype == (q.dtype if name == "forces" else torch.float64)
        np.testing.assert_allclose(value.cpu().numpy(), reference[name], **tolerance)
