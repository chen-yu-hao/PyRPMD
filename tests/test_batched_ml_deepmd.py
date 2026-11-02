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


def _native_reference(model, q, cell=None, symbols=None):
    species = ["O", "H", "H"] if symbols is None else symbols
    type_map = model.get_type_map()
    types = torch.tensor([type_map.index(s) for s in species], dtype=torch.long).expand(len(q), -1)
    coordinates = torch.tensor(q * BOHR_ANGSTROM, dtype=torch.float64)
    box = None if cell is None else torch.tensor(
        cell * BOHR_ANGSTROM, dtype=torch.float64,
    ).reshape(1, 9).expand(len(q), 9)
    result = model(coordinates, types, box)
    return {
        "energy": result["energy"].detach().numpy().reshape(-1) * EV_HARTREE,
        "forces": result["force"].detach().numpy() * EV_HARTREE * BOHR_ANGSTROM,
        "virial": result["virial"].detach().numpy().reshape(-1, 3, 3) * EV_HARTREE,
    }


@pytest.mark.parametrize("frames", [1, 3, 5])
@pytest.mark.parametrize("periodic", [False, True])
def test_sparse_batch_matches_native_forward_and_individual_frames(frames, periodic):
    model = _model()
    cell = np.array([[8.0, 0.3, 0.0], [0.0, 8.3, 0.2], [0.0, 0.0, 8.1]]) if periodic else None
    provider = DeepMDBatchedPotential(["O", "D", "T"], model, cell=cell,
                                     pbc=[periodic]*3, virial=True)
    q = _coordinates(frames)
    expected = _native_reference(model, q, cell=cell)
    actual = provider.evaluate(q)
    singles = [provider.evaluate(frame[None]) for frame in q]
    for name in ("energy", "forces", "virial"):
        np.testing.assert_allclose(getattr(actual, name), expected[name], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(
            getattr(actual, name), np.concatenate([getattr(one, name) for one in singles]),
            rtol=1e-10, atol=1e-12,
        )
    assert provider.metadata["native_batch"] is True
    assert provider.metadata["trajectory_batch_safe"] is True
    assert provider.metadata["batch_independent"] is True


def test_torchscript_export_load_and_native_model_identity(tmp_path):
    model = _model()
    path = tmp_path / "random-deepmd.pth"
    torch.jit.save(torch.jit.script(model), str(path))
    provider = DeepMDBatchedPotential(["O", "H", "H"], path, virial=True)
    q = _coordinates(2)
    expected = _native_reference(model, q)
    actual = provider.evaluate(q)
    for name in ("energy", "forces", "virial"):
        np.testing.assert_allclose(getattr(actual, name), expected[name], rtol=1e-10, atol=1e-12)


def test_one_atomic_model_call_and_tensor_evaluation_has_no_host_transfers(monkeypatch):
    model = _model()
    provider = DeepMDBatchedPotential(["O", "H", "H"], model, virial=True,
                                     cell=np.eye(3)*8, pbc=[True]*3)
    calls = []
    original = model.atomic_model.forward_common_atomic

    def count(extended, types, nlist, **kwargs):
        calls.append((len(extended), tuple(nlist.shape)))
        return original(extended, types, nlist, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("evaluate_tensor must not transfer tensors to NumPy/CPU")

    monkeypatch.setattr(model.atomic_model, "forward_common_atomic", count)
    q = torch.tensor(_coordinates(4), dtype=torch.float64, requires_grad=True)
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", forbidden)
        patch.setattr(torch.Tensor, "numpy", forbidden)
        with torch.no_grad():
            result = provider.evaluate_tensor(q)
    assert calls == [(4, (4, 3, 12))]
    assert q.grad is None
    for value in (result.energy, result.forces, result.virial):
        assert value.dtype == q.dtype and value.device == q.device
        assert not value.requires_grad
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("periodic", [False, True])
def test_forces_and_virial_match_independent_finite_differences(periodic):
    cell = np.eye(3) * 8 if periodic else None
    provider = DeepMDBatchedPotential(["O", "H", "H"], _model(), cell=cell,
                                     pbc=[periodic]*3, virial=True)
    q = _coordinates(2)
    result = provider.evaluate(q)
    step = 1e-5
    for atom, axis in [(0, 0), (1, 2), (2, 1)]:
        plus, minus = q.copy(), q.copy()
        plus[:, atom, axis] += step
        minus[:, atom, axis] -= step
        finite = -(provider.evaluate(plus).energy - provider.evaluate(minus).energy) / (2*step)
        np.testing.assert_allclose(result.forces[:, atom, axis], finite, rtol=3e-7, atol=3e-10)
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
        np.testing.assert_allclose(result.virial[:, first, second], finite, rtol=3e-7, atol=3e-10)
        if periodic:
            provider.set_cell(cell)


@pytest.mark.parametrize("factor", [1-1e-6, 1.0, 1+1e-6])
def test_cutoff_boundary_matches_native_model(factor):
    model = _model(cutoff=2.0)
    provider = DeepMDBatchedPotential(["O", "H", "H"], model, virial=True)
    q = np.array([[[0., 0., 0.], [2.0*factor/BOHR_ANGSTROM, 0., 0.],
                   [-9., 1., 2.]]])
    expected = _native_reference(model, q)
    actual = provider.evaluate(q)
    for name in ("energy", "forces", "virial"):
        np.testing.assert_allclose(getattr(actual, name), expected[name], rtol=1e-10, atol=1e-12)


def test_trained_neighbor_selection_overflow_matches_native_distance_policy():
    model = _model(sel=(1, 2), cutoff=5.0)
    symbols = ["O", "H", "H", "O", "H", "H", "H"]
    generator = np.random.default_rng(984)
    q = generator.uniform(-2, 2, size=(3, len(symbols), 3))
    provider = DeepMDBatchedPotential(symbols, model, virial=True)
    expected = _native_reference(model, q, symbols=symbols)
    actual = provider.evaluate(q)
    for name in ("energy", "forces", "virial"):
        np.testing.assert_allclose(getattr(actual, name), expected[name], rtol=1e-10, atol=1e-12)


def test_ambiguous_tie_at_trained_selection_boundary_is_explicit():
    provider = DeepMDBatchedPotential(["O", "H", "H", "H"], _model(sel=(0, 1)))
    q = np.array([[[0., 0., 0.], [2., 0., 0.], [-2., 0., 0.], [0., 2., 0.]]])
    with pytest.raises(PotentialError, match="equidistant"):
        provider.evaluate(q)


@pytest.mark.parametrize("kwargs, message", [
    ({"dtype": "float32"}, "float64"),
    ({"charge": 1}, "neutral"),
    ({"multiplicity": 2}, "multiplicity"),
    ({"cell": np.eye(3)*8, "pbc": [True, False, False]}, "partial PBC"),
])
def test_unsupported_modes_fail_explicitly(kwargs, message):
    with pytest.raises((ValueError, PotentialError), match=message):
        DeepMDBatchedPotential(["O", "H", "H"], _model(), **kwargs)


def test_model_precision_metadata_cannot_be_overridden_by_casting_parameters():
    model = _model(precision="float32").double()
    with pytest.raises(ValueError, match="precision"):
        DeepMDBatchedPotential(["O", "H", "H"], model)


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


def test_rebind_and_close_release_model():
    provider = DeepMDBatchedPotential(["O", "D", "T"], _model())
    q = _coordinates(2)
    provider.evaluate(q)
    provider.bind_system({"symbols": ["O", "H", "H"], "cell": np.eye(3)*8, "pbc": [True]*3})
    expected = _native_reference(provider.model, q, cell=np.eye(3)*8)
    np.testing.assert_allclose(provider.evaluate(q).energy, expected["energy"], rtol=1e-10, atol=1e-12)
    provider.close()
    provider.close()
    assert provider.model is None
    assert provider._neighbors is None
    with pytest.raises(PotentialError, match="closed"):
        provider.evaluate(q)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_batch_retains_device():
    provider = DeepMDBatchedPotential(["O", "H", "H"], _model(), device="cuda",
                                     cell=np.eye(3)*8, pbc=[True]*3, virial=True)
    q = torch.tensor(_coordinates(2), dtype=torch.float64, device="cuda")
    result = provider.evaluate_tensor(q)
    reference = _native_reference(_model(), q.cpu().numpy(), cell=np.eye(3)*8)
    for name in ("energy", "forces", "virial"):
        value = getattr(result, name)
        assert value.device == q.device
        np.testing.assert_allclose(value.cpu().numpy(), reference[name], rtol=1e-10, atol=1e-12)
