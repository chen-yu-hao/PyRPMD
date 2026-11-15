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


def _changing_topology_geometry():
    cube = np.array([[x, y, z] for x in (0., 1.) for y in (0., 1.) for z in (0., 1.)])
    cell_angstrom = np.array([[18., 0.6, 0.2], [0., 17., 0.3], [0., 0., 19.]])
    dense = 0.8 * cube + 0.3
    isolated = 6.0 * cube + 0.3
    components = dense.copy()
    components[4:] = [[7., 1., 1.], [1., 7., 1.], [7., 7., 1.], [7., 7., 7.]]
    images = dense.copy()
    images[::2] += cell_angstrom[0]
    q = np.stack((dense, isolated, components, images)) / BOHR_ANGSTROM
    cell = cell_angstrom / BOHR_ANGSTROM
    return q, cell


@pytest.mark.parametrize("pbc", [[False] * 3, [True] * 3, [True, False, True]])
def test_mace_reused_neighbor_buffers_preserve_changing_topology(mace_model, pbc):
    # Successive frames shrink and grow the native neighbor buffers: a compact
    # cluster, isolated atoms, two components, and atoms in other periodic images.
    dtype, model = mace_model
    q, cell = _changing_topology_geometry()
    symbols = ["D", "T", "H", "O"] * 2
    tolerance = {"rtol": 1e-10, "atol": 2e-14} if dtype == "float64" else {
        "rtol": 3e-5, "atol": 3e-9,
    }

    with MACEBatchedPotential(symbols, deepcopy(model), dtype=dtype, cell=cell,
                              pbc=pbc, virial=True) as provider:
        # A strided caller tensor must retain its data, gradient flag and layout.
        storage = torch.zeros((4, 8, 6), dtype=provider.dtype)
        storage[..., ::2] = torch.as_tensor(q, dtype=provider.dtype)
        tensor = storage[..., ::2].requires_grad_(True)
        before = tensor.detach().clone()
        graph = provider._graph(tensor)
        assert graph["edge_index"].is_contiguous()
        edge_frame = graph["batch"][graph["edge_index"][0]]
        torch.testing.assert_close(edge_frame, graph["batch"][graph["edge_index"][1]])
        counts = torch.bincount(edge_frame, minlength=4)
        assert counts[0] == 56 and counts[1] == 0 and counts[2] == 12
        saved = {key: graph[key].clone() for key in ("edge_index", "unit_shifts", "shifts")}
        # Reusing the calculator after graph assembly cannot alter that graph.
        for frame in (0, 1, 3, 2):
            provider._neighbors.compute(
                torch.as_tensor(q[frame], dtype=provider.dtype) / provider.length_factor,
                provider._cell_tensor(), provider._pbc_tensor(), quantities="PS", copy=False,
            )
        for key, value in saved.items():
            assert torch.equal(graph[key], value)

        result = provider.evaluate_tensor(tensor)
        with _ase_reference(model, dtype, symbols, cell, pbc) as scalar:
            expected = scalar.evaluate(q)
        np.testing.assert_allclose(result.energy.numpy(), expected.energy, **tolerance)
        np.testing.assert_allclose(result.forces.numpy(), expected.forces, **tolerance)
        # Float32 strain gradients cancel larger terms for atoms in other
        # periodic images. The published adapter has the same ~2.3e-8 AU
        # near-zero virial differences from ASE for this geometry.
        virial_tolerance = {**tolerance, "atol": 3e-8} if dtype == "float32" else tolerance
        if any(pbc):
            np.testing.assert_allclose(result.virial.numpy(), expected.virial, **virial_tolerance)
        # Include singleton and reordered calls after the much larger graph.
        for selection in ([3], [2, 0], [3, 1, 0, 2]):
            selected = provider.evaluate_tensor(tensor[selection])
            for name in ("energy", "forces", "virial"):
                torch.testing.assert_close(getattr(selected, name), getattr(result, name)[selection],
                                           **(virial_tolerance if name == "virial" else tolerance))
        assert torch.equal(tensor.detach(), before)
        assert not tensor.is_contiguous() and tensor.requires_grad and tensor.grad is None


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
@pytest.mark.parametrize("custom_stream", [False, True])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_mace_cuda_tensor_matches_cpu_without_changing_device(custom_stream, dtype, monkeypatch):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    model = make_random_mace(dtype=dtype)
    q, cell = _changing_topology_geometry()
    symbols = ["D", "T", "H", "O"] * 2
    pbc = [True, False, True]
    tolerance = {"rtol": 1e-10, "atol": 2e-14} if dtype == "float64" else {
        "rtol": 3e-5, "atol": 3e-9,
    }
    virial_tolerance = {**tolerance, "atol": 3e-8} if dtype == "float32" else tolerance
    with MACEBatchedPotential(symbols, deepcopy(model), dtype=dtype, cell=cell,
                              pbc=pbc, virial=True) as cpu:
        expected = cpu.evaluate(q)
    with MACEBatchedPotential(symbols, model, device="cuda:0", dtype=dtype, cell=cell,
                              pbc=pbc, virial=True) as gpu:
        outer_stream = torch.cuda.current_stream("cuda:0")
        stream = torch.cuda.Stream(device="cuda:0") if custom_stream else outer_stream
        stream.wait_stream(outer_stream)
        inference_streams = []
        hook = gpu.model.register_forward_pre_hook(
            lambda _, args: inference_streams.append(torch.cuda.current_stream("cuda:0"))
        )
        try:
            with torch.cuda.stream(stream):
                tensor = torch.as_tensor(q, dtype=getattr(torch, dtype), device="cuda:0")
                result = gpu.evaluate_tensor(tensor)
                # Reuse the native output buffers and cached templates across
                # different batch sizes, including an empty-neighbor frame.
                for selection in ([3], [1, 0], [3, 1, 0, 2]):
                    selected = gpu.evaluate_tensor(tensor[selection])
                    for name in ("energy", "forces", "virial"):
                        torch.testing.assert_close(
                            getattr(selected, name), getattr(result, name)[selection],
                            **(virial_tolerance if name == "virial" else tolerance),
                        )
                assert torch.cuda.current_stream("cuda:0") == stream
            assert torch.cuda.current_stream("cuda:0") == outer_stream
            assert inference_streams == [stream] * 4
            stream.synchronize()
        finally:
            hook.remove()
    for value in (result.energy, result.forces, result.virial):
        assert value.device == torch.device("cuda:0")
        assert value.dtype == getattr(torch, dtype)
    np.testing.assert_allclose(result.energy.cpu().numpy(), expected.energy, **tolerance)
    np.testing.assert_allclose(result.forces.cpu().numpy(), expected.forces, **tolerance)
    np.testing.assert_allclose(result.virial.cpu().numpy(), expected.virial, **virial_tolerance)


@pytest.mark.parametrize("bad", [0, -1, True, False, 1.5, 2.0, "2"])
def test_mace_rejects_invalid_max_batch_frames(mace_model, bad):
    dtype, model = mace_model
    with pytest.raises(ValueError, match="max_batch_frames"):
        MACEBatchedPotential(["D", "H"], deepcopy(model), dtype=dtype, max_batch_frames=bad)


@pytest.mark.parametrize("batch,expected_chunks", [(1, [1]), (2, [2]), (5, [2, 2, 1]), (7, [2, 2, 2, 1])])
@pytest.mark.parametrize("virial", [False, True])
@pytest.mark.parametrize("pbc", [[False, False, False], [True, True, True], [True, False, True]])
def test_mace_chunked_batch_matches_unchunked_within_1e13_au(batch, expected_chunks, virial, pbc):
    model = make_random_mace(dtype="float64")
    q, cell = _geometry(batch)
    symbols = ["D", "T", "H", "O"]
    chunk_calls = []
    chunked_model = deepcopy(model)
    hook = chunked_model.register_forward_pre_hook(
        lambda _, args: chunk_calls.append(len(args[0]["ptr"]) - 1)
    )
    try:
        with MACEBatchedPotential(symbols, deepcopy(model), dtype="float64",
                                  cell=cell, pbc=pbc, virial=virial) as unchunked, \
             MACEBatchedPotential(symbols, chunked_model, dtype="float64",
                                  cell=cell, pbc=pbc, virial=virial,
                                  max_batch_frames=2) as chunked:
            assert unchunked.metadata["max_batch_frames"] is None
            assert unchunked.metadata["model_calls_per_batch"] == 1
            assert chunked.metadata["max_batch_frames"] == 2
            assert "model_calls_per_batch" not in chunked.metadata

            expected = unchunked.evaluate(q)
            actual = chunked.evaluate(q)
            assert chunk_calls == expected_chunks
            assert np.max(np.abs(actual.energy - expected.energy)) < 1e-13
            assert np.max(np.abs(actual.forces - expected.forces)) < 1e-13
            if virial:
                assert actual.virial is not None
                assert np.max(np.abs(actual.virial - expected.virial)) < 1e-13
            else:
                assert actual.virial is None

            # Second call hits cached templates for both full and remainder chunks.
            chunk_calls.clear()
            tensor_in = torch.as_tensor(q, dtype=torch.float64)
            t_expected = unchunked.evaluate_tensor(tensor_in)
            t_actual = chunked.evaluate_tensor(tensor_in)
            assert chunk_calls == expected_chunks
            assert torch.max(torch.abs(t_actual.energy - t_expected.energy)).item() < 1e-13
            assert torch.max(torch.abs(t_actual.forces - t_expected.forces)).item() < 1e-13
            if virial:
                assert torch.max(torch.abs(t_actual.virial - t_expected.virial)).item() < 1e-13
    finally:
        hook.remove()


def test_mace_chunking_handles_empty_edge_chunks_and_strided_inputs():
    model = make_random_mace(dtype="float64")
    q, cell = _changing_topology_geometry()
    symbols = ["D", "T", "H", "O"] * 2
    pbc = [True, True, True]
    storage = torch.zeros((4, 8, 6), dtype=torch.float64)
    storage[..., ::2] = torch.as_tensor(q, dtype=torch.float64)
    strided = storage[..., ::2].requires_grad_(True)
    before = strided.detach().clone()

    with MACEBatchedPotential(symbols, deepcopy(model), cell=cell, pbc=pbc, virial=True) as full, \
         MACEBatchedPotential(symbols, deepcopy(model), cell=cell, pbc=pbc, virial=True,
                              max_batch_frames=1) as per_frame:
        expected = full.evaluate_tensor(strided)
        actual = per_frame.evaluate_tensor(strided)
    assert torch.max(torch.abs(actual.energy - expected.energy)).item() < 1e-13
    assert torch.max(torch.abs(actual.forces - expected.forces)).item() < 1e-13
    assert torch.max(torch.abs(actual.virial - expected.virial)).item() < 1e-13
    assert torch.equal(strided.detach(), before)
    assert not strided.is_contiguous() and strided.requires_grad and strided.grad is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("custom_stream", [False, True])
def test_mace_cuda_chunking_matches_unchunked_and_preserves_stream(custom_stream):
    model = make_random_mace(dtype="float64")
    q, cell = _geometry(5)
    symbols = ["D", "T", "H", "O"]
    pbc = [True, True, True]
    with MACEBatchedPotential(symbols, deepcopy(model), device="cuda:0", dtype="float64",
                              cell=cell, pbc=pbc, virial=True) as full, \
         MACEBatchedPotential(symbols, model, device="cuda:0", dtype="float64",
                              cell=cell, pbc=pbc, virial=True, max_batch_frames=2) as chunked:
        outer_stream = torch.cuda.current_stream("cuda:0")
        stream = torch.cuda.Stream(device="cuda:0") if custom_stream else outer_stream
        stream.wait_stream(outer_stream)
        seen_streams, seen_sizes = [], []

        def record(_, args):
            seen_streams.append(torch.cuda.current_stream("cuda:0"))
            seen_sizes.append(len(args[0]["ptr"]) - 1)

        hook = chunked.model.register_forward_pre_hook(record)
        try:
            with torch.cuda.stream(stream):
                tensor = torch.as_tensor(q, dtype=torch.float64, device="cuda:0")
                expected = full.evaluate_tensor(tensor)
                actual = chunked.evaluate_tensor(tensor)
                assert torch.cuda.current_stream("cuda:0") == stream
            stream.synchronize()
        finally:
            hook.remove()
    assert seen_sizes == [2, 2, 1]
    assert seen_streams == [stream] * 3
    assert torch.max(torch.abs(actual.energy - expected.energy)).item() < 1e-13
    assert torch.max(torch.abs(actual.forces - expected.forces)).item() < 1e-13
    assert torch.max(torch.abs(actual.virial - expected.virial)).item() < 1e-13
