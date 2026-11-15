"""Public registration, strict tensor boundaries and scientific identity."""

import copy
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest

from pyrpmd.batched import validate_batch_potential
from pyrpmd.input import InputError, JobSpec
from pyrpmd.io import fingerprint
from pyrpmd.potentials import PotentialError, make_potential, validate_periodic_spec
from pyrpmd.potentials._batched_ml_base import _TensorBatchPotential
from pyrpmd.potentials.batched_ml import BATCHED_ML_PROVIDERS
from pyrpmd.propagation import scientific_config
from pyrpmd.system import System
from test_input_cli import small_collision


class AnalyticBatch(_TensorBatchPotential):
    def __init__(self, **kwargs):
        self.calls = 0
        self._setup(["D", "T"], "test-analytic", **kwargs)

    def _evaluate_tensor(self, q):
        from pyrpmd.torch_backend import TensorPESResult
        self.calls += 1
        virial = -self.torch.einsum("bna,bnc->bac", q, q) if self.virial else None
        return TensorPESResult((q*q).sum(dim=(1, 2))/2, -q, virial)


def test_import_is_lazy_in_an_environment_without_torch():
    code = """
import sys
class DenyTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith(('torch.', 'mace.', 'nequip.', 'metatomic.', 'fairchem.')):
            raise AssertionError('eager optional import: ' + fullname)
sys.meta_path.insert(0, DenyTorch())
import pyrpmd.potentials
import pyrpmd.potentials.batched_ml
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("kind", sorted(BATCHED_ML_PROVIDERS))
def test_provider_schema_registration_is_lazy_and_preserves_physical_options(kind, tmp_path):
    config = small_collision()
    config["potential"] = {"provider": kind, "model": "models/local.pt", "dtype": "float32",
                           "virial": True, "params": {"head": "selected"}}
    resolved, system = JobSpec.from_mapping(config, base_directory=tmp_path).resolve()
    spec = resolved["potential"]
    assert spec["model"] == str(tmp_path / "models/local.pt")
    with patch("pyrpmd.potentials.adapters.make_named_potential") as factory:
        make_potential(spec, system)
    assert factory.call_args.args == (kind,)
    assert factory.call_args.kwargs["params"] == {
        "dtype": "float32", "virial": True, "head": "selected",
    }
    assert spec["params"] == {"head": "selected"}


@pytest.mark.parametrize("kind", sorted(BATCHED_ML_PROVIDERS))
def test_new_provider_named_registry_dispatches_without_ase(kind):
    from pyrpmd.potentials.adapters import make_named_potential
    with patch("pyrpmd.potentials.batched_ml.make_batched_ml_potential") as factory:
        make_named_potential(kind, symbols=["D", "T"], model="local.pt", device="cuda:1",
                             charge=0, multiplicity=1, params={"dtype": "float64"})
    assert factory.call_args.args == (kind,)
    assert factory.call_args.kwargs["device"] == "cuda:1"
    assert factory.call_args.kwargs["symbols"] == ["D", "T"]


@pytest.mark.parametrize("option", [
    {"dtype": "float16"}, {"dtype": 64}, {"virial": "auto"},
    {"dtype": "float64", "params": {"dtype": "float32"}},
    {"virial": True, "params": {"virial": False}},
    {"max_batch_frames": 0}, {"max_batch_frames": -1},
    {"max_batch_frames": True}, {"max_batch_frames": False},
    {"max_batch_frames": 1.5}, {"max_batch_frames": "2"},
    {"max_batch_frames": "AUTO"}, {"max_batch_frames": ""},
    {"params": {"max_batch_frames": 0}}, {"params": {"max_batch_frames": -2}},
    {"params": {"max_batch_frames": True}}, {"params": {"max_batch_frames": 2.0}},
    {"params": {"max_batch_frames": "none"}},
    {"max_batch_frames": 2, "params": {"max_batch_frames": 4}},
    {"max_batch_frames": "auto", "params": {"max_batch_frames": "auto"}},
])
def test_schema_rejects_implicit_precision_and_ambiguous_options(option):
    config = small_collision()
    config["potential"] = {"provider": "mace-batched", "model": "missing.model", **option}
    with pytest.raises(InputError):
        JobSpec.from_mapping(config).resolve()


def test_non_batched_provider_rejects_top_level_max_batch_frames():
    with pytest.raises(PotentialError, match="batched ML"):
        validate_periodic_spec({"provider": "builtin", "name": "harmonic", "max_batch_frames": 2})


@pytest.mark.parametrize("kind", sorted(BATCHED_ML_PROVIDERS))
@pytest.mark.parametrize("via_params", [True, False])
@pytest.mark.parametrize("setting", [3, "auto"])
def test_schema_accepts_and_forwards_max_batch_frames(kind, via_params, setting, tmp_path):
    config = small_collision()
    if via_params:
        config["potential"] = {
            "provider": kind, "model": "models/local.pt",
            "params": {"max_batch_frames": setting},
        }
    else:
        config["potential"] = {
            "provider": kind, "model": "models/local.pt", "max_batch_frames": setting,
        }
    resolved, system = JobSpec.from_mapping(config, base_directory=tmp_path).resolve()
    spec = resolved["potential"]
    with patch("pyrpmd.potentials.adapters.make_named_potential") as factory:
        make_potential(spec, system)
    assert factory.call_args.args == (kind,)
    assert factory.call_args.kwargs["params"]["max_batch_frames"] == setting
    baseline = copy.deepcopy(resolved)
    baseline["potential"].pop("max_batch_frames", None)
    baseline["potential"].pop("params", None)
    assert scientific_config(resolved) == scientific_config(baseline)


@pytest.mark.parametrize("kind", sorted(BATCHED_ML_PROVIDERS))
def test_gpu_worker_allowlist_includes_new_providers(kind):
    # Verify GPU routing without creating a CUDA context on CPU CI.
    from types import SimpleNamespace
    from pyrpmd.parallel import execution
    provider = SimpleNamespace(metadata={"execution_device": "cuda:0"})
    plan = SimpleNamespace(mode="gpu", devices=["cuda:0"], trajectories_per_gpu=1)
    context = SimpleNamespace(plan=plan, slot=0, comm=None)
    with patch.object(execution, "_CONTEXT", context):
        with patch("pyrpmd.potentials.make_potential", return_value=provider) as factory:
            assert execution.make_task_potential({"provider": kind}, None) is provider
    assert factory.call_args.args[0] == {"provider": kind, "device": "cuda:0"}


@pytest.mark.parametrize("kind", sorted(BATCHED_ML_PROVIDERS))
def test_scientific_model_choices_remain_in_identity(kind):
    system = System(("H",), np.array([1837.]), np.zeros((1, 3)))
    config = {"task": "pimd", "beads": 2, "potential": {
        "provider": kind, "model": "local.pt", "dtype": "float64", "params": {"head": "one"},
    }}
    original = fingerprint(config, system)
    for changes in ({"dtype": "float32"}, {"model": "different.pt"},
                    {"params": {"head": "two"}}, {"virial": True}):
        updated = copy.deepcopy(config)
        updated["potential"].update(changes)
        assert fingerprint(updated, system) != original
        assert scientific_config(updated)["potential"] == updated["potential"]
    for old in ("mace", "nequip", "fairchem", "deepmd"):
        updated = copy.deepcopy(config)
        updated["potential"]["provider"] = old
        assert fingerprint(updated, system) != original


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_tensor_contract_numpy_boundary_independence_and_no_grad(dtype):
    torch = pytest.importorskip("torch")
    pes = AnalyticBatch(dtype=dtype, virial=True)
    validate_batch_potential(pes)
    q = torch.arange(24, dtype=getattr(torch, dtype)).reshape(4, 2, 3) / 10
    with torch.no_grad():
        out = pes.evaluate_tensor(q)
    assert out.energy.dtype == getattr(torch, dtype)
    assert out.energy.device == q.device and not out.forces.requires_grad
    np.testing.assert_allclose(out.energy.numpy(), np.sum(q.numpy()**2, axis=(1, 2))/2, rtol=2e-7)
    np.testing.assert_allclose(out.forces.numpy(), -q.numpy())
    np.testing.assert_allclose(out.virial.numpy(), -np.einsum("bna,bnc->bac", q.numpy(), q.numpy()), rtol=2e-7)
    assert pes.calls == 1
    npout = pes.evaluate(q.numpy())
    assert npout.energy.dtype == np.float64
    assert pes.metadata["native_batch"] and pes.metadata["batch_independent"]
    np.testing.assert_array_equal(npout.energy, out.energy.numpy())
    np.testing.assert_array_equal(pes.evaluate(q.flip(0).numpy()).energy, npout.energy[::-1])


def test_bad_tensor_contracts_fail_before_model_call():
    torch = pytest.importorskip("torch")
    pes = AnalyticBatch()
    for q in (torch.ones((2, 2, 3), dtype=torch.float32), torch.zeros((0, 2, 3)),
              torch.ones((2, 1, 3), dtype=torch.float64),
              torch.full((2, 2, 3), float("nan"), dtype=torch.float64),
              np.zeros((2, 2, 3))):
        with pytest.raises(PotentialError):
            pes.evaluate_tensor(q)
    with torch.inference_mode(), pytest.raises(PotentialError, match="inference_mode"):
        pes.evaluate_tensor(torch.zeros((2, 2, 3), dtype=torch.float64))
    with torch.autocast(device_type="cpu"), pytest.raises(PotentialError, match="autocast"):
        pes.evaluate_tensor(torch.zeros((2, 2, 3), dtype=torch.float64))
    assert pes.calls == 0
    pes.close()
    with pytest.raises(PotentialError, match="closed"):
        pes.evaluate_tensor(torch.ones((1, 2, 3), dtype=torch.float64))
    with pytest.raises(PotentialError, match="closed"):
        pes.evaluate(np.zeros((1, 2, 3)))


def test_batch_failure_is_not_retried_or_split():
    torch = pytest.importorskip("torch")
    pes = AnalyticBatch()
    with patch.object(pes, "_evaluate_tensor", side_effect=RuntimeError("CUDA out of memory")) as evaluate:
        with pytest.raises(PotentialError, match=r"out of memory.*max_batch_frames"):
            pes.evaluate_tensor(torch.ones((8, 2, 3), dtype=torch.float64))
    assert evaluate.call_count == 1

    chunked = AnalyticBatch(max_batch_frames=3)
    seen = []

    def fail_second(q):
        seen.append(len(q))
        if len(seen) == 2:
            raise RuntimeError("CUDA out of memory")
        return AnalyticBatch._evaluate_tensor(chunked, q)

    with patch.object(chunked, "_evaluate_tensor", side_effect=fail_second):
        with pytest.raises(PotentialError, match=r"out of memory.*below 3"):
            chunked.evaluate_tensor(torch.ones((8, 2, 3), dtype=torch.float64))
    assert seen == [3, 3]


@pytest.mark.parametrize("bad", [0, -1, True, False, 1.5, 2.0, "2", "AUTO", "", float("nan"), float("inf")])
def test_invalid_max_batch_frames_rejected_at_construction(bad):
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="max_batch_frames"):
        AnalyticBatch(max_batch_frames=bad)


def test_auto_max_batch_frames_cpu_and_vram_calibration_contract():
    torch = pytest.importorskip("torch")
    mib = 1024 ** 2
    q7 = torch.arange(42, dtype=torch.float64).reshape(7, 2, 3) / 10.0 + 0.05
    ref = AnalyticBatch(virial=True).evaluate_tensor(q7)

    # On CPU, 'auto' stays unchunked with zero extra calibration calls.
    cpu_auto = AnalyticBatch(virial=True, max_batch_frames="auto")
    assert cpu_auto.max_batch_frames == "auto"
    assert cpu_auto.resolved_max_batch_frames is None
    assert cpu_auto.metadata["max_batch_frames"] == "auto"
    assert cpu_auto.metadata["resolved_max_batch_frames"] is None
    cpu_out = cpu_auto.evaluate_tensor(q7)
    assert cpu_auto.calls == 1
    torch.testing.assert_close(cpu_out.energy, ref.energy, rtol=0.0, atol=1e-14)

    # Simulate CUDA VRAM calibration: 1000 MiB/frame, 4600 MiB available -> int(4600*0.65 // 1000) == 2.
    auto_pes = AnalyticBatch(
        cell=np.eye(3) * 10.0, pbc=[True, True, True], virial=True, max_batch_frames="auto",
    )
    borrowed = auto_pes.as_torch_pes()
    auto_pes._auto_calibrated = False
    seen = []
    real_eval = auto_pes._evaluate_tensor

    with patch.object(auto_pes, "_evaluate_tensor", side_effect=lambda x: (seen.append(len(x)), real_eval(x))[1]), \
         patch.object(torch.cuda, "memory_allocated", side_effect=[100 * mib, 102 * mib]), \
         patch.object(torch.cuda, "reset_peak_memory_stats"), \
         patch.object(torch.cuda, "max_memory_allocated", return_value=1100 * mib), \
         patch.object(torch.cuda, "memory_reserved", return_value=1202 * mib), \
         patch.object(torch.cuda, "mem_get_info", return_value=(3500 * mib, 16000 * mib)):
        out1 = auto_pes.evaluate_tensor(q7)
        assert auto_pes.resolved_max_batch_frames == 2
        assert auto_pes.metadata["resolved_max_batch_frames"] == 2
        assert borrowed.metadata["resolved_max_batch_frames"] == 2
        assert seen == [1, 2, 2, 2, 1]
        seen.clear()
        out2 = auto_pes.evaluate_tensor(q7)
        assert seen == [2, 2, 2, 1]
    assert torch.equal(out1.energy, out2.energy)
    torch.testing.assert_close(out2.energy, ref.energy, rtol=0.0, atol=1e-14)
    torch.testing.assert_close(out2.forces, ref.forces, rtol=0.0, atol=1e-14)
    torch.testing.assert_close(out2.virial, ref.virial, rtol=0.0, atol=1e-14)

    # Small cell compression (<15%) keeps calibration; large compression (>15%) invalidates it.
    auto_pes.set_cell(np.eye(3) * 9.8)
    assert auto_pes.resolved_max_batch_frames == 2
    auto_pes.set_cell(np.eye(3) * 8.0)
    assert auto_pes.resolved_max_batch_frames is None
    assert borrowed.metadata["resolved_max_batch_frames"] is None

    # Tight VRAM -> resolved == 1 reuses the calibration frame without an extra model call.
    auto_pes._auto_calibrated = False
    seen.clear()
    with patch.object(auto_pes, "_evaluate_tensor", side_effect=lambda x: (seen.append(len(x)), real_eval(x))[1]), \
         patch.object(torch.cuda, "memory_allocated", side_effect=[100 * mib, 101 * mib]), \
         patch.object(torch.cuda, "reset_peak_memory_stats"), \
         patch.object(torch.cuda, "max_memory_allocated", return_value=1300 * mib), \
         patch.object(torch.cuda, "memory_reserved", return_value=1351 * mib), \
         patch.object(torch.cuda, "mem_get_info", return_value=(200 * mib, 16000 * mib)):
        out_tight = auto_pes.evaluate_tensor(q7[:4])
    assert auto_pes.resolved_max_batch_frames == 1
    assert seen == [1, 1, 1, 1]
    torch.testing.assert_close(out_tight.energy, ref.energy[:4], rtol=0.0, atol=1e-14)

    # Rebinding system resets calibration.
    auto_pes._auto_calibrated = True
    auto_pes.bind_system({"symbols": ["H", "H"], "cell": np.eye(3) * 10.0, "pbc": [True, True, True]})
    assert auto_pes.resolved_max_batch_frames is None


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("virial", [False, True])
@pytest.mark.parametrize("frames,expected_chunks", [(1, [1]), (2, [2]), (5, [2, 2, 1]), (7, [2, 2, 2, 1])])
def test_deterministic_sub_batch_chunking_contract(dtype, virial, frames, expected_chunks):
    torch = pytest.importorskip("torch")
    full = AnalyticBatch(dtype=dtype, virial=virial)
    chunked = AnalyticBatch(dtype=dtype, virial=virial, max_batch_frames=2)
    assert full.metadata["max_batch_frames"] is None
    assert chunked.metadata["max_batch_frames"] == 2
    assert chunked.as_torch_pes().metadata["max_batch_frames"] == 2 if dtype == "float64" else True

    seen = []
    real_eval = chunked._evaluate_tensor
    q = torch.arange(frames * 6, dtype=getattr(torch, dtype)).reshape(frames, 2, 3) / 10.0 + 0.05
    expected = full.evaluate_tensor(q)
    with patch.object(chunked, "_evaluate_tensor", side_effect=lambda x: (seen.append(len(x)), real_eval(x))[1]):
        actual = chunked.evaluate_tensor(q)
    assert seen == expected_chunks
    torch.testing.assert_close(actual.energy, expected.energy, rtol=0.0, atol=1e-14)
    torch.testing.assert_close(actual.forces, expected.forces, rtol=0.0, atol=1e-14)
    if virial:
        torch.testing.assert_close(actual.virial, expected.virial, rtol=0.0, atol=1e-14)
    else:
        assert actual.virial is None


def test_native_reduction_dtypes_are_preserved_and_checked():
    torch = pytest.importorskip("torch")
    from pyrpmd.torch_backend import TensorPESResult

    class MixedReductionBatch(AnalyticBatch):
        def __init__(self):
            super().__init__(dtype="float32", virial=True)
            self.energy_dtype = self.virial_dtype = torch.float64
            self.metadata.update(energy_dtype="float64", virial_dtype="float64")

        def _evaluate_tensor(self, q):
            self.calls += 1
            geometry = q.to(dtype=torch.float64)
            energy = (geometry * geometry).sum(dim=(1, 2)) / 2
            virial = -torch.einsum("bna,bnc->bac", geometry, geometry)
            return TensorPESResult(energy, -q, virial)

    pes = MixedReductionBatch()
    q = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3) / 10
    result = pes.evaluate_tensor(q)
    assert result.energy.dtype == result.virial.dtype == torch.float64
    assert result.forces.dtype == torch.float32
    reference = q.to(dtype=torch.float64)
    torch.testing.assert_close(result.energy, (reference * reference).sum((1, 2)) / 2)
    torch.testing.assert_close(result.virial, -torch.einsum("bna,bnc->bac", reference, reference))
    np.testing.assert_array_equal(pes.evaluate(q.numpy()).virial, result.virial.numpy())
    with pytest.raises(ValueError, match="requires float64"):
        pes.as_torch_pes()
    with patch.object(pes, "_evaluate_tensor", return_value=TensorPESResult(
            result.energy, result.forces, result.virial.float())):
        with pytest.raises(PotentialError, match="virial must be torch.float64"):
            pes.evaluate_tensor(q)


def test_float64_torch_bridge_checks_requested_virial_precision():
    torch = pytest.importorskip("torch")
    pes = AnalyticBatch(virial=True)
    pes.virial_dtype = torch.float32
    with pytest.raises(ValueError, match="requires float64"):
        pes.as_torch_pes()


def test_binding_caches_are_rebuilt_and_failed_binding_preserves_state():
    pytest.importorskip("torch")
    pes = AnalyticBatch(cell=np.eye(3)*8, pbc=[True, True, True])
    old = pes._cell_tensor().clone()
    with pytest.raises(PotentialError, match="composition"):
        pes.bind_system({"symbols": ["He", "H"], "cell": np.eye(3)*5})
    np.testing.assert_array_equal(pes._cell_tensor().numpy(), old.numpy())
    pes.bind_system({"symbols": ["H", "H"], "cell": np.eye(3)*10})
    np.testing.assert_array_equal(pes._cell_tensor().numpy(), np.eye(3)*10)
    pes.set_cell(np.eye(3)*11)
    np.testing.assert_array_equal(pes._cell_tensor().numpy(), np.eye(3)*11)


def test_model_precision_is_checked_without_changing_parameters():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(3, 1, dtype=torch.float32)
    with pytest.raises(ValueError, match="no implicit precision"):
        AnalyticBatch()._prepare_module(model)
    assert model.weight.dtype == torch.float32
    model.metadata = {"batch_independent": False}
    with pytest.raises(PotentialError, match="stateless"):
        AnalyticBatch(dtype="float32")._prepare_module(model)


def test_explicit_missing_cuda_has_no_fallback():
    torch = pytest.importorskip("torch")
    from pyrpmd.potentials import BackendUnavailableError
    with patch.object(torch.cuda, "is_available", return_value=False):
        with pytest.raises(BackendUnavailableError, match="no CPU fallback"):
            AnalyticBatch(device="cuda")


def test_float64_provider_runs_in_existing_torch_dynamics_without_backend_edits():
    torch = pytest.importorskip("torch")
    from pyrpmd.torch_backend import TorchDynamics, TorchPES, TorchState
    pes = AnalyticBatch()
    wrapped = pes.as_torch_pes()
    assert isinstance(wrapped, TorchPES)
    q = torch.arange(24, dtype=torch.float64).reshape(2, 2, 2, 3) / 100
    state = TorchState(q.clone(), torch.zeros_like(q))
    dynamics = TorchDynamics(wrapped, [1837., 3674.], beta=100., nbeads=2, dt=0.01)
    dynamics.propagate(state, 2)
    assert pes.calls > 0
    assert torch.isfinite(state.q).all()
    with pytest.raises(ValueError, match="requires float64"):
        AnalyticBatch(dtype="float32").as_torch_pes()
