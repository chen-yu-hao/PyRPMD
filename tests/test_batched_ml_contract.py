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
])
def test_schema_rejects_implicit_precision_and_ambiguous_options(option):
    config = small_collision()
    config["potential"] = {"provider": "mace-batched", "model": "missing.model", **option}
    with pytest.raises(InputError):
        JobSpec.from_mapping(config).resolve()


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
    with patch.object(pes, "_evaluate_tensor", side_effect=RuntimeError("out of memory")) as evaluate:
        with pytest.raises(PotentialError, match="out of memory"):
            pes.evaluate_tensor(torch.ones((8, 2, 3), dtype=torch.float64))
    assert evaluate.call_count == 1


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
