"""Selected optional resources must distinguish valid input from ready execution."""
import json
from types import SimpleNamespace

import pytest

from pyrpmd import resources
from pyrpmd.cli import main
from pyrpmd.runner import prepare
from pyrpmd.scaffold import init_project


def test_serial_resources_do_not_import_optional_backends(monkeypatch):
    def fail(name):
        raise AssertionError("unexpected optional import " + name)
    monkeypatch.setattr(resources.importlib, "import_module", fail)
    assert resources.resource_report({"task": "fes", "parallel": {"backend": "serial"}}) == {"available": True, "checks": {}}


def test_mpi_missing_package_is_unavailable_without_launch(monkeypatch):
    monkeypatch.setattr(resources.importlib.util, "find_spec", lambda name: None)
    result = resources.resource_report({"parallel": {"backend": "mpi", "trajectory_workers": 2}})
    assert result["available"] is False
    assert result["checks"]["mpi"]["required_ranks"] == 2
    assert "mpi extra" in result["checks"]["mpi"]["hint"]


def test_mpi_requires_matching_launcher_context(monkeypatch):
    monkeypatch.setattr(resources.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(resources.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    for name in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE", "MV2_COMM_WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)
    config = {"parallel": {"backend": "mpi", "trajectory_workers": 2}}
    assert resources.resource_report(config)["available"] is False
    monkeypatch.setenv("PMI_SIZE", "2")
    assert resources.resource_report(config)["available"] is True
    monkeypatch.setenv("PMI_SIZE", "3")
    assert resources.resource_report(config)["available"] is False


def test_mpi_runtime_loader_failure_remains_unavailable(monkeypatch):
    monkeypatch.setattr(resources.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(resources.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stderr="libmpi missing"))
    monkeypatch.setenv("PMI_SIZE", "2")
    report = resources.resource_report({"parallel": {"backend": "mpi", "trajectory_workers": 2}})
    assert not report["available"]
    assert "libmpi missing" in report["checks"]["mpi"]["reason"]


@pytest.mark.parametrize("version,available", [(7, False), (8, True)])
def test_plumed_probe_only_queries_api_without_input_or_init(monkeypatch, version, available):
    calls = []
    class Kernel:
        def cmd(self, name, output):
            calls.append(name)
            output[0] = version
        def finalize(self):
            calls.append("finalize")
    monkeypatch.setattr(resources.importlib, "import_module", lambda name: SimpleNamespace(Plumed=lambda **kwargs: Kernel()))
    report = resources.resource_report({"task": "plumed"})
    assert report["available"] is available
    assert calls == ["getApiVersion", "finalize"]


def test_missing_plumed_kernel_is_reported_with_hint(monkeypatch):
    def fail(name):
        raise RuntimeError("libplumedKernel missing")
    monkeypatch.setattr(resources.importlib, "import_module", fail)
    report = resources.resource_report({"task": "plumed"})
    assert report["available"] is False
    assert "PLUMED_KERNEL" in report["checks"]["plumed"]["hint"]


@pytest.mark.parametrize("device,available", [("cpu", True), ("cuda:0", True), ("cuda:2", False)])
def test_torch_probe_checks_requested_device_without_allocating(monkeypatch, device, available):
    fake = SimpleNamespace(
        device=lambda value: SimpleNamespace(type=value.split(":")[0], index=int(value.split(":")[1]) if ":" in value else None),
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1))
    monkeypatch.setattr(resources.importlib, "import_module", lambda name: fake)
    report = resources.resource_report({"sampling": {"engine": "torch", "device": device}})
    assert report["available"] is available


def test_native_gpu_probe_does_not_require_python_torch_or_onnx(monkeypatch):
    def fail(name):
        raise AssertionError("unexpected optional import " + name)
    monkeypatch.setattr(resources.importlib, "import_module", fail)
    report = resources.resource_report({
        "potential": {"provider": "cabi", "params": {"device": "cuda:0"}},
    })
    assert report["available"] is True
    assert report["checks"]["native_gpu"]["deferred"] is True


def test_validate_valid_but_not_ready_and_prepare_skips_numerical_work(tmp_path, monkeypatch, capsys):
    starter = init_project(tmp_path / "project")
    unavailable = {"available": False, "checks": {"mpi": {"available": False, "hint": "launch with MPI"}}}
    monkeypatch.setattr(resources, "resource_report", lambda config: unavailable)
    assert main(["validate", starter["input"]]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["valid"] is True and result["ready"] is False
    assert result["resources"] == unavailable
    assert "input_warnings" in result
    from pyrpmd import runner
    original = runner._prepare
    seen = []
    def checked(*args, **kwargs):
        seen.append(kwargs["numerical"])
        return original(*args, **kwargs)
    monkeypatch.setattr(runner, "_prepare", checked)
    assert prepare(starter["input"], numerical=True)["ready"] is False
    assert seen == [False]
    assert not (tmp_path / "project/run").exists()
