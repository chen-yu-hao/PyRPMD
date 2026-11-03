"""Numerical-reference and honest-failure tests for the T19 measurement driver."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.efficiency import t19_cuda as benchmark


def _spec(tmp_path):
    model = tmp_path / "explicit-untrained-fixture.bin"
    model.write_bytes(b"analytic test fixture; not an ML model")
    return {"name": "fixture", "provider": "mace-batched", "model": str(model),
            "sha256": benchmark.sha256(model), "model_kind": "untrained", "dtype": "float64",
            "params": {}, "geometry": {"species": ["H", "O"], "spacing_bohr": 4.0,
                                       "jitter_bohr": .02, "pbc": [True] * 3}}


def _manifest(tmp_path, specs):
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"schema": "pyrpmd.t19-models.v1", "models": specs}))
    return path


@pytest.mark.parametrize("change", [
    {"beads": (8, 8)}, {"atoms": (False,)}, {"replicas": 0}, {"warmup": 0},
    {"repeats": 2}, {"threads": True}, {"seed": -1}, {"device": "auto"},
    {"device": "mps"}, {"suite": "kernel"}, {"rtol": float("nan")},
    {"atol": 0}, {"normal_mode_layout": "unknown"},
    {"production_steps": 63}, {"production_steps": 16},
    {"providers": ()}, {"providers": ("mace-batched", "mace-batched")}, {"providers": ("unsupported",)},
])
def test_settings_reject_ambiguous_or_unmeasurable_requests(change):
    with pytest.raises(ValueError):
        benchmark.Settings(**change)


def test_manifest_is_explicit_local_float64_and_hash_pinned(tmp_path):
    spec = _spec(tmp_path)
    spec["model"] = Path(spec["model"]).name
    loaded = benchmark.load_manifest(_manifest(tmp_path, [spec]))
    assert loaded[0]["model"] == str(tmp_path / spec["model"])
    assert loaded[0]["sha256"] == spec["sha256"]
    for change in ({"dtype": "float32"}, {"sha256": ""}, {"model_kind": "unknown"},
                   {"provider": "ase"}, {"model": "https://example.invalid/model"},
                   {"params": {"device": "cpu"}}, {"params": {"dtype": "float32"}},
                   {"structures": {"64": "n64.npz"}}):
        with pytest.raises(ValueError):
            benchmark.load_manifest(_manifest(tmp_path, [dict(spec, **change)]))
    with pytest.raises(ValueError, match="unique"):
        benchmark.load_manifest(_manifest(tmp_path, [spec, spec]))


def test_generated_geometry_is_seeded_in_atomic_units_without_global_rng_change(tmp_path):
    spec = _spec(tmp_path)
    np.random.seed(8723)
    before = np.random.get_state()
    q, symbols, cell, pbc, metadata = benchmark.geometry(spec, 64, 8, 100)
    after = np.random.get_state()
    assert q.shape == (8, 64, 3) and q.dtype == np.float64
    assert symbols == ["H", "O"] * 32
    np.testing.assert_array_equal(cell, np.eye(3) * 16)
    np.testing.assert_array_equal(pbc, [True] * 3)
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    again = benchmark.geometry(spec, 64, 8, 100)
    np.testing.assert_array_equal(q, again[0])
    assert metadata["coordinates_sha256"] == again[-1]["coordinates_sha256"]
    assert metadata["kind"] == "synthetic_lattice_performance_probe"
    assert not np.array_equal(q, benchmark.geometry(spec, 64, 8, 101)[0])


def test_structure_precision_and_periodicity_are_not_silently_changed(tmp_path):
    path = tmp_path / "structure.npz"
    q = np.arange(6, dtype=np.float64).reshape(2, 3)
    spec = {"structures": {"2": str(path)}, "jitter_bohr": 0.0}
    np.savez(path, positions=q, symbols=np.array(["H", "H"]))
    result = benchmark.geometry(spec, 2, 3, 4)
    np.testing.assert_array_equal(result[0], np.broadcast_to(q, (3, 2, 3)))
    assert result[-1]["sha256"] == benchmark.sha256(path)
    np.savez(path, positions=q.astype(np.float32), symbols=np.array(["H", "H"]))
    with pytest.raises(ValueError, match="float64"):
        benchmark.geometry(spec, 2, 3, 4)
    np.savez(path, positions=q, symbols=np.array(["H", "H"]), pbc=np.array([True] * 3))
    with pytest.raises(ValueError, match="require a cell"):
        benchmark.geometry(spec, 2, 3, 4)


def test_error_table_has_true_zero_reference_semantics_and_rejects_nan():
    exact = benchmark.error_metrics([0., 0.], [0., 0.], rtol=1e-9, atol=1e-10)
    assert exact["passed"] and exact["relative_l2"] == 0
    wrong = benchmark.error_metrics([1., -1.], [0., 0.], rtol=1e-9, atol=1e-10)
    assert wrong["max_absolute"] == 1 and wrong["rms"] == 1
    assert wrong["relative_l2"] is None and wrong["passed"] is False
    with pytest.raises(ValueError, match="finite"):
        benchmark.error_metrics([np.nan], [0.], rtol=1e-9, atol=1e-10)
    large = benchmark.error_metrics([1e200, -1e200], [0., 0.], rtol=1e-9, atol=1e-10)
    assert large["rms"] == pytest.approx(1e200)
    json.dumps(large, allow_nan=False)


def test_timing_waits_for_cuda_work_and_excludes_warmup(monkeypatch):
    sequence = []
    clock = iter([1., 1.003, 2., 2.001, 3., 3.002])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(clock))
    timing = benchmark.timed_calls(lambda: sequence.append("work"),
                                   lambda: sequence.append("synchronize"), warmup=2, repeats=3)
    assert sequence == ["synchronize", "work", "synchronize"] * 5
    assert timing["median_ms"] == pytest.approx(2.0)
    assert len(timing["seconds"]) == 3


def _blocked_environment(device):
    return {"requested_device": device, "blocking_reasons": ["CUDA unavailable; no CPU fallback"],
            "cuda_available": False, "cuda_build_version": None, "gpu": None, "nvidia_smi": None}


def test_cuda_metadata_keeps_actual_torch_build_and_driver_distinct(monkeypatch):
    fake = SimpleNamespace(
        __version__="test-torch", version=SimpleNamespace(cuda="12.6"),
        get_num_threads=lambda: 1, get_num_interop_threads=lambda: 1,
        empty=lambda *a, **kw: SimpleNamespace(device="cuda:0"),
        cuda=SimpleNamespace(is_available=lambda: True,
                             get_device_properties=lambda device: SimpleNamespace(
                                 name="Test GPU", total_memory=123456, major=8, minor=0, uuid="GPU-test")))
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setattr(benchmark, "_command", lambda *a: "0, GPU-test, Test GPU, 550.123, 16384")
    record = benchmark.environment("cuda:0")
    assert record["blocking_reasons"] == []
    assert record["versions"]["torch"] == "test-torch"
    assert record["cuda_build_version"] == "12.6"
    assert record["cuda_driver_version"] == "550.123"
    assert record["gpu"]["model"] == "Test GPU"


def test_missing_cuda_reports_every_requested_case_without_running_cpu(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "environment", _blocked_environment)
    monkeypatch.setattr(benchmark, "source_snapshot", lambda: {"git_head": "test", "torch_workflow_available": False})
    monkeypatch.setattr(benchmark, "benchmark_provider", lambda *args: pytest.fail("must not evaluate a CPU fallback"))
    monkeypatch.setattr(benchmark, "benchmark_workflow", lambda *args: pytest.fail("must not run a CPU fallback"))
    output = tmp_path / "blocked.json"
    report = benchmark.run(output=output)
    assert len(report["cases"]) == 63  # five ML providers + two workflows, each 3 x 3
    assert all(row["status"] == "blocked" and "timings" not in row for row in report["cases"])
    assert report["execution_complete"] and not report["cuda_matrix_complete"]
    assert report["requested_cases_passed"] is False
    assert json.loads(output.read_text()) == json.loads(json.dumps(report))
    rendered = benchmark.markdown(report)
    assert "| blocked | — | — | — |" in rendered
    assert "PR #84" in rendered
    assert not list(tmp_path.glob("*.tmp"))


def test_explicit_provider_subset_can_complete_without_claiming_all_of_t19(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    # Only routing/coverage is mocked here; these are not GPU measurements.
    monkeypatch.setattr(benchmark, "environment", lambda device: {
        "blocking_reasons": [], "gpu": {"model": "test-fixture"},
        "cuda_driver_version": "test-driver", "cuda_build_version": "test-build"})
    monkeypatch.setattr(benchmark, "source_snapshot", lambda: {"git_head": "test", "torch_workflow_available": True})
    monkeypatch.setattr(benchmark, "benchmark_provider", lambda *a: {"status": "passed"})
    monkeypatch.setattr(benchmark, "benchmark_workflow", lambda *a: {"status": "passed"})
    settings = benchmark.Settings(providers=("mace-batched",))
    report = benchmark.run(settings, manifest=_manifest(tmp_path, [_spec(tmp_path)]))
    assert len(report["cases"]) == 27  # One framework and two workflows, full size grid.
    assert report["requested_cases_passed"] is True
    assert {row["provider"] for row in report["cases"] if row["suite"] == "ml"} == {"mace-batched"}
    assert report["cuda_matrix_complete"] is False


def test_cli_returns_nonzero_and_saves_blockers(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "environment", _blocked_environment)
    monkeypatch.setattr(benchmark, "source_snapshot", lambda: {"git_head": "test", "torch_workflow_available": False})
    output, markdown = tmp_path / "blocked.json", tmp_path / "blocked.md"
    status = benchmark.main(["--output", str(output), "--markdown", str(markdown),
                             "--atoms", "64", "--beads", "8"])
    assert status == 2 and output.is_file() and markdown.is_file()
    assert len(json.loads(output.read_text())["cases"]) == 7


def test_atomic_report_failure_preserves_prior_record(tmp_path, monkeypatch):
    output = tmp_path / "result.json"
    benchmark.atomic_json(output, {"checkpoint": 1})
    def fail(*args):
        raise OSError("publication interrupted")
    monkeypatch.setattr(benchmark.os, "replace", fail)
    with pytest.raises(OSError, match="interrupted"):
        benchmark.atomic_json(output, {"checkpoint": 2})
    assert json.loads(output.read_text()) == {"checkpoint": 1}
    assert list(tmp_path.iterdir()) == [output]


def test_source_provenance_follows_imported_package_not_benchmark_checkout(tmp_path):
    package = tmp_path / "pyrpmd"
    for directory in (package, package / "potentials", package / "workflows"):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "__init__.py").write_text("")
    source = package / "torch_backend.py"
    source.write_text("marker = 'different tested source'\n")
    child = subprocess.run(
        [sys.executable, "-c", "import json; from benchmarks.efficiency.t19_cuda import source_snapshot; print(json.dumps(source_snapshot()))"],
        cwd=Path(benchmark.__file__).resolve().parents[2],
        env=dict(os.environ, PYTHONPATH=str(tmp_path)), capture_output=True, text=True, check=True)
    snapshot = json.loads(child.stdout)
    assert snapshot["package_path"] == str(package / "__init__.py")
    assert snapshot["modules"]["pyrpmd.torch_backend"]["sha256"] == benchmark.sha256(source)
    assert snapshot["benchmark"]["path"] == str(Path(benchmark.__file__).resolve())
    assert snapshot["torch_workflow_available"] is False


def _analytic_provider(symbols, *, device, dtype, **kwargs):
    torch = pytest.importorskip("torch")
    from pyrpmd.potentials._batched_ml_base import _TensorBatchPotential
    from pyrpmd.torch_backend import TensorPESResult

    class Analytic(_TensorBatchPotential):
        def __init__(self):
            self._setup(symbols, "analytic-fixture", device=device, dtype=dtype,
                        cell=kwargs.get("cell"), pbc=kwargs.get("pbc"))

        def _evaluate_tensor(self, q):
            # Quartic potential: independent closed-form force reference.
            return TensorPESResult((.5 * q.square() + .025 * q**4).sum(dim=(1, 2)), -q - .1 * q**3)

    return Analytic()


def test_actual_tensor_measurement_preserves_fields_and_flags_wrong_force(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from pyrpmd.potentials import batched_ml
    instances = []
    def factory(kind, **kwargs):
        provider = _analytic_provider(**kwargs)
        instances.append(provider)
        return provider
    monkeypatch.setattr(batched_ml, "make_batched_ml_potential", factory)
    settings = benchmark.Settings(device="cpu", atoms=(4,), beads=(3,), warmup=1, repeats=3)
    spec = _spec(tmp_path)
    result = benchmark.benchmark_provider(spec, 4, 3, settings)
    assert result["status"] == "passed" and result["model_kind"] == "untrained"
    assert set(result["timings"]) == {"cpu_numpy", "target_numpy", "target_tensor"}
    assert all(len(v["seconds"]) == 3 and v["median_ms"] > 0 for v in result["timings"].values())
    assert result["peak_cuda_allocated_bytes"] is None
    assert all(provider.closed for provider in instances)
    q = np.array([[[.1, -.2, .3]]], dtype=np.float64)
    with _analytic_provider(["H"], device="cpu", dtype="float64") as provider:
        force = provider.evaluate(q).forces[0, 0, 0]
        plus, minus = q.copy(), q.copy()
        plus[0, 0, 0] += 1e-5
        minus[0, 0, 0] -= 1e-5
        derivative = -(provider.evaluate(plus).energy[0] - provider.evaluate(minus).energy[0]) / 2e-5
        assert force == pytest.approx(derivative, rel=1e-8, abs=1e-10)
    bad = copy.deepcopy(result["correctness"])
    bad["tensor"]["forces"]["passed"] = False
    assert benchmark._passed(bad) is False


def test_model_hash_mismatch_fails_before_loading_any_model(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from pyrpmd.potentials import batched_ml
    monkeypatch.setattr(batched_ml, "make_batched_ml_potential", lambda *a, **kw: pytest.fail("model was changed"))
    spec = _spec(tmp_path)
    Path(spec["model"]).write_bytes(b"different model")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        benchmark.benchmark_provider(spec, 4, 3, benchmark.Settings(device="cpu"))


def test_workflow_numpy_reference_detects_observable_and_transport_corruption():
    # Analytic free ring with all beads colocated, hence zero spring energy.
    beads, atoms, beta = 8, 2, 4.
    masses = np.array([2., 3.])
    velocities = np.arange(1, 7, dtype=float).reshape(2, 3) * .01
    times = np.arange(4, dtype=float) * .125
    positions = times[:, None, None] * velocities
    q = np.broadcast_to(positions[-1], (beads, atoms, 3)).copy()
    p = np.broadcast_to(velocities * masses[:, None], q.shape).copy()
    kinetic = np.sum(p**2 / (2 * masses[None, :, None]))
    observables = np.tile([0, 1.5 * atoms * beads / beta, 1.5 * atoms / beta,
                           1.5 * atoms / beta, kinetic], (len(times), 1))
    stream = {"state": SimpleNamespace(q=q, p=p), "positions": positions,
              "velocities": np.broadcast_to(velocities, positions.shape).copy(), "times": times,
              "observables": observables, "radii_gyration_squared": np.zeros((len(times), atoms))}
    settings = benchmark.Settings()
    result = {"streams": [stream]}
    check = benchmark.workflow_reference(result, task="diffusion", beta=beta, masses=masses, settings=settings)
    assert benchmark._passed(check)
    stream["positions"][1, 0, 0] += .1
    check = benchmark.workflow_reference(result, task="diffusion", beta=beta, masses=masses, settings=settings)
    assert check["0"]["ballistic_positions"]["passed"] is False
    stream["observables"][-1, 2] += .1
    check = benchmark.workflow_reference(result, task="diffusion", beta=beta, masses=masses, settings=settings)
    assert check["0"]["observables"]["passed"] is False


@pytest.mark.parametrize("corruption", ["empty", "partial", "step", "time", "frames",
                                        "sample_clock", "sample_time", "missing_data", "shape", "replica"])
def test_workflow_cannot_report_unperformed_steps_as_a_speedup(corruption):
    settings = benchmark.Settings(device="cpu", beads=(4,), atoms=(2,),
                                  production_steps=8, equilibration_steps=2, sample_interval=2)
    config = benchmark.workflow_configuration("pimd", 4, settings, "cpu")
    stream = {"replica": 0, "complete": True, "elapsed": 10, "frames": 5,
              "state": SimpleNamespace(q=np.zeros((4, 2, 3)), p=np.zeros((4, 2, 3)), step=10, time=1.25),
              "sample_steps": np.arange(2, 11, 2), "times": np.arange(5, dtype=float) * .25,
              "observables": np.zeros((5, 5)), "positions": np.zeros((5, 2, 3)),
              "velocities": np.zeros((5, 2, 3)), "radii_gyration_squared": np.zeros((5, 2))}
    result = {"summary": {}, "streams": [stream]}
    benchmark.validate_workflow_completion(result, config, 2)
    if corruption == "empty":
        result["streams"] = []
        with pytest.raises(ValueError, match="nonempty"):
            benchmark.workflow_reference(result, task="pimd", beta=4., masses=np.ones(2), settings=settings)
        assert benchmark._passed({}) is False
    elif corruption == "partial":
        stream["complete"] = False
    elif corruption == "step":
        stream["state"].step = 9
    elif corruption == "time":
        stream["state"].time = 1.0
    elif corruption == "frames":
        stream["frames"] = 4
    elif corruption == "sample_clock":
        stream["sample_steps"][-1] = 9
    elif corruption == "sample_time":
        stream["times"][-1] = .75
    elif corruption == "missing_data":
        stream["positions"] = stream["positions"][:-1]
    elif corruption == "shape":
        stream["state"].q = np.zeros((1, 2, 3))
    elif corruption == "replica":
        stream["replica"] = 1
    with pytest.raises(ValueError, match="workflow"):
        benchmark.validate_workflow_completion(result, config, 2)


def test_pr84_workflow_cpu_if_present(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("pyrpmd.workflows.torch_materials")
    settings = benchmark.Settings(device="cpu", beads=(4,), atoms=(2,), repeats=3, warmup=1,
                                  production_steps=8, equilibration_steps=2, sample_interval=2)
    for task in ("pimd", "diffusion"):
        result = benchmark.benchmark_workflow(task, 2, 4, settings, tmp_path)
        assert result["status"] == "passed" and result["physical_converged"] is False
        assert result["steps_per_replica"] == 10


def _cuda_torch():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware is required")
    return torch


def test_cuda_nondefault_stream_compares_actual_tensor_forces_to_cpu():
    _cuda_torch()
    q = np.random.default_rng(19064).normal(size=(8, 4, 3))
    with _analytic_provider(["H"] * 4, device="cpu", dtype="float64") as cpu:
        with _analytic_provider(["H"] * 4, device="cuda:0", dtype="float64") as gpu:
            assert benchmark._passed(benchmark.stream_probe(gpu, q, cpu, benchmark.Settings()))


def test_cuda_mace_graph_caller_stream_with_changing_batch_size():
    _cuda_torch()
    pytest.importorskip("mace.modules")
    pytest.importorskip("vesin_torch")
    from examples.potentials._random_mace import make_random_mace
    from pyrpmd.potentials.batched_ml import MACEBatchedPotential

    rng = np.random.default_rng(19064)
    base = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.], [0., 0., 2.]])
    q = base + rng.normal(scale=.01, size=(8, 4, 3))
    model = make_random_mace()
    with MACEBatchedPotential(["H"] * 4, copy.deepcopy(model), device="cpu") as cpu:
        with MACEBatchedPotential(["H"] * 4, copy.deepcopy(model), device="cuda:0") as gpu:
            assert benchmark._passed(benchmark.stream_probe(gpu, q, cpu, benchmark.Settings()))


def test_cuda_pr84_public_pimd_and_diffusion_if_present(tmp_path):
    _cuda_torch()
    pytest.importorskip("pyrpmd.workflows.torch_materials")
    settings = benchmark.Settings(beads=(8,), atoms=(4,), repeats=3, warmup=1,
                                  production_steps=8, equilibration_steps=2, sample_interval=2)
    for task in ("pimd", "diffusion"):
        result = benchmark.benchmark_workflow(task, 4, 8, settings, tmp_path)
        assert result["status"] == "passed" and result["physical_converged"] is False
