"""Dense solver equivalence, resource admission, task concurrency and restart."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
from scipy.linalg.lapack import ztrsyl

from pyrpmd.analysis import subspace_corrections
from pyrpmd.analysis.subspace_model import SubspaceModel
from pyrpmd.analysis.subspace_quantum import _GridOperators, _triangular_sylvester, exact_quantum_rate
import pyrpmd.analysis.subspace_execution as execution


def cases():
    x = np.linspace(-5., 5., 65)
    m = SubspaceModel((x,), .3 / np.cosh(x)**2, [1.], 0., reactant_reference=-3.)
    return {"one": dict(model=m, beta=2., nbeads=4,
                        rpmd_options=dict(windows=7, walkers=4, burn_in=2, sweeps=4,
                                          leapfrog_steps=2, trajectories=16, md_time=1., md_dt=.05,
                                          bootstrap=6, seed=123))}


@pytest.mark.parametrize("shape", [(8,), (8, 9), (8, 9, 10)])
def test_dvr_assembly_matches_kronecker_sum_and_arbitrary_truncation(shape):
    axes = tuple(np.linspace(-3, 3, n) for n in shape)
    model = SubspaceModel(axes, np.arange(np.prod(shape)).reshape(shape) * .001, np.ones(len(shape)), 0.)
    ops = _GridOperators(model)
    reference = np.diag(model.values.ravel())
    for a, matrix in enumerate(ops.kinetic):
        reference += np.kron(np.kron(np.eye(int(np.prod(shape[:a]))), matrix), np.eye(int(np.prod(shape[a + 1:]))))
    np.testing.assert_allclose(ops.dense_hamiltonian(), reference, atol=1e-15)
    keep = np.random.default_rng(4).permutation(ops.size)[::2]
    np.testing.assert_allclose(ops.dense_hamiltonian(keep), reference[np.ix_(keep, keep)], atol=1e-15)


def test_blocked_sylvester_matches_lapack_and_equation_residual():
    rng = np.random.default_rng(4)
    n = 145
    t = np.triu((rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) * .01)
    t[np.diag_indices(n)] = np.arange(n) * .03 - .3j
    c = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    expected, scale, info = ztrsyl(t, t, c, trana="C", tranb="N", isgn=-1)
    actual, actual_scale, actual_info, method = _triangular_sylvester(t, c)
    assert method == "blocked_lapack" and actual_info == info == 0
    np.testing.assert_allclose(actual / actual_scale, expected / scale, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(t.conj().T @ actual - actual @ t, c, rtol=1e-11, atol=1e-11)


def test_sylvester_global_scaling_fallback():
    t = np.eye(70, dtype=complex)
    c = np.eye(70, dtype=complex)
    expected, scale, info = ztrsyl(t, t, c, trana="C", tranb="N", isgn=-1)
    actual, actual_scale, actual_info, method = _triangular_sylvester(t, c)
    assert method == "lapack_fallback"
    assert actual_scale == scale and actual_info == info
    np.testing.assert_array_equal(actual, expected)


def test_dense_memory_budget_fails_before_allocating(monkeypatch):
    monkeypatch.setattr(_GridOperators, "dense_hamiltonian", lambda *args: pytest.fail("allocated dense DVR"))
    with pytest.raises(MemoryError, match="budget"):
        exact_quantum_rate(cases()["one"]["model"], 2., memory_limit_mb=1.)


def test_serial_and_process_corrections_match_and_threads_are_bounded(tmp_path):
    serial = subspace_corrections(cases(), execution={"backend": "serial", "cpus": 1})["one"]
    parallel = subspace_corrections(cases(), execution={"cpus": 2, "workers": 2,
                                                       "checkpoint_dir": str(tmp_path)})["one"]
    assert parallel.factor == pytest.approx(serial.factor, rel=1e-12)
    assert parallel.log_standard_error == pytest.approx(serial.log_standard_error, rel=1e-12)
    for result in (parallel.quantum, parallel.rpmd):
        assert result.diagnostics["execution"]["threads"] == 1
        assert all(p["num_threads"] == 1 for p in result.diagnostics["execution"]["blas_pools"])
    assert len(list(tmp_path.glob("*.json"))) == 2
    resumed = subspace_corrections(cases(), execution={"cpus": 1, "checkpoint_dir": str(tmp_path)})["one"]
    assert resumed.quantum.diagnostics["checkpoint_reused"]
    assert resumed.rpmd.diagnostics["checkpoint_reused"]
    np.testing.assert_array_equal(resumed.rpmd.kappa, parallel.rpmd.kappa)
    assert resumed.factor == parallel.factor


def test_completed_quantum_task_survives_rpmd_failure(tmp_path, monkeypatch):
    real = execution._run_task

    def fail_rpmd(task, *args):
        if task["kind"] == "rpmd":
            raise RuntimeError("interrupted RPMD task")
        return real(task, *args)

    settings = dict(backend="serial", checkpoint_dir=str(tmp_path))
    monkeypatch.setattr(execution, "_run_task", fail_rpmd)
    with pytest.raises(RuntimeError, match="interrupted"):
        subspace_corrections(cases(), execution=settings)
    assert len(list(tmp_path.glob("*.json"))) == 1
    monkeypatch.setattr(execution, "_run_task", real)
    resumed = subspace_corrections(cases(), execution=settings)["one"]
    assert resumed.quantum.diagnostics["checkpoint_reused"]
    assert not resumed.rpmd.diagnostics["checkpoint_reused"]
    changed = cases()
    changed["one"]["rpmd_options"]["seed"] += 1
    next_result = subspace_corrections(changed, execution=settings)["one"]
    assert next_result.quantum.diagnostics["checkpoint_reused"]
    assert not next_result.rpmd.diagnostics["checkpoint_reused"]


def test_admission_rejects_over_budget_before_computing(monkeypatch):
    monkeypatch.setattr(execution, "_run_task", lambda *args: pytest.fail("started over-budget task"))
    with pytest.raises(MemoryError, match="budget"):
        subspace_corrections(cases(), execution={"memory_limit_mb": 1})


def test_invalid_option_cannot_alias_a_cached_none_default(tmp_path):
    inputs = cases()
    inputs["one"]["quantum_options"] = {"cap_strength": None}
    settings = dict(backend="serial", checkpoint_dir=str(tmp_path))
    subspace_corrections(inputs, execution=settings)
    inputs["one"]["quantum_options"]["cap_strength"] = float("nan")
    with pytest.raises(ValueError, match="JSON"):
        subspace_corrections(inputs, execution=settings)


def test_memory_budget_serializes_tasks_with_multiple_workers():
    result = subspace_corrections(cases(), execution={"cpus": 2, "workers": 2, "memory_limit_mb": 70})["one"]
    q, r = (s.diagnostics["execution"] for s in (result.quantum, result.rpmd))
    assert q["end_time_ns"] <= r["start_time_ns"]


def pipeline_fixture(monkeypatch, directory):
    from types import SimpleNamespace
    from pyrpmd.analysis import subspace_pipeline as pipeline

    one = cases()["one"]
    m1 = one["model"]
    y = np.linspace(-2, 2, 9)
    m2 = SubspaceModel((m1.axes[0], y), m1.values[:, None] + .2 * y[None, :]**2,
                       [1., 1.], 0., reactant_reference=-3.)
    run = SimpleNamespace(root=directory, beta=2., nbeads=4, xi_star=1., provider=SimpleNamespace())
    monkeypatch.setattr(pipeline, "pipeline_identity", lambda *args: "fixture-identity")
    mean = SimpleNamespace(curvature_width=lambda *args, **kwargs: {"maximum": 0.})
    monkeypatch.setattr(pipeline, "load_rate_run", lambda *args: run)
    monkeypatch.setattr(pipeline, "build_path", lambda *args: (mean, None, np.zeros((4, 2)), {}))
    monkeypatch.setattr(pipeline, "_static_curvature_width", lambda *args: {"maximum": 0., "candidate_direction": [0., 1.]})
    monkeypatch.setattr(pipeline, "path_model", lambda *args, **kwargs: (m1, {}, {"dimension": 1}))
    monkeypatch.setattr(pipeline, "plane_model", lambda *args: (m2, {}, {"dimension": 2}, None))
    monkeypatch.setattr(pipeline, "rph_model", lambda *args: (m2, {}, {"dimension": 2}, None))
    monkeypatch.setattr(pipeline, "correct_rate_run", lambda *args, **kwargs: {})
    return dict(source={"rate_run": str(directory)}, subspace={"dimension": [1, 2]},
                rpmd=one["rpmd_options"], output={"directory": str(directory)})


def test_pipeline_keeps_dimension_records_and_uses_task_cache(tmp_path, monkeypatch):
    from pyrpmd.analysis import run_subspace_pipeline

    spec = pipeline_fixture(monkeypatch, tmp_path)
    spec["execution"] = dict(backend="process", cpus=2, workers=2)
    report = run_subspace_pipeline(spec)
    assert report["dimensions_computed"] == [1, 2]
    assert set(report["corrections"]) == {"1", "2"}
    assert (tmp_path / "C_S_1d.json").is_file() and (tmp_path / "C_S_2d.json").is_file()
    assert len(list((tmp_path / "tasks").glob("*.json"))) == 4
    resumed = run_subspace_pipeline(spec)
    assert all(r[k]["diagnostics"]["checkpoint_reused"] for r in resumed["corrections"].values()
               for k in ("quantum", "rpmd"))


@pytest.mark.parametrize("mode", ["normal", "failure", "preparation_failure", "pipeline"])
def test_mpi_tasks_complete_or_fail_collectively_and_resume(tmp_path, mode):
    pytest.importorskip("mpi4py.MPI")
    launcher = shutil.which("mpiexec")
    if not launcher:
        pytest.skip("MPI launcher unavailable")
    worker = Path(__file__).with_name("subspace_mpi_worker.py")
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    subprocess.run([launcher, "-n", "2", sys.executable, str(worker), str(tmp_path), mode],
                   env=env, check=True, capture_output=True, text=True, timeout=90)
    report = json.loads((tmp_path / "mpi.json").read_text())
    assert report["ranks"] == 2
    if mode == "pipeline":
        assert report["dimensions"] == [1, 2]
        assert report["root_wrote_report"]
    elif mode == "normal":
        assert report["factor"] > 0
        assert report["all_reused"]
        assert len(set(report["pids"])) == 2
    else:
        assert report["collective_failure"]


def test_execution_cpus_auto_and_invalid_values(monkeypatch):
    monkeypatch.setattr(execution, "_affinity_cpus", lambda: 6)
    assert execution._options({"cpus": "auto"})["cpus"] == 6
    assert execution._options({})["cpus"] == 1
    for bad in ("all", 0, -2, True):
        with pytest.raises(ValueError, match="cpus"):
            execution._options({"cpus": bad})


def test_dynamic_scheduler_avoids_quarter_cpu_cap_on_mixed_1d_2d_tasks(monkeypatch):
    from concurrent.futures import Future

    x1 = np.linspace(-5.0, 5.0, 65)
    m1 = SubspaceModel((x1,), 0.3 / np.cosh(x1) ** 2, [1.0], 0.0, reactant_reference=-3.0)
    x2 = np.linspace(-6.0, 6.0, 81)
    y2 = np.linspace(-3.0, 3.0, 45)  # 3645 grid points -> realistic 2D Sylvester task
    X2, Y2 = np.meshgrid(x2, y2, indexing="ij")
    m2 = SubspaceModel((x2, y2), 0.015 / np.cosh(X2) ** 2 + 0.02 * Y2**2,
                       [1.0, 1.0], 0.0, reactant_reference=-4.0)
    mixed = {
        "1d": dict(model=m1, beta=1500.0, nbeads=64,
                   rpmd_options=dict(windows=24, walkers=16, burn_in=20, sweeps=40, trajectories=128, seed=1)),
        "2d": dict(model=m2, beta=1500.0, nbeads=128,
                   quantum_options=dict(method="sylvester", dense_limit=5000),
                   rpmd_options=dict(windows=48, walkers=32, burn_in=200, sweeps=400, trajectories=512, seed=1)),
    }
    monkeypatch.setattr(execution, "_affinity_cpus", lambda: 8)
    monkeypatch.setattr(execution, "_native_openmp", lambda: True)
    opts = execution._options({"backend": "process", "cpus": "auto"})
    assert opts["cpus"] == 8

    def run_sim(tasks_list):
        for t in tasks_list:
            t["blas_threads"] = opts["blas_threads"]
        profiles = {t["key"]: execution._task_profile(t, opts["cpus"], opts["blas_threads"]) for t in tasks_list}
        events = []
        max_concurrent_cpus = 0
        in_flight = {}
        finish_at = {}
        sim_now = 0.0

        class FakePool:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def submit(self, fn, task, threads, mem):
                nonlocal max_concurrent_cpus
                fut = Future()
                dur = execution._est_wall(profiles[task["key"]], threads)
                in_flight[fut] = (task["key"], threads)
                finish_at[fut] = sim_now + dur
                max_concurrent_cpus = max(max_concurrent_cpus, sum(c for _, c in in_flight.values()))
                events.append((task["key"], threads))
                res = (
                    execution.QuantumRateResult(1500.0, 1.0, 1.0, 1.0, "mock")
                    if task["kind"] == "quantum"
                    else execution.RingPolymerRateResult(
                        1500.0, 1, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
                        np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(2), np.zeros(2),
                    )
                )
                fut.set_result(res)
                return fut

        def fake_wait(fs, return_when=None):
            nonlocal sim_now
            first = min(fs, key=lambda f: finish_at[f])
            sim_now = finish_at.pop(first)
            in_flight.pop(first, None)
            return {first}, set(fs) - {first}

        monkeypatch.setattr(execution.time, "perf_counter", lambda: sim_now)
        monkeypatch.setattr(execution, "ProcessPoolExecutor", FakePool)
        monkeypatch.setattr(execution, "wait", fake_wait)
        checkpoints = execution._Checkpoints(tasks_list, opts)
        execution._local(tasks_list, opts, checkpoints, {})
        return dict(events), max_concurrent_cpus

    all_tasks = execution._tasks(mixed)
    alloc, max_concurrent_cpus = run_sim(all_tasks)
    assert max_concurrent_cpus <= 8
    assert alloc[("1d", "quantum")] == 1
    assert alloc[("2d", "rpmd")] >= 6

    pair_only = [t for t in execution._tasks(mixed) if t["key"] in (("1d", "quantum"), ("2d", "rpmd"))]
    pair_alloc, _ = run_sim(pair_only)
    assert pair_alloc[("1d", "quantum")] == 1
    assert pair_alloc[("2d", "rpmd")] == 8

    # Platforms built without OpenMP (e.g. macOS AppleClang / Windows MSVC) cap single-process RPMD at 1 thread.
    monkeypatch.setattr(execution, "_native_openmp", lambda: False)
    no_omp_alloc, no_omp_max = run_sim(execution._tasks(mixed))
    assert no_omp_max <= 8
    assert no_omp_alloc[("2d", "rpmd")] == 1


def test_thread_count_determinism_and_cpus_auto_cli(tmp_path, monkeypatch):
    from pyrpmd.cli import main as cli_main

    x1 = np.linspace(-5.0, 5.0, 41)
    m1 = SubspaceModel((x1,), 0.3 / np.cosh(x1) ** 2, [1.0], 0.0, reactant_reference=-3.0)
    x2 = np.linspace(-5.0, 5.0, 29)
    y2 = np.linspace(-2.0, 2.0, 15)  # 435 points > 256
    X2, Y2 = np.meshgrid(x2, y2, indexing="ij")
    m2 = SubspaceModel((x2, y2), 0.3 / np.cosh(X2) ** 2 + 0.2 * Y2**2, [1.0, 1.0], 0.0, reactant_reference=-3.0)
    mixed = {
        "1d": dict(model=m1, beta=2.0, nbeads=4,
                   rpmd_options=dict(windows=7, walkers=4, burn_in=2, sweeps=4, leapfrog_steps=2, trajectories=16, md_time=1.0, md_dt=0.05, bootstrap=6, seed=11)),
        "2d": dict(model=m2, beta=2.0, nbeads=4,
                   rpmd_options=dict(windows=7, walkers=4, burn_in=2, sweeps=4, leapfrog_steps=2, trajectories=16, md_time=1.0, md_dt=0.05, bootstrap=6, seed=11)),
    }

    ckpt = tmp_path / "ckpt"
    serial = subspace_corrections(mixed, execution={"backend": "serial", "cpus": 1})
    proc2 = subspace_corrections(mixed, execution={"backend": "process", "cpus": 2})
    auto = subspace_corrections(mixed, execution={"backend": "process", "cpus": "auto", "checkpoint_dir": str(ckpt)})
    reused = subspace_corrections(mixed, execution={"backend": "serial", "cpus": 1, "checkpoint_dir": str(ckpt)})

    for dim in ("1d", "2d"):
        assert proc2[dim].factor == pytest.approx(serial[dim].factor, rel=1e-12)
        assert auto[dim].factor == pytest.approx(serial[dim].factor, rel=1e-12)
        assert auto[dim].log_standard_error == pytest.approx(serial[dim].log_standard_error, rel=1e-12)
        np.testing.assert_array_equal(auto[dim].rpmd.kappa, serial[dim].rpmd.kappa)
        np.testing.assert_array_equal(proc2[dim].rpmd.kappa, serial[dim].rpmd.kappa)
        assert reused[dim].quantum.diagnostics["checkpoint_reused"]
        assert reused[dim].rpmd.diagnostics["checkpoint_reused"]

    cli_dir = tmp_path / "cli_run"
    cli_dir.mkdir()
    pipeline_fixture(monkeypatch, cli_dir)
    toml = cli_dir / "correct.toml"
    toml.write_text(
        f"""
[source]
rate_run = "{cli_dir.as_posix()}"

[subspace]
dimension = [1, 2]

[rpmd]
windows = 7
walkers = 4
burn_in = 2
sweeps = 4
leapfrog_steps = 2
trajectories = 16
md_time = 1.0
md_dt = 0.05
bootstrap = 6
seed = 123

[execution]
backend = "process"
cpus = "auto"

[output]
directory = "{(cli_dir / "out").as_posix()}"
""",
        encoding="utf-8",
    )
    assert cli_main(["subspace-correct", str(toml)]) == 0
    report = json.loads((cli_dir / "out" / "subspace_pipeline.json").read_text(encoding="utf-8"))
    assert report["dimensions_computed"] == [1, 2]
