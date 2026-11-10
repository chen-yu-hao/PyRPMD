"""Real spawn transport, stage-cache integrity, and pipeline orchestration."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyrpmd.analysis.subspace_correction import subspace_correction, _matched_rpmd_model
from pyrpmd.analysis.subspace_execution import (execute_rate_tasks, execution_options,
                                               freeze_quantum_options, snapshot_model)
from pyrpmd.analysis.subspace_model import SubspaceModel, model_from_function


def model():
    x = np.linspace(-4., 4., 65)
    return SubspaceModel((x,), .2 / np.cosh(x)**2, [1.], 0., "asymptotic", -3., {})


def request():
    return {"model": model(), "beta": 3., "nbeads": 2,
            "rpmd_options": {"windows": 6, "walkers": 3, "burn_in": 2, "sweeps": 5,
                             "trajectories": 8, "md_time": 1., "md_dt": .05, "bootstrap": 8, "seed": 19}}


@pytest.mark.parametrize("scheduling", ["static", "dynamic"])
def test_spawn_results_match_serial(scheduling):
    r = request()
    expected = execute_rate_tasks([r], {"backend": "serial"})[0]
    actual = execute_rate_tasks([r], {"backend": "process", "workers": 2, "scheduling": scheduling})[0]
    assert actual[0].rate == expected[0].rate
    assert actual[1].rate == expected[1].rate
    np.testing.assert_array_equal(actual[1].kappa, expected[1].kappa)


def test_process_correction_api():
    r = request()
    a = subspace_correction(r["model"], r["beta"], r["nbeads"], rpmd_options=r["rpmd_options"])
    b = subspace_correction(r["model"], r["beta"], r["nbeads"], rpmd_options=r["rpmd_options"],
                            execution={"backend": "process", "workers": 2})
    assert a.factor == b.factor
    assert a.log_standard_error == b.log_standard_error


def test_spawn_failure_propagates():
    r = request(); r["quantum_options"] = {"method": "invalid"}
    with pytest.raises(ValueError, match="method"):
        execute_rate_tasks([r], {"backend": "process", "workers": 2})


def test_completed_cache_keeps_full_time_series_and_detects_tampering(tmp_path, monkeypatch):
    from pyrpmd.analysis import subspace_quantum
    opts = {"cache_directory": str(tmp_path), "reuse": True}
    a = execute_rate_tasks([request()], opts)[0]
    assert not a[0].diagnostics["stage_cache_hit"]
    original = subspace_quantum.exact_quantum_rate
    def forbidden(*args, **kwargs): raise AssertionError("valid completed cache was recomputed")
    monkeypatch.setattr(subspace_quantum, "exact_quantum_rate", forbidden)
    b = execute_rate_tasks([request()], opts)[0]
    assert b[0].diagnostics["stage_cache_hit"] and b[1].diagnostics["stage_cache_hit"]
    np.testing.assert_array_equal(a[1].time, b[1].time)
    np.testing.assert_array_equal(a[1].kappa, b[1].kappa)
    monkeypatch.setattr(subspace_quantum, "exact_quantum_rate", original)
    target = next(tmp_path.glob("quantum-*.npz")); target.write_bytes(b"corrupt")
    c = execute_rate_tasks([request()], opts)[0]
    assert not c[0].diagnostics["stage_cache_hit"] and c[1].diagnostics["stage_cache_hit"]
    assert c[0].rate == a[0].rate


@pytest.mark.parametrize("change", ["beta", "seed", "values", "mass", "reference"])
def test_cache_identity_covers_scientific_settings(tmp_path, change):
    opts = {"cache_directory": str(tmp_path), "reuse": True}
    execute_rate_tasks([request()], opts)
    r = request()
    if change == "beta": r["beta"] = 3.1
    elif change == "seed": r["rpmd_options"]["seed"] += 1
    elif change == "values": r["model"] = replace(r["model"], values=r["model"].values * 1.01)
    elif change == "mass": r["model"] = replace(r["model"], masses=np.array([1.1]))
    elif change == "reference": r["model"] = replace(r["model"], reactant_reference=-2.9)
    result = execute_rate_tasks([r], opts)[0]
    assert not result[1].diagnostics["stage_cache_hit"]
    assert result[0].diagnostics["stage_cache_hit"] == (change == "seed")


def test_memory_guard_and_thread_limit_restore():
    from threadpoolctl import threadpool_info
    before = [(p["filepath"], p["num_threads"]) for p in threadpool_info() if p["user_api"] == "blas"]
    with pytest.raises(MemoryError):
        execute_rate_tasks([request()], {"memory_budget_mb": .001})
    after = [(p["filepath"], p["num_threads"]) for p in threadpool_info() if p["user_api"] == "blas"]
    assert before == after


def test_freeze_callable_options_is_exact():
    m = model(); fn = lambda p: .1 * np.maximum(abs(p[..., 0]) - 2.5, 0)**2
    opts = freeze_quantum_options(m, {"absorber": fn, "truncation_energy": lambda p: np.ones(len(p))})
    np.testing.assert_array_equal(opts["absorber"], fn(m.grid_points().reshape(-1, 1)))
    np.testing.assert_array_equal(opts["truncation_energy"], np.ones(m.shape))


def test_callback_models_rejected_before_parallel_launch():
    r = request(); r["model"] = model_from_function(lambda q: q[..., 0]**2, [np.linspace(-4, 4, 20)])
    with pytest.raises(ValueError, match="static-grid"):
        execute_rate_tasks([r], {"backend": "process", "workers": 2})


@pytest.mark.parametrize("options", [{"workers": True}, {"backend": "unknown"}, {"workers": 2},
                                      {"typo": 1}, {"memory_budget_mb": -1}, {"reuse": "yes"}])
def test_execution_options_reject_invalid_controls(options):
    with pytest.raises(ValueError): execution_options(options)


def test_mpi_requires_an_actual_launcher():
    with pytest.raises(ValueError, match="MPI launch"):
        execute_rate_tasks([request()], {"backend": "mpi"})


@pytest.mark.parametrize("changed", ["mass", "normalization", "values"])
def test_model_matching_fails_before_computation(changed):
    a = model()
    b = replace(a, masses=[2.]) if changed == "mass" else (
        replace(a, reactant="well") if changed == "normalization" else replace(a, values=a.values * 2.))
    with pytest.raises(ValueError): _matched_rpmd_model(a, b)


def test_fine_and_coarse_models_share_exact_grid_values():
    fine = model()
    coarse = replace(fine, axes=(fine.axes[0][::2],), values=fine.values[::2])
    assert _matched_rpmd_model(coarse, fine) is fine
    shifted = replace(coarse, axes=(coarse.axes[0] + .001,))
    with pytest.raises(ValueError, match="subset"):
        _matched_rpmd_model(shifted, fine)


def test_pipeline_numeric_tasks_and_reuse(tmp_path, monkeypatch):
    from pyrpmd.analysis import subspace_pipeline as pipeline
    closed = []
    fake = SimpleNamespace(root=tmp_path, beta=3., nbeads=2, xi_star=.4,
                           provider=SimpleNamespace(close=lambda: closed.append(True)))
    monkeypatch.setattr(pipeline, "load_rate_run", lambda directory: fake)
    mean = SimpleNamespace(curvature_width=lambda *a, **kw: {"maximum": 0.})
    monkeypatch.setattr(pipeline, "build_path", lambda *a: (mean, np.ones(4, bool), np.zeros((4, 1)), {}))
    monkeypatch.setattr(pipeline, "_static_curvature_width", lambda *a: {"maximum": 0.})
    monkeypatch.setattr(pipeline, "path_model", lambda *a: (model(), {}, {"grid_points": 65}))
    monkeypatch.setattr(pipeline, "correct_rate_run", lambda root, correction, output: {"task": "rate", "factor": correction.factor})
    spec = {"source": {"rate_run": str(tmp_path)}, "rpmd": request()["rpmd_options"],
            "output": {"reuse": True}, "subspace": {"dimension": 1}}
    a = pipeline.run_subspace_pipeline(spec)
    b = pipeline.run_subspace_pipeline(spec)
    assert a["corrections"]["1"]["factor"] == b["corrections"]["1"]["factor"]
    assert b["corrections"]["1"]["quantum"]["diagnostics"]["stage_cache_hit"]
    assert not b["physical_converged"] and len(closed) == 2
    assert (tmp_path / "subspace" / "C_S_1d.json").is_file()


def test_projection_failure_closes_provider(tmp_path, monkeypatch):
    from pyrpmd.analysis import subspace_pipeline as pipeline
    closed = []
    fake = SimpleNamespace(provider=SimpleNamespace(close=lambda: closed.append(True)))
    monkeypatch.setattr(pipeline, "load_rate_run", lambda directory: fake)
    def fail(*args): raise ValueError("projection failed")
    monkeypatch.setattr(pipeline, "_prepare_models", fail)
    with pytest.raises(ValueError, match="projection failed"):
        pipeline._prepare_pipeline({"source": {"rate_run": str(tmp_path)}}, None)
    assert closed == [True]
