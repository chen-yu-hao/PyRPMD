"""Analytic averaging and accounting tests; no native preparation is run."""
import json
import math
import pickle

import numpy as np
import pytest

from benchmarks.h3 import preparation_audit as audit


ENERGY = "primitive_internal_per_bead_hartree"


def test_trapezoidal_mean_uses_time_weights_and_interpolates_window_boundaries():
    # Nonuniform observations make an unweighted sample mean incorrect.
    times = [0.0, 0.1, 0.3, 1.0, 2.0]
    trace = [{"time_fs": t, ENERGY: 7-2*t} for t in times]
    result = audit.time_window_average(trace, 0.2, 1.4)
    assert result["values"][ENERGY] == pytest.approx(5.4, abs=1e-14)
    assert result["duration_fs"] == pytest.approx(1.2)
    assert result["dense_points"] == 2
    assert result["integration_nodes"] == 4
    assert result["interpolated_boundaries"] == {"start": True, "end": True}
    assert result["values"][ENERGY] != pytest.approx(np.mean([row[ENERGY] for row in trace]))
    with pytest.raises(ValueError, match="beyond"):
        audit.time_window_average(trace, -0.1, 1.0)
    trace[2][ENERGY] = None
    with pytest.raises(ValueError, match="Missing"):
        audit.time_window_average(trace, 0.2, 1.4)


def test_sparse_trace_preserves_off_grid_exact_endpoints_and_original_values():
    times = sorted(set(np.round(np.arange(0, 11.5, 0.2), 10)) | {3.3})
    trace = [{"time_fs": float(t), ENERGY: math.sin(t)} for t in times]
    compact, metadata = audit.sparse_trace(trace, 3.3)
    restored = [dict(zip(compact["columns"], row)) for row in compact["records"]]
    assert [row["time_fs"] for row in restored] == [0, 3.3, 5, 10, 11.4]
    assert all(row in trace for row in restored)
    assert compact["units"][ENERGY] == "hartree"
    assert metadata["dense_records"] == len(trace)
    assert metadata["sparse_records"] == 5
    assert metadata["mandatory_endpoints_fs"] == [0, 3.3, 11.4]
    assert metadata["maximum_sparse_grid_offset_fs"] == 0
    with pytest.raises(ValueError, match="lacks required endpoint"):
        audit.sparse_trace(trace, 3.31)


def test_hold_boundary_roundoff_does_not_include_undefined_ramp_drift():
    field = "hold_energy_drift_per_bead_hartree"
    start = float(np.nextafter(50.0, np.inf))
    trace = [{"time_fs": 49.8, field: None}, {"time_fs": start, field: 0.0},
             {"time_fs": 50.2, field: 0.4}, {"time_fs": 50.4, field: 0.8}]
    mean = audit.time_window_average(trace, 50.0, 50.4)
    assert mean["start_fs"] == start
    assert mean["values"][field] == pytest.approx(0.4, abs=1e-13)
    assert mean["interpolated_boundaries"] == {"start": False, "end": False}


def test_native_hold_primary_and_trace_final_window_are_separate():
    times = np.arange(0, 150.01, 0.2)
    trace = [{"time_fs": float(t), ENERGY: float(2+t)} for t in times]
    row = {"id": 1, "status": "prepared", "preparation_study": {
        "trace": trace,
        "switch": {"status": "completed", "switch_time_fs": 50.0,
                   "hold_time_fs": 100.0, "accepted_hold_duration_fs": 100.0,
                   "dt_fs": 0.02, "trace_stride_fs": 0.2,
                   "hold_time_average": {ENERGY: 102.001}},
    }}
    reduced = audit.reduce_audit_row(row)["preparation_study"]
    windows = reduced["time_averages"]
    assert windows["hold"]["source"] == "native_every_step_trapezoid"
    assert windows["hold"]["values"][ENERGY] == 102.001
    assert windows["hold"]["native_steps"] == 5000
    assert windows["hold_trace_check"]["values"][ENERGY] == pytest.approx(102.0)
    assert windows["final_50fs"]["values"][ENERGY] == pytest.approx(127.0)
    assert windows["final_50fs"]["start_fs"] == 100.0
    assert len(reduced["trace"]["records"]) == 31
    assert reduced["trace_sampling"]["dense_records"] == 751
    assert reduced["trace_sampling"]["averaging_before_thinning"] is True
    json.dumps(row, allow_nan=False)


def _mean_row(identity, mean, dense_points):
    return {"id": identity, "status": "prepared", "preparation_study": {
        "time_averages": {
            "hold": {"source": "native_every_step_trapezoid", "start_fs": 0.,
                     "end_fs": 100., "duration_fs": 100., "values": {ENERGY: mean}},
            "hold_trace_check": {"source": "dense_trace_trapezoid", "start_fs": 0.,
                     "end_fs": 100., "duration_fs": 100., "dense_points": dense_points,
                     "values": {ENERGY: mean+0.125}},
        }}}


def test_sem_is_across_trajectory_means_with_equal_weight_and_failure_denominator():
    rows = [_mean_row(1, 1., 2), _mean_row(2, 3., 1001), _mean_row(3, 5., 2001),
            {"id": 4, "status": "failed", "reason": "preparation failed"}]
    summary = audit.summarize_time_averages(iter(rows))
    stats = summary["windows"]["hold"]["values"][ENERGY]
    assert (summary["attempts"], summary["prepared"], summary["failed"]) == (4, 3, 1)
    assert stats["n"] == 3
    assert stats["mean"] == 3.0
    assert stats["sem"] == pytest.approx(2/math.sqrt(3))
    assert summary["native_minus_trace_hold"][ENERGY]["mean"] == -0.125
    single = audit.summarize_time_averages(rows[:1])
    assert single["windows"]["hold"]["values"][ENERGY]["sem"] is None
    json.dumps(single, allow_nan=False)


def test_worker_reduction_failure_is_explicit_and_removes_dense_payload(monkeypatch):
    original = {"id": 7, "status": "prepared", "preparation_study": {"trace": []}}
    monkeypatch.setattr(audit, "_ORIGINAL_AUDIT_ONE", lambda identity: original)
    result = audit.audit_one(7)
    assert result["status"] == "failed"
    assert result["preparation_status"] == "prepared"
    assert result["failure_stage"] == "audit_reduction"
    assert "trace" not in result["preparation_study"]


def test_parent_binding_restores_on_exception_and_worker_is_importable():
    previous = audit.study._audit_one
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        with audit.audit_callable_binding():
            assert audit.study._audit_one is audit.audit_one
            assert pickle.loads(pickle.dumps(audit.audit_one)) is audit.audit_one
            raise RuntimeError("synthetic interruption")
    assert audit.study._audit_one is previous
    cases = audit.planned_cases()
    assert len(cases) == 12
    assert sum(case["config"]["samples"] for case in cases) == 6144
    assert all(case["config"]["id_offset"] == 910000 for case in cases)
