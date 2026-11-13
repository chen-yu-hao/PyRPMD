"""Subspace box planning and exact-solver capacity (issue #195).

A thermoneutral reaction whose barrier is lower than ``product_drop_kt``
(H + H2 above 322 K) used to stretch the 2D box to the end of the path
continuation, and a grid too large for the dense route then failed only
inside the solver, with a generic ``ValueError`` after all sampling.
"""

import json

import numpy as np
import pytest

from pyrpmd.analysis.subspace_model import SubspaceModel
from pyrpmd.analysis.subspace_pipeline import _path_box
from pyrpmd.analysis.subspace_quantum import (DenseMemoryError, SubspaceCapacityError, _GridOperators,
                                              exact_quantum_rate, quantum_route_plan)

from test_subspace_validity import CONVERGED, pipeline_spec

KB = 3.166811563e-6  # Eh / K
BARRIER = 9.6133 / 627.5095  # Eh, BKMP2 H + H2


# --- P0: the product absorber and box along the path -------------------------

def symmetric_profile(continuation, step=2.0, width=40.0):
    """Thermoneutral Eckart-like barrier along a path continued `continuation` beyond the TS."""
    arc = np.arange(-continuation, continuation + step / 2, step)
    return arc + continuation, BARRIER / np.cosh(arc / width) ** 2, int(np.argmin(abs(arc)))


@pytest.mark.parametrize("temperature", [250.0, 300.0, 350.0, 400.0])
def test_box_does_not_depend_on_how_far_the_path_was_continued(temperature):
    beta = 1.0 / (KB * temperature)
    boxes = []
    for continuation in (600.0, 1200.0, 4000.0):
        arc, energies, k_ts = symmetric_profile(continuation)
        box = _path_box(arc, energies, k_ts, beta, {})
        boxes.append(np.subtract(box["box_arc"], arc[k_ts]))
        assert box["product_absorber_criterion"] != "path_end"
    np.testing.assert_allclose(boxes[1:], [boxes[0]] * 2, atol=1e-9)


def test_product_absorber_is_continuous_through_the_barrier_over_15_kt():
    crossing = BARRIER / (15.0 * KB)  # 322.5 K for BKMP2
    arc, energies, k_ts = symmetric_profile(4000.0)
    below = _path_box(arc, energies, k_ts, 1.0 / (KB * (crossing - 1.0)), {})
    above = _path_box(arc, energies, k_ts, 1.0 / (KB * (crossing + 1.0)), {})
    # The old rule placed the absorber at the path end (4000 beyond the TS) above the crossing.
    assert above["product_absorber_criterion"] == "reference_kt"
    assert abs(above["absorber_arc"][1] - below["absorber_arc"][1]) < 20.0
    assert above["absorber_arc"][1] - arc[k_ts] < 400.0


def test_box_is_unchanged_where_the_drop_criterion_comes_first():
    """Exothermic, or a barrier well above 15 kT: the validated boxes stay where they were."""
    arc, energies, k_ts = symmetric_profile(1500.0)
    beta = 1.0 / (KB * 200.0)
    box = _path_box(arc, energies, k_ts, beta, {})
    old = int(np.flatnonzero((np.arange(len(arc)) > k_ts) & (energies < energies[k_ts:].max() - 15.0 / beta))[0])
    assert box["product_absorber_criterion"] == "product_drop_kt" and box["product_index"] == old
    exothermic = np.where(np.arange(len(arc)) > k_ts, energies - 0.05 * (1 - energies / BARRIER), energies)
    box = _path_box(arc, exothermic, k_ts, 1.0 / (KB * 400.0), {})
    assert box["product_absorber_criterion"] == "product_drop_kt"


# --- P2: a structured, solver-side capacity plan -----------------------------

def well(n=24):
    """A 2D barrier between two harmonic channels; about half the grid lies below 0.4 Eh."""
    x = np.linspace(-6.0, 6.0, n)
    y = np.linspace(-3.0, 3.0, n)
    values = 0.02 / np.cosh(x)[:, None] ** 2 + 0.05 * y[None, :] ** 2
    return SubspaceModel((x, y), values, [1.0, 1.0], 0.0, "asymptotic", -4.0)


DENSE = dict(truncation_energy=0.2, energy_floor=0.0, normalization="slice")


def no_dense_matrix(monkeypatch):
    monkeypatch.setattr(_GridOperators, "dense_hamiltonian", lambda *args: pytest.fail("built a dense DVR"))


def test_dense_only_options_over_the_limit_give_a_structured_diagnostic(monkeypatch):
    model = well()
    plan = quantum_route_plan(model, dense_limit=50, **DENSE)
    retained = int(np.count_nonzero(model.values <= 0.2))
    assert not plan["feasible"] and plan["code"] == "dense_limit_exceeded" and plan["route"] == "sylvester"
    assert plan["retained_points"] == retained and plan["dense_limit"] == 50
    assert plan["required_dense_limit"] == retained and plan["estimated_dense_memory_mb"] > 64
    assert set(plan["dense_only_features"]) == {"truncation_energy", "energy_floor", "slice normalization"}
    assert any("dense_limit" in alternative for alternative in plan["alternatives"])
    json.dumps(plan, allow_nan=False)
    no_dense_matrix(monkeypatch)
    with pytest.raises(SubspaceCapacityError) as caught:
        exact_quantum_rate(model, 40.0, dense_limit=50, **DENSE)
    assert isinstance(caught.value, ValueError) and caught.value.code == "dense_limit_exceeded"
    assert caught.value.diagnostic == plan
    assert "need the dense route" not in str(caught.value) and str(retained) in str(caught.value)


def test_dense_only_options_cannot_take_the_iterative_route(monkeypatch):
    no_dense_matrix(monkeypatch)
    with pytest.raises(SubspaceCapacityError) as caught:
        exact_quantum_rate(well(), 40.0, method="flux_eigenstates", energy_floor=0.0)
    assert caught.value.code == "dense_route_required"
    assert caught.value.diagnostic["dense_only_features"] == ["energy_floor"]


def test_memory_budget_is_a_capacity_error_and_still_a_memory_error(monkeypatch):
    no_dense_matrix(monkeypatch)
    with pytest.raises(DenseMemoryError, match="budget") as caught:
        exact_quantum_rate(well(), 40.0, memory_limit_mb=1.0, **DENSE)
    assert isinstance(caught.value, MemoryError) and isinstance(caught.value, SubspaceCapacityError)
    assert caught.value.code == "dense_memory_exceeded"
    assert caught.value.diagnostic["memory_limit_mb"] == 1.0


def test_plain_grids_still_switch_to_the_iterative_route():
    plan = quantum_route_plan(well(), dense_limit=50)
    assert plan["feasible"] and plan["route"] == "flux_eigenstates" and plan["required_dense_limit"] is None


def test_plan_is_the_route_and_memory_the_solver_uses():
    model = well(16)
    plan = quantum_route_plan(model, dense_limit=4000, **DENSE)
    result = exact_quantum_rate(model, 40.0, dense_limit=4000, **DENSE)
    assert plan["feasible"] and result.method == plan["route"] == "sylvester"
    assert result.diagnostics["estimated_peak_memory_mb"] == plan["estimated_dense_memory_mb"]


# --- pipeline: rejected before solving, with the plan in the report -----------

def oversized_plane():
    return well(), {"dense_limit": 50, **DENSE}


@pytest.mark.parametrize("on_invalid", ["stop", "diagnose"])
def test_pipeline_rejects_an_oversized_model_before_solving(tmp_path, monkeypatch, on_invalid):
    from pyrpmd.analysis import run_subspace_pipeline
    from pyrpmd.analysis import subspace_pipeline as pipeline
    from pyrpmd.analysis.subspace_correction import CORRECTED_SCHEMA

    stale = tmp_path / "rate_quantum_corrected.json"
    stale.write_text(json.dumps({"schema": CORRECTED_SCHEMA}))
    spec, _ = pipeline_spec(monkeypatch, tmp_path, termination=CONVERGED, dimension=2,
                            validity={"on_invalid": on_invalid})
    model, quantum = oversized_plane()
    monkeypatch.setattr(pipeline, "plane_model",
                        lambda run, path, direction, options: (model, quantum, {"dimension": 2}, None))
    monkeypatch.setattr(pipeline, "subspace_correction", lambda **case: pytest.fail("solved an oversized model"))
    report = run_subspace_pipeline(spec)
    assert report["status"] == "capacity_exceeded" and not report["validity"]["valid"]
    assert report["applied_dimension"] is None and report["corrected_rate"] is None
    assert report["dimensions_computed"] == [] and report["validity"]["rejected_before_solving"] == [2]
    capacity = report["validity"]["capacity_exceeded"]["2"]
    assert capacity["code"] == "dense_limit_exceeded" and capacity["retained_points"] > 50
    assert report["models"]["2"]["quantum_plan"] == capacity
    gate = report["validity"]["dimensions"]["2"]["gates"][-1]
    assert gate["gate"] == "capacity" and not gate["passed"] and not gate["overridden"]
    assert report["validity"]["failures"]["2"][-1].startswith("dense_limit_exceeded:")
    assert not stale.exists() and not (tmp_path / "C_S_2d.json").exists()


def test_pipeline_reports_the_plan_of_a_model_it_solves(tmp_path, monkeypatch):
    from pyrpmd.analysis import run_subspace_pipeline

    spec, _ = pipeline_spec(monkeypatch, tmp_path, termination=CONVERGED)
    report = run_subspace_pipeline(spec)
    assert report["status"] == "complete"
    plan = report["models"]["1"]["quantum_plan"]
    assert plan["feasible"] and plan["route"] == report["corrections"]["1"]["quantum"]["method"]
    assert report["validity"]["capacity_exceeded"] == {}
