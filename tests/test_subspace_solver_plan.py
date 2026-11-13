"""Solver capacity, temperature-robust grids and failure isolation (issue #195).

A P = 256 H + H2 rate run at 350/400 K ended its subspace stage in a generic
``ValueError`` from the quantum solver after the full PMF/recrossing run: the
2D box had jumped to the end of the reaction path once ``product_drop_kt`` kT
exceeded the barrier, the retained grid exceeded ``dense_limit``, and the
automatic route chose the iterative solver, which cannot honour the energy
truncation, energy floor or slice normalization of the 2D model.
"""

import json

import numpy as np
import pytest

from pyrpmd.analysis import DenseRouteUnavailable, exact_quantum_rate, quantum_solver_plan
from pyrpmd.analysis.subspace_correction import validity_gates
from pyrpmd.analysis.subspace_model import SubspaceModel
from pyrpmd.analysis.subspace_pipeline import _product_absorber_start
from pyrpmd.analysis.subspace_quantum import _GridOperators
import pyrpmd.analysis.subspace_execution as execution

from test_subspace_validity import CONVERGED, barrier, pipeline_spec

KB = 3.166811563e-6  # hartree per kelvin


def plane(nx=81, ny=15):
    x, y = np.linspace(-8.0, 8.0, nx), np.linspace(-2.0, 2.0, ny)
    values = 1.0 / np.cosh(x)[:, None] ** 2 + 0.5 * y[None, :] ** 2
    return SubspaceModel((x, y), values, [1.0, 1.0], 0.0, reactant_reference=-5.0)


TWO_D = {"truncation_energy": 1.5, "energy_floor": 0.0, "normalization": "slice"}


# --- quantum solver plan ------------------------------------------------------

def test_dense_only_options_beyond_dense_limit_fail_before_any_solve(monkeypatch):
    model = plane()
    retained = int(np.count_nonzero(model.values <= TWO_D["truncation_energy"]))
    plan = quantum_solver_plan(model, dense_limit=retained - 1, **TWO_D)
    assert not plan["feasible"] and plan["status"] == "dense_limit_exceeded"
    assert plan["route"] == "flux_eigenstates" and plan["retained_points"] == retained
    assert plan["required_dense_limit"] == retained and plan["grid_points"] == model.values.size
    assert plan["dense_only_options"] == ["truncation_energy", "energy_floor", "slice normalization"]
    assert plan["dense_memory_mb"] > 0 and any("dense_limit" in a for a in plan["alternatives"])

    monkeypatch.setattr(_GridOperators, "dense_hamiltonian", lambda *a, **k: pytest.fail("allocated dense DVR"))
    with pytest.raises(DenseRouteUnavailable, match="need the dense route") as raised:
        exact_quantum_rate(model, 2.0, dense_limit=retained - 1, **TWO_D)
    # Still a ValueError, now with the numbers needed to act on it.
    assert isinstance(raised.value, ValueError)
    assert raised.value.diagnostic["retained_points"] == retained
    assert raised.value.diagnostic["dense_limit"] == retained - 1
    assert str(retained) in str(raised.value)


def test_the_plan_is_the_route_the_solver_takes():
    model = plane(41, 9)
    plan = quantum_solver_plan(model, dense_limit=8000, **TWO_D)
    assert plan["feasible"] and plan["route"] == "sylvester"
    result = exact_quantum_rate(model, 2.0, dense_limit=8000, **TWO_D)
    assert result.method == "sylvester" and result.rate > 0
    recorded = result.diagnostics["solver_plan"]
    assert recorded["retained_points"] == plan["retained_points"]
    assert recorded["dense_memory_mb"] == pytest.approx(plan["dense_memory_mb"])
    # The scheduler budgets the same estimate.
    assert execution._memory_mb("quantum", model, 2.0, 4, dict(TWO_D, dense_limit=8000)) == \
        pytest.approx(plan["estimated_memory_mb"])
    # Without dense-only options the iterative route stays available.
    free = quantum_solver_plan(model, dense_limit=10)
    assert free["feasible"] and free["route"] == "flux_eigenstates" and free["required_dense_limit"] is None


def test_explicit_iterative_route_and_memory_budget_are_planned():
    model = plane(41, 9)
    conflict = quantum_solver_plan(model, method="flux_eigenstates", energy_floor=0.0)
    assert not conflict["feasible"] and conflict["status"] == "dense_route_required"
    with pytest.raises(DenseRouteUnavailable, match="energy_floor need the dense route"):
        exact_quantum_rate(model, 2.0, method="flux_eigenstates", energy_floor=0.0)
    budget = quantum_solver_plan(model, dense_limit=8000, memory_limit_mb=1.0, **TWO_D)
    assert not budget["feasible"] and budget["status"] == "memory_limit_exceeded"
    with pytest.raises(ValueError, match="dense_limit"):
        quantum_solver_plan(model, dense_limit=0)


# --- product absorber ---------------------------------------------------------

def thermoneutral(barrier_height=0.0153, length=600.0, width=60.0):
    """A symmetric H + H2-like profile along arc length (barrier ~ BKMP2)."""
    arc = np.linspace(-length, length, 1201)
    return arc, barrier_height / np.cosh(arc / width) ** 2, 600


def former_rule(energies, k_ts, beta, drop_kt):
    top = float(energies[k_ts:].max())
    hits = np.flatnonzero((np.arange(len(energies)) > k_ts) & (energies < top - drop_kt / beta))
    return int(hits[0]) if len(hits) else len(energies) - 1


def test_product_absorber_is_unchanged_while_the_channel_can_drop_15_kt():
    arc, energies, k_ts = thermoneutral()
    for temperature in (150.0, 200.0, 250.0, 300.0):
        beta = 1.0 / (KB * temperature)
        k_p, record = _product_absorber_start(energies, k_ts, beta, 15.0, 0.3)
        assert record["criterion"] == "product_drop"
        assert k_p == former_rule(energies, k_ts, beta, 15.0)


def test_product_absorber_does_not_jump_to_the_end_of_the_path_at_high_temperature():
    """Issue #195: above ~320 K the former rule put the absorber at the path end."""
    arc, energies, k_ts = thermoneutral()
    temperatures = np.arange(200.0, 601.0, 5.0)
    starts, criteria = [], []
    for temperature in temperatures:
        beta = 1.0 / (KB * temperature)
        k_p, record = _product_absorber_start(energies, k_ts, beta, 15.0, 0.3)
        starts.append(arc[k_p])
        criteria.append(record["criterion"])
        if record["criterion"] == "product_asymptote":
            assert record["energy_above_end"] <= record["asymptote_tolerance"]
            assert 15.0 / beta > record["available_drop"] - record["asymptote_tolerance"]
    assert former_rule(energies, k_ts, 1.0 / (KB * 400.0), 15.0) == len(arc) - 1
    assert "product_asymptote" in criteria and "path_end" not in criteria
    # Monotone in temperature until the asymptote criterion takes over, and
    # bounded far from the end of the path everywhere.
    assert max(starts) < 0.5 * arc[-1]
    assert np.max(np.abs(np.diff(starts))) < 0.1 * (arc[-1] - arc[k_ts])


def test_product_absorber_waits_until_a_product_well_has_settled():
    arc = np.linspace(-300.0, 300.0, 601)
    energies = 0.0153 / np.cosh(arc / 40.0) ** 2 - 0.004 * np.exp(-((arc - 120.0) / 25.0) ** 2)
    beta = 1.0 / (KB * 500.0)  # 15 kT exceeds the barrier
    k_p, record = _product_absorber_start(energies, 300, beta, 15.0, 0.3)
    assert record["criterion"] == "product_asymptote"
    assert np.all(np.abs(energies[k_p:] - energies[-1]) <= record["asymptote_tolerance"])
    assert arc[k_p] > 120.0


def test_a_channel_that_never_descends_keeps_the_whole_path():
    energies = np.concatenate([np.linspace(0.0, 0.01, 50), np.linspace(0.01, 0.02, 50)])
    k_p, record = _product_absorber_start(energies, 49, 1.0 / (KB * 300.0), 15.0, 0.3)
    assert record["criterion"] == "path_end" and k_p == len(energies) - 1


# --- grid gate ------------------------------------------------------------------

def test_outside_grid_reason_names_the_estimator_that_left_the_grid():
    exits = {"ti_sample_outside_fraction": 0.33, "surface_sample_outside_fraction": 0.02,
             "recrossing_contaminated_flux_fraction": 0.0, "estimator_outside_fraction": 0.33}
    gates = validity_gates({}, {"grid_exit": exits})
    assert not gates["valid"]
    reason = gates["failures"][0]
    assert "outside its grid" in reason and "ti 0.33" in reason and "dominant: ti" in reason
    assert gates["gates"][0]["detail"]["dominant_component"] == "ti"


# --- pipeline -------------------------------------------------------------------

def test_pipeline_refuses_an_unsolvable_dimension_before_solving_and_keeps_the_others(tmp_path, monkeypatch):
    from pyrpmd.analysis import run_subspace_pipeline
    from pyrpmd.analysis import subspace_pipeline as pipeline

    model2 = plane()
    retained = int(np.count_nonzero(model2.values <= TWO_D["truncation_energy"]))
    spec, built = pipeline_spec(monkeypatch, tmp_path, termination=CONVERGED, plane=model2, dimension=[1, 2],
                                plane_quantum=dict(TWO_D, dense_limit=retained - 1))
    real = pipeline.subspace_correction

    def solve(**case):
        assert case["model"] is not model2, "solved a model the solver plan refused"
        return real(**case)

    monkeypatch.setattr(pipeline, "subspace_correction", solve)
    report = run_subspace_pipeline(spec)
    assert report["status"] == "complete" and report["applied_dimension"] == 1
    assert report["dimensions_computed"] == [1]
    assert report["validity"]["refused_by_solver_plan"] == [2]
    assert report["validity"]["rejected_before_solving"] == [2]
    plan = report["solver_plans"]["2"]
    assert plan["status"] == "dense_limit_exceeded" and plan["retained_points"] == retained
    assert plan["dense_limit"] == retained - 1 and plan["dense_memory_mb"] > 0
    assert any(f"quantum.dense_limit = {retained}" in r for r in plan["recommendations"])
    assert report["solver_plans"]["1"]["feasible"]
    gate = report["validity"]["dimensions"]["2"]["gates"][-1]
    assert gate["gate"] == "solver" and not gate["passed"] and not gate["overridden"]


def test_pipeline_reports_solver_capacity_when_nothing_can_be_solved(tmp_path, monkeypatch):
    from pyrpmd.analysis import run_subspace_pipeline
    from pyrpmd.analysis import subspace_pipeline as pipeline

    spec, _ = pipeline_spec(monkeypatch, tmp_path, termination=CONVERGED, plane=plane(), dimension=2,
                            plane_quantum=dict(TWO_D, dense_limit=10), validity={"on_invalid": "diagnose"})
    monkeypatch.setattr(pipeline, "subspace_correction", lambda **case: pytest.fail("solved a refused model"))
    report = run_subspace_pipeline(spec)
    assert report["status"] == "solver_capacity_exceeded" and report["corrected_rate"] is None
    assert report["applied_dimension"] is None and report["dimensions_computed"] == []
    assert "solver capacity" in report["validity"]["failures"]["2"][0]
    saved = json.loads((tmp_path / "subspace_pipeline.json").read_text())
    assert saved["status"] == "solver_capacity_exceeded"
    assert saved["solver_plans"]["2"]["route"] == "flux_eigenstates"


def test_one_failed_dimension_does_not_discard_the_others(tmp_path, monkeypatch):
    from pyrpmd.analysis import run_subspace_pipeline
    from pyrpmd.analysis import subspace_pipeline as pipeline

    model2 = plane(41, 9)
    spec, _ = pipeline_spec(monkeypatch, tmp_path, termination=CONVERGED, plane=model2, dimension=[1, 2])
    real = pipeline.subspace_correction

    def solve(**case):
        if case["model"] is model2:
            raise FloatingPointError("non-finite 2D flux")
        return real(**case)

    monkeypatch.setattr(pipeline, "subspace_correction", solve)
    report = run_subspace_pipeline(spec)
    assert report["status"] == "complete" and report["applied_dimension"] == 1
    assert report["validity"]["failed_to_solve"] == [2]
    assert report["solve_failures"]["2"] == {"type": "FloatingPointError", "message": "non-finite 2D flux"}
    assert "solve failed" in report["validity"]["failures"]["2"][0]

    spec, _ = pipeline_spec(monkeypatch, tmp_path, termination=CONVERGED, plane=model2, dimension=2)
    failed = run_subspace_pipeline(spec)
    assert failed["status"] == "failed" and failed["corrected_rate"] is None
    assert failed["solve_failures"]["2"]["type"] == "FloatingPointError"


def test_the_native_stage_records_plans_and_raises_on_a_failed_explicit_stage(tmp_path, monkeypatch):
    from test_rate_subspace import saved_run
    from pyrpmd.workflows import rate_subspace as stage

    config, _, root = saved_run(tmp_path)
    monkeypatch.setattr(stage, "_check_centroid_path", lambda *a: None)
    report = {"status": "failed", "dimensions_computed": [], "applied_dimension": None, "corrections": {},
              "corrected_rate": None, "validity": {"valid": False}, "warnings": [],
              "solver_plans": {"2": {"feasible": True, "route": "sylvester"}},
              "solve_failures": {"2": {"type": "FloatingPointError", "message": "non-finite 2D flux"}}}
    monkeypatch.setattr("pyrpmd.analysis.subspace_pipeline.run_subspace_pipeline", lambda *a, **k: dict(report))
    with pytest.warns(RuntimeWarning, match="non-finite 2D flux"):
        status = stage.postprocess_rate_subspace(config, root)
    assert status["status"] == "failed" and status["solver_plans"]["2"]["route"] == "sylvester"
    with pytest.raises(RuntimeError, match="2D: FloatingPointError"):
        stage.postprocess_rate_subspace(config, root, strict=True)
    saved = json.loads((root / "subspace" / "status.json").read_text())
    assert saved["solve_failures"]["2"]["message"] == "non-finite 2D flux"

    report.update(status="solver_capacity_exceeded", solve_failures={})
    assert stage.postprocess_rate_subspace(config, root, strict=True)["status"] == "solver_capacity_exceeded"
