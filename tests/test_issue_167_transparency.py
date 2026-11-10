"""Regression coverage for issue #167 backend transparency and projections."""

import copy

import numpy as np
import pytest

from pyrpmd.block_propagation import cv_descriptor as block_cv_descriptor
from pyrpmd.cv_periodic import FractionalCoordinate
from pyrpmd.propagation import combine_propagation_stats, cv_descriptor
from pyrpmd.runner import _append_backend_warnings, _summary_propagation
from pyrpmd.system import System


def test_fractional_cv_lowering_preserves_affine_origin_for_both_native_abis():
    system = System(
        ("H",), [2.0], [[0.0, 0.0, 0.0]],
        cell=np.array([[5.0, 0.0, 0.0], [0.7, 6.0, 0.0], [0.2, 0.4, 7.0]]),
        pbc=(True, True, True),
    )
    cv = FractionalCoordinate(0, 0, system.cell, system.pbc, system.masses, origin=0.13)
    expected = np.asarray(cv._affine_gradient()).reshape(-1)

    direct = cv_descriptor(cv, system.natoms)["primitive"]
    tagged = block_cv_descriptor(cv, system.natoms)

    np.testing.assert_allclose(direct["linear"], expected)
    assert direct["constant"] == -0.13
    np.testing.assert_allclose(tagged["weights"], expected)
    assert tagged["constant"] == -0.13


def test_summary_propagation_combines_nested_streams_without_mutating_records():
    first = {"native_steps": 3, "python_steps": 2, "fallback_reasons": {"cv": 2}}
    second = {"native_steps": 4, "python_steps": 1, "fallback_reasons": {"cv": 1, "pes": 1}}
    result = _summary_propagation({"streams": [{"propagation": first}, {"propagation": second}]})
    assert result["propagation"] == {
        "native_steps": 7, "python_steps": 3,
        "fallback_reasons": {"cv": 3, "pes": 1},
    }
    assert first["fallback_reasons"] == {"cv": 2}
    assert combine_propagation_stats([]) == {
        "native_steps": 0, "python_steps": 0, "fallback_reasons": {},
    }


def test_large_python_backend_plan_adds_one_actionable_warning():
    config = {"_warnings": []}
    _append_backend_warnings(config, [{
        "stage": "umbrella", "backend": "python", "planned_steps": 1_000_001,
        "reason": "CV kind 'fractional' has no native kernel",
    }])
    assert len(config["_warnings"]) == 1
    assert "1,000,001" in config["_warnings"][0]
    assert "fractional" in config["_warnings"][0]


def test_missing_runtime_counts_are_unknown_not_zero():
    result = _summary_propagation({"task": "legacy_workflow"})
    assert result["propagation"]["native_steps"] is None
    assert result["propagation"]["python_steps"] is None
    assert result["propagation"]["measurement_available"] is False


def test_aggregate_with_nested_details_is_counted_once():
    counts = {"native_steps": 5, "python_steps": 0, "fallback_reasons": {}}
    result = _summary_propagation({"streams": [{"propagation": {**counts, "details": counts}}]})
    assert result["propagation"]["native_steps"] == 5


def _rate_plan_config():
    system=System(("H","H","H"),[1837.,1837.,1837.],[[-1.757,0.,0.],[0.,0.,0.],[1.757,0.,0.]])
    config={"task":"rate","beta":100.,"beads":4,"dt_au":.02,
        "potential":{"provider":"free"},
        "reaction":{"fragments":[[0],[1,2]],"forming_bonds":[[0,1]],"breaking_bonds":[[1,2]],"rinf":10.},
        "sampling":{"equilibration_steps":2,"production_steps":8,"sample_interval":2,"replicas":1},
        "umbrella":{"centers":[0.,.5,1.],"kappa":.2},
        "analysis":{"bins":12,"bootstrap":0},
        "recrossing":{"child_steps":8}}
    return config,system


@pytest.mark.parametrize("provider", ["python","cabi","f2py"])
def test_external_native_binding_is_unknown_without_loading_provider(provider,monkeypatch):
    from pyrpmd import potentials
    from pyrpmd.runner import propagation_backend_plan
    config,system=_rate_plan_config()
    config["potential"]={"provider":provider,"factory":"unavailable.py:build","library":"missing.so"}
    monkeypatch.setattr(potentials,"make_potential",lambda *a: pytest.fail("external provider must not load"))
    stages=propagation_backend_plan(config,system)
    assert stages and all(stage["backend"]=="unknown" for stage in stages)
    assert all("deferred" in stage["reason"] for stage in stages)


def test_explicit_python_remains_known_for_a_deferred_provider():
    from pyrpmd.runner import propagation_backend_plan
    config,system=_rate_plan_config()
    config["potential"]={"provider":"python","factory":"unavailable.py:build"}
    config["dynamics"]={"loop_backend":"python"}
    assert all(s["backend"]=="python" for s in propagation_backend_plan(config,system))


def test_gas_rate_parent_and_child_native_descriptors_are_not_reported_as_python():
    from pyrpmd.dynamics import native_capabilities
    from pyrpmd.runner import propagation_backend_plan
    if not native_capabilities()["available"]:
        pytest.skip("native extension unavailable")
    config,system=_rate_plan_config()
    before=copy.deepcopy(config)
    stages={s["stage"]:s for s in propagation_backend_plan(config,system)}
    assert config==before
    for stage in ("umbrella","recrossing parent","recrossing child"):
        assert stages[stage]["backend"]=="native", stages[stage]
        assert stages[stage]["reason"] is None
    config["recrossing"]["dynamics"]="trpmd"
    child=next(s for s in propagation_backend_plan(config,system) if s["stage"]=="recrossing child")
    assert child["backend"]=="python" and "TRPMD" in child["reason"]


def test_site_rate_python_parent_and_child_are_distinct_from_native_umbrella():
    from pyrpmd.input import JobSpec
    from pyrpmd.runner import propagation_backend_plan
    from pathlib import Path
    config,system=JobSpec.from_file(Path(__file__).resolve().parents[1]/"examples/tutorials/periodic_site_rate/input.inp").resolve()
    stages={s["stage"]:s for s in propagation_backend_plan(config,system)}
    assert stages["site parent"]["backend"]=="python"
    assert stages["site child"]["backend"]=="python"
    assert "checkpointed" in stages["site parent"]["reason"]


def test_history_dependent_fes_plan_does_not_invent_an_umbrella_stage():
    from pyrpmd.runner import propagation_backend_plan
    config,system=_rate_plan_config()
    config.update(task="fes",fes={"method":"metadynamics"})
    stages=propagation_backend_plan(config,system)
    assert [s["stage"] for s in stages]==["metadynamics"]
    assert stages[0]["backend"]=="unknown"



def _umbrella_budget_config():
    system=System(("H",),[1837.],[[0.,0.,0.]])
    config={"task":"fes","beta":100.,"beads":4,"dt_au":.02,"native":False,
            "potential":{"provider":"free"},
            "sampling":{"equilibration_steps":20,"production_steps":100,"replicas":2,"sample_interval":5},
            "cvs":[{"name":"x","kind":"cartesian","atom":0,"axis":"x"}],
            "umbrella":{"cvs":["x"],"centers":[-.5,0.,.5],"stiffness":.2}}
    return config,system


def test_umbrella_budget_includes_each_window_replica_and_local_override():
    from pyrpmd.runner import propagation_backend_plan
    config,system=_umbrella_budget_config()
    assert propagation_backend_plan(config,system)[0]["planned_steps"]==720
    config["umbrella"].pop("centers")
    config["umbrella"]["windows"]=[{"id":"left","center":-.2},
        {"id":"right","center":.2,"sampling":{"equilibration":{"steps":7},"production":{"steps":13}}}]
    assert propagation_backend_plan(config,system)[0]["planned_steps"]==280


def test_recrossing_budgets_include_parent_equilibration_and_both_child_signs():
    from pyrpmd.runner import propagation_backend_plan
    config,system=_rate_plan_config()
    config["recrossing"].update(parent_replicas=3,parent_samples=5,equilibration_steps=7,
                               parent_interval_steps=11,child_pairs=4,child_steps=13)
    budgets={stage["stage"]:stage["planned_steps"] for stage in propagation_backend_plan(config,system)}
    assert budgets["recrossing parent"]==3*(7+5*11)
    assert budgets["recrossing child"]==3*5*4*2*13
    assert budgets["umbrella"]==3*(2+8)


def test_diffusion_equilibration_and_production_have_separate_budgets():
    from pyrpmd.runner import propagation_backend_plan
    config,system=_umbrella_budget_config()
    config["task"]="diffusion"
    config.pop("umbrella");config.pop("cvs")
    stages={stage["stage"]:stage for stage in propagation_backend_plan(config,system)}
    assert stages["sampling"]["planned_steps"]==2*20
    assert stages["production"]["planned_steps"]==2*100


def test_adaptive_unknown_budget_does_not_emit_false_million_step_warning():
    from pyrpmd.runner import propagation_backend_plan
    config,system=_umbrella_budget_config()
    config["sampling"]["production_steps"]=2_000_000
    config["adaptive_umbrella"]={"enabled":True}
    plan=propagation_backend_plan(config,system)
    assert plan[0]["planned_steps"] is None
    _append_backend_warnings(config,plan)
    assert config["_warnings"]==[]


def test_million_step_warning_uses_expanded_window_budget():
    from pyrpmd.runner import propagation_backend_plan
    config,system=_umbrella_budget_config()
    config["sampling"]["production_steps"]=200_000
    plan=propagation_backend_plan(config,system)
    assert plan[0]["planned_steps"]==1_200_120
    _append_backend_warnings(config,plan)
    assert len(config["_warnings"])==1 and "1,200,120" in config["_warnings"][0]
