"""Regression coverage for issue #167 backend transparency and projections."""

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
    assert _summary_propagation(result) == result
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


def test_summary_counts_collision_records_and_does_not_recount_nested_totals():
    collision = {"native_steps": 7, "python_steps": 2}
    nested = {"native_steps": 3, "python_steps": 1, "fallback_reasons": {"cv": 1},
              "streams": [{"native_steps": 3, "python_steps": 1, "fallback_reasons": {"cv": 1}}]}
    result = _summary_propagation({"records": [{"propagation": collision}],
                                   "stages": [{"propagation": nested}]})
    assert result["propagation"] == {
        "native_steps": 10, "python_steps": 3, "fallback_reasons": {"cv": 1}}


def _plan_input():
    system = System(("H", "H", "H"), [2., 2., 2.],
                    [[-1.5, 0., 0.], [0., 0., 0.], [1.5, 0., 0.]], ((0,), (1, 2)))
    config = {
        "task": "rate", "beta": 3., "beads": 4, "dt_au": .02,
        "potential": {"provider": "free"},
        "sampling": {"equilibration_steps": 10, "production_steps": 100, "replicas": 2},
        "cvs": [{"kind": "cartesian", "atom": 0, "axis": 0}],
        "umbrella": {"centers": [-1., 0., 1.], "stiffness": 2.},
        "reaction": {"fragments": [[0], [1, 2]], "forming_bonds": [[0, 1]],
                     "breaking_bonds": [[1, 2]], "rinf": 10.},
        "recrossing": {"parent_replicas": 2, "parent_samples": 3, "equilibration_steps": 5,
                       "parent_interval_steps": 7, "child_pairs": 4, "child_steps": 11},
    }
    return config, system


def test_backend_plan_preserves_native_rate_stages_and_separate_budgets():
    import pytest
    from pyrpmd.dynamics import native_capabilities
    from pyrpmd.runner import propagation_backend_plan
    if not native_capabilities()["available"]:
        pytest.skip("native build required")
    config, system = _plan_input()
    plan = {item["stage"]: item for item in propagation_backend_plan(config, system)}
    assert plan["umbrella"]["planned_steps"] == 3 * 2 * 110
    assert plan["recrossing parent"]["planned_steps"] == 2 * (5 + 3 * 7)
    assert plan["recrossing child"]["planned_steps"] == 2 * 2 * 3 * 4 * 11
    assert plan["recrossing parent"]["backend"] == "native"
    assert plan["recrossing child"]["backend"] == "native"


def test_external_binding_is_unknown_without_loading_provider(monkeypatch):
    from pyrpmd.runner import propagation_backend_plan
    from pyrpmd import potentials
    config, system = _plan_input()
    config["potential"] = {"provider": "c_abi", "library": "not-loaded.so"}
    monkeypatch.setattr(potentials, "make_potential", lambda *a: (_ for _ in ()).throw(AssertionError("loaded PES")))
    plan = propagation_backend_plan(config, system)
    assert {item["backend"] for item in plan} == {"unknown"}
    config["dynamics"] = {"loop_backend": "python"}
    assert {item["backend"] for item in propagation_backend_plan(config, system)} == {"python"}


def test_rate_nonroot_does_not_iterate_root_only_parent_results(monkeypatch, tmp_path):
    from pyrpmd.workflows import rate
    config, system = _plan_input()
    sentinel = object()
    seen = []
    def root_call(function, *args, **kwargs):
        seen.append(function.__name__)
        if function is rate._prepare:
            return config, system
        if function is rate._pmf_and_parent_tasks:
            return {}, [object()]
        if function is rate._parent_propagation:
            assert args == (None,)
            return []
        if function is rate._child_tasks:
            return []
        if function is rate._aggregate:
            return sentinel
        if function.__name__ == "finish_rate":
            assert args[-1] is sentinel
            return sentinel
        raise AssertionError(function)
    monkeypatch.setattr(rate, "root_call", root_call)
    monkeypatch.setattr(rate, "sample_umbrellas", lambda *a, **kw: None)
    monkeypatch.setattr(rate, "run_tasks", lambda *a, **kw: None)
    monkeypatch.setattr(rate, "_dispatch_children", lambda *a, **kw: None)
    assert rate.run_rate(config, system, tmp_path) is sentinel
    assert "_parent_propagation" in seen


def test_site_nonroot_passes_root_only_parents_without_iteration(monkeypatch, tmp_path):
    from pyrpmd.workflows import site_rate as site
    config, system = _plan_input()
    config["task"] = "site_rate"
    sentinel = object()
    def root_call(function, *args, **kwargs):
        if function is site.validate_site_rate:
            return None
        if function is site._estimate_pmf:
            return {}, []
        if function is site._child_tasks:
            return []
        if function is site._aggregate:
            assert args[-1] is None
            return sentinel
        raise AssertionError(function)
    monkeypatch.setattr(site, "root_call", root_call)
    monkeypatch.setattr(site, "run_fes", lambda *a, **kw: None)
    monkeypatch.setattr(site, "map_trajectory_tasks", lambda *a, **kw: None)
    assert site.run_site_rate(config, system, tmp_path) is sentinel


def test_independent_pmf_plan_uses_its_backend_clock_and_thermostat():
    import copy
    import pytest
    from pyrpmd.dynamics import native_capabilities
    from pyrpmd.input import JobSpec
    from pyrpmd.runner import propagation_backend_plan
    from test_rate_pmf_input import configuration, independent_pmf
    if not native_capabilities()["available"]:
        pytest.skip("native build required")
    raw = configuration()
    raw["native"] = True
    raw["rate"] = {"pmf": independent_pmf()}
    for overrides in ({"loop_backend": "python"}, {"native": False}, {}):
        request = copy.deepcopy(raw)
        request["rate"]["pmf"]["dynamics"].update(overrides)
        resolved, system = JobSpec.from_mapping(request).resolve()
        plan = {item["stage"]: item for item in propagation_backend_plan(resolved, system)}
        assert plan["umbrella"]["backend"] == "python"
        assert plan["umbrella"]["planned_steps"] == 4 * 2 * (10 + 50)
        assert plan["recrossing child"]["backend"] == "native"
        if not overrides:
            assert "PILE-L" in plan["umbrella"]["reason"]


def test_surface_plan_uses_declared_surface_before_testing_native_support(monkeypatch):
    from pyrpmd.runner import propagation_backend_plan
    from pyrpmd.workflows import surface
    from test_surface_workflow import cluster_job
    from pyrpmd.input import JobSpec
    raw = cluster_job()
    raw["native"] = True
    resolved, system = JobSpec.from_mapping(raw).resolve()
    original = surface.surface_definition
    seen = []
    def definition(*args):
        result = original(*args)
        coordinate = result["coordinate"]
        dividing = coordinate.dividing_surface
        def checked(value):
            seen.append(value)
            return dividing(value)
        coordinate.dividing_surface = checked
        return result
    monkeypatch.setattr(surface, "surface_definition", definition)
    propagation_backend_plan(resolved, system)
    assert seen == [raw["surface"]["dividing_surface"]]


@pytest.mark.parametrize("collection", [list, tuple])
def test_summary_preserves_legacy_propagation_collection_and_totals(collection):
    streams = collection([
        {"sampling": {"native_steps": 7, "python_steps": 0}, "production": None},
        {"sampling": {"native_steps": 0, "python_steps": 3,
                      "fallback_reasons": {"provider": 3}}},
    ])
    source = {"propagation": streams}
    result = _summary_propagation(source)
    assert result["propagation"] is streams
    assert "propagation_totals" not in source
    assert "propagation_streams" not in result
    assert result["propagation_totals"] == {
        "native_steps": 7, "python_steps": 3, "fallback_reasons": {"provider": 3}}
    assert _summary_propagation(result) == result


def test_backend_plan_borrows_provider_without_reconstructing_or_closing(monkeypatch):
    from unittest.mock import Mock
    from pyrpmd import potentials
    from pyrpmd.runner import propagation_backend_plan
    config, system = _plan_input()
    provider = potentials.FreePotential()
    provider.close = Mock()
    factory = Mock(side_effect=AssertionError("borrowed provider was reconstructed"))
    monkeypatch.setattr(potentials, "make_potential", factory)
    propagation_backend_plan(config, system, potential=provider)
    factory.assert_not_called()
    provider.close.assert_not_called()
