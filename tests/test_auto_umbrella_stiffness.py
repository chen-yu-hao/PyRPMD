"""Pilot-only spring tuning, bounded decisions and frozen resume behavior."""

import copy
import json

import numpy as np
import pytest

from pyrpmd.system import System
from pyrpmd.workflows import adaptive_umbrella as adaptive
from pyrpmd.workflows.fes import run_fes


def _config():
    return {
        "task": "fes", "beta": 1., "beads": 2, "dt_au": .03,
        "native": False, "seed": 91,
        "potential": {"provider": "builtin", "name": "harmonic", "params": {"k": .2}},
        "sampling": {"equilibration_steps": 10, "production_steps": 40,
                     "sample_interval": 1, "andersen_interval": 10, "replicas": 1},
        "cvs": [{"name": "x", "kind": "cartesian", "atom": 0, "axis": "x"}],
        "umbrella": {"centers": [[-1.], [1.]], "stiffness": 30.},
        "adaptive_umbrella": {"adapt_stiffness": True, "pilot_steps": 150,
                              "pilot_equilibration_steps": 30, "max_rounds": 8,
                              "max_windows": 32, "target_overlap": .001,
                              "min_effective_samples": 1.},
        "analysis": {"bins": [6], "bootstrap": 0},
        "output": {"checkpoint_steps": 25},
    }


def _decision(shift=0., width=.05, *, connected=False, enough=True, **state):
    centers = np.array([[0.], [1.]])
    noise = np.array([-1., 1.] * 100)[:, None] * width
    streams = [{"window": i, "samples": center + shift + noise}
               for i, center in enumerate(centers)]
    report = {"overlap_connected_at_target": connected,
              "effective_samples_target_met": enough}
    return adaptive._stiffness_decision(
        streams, centers, np.array([[20.]]), 1., [None], report, state)


def test_stiffness_tuning_is_opt_in_and_boolean():
    assert adaptive.adaptive_options({})["adapt_stiffness"] is False
    for value in (1, "true", None):
        with pytest.raises(ValueError, match="adapt_stiffness must be boolean"):
            adaptive.adaptive_options({"adaptive_umbrella": {"adapt_stiffness": value}})


def test_displaced_windows_strengthen_and_require_another_pilot():
    decision = _decision(shift=.8)
    assert decision["factor"] == 2.
    assert decision["changed"]
    assert not decision["centering_target_met"]
    assert decision["next_stiffness"] == [[40.]]
    assert decision["stiffness_floor_scale"] == 2.


def test_narrow_disconnected_windows_soften_but_connected_windows_do_not():
    assert _decision()["factor"] == .5
    assert _decision(connected=True)["factor"] == 1.
    assert _decision(width=.4)["factor"] == 1.
    assert _decision(enough=False)["factor"] == 1.


def test_failed_centering_sets_a_floor_that_prevents_oscillation():
    decision = _decision(stiffness_scale=2., stiffness_floor_scale=2.)
    assert decision["factor"] == 1.
    assert decision["reason"] == "centering_floor"
    assert _decision(stiffness_scale=1 / 16)["factor"] == 1.
    upper = _decision(shift=.8, stiffness_scale=16.)
    assert upper["factor"] == 1.
    assert not upper["centering_target_met"]


def test_stiffness_decision_is_invariant_to_coordinate_units():
    base = _decision(shift=.8)
    centers = np.array([[0.], [100.]])
    streams = [{"window": i, "samples": c + 80 + np.array([-5., 5.] * 100)[:, None]}
               for i, c in enumerate(centers)]
    scaled = adaptive._stiffness_decision(
        streams, centers, np.array([[.002]]), 1., [None],
        {"overlap_connected_at_target": False, "effective_samples_target_met": True}, {})
    assert scaled["factor"] == base["factor"]
    np.testing.assert_allclose(scaled["displacement_per_spacing"], base["displacement_per_spacing"])


def test_budget_extension_restores_the_last_proposed_spring(tmp_path):
    config = _config()
    config["adaptive_umbrella"]["max_rounds"] = 1
    system = System(("H",), np.array([1.]), np.zeros((1, 3)))
    with pytest.raises(RuntimeError, match="production was not started"):
        run_fes(config, system, tmp_path)
    path = tmp_path / "adaptive-plan.json"
    stopped = json.loads(path.read_text())
    proposal = stopped["rounds"][-1]["stiffness_update"]
    assert proposal["changed"]
    adaptive.extend_adaptive_budget(tmp_path, {"max_rounds": 8})
    resumed = json.loads(path.read_text())
    assert resumed["stiffness"] == proposal["next_stiffness"]
    assert resumed["stiffness_scale"] == proposal["stiffness_scale"]
    assert resumed["stiffness_floor_scale"] == proposal["stiffness_floor_scale"]


def test_changed_spring_is_frozen_and_resume_keeps_production_exact(tmp_path, monkeypatch):
    config = _config()
    system = System(("H",), np.array([1.]), np.zeros((1, 3)))
    expected = run_fes(config, system, tmp_path / "full", return_data=True)
    plan = json.loads((tmp_path / "full/adaptive-plan.json").read_text())
    assert any(r["stiffness_update"]["changed"] for r in plan["rounds"])
    assert plan["stiffness"] == plan["rounds"][-1]["stiffness"]
    assert not plan["rounds"][-1]["stiffness_update"]["changed"]
    assert all(s["stiffness"].tolist() == plan["stiffness"] for s in expected["streams"])
    assert all(len(s["samples"]) == 40 for s in expected["streams"])
    original = adaptive._record_round

    def interrupt_after_change(*args, **kwargs):
        result = original(*args, **kwargs)
        if result["rounds"][-1]["stiffness_update"]["changed"]:
            raise RuntimeError("after spring decision")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(adaptive, "_record_round", interrupt_after_change)
        with pytest.raises(RuntimeError, match="after spring decision"):
            run_fes(copy.deepcopy(config), system, tmp_path / "resumed", return_data=True)
    actual = run_fes(config, system, tmp_path / "resumed", return_data=True)
    for left, right in zip(expected["streams"], actual["streams"]):
        np.testing.assert_array_equal(left["samples"], right["samples"])
        np.testing.assert_array_equal(left["last_q"], right["last_q"])
