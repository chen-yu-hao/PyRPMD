"""Pilot decisions, frozen-production isolation, bounded failure and restart."""
import copy
import json
import numpy as np
import pytest

from pyrpmd.system import System
from pyrpmd.workflows.fes import run_fes, reanalyze_fes
from pyrpmd.workflows.adaptive_umbrella import adaptive_options, diagnose_and_refine
import pyrpmd.workflows.adaptive_umbrella as adaptive


def _system():
    return System(("H",), np.array([1.0]), np.zeros((1, 3)))


def _config(dimension=1):
    return {"task": "fes", "beta": 1.0, "beads": 2, "dt_au": 0.03,
            "native": False, "seed": 91,
            "potential": {"provider": "builtin", "name": "harmonic", "params": {"k": 0.2}},
            "sampling": {"equilibration_steps": 10, "production_steps": 40,
                         "sample_interval": 1, "andersen_interval": 10, "replicas": 1},
            "cvs": [{"name": axis, "kind": "cartesian", "atom": 0, "axis": axis}
                    for axis in ("x", "y")[:dimension]],
            "umbrella": {"centers": [[-1.0], [1.0]] if dimension == 1 else [[-.3, -.3], [-.3, .3], [.3, -.3], [.3, .3]],
                         "stiffness": 30.0 if dimension == 1 else 1.0},
            "adaptive_umbrella": {"enabled": True, "pilot_steps": 150, "pilot_equilibration_steps": 30,
                                  "max_rounds": 4, "max_windows": 16,
                                  "target_overlap": 0.001, "min_effective_samples": 1.0},
            "analysis": {"bins": [6] * dimension, "bootstrap": 0},
            "output": {"checkpoint_steps": 25}}


@pytest.mark.parametrize("dimension", [1, 2])
def test_midpoint_refinement_uses_actual_overlap_and_is_deterministic(dimension):
    centers = np.array([[-2.0], [2.0]]) if dimension == 1 else np.array([[-2., -2.], [-2., 2.], [2., -2.], [2., 2.]])
    rng = np.random.default_rng(1)
    streams = [{"window": i, "samples": c + rng.normal(0, 0.025, (100, dimension))}
               for i, c in enumerate(centers)]
    options = adaptive_options(_config())
    result = diagnose_and_refine(streams, centers, np.eye(dimension) * 20, 1., (None,) * dimension, options)
    assert not result["target_met"]
    assert len(result["inserted"]) == len(centers) - 1
    assert result == diagnose_and_refine(streams, centers, np.eye(dimension) * 20, 1., (None,) * dimension, options)
    if dimension == 1:
        assert result["inserted"][0]["center"] == [0.0]


def test_actual_adaptation_freezes_then_uses_only_independent_production(tmp_path):
    cfg = _config()
    result = run_fes(cfg, _system(), tmp_path, return_data=True)
    plan = json.loads((tmp_path / "adaptive-plan.json").read_text())
    assert plan["status"] == "frozen"
    assert plan["rounds"][0]["inserted"]
    assert len(plan["centers"]) > 2
    assert plan["production_seed"] not in [r["seed"] for r in plan["rounds"]]
    assert result["summary"]["samples"] == len(plan["centers"]) * 40
    assert all(len(s["samples"]) == 40 for s in result["streams"])
    assert result["summary"]["adaptive_umbrella"]["pilot_samples_used_in_production"] == 0
    again = reanalyze_fes(cfg, _system(), tmp_path, return_data=True)
    np.testing.assert_array_equal(result["data"]["free_energy"], again["data"]["free_energy"])


def test_two_dimensional_adaptive_workflow(tmp_path):
    cfg = _config(dimension=2)
    result = run_fes(cfg, _system(), tmp_path, return_data=True)
    assert result["data"]["free_energy"].shape == (6, 6)
    assert result["summary"]["adaptive_umbrella"]["target_met"]
    assert result["summary"]["samples"] == result["summary"]["windows"] * 40


def test_saved_pilot_decision_resumes_production_exactly(tmp_path, monkeypatch):
    cfg = _config(dimension=2)
    full = run_fes(cfg, _system(), tmp_path / "full", return_data=True)
    original = adaptive._record_round
    def interrupt_after_durable_decision(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("interrupted after persisted pilot decision")
    with monkeypatch.context() as patch:
        patch.setattr(adaptive, "_record_round", interrupt_after_durable_decision)
        with pytest.raises(RuntimeError, match="persisted pilot decision"):
            run_fes(cfg, _system(), tmp_path / "resume", return_data=True)
    assert json.loads((tmp_path / "resume/adaptive-plan.json").read_text())["status"] == "frozen"
    resumed = run_fes(cfg, _system(), tmp_path / "resume", return_data=True)
    for a, b in zip(full["streams"], resumed["streams"]):
        for name in ("samples", "last_q", "last_p", "energies"):
            np.testing.assert_array_equal(a[name], b[name])


def test_adaptation_budget_exhaustion_does_not_start_production(tmp_path):
    cfg = _config()
    cfg["adaptive_umbrella"]["max_rounds"] = 1
    with pytest.raises(RuntimeError, match="production was not started"):
        run_fes(cfg, _system(), tmp_path)
    plan = json.loads((tmp_path / "adaptive-plan.json").read_text())
    assert plan["status"] == "incomplete"
    assert not (tmp_path / "checkpoints").exists()
    assert json.loads((tmp_path / "summary.json").read_text())["production_started"] is False


def test_default_budget_accepts_a_96_window_reference():
    cfg = _config()
    cfg["adaptive_umbrella"] = {"enabled": True}
    cfg["umbrella"]["centers"] = np.linspace(-1, 1, 96).tolist()
    adaptive.validate_adaptive_umbrella(cfg, _system())
    options = adaptive_options(cfg)
    assert options["max_windows"] >= 96
    assert options["max_pilot_steps"] >= options["pilot_steps"]


def test_enhanced_sampling_default_is_longer_and_explicit_lengths_win():
    from pyrpmd.input import JobSpec
    spec = JobSpec.from_file("examples/fes/adaptive_harmonic.inp")
    spec.data["sampling"].pop("production_steps")
    enhanced, _ = spec.resolve()
    spec.data["adaptive_umbrella"]["enabled"] = False
    ordinary, _ = spec.resolve()
    assert enhanced["sampling"]["production_steps"] == 5 * ordinary["sampling"]["production_steps"]
    spec.data["adaptive_umbrella"]["enabled"] = True
    spec.data["sampling"]["production_steps"] = 100
    explicit, _ = spec.resolve()
    assert explicit["sampling"]["production_steps"] == 100


def test_pilot_duration_tracks_ess_deficit_and_respects_the_cap():
    options = {**adaptive_options(_config()), "min_effective_samples": 20., "max_pilot_steps": 250}
    report = {"effective_samples_target_met": True, "effective_samples_per_window": [40., 25.]}
    assert adaptive._next_pilot_steps(100, report, options) == 100
    report.update(effective_samples_target_met=False, effective_samples_per_window=[2., 25.])
    assert adaptive._next_pilot_steps(100, report, options) == 200
    assert adaptive._next_pilot_steps(200, report, options) == 250
    assert adaptive._next_pilot_steps(250, report, options) == 250


def test_pilot_cap_stops_unproductive_ess_rounds(tmp_path):
    cfg = _config(dimension=2)
    cfg["adaptive_umbrella"].update(max_pilot_steps=150, min_effective_samples=1e6)
    with pytest.raises(RuntimeError, match="max_pilot_steps"):
        run_fes(cfg, _system(), tmp_path)
    plan = json.loads((tmp_path / "adaptive-plan.json").read_text())
    assert len(plan["rounds"]) == 1
    assert not (tmp_path / "checkpoints").exists()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["minimum_effective_samples"] < 1e6


def test_production_ess_is_not_inferred_from_pilot_success(tmp_path):
    summary = {"effective_samples_per_stream": [1., 2., 30., 40.], "warnings": []}
    plan = {"centers": [[0.], [1.]], "options": {"min_effective_samples": 20.},
            "rounds": [{"pilot": {"production_steps": 1000}}], "production_seed": 123}
    adaptive._annotate(summary, plan, tmp_path, False)
    assert summary["adaptive_umbrella"]["target_met"]
    assert not summary["adaptive_umbrella"]["production_effective_samples_target_met"]
    assert summary["adaptive_umbrella"]["production_effective_samples_per_window"] == [3., 70.]
    assert any("Frozen production" in warning for warning in summary["warnings"])


def test_legacy_plan_keeps_its_saved_defaults(tmp_path):
    cfg = _config()
    cfg["adaptive_umbrella"] = {"enabled": True}
    options = adaptive_options(cfg)
    cv, _, centers, stiffness = adaptive.resolve_umbrellas(cfg, _system())
    plan = adaptive._open_plan(cfg, _system(), tmp_path, centers, stiffness, options)
    plan["schema_version"] = 1
    plan.pop("next_pilot_steps")
    plan["options"].update(max_rounds=4, max_windows=64)
    plan["options"].pop("max_pilot_steps")
    adaptive._save_plan(tmp_path, plan)
    reopened = adaptive._open_plan(cfg, _system(), tmp_path, centers, stiffness, options)
    assert reopened["options"]["max_rounds"] == 4
    assert "next_pilot_steps" not in reopened


def test_public_resume_extends_budget_without_resampling_completed_rounds(tmp_path):
    from pyrpmd.runner import run_resolved, resume_run
    from pyrpmd.checkpoint import load_checkpoint
    cfg = _config()
    full = run_fes(cfg, _system(), tmp_path / "full", return_data=True)
    cfg["adaptive_umbrella"]["max_rounds"] = 1
    output = tmp_path / "resume"
    with pytest.raises(RuntimeError, match="production was not started"):
        run_resolved(cfg, _system(), output=output)
    manifest = (output / "manifest.json").read_bytes()
    checkpoints = {p: p.read_bytes() for p in (output / "adaptation").rglob("*.npz")}
    resumed = resume_run(output, adaptive_budget={"max_rounds": 4})
    assert resumed["adaptive_umbrella"]["target_met"]
    assert (output / "manifest.json").read_bytes() == manifest
    assert all(p.read_bytes() == data for p, data in checkpoints.items())
    for stream in full["streams"]:
        path = output / "checkpoints" / f'window_{stream["window"]:05d}_replica_{stream["replica"]:04d}.npz'
        with load_checkpoint(path) as saved:
            np.testing.assert_array_equal(saved["values"], stream["samples"])
            np.testing.assert_array_equal(saved["q"], stream["last_q"])
    plan = json.loads((output / "adaptive-plan.json").read_text())
    assert plan["budget_extensions"][0]["previous"]["max_rounds"] == 1
    with pytest.raises(RuntimeError, match="incomplete"):
        resume_run(output, adaptive_budget={"max_rounds": 8})


@pytest.mark.parametrize("limits", [{"max_rounds": 0}, {"max_windows": 1}, {"target_overlap": .001}, {"max_rounds": True}])
def test_resume_budget_cannot_change_science_or_decrease_limits(tmp_path, limits):
    cfg = _config()
    cfg["adaptive_umbrella"]["max_rounds"] = 1
    with pytest.raises(RuntimeError):
        run_fes(cfg, _system(), tmp_path)
    original = (tmp_path / "adaptive-plan.json").read_bytes()
    with pytest.raises(ValueError):
        adaptive.extend_adaptive_budget(tmp_path, limits)
    assert (tmp_path / "adaptive-plan.json").read_bytes() == original


def test_rate_flux_measure_is_retained_in_pilot_and_production_config():
    cfg = _config()
    cfg["task"] = "rate"
    cfg["analysis"]["measure"] = "rate_flux"
    frozen = adaptive._stage_config(cfg, [[0], [0.5], [1.0]], [[30.]], 111)
    assert frozen["task"] == "rate"
    assert frozen["analysis"]["measure"] == "rate_flux"
    assert frozen["umbrella"]["centers_au"] == [[0], [0.5], [1.0]]
    assert "adaptive_umbrella" not in frozen


def test_adaptive_rate_pmf_reaches_actual_recrossing_children(tmp_path):
    from pyrpmd.workflows.rate import run_rate
    system = System(("H", "H", "H"), np.array([2., 2., 2.]),
                    np.array([[-1.5, 0., 0.], [0., 0., 0.], [1.5, 0., 0.]]), ((0,), (1, 2)))
    cfg = {"task": "rate", "beta": 5., "temperature_K": 300., "beads": 2,
           "dt_au": .003, "seed": 17, "native": False, "potential": {"provider": "free"},
           "sampling": {"equilibration_steps": 50, "production_steps": 200,
                        "sample_interval": 2, "andersen_interval": 20},
           "reaction": {"fragments": [[0], [1, 2]], "forming_bonds": [[0, 1]],
                        "breaking_bonds": [[1, 2]], "rinf": 4.},
           "umbrella": {"centers": np.linspace(0, 1, 11).tolist(), "kappa": 10.},
           "adaptive_umbrella": {"pilot_steps": 100, "pilot_equilibration_steps": 20,
                                 "target_overlap": 0.0001, "min_effective_samples": 1., "max_rounds": 2},
           "analysis": {"bins": 30, "bootstrap": 0}, "output": {"checkpoint_steps": 50},
           "recrossing": {"dividing_surface": 1., "parent_replicas": 1, "parent_samples": 2,
                          "equilibration_steps": 20, "parent_interval_steps": 5,
                          "child_pairs": 1, "child_steps": 20, "bootstrap": 0}}
    result = run_rate(cfg, system, tmp_path)
    assert result["child_trajectories"] == 4
    assert result["pmf"]["sampling_measure"] == "rate_flux"
    plan = json.loads((tmp_path / "umbrella/adaptive-plan.json").read_text())
    assert plan["status"] == "frozen" and plan["pilot_samples_used_in_production"] == 0


def test_adaptive_plan_corruption_is_rejected(tmp_path):
    cfg = _config(2)
    run_fes(cfg, _system(), tmp_path)
    path = tmp_path / "adaptive-plan.json"
    plan = json.loads(path.read_text())
    plan["centers"][0][0] += 0.1
    path.write_text(json.dumps(plan))
    with pytest.raises(RuntimeError, match="checksum"):
        run_fes(cfg, _system(), tmp_path)
