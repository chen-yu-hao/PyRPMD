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


def test_external_thread_executor_preserves_pilot_and_production_streams(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    cfg = _config(dimension=2)
    expected = run_fes(cfg, _system(), tmp_path / "serial", return_data=True)
    with ThreadPoolExecutor(max_workers=2) as executor:
        actual = run_fes(cfg, _system(), tmp_path / "threads", executor=executor,
                         return_data=True)
    plans = [json.loads((tmp_path / path / "adaptive-plan.json").read_text())
             for path in ("serial", "threads")]
    assert plans[0] == plans[1]
    assert actual["summary"]["samples"] == expected["summary"]["samples"] == 160
    assert len(actual["streams"]) == len(expected["streams"]) == 4
    np.testing.assert_array_equal(actual["data"]["free_energy"], expected["data"]["free_energy"])
    for a, b in zip(actual["streams"], expected["streams"]):
        for key in ("samples", "last_q", "last_p", "energies"):
            np.testing.assert_array_equal(a[key], b[key])


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
    plan = {"centers": [[0.], [1.]],
            "options": {"min_effective_samples": 20., "target_overlap": .03},
            "rounds": [{"pilot": {"production_steps": 1000}}], "production_seed": 123}
    result = {"summary": summary,
              "streams": [{"window": i // 2, "samples": np.zeros((40, 1))} for i in range(4)],
              "data": {"overlap": np.full((4, 4), .25), "connected": True,
                       "effective_samples_per_stream": [1., 2., 30., 40.]}}
    adaptive._annotate(result, plan, tmp_path)
    assert summary["adaptive_umbrella"]["target_met"]
    assert summary["adaptive_umbrella"]["pilot_target_met"]
    assert not summary["adaptive_umbrella"]["production_target_met"]
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


def test_connected_overlap_does_not_hide_two_dimensional_coverage_holes():
    # All windows explore the same central basin; their overlap graph is
    # excellent, but it gives no evidence of the surrounding declared domain.
    rng = np.random.default_rng(721)
    centers = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    streams = [{"window": i, "samples": rng.normal(0, .03, (500, 2))} for i in range(4)]
    cfg = _config(2)
    cfg["adaptive_umbrella"].update(coverage_bins=[3, 3], coverage_ranges=[[-1.5, 1.5]] * 2,
                                    min_bin_effective_samples=5, target_coverage=1.)
    options = adaptive_options(cfg)
    report = diagnose_and_refine(streams, centers, np.eye(2) * .1, 1., (None, None), options)
    assert report["overlap_connected_at_target"]
    assert report["effective_samples_target_met"]
    assert not report["target_met"]
    assert report["coverage"]["fraction"] == pytest.approx(1 / 9)
    assert len(report["inserted"]) == 4  # other four deficient cells already have centers
    assert all(item["reason"] == "coverage" for item in report["inserted"])
    assert report == diagnose_and_refine(streams, centers, np.eye(2) * .1, 1., (None, None), options)


def test_coverage_local_ess_accounts_for_weights_and_correlations():
    options = {"coverage_bins": [2], "coverage_ranges": [[-1., 1.]],
               "min_bin_effective_samples": 1.55, "target_coverage": 1.}
    values = np.array([[-.5]] * 3 + [[.5]] * 3)
    weights = np.array([1., 1., 2., 3., 3., 3.])
    factors = np.array([1., 1., 2., 2., 2., 2.])
    report, _, _, _ = adaptive._coverage_diagnostic(values, weights, factors, [None], options)
    np.testing.assert_allclose(report["effective_samples"], [1.6, 1.5])
    assert report["sample_counts"] == [3, 3]
    assert report["fraction"] == .5 and not report["target_met"]
    # ESS is independent of the arbitrary normalization of target weights.
    scaled = adaptive._coverage_diagnostic(values, 7 * weights, factors, [None], options)[0]
    np.testing.assert_allclose(scaled["effective_samples"], report["effective_samples"])


def test_coverage_wraps_periodic_samples_into_the_declared_domain():
    options = {"coverage_bins": [2], "coverage_ranges": [[-np.pi, np.pi]],
               "min_bin_effective_samples": 1., "target_coverage": 1.}
    values = np.array([[-np.pi + .1], [np.pi + .1]])
    report = adaptive._coverage_diagnostic(values, np.ones(2), np.ones(2), [2 * np.pi], options)[0]
    assert report["sample_counts"] == [2, 0]
    assert report["effective_samples"] == [2., 0.]
    assert not report["target_met"]


def test_coverage_cannot_relax_per_window_ess_or_exceed_window_budget():
    cfg = _config()
    cfg["adaptive_umbrella"].update(coverage_bins=[3], coverage_ranges=[[-1.5, 1.5]],
                                    min_bin_effective_samples=1., target_coverage=1.,
                                    min_effective_samples=1000, max_windows=3)
    rng = np.random.default_rng(3)
    centers = np.array([[-.1], [.1]])
    streams = [{"window": i, "samples": rng.normal(0, .03, (100, 1))} for i in range(2)]
    report = diagnose_and_refine(streams, centers, np.eye(1), 1., [None], adaptive_options(cfg))
    assert report["overlap_connected_at_target"]
    assert not report["effective_samples_target_met"]
    assert not report["target_met"]
    assert len(report["next_centers"]) == 3
    assert any(item["reason"] == "max_windows" for item in report["rejected_insertions"])


@pytest.mark.parametrize("periodic", [False, True])
def test_ess_detects_slow_amplitudes_hidden_by_fast_sign_changes(periodic):
    rng = np.random.default_rng(73)
    time = np.arange(4000)
    signs = rng.choice([-1., 1.], len(time))
    if periodic:
        # Opposite circular directions switch rapidly; their common axis drifts slowly.
        values = (.7 * np.sin(2 * np.pi * time / 2000) + np.pi * (signs > 0))[:, None]
        periods = [2 * np.pi]
    else:
        values = (signs * (1 + .8 * np.sin(2 * np.pi * time / 2000)))[:, None]
        periods = [None]
    cfg = _config()
    cfg["adaptive_umbrella"]["min_effective_samples"] = 100
    streams = [{"window": 0, "samples": values}]
    report = diagnose_and_refine(streams, [[0.]], np.eye(1), 1., periods, adaptive_options(cfg))
    assert report["effective_samples_per_window"][0] < 100
    assert not report["target_met"]


@pytest.mark.parametrize("extra, message", [
    ({"coverage_bins": [2, 2]}, "supplied together"),
    ({"coverage_ranges": [[0., 1.]]}, "supplied together"),
    ({"coverage_bins": [2, True], "coverage_ranges": [[0., 1.]] * 2}, "integer"),
    ({"coverage_bins": [1001, 1000], "coverage_ranges": [[0., 1.]] * 2}, "million"),
    ({"coverage_bins": [2, 2], "coverage_ranges": [[0., 1.]]}, "dimension"),
    ({"target_coverage": 1.1}, "exceed one"),
    ({"min_bin_effective_samples": 0}, "positive"),
])
def test_coverage_options_are_validated(extra, message):
    cfg = _config(2)
    cfg["adaptive_umbrella"].update(extra)
    with pytest.raises(ValueError, match=message):
        adaptive_options(cfg)


def test_coverage_bounds_resolve_units_and_reject_wrong_dimensions():
    cfg = _config(2)
    cfg["adaptive_umbrella"].update(coverage_bins=[2, 3],
                                    coverage_ranges=[["-1 bohr", "1 bohr"], ["0 bohr", "2 bohr"]])
    cv, _, _, _ = adaptive.resolve_umbrellas(cfg, _system())
    options = adaptive._resolve_coverage_options(adaptive_options(cfg), cv)
    assert options["coverage_ranges"] == [[-1., 1.], [0., 2.]]
    cfg["adaptive_umbrella"].update(coverage_bins=[2], coverage_ranges=[[0., 1.]])
    with pytest.raises(ValueError, match="selected CVs"):
        adaptive.validate_adaptive_umbrella(cfg, _system())
    cfg["adaptive_umbrella"].update(coverage_bins=[2, 2], coverage_ranges=[[0., 1.], [2., 1.]])
    with pytest.raises(ValueError, match="lower < upper"):
        adaptive.validate_adaptive_umbrella(cfg, _system())


def test_optional_coverage_workflow_and_reanalysis_preserve_frozen_production(tmp_path):
    cfg = _config(2)
    cfg["adaptive_umbrella"].update(coverage_bins=[2, 2], coverage_ranges=[[-.6, .6]] * 2,
                                    min_bin_effective_samples=.1, target_coverage=1.)
    result = run_fes(cfg, _system(), tmp_path, return_data=True)
    plan = json.loads((tmp_path / "adaptive-plan.json").read_text())
    assert plan["status"] == "frozen"
    assert plan["rounds"][-1]["coverage"]["target_met"]
    assert result["summary"]["adaptive_umbrella"]["coverage"]["target_met"]
    assert result["summary"]["samples"] == len(plan["centers"]) * 40
    again = reanalyze_fes(cfg, _system(), tmp_path, return_data=True)
    np.testing.assert_array_equal(result["data"]["free_energy"], again["data"]["free_energy"])


def test_incomplete_coverage_does_not_start_production(tmp_path):
    cfg = _config(2)
    cfg["adaptive_umbrella"].update(coverage_bins=[4, 4], coverage_ranges=[[-10., 10.]] * 2,
                                    min_bin_effective_samples=5., target_coverage=1., max_rounds=1)
    with pytest.raises(RuntimeError, match="production was not started"):
        run_fes(cfg, _system(), tmp_path)
    plan = json.loads((tmp_path / "adaptive-plan.json").read_text())
    assert plan["rounds"][0]["overlap_connected_at_target"]
    assert plan["rounds"][0]["effective_samples_target_met"]
    assert not plan["rounds"][0]["coverage"]["target_met"]


def test_coverage_shortfall_extends_pilots_and_stops_at_sampling_cap(tmp_path):
    # Every existing window has adequate global ESS, but a deliberately broad
    # declared domain has unsupported cells. With no spare window slots the
    # only permitted adaptation is a longer pilot, bounded by max_pilot_steps.
    cfg = _config(2)
    cfg["adaptive_umbrella"].update(
        coverage_bins=[4, 4], coverage_ranges=[[-10., 10.]] * 2,
        min_bin_effective_samples=5., target_coverage=1.,
        min_effective_samples=.1, max_windows=4, max_rounds=8,
        pilot_steps=50, max_pilot_steps=200,
    )
    with pytest.raises(RuntimeError, match="max_pilot_steps before coverage target"):
        run_fes(cfg, _system(), tmp_path)
    plan = json.loads((tmp_path / "adaptive-plan.json").read_text())
    assert [r["pilot"]["production_steps"] for r in plan["rounds"]] == [50, 100, 200]
    assert all(r["effective_samples_target_met"] for r in plan["rounds"])
    assert all(not r["inserted"] and not r["coverage"]["target_met"] for r in plan["rounds"])
    assert plan["status"] == "incomplete"
    assert not (tmp_path / "checkpoints").exists()
    assert not (tmp_path / "checkpoints").exists()


@pytest.mark.parametrize("schema", [1, 2])
def test_legacy_frozen_plan_preserves_samples_and_labels_its_original_ess_rule(tmp_path, schema):
    cfg = _config(2)
    original = run_fes(cfg, _system(), tmp_path, return_data=True)
    plan = json.loads((tmp_path / "adaptive-plan.json").read_text())
    plan["schema_version"] = schema
    plan.pop("overlap_metric")
    for report in plan["rounds"]:
        report.pop("effective_sample_observables")
        report.pop("overlap_metric")
    adaptive._save_plan(tmp_path, plan)  # emulate a checksum-valid pre-enhancement plan
    resumed = reanalyze_fes(cfg, _system(), tmp_path, return_data=True)
    np.testing.assert_array_equal(resumed["data"]["free_energy"], original["data"]["free_energy"])
    summary = resumed["summary"]
    assert summary["adaptive_umbrella"]["overlap_metric"] == adaptive.LEGACY_OVERLAP_METRIC
    assert summary["adaptive_umbrella"]["effective_sample_observables"].startswith("Legacy saved pilot")
    assert any("legacy coordinate-only pilot ESS" in warning for warning in summary["warnings"])


def test_normalized_overlap_does_not_refine_64_almost_identical_windows():
    # Transition probabilities are diluted to 1/K even when all K states have
    # almost exactly the same support. Inserting more windows makes that old
    # diagnostic worse, eventually exhausting max_windows on an ideal case.
    rng = np.random.default_rng(24)
    centers = np.linspace(-1e-4, 1e-4, 64)[:, None]
    streams = [{"window": i, "samples": center + rng.normal(size=(400, 1))}
               for i, center in enumerate(centers)]
    options = adaptive_options({"adaptive_umbrella": {"enabled": True}})
    current = diagnose_and_refine(streams, centers, np.eye(1), 1., [None], options)
    legacy = diagnose_and_refine(streams, centers, np.eye(1), 1., [None], options,
                                 overlap_metric=adaptive.LEGACY_OVERLAP_METRIC)
    assert current["overlap_metric"] == adaptive.NORMALIZED_OVERLAP_METRIC
    assert current["target_met"] and current["inserted"] == []
    assert current["spanning_overlap"] > .999
    assert legacy["effective_samples_target_met"]
    assert not legacy["overlap_connected_at_target"]
    assert legacy["spanning_overlap"] == pytest.approx(1 / 64, rel=1e-4)
    assert len(legacy["inserted"]) == 63


def test_normalized_overlap_distinguishes_identical_and_disjoint_support():
    # Exact identical states with unequal sample counts must still score one.
    counts = np.array([1., 99., 500.])
    identical = np.broadcast_to(counts / counts.sum(), (3, 3))
    np.testing.assert_allclose(
        adaptive._overlap_scores(identical, adaptive.NORMALIZED_OVERLAP_METRIC), 1.)
    # A disconnected empirical overlap matrix must stay disconnected.
    np.testing.assert_array_equal(
        adaptive._overlap_scores(np.eye(3), adaptive.NORMALIZED_OVERLAP_METRIC), np.eye(3))
    rng = np.random.default_rng(15)
    centers = np.array([[-3.], [3.]])
    streams = [{"window": i, "samples": center + rng.normal(0, .01, (400, 1))}
               for i, center in enumerate(centers)]
    report = diagnose_and_refine(streams, centers, np.eye(1) * 100, 1., [None],
                                 adaptive_options({"adaptive_umbrella": {"enabled": True}}))
    assert not report["target_met"]
    assert report["spanning_overlap"] < 1e-8
    assert report["inserted"][0]["center"] == [0.]


@pytest.mark.parametrize("schema", [1, 2, 3])
def test_saved_plan_schema_keeps_its_overlap_decisions(tmp_path, schema):
    cfg = _config()
    cfg["umbrella"].update(centers=[[-.1], [.1]], stiffness=1.)
    cfg["adaptive_umbrella"].update(target_overlap=.75, max_rounds=1,
                                    min_effective_samples=20.)
    options = adaptive_options(cfg)
    cv, _, centers, stiffness = adaptive.resolve_umbrellas(cfg, _system())
    plan = adaptive._open_plan(cfg, _system(), tmp_path, centers, stiffness, options)
    assert plan["schema_version"] == 3
    plan["schema_version"] = schema
    if schema < 3:
        plan.pop("overlap_metric")
    if schema == 1:
        plan.pop("next_pilot_steps")
    adaptive._save_plan(tmp_path, plan)
    plan = adaptive._open_plan(cfg, _system(), tmp_path, centers, stiffness, options)
    rng = np.random.default_rng(123)
    streams = [{"window": i, "samples": center + rng.normal(size=(400, 1))}
               for i, center in enumerate(centers)]
    adaptive._record_round(plan, streams, cfg, _system(), str(tmp_path), cv, options,
                           {"equilibration_steps": 30, "production_steps": 400})
    decision = plan["rounds"][-1]
    if schema < 3:
        assert decision["overlap_metric"] == adaptive.LEGACY_OVERLAP_METRIC
        assert plan["status"] == "incomplete"
        assert not decision["target_met"]
    else:
        assert decision["overlap_metric"] == adaptive.NORMALIZED_OVERLAP_METRIC
        assert plan["status"] == "frozen"
        assert decision["target_met"]


def _synthetic_production(centers, *, scale=.1, stiffness=1., replicas=2):
    from pyrpmd.analysis.fes import reconstruct_fes
    rng = np.random.default_rng(73)
    streams = []
    # Unequal lengths and interleaved windows exercise grouping by identity,
    # not an assumption that every window has equal, contiguous replicas.
    for replica in range(replicas):
        for window, center in enumerate(centers):
            streams.append({"window": window, "replica": replica,
                            "samples": np.asarray(center) + rng.normal(0, scale, (200 + 20 * replica, 1)),
                            "center": center, "stiffness": [[stiffness]]})
    data = reconstruct_fes([s["samples"] for s in streams],
                           [s["center"] for s in streams],
                           [s["stiffness"] for s in streams], 1., bootstrap=0)
    return {"summary": {"effective_samples_per_stream": data["effective_samples_per_stream"],
                        "converged": False, "warnings": []},
            "data": data, "streams": streams}


def _diagnostic_plan(centers, options):
    return {"schema_version": 3, "overlap_metric": adaptive.NORMALIZED_OVERLAP_METRIC,
            "centers": centers, "options": options, "production_seed": 123,
            "rounds": [{"pilot": {"production_steps": 100},
                        "effective_sample_observables": adaptive.CV_CORRELATION_OBSERVABLES}]}


def test_production_coverage_is_recomputed_without_another_mbar_solve(tmp_path, monkeypatch):
    centers = [[-.1], [.1]]
    options = adaptive_options({"adaptive_umbrella": {
        "enabled": True, "coverage_bins": [4], "coverage_ranges": [[-2., 2.]],
        "target_coverage": 1., "min_bin_effective_samples": 5.}})
    result = _synthetic_production(centers)
    plan = _diagnostic_plan(centers, options)
    plan["rounds"][-1]["coverage"] = {"fraction": 1., "target_met": True}
    def no_new_solver(*args, **kwargs):
        raise AssertionError("production diagnostics must reuse the existing MBAR fit")
    monkeypatch.setattr(adaptive, "solve_mbar", no_new_solver)
    adaptive._annotate(result, plan, tmp_path, periods=[None])
    report = result["summary"]["adaptive_umbrella"]
    assert report["target_met"] and report["pilot_target_met"]
    assert report["pilot_coverage"]["target_met"]
    assert report["coverage_scope"] == "discarded_adaptation_pilots"
    assert report["production_overlap_connected_at_target"]
    assert report["production_effective_samples_target_met"]
    assert report["production_coverage"]["scope"] == "frozen_production"
    assert report["production_coverage"]["fraction"] == .5
    assert not report["production_coverage"]["target_met"]
    assert not report["production_target_met"]
    assert result["summary"]["converged"] is False
    assert any("pilot coverage is reported separately" in text for text in result["summary"]["warnings"])


def test_production_overlap_is_not_inferred_from_accepted_pilot(tmp_path):
    centers = [[-3.], [3.]]
    options = adaptive_options({"adaptive_umbrella": {"enabled": True}})
    result = _synthetic_production(centers, scale=.01, stiffness=100.)
    adaptive._annotate(result, _diagnostic_plan(centers, options), tmp_path, periods=[None])
    report = result["summary"]["adaptive_umbrella"]
    assert report["pilot_target_met"]
    assert report["production_effective_samples_target_met"]
    assert not report["production_overlap_connected_at_target"]
    assert not report["production_analysis_overlap_connected"]
    assert not report["production_target_met"]
    assert report["production_spanning_overlap"] < 1e-8


@pytest.mark.parametrize("metric", [adaptive.LEGACY_OVERLAP_METRIC, adaptive.NORMALIZED_OVERLAP_METRIC])
def test_production_replica_grouping_matches_independent_pooled_fit(metric):
    centers = [[-.2], [.2]]
    options = adaptive_options({"adaptive_umbrella": {"enabled": True}})
    result = _synthetic_production(centers, scale=1.)
    plan = _diagnostic_plan(centers, options)
    if metric == adaptive.LEGACY_OVERLAP_METRIC:
        plan["schema_version"] = 2
    production = adaptive._production_diagnostic(result, plan, [None])
    independent = diagnose_and_refine(result["streams"], centers, np.eye(1), 1., [None], options,
                                      overlap_metric=metric)
    assert production["overlap_metric"] == metric
    assert production["spanning_overlap"] == pytest.approx(independent["spanning_overlap"])
    np.testing.assert_allclose(production["effective_samples_per_window"],
                               independent["effective_samples_per_window"])
    assert production["target_met"] == independent["target_met"]


def test_summary_only_diagnostics_are_computed_before_root_broadcast(tmp_path, monkeypatch):
    import pyrpmd.parallel.execution as execution
    cfg = _config(2)
    cfg["adaptive_umbrella"].update(coverage_bins=[2, 2], coverage_ranges=[[-.6, .6]] * 2,
                                    min_bin_effective_samples=.1, target_coverage=1.)
    original_broadcast = execution.broadcast
    summaries = []
    def inspect_payload(payload):
        if isinstance(payload, tuple) and len(payload) == 2 and isinstance(payload[0], dict):
            value = payload[0]
            assert not {"summary", "data", "streams"}.issubset(value)
            if value.get("task") == "fes":
                assert "production_target_met" in value["adaptive_umbrella"]
                summaries.append(copy.deepcopy(value))
        return original_broadcast(payload)
    with monkeypatch.context() as patch:
        patch.setattr(execution, "broadcast", inspect_payload)
        sampled = run_fes(cfg, _system(), tmp_path)
        reanalyzed = reanalyze_fes(cfg, _system(), tmp_path)
    assert len(summaries) == 2
    assert sampled["adaptive_umbrella"] == reanalyzed["adaptive_umbrella"]
    assert "data" not in sampled and isinstance(sampled["streams"], int)
    full = reanalyze_fes(cfg, _system(), tmp_path, return_data=True)
    assert {"summary", "data", "streams"}.issubset(full)
    assert full["summary"]["adaptive_umbrella"] == sampled["adaptive_umbrella"]
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["adaptive_umbrella"] == sampled["adaptive_umbrella"]
