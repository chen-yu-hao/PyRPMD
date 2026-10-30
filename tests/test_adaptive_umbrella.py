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
    assert not (tmp_path / "checkpoints").exists()


def test_legacy_frozen_plan_preserves_samples_and_labels_its_original_ess_rule(tmp_path):
    cfg = _config(2)
    original = run_fes(cfg, _system(), tmp_path, return_data=True)
    plan = json.loads((tmp_path / "adaptive-plan.json").read_text())
    for report in plan["rounds"]:
        report.pop("effective_sample_observables")
    adaptive._save_plan(tmp_path, plan)  # emulate a checksum-valid pre-enhancement plan
    resumed = reanalyze_fes(cfg, _system(), tmp_path, return_data=True)
    np.testing.assert_array_equal(resumed["data"]["free_energy"], original["data"]["free_energy"])
    summary = resumed["summary"]
    assert summary["adaptive_umbrella"]["effective_sample_observables"].startswith("Legacy saved pilot")
    assert any("legacy coordinate-only pilot ESS" in warning for warning in summary["warnings"])
