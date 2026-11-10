"""Adaptive recrossing decisions preserve support and restart semantics."""

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import json

import numpy as np
import pytest

from pyrpmd.system import System
from pyrpmd.workflows import auto_recrossing as auto


def setup():
    system = System(
        ("H", "H", "H"), np.array([2., 2., 2.]),
        np.array([[-1.5, 0., 0.], [0., 0., 0.], [1.5, 0., 0.]]), ((0,), (1, 2)),
    )
    config = {
        "task": "rate", "beta": 10., "beads": 1, "dt_au": 1e-4, "seed": 7,
        "native": False, "potential": {"provider": "free"}, "sampling": {},
        "reaction": {"fragments": [[0], [1, 2]], "forming_bonds": [[0, 1]],
                     "breaking_bonds": [[1, 2]], "rinf": 10.},
        "recrossing": {"equilibration_steps": 0, "parent_interval_steps": 1,
                       "parent_replicas": 2, "parent_samples": 10,
                       "child_steps": 40, "bootstrap": 200},
    }
    pmf = {
        "ready": True, "status": "ready", "method": "ui", "xi_star": 1.,
        "warnings": [], "k_qtst_cm3_s": 12., "free_energy_barrier_hartree": .01,
        "pmf_bootstrap_barriers": [.01] * 20,
        "uncertainty": {"available": True, "reasons": []},
        "bootstrap": {"requested": 20, "attempted": 20, "valid": 20,
                      "block_plan": {"status": "adequate"}},
    }
    tasks = [{"chain": 0, "initial_q": system.geometry[None].copy(), "xi_star": 1.}]
    return config, system, pmf, tasks


def run(directory, *, max_rounds=3, **changes):
    config, system, pmf, tasks = setup()
    pmf.update(changes)
    return auto.run_auto_recrossing(config, system, directory, pmf, tasks,
                                    target_relative_error=.1, max_rounds=max_rounds)


def test_real_free_flux_converges_and_complete_replay_never_propagates(tmp_path, monkeypatch):
    result = run(tmp_path)
    assert result["sampling_converged"]
    assert not result["converged"]
    assert result["rate_cm3_s"] == pytest.approx(12.)
    assert result["parent_configurations"] == 20
    assert result["auto_recrossing"]["rounds"] == 1
    arrays = (tmp_path / "recrossing.npz").read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("replayed a completed recrossing ensemble")

    monkeypatch.setattr(auto, "_parent_task", forbidden)
    monkeypatch.setattr(auto, "_dispatch_children", forbidden)
    assert run(tmp_path) == result
    assert (tmp_path / "recrossing.npz").read_bytes() == arrays


def drifting_children(tasks, config, dispatch):
    records = []
    for task in tasks:
        n = task["child_steps"]
        numerator = np.ones(n)
        # Every first-round pair returns in the middle of the selected tail.
        if n == 40:
            numerator[35:] = 0.
        records.append({"chain": task["chain"], "parent": task["parent"],
                        "pair": task["pair"], "numerator": numerator,
                        "denominator": 1., "energy_drift": 0.,
                        "endpoint_energy_error_hartree_per_bead": 0.,
                        "endpoint_energy_error_kbt": 0.})
    return records


def test_time_extension_uses_fresh_round_and_does_not_pool_old_pairs(tmp_path, monkeypatch):
    monkeypatch.setattr(auto, "_dispatch_children", drifting_children)
    result = run(tmp_path)
    assert result["sampling_converged"]
    assert result["auto_recrossing"]["rounds"] == 2
    assert result["auto_recrossing"]["options"]["child_steps"] == 80
    assert result["child_pairs"] == 40
    first = json.loads((tmp_path / "auto-rate/recrossing/round-001/report.json").read_text())
    assert first["rate_cm3_s"] is None
    assert first["auto_recrossing"]["actions"] == ["extend_child_time"]
    # The same origins and deterministic momenta are refinements, not another
    # independent dataset that can be concatenated with round two.
    assert result["parent_configurations"] == 20


def test_budget_exhaustion_withholds_formal_rate(tmp_path, monkeypatch):
    monkeypatch.setattr(auto, "_dispatch_children", drifting_children)
    result = run(tmp_path, max_rounds=1)
    assert not result["sampling_converged"]
    assert result["status"] == "auto_rate_budget_exhausted"
    assert result["rate_cm3_s"] is None
    assert result["rate_estimate_cm3_s"] > 0


def test_unsupported_pmf_uncertainty_stops_without_false_success(tmp_path):
    result = run(tmp_path, uncertainty={"available": False, "reasons": ["no_blocks"]})
    assert not result["sampling_converged"]
    assert result["rate_cm3_s"] is None
    assert result["auto_recrossing"]["rounds"] == 1
    assert "pmf_uncertainty_unavailable" in result["auto_recrossing"]["quality"]["reasons"]


def test_resume_rejects_changed_pmf_and_missing_committed_arrays(tmp_path):
    run(tmp_path)
    with pytest.raises((ValueError, RuntimeError), match="another PMF"):
        run(tmp_path, free_energy_barrier_hartree=.02)
    (tmp_path / "auto-rate/recrossing/round-001/recrossing.npz").unlink()
    with pytest.raises((ValueError, RuntimeError), match="missing its saved arrays"):
        run(tmp_path)


def test_interruption_reuses_parent_checkpoints(tmp_path, monkeypatch):
    original = auto._dispatch_children

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("after constrained parents")

    monkeypatch.setattr(auto, "_dispatch_children", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path)
    path = tmp_path / "auto-rate/recrossing/round-001/checkpoints/parent_0000.npz"
    with np.load(path) as saved:
        original_snapshots = saved["snapshots"].copy()
    monkeypatch.setattr(auto, "_dispatch_children", original)
    result = run(tmp_path)
    assert result["sampling_converged"]
    with np.load(path) as saved:
        np.testing.assert_array_equal(saved["snapshots"], original_snapshots)


def test_noise_requests_samples_not_longer_children_and_tunes_parent_spacing():
    config, _, _, _ = setup()
    options = auto._recrossing_options(config)
    options.update(parent_replicas=4, parent_samples=16, child_pairs=2)
    result = {
        "kappa": .5, "kappa_standard_error": .1, "plateau_drift": 0.,
        "plateau_detected": False,
        "plateau_diagnostics": {"tail_samples": 10, "absolute_tolerance": .05,
                                "maximum_tail_deviation": .01},
        "rate_uncertainty": {"parent_block_lengths": {"0": 3},
                             "recrossing_support": {"independent_parent_chains": 4,
                                                     "nonzero_flux_parent_configurations": 64}},
    }
    changed, actions = auto._next_options(result, [], options, .1)
    assert changed["child_steps"] == options["child_steps"]
    assert changed["parent_samples"] > options["parent_samples"]
    assert changed["parent_interval_steps"] == 3 * options["parent_interval_steps"]
    assert "increase_parent_spacing" in actions


@pytest.mark.parametrize("radius, drift, drift_error, extends_time", [
    (.20, 0., .05, False),
    (.20, .15, .10, False),
    (.20, .15, .01, True),
    (0., 0., 0., True),
])
def test_resolved_tail_shape_distinguishes_time_from_sampling(radius, drift, drift_error, extends_time):
    config, _, _, _ = setup()
    options = auto._recrossing_options(config)
    result = {
        "kappa": .5, "kappa_standard_error": .1, "plateau_drift": drift,
        "plateau_drift_standard_error": drift_error, "plateau_detected": False,
        "plateau_diagnostics": {"tail_samples": 10, "absolute_tolerance": .05,
                                "maximum_tail_deviation": .15,
                                "bootstrap_95_envelope_radius": radius},
        "rate_uncertainty": {"parent_block_lengths": {"0": 1},
                             "recrossing_support": {"independent_parent_chains": 2,
                                                     "nonzero_flux_parent_configurations": 20}},
    }
    changed, actions = auto._next_options(result, [], options, .1)
    assert ("extend_child_time" in actions) == extends_time
    assert changed["child_steps"] == options["child_steps"] * (2 if extends_time else 1)
    if not extends_time:
        assert changed["parent_samples"] > options["parent_samples"]


def test_parent_chain_heuristic_never_reduces_user_counts_above_its_cap():
    config, _, _, _ = setup()
    options = auto._recrossing_options(config)
    options["parent_replicas"] = 32
    result = {
        "kappa": .5, "kappa_standard_error": .1,
        "plateau_diagnostics": {"tail_samples": 10},
        "rate_uncertainty": {"parent_block_lengths": {"0": options["parent_samples"]},
                             "recrossing_support": {"independent_parent_chains": 32,
                                                     "nonzero_flux_parent_configurations": 320}},
    }
    changed, actions = auto._next_options(result, [], options, .1)
    assert changed["parent_replicas"] == 32
    assert "increase_parent_chains" not in actions


def test_nested_variance_allocation_prefers_pairs_only_for_within_parent_noise():
    config, _, _, _ = setup()
    options = auto._recrossing_options(config)
    options.update(child_pairs=2, parent_samples=20)
    within = [{"chain": p // 10, "parent": p, "pair": pair,
               "numerator": np.full(40, float(pair)), "denominator": 1.}
              for p in range(20) for pair in range(2)]
    between = deepcopy(within)
    for record in between:
        record["numerator"][:] = float(record["parent"] % 2)
    assert auto._pair_allocation(within, options, .5) > options["child_pairs"]
    assert auto._pair_allocation(between, options, .5) == 1


def test_executor_iterators_keep_parents_children_and_propagation(tmp_path):
    config, system, pmf, tasks = setup()
    with ThreadPoolExecutor(max_workers=2) as executor:
        result = auto.run_auto_recrossing(config, system, tmp_path, pmf, tasks,
                                         target_relative_error=.1, max_rounds=2, executor=executor)
    assert result["sampling_converged"]
    assert result["parent_configurations"] == 20
    assert len(result["propagation"]["parents"]) == 2
    assert sum(item["python_steps"] for item in result["propagation"]["parents"]) == 20
    assert len(result["propagation"]["children"]) == 40


def test_user_starting_counts_are_recorded_without_hidden_floors(tmp_path):
    config, system, pmf, tasks = setup()
    requested = dict(parent_replicas=1, parent_samples=2, child_pairs=1,
                     child_steps=20, bootstrap=100)
    config["recrossing"].update(requested)
    result = auto.run_auto_recrossing(config, system, tmp_path, pmf, tasks,
                                     target_relative_error=.1, max_rounds=1)
    plan = json.loads((tmp_path / "auto-rate/recrossing/round-001/plan.json").read_text())
    for key, value in requested.items():
        assert plan["options"][key] == value
    assert result["child_trajectories"] == 4
    assert not result["sampling_converged"]
    assert result["auto_recrossing"]["next_options"]["parent_replicas"] == 2


def test_trpmd_is_rejected_before_any_round_is_created(tmp_path):
    config, system, pmf, tasks = setup()
    config["recrossing"]["dynamics"] = "trpmd"
    with pytest.raises(ValueError, match="requires RPMD"):
        auto.run_auto_recrossing(config, system, tmp_path, pmf, tasks,
                                 target_relative_error=.1, max_rounds=2)
    assert not (tmp_path / "auto-rate").exists()


def test_unpublished_rate_cannot_be_declared_converged(tmp_path):
    result = run(tmp_path)
    result["rate_cm3_s"] = None
    quality = auto._quality(result, .1)
    assert not quality["passed"]
    assert "rate_publication_prerequisites_not_met" in quality["reasons"]


def test_nonfinite_diagnostics_are_null_and_replay_identically(tmp_path, monkeypatch):
    aggregate = auto._aggregate

    def with_missing_diagnostic(*args, **kwargs):
        result = aggregate(*args, **kwargs)
        result["unsupported_diagnostic"] = float("nan")
        return result

    monkeypatch.setattr(auto, "_aggregate", with_missing_diagnostic)
    result = run(tmp_path)
    assert result["unsupported_diagnostic"] is None
    assert run(tmp_path) == result
    report = json.loads((tmp_path / "auto-rate/recrossing/round-001/report.json").read_text())
    assert report["unsupported_diagnostic"] is None
