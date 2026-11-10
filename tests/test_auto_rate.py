"""Automatic PMF decisions, durable budgets and real stream continuation."""

from copy import deepcopy
import json

import numpy as np
import pytest

from pyrpmd.system import System
from pyrpmd.workflows import auto_rate
from pyrpmd.workflows.fes import sample_umbrellas


def _config():
    return {"task": "rate", "beta": 1., "temperature_K": 300., "beads": 1,
            "dt_au": .01, "seed": 7, "native": False,
            "potential": {"provider": "free"},
            "sampling": {"equilibration_steps": 5, "production_steps": 20,
                         "sample_interval": 1, "andersen_interval": 10, "replicas": 2},
            "umbrella": {"centers_au": [[0.], [.5], [1.]], "stiffness_au": [[1.]]},
            "adaptive_umbrella": {"enabled": True, "max_windows": 10},
            "rate": {"auto": {"max_rounds": 3, "target_relative_error": .1}},
            "recrossing": {"equilibration_steps": 5, "parent_interval_steps": 2,
                           "child_steps": 20},
            "output": {"checkpoint_steps": 10}}


def _pmf(error=.01, ess=40):
    return {"ready": True, "status": "ready", "warnings": [], "method": "ui",
            "barrier_standard_error_hartree": error,
            "free_energy_barrier_hartree": 0., "pmf_bootstrap_barriers": [-error, error],
            "uncertainty": {"available": True, "reasons": []},
            "effective_samples_per_stream": [ess] * 6}


def _state():
    config = _config()
    frozen = deepcopy(config)
    frozen.pop("adaptive_umbrella")
    return {"sampling_config": config, "frozen_config": frozen}


def test_pmf_decides_time_independent_trajectories_and_supported_precision():
    assert auto_rate._pmf_decision(_pmf(), _state(), _config())["action"] == "recrossing"
    assert auto_rate._pmf_decision(_pmf(.2, 5), _state(), _config())["action"] == "extend_time"
    assert auto_rate._pmf_decision(_pmf(.2, 40), _state(), _config())["action"] == "add_replicas"
    limited = _pmf(0., 40)
    limited["uncertainty"].update(block_status="limited_blocks", reasons=["pmf_bootstrap_limited_blocks"])
    assert auto_rate._pmf_decision(limited, _state(), _config())["action"] == "extend_time"


def test_window_coverage_uses_effective_samples_not_just_nonempty_bins():
    pmf = _pmf()
    pmf["ready"] = False
    pmf["adaptive_umbrella"] = {
        "production_coverage": {"target_met": False, "ranges": [[0., 1.]], "bins": [2],
                                "sample_counts": [100, 100], "effective_samples": [20., 1.],
                                "min_bin_effective_samples": 5.}}
    decision = auto_rate._pmf_decision(pmf, _state(), _config())
    assert decision["action"] == "refine_windows"
    assert .75 in decision["centers"]
    pmf["stream_state_checks"] = [{"passed": False}]
    assert auto_rate._pmf_decision(pmf, _state(), _config())["action"] == "stop"


def test_nonlinear_qtst_error_must_meet_precision_target():
    pmf = _pmf(.01)
    pmf["pmf_bootstrap_barriers"] = [-.4, .4]
    decision = auto_rate._pmf_decision(pmf, _state(), _config())
    assert decision["pmf_log_standard_error"] < decision["target"]
    assert decision["qtst_relative_standard_error"] > decision["target"]
    assert decision["action"] == "add_replicas"


def _mock_sampling(monkeypatch, *, interrupt=False, never_ready=False):
    calls = []
    stopped = False
    monkeypatch.setattr(auto_rate, "_prepare", lambda config, system, *a: (config, system))
    monkeypatch.setattr(auto_rate, "_production_hook", lambda *a: None)

    def sample(config, system, output, **kwargs):
        nonlocal stopped
        steps = kwargs.get("production_steps", config["sampling"]["production_steps"])
        calls.append((steps, kwargs.get("replicas", 2)))
        if interrupt and steps > 20 and not stopped:
            stopped = True
            raise RuntimeError("interrupted extension")
        frozen = deepcopy(config)
        frozen.pop("adaptive_umbrella", None)
        return {"sampling_config": frozen, "steps": steps}

    def analyze(sampled, *args):
        return (_pmf(.2, 5) if never_ready or sampled["steps"] == 20 else _pmf()), [{}]

    monkeypatch.setattr(auto_rate, "sample_umbrellas", sample)
    monkeypatch.setattr(auto_rate, "_pmf_and_parent_tasks", analyze)
    monkeypatch.setattr("pyrpmd.workflows.auto_recrossing.run_auto_recrossing",
                        lambda *a, **k: {"task": "rate", "status": "plateau_estimate",
                                         "sampling_converged": True, "rate_cm3_s": 1.,
                                         "converged": False})
    monkeypatch.setattr("pyrpmd.workflows.rate_subspace.finish_rate", lambda c, p, r: r)
    return calls


def test_resume_starts_latest_durable_budget_and_completed_run_does_no_sampling(tmp_path, monkeypatch):
    calls = _mock_sampling(monkeypatch, interrupt=True)
    system = System(("H",), np.array([1.]), np.zeros((1, 3)))
    config = _config()
    with pytest.raises(RuntimeError, match="interrupted extension"):
        auto_rate.run_auto_rate(config, system, tmp_path)
    saved = json.loads((tmp_path / "auto-rate/state.json").read_text())
    assert saved["production_steps"] == 40 and saved["round"] == 1
    config["parallel"] = {"backend": "serial", "threads_per_rank": 2}
    config["dynamics"] = {"loop_backend": "python", "block_steps": 16}
    result = auto_rate.run_auto_rate(config, system, tmp_path)
    assert result["sampling_converged"] and not result["converged"]
    assert calls == [(20, 2), (40, 2), (40, 2)]
    saved = json.loads((tmp_path / "auto-rate/state.json").read_text())
    assert saved["frozen_config"]["parallel"] == config["parallel"]
    assert saved["frozen_config"]["dynamics"]["block_steps"] == 16
    assert auto_rate.run_auto_rate(config, system, tmp_path) == result
    assert len(calls) == 3


def test_budget_exhaustion_never_runs_recrossing_or_publishes_rate(tmp_path, monkeypatch):
    calls = _mock_sampling(monkeypatch, never_ready=True)
    monkeypatch.setattr("pyrpmd.workflows.auto_recrossing.run_auto_recrossing",
                        lambda *a, **k: pytest.fail("recrossing started before PMF precision"))
    result = auto_rate.run_auto_rate(_config(), System(("H",), np.ones(1), np.zeros((1, 3))), tmp_path)
    assert calls == [(20, 2), (40, 2), (80, 2)]
    assert result["status"] == "auto_rate_incomplete"
    assert not result["sampling_converged"] and result["rate_cm3_s"] is None
    assert result["automatic_rate"]["reason"] == "pmf_budget_exhausted"


def test_optional_postprocessing_reads_an_already_published_unchanged_rate(tmp_path, monkeypatch):
    _mock_sampling(monkeypatch)
    source = []

    def finish(config, output, result):
        source.append((tmp_path / "summary.json").read_bytes())
        assert json.loads(source[0])["sampling_converged"]
        return {**result, "subspace_correction": {"status": "complete"}}

    monkeypatch.setattr("pyrpmd.workflows.rate_subspace.finish_rate", finish)
    result = auto_rate.run_auto_rate(_config(), System(("H",), np.ones(1), np.zeros((1, 3))), tmp_path)
    assert result["subspace_correction"]["status"] == "complete"
    assert (tmp_path / "summary.json").read_bytes() == source[0]


def test_production_coverage_failure_starts_an_independent_generation(tmp_path, monkeypatch):
    _mock_sampling(monkeypatch)
    sample = auto_rate.sample_umbrellas
    generations = []

    def record(config, system, output, **kwargs):
        generations.append((str(output), config["seed"], deepcopy(config["umbrella"])))
        return sample(config, system, output, **kwargs)

    def analyze(*args):
        pmf = _pmf()
        if len(generations) == 1:
            pmf["ready"] = False
            pmf["adaptive_umbrella"] = {"production_overlap_components": [0, 0, 1]}
        return pmf, [{}]

    monkeypatch.setattr(auto_rate, "sample_umbrellas", record)
    monkeypatch.setattr(auto_rate, "_pmf_and_parent_tasks", analyze)
    result = auto_rate.run_auto_rate(_config(), System(("H",), np.ones(1), np.zeros((1, 3))), tmp_path)
    assert result["sampling_converged"]
    assert len(generations) == 2
    assert generations[0][0] != generations[1][0]
    assert generations[0][1] != generations[1][1]
    assert len(generations[1][2]["centers_au"]) == 4
    assert [.75] in generations[1][2]["centers_au"]
    state = json.loads((tmp_path / "auto-rate/state.json").read_text())
    assert state["generation"] == 1
    assert [row["generation"] for row in state["history"]] == [0, 1]


def test_more_replicas_preserve_existing_checkpoints_and_extend_same_random_stream(tmp_path):
    config = _config()
    config["task"] = "fes"
    config.pop("rate")
    config.pop("adaptive_umbrella")
    config["cvs"] = [{"name": "x", "kind": "cartesian", "atom": 0, "axis": "x"}]
    config["umbrella"]["centers_au"] = [[0.]]
    system = System(("H",), np.array([1.]), np.zeros((1, 3)))
    original = sample_umbrellas(config, system, tmp_path / "resume")["streams"]
    original = list(original)
    checkpoints = {p: p.read_bytes() for p in (tmp_path / "resume/checkpoints").glob("*.npz")}
    expanded = list(sample_umbrellas(config, system, tmp_path / "resume", replicas=4)["streams"])
    assert len(expanded) == 4
    assert all(p.read_bytes() == content for p, content in checkpoints.items())
    for a, b in zip(original, expanded):
        np.testing.assert_array_equal(a["samples"], b["samples"])
    assert not np.array_equal(expanded[0]["samples"], expanded[2]["samples"])
    extended = list(sample_umbrellas(config, system, tmp_path / "resume",
                                    replicas=4, production_steps=40)["streams"])
    uninterrupted = list(sample_umbrellas(config, system, tmp_path / "full",
                                         replicas=4, production_steps=40)["streams"])
    for old, actual, expected in zip(expanded, extended, uninterrupted):
        np.testing.assert_array_equal(actual["samples"][:20], old["samples"])
        np.testing.assert_array_equal(actual["samples"], expected["samples"])
        np.testing.assert_array_equal(actual["last_q"], expected["last_q"])
    with pytest.raises(ValueError, match="replicas"):
        sample_umbrellas(config, system, tmp_path / "invalid", replicas=1)
