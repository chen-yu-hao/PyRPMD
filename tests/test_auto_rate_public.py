"""Small real-dynamics automatic jobs exercise public run and resume."""

import json
from pathlib import Path

import numpy as np

from pyrpmd.constants import KB_AU
from pyrpmd.input import JobSpec
from pyrpmd.runner import resume_run, run
from pyrpmd.system import System


def _checkpoints(directory):
    return {path: path.read_bytes() for path in directory.rglob("*.npz")}


def test_public_toy_auto_budget_exhaustion_resumes_without_repeating_pilots(tmp_path):
    source = Path(__file__).resolve().parents[1] / "examples" / "rate" / "automatic.inp"
    spec = JobSpec.from_file(source)
    spec.data.update(beads=1, seed=72)
    spec.data["dynamics"] = {"timestep": "0.01 fs", "loop_backend": "python"}
    spec.data["sampling"] = {"equilibration_steps": 2, "production_steps": 8,
                             "sample_interval": 1}
    spec.data["umbrella"] = {"centers": [-.05, .5, 1.1]}
    spec.data["adaptive_umbrella"] = {
        "pilot_steps": 8, "pilot_equilibration_steps": 2, "max_rounds": 1,
        "max_windows": 3, "min_effective_samples": 1e6,
    }
    output = tmp_path / "toy"
    result = run(spec, output=output)
    assert result["status"] == "auto_rate_incomplete"
    assert result["automatic_rate"]["reason"] == "aus_budget_exhausted"
    assert not result["sampling_converged"] and result["rate_cm3_s"] is None
    plan = json.loads((output / "umbrella" / "adaptive-plan.json").read_text())
    assert plan["status"] == "incomplete" and len(plan["rounds"]) == 1
    assert not (output / "umbrella" / "checkpoints").exists()
    assert not (output / "auto-rate" / "recrossing").exists()
    checkpoints = _checkpoints(output)
    assert len(checkpoints) == 6  # Three windows, two independent pilot replicas.
    manifest = (output / "manifest.json").read_bytes()
    resumed = resume_run(output)
    assert resumed["status"] == result["status"]
    assert resumed["automatic_rate"] == result["automatic_rate"]
    assert _checkpoints(output) == checkpoints
    assert (output / "manifest.json").read_bytes() == manifest
    summary = json.loads((output / "results" / "summary.json").read_text())
    assert summary["sampling_converged"] is False and summary["rate_cm3_s"] is None


def test_public_independent_pmf_stage_preserves_unfinished_window_tuning_on_resume(tmp_path):
    # A free model exercises sampling/checkpoint plumbing, not a physical H3 rate.
    system = System(("H", "H", "H"), np.array([2., 2., 2.]),
                    np.array([[-1.5, 0., 0.], [0., 0., 0.], [1.5, 0., 0.]]),
                    ((0,), (1, 2)))
    spec = JobSpec.from_mapping({
        "schema_version": 2, "task": "rate", "temperature": 1 / (5 * KB_AU),
        "beads": 2, "seed": 17, "native": False, "system": system,
        "potential": {"provider": "free"},
        "dynamics": {"timestep": .003, "loop_backend": "python"},
        "reaction": {"definition": "explicit", "fragments": [[0], [1, 2]],
                     "forming_bonds": [[0, 1]], "breaking_bonds": [[1, 2]], "rinf": 4.},
        "preparation": {"ts_action": "none", "check_channel": False},
        "rate": {
            "auto": {"target_relative_error": .01, "max_rounds": 2},
            "pmf_points": 41, "pmf_search_range": [0., 1.],
            "pmf": {
                "method": "umbrella", "estimator": "ui",
                "dynamics": {"timestep": .003},
                "sampling": {"equilibration_steps": 50, "production_steps": 200,
                             "sample_interval": 2, "andersen_interval": 20},
                "umbrella": {"centers": np.linspace(0, 1, 11).tolist(), "kappa": 10.},
                "adaptive_umbrella": {
                    "pilot_steps": 100, "pilot_equilibration_steps": 20,
                    "target_overlap": .0001, "min_effective_samples": 1., "max_rounds": 2,
                    "coverage_bins": [1], "coverage_ranges": [[0., 1.]],
                    "min_bin_effective_samples": .1,
                },
                "analysis": {"bins": 30, "bootstrap": 100, "ranges": [[-.2, 1.2]]},
            },
        },
        "output": {"checkpoint_steps": 50},
    })
    output = tmp_path / "independent-pmf"
    result = run(spec, output=output)
    assert result["status"] == "auto_rate_incomplete"
    assert result["automatic_rate"]["reason"] == "aus_budget_exhausted"
    assert result["rate_cm3_s"] is None and not result["sampling_converged"]
    plan = json.loads((output / "umbrella" / "adaptive-plan.json").read_text())
    assert plan["status"] == "incomplete" and plan["pilot_samples_used_in_production"] == 0
    assert plan["schema_version"] == 4
    assert len(plan["rounds"]) == 2
    assert any(r["stiffness_update"]["changed"] for r in plan["rounds"])
    assert not (output / "rate_pmf.json").exists()
    assert not (output / "umbrella" / "checkpoints").exists()
    checkpoints = _checkpoints(output)
    resumed = resume_run(output)
    assert resumed["automatic_rate"] == result["automatic_rate"]
    assert _checkpoints(output) == checkpoints
    assert not (output / "auto-rate" / "recrossing").exists()
