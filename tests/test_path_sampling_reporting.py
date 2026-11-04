"""Path-sampling reports retain candidate rates and their exact qualifications."""

import json

import pytest

from pyrpmd.analysis.reporting import collect_run, generate_report


def saved_run(directory):
    (directory / "input").mkdir(parents=True)
    (directory / "results").mkdir()
    forbidden = directory / "unavailable_model.py"
    forbidden.write_text("raise AssertionError('report imported input code')\n")
    config = {
        "task": "path_sampling", "beads": 1, "temperature_K": 300., "dt_au": .04,
        "potential": {"provider": "python", "factory": str(forbidden) + ":build"},
        "path_sampling": {"method": "repptis", "interfaces": [-.8, -.4]},
    }
    summary = {
        "task": "path_sampling", "method": "repptis", "status": "complete",
        "samples": 100, "flux_au": .1, "flux_s_inv": 4.1e15,
        "overall_crossing_probability": .3,
        "rate_estimate_au": .03, "rate_estimate_s_inv": 1.2e15,
        "rate_s_inv": None, "converged": False, "physical_converged": False,
        "rate_standard_error_block_conditional_on_flux_au": .004,
        "qualification": "Memory loss beyond neighboring interfaces must be checked.",
        "uncertainty_qualification": "Supplied-flux uncertainty is excluded.",
        "interfaces": [{"interface": -.8, "p_LR": .4, "p_RL": .7}],
    }
    (directory / "input" / "resolved.json").write_text(json.dumps(config))
    (directory / "results" / "summary.json").write_text(json.dumps(summary))
    return summary


def test_saved_path_sampling_report_never_constructs_provider(tmp_path, monkeypatch):
    import pyrpmd.potentials

    def forbidden(*args, **kwargs):
        raise AssertionError("report constructed a potential")

    monkeypatch.setattr(pyrpmd.potentials, "make_potential", forbidden)
    summary = saved_run(tmp_path / "run")
    report = collect_run(tmp_path / "run")
    metrics = report["metrics"]
    for key in ("method", "interfaces", "flux_au", "rate_estimate_au", "rate_estimate_s_inv",
                "rate_s_inv", "overall_crossing_probability",
                "rate_standard_error_block_conditional_on_flux_au", "physical_converged"):
        assert metrics[key] == summary[key]
    assert "rate_standard_error_au" not in metrics
    assert metrics["rate_s_inv"] is None
    codes = {entry["code"] for entry in report["findings"]}
    assert {"path_sampling_rate_withheld", "path_sampling_qualification", "path_sampling_uncertainty"} <= codes
    assert report["converged"] is False


def test_exported_report_contains_path_sampling_candidate_and_conditional_error(tmp_path):
    pytest.importorskip("matplotlib")
    summary = saved_run(tmp_path / "run")
    output = tmp_path / "report"
    generate_report(tmp_path / "run", output, formats=("svg",))
    report = json.loads((output / "report.json").read_text())
    metrics = report["runs"][0]["metrics"]
    assert metrics["rate_estimate_au"] == summary["rate_estimate_au"]
    assert metrics["rate_standard_error_block_conditional_on_flux_au"] == .004
    assert metrics["rate_s_inv"] is None
    assert "rate_standard_error_au" not in metrics
    assert "Supplied-flux uncertainty is excluded." in (output / "report.md").read_text()
