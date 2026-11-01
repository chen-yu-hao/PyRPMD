"""Reproduce longer-budget follow-ups of both failed endpoint-voltage P=4 cases.

Run from the repository root after the nine-point full study. The original
parameter ordering, seed, model, stiffness and diagnostic thresholds stay fixed.
The initial failed result remains in the sibling full-study evidence directory.
"""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/electrochemistry"))
import adaptive_nrpmd_study as study


def main():
    output = Path(__file__).resolve().parent
    original = ROOT / "docs/validation/adaptive-nrpmd-study-2026-11-01"
    original_reports = [original / name / "report.json" for name in ("Um0_6_P4", "Up0_6_P4")]
    failed = [json.loads(path.read_text()) for path in original_reports]
    if any("target_effective_samples_below_threshold" not in item["production_reweighting"]["unavailable_reasons"] for item in failed):
        raise ValueError("these fixed follow-ups require both recorded original effective-sample failures")
    config = replace(study.StudyConfig(**failed[0]["config"]), pilot_samples=512,
                     pilot_burn_in=256, production_samples=2048,
                     production_burn_in=512)
    plan = {
        "reason": "both_original_endpoint_voltage_P4_cases_failed_positive_effective_sample_gate",
        "selection": "all_points_failing_original_positive_effective_sample_gate; no_seed_or_threshold_selection",
        "selected_cases": [{"potential_V": voltage, "beads": 4} for voltage in (-0.6, 0.6)],
        "original_report_sha256": {path.parent.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in original_reports},
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config": asdict(config), "source_identity": study.source_identity(),
    }
    study._write_json(output / "plan.json", plan)
    cases = []
    for voltage in (-0.6, 0.6):
        report = study.run_case(asdict(config), voltage, 4, output, resume=True)
        keys = ("point_id", "potential_V", "beads", "observables", "production_reweighting",
                "bootstrap", "independent_chain_results", "charge", "timing_seconds")
        case = {key: report[key] for key in keys}
        case.update(report=report["point_id"]+"/report.json",
                    dynamics_available=report["dynamics"]["available"],
                    path_stopping_reason=report["path"]["stopping_reason"])
        cases.append(case)
        print(report["point_id"], report["production_reweighting"]["unavailable_reasons"], flush=True)
    study._write_json(output / "summary.json", {
        "schema_version": 1, "plan": plan, "cases": cases,
        "physical_converged": False, "rate_constant": None,
    })


if __name__ == "__main__":
    main()
