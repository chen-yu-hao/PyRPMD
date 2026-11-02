"""Scientific eligibility and uncertainty gates for the complete-cohort plot."""

from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.water import plot_native_campaign as plotter
from benchmarks.water.campaign import sha256
from benchmarks.water.native_reference import Config
from pyrpmd.constants import KCAL_MOL_TO_AU


def synthetic_fixture(directory):
    """Analytic curves explicitly marked synthetic; no molecular dynamics."""
    directory.mkdir(parents=True, exist_ok=True)
    protocol = {"fixed_stage": {"seeds": list(range(1101, 1109)), "NVE_segments_per_chain": 25},
                "predeclared_eligibility_checks": {"max_split_mean_potential_difference_kcal_mol_water": .05},
                "literature_comparison": {"reference": .199, "standard_error": .002}}
    protocol_path = directory / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, sort_keys=True))
    chains, arrays = [], {}
    lag = np.linspace(0, 2, 401)
    for index, seed in enumerate(protocol["fixed_stage"]["seeds"]):
        gk, msd = .15 + .01*index, .154 + .01*index
        config = Config(seed=seed)
        chains.append({"seed": seed, "complete": True, "config": asdict(config), "committed_steps": config.total_steps,
                       "incomplete_segment": None,
                       "segments": [{"segment": number, "included_in_mean": True} for number in range(25)],
                       "mean_D_VACF_2ps": gk, "mean_D_MSD_1_2ps": msd,
                       "sampling": {"potential_per_molecule_hartree": {
                           "half_mean_difference": (-.16 if index == 1 else .005*index) * KCAL_MOL_TO_AU}}})
        arrays[f"seed_{seed}_lag_time_ps"] = lag
        arrays[f"seed_{seed}_mean_D_VACF_angstrom2_ps"] = gk * (1 - np.exp(-lag / .04))
        arrays[f"seed_{seed}_mean_msd_angstrom2"] = 6 * msd * lag
    np.savez_compressed(directory / "curves.npz", **arrays)
    report = {"schema": "pyrpmd.water-native-chain-analysis.v1", "synthetic_data": True,
              "external_reference": True, "pyrpmd_production_trajectory": False,
              "all_predeclared_chains_complete": True, "protocol_sha256": sha256(protocol_path),
              "chains": chains, "eligibility_checks": {"fixed_stage_complete": True,
                                                         "sampling_potential_per_molecule_hartree": False},
              "literature_comparison": None, "source_sha256": {"synthetic_fixture": sha256(Path(__file__))},
              "artifacts": {"curves.npz": {"sha256": sha256(directory / "curves.npz")}}}
    for label, key in (("VACF", "mean_D_VACF_2ps"), ("MSD", "mean_D_MSD_1_2ps")):
        report[f"diffusion_{label}_angstrom2_ps"] = {"chains": 8, "degrees_of_freedom": 7,
                                                   **plotter.independent_summary([row[key] for row in chains])}
    (directory / "report.json").write_text(json.dumps(report, sort_keys=True))
    return protocol_path, report


@pytest.fixture
def cohort(tmp_path):
    protocol, report = synthetic_fixture(tmp_path)
    return tmp_path, protocol, report


def save_report(cohort, report):
    (cohort[0] / "report.json").write_text(json.dumps(report, sort_keys=True))


def test_partial_cohort_is_rejected_before_curves_or_reference_are_read(cohort):
    directory, protocol, report = cohort
    report["all_predeclared_chains_complete"] = False
    save_report(cohort, report)
    (directory / "curves.npz").unlink()
    with pytest.raises(ValueError, match="complete all eight"):
        plotter.load_complete_cohort(directory, protocol)


def test_uncertainty_is_across_eight_means_not_two_hundred_segments(cohort):
    summary = plotter.independent_summary(np.arange(8, dtype=float))
    assert summary["mean"] == 3.5
    assert summary["standard_error"] == pytest.approx(np.sqrt(.75))
    directory, protocol, report = cohort
    report["diffusion_VACF_angstrom2_ps"]["standard_error"] /= np.sqrt(25)
    save_report(cohort, report)
    with pytest.raises(ValueError, match="independent-chain means"):
        plotter.load_complete_cohort(directory, protocol)


@pytest.mark.parametrize("change", ["duplicate_seed", "dropped_segment", "missing_preparation", "different_preparation_duration"])
def test_incomplete_or_duplicated_evidence_cannot_be_aggregated(cohort, change):
    directory, protocol, report = cohort
    if change == "duplicate_seed":
        report["chains"][-1]["seed"] = 1101
    elif change == "dropped_segment":
        report["chains"][0]["segments"].pop()
    elif change == "missing_preparation":
        report["chains"][0]["sampling"]["potential_per_molecule_hartree"]["half_mean_difference"] = None
    else:
        config = report["chains"][0]["config"]
        config["equilibration_steps"] *= 5
        report["chains"][0]["committed_steps"] = config["equilibration_steps"] + config["segment_steps"] * config["segments"]
    save_report(cohort, report)
    with pytest.raises(ValueError):
        plotter.load_complete_cohort(directory, protocol)


def test_curve_tampering_is_rejected(cohort):
    directory, protocol, report = cohort
    with (directory / "curves.npz").open("ab") as stream:
        stream.write(b"changed artifact")
    with pytest.raises(ValueError, match="curve checksum"):
        plotter.load_complete_cohort(directory, protocol)


def test_curve_endpoint_must_match_the_recorded_chain_estimate(cohort):
    directory, protocol, report = cohort
    report["chains"][0]["mean_D_VACF_2ps"] += .1
    report["diffusion_VACF_angstrom2_ps"].update(plotter.independent_summary(
        [row["mean_D_VACF_2ps"] for row in report["chains"]]))
    save_report(cohort, report)
    with pytest.raises(ValueError, match="curve endpoint"):
        plotter.load_complete_cohort(directory, protocol)


def test_failed_preparation_keeps_paper_comparison_disabled(cohort):
    directory, protocol, report = cohort
    loaded = plotter.load_complete_cohort(directory, protocol)
    assert loaded["paper_allowed"] is False
    report["literature_comparison"] = {"invented": "eligible despite failed check"}
    save_report(cohort, report)
    with pytest.raises(ValueError, match="comparison eligibility"):
        plotter.load_complete_cohort(directory, protocol)
