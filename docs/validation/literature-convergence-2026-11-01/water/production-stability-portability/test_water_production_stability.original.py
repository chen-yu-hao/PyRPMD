"""Excluded synthetic cases for the additional descriptive production split."""

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.water import plot_production_stability as plotter
from benchmarks.water.campaign import sha256
from test_water_native_campaign_plot import synthetic_fixture


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    protocol, report = synthetic_fixture(tmp_path / "analysis")
    completed = []
    for index, chain in enumerate(report["chains"]):
        chain["manifest_sha256"] = hashlib.sha256(str(chain["seed"]).encode()).hexdigest()
        for segment in chain["segments"]:
            sign = -1 if segment["segment"] < 12 else 1 if segment["segment"] > 12 else 0
            segment["D_VACF_2ps"] = chain["mean_D_VACF_2ps"] + sign * (index + 1) * .0005
            segment["D_MSD_1_2ps"] = chain["mean_D_MSD_1_2ps"] + sign * (8 - index) * .001
        completed.append({key: chain[key] for key in ("seed", "manifest_sha256", "committed_steps", "complete")})
    (protocol.parent / "report.json").write_text(json.dumps(report, sort_keys=True))
    supplement = {"fixed_seeds": list(range(1101, 1109)), "early_segments": list(range(12)),
                  "late_segments": list(range(13, 25)), "original_protocol_sha256": sha256(protocol),
                  "frozen_analysis_sources_sha256": {name: sha256(plotter.ROOT / name) for name in (
                      "benchmarks/water/analyze_native.py", "benchmarks/water/plot_native_campaign.py")},
                  "statistics": {"interpretation": "EXCLUDED SYNTHETIC FIXTURE: descriptive paired t interval."}}
    supplement_path = tmp_path / "synthetic-supplement.json"
    supplement_path.write_text(json.dumps(supplement, sort_keys=True))
    monkeypatch.setattr(plotter, "SUPPLEMENT_SHA256", sha256(supplement_path))
    proof = {"schema": "pyrpmd.water-original-coordinator-normal-exit.v1", "process_exit_code": 0,
             "pool_drained": True, "all_original_leases_acquired_and_released": True,
             "coordinator_finished_event": {"event": "coordinator_finished", "failed": False},
             "completed_chains": completed}
    proof_path = tmp_path / "synthetic-exit-proof.json"
    proof_path.write_text(json.dumps(proof, sort_keys=True))
    return protocol.parent, protocol, supplement_path, proof_path


def test_paired_chain_variance_and_middle_retention_are_distinct():
    differences = np.arange(1, 9) * .003
    shared_offsets = np.arange(8) * 12.0
    values = np.repeat(shared_offsets[:, None], 25, axis=1)
    values[:, :12] -= differences[:, None] / 2
    values[:, 13:] += differences[:, None] / 2
    original = plotter.paired_halves(values)
    values[:, 12] -= 25.0
    revised = plotter.paired_halves(values)
    np.testing.assert_allclose(revised["paired_chain_differences"], differences, atol=1e-13)
    assert revised["paired_difference"]["chains"] == 8
    assert revised["paired_difference"]["degrees_of_freedom"] == 7
    assert revised["paired_difference"]["standard_error"] == pytest.approx(differences.std(ddof=1) / np.sqrt(8))
    assert revised["paired_difference"] == original["paired_difference"]
    np.testing.assert_allclose(np.asarray(revised["all25_chain_means"]) - original["all25_chain_means"], -1)


def test_missing_or_nonfinite_segment_is_not_excluded():
    with pytest.raises(ValueError, match="eight by 25"):
        plotter.paired_halves(np.zeros((8, 24)))
    values = np.zeros((8, 25))
    values[7, 12] = np.nan
    with pytest.raises(ValueError, match="finite"):
        plotter.paired_halves(values)


@pytest.mark.parametrize("fault", ["manifest", "exit", "leases"])
def test_completed_report_alone_does_not_establish_writer_shutdown(cohort, fault):
    plotter.load_diagnostics(*cohort)
    proof = json.loads(cohort[-1].read_text())
    if fault == "manifest":
        proof["completed_chains"][7]["manifest_sha256"] = "0" * 64
    elif fault == "exit":
        proof["process_exit_code"] = 1
    else:
        proof["all_original_leases_acquired_and_released"] = False
    cohort[-1].write_text(json.dumps(proof, sort_keys=True))
    with pytest.raises(ValueError):
        plotter.load_diagnostics(*cohort)


def test_plot_retains_every_segment_without_adding_a_gate(cohort, tmp_path):
    output = tmp_path / "figures"
    result = plotter.plot(*cohort, output)
    assert result["synthetic_data"] is True
    assert result["new_hard_gate"] is False
    assert result["literature_reproduced"] is False
    assert result["original_eligibility_checks"]["sampling_potential_per_molecule_hartree"] is False
    for label, expected in (("VACF", .0045), ("MSD", .009)):
        assert result["summaries"][label]["paired_difference"]["mean"] == pytest.approx(expected)
    with (output / "production-segment-means.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 400
    assert len([row for row in rows if row["segment"] == "12"]) == 16
    assert all(row["retained_in_primary"] == "True" for row in rows)
    assert all(sha256(output / name) == expected for name, expected in result["artifacts"].items())
