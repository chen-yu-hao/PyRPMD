"""Only the new scalar/RDF contracts; complete-cohort gates are tested separately."""

import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.water import plot_preparation_diagnostics as plotter
from benchmarks.water.campaign import sha256
from test_water_native_campaign_plot import synthetic_fixture


def prepare_fixture(directory):
    protocol_path, report = synthetic_fixture(directory)
    protocol = json.loads(protocol_path.read_text())
    limits = protocol["predeclared_eligibility_checks"]
    limits.update(max_split_mean_temperature_difference_K=5,
                  max_split_mean_OH_difference_angstrom=.001,
                  max_split_mean_angle_difference_degrees=.1)
    protocol_path.write_text(json.dumps(protocol, sort_keys=True))
    report["protocol_sha256"] = sha256(protocol_path)
    with np.load(directory / "curves.npz", allow_pickle=False) as data:
        curves = {key: data[key].copy() for key in data.files}
    radius = np.arange(.0125, 9, .025)
    base = np.where(radius > 2, 1 + 2 * np.exp(-((radius - 2.8) / .25)**2), 0)
    for index, chain in enumerate(report["chains"]):
        for key, _, _, _, _ in plotter.SCALARS:
            delta = chain["sampling"].get(key, {}).get("half_mean_difference", .0001 * index)
            chain["sampling"][key] = {"last_50ps_half_means": [0, delta],
                                      "half_mean_difference": delta,
                                      "half_window_observation_counts": [250, 250],
                                      "expected_observations_per_half_window": 250}
        prefix = f"seed_{chain['seed']}_"
        curves[prefix + "rdf_r_angstrom"] = radius.copy()
        # A large correlated chain offset cancels only in a paired interval.
        curves[prefix + "rdf_early_late"] = base + .08 * index
        curves[prefix + "rdf_late"] = base + .08 * index + .002 * index
    np.savez_compressed(directory / "curves.npz", **curves)
    report["artifacts"]["curves.npz"]["sha256"] = sha256(directory / "curves.npz")
    (directory / "report.json").write_text(json.dumps(report, sort_keys=True))
    return protocol_path, report, curves


def test_paired_rdf_bands_cancel_shared_chain_offsets_and_add_no_gate(tmp_path):
    protocol, _, _ = prepare_fixture(tmp_path / "data")
    result = plotter.plot(tmp_path / "data", protocol, tmp_path / "figures")
    assert result["synthetic_data"] is True
    assert result["RDF_hard_gate_adopted"] is False
    assert result["scalar_thresholds_changed"] is False
    assert result["diffusion_evidence"] is False
    assert result["fixed_seeds"] == list(range(1101, 1109))
    with (tmp_path / "figures/preparation-rdf.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    delta = .002 * np.arange(8)
    half = student_t.ppf(.975, 7) * delta.std(ddof=1) / np.sqrt(8)
    for row in rows:
        assert float(row["paired_difference_mean"]) == pytest.approx(delta.mean(), abs=1e-14)
        assert float(row["paired_difference_pointwise_CI95_halfwidth"]) == pytest.approx(half, abs=1e-14)
        assert all(f"seed_{seed}_late_minus_early" in row for seed in range(1101, 1109))


@pytest.mark.parametrize("invalid", ["different_grid", "nonfinite_rdf", "missing_window_sample"])
def test_new_preparation_data_contract_rejects_incomplete_or_incomparable_curves(tmp_path, invalid):
    protocol, report, curves = prepare_fixture(tmp_path)
    if invalid == "missing_window_sample":
        report["chains"][3]["sampling"]["mean_OH_angstrom"]["half_window_observation_counts"] = [249, 250]
        message = "all registered finite observations"
    else:
        if invalid == "different_grid":
            curves["seed_1104_rdf_r_angstrom"] += .001
        else:
            curves["seed_1104_rdf_late"][8] = np.nan
        np.savez_compressed(tmp_path / "curves.npz", **curves)
        report["artifacts"]["curves.npz"]["sha256"] = sha256(tmp_path / "curves.npz")
        message = "identical radial grid"
    (tmp_path / "report.json").write_text(json.dumps(report, sort_keys=True))
    with pytest.raises(ValueError, match=message):
        plotter.load_diagnostics(tmp_path, protocol)
