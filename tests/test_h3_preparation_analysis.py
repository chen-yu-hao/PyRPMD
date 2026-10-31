"""Independent counting/unknown-bound checks; no PES or trajectory simulation."""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
import unittest

import numpy as np
import pytest

from benchmarks.h3 import analyze_preparation_study as analysis

from benchmarks.h3.analyze_preparation_study import (
    Case, POSSIBLE, binomial_estimate, clopper_pearson, pair_cases, paired_estimate, paired_audit_means,
    unresolved_diagnostics,
)


class PreparationAnalysisTests(unittest.TestCase):
    def test_binomial_boundaries_and_unknown_denominator(self):
        low, high = clopper_pearson(0, 20)
        self.assertEqual(low, 0)
        self.assertAlmostEqual(high, 1-.025**(1/20))
        reverse = clopper_pearson(20, 20)
        self.assertAlmostEqual(reverse[0], 1-high)
        self.assertEqual(reverse[1], 1)
        result = binomial_estimate(["product", "nonreactive", "unresolved", "failed"], 8.)
        self.assertEqual(result["samples"], 4)
        self.assertEqual(result["unknown"], 2)
        self.assertEqual(result["sigma_identification_lower_angstrom2"], 2.)
        self.assertEqual(result["sigma_identification_upper_angstrom2"], 6.)

    def test_known_discordances_and_sign(self):
        a = ["nonreactive"]*4+["product"]*3
        b = ["product"]*2+["nonreactive"]*2+["nonreactive"]+["product"]*2
        result = paired_estimate(a, b, 7.)
        self.assertEqual(result["gained_reactivity_definite"], 2)
        self.assertEqual(result["lost_reactivity_definite"], 1)
        self.assertEqual(result["delta_sigma_angstrom2"], 1.)
        reverse = paired_estimate(b, a, 7.)
        self.assertAlmostEqual(reverse["delta_sigma_conservative_ci95_lower_angstrom2"],
                               -result["delta_sigma_conservative_ci95_upper_angstrom2"])
        self.assertAlmostEqual(reverse["delta_sigma_conservative_ci95_upper_angstrom2"],
                               -result["delta_sigma_conservative_ci95_lower_angstrom2"])

    def test_unknown_identification_bounds_exhaust_all_assignments(self):
        a = ["product", "nonreactive", "unresolved", "failed", "unresolved"]
        b = ["unresolved", "failed", "product", "nonreactive", "failed"]
        result = paired_estimate(a, b, 5.)
        gains, losses, differences = [], [], []
        assignments = [list(itertools.product(POSSIBLE[x], POSSIBLE[y])) for x, y in zip(a, b)]
        for completed in itertools.product(*assignments):
            plus = sum(x == 0 and y == 1 for x, y in completed)
            minus = sum(x == 1 and y == 0 for x, y in completed)
            gains.append(plus)
            losses.append(minus)
            differences.append(plus-minus)
            aa = ["product" if x else "nonreactive" for x, _ in completed]
            bb = ["product" if y else "nonreactive" for _, y in completed]
            resolved = paired_estimate(aa, bb, 5.)
            self.assertLessEqual(result["delta_sigma_conservative_ci95_lower_angstrom2"],
                                 resolved["delta_sigma_conservative_ci95_lower_angstrom2"])
            self.assertGreaterEqual(result["delta_sigma_conservative_ci95_upper_angstrom2"],
                                    resolved["delta_sigma_conservative_ci95_upper_angstrom2"])
        self.assertEqual(result["gained_reactivity_definite"], min(gains))
        self.assertEqual(result["gained_reactivity_possible"], max(gains))
        self.assertEqual(result["lost_reactivity_definite"], min(losses))
        self.assertEqual(result["lost_reactivity_possible"], max(losses))
        self.assertEqual(result["delta_sigma_identification_lower_angstrom2"], min(differences))
        self.assertEqual(result["delta_sigma_identification_upper_angstrom2"], max(differences))
        self.assertIsNone(result["delta_sigma_angstrom2"])

    @staticmethod
    def case(variant, ids=(10, 11)):
        config = {"method": "qct" if variant.startswith("qct_") else "ne_rpmd", "v": 0,
                  "total_energy_ev": .55, "beads": 1 if variant.startswith("qct_") else 32,
                  "samples": len(ids), "seed": 9, "bmin": 0., "bmax": 3.5,
                  "bond_cutoff_bohr": 2.5, "beta_override": None}
        summary = {"omega_au": .02, "collision_energy_ev": .28, "beta_au": 280.}
        rows = {i: {"id": i, "status": "nonreactive", "b_bohr": i/10,
                    "initial_axial_bohr": 10., "initial_bond_bohr": 1.4,
                    "initial_rg_bohr": [.1, .2, .2]} for i in ids}
        return Case(Path("unused"), variant, config, summary, rows, [])

    @staticmethod
    def audit_case(variant, values=None):
        values = {10: 100., 11: 200.} if values is None else values
        case = PreparationAnalysisTests.case(variant, tuple(values))
        case.summary = {}
        spec = analysis._expected_preparation(variant)
        start, duration = spec["switch_time_fs"], spec["hold_time_fs"]
        case.rows = {}
        for identity, value in values.items():
            if value is None:
                case.rows[identity] = {"id": identity, "status": "failed", "reason": "test preparation failure"}
                continue
            means = {key: value for key in analysis.NATIVE_HOLD_COLUMNS}
            native = {"status": "completed", "switch_time_fs": start, "hold_time_fs": duration,
                      "accepted_hold_duration_fs": duration, "dt_fs": .02, "trace_stride_fs": .2, "omega_au": .02,
                      "hold_time_average": dict(means)}
            windows = {}
            for name, source in analysis.AUDIT_WINDOWS.items():
                length = 50. if name == "final_50fs" else duration
                windows[name] = {"source": source, "start_fs": start+duration-length,
                                 "end_fs": start+duration, "duration_fs": length, "values": dict(means)}
            windows["hold"].update(native_dt_fs=.02, native_steps=5000)
            case.rows[identity] = {"id": identity, "status": "prepared", "preparation_study": {
                "kind": spec["kind"], "conditioning": spec["conditioning"], "energy_convention": spec["energy_convention"],
                "collision_energy_ev": .55-.01/analysis.EV_TO_AU, "nominal_internal_label_ev": .01/analysis.EV_TO_AU,
                "switch": native, "time_averages": windows,
                "before": {"primitive_full_ev": value}, "after": {"primitive_full_ev": value},
                "trace": [{"time_fs": time, "primitive_internal_per_bead_hartree": value}
                          for time in (start, start+duration)]}}
        return case

    def test_pairing_checks_ids_and_serialized_fields(self):
        a, b = self.case("ne_paper_subset"), self.case("ne_uncut")
        result = pair_cases(a, b, "check", same_conditioning=False)
        self.assertEqual(result["paired_samples"], 2)
        b.rows[10]["initial_axial_bohr"] += .1
        self.assertEqual(pair_cases(a, b, "check", same_conditioning=False)
                         ["serialized_launch_checks"]["initial_axial_bohr"]["different"], 1)
        with self.assertRaisesRegex(ValueError, "initial_axial"):
            pair_cases(a, b, "check", same_conditioning=True)
        b.rows[10]["b_bohr"] += .1
        with self.assertRaisesRegex(ValueError, "b_bohr"):
            pair_cases(a, b, "check", same_conditioning=False)
        with self.assertRaisesRegex(ValueError, "equal ID sets"):
            pair_cases(a, self.case("ne_uncut", (11, 12)), "check", same_conditioning=False)

    def test_endpoint_derivative_uses_centroid_momentum(self):
        case = self.case("ne_uncut", (10,))
        q = np.array([[[0., 0., 0.], [3., 0., 0.], [0., 4., 0.]]])
        # Expansion r_dot = r / tau with tau=2 atomic time units.
        mass = 1837.1526474396187
        case.rows[10].update(status="unresolved", endpoint_q_bohr=q.tolist(),
                             endpoint_p_au=(mass*q/2).tolist())
        result = unresolved_diagnostics(case)[0]
        self.assertTrue(result["all_centroid_pair_distances_increasing"])
        self.assertTrue(result["all_centroid_pairs_beyond_bond_cutoff"])
        for row in result["centroid_pairs"]:
            self.assertAlmostEqual(row["separation_derivative_bohr_per_atomic_time"],
                                   row["centroid_distance_bohr"]/2)
        self.assertEqual(case.rows[10]["status"], "unresolved")
        self.assertFalse(result["status_changed"])

    def test_audit_sem_is_of_paired_means_and_failures_are_counted(self):
        a = self.audit_case("direct-none", {10: 100., 11: 200.})
        b = self.audit_case("switch200-none", {11: 203., 10: 105.})
        row = next(row for row in paired_audit_means(a, b, "check") if row["column"] == "bond_spread_bohr")
        self.assertEqual(row["successful_pairs"], 2)
        self.assertEqual(row["mean_difference"], 4.)
        self.assertAlmostEqual(row["sem_difference"], 1.)
        b.rows[10]["status"] = "failed"
        row = next(row for row in paired_audit_means(a, b, "check") if row["column"] == "bond_spread_bohr")
        self.assertEqual(row["pairs_with_a_failure"], 1)
        self.assertEqual(row["successful_pairs"], 1)
        self.assertEqual(row["mean_difference"], 3.)
        self.assertIsNone(row["sem_difference"])


def test_single_successful_audit_keeps_mean_without_zero_sem_band(tmp_path, monkeypatch):
    import pytest
    pytest.importorskip("matplotlib")
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from benchmarks.h3 import analyze_preparation_study as analysis

    case = PreparationAnalysisTests.audit_case("direct-none", {10: .01, 11: None})
    case.rows[10]["preparation_study"]["trace"][1]["primitive_internal_per_bead_hartree"] = .02
    bands = []
    original = Axes.fill_between

    def record(ax, *args, **kwargs):
        band = original(ax, *args, **kwargs)
        bands.append((ax, band))
        return band

    monkeypatch.setattr(Axes, "fill_between", record)
    monkeypatch.setattr(Figure, "savefig", lambda *args, **kwargs: None)
    assert analysis.audit_plot([case], {}, tmp_path)
    assert len(bands) == 1
    ax, band = bands[0]
    np.testing.assert_allclose(ax.lines[0].get_ydata(), np.array([.01, .02])/analysis.EV_TO_AU)
    assert len(band.get_paths()) == 0, "Unavailable SEM must not become a zero-error band"


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e999", '{"value": 1, "value": 2}'])
def test_json_rejects_nonfinite_and_ambiguous_numbers(literal):
    with pytest.raises(ValueError, match="Nonfinite|Duplicate"):
        analysis._strict_json(literal, "fixture")


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), True])
def test_successful_audit_cannot_silently_drop_a_mean(value):
    case = PreparationAnalysisTests.audit_case("direct-none")
    case.rows[10]["preparation_study"]["time_averages"]["hold"]["values"]["bond_spread_bohr"] = value
    with pytest.raises(ValueError, match="numeric|integral"):
        analysis.audit_statistics(case)


def test_successful_audit_requires_all_columns_and_consistent_single_frames():
    case = PreparationAnalysisTests.audit_case("direct-none")
    del case.rows[10]["preparation_study"]["time_averages"]["hold"]["values"]["bond_spread_bohr"]
    with pytest.raises(ValueError, match="columns"):
        analysis.audit_statistics(case)
    case = PreparationAnalysisTests.audit_case("direct-none")
    case.rows[10]["preparation_study"]["after"]["missing_in_other_row_ev"] = 1.
    with pytest.raises(ValueError, match="columns"):
        analysis.audit_statistics(case)


@pytest.mark.parametrize("field,value", [("duration_fs", 50.), ("end_fs", 250.), ("source", "dense_trace_trapezoid"),
                                         ("native_dt_fs", .1), ("native_steps", True)])
def test_paired_audits_refuse_mismatched_hold_provenance(field, value):
    a = PreparationAnalysisTests.audit_case("direct-none")
    b = PreparationAnalysisTests.audit_case("switch200-none")
    b.rows[10]["preparation_study"]["time_averages"]["hold"][field] = value
    with pytest.raises(ValueError, match="window|source|timestep"):
        paired_audit_means(a, b, "test")


def test_zero_success_audits_and_pairs_retain_every_failure():
    a = PreparationAnalysisTests.audit_case("direct-none", {10: None, 11: None})
    b = PreparationAnalysisTests.audit_case("switch200-none", {10: .1, 11: None})
    assert analysis.audit_statistics(a) == []
    assert analysis.audit_case_record(a)["attempts"] == 2
    assert analysis.audit_case_record(a)["failed"] == 2
    assert paired_audit_means(a, b, "test") == []
    outcome = analysis.audit_pair_record(a, b, "test")
    assert {key: outcome[key] for key in ("attempted_pairs", "successful_pairs", "pairs_with_a_failure",
                                         "reference_failed", "variant_failed", "both_failed")} == {
        "attempted_pairs": 2, "successful_pairs": 0, "pairs_with_a_failure": 2,
        "reference_failed": 2, "variant_failed": 1, "both_failed": 1}


@pytest.mark.parametrize("field,value", [("dt_fs", 2.), ("beta_override", 9.), ("integrator", "different"),
                                         ("bond_cutoff_bohr", 5.), ("extended_cap_fs", 2000.)])
def test_collision_pair_rejects_changes_beyond_preparation(field, value):
    a, b = PreparationAnalysisTests.case("ne_uncut"), PreparationAnalysisTests.case("ne_switch200")
    b.config[field] = value
    with pytest.raises(ValueError, match=field):
        pair_cases(a, b, "check", same_conditioning=True)


def test_legacy_optional_override_and_declared_subset_are_compatible():
    a, b = PreparationAnalysisTests.case("ne_paper_subset"), PreparationAnalysisTests.case("ne_uncut")
    a.config["samples"] = 4096
    b.config["collision_energy_override_au"] = None
    assert pair_cases(a, b, "check", same_conditioning=False)["physical_configuration_verified"]
    b.variant = "ne_switch200"
    with pytest.raises(ValueError, match="samples"):
        pair_cases(a, b, "check", same_conditioning=False)


def test_only_declared_action_total_control_can_change_collision_energy():
    a, b = PreparationAnalysisTests.case("qct_action_collision"), PreparationAnalysisTests.case("qct_action_total")
    b.config["collision_energy_override_au"] = .02
    b.summary["collision_energy_ev"] = .02/analysis.EV_TO_AU
    assert pair_cases(a, b, "check", same_conditioning=True)["paired_samples"] == 2
    b.variant = "qct_harmonic"
    with pytest.raises(ValueError, match="collision energies"):
        pair_cases(a, b, "check", same_conditioning=True)


def test_redundant_nominal_collision_override_does_not_break_pairing():
    a, b = PreparationAnalysisTests.case("ne_uncut"), PreparationAnalysisTests.case("ne_switch200")
    b.config["collision_energy_override_au"] = b.summary["collision_energy_ev"]*analysis.EV_TO_AU
    assert pair_cases(a, b, "check", same_conditioning=True)["physical_configuration_verified"]
    b.config["collision_energy_override_au"] *= 2
    with pytest.raises(ValueError, match="override"):
        pair_cases(a, b, "check", same_conditioning=True)


def _write_case(path, case, *, audit=False):
    path.mkdir(parents=True)
    config = {**case.config, "id_offset": min(case.rows)}
    summary = {**case.summary, "config": config}
    if audit:
        from benchmarks.h3.preparation_audit import summarize_time_averages
        preparation = analysis._expected_preparation(case.variant)
        reduction = summarize_time_averages(case.rows.values())
        summary.update({key: reduction[key] for key in ("attempts", "prepared", "failed")})
        summary["preparation"] = preparation
        for name, data in (("preparation.json", preparation), ("provenance.json", {"study_spec": preparation}),
                           ("audit_time_averages.json", reduction)):
            (path/name).write_text(json.dumps(data))
    else:
        summary.update(launched_attempts=len(case.rows), counts={s: sum(r["status"] == s for r in case.rows.values())
                                                               for s in analysis.STATUSES})
    (path/"config.json").write_text(json.dumps(config))
    (path/"summary.json").write_text(json.dumps(summary))
    (path/"trajectories.jsonl").write_text("\n".join(json.dumps(row) for row in case.rows.values())+"\n")


def test_relocated_completed_audit_validates_counts_and_reduction(tmp_path):
    case = PreparationAnalysisTests.audit_case("direct-none", {10: .1, 11: None})
    path = tmp_path/"extracted"/"data"/"state_preparation_study"/"isolated_audit"/"case"
    _write_case(path, case, audit=True)
    loaded = analysis.load_case(path, case.variant, audit=True)
    assert analysis.audit_case_record(loaded)["failed"] == 1
    assert all(Path(row["path"]).is_relative_to(path) for row in loaded.sources)
    summary_path = path/"summary.json"
    summary = json.loads(summary_path.read_text())
    summary.update(prepared=2, failed=0)
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="count mismatch"):
        analysis.load_case(path, case.variant, audit=True)


def test_completed_audit_cannot_relabel_or_supply_stale_means(tmp_path):
    case = PreparationAnalysisTests.audit_case("direct-none", {10: .1, 11: .3})
    path = tmp_path/"case"
    _write_case(path, case, audit=True)
    with pytest.raises(ValueError, match="variant"):
        analysis.load_case(path, "switch200-none", audit=True)
    mean_path = path/"audit_time_averages.json"
    data = json.loads(mean_path.read_text())
    data["windows"]["hold"]["values"]["bond_spread_bohr"]["mean"] = 100.
    mean_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="disagrees with ledger"):
        analysis.load_case(path, case.variant, audit=True)


@pytest.mark.parametrize("field", ["duration_fs", "native_difference", "missing_difference"])
def test_completed_audit_revalidates_window_duration_and_native_trace_crosscheck(tmp_path, field):
    case = PreparationAnalysisTests.audit_case("direct-none", {10: .1, 11: .3})
    path = tmp_path/"case"
    _write_case(path, case, audit=True)
    mean_path = path/"audit_time_averages.json"
    data = json.loads(mean_path.read_text())
    if field == "duration_fs":
        data["windows"]["hold"]["duration_fs"] = 999.
    elif field == "native_difference":
        data["native_minus_trace_hold"]["bond_spread_bohr"]["mean"] = 999.
    else:
        del data["native_minus_trace_hold"]
    mean_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="reduction"):
        analysis.load_case(path, case.variant, audit=True)


def test_successful_ne_collision_energy_must_match_summary():
    case = PreparationAnalysisTests.case("ne_uncut", (10,))
    spec = analysis._expected_preparation(case.variant)
    harmonic = .01/analysis.EV_TO_AU
    collision = case.config["total_energy_ev"]-harmonic
    case.summary.update(preparation=spec, collision_energy_ev=collision)
    case.rows[10]["preparation_study"] = {"kind": "direct", "conditioning": "none",
        "energy_convention": "fixed_collision", "collision_energy_ev": collision,
        "nominal_internal_label_ev": harmonic, "before": {"energy_ev": .1}, "after": {"energy_ev": .2}}
    analysis._validate_preparation(case, spec)
    case.rows[10]["preparation_study"]["collision_energy_ev"] *= 2
    with pytest.raises(ValueError, match="collision energy disagrees"):
        analysis._validate_preparation(case, spec)


def test_completed_collision_ledger_cannot_hide_nonfinite_diagnostics(tmp_path):
    case = PreparationAnalysisTests.case("ne_paper_full")
    case.rows[10]["max_observed_energy_drift_ev"] = float("nan")
    path = tmp_path/"case"
    _write_case(path, case)
    with pytest.raises(ValueError, match="Nonfinite"):
        analysis.load_case(path, case.variant)


def test_output_generation_preserves_previous_results_on_failure_and_clears_stale_data(tmp_path):
    output = tmp_path/"analysis"
    output.mkdir()
    csv = output/"audit_statistics.csv"
    pdf = output/"pyrpmd_state_preparation_report.pdf"
    custom = output/"custom_export.pdf"
    csv.write_text("previous means\n")
    pdf.write_bytes(b"previous generated PDF")
    custom.write_bytes(b"user export")
    with pytest.raises(RuntimeError, match="plot failed"):
        with analysis._output_generation(output) as stage:
            analysis._csv(stage/csv.name, [])
            raise RuntimeError("plot failed")
    assert csv.read_text() == "previous means\n"
    assert pdf.exists()
    with analysis._output_generation(output) as stage:
        analysis._csv(stage/csv.name, [])
        analysis._json(stage/"preparation_analysis.json", {"current": True})
    assert csv.read_text() == ""
    assert not pdf.exists()
    assert custom.read_bytes() == b"user export"


def test_all_failed_audit_plot_replaces_stale_image_with_explicit_placeholder(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure
    captured = []
    def save(figure, path, **kwargs):
        captured.extend(text.get_text() for ax in figure.axes for text in ax.texts)
        Path(path).write_bytes(b"current empty-data figure")
    monkeypatch.setattr(Figure, "savefig", save)
    image = tmp_path/"preparation_energy_hold.png"
    image.write_bytes(b"old populated figure")
    case = PreparationAnalysisTests.audit_case("direct-none", {10: None, 11: None})
    assert analysis.audit_plot([case], {}, tmp_path) is False
    assert image.read_bytes() == b"current empty-data figure"
    assert any("all failed" in text for text in captured)


if __name__ == "__main__":
    unittest.main()
