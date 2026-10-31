"""Analyze the controlled H + H2 preparation study without running dynamics.

Only completed case directories (a final summary plus a complete ledger) enter
the estimates. Paired comparisons use exactly the same trajectory IDs, retain
unresolved/failed attempts, and distinguish nominal labels from collision and
full-PES action energies. This module never modifies simulation inputs or data.
"""
from __future__ import annotations

import argparse
import base64
import csv
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy.stats import beta as beta_distribution

from pyrpmd.constants import BOHR_ANGSTROM, EV_TO_AU, FS_TO_AU

ROOT = Path(__file__).resolve().parents[2]
POINTS = ((0, .55), (1, .90), (1, 1.10))
AREA_FACTOR = BOHR_ANGSTROM ** 2
MASS_AU = 1837.1526474396187
STATUSES = ("product", "nonreactive", "unresolved", "failed")
POSSIBLE = {"product": (1,), "nonreactive": (0,), "unresolved": (0, 1), "failed": (0, 1)}
LABELS = {
    "ne_paper_subset": "Paper-conditioned NE: paired subset",
    "ne_paper_full": "Paper-conditioned NE: full original sample",
    "ne_uncut": "Uncut NE",
    "ne_switch200": "Uncut NE + 200 fs switch",
    "qct_harmonic": "Harmonic-prepared QCT",
    "qct_action_collision": "Full-PES action QCT: fixed collision energy",
    "qct_action_total": "Full-PES action QCT: fixed actual asymptotic total energy",
}
AUDIT_LABELS = {
    "direct-paper": "Paper-conditioned, sudden",
    "direct-none": "Uncut, sudden",
    "switch50-none": "Uncut, 50 fs switch",
    "switch200-none": "Uncut, 200 fs switch",
}
COLORS = {"ne_paper_subset": "#55657c", "ne_uncut": "#c76b30",
          "ne_switch200": "#147d91", "qct_harmonic": "#55657c",
          "qct_action_collision": "#4e8a43", "qct_action_total": "#8558a5"}
AUDIT_COLORS = dict(zip(AUDIT_LABELS, ("#55657c", "#c76b30", "#a68637", "#147d91")))
NATIVE_HOLD_COLUMNS = frozenset(("primitive_internal_per_bead_hartree",
    "kinetic_per_bead_hartree", "full_potential_per_bead_hartree",
    "spring_per_bead_hartree", "centroid_harmonic_energy_hartree", "bond_spread_bohr"))
AUDIT_WINDOWS = {"hold": "native_every_step_trapezoid",
                 "hold_trace_check": "dense_trace_trapezoid", "final_50fs": "dense_trace_trapezoid"}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(text, source):
    """Reject ambiguous keys and every nonfinite JSON number, including 1e999."""
    def number(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"Nonfinite JSON number in {source}: {value}")
        return result
    def constant(value):
        raise ValueError(f"Nonfinite JSON number in {source}: {value}")
    def mapping(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key in {source}: {key}")
            result[key] = value
        return result
    try:
        return json.loads(text, parse_constant=constant, parse_float=number, object_pairs_hook=mapping)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {source}: {exc}") from exc


def _read_json(path):
    return _strict_json(Path(path).read_bytes(), path)


def _finite(value, context):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Missing or nonfinite numeric {context}")
    return float(value)


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _csv(path, rows):
    if not rows:
        # Empty current results must not leave a prior generation's statistics.
        Path(path).write_text("")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, separators=(",", ":")) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()} for row in rows)


@contextmanager
def _output_generation(output):
    """Generate in isolation; failed validation/plotting leaves old results intact."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="preparation-analysis-", dir=output.parent) as temporary:
        stage = Path(temporary)
        (stage/"figures").mkdir()
        yield stage
        files = sorted((path for path in stage.rglob("*") if path.is_file()),
                       key=lambda path: (path.name == "preparation_analysis.json", str(path)))
        for path in files:
            target = output/path.relative_to(stage)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(path, target)
        # This exact managed PDF belongs to the previous analysis. Custom
        # user-named exports and all unrelated files are preserved.
        (output/"pyrpmd_state_preparation_report.pdf").unlink(missing_ok=True)


def clopper_pearson(k, n, alpha=.05):
    """Two-sided exact binomial interval, including k=0 and k=n."""
    if not isinstance(k, (int, np.integer)) or not isinstance(n, (int, np.integer)) or not 0 <= k <= n or n < 1:
        raise ValueError("Require integer counts 0 <= k <= n, n >= 1")
    if not 0 < alpha < 1:
        raise ValueError("Require 0 < alpha < 1")
    return (0. if k == 0 else float(beta_distribution.ppf(alpha/2, k, n-k+1)),
            1. if k == n else float(beta_distribution.ppf(1-alpha/2, k+1, n-k)))


def binomial_estimate(statuses, area):
    """Identification and 95% confidence bounds with unknown outcomes retained."""
    statuses = list(statuses)
    if not statuses or any(s not in STATUSES for s in statuses) or not math.isfinite(area) or area <= 0:
        raise ValueError("Invalid status sample or area")
    n = len(statuses)
    counts = {s: statuses.count(s) for s in STATUSES}
    k, unknown = counts["product"], counts["unresolved"] + counts["failed"]
    low, high = clopper_pearson(k, n)
    possible_high = clopper_pearson(k+unknown, n)[1]
    return {"samples": n, **counts, "unknown": unknown,
            "sigma_confirmed_angstrom2": area*k/n,
            "sigma_identification_lower_angstrom2": area*k/n,
            "sigma_identification_upper_angstrom2": area*(k+unknown)/n,
            "sigma_confirmed_ci95_lower_angstrom2": area*low,
            "sigma_confirmed_ci95_upper_angstrom2": area*high,
            "sigma_ci95_including_unknown_lower_angstrom2": area*low,
            "sigma_ci95_including_unknown_upper_angstrom2": area*possible_high}


def paired_estimate(reference_statuses, variant_statuses, area):
    """Conservative paired difference, variant minus reference.

    Let p+ = P(reference=0, variant=1), p- = P(reference=1, variant=0).
    Each gets a two-sided 97.5% Clopper-Pearson interval. Their simultaneous
    coverage is at least 95% by Bonferroni, so area*[L+ - U-, U+ - L-]
    covers the difference. Unknown observations expand each event count from
    its definite to its possible count; no pair is deleted or set to zero.
    """
    first, second = list(reference_statuses), list(variant_statuses)
    if len(first) != len(second) or not first or any(s not in POSSIBLE for s in first+second):
        raise ValueError("Paired status samples must have the same nonzero length")
    if not math.isfinite(area) or area <= 0:
        raise ValueError("Invalid area")
    plus_definite = plus_possible = minus_definite = minus_possible = 0
    empirical_low = empirical_high = 0
    table = {f"{a}|{b}": 0 for a in STATUSES for b in STATUSES}
    unknown_pairs = 0
    for a, b in zip(first, second):
        av, bv = POSSIBLE[a], POSSIBLE[b]
        table[f"{a}|{b}"] += 1
        plus_definite += av == (0,) and bv == (1,)
        plus_possible += 0 in av and 1 in bv
        minus_definite += av == (1,) and bv == (0,)
        minus_possible += 1 in av and 0 in bv
        empirical_low += min(bv)-max(av)
        empirical_high += max(bv)-min(av)
        unknown_pairs += len(av) > 1 or len(bv) > 1
    n = len(first)
    plus = (clopper_pearson(plus_definite, n, .025)[0],
            clopper_pearson(plus_possible, n, .025)[1])
    minus = (clopper_pearson(minus_definite, n, .025)[0],
             clopper_pearson(minus_possible, n, .025)[1])
    return {"paired_samples": n, "unknown_pairs": unknown_pairs,
            "gained_reactivity_definite": plus_definite, "gained_reactivity_possible": plus_possible,
            "lost_reactivity_definite": minus_definite, "lost_reactivity_possible": minus_possible,
            "delta_sigma_angstrom2": area*empirical_low/n if not unknown_pairs else None,
            "delta_sigma_identification_lower_angstrom2": area*empirical_low/n,
            "delta_sigma_identification_upper_angstrom2": area*empirical_high/n,
            "delta_sigma_conservative_ci95_lower_angstrom2": area*(plus[0]-minus[1]),
            "delta_sigma_conservative_ci95_upper_angstrom2": area*(plus[1]-minus[0]),
            "gain_probability_ci97_5": list(plus), "loss_probability_ci97_5": list(minus),
            "status_transition_counts": {key: count for key, count in table.items() if count},
            "interval_scope": "Per comparison; no adjustment across the 12 different comparisons; unknowns included"}


@dataclass
class Case:
    path: Path
    variant: str
    config: dict
    summary: dict
    rows: dict[int, dict]
    sources: list[dict]

    @property
    def key(self):
        return self.variant, int(self.config["v"]), round(float(self.config["total_energy_ev"]), 6)

    @property
    def area(self):
        return math.pi * (self.config["bmax"]**2-self.config["bmin"]**2)*AREA_FACTOR

    def subset(self, ids, variant):
        ids = set(ids)
        if not ids or not ids.issubset(self.rows):
            raise ValueError("Requested exact paired IDs are absent from the original case")
        return replace(self, variant=variant, rows={i: self.rows[i] for i in sorted(ids)})


def _expected_preparation(variant):
    if variant in ("ne_paper_full", "ne_paper_subset", "qct_harmonic"):
        return None  # Historical baseline predates PreparationSpec.
    if variant not in LABELS and variant not in AUDIT_LABELS:
        raise ValueError(f"Unknown preparation variant: {variant}")
    audit = variant in AUDIT_LABELS
    switch = 50. if variant == "switch50-none" else 200. if variant in ("switch200-none", "ne_switch200") else 0.
    action = variant in ("qct_action_collision", "qct_action_total")
    return {"kind": "action_qct" if action else "switch" if switch else "direct",
            "conditioning": "paper" if variant == "direct-paper" else "none",
            "switch_time_fs": switch, "hold_time_fs": 100. if audit else 0.,
            "preparation_dt_fs": .02, "trace_stride_fs": .2,
            "energy_convention": "fixed_total" if variant == "qct_action_total" else "fixed_collision"}


def _validate_preparation(case, preparation):
    expected = _expected_preparation(case.variant)
    if expected is None:
        return
    if preparation != expected or case.summary.get("preparation") != preparation:
        raise ValueError(f"Preparation metadata does not match variant {case.variant}: {case.path}")
    if "collision_energy_ev" in case.summary:
        collision = _finite(case.summary["collision_energy_ev"], "summary collision energy")
        harmonic = (case.config["v"]+.5)*_finite(case.summary.get("omega_au"), "summary omega")/EV_TO_AU
        if collision <= 0:
            raise ValueError(f"Nonpositive collision energy: {case.path}")
        override = case.config.get("collision_energy_override_au")
        if override is not None and not math.isclose(_finite(override, "collision override")/EV_TO_AU,
                                                     collision, rel_tol=2e-12, abs_tol=1e-12):
            raise ValueError(f"Collision override disagrees with summary: {case.path}")
        if preparation["energy_convention"] == "fixed_collision" and not math.isclose(
                collision+harmonic, case.config["total_energy_ev"], rel_tol=2e-12, abs_tol=1e-12):
            raise ValueError(f"Fixed-collision energy convention mismatch: {case.path}")
    for row in case.rows.values():
        if row["status"] == "failed":
            continue
        study = row.get("preparation_study", {})
        if any(study.get(key) != preparation[key] for key in ("kind", "conditioning", "energy_convention")):
            raise ValueError(f"Successful row has inconsistent preparation metadata: {case.path}, ID {row['id']}")
        row_collision = _finite(study.get("collision_energy_ev"), f"collision energy, ID {row['id']}")
        if "collision_energy_ev" not in case.summary:
            # Isolated-audit summaries omit scattering energies; the recorded
            # native frequency defines their original harmonic launch label.
            harmonic = (case.config["v"]+.5)*_finite(study.get("switch", {}).get("omega_au"), "audit native omega")/EV_TO_AU
            collision = case.config["total_energy_ev"]-harmonic
        if row_collision <= 0 or not math.isclose(row_collision, collision, rel_tol=2e-12, abs_tol=1e-12):
            raise ValueError(f"Successful row collision energy disagrees with its configuration/summary: {case.path}, ID {row['id']}")
        if not math.isclose(_finite(study.get("nominal_internal_label_ev"), "nominal internal label"), harmonic,
                            rel_tol=2e-12, abs_tol=1e-12):
            raise ValueError(f"Successful row harmonic energy label mismatch: {case.path}, ID {row['id']}")
        for stage in ("before", "after"):
            values = study.get(stage)
            if not isinstance(values, dict) or not values:
                raise ValueError(f"Successful row lacks {stage} diagnostics: {case.path}, ID {row['id']}")
            for key, value in values.items():
                _finite(value, f"{stage}.{key}, ID {row['id']}")
        if preparation["switch_time_fs"] or preparation["hold_time_fs"]:
            native = study.get("switch", {})
            if native.get("status") != "completed" or any(
                    native.get(key) != preparation[spec_key] for key, spec_key in
                    (("switch_time_fs", "switch_time_fs"), ("hold_time_fs", "hold_time_fs"),
                     ("dt_fs", "preparation_dt_fs"), ("trace_stride_fs", "trace_stride_fs"))):
                raise ValueError(f"Successful row has incomplete/mismatched native preparation: {case.path}, ID {row['id']}")
        if preparation["kind"] == "action_qct":
            for key in ("orbit_energy_ev", "actual_asymptotic_total_energy_ev", "orbit_phase", "action_quantum_number"):
                _finite(study.get(key), f"{key}, ID {row['id']}")
            if (study["action_quantum_number"] != case.config["v"]+.5 or not 0 <= study["orbit_phase"] < 1 or
                    not math.isclose(study["orbit_energy_ev"]+collision, study["actual_asymptotic_total_energy_ev"],
                                     rel_tol=2e-12, abs_tol=1e-12)):
                raise ValueError(f"Action state/energy metadata mismatch: {case.path}, ID {row['id']}")
            if preparation["energy_convention"] == "fixed_total" and not math.isclose(
                    study["actual_asymptotic_total_energy_ev"], case.config["total_energy_ev"], rel_tol=2e-12, abs_tol=1e-12):
                raise ValueError(f"Fixed-total action energy mismatch: {case.path}, ID {row['id']}")


def load_case(path, variant, *, audit=False):
    """A summary is the completion marker; in-flight ledgers are not analyzed."""
    path = Path(path)
    summary_path = path/"summary.json"
    if not summary_path.exists():
        return None
    summary = _read_json(summary_path)
    config_path, ledger_path = path/"config.json", path/"trajectories.jsonl"
    config = _read_json(config_path)
    if summary["config"] != config:
        raise ValueError(f"Summary/config mismatch: {path}")
    for key, minimum in (("id_offset", 0), ("samples", 1), ("beads", 1)):
        if type(config.get(key)) is not int or config[key] < minimum:
            raise ValueError(f"Invalid {key} in completed case: {path}")
    expected_method = "qct" if variant.startswith("qct_") else "ne_rpmd"
    if config.get("method") != expected_method or config["beads"] != (1 if expected_method == "qct" else 32):
        raise ValueError(f"Method/beads do not match variant {variant}: {path}")
    rows = {}
    with ledger_path.open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = _strict_json(line, f"{ledger_path}, line {number}")
            if not isinstance(row, dict):
                raise ValueError(f"Ledger row is not an object: {ledger_path}, line {number}")
            i = row.get("id")
            if type(i) is not int or i in rows:
                raise ValueError(f"Duplicate or invalid trajectory ID in {ledger_path}")
            rows[i] = row
    expected = set(range(config["id_offset"], config["id_offset"]+config["samples"]))
    if set(rows) != expected:
        raise ValueError(f"Final ledger does not contain its exact declared ID range: {path}")
    if audit:
        if any(r.get("status") not in ("prepared", "failed") for r in rows.values()):
            raise ValueError(f"Invalid audit status: {path}")
        totals = {"attempts": len(rows), "prepared": sum(r["status"] == "prepared" for r in rows.values()),
                  "failed": sum(r["status"] == "failed" for r in rows.values())}
        if any(type(summary.get(key)) is not int or summary[key] != value for key, value in totals.items()):
            raise ValueError(f"Audit count mismatch: {path}")
    else:
        if any(r.get("status") not in STATUSES for r in rows.values()):
            raise ValueError(f"Invalid collision status: {path}")
        counts = {s: sum(r["status"] == s for r in rows.values()) for s in STATUSES}
        if counts != summary["counts"] or len(rows) != summary["launched_attempts"]:
            raise ValueError(f"Summary/ledger outcome mismatch: {path}")
    source_paths = [summary_path, config_path, ledger_path]
    for name in ("preparation.json", "provenance.json", "audit_time_averages.json"):
        if (path/name).exists():
            source_paths.append(path/name)
    case = Case(path, variant, config, summary, rows,
                [{"path": str(p.resolve()), "sha256": _sha(p)} for p in source_paths])
    if _expected_preparation(variant) is not None:
        preparation = _read_json(path/"preparation.json")
        _validate_preparation(case, preparation)
        provenance = _read_json(path/"provenance.json")
        if provenance.get("study_spec") != preparation:
            raise ValueError(f"Preparation/provenance mismatch: {path}")
    if audit:
        _validate_audit_case(case)
        _validate_audit_reduction(case, _read_json(path/"audit_time_averages.json"))
    return case


def case_record(case):
    records = list(case.rows.values())
    config = case.config
    initial = [r["preparation_study"] for r in records if "preparation_study" in r]
    actual_total = [r["actual_asymptotic_total_energy_ev"] for r in initial if "actual_asymptotic_total_energy_ev" in r]
    internal = [r["orbit_energy_ev"] for r in initial if "orbit_energy_ev" in r]
    for data in (actual_total, internal):
        if data and not np.allclose(data, data[0], atol=2e-12, rtol=0):
            raise ValueError(f"Action energy not constant within {case.path}")
    omega = float(case.summary["omega_au"])
    return {"variant": case.variant, "label": LABELS[case.variant], "v": config["v"],
            "nominal_total_label_ev": config["total_energy_ev"],
            "collision_energy_ev": case.summary["collision_energy_ev"],
            "harmonic_internal_label_ev": (config["v"]+.5)*omega/EV_TO_AU,
            "action_internal_energy_ev": internal[0] if internal else None,
            "actual_asymptotic_total_energy_ev": actual_total[0] if actual_total else None,
            "energy_scope": ("constant full-PES semiclassical action orbit" if internal else
                             "full-PES action preparation attempted; no orbit energy is available" if case.variant.startswith("qct_action_") else
                             "nominal harmonic label; not a constant full-PES internal energy"),
            "beads": config["beads"], "beta_au": case.summary["beta_au"],
            "id_min": min(case.rows), "id_max": max(case.rows),
            "bmin_bohr": config["bmin"], "bmax_bohr": config["bmax"], "area_angstrom2": case.area,
            **binomial_estimate((r["status"] for r in records), case.area),
            "preparation_rejections": sum(r.get("preparation_draws", 1)-1 for r in records),
            "max_observed_energy_drift_ev": max((r["max_observed_energy_drift_ev"] for r in records
                                                 if "max_observed_energy_drift_ev" in r), default=None),
            "native_steps": sum(r.get("native_steps", 0) for r in records),
            "python_steps": sum(r.get("python_steps", 0) for r in records),
            "path": str(case.path.resolve())}


def _validate_pair_config(reference, variant, *, audit=False):
    """Permit only the two differences explicitly present in this study design."""
    configs = [dict(case.config) for case in (reference, variant)]
    for config in configs:
        # This optional field did not exist in the frozen baseline schema.
        config.setdefault("collision_energy_override_au", None)
    variants = {reference.variant, variant.variant}
    ignored = set()
    if not audit and variants == {"ne_paper_subset", "ne_uncut"}:
        ignored.add("samples")  # subset() deliberately preserves full-run config.
    retuned = not audit and variants == {"qct_action_collision", "qct_action_total"}
    if retuned:
        ignored.add("collision_energy_override_au")
    if not audit:
        for case, config in zip((reference, variant), configs):
            override = config["collision_energy_override_au"]
            if override is not None:
                collision = _finite(case.summary.get("collision_energy_ev"), "paired collision energy")
                if not math.isclose(_finite(override, "paired collision_energy_override_au"), collision*EV_TO_AU,
                                    rel_tol=2e-12, abs_tol=1e-14):
                    raise ValueError("Paired collision_energy_override_au disagrees with its declared collision energy")
                if not retuned:
                    # An explicit override equal to the nominal collision
                    # energy has the same physical meaning as the default.
                    config["collision_energy_override_au"] = None
    for key in sorted((configs[0].keys() | configs[1].keys())-ignored):
        if key not in configs[0] or key not in configs[1] or configs[0][key] != configs[1][key]:
            raise ValueError(f"Paired physical/RNG configuration mismatch for {key}")
    for key in ("omega_au", "beta_au"):
        if reference.summary.get(key) != variant.summary.get(key):
            raise ValueError(f"Paired derived physical configuration mismatch for {key}")
    if not audit and not retuned and reference.summary["collision_energy_ev"] != variant.summary["collision_energy_ev"]:
        raise ValueError("Paired collision energies differ outside the declared fixed-total control")


def pair_cases(reference, variant, comparison, *, same_conditioning):
    if set(reference.rows) != set(variant.rows):
        raise ValueError("Paired comparison requires exactly equal ID sets, not sample-size matching")
    _validate_pair_config(reference, variant)
    checks = {key: {"checked": 0, "missing": 0, "different": 0} for key in
              ("b_bohr", "initial_axial_bohr", "initial_bond_bohr", "incoming_A_ring_radius_bohr")}
    final_radius_max_difference = 0.
    pairs = [(reference.rows[i], variant.rows[i]) for i in sorted(reference.rows)]
    for first, second in pairs:
        for key in checks:
            if key == "incoming_A_ring_radius_bohr":
                a = first.get("initial_rg_bohr", [None])[0]
                b = second.get("initial_rg_bohr", [None])[0]
            else:
                a, b = first.get(key), second.get(key)
            if a is None or b is None:
                checks[key]["missing"] += 1
                if ((a is None and first["status"] != "failed") or
                        (b is None and second["status"] != "failed")):
                    raise ValueError(f"Successful launch lacks paired field {key}, ID {first['id']}")
                continue
            _finite(a, f"paired {key}")
            _finite(b, f"paired {key}")
            checks[key]["checked"] += 1
            if a != b:
                checks[key]["different"] += 1
                if key == "b_bohr" or same_conditioning:
                    raise ValueError(f"Paired launch field {key} differs for trajectory {first['id']}")
        study = second.get("preparation_study", {})
        if same_conditioning and "final_rg_bohr" in study and "initial_rg_bohr" in second:
            # The original radius is evaluated before the launch translation;
            # the final radius is recomputed afterwards. Coordinate subtraction
            # introduces rounding even when the actual A q,p arrays are fixed.
            difference = abs(study["final_rg_bohr"][0]-second["initial_rg_bohr"][0])
            final_radius_max_difference = max(final_radius_max_difference, difference)
            if difference > 5e-13:
                raise ValueError("Preparation changed the serialized incoming A ring radius")
    a, b = case_record(reference), case_record(variant)
    return {"comparison": comparison, "reference_variant": reference.variant, "variant": variant.variant,
            "v": a["v"], "nominal_total_label_ev": a["nominal_total_label_ev"],
            "reference_collision_energy_ev": a["collision_energy_ev"],
            "variant_collision_energy_ev": b["collision_energy_ev"],
            "reference_actual_asymptotic_total_energy_ev": a["actual_asymptotic_total_energy_ev"],
            "variant_actual_asymptotic_total_energy_ev": b["actual_asymptotic_total_energy_ev"],
            "reference_reactive": a["product"], "variant_reactive": b["product"],
            "reference_unknown": a["unknown"], "variant_unknown": b["unknown"],
            "id_min": min(reference.rows), "id_max": max(reference.rows),
            "exact_ids_verified": True, "same_conditioning": same_conditioning,
            "physical_configuration_verified": True,
            "serialized_launch_checks": checks,
            "post_preparation_A_radius_max_abs_difference_bohr": final_radius_max_difference,
            "post_preparation_A_radius_roundoff_tolerance_bohr": 5e-13,
            "full_A_qp_and_orientation_serialized": False,
            "pairing_scope": ("Same launch RNG and physical configuration; measured b/axial separation/A radius/initial bond checked where preparation succeeded. "
                              "Full A q,p and orientation are not serialized. The frozen preparation source preserves A q,p."
                              if same_conditioning else
                              "Same IDs and launch RNG; measured b checked. Cutoff rejection redraws the whole initial state, "
                              "so removing it can also change A ring and axial separation on rejected IDs."),
            **paired_estimate((x["status"] for x, _ in pairs), (y["status"] for _, y in pairs), reference.area)}


def unresolved_diagnostics(case):
    output = []
    for i, row in sorted(case.rows.items()):
        if row["status"] != "unresolved":
            continue
        entry = {"variant": case.variant, "v": case.config["v"],
                 "nominal_total_label_ev": case.config["total_energy_ev"], "id": i,
                 "original_status": row["status"], "original_reason": row.get("reason"),
                 "time_fs": row.get("time_fs"), "b_bohr": row.get("b_bohr"),
                 "initial_centroid_bond_bohr": row.get("initial_bond_bohr"),
                 "initial_ring_hamiltonian_per_bead_ev": row.get("initial_ring_hamiltonian_per_bead_ev"),
                 "max_observed_energy_drift_ev": row.get("max_observed_energy_drift_ev"),
                 "initial_internal_diagnostics": row.get("preparation_study", {}).get("before"),
                 "post_preparation_internal_diagnostics": row.get("preparation_study", {}).get("after"),
                 "original_classifier_diagnostics": row.get("final_diagnostics"),
                 "status_changed": False}
        if "endpoint_q_bohr" in row and "endpoint_p_au" in row:
            q, p = np.asarray(row["endpoint_q_bohr"]), np.asarray(row["endpoint_p_au"])
            if q.shape != p.shape or q.ndim != 3 or q.shape[1:] != (3, 3) or not np.isfinite(q).all() or not np.isfinite(p).all():
                raise ValueError("Invalid saved unresolved endpoint")
            centers, velocities = q.mean(axis=0), p.mean(axis=0)/MASS_AU
            distances = []
            for a, b in ((0, 1), (0, 2), (1, 2)):
                vector = centers[b]-centers[a]
                length = float(np.linalg.norm(vector))
                speed = float(np.dot(vector, velocities[b]-velocities[a])/length) if length else None
                distances.append({"atoms": [a, b], "centroid_distance_bohr": length,
                                  "separation_derivative_bohr_per_atomic_time": speed,
                                  "separation_derivative_bohr_per_fs": speed*FS_TO_AU if speed is not None else None})
            entry.update(endpoint_saved=True, centroid_pairs=distances,
                         all_centroid_pairs_beyond_bond_cutoff=all(x["centroid_distance_bohr"] > case.config["bond_cutoff_bohr"] for x in distances),
                         all_centroid_pair_distances_increasing=all(x["separation_derivative_bohr_per_atomic_time"] is not None and x["separation_derivative_bohr_per_atomic_time"] > 0 for x in distances))
        else:
            entry["endpoint_saved"] = False
        entry["interpretation_scope"] = ("Endpoint geometry/velocity diagnostic only. Large increasing distances may be consistent "
                                         "with three-fragment separation, but do not redefine the original 2+1 classifier or resolve this attempt.")
        output.append(entry)
    return output


def mean_sem(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Mean/SEM needs finite, nonempty trajectory values")
    return {"n": len(values), "mean": float(values.mean()),
            "sem": float(values.std(ddof=1)/math.sqrt(len(values))) if len(values)>1 else None}


def _display_unit(column):
    if column.endswith("hartree"):
        return "eV", 1/EV_TO_AU
    if column.endswith("_bohr"):
        return "bohr", 1.
    if column.endswith("_fs"):
        return "fs", 1.
    if column == "lambda":
        return "dimensionless", 1.
    return "atomic units", 1.


def _validate_audit_case(case):
    """A successful preparation must contribute a complete comparable diagnostic."""
    if any(row.get("status") not in ("prepared", "failed") for row in case.rows.values()):
        raise ValueError(f"Invalid audit status: {case.path}")
    accepted = [row for row in case.rows.values() if row["status"] == "prepared"]
    columns = {}
    expected = _expected_preparation(case.variant)
    if case.variant not in AUDIT_LABELS:
        raise ValueError(f"Not an isolated audit variant: {case.variant}")
    start, duration = expected["switch_time_fs"], expected["hold_time_fs"]
    for row in accepted:
        context = f"{case.path}, ID {row['id']}"
        study = row.get("preparation_study", {})
        native = study.get("switch", {})
        if native.get("status") != "completed":
            raise ValueError(f"Successful audit lacks completed native preparation: {context}")
        for key, value in (("switch_time_fs", start), ("hold_time_fs", duration),
                           ("accepted_hold_duration_fs", duration), ("dt_fs", expected["preparation_dt_fs"])):
            if not math.isclose(_finite(native.get(key), f"native {key}: {context}"), value,
                                rel_tol=1e-12, abs_tol=1e-9):
                raise ValueError(f"Mismatched native {key}: {context}")
        averages = study.get("time_averages")
        if not isinstance(averages, dict) or set(averages) != set(AUDIT_WINDOWS):
            raise ValueError(f"Successful audit lacks complete time windows: {context}")
        for name, source in AUDIT_WINDOWS.items():
            window = averages[name]
            window_start = start+duration-50. if name == "final_50fs" else start
            window_duration = 50. if name == "final_50fs" else duration
            if not isinstance(window, dict) or window.get("source") != source:
                raise ValueError(f"Invalid {name} averaging source: {context}")
            for key, value in (("start_fs", window_start), ("end_fs", start+duration),
                               ("duration_fs", window_duration)):
                if not math.isclose(_finite(window.get(key), f"{name}.{key}: {context}"), value,
                                    rel_tol=1e-12, abs_tol=1e-9):
                    raise ValueError(f"Mismatched {name} time window {key}: {context}")
            values = window.get("values")
            if not isinstance(values, dict) or not NATIVE_HOLD_COLUMNS <= values.keys():
                raise ValueError(f"Successful audit lacks required {name} mean columns: {context}")
            if name in columns and columns[name] != set(values):
                raise ValueError(f"Successful audit mean columns differ for {name}: {context}")
            columns[name] = set(values)
            for key, value in values.items():
                _finite(value, f"{name}.{key}: {context}")
            if name == "hold":
                if set(values) != NATIVE_HOLD_COLUMNS or values != native.get("hold_time_average"):
                    raise ValueError(f"Hold means disagree with the native integral: {context}")
                if (window.get("native_dt_fs") != native["dt_fs"] or
                        type(window.get("native_steps")) is not int or
                        window["native_steps"] != round(duration/native["dt_fs"])):
                    raise ValueError(f"Invalid native hold timestep/count: {context}")
        for stage in ("before", "after"):
            values = study.get(stage)
            if not isinstance(values, dict) or not values:
                raise ValueError(f"Successful audit lacks {stage} diagnostics: {context}")
            if stage in columns and columns[stage] != set(values):
                raise ValueError(f"Successful audit {stage} diagnostic columns differ: {context}")
            columns[stage] = set(values)
            for key, value in values.items():
                _finite(value, f"{stage}.{key}: {context}")
    return accepted


def audit_case_record(case):
    accepted = _validate_audit_case(case)
    expected = _expected_preparation(case.variant)
    return {"variant": case.variant, "v": case.config["v"],
            "nominal_total_label_ev": case.config["total_energy_ev"],
            "attempts": len(case.rows), "prepared": len(accepted), "failed": len(case.rows)-len(accepted),
            "id_min": min(case.rows), "id_max": max(case.rows),
            "hold_time_fs": expected["hold_time_fs"], "native_dt_fs": expected["preparation_dt_fs"],
            "statistics_scope": "Every attempted audit is counted; means and SEM are unavailable when no preparation succeeds"}


def _validate_audit_reduction(case, reduction):
    """Authenticate the separately archived reduction against the actual ledger."""
    outcomes = audit_case_record(case)
    if any(type(reduction.get(key)) is not int or reduction[key] != outcomes[key]
           for key in ("attempts", "prepared", "failed")):
        raise ValueError(f"Audit reduction outcome mismatch: {case.path}")
    expected = {}
    for row in audit_statistics(case):
        if row["window"] in AUDIT_WINDOWS:
            expected.setdefault(row["window"], {})[row["column"]] = row
    windows = reduction.get("windows")
    if not isinstance(windows, dict) or set(windows) != set(expected):
        raise ValueError(f"Audit reduction window mismatch: {case.path}")
    for name, columns in expected.items():
        window = windows[name]
        if set(window.get("values", {})) != set(columns):
            raise ValueError(f"Audit reduction mean columns mismatch: {case.path}")
        for column, row in columns.items():
            values = window["values"][column]
            if (window.get("source") != row["source"] or
                    any(not math.isclose(_finite(window.get(key), f"reduction {key}"), row[key], abs_tol=1e-9)
                        for key in ("start_fs", "end_fs")) or
                    not math.isclose(_finite(window.get("duration_fs"), "reduction duration_fs"), row["end_fs"]-row["start_fs"],
                                     rel_tol=1e-12, abs_tol=1e-9) or
                    type(values.get("n")) is not int or values["n"] != outcomes["prepared"]):
                raise ValueError(f"Audit reduction metadata/count mismatch: {case.path}")
            for key, expected_key in (("mean", "mean_native_unit"), ("sem", "sem_native_unit")):
                value, target = values.get(key), row[expected_key]
                if target is None:
                    if value is not None:
                        raise ValueError(f"Audit reduction invents unavailable SEM: {case.path}")
                elif not math.isclose(_finite(value, f"reduction {key}"), target, rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError(f"Audit reduction {key} disagrees with ledger: {case.path}")
    differences = {}
    for row in case.rows.values():
        if row["status"] == "prepared":
            averages = row["preparation_study"]["time_averages"]
            for column in NATIVE_HOLD_COLUMNS:
                differences.setdefault(column, []).append(averages["hold"]["values"][column]-averages["hold_trace_check"]["values"][column])
    recorded = reduction.get("native_minus_trace_hold")
    if not isinstance(recorded, dict) or set(recorded) != set(differences):
        raise ValueError(f"Audit native-minus-trace reduction columns mismatch: {case.path}")
    for column, values in differences.items():
        expected_stats = mean_sem(values)
        observed = recorded[column]
        if type(observed.get("n")) is not int or observed["n"] != expected_stats["n"]:
            raise ValueError(f"Audit native-minus-trace reduction count mismatch: {case.path}")
        for key in ("mean", "sem"):
            target, value = expected_stats[key], observed.get(key)
            if ((target is None and value is not None) or
                    (target is not None and not math.isclose(_finite(value, f"native-minus-trace {key}"), target,
                                                             rel_tol=1e-10, abs_tol=1e-12))):
                raise ValueError(f"Audit native-minus-trace reduction {key} disagrees with ledger: {case.path}")


def audit_statistics(case):
    """Every replicate is a trajectory mean; time points never inflate N."""
    accepted = _validate_audit_case(case)
    grouped = {}
    units = {}
    for row in accepted:
        trace = row["preparation_study"].get("trace", {})
        if isinstance(trace, dict):
            units.update(trace.get("units", {}))
    for row in accepted:
        study = row["preparation_study"]
        for window, data in study.get("time_averages", {}).items():
            for column, value in data.get("values", {}).items():
                # Validation above requires every successful ID in every mean.
                key = (window, column, data["source"], round(data["start_fs"], 8), round(data["end_fs"], 8))
                grouped.setdefault(key, []).append(value)
    rows = []
    for (window, column, source, start, end), values in sorted(grouped.items()):
        unit, factor = _display_unit(column)
        native = mean_sem(values)
        display = mean_sem(np.asarray(values)*factor)
        rows.append({"variant": case.variant, "v": case.config["v"],
                     "nominal_total_label_ev": case.config["total_energy_ev"],
                     "attempts": len(case.rows), "prepared": len(accepted),
                     "failed": len(case.rows)-len(accepted), "window": window,
                     "source": source, "start_fs": start, "end_fs": end,
                     "column": column, "n_trajectory_means": native["n"],
                     "native_unit": units.get(column, "hartree" if column.endswith("hartree") else unit),
                     "mean_native_unit": native["mean"], "sem_native_unit": native["sem"],
                     "display_unit": unit, "mean": display["mean"], "sem": display["sem"],
                     "energy_reference": ("Includes common BKMP2 Vmin; original PES zero is three separated H atoms"
                                          if column in ("harmonic_potential_per_bead_hartree", "full_potential_per_bead_hartree", "ring_hamiltonian_per_bead_hartree") else
                                          "K + Vfull - Vmin - spring per bead"
                                          if column == "primitive_internal_per_bead_hartree" else
                                          "Zero at harmonic minimum" if column == "centroid_harmonic_energy_hartree" else ""),
                     "statistics_scope": "Conditional on successful preparation; SEM across trajectory means, not time points"})
    # Initial one-frame fluctuations are intentionally tabulated separately.
    for stage in ("before", "after"):
        columns = sorted({k for row in accepted for k in row["preparation_study"].get(stage, {})})
        for column in columns:
            values = [row["preparation_study"][stage][column] for row in accepted
                      if column in row["preparation_study"].get(stage, {})]
            data = mean_sem(values)
            rows.append({"variant": case.variant, "v": case.config["v"],
                         "nominal_total_label_ev": case.config["total_energy_ev"],
                         "attempts": len(case.rows), "prepared": len(accepted), "failed": len(case.rows)-len(accepted),
                         "window": stage+"_single_frame", "source": "initial_or_final_single_frame",
                         "start_fs": None, "end_fs": None, "column": column,
                         "n_trajectory_means": data["n"], "mean_native_unit": data["mean"],
                         "sem_native_unit": data["sem"], "display_unit": "eV" if column.endswith("_ev") else "bohr",
                         "mean": data["mean"], "sem": data["sem"],
                         "statistics_scope": "Single-frame ensemble statistic; not a time-averaged state energy"})
    return rows


def preparation_numerics(case, *, audit=False):
    """Observed work/drift maxima are numerical checks, not statistical SEM."""
    summaries = [r.get("preparation_study", {}).get("switch") for r in case.rows.values()]
    summaries = [s for s in summaries if s is not None and s.get("status") == "completed"]
    if not summaries:
        return None
    for summary in summaries:
        for key in ("switch_time_fs", "hold_time_fs", "dt_fs", "minimum_bond_bohr", "maximum_bond_bohr",
                    "maximum_absolute_energy_balance_residual_per_bead_hartree", "final_energy_balance_residual_per_bead_hartree"):
            _finite(summary.get(key), f"completed native {key}: {case.path}")
        if summary["hold_time_fs"] > 0:
            _finite(summary.get("maximum_absolute_hold_energy_drift_per_bead_hartree"), f"completed native hold drift: {case.path}")
        if any(summary[key] != summaries[0][key] for key in ("switch_time_fs", "hold_time_fs", "dt_fs")):
            raise ValueError(f"Cannot pool native preparations with different windows/timesteps: {case.path}")
    def maximum(key):
        values = [abs(s[key]) for s in summaries if s.get(key) is not None]
        return max(values)/EV_TO_AU if values else None
    result = {"scope": "isolated_audit" if audit else "collision_preparation", "variant": case.variant,
            "v": case.config["v"], "nominal_total_label_ev": case.config["total_energy_ev"],
            "attempts": len(case.rows), "native_preparations_completed": len(summaries),
            "switch_time_fs": summaries[0]["switch_time_fs"], "hold_time_fs": summaries[0]["hold_time_fs"],
            "native_dt_fs": summaries[0]["dt_fs"],
            "max_absolute_work_balance_residual_ev": maximum("maximum_absolute_energy_balance_residual_per_bead_hartree"),
            "max_absolute_final_work_balance_residual_ev": maximum("final_energy_balance_residual_per_bead_hartree"),
            "max_absolute_hold_energy_drift_ev": maximum("maximum_absolute_hold_energy_drift_per_bead_hartree")
            if summaries[0]["hold_time_fs"] > 0 else None,
            "minimum_bond_bohr": min(s["minimum_bond_bohr"] for s in summaries),
            "maximum_bond_bohr": max(s["maximum_bond_bohr"] for s in summaries),
            "scope_note": "Native observed maxima over accepted steps; completion is not a bound-state classification; no comparison to statistical SEM or relabeling"}
    for name in ("centroid_bond_bohr", "minimum_bead_bond_bohr", "maximum_bead_bond_bohr"):
        values = [(row["preparation_study"]["after"][name], row["id"]) for row in case.rows.values()
                  if name in row.get("preparation_study", {}).get("after", {})]
        if values:
            low, high = min(values), max(values)
            stem = "final_"+name.removesuffix("_bohr")
            result.update({stem+"_min_bohr": low[0], stem+"_min_id": low[1],
                           stem+"_max_bohr": high[0], stem+"_max_id": high[1]})
    return result


def audit_pair_record(reference, variant, comparison):
    """Outcome accounting exists even when no pair has two successful members."""
    if set(reference.rows) != set(variant.rows):
        raise ValueError("Audit pairing requires the same exact attempted IDs")
    _validate_pair_config(reference, variant, audit=True)
    for case in (reference, variant):
        _validate_audit_case(case)
    pairs = [(reference.rows[i], variant.rows[i]) for i in sorted(reference.rows)]
    successful = sum(a["status"] == b["status"] == "prepared" for a, b in pairs)
    return {"comparison": comparison, "reference_variant": reference.variant, "variant": variant.variant,
            "v": variant.config["v"], "nominal_total_label_ev": variant.config["total_energy_ev"],
            "attempted_pairs": len(pairs), "successful_pairs": successful,
            "pairs_with_a_failure": len(pairs)-successful,
            "reference_failed": sum(a["status"] == "failed" for a, _ in pairs),
            "variant_failed": sum(b["status"] == "failed" for _, b in pairs),
            "both_failed": sum(a["status"] == b["status"] == "failed" for a, b in pairs),
            "exact_ids_verified": True, "physical_configuration_verified": True,
            "hold_time_fs": 100.,
            "scope": "All attempted pairs are counted; means require both preparations to succeed; matched native 100 fs hold windows"}


def paired_audit_means(reference, variant, comparison):
    """Paired SEM for native hold means, with failed-preparation counts visible."""
    outcomes = audit_pair_record(reference, variant, comparison)
    pairs = [(reference.rows[i], variant.rows[i]) for i in sorted(reference.rows)
             if reference.rows[i]["status"] == variant.rows[i]["status"] == "prepared"]
    if not pairs:
        return []
    columns = set(pairs[0][0]["preparation_study"]["time_averages"]["hold"]["values"])
    differences = {key: [] for key in columns}
    for a, b in pairs:
        av, bv = [r["preparation_study"]["time_averages"]["hold"]["values"] for r in (a, b)]
        if set(av) != columns or set(bv) != columns:
            raise ValueError("Paired native hold mean columns differ")
        for key in columns:
            differences[key].append(bv[key]-av[key])
    rows = []
    for column, values in sorted(differences.items()):
        unit, factor = _display_unit(column)
        stats = mean_sem(np.asarray(values)*factor)
        rows.append({**outcomes,
                     "column": column, "display_unit": unit,
                     "mean_difference": stats["mean"], "sem_difference": stats["sem"],
                     "scope": "Variant minus reference native 100 fs hold means; conditional on both preparations succeeding; finite-window diagnostic, not a pure-state or adiabatic-convergence test"})
    return rows


def _plot_style():
    # Plotting is optional for importing/testing the numerical analysis tools.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelsize": 11, "axes.titlesize": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#9aa4af", "axes.labelcolor": "#243449",
                         "xtick.color": "#465569", "ytick.color": "#465569",
                         "figure.facecolor": "white", "savefig.facecolor": "white",
                         "svg.fonttype": "none"})
    return plt


def collision_plot(records, reference, output):
    plt = _plot_style()
    lookup = {(r["variant"], r["v"], round(r["nominal_total_label_ev"], 6)): r for r in records}
    fig, axes = plt.subplots(2, 3, figsize=(12.7, 8.6))
    fig.subplots_adjust(left=.075, right=.98, top=.81, bottom=.20, hspace=.76, wspace=.28)
    for column, (v, energy) in enumerate(POINTS):
        for row, variants in enumerate((("ne_paper_subset", "ne_uncut", "ne_switch200"),
                                        ("qct_harmonic", "qct_action_collision", "qct_action_total"))):
            ax = axes[row, column]
            ax.set_xlim(-.55, 2.55)
            ax.grid(axis="y", color="#e9edf2", linewidth=.8)
            maxima = []
            labels = []
            for x, variant in enumerate(variants):
                record = lookup.get((variant, v, energy))
                label = (("Paper cutoff", "Uncut", "Switch 200 fs") if row == 0 else
                         ("Harmonic", "Action\nfixed Ecol", "Action\nfixed actual Etot"))[x]
                if record is None:
                    ax.text(x, .08, "pending", transform=ax.get_xaxis_transform(), ha="center", color="#97a0aa", fontsize=9)
                    labels.append(label+"\n(pending)")
                    continue
                center = record["sigma_confirmed_angstrom2"]
                low, high = (record["sigma_ci95_including_unknown_lower_angstrom2"],
                             record["sigma_ci95_including_unknown_upper_angstrom2"])
                ax.errorbar(x, center, yerr=[[center-low], [high-center]], color=COLORS[variant],
                            marker="o", markersize=7, capsize=4, linewidth=1.6, zorder=4)
                upper = record["sigma_identification_upper_angstrom2"]
                if upper > center:
                    ax.plot([x, x], [center, upper], color=COLORS[variant], lw=6, alpha=.4, zorder=3)
                    ax.plot(x, upper, marker="_", color=COLORS[variant], ms=12, mew=2)
                label += f"\nN={record['samples']}" if row == 0 else f"\n{record['collision_energy_ev']:.4f} eV"
                labels.append(label)
                annotation = f"{record['product']} R"
                if record["unknown"]:
                    annotation += f" + {record['unknown']} ?"
                ax.annotate(annotation, (x, high), xytext=(0, 6), textcoords="offset points",
                            ha="center", fontsize=8.5, color=COLORS[variant])
                maxima.append(high)
            if row == 0 and (v, energy) in reference:
                value = reference[v, energy]
                ax.axhspan(value["lower"], value["upper"], color="#8190a3", alpha=.10, linewidth=0)
                ax.axhline(value["value"], color="#8190a3", ls=(0, (4, 3)), lw=1.1)
                maxima.append(value["upper"])
            ax.set_ylim(0, max(maxima, default=.1)*1.28)
            ax.set_xticks(range(3), labels, fontsize=9)
            ax.set_title(f"v = {v}  |  nominal label {energy:.2f} eV", loc="left", fontweight="semibold", pad=12)
            if column == 0:
                ax.set_ylabel(r"Reactive cross section / $\mathrm{\AA^2}$")
            if row == 0:
                record = lookup.get(("ne_paper_subset", v, energy))
                if record:
                    ax.text(.97, .94, f"Ecol = {record['collision_energy_ev']:.4f} eV", transform=ax.transAxes,
                            ha="right", va="top", fontsize=9, color="#4b596b")
    fig.text(.075, .955, "Initial preparation changes the scattering ensemble", fontsize=19, fontweight="bold", color="#172a42")
    fig.text(.075, .915, "H + H2 on BKMP2  |  controlled sensitivity study, not a fitted reproduction", fontsize=11, color="#536176")
    fig.text(.075, .872, "NE-RPMD, P = 32  ·  same collision energy, exact paired N = 1024", fontsize=11, fontweight="semibold", color="#243449")
    fig.text(.075, .482, "QCT, P = 1  ·  exact paired N = 4096  ·  actual collision energy below each variant", fontsize=11, fontweight="semibold", color="#243449")
    fig.text(.075, .092, "Points: confirmed reactions / all attempts; bars: exact 95% binomial bounds including unknown outcomes.  R = reactive; ? = unresolved/failed.", fontsize=9, color="#47566b")
    fig.text(.075, .066, "NE only: dashed line and pale band are the digitized 2021 NE curve and digitization tolerance, not a confidence interval.", fontsize=9, color="#47566b")
    fig.text(.075, .040, "Nominal labels use harmonic (v + 1/2)hbar*omega. Action QCT uses a constant full-PES semiclassical internal energy; the energy constraints differ.", fontsize=9, color="#47566b")
    for extension in ("png", "svg"):
        fig.savefig(Path(output)/f"preparation_scattering.{extension}", dpi=300)
    plt.close(fig)


def audit_plot(cases, quantum_levels, output):
    """Sparse frames visualize the hold; quantitative means use native integrals."""
    from matplotlib.lines import Line2D
    plt = _plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.7, 5.4))
    fig.subplots_adjust(left=.075, right=.98, bottom=.30, top=.73, wspace=.28)
    plotted = False
    for ax, (v, energy) in zip(axes, POINTS):
        ax.set_title(f"v = {v}  |  nominal label {energy:.2f} eV", loc="left", fontweight="semibold")
        panel_plotted = False
        panel_cases = [case for case in cases if (case.config["v"], case.config["total_energy_ev"]) == (v, energy)]
        for case in panel_cases:
            _validate_audit_case(case)
            by_time = {}
            for row in case.rows.values():
                if row["status"] != "prepared":
                    continue
                study = row["preparation_study"]
                switch = study["switch"]["switch_time_fs"]
                trace = study.get("trace", [])
                if isinstance(trace, dict) and (len(set(trace.get("columns", []))) != len(trace.get("columns", [])) or
                        any(len(record) != len(trace["columns"]) for record in trace.get("records", []))):
                    raise ValueError(f"Invalid compact audit trace columns: {case.path}, ID {row['id']}")
                frames = ([dict(zip(trace["columns"], record)) for record in trace["records"]]
                          if isinstance(trace, dict) else trace)
                if not frames:
                    raise ValueError(f"Successful audit lacks saved trace: {case.path}, ID {row['id']}")
                times = [_finite(frame.get("time_fs"), "audit trace time") for frame in frames]
                if any(b <= a for a, b in zip(times, times[1:])):
                    raise ValueError(f"Audit trace times are not increasing: {case.path}, ID {row['id']}")
                if not any(math.isclose(t, switch+100., abs_tol=1e-8) for t in times):
                    raise ValueError(f"Successful audit trace lacks completed hold endpoint: {case.path}, ID {row['id']}")
                for frame in frames:
                    t = round(frame["time_fs"]-switch, 7)
                    value = frame.get("primitive_internal_per_bead_hartree")
                    if t >= -1e-7:
                        _finite(value, "successful audit trace primitive diagnostic")
                        by_time.setdefault(t, []).append(value/EV_TO_AU)
            if not by_time:
                continue
            times = sorted(by_time)
            values = [mean_sem(by_time[t]) for t in times]
            means = np.array([r["mean"] for r in values])
            # A single successful trajectory has no estimable ensemble SEM.
            # Preserve that absence instead of drawing a zero-width error band.
            sems = np.array([np.nan if r["sem"] is None else r["sem"] for r in values])
            ax.plot(times, means, color=AUDIT_COLORS[case.variant], lw=1.0, marker=".", markersize=3)
            ax.fill_between(times, means-sems, means+sems, color=AUDIT_COLORS[case.variant], alpha=.10, linewidth=0)
            plotted = True
            panel_plotted = True
        if not panel_plotted:
            label = ("No successful preparations\n"+str(sum(len(case.rows) for case in panel_cases))+" attempted; all failed"
                     if panel_cases else "Audit pending")
            ax.text(.5, .45, label, transform=ax.transAxes, ha="center", color="#6b7885", fontsize=9)
        if v in quantum_levels:
            ax.axhline(quantum_levels[v], color="#283a4e", ls=(0, (4, 3)), lw=1)
        ax.grid(color="#e9edf2", linewidth=.8)
        ax.set_xlabel("Time after switch / fs")
        ax.set_xlim(0, 100)
    axes[0].set_ylabel("Mean primitive internal estimator / eV")
    handles = [Line2D([0], [0], color=AUDIT_COLORS[k], lw=2, label=v) for k, v in AUDIT_LABELS.items()]
    handles.append(Line2D([0], [0], color="#283a4e", lw=1, ls="--", label="Exact isolated-H2 level (reference only)"))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.07, .035), frameon=False, ncol=3, fontsize=9)
    fig.text(.075, .93, "Isolated H2: energy diagnostics during the final 100 fs hold", fontsize=16, fontweight="bold", color="#172a42")
    fig.text(.075, .855, "Sparse 5 fs frames; shading is one SEM across independent trajectory IDs. Native every-step hold means are tabulated separately.", fontsize=10, color="#536176")
    fig.text(.075, .185, "Sparse frames do not resolve the fastest oscillations; they are not used to calculate any hold average.", fontsize=9, color="#536176")
    fig.text(.075, .155, "K + Vfull - Vmin - spring per bead is a primitive diagnostic, not the conserved ring Hamiltonian or a pure-state population.", fontsize=9, color="#536176")
    # Publish a current placeholder when no trajectory succeeded; never reuse
    # an earlier generation's figure merely because its path still exists.
    for extension in ("png", "svg"):
        fig.savefig(Path(output)/f"preparation_energy_hold.{extension}", dpi=300)
    plt.close(fig)
    return plotted


def _number(value, digits=5):
    return "—" if value is None else f"{value:.{digits}f}"


def _table(headers, rows):
    return ("<table><thead><tr>"+"".join(f"<th>{html.escape(str(h))}</th>" for h in headers)+
            "</tr></thead><tbody>"+"".join("<tr>"+"".join(f"<td>{html.escape(str(c))}</td>" for c in row)+"</tr>" for row in rows)+"</tbody></table>")


def html_report(report, output):
    records, pairs = report["cross_sections"], report["paired_comparisons"]
    rows = [[r["v"], f'{r["nominal_total_label_ev"]:.2f}', LABELS[r["variant"]],
             f'{r["collision_energy_ev"]:.6f}', r["samples"], f'{r["product"]} / {r["unknown"]}',
             _number(r["sigma_confirmed_angstrom2"]),
             f'[{r["sigma_identification_lower_angstrom2"]:.5f}, {r["sigma_identification_upper_angstrom2"]:.5f}]',
             f'[{r["sigma_ci95_including_unknown_lower_angstrom2"]:.5f}, {r["sigma_ci95_including_unknown_upper_angstrom2"]:.5f}]'] for r in records]
    pair_rows = [[r["v"], f'{r["nominal_total_label_ev"]:.2f}', r["comparison"], r["paired_samples"],
                  f'{r["gained_reactivity_definite"]}–{r["gained_reactivity_possible"]} / {r["lost_reactivity_definite"]}–{r["lost_reactivity_possible"]}',
                  f'[{r["delta_sigma_identification_lower_angstrom2"]:.5f}, {r["delta_sigma_identification_upper_angstrom2"]:.5f}]',
                  f'[{r["delta_sigma_conservative_ci95_lower_angstrom2"]:.5f}, {r["delta_sigma_conservative_ci95_upper_angstrom2"]:.5f}]'] for r in pairs]
    primary_audit = [r for r in report["audit_statistics"] if r["window"] == "hold" and r["column"] == "primitive_internal_per_bead_hartree"]
    audit_index = {(r["variant"], r["v"], r["nominal_total_label_ev"]): r for r in primary_audit}
    audit_rows = []
    for case in report["audit_cases"]:
        mean = audit_index.get((case["variant"], case["v"], case["nominal_total_label_ev"]), {})
        audit_rows.append([case["v"], f'{case["nominal_total_label_ev"]:.2f}', AUDIT_LABELS[case["variant"]],
                           case["attempts"], case["failed"], case["prepared"],
                           _number(mean.get("mean")), _number(mean.get("sem"))])
    energy_rows = [[r["v"], _number(r["harmonic_internal_label_ev"], 7), _number(r["action_internal_energy_ev"], 7),
                    _number(r["exact_quantum_internal_energy_ev"], 7), _number(r["action_minus_exact_ev"], 7)] for r in report["internal_energy_references"]]
    status = ("COMPLETE" if report["complete"] else f"PARTIAL: {len(report['pending_cases'])} planned case(s) pending")
    sections = [f"<h1>BKMP2 initial-preparation sensitivity study</h1><p class='status'>{status}</p>",
        "<p>These calculations test how the initial ensemble changes H + H2 scattering. They do not establish a new pure-state NE-RPMD method, fit a published curve, or reproduce the paper's externally sourced QCT data. Only completed production cases enter this report; pilot and smoke cases are excluded.</p>",
        "<h2>Cross sections and exact paired comparisons</h2>",
        "<p>The NE panel uses the first exact 1024 original trajectory IDs and the same 1024 IDs for the new variants. The original full NE estimate (4096 IDs) remains a separate row below and is never treated as the paired baseline. All QCT comparisons use the same 4096 IDs. Removing the centroid cutoff can redraw the whole initial state on rejected IDs; the subsequent switch comparison keeps the same uncut launch and changes only BC internal preparation.</p>",
        "<img src='data:image/png;base64,"+base64.b64encode((Path(output)/"figures/preparation_scattering.png").read_bytes()).decode()+"'>",
        _table(["v", "Nominal label / eV", "Preparation", "Ecol / eV", "N", "Reactive / unknown", "Confirmed sigma / A^2", "Identification bounds / A^2", "95% bounds incl. unknown / A^2"], rows),
        "<p>Every attempted trajectory remains in the denominator. Unknown includes unresolved and failed attempts. The empirical identification interval assigns each unknown either nonreactive or reactive; the wider interval combines that uncertainty with exact two-sided binomial sampling bounds. These are truncated-disk cross sections at bmax = 3.5 bohr (v=0) or 4.0 bohr (v=1), not an assertion that all larger impact parameters contribute zero.</p>",
        "<h3>Paired differences: variant minus reference</h3>",
        _table(["v", "Nominal label / eV", "Contrast", "Paired N", "Gain range / loss range", "Delta identification bounds / A^2", "Conservative 95% delta bounds / A^2"], pair_rows),
        "<p>Gain means reference nonreactive and variant reactive; loss is the converse. Each discordance probability has a two-sided 97.5% Clopper–Pearson interval. Bonferroni gives at least 95% simultaneous coverage for the two probabilities, and subtracting their intervals gives the reported paired difference bounds. Unknown outcomes expand definite event counts to possible counts. Coverage is per contrast, with no adjustment across the 12 contrasts. No unknown pair is dropped. The JSON records exact ID and serialized launch-field checks; full incoming-atom q,p and molecular orientation are not present in the ledgers and are not claimed to have been compared as arrays.</p>",
        "<h2>Energy conventions</h2>",
        _table(["v", "Harmonic label / eV", "Full-PES action / eV", "Exact radial quantum / eV", "Action - quantum / eV"], energy_rows),
        "<p>The nominal total label is Ecol + (v+1/2)hbar*omega with omega = 4407 cm^-1. A harmonic phase draw propagated on BKMP2 does not have a constant actual full-PES internal energy. Full-PES QCT solves J(E) = (1/pi) integral sqrt(2 mu [E-V(r)]) dr = v+1/2 and samples the periodic orbit uniformly in time. Its energy is semiclassical, and differs from the independently converged radial Schrodinger eigenvalue. Both use the same isolated-H2 well minimum as zero. Fixed-collision action QCT keeps the original Ecol and changes actual asymptotic total energy. Fixed-total action QCT changes Ecol so its actual asymptotic total equals the displayed nominal label. These controls answer different questions.</p>",
        "<p>The NE primitive internal quantity K + Vfull - Vmin - spring (per bead) is an ensemble diagnostic, not a single-trajectory physical energy, a state population, or the conserved H_P/P. Smooth switching does not by itself prove excited-state preservation. An apparent match to a quantum energy reference is insufficient to establish the correct state-selected ensemble.</p>"]
    if report["audit_cases"]:
        sections += ["<h2>Isolated preparation audit</h2>",
            _table(["v", "Nominal label / eV", "Preparation", "Attempts", "Failed", "Independent means", "100 fs hold mean / eV", "SEM / eV"], audit_rows),
            "<p>Primary means use the native trapezoidal integral at every accepted timestep during the full 100 fs hold. SEM is computed across independent per-trajectory means. The dense 0.2 fs trace integral and final-50-fs average are separate secondary checks in audit_statistics.csv. The high-variance initial and final single-frame statistics are separate rows, never substituted for the hold mean. Audit means are conditional on successful preparation; failures remain counted. Collision variants launch immediately after the ramp, with no extra hold; only these isolated diagnostics append the 100 fs NVE hold. Their time means therefore characterize subsequent isolated evolution, not the instantaneous collision launch values.</p>"]
        image = Path(output)/"figures/preparation_energy_hold.png"
        if image.exists():
            sections.append("<img src='data:image/png;base64,"+base64.b64encode(image.read_bytes()).decode()+"'>")
        paired_means = {(r["reference_variant"], r["variant"], r["v"], r["nominal_total_label_ev"]): r
                        for r in report["audit_paired_time_means"] if r["column"] == "primitive_internal_per_bead_hartree"}
        paired_energy_rows = []
        for row in report["audit_paired_outcomes"]:
            mean = paired_means.get((row["reference_variant"], row["variant"], row["v"], row["nominal_total_label_ev"]), {})
            paired_energy_rows.append([row["v"], f'{row["nominal_total_label_ev"]:.2f}', row["comparison"],
                                       row["successful_pairs"], row["pairs_with_a_failure"],
                                       _number(mean.get("mean_difference"), 6), _number(mean.get("sem_difference"), 6)])
        sections += ["<h3>Paired native hold-mean differences</h3>",
                     _table(["v", "Nominal label / eV", "Contrast", "Paired N", "Pairs with failure", "Mean difference / eV", "Paired SEM / eV"], paired_energy_rows),
                     "<p>These differences use the exact shared audit IDs and subtract each trajectory's native hold mean before calculating SEM. They describe the finite 100 fs window and ramp-duration sensitivity; they are not state-population estimates or proof of an adiabatic limit. Any pair with a preparation failure is counted and the mean is explicitly conditional on both preparations succeeding.</p>"]
    if report["preparation_numerics"]:
        numeric_rows = [[r["scope"], r["variant"], r["v"], f'{r["nominal_total_label_ev"]:.2f}',
                         r["native_preparations_completed"], _number(r["max_absolute_work_balance_residual_ev"], 7),
                         _number(r["max_absolute_hold_energy_drift_ev"], 7)] for r in report["preparation_numerics"]]
        sections += ["<h2>Native preparation numerical checks</h2>",
            _table(["Scope", "Variant", "v", "Nominal label / eV", "Completed", "Max work-balance residual / eV", "Max hold drift / eV"], numeric_rows),
            "<p>Work balance is H_lambda/P - initial harmonic H/P - integrated switching work/P. The hold drift measures H_full/P from the beginning of the NVE hold. These maxima are numerical integration diagnostics, distinct from trajectory-to-trajectory SEM or proof of quantum-state accuracy.</p>"]
    if report["unresolved_endpoints"]:
        entries = report["unresolved_endpoints"]
        sections += ["<h2>Unresolved endpoints remain unknown</h2>",
                     f"<p>{len(entries)} unresolved production endpoints are retained with their original labels. The separate unresolved_endpoint_diagnostics.json records all centroid pair distances and their derivatives, saved initial bond/energy diagnostics, and observed integration drift. Widely separated, increasingly distant pairs are consistent with possible three-fragment separation, not evidence that a longer 2+1 collision lifetime has been measured. These diagnostics do not redefine the classifier or relabel any event.</p>"]
    sections += ["<h2>Scope and reproducibility</h2>",
        "<p>The 2021 NE reference is digitized from Marjollet and Welsch, DOI <a href='https://doi.org/10.1002/qua.26447'>10.1002/qua.26447</a>. Its band is a digitization tolerance, not a statistical interval; interpolation-model error and published sampling uncertainty are not quantified. The QCT panel contains only the present calculations. Source and completed-ledger SHA256 hashes are included in preparation_analysis.json. The analyzer does not run dynamics, change any source ledger, extend trajectories, or choose parameters to match the literature.</p>",
        "<p>Pending cases: "+html.escape(", ".join(report["pending_cases"]) or "none")+".</p>"]
    style = "body{font:15px/1.55 system-ui,sans-serif;max-width:1250px;margin:40px auto;padding:0 28px;color:#243449}h1{font-size:30px}h2{margin-top:32px}h3{margin-top:26px}.status{font-weight:700;color:#147d91}table{border-collapse:collapse;width:100%;font-size:12px;margin:18px 0}th,td{border-bottom:1px solid #dce3eb;padding:8px 7px;text-align:left}th{background:#eef3f7}img{width:100%;height:auto}a{color:#147d91}"
    (Path(output)/"preparation_report.html").write_text("<!doctype html><html lang='en'><meta charset='utf-8'><title>BKMP2 preparation study</title><style>"+style+"</style><body>"+"\n".join(sections)+"</body></html>\n")


def _check_design(case, v, energy, samples, offset):
    expected = {"v": v, "total_energy_ev": energy, "samples": samples, "id_offset": offset,
                "seed": 20261031, "bmin": 0., "bmax": 3.5 if v == 0 else 4., "dt_fs": .02,
                "time_cap_fs": 250., "extended_cap_fs": 1000., "initial_axial_bohr": 10.,
                "exit_bohr": 10., "bond_cutoff_bohr": 2.5, "persistence_fs": 1., "check_fs": .2,
                "dmin_bohr": 1.078, "integrator": "normal-mode-bab", "beta_override": None}
    if any(case.config.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Case does not match the registered production design: {case.path}")


def analyze(baseline, study, output):
    baseline, study, output = map(Path, (baseline, study, output))
    output.mkdir(parents=True, exist_ok=True)
    (output/"figures").mkdir(exist_ok=True)
    cases, pending, audits = [], [], []
    audit_manifest_path = study/"isolated_audit/audit_manifest.json"
    audit_manifest_bytes = audit_manifest_path.read_bytes() if audit_manifest_path.exists() else None
    audit_manifest = _strict_json(audit_manifest_bytes, audit_manifest_path) if audit_manifest_bytes is not None else {}
    completed_audits = set(audit_manifest.get("completed_cases", []))
    for v, energy in POINTS:
        for method, beads, variant in (("ne_rpmd", 32, "ne_paper_full"), ("qct", 1, "qct_harmonic")):
            path = baseline/"main"/f"{method}-v{v}-Et{energy:.3f}-P{beads}"
            case = load_case(path, variant)
            if case is None:
                raise FileNotFoundError(f"Required original case missing: {path}")
            _check_design(case, v, energy, 4096, 10000 if method == "ne_rpmd" else 20000)
            cases.append(case)
            if method == "ne_rpmd":
                ids = range(case.config["id_offset"], case.config["id_offset"]+1024)
                cases.append(case.subset(ids, "ne_paper_subset"))
        for name, variant in (("ne-direct-none", "ne_uncut"), ("ne-switch200-none", "ne_switch200"),
                              ("qct-action-fixed_collision", "qct_action_collision"),
                              ("qct-action-fixed_total", "qct_action_total")):
            path = study/"collisions"/f"{name}-v{v}-Et{energy:.3f}"
            case = load_case(path, variant)
            if case is not None:
                _check_design(case, v, energy, 1024 if name.startswith("ne-") else 4096,
                              10000 if name.startswith("ne-") else 20000)
            (cases if case is not None else pending).append(case if case is not None else str(path.resolve()))
        for variant in AUDIT_LABELS:
            # The standalone audit wrapper deliberately keeps a separate tree.
            # Never discover smoke or arbitrary recursive summaries here.
            case_name = f"v{v}_E{energy:.2f}".replace(".", "p")+"_"+variant.replace("-", "_")
            path = study/"isolated_audit"/case_name
            case = load_case(path, variant, audit=True) if case_name in completed_audits else None
            if case is not None:
                _check_design(case, v, energy, 512, 910000)
                declared = [row for row in audit_manifest.get("cases", []) if row.get("name") == case_name]
                if len(declared) != 1 or declared[0].get("config") != case.config or declared[0].get("preparation") != case.summary["preparation"]:
                    raise ValueError(f"Audit manifest/configuration mismatch: {path}")
            (audits if case is not None else pending).append(case if case is not None else str(path.resolve()))
    records = [case_record(case) for case in cases]
    index = {case.key: case for case in cases}
    pairs = []
    for v, energy in POINTS:
        for first, second, comparison, same in (
            ("ne_paper_subset", "ne_uncut", "Remove centroid cutoff", False),
            ("ne_uncut", "ne_switch200", "Add 200 fs switch to uncut NE", True),
            ("qct_harmonic", "qct_action_collision", "Use action QCT at fixed Ecol", True),
            ("qct_action_collision", "qct_action_total", "Retune action QCT to fixed actual Etot", True)):
            if (first, v, energy) in index and (second, v, energy) in index:
                pairs.append(pair_cases(index[first, v, energy], index[second, v, energy], comparison, same_conditioning=same))
    reference_path = baseline/"literature_reference/marjollet_2021_requested_energy_interpolation.csv"
    with reference_path.open(newline="") as stream:
        reference = {(int(r["v"]), round(float(r["requested_total_energy_ev"]), 6)):
                     {"value": float(r["linear_interpolated_cross_section_angstrom2"]),
                      "lower": float(r["digitization_only_lower_angstrom2"]),
                      "upper": float(r["digitization_only_upper_angstrom2"])} for r in csv.DictReader(stream)}
    h2_path = baseline/"h2_reference.json"
    h2 = _read_json(h2_path)
    quantum_levels = {r["v"]: r["energy_above_well_minimum_eV"] for r in h2["levels"]}
    energies = []
    for v in (0, 1):
        values = [r for r in records if r["v"] == v and r["action_internal_energy_ev"] is not None]
        if values:
            value = values[0]
            energies.append({"v": v, "harmonic_internal_label_ev": value["harmonic_internal_label_ev"],
                             "action_internal_energy_ev": value["action_internal_energy_ev"],
                             "exact_quantum_internal_energy_ev": quantum_levels[v],
                             "action_minus_exact_ev": value["action_internal_energy_ev"]-quantum_levels[v]})
    audit_cases = [audit_case_record(case) for case in audits]
    audit_rows = [row for case in audits for row in audit_statistics(case)]
    audit_index = {case.key: case for case in audits}
    audit_pairs, audit_outcomes = [], []
    for v, energy in POINTS:
        for first, second, label in (("direct-paper", "direct-none", "Remove centroid cutoff"),
                                     ("direct-none", "switch200-none", "Add 200 fs switch"),
                                     ("switch50-none", "switch200-none", "Increase switch from 50 to 200 fs")):
            if (first, v, energy) in audit_index and (second, v, energy) in audit_index:
                audit_outcomes.append(audit_pair_record(audit_index[first, v, energy], audit_index[second, v, energy], label))
                audit_pairs.extend(paired_audit_means(audit_index[first, v, energy], audit_index[second, v, energy], label))
    numeric_rows = [row for case in cases for row in [preparation_numerics(case)] if row is not None]
    numeric_rows += [row for case in audits for row in [preparation_numerics(case, audit=True)] if row is not None]
    unknown = [row for case in cases if case.variant != "ne_paper_subset" for row in unresolved_diagnostics(case)]
    sources = {r["path"]: r for case in cases+audits for r in case.sources}
    for path in (reference_path, h2_path, Path(__file__)):
        sources[str(path.resolve())] = {"path": str(path.resolve()), "sha256": _sha(path)}
    audit_definitions = audit_manifest.get("energy_definitions", {})
    if audit_manifest_bytes is not None:
        sources[str(audit_manifest_path.resolve())] = {"path": str(audit_manifest_path.resolve()),
                                                     "sha256": hashlib.sha256(audit_manifest_bytes).hexdigest()}
    code_hashes = {}
    for case in cases+audits:
        path = case.path/"provenance.json"
        if path.exists():
            for source in _read_json(path).get("files", []):
                code_hashes.setdefault(source["path"], set()).add(source["sha256"])
    if "wrapper_source" in audit_manifest:
        source = audit_manifest["wrapper_source"]
        code_hashes.setdefault(source["path"], set()).add(source["sha256"])
    report = {"schema": "pyrpmd.h3.preparation-analysis.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "complete": not pending and audit_manifest.get("status") == "completed", "pending_cases": pending,
              "audit_manifest_status": audit_manifest.get("status", "not_started"),
              "scope": "Preparation sensitivity; completed production cases only; no fitting or full-replication claim",
              "cross_sections": records, "paired_comparisons": pairs, "internal_energy_references": energies,
              "audit_statistics": audit_rows, "unresolved_endpoints": unknown,
              "audit_cases": audit_cases, "audit_paired_outcomes": audit_outcomes,
              "audit_paired_time_means": audit_pairs,
              "audit_energy_definitions": audit_definitions,
              "preparation_numerics": numeric_rows,
              "recorded_run_code_hashes": [{"path": path, "sha256_values": sorted(values)} for path, values in sorted(code_hashes.items())],
              "statistics": {"cross_sections": "Exact two-sided 95% Clopper-Pearson, including unknown identification bounds",
                             "paired": "Difference of simultaneous 97.5% CP discordance intervals; Bonferroni >=95% per contrast",
                             "audit": "SEM over independent trajectory time means; no temporal independence assumption"},
              "sources": list(sources.values())}
    with _output_generation(output) as generated:
        _json(generated/"preparation_analysis.json", report)
        _csv(generated/"preparation_cross_sections.csv", records)
        _csv(generated/"preparation_paired_comparisons.csv", pairs)
        _csv(generated/"audit_statistics.csv", audit_rows)
        _csv(generated/"audit_cases.csv", audit_cases)
        _csv(generated/"audit_paired_outcomes.csv", audit_outcomes)
        _csv(generated/"audit_paired_time_means.csv", audit_pairs)
        _csv(generated/"preparation_numerics.csv", numeric_rows)
        _csv(generated/"internal_energy_references.csv", energies)
        _json(generated/"unresolved_endpoint_diagnostics.json", {
            "scope": "Diagnostic only; original unresolved statuses retained in every estimate and denominator",
            "mass_au": MASS_AU, "velocity_definition": "centroid p / atomic mass; pair derivative = rhat dot (vj - vi)",
            "energy_definitions": {
                "initial_ring_hamiltonian_per_bead_ev": "Full collision H_P/P, with original BKMP2 zero at three separated H atoms; includes ring springs and relative translation; not primitive internal energy",
                "internal_full_potential_ev": "BC mean full isolated potential minus its well minimum",
                "internal_primitive_full_ev": "BC (K + Vfull - Vmin - spring)/P diagnostic",
                "internal_ring_hamiltonian_relative_ev": "BC (K + Vfull - Vmin + spring)/P; not the primitive diagnostic"},
            "trajectories": unknown})
        collision_plot(records, reference, generated/"figures")
        audit_plot(audits, quantum_levels, generated/"figures")
        html_report(report, generated)
    print(json.dumps({"complete": report["complete"], "completed_collision_estimates": len(records),
                      "paired_contrasts": len(pairs), "completed_audits": len(audits),
                      "unresolved_endpoints": len(unknown), "pending": len(pending), "output": str(output)}))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT/"runs/state_selected_reproduction")
    parser.add_argument("--study", type=Path, default=ROOT/"runs/state_preparation_study")
    parser.add_argument("--output", type=Path, default=ROOT/"runs/state_preparation_study/analysis")
    args = parser.parse_args()
    analyze(args.baseline, args.study, args.output)


if __name__ == "__main__":
    main()
