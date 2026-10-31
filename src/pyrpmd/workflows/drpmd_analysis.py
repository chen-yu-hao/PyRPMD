"""dRPMD outcome estimators, uncertainty and summary artifacts.

Collision assembly, centroid connectivity and flux estimators are adapted from
chen-yu-hao/RPMDrate, commit 9a44d6b378d8bf835d67e1b3b7f78b27bdc84a78,
rpmdrate/drpmd.py (MIT). Original RPMDrate copyright (c) 2012 Joshua W. Allen,
William H. Green and Yury V. Suleimanov; see THIRD_PARTY_NOTICES.md and LICENSE.
"""

from __future__ import annotations

from collections import Counter
import csv
import json
import math
from pathlib import Path

import numpy as np

from .drpmd_classification import _STATUSES
from .drpmd_sampling import _field, validate_fragments

RATE_AU_TO_CM3_S = (5.29177210903e-9) ** 3 / 2.4188843265857e-17


def _wilson(successes, n, z=1.959963984540054):
    p = successes / n
    denominator = 1 + z * z / n
    middle = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, middle - half), min(1.0, middle + half)]


def _electronic_summary(records, config, weights, channels, allowed):
    """Joint signed estimates; one parent draw is shared by every output column."""
    na = config["nonadiabatic"]
    if set(channels) & {"total_reaction", "nonreactive"}:
        raise ValueError("NRPMD product channel names total_reaction and nonreactive are reserved")
    labels = na.get("state_labels", [str(i) for i in range(na["nstates"])])
    population = np.asarray([r["electronic_populations"] for r in records], dtype=float)
    if population.shape != (len(records), na["nstates"]) or not np.isfinite(population).all():
        raise ValueError("NRPMD results have missing or invalid electronic populations")
    norm_error = float(np.max(np.abs(population.sum(axis=1) - 1.0)))
    if norm_error > 1e-8:
        raise ValueError("NRPMD electronic populations do not sum to one")
    indicators = {
        "total_reaction": np.array([r["status"] == "product" for r in records]),
        "nonreactive": np.array([r["status"] == "nonreactive" for r in records]),
    }
    indicators.update({name: np.array([r["status"] == "product" and r["channel"] == name
                                      for r in records]) for name in channels})
    columns, values, entries = [], [], {}
    fixed = config["collision"].get("sampling", "thermal_flux") == "fixed_energy"
    for channel, indicator in indicators.items():
        joint = weights[:, None] * indicator[:, None] * population
        chemical = float(np.mean(weights * indicator))
        means = joint.mean(axis=0)
        entries[channel] = {
            "signed_observed_contributions": means.tolist(),
            "rate_cm3_molecule_s": means.tolist() if allowed and not fixed else None,
            "cross_section_bohr2": means.tolist() if allowed and fixed else None,
            "parent_cluster_bootstrap95": None,
            "chemical_observed_contribution": chemical,
            "electronic_sum_error": float(means.sum() - chemical),
        }
        columns.extend([[channel, label] for label in labels])
        values.append(joint)
    matrix = np.concatenate(values, axis=1)
    covariance = None
    parents = sorted({r["parent_id"] for r in records})
    if allowed and len(parents) >= 2:
        groups = [np.array([i for i, r in enumerate(records) if r["parent_id"] == parent])
                  for parent in parents]
        rng = np.random.default_rng(np.random.SeedSequence([int(config.get("seed", 0)), 7]))
        draws = np.empty((400, matrix.shape[1]))
        for i in range(len(draws)):
            idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
            draws[i] = matrix[idx].mean(axis=0)
        covariance = np.atleast_2d(np.cov(draws, rowvar=False, ddof=1)).tolist()
        bounds = np.quantile(draws, [0.025, 0.975], axis=0).T
        for i, entry in enumerate(entries.values()):
            entry["parent_cluster_bootstrap95"] = bounds[i * len(labels):(i+1) * len(labels)].tolist()
    return {
        "method": "nrpmd", "experimental": True, "physically_validated": False,
        "initial_state": na["initial_state"], "state_labels": labels,
        "estimator": "bead_mean_MMST_signed", "population_sum_max_error": norm_error,
        "final_estimates_available": bool(allowed),
        "channels": entries, "covariance_columns": columns,
        "parent_cluster_bootstrap_covariance": covariance,
        "units": "bohr^2" if fixed else "cm^3 molecule^-1 s^-1",
        "note": "Signed mapping populations are not probabilities and are never clipped. Observed contributions use all launched trajectories in the denominator and are diagnostics only when unresolved/failed trajectories or unvalidated cutoffs/classification withhold final estimates. No binomial interval or probability bound applies. This is specified-incident-state scattering, not an electronic thermal-equilibrium rate.",
    }


def summarize(records, config, system):
    n = len(records)
    if not n:
        raise ValueError("Cannot summarize zero trajectories")
    counts = Counter(r["status"] for r in records)
    if set(counts) - set(_STATUSES):
        raise ValueError("Unrecognized trajectory status")
    masses = np.asarray(_field(system, "masses"), dtype=float)
    fragments = validate_fragments(_field(system, "fragments"), len(masses))
    fm = np.array([masses[list(f)].sum() for f in fragments])
    mu, beta = fm.prod() / fm.sum(), config["beta"]
    area = math.pi * config["collision"]["bmax_au"] ** 2
    sampling = config["collision"].get("sampling", "thermal_flux")
    electronic_weight = float(config.get("reaction", {}).get("electronic_weight", 1.0))
    if electronic_weight < 0 or not math.isfinite(electronic_weight):
        raise ValueError("Electronic weight must be finite and nonnegative")
    if sampling == "thermal_flux":
        weights = np.full(
            n,
            area
            * math.sqrt(8 / (math.pi * beta * mu))
            * RATE_AU_TO_CM3_S
            * electronic_weight,
        )
    elif sampling == "maxwell":
        weights = np.array(
            [
                area * r["launch"]["speed_au"] * RATE_AU_TO_CM3_S * electronic_weight
                for r in records
            ]
        )
    else:
        weights = np.full(n, area)
    unknown = np.array([r["status"] in ("unresolved", "failed") for r in records])
    outcomes = config.get("outcomes", {})
    classifier_unvalidated = bool(outcomes.get("classifier_unvalidated", False))
    cutoff_unvalidated = bool(
        config.get("collision", {}).get("bmax_unvalidated", False)
    )
    rate_allowed = (
        sampling != "fixed_energy"
        and not cutoff_unvalidated
        and not classifier_unvalidated
    )
    channels = sorted(
        set(config.get("outcomes", {}).get("channels", {}))
        | {r["channel"] for r in records if r["status"] == "product"}
    )
    parent_ids = sorted({r["parent_id"] for r in records})

    def estimate(indicator):
        indicator = np.asarray(indicator, dtype=bool)
        values = weights * indicator
        lower, upper = float(values.mean()), float((values + weights * unknown).mean())
        point = lower if not unknown.any() and rate_allowed else None
        entry = {
            "count": int(indicator.sum()),
            "rate_cm3_molecule_s": point,
            "event_fraction": float(indicator.mean()),
            "candidate_only": classifier_unvalidated,
            "observed_lower_cm3_molecule_s": (
                lower
                if sampling != "fixed_energy" and not classifier_unvalidated
                else None
            ),
            "possible_upper_cm3_molecule_s": (
                upper
                if sampling != "fixed_energy" and not classifier_unvalidated
                else None
            ),
            "cross_section_bohr2": (
                lower
                if sampling == "fixed_energy"
                and not unknown.any()
                and not classifier_unvalidated
                and not cutoff_unvalidated
                else None
            ),
            "iid_wilson95_cm3_molecule_s": None,
            "parent_cluster_bootstrap95": None,
        }
        if sampling == "thermal_flux" and not unknown.any() and rate_allowed:
            entry["iid_wilson95_cm3_molecule_s"] = [
                float(weights[0] * p) for p in _wilson(int(indicator.sum()), n)
            ]
        # Parent-level resampling preserves dependence caused by reusing a chain.
        if len(parent_ids) >= 2 and not unknown.any() and rate_allowed:
            rng = np.random.default_rng(
                np.random.SeedSequence([int(config.get("seed", 0)), 7])
            )
            groups = [
                np.array([i for i, r in enumerate(records) if r["parent_id"] == parent])
                for parent in parent_ids
            ]
            draws = []
            for _ in range(400):
                idx = np.concatenate(
                    [groups[i] for i in rng.integers(0, len(groups), len(groups))]
                )
                draws.append(float(values[idx].mean()))
            entry["parent_cluster_bootstrap95"] = [
                float(x) for x in np.quantile(draws, [0.025, 0.975])
            ]
        return entry

    summary = {
        "format": "PyRPMD.drpmd.summary.v1",
        "task": "drpmd",
        "trajectories": n,
        "counts": {s: counts[s] for s in _STATUSES},
        "parents": len(parent_ids),
        "all_trajectories_resolved": not bool(unknown.any()),
        "temperature_K": None if config.get("initial_ensemble") else config["temperature_K"],
        "beads": config["beads"],
        "sampling": sampling,
        "electronic_weight": electronic_weight,
        "classification_unvalidated": classifier_unvalidated,
        "classification": {
            "mode": outcomes.get(
                "classifier_mode",
                (
                    "python_factory"
                    if outcomes.get("factory")
                    else "explicit_connectivity"
                ),
            ),
            "unvalidated": classifier_unvalidated,
            "bond_cutoffs_bohr": outcomes.get(
                "bond_cutoffs_au", outcomes.get("bond_cutoffs_bohr", {})
            ),
            "inferred_radii_angstrom": outcomes.get("inferred_radii_angstrom", {}),
            "inference_scale": outcomes.get("inference_scale"),
            "reference_bonds": outcomes.get("reference_bonds"),
            "reactant_formulas": outcomes.get("reactant_formulas"),
            "assumptions": outcomes.get("classification_assumptions", []),
        },
        "total_reaction": estimate([r["status"] == "product" for r in records]),
        "channels": {
            channel: estimate(
                [r["status"] == "product" and r["channel"] == channel for r in records]
            )
            for channel in channels
        },
        "uncertainty_note": "Outcome bounds are not confidence limits. IID Wilson intervals ignore parent-chain dependence; parent-cluster bootstrap requires enough independent chains and does not establish numerical or cutoff convergence.",
        "convergence_note": "Check timestep, bead count, thermalization, sample autocorrelation, R0, bmax, maximum time and classification. Fixed-energy data are not a thermal rate.",
    }

    if config.get("initial_ensemble"):
        summary["initialization_mode"] = "independent"
        summary["beta_au"] = beta
        summary["temperature_interpretation"] = config.get(
            "temperature_interpretation", "auxiliary_ring_parameter_not_thermal_temperature")
        summary["convergence_note"] = (
            "Approximate specified-initial-vibrational-state scattering. Check the "
            "harmonic reference, preparation beta, beads, timestep, precollision "
            "state stability, R0, bmax, maximum time and classifier. These are "
            "initial-state-resolved cross sections, not final quantum state-to-state "
            "probabilities or a thermal rate.")
    else:
        summary["initialization_mode"] = "parent_chain"
    if sampling == "fixed_energy":
        from pyrpmd.analysis.scattering import summarize_fixed_energy_scattering

        scattering = summarize_fixed_energy_scattering(
            records, area_bohr2=area, initialization_mode=summary["initialization_mode"],
            channels=channels, collision_energy_au=config["collision"].get("energy_au"),
            classifier_validated=not classifier_unvalidated,
            cutoff_validated=not cutoff_unvalidated,
            impact_sampling=config["collision"].get("impact_sampling", "uniform_area"),
            bootstrap_seed=int(config.get("seed", 0)),
        )
        summary["fixed_energy_scattering"] = scattering
        summary["total_reaction"].update(scattering["total_reaction"])
        for channel, entry in scattering["channels"].items():
            summary["channels"][channel].update(entry)
        summary["uncertainty_note"] = (
            "Outcome-assignment bounds retain every launched trajectory and are not "
            "confidence limits. IID Wilson intervals require independent complete "
            "initial conditions; parent-chain samples use eligible cluster bootstrap "
            "intervals. Availability and reasons are recorded in fixed_energy_scattering.")

    if config.get("nonadiabatic"):
        summary["nonadiabatic"] = _electronic_summary(
            records, config, weights, channels,
            not unknown.any() and not cutoff_unvalidated and not classifier_unvalidated)
    return summary


def _write_summary(results, config, system, output, expected, *, write_metadata):
    records = [record for parent in results for record in parent["records"]]
    out = Path(output)
    # Per-trajectory commits are authoritative after controlled extensions or a
    # crash before a parent ledger update. Preserve parent-only sampling fields.
    for i, record in enumerate(records):
        path = out / "trajectories" / f"trajectory-{record['trajectory_id']:08d}.json"
        if path.exists():
            current = json.loads(path.read_text())
            if (current.get("physics_hash") != record.get("physics_hash")
                    or current.get("trajectory_id") != record["trajectory_id"]
                    or current.get("steps", -1) < record.get("steps", 0)):
                raise ValueError("Trajectory result conflicts with its parent ledger")
            records[i] = {**record, **current}
    records.sort(key=lambda record: record["trajectory_id"])
    if [r["trajectory_id"] for r in records] != list(range(expected)):
        raise RuntimeError("Trajectory ledger has missing or duplicate global IDs")
    summary = summarize(records, config, system)
    summary["parent_thermal_steps"] = {
        str(p["parent_id"]): p["thermal_steps"] for p in results
    }
    summary["records"] = records
    extension_path = out / "results" / "extensions.json"
    if extension_path.exists():
        ledger = json.loads(extension_path.read_text())
        if (ledger.get("format") != "PyRPMD.drpmd.extensions.v1"
                or ledger.get("physics_hash") != records[0].get("physics_hash")
                or not ledger.get("requests")):
            raise ValueError("Extension ledger does not match this trajectory ensemble")
        summary["original_max_steps"] = ledger["original_max_steps"]
        summary["effective_max_steps"] = ledger["requests"][-1]["max_steps"]
        summary["extensions"] = ledger
    if config.get("initial_ensemble"):
        from pyrpmd.initial_states import prepare_initial_ensemble
        metadata = prepare_initial_ensemble(
            config["initial_ensemble"], system, config["beads"], config["beta"]).metadata
        summary["initial_ensemble"] = metadata
        write_metadata(out / "states" / "initial-ensemble.json", metadata)
    write_metadata(out / "results" / "summary.json", summary)
    with (out / "results" / "channels.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "channel",
                "count",
                "rate_cm3_molecule_s",
                "observed_lower_cm3_molecule_s",
                "possible_upper_cm3_molecule_s",
                "cross_section_bohr2",
                "observed_lower_bohr2",
                "possible_upper_bohr2",
                "iid_wilson95_lower_bohr2",
                "iid_wilson95_upper_bohr2",
                "parent_cluster_bootstrap95_lower_bohr2",
                "parent_cluster_bootstrap95_upper_bohr2",
            ]
        )
        for channel, entry in {
            "total_reaction": summary["total_reaction"],
            **summary["channels"],
        }.items():
            writer.writerow(
                [
                    channel,
                    entry["count"],
                    entry["rate_cm3_molecule_s"],
                    entry["observed_lower_cm3_molecule_s"],
                    entry["possible_upper_cm3_molecule_s"],
                    entry.get("cross_section_bohr2"),
                    entry.get("observed_lower_bohr2"),
                    entry.get("possible_upper_bohr2"),
                    *(entry.get("iid_wilson95_bohr2") or [None, None]),
                    *(entry.get("parent_cluster_bootstrap95_bohr2") or [None, None]),
                ]
            )
    if "nonadiabatic" in summary:
        with (out / "results" / "electronic_channels.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["chemical_channel", "electronic_state", "signed_observed_contribution",
                             "rate_cm3_molecule_s", "cross_section_bohr2"])
            na = summary["nonadiabatic"]
            for channel, entry in na["channels"].items():
                for i, label in enumerate(na["state_labels"]):
                    writer.writerow([channel, label, entry["signed_observed_contributions"][i],
                                     entry["rate_cm3_molecule_s"][i] if entry["rate_cm3_molecule_s"] is not None else None,
                                     entry["cross_section_bohr2"][i] if entry["cross_section_bohr2"] is not None else None])
    return summary
