"""Explicit, unit-aware comparisons of saved PyRPMD results."""

from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from scipy.constants import R

from .reporting import _selected_cv_names


RATE_UNITS = {"cm3 molecule^-1 s^-1", "s^-1"}
SCAN_LABELS = {"beads": "Number of beads", "dt_fs": "Time step (fs)",
               "rinf_bohr": "Reactant separation Rinf (bohr)",
               "bmax_bohr": "Impact-parameter cutoff (bohr)", "samples": "PMF production sample count"}


def _finite(value):
    if value is None or isinstance(value, (bool, str, dict, list)):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _cv_references(config, system, selected):
    """Read effective references without constructing CVs or executing input."""
    specifications = config.get("cvs", [])
    if isinstance(specifications, dict):
        definitions = specifications
    else:
        definitions = {spec.get("name", f"CV {index + 1}"): spec
                       for index, spec in enumerate(specifications)}
    references = {}
    for name in selected:
        spec = definitions.get(name, {})
        if spec.get("kind") not in ("rpmdrate", "rpmdrate_xi"):
            continue
        try:
            reference = np.asarray(spec.get("reference", system.get("geometry")), dtype=float)
            natoms = len(system.get("masses", system.get("symbols", [])))
        except (TypeError, ValueError):
            return None
        if not natoms or reference.shape != (natoms, 3) or not np.isfinite(reference).all():
            return None
        references[name] = reference.tolist()
    return references


def _identity(run, *, isotope=False, varying=None):
    config, system = run.get("_config", {}), run.get("_system", {})
    if not config or not system or run.get("_comparison_identity_available") is False:
        return None
    if (run["task"] in ("fes", "metadynamics", "opes")
            and run.get("_measure", run.get("metadata", {}).get("measure")) not in ("cv_density", "rate_flux")):
        return None
    fes = run.get("_fes_result")
    selected = _selected_cv_names(config, run["task"], None if fes is None else fes.method)
    references = _cv_references(config, system, selected)
    if references is None:
        return None
    raw_system = system
    system = {k: system.get(k) for k in ("symbols", "masses", "fragments", "charge", "multiplicity", "cell", "pbc", "fixed_atoms")}
    if raw_system.get("fixed_atoms"):
        geometry = raw_system.get("geometry", [])
        system["fixed_geometry"] = [geometry[i] for i in raw_system["fixed_atoms"]] if geometry else None
    if run["task"] == "rate":
        # The rate reaction coordinate uses the saved TS bond-length reference.
        system["transition_state_geometry"] = raw_system.get("geometry")
    if isotope:
        system.pop("masses", None)
        system["symbols"] = ["H" if str(s) in ("D", "T") else s for s in system.get("symbols") or []]
    reaction = dict(run.get("_prepared_reaction", config.get("reaction", {})))
    if varying == "rinf_bohr":
        for key in ("rinf", "Rinf", "rinf_au"):
            reaction.pop(key, None)
    identity = {"task": "fes" if fes is not None else run["task"], "system": system, "potential": config.get("potential"),
                "model_files": run.get("_model_files"),
                "reaction": reaction, "site_rate": config.get("site_rate"), "surface": config.get("surface"),
                "cvs": config.get("cvs"),
                "selected_cvs": selected, "cv_references": references,
                "static_walls": config.get("fes", {}).get("metadynamics", config.get("metadynamics", {})).get("walls"),
                "measure": run.get("_measure", run.get("metadata", {}).get("measure"))}
    if run["task"] == "drpmd":
        collision = dict(config.get("collision", {}))
        collision.pop("trajectories", None)  # Statistical size is not the sampled observable.
        if varying == "bmax_bohr":
            for key in ("bmax", "bmax_au"):
                collision.pop(key, None)
        identity["scattering"] = {
            "collision": collision, "sampling": config.get("sampling"),
            "outcomes": config.get("outcomes"), "initial_ensemble": config.get("initial_ensemble"),
            "nonadiabatic": config.get("nonadiabatic"),
        }
    return json.dumps(identity, sort_keys=True, ensure_ascii=False)


def _scan_rates(run):
    if run.get("_rate") is not None:
        return [("rate", run["_rate"])]
    if (run["task"] != "drpmd"
            or run.get("_summary", {}).get("sampling") not in ("thermal_flux", "maxwell")
            or run.get("_config", {}).get("initial_ensemble")):
        return []
    rates = []
    for name, entry in run["metrics"].get("channels", {}).items():
        value = _finite(entry.get("rate_cm3_molecule_s"))
        if entry.get("estimate_accepted") is True and value is not None and value >= 0:
            rates.append((name, {"value": value, "unit": "cm3 molecule^-1 s^-1", "standard_error": None,
                                 "source_statistics": entry,
                                 "uncertainty_note": run["metrics"].get("channel_uncertainty_note")}))
    return rates


def read_reference(path):
    """Read explicitly supplied rates; no digitization or data are invented."""
    points = []
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"series", "temperature_K", "rate", "rate_unit", "source"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("Reference CSV requires series,temperature_K,rate,rate_unit,source")
        for line, row in enumerate(reader, 2):
            try:
                temperature, rate = float(row["temperature_K"]), float(row["rate"])
                error = float(row["standard_error"]) if row.get("standard_error", "").strip() else None
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric reference data at line {line}") from exc
            unit, source, series = (row[name].strip() for name in ("rate_unit", "source", "series"))
            if (not np.isfinite(temperature) or temperature <= 0 or not np.isfinite(rate) or rate <= 0
                    or error is not None and (not np.isfinite(error) or error < 0)):
                raise ValueError(f"Reference temperature/rate must be finite and positive, SE nonnegative: line {line}")
            if unit not in RATE_UNITS or not source or not series:
                raise ValueError(f"Reference series/source and a canonical rate_unit are required: line {line}")
            points.append({"series": series, "temperature_K": temperature, "rate": rate,
                           "rate_unit": unit, "standard_error": error, "source": source})
    if not points:
        raise ValueError("Reference CSV is empty")
    return points


def fit_arrhenius(temperatures, rates):
    """Unweighted descriptive fit of ln(k) vs 1000/T, without inferred rate SE."""
    t, k = np.asarray(temperatures, float), np.asarray(rates, float)
    if (t.ndim != 1 or k.shape != t.shape or len(t) < 3 or not np.isfinite(t).all()
            or not np.isfinite(k).all() or np.any(t <= 0) or np.any(k <= 0)):
        raise ValueError("Arrhenius fitting requires at least three finite positive temperatures and rates")
    if len(np.unique(t)) != len(t):
        raise ValueError("Repeated temperatures need an independently justified replicate aggregation before fitting")
    x, y = 1000.0 / t, np.log(k)
    xc = x - x.mean()
    slope = float(xc @ (y - y.mean()) / (xc @ xc))
    intercept = float(y.mean() - slope * x.mean())
    residual = y - (intercept + slope * x)
    total = float(np.sum((y - y.mean()) ** 2))
    return {"model": "k=A*exp(-Ea/(R*T))", "weighting": "unweighted log rates",
            "slope_per_1000_over_T": slope, "log_A": intercept,
            "A": float(np.exp(intercept)) if intercept <= np.log(np.finfo(float).max) else None,
            "activation_energy_kJ_mol": -slope * R,
            "r_squared_log": 1 - float(residual @ residual) / total if total > 0 else None,
            "temperature_range_K": [float(t.min()), float(t.max())], "points": len(t),
            "interpretation": "Descriptive fit over this interval; not a tunneling model or a convergence/uncertainty certificate."}


def _description(run):
    metadata = run["metadata"]
    parts = [run["label"]]
    if metadata.get("fes_method"):
        parts.append(metadata["fes_method"])
    for name, suffix in (("temperature_K", "K"), ("beads", "beads")):
        value = _finite(metadata.get(name))
        if value is not None:
            parts.append(f"{value:g} {suffix}")
    return ", ".join(parts)


def _make_groups(runs, series, varying):
    groups = defaultdict(list)
    if series is not None:
        if len(series) != len(runs) or any(not isinstance(s, str) or not s.strip() for s in series):
            raise ValueError("series must provide a nonempty grouping name per run")
        for name, run in zip(series, runs):
            groups[name].append(run)
        for name, group in groups.items():
            identities = {_identity(run, varying=varying) for run in group}
            if len(group) > 1 and (None in identities or len(identities) != 1):
                raise ValueError(f"Series {name!r} mixes incompatible or unavailable saved system/PES/reaction/measure identities")
        return groups
    keys = {}
    for index, run in enumerate(runs):
        identity = _identity(run, varying=varying)
        meta = run["metadata"]
        numeric = tuple((key, meta.get(key)) for key in ("beads", "dt_fs", "integrator") if key != varying)
        signature = (identity, numeric) if identity is not None else ("unavailable", index)
        if signature not in keys:
            p = meta.get("beads")
            keys[signature] = f"{run['task']}" + (f", P={p}" if p is not None and varying != "beads" else "") + f" (set {len(keys)+1})"
        groups[keys[signature]].append(run)
    return groups


def collect_comparisons(runs, *, series=None, reference=None, include_estimates=False,
                        arrhenius_fit=False, kie=None, convergence=None):
    result = {"plots": [], "findings": [], "skipped": [], "metrics": {}}
    if convergence is not None and convergence not in SCAN_LABELS:
        raise ValueError("Unknown convergence variable")
    groups = _make_groups(runs, series, convergence)
    result["metrics"]["series_members"] = {name: [run["id"] for run in group] for name, group in groups.items()}
    # Curves are overlaid only when saved measures, physical identities and axes
    # agree. Different temperatures/bead counts remain labelled on each curve.
    overlays = defaultdict(list)
    for run in runs:
        for plot in run["_plots"]:
            if "_compare" not in plot:
                continue
            identity = _identity(run, varying=convergence)
            if identity is None:
                continue
            key = (plot["_compare"], plot["xlabel"], plot["ylabel"], identity)
            overlays[key].append((run, plot))
    for i, ((kind, xlabel, ylabel, _), members) in enumerate(overlays.items()):
        if len(members) < 2:
            continue
        result["plots"].append({"id": f"comparison-{kind}-{i+1:02d}", "kind": "line",
            "title": "PMF comparison" if kind == "pmf" else "Transmission comparison", "xlabel": xlabel, "ylabel": ylabel,
            "series": [{**plot["series"][0], "label": _description(run)} for run, plot in members],
            "caption": "Saved gauges and uncertainty bands are retained: block-bootstrap bands are 1 SE; replica spread is descriptive 1 SD, not a standard error. Each label identifies its temperature and bead count; overlays do not establish convergence."})
    rate_series = defaultdict(list)
    accepted = {}
    fits = []
    for name, group in groups.items():
        entries = []
        for run in group:
            rate = run.get("_rate")
            temperature = _finite(run["metadata"].get("temperature_K"))
            if rate is None or temperature is None or temperature <= 0:
                continue
            value = _finite(rate["value"])
            if value is not None and value > 0:
                entries.append((temperature, value, rate["standard_error"], rate["unit"], run))
            elif value is not None:
                result["findings"].append({"level": "warning", "code": "nonpositive_log_rate", "run": run["label"],
                    "message": "Nonpositive saved rate omitted from logarithmic rate plots; value retained in the report."})
            if include_estimates and value is None and _finite(rate["estimate"]) is not None and rate["estimate"] > 0:
                rate_series[rate["unit"]].append({"label": f"{run['label']} (withheld diagnostic)",
                    "x": [1000 / temperature], "y": [rate["estimate"]], "style": "points"})
        entries.sort(key=lambda entry: entry[0])
        accepted[name] = entries
        units = {entry[3] for entry in entries}
        if len(units) > 1:
            raise ValueError(f"Series {name!r} mixes first-order and bimolecular rates")
        if not entries:
            if any("_rate" in run for run in group):
                result["skipped"].append({"figure": "Arrhenius rate", "run": name,
                    "reason": "No positive source-accepted rates with saved temperatures"})
            continue
        unit = entries[0][3]
        rate_series[unit].append({"label": name, "x": [1000 / e[0] for e in entries], "y": [e[1] for e in entries],
            "yerr": [np.nan if e[2] is None else e[2] for e in entries], "style": "points"})
        if arrhenius_fit:
            try:
                fit = fit_arrhenius([e[0] for e in entries], [e[1] for e in entries])
            except ValueError as error:
                result["skipped"].append({"figure": "Arrhenius fit", "run": name, "reason": str(error)})
            else:
                fits.append({"series": name, "rate_unit": unit, **fit})
                x = np.linspace(min(1000 / e[0] for e in entries), max(1000 / e[0] for e in entries), 100)
                rate_series[unit].append({"label": name + " (descriptive fit)", "x": x,
                    "y": np.exp(fit["log_A"] + fit["slope_per_1000_over_T"] * x), "style": "dashed"})
    if reference:
        references = read_reference(reference)
        result["metrics"]["reference_data"] = references
        grouped = defaultdict(list)
        for point in references:
            grouped[(point["rate_unit"], point["series"])].append(point)
        for (unit, name), points in grouped.items():
            points.sort(key=lambda point: point["temperature_K"])
            rate_series[unit].append({"label": name + " (reference)", "x": [1000 / p["temperature_K"] for p in points],
                "y": [p["rate"] for p in points], "yerr": [p["standard_error"] if p["standard_error"] is not None else np.nan for p in points],
                "style": "points"})
    result["metrics"]["arrhenius_fits"] = fits
    for i, (unit, curves) in enumerate(rate_series.items()):
        result["plots"].append({"id": f"comparison-rates-{i+1:02d}", "kind": "line", "title": "Temperature-dependent rates",
            "xlabel": "1000 / T (K^-1)", "ylabel": f"Rate coefficient ({unit})", "yscale": "log", "series": curves,
            "caption": "Saved source-accepted rates and explicitly supplied references. Error bars are 1 SE, where available. Explicitly requested withheld diagnostics are labelled and never fitted. First-order and bimolecular rates are separated."})
    if kie is not None:
        if len(kie) != 2 or kie[0] == kie[1] or any(name not in accepted for name in kie):
            raise ValueError("KIE requires two distinct existing series names (numerator, denominator)")
        numerator, denominator = accepted[kie[0]], accepted[kie[1]]
        def indexed(entries):
            values = {}
            for entry in entries:
                if entry[0] in values:
                    raise ValueError("KIE requires one accepted result per series and temperature; replicate aggregation must be explicit")
                values[entry[0]] = entry
            return values
        num, den = indexed(numerator), indexed(denominator)
        ratios = []
        for temperature in sorted(set(num) & set(den)):
            n, d = num[temperature], den[temperature]
            if n[3] != d[3]:
                raise ValueError("KIE numerator and denominator rate units differ")
            ni, di = _identity(n[4], isotope=True), _identity(d[4], isotope=True)
            if ni is None or di is None or ni != di:
                raise ValueError("KIE requires compatible saved reaction/PES/system identities apart from isotope masses")
            nm, dm = n[4]["_system"].get("masses"), d[4]["_system"].get("masses")
            if nm is None or dm is None or np.array_equal(np.asarray(nm), np.asarray(dm)):
                raise ValueError("KIE requires a recorded isotope mass change; identical or missing masses only define a generic rate ratio")
            ratio = n[1] / d[1]
            stderr = None if n[2] is None or d[2] is None else ratio * np.hypot(n[2] / n[1], d[2] / d[1])
            ratios.append({"temperature_K": temperature, "ratio": ratio, "standard_error": stderr})
        result["metrics"]["kinetic_isotope_effect"] = {"numerator": kie[0], "denominator": kie[1], "data": ratios,
            "uncertainty": "First-order delta method assuming independent numerator and denominator; no covariance or interpolation inferred."}
        if ratios:
            result["plots"].append({"id": "comparison-kie", "kind": "line", "title": f"Kinetic isotope effect: {kie[0]} / {kie[1]}",
                "xlabel": "Temperature (K)", "ylabel": "Rate ratio (dimensionless)",
                "series": [{"label": f"{kie[0]} / {kie[1]}", "x": [p["temperature_K"] for p in ratios],
                    "y": [p["ratio"] for p in ratios], "yerr": [np.nan if p["standard_error"] is None else p["standard_error"] for p in ratios]}],
                "caption": "Matched identical temperatures only. 1 SE uses independent-rate delta propagation where both rate errors exist; correlated runs require covariance-aware analysis."})
        else:
            result["skipped"].append({"figure": "KIE", "reason": "No matching temperatures with positive source-accepted rates"})
    if convergence:
        scans = defaultdict(list)
        scan_metrics = []
        for name, group in groups.items():
            buckets = defaultdict(list)
            for run in group:
                meta = run["metadata"]
                x, temperature = _finite(meta.get(convergence)), _finite(meta.get("temperature_K"))
                if x is None or temperature is None:
                    continue
                other = tuple((k, meta.get(k)) for k in ("beads", "dt_fs", "rinf_bohr", "bmax_bohr", "integrator") if k != convergence)
                for observable, rate in _scan_rates(run):
                    if _finite(rate["value"]) is not None:
                        kind = "scattering" if "source_statistics" in rate else "standard_error"
                        buckets[(rate["unit"], temperature, other, observable, kind)].append((x, rate, run))
            for (unit, temperature, _, observable, kind), points in buckets.items():
                points.sort(key=lambda p: p[0])
                if len({p[0] for p in points}) < 2:
                    continue
                label = f"{name}, {temperature:g} K" + (f", {observable}" if kind == "scattering" else "")
                scans[(unit, kind)].append({"label": label, "x": [p[0] for p in points],
                    "y": [p[1]["value"] for p in points], "yerr": [np.nan if p[1]["standard_error"] is None else p[1]["standard_error"] for p in points]})
                record = {"series": name, "temperature_K": temperature, "variable": convergence,
                          "observable": observable, "rate_unit": unit,
                          "run_ids": [p[2]["id"] for p in points], "physical_convergence_certified": False}
                if kind == "scattering":
                    record["source_statistics"] = [p[1]["source_statistics"] for p in points]
                    record["uncertainty_notes"] = [p[1]["uncertainty_note"] for p in points]
                scan_metrics.append(record)
        for i, ((unit, kind), curves) in enumerate(scans.items()):
            result["plots"].append({"id": f"comparison-scan-{i+1:02d}", "kind": "line", "title": "Numerical-parameter scan",
                "xlabel": SCAN_LABELS[convergence], "ylabel": f"Rate coefficient ({unit})", "series": curves,
                "caption": "Comparable source-accepted rates with fixed saved temperature and other listed numerical controls. "
                           + ("Scattering point estimates only; saved parent-cluster 95% intervals, IID Wilson intervals and outcome bounds remain distinctly labelled in metrics. "
                              if kind == "scattering" else "Bands are saved 1 SE. ")
                           + "Visual agreement does not certify convergence."})
        if not scans:
            result["skipped"].append({"figure": "parameter scan", "reason": "Need two comparable source-accepted rates differing in the selected parameter"})
        result["metrics"]["parameter_scans"] = scan_metrics
    return result
