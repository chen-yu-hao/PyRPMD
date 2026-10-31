"""Saved-data adapters for scattering, materials, transport and PCET reports.

These adapters never construct a potential or execute an input file.  Existing
workflow acceptance decisions and uncertainty labels are retained verbatim.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re

import numpy as np

from ..checkpoint import load_checkpoint
from ..constants import ATOMIC_TIME_S, BOHR_ANGSTROM, KB_AU
from ..units import unit_factor


_TIME_FS = ATOMIC_TIME_S / 1e-15
_STATUSES = ("product", "nonreactive", "unresolved", "failed")
_OBSERVABLES = (
    "potential_energy", "kinetic_primitive", "kinetic_centroid_virial",
    "internal_energy_centroid_virial", "ring_polymer_hamiltonian",
)


def _finite(value):
    return (not isinstance(value, (bool, np.bool_))
            and isinstance(value, (int, float, np.integer, np.floating))
            and np.isfinite(value))


def _vector(value, name, size=None, *, finite=True):
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or not len(result) or (size is not None and len(result) != size):
        raise ValueError(f"{name} has an invalid one-dimensional shape")
    if finite and not np.isfinite(result).all():
        raise ValueError(f"{name} contains nonfinite values")
    return result


def _clock(value, name="time"):
    result = _vector(value, name)
    if np.any(np.diff(result) <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    return result


def _safe_path(path, root):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("saved artifact resolves outside the selected run")
    return resolved


def _snapshot(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest, "size": path.stat().st_size}


def _verify_snapshot(path, before, result, **extra):
    after = _snapshot(path)
    if before != after:
        raise ValueError(f"Saved source changed while reading: {path}")
    for record in result["source_records"]:
        if record["path"] == before["path"] and record["sha256"] != before["sha256"]:
            raise ValueError(f"Saved source changed between reads: {path}")
    result["source_records"].append({**before, **extra})


def _npz(path, root, keys, result):
    path = _safe_path(path, root)
    before = _snapshot(path)
    with load_checkpoint(path) as archive:
        data = {key: archive[key].copy() for key in keys if key in archive}
        integrity = archive.envelope["integrity"]
    _verify_snapshot(path, before, result, integrity=integrity)
    return data


def _metadata(arrays):
    if "metadata" not in arrays:
        return {}
    metadata = json.loads(str(arrays["metadata"]))
    if not isinstance(metadata, dict):
        raise ValueError("trajectory metadata must be a JSON object")
    return metadata


def _series(label, x, y, **kwargs):
    return {"label": str(label), "x": np.asarray(x).tolist(),
            "y": np.asarray(y).tolist(), **kwargs}


def _finding(result, code, message, level="warning"):
    result["findings"].append({"level": level, "code": code, "message": message})


def _skip(result, figure, reason):
    result["skipped"].append({"figure": figure, "reason": str(reason)})


def _plot(result, prefix, label, suffix, title, xlabel, ylabel, **kwargs):
    result["plots"].append({"id": f"{prefix}-{suffix}",
                            "title": f"{label}: {title}" if label else title,
                            "xlabel": xlabel, "ylabel": ylabel, **kwargs})


def _channel_entry(entry):
    """Keep asymmetric intervals as intervals, never reinterpret them as SE."""
    answer = {key: entry.get(key) for key in (
        "count", "event_fraction", "rate_cm3_molecule_s", "cross_section_bohr2",
        "observed_lower_cm3_molecule_s", "possible_upper_cm3_molecule_s",
        "parent_cluster_bootstrap95", "candidate_only",
    )}
    for name in ("event_fraction", "rate_cm3_molecule_s", "cross_section_bohr2",
                 "observed_lower_cm3_molecule_s", "possible_upper_cm3_molecule_s"):
        if answer[name] is not None and not _finite(answer[name]):
            raise ValueError(f"{name} must be finite or unavailable")
    interval = answer["parent_cluster_bootstrap95"]
    if interval is not None:
        bounds = _vector(interval, "parent-cluster interval", 2)
        if bounds[1] < bounds[0]:
            raise ValueError("parent-cluster interval endpoints are reversed")
        answer["parent_cluster_bootstrap95"] = bounds.tolist()
    return answer


def _centroid_distances(data, metadata, times, path, config, result, prefix, label, index):
    """Display specified reaction pairs or explicitly nonbonded pair distances."""
    figure = "centroid-distances-" + path.stem
    if "centroid" not in data:
        _skip(result, figure, "Saved centroid coordinates are unavailable.")
        return
    try:
        system = config.get("system", {})
        if isinstance(system, dict) and np.asarray(system.get("pbc", False), dtype=bool).any():
            raise ValueError("Centroid pair-distance panels require nonperiodic dRPMD trajectories.")
        if metadata.get("units", {}).get("q", "bohr") != "bohr":
            raise ValueError("Saved centroid coordinates must be declared in bohr.")
        centroid = np.asarray(data["centroid"], dtype=float)
        if (centroid.ndim != 3 or centroid.shape[0] != len(times)
                or centroid.shape[1] < 2 or centroid.shape[2] != 3
                or not np.isfinite(centroid).all()):
            raise ValueError("Centroid coordinates must be finite with shape (saved frames, atoms >= 2, 3).")
        natoms = centroid.shape[1]
        reaction = config.get("reaction", {})
        if not isinstance(reaction, dict):
            raise ValueError("Saved reaction pair definitions must be a mapping.")
        pairs = {}
        for key, role in (("forming_bonds", "forming"), ("breaking_bonds", "breaking")):
            definitions = reaction.get(key, [])
            if not isinstance(definitions, (list, tuple)):
                raise ValueError("Explicit forming/breaking pairs must be lists of atom-index pairs.")
            for pair in definitions:
                if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                        or any(not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < natoms for i in pair)
                        or pair[0] == pair[1]):
                    raise ValueError("Explicit forming/breaking pairs require two distinct valid atom indices.")
                canonical = tuple(sorted(pair))
                pairs.setdefault(canonical, []).append(role)
        specified = bool(pairs)
        if not pairs:
            if natoms > 6:
                raise ValueError("More than six atoms and no explicit forming/breaking pairs; pair selection is required.")
            pairs = {pair: [] for pair in combinations(range(natoms), 2)}
        series = []
        for (first, second), roles in pairs.items():
            distance = np.linalg.norm(centroid[:, first] - centroid[:, second], axis=1) * BOHR_ANGSTROM
            if not np.isfinite(distance).all():
                raise ValueError("Centroid pair distances are nonfinite.")
            pair_label = f"Atoms {first + 1}–{second + 1}"
            if roles:
                pair_label += " (" + "/".join(dict.fromkeys(roles)) + ")"
            series.append(_series(pair_label, times * _TIME_FS, distance))
        _plot(result, prefix, label, f"centroid-distances-{index+1}",
              f"Centroid pair distances: {path.stem}", "Time (fs)", "Centroid pair distance (Å)",
              kind="line", series=series,
              caption=("Explicit forming/breaking pairs from the saved reaction definition. " if specified else
                       "All atom pairs for this small system; these labels do not infer chemical bonds. ")
                      + "Atom indices in the legend are 1-based. Distances are between ring-polymer centroids, not bead-averaged bond lengths. Saved frames are retained; no classification changes or interpolation are applied.")
    except (ValueError, TypeError, KeyError) as exc:
        _skip(result, figure, exc)


def _scattering(root, summary, config, result, prefix, label, factor, energy_unit):
    counts = summary.get("counts", {})
    total = summary.get("trajectories")
    valid_counts = (isinstance(counts, dict)
                    and isinstance(total, int) and not isinstance(total, bool) and total > 0
                    and all(isinstance(counts.get(k), int) and not isinstance(counts[k], bool)
                            and counts[k] >= 0 for k in _STATUSES)
                    and sum(counts[k] for k in _STATUSES) == total)
    if valid_counts:
        result["metrics"]["outcomes"] = {"trajectories": total, "counts": counts,
                                          "fractions": {k: counts[k] / total for k in _STATUSES}}
        _plot(result, prefix, label, "outcomes", "Trajectory outcomes", "Outcome", "Trajectories",
              kind="bar", labels=list(_STATUSES), y=[counts[k] for k in _STATUSES],
              caption="All launched trajectories are retained, including unresolved and failed outcomes.")
        if counts["unresolved"] or counts["failed"]:
            _finding(result, "unresolved_scattering", "Unresolved or failed trajectories remain; outcome bounds are not confidence intervals.")
    else:
        _skip(result, "outcomes", "Missing or inconsistent trajectory counts.")
    channels = summary.get("channels", {})
    if not isinstance(channels, dict):
        raise ValueError("scattering channels must be a mapping")
    rows = {"total_reaction": summary["total_reaction"]} if isinstance(summary.get("total_reaction"), dict) else {}
    rows.update(channels)
    accepted, channel_metrics = {}, {}
    fixed = summary.get("sampling") == "fixed_energy"
    key = "cross_section_bohr2" if fixed else "rate_cm3_molecule_s"
    withheld = (summary.get("all_trajectories_resolved") is False
                or summary.get("classification_unvalidated") is True
                or (valid_counts and bool(counts["unresolved"] or counts["failed"])))
    for name, entry in rows.items():
        if not isinstance(entry, dict):
            _skip(result, f"channel-{name}", "Channel entry is not a mapping.")
            continue
        try:
            channel_metrics[name] = _channel_entry(entry)
        except (ValueError, TypeError) as exc:
            _skip(result, f"channel-{name}", exc)
            continue
        value = entry.get(key)
        if not withheld and not entry.get("candidate_only", False) and _finite(value) and value >= 0:
            accepted[name] = float(value)
    result["metrics"]["channels"] = channel_metrics
    result["metrics"]["channel_statistics_note"] = (
        "Event fractions use all launched trajectories and are not thermal branching ratios for unequal velocity weights. "
        "Parent-cluster 95% intervals preserve the producer's dependence model; outcome bounds are not confidence intervals."
    )
    if accepted:
        errors = [[], []]
        for name, value in accepted.items():
            interval = channel_metrics[name]["parent_cluster_bootstrap95"]
            if interval is not None and interval[0] <= value <= interval[1]:
                errors[0].append(value - interval[0])
                errors[1].append(interval[1] - value)
            else:
                errors[0].append(None)
                errors[1].append(None)
                if interval is not None:
                    _finding(result, "interval_excludes_point", f"The saved parent-cluster interval for {name} excludes its point estimate; it remains in metrics but is not represented as an error bar.")
        _plot(result, prefix, label, "channel-estimates", "Supported channel estimates", "Channel",
              "Cross section (bohr²)" if fixed else "Rate (cm³ molecule⁻¹ s⁻¹)",
              kind="bar", labels=list(accepted), y=list(accepted.values()), yerr=errors,
              caption="Only supported published estimate fields are plotted. Error bars are saved asymmetric parent-cluster 95% intervals, not standard errors; unavailable intervals are omitted.")
    elif rows:
        _skip(result, "channel-estimates", "No supported rate or cross-section estimates; diagnostic bounds remain in report metrics.")
    _electronic(summary, result, prefix, label)
    energy_series, population_series = [], []
    diagnostics = []
    paths = [p for p in sorted((root / "trajectories").glob("*.npz"))
             if not p.name.endswith(".checkpoint.npz")][:4]
    for index, path in enumerate(paths):
        try:
            data = _npz(path, root, ("time_au", "energy_per_bead", "electronic_populations", "centroid", "metadata"), result)
            result["sources"].append(path)
            metadata = _metadata(data)
            units = metadata.get("units", {})
            if units.get("energy", "hartree") != "hartree" or units.get("time", "atomic") not in ("atomic", "atomic_time"):
                raise ValueError("unsupported scattering trajectory units")
            times = _clock(data["time_au"])
            energy = _vector(data["energy_per_bead"], "energy_per_bead", len(times))
            delta = energy - energy[0]
            energy_series.append(_series(path.stem, times * _TIME_FS, delta * factor))
            diagnostics.append({"trajectory": path.stem,
                                "maximum_saved_frame_energy_error_hartree_per_bead": float(np.abs(delta).max()),
                                "reference": "first_saved_frame"})
            if "electronic_populations" in data and not population_series:
                pops = np.asarray(data["electronic_populations"], dtype=float)
                if pops.ndim != 2 or pops.shape[0] != len(times) or not np.isfinite(pops).all():
                    raise ValueError("electronic population traces have an invalid shape or values")
                for state in range(pops.shape[1]):
                    population_series.append(_series(f"{path.stem}, state {state}", times * _TIME_FS, pops[:, state]))
            _centroid_distances(data, metadata, times, path, config, result, prefix, label, index)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            _skip(result, "trajectory-" + path.name, exc)
    if energy_series:
        _plot(result, prefix, label, "scattering-energy", "Scattering energy diagnostics", "Time (fs)",
              f"Δ(H/P) ({energy_unit})", kind="line", series=energy_series,
              caption="First four saved trajectories at most; differences use each first saved frame. Saved frames can miss excursions between output steps.")
        result["metrics"]["scattering_energy"] = diagnostics
    if population_series:
        _plot(result, prefix, label, "electronic-populations", "Signed electronic population estimator", "Time (fs)",
              "Signed mapping population", kind="line", series=population_series,
              caption="First available trajectory. Signed MMST estimators are not probabilities and are not clipped.")


def _electronic(summary, result, prefix, label):
    electronic = summary.get("nonadiabatic")
    if not isinstance(electronic, dict):
        return
    labels = electronic.get("state_labels", [])
    entry = electronic.get("channels", {}).get("total_reaction", {})
    values = entry.get("signed_observed_contributions")
    if values is None:
        return
    try:
        values = _vector(values, "signed electronic contributions", len(labels))
    except (ValueError, TypeError) as exc:
        _skip(result, "electronic-contributions", exc)
        return
    result["metrics"]["electronic"] = {
        "estimator": electronic.get("estimator"), "units": electronic.get("units"),
        "final_estimates_available": electronic.get("final_estimates_available", False),
        "total_reaction": entry,
    }
    _plot(result, prefix, label, "electronic-contributions", "Observed signed electronic contributions",
          "Electronic state", f"Signed contribution ({electronic.get('units', 'producer units')})",
          kind="bar", labels=[str(x) for x in labels], y=values.tolist(),
          caption="Diagnostic observed contributions retain signs. They are not final rates when the producer withholds final_estimates_available, and are not binomial probabilities.")


def _materials(root, summary, config, result, prefix, label, factor, energy_unit):
    is_pimd = summary.get("task", config.get("task")) == "pimd"
    series, gyration, energy_errors = [], [], []
    beads = config.get("beads")
    if not is_pimd and (not isinstance(beads, int) or isinstance(beads, bool) or beads < 1):
        _skip(result, "nve-energy", "A positive saved bead count is required for per-bead normalization.")
    for path in sorted((root / "samples").glob("materials_*.npz"))[:4]:
        try:
            data = _npz(path, root, ("times", "observables", "radii_gyration_squared", "metadata"), result)
            result["sources"].append(path)
            metadata = _metadata(data)
            if metadata.get("energy_unit", "hartree") != "hartree" or metadata.get("time_unit", "atomic_time") not in ("atomic", "atomic_time"):
                raise ValueError("unsupported materials trajectory units")
            times = _clock(data["times"])
            obs = np.asarray(data["observables"], dtype=float)
            names = metadata.get("observable_names", list(_OBSERVABLES))
            if (not isinstance(names, (list, tuple)) or len(set(names)) != len(names)
                    or obs.shape != (len(times), len(names)) or not np.isfinite(obs).all()):
                raise ValueError("materials observables disagree with their metadata")
            if is_pimd:
                for name in ("potential_energy", "internal_energy_centroid_virial"):
                    if name in names:
                        series.append(_series(f"{path.stem}: {name}", times * _TIME_FS, obs[:, names.index(name)] * factor))
            elif isinstance(beads, int) and not isinstance(beads, bool) and beads > 0 and "ring_polymer_hamiltonian" in names:
                hp = obs[:, names.index("ring_polymer_hamiltonian")]
                delta = (hp - hp[0]) / beads
                series.append(_series(path.stem, times * _TIME_FS, delta * factor))
                diagnostic = {"replica": path.stem, "maximum_saved_frame_error_hartree_per_bead": float(np.abs(delta).max())}
                beta = config.get("beta")
                if not (_finite(beta) and beta > 0):
                    temperature = config.get("temperature_K")
                    beta = 1 / (KB_AU * temperature) if _finite(temperature) and temperature > 0 else None
                if beta is not None and _finite(beta) and beta > 0:
                    diagnostic["maximum_saved_frame_error_kbt_per_bead"] = float(np.abs(delta).max() * beta)
                energy_errors.append(diagnostic)
            if "radii_gyration_squared" in data:
                rg = np.asarray(data["radii_gyration_squared"], dtype=float)
                if rg.ndim != 2 or rg.shape[0] != len(times) or not rg.shape[1] or not np.isfinite(rg).all() or np.any(rg < 0):
                    raise ValueError("radius-of-gyration samples have invalid values")
                gyration.append(_series(path.stem, times * _TIME_FS, np.sqrt(rg.mean(axis=1)) * BOHR_ANGSTROM))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            _skip(result, "materials-" + path.name, exc)
    if series:
        _plot(result, prefix, label, "materials-energy", "Equilibrium energy estimators" if is_pimd else "NVE energy diagnostics",
              "Sampling time (fs)" if is_pimd else "Time (fs)", f"Energy ({energy_unit})" if is_pimd else f"Δ(H/P) ({energy_unit})",
              kind="line", series=series,
              caption=("Thermostatted PIMD sampling time is not a physical dynamics clock; energy fluctuations are not conservation failures."
                       if is_pimd else "Energy-zero-independent per-bead differences from the first saved frame. Saved-frame conservation does not establish timestep convergence."))
    if gyration:
        _plot(result, prefix, label, "gyration", "Ring-polymer size", "Sampling time (fs)" if is_pimd else "Time (fs)",
              "√mean(Rg²) (Å)", kind="line", series=gyration,
              caption="Square root of the atom-averaged squared ring-polymer radius of gyration; first four saved replicas at most.")
    if energy_errors:
        result["metrics"]["nve_energy"] = energy_errors
    if is_pimd:
        result["metrics"]["equilibrium_estimators"] = summary.get("estimators", {})
        result["metrics"]["sampling_diagnostics"] = summary.get("sampling_diagnostics", [])


def _diffusion(root, summary, result, prefix, label):
    data = summary.get("diffusion")
    if not isinstance(data, dict):
        path = root / "results" / "diffusion.json"
        if not path.is_file():
            _skip(result, "diffusion", "Saved diffusion analysis is unavailable.")
            return
        path = _safe_path(path, root)
        before = _snapshot(path)
        data = json.loads(path.read_text())
        _verify_snapshot(path, before, result)
        result["sources"].append(path)
    if not isinstance(data, dict):
        raise ValueError("diffusion analysis must be a JSON object")
    units = data.get("input", {})
    if units.get("msd_unit", "bohr^2") != "bohr^2":
        raise ValueError("unsupported diffusion MSD units")
    times = _clock(data["lag_time_fs"], "lag time")
    msd = _vector(data["msd_au"], "MSD", len(times))
    _plot(result, prefix, label, "msd", "Centroid mean-square displacement", "Lag time (fs)", "MSD (Å²)",
          kind="line", series=[_series("MSD", times, msd * BOHR_ANGSTROM**2)],
          caption="Saved unwrapped-centroid estimator. The producer's diffusion acceptance and fitting range are retained in report metrics.")
    if data.get("vacf_au") is not None:
        velocity = _vector(data["vacf_au"], "VACF", len(times))
        _plot(result, prefix, label, "vacf", "Velocity autocorrelation", "Lag time (fs)", "VACF (Å² fs⁻²)",
              kind="line", series=[_series("VACF", times, velocity * (BOHR_ANGSTROM / _TIME_FS)**2)])
    if data.get("vacf_integral_m2_s") is not None:
        integral = _vector(data["vacf_integral_m2_s"], "VACF integral", len(times))
        _plot(result, prefix, label, "vacf-integral", "Integrated velocity autocorrelation", "Lag time (fs)", "Running integral (m² s⁻¹)",
              kind="line", series=[_series("VACF integral / dimension", times, integral)],
              caption="A finite-time running integral is a diagnostic; it is not automatically a converged diffusion coefficient.")
    result["metrics"]["diffusion"] = {key: data.get(key) for key in (
        "status", "diffusion_m2_s", "diffusion_tensor_m2_s", "raw_fit", "uncertainty", "input", "diagnostics", "physical_converged",
    )}
    if data.get("diffusion_m2_s") is None:
        _finding(result, "diffusion_withheld", "The saved analysis withholds the diffusion coefficient; MSD/VACF curves are diagnostic only.")


def _pcet(summary, result, prefix, label):
    rows = summary.get("rates", [])
    if not isinstance(rows, list):
        raise ValueError("PCET rates must be a row list")
    groups = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict) or not _finite(row.get("temperature_K")) or not _finite(row.get("potential_V")):
            _skip(result, "pcet-rate-row", "A PCET row lacks finite temperature or potential.")
            continue
        groups[(str(row.get("isotope", "unknown")), float(row["temperature_K"]))].append(row)
    series = []
    for (isotope, temperature), entries in sorted(groups.items()):
        entries = sorted(entries, key=lambda row: row["potential_V"])
        for direction in ("reduction", "oxidation"):
            x, y = [], []
            for row in entries:
                # Log-domain fields preserve valid very small rates without underflow.
                value = row.get("log_" + direction + "_rate_s")
                if _finite(value):
                    log10rate = value / np.log(10)
                else:
                    value = row.get(direction + "_rate_s")
                    if not _finite(value) or value <= 0:
                        continue
                    log10rate = np.log10(value)
                x.append(row["potential_V"])
                y.append(log10rate)
            if x:
                series.append(_series(f"{isotope}, {temperature:g} K, {direction}", x, y,
                                      style="solid" if direction == "reduction" else "dashed"))
    if series:
        reference = str(summary.get("potential_reference", "saved reference"))
        _plot(result, prefix, label, "pcet-rates", "Local PCET model rates", f"Electrode potential (V vs {reference})", "log₁₀(k / s⁻¹)",
              kind="line", series=series,
              caption="Local first-order golden-rule model rates. Numerical convergence does not validate the physical applicability of the electrode model.")
    else:
        _skip(result, "pcet-rates", "No finite positive rates or finite log-domain rates.")
    kie_groups = defaultdict(list)
    for row in summary.get("kinetic_isotope_effects", []):
        if isinstance(row, dict) and _finite(row.get("temperature_K")) and _finite(row.get("potential_V")):
            kie_groups[float(row["temperature_K"])].append(row)
    kie_series = []
    for temperature, entries in sorted(kie_groups.items()):
        for direction in ("reduction", "oxidation"):
            selected = sorted((r for r in entries if _finite(r.get(direction + "_KIE")) and r[direction + "_KIE"] > 0),
                              key=lambda r: r["potential_V"])
            if selected:
                kie_series.append(_series(f"{temperature:g} K, {direction}", [r["potential_V"] for r in selected],
                                          [r[direction + "_KIE"] for r in selected],
                                          style="solid" if direction == "reduction" else "dashed"))
    if kie_series:
        _plot(result, prefix, label, "pcet-kie", "PCET kinetic isotope effect", "Electrode potential (V, saved reference)",
              "kH / kD", kind="line", series=kie_series,
              caption="Ratios are the producer's matched-temperature and matched-potential H/D model results.")
    result["metrics"]["pcet"] = {key: summary.get(key) for key in (
        "method", "normalization", "rate_unit", "potential_reference", "numerical_convergence",
        "physical_applicability_unverified", "rates", "kinetic_isotope_effects",
    )}


def collect_observables(run_path: Path, summary: dict, config: dict, *,
                        prefix: str, label: str, energy_unit="kcal/mol") -> dict:
    """Collect plot specifications and diagnostics from existing workflow output.

    Plot arrays are not decimated; individual trajectory/replica figures select
    at most the first four files and explicitly identify that scope.
    """
    root = Path(run_path).resolve()
    if not isinstance(summary, dict) or not isinstance(config, dict):
        raise ValueError("summary and config must be mappings")
    prefix = re.sub(r"[^A-Za-z0-9_-]+", "-", str(prefix)).strip("-") or "run"
    factor = 1 / unit_factor(energy_unit, "energy")
    result = {"plots": [], "metrics": {}, "findings": [], "skipped": [], "sources": [], "source_records": []}
    task = summary.get("task", config.get("task"))
    adapters = []
    if task == "drpmd":
        adapters.append(("scattering", lambda: _scattering(root, summary, config, result, prefix, label, factor, energy_unit)))
    if task in ("pimd", "diffusion"):
        adapters.append(("materials", lambda: _materials(root, summary, config, result, prefix, label, factor, energy_unit)))
    if task == "diffusion":
        adapters.append(("diffusion", lambda: _diffusion(root, summary, result, prefix, label)))
    if task == "pcet":
        adapters.append(("pcet", lambda: _pcet(summary, result, prefix, label)))
    for name, adapter in adapters:
        try:
            adapter()
        except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
            _finding(result, "malformed_" + name, f"Could not completely analyze saved {name} data: {exc}")
            _skip(result, name, exc)
    result["sources"] = list(dict.fromkeys(result["sources"]))
    return result
