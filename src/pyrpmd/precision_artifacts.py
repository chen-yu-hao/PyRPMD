"""Read published thermal-rate diagnostics without inventing precision evidence.

The adapter retains the producer's uncertainty scope and assumptions. In
particular, a legacy bootstrap standard error is not proof of independence of
PMF and recrossing samples, and parent configurations are not parent chains.
No potential is loaded and no dynamics or statistical resampling is performed.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .io import file_sha256
from .rate_artifacts import active_rate_directory


def _safe_path(root, relative):
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Rate artifact must remain inside its run directory: {relative}")
    return path


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _number(value, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value) or (nonnegative and value < 0):
        return None
    return value


def _count(value):
    return value if type(value) is int and value >= 0 else None


def _flag(value):
    return value if type(value) is bool else None


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_rate_diagnostics(directory):
    """Return a finite JSON snapshot of the active, atomically published rate.

    Missing diagnostics remain ``None`` and are listed in ``missing``. The
    snapshot is suitable for pilot planning; using it for production acceptance
    additionally requires the controller's frozen-design and stream-provenance
    checks. ``source_family_id`` identifies the original run across extensions,
    while ``source_ids`` identify this exact published snapshot. Neither is an
    independent-sample count.
    """
    root = Path(directory).resolve()
    if not (root / "summary.json").is_file() and not (root / "results/summary.json").is_file():
        candidate = root / "run"
        if candidate.is_dir():
            root = candidate.resolve()
    hashes, absent = {}, set()

    def read_json(relative, *, required=False):
        path = _safe_path(root, relative)
        if not path.is_file():
            absent.add(str(relative))
            if required:
                raise ValueError(f"Missing rate artifact: {relative}")
            return {}
        before = file_sha256(path)
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object: {relative}")
        if file_sha256(path) != before:
            raise ValueError("Rate artifacts changed during diagnostic snapshot; retry")
        hashes[str(relative)] = before
        return _json_safe(value)

    def read_arrays(relative):
        path = _safe_path(root, relative)
        if not path.is_file():
            absent.add(str(relative))
            return {}
        before = file_sha256(path)
        with np.load(path, allow_pickle=False) as saved:
            arrays = {name: saved[name] for name in saved.files}
        if file_sha256(path) != before:
            raise ValueError("Rate artifacts changed during diagnostic snapshot; retry")
        hashes[str(relative)] = before
        return arrays

    publication = read_json("results/summary.json")
    # The old root summary remains the original generation after extension.
    if "rate_extension" in publication:
        summary = publication
    else:
        summary = read_json("summary.json") or publication
    if not summary:
        raise ValueError("A saved thermal rate summary is required")
    artifacts = active_rate_directory(root, publication or summary)
    relative = artifacts.relative_to(root)
    if artifacts != root:
        read_json(relative / "receipt.json", required=True)
    if summary.get("task") != "rate":
        raise ValueError("Precision diagnostics currently accept only thermal gas RPMD rate runs")
    config = read_json("input/resolved.json")
    manifest = read_json("manifest.json")
    recrossing_config = _mapping(config.get("recrossing", _mapping(config.get("rate")).get("recrossing")))
    if (summary.get("dynamics") == "trpmd" or summary.get("status") == "trpmd_sensitivity_estimate"
            or recrossing_config.get("dynamics") == "trpmd"):
        raise ValueError("TRPMD sensitivity estimates are outside thermal RPMD precision acceptance")
    if config.get("task", "rate") != "rate":
        raise ValueError("Saved input and thermal rate summary disagree on task")

    pmf = read_json(relative / "rate_pmf.json")
    if not pmf:
        pmf = _mapping(summary.get("pmf"))
    stored_pmf = _mapping(summary.get("pmf"))
    for key in ("method", "xi_star", "free_energy_barrier_hartree"):
        if key in pmf and key in stored_pmf and pmf[key] != stored_pmf[key]:
            raise ValueError(f"Rate summary and PMF disagree on {key}")
    if pmf.get("sampling_measure", "rate_flux") != "rate_flux":
        raise ValueError("Thermal rate PMF must use the rate_flux measure")
    pmf_arrays = read_arrays(relative / "rate_pmf.npz")
    for key in ("method", "ready", "sampling_measure"):
        if key in pmf_arrays:
            value = pmf_arrays[key]
            if value.ndim != 0:
                raise ValueError(f"PMF {key} must be scalar")
            if key in pmf and pmf[key] != value.item():
                raise ValueError(f"PMF JSON and NPZ disagree on {key}")
    umbrella = read_json(relative / "umbrella/summary.json")
    umbrella_arrays = read_arrays(relative / "umbrella/fes.npz")
    overlap = pmf_arrays.get("overlap", umbrella_arrays.get("overlap"))
    if overlap is not None:
        if (overlap.dtype.kind not in "iuf" or overlap.ndim != 2 or overlap.shape[0] != overlap.shape[1]
                or not np.isfinite(overlap).all() or np.any(overlap < 0)):
            raise ValueError("Invalid saved umbrella overlap matrix")
    uncertainty = _mapping(summary.get("rate_uncertainty"))
    parent_support = _mapping(uncertainty.get("recrossing_support"))
    pmf_uncertainty = _mapping(pmf.get("uncertainty", pmf.get("barrier_uncertainty")))
    ess = pmf.get("effective_samples_per_stream", umbrella.get("effective_samples_per_stream",
                  _json_safe(umbrella_arrays.get("effective_samples_per_stream"))))
    if ess is not None:
        try:
            ess_array = np.asarray(ess, dtype=float)
        except (TypeError, ValueError):
            ess_array = np.array([np.nan])
        if (ess_array.ndim != 1 or not ess_array.size or not np.isfinite(ess_array).all()
                or np.any(ess_array < 0)):
            ess = None
        else:
            ess = ess_array.tolist()
    rec = read_arrays(relative / "recrossing.npz")
    parent_count = _count(summary.get("parent_configurations"))
    chain_count_from_ids = None
    horizon = None
    steps = None
    if rec:
        needed = {"parent_ids", "parent_numerator", "parent_denominator", "time_au", "kappa"}
        if not needed.issubset(rec):
            raise ValueError("Saved recrossing artifact lacks parent identity/flux/time records")
        ids, num, den, times, curve = (rec[k] for k in (
            "parent_ids", "parent_numerator", "parent_denominator", "time_au", "kappa"))
        if (ids.ndim != 2 or ids.shape[1] != 2 or ids.dtype.kind not in "iu"
                or np.any(ids < 0) or len({tuple(x) for x in ids.tolist()}) != len(ids)
                or any(value.dtype.kind not in "iuf" for value in (num, den, times, curve))
                or num.ndim != 2 or num.shape[0] != len(ids)
                or den.shape != (len(ids),) or not np.isfinite(num).all()
                or not np.isfinite(den).all() or np.any(den < 0) or den.sum() <= 0
                or np.any(num[den == 0] != 0)
                or times.shape != (num.shape[1],) or not len(times)
                or not np.isfinite(times).all() or np.any(times <= 0) or np.any(np.diff(times) <= 0)
                or curve.shape != times.shape or not np.isfinite(curve).all()):
            raise ValueError("Invalid saved recrossing parent/flux/time arrays")
        if not np.allclose(num.sum(axis=0) / den.sum(), curve, rtol=1e-10, atol=1e-12):
            raise ValueError("Saved recrossing curve disagrees with parent flux sums")
        if parent_count is not None and parent_count != len(ids):
            raise ValueError("Saved parent configuration count disagrees with parent identities")
        parent_count = len(ids)
        chain_count_from_ids = len(np.unique(ids[:, 0]))
        horizon, steps = float(times[-1]), len(times)
    independent = _count(parent_support.get("independent_parent_chains"))
    if independent is not None and chain_count_from_ids is not None and independent > chain_count_from_ids:
        raise ValueError("Independent parent-chain count exceeds saved chain identities")
    child_pairs = _count(summary.get("child_pairs"))
    checked = _count(summary.get("energy_diagnostic_child_pairs"))
    energy = _number(summary.get("maximum_endpoint_energy_error_kbt"), nonnegative=True)
    energy_complete = None if checked is None or child_pairs is None else checked == child_pairs and child_pairs > 0
    if checked is not None and child_pairs is not None and checked > child_pairs:
        raise ValueError("Energy diagnostic count exceeds child-pair count")
    production = _mapping(summary.get("precision_provenance"))
    # Preserve producer declarations for the controller to authenticate against
    # its frozen ledger. Legacy status='plateau_estimate' is not production.
    phase = "unknown"
    if production.get("format") == "PyRPMD.precision.production.v1":
        if production.get("phase") == "pilot":
            phase = "pilot"
        elif (production.get("phase") == "production"
              and production.get("frozen_before_sampling") is True
              and production.get("independent_from_pilot") is True
              and isinstance(production.get("epoch_id"), str) and production["epoch_id"]
              and isinstance(production.get("design_id"), str) and production["design_id"]
              and production.get("physics_fingerprint") == manifest.get("physics_fingerprint")
              and manifest.get("physics_fingerprint") is not None):
            phase = "production"

    missing = []
    fields = {
        "pmf": {
            "ready": _flag(pmf.get("ready")),
            "method": pmf.get("method"),
            "barrier_hartree": _number(pmf.get("free_energy_barrier_hartree")),
            "barrier_standard_error_hartree": _number(pmf.get("barrier_standard_error_hartree"), nonnegative=True),
            "uncertainty": pmf_uncertainty,
            "overlap_connected": _flag(pmf.get("overlap_connected", umbrella.get("overlap_connected"))),
            "overlap": _json_safe(overlap),
            "effective_samples_per_stream": ess,
            "minimum_effective_samples": min(ess) if ess else None,
            "weight_ess": _number(pmf.get("weight_ess"), nonnegative=True),
            "support": _mapping(pmf.get("support")),
            "bootstrap": _mapping(pmf.get("bootstrap")),
            "shared_source_sha256": pmf.get("source_sha256"),
        },
        "recrossing": {
            "kappa": _number(summary.get("kappa")),
            "standard_error": _number(summary.get("kappa_standard_error"), nonnegative=True),
            "uncertainty_scope": parent_support.get("scope", uncertainty.get("recrossing_scope")),
            "uncertainty": parent_support,
            "independent_parent_chains": independent,
            "saved_chain_ids_count": chain_count_from_ids,
            "parent_configurations": parent_count,
            "effective_parent_blocks": parent_support.get("effective_parent_blocks"),
            "child_pairs": child_pairs,
            "child_steps": steps,
            "child_horizon_au": horizon,
            "child_time_fs": _number(summary.get("child_time_fs"), nonnegative=True),
            "plateau_detected": _flag(summary.get("plateau_detected")),
            "plateau_diagnostics": _mapping(summary.get("plateau_diagnostics")),
            "plateau_drift": _number(summary.get("plateau_drift")),
            "plateau_drift_standard_error": _number(summary.get("plateau_drift_standard_error"), nonnegative=True),
            "variance_components": None,
        },
        "energy": {
            "maximum_endpoint_error_kbt_per_bead": energy,
            "diagnosed_child_pairs": checked,
            "complete": energy_complete,
            "within_workflow_tolerance": energy <= .1 if energy is not None and energy_complete is True else None,
            "workflow_tolerance_kbt_per_bead": .1,
            "scope": "child_endpoints_only",
            "maximum_scaled_energy_drift": _number(summary.get("maximum_scaled_energy_drift"), nonnegative=True),
        },
    }
    for section, names in {
        "pmf": ("barrier_standard_error_hartree", "overlap_connected", "minimum_effective_samples"),
        "recrossing": ("standard_error", "independent_parent_chains", "plateau_detected", "variance_components"),
        "energy": ("maximum_endpoint_error_kbt_per_bead", "complete"),
    }.items():
        missing.extend(section + "." + name for name in names if fields[section][name] is None)
    # Independence/covariance and frozen production cannot be inferred from an
    # RNG seed or from a point estimate passing the legacy plateau check.
    missing.extend(["pmf_recrossing_covariance_or_independence_evidence", "validated_frozen_production_provenance"])
    fingerprint = manifest.get("physics_fingerprint")
    if fingerprint is None:
        missing.append("physics_fingerprint")
    identity_files = {name: hashes[name] for name in ("manifest.json", "input/resolved.json") if name in hashes}
    family = _digest({"physics_fingerprint": fingerprint, "inputs": identity_files}) if identity_files else None
    if family is None:
        missing.append("source_family_identity")
    snapshot = _digest({"physics_fingerprint": fingerprint, "artifacts": hashes})
    # Catch publication switches and modifications after individual reads.
    for name, digest in hashes.items():
        if file_sha256(_safe_path(root, name)) != digest:
            raise ValueError("Rate artifacts changed during diagnostic snapshot; retry")
    if any(_safe_path(root, name).is_file() for name in absent):
        raise ValueError("Rate artifacts changed during diagnostic snapshot; retry")
    return {
        "format": "PyRPMD.precision.rate-diagnostics.v1",
        "source_ids": [snapshot],
        "source_family_id": family,
        "artifact_sha256": hashes,
        "published_generation": str(relative) if artifacts != root else "original",
        "physics_fingerprint": fingerprint,
        "task": "rate",
        "source_status": summary.get("status"),
        "rate_cm3_s": _number(summary.get("rate_cm3_s")),
        "rate_estimate_cm3_s": _number(summary.get("rate_estimate_cm3_s")),
        "uncertainty": {
            "raw_rate_standard_error_cm3_s": _number(summary.get("rate_standard_error_cm3_s"), nonnegative=True),
            "raw_rate_percentile_95_ci_cm3_s": summary.get("rate_percentile_95_ci_cm3_s"),
            "recrossing_only_conditional_standard_error_cm3_s": _number(summary.get("conditional_rate_standard_error_cm3_s"), nonnegative=True),
            "producer_metadata": uncertainty,
            "covariance_pmf_recrossing": None,
            "validated_total_rate_error": None,
        },
        "provenance": {"phase": phase, "epoch_id": production.get("epoch_id"),
                       "design_id": production.get("design_id"), "record": production,
                       "validated_for_acceptance": False},
        **fields,
        "missing": missing,
        "warnings": summary.get("warnings", []),
        "metadata": {"resolved_input": config, "manifest": manifest, "pmf": pmf,
                     "umbrella": umbrella},
    }


def read_multiple_rate_diagnostics(directories):
    """Read independent candidate runs, rejecting duplicate/overlapping runs.

    This is a provenance guard, not certification that distinct runs are
    independent. Shared source PMFs and reused parent states still require the
    controller's joint-covariance/independence checks.
    """
    results, sources, families = [], set(), set()
    for directory in directories:
        result = read_rate_diagnostics(directory)
        ids = set(result["source_ids"])
        family = result["source_family_id"]
        if sources.intersection(ids) or (family is not None and family in families):
            raise ValueError("Duplicate or overlapping published rate source")
        sources.update(ids)
        if family is not None:
            families.add(family)
        results.append(result)
    return results
