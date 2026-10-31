"""Provenance-preserving, offline rate postprocessing for saved PyRPMD runs.

No potential is loaded and no dynamics is run. The one-dimensional adapter
recomputes the existing gas-rate convention. Two-dimensional specifications
must distinguish coordinate density from a flux-weighted sampling measure.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..constants import AU_RATE_TO_CM3_S
from ..checkpoint import load_checkpoint
from ..io import file_sha256, write_json


def _read_json(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def _load_npz(path):
    with load_checkpoint(path) as source:
        return dict(source.items())


def _scalar(value, name):
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    return array.item()


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be positive and finite")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _root(directory, marker):
    directory = Path(directory).resolve()
    if (directory / marker).is_file():
        return directory
    if (directory / "run" / marker).is_file():
        return directory / "run"
    raise ValueError(f"No {marker} in {directory} (or its run subdirectory)")


def _metadata(root):
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        raise ValueError("Saved summary.json is required; NPZ alone has insufficient measure metadata")
    summary = _read_json(summary_path)
    config_path = root / "input" / "resolved.json"
    config = _read_json(config_path) if config_path.is_file() else {}
    files = [summary_path]
    if config_path.is_file():
        files.append(config_path)
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    if manifest_path.is_file():
        files.append(manifest_path)
    return summary, config, manifest, files


def _beta(config, spec):
    saved, supplied = config.get("beta"), spec.get("beta")
    if saved is None and supplied is None:
        raise ValueError("beta in atomic units is required when saved input/resolved.json is unavailable")
    beta = _positive(saved if saved is not None else supplied, "beta")
    if supplied is not None and not np.isclose(beta, _positive(supplied, "beta"), rtol=1e-12, atol=0):
        raise ValueError("Specification beta disagrees with the saved run temperature")
    return beta


def _legacy_rate(root, *, bootstrap, seed):
    from .rate_postprocess import channel_rates

    summary, config, manifest, files = _metadata(root)
    if summary.get("task") != "rate" or config.get("task", "rate") != "rate":
        raise ValueError("The automatic adapter accepts a gas rate run, not surface/site rates or network entries")
    beta = _beta(config, {})
    pmf_path, arrays_path = root / "rate_pmf.json", root / "rate_pmf.npz"
    pmf, arrays = _read_json(pmf_path), _load_npz(arrays_path)
    files.extend([pmf_path, arrays_path])
    if pmf.get("sampling_measure") != "rate_flux" or _scalar(arrays.get("sampling_measure"), "sampling_measure") != "rate_flux":
        raise ValueError("A gas rate PMF must use the rate_flux sampling measure")
    if pmf.get("method") != _scalar(arrays.get("method"), "method"):
        raise ValueError("rate_pmf.json and rate_pmf.npz disagree on the PMF method")
    ready = bool(pmf.get("ready", False))
    if ready != bool(_scalar(arrays.get("ready"), "ready")):
        raise ValueError("Saved PMF readiness is inconsistent")
    result = {
        "method": "saved_rate_1d", "measure": "rate_flux", "beta_au": beta,
        "normalization_kind": "asymptotic_bimolecular_flux",
        "source_status": summary.get("status"), "source_converged": summary.get("converged", False),
        "physics_fingerprint": manifest.get("physics_fingerprint"),
        "rate_cm3_s": None, "rate_estimate_cm3_s": None, "converged": False,
        "warnings": list(summary.get("warnings", [])),
    }
    if not ready:
        result.update(status="source_pmf_not_ready", static=None, channels=None)
        return result, files
    xi = float(pmf["xi_star"])
    if not np.isclose(xi, _scalar(arrays["xi_star"], "xi_star"), rtol=1e-12, atol=1e-14):
        raise ValueError("Saved PMF dividing surfaces disagree")
    source_pmf = summary.get("pmf", {})
    for key in ("method", "xi_star", "free_energy_barrier_hartree"):
        if source_pmf.get(key) != pmf.get(key):
            raise ValueError(f"Source summary and PMF disagree on {key}; do not mix runs")
    rinf = _positive(pmf["rinf_bohr"], "rinf_bohr")
    mass = _positive(pmf["reduced_mass_au"], "reduced_mass_au")
    prefactor = 4 * np.pi * rinf**2 / np.sqrt(2 * np.pi * beta * mass)
    if not np.isclose(prefactor, pmf["reactant_prefactor_au"], rtol=1e-10, atol=0):
        raise ValueError("Saved entrance prefactor disagrees with temperature, Rinf or reduced mass")
    barrier = float(pmf["free_energy_barrier_hartree"])
    if not np.isfinite(barrier):
        raise ValueError("The PMF barrier must be finite")
    from .fes import _rate_support, _rate_value

    grid = np.asarray(arrays["xi"], float)
    free = np.asarray(arrays["free_energy"], float)
    mask = np.asarray(arrays["mask"])
    if (grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all()
            or np.any(np.diff(grid) <= 0) or free.shape != grid.shape
            or mask.shape != grid.shape or mask.dtype.kind != "b"):
        raise ValueError("Invalid saved PMF grid, free-energy curve or mask")
    reference = float(pmf.get("reference", 0.0))
    if "reference" in arrays and not np.isclose(reference, _scalar(arrays["reference"], "reference"), rtol=0, atol=1e-14):
        raise ValueError("Saved PMF reference coordinates disagree")
    rb, sb, reason = _rate_support(grid, mask | ~np.isfinite(free), reference, xi)
    if reason:
        raise ValueError(f"Saved ready PMF lacks endpoint/interval support: {reason}")
    curve_barrier = _rate_value(free, sb) - _rate_value(free, rb)
    if not np.isclose(curve_barrier, barrier, rtol=1e-10, atol=1e-13):
        raise ValueError("Saved PMF curve barrier disagrees with rate_pmf.json")
    electronic = float(pmf.get("electronic_weight", 1.0))
    symmetry = _positive(pmf.get("symmetry_factor", 1.0), "symmetry_factor")
    if not np.isfinite(electronic) or electronic < 0:
        raise ValueError("electronic_weight must be finite and nonnegative")
    qtst = float(prefactor * np.exp(-beta * barrier) * electronic * symmetry)
    # The producer permits zero electronic weight and Boltzmann underflow.
    # Preserve those finite zeros; channel_rates reports zero_static_rate and
    # does not promote them to a validated kinetic rate or branching fraction.
    if not np.isfinite(qtst) or qtst < 0:
        raise ValueError("QTST must be finite and nonnegative; inspect the barrier and statistical weights")
    if not np.isclose(qtst, pmf["k_qtst_au"], rtol=1e-10, atol=0):
        raise ValueError("Recomputed QTST disagrees with saved PMF normalization")
    static = {
        "k_tst_au": qtst, "k_tst_cm3_s": qtst * AU_RATE_TO_CM3_S,
        "reference_rate_au": prefactor, "free_energy_barrier_hartree": barrier,
        "xi_star": xi, "electronic_weight": electronic, "symmetry_factor": symmetry,
        "underflow": bool(qtst == 0 and electronic > 0),
        "zero_rate_reason": (
            "zero_electronic_weight" if electronic == 0 else
            "floating_point_underflow" if qtst == 0 else None
        ),
        "measure": "rate_flux", "converged": False,
    }
    recrossing_path = root / "recrossing.npz"
    if not recrossing_path.is_file():
        result.update(status="static_only", static=static, channels=None)
        return result, files
    saved = _load_npz(recrossing_path)
    files.append(recrossing_path)
    if "dt_au" in config:
        dt = _positive(config["dt_au"], "dt_au")
        if not np.allclose(saved["time_au"], np.arange(1, len(saved["time_au"]) + 1) * dt, rtol=1e-12, atol=1e-12):
            raise ValueError("Saved recrossing time grid disagrees with the run timestep")
    numerator = np.asarray(saved["parent_numerator"], float)
    denominator = np.asarray(saved["parent_denominator"], float)
    if numerator.ndim != 2 or denominator.sum() <= 0:
        raise ValueError("Invalid saved single-channel parent flux arrays")
    curve = numerator.sum(axis=0) / denominator.sum()
    if curve.shape != saved["kappa"].shape or not np.allclose(curve, saved["kappa"], rtol=1e-10, atol=1e-12):
        raise ValueError("Saved kappa curve disagrees with signed parent flux sums")
    recrossing = config.get("recrossing", config.get("rate", {}).get("recrossing", {}))
    channels = channel_rates(
        qtst, saved["time_au"], numerator[:, :, None], denominator,
        saved["parent_ids"], ["reaction"],
        plateau_fraction=recrossing.get("plateau_fraction", .25),
        plateau_tolerance=recrossing.get("plateau_tolerance", .1),
        bootstrap=bootstrap, seed=seed,
    )
    tail = min(len(curve), max(2, int(np.ceil(len(curve) * recrossing.get("plateau_fraction", .25)))))
    estimate = float(qtst * curve[-tail:].mean() * AU_RATE_TO_CM3_S)
    source_estimate = summary.get("rate_estimate_cm3_s")
    if source_estimate is not None and not np.isclose(estimate, source_estimate, rtol=1e-10, atol=0):
        raise ValueError("Recomputed rate disagrees with saved summary; do not mix PMF and recrossing outputs")
    # An old summary contains energy/trajectory checks absent from the NPZ.
    # Their failure cannot be erased by repeating a statistical calculation.
    energy_error = summary.get("maximum_endpoint_energy_error_kbt")
    source_publishable = (
        summary.get("rate_cm3_s") is not None
        and summary.get("status") == "plateau_estimate"
        and summary.get("plateau_detected") is True
        and energy_error is not None and np.isfinite(energy_error) and 0 <= energy_error <= .1
        and channels.get("status") == "conditional_plateau_estimate"
    )
    result.update(
        status="postprocessed_plateau_estimate" if source_publishable else "source_validation_incomplete",
        static=static, channels=channels, rate_estimate_cm3_s=estimate,
        rate_cm3_s=estimate if source_publishable else None,
        source_energy_error_kbt=summary.get("maximum_endpoint_energy_error_kbt"),
        source_plateau_detected=summary.get("plateau_detected", False),
        uncertainty_scope="channel bootstrap conditional on the saved static PMF; source joint uncertainty retained separately",
        source_rate_standard_error_cm3_s=summary.get("rate_standard_error_cm3_s"),
        source_rate_uncertainty=summary.get("rate_uncertainty"),
    )
    return result, files


_SPEC_FIELDS = {
    "schema", "schema_version", "method", "measure", "beta", "curve", "normal_speed", "inverse_mass",
    "reference_curve", "reference_normal_speed", "reference_inverse_mass", "reference_rate_au",
    "reference_definition", "reactant_mask", "reactant_mask_key", "reactant_volume_au",
    "reactant_definition", "surface", "reference", "flux_axis", "quadrature_order",
    "channel_flux", "surface_identity", "plateau_fraction", "plateau_tolerance",
}


def _import_channels(root, spec, static, *, bootstrap, seed):
    from .rate_postprocess import channel_rates

    path = Path(spec["channel_flux"])
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    arrays = _load_npz(path)
    expected = {
        "schema": "pyrpmd.channel_flux", "schema_version": 1,
        "estimator": "signed_flux_side", "time_unit": "atomic",
    }
    for key, value in expected.items():
        if key not in arrays or _scalar(arrays[key], key) != value:
            raise ValueError(f"channel_flux requires {key}={value!r}")
    identity = spec.get("surface_identity")
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError("Imported channel flux requires an explicit surface_identity in the specification")
    flux_identity = _scalar(arrays.get("surface_identity"), "surface_identity")
    if flux_identity != identity:
        raise ValueError("Channel flux belongs to a different dividing surface / physics identity")
    channels = channel_rates(
        static["k_tst_au"], arrays["time_au"], arrays["parent_numerator"],
        arrays["parent_denominator"], arrays["parent_ids"], arrays["channel_names"].tolist(),
        plateau_fraction=spec.get("plateau_fraction", .25),
        plateau_tolerance=spec.get("plateau_tolerance", .1),
        bootstrap=bootstrap, seed=seed, source_identity=identity, flux_identity=flux_identity,
    )
    return channels, path


def _fes_rate(root, spec, *, bootstrap, seed):
    from .rate_postprocess import surface_rate_2d, flux_slice_rate_2d

    unknown = set(spec) - _SPEC_FIELDS
    if unknown:
        raise ValueError(f"Unknown rate postprocessing keys: {sorted(unknown)}")
    if spec.get("schema", "pyrpmd.rate_postprocess") != "pyrpmd.rate_postprocess" or spec.get("schema_version", 1) != 1:
        raise ValueError("Unsupported rate postprocessing specification schema")
    summary, config, manifest, files = _metadata(root)
    if summary.get("dimension") != 2 or summary.get("energy_unit") != "hartree":
        raise ValueError("A two-dimensional FES with explicit hartree energy metadata is required")
    units = summary.get("units")
    if not isinstance(units, list) or len(units) != 2 or not all(isinstance(u, str) and u for u in units):
        raise ValueError("Two explicit CV units are required")
    if summary.get("overlap_connected") is not True:
        raise ValueError("A disconnected or unverified global FES cannot define a rate")
    beta = _beta(config, spec)
    path = root / "fes.npz"
    data = _load_npz(path)
    files.append(path)
    if "energy_unit" in data and _scalar(data["energy_unit"], "energy_unit") != summary["energy_unit"]:
        raise ValueError("NPZ and summary energy units disagree")
    if "cv_units" in data and data["cv_units"].tolist() != units:
        raise ValueError("NPZ and summary CV units disagree")
    for origin, value in (("summary", summary.get("beta")), ("NPZ", data.get("beta"))):
        if value is not None and not np.isclose(_positive(_scalar(value, "beta"), "beta"), beta, rtol=1e-12, atol=0):
            raise ValueError(f"{origin} beta disagrees with the saved run temperature")
    measure = summary.get("measure")
    if "measure" in spec and spec["measure"] != measure:
        raise ValueError("Requested measure disagrees with the saved FES measure")
    for measure_key in ("sampling_measure", "measure"):
        if measure_key in data and _scalar(data[measure_key], measure_key) != measure:
            raise ValueError("NPZ and summary sampling measures disagree")
    args = (data["free_energy"], [data["edges_0"], data["edges_1"]])
    common = {"beta": beta, "mask": data["mask"], "measure": measure}
    method = spec.get("method")
    if method == "flux_slice_2d":
        if any(key in spec for key in (
            "curve", "normal_speed", "inverse_mass", "reactant_volume_au", "reference_curve",
            "reference_normal_speed", "reference_inverse_mass", "reactant_mask", "reactant_mask_key",
            "reactant_definition", "quadrature_order",
        )):
            raise ValueError("rate_flux slices already contain the flux metric; density/curve options are incompatible")
        axis = spec.get("flux_axis", 0)
        saved_axis = summary.get("flux_axis")
        if saved_axis is None or saved_axis != axis:
            raise ValueError("An explicit matching flux_axis in the saved FES metadata is required")
        if "flux_axis" in data and _scalar(data["flux_axis"], "flux_axis") != saved_axis:
            raise ValueError("NPZ and summary flux_axis disagree")
        if not isinstance(spec.get("reference_definition"), str) or not spec["reference_definition"].strip():
            raise ValueError("Explain the physical reference normalization in reference_definition")
        static = flux_slice_rate_2d(
            *args, **common, surface=spec["surface"], reference=spec["reference"],
            reference_rate_au=spec["reference_rate_au"], flux_axis=axis,
        )
    elif method == "surface_2d":
        if any(key in spec for key in ("surface", "reference", "flux_axis")):
            raise ValueError("surface_2d uses explicit curves, not rate_flux slice options")
        reactant_mask = spec.get("reactant_mask")
        if "reactant_mask_key" in spec:
            if reactant_mask is not None:
                raise ValueError("Choose reactant_mask or reactant_mask_key, not both")
            reactant_mask = data[spec["reactant_mask_key"]]
        is_volume = spec.get("reactant_volume_au") is not None
        explanation = "reactant_definition" if is_volume else "reference_definition"
        if not isinstance(spec.get(explanation), str) or not spec[explanation].strip():
            raise ValueError(f"Explain the physical normalization in {explanation}")
        allowed = (
            "curve", "normal_speed", "inverse_mass", "reference_curve", "reference_normal_speed",
            "reference_inverse_mass", "reference_rate_au", "reactant_volume_au", "quadrature_order",
        )
        options = {key: spec[key] for key in allowed if key in spec}
        static = surface_rate_2d(*args, **common, reactant_mask=reactant_mask, **options)
    else:
        raise ValueError("method must be surface_2d or flux_slice_2d")
    result = {
        "method": method, "measure": measure, "beta_au": beta, "cv_units": units,
        "static": static, "channels": None, "status": "static_tst_estimate",
        "rate_cm3_s": None, "rate_estimate_cm3_s": None, "converged": False,
        "source_status": summary.get("status"), "source_converged": summary.get("converged", False),
        "physics_fingerprint": manifest.get("physics_fingerprint"),
        "specification": spec,
        "warnings": list(summary.get("warnings", [])) + [
            "Finite histogram support does not establish transverse-domain, bead, timestep or sampling convergence.",
            "The static factor is a TST estimate; product rates require matched signed channel fluxes.",
        ],
    }
    if spec.get("channel_flux"):
        result["channels"], channel_path = _import_channels(root, spec, static, bootstrap=bootstrap, seed=seed)
        files.append(channel_path)
        result["status"] = "channel_estimates_require_physical_validation"
        result["rate_estimate_cm3_s"] = result["channels"]["total_rate_estimate_cm3_s"]
        # An imported statistical record alone supplies no energy-conservation,
        # product-commitment or quantum discretization convergence evidence.
        result["warnings"].append("Imported channel rates remain estimates; trajectory quality and final product commitment must be checked separately.")
    return result, files


def postprocess_rate(directory, *, spec=None, output=None, bootstrap=200, seed=0):
    """Recompute saved rates, or integrate a specified two-dimensional FES.

    With no spec, directory is an existing gas-rate run. A JSON path or mapping
    selects a 2D method. All numerical inputs use the run's atomic/CV units.
    The output is a new JSON and never replaces the saved scientific inputs.
    """
    if isinstance(bootstrap, (bool, np.bool_)) or not isinstance(bootstrap, (int, np.integer)) or bootstrap < 0 or bootstrap == 1:
        raise ValueError("bootstrap must be zero or an integer of at least two")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    specification_path = None
    if isinstance(spec, (str, Path)):
        specification_path = Path(spec).resolve()
        spec = _read_json(specification_path)
    elif spec is not None and not isinstance(spec, dict):
        raise ValueError("spec must be a JSON path or a mapping")
    root = _root(directory, "rate_pmf.json" if spec is None else "fes.npz")
    if spec is None:
        result, files = _legacy_rate(root, bootstrap=int(bootstrap), seed=int(seed))
    else:
        result, files = _fes_rate(root, dict(spec), bootstrap=int(bootstrap), seed=int(seed))
    if specification_path is not None:
        files.append(specification_path)
    target = Path(output).resolve() if output else root / "rate_postprocess.json"
    protected = {path.resolve() for path in files}
    protected.update((root / name).resolve() for name in (
        "summary.json", "rate_pmf.json", "rate_pmf.npz", "recrossing.npz", "fes.npz",
        "fes.h5", "results/summary.json", "manifest.json", "input/resolved.json", "input/system.json",
    ))
    if target in protected:
        raise ValueError("Postprocessing output must not overwrite a source scientific file")
    if target.exists():
        try:
            previous = _read_json(target)
        except (ValueError, OSError, UnicodeError) as exc:
            raise ValueError("Postprocessing output must not overwrite an existing scientific or unrelated file") from exc
        if previous.get("schema") != "pyrpmd.rate_postprocess":
            raise ValueError("Postprocessing output must not overwrite an existing scientific or unrelated file")
    result.update(
        schema="pyrpmd.rate_postprocess", schema_version=1,
        energy_unit="hartree", time_unit="atomic", rate_unit="cm^3 molecule^-1 s^-1",
        source_directory=str(root), output=str(target),
        source_files=[{"path": str(path), "sha256": file_sha256(path)} for path in dict.fromkeys(files)],
        analysis={"bootstrap": int(bootstrap), "seed": int(seed)},
    )
    result = _json_safe(result)
    write_json(target, result)
    return result
