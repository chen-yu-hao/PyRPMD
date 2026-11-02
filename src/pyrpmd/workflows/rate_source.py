"""Validated saved FES sources for gas-phase RPMD rate calculations.

A time-dependent bias estimate is never substituted for a rate-flux PMF.
Non-umbrella sources require an explicit opt-in and reweighting provenance.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from ..analysis.fes_result import FESResult
from ..cv import RPMDrateCoordinate, make_cvs
from ..io import file_sha256, fingerprint, jsonable
from ..input import fes_method, rate_pmf_options
from ..propagation import scientific_config
from ..system import System


_SOURCE_FILES = {
    "data_file": "fes.npz",
    "config_file": "input/resolved.json",
    "system_file": "input/system.json",
    "manifest_file": "manifest.json",
}


def source_paths(raw, base=None):
    """Resolve exact source files so the destination manifest can hash them."""
    rate_pmf_options({"rate": {"pmf_source": raw}})
    if isinstance(raw, str):
        raw = {"directory": raw}
    root = Path(base or ".")
    directory = (root / raw["directory"]).resolve()
    result = {
        "directory": str(directory),
        "allow_non_umbrella": raw.get("allow_non_umbrella", False),
    }
    for name, relative in _SOURCE_FILES.items():
        path = directory / relative
        if name in raw and Path(raw[name]).resolve() != path:
            raise ValueError(f"rate.pmf_source.{name} must belong to directory")
        if not path.is_file():
            raise ValueError(f"Missing rate PMF source file: {path}")
        result[name] = str(path)
    for name, choices in (
        ("summary_file", ("summary.json", "results/summary.json")),
        ("metadata_file", ("fes.json",)),
    ):
        found = next((directory / choice for choice in choices if (directory / choice).is_file()), None)
        if name in raw and (found is None or Path(raw[name]).resolve() != found):
            raise ValueError(f"rate.pmf_source.{name} must belong to directory")
        if found is not None:
            result[name] = str(found)
    return result


def _coordinate_identity(cv):
    return jsonable({
        "estimator": cv.estimator,
        "fragments": cv.com.fragments,
        "masses": cv.masses,
        "reference": cv.reference,
        "forming": cv.forming,
        "breaking": cv.breaking,
        "rinf": cv.rinf,
        "permutations": cv.permutations,
    })


def read_source(config, system):
    """Validate source provenance, ensemble, units and reaction coordinate."""
    from .rate import rate_coordinate

    paths = source_paths(rate_pmf_options(config)["source"])
    result = FESResult.read(paths["directory"])
    if result.method != "umbrella":
        if (not paths["allow_non_umbrella"] or result.metadata.get("reweighted") is not True
                or result.metadata.get("estimator") in ("bias_estimate", "kde")):
            raise ValueError(
                "MetaD/OPES PMFs must be reweighted before use by rate; "
                "supply a reweighted rate-flux FESResult and explicitly set "
                "rate.pmf_source.allow_non_umbrella=true. Raw bias estimates are not accepted."
            )
    if result.measure != "rate_flux":
        raise ValueError("rate.pmf_source requires a rate_flux PMF, including the flux metric")
    if result.free_energy.ndim != 1 or result.cv_units != ("dimensionless",) or result.energy_unit != "hartree":
        raise ValueError("rate.pmf_source requires one dimensionless RPMDrate coordinate and hartree energies")
    source = json.loads(Path(paths["config_file"]).read_text())
    source_system = System.from_dict(json.loads(Path(paths["system_file"]).read_text()))
    manifest = json.loads(Path(paths["manifest_file"]).read_text())
    schema = manifest.get("schema_version", 1)
    if (type(schema) is not int or schema not in (1, 2)
            or ("schema" in manifest and manifest["schema"] != "pyrpmd.run")
            or (schema == 2 and manifest.get("schema") != "pyrpmd.run")):
        raise ValueError("Unsupported rate PMF source manifest schema")
    if manifest.get("physics_fingerprint") not in (
        fingerprint(source, source_system), fingerprint(scientific_config(source), source_system)
    ):
        raise ValueError("rate PMF source configuration does not match its manifest")
    for filename, digest in manifest.get("files", {}).items():
        if not Path(filename).is_file() or file_sha256(filename) != digest:
            raise ValueError(f"rate PMF source input/model file changed or is missing: {filename}")
    if (source.get("task") not in ("fes", "metadynamics", "opes")
            or fes_method(source) != result.method):
        raise ValueError("rate PMF source method disagrees with its saved input")
    source_measure = source.get("analysis", {}).get("measure", "cv_density")
    if source_measure != result.measure:
        # The built-in frozen estimator removes only the historical bias.
        # Its reweighted flag cannot certify a change in the target measure.
        if result.metadata.get("estimator") == "reweighted_histogram":
            raise ValueError(
                "Frozen-bias reweighting preserves its sampling measure; "
                "a rate_flux source must have sampled the flux metric"
            )
        if result.metadata.get("reweighted") is not True:
            raise ValueError("rate PMF source measure differs from its sampling ensemble without reweighting provenance")
    try:
        beta_matches = np.isclose(source.get("beta", np.nan), config["beta"], rtol=1e-13, atol=0)
        if "beta" in result.metadata:
            beta_matches = beta_matches and np.isclose(result.metadata["beta"], config["beta"], rtol=1e-13, atol=0)
    except (TypeError, ValueError):
        beta_matches = False
    if source.get("beads") != config["beads"] or not beta_matches:
        raise ValueError("rate PMF source temperature or bead count differs")
    if jsonable(source.get("potential")) != jsonable(config.get("potential")):
        raise ValueError("rate PMF source PES specification differs")
    if jsonable(source_system.to_dict()) != jsonable(system.to_dict()):
        raise ValueError("rate PMF source system or reference geometry differs")
    names = source.get(result.method, {}).get("cvs")
    if isinstance(names, str):
        names = [names]
    cv = make_cvs(source.get("cvs", []), source_system, names)
    if (cv.dimension != 1 or len(cv.cvs) != 1
            or not isinstance(cv.cvs[0], RPMDrateCoordinate)
            or _coordinate_identity(cv.cvs[0]) != _coordinate_identity(rate_coordinate(config, system))):
        raise ValueError("rate PMF source must use the same centroid RPMDrate reaction coordinate")
    # Additional static walls change the constrained ensemble; the current rate
    # parent sampler does not include such walls and must not silently omit them.
    if source.get("metadynamics", {}).get("walls"):
        raise ValueError("rate PMF source cannot include static MetaD walls")
    return result, paths


def source_estimate(config, system):
    """Authenticate the source before using the common supported FES estimator."""
    from .rate_curve import estimate_fes_curve

    result, paths = read_source(config, system)
    warnings = [
        "Rate PMF loaded from a saved FESResult; no new biased trajectories were sampled.",
        "The saved pointwise uncertainty does not supply paired endpoint covariance; uncertainty for the re-referenced rate PMF, its barrier and the total rate is unavailable.",
    ]
    if result.method != "umbrella":
        warnings.append("MetaD/OPES PMF requires reweighting; this source explicitly declares a reweighted rate-flux estimate.")
    estimate = estimate_fes_curve(
        result, rate_pmf_options(config),
        uncertainty_reason="source_missing_paired_endpoint_covariance",
        unsupported_status="unsupported_source_pmf", warnings=warnings,
        strict_search=True,
    )
    estimate.update(
        source=paths, source_method=result.method,
        source_reweighted=result.metadata.get("reweighted", False),
        source_sha256={key: file_sha256(path) for key, path in paths.items() if key.endswith("_file")},
    )
    return estimate
