"""Optional, root-only postprocessing of a native gas-phase rate run.

Scientific output is written only under RUN/subspace. The explicit stage also
updates advisory progress/log files at the run root. Source rates and their
publication gates stay unchanged; a failed correction cannot overwrite a rate.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import warnings

import numpy as np

from ..input_schema import InputError
from ..input_subspace import rate_subspace_options
from ..io import fingerprint, referenced_files, write_json
from ..propagation import scientific_config


def validate_saved_rate(config, system, directory):
    """Read-only identity check, allowing changes to postprocessing, not physics.

    The .inp itself may change to add/tune this stage: its resolved scientific
    identity is checked instead of requiring the old input's byte hash. Every
    other recorded model/source file must retain its manifest hash.
    """
    from ..runner import _read_identity, _validate_resume_identity

    root = Path(directory).resolve()
    manifest_path = root / "manifest.json"
    resolved_path = root / "input" / "resolved.json"
    if not manifest_path.is_file() or not resolved_path.is_file():
        raise InputError("The subspace stage needs an existing rate run with manifest.json and input/resolved.json")
    if (root / "results" / "extensions.json").exists():
        raise InputError("Subspace postprocessing of rate-extension generations is not supported by this stage")
    stored = json.loads(resolved_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if stored.get("task") != "rate":
        raise InputError("The subspace stage requires a saved gas-phase rate run")
    stored_system = _read_identity(stored, root)
    identity_config = stored if manifest.get("schema_version", 1) == 1 else scientific_config(stored)
    expected = fingerprint(identity_config, stored_system)
    if manifest.get("physics_fingerprint") != expected:
        raise InputError("Saved rate inputs do not match the manifest physics fingerprint")
    # Do not weaken the generic restart contract for unrelated workflows.
    checked = deepcopy(manifest)
    source = stored.get("_input_path")
    mutable_input = (source if source and Path(source).suffix.lower() in (".inp", ".toml") else None)
    checked["files"] = {p: sha for p, sha in manifest.get("files", {}).items()
                        if p != mutable_input}
    _validate_resume_identity(config, system, root, checked, referenced_files(config),
                              fingerprint(scientific_config(config), system))
    return root


def _source_identity(root, options):
    from ..analysis.subspace_pipeline import pipeline_identity

    return pipeline_identity(root, options)


def _check_centroid_path(root, config):
    from ..analysis.centroid_path import mean_path_from_rate_run

    mean = mean_path_from_rate_run(root)
    if (mean.nbeads != config["beads"]
            or not np.isclose(mean.beta, config["beta"], rtol=1e-9, atol=0)):
        raise InputError("centroid_path.npz temperature/bead count does not match the rate run")
    if not np.all(mean.complete) or np.any(mean.count <= 0):
        raise InputError("centroid_path.npz has incomplete window moments; rerun sampling with "
                         "output.centroid_path=true from the start, not only after a checkpoint")
    if np.count_nonzero(mean.locked()) < 2:
        raise InputError("Subspace correction needs at least two orientation-locked umbrella windows")


def postprocess_rate_subspace(config, directory, *, strict=False):
    """Run the shared analysis pipeline on the calling (root) rank only.

    Full rate runs record optional-stage failures and retain their original
    result. Explicit stage-only calls re-raise after recording failure, so CLI
    scripts receive a nonzero exit status. Numerical solvers are not MPI jobs.
    """
    from ..analysis.subspace_pipeline import run_subspace_pipeline

    options = rate_subspace_options(config)
    if options is None or not options["enabled"]:
        return None
    root = Path(directory).resolve()
    target = root / "subspace"
    status_path = target / "status.json"
    status = {"schema": "pyrpmd.rate_subspace", "schema_version": 1, "mode": options["mode"],
              "directory": str(target), "physical_converged": False}
    try:
        if (root / "results" / "extensions.json").exists():
            raise InputError("Subspace postprocessing of rate-extension generations is not supported by this stage")
        _check_centroid_path(root, config)
        identity = _source_identity(root, options)
        specification = {
            "source": {"rate_run": str(root)},
            "subspace": {k: options[k] for k in ("dimension", "threshold", "environment", "curvature",
                         "zero_point", "dvr_points", "dvr_half_width", "dvr_spacing",
                         "hessian_displacement") if k in options},
            "validity": options.get("validity", {}),
            "path": options["path"], "quantum": options["quantum"], "rpmd": options["rpmd"],
            "output": {"directory": str(target), "reuse": options["reuse"]},
        }
        write_json(target / "request.json", {"options": options, "reuse_key": identity})
        write_json(status_path, {**status, "status": "running"})
        report = run_subspace_pipeline(specification, apply=options["mode"] == "apply", reuse_key=identity)
        # "invalid_model": the stage ran, but no subspace model passed its
        # validity gates, so there is no factor and no corrected rate.
        # "solver_capacity_exceeded": a model's solves were refused before
        # starting (route or memory); "solver_plans" says what they need.
        # "failed": every solve that was started raised ("solve_failures").
        status.update(status=report.get("status", "complete"), reuse_key=identity,
                      validity=report.get("validity"),
                      dimensions_computed=report["dimensions_computed"],
                      selected_dimension=report["applied_dimension"],
                      correction_factors={d: c["factor"] for d, c in report["corrections"].items()},
                      solver_plans=report.get("solver_plans", {}),
                      solve_failures=report.get("solve_failures", {}),
                      warnings=report.get("warnings", []),
                      report=str(target / "subspace_pipeline.json"))
        corrected = report.get("corrected_rate")
        if corrected is not None:
            status.update(corrected_rate_file=str(target / "rate_quantum_corrected.json"),
                          corrected_rate_cm3_s=corrected["corrected_rate_cm3_s"],
                          corrected_rate_estimate_cm3_s=corrected["corrected_rate_estimate_cm3_s"],
                          corrected_rate_standard_error_cm3_s=corrected["corrected_rate_standard_error_cm3_s"])
    except Exception as error:
        status.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        if isinstance(getattr(error, "diagnostic", None), dict):
            status["error"]["diagnostic"] = error.diagnostic
        write_json(status_path, status)
        if strict:
            raise
        warnings.warn(f"Rate subspace postprocessing failed; the uncorrected RPMD result is intact. "
                      f"See {status_path}: {error}", RuntimeWarning, stacklevel=2)
        return status
    write_json(status_path, status)
    if status["status"] == "failed":
        failures = "; ".join(f"{d}D: {f['type']}: {f['message']}" for d, f in status["solve_failures"].items())
        if strict:
            raise RuntimeError(f"subspace stage failed ({failures}); see {status_path}")
        warnings.warn(f"Rate subspace postprocessing failed; the uncorrected RPMD result is intact. "
                      f"See {status_path}: {failures}", RuntimeWarning, stacklevel=2)
    return status


def finish_rate(config, directory, result):
    """Attach stage status to the return value, without rewriting source summary.json."""
    options = rate_subspace_options(config)
    if options is None or not options["enabled"]:
        return result
    stage = postprocess_rate_subspace(config, directory)
    return {**result, "subspace_correction": stage}


def run_subspace_stage(config, system, directory):
    """Native runner's read-only-source route: no preparation or MD dispatch."""
    options = rate_subspace_options(config)
    if options is None or not options["enabled"]:
        raise InputError("--stage subspace requires rate.subspace_correction.enabled=true")
    root = validate_saved_rate(config, system, directory)
    from ..progress import RunProgress

    with RunProgress(root, {"task": "subspace", "resume": True}) as progress:
        status = postprocess_rate_subspace(config, root, strict=True)
        result = {"task": "rate", "stage": "subspace", "output": str(root),
                  "subspace_correction": status, "physical_converged": False}
        progress.result = result
        return result
