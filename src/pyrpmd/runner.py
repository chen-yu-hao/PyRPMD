"""Public run/prepare/resume entry points shared by Python and the CLI."""

from __future__ import annotations
import copy
import importlib.metadata
import json
import platform
from pathlib import Path
import shutil
import sys
from . import __version__
from .input import JobSpec, InputError
from .io import write_json, referenced_files, fingerprint, jsonable
from .system import System


def _spec(value):
    if isinstance(value, JobSpec):
        return value
    if isinstance(value, (str, Path)):
        return JobSpec.from_file(value)
    return JobSpec.from_mapping(value)


def _initialize_output(config, system, output, resume=False):
    path = Path(output)
    manifest_path = path / "manifest.json"
    hashes = referenced_files(config)
    digest = fingerprint(config, system)
    if resume:
        if not manifest_path.is_file():
            raise InputError(f"No run manifest in {path}")
        old = json.loads(manifest_path.read_text())
        if old["physics_fingerprint"] != digest:
            raise InputError("Cannot resume with changed scientific settings or system")
        for file, sha in old.get("files", {}).items():
            if hashes.get(file) != sha:
                raise InputError(f"Input/model file changed or is missing: {file}")
        return
    if path.exists() and any(path.iterdir()):
        raise InputError(
            f"Output directory is not empty: {path}; choose another directory or resume"
        )
    path.mkdir(parents=True, exist_ok=True)
    for child in ("input", "results", "checkpoints", "samples", "logs"):
        (path / child).mkdir(exist_ok=True)
    versions = {}
    for name in ("numpy", "scipy", "h5py", "mpi4py", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    from .dynamics import native_capabilities

    manifest = {
        "software": "PyRPMD",
        "version": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": versions,
        "files": hashes,
        "physics_fingerprint": digest,
        "native": native_capabilities(),
        "parallel": config.get("parallel", {}),
        "preset": config.get("preset", "pilot"),
        "hamiltonian_convention": "sum_beads_beta_over_P",
        "units": {
            "length": "bohr",
            "momentum": "atomic",
            "energy": "hartree",
            "time": "atomic",
        },
    }
    write_json(manifest_path, manifest)
    write_json(path / "input" / "resolved.json", config)
    write_json(path / "input" / "system.json", system.to_dict())
    if config.get("_input_path"):
        source = Path(config["_input_path"])
        shutil.copy2(source, path / "input" / ("original" + source.suffix))


def run(job, *, output=None, resume=False):
    """Execute a JobSpec, mapping, .inp or .py file and return the root summary."""
    spec = _spec(job)
    if output is not None:
        spec = spec.with_overrides(output=output)
    config, system = spec.resolve()
    return run_resolved(config, system, output=output, resume=resume)


def run_resolved(config, system, *, output=None, resume=False):
    config = copy.deepcopy(config)
    target = str(Path(output or config["output"]["directory"]).resolve())
    config["output"]["directory"] = target
    config["resume"] = bool(resume)
    from .parallel import root_call, is_root

    root_call(_initialize_output, config, system, target, resume)
    if config["task"] == "drpmd":
        from .workflows.drpmd import run_drpmd

        result = run_drpmd(config, system, Path(target))
    elif config["task"] == "fes":
        from .workflows.fes import run_fes

        result = run_fes(config, system, Path(target))
    elif config["task"] == "rate":
        from .workflows.rate import run_rate

        result = run_rate(config, system, Path(target))
    else:
        raise InputError("Unknown workflow")
    if is_root() and result is not None:
        result = {
            **result,
            "software": "PyRPMD",
            "version": __version__,
            "output": target,
            "preset": config.get("preset", "pilot"),
            "input_warnings": config.get("_warnings", []),
        }
        write_json(Path(target) / "results" / "summary.json", result)
    return result


def resume_run(directory, *, parallel=None):
    path = Path(directory).resolve()
    if path.name == "checkpoints":
        path = path.parent
    config = json.loads((path / "input" / "resolved.json").read_text())
    system = System.from_dict(json.loads((path / "input" / "system.json").read_text()))
    if config.get("_base_directory") and config["_base_directory"] not in sys.path:
        sys.path.insert(0, config["_base_directory"])
    if parallel is not None:
        config["parallel"].update(parallel)
    return run_resolved(config, system, output=path, resume=True)


def analyze_run(directory, *, recompute=False, bins=None, bootstrap=None):
    """Read results, or reweight saved FES samples without new trajectories."""
    path = Path(directory).resolve()
    if not recompute and bins is None and bootstrap is None:
        summary = path / "results" / "summary.json"
        if not summary.is_file():
            summary = path / "summary.json"
        return json.loads(summary.read_text())
    config = json.loads((path / "input" / "resolved.json").read_text())
    if config["task"] != "fes":
        raise InputError(
            "Recomputing analysis is supported for FES; omit --recompute to read rate/dRPMD results"
        )
    system = System.from_dict(json.loads((path / "input" / "system.json").read_text()))
    if config.get("_base_directory") and config["_base_directory"] not in sys.path:
        sys.path.insert(0, config["_base_directory"])
    analysis = config.setdefault("analysis", {})
    if bins is not None:
        if not bins or any(
            not isinstance(v, int) or isinstance(v, bool) or v < 2 for v in bins
        ):
            raise InputError("Analysis bins must be integers >= 2")
        analysis["bins"] = bins[0] if len(bins) == 1 else bins
    if bootstrap is not None:
        if (
            not isinstance(bootstrap, int)
            or isinstance(bootstrap, bool)
            or bootstrap < 0
        ):
            raise InputError("Bootstrap replicate count must be an integer >= 0")
        analysis["bootstrap"] = bootstrap
    from .workflows.fes import reanalyze_fes

    result = reanalyze_fes(config, system, path)
    from .parallel import root_call

    result = {
        **result,
        "software": "PyRPMD",
        "version": __version__,
        "output": str(path),
        "preset": config.get("preset", "pilot"),
        "analysis_settings": analysis,
        "input_warnings": config.get("_warnings", []),
    }
    root_call(write_json, path / "results" / "summary.json", result)
    return result


def prepare(job, *, numerical=False):
    spec = _spec(job)
    config, system = spec.resolve()
    result = {
        "task": config["task"],
        "resolved": config,
        "system": system.to_dict(),
        "diagnostics": config.get("_warnings", []),
        "ready": True,
    }
    if config["task"] == "rate":
        from .workflows.prepare import prepare_rate

        result["reaction_preparation"] = prepare_rate(
            config, system, numerical=numerical
        )
        result["ready"] = result["reaction_preparation"].get("ready", False)
    elif config["task"] == "drpmd":
        result["ready"] = not (
            config["collision"].get("bmax_unvalidated", False)
            or config["outcomes"].get("classifier_unvalidated", False)
        )
        if not result["ready"]:
            result["diagnostics"].append(
                "Pilot simulations are available; production estimation requires a converged bmax and an explicit validated outcome classifier"
            )
    elif config["task"] == "fes":
        from .workflows.fes import resolve_umbrellas

        cv, specs, centers, stiffness = resolve_umbrellas(config, system)
        result["umbrella_preparation"] = {
            "dimension": cv.dimension,
            "windows": len(centers),
            "cv_units": cv.units,
            "centers_au": centers,
            "stiffness_au": stiffness,
        }
    if numerical:
        from .potentials import make_potential

        # The provider itself validates array shape, status and finite results.
        provider = make_potential(config["potential"], system)
        try:
            x = system.geometry[None, :, :]
            evaluated = provider.evaluate(x)
            from .potentials import validate_result

            validate_result(evaluated, x.shape)
            result["pes_check"] = {
                "energy_hartree": evaluated.energy.tolist(),
                "status": evaluated.status.tolist(),
            }
        finally:
            provider.close()
    return result
