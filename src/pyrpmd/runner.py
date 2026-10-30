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


def _read_identity(config, directory):
    """Restore a molecular System or the explicit system-free PCET model."""
    path = Path(directory) / "input"
    if config["task"] == "pcet":
        from .pcet_input import PCETModelIdentity

        return PCETModelIdentity.from_dict(json.loads((path / "model.json").read_text()))
    return System.from_dict(json.loads((path / "system.json").read_text()))


def _spec(value):
    if isinstance(value, JobSpec):
        return value
    if isinstance(value, (str, Path)):
        return JobSpec.from_file(value)
    return JobSpec.from_mapping(value)


def _initialize_output(config, system, output, resume=False):
    from .propagation import scientific_config

    path = Path(output)
    manifest_path = path / "manifest.json"
    hashes = referenced_files(config)
    digest = fingerprint(scientific_config(config), system)
    if resume:
        if not manifest_path.is_file():
            raise InputError(f"No run manifest in {path}")
        old = json.loads(manifest_path.read_text())
        schema = old.get("schema_version", 1)
        if type(schema) is not int or schema not in (1, 2):
            raise InputError(f"Unsupported run manifest schema version: {schema}")
        # Legacy manifests omitted the schema name. An explicit foreign name
        # must never be reinterpreted as a PyRPMD manifest during migration.
        if (schema == 2 or "schema" in old) and old.get("schema") != "pyrpmd.run":
            raise InputError("Unknown run manifest schema")
        old_digest = old["physics_fingerprint"]
        if schema == 1:
            # Authenticate the saved input before migrating an old identity.
            # A v0.1 run may have included execution-only block/backend options.
            stored_config = json.loads((path / "input" / "resolved.json").read_text())
            stored_system = _read_identity(stored_config, path)
            if fingerprint(stored_config, stored_system) != old_digest:
                raise InputError("Legacy saved input does not match its run manifest")
            old_digest = fingerprint(scientific_config(stored_config), stored_system)
        if old_digest != digest:
            raise InputError("Cannot resume with changed scientific settings or system")
        for file, sha in old.get("files", {}).items():
            if hashes.get(file) != sha:
                raise InputError(f"Input/model file changed or is missing: {file}")
        if schema == 1:
            write_json(manifest_path, {
                **old,
                "schema": "pyrpmd.run",
                "schema_version": 2,
                "checkpoint_schema_version": 2,
                "legacy_physics_fingerprint": old["physics_fingerprint"],
                "physics_fingerprint": digest,
                "physics_identity": "execution-independent-v2",
                "migrated_by_version": __version__,
            })
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
        "schema": "pyrpmd.run",
        "schema_version": 2,
        "checkpoint_schema_version": 2,
        "input_schema_version": config.get("schema_version", 1),
        "physics_identity": "execution-independent-v2",
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
        "hamiltonian_convention": (
            "one_proton_vibronic_golden_rule" if config["task"] == "pcet"
            else "sum_beads_beta_over_P"
        ),
        "units": {
            "length": "bohr",
            "momentum": "atomic",
            "energy": "hartree",
            "time": "atomic",
        },
    }
    if config["task"] == "pcet":
        manifest["model_schema"] = "pyrpmd.pcet-model"
        manifest["checkpoint_schema"] = "pyrpmd.pcet-case"
        manifest["checkpoint_schema_version"] = 1
        manifest["normalization"] = "local_first_order"
        manifest["units"].pop("momentum")
        manifest["units"]["rate"] = "s^-1"
        manifest["units"]["electrode_potential"] = "V"
    write_json(manifest_path, manifest)
    write_json(path / "input" / "resolved.json", config)
    identity_file = "model.json" if config["task"] == "pcet" else "system.json"
    write_json(path / "input" / identity_file, system.to_dict())
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
    if config["task"] == "pcet":
        from .workflows.pcet import run_pcet

        result = run_pcet(config, system, Path(target))
    elif config["task"] == "drpmd":
        from .workflows.drpmd import run_drpmd

        result = run_drpmd(config, system, Path(target))
    elif config["task"] == "fes":
        from .workflows.fes import run_fes

        result = run_fes(config, system, Path(target))
    elif config["task"] == "rate":
        from .workflows.rate import run_rate

        result = run_rate(config, system, Path(target))
    elif config["task"] == "metadynamics":
        from .workflows.metadynamics import run_metadynamics

        result = run_metadynamics(config, system, Path(target))
    elif config["task"] == "network":
        from .workflows.network import run_network

        result = run_network(config, system, Path(target))
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


def resume_run(directory, *, parallel=None, execution=None):
    path = Path(directory).resolve()
    if path.name == "checkpoints":
        path = path.parent
    config = json.loads((path / "input" / "resolved.json").read_text())
    system = _read_identity(config, path)
    if config["task"] == "pcet" and (parallel is not None or execution is not None):
        raise InputError("PCET resume does not accept trajectory parallelism or propagation overrides")
    if config.get("_base_directory") and config["_base_directory"] not in sys.path:
        sys.path.insert(0, config["_base_directory"])
    if parallel is not None:
        config["parallel"].update(parallel)
    if execution is not None:
        if not isinstance(execution, dict) or set(execution) - {"loop_backend", "block_steps"}:
            raise InputError("Resume execution overrides accept only loop_backend and block_steps")
        if "loop_backend" in execution and execution["loop_backend"] not in ("auto", "native", "python"):
            raise InputError("loop_backend must be auto/native/python")
        if "block_steps" in execution:
            value = execution["block_steps"]
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65536:
                raise InputError("block_steps must be an integer in 1..65536")
        config.setdefault("dynamics", {}).update(execution)
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
    if config["task"] not in ("fes", "metadynamics"):
        raise InputError(
            "Recomputing analysis supports FES and metadynamics; omit --recompute to read other results"
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
    if config["task"] == "metadynamics":
        from .workflows.metadynamics import reanalyze_metadynamics

        result = reanalyze_metadynamics(config, system, path)
    else:
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
    if config["task"] == "pcet":
        from .workflows.pcet import prepare_pcet

        return {
            "task": "pcet", "resolved": config, "model": system.to_dict(),
            "diagnostics": config.get("_warnings", []),
            "ready": True, "pcet_preparation": prepare_pcet(config, system, numerical=numerical),
        }
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
    elif config["task"] == "metadynamics":
        from .workflows.metadynamics import validate_metadynamics

        validate_metadynamics(config, system)
        result["metadynamics_preparation"] = config["metadynamics"]
    elif config["task"] == "network":
        from .workflows.network import prepare_network

        result["network_preparation"] = prepare_network(config, system, numerical=numerical)
        result["ready"] = result["network_preparation"].get("ready", False)
        return result
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
