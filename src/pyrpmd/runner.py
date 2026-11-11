"""Public run/prepare/resume entry points shared by Python and the CLI."""

from __future__ import annotations

import numpy as np
import copy
import importlib.metadata
import json
import platform
from collections.abc import Mapping
from pathlib import Path
import shutil
import sys
from . import __version__
from .input import JobSpec, InputError, fes_method, normalize_fes_task
from .io import write_json, referenced_files, fingerprint, jsonable, _windows_drive_files
from .system import System
from .stability import classify_stability


def _planned_propagation_steps(config):
    """Conservatively estimate the declared propagation work budget."""
    sampling = config.get("sampling", {})
    total = 0
    if isinstance(sampling, Mapping):
        total += int(sampling.get("equilibration_steps", 0) or 0)
        total += int(sampling.get("production_steps", 0) or 0)
        total *= max(1, int(sampling.get("replicas", 1) or 1))
    collision = config.get("collision", {})
    if isinstance(collision, Mapping):
        total += int(collision.get("max_steps", 0) or 0) * max(
            1, int(collision.get("trajectories", 1) or 1))
    recrossing = config.get("recrossing", {})
    if isinstance(recrossing, Mapping):
        parents = int(recrossing.get("parent_replicas", 1) or 1) * int(
            recrossing.get("parent_samples", 1) or 1)
        total += parents * int(recrossing.get("child_pairs", 1) or 1) * int(
            recrossing.get("child_steps", 0) or 0)
        total += int(recrossing.get("parent_replicas", 1) or 1) * int(
            recrossing.get("parent_samples", 1) or 1) * int(
            recrossing.get("parent_interval_steps", 0) or 0)
    return max(0, total)


def propagation_backend_plan(config, system, *, potential=None):
    """Describe the backend selected for each planned workflow stage.

    This constructs provider/CV descriptors but never evaluates a PES or
    advances a state, making it safe for ``pyrpmd validate``. An optional live
    potential is borrowed from preparation and remains owned by its caller.
    """
    if (config.get("task") in ("pimd", "diffusion")
            and config.get("sampling", {}).get("engine") == "torch"):
        return [{
            "stage": "production" if config["task"] == "diffusion" else "propagation",
            "backend": "unknown",
            "reason": "Torch tensor workflow; device and execution backend are confirmed at runtime",
            "planned_steps": _planned_propagation_steps(config),
        }]

    from .propagation import native_support_reason

    if config.get("task") == "rate":
        from .workflows.prepare import prepare_rate
        from .workflows.rate_pmf import configure_rate_umbrella
        try:
            preparation = prepare_rate(config, system, numerical=False)
            config = configure_rate_umbrella(config, system, preparation["reaction"])
        except (KeyError, ValueError):
            # Inferred reactions need numerical preparation before their CVs
            # exist; the stage below reports that unavailable plan explicitly.
            pass

    options = config.get("dynamics", {})
    if not isinstance(options, Mapping):
        options = {}
    provider = potential
    provider_owned = potential is None and isinstance(config.get("potential"), Mapping)
    dynamics = None
    base_reason = None
    base_backend = "native"
    try:
        from .potentials import make_potential
        from .dynamics import Dynamics
        potential_spec = config["potential"]
        kind = (potential_spec.get("provider", potential_spec.get("type", "builtin"))
                if isinstance(potential_spec, Mapping) else None)
        cheap_kinds = {"builtin", "free", "harmonic", "morse_pair", "double_well", "periodic_cosine"}
        if options.get("loop_backend") == "python" or not config.get("native", options.get("native", True)):
            base_reason = "explicit Python loop backend or native=False"
            base_backend = "python"
        elif config.get("parallel", {}).get("bead_workers", 1) > 1:
            base_reason = "distributed bead PES callbacks require Python scheduling"
            base_backend = "python"
        elif kind is not None and kind not in cheap_kinds:
            base_reason = f"PES provider '{kind}' binding is checked at runtime"
            base_backend = "unknown"
            provider = None
            dynamics = None
        else:
            provider = potential if potential is not None else make_potential(potential_spec, system)
            dynamics = Dynamics(
                provider, system.masses, config["beta"], config["beads"], config["dt_au"],
                native=config.get("native", options.get("native", True)),
                threads=config.get("parallel", {}).get("threads_per_rank", 1),
                fixed_atoms=getattr(system, "fixed_atoms", ()),
                fixed_positions=system.geometry if getattr(system, "fixed_atoms", ()) else None,
                **{key: options[key] for key in ("loop_backend", "block_steps", "integrator")
                   if key in options},
            )
            base_reason = native_support_reason(dynamics)
            base_backend = "python" if base_reason else "native"
    except Exception as exc:
        base_reason = f"validation could not construct native plan: {exc}"
        base_backend = "unknown"

    planned = _planned_propagation_steps(config)
    stages = []

    def add(stage, reason=None, *, steps=planned, backend=None):
        selected = base_reason if reason is None else reason
        selected_backend = base_backend if reason is None else "python"
        stages.append({"stage": stage, "backend": backend or selected_backend,
                       "reason": selected, "planned_steps": steps})

    task = config.get("task")
    if task in ("fes", "rate", "surface_rate", "site_rate"):
        try:
            from .input import rate_pmf_options, rate_pmf_stage
            from .workflows.fes import resolve_window_plan
            from .workflows.umbrella_plan import window_sampling
            from .bias import UmbrellaBias
            stage_config = (rate_pmf_stage(config, system) if task == "rate" else None) or config
            external = task == "rate" and rate_pmf_options(config)["source"] is not None
            if stage_config is not config:
                # Independent PMF stages own their integrator, clock, native
                # switch and thermostat; use the same resolved stage as run_rate.
                stages.extend(propagation_backend_plan(stage_config, system, potential=provider))
            elif not external:
                cv, _, windows = resolve_window_plan(stage_config, system)
                steps = sum(
                    (int(sampling.get("equilibration_steps", 0))
                     + int(sampling.get("production_steps", 0)))
                    * int(sampling.get("replicas", 1))
                    for sampling in (window_sampling(stage_config, w) for w in windows)
                )
                reason = None
                backend = None
                from .workflows.sampling import fes_sampler
                sampler = fes_sampler(stage_config)
                if (sampler.thermostat == "pile-l" and sampler.thermostat_spec is None
                        and options.get("loop_backend", "auto") != "native"):
                    reason = "PILE-L umbrella force steps use Python scheduling"
                elif sampler.thermostat != "andersen":
                    reason = "sampling operator backend is checked at runtime"
                    backend = "unknown"
                elif dynamics is not None:
                    reason = native_support_reason(
                        dynamics, bias=UmbrellaBias(cv, windows[0].center, windows[0].stiffness),
                        observe=cv)
                add("umbrella", reason, steps=steps, backend=backend)
        except Exception as exc:
            add("umbrella", f"CV/bias plan unavailable: {exc}", backend="unknown")
    elif task == "diffusion" and config.get("electronic_friction") is not None:
        provider_kind = config.get("electronic_friction", {}).get("provider", "constant")
        if provider_kind != "constant":
            add("production", "electronic friction provider callbacks require the Python loop")
        elif dynamics is None or base_backend != "native":
            add("production")
        else:
            # Constant scalar or diagonal centroid friction has a fused native
            # block; ask the same eligibility check that execution uses.
            friction = None
            try:
                from .electronic_friction import ElectronicFrictionDynamics, make_electronic_friction
                from .friction_block import native_friction_plan
                from .propagation import NativeUnavailableError
                friction = make_electronic_friction(config, system)
                driver = ElectronicFrictionDynamics(
                    provider, system.masses, config["beta"], config["beads"], config["dt_au"], friction,
                    native=config.get("native", options.get("native", True)),
                    threads=config.get("parallel", {}).get("threads_per_rank", 1),
                    fixed_atoms=getattr(system, "fixed_atoms", ()),
                    fixed_positions=system.geometry if getattr(system, "fixed_atoms", ()) else None,
                    **{key: options[key] for key in ("loop_backend", "block_steps", "integrator")
                       if key in options},
                )
                try:
                    native_friction_plan(driver)
                    add("production")
                except NativeUnavailableError as exc:
                    add("production", str(exc))
            except Exception as exc:
                add("production", f"electronic-friction plan could not be inspected: {exc}",
                    backend="unknown")
            finally:
                if callable(getattr(friction, "close", None)):
                    friction.close()
    elif task != "diffusion":
        add("propagation")

    if task in ("rate", "surface_rate", "site_rate"):
        from .workflows.rate_options import _recrossing_options, _recrossing_spec
        recrossing = _recrossing_options(config)
        parent_steps = recrossing["parent_replicas"] * (
            recrossing["equilibration_steps"]
            + recrossing["parent_samples"] * recrossing["parent_interval_steps"])
        child_steps = (2 * recrossing["parent_replicas"] * recrossing["parent_samples"]
                       * recrossing["child_pairs"] * recrossing["child_steps"])
        if task == "site_rate":
            add("site parent", "centroid RATTLE parent sampling uses the Python loop", steps=parent_steps)
            add("site child", "checkpointed child classification uses the Python loop", steps=child_steps)
        else:
            parent_reason = child_reason = None
            parent_backend = child_backend = base_backend
            if dynamics is not None:
                try:
                    from .block_propagation import _native_plan, NotNative
                    from .propagation import NativeUnavailableError
                    from .workflows.rate_constraints import rate_coordinate
                    dividing_surface = 1.0
                    if task == "surface_rate":
                        from .workflows.surface import surface_definition
                        dividing_surface = surface_definition(config, system)["dividing_surface_bohr"]
                    surface = rate_coordinate(config, system).dividing_surface(dividing_surface)
                    child_reason = native_support_reason(dynamics, observe=surface)
                    child_backend = "python" if child_reason else "native"
                    if dynamics.loop_backend == "python":
                        parent_reason = "explicit Python loop backend"
                    else:
                        try:
                            _native_plan(dynamics, None, None, surface, (), 1e-10, False, 0.0)
                        except (NotNative, NativeUnavailableError) as exc:
                            parent_reason = str(exc)
                    parent_backend = "python" if parent_reason else "native"
                except Exception as exc:
                    parent_reason = child_reason = f"rate surface plan unavailable: {exc}"
                    parent_backend = child_backend = "unknown"
            raw = _recrossing_spec(config)
            if raw.get("dynamics", "rpmd") == "trpmd" or raw.get("child_batch_pairs", 1) > 1:
                child_reason = "TRPMD or batched child propagation uses Python scheduling"
                child_backend = "python"
            add("recrossing parent", parent_reason, steps=parent_steps, backend=parent_backend)
            add("recrossing child", child_reason, steps=child_steps, backend=child_backend)
    elif task == "diffusion" and config.get("electronic_friction") is None:
        add("production")

    if provider_owned and provider is not None:
        close = getattr(provider, "close", None)
        if callable(close):
            close()
    return stages


def _append_backend_warnings(config, plan):
    """Warn once when a large declared stage resolves to Python."""
    warnings = config.setdefault("_warnings", [])
    for item in plan:
        if item.get("backend") == "python" and item.get("planned_steps", 0) > 1_000_000:
            message = (f"{item['stage']}: Python propagation loop selected for "
                       f"{item['planned_steps']:,} planned steps ({item.get('reason')}); "
                       "this may be substantially slower than the native loop.")
            if message not in warnings:
                warnings.append(message)


def _summary_propagation(result):
    """Expose propagation totals while preserving each workflow's public shape."""
    if not isinstance(result, Mapping):
        return result
    existing = result.get("propagation")
    if (isinstance(existing, Mapping)
            and {"native_steps", "python_steps", "fallback_reasons"}.issubset(existing)):
        return result
    records = []

    def visit(value):
        if isinstance(value, Mapping):
            if {"native_steps", "python_steps"}.issubset(value):
                records.append(value)
                return  # This node already aggregates any nested stage details.
            for key, child in value.items():
                if key != "propagation_totals":
                    visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(result)
    from .propagation import combine_propagation_stats
    combined = combine_propagation_stats(records)
    if isinstance(existing, Mapping):
        # Preserve stage-specific detail while making the aggregate counters
        # available at the stable top level expected by result consumers.
        return {**result, "propagation": {**existing, **combined}}
    if isinstance(existing, (list, tuple)):
        return {**result, "propagation_totals": combined}
    return {**result, "propagation": combined}


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


def _validate_resume_identity(config, system, path, old, hashes, digest):
    """Authenticate a resume without changing saved inputs or manifest bytes."""
    from .propagation import scientific_config

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
    migrated_fes = False
    if old_digest != digest:
        stored_config = json.loads((path / "input" / "resolved.json").read_text())
        stored_system = _read_identity(stored_config, path)
        authenticated = fingerprint(scientific_config(stored_config), stored_system) == old_digest
        if authenticated and stored_config.get("task") in ("fes", "metadynamics", "opes"):
            normalized = normalize_fes_task(copy.deepcopy(stored_config), warn=False)
            migrated_fes = fingerprint(scientific_config(normalized), stored_system) == digest
        if not migrated_fes:
            raise InputError("Cannot resume with changed scientific settings or system")
    mutable_input = None
    if config.get("task") == "rate" and "subspace_correction" in config.get("rate", {}):
        stored_config = json.loads((path / "input" / "resolved.json").read_text())
        # Physics equality was established above. Allow editing only the .inp
        # wrapper to tune postprocessing, never a PES/model dependency.
        source = stored_config.get("_input_path")
        if source and Path(source).suffix.lower() in (".inp", ".toml"):
            mutable_input = source
    for file, sha in old.get("files", {}).items():
        if file == mutable_input:
            continue
        if hashes.get(file) != sha:
            raise InputError(f"Input/model file changed or is missing: {file}")

    # Before f9e85b4, explicit drive-letter inputs could be absent from files.
    # Today's content cannot establish the provenance of an existing trajectory.
    # Use the saved platform and saved references, not execution overrides or
    # the current host; a copied Windows run has the same evidence gap.
    if str(old.get("platform", "")).startswith("Windows"):
        stored_config = json.loads((path / "input" / "resolved.json").read_text())
        missing = sorted(_windows_drive_files(stored_config, hashes)
                         - old.get("files", {}).keys())
        if missing:
            raise InputError(
                "Cannot resume this Windows run: its manifest has no recorded "
                "source/model hashes for: " + ", ".join(missing) + ". "
                "Older drive-letter path handling (before f9e85b4) could omit "
                "these inputs. Their original contents cannot be verified; "
                "hashing the current files cannot prove their historical source. "
                "Keep the old run for inspection and rerun the original input "
                "in a new, empty output directory (pyrpmd run INPUT --output NEW_RUN). "
                "Do not edit the manifest or copy current hashes into it."
            )
    return schema, migrated_fes


def _check_extension_provenance(directory):
    """Refuse unprovable Windows extensions before acquiring/writing a lock."""
    from .propagation import scientific_config

    path = Path(directory).resolve()
    if path.name == "checkpoints":
        path = path.parent
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise InputError(f"No run manifest in {path}")
    old = json.loads(manifest_path.read_text())
    if not str(old.get("platform", "")).startswith("Windows"):
        return
    config = json.loads((path / "input" / "resolved.json").read_text())
    system = _read_identity(config, path)
    if config.get("_base_directory") and config["_base_directory"] not in sys.path:
        sys.path.insert(0, config["_base_directory"])
    _validate_resume_identity(
        config, system, path, old, referenced_files(config),
        fingerprint(scientific_config(config), system),
    )


def _initialize_output(config, system, output, resume=False, initialized_from=None):
    from .propagation import scientific_config

    path = Path(output)
    if path.exists() and not path.is_dir():
        raise InputError(f"Output path is not a directory: {path}; choose a directory for the run")
    manifest_path = path / "manifest.json"
    hashes = referenced_files(config)
    digest = fingerprint(scientific_config(config), system)
    if resume:
        if not manifest_path.is_file():
            raise InputError(f"No run manifest in {path}")
        old = json.loads(manifest_path.read_text())
        schema, migrated_fes = _validate_resume_identity(
            config, system, path, old, hashes, digest,
        )
        if schema == 1 or migrated_fes:
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
            if migrated_fes:
                write_json(path / "input" / "resolved.json", config)
        return
    if path.exists() and any(path.iterdir()):
        raise InputError(
            f"Output directory is not empty: {path}; choose another directory or resume"
        )
    path.mkdir(parents=True, exist_ok=True)
    for child in ("input", "results", "checkpoints", "samples", "logs"):
        (path / child).mkdir(exist_ok=True)
    versions = {}
    for name in ("numpy", "scipy", "h5py", "mpi4py", "torch", "ase", "openmm", "lammps"):
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
    if config.get("warm_start"):
        # Provenance only: the imported configuration itself is part of the
        # resolved input and therefore of this run's own physics fingerprint.
        manifest["initialized_from"] = {
            **config["warm_start"]["initialized_from"],
            **({"run": str(initialized_from)} if initialized_from is not None else {}),
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


def run(job, *, output=None, resume=False, init_from=None, init_index=None, init_bias=False, stage="all"):
    """Execute a JobSpec, mapping, .inp or .py file and return the root summary.

    ``init_from`` warm-starts a *new* run from a configuration saved by an
    earlier run (a run directory or one of its checkpoint files). The new run
    keeps its own scientific settings and fingerprint; see
    :mod:`pyrpmd.warm_start`. ``init_index`` selects one configuration from a
    checkpoint holding several, and ``init_bias=True`` additionally imports
    the parent's metadynamics hills as an initial bias.
    ``stage="subspace"`` validates an existing rate run and executes only its
    configured subspace postprocessing, without preparation or new dynamics.
    """
    spec = _spec(job)
    if output is not None:
        spec = spec.with_overrides(output=output)
    config, system = spec.resolve()
    return run_resolved(config, system, output=output, resume=resume, init_from=init_from,
                        init_index=init_index, init_bias=init_bias, stage=stage)


def run_resolved(config, system, *, output=None, resume=False, init_from=None, init_index=None,
                 init_bias=False, stage="all"):
    config = normalize_fes_task(copy.deepcopy(config))
    if stage not in ("all", "subspace"):
        raise InputError("stage must be all or subspace")
    if stage == "subspace":
        if config.get("task") != "rate":
            raise InputError("--stage subspace requires task='rate'")
        if resume or init_from is not None or init_index is not None or init_bias:
            raise InputError("--stage subspace cannot be combined with resume or warm-start options")
        from .parallel import root_call
        from .workflows.rate_subspace import run_subspace_stage

        return root_call(run_subspace_stage, config, system, output or config["output"]["directory"])
    if config.get("task") == "rate":
        from .input_subspace import normalize_rate_subspace

        normalize_rate_subspace(config)
    parent_run = None
    if init_from is None and (init_index is not None or init_bias):
        raise InputError("init_index and init_bias require init_from")
    if init_from is not None:
        if resume:
            raise InputError("A warm start creates a new run; it cannot be combined with resume. "
                             "Resume the warm-started run without init_from.")
        from . import warm_start

        config, parent_run = warm_start.prepare(config, system, init_from, index=init_index,
                                                import_bias_history=init_bias)
    from .workflows import validate_bond_constraint_workflow
    from .workflows.sampling import validate_andersen_schedule

    validate_bond_constraint_workflow(system, config.get("task"))
    validate_andersen_schedule(config, config.get("task"))
    if ((config.get("task") in ("pimd", "diffusion") and "path_integral" not in config)
            or config.get("sampling", {}).get("engine") == "torch"
            or getattr(system, "bond_constraints", ())):
        # All materials backends share preflight, including direct resolved
        # calls, before creating or migrating an output directory.
        from .workflows.materials import resolve_materials
        resolve_materials(config, system)
    if (config.get("task") == "plumed" and resume
            and config.get("plumed", {}).get("restart") != "files"):
        raise InputError("PLUMED automatic resume is unsupported because its opaque kernel history is not in the MD checkpoint")
    if config.get("task") == "rate" and config.get("rate", {}).get("pmf_source") not in (None, "umbrella"):
        from .workflows.rate_source import source_paths

        config["rate"]["pmf_source"] = source_paths(config["rate"]["pmf_source"], config.get("_base_directory"))
    if "electronic_friction" in config:
        from .electronic_friction import validate_electronic_friction
        validate_electronic_friction(config, system)
    if ("nonadiabatic" in config
            or config.get("potential", {}).get("provider", config.get("potential", {}).get("type")) == "diabatic_python"):
        from .input import validate_nonadiabatic
        validate_nonadiabatic(config, system=system)
    target = str(Path(output or config["output"]["directory"]).resolve())
    config["output"]["directory"] = target
    config["resume"] = bool(resume)
    from .parallel import root_call, is_root

    maturity = classify_stability(config)
    if is_root() and maturity["level"] == "research":
        import warnings

        warnings.warn(f"Research-level workflow: {maturity['scope']}", UserWarning, stacklevel=2)
    # Reject before initialization can migrate a legacy manifest bound by the ledger.
    if (resume and config["task"] == "rate"
            and (Path(target) / "results" / "extensions.json").is_file()):
        raise InputError("This rate run has an authorized extension budget; use resume_run or pyrpmd resume")
    if "tempering" in config:
        from .workflows.tempering import validate_tempering

        root_call(validate_tempering, config, system)
    if config["task"] == "fes":
        from .workflows.fes_dispatch import validate_fes

        root_call(validate_fes, config, system)
    if config["task"] == "path_sampling":
        from .workflows.path_sampling import resolve_path_sampling

        root_call(resolve_path_sampling, config, system)
    if (parent_run is not None and Path(parent_run).resolve() == Path(target).resolve()):
        raise InputError("A warm start needs a new output directory, not the parent run directory")
    backend_plan = [] if config.get("task") == "pcet" else propagation_backend_plan(config, system)
    _append_backend_warnings(config, backend_plan)

    root_call(_initialize_output, config, system, target, resume, parent_run)
    if config["task"] == "path_sampling":
        from .workflows.path_sampling import run_path_sampling

        result = run_path_sampling(config, system, Path(target))
    elif "tempering" in config:
        from .workflows.tempering import run_tempering

        result = run_tempering(config, system, Path(target))
    elif config["task"] == "spectroscopy":
        from .workflows.spectroscopy import run_spectroscopy

        result = run_spectroscopy(config, system, Path(target))
    elif config["task"] == "plumed":
        from .workflows.plumed import run_plumed

        result = run_plumed(config, system, Path(target))
    elif config["task"] == "pcet":
        from .workflows.pcet import run_pcet

        result = run_pcet(config, system, Path(target))
    elif config["task"] == "nonadiabatic":
        from .workflows.nonadiabatic import run_nonadiabatic

        result = run_nonadiabatic(config, system, Path(target))
    elif config["task"] == "drpmd":
        from .workflows.drpmd import run_drpmd

        result = run_drpmd(config, system, Path(target))
    elif config["task"] == "fes":
        from .workflows.fes_dispatch import run_fes

        result = run_fes(config, system, Path(target))
    elif config["task"] == "rate":
        from .workflows.rate import run_rate

        result = run_rate(config, system, Path(target))
    elif config["task"] == "surface_rate":
        from .workflows.surface import run_surface_rate

        result = run_surface_rate(config, system, Path(target))
    elif config["task"] == "network":
        from .workflows.network import run_network

        result = run_network(config, system, Path(target))
    elif config["task"] == "pimd" and "path_integral" in config:
        from .workflows.path_integral import run_path_integral

        result = run_path_integral(config, system, Path(target))
    elif config["task"] in ("pimd", "diffusion"):
        from .workflows.materials import run_materials

        result = run_materials(config, system, Path(target))
    elif config["task"] == "site_rate":
        from .workflows.site_rate import run_site_rate

        result = run_site_rate(config, system, Path(target))
    else:
        raise InputError("Unknown workflow")
    if is_root() and result is not None:
        result = _summary_propagation(result)
        result = {
            **result,
            "software": "PyRPMD",
            "version": __version__,
            "output": target,
            "preset": config.get("preset", "pilot"),
            "input_warnings": config.get("_warnings", []),
            "stability": classify_stability(config),
            "propagation_plan": backend_plan,
        }
        if config.get("warm_start"):
            from .warm_start import report

            result["warm_start"] = report(config)
        # dRPMD returns records for API compatibility, but persists the full
        # table separately so the run's summary stays bounded in size.
        stored = ({key: value for key, value in result.items() if key != "records"}
                  if config["task"] == "drpmd" else result)
        write_json(Path(target) / "results" / "summary.json", stored)
    return result


def _resume_parallel(current, overrides):
    """Apply execution-only overrides without silently retaining old aliases."""
    from .input import _SECTIONS, _unknown_fields
    from .parallel import execution_plan

    if not isinstance(overrides, Mapping) or any(not isinstance(key, str) for key in overrides):
        raise InputError("Resume parallel overrides must be a mapping with string field names")
    unknown = set(overrides) - _SECTIONS["parallel"]
    if unknown:
        raise _unknown_fields(unknown, _SECTIONS["parallel"], "parallel")
    aliases = ("workers", "trajectory_workers", "trajectory_groups")
    counts = []
    for key in aliases:
        if key in overrides:
            value = overrides[key]
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                    or value < 1):
                raise InputError(f"parallel.{key} must be an integer >= 1")
            counts.append(int(value))
    if counts and any(value != counts[0] for value in counts):
        raise InputError("Resume workers, trajectory_workers and trajectory_groups overrides disagree")
    merged = copy.deepcopy(current)
    merged.update(overrides)
    if counts:
        merged.pop("workers", None)
        merged["trajectory_workers"] = merged["trajectory_groups"] = counts[0]
    if ("scheduling" in overrides or "backend" in overrides) and "coordinator_ranks" not in overrides:
        merged["coordinator_ranks"] = int(
            merged.get("scheduling", "static") == "dynamic"
            and merged.get("backend", "serial") == "mpi"
        )
    try:
        execution_plan(merged)
    except (ValueError, TypeError) as error:
        raise InputError(str(error)) from error
    return merged


def resume_run(directory, *, parallel=None, execution=None, adaptive_budget=None):
    path = Path(directory).resolve()
    if path.name == "checkpoints":
        path = path.parent
    if not (path / "manifest.json").is_file():
        raise InputError(
            f"No run manifest in {path}; pass the run directory containing manifest.json "
            "or its checkpoints subdirectory"
        )
    resolved_path = path / "input" / "resolved.json"
    if not resolved_path.is_file():
        raise InputError(f"Cannot resume {path}: saved input/resolved.json is missing")
    config = json.loads(resolved_path.read_text())
    system = _read_identity(config, path)
    if config["task"] == "pcet" and (parallel is not None or execution is not None):
        raise InputError("PCET resume does not accept trajectory parallelism or propagation overrides")
    if config.get("_base_directory") and config["_base_directory"] not in sys.path:
        sys.path.insert(0, config["_base_directory"])
    extension_path = path / "results" / "extensions.json"
    if config["task"] == "rate" and extension_path.is_file():
        ledger = json.loads(extension_path.read_text())
        if not isinstance(ledger, dict) or ledger.get("format") != "PyRPMD.rate.extensions.v1":
            raise InputError("Unknown rate production extension ledger")
        if parallel is not None or execution is not None or adaptive_budget is not None:
            raise InputError("Resume the rate extension without execution or adaptive-budget overrides")
        requests = ledger.get("requests")
        if not isinstance(requests, list) or not requests or not isinstance(requests[-1], dict):
            raise InputError("Invalid rate production extension requests")
        # Completed requests also retain their effective total budget. Falling
        # through to the original short configuration would publish stale rates.
        return extend_run(path, production_steps=requests[-1].get("production_steps"))
    if config["task"] == "drpmd" and extension_path.is_file():
        ledger = json.loads(extension_path.read_text())
        if ledger.get("format") != "PyRPMD.drpmd.extensions.v1":
            raise InputError("Unknown unresolved-trajectory extension ledger")
        requests = ledger.get("requests", [])
        if requests and requests[-1].get("status") == "running":
            if parallel is not None or execution is not None or adaptive_budget is not None:
                raise InputError("Resume the pending extension without execution or adaptive-budget overrides")
            target = requests[-1].get("max_steps")
            if type(target) is not int or target <= config["collision"]["max_steps"]:
                raise InputError("Invalid pending extension budget")
            # extend_run authenticates the original input/manifest and every
            # selected checkpoint before continuing the already authorized work.
            return extend_run(path, max_time=target * config["dt_au"])
    if parallel is not None:
        config["parallel"] = _resume_parallel(config.get("parallel", {}), parallel)
    if execution is not None:
        allowed = {"loop_backend", "block_steps"}
        mapping = config["task"] == "nonadiabatic" and config["method"] in {"nrpmd", "sm-nrpmd"}
        if mapping:
            allowed |= {"electronic_backend", "electronic_device"}
        if not isinstance(execution, dict) or set(execution) - allowed:
            names = ("loop_backend, block_steps, electronic_backend and electronic_device"
                     if mapping else "loop_backend and block_steps")
            raise InputError("Resume execution overrides accept only " + names)
        if "loop_backend" in execution and execution["loop_backend"] not in ("auto", "native", "python"):
            raise InputError("loop_backend must be auto/native/python")
        if "block_steps" in execution:
            value = execution["block_steps"]
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65536:
                raise InputError("block_steps must be an integer in 1..65536")
        config.setdefault("dynamics", {}).update(execution)
        if mapping:
            from .nonadiabatic_input import _choice
            import re
            options = config["dynamics"]
            backend = _choice(options.get("electronic_backend", "numpy"),
                              "electronic_backend", {"numpy", "native", "torch"})
            device = options.get("electronic_device", "cpu")
            if not isinstance(device, str) or re.fullmatch(r"cpu|cuda(?::[0-9]+)?", device) is None:
                raise InputError("electronic_device must be cpu or cuda[:index]")
            if backend != "torch" and device != "cpu":
                raise InputError("a CUDA electronic_device requires electronic_backend='torch'")
    if adaptive_budget is not None:
        from .parallel import root_call
        from .workflows.adaptive_umbrella import extend_adaptive_budget

        adaptive = config.get("adaptive_umbrella")
        if config["task"] == "rate" and "pmf" in config.get("rate", {}):
            adaptive = config.get("rate", {}).get("pmf_stage", {}).get("config", {}).get("adaptive_umbrella")
        if config["task"] not in ("fes", "rate") or not adaptive:
            raise InputError("adaptive budget overrides require an adaptive FES or rate run")
        # Authenticate saved inputs/model files before modifying the plan.
        root_call(_initialize_output, config, system, path, True)
        pilot_output = path / "umbrella" if config["task"] == "rate" else path
        root_call(extend_adaptive_budget, pilot_output, adaptive_budget)
    return run_resolved(config, system, output=path, resume=True)


def extend_run(directory, *, max_time=None, production_steps=None):
    """Continue a saved run to one explicitly increased total budget.

    ``max_time`` extends unresolved dRPMD launches. ``production_steps`` extends
    every existing fixed-window gas-rate umbrella stream, excluding equilibration,
    and generates fresh recrossing trajectories for the resulting PMF. Supply
    exactly one budget. Repeating its total recovers or returns the same result.
    Original input and manifest stay unchanged; a separate ledger authorizes work.
    """
    if (max_time is None) == (production_steps is None):
        raise InputError("Specify exactly one of max_time or production_steps")
    from .parallel import root_call

    root_call(_check_extension_provenance, directory)
    if production_steps is not None:
        from .workflows.rate_extend import extend_rate

        return extend_rate(directory, production_steps=production_steps)
    from .workflows.extend import extend_drpmd

    return extend_drpmd(directory, max_time=max_time)


def analyze_run(directory, *, recompute=False, bins=None, bootstrap=None, diffusion=None):
    """Read results or reanalyze saved FES/transport data without new trajectories."""
    path = Path(directory).resolve()
    if not recompute and bins is None and bootstrap is None and diffusion is None:
        summary = path / "results" / "summary.json"
        if not summary.is_file():
            summary = path / "summary.json"
        result = json.loads(summary.read_text())
        if "rate_extension" in result:
            from .rate_artifacts import active_rate_directory

            active_rate_directory(path, result)
        if result.get("task") in ("fes", "metadynamics", "opes"):
            from .analysis.fes_result import FESResult

            if (path / "fes.npz").is_file():
                return FESResult.read(path, summary=result).summary()
        return result
    config = json.loads((path / "input" / "resolved.json").read_text())
    if "tempering" in config:
        if bins is not None or bootstrap is not None or diffusion is not None:
            raise InputError("Tempering reanalysis does not accept FES/diffusion overrides")
        from .workflows.tempering import reanalyze_tempering
        from .parallel import root_call

        system = _read_identity(config, path)
        _initialize_output(config, system, path, resume=True)
        result = reanalyze_tempering(config, system, path)
        if result is None:
            return None
        result = {
            **result, "software": "PyRPMD", "version": __version__,
            "output": str(path), "preset": config.get("preset", "pilot"),
            "input_warnings": config.get("_warnings", []),
            "stability": classify_stability(config),
        }
        root_call(write_json, path / "results" / "summary.json", result)
        return result
    if config["task"] == "nonadiabatic":
        if bins is not None or bootstrap is not None or diffusion is not None:
            raise InputError("Nonadiabatic time-series analysis does not accept FES/diffusion overrides")
        from .workflows.nonadiabatic import analyze_nonadiabatic

        system = _read_identity(config, path)
        _initialize_output(config, system, path, resume=True)
        result = analyze_nonadiabatic(config, system, path)
        result.update(software="PyRPMD", version=__version__, output=str(path),
                      preset=config.get("preset", "pilot"),
                      input_warnings=config.get("_warnings", []), stability=classify_stability(config))
        write_json(path / "results" / "summary.json", result)
        return result
    if config["task"] not in ("fes", "metadynamics", "opes", "diffusion", "spectroscopy"):
        raise InputError(
            "Recomputing analysis supports FES, metadynamics, OPES, diffusion, spectroscopy and nonadiabatic populations; omit --recompute to read other results"
        )
    system = System.from_dict(json.loads((path / "input" / "system.json").read_text()))
    if config.get("_base_directory") and config["_base_directory"] not in sys.path:
        sys.path.insert(0, config["_base_directory"])
    if config["task"] == "spectroscopy":
        if bins is not None or bootstrap is not None or diffusion is not None:
            raise InputError("spectroscopy reanalysis does not use FES/diffusion overrides")
        from .workflows.spectroscopy import reanalyze_spectroscopy
        result = reanalyze_spectroscopy(config, system, path)
        write_json(path / "results" / "summary.json", result)
        return result
    if config["task"] == "diffusion":
        from .input import _SECTIONS
        from .units import parse_quantity
        from .workflows.materials import reanalyze_diffusion, validate_materials
        from .parallel import root_call

        if bins is not None:
            raise InputError("Diffusion analysis does not use histogram bins")
        if diffusion is not None:
            if not isinstance(diffusion, dict) or set(diffusion) - _SECTIONS["diffusion"]:
                raise InputError("diffusion overrides must contain documented diffusion analysis options")
            config.setdefault("diffusion", {}).update(copy.deepcopy(diffusion))
        if bootstrap is not None:
            config.setdefault("diffusion", {})["bootstrap"] = bootstrap
        options = config.setdefault("diffusion", {})
        if "fit_range" in options:
            if not isinstance(options["fit_range"], (list, tuple)) or len(options["fit_range"]) != 2:
                raise InputError("diffusion.fit_range must be [start, stop] times")
            options["fit_range"] = [None if v is None else parse_quantity(v, "time")
                                    for v in options["fit_range"]]
        validate_materials(config, system)
        result = reanalyze_diffusion(config, system, path)
        if result is None:
            return None
        result = {
            **result, "software": "PyRPMD", "version": __version__,
            "output": str(path), "preset": config.get("preset", "pilot"),
            "input_warnings": config.get("_warnings", []),
            "stability": classify_stability(config),
        }
        root_call(write_json, path / "results" / "summary.json", result)
        return result
    if diffusion is not None:
        raise InputError("Diffusion analysis overrides require a diffusion run")
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
    from .workflows.fes_dispatch import reanalyze_fes
    from .analysis.fes_result import FESResult

    result = reanalyze_fes(config, system, path)
    if result is None:
        return None
    if fes_method(config) != "steered":
        result = FESResult.read(path, summary=result).summary()
    from .parallel import root_call

    result = {
        **result,
        "software": "PyRPMD",
        "version": __version__,
        "output": str(path),
        "preset": config.get("preset", "pilot"),
        "analysis_settings": analysis,
        "input_warnings": config.get("_warnings", []),
        "stability": classify_stability(config),
    }
    root_call(write_json, path / "results" / "summary.json", result)
    return result


def prepare(job, *, numerical=False):
    from .resources import resource_report

    spec = _spec(job)
    config, system = spec.resolve()
    resources = resource_report(config)
    result = _prepare(spec, numerical=numerical and resources["available"], _resolved=(config, system))
    result.update(resources=resources, ready=bool(result.get("ready", True) and resources["available"]),
                  stability=classify_stability(config))
    return result


def _prepare(job, *, numerical=False, _resolved=None):
    spec = _spec(job)
    config, system = spec.resolve() if _resolved is None else _resolved
    backend_plan = []
    if config["task"] == "nonadiabatic":
        from .workflows.nonadiabatic import prepare_nonadiabatic

        preparation = prepare_nonadiabatic(config, system, numerical=numerical)
        backend_plan = propagation_backend_plan(config, system)
        _append_backend_warnings(config, backend_plan)
        return {
            "task": "nonadiabatic", "resolved": config, "system": system.to_dict(),
            "diagnostics": config.get("_warnings", []), "ready": True,
            "propagation_plan": backend_plan,
            "nonadiabatic_preparation": preparation,
        }
    if config["task"] == "pcet":
        from .workflows.pcet import prepare_pcet

        return {
            "task": "pcet", "resolved": config, "model": system.to_dict(),
            "diagnostics": config.get("_warnings", []),
            "propagation_plan": backend_plan,
            "ready": True, "pcet_preparation": prepare_pcet(config, system, numerical=numerical),
        }
    result = {
        "task": config["task"],
        "resolved": config,
        "system": system.to_dict(),
        "diagnostics": config.get("_warnings", []),
        "propagation_plan": backend_plan,
        "ready": True,
    }
    if config["task"] == "fes":
        result["method"] = fes_method(config)
    if config["task"] == "rate":
        from .workflows.prepare import prepare_rate, prepared_system
        from .workflows.rate_plan import rate_pmf_plan

        result["pmf_plan"] = rate_pmf_plan(config)

        result["reaction_preparation"] = prepare_rate(
            config, system, numerical=numerical
        )
        result["ready"] = result["reaction_preparation"].get("ready", False)
        system = prepared_system(system, result["reaction_preparation"])
        result["system"] = system.to_dict()
        if numerical and result["ready"] and "pmf" in config.get("rate", {}):
            from .workflows.rate import _prepare

            _prepare(config, system, ".", _preparation=result["reaction_preparation"], _write=False)
        reaction = result["reaction_preparation"]["reaction"]
        if config.get("reaction", {}).get("definition"):
            from .workflows.rate_options import _recrossing_options, _parent_thermostat

            options = _recrossing_options(config)
            interval, schedule = _parent_thermostat(config)
            configurations = options["parent_replicas"] * options["parent_samples"]
            pairs = configurations * options["child_pairs"]
            result["recrossing_preparation"] = {
                **options,
                "parent_refresh_interval_steps": interval,
                "parent_refresh_schedule": schedule,
                "planned_configurations": configurations,
                "planned_pairs": pairs,
                "planned_trajectories": 2 * pairs,
            }
            if all(reaction.get(key) for key in ("fragments", "forming_bonds", "breaking_bonds")):
                from .input_options import rate_pmf_options
                from .input_rate_pmf import rate_pmf_stage
                from .workflows.fes import validate_fes_request
                from .workflows.rate_pmf import configure_rate_umbrella
                from .workflows.umbrella_plan import plan_preview

                planned = configure_rate_umbrella(config, system, reaction)
                pmf_options = rate_pmf_options(planned)
                if pmf_options["source"] is None and pmf_options.get("sampler", "umbrella") == "umbrella":
                    stage = rate_pmf_stage(planned, system)
                    sampling_config = planned if stage is None else stage
                    validate_fes_request(sampling_config, system)
                    preview = plan_preview(sampling_config, system)
                    if preview["dimension"] == 1:
                        # Report the harmonic thermal width alongside actual
                        # center spacing; nominal spring coefficients alone
                        # obscure when Gaussian UI windows are too broad.
                        locations = np.asarray(
                            [w["center_au"][0] for w in preview["windows"]], float
                        )
                        for window in preview["windows"]:
                            spring = float(window["stiffness_au"][0][0])
                            gaps = np.abs(locations - window["center_au"][0])
                            gaps = gaps[gaps > 1e-12]
                            window["thermal_sigma_xi"] = float(
                                np.sqrt(1.0 / (config["beta"] * spring))
                            )
                            window["nearest_center_spacing_xi"] = (
                                float(gaps.min()) if len(gaps) else None
                            )
                    result["umbrella_preparation"] = {
                        "dimension": preview["dimension"],
                        "cv_units": preview["cv_units"],
                        "windows": len(preview["windows"]),
                        "window_plan": preview["windows"],
                    }
    elif config["task"] == "drpmd":
        result["ready"] = not (
            config["collision"].get("bmax_unvalidated", False)
            or config["outcomes"].get("classifier_unvalidated", False)
        )
        if not result["ready"]:
            result["diagnostics"].append(
                "Pilot simulations are available; production estimation requires a converged bmax and an explicit validated outcome classifier"
            )
    elif config["task"] == "surface_rate" or (config["task"] == "fes" and fes_method(config) == "umbrella"):
        from .workflows.fes import resolve_umbrellas

        if config.get("surface", {}).get("free_energy"):
            from .workflows.surface_source import source_context
            context = source_context(config, system, verify=True)
            result["source_preparation"] = context["paths"]
        else:
            cv, specs, centers, stiffness = resolve_umbrellas(config, system)
            result["umbrella_preparation"] = {
                "dimension": cv.dimension,
                "windows": len(centers),
                "cv_units": cv.units,
                "centers_au": centers,
                "stiffness_au": stiffness,
            }
            from .workflows.umbrella_plan import plan_preview

            result["umbrella_preparation"]["window_plan"] = plan_preview(config, system)["windows"]
        if config["task"] == "surface_rate":
            from .workflows.surface import surface_definition

            definition = surface_definition(config, system)
            result["surface_preparation"] = {
                key: value for key, value in definition.items() if key != "coordinate"
            }
    elif config["task"] == "fes" and fes_method(config) == "metadynamics":
        from .workflows.metadynamics import validate_metadynamics

        validate_metadynamics(config, system)
        result["metadynamics_preparation"] = config["metadynamics"]
    elif config["task"] == "fes" and fes_method(config) == "opes":
        from .workflows.opes import validate_opes

        validate_opes(config, system)
        result["opes_preparation"] = config["opes"]
    elif config["task"] == "fes" and fes_method(config) in ("abf", "eabf", "steered", "ves"):
        from .workflows.fes_dispatch import validate_fes

        validate_fes(config, system)
        method = fes_method(config)
        result[method + "_preparation"] = config[method]
    elif config["task"] == "network":
        from .workflows.network import prepare_network

        result["network_preparation"] = prepare_network(config, system, numerical=numerical)
        result["ready"] = result["network_preparation"].get("ready", False)
        result["propagation_plan"] = propagation_backend_plan(config, system)
        _append_backend_warnings(config, result["propagation_plan"])
        return result
    elif config["task"] == "spectroscopy":
        from .workflows.spectroscopy import prepare_spectroscopy

        result["spectroscopy_preparation"] = prepare_spectroscopy(config, system)
    elif config["task"] == "pimd" and "path_integral" in config:
        from .workflows.path_integral import resolve_path_integral

        method, options = resolve_path_integral(config, system)
        result["path_integral_preparation"] = {
            "method": method, "equilibrium_only": True,
            "ensemble": "npt" if options["barostat"] is not None else "fixed_cell",
            "trajectory_storage": options["storage"],
            "sampling_thermostat": options["thermostat"],
            "replicas": options["replicas"],
            "temperature_interpretation": "auxiliary_only_projection_sets_distribution" if method == "pigs" else "physical_temperature",
            "observable": "projected_midpoint_potential" if method == "pigs" else
                "fermionic_sign_reweighted_thermodynamics" if method == "fermionic" else
                "even_bead_potential_alpha_zero_only" if method == "suzuki_chin" else "symmetric_bead_potential",
        }
        if method in ("bosonic", "fermionic") and options["barostat"] is not None:
            result["path_integral_preparation"]["barostat_coordinate_scaling"] = "all_beads"
    elif config["task"] in ("pimd", "diffusion"):
        from .workflows.materials import resolve_materials

        materials_options = resolve_materials(config, system)
        native_pimd = config.get("sampling", {}).get("engine", "legacy") == "native"
        if native_pimd:
            barostat = copy.deepcopy(config.get("barostat", {"mode": "none"}))
            if barostat.get("mode", "none") == "none":
                barostat = None
            storage = "native_pimd_journal"
        else:
            barostat = materials_options["barostat"]
            storage = materials_options["storage"]
        result["materials_preparation"] = {
            "cell_bohr": system.to_dict().get("cell"),
            "pbc": list(getattr(system, "pbc", (False, False, False))),
            "ensemble": "npt" if barostat is not None else "fixed_cell",
            "barostat": barostat,
            "fixed_atoms": list(system.fixed_atoms),
            "trajectory_storage": storage,
            "sampling_thermostat": config["sampling"]["thermostat"],
            "production_method": ("centroid_ef_rpmd" if "electronic_friction" in config else
                                  "unthermostatted_rpmd" if config["task"] == "diffusion" else "pimd_sampling"),
            "sampling": config["sampling"],
            "analysis": config.get("diffusion", {}),
        }
        if native_pimd:
            result["materials_preparation"]["engine"] = "native"
        elif config.get("sampling", {}).get("engine") == "torch":
            result["materials_preparation"].update(
                engine="torch", device=materials_options["device"], dtype="float64",
                normal_mode_layout=materials_options["normal_mode_layout"],
                sampling_integrator="bcocb", batch_replicas=materials_options["replicas"],
                checkpoint="atomic_tensor_batch_with_owned_rng_states",
            )
    elif config["task"] == "site_rate":
        from .workflows.site_rate import validate_site_rate

        result["site_rate_preparation"] = validate_site_rate(config, system)
    if numerical:
        from .potentials import make_potential

        # The provider itself validates array shape, status and finite results.
        provider = make_potential(config["potential"], system)
        try:
            x = system.geometry[None, :, :]
            if (config["task"] == "spectroscopy"
                    or (getattr(provider, "metadata", {}) or {}).get("ring_polymer_coupled")):
                x = np.repeat(x, config["beads"], axis=0)
            from .potentials import evaluate_checked, validate_result

            specialized_check = None
            if config["task"] in ("pimd", "diffusion") and "path_integral" not in config:
                from .workflows.pimd import select_engine

                specialized_check = select_engine(config).check(config, system, provider)
            if specialized_check is not None:
                result["pes_check"] = specialized_check
            elif config["task"] == "pimd" and "path_integral" in config:
                from .workflows.path_integral import _action

                action = _action(config, system, provider)
                q = np.repeat(system.geometry[None], config["beads"], axis=0)
                checked = evaluate_checked(action, q)
                result["pes_check"] = {
                    "energy_hartree": checked.energy.tolist(),
                    "status": checked.status.tolist(),
                    "sampling_action_potential": True,
                    "auxiliary_beta_au": action.beta,
                    "force_max_au": float(np.max(np.abs(checked.forces))),
                }
                if config["path_integral"].get("method", "suzuki_chin") == "suzuki_chin":
                    volume = None if system.cell is None else float(abs(np.linalg.det(system.cell)))
                    result["pes_check"]["initial_sc_estimators"] = action.estimators(
                        q, volume=volume,
                        finite_difference_step=config["path_integral"].get("beta_fd_step", 1e-3),
                        require_pressure=options["barostat"] is not None)
                    if options["barostat"] is not None:
                        from .barostat import BZPBarostat
                        from .dynamics import State
                        piston = BZPBarostat(action, system, action.beta, config["beads"], **options["barostat"])
                        result["pes_check"]["initial_action_virial_hartree"] = checked.virial.tolist()
                        result["pes_check"]["initial_configurational_pressure_au"] = piston.pressure(State(q, np.zeros_like(q)))
                elif config["path_integral"].get("method") in ("bosonic", "fermionic"):
                    volume = float(abs(np.linalg.det(system.cell))) if all(system.pbc) else None
                    result["pes_check"]["initial_exchange_estimators"] = action.estimators(
                        q, volume=volume, require_pressure=options["barostat"] is not None)
                    if action.periodic_mode == "full_winding":
                        result["pes_check"]["winding_kernel_plans"] = copy.deepcopy(
                            action.metadata["winding_kernels"])
                    if options["barostat"] is not None:
                        from .workflows.path_integral import _make_barostat
                        from .dynamics import State
                        piston = _make_barostat(action, system, config["beads"], options["barostat"])
                        result["pes_check"]["initial_configurational_pressure_au"] = piston.pressure(
                            State(q, np.zeros_like(q)))
            elif "nonadiabatic" in config:
                evaluated = provider.evaluate(x)
                from .nonadiabatic import validate_diabatic_result
                from .potentials.diabatic import check_diabatic_potential
                checked = validate_diabatic_result(evaluated, x.shape, config["nonadiabatic"]["nstates"])
                result["pes_check"] = {
                    "matrix_hartree": checked.matrix.real.tolist(),
                    **check_diabatic_potential(provider, x, config["nonadiabatic"]["nstates"]),
                }
                if checked.matrix.dtype.kind == "c":
                    result["pes_check"]["matrix_imag_hartree"] = checked.matrix.imag.tolist()
            else:
                evaluated = provider.evaluate(x)
                validate_result(evaluated, x.shape)
                result["pes_check"] = {
                    "energy_hartree": evaluated.energy.tolist(),
                    "status": evaluated.status.tolist(),
                }
                if config["task"] == "spectroscopy":
                    result["spectroscopy_preparation"] = prepare_spectroscopy(
                        config, system, potential=provider
                    )
                if config["task"] == "pimd" and "path_integral" not in config and materials_options["barostat"] is not None:
                    from .barostat import BZPBarostat
                    from .dynamics import State
                    piston = BZPBarostat(provider, system, config["beta"], config["beads"],
                                        **materials_options["barostat"])
                    q = np.repeat(x, config["beads"], axis=0)
                    result["pes_check"]["initial_configurational_pressure_au"] = piston.pressure(State(q, np.zeros_like(q)))
            backend_plan = propagation_backend_plan(config, system, potential=provider)
        finally:
            close = getattr(provider, "close", None)
            if callable(close):
                close()
        if "electronic_friction" in config:
            from .electronic_friction import make_electronic_friction, validate_friction_result
            friction = make_electronic_friction(config, system)
            try:
                checked = validate_friction_result(friction.evaluate(x), x.shape)
                result["friction_check"] = {"tensor_atomic_units": checked.tensor.tolist(),
                                            "method": "markovian_centroid_ef_rpmd"}
            finally:
                close = getattr(friction, "close", None)
                if callable(close):
                    close()
    if not numerical:
        backend_plan = propagation_backend_plan(config, system)
    result["propagation_plan"] = backend_plan
    _append_backend_warnings(config, backend_plan)
    return result
