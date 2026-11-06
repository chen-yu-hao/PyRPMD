"""Input-file access to the existing explicitly clocked PLUMED bridge."""
from __future__ import annotations

from contextlib import chdir
from collections.abc import Mapping
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
from ..warm_start import initial_ring

from ..checkpoint import atomic_savez, load_checkpoint
from ..dynamics import Dynamics, State
from ..plumed import PlumedBias, file_restart_plan
from ..potentials import make_potential
from ..propagation import loop_options, scientific_config
from ..io import fingerprint, fingerprint_payload, referenced_files, write_json


FIELDS = {"file", "input", "kernel", "charges", "restart"}


def resolve_plumed_job(data, *, base_directory=None, input_path=None):
    from ..input import JobSpec, InputError, _path
    base = Path(base_directory or ".").resolve()
    if not isinstance(data, Mapping) or data.get("task") != "plumed":
        raise InputError("resolve_plumed_job requires task='plumed'")
    raw = deepcopy(dict(data))
    table = raw.pop("plumed", None)
    if not isinstance(table, dict) or set(table) - FIELDS:
        raise InputError("[plumed] must contain file or input, optional kernel, charges and restart")
    if ("file" in table) == ("input" in table):
        raise InputError("choose exactly one of plumed.file or plumed.input")
    forbidden = {
        "path_integral", "active_learning", "barostat", "electronic_friction",
        "reaction", "collision", "outcomes", "cvs", "umbrella", "analysis", "rate",
        "recrossing", "network", "metadynamics", "opes", "adaptive_umbrella", "pcet",
        "diffusion", "site_rate", "surface", "nonadiabatic", "replica_exchange", "initial_ensemble",
    }
    if forbidden.intersection(raw):
        raise InputError("PLUMED workflow does not support sections: "
                         + ", ".join(sorted(forbidden.intersection(raw))))
    if "file" in table:
        if not isinstance(table["file"], str) or not table["file"]:
            raise InputError("plumed.file must be a nonempty file path")
        table["file"] = _path(base, table["file"])
        if not Path(table["file"]).is_file():
            raise InputError("PLUMED input file is missing")
        text = Path(table["file"]).read_text(encoding="utf-8")
    else:
        text = table["input"]
    if not isinstance(text, str) or not text.strip() or "\x00" in text:
        raise InputError("PLUMED input must be nonempty text")
    if "restart" in table:
        if table["restart"] != "files":
            raise InputError("plumed.restart must be 'files' when specified")
        try:
            file_restart_plan(text)
        except ValueError as error:
            raise InputError(str(error)) from error
    if "kernel" in table:
        if not isinstance(table["kernel"], str) or not table["kernel"]:
            raise InputError("plumed.kernel must be a nonempty library path")
        table["kernel"] = _path(base, table["kernel"])
        if not Path(table["kernel"]).is_file():
            raise InputError("PLUMED kernel library is missing")
    output = raw.setdefault("output", {})
    if not isinstance(output, Mapping):
        raise InputError("output must be a table")
    unsupported_output = {"trajectory_stride", "trajectory_storage", "chunk_frames", "diagnostics", "store_configurations"}
    if unsupported_output.intersection(output):
        raise InputError("PLUMED workflow does not support output controls: "
                         + ", ".join(sorted(unsupported_output.intersection(output))))
    output.setdefault("directory", "runs/plumed")
    raw["task"] = "pimd"
    resolved, system = JobSpec(raw, base, input_path).resolve()
    from . import validate_bond_constraint_workflow
    try:
        validate_bond_constraint_workflow(system, "plumed")
    except ValueError as error:
        raise InputError(str(error)) from error
    parallel = resolved["parallel"]
    if (parallel["backend"] != "serial" or parallel["mode"] != "serial"
            or parallel["trajectory_workers"] != 1 or parallel["bead_workers"] != 1
            or parallel.get("pes_ranks", 1) != 1):
        raise InputError("PLUMED input workflow requires serial execution")
    if (resolved["sampling"].get("replicas", 1) != 1 or system.fixed_atoms
            or (any(system.pbc) and not all(system.pbc))):
        raise InputError("PLUMED workflow requires one all-mobile trajectory and all/no PBC")
    if resolved["sampling"].get("engine", "legacy") != "legacy" or resolved.get("dynamics", {}).get("loop_backend") == "native":
        raise InputError("PLUMED requires Python step orchestration")
    if resolved["sampling"]["dt_au"] != resolved["dt_au"]:
        raise InputError("PLUMED requires sampling.timestep equal to dynamics.timestep for its fixed update clock")
    if resolved["sampling"]["thermostat"] not in ("andersen", "pile-l"):
        raise InputError("PLUMED workflow supports only Andersen or PILE-L sampling")
    if {"parents", "thermostat_params"}.intersection(data.get("sampling", {})):
        raise InputError("PLUMED workflow does not use parents or NHC thermostat parameters")
    if "charges" in table:
        charges = np.asarray(table["charges"])
        if charges.dtype.kind not in "fiu" or charges.shape != (system.natoms,) or not np.isfinite(charges).all():
            raise InputError("plumed.charges must be a finite real vector of length N")
    resolved.update(task="plumed", plumed=table, _input_path=input_path)
    if table.get("restart") == "files":
        resolved["_warnings"].append("PLUMED file restart uses a restricted fixed-hill profile, round-trip output precision and FLUSH STRIDE=1. Relative FILE paths use RUN/plumed.")
    else:
        resolved["_warnings"].append("PLUMED owns its history/files; automatic kernel restart is not supported. Relative PLUMED FILE paths use RUN/plumed.")
    return resolved, system


def run_plumed(config, system, output):
    from . import validate_bond_constraint_workflow
    validate_bond_constraint_workflow(system, "plumed")
    if config.get("plumed", {}).get("restart") == "files":
        return _run_file_restart(config, system, output)
    if config.get("resume"):
        raise ValueError("PLUMED kernel history cannot be restored from an MD-only checkpoint; automatic resume is unsupported")
    output = Path(output).resolve()
    work = output / "plumed"
    work.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(parents=True, exist_ok=True)
    table, sampling = config["plumed"], config["sampling"]
    script = Path(table["file"]).read_text(encoding="utf-8") if "file" in table else table["input"]
    provider = make_potential(config["potential"], system)
    steps, positions, biases = [], [], []
    snapshot = output / "checkpoints" / "plumed-md-state.npz"
    try:
        dynamics = Dynamics(provider, system.masses, config["beta"], config["beads"], config["dt_au"],
            seed=config.get("seed", 0), native=config.get("native", config.get("dynamics", {}).get("native", True)),
            threads=config.get("parallel", {}).get("threads_per_rank", 1),
            **loop_options(config))
        state = State(initial_ring(config, system), dynamics.sample_momenta())
        total = sampling["equilibration_steps"] + sampling["production_steps"]
        def save():
            atomic_savez(snapshot, kind="PyRPMD.plumed-md-state.v1", q=state.q, p=state.p,
                step=state.step, time=state.time, rng=json.dumps(dynamics.rng.bit_generator.state),
                kernel_state_saved=False, sample_steps=np.asarray(steps, dtype=np.int64),
                positions=np.asarray(positions).reshape(-1, config["beads"], system.natoms, 3),
                physical_bias=np.asarray(biases))
        save()
        try:
            with chdir(work), PlumedBias(script, system.masses, config["beta"], config["beads"], config["dt_au"],
                    cell=system.cell, pbc=system.pbc, charges=table.get("charges"),
                    kernel_path=table.get("kernel"), log_file=str(output / "logs" / "plumed.log")) as bridge:
                while state.step < total and not bridge.stop_requested:
                    bridge.step(dynamics, state, thermostat=sampling["thermostat"],
                        gamma_centroid=sampling.get("gamma_centroid", 1.0),
                        andersen_interval=sampling["andersen_interval"])
                    production = state.step - sampling["equilibration_steps"]
                    if production > 0 and production % sampling["sample_interval"] == 0:
                        bias = bridge.evaluate(state.q).physical_energy
                        steps.append(state.step)
                        positions.append(state.q.copy())
                        biases.append(bias)
                    if state.step % config["output"]["checkpoint_steps"] == 0:
                        save()
                stopped = bridge.stop_requested
        except BaseException as error:
            write_json(output / "results" / "plumed-failure.json", {
                "task": "plumed", "status": "failed", "last_accepted_step": state.step,
                "error_type": type(error).__name__, "error": str(error),
                "md_snapshot": str(snapshot), "kernel_state_saved": False,
                "restart_supported": False,
            })
            raise
        finally:
            save()
        return {"task": "plumed", "status": "plumed_stop" if stopped else "complete",
            "steps": state.step, "samples": len(steps), "restart_supported": False,
            "md_snapshot": str(snapshot), "plumed_directory": str(work),
            "biased_dynamics": True, "physical_rate": None}
    finally:
        provider.close()


_FILE_CHECKPOINT = "PyRPMD.plumed-file-state.v1"


def _file_path(work, name):
    """Reject symlinks before either checkpoint reads or restoration writes."""
    path = work / name
    current = path
    while current != work.parent:
        if current.is_symlink():
            raise ValueError("PLUMED restart files and their parents must not be symlinks")
        current = current.parent
    if path.exists() and not path.is_file():
        raise ValueError("PLUMED restart output path is not a regular file")
    return path


def _restore_files(work, contents):
    # Validate every destination before modifying any file. The atomic NPZ is
    # the transaction's source of truth: an interrupted multi-file restore can
    # be replayed safely, and never causes the MD checkpoint to advance.
    paths = {name: _file_path(work, name) for name in contents}
    for name, data in contents.items():
        path = paths[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        if data is None:
            path.unlink(missing_ok=True)
            continue
        descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def _truncate_clock_replay(work, contents):
    """Remove the restored-endpoint PRINT row without replacing open inodes."""
    for name, prefix in contents.items():
        path = _file_path(work, name)
        if prefix is None or not path.is_file() or not path.read_bytes().startswith(prefix):
            raise ValueError("PLUMED RESTART did not append to the verified checkpoint file prefix")
    for name, prefix in contents.items():
        # FLUSH STRIDE=1 has flushed the replayed update. Keeping the inode lets
        # PLUMED's existing append-mode handles continue writing at the new EOF.
        with _file_path(work, name).open("r+b") as stream:
            stream.truncate(len(prefix))
            stream.flush()
            os.fsync(stream.fileno())


def _restart_identity(config, system, plan):
    return fingerprint_payload({
        "profile": "PyRPMD.plumed-fixed-hills.v1",
        "physics": fingerprint(scientific_config(config), system),
        "input_files": referenced_files(config),
        "input_lines": plan.input_lines,
        "output_files": plan.files,
    })


def _read_file_checkpoint(snapshot, identity, shape, config, plan):
    """Validate the complete payload before restoring RNG, files or a kernel."""
    with load_checkpoint(snapshot, expected_kind=_FILE_CHECKPOINT, require_integrity=True) as saved:
        arrays = {name: saved[name] for name in saved.files}
    required = {"q", "p", "step", "time", "rng", "identity", "sample_steps", "positions",
                "physical_bias", "file_manifest", "kernel_state_saved", "plumed_file_state_saved",
                "api_version"} | {f"file_{index}" for index in range(len(plan.files))}
    if set(arrays) != required:
        raise ValueError("PLUMED checkpoint field inventory is invalid")
    if arrays["identity"].shape != () or str(arrays["identity"].item()) != identity:
        raise ValueError("PLUMED checkpoint scientific identity or input files changed")
    for name in ("q", "p"):
        array = arrays[name]
        if array.shape != shape or array.dtype != np.float64 or not np.isfinite(array).all():
            raise ValueError("invalid PLUMED checkpoint coordinates/momenta")
    step = arrays["step"]
    total = config["sampling"]["equilibration_steps"] + config["sampling"]["production_steps"]
    if step.shape != () or step.dtype.kind not in "iu" or not 0 <= int(step) <= total:
        raise ValueError("invalid PLUMED checkpoint step")
    step = int(step)
    time = arrays["time"]
    if (time.shape != () or time.dtype.kind not in "fiu" or not math.isfinite(float(time))
            or not math.isclose(float(time), step * config["dt_au"], rel_tol=1e-10, abs_tol=1e-10)):
        raise ValueError("PLUMED checkpoint time does not match its step")
    sampling = config["sampling"]
    expected_steps = np.arange(sampling["equilibration_steps"] + sampling["sample_interval"],
                               step + 1, sampling["sample_interval"], dtype=np.int64)
    actual_steps = arrays["sample_steps"]
    if actual_steps.dtype.kind not in "iu" or not np.array_equal(actual_steps, expected_steps):
        raise ValueError("PLUMED checkpoint sample clocks are inconsistent")
    for name, expected_shape in (("positions", (len(expected_steps), *shape)),
                                 ("physical_bias", (len(expected_steps),))):
        array = arrays[name]
        if array.shape != expected_shape or array.dtype != np.float64 or not np.isfinite(array).all():
            raise ValueError("invalid PLUMED checkpoint sample arrays")
    if len(expected_steps) and expected_steps[-1] == step and not np.array_equal(arrays["positions"][-1], arrays["q"]):
        raise ValueError("PLUMED checkpoint final sample does not match its MD state")
    if (arrays["kernel_state_saved"].shape != () or arrays["kernel_state_saved"].dtype.kind != "b"
            or bool(arrays["kernel_state_saved"]) or arrays["plumed_file_state_saved"].shape != ()
            or arrays["plumed_file_state_saved"].dtype.kind != "b" or not bool(arrays["plumed_file_state_saved"])):
        raise ValueError("invalid PLUMED checkpoint file-state markers")
    api = arrays["api_version"]
    if api.shape != () or api.dtype.kind not in "iu" or int(api) < 0 or (step and int(api) < 8):
        raise ValueError("invalid PLUMED checkpoint API version")
    try:
        if arrays["rng"].shape != () or arrays["file_manifest"].shape != ():
            raise ValueError("expected JSON scalar")
        rng = json.loads(str(arrays["rng"].item()))
        probe = np.random.default_rng()
        probe.bit_generator.state = rng
        manifest = json.loads(str(arrays["file_manifest"].item()))
    except (ValueError, TypeError, KeyError) as error:
        raise ValueError("invalid PLUMED checkpoint RNG/file manifest") from error
    expected_manifest = [{"path": name, "array": f"file_{index}"}
                         for index, name in enumerate(plan.files)]
    if not isinstance(manifest, list) or len(manifest) != len(expected_manifest):
        raise ValueError("PLUMED checkpoint file inventory changed")
    contents = {}
    for actual, expected in zip(manifest, expected_manifest):
        if (not isinstance(actual, dict) or set(actual) != {"path", "array", "present"}
                or any(actual[key] != value for key, value in expected.items())
                or type(actual["present"]) is not bool):
            raise ValueError("PLUMED checkpoint file inventory changed")
        blob = arrays[expected["array"]]
        if (blob.dtype != np.uint8 or blob.ndim != 1
                or (not actual["present"] and (blob.size or step))):
            raise ValueError("invalid PLUMED checkpoint file bytes")
        contents[expected["path"]] = blob.tobytes() if actual["present"] else None
    return arrays, rng, contents


def _run_file_restart(config, system, output):
    table, sampling = config["plumed"], config["sampling"]
    script = Path(table["file"]).read_text(encoding="utf-8") if "file" in table else table["input"]
    plan = file_restart_plan(script)
    identity = _restart_identity(config, system, plan)
    output = Path(output).resolve()
    work = output / "plumed"
    snapshot = output / "checkpoints" / "plumed-md-state.npz"
    shape = (config["beads"], system.natoms, 3)
    resume = bool(config.get("resume"))
    restored = _read_file_checkpoint(snapshot, identity, shape, config, plan) if resume else None
    # Paths are checked before kernel construction, including a dangling
    # symlink. Fresh direct-API runs must not replace existing histories.
    for name in plan.files:
        path = _file_path(work, name)
        if not resume and path.exists():
            raise ValueError("PLUMED history already exists; choose a new output or resume")
    work.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(parents=True, exist_ok=True)
    for name in plan.files:
        _file_path(work, name).parent.mkdir(parents=True, exist_ok=True)
    provider = make_potential(config["potential"], system)
    try:
        dynamics = Dynamics(provider, system.masses, config["beta"], config["beads"], config["dt_au"],
            seed=config.get("seed", 0), native=config.get("native", config.get("dynamics", {}).get("native", True)),
            threads=config.get("parallel", {}).get("threads_per_rank", 1), **loop_options(config))
        if restored is None:
            state = State(initial_ring(config, system), dynamics.sample_momenta())
            steps, positions, biases, api_version = [], [], [], 0
        else:
            arrays, rng, contents = restored
            state = State(arrays["q"], arrays["p"], step=int(arrays["step"]), time=float(arrays["time"]))
            dynamics.rng.bit_generator.state = rng
            steps = arrays["sample_steps"].tolist()
            positions = list(arrays["positions"])
            biases = arrays["physical_bias"].tolist()
            api_version = int(arrays["api_version"])
            _restore_files(work, contents)
        checkpoint_step = state.step

        def save(*, initial=False):
            nonlocal checkpoint_step
            manifest, blobs = [], {}
            for index, name in enumerate(plan.files):
                path = _file_path(work, name)
                present = path.is_file() and not initial
                if state.step and not present:
                    raise ValueError(f"PLUMED did not create declared restart file {name!r}")
                manifest.append({"path": name, "array": f"file_{index}", "present": present})
                blobs[f"file_{index}"] = np.frombuffer(path.read_bytes() if present else b"", dtype=np.uint8)
            atomic_savez(snapshot, kind=_FILE_CHECKPOINT, q=state.q, p=state.p,
                step=state.step, time=state.time, rng=json.dumps(dynamics.rng.bit_generator.state),
                identity=identity, api_version=api_version, kernel_state_saved=False, plumed_file_state_saved=True,
                file_manifest=json.dumps(manifest, sort_keys=True), **blobs,
                sample_steps=np.asarray(steps, dtype=np.int64),
                positions=np.asarray(positions, dtype=np.float64).reshape(-1, *shape),
                physical_bias=np.asarray(biases, dtype=np.float64))
            checkpoint_step = state.step

        if not resume:
            save(initial=True)
        total = sampling["equilibration_steps"] + sampling["production_steps"]
        try:
            with chdir(work), PlumedBias(plan.input_lines, system.masses, config["beta"], config["beads"], config["dt_au"],
                    cell=system.cell, pbc=system.pbc, charges=table.get("charges"), kernel_path=table.get("kernel"),
                    log_file=str(output / "logs" / "plumed.log"), restart=resume and state.step > 0,
                    start_step=state.step) as bridge:
                if api_version and api_version != bridge.api_version:
                    raise ValueError("PLUMED API version changed since checkpoint")
                api_version = bridge.api_version
                if resume and state.step > 0:
                    bridge._prime_file_restart(state.q)
                    _truncate_clock_replay(work, contents)
                while state.step < total and not bridge.stop_requested:
                    bridge.step(dynamics, state, thermostat=sampling["thermostat"],
                        gamma_centroid=sampling.get("gamma_centroid", 1.0),
                        andersen_interval=sampling["andersen_interval"])
                    production = state.step - sampling["equilibration_steps"]
                    if production > 0 and production % sampling["sample_interval"] == 0:
                        bias = bridge.evaluate(state.q).physical_energy
                        steps.append(state.step)
                        positions.append(state.q.copy())
                        biases.append(bias)
                    if state.step % config["output"]["checkpoint_steps"] == 0:
                        save()
                stopped = bridge.stop_requested
                if state.step:
                    save()
        except BaseException as error:
            # Never overwrite a consistent checkpoint with possibly partial
            # kernel output from a failed update/force/diagnostic/finalize.
            write_json(output / "results" / "plumed-failure.json", {
                "task": "plumed", "status": "failed", "last_accepted_step": state.step,
                "checkpoint_step": checkpoint_step, "error_type": type(error).__name__, "error": str(error),
                "md_snapshot": str(snapshot), "kernel_state_saved": False,
                "plumed_file_state_saved": True, "restart_supported": True,
            })
            raise
        return {"task": "plumed", "status": "plumed_stop" if stopped else "complete",
            "steps": state.step, "samples": len(steps), "restart_supported": True,
            "restart_profile": "fixed-hills-v1", "checkpoint_step": checkpoint_step,
            "md_snapshot": str(snapshot), "plumed_directory": str(work),
            "plumed_api_version": api_version, "biased_dynamics": True, "physical_rate": None}
    finally:
        provider.close()
