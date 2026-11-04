"""Inspect PIMD restart records and migrate legacy trajectory storage.

Storage migration preserves the selected engine, all scientific fingerprints,
phase clocks and RNG state. It is not a conversion between NumPy, C++ and Torch
propagators: their random streams and sampling maps are different. Native and
Torch records can be inspected without importing their execution runtimes.

Run ``python -m pyrpmd.pimd_checkpoint --help`` for the offline command line.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import sys

import numpy as np

from .checkpoint import CheckpointError, load_checkpoint

_LEGACY_KINDS = {"PyRPMD.materials.v1": "memory",
                 "PyRPMD.materials-stream.v1": "stream"}
_TORCH_KIND = "PyRPMD.materials-torch.v1"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def _scalar(data, name, kinds):
    if name not in data:
        raise CheckpointError(f"PIMD checkpoint is missing {name!r}")
    value = np.asarray(data[name])
    if value.shape != () or value.dtype.kind not in kinds:
        raise CheckpointError(f"Invalid PIMD checkpoint scalar {name!r}")
    return value.item()


def _invalid_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _json(data, name, expected=dict):
    try:
        value = json.loads(_scalar(data, name, "US"), parse_constant=_invalid_constant)
    except (ValueError, TypeError) as error:
        raise CheckpointError(f"Invalid PIMD checkpoint JSON {name!r}") from error
    if not isinstance(value, expected):
        raise CheckpointError(f"Invalid PIMD checkpoint JSON {name!r}")
    return value


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise CheckpointError(f"Invalid PIMD checkpoint {name}")
    return value


def _fingerprint(value):
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise CheckpointError("Invalid PIMD checkpoint fingerprint")
    return value


def _state_arrays(data, dimensions):
    shape = None
    for name in ("q", "p"):
        if name not in data:
            raise CheckpointError(f"PIMD checkpoint is missing {name!r}")
        array = data[name]
        if (array.dtype.kind != "f" or array.dtype.itemsize != 8
                or array.ndim != dimensions or array.shape[-1] != 3
                or any(n < 1 for n in array.shape) or not np.isfinite(array).all()
                or (shape is not None and array.shape != shape)):
            raise CheckpointError(f"Invalid PIMD checkpoint {name} state")
        shape = array.shape
    return shape


def _legacy_rng(state):
    """Validate the generator owned by the legacy workflow without drawing."""
    if (not isinstance(state, dict) or set(state) != {"bit_generator", "state", "has_uint32", "uinteger"}
            or state.get("bit_generator") != "PCG64"
            or not isinstance(state.get("state"), dict)
            or set(state["state"]) != {"state", "inc"}):
        raise CheckpointError("Invalid legacy PIMD PCG64 RNG state")
    integers = {**state["state"], "has_uint32": state["has_uint32"], "uinteger": state["uinteger"]}
    limits = {"state": 2**128, "inc": 2**128, "has_uint32": 2, "uinteger": 2**32}
    if (any(type(value) is not int or not 0 <= value < limits[name]
            for name, value in integers.items()) or state["state"]["inc"] % 2 != 1):
        raise CheckpointError("Invalid legacy PIMD PCG64 RNG state")
    try:
        # A separate, explicitly seeded generator cannot consume simulation or
        # global RNG state. The assignment also applies NumPy's format checks.
        generator = np.random.PCG64(0)
        generator.state = state
    except (TypeError, ValueError, OverflowError) as error:
        raise CheckpointError("Invalid legacy PIMD PCG64 RNG state") from error
    return "PCG64"


def _legacy_info(data, kind):
    from .trajectory import _descriptor

    shape = _state_arrays(data, 3)
    step = _integer(_scalar(data, "step", "iu"), "step")
    if _integer(_scalar(data, "elapsed", "iu"), "elapsed") != step:
        raise CheckpointError("Legacy PIMD checkpoint phase clocks differ")
    time = _scalar(data, "time", "fiu")
    if not np.isfinite(time) or time < 0:
        raise CheckpointError("Invalid legacy PIMD checkpoint time")
    algorithm = _legacy_rng(_json(data, "rng"))
    _json(data, "propagation")
    storage = _LEGACY_KINDS[kind]
    if storage == "stream":
        frames = _descriptor(_json(data, "trajectory"))["frames"]
    else:
        if "sample_steps" not in data:
            raise CheckpointError("Legacy PIMD checkpoint is missing sample_steps")
        steps = data["sample_steps"]
        if (steps.ndim != 1 or steps.dtype.kind not in "iu" or np.any(steps < 0)
                or np.any(steps > step) or (len(steps) > 1 and np.any(steps[1:] <= steps[:-1]))):
            raise CheckpointError("Invalid legacy PIMD sample clock")
        frames, atoms = len(steps), shape[-2]
        for name, tail in (("positions", (atoms, 3)), ("velocities", (atoms, 3)),
                           ("radii_gyration_squared", (atoms,))):
            if name not in data:
                raise CheckpointError(f"Legacy PIMD checkpoint is missing {name}")
            value = data[name]
            if (value.dtype.kind != "f" or value.dtype.itemsize != 8
                    or value.shape != (frames, *tail) or not np.isfinite(value).all()):
                raise CheckpointError(f"Invalid legacy PIMD trajectory {name}")
        if "observables" not in data:
            raise CheckpointError("Legacy PIMD checkpoint is missing observables")
        values = data["observables"]
        if (values.dtype.kind != "f" or values.dtype.itemsize != 8 or values.ndim != 2
                or values.shape[0] != frames or values.shape[1] not in (2, 5)
                or not np.isfinite(values).all()):
            raise CheckpointError("Invalid legacy PIMD observables")
    return {"engine": "legacy", "storage": storage, "step": step,
            "time_au": float(time), "frames": frames,
            "fingerprint": _fingerprint(_scalar(data, "fingerprint", "US")),
            "rng": {"algorithm": algorithm, "generators": 1},
            "external_trajectory_required": storage == "stream"}


def _native_info(data):
    from .native_ensemble import _ARRAYS, _FORMAT, _state_digest

    metadata = _json(data, "metadata")
    expected = {"step", "force_evaluations", "rng", "format", "fingerprint", "digest"}
    if (set(data.files) != set(_ARRAYS) | {"metadata"} or set(metadata) != expected
            or metadata.get("format") != _FORMAT):
        raise CheckpointError("Invalid native PIMD checkpoint fields")
    shape = _state_arrays(data, 3)
    arrays = {name: data[name] for name in _ARRAYS}
    shapes = {"energy": (shape[0],), "force": shape, "virial": (shape[0], 3, 3),
              "cell": (3, 3), "cell_p": (3, 3)}
    for name, array in arrays.items():
        if (array.dtype.kind != "f" or array.dtype.itemsize != 8 or not np.isfinite(array).all()
                or (name in shapes and array.shape != shapes[name])):
            raise CheckpointError(f"Invalid native PIMD checkpoint {name}")
    _integer(metadata["force_evaluations"], "force_evaluations")
    if not isinstance(metadata["rng"], str) or not metadata["rng"]:
        raise CheckpointError("Invalid native PIMD serialized RNG")
    if metadata["digest"] != _state_digest({**metadata, **arrays}):
        raise CheckpointError("Native PIMD state digest differs")
    return {"engine": "native", "storage": "native-state",
            "step": _integer(metadata["step"], "step"),
            "fingerprint": _fingerprint(metadata["fingerprint"]),
            "rng": {"algorithm": "native-cxx17", "generators": 1},
            "state_digest_verified": True,
            # A workflow run additionally needs its journal and sample chunks;
            # a standalone NativeEnsemble uses only this state and its config.
            "external_trajectory_required": "for native PIMD workflow replay"}


def _torch_info(data):
    from .trajectory import _descriptor
    from .workflows.torch_materials import _FIELDS, _validate_runtime

    if set(data.files) != _FIELDS:
        raise CheckpointError("Invalid Torch PIMD checkpoint fields")
    shape = _state_arrays(data, 4)
    step = _integer(_scalar(data, "step", "iu"), "step")
    time = _scalar(data, "time", "fiu")
    if not np.isfinite(time) or time < 0:
        raise CheckpointError("Invalid Torch PIMD checkpoint time")
    _scalar(data, "production_started", "b")
    _json(data, "sources")
    runtime = _json(data, "runtime")
    if runtime.get("normal_mode_layout") not in ("rfft", "rpmdrate"):
        raise CheckpointError("Invalid Torch PIMD normal-mode layout")
    runtime = _validate_runtime(runtime, {"device": runtime.get("device"),
                                         "normal_mode_layout": runtime.get("normal_mode_layout")})
    for name in ("rng_sampling", "rng_production"):
        value = data[name]
        if value.dtype != np.uint8 or value.ndim != 1 or not value.size:
            raise CheckpointError(f"Invalid Torch PIMD checkpoint {name}")
    descriptors = [_descriptor(value) for value in _json(data, "trajectories", list)]
    if len(descriptors) != shape[0]:
        raise CheckpointError("Torch PIMD trajectory count differs from batch state")
    return {"engine": "torch", "storage": "stream", "step": step, "time_au": float(time),
            "replicas": shape[0], "frames_per_replica": [value["frames"] for value in descriptors],
            "fingerprint": _fingerprint(_scalar(data, "fingerprint", "US")),
            "rng": {"algorithm": "torch", "generators": 2},
            "runtime": {name: runtime[name] for name in ("torch", "device", "normal_mode_layout")},
            "external_trajectory_required": True}


def inspect_pimd_checkpoint(path):
    """Read local state metadata, without constructing a PES or execution engine.

    All NPZ members are integrity/CRC checked. The report does not certify a
    resume: checking the original run configuration, model files, runtime ABI
    and any external trajectory prefix remains the workflow reader's job.
    Native pre-envelope files additionally have their original state digest
    verified; unchecksummed legacy containers are explicitly identified.
    """
    with load_checkpoint(path) as data:
        kind = data.envelope.get("kind")
        if kind is None:
            if "metadata" in data:
                kind = _json(data, "metadata").get("format")
            elif {"elapsed", "rng", "q", "p"}.issubset(data.files):
                kind = "PyRPMD.materials-stream.v1" if "trajectory" in data else "PyRPMD.materials.v1"
        if not isinstance(kind, str):
            raise CheckpointError(f"Unsupported PIMD checkpoint kind: {kind!r}")
        if kind in _LEGACY_KINDS:
            info = _legacy_info(data, kind)
        elif kind == "pyrpmd.native-ensemble.v1":
            info = _native_info(data)
        elif kind == _TORCH_KIND:
            info = _torch_info(data)
        else:
            raise CheckpointError(f"Unsupported PIMD checkpoint kind: {kind!r}")
        info.update(path=str(Path(path).resolve()), payload_format=kind,
                    container_integrity=data.envelope["integrity"],
                    resume_validated=False, external_trajectory_verified=False,
                    arrays={name: {"shape": list(data[name].shape), "dtype": data[name].dtype.str}
                            for name in data.files})
        return info


def migrate_legacy_checkpoint(config, system, output, replica=0, *, storage):
    """Atomically convert one stopped legacy replica between memory and stream.

    Completed and partial checkpoints are supported. Validation includes the
    scientific identity and every committed frame before anything is published.
    Every non-trajectory payload array is copied without reserializing metadata.
    Source chunks are retained, so a failed replacement leaves the old checkpoint
    usable; unreferenced chunks never advance a trajectory. As in propagation,
    there must be one writer per replica, with no concurrent running simulation.

    The caller's config, saved inputs and run manifest are never changed. A later
    resume follows the storage requested by its configuration. The returned
    detached workflow view describes the new checkpoint without propagating.
    """
    from .workflows import materials

    if storage not in ("memory", "stream"):
        raise ValueError("storage must be memory or stream")
    if (config.get("task") not in ("pimd", "diffusion")
            or config.get("sampling", {}).get("engine", "legacy") != "legacy"):
        raise ValueError("storage migration requires a legacy pimd/diffusion run; it cannot change engines")
    replica = _integer(replica, "replica")
    changed = copy.deepcopy(config)
    changed.setdefault("output", {})["trajectory_storage"] = storage
    options = materials.resolve_materials(changed, system)
    if replica >= options["replicas"]:
        raise ValueError("replica is outside the configured replica count")
    schema = materials._trajectory_schema(system, options)
    checkpoint = materials._checkpoint_path(output, replica)
    with load_checkpoint(checkpoint, require_integrity=True) as data:
        kind = data.envelope.get("kind")
        if not isinstance(kind, str) or kind not in _LEGACY_KINDS:
            raise CheckpointError("Storage migration requires a legacy materials checkpoint")
        _legacy_info(data, kind)
        # Preserve unknown auxiliary payload arrays too; future state must not
        # disappear merely because this storage utility does not interpret it.
        payload = {name: data[name] for name in data.files if name not in schema and name != "trajectory"}
    saved = materials._read_stream(changed, system, output, replica, options, materialize=False)
    if saved["storage"] == storage:
        return saved
    if storage == "stream":
        writer = materials.TrajectoryStream(materials._stream_directory(output, replica),
                                            schema=schema, chunk_frames=options["chunk_frames"])
        writer.extend({name: saved[name] for name in schema})
        payload["trajectory"] = np.asarray(json.dumps(writer.prepare_commit()))
        kind = materials.STREAM_CHECKPOINT_KIND
    else:
        payload.update(materials.materialize_stream(saved["trajectory_directory"], saved["trajectory"]))
        kind = materials.CHECKPOINT_KIND
    # Keep the workflow's established publication/fault-injection boundary.
    materials.atomic_savez(checkpoint, kind=kind, **payload)
    return materials._read_stream(changed, system, output, replica, options, materialize=False)


def _saved_run(directory):
    from .io import fingerprint, referenced_files
    from .propagation import scientific_config
    from .runner import _read_identity, _validate_resume_identity

    path = Path(directory).resolve()
    if path.name == "checkpoints":
        path = path.parent
    config = json.loads((path / "input" / "resolved.json").read_text(encoding="utf-8"))
    system = _read_identity(config, path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    before = list(sys.path)
    try:
        base = config.get("_base_directory")
        if base and base not in sys.path:
            sys.path.insert(0, base)
        _validate_resume_identity(config, system, path, manifest, referenced_files(config),
                                  fingerprint(scientific_config(config), system))
    finally:
        sys.path[:] = before
    return path, config, system


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="read checkpoint metadata without a PES or execution runtime")
    inspect.add_argument("checkpoint", type=Path)
    migrate = commands.add_parser("migrate", help="convert a stopped legacy replica's trajectory storage")
    migrate.add_argument("run", type=Path, help="run directory containing manifest.json and input/")
    migrate.add_argument("--storage", required=True, choices=("memory", "stream"))
    migrate.add_argument("--replica", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            result = inspect_pimd_checkpoint(args.checkpoint)
        else:
            path, config, system = _saved_run(args.run)
            saved = migrate_legacy_checkpoint(config, system, path, args.replica, storage=args.storage)
            result = inspect_pimd_checkpoint(saved["checkpoint"])
            result.update(complete=saved["complete"], frames=saved["frames"],
                          configured_resume_storage=config.get("output", {}).get("trajectory_storage", "stream"))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
