"""Replicated, bead-resolved and native multiple-walker PLUMED workflow."""

from __future__ import annotations

from contextlib import chdir
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
import uuid

import numpy as np

from ..checkpoint import atomic_savez, load_checkpoint
from ..dynamics import Dynamics, State
from ..io import fingerprint, fingerprint_payload, referenced_files, write_json
from ..parallel import broadcast, is_root, root_call, run_tasks, task_potential
from ..plumed_extended import (
    ExtendedPlumedBias,
    configure_multiple_walkers,
    ensure_checkpoint_flush,
    plumed_atom_map,
    validate_managed_output_paths,
)
from ..propagation import loop_options, scientific_config


_CHECKPOINT_KIND = "PyRPMD.plumed-external-files.v2"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _replica_seed(seed, replica):
    value = int(seed) + int(replica)
    if value < 0 or value > np.iinfo(np.uint64).max:
        raise ValueError("PLUMED replica seed is outside NumPy's uint64 range")
    return value


def _managed_path(root: Path, relative: str) -> Path:
    path = root / relative
    current = path
    while current != root.parent:
        if current.is_symlink():
            raise ValueError(
                "PLUMED managed files and their parents must not be symlinks"
            )
        current = current.parent
    if path.exists() and not path.is_file():
        raise ValueError("PLUMED managed path is not a regular file")
    return path


def _inventory(root: Path) -> dict[str, bytes]:
    if not root.exists():
        return {}
    if root.is_symlink() or not root.is_dir():
        raise ValueError("PLUMED state root must be a real directory")
    result: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(
                "PLUMED managed files and directories must not be symlinks"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("PLUMED state tree contains a non-regular file")
        relative = path.relative_to(root).as_posix()
        result[relative] = path.read_bytes()
    return result


def _manifest_arrays(local: Path, shared: Path | None):
    manifest, arrays = [], {}
    for scope, root in (("replica", local), ("shared", shared)):
        if root is None:
            continue
        for relative, data in _inventory(root).items():
            name = f"file_{len(manifest)}"
            manifest.append(
                {
                    "scope": scope,
                    "path": relative,
                    "array": name,
                    "offset": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
            arrays[name] = np.frombuffer(data, dtype=np.uint8).copy()
    return manifest, arrays


def _manifest_contents(manifest, arrays):
    contents = {"replica": {}, "shared": {}}
    if not isinstance(manifest, list):
        raise ValueError("PLUMED checkpoint file manifest must be a list")
    for index, entry in enumerate(manifest):
        if (
            not isinstance(entry, dict)
            or set(entry) != {"scope", "path", "array", "offset", "sha256"}
            or entry["scope"] not in contents
            or entry["array"] != f"file_{index}"
            or not isinstance(entry["path"], str)
            or not entry["path"]
            or Path(entry["path"]).is_absolute()
            or ".." in Path(entry["path"]).parts
            or not isinstance(entry["offset"], int)
            or entry["offset"] < 0
            or not isinstance(entry["sha256"], str)
            or _SHA256.fullmatch(entry["sha256"]) is None
        ):
            raise ValueError("PLUMED checkpoint file manifest is invalid")
        blob = arrays.get(entry["array"])
        if (
            not isinstance(blob, np.ndarray)
            or blob.dtype != np.uint8
            or blob.ndim != 1
            or blob.size != entry["offset"]
        ):
            raise ValueError("PLUMED checkpoint file bytes are invalid")
        data = blob.tobytes()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(
                "PLUMED checkpoint file hash does not match its payload"
            )
        if entry["path"] in contents[entry["scope"]]:
            raise ValueError("PLUMED checkpoint file paths are duplicated")
        contents[entry["scope"]][entry["path"]] = data
    return contents


def _atomic_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _verify_restore_tree(root: Path, expected: dict[str, bytes]):
    """Verify every live prefix and atomically truncate crash tails."""
    current = _inventory(root)
    if set(current) != set(expected):
        raise ValueError(
            "PLUMED external file inventory differs from the checkpoint"
        )
    for relative, saved in expected.items():
        actual = current[relative]
        if (
            len(actual) < len(saved)
            or hashlib.sha256(actual[: len(saved)]).digest()
            != hashlib.sha256(saved).digest()
        ):
            raise ValueError(
                f"PLUMED external file {relative!r} does not match the "
                "checkpoint prefix"
            )
    for relative, saved in expected.items():
        _atomic_bytes(_managed_path(root, relative), saved)


def _verify_exact_tree(root: Path, expected: dict[str, bytes]):
    current = _inventory(root)
    if set(current) != set(expected):
        raise ValueError(
            "PLUMED shared file inventory differs between replica checkpoints"
        )
    for relative, saved in expected.items():
        if current[relative] != saved:
            raise ValueError(
                f"PLUMED shared file {relative!r} differs between replica "
                "checkpoints"
            )


def _restore_open_tree(root: Path, expected: dict[str, bytes]):
    """Undo restart-clock replay without replacing PLUMED's open inodes."""
    current = _inventory(root)
    if set(current) != set(expected):
        raise ValueError(
            "PLUMED restart replay changed the external file inventory"
        )
    for relative, saved in expected.items():
        actual = current[relative]
        if len(actual) < len(saved) or actual[: len(saved)] != saved:
            raise ValueError(
                "PLUMED restart replay rewrote a checkpoint file prefix"
            )
    for relative, saved in expected.items():
        path = _managed_path(root, relative)
        with path.open("r+b") as stream:
            stream.seek(0)
            stream.write(saved)
            stream.truncate(len(saved))
            stream.flush()
            os.fsync(stream.fileno())


def _identity(config, system, script, replica):
    return fingerprint_payload(
        {
            "profile": "PyRPMD.plumed-external-files.v2",
            "physics": fingerprint(scientific_config(config), system),
            "input_files": referenced_files(config),
            "configured_input": script,
            "replica": int(replica),
        }
    )


def _load_checkpoint(snapshot, identity, shape, total, mode, walkers, replica):
    with load_checkpoint(
        snapshot, expected_kind=_CHECKPOINT_KIND, require_integrity=True
    ) as saved:
        arrays = {name: saved[name] for name in saved.files}
    fixed = {
        "q",
        "p",
        "step",
        "time",
        "rng",
        "identity",
        "api_version",
        "sample_steps",
        "positions",
        "physical_bias",
        "file_manifest",
        "mode",
        "walkers",
        "replica",
        "kernel_state_saved",
        "plumed_file_state_saved",
    }
    dynamic = {name for name in arrays if name.startswith("file_")}
    if set(arrays) != fixed | dynamic:
        raise ValueError("PLUMED v2 checkpoint field inventory is invalid")
    scalar_strings = {
        "identity": identity,
        "mode": mode,
        "walkers": walkers,
    }
    for name, expected in scalar_strings.items():
        value = arrays[name]
        if value.shape != () or str(value.item()) != expected:
            raise ValueError(
                f"PLUMED checkpoint {name} does not match this run"
            )
    saved_replica = arrays["replica"]
    if (
        saved_replica.shape != ()
        or saved_replica.dtype.kind not in "iu"
        or int(saved_replica) != replica
    ):
        raise ValueError("PLUMED checkpoint replica ID is invalid")
    for name in ("q", "p"):
        value = arrays[name]
        if (
            value.shape != shape
            or value.dtype != np.float64
            or not np.isfinite(value).all()
        ):
            raise ValueError("invalid PLUMED checkpoint coordinates/momenta")
    step_value = arrays["step"]
    if (
        step_value.shape != ()
        or step_value.dtype.kind not in "iu"
        or not 0 <= int(step_value) <= total
    ):
        raise ValueError("invalid PLUMED checkpoint step")
    step = int(step_value)
    time_value = arrays["time"]
    if (
        time_value.shape != ()
        or time_value.dtype.kind not in "fiu"
        or not math.isfinite(float(time_value))
    ):
        raise ValueError("invalid PLUMED checkpoint time")
    api = arrays["api_version"]
    if (
        api.shape != ()
        or api.dtype.kind not in "iu"
        or int(api) < 0
        or (step > 0 and int(api) < 8)
    ):
        raise ValueError("invalid PLUMED checkpoint API version")
    try:
        if arrays["rng"].shape != () or arrays["file_manifest"].shape != ():
            raise ValueError("expected JSON scalars")
        rng = json.loads(str(arrays["rng"].item()))
        probe = np.random.default_rng()
        probe.bit_generator.state = rng
        manifest = json.loads(str(arrays["file_manifest"].item()))
    except (ValueError, TypeError, KeyError) as error:
        raise ValueError(
            "invalid PLUMED checkpoint RNG/file manifest"
        ) from error
    contents = _manifest_contents(manifest, arrays)
    sample_steps = arrays["sample_steps"]
    positions = arrays["positions"]
    biases = arrays["physical_bias"]
    if (
        sample_steps.ndim != 1
        or sample_steps.dtype.kind not in "iu"
        or (sample_steps.size and np.any(np.diff(sample_steps) <= 0))
        or (sample_steps.size and int(sample_steps[-1]) > step)
        or positions.shape != (sample_steps.size, *shape)
        or positions.dtype != np.float64
        or not np.isfinite(positions).all()
        or biases.shape != (sample_steps.size,)
        or biases.dtype != np.float64
        or not np.isfinite(biases).all()
    ):
        raise ValueError("invalid PLUMED checkpoint sample arrays")
    for marker, expected in (
        ("kernel_state_saved", False),
        ("plumed_file_state_saved", True),
    ):
        value = arrays[marker]
        if (
            value.shape != ()
            or value.dtype.kind != "b"
            or bool(value) is not expected
        ):
            raise ValueError("invalid PLUMED checkpoint state marker")
    return arrays, rng, contents


def _save_checkpoint(
    snapshot,
    *,
    identity,
    mode,
    walkers,
    replica,
    state,
    dynamics,
    api_version,
    steps,
    positions,
    biases,
    shape,
    work,
    shared,
):
    manifest, blobs = _manifest_arrays(work, shared)
    atomic_savez(
        snapshot,
        kind=_CHECKPOINT_KIND,
        q=state.q,
        p=state.p,
        step=state.step,
        time=state.time,
        rng=json.dumps(dynamics.rng.bit_generator.state, sort_keys=True),
        identity=identity,
        api_version=api_version,
        mode=mode,
        walkers=walkers,
        replica=replica,
        kernel_state_saved=False,
        plumed_file_state_saved=True,
        file_manifest=json.dumps(manifest, sort_keys=True),
        sample_steps=np.asarray(steps, dtype=np.int64),
        positions=np.asarray(positions, dtype=np.float64).reshape(-1, *shape),
        physical_bias=np.asarray(biases, dtype=np.float64),
        **blobs,
    )


class _FilesystemSync:
    """Filesystem rendezvous for process walkers without hidden workers."""

    def __init__(self, root, replicas, replica):
        self.root = Path(root)
        self.replicas = int(replicas)
        self.replica = int(replica)
        self.phase = 0
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def abort_path(self):
        return self.root / "ABORT"

    def abort(self, message):
        try:
            descriptor = os.open(
                self.abort_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
        except FileExistsError:
            return
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(str(message).encode("utf-8", "replace"))
            stream.flush()
            os.fsync(stream.fileno())

    def exchange(self, payload=b""):
        if isinstance(payload, str):
            payload = payload.encode("utf-8", "replace")
        if self.abort_path.exists():
            raise RuntimeError(
                "PLUMED walker peer aborted: "
                + self.abort_path.read_text(errors="replace")
            )
        directory = self.root / f"phase-{self.phase:012d}"
        self.phase += 1
        directory.mkdir(parents=True, exist_ok=True)
        marker = directory / f"{self.replica:06d}"
        descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        expected = [directory / f"{index:06d}" for index in range(self.replicas)]
        while not all(path.is_file() for path in expected):
            if self.abort_path.exists():
                raise RuntimeError(
                    "PLUMED walker peer aborted: "
                    + self.abort_path.read_text(errors="replace")
                )
            time.sleep(0.005)
        return [path.read_bytes() for path in expected]


def _mpi_communicators(config, replica, replicas, shared_walkers):
    if config.get("parallel", {}).get("backend") != "mpi":
        return None, None
    from mpi4py import MPI

    world = MPI.COMM_WORLD
    parallel = config["parallel"]
    group_size = parallel.get("bead_workers", 1) * parallel.get("pes_ranks", 1)
    leaders = [
        parallel.get("coordinator_ranks", 0) + index * group_size
        for index in range(replicas)
    ]
    if world.Get_rank() != leaders[replica]:
        raise RuntimeError(
            "PLUMED replica task did not execute on its MPI group leader"
        )
    multi = None
    if shared_walkers:
        world_group = world.Get_group()
        leader_group = world_group.Incl(leaders)
        try:
            multi = world.Create_group(leader_group, tag=3617)
        finally:
            leader_group.Free()
            world_group.Free()
        if (
            multi == MPI.COMM_NULL
            or multi.Get_size() != replicas
            or multi.Get_rank() != replica
        ):
            if multi != MPI.COMM_NULL:
                multi.Free()
            raise RuntimeError(
                "PLUMED MPI leader communicator does not match replica IDs"
            )
    return MPI.COMM_SELF, multi


def _errors(comm, sync, local_error):
    if comm is not None:
        return comm.allgather(local_error)
    if sync is not None:
        return [
            value.decode("utf-8", "replace")
            for value in sync.exchange(local_error or "")
        ]
    return [local_error]


def _coordinated_call(comm, sync, function, *, active=True):
    error = None
    caught = None
    if active:
        try:
            function()
        except BaseException as exc:  # synchronize before propagating
            caught = exc
            error = f"{type(exc).__name__}: {exc}"
    errors = _errors(comm, sync, error)
    if any(errors):
        if caught is not None:
            raise caught
        raise RuntimeError(
            "PLUMED walker peer failed: "
            + "; ".join(value for value in errors if value)
        )


def _collective_any(comm, sync, requested):
    if comm is not None:
        return bool(comm.allreduce(int(bool(requested))))
    if sync is not None:
        values = sync.exchange("1" if requested else "0")
        return any(value == b"1" for value in values)
    return bool(requested)


def _advance(task, bridge, dynamics, state, comm, sync):
    sampling = task["config"]["sampling"]
    kwargs = {
        "thermostat": sampling["thermostat"],
        "gamma_centroid": sampling.get("gamma_centroid", 1.0),
        "andersen_interval": sampling["andersen_interval"],
    }
    if task["walkers"] == "files":
        for turn in range(task["replicas"]):
            _coordinated_call(
                comm,
                sync,
                lambda: bridge.step(dynamics, state, **kwargs),
                active=task["replica"] == turn,
            )
    elif task["walkers"] == "mpi":
        _coordinated_call(
            comm, sync, lambda: bridge.step(dynamics, state, **kwargs)
        )
    else:
        bridge.step(dynamics, state, **kwargs)


def _prime(task, bridge, state, comm, sync):
    if task["walkers"] == "files":
        for turn in range(task["replicas"]):
            _coordinated_call(
                comm,
                sync,
                lambda: bridge._prime_file_restart(state.q),
                active=task["replica"] == turn,
            )
    elif task["walkers"] == "mpi":
        _coordinated_call(
            comm, sync, lambda: bridge._prime_file_restart(state.q)
        )
    else:
        bridge._prime_file_restart(state.q)


def _configured_script(table, replicas, replica):
    raw = (
        Path(table["file"]).read_text(encoding="utf-8")
        if "file" in table
        else table["input"]
    )
    script = configure_multiple_walkers(
        raw,
        walkers=table.get("walkers", "independent"),
        count=replicas,
        walker_id=replica,
        read_stride=table.get("walker_read_stride", 1),
    )
    if table.get("restart") == "files":
        script = ensure_checkpoint_flush(script)
    validate_managed_output_paths(script)
    return script


def _replica_worker(task):
    config, system = task["config"], task["system"]
    output = Path(task["output"]).resolve()
    replica, replicas = task["replica"], task["replicas"]
    mode, walkers = task["mode"], task["walkers"]
    table, sampling = config["plumed"], config["sampling"]
    work = output / "plumed" / f"replica-{replica:04d}"
    shared = output / "plumed" / "shared" if walkers == "files" else None
    snapshot = output / "checkpoints" / f"plumed-replica-{replica:04d}.npz"
    script = _configured_script(table, replicas, replica)
    identity = _identity(config, system, script, replica)
    shape = (config["beads"], system.natoms, 3)
    total = sampling["equilibration_steps"] + sampling["production_steps"]
    resume = bool(config.get("resume"))
    restart_files = table.get("restart") == "files"
    communicator, leader_comm = _mpi_communicators(
        config, replica, replicas, walkers in ("files", "mpi")
    )
    sync = (
        _FilesystemSync(task["sync_root"], replicas, replica)
        if walkers == "files" and leader_comm is None
        else None
    )
    checkpoint_step = 0
    try:
        work.mkdir(parents=True, exist_ok=True)
        if shared is not None:
            shared.mkdir(parents=True, exist_ok=True)

        restored = None
        if resume:
            restored = _load_checkpoint(
                snapshot,
                identity,
                shape,
                total,
                mode,
                walkers,
                replica,
            )
            arrays, rng_state, contents = restored
            _coordinated_call(
                leader_comm,
                sync,
                lambda: _verify_restore_tree(work, contents["replica"]),
            )
            if shared is not None:
                _coordinated_call(
                    leader_comm,
                    sync,
                    lambda: _verify_restore_tree(shared, contents["shared"]),
                    active=replica == 0,
                )
                _coordinated_call(
                    leader_comm,
                    sync,
                    lambda: _verify_exact_tree(shared, contents["shared"]),
                    active=replica != 0,
                )
        else:
            # Resolver validation happens before output initialization.  This
            # second check is deliberately before PES/RNG/kernel construction.
            if _inventory(work):
                raise ValueError(
                    "PLUMED replica directory is not empty; choose a new output"
                )
            if shared is not None:
                def check_shared_empty():
                    if _inventory(shared):
                        raise ValueError(
                            "PLUMED shared walker directory is not empty; "
                            "choose a new output"
                        )

                _coordinated_call(
                    leader_comm,
                    sync,
                    check_shared_empty,
                    active=replica == 0,
                )

        with task_potential(config["potential"], system) as provider:
            dynamics = Dynamics(
                provider,
                system.masses,
                config["beta"],
                config["beads"],
                config["dt_au"],
                seed=_replica_seed(config.get("seed", 0), replica),
                native=config.get(
                    "native", config.get("dynamics", {}).get("native", True)
                ),
                threads=config.get("parallel", {}).get("threads_per_rank", 1),
                **loop_options(config),
            )
            if restored is None:
                state = State(
                    np.repeat(system.geometry[None], config["beads"], axis=0),
                    dynamics.sample_momenta(),
                )
                steps, positions, biases, api_version = [], [], [], 0
                contents = {"replica": {}, "shared": {}}
            else:
                state = State(
                    arrays["q"],
                    arrays["p"],
                    step=int(arrays["step"]),
                    time=float(arrays["time"]),
                )
                if not math.isclose(
                    state.time,
                    state.step * config["dt_au"],
                    rel_tol=1e-10,
                    abs_tol=1e-10,
                ):
                    raise ValueError(
                        "PLUMED checkpoint time does not match its step"
                    )
                dynamics.rng.bit_generator.state = rng_state
                steps = arrays["sample_steps"].tolist()
                positions = list(arrays["positions"])
                biases = arrays["physical_bias"].tolist()
                api_version = int(arrays["api_version"])
            checkpoint_step = state.step

            def save():
                nonlocal checkpoint_step
                _coordinated_call(
                    leader_comm,
                    sync,
                    lambda: _save_checkpoint(
                        snapshot,
                        identity=identity,
                        mode=mode,
                        walkers=walkers,
                        replica=replica,
                        state=state,
                        dynamics=dynamics,
                        api_version=api_version,
                        steps=steps,
                        positions=positions,
                        biases=biases,
                        shape=shape,
                        work=work,
                        shared=shared,
                    ),
                )
                checkpoint_step = state.step

            if restored is None and restart_files:
                save()
            stopped = False
            try:
                with chdir(work), ExtendedPlumedBias(
                    script,
                    system.masses,
                    config["beta"],
                    config["beads"],
                    config["dt_au"],
                    mode=mode,
                    cell=system.cell,
                    pbc=system.pbc,
                    charges=table.get("charges"),
                    kernel_path=table.get("kernel"),
                    log_file=str(
                        output / "logs" / f"plumed-{replica:04d}.log"
                    ),
                    restart=resume and state.step > 0,
                    start_step=state.step,
                    replica_comm=communicator,
                    multi_sim_comm=(leader_comm if walkers == "mpi" else None),
                ) as bridge:
                    if api_version and api_version != bridge.api_version:
                        raise ValueError(
                            "PLUMED API version changed since checkpoint"
                        )
                    api_version = bridge.api_version
                    if resume and state.step > 0:
                        _prime(task, bridge, state, leader_comm, sync)
                        _coordinated_call(
                            leader_comm,
                            sync,
                            lambda: _restore_open_tree(
                                work, contents["replica"]
                            ),
                        )
                        if shared is not None:
                            _coordinated_call(
                                leader_comm,
                                sync,
                                lambda: _restore_open_tree(
                                    shared, contents["shared"]
                                ),
                                active=replica == 0,
                            )
                            _coordinated_call(
                                leader_comm,
                                sync,
                                lambda: _verify_exact_tree(
                                    shared, contents["shared"]
                                ),
                                active=replica != 0,
                            )
                    while state.step < total and not stopped:
                        _advance(task, bridge, dynamics, state, leader_comm, sync)
                        production = state.step - sampling["equilibration_steps"]
                        if (
                            production > 0
                            and production % sampling["sample_interval"] == 0
                        ):
                            def sample():
                                result = bridge.evaluate(state.q)
                                steps.append(state.step)
                                positions.append(state.q.copy())
                                biases.append(result.physical_energy)

                            _coordinated_call(
                                leader_comm,
                                sync,
                                sample,
                            )
                        stopped = (
                            _collective_any(
                                leader_comm, sync, bridge.stop_requested
                            )
                            if walkers != "independent"
                            else bridge.stop_requested
                        )
                        if (
                            restart_files
                            and state.step
                            % config["output"]["checkpoint_steps"]
                            == 0
                        ):
                            save()
                    if restart_files and state.step != checkpoint_step:
                        save()
            except BaseException as error:
                write_json(
                    output
                    / "results"
                    / f"plumed-failure-{replica:04d}.json",
                    {
                        "task": "plumed",
                        "status": "failed",
                        "replica": replica,
                        "last_accepted_step": state.step,
                        "checkpoint_step": checkpoint_step,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "md_snapshot": str(snapshot) if restart_files else None,
                        "kernel_state_saved": False,
                        "plumed_file_state_saved": restart_files,
                        "restart_supported": restart_files,
                    },
                )
                raise
            return {
                "replica": replica,
                "status": "plumed_stop" if stopped else "complete",
                "steps": state.step,
                "samples": len(steps),
                "checkpoint_step": checkpoint_step,
                "md_snapshot": str(snapshot) if restart_files else None,
                "plumed_directory": str(work),
                "plumed_api_version": api_version,
            }
    except BaseException as error:
        if sync is not None:
            sync.abort(f"replica {replica}: {type(error).__name__}: {error}")
        raise
    finally:
        if leader_comm is not None:
            leader_comm.Free()


def _preflight(config, output):
    output = Path(output)
    replicas = config["sampling"].get("replicas", 1)
    walkers = config["plumed"].get("walkers", "independent")
    resume = bool(config.get("resume"))
    root = output / "plumed"
    snapshots = [
        output / "checkpoints" / f"plumed-replica-{replica:04d}.npz"
        for replica in range(replicas)
    ]
    if resume:
        missing = [str(path) for path in snapshots if not path.is_file()]
        for replica in range(replicas):
            directory = root / f"replica-{replica:04d}"
            if not directory.is_dir() or directory.is_symlink():
                missing.append(str(directory))
        if walkers == "files":
            shared = root / "shared"
            if not shared.is_dir() or shared.is_symlink():
                missing.append(str(shared))
        if missing:
            raise ValueError(
                "PLUMED resume requires every checkpoint and managed state "
                "directory; missing: " + ", ".join(missing)
            )
    else:
        if root.exists() and (
            root.is_symlink()
            or not root.is_dir()
            or any(root.iterdir())
        ):
            raise ValueError(
                "PLUMED managed state directory is not empty; choose a new output"
            )
        if any(path.exists() for path in snapshots):
            raise ValueError(
                "PLUMED replica checkpoint already exists; choose a new output"
            )
    for child in ("plumed", "logs", "checkpoints", "results"):
        (output / child).mkdir(parents=True, exist_ok=True)


def _remove_sync(path):
    shutil.rmtree(path, ignore_errors=True)


def run_extended_plumed(config, system, output):
    """Dispatch one PLUMED instance per replica and collate on world root."""
    output = Path(output).resolve()
    table = config["plumed"]
    mode = table.get("mode", "centroid")
    walkers = table.get("walkers", "independent")
    replicas = config["sampling"].get("replicas", 1)
    if config.get("resume") and table.get("restart") != "files":
        raise ValueError(
            "PLUMED extended resume requires plumed.restart='files'"
        )
    root_call(_preflight, config, output)
    token = broadcast(uuid.uuid4().hex if is_root() else None)
    sync_root = output / "checkpoints" / f".plumed-sync-{token}"
    if is_root() and mode == "beads":
        write_json(
            output / "results" / "plumed-atom-map.json",
            {
                "mode": "beads",
                "ordering": "bead-major",
                "nbeads": config["beads"],
                "natoms": system.natoms,
                "map": plumed_atom_map(
                    config["beads"], system.natoms, mode
                ),
            },
        )
    tasks = [
        {
            "config": config,
            "system": system,
            "output": str(output),
            "replica": replica,
            "replicas": replicas,
            "mode": mode,
            "walkers": walkers,
            "sync_root": str(sync_root),
        }
        for replica in range(replicas)
    ]
    try:
        results = run_tasks(_replica_worker, tasks, config.get("parallel", {}))
    finally:
        root_call(_remove_sync, sync_root)
    if results is None:
        return None
    statuses = {item["status"] for item in results}
    return {
        "task": "plumed",
        "status": "complete" if statuses == {"complete"} else "plumed_stop",
        "mode": mode,
        "walkers": walkers,
        "replicas": replicas,
        "replica_results": results,
        "restart_supported": table.get("restart") == "files",
        "restart_profile": (
            "external-files-v2" if table.get("restart") == "files" else None
        ),
        "biased_dynamics": True,
        "physical_rate": None,
    }


__all__ = ["run_extended_plumed"]
