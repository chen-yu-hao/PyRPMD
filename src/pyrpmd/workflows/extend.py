"""Controlled budget extensions of completed, unresolved dRPMD trajectories.

The original scientific input and manifest remain immutable. A separate atomic
ledger authorizes a larger total propagation budget, while each trajectory
continues from its full checkpoint under its original physics identity.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

from .. import __version__
from ..dynamics import Dynamics, State
from ..input import InputError
from ..io import fingerprint, referenced_files, write_json
from ..parallel import execution_plan, make_task_potential, root_call, run_tasks
from ..propagation import loop_options, scientific_config
from ..units import time_steps
from . import drpmd


_FORMAT = "PyRPMD.drpmd.extensions.v1"
_LOCKS = {}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _lock(directory):
    """Hold an OS lock across workers; an interrupted process releases it."""
    path = Path(directory)
    if not (path / "manifest.json").is_file():
        raise InputError(f"No run manifest in {path}")
    stream = (path / "checkpoints" / "extensions.lock").open("a+b")
    try:
        if os.name == "nt":
            import msvcrt

            if stream.seek(0, 2) == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        stream.close()
        raise InputError("Another budget extension is active for this run") from exc
    _LOCKS[str(path)] = stream


def _unlock(directory):
    stream = _LOCKS.pop(str(Path(directory)), None)
    if stream is not None:
        stream.close()


def _authenticate(config, system, path):
    """The resume identity checks, performed without migrating the manifest."""
    manifest = json.loads((path / "manifest.json").read_text())
    schema = manifest.get("schema_version", 1)
    if type(schema) is not int or schema not in (1, 2):
        raise InputError(f"Unsupported run manifest schema version: {schema}")
    if (schema == 2 or "schema" in manifest) and manifest.get("schema") != "pyrpmd.run":
        raise InputError("Unknown run manifest schema")
    identity_config = config if schema == 1 else scientific_config(config)
    if fingerprint(identity_config, system) != manifest.get("physics_fingerprint"):
        raise InputError("Saved scientific settings or system do not match the run manifest")
    hashes = referenced_files(config)
    for file, sha in manifest.get("files", {}).items():
        if hashes.get(file) != sha:
            raise InputError(f"Input/model file changed or is missing: {file}")


def _checkpoint(path, record, physics_hash, target, dt):
    file = path / "trajectories" / f"trajectory-{record['trajectory_id']:08d}.checkpoint.npz"
    if not file.is_file():
        raise InputError(f"Unresolved trajectory needs its full checkpoint: {file}")
    meta, arrays = drpmd._load_npz(file, expected_kind="PyRPMD.drpmd.trajectory-checkpoint.v1")
    if (meta.get("physics_hash") != physics_hash
            or meta.get("trajectory_id") != record["trajectory_id"]
            or meta.get("parent_id") != record["parent_id"]):
        raise InputError("Trajectory checkpoint does not match its saved launch identity")
    done = meta.get("done")
    if (type(done) is not int or done < record["steps"] or done > target
            or meta.get("step") != done
            or not math.isclose(meta.get("time", float("nan")), done * dt,
                                rel_tol=1e-9, abs_tol=dt * 1e-6)):
        raise InputError("Trajectory checkpoint has a conflicting propagation clock or budget")
    if meta.get("status", "unresolved") not in drpmd._STATUSES:
        raise InputError("Trajectory checkpoint has an unknown outcome status")
    if meta.get("launch") != record["launch"]:
        raise InputError("Trajectory checkpoint launch metadata conflicts with its result")
    state = State(arrays["q"], arrays["p"], meta["step"], meta["time"])
    return state


def _prepare_extension(directory, max_time):
    from ..runner import _read_identity

    path = Path(directory)
    config = json.loads((path / "input" / "resolved.json").read_text())
    if config.get("task") != "drpmd":
        raise InputError("Budget extension currently supports only dRPMD scattering runs")
    if config.get("nonadiabatic"):
        raise InputError("Nonadiabatic trajectory extension is not implemented")
    system = _read_identity(config, path)
    if config.get("_base_directory") and config["_base_directory"] not in sys.path:
        sys.path.insert(0, config["_base_directory"])
    _authenticate(config, system, path)
    target = time_steps(max_time, config["dt_au"])
    original = config["collision"]["max_steps"]
    if target <= original:
        raise InputError("The new total max-time must exceed the original trajectory budget")
    plan = execution_plan(config.get("parallel"))
    n = config["collision"]["trajectories"]
    default_parents = plan.trajectory_groups if plan.backend == "mpi" else plan.trajectory_workers
    parent_count = min(n, config.get("sampling", {}).get("parents", default_parents))
    # Hash BEFORE changing max_steps. This also rechecks local model/classifier
    # dependencies which enter dRPMD's more specific trajectory identity.
    physics_hash = drpmd._physics_hash(config, system, parent_count)
    summary_file = path / "results" / "summary.json"
    previous_summary = json.loads(summary_file.read_text()) if summary_file.is_file() else {}
    cached = {record["trajectory_id"]: record for record in previous_summary.get("records", [])}
    records, allocations, start = [], {}, 0
    for parent_id in range(parent_count):
        count = n // parent_count + int(parent_id < n % parent_count)
        allocations[parent_id] = list(range(start, start + count))
        start += count
    owners = {trajectory_id: parent for parent, ids in allocations.items() for trajectory_id in ids}
    for trajectory_id in range(n):
        file = path / "trajectories" / f"trajectory-{trajectory_id:08d}.json"
        if not file.is_file():
            raise InputError("The original run has unfinished launches; resume it before extending time limits")
        current = json.loads(file.read_text())
        old = cached.get(trajectory_id, {})
        if (current.get("physics_hash") != physics_hash
                or current.get("trajectory_id") != trajectory_id
                or type(current.get("parent_id")) is not int or current["parent_id"] < 0
                or (not config.get("initial_ensemble")
                    and current["parent_id"] != owners[trajectory_id])
                or current.get("status") not in drpmd._STATUSES
                or type(current.get("steps")) is not int or current["steps"] < 0
                or not isinstance(current.get("launch"), dict)):
            raise InputError("Saved trajectory result does not match the original run identity")
        if old and (old.get("physics_hash") != physics_hash
                    or old.get("steps", 0) > current["steps"]):
            raise InputError("Saved summary conflicts with an authoritative trajectory result")
        records.append({**old, **current})
    if config.get("initial_ensemble"):
        # Selected-state preparations may have been resumed with a different
        # work allocation. Their recorded parent labels are scheduling metadata.
        allocations = {}
        for record in records:
            allocations.setdefault(record["parent_id"], []).append(record["trajectory_id"])

    ledger_file = path / "results" / "extensions.json"
    if ledger_file.is_file():
        ledger = json.loads(ledger_file.read_text())
        if (ledger.get("format") != _FORMAT or ledger.get("physics_hash") != physics_hash
                or ledger.get("original_max_steps") != original
                or ledger.get("dt_au") != config["dt_au"]
                or not isinstance(ledger.get("requests"), list) or not ledger["requests"]):
            raise InputError("Extension ledger does not match this run")
        budgets = [item.get("max_steps") for item in ledger["requests"]]
        if (any(type(value) is not int or value <= original for value in budgets)
                or any(a >= b for a, b in zip(budgets, budgets[1:]))
                or any(item.get("status") not in ("running", "completed", "superseded")
                       for item in ledger["requests"])):
            raise InputError("Extension ledger contains an invalid budget history")
        if target < budgets[-1]:
            raise InputError("The requested max-time would reduce an already authorized extension budget")
    else:
        ledger = {"format": _FORMAT, "physics_hash": physics_hash,
                  "original_max_steps": original, "dt_au": config["dt_au"], "requests": []}
    if any(record["steps"] > target for record in records):
        raise InputError("A trajectory already exceeds the requested total time limit")
    selected = [record for record in records
                if record["status"] == "unresolved" and record["steps"] < target]
    # Validate every selected checkpoint before authorizing or propagating any
    # new segment. _trajectory restores it again inside its worker.
    for record in selected:
        state = _checkpoint(path, record, physics_hash, target, config["dt_au"])
        if state.q.shape != (config["beads"], len(drpmd._field(system, "masses")), 3):
            raise InputError("Trajectory checkpoint shape differs from the saved physical system")
    if not ledger["requests"] or target > ledger["requests"][-1]["max_steps"]:
        if ledger["requests"] and ledger["requests"][-1]["status"] == "running":
            ledger["requests"][-1].update(status="superseded", superseded_by_max_steps=target)
        ledger["requests"].append({
            "max_steps": target, "max_time_au": target * config["dt_au"],
            "status": "running", "requested_at": _now(),
            "trajectory_ids": [record["trajectory_id"] for record in selected],
            "from_steps": {str(record["trajectory_id"]): record["steps"] for record in selected},
        })
    elif selected:
        # A crash can occur after one trajectory result commits but before the
        # batch ledger or summary commits. Repeating the same request is safe.
        ledger["requests"][-1]["status"] = "running"
    budget_config = copy.deepcopy(config)
    budget_config["collision"]["max_steps"] = target
    budget_config["resume"] = True
    tasks = []
    for parent_id, ids in allocations.items():
        pending = [record for record in selected if record["trajectory_id"] in ids]
        if pending:
            tasks.append({"config": budget_config, "system": system, "output": str(path),
                          "parent_id": parent_id, "records": pending, "physics_hash": physics_hash})
    thermal_steps = previous_summary.get("parent_thermal_steps", {})
    parents = []
    for parent_id, ids in allocations.items():
        steps = thermal_steps.get(str(parent_id))
        checkpoint = path / "checkpoints" / f"parent-{parent_id:06d}.npz"
        if steps is None and checkpoint.is_file():
            meta, _ = drpmd._load_npz(checkpoint)
            if meta.get("physics_hash") != physics_hash or meta.get("parent_id") != parent_id:
                raise InputError("Parent checkpoint identity conflicts with the saved run")
            steps = meta.get("step")
        if steps is None and config.get("initial_ensemble"):
            steps = 0
        parents.append({"parent_id": parent_id, "thermal_steps": steps,
                        "records": [records[i] for i in ids]})
    write_json(ledger_file, ledger)
    return {"config": config, "system": system, "tasks": tasks,
            "parents": parents, "ledger": ledger, "output": str(path), "expected": n}


def _extend_parent(task):
    config, system, out = task["config"], task["system"], Path(task["output"])
    if config.get("_base_directory") and config["_base_directory"] not in sys.path:
        sys.path.insert(0, config["_base_directory"])
    potential = make_task_potential(config["potential"], system)
    try:
        plan = execution_plan(config.get("parallel"))
        dynamics = Dynamics(
            potential, np.asarray(drpmd._field(system, "masses"), dtype=float),
            config["beta"], config["beads"], config["dt_au"],
            seed=np.random.SeedSequence([int(config.get("seed", 0)), 3, task["parent_id"]]),
            native=config.get("native", config.get("dynamics", {}).get("native", True)),
            threads=plan.threads, **loop_options(config),
        )
        classifier = drpmd._make_classifier(config, system)
        updated = []
        for previous in task["records"]:
            state = _checkpoint(out, previous, task["physics_hash"],
                                config["collision"]["max_steps"], config["dt_au"])
            current = drpmd._trajectory(
                dynamics, state, classifier, config, out,
                previous["trajectory_id"], task["parent_id"], previous["launch"],
                task["physics_hash"], True, extend_unresolved=True,
            )
            # Preserve initial-state and parent-sampling metadata carried by
            # earlier workflow versions outside the core trajectory function.
            current = {**previous, **current}
            drpmd._atomic_json(out / "trajectories" / f"trajectory-{current['trajectory_id']:08d}.json", current)
            updated.append(current)
        return updated
    finally:
        if hasattr(potential, "close"):
            potential.close()


def _finish_extension(prepared, results):
    path = Path(prepared["output"])
    updated = {record["trajectory_id"]: record for batch in results for record in batch}
    for parent in prepared["parents"]:
        parent["records"] = [updated.get(record["trajectory_id"], record) for record in parent["records"]]
    ledger = prepared["ledger"]
    request = ledger["requests"][-1]
    request["status"] = "completed"
    request.setdefault("completed_at", _now())
    request["outcomes"] = {
        str(record["trajectory_id"]): {"status": record["status"], "steps": record["steps"]}
        for parent in prepared["parents"] for record in parent["records"]
        if record["trajectory_id"] in request["trajectory_ids"]
    }
    write_json(path / "results" / "extensions.json", ledger)
    summary = drpmd._write_summary(prepared["parents"], prepared["config"],
                                   prepared["system"], path, prepared["expected"])
    summary.update({
        "software": "PyRPMD", "version": __version__, "output": str(path),
        "preset": prepared["config"].get("preset", "pilot"),
        "input_warnings": prepared["config"].get("_warnings", []),
        "original_max_steps": ledger["original_max_steps"],
        "effective_max_steps": request["max_steps"], "extensions": ledger,
    })
    write_json(path / "results" / "summary.json", summary)
    return summary


def extend_drpmd(directory, *, max_time):
    """Raise the total time limit, continuing only unresolved saved launches.

    All ranks participate under an existing MPI execution plan. Repeating the
    same requested total limit recovers an interrupted extension and is otherwise
    idempotent. Completed chemical outcomes and failed trajectories are retained.
    No parent thermalization, initial-state sampling or collision launch occurs.
    """
    path = Path(directory).resolve()
    if path.name == "checkpoints":
        path = path.parent
    root_call(_lock, path)
    try:
        prepared = root_call(_prepare_extension, path, max_time)
        results = run_tasks(_extend_parent, prepared["tasks"], prepared["config"].get("parallel"))
        return root_call(_finish_extension, prepared, results)
    finally:
        root_call(_unlock, path)
