"""Authenticated, resumable production extensions of fixed-window gas rates.

Umbrella checkpoints retain their original scientific identity and continue
under an explicit total-budget authorization. Every larger target gets a fresh
PMF/recrossing generation. A single atomic root summary publishes that complete
generation, preserving the previous result throughout interrupted work.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

from .. import __version__
from ..input import InputError, rate_pmf_options
from ..io import file_sha256, write_json
from ..parallel import root_call, run_tasks
from ..rate_artifacts import (EXTENSION_FORMAT, RECEIPT_FORMAT, generation_name,
                              read_generation)
from . import fes, rate
from .extend import _authenticate, _lock, _unlock


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise InputError(f"Expected a JSON object: {path}")
    return value


def _enabled(value):
    return bool(value) and (not isinstance(value, dict) or value.get("enabled", True))


def _pointer(request):
    return {"format": EXTENSION_FORMAT,
            "production_steps": request["production_steps"],
            "generation": request["generation"],
            "receipt_sha256": request["receipt_sha256"]}


def _checkpoint_hashes(records, path):
    return {str(Path(record["checkpoint"]).relative_to(path)):
            file_sha256(record["checkpoint"]) for record in records}


def _snapshot_checkpoint(source, target):
    """Archive a bounded state/descriptor without copying immutable chunks."""
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=target.name, suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as destination, source.open("rb") as origin:
            shutil.copyfileobj(origin, destination)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _validate_ledger(ledger, identity, original):
    if any(ledger.get(key) != value for key, value in identity.items()):
        raise InputError("Rate extension ledger does not match the original run identity")
    requests = ledger.get("requests")
    if not isinstance(requests, list) or not requests:
        raise InputError("Rate extension ledger has no authorized requests")
    previous = original
    for index, request in enumerate(requests):
        if not isinstance(request, dict):
            raise InputError("Invalid rate extension request")
        target = request.get("production_steps")
        if (type(target) is not int or target <= previous
                or request.get("from_production_steps") != previous
                or request.get("generation") != generation_name(target)
                or request.get("status") not in ("running", "completed")
                or (index < len(requests) - 1 and request.get("status") != "completed")):
            raise InputError("Rate extension ledger contains an invalid budget history")
        if request["status"] == "completed":
            digest = request.get("receipt_sha256")
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)):
                raise InputError("Completed rate extension lacks its generation receipt")
        previous = target


def _prepare_extension(directory, target):
    from ..runner import _read_identity

    path = Path(directory)
    config = _json(path / "input" / "resolved.json")
    if config.get("task") != "rate":
        raise InputError("Production-step extension supports only fixed-window gas-phase rate runs")
    if (_enabled(config.get("adaptive_umbrella"))
            or _enabled(config.get("replica_exchange"))):
        raise InputError("Rate production extension requires fixed windows without adaptive umbrellas or replica exchange")
    if any(config.get(key) for key in ("source_pmf", "pmf_source", "saved_pmf")):
        raise InputError("Rate production extension requires local umbrella checkpoints, not an external PMF")
    if config.get("_base_directory") and config["_base_directory"] not in sys.path:
        sys.path.insert(0, config["_base_directory"])
    system = _read_identity(config, path)
    if any(system.pbc) or system.fixed_atoms:
        raise InputError("Production-step extension supports only nonperiodic gas-phase rate runs")
    _authenticate(config, system, path)
    original = fes.checked_integer(config["sampling"]["production_steps"],
                                   "sampling.production_steps")
    if target <= original:
        raise InputError("The new total production-steps must exceed the original production budget")
    preparation = _json(path / "preparation.json")
    if preparation.get("ready") is not True:
        raise InputError("The saved rate preparation is not ready")
    # The original checkpoint fingerprints independently authenticate this
    # prepared reaction. No TS search, provider construction or input rewrite.
    prepared_config = rate._prepare(config, system, path,
                                    _preparation=preparation, _write=False)
    cv, _, _, _ = fes.resolve_umbrellas(prepared_config, system)
    if cv.dimension != 1:
        raise InputError("Rate production extension currently requires one reaction coordinate")
    sampling_config = copy.deepcopy(prepared_config)
    if rate_pmf_options(prepared_config)["method"] == "mbar":
        sampling_config["analysis"]["bootstrap"] = 0
    manifest = _json(path / "manifest.json")
    identity = {
        "format": EXTENSION_FORMAT,
        "physics_fingerprint": manifest["physics_fingerprint"],
        "original_production_steps": original,
        "input_sha256": file_sha256(path / "input" / "resolved.json"),
        "system_sha256": file_sha256(path / "input" / "system.json"),
        "manifest_sha256": file_sha256(path / "manifest.json"),
        "preparation_sha256": file_sha256(path / "preparation.json"),
    }
    ledger_path = path / "results" / "extensions.json"
    if ledger_path.exists():
        ledger = _json(ledger_path)
        _validate_ledger(ledger, identity, original)
        last = ledger["requests"][-1]
        maximum = last["production_steps"]
        minimum = (maximum if last["status"] == "completed"
                   else last["from_production_steps"])
        if target < maximum:
            raise InputError("The requested production-steps would reduce an already authorized budget")
        if target > maximum and last["status"] != "completed":
            raise InputError("Resume the pending rate extension before requesting a larger production budget")
    else:
        summary = _json(path / "results" / "summary.json")
        if summary.get("task") != "rate":
            raise InputError("Finish the original rate run before extending its production budget")
        if "rate_extension" in summary:
            raise InputError("Published rate extension is missing its authorization ledger")
        ledger = {**identity, "requests": []}
        minimum = maximum = original
    records = fes.validate_fes_checkpoints(
        sampling_config, system, path / "umbrella", production_steps=maximum,
    )
    neq = config["sampling"]["equilibration_steps"]
    if any(record["elapsed"] < neq + minimum for record in records):
        raise InputError("An umbrella checkpoint is incomplete for the last completed production budget; resume the run first")
    if ledger["requests"] and ledger["requests"][-1]["status"] == "running":
        last = ledger["requests"][-1]
        binding = path / last["generation"] / "generation.json"
        if binding.is_file():
            saved = _json(binding)
            if (saved.get("format") != "PyRPMD.rate.generation.v1"
                    or saved.get("production_steps") != maximum
                    or saved.get("physics_fingerprint") != identity["physics_fingerprint"]
                    or saved.get("source_checkpoints") != _checkpoint_hashes(records, path)
                    or any(record["elapsed"] != neq + maximum for record in records)):
                raise InputError("Pending rate generation does not match its completed umbrella sources")
    if ledger["requests"] and ledger["requests"][-1]["status"] == "completed":
        last = ledger["requests"][-1]
        generation, _, _ = read_generation(path, _pointer(last))
        source = _json(generation / "generation.json")["source_checkpoints"]
        if _checkpoint_hashes(records, path) != source:
            raise InputError("Umbrella checkpoints differ from the completed rate generation")
    if ledger["requests"] and target == ledger["requests"][-1]["production_steps"]:
        if ledger["requests"][-1]["status"] == "completed":
            return {"completed": True, "output": str(path), "ledger": ledger}
    else:
        relative = generation_name(target)
        if (path / relative).exists():
            raise InputError("An unrecognized rate generation already occupies the requested output")
        ledger["requests"].append({
            "production_steps": target, "from_production_steps": maximum,
            "generation": relative, "status": "running", "requested_at": _now(),
        })
        # Authorize only after every input and expected window passed preflight.
        write_json(ledger_path, ledger)
    generation = path / ledger["requests"][-1]["generation"]
    generation.mkdir(parents=True, exist_ok=True)
    return {"completed": False, "output": str(path), "generation": str(generation),
            "config": prepared_config, "sampling_config": sampling_config,
            "system": system, "preparation": preparation, "ledger": ledger,
            "checkpoint_records": records}


def _bind_generation(prepared, pmf, tasks):
    path, generation = Path(prepared["output"]), Path(prepared["generation"])
    target = prepared["ledger"]["requests"][-1]["production_steps"]
    records = fes.validate_fes_checkpoints(prepared["sampling_config"],
        prepared["system"], path / "umbrella", production_steps=target)
    neq = prepared["config"]["sampling"]["equilibration_steps"]
    if any(record["elapsed"] != neq + target for record in records):
        raise InputError("A rate generation requires complete umbrella production")
    starts = {str(task["chain"]): hashlib.sha256(
        np.ascontiguousarray(task["initial_q"], dtype="<f8").tobytes()).hexdigest()
        for task in tasks}
    sources = {}
    for record in records:
        source = Path(record["checkpoint"])
        relative = str(source.relative_to(path))
        snapshot = generation / relative
        if snapshot.exists():
            if file_sha256(snapshot) != file_sha256(source):
                raise InputError("Archived umbrella checkpoint differs from the generation source")
        else:
            _snapshot_checkpoint(source, snapshot)
        sources[relative] = str(source.with_suffix(".chunks").relative_to(path))
    identity = {
        "format": "PyRPMD.rate.generation.v1",
        "physics_fingerprint": prepared["ledger"]["physics_fingerprint"],
        "production_steps": target, "source_checkpoints": _checkpoint_hashes(records, path),
        "sample_stream_sources": sources,
        "pmf_method": pmf["method"], "xi_star": pmf["xi_star"],
        "parent_initial_q_sha256": starts,
    }
    file = generation / "generation.json"
    if file.exists():
        if _json(file) != identity:
            raise InputError("Rate generation source, surface or parent starting states changed")
    else:
        # Do not adopt caches whose parent origins were never committed.
        if any((generation / "checkpoints").glob("*.npz")):
            raise InputError("Recrossing checkpoints lack a rate generation identity")
        write_json(file, identity)
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _publish_completed(directory, ledger):
    path = Path(directory)
    request = ledger["requests"][-1]
    _, summary, _ = read_generation(path, _pointer(request))
    result = {**summary, "rate_extension": _pointer(request), "extensions": ledger}
    current = path / "results" / "summary.json"
    if not current.is_file() or _json(current) != result:
        write_json(current, result)
    ledger_path = path / "results" / "extensions.json"
    if _json(ledger_path) != ledger:
        write_json(ledger_path, ledger)
    return result


def _finish_extension(prepared, summary):
    path, generation = Path(prepared["output"]), Path(prepared["generation"])
    ledger = copy.deepcopy(prepared["ledger"])
    request = ledger["requests"][-1]
    config = prepared["config"]
    if summary.get("kappa") is not None and not (generation / "recrossing.npz").is_file():
        raise InputError("The rate result is missing its recrossing arrays")
    summary = {**summary, "software": "PyRPMD", "version": __version__,
               "output": str(path), "preset": config.get("preset", "pilot"),
               "input_warnings": config.get("_warnings", []),
               "original_production_steps": ledger["original_production_steps"],
               "effective_production_steps": request["production_steps"]}
    write_json(generation / "summary.json", summary)
    # Hash the published scientific payload, not mutable shared checkpoints.
    names = ["generation.json", "summary.json", "preparation.json", "rate_pmf.json",
             "rate_pmf.npz", "rate_pmf.csv", "recrossing.npz", "recrossing.csv",
             "umbrella/fes.npz", "umbrella/fes.h5", "umbrella/fes.csv",
             "umbrella/summary.json"]
    names.extend(str(file.relative_to(generation)) for file in sorted(
        (generation / "umbrella" / "checkpoints").glob("window_*_replica_*.npz")))
    receipt = {"format": RECEIPT_FORMAT, "generation": request["generation"],
               "physics_fingerprint": ledger["physics_fingerprint"],
               "production_steps": request["production_steps"],
               "artifacts": {name: file_sha256(generation / name)
                             for name in names if (generation / name).is_file()}}
    write_json(generation / "receipt.json", receipt)
    request.update(status="completed", receipt_sha256=file_sha256(generation / "receipt.json"),
                   completed_at=_now())
    # Root summary commits first. If ledger publication then fails, recovery
    # reuses the completed receipt instead of rewriting the active generation.
    return _publish_completed(path, ledger)


def _recover_receipt(prepared):
    generation = Path(prepared["generation"])
    if not (generation / "receipt.json").is_file():
        return None
    ledger = copy.deepcopy(prepared["ledger"])
    request = ledger["requests"][-1]
    request.update(status="completed", receipt_sha256=file_sha256(generation / "receipt.json"))
    # Validation must precede publishing a receipt left by an interrupted commit.
    read_generation(prepared["output"], _pointer(request))
    source = _json(generation / "generation.json")["source_checkpoints"]
    if _checkpoint_hashes(prepared["checkpoint_records"], Path(prepared["output"])) != source:
        raise InputError("Umbrella checkpoints differ from the completed rate generation")
    request.setdefault("completed_at", _now())
    return _publish_completed(prepared["output"], ledger)


def extend_rate(directory, *, production_steps):
    """Continue every existing umbrella stream to a larger total production target.

    Equilibration and all scientific settings remain unchanged. Each larger
    target receives fresh constrained parents and children initialized from the
    extended PMF samples. Equal targets recover or return the same generation.
    """
    target = fes.checked_integer(production_steps, "production_steps")
    path = Path(directory).resolve()
    if path.name == "checkpoints":
        path = path.parent
    root_call(_lock, path)
    try:
        prepared = root_call(_prepare_extension, path, target)
        if prepared["completed"]:
            return root_call(_publish_completed, path, prepared["ledger"])
        recovered = root_call(_recover_receipt, prepared)
        if recovered is not None:
            return recovered
        config, system = prepared["config"], prepared["system"]
        # Preparation runs on root. MPI workers also need the saved import
        # directory for module factories and file plugins' sibling imports.
        if config.get("_base_directory") and config["_base_directory"] not in sys.path:
            sys.path.insert(0, config["_base_directory"])
        generation = Path(prepared["generation"])
        root_call(write_json, generation / "preparation.json", prepared["preparation"])
        sampled = fes.run_fes(prepared["sampling_config"], system, generation / "umbrella",
                              return_data=True, production_steps=target,
                              checkpoint_output=path / "umbrella")
        options = rate._recrossing_options(config)
        pmf, parent_tasks = root_call(rate._pmf_and_parent_tasks,
            sampled, config, system, str(generation), options)
        generation_identity = root_call(_bind_generation, prepared, pmf, parent_tasks)
        for task in parent_tasks:
            task["generation_identity"] = generation_identity

        def dispatch(function, tasks):
            return run_tasks(function, tasks, config.get("parallel", {}))

        if parent_tasks:
            parents = dispatch(rate._parent_task, parent_tasks)
            tasks = root_call(rate._child_tasks, parents, config, system,
                              str(generation), options, pmf)
            children = rate._dispatch_children(tasks, config, dispatch)
        else:
            children = []
        summary = root_call(rate._aggregate, children, pmf, config, str(generation), options)
        return root_call(_finish_extension, prepared, summary)
    finally:
        root_call(_unlock, path)
