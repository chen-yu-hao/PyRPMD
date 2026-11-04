"""Distributed OPES workflow helpers."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np

from . import _opes_serial as _serial
for _name, _value in vars(_serial).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

from ..analysis.fes_result import FESResult, write_fes_result
from ..checkpoint import load_checkpoint
from ..opes import SharedOPESBias
from ..walker_exchange import (
    AsyncWalkerStore, allgather_centers, ensure_manifest, make_record, mpi_world,
    read_committed_records, validate_record)

MPI_KIND = "PyRPMD.shared-opes.mpi.v1"
ASYNC_KIND = "PyRPMD.shared-opes.async-walker.v1"

from ._distributed_opes_common import (
    _analyze_shared, _new_shared_bias, _shared_fingerprint,
    _validate_shared_options)

def _async_identity(config):
    walkers = config["sampling"]["replicas"]
    raw = os.environ.get("PYRPMD_WALKER_ID")
    if raw is not None:
        try:
            walker = int(raw)
        except ValueError as error:
            raise ValueError("PYRPMD_WALKER_ID must be an integer") from error
        if not 0 <= walker < walkers:
            raise ValueError("PYRPMD_WALKER_ID is outside sampling.replicas")
        declared = os.environ.get("PYRPMD_WALKERS")
        if declared is not None and int(declared) != walkers:
            raise ValueError("PYRPMD_WALKERS differs from sampling.replicas")
        return walker
    try:
        comm = mpi_world()
    except RuntimeError as error:
        raise RuntimeError("async-files requires PYRPMD_WALKER_ID or MPI") from error
    if comm.Get_size() != walkers:
        raise ValueError("async-files MPI size must equal sampling.replicas")
    return comm.Get_rank()


def _async_path(output, walker):
    return Path(output) / "checkpoints" / f"opes_shared_async_{walker:04d}.npz"


def _read_async(config, system, output, walker, cv, options):
    path = _async_path(output, walker)
    with load_checkpoint(path, expected_kind=ASYNC_KIND, require_integrity=True) as data:
        if not fingerprint_matches(
                data["fingerprint"], checkpoint_cv_config(config), system,
                {"opes_shared": 1, "walkers_mode": "async-files", "walker": walker}):
            raise ValueError("asynchronous OPES checkpoint physics differs")
        elapsed_array = np.asarray(data["elapsed"])
        if (elapsed_array.shape != () or elapsed_array.dtype.kind not in "iu"
                or not 0 <= int(elapsed_array) <= options["neq"] + options["nprod"]):
            raise ValueError("asynchronous OPES checkpoint has an invalid clock")
        elapsed = int(elapsed_array)
        if int(np.asarray(data["step"])) != elapsed:
            raise ValueError("asynchronous OPES state step disagrees with clock")
        dt = config["sampling"].get("dt_au", config["dt_au"])
        if not np.isclose(float(data["time"]), elapsed * dt, rtol=1e-10, atol=1e-12):
            raise ValueError("asynchronous OPES time disagrees with clock")
        shape = (config["beads"], len(system.masses), 3)
        if (data["q"].shape != shape or data["p"].shape != shape
                or not np.isfinite(data["q"]).all() or not np.isfinite(data["p"]).all()):
            raise ValueError("asynchronous OPES state shape or values differ from system")
        expected = max(0, elapsed - options["neq"]) // options["interval"]
        samples = data["samples"].copy()
        steps = data["sample_steps"].copy()
        sampled_bias = data["sample_bias"].copy()
        energies = data["sample_energies"].copy()
        expected_steps = (options["neq"] + np.arange(1, expected + 1)
                          * options["interval"])
        if (samples.shape != (expected, cv.dimension)
                or steps.shape != (expected,) or not np.array_equal(steps, expected_steps)
                or sampled_bias.shape != (expected,) or energies.shape != (expected,)
                or not np.isfinite(samples).all()
                or not np.isfinite(sampled_bias).all()
                or not np.isfinite(energies).all()):
            raise ValueError("asynchronous OPES samples disagree with clock")
        rng = json.loads(str(data["rng"]))
        np.random.default_rng().bit_generator.state = rng
        pending = (json.loads(str(data["pending_record"]))
                   if "pending_record" in data else None)
        if pending is not None:
            pending = validate_record(pending, method="opes", walker=walker)
        bias_state = json.loads(str(data["bias_state"]))
        _new_shared_bias(
            cv, options, config["beta"], config["sampling"]["replicas"],
            bias_state)
        return {"elapsed": elapsed, "q": data["q"].copy(), "p": data["p"].copy(),
                "rng": rng, "bias_state": bias_state,
                "exchange": json.loads(str(data["exchange"])),
                "pending_record": pending, "samples": list(samples),
                "sample_steps": list(steps), "sample_bias": list(sampled_bias),
                "sample_energies": list(energies),
                "propagation": json.loads(str(data["propagation"])),
                "operator_state": json.loads(str(data["operator_state"])),
                "checkpoint": str(path)}


def _collect_async(config, system, output, cv, options, *, return_data):
    walkers = config["sampling"]["replicas"]
    streams = [_read_async(config, system, output, index, cv, options)
               for index in range(walkers)]
    records = []
    log_dir = Path(output) / "walkers" / "opes"
    ensure_manifest(
        log_dir, {"schema_version": 1, "method": "opes",
                  "walkers": walkers,
                  "fingerprint": _shared_fingerprint(
                      config, system, "async-files")})
    for index in range(walkers):
        local, _ = read_committed_records(
            log_dir / f"walker-{index:04d}.jsonl", 0,
            method="opes", walker=index)
        records.extend(local)
    committed = {(record["event"]["walker"], record["event"]["sequence"]): record
                 for record in records}
    for index, stream in enumerate(streams):
        expected = max(0, stream["elapsed"] - options["neq"]) // options["pace"]
        owned = [record for record in records if record["event"]["walker"] == index]
        if len(owned) != expected:
            raise ValueError(
                "asynchronous OPES log count disagrees with its walker checkpoint; resume that walker")
        for raw in stream["bias_state"]["records"]:
            saved_record = validate_record(raw, method="opes")
            identity = (saved_record["event"]["walker"],
                        saved_record["event"]["sequence"])
            if committed.get(identity) != saved_record:
                raise ValueError(
                    "asynchronous OPES checkpoint references a missing or conflicting log record")
        pending = stream["pending_record"]
        if pending is not None:
            identity = (pending["event"]["walker"], pending["event"]["sequence"])
            if committed.get(identity) != pending:
                raise ValueError(
                    "asynchronous OPES has an unpublished pending record; resume that walker")
    bias = _new_shared_bias(cv, options, config["beta"], walkers)
    bias.merge_records(records)
    clocks = [np.asarray(stream["sample_steps"]) for stream in streams]
    if any(not np.array_equal(clocks[0], clock) for clock in clocks[1:]):
        raise ValueError("asynchronous OPES sample clocks differ")
    samples = np.stack([np.asarray(stream["samples"]) for stream in streams], axis=1)
    complete = all(stream["elapsed"] == options["neq"] + options["nprod"]
                   for stream in streams)
    return _analyze_shared(
        bias, samples, config, system, output, options, mode="async-files",
        checkpoint_files=[stream["checkpoint"] for stream in streams],
        complete=complete, return_data=return_data)


