"""Distributed shared-metadynamics workflow helpers."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np

from . import _shared_metadynamics_serial as _serial
for _name, _value in vars(_serial).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

from ..checkpoint import atomic_savez, load_checkpoint
from ..shared_metadynamics import AsynchronousMetadynamicsBias
from ..walker_exchange import (
    AsyncWalkerStore, allgather_centers, ensure_manifest, make_record, mpi_world,
    read_committed_records, validate_record)

MPI_KIND = "PyRPMD.shared-metadynamics.mpi.v1"
ASYNC_KIND = "PyRPMD.shared-metadynamics.async-walker.v1"

def _new_async_bias(cv, config, options, records=None):
    return AsynchronousMetadynamicsBias(
        cv, options["sigma"], options["height"], config["beta"],
        walkers=config["sampling"]["replicas"], mode=options["mode"],
        bias_factor=options["bias_factor"], records=records)


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
        raise RuntimeError(
            "async-files requires PYRPMD_WALKER_ID or an MPI launch") from error
    if comm.Get_size() != walkers:
        raise ValueError("async-files MPI size must equal sampling.replicas")
    return comm.Get_rank()


def _async_path(output, walker):
    return Path(output) / "checkpoints" / f"shared_metadynamics_async_{walker:04d}.npz"


def _read_async_walker(config, system, output, walker, cv, options):
    path = _async_path(output, walker)
    with load_checkpoint(path, expected_kind=ASYNC_KIND, require_integrity=True) as data:
        if not fingerprint_matches(
                data["fingerprint"], config, system,
                {"shared_metadynamics": 2, "walkers_mode": "async-files",
                 "walker": walker}):
            raise ValueError("asynchronous metadynamics checkpoint physics differs")
        elapsed_array = np.asarray(data["elapsed"])
        if (elapsed_array.shape != () or elapsed_array.dtype.kind not in "iu"
                or not 0 <= int(elapsed_array) <= options["neq"] + options["nprod"]):
            raise ValueError("invalid asynchronous metadynamics clock")
        elapsed = int(elapsed_array)
        shape = (config["beads"], system.natoms, 3)
        if (data["q"].shape != shape or data["p"].shape != shape
                or not np.isfinite(data["q"]).all() or not np.isfinite(data["p"]).all()):
            raise ValueError("asynchronous metadynamics state shape or values mismatch")
        samples = data["samples"].copy()
        steps = data["sample_steps"].copy()
        biases = data["sample_bias"].copy()
        expected = max(0, elapsed - options["neq"]) // options["interval"]
        if (samples.shape != (expected, cv.dimension) or steps.shape != (expected,)
                or biases.shape != (expected,) or not np.isfinite(samples).all()
                or not np.isfinite(biases).all()):
            raise ValueError("invalid asynchronous metadynamics samples")
        expected_steps = (options["neq"] + np.arange(1, expected + 1)
                          * options["interval"])
        if not np.array_equal(steps, expected_steps):
            raise ValueError("invalid asynchronous metadynamics sample clock")
        dt = config["sampling"].get("dt_au", config["dt_au"])
        if not np.isclose(float(data["time"]), elapsed * dt, rtol=1e-10, atol=1e-12):
            raise ValueError("asynchronous metadynamics time disagrees with clock")
        configurations = data["configurations"].copy()
        expected_shape = ((expected, *shape) if options["store_configurations"]
                          else (0, *shape))
        if (configurations.shape != expected_shape
                or not np.isfinite(configurations).all()):
            raise ValueError("invalid asynchronous metadynamics configurations")
        rng = json.loads(str(data["rng"]))
        np.random.default_rng().bit_generator.state = rng
        pending = (json.loads(str(data["pending_record"]))
                   if "pending_record" in data else None)
        if pending is not None:
            pending = validate_record(
                pending, method="metadynamics", walker=walker)
        records = json.loads(str(data["records"]))
        _new_async_bias(cv, config, options, records=records)
        return {"elapsed": elapsed, "q": data["q"].copy(), "p": data["p"].copy(),
                "rng": rng, "records": records,
                "exchange": json.loads(str(data["exchange"])),
                "pending_record": pending,
                "samples": list(samples), "sample_steps": list(steps),
                "sample_bias": list(biases),
                "configurations": list(configurations), "checkpoint": str(path)}


def _collect_async(config, system, output, cv, options, *, return_data):
    walkers = config["sampling"]["replicas"]
    streams = [_read_async_walker(config, system, output, index, cv, options)
               for index in range(walkers)]
    records = []
    log_dir = Path(output) / "walkers" / "metadynamics"
    ensure_manifest(
        log_dir, {"schema_version": 1, "method": "metadynamics",
                  "walkers": walkers,
                  "fingerprint": _fingerprint(
                      config, system,
                      {"shared_metadynamics": 2,
                       "walkers_mode": "async-files"})})
    for index in range(walkers):
        local, _ = read_committed_records(
            log_dir / f"walker-{index:04d}.jsonl", 0,
            method="metadynamics", walker=index)
        records.extend(local)
    committed = {(record["event"]["walker"], record["event"]["sequence"]): record
                 for record in records}
    for index, stream in enumerate(streams):
        expected = max(0, stream["elapsed"] - options["neq"]) // options["pace"]
        owned = [record for record in records if record["event"]["walker"] == index]
        if len(owned) != expected:
            raise ValueError(
                "asynchronous metadynamics log count disagrees with its walker checkpoint; resume that walker")
        for raw in stream["records"]:
            saved_record = validate_record(raw, method="metadynamics")
            identity = (saved_record["event"]["walker"],
                        saved_record["event"]["sequence"])
            if committed.get(identity) != saved_record:
                raise ValueError(
                    "asynchronous metadynamics checkpoint references a missing or conflicting log record")
        pending = stream["pending_record"]
        if pending is not None:
            identity = (pending["event"]["walker"], pending["event"]["sequence"])
            if committed.get(identity) != pending:
                raise ValueError(
                    "asynchronous metadynamics has an unpublished pending record; resume that walker")
    bias = _new_async_bias(cv, config, options, records=records)
    clocks = [np.asarray(stream["sample_steps"]) for stream in streams]
    if any(not np.array_equal(clocks[0], clock) for clock in clocks[1:]):
        raise ValueError("asynchronous walker sample clocks differ")
    samples = np.stack([np.asarray(stream["samples"]) for stream in streams], axis=1)
    sampled_bias = np.stack(
        [np.asarray(stream["sample_bias"]) for stream in streams], axis=1)
    configurations = []
    if options["store_configurations"]:
        configurations = np.stack(
            [np.asarray(stream["configurations"]) for stream in streams], axis=1)
    ensemble = {"states": [], "rng": [], "bias": bias,
                "elapsed": min(stream["elapsed"] for stream in streams),
                "samples": list(samples), "sample_steps": list(clocks[0]),
                "sample_bias": list(sampled_bias),
                "configurations": list(configurations)}
    result = _serial._analyze(
        config, system, output, cv, options, ensemble, return_data)
    summary = result["summary"] if return_data else result
    summary["walkers_mode"] = "async-files"
    summary["physical_converged"] = False
    summary["status"] = ("finite_time_async_bias_estimate" if all(
        stream["elapsed"] == options["neq"] + options["nprod"] for stream in streams)
        else "partial_async_bias_estimate")
    summary["files"]["checkpoints"] = [stream["checkpoint"] for stream in streams]
    summary["files"]["logs"] = str(log_dir)
    summary["warnings"] = [
        warning for warning in summary["warnings"]
        if "same pre-deposition bias" not in warning]
    summary["warnings"].append(
        "Asynchronous hills retain the bias visible to their writer at publication; merged histories are eventually consistent, not trajectory-wise identical to mpi-sync.")
    write_json(Path(output) / "summary.json", summary)
    return result


