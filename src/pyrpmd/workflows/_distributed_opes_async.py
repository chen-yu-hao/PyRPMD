"""Distributed OPES workflow helpers."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import numpy as np
from ..checkpoint import atomic_savez
from ..dynamics import Dynamics, State
from ..parallel import make_task_potential
from ..propagation import loop_options
from ..walker_exchange import AsyncWalkerStore, make_record
from .opes import resolve_opes, _distributed_fes_sampler as fes_sampler

MPI_KIND = "PyRPMD.shared-opes.mpi.v1"
ASYNC_KIND = "PyRPMD.shared-opes.async-walker.v1"

from ._distributed_opes_common import (
    _analyze_shared, _new_shared_bias, _shared_fingerprint,
    _validate_shared_options)
from ._distributed_opes_async_io import (
    _async_identity, _async_path, _collect_async, _read_async)

def _run_async(config, system, output, *, return_data=False):
    cv, _, options = resolve_opes(config, system)
    walkers = _validate_shared_options(config, options)
    walker = _async_identity(config)
    sampler = fes_sampler(config)
    checkpoint = _async_path(output, walker)
    restored = (_read_async(config, system, output, walker, cv, options)
                if checkpoint.exists() else None)
    provider = make_task_potential(config["potential"], system)
    try:
        dynamics = Dynamics(
            provider, system.masses, config["beta"], config["beads"],
            config["sampling"].get("dt_au", config["dt_au"]),
            seed=np.random.SeedSequence([int(config.get("seed", 0)), 811, walker]),
            native=config.get("native", config.get("dynamics", {}).get("native", True)),
            threads=config.get("parallel", {}).get("threads_per_rank", 1),
            **loop_options(config))
        if restored is None:
            state = State(np.repeat(system.geometry[None], config["beads"], axis=0),
                          dynamics.sample_momenta())
            bias = _new_shared_bias(cv, options, config["beta"], walkers)
            elapsed = 0
            saved = {"samples": [], "sample_steps": [], "sample_bias": [],
                     "sample_energies": []}
            exchange_state = None
            pending_record = None
        else:
            elapsed = restored["elapsed"]
            state = State(restored["q"], restored["p"], elapsed,
                          elapsed * config["sampling"].get("dt_au", config["dt_au"]))
            dynamics.rng.bit_generator.state = restored["rng"]
            dynamics.propagation_stats = restored["propagation"]
            if sampler.operator_checkpoint and restored["operator_state"] is not None:
                dynamics.load_operator_state_dict(restored["operator_state"])
            bias = _new_shared_bias(
                cv, options, config["beta"], walkers, restored["bias_state"])
            saved = {key: restored[key] for key in (
                "samples", "sample_steps", "sample_bias", "sample_energies")}
            exchange_state = restored["exchange"]
            pending_record = restored.get("pending_record")
        log_dir = Path(output) / "walkers" / "opes"
        with AsyncWalkerStore(
                log_dir, method="opes", walker=walker,
                walkers=walkers, state=exchange_state,
                identity={"schema_version": 1, "method": "opes",
                          "walkers": walkers,
                          "fingerprint": _shared_fingerprint(
                              config, system, "async-files")}) as exchange:
            if restored is None and exchange.next_sequence:
                raise ValueError(
                    "asynchronous OPES log exists without its walker checkpoint")
            bias.merge_records(exchange.poll().records)

            def save():
                atomic_savez(
                    checkpoint, kind=ASYNC_KIND,
                    fingerprint=_shared_fingerprint(
                        config, system, "async-files", walker=walker),
                    elapsed=elapsed, step=elapsed, time=state.time,
                    q=state.q, p=state.p,
                    rng=json.dumps(dynamics.rng.bit_generator.state),
                    bias_state=json.dumps(bias.state_dict(), allow_nan=False),
                    exchange=json.dumps(exchange.state_dict(), allow_nan=False),
                    pending_record=json.dumps(pending_record, allow_nan=False),
                    samples=np.asarray(saved["samples"]).reshape(-1, cv.dimension),
                    sample_steps=np.asarray(saved["sample_steps"], dtype=np.int64),
                    sample_bias=np.asarray(saved["sample_bias"]),
                    sample_energies=np.asarray(saved["sample_energies"]),
                    propagation=json.dumps(dynamics.propagation_stats),
                    operator_state=json.dumps(
                        dynamics.operator_state_dict()
                        if sampler.operator_checkpoint else None))

            if pending_record is not None:
                exchange.ensure_record(pending_record)
                bias.merge_records(exchange.poll().records)
                pending_record = None
                save()
            else:
                save()
            while elapsed < options["neq"] + options["nprod"]:
                bias.merge_records(exchange.poll().records)
                previous = state.copy()
                old_rng = copy.deepcopy(dynamics.rng.bit_generator.state)
                old_stats = copy.deepcopy(dynamics.propagation_stats)
                old_operator = (dynamics.operator_state_dict()
                                if sampler.operator_checkpoint else None)
                production = elapsed + 1 - options["neq"]
                record = production > 0 and production % options["interval"] == 0
                due = production > 0 and production % options["pace"] == 0
                try:
                    sampler.step(dynamics, state, bias)
                    value = (np.asarray(cv.value(state.q), dtype=np.float64)
                             if record or due else None)
                    if value is not None and (value.shape != (cv.dimension,)
                                              or not np.isfinite(value).all()):
                        raise ValueError("CV returned invalid OPES observation")
                    if record:
                        observed = float(bias.physical_energy(value))
                        energy = float(dynamics.energy(state, bias))
                        if not np.isfinite([observed, energy]).all():
                            raise ValueError("asynchronous OPES observables must be finite")
                except BaseException:
                    state = previous
                    dynamics.rng.bit_generator.state = old_rng
                    dynamics.propagation_stats = old_stats
                    if old_operator is not None:
                        dynamics.load_operator_state_dict(old_operator)
                    save()
                    raise
                elapsed += 1
                dynamics._record_python_steps(1)
                if record:
                    saved["samples"].append(value.copy())
                    saved["sample_steps"].append(elapsed)
                    saved["sample_bias"].append(observed)
                    saved["sample_energies"].append(energy)
                if due:
                    payload = bias.prepare_local_payload(value)
                    pending_record = make_record(
                        "opes", walker, exchange.next_sequence,
                        step=state.step, time=state.time, payload=payload)
                    save()
                    exchange.ensure_record(pending_record)
                    bias.merge_records(exchange.poll().records)
                    pending_record = None
                    save()
                elif elapsed % config.get("output", {}).get(
                        "checkpoint_steps", 1000) == 0:
                    save()
            bias.merge_records(exchange.poll().records)
            save()
        if all(_async_path(output, index).exists() for index in range(walkers)):
            streams = [_read_async(config, system, output, index, cv, options)
                       for index in range(walkers)]
            if all(stream["elapsed"] == options["neq"] + options["nprod"]
                   for stream in streams):
                return _collect_async(
                    config, system, output, cv, options, return_data=return_data)
        return {"task": "opes", "status": "walker_complete_pending_merge",
                "walkers_mode": "async-files", "walker": walker,
                "checkpoint": str(checkpoint), "converged": False,
                "physical_converged": False}
    finally:
        if hasattr(provider, "close"):
            provider.close()


