"""Distributed shared-metadynamics workflow helpers."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import numpy as np
from ..bias import CompositeBias
from ..checkpoint import atomic_savez
from ..dynamics import Dynamics, State
from ..potentials import make_potential
from ..propagation import loop_options
from ..units import parse_quantity
from ..walker_exchange import AsyncWalkerStore, make_record
from .common import _fingerprint
from .metadynamics import resolve_metadynamics, resolve_walls
from .shared_metadynamics import ASYNC_KIND, validate_shared

from ._distributed_meta_async_io import (
    _async_identity, _async_path, _collect_async, _new_async_bias,
    _read_async_walker)

def _run_async(config, system, output, *, return_data=False):
    cv, _, options = resolve_metadynamics(config, system)
    checked = copy.deepcopy(config)
    checked["parallel"] = {}
    walkers = validate_shared(checked, options)
    walker = _async_identity(config)
    checkpoint = _async_path(output, walker)
    restored = (_read_async_walker(config, system, output, walker, cv, options)
                if checkpoint.exists() else None)
    provider = make_potential(config["potential"], system)
    try:
        dynamics = Dynamics(
            provider, system.masses, config["beta"], config["beads"],
            config["sampling"].get("dt_au", config["dt_au"]),
            seed=np.random.SeedSequence([config.get("seed", 0), 2267, walker]),
            native=config.get("native", config.get("dynamics", {}).get("native", True)),
            threads=config.get("parallel", {}).get("threads_per_rank", 1),
            fixed_atoms=system.fixed_atoms, fixed_positions=system.geometry,
            **loop_options(config))
        if restored is None:
            state = State(np.repeat(system.geometry[None], config["beads"], axis=0),
                          dynamics.sample_momenta())
            bias, elapsed = _new_async_bias(cv, config, options), 0
            saved = {"samples": [], "sample_steps": [], "sample_bias": [],
                     "configurations": []}
            exchange_state = None
            pending_record = None
        else:
            state = State(restored["q"], restored["p"], restored["elapsed"],
                          restored["elapsed"] * config["sampling"].get(
                              "dt_au", config["dt_au"]))
            dynamics.integrator._check(state.q, state.p)
            dynamics._check_fixed(state.q, state.p)
            dynamics.rng.bit_generator.state = restored["rng"]
            bias = _new_async_bias(cv, config, options, restored["records"])
            elapsed = restored["elapsed"]
            saved = {key: restored[key] for key in (
                "samples", "sample_steps", "sample_bias", "configurations")}
            exchange_state = restored["exchange"]
            pending_record = restored.get("pending_record")
        walls = resolve_walls(config, system)
        total_bias = CompositeBias(bias, walls)
        log_dir = Path(output) / "walkers" / "metadynamics"
        with AsyncWalkerStore(
                log_dir, method="metadynamics", walker=walker,
                walkers=walkers, state=exchange_state,
                identity={"schema_version": 1, "method": "metadynamics",
                          "walkers": walkers,
                          "fingerprint": _fingerprint(
                              config, system,
                              {"shared_metadynamics": 2,
                               "walkers_mode": "async-files"})}) as exchange:
            if restored is None and exchange.next_sequence:
                raise ValueError(
                    "asynchronous metadynamics log exists without its walker checkpoint")
            bias.merge_records(exchange.poll().records)

            def save():
                shape = (config["beads"], system.natoms, 3)
                atomic_savez(
                    checkpoint, kind=ASYNC_KIND,
                    fingerprint=_fingerprint(
                        config, system,
                        {"shared_metadynamics": 2, "walkers_mode": "async-files",
                         "walker": walker}),
                    elapsed=elapsed, time=state.time, q=state.q, p=state.p,
                    rng=json.dumps(dynamics.rng.bit_generator.state),
                    records=json.dumps(bias.records, allow_nan=False),
                    exchange=json.dumps(exchange.state_dict(), allow_nan=False),
                    pending_record=json.dumps(pending_record, allow_nan=False),
                    sample_steps=np.asarray(saved["sample_steps"], dtype=np.int64),
                    samples=np.asarray(saved["samples"]).reshape(-1, cv.dimension),
                    sample_bias=np.asarray(saved["sample_bias"]),
                    configurations=np.asarray(saved["configurations"]).reshape(-1, *shape))

            if pending_record is not None:
                exchange.ensure_record(pending_record)
                bias.merge_records(exchange.poll().records)
                pending_record = None
                save()
            else:
                save()
            sampling = config["sampling"]
            while elapsed < options["neq"] + options["nprod"]:
                bias.merge_records(exchange.poll().records)
                previous = state.copy()
                rng = copy.deepcopy(dynamics.rng.bit_generator.state)
                try:
                    active = (total_bias if elapsed >= options["neq"] or
                              config["metadynamics"].get(
                                  "walls_during_equilibration", True) else bias)
                    if sampling.get("thermostat", "andersen") == "pile-l":
                        dynamics.sample_step(
                            state, active,
                            gamma_centroid=parse_quantity(
                                sampling.get("gamma_centroid", "0.01 fs^-1"),
                                "frequency"), method=sampling.get("integrator"))
                    else:
                        if state.step % sampling.get("andersen_interval", 10) == 0:
                            state.p[:] = dynamics.sample_momenta()
                        dynamics.step(state, active)
                except BaseException:
                    state = previous
                    dynamics.rng.bit_generator.state = rng
                    save()
                    raise
                production = elapsed + 1 - options["neq"]
                record = production > 0 and production % options["interval"] == 0
                deposit = production > 0 and production % options["pace"] == 0
                value = np.asarray(cv.value(state.q), dtype=np.float64) if record or deposit else None
                if record:
                    saved["samples"].append(value.copy())
                    saved["sample_steps"].append(elapsed + 1)
                    saved["sample_bias"].append(float(bias.physical_energy(value)))
                    if options["store_configurations"]:
                        saved["configurations"].append(state.q.copy())
                elapsed += 1
                if deposit:
                    # Two-phase publication: persist the exact prepared record
                    # with the post-step state, then idempotently ensure it is
                    # in the append-only log.  Every crash window resumes
                    # without losing or duplicating this deposition.
                    payload = bias.prepare_local_payload(value)
                    pending_record = make_record(
                        "metadynamics", walker, exchange.next_sequence,
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
        # The last finishing process performs aggregate analysis. Earlier
        # finishers return an honest pending status without waiting for peers.
        if all(_async_path(output, index).exists() for index in range(walkers)):
            streams = [_read_async_walker(
                config, system, output, index, cv, options) for index in range(walkers)]
            if all(stream["elapsed"] == options["neq"] + options["nprod"]
                   for stream in streams):
                return _collect_async(
                    config, system, output, cv, options, return_data=return_data)
        return {"task": "metadynamics", "status": "walker_complete_pending_merge",
                "walkers_mode": "async-files", "walker": walker,
                "checkpoint": str(checkpoint), "physical_converged": False,
                "converged": False}
    finally:
        if hasattr(provider, "close"):
            provider.close()


