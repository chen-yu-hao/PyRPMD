"""Shared scheduling and transactional propagation for biased CV sampling.

Algorithms supply a bias update policy and their reconstruction.  This module
owns the integration/sample clock, CV observations, thermostat, rollback and
checkpoint boundaries.  Fixed biases retain the native block fast path.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from ..block_propagation import PropagationError
from ..parallel import run_tasks
from ..thermostats import NoseHooverChain, nhc_options
from ..units import parse_quantity


@dataclass
class SamplingRuntime:
    state: object
    elapsed: int = 0
    values: list = field(default_factory=list)
    clocks: list = field(default_factory=list)
    biases: list = field(default_factory=list)
    energies: list = field(default_factory=list)
    chain: object = None


def dispatch_replicas(config, system, output, worker, executor=None):
    """Schedule independent stream identities without changing their seeds."""
    tasks = [{"config": copy.deepcopy(config), "system": system,
              "output": str(output), "replica": replica}
             for replica in range(config.get("sampling", {}).get("replicas", 1))]
    return dispatch_tasks(worker, tasks, config, executor)


def dispatch_tasks(worker, tasks, config, executor=None):
    """Use the same serial/process/MPI scheduler for windows and walkers."""
    return (executor.map(worker, tasks) if executor is not None else
            run_tasks(worker, tasks, config.get("parallel", {})))


def _fixed_step(runtime, dynamics, cv, bias, *, neq, total, interval,
                checkpoint_steps, sampling, sampler, fused):
    """Advance an unchanged bias, retaining already committed native steps."""
    state = runtime.state
    andersen = sampling.get("andersen_interval", 10)
    initial_elapsed = runtime.elapsed
    thermostat_p = thermostat_rng = None
    if sampler.thermostat == "andersen" and state.step % andersen == 0:
        thermostat_p = state.p.copy()
        thermostat_rng = copy.deepcopy(dynamics.rng.bit_generator.state)
    try:
        if thermostat_p is not None:
            state.p[:] = dynamics.sample_momenta()
        if fused:
            count = min(dynamics.block_steps, 4096, total - runtime.elapsed,
                        andersen - state.step % andersen,
                        checkpoint_steps - runtime.elapsed % checkpoint_steps)
            production = runtime.elapsed + np.arange(1, count + 1) - neq
            observe = (production > 0) & (production % interval == 0)
            block = dynamics.propagate_block(
                state, count, bias=bias, cv=cv, observe=observe,
                check_energy=True, raise_on_error=False,
            )
            runtime.elapsed += block.completed
            runtime.values.extend(block.values.copy())
            runtime.energies.extend(block.energies.tolist())
            if block.error is not None:
                raise PropagationError(block)
        else:
            previous = state.copy()
            previous_rng = copy.deepcopy(dynamics.rng.bit_generator.state)
            try:
                sampler.step(dynamics, state, bias)
                production = runtime.elapsed + 1 - neq
                record = production > 0 and production % interval == 0
                if record:
                    value = np.asarray(cv.value(state.q), dtype=float)
                    if value.shape != (cv.dimension,) or not np.isfinite(value).all():
                        raise ValueError("CV returned invalid observation")
                    energy = float(dynamics.energy(state, bias))
                    if not np.isfinite(energy):
                        raise FloatingPointError("nonfinite total ring energy")
            except Exception:
                state.q[:] = previous.q
                state.p[:] = previous.p
                state.step, state.time = previous.step, previous.time
                dynamics.rng.bit_generator.state = previous_rng
                raise
            if record:
                runtime.values.append(value.copy())
                runtime.energies.append(energy)
            runtime.elapsed += 1
            dynamics._record_python_steps(1)
    except Exception:
        if runtime.elapsed == initial_elapsed and thermostat_p is not None:
            state.p[:] = thermostat_p
            dynamics.rng.bit_generator.state = thermostat_rng
        raise


def _history_step(runtime, dynamics, cv, update, *, neq, interval, sampling,
                  active_bias, energy_bias, fallback_reason):
    """Integrate with frozen history, observe, then commit a bias update."""
    state = runtime.state
    previous = state.copy()
    old_rng = copy.deepcopy(dynamics.rng.bit_generator.state)
    old_stats = copy.deepcopy(dynamics.propagation_stats)
    old_bath = runtime.chain.state_dict() if runtime.chain else None
    production = runtime.elapsed + 1 - neq
    record = production > 0 and production % interval == 0
    due = production > 0 and update.due(production)
    old_bias = update.snapshot() if due else None
    try:
        if (runtime.chain is not None and runtime.elapsed == neq
                and sampling.get("thermostat_params", {}).get("reset_after_equilibration", False)):
            runtime.chain = NoseHooverChain(dynamics, **nhc_options(sampling))
        force_bias = active_bias(runtime.elapsed, update.bias) if active_bias else update.bias
        thermostat = sampling.get("thermostat", "andersen")
        if thermostat == "andersen" and state.step % sampling.get("andersen_interval", 10) == 0:
            state.p[:] = dynamics.sample_momenta()
        if runtime.chain is not None:
            runtime.chain.step(state, force_bias)
        elif thermostat == "pile-l":
            gamma = parse_quantity(sampling.get("gamma_centroid", "0.01 fs^-1"), "frequency")
            dynamics.sample_step(state, force_bias, gamma_centroid=gamma)
        else:
            dynamics.step(state, force_bias)
        value = cv.value(state.q) if record or due else None
        observed = float(update.bias.physical_energy(value)) if record else None
        energy = float(dynamics.energy(state, energy_bias(update.bias) if energy_bias else update.bias)) if record else None
        if record and not np.isfinite([observed, energy]).all():
            raise ValueError("biased sampling observables must be finite")
        if due:
            update.apply(value, state)
    except BaseException:
        runtime.state = previous
        dynamics.rng.bit_generator.state = old_rng
        dynamics.propagation_stats = old_stats
        if runtime.chain is not None:
            runtime.chain.load_state_dict(old_bath)
        if due:
            update.restore(old_bias)
        raise
    runtime.elapsed += 1
    dynamics._record_python_steps(1)
    if fallback_reason:
        reasons = dynamics.propagation_stats.setdefault("fallback_reasons", {})
        reasons[fallback_reason] = reasons.get(fallback_reason, 0) + 1
    if record:
        runtime.values.append(value.copy())
        runtime.clocks.append(state.step)
        runtime.biases.append(observed)
        runtime.energies.append(energy)


def run_biased_sampling(runtime, dynamics, cv, bias=None, *, neq, nprod, interval,
                        checkpoint_steps, sampling, save=None, update=None,
                        sampler=None, active_bias=None, energy_bias=None,
                        fallback_reason=None, initial_save=True, final_save=True,
                        stop=None):
    """Run or resume one stream up to a checkpoint/synchronization boundary.

    ``update`` has ``bias``, ``due``, ``snapshot``, ``apply`` and ``restore``;
    a fixed bias instead supplies the existing checked ``sampler``.  ``save``
    receives the current runtime and must atomically commit its state/history.
    ``stop`` is an absolute clock used by the multiple-walker block scheduler.
    """
    total = neq + nprod if stop is None else min(stop, neq + nprod)
    fused = (update is None and sampler.thermostat == "andersen"
             and dynamics.can_fuse(bias=bias, observe=cv))
    if initial_save and save is not None:
        save(runtime)
    while runtime.elapsed < total:
        try:
            if update is None:
                _fixed_step(runtime, dynamics, cv, bias, neq=neq, total=total,
                            interval=interval, checkpoint_steps=checkpoint_steps,
                            sampling=sampling, sampler=sampler, fused=fused)
            else:
                _history_step(runtime, dynamics, cv, update, neq=neq,
                              interval=interval, sampling=sampling,
                              active_bias=active_bias, energy_bias=energy_bias,
                              fallback_reason=fallback_reason)
        except BaseException as error:
            # Fixed-bias propagation historically rolls back ordinary
            # Exceptions, while an asynchronous interrupt may land inside an
            # uncommitted Python/native map. Preserve the last durable prefix
            # in that case. History-dependent steps explicitly roll back all
            # BaseExceptions and can safely commit their restored state.
            if save is not None and (update is not None or isinstance(error, Exception)):
                save(runtime)
            raise
        if save is not None and runtime.elapsed % checkpoint_steps == 0:
            save(runtime)
    if final_save and save is not None:
        save(runtime)
    return runtime
