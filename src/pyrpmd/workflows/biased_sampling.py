"""Shared scheduling and transactional propagation for biased CV sampling.

Algorithms supply a bias update policy and their reconstruction.  This module
owns the integration/sample clock, CV observations, thermostat, rollback and
checkpoint boundaries.  Fixed biases retain the native block fast path.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from ..block_propagation import PropagationError, NotNative, _native_plan
from ..momentum_refresh import plan_refresh
from ..sampling_block import propagate_sampling_block, _eligibility as _sampling_native_eligibility
from ..dynamics import State
from ..parallel import run_tasks
from ..thermostats import NoseHooverChain, nhc_options
from ..trajectory_batch import map_trajectory_tasks
from ..units import parse_quantity
from .sampling import AndersenSchedule


@dataclass
class SamplingRuntime:
    state: object
    elapsed: int = 0
    values: list = field(default_factory=list)
    clocks: list = field(default_factory=list)
    biases: list = field(default_factory=list)
    energies: list = field(default_factory=list)
    chain: object = None
    configurations: list | None = None
    collision_clock: AndersenSchedule | None = None
    operator_checkpoint: bool = False
    # R10 execution-contract marker. A missing marker means a legacy
    # checkpoint and must retain the scalar history path byte-for-byte.
    native_path_contract: int = 0
    # Optional pyrpmd.analysis.centroid_path.CentroidAccumulator. It sees the
    # bead positions of every recorded production sample (fixed bias only).
    centroid_accumulator: object | None = None


def _pair_distance_report(q):
    """Smallest bead and centroid interatomic distances of a ring polymer."""
    q = np.asarray(q, float)
    if q.ndim != 3 or q.shape[1] < 2:
        return None
    i, j = np.triu_indices(q.shape[1], 1)
    bead = np.linalg.norm(q[:, i] - q[:, j], axis=-1)            # [P, pairs]
    centroid = q.mean(axis=0)
    centroid_pairs = np.linalg.norm(centroid[i] - centroid[j], axis=-1)
    flat = int(np.argmin(bead))
    b, pair = np.unravel_index(flat, bead.shape)
    return {
        "unit": "bohr",
        "minimum_bead_pair_distance": float(bead[b, pair]),
        "minimum_bead": int(b),
        "minimum_pair": [int(i[pair]), int(j[pair])],
        "per_pair_bead_minimum": {f"{int(a)}-{int(c)}": float(v)
                                  for a, c, v in zip(i, j, bead.min(axis=0))},
        "centroid_pair_distances": {f"{int(a)}-{int(c)}": float(v)
                                    for a, c, v in zip(i, j, centroid_pairs)},
    }


def _record_pair_distance(path, runtime, start):
    """Record every accepted state in an explicitly requested short preflight."""
    import json
    from pathlib import Path

    report = _pair_distance_report(runtime.state.q)
    if report is None:
        raise ValueError("pair-distance monitoring requires at least two atoms")
    minimum = report["minimum_bead_pair_distance"]
    row = {"start_step": int(start), "end_step": int(runtime.elapsed),
           "minimum_bead_pair_distance": minimum, "finite": bool(np.isfinite(minimum))}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a") as handle:
        handle.write(json.dumps(row, allow_nan=False) + "\n")


def sampling_failure_record(runtime, dynamics, cv, error, *, identity, neq, nprod,
                            bias_state=None, checkpoint=None):
    """Structured evidence for a biased history that could not continue.

    The recorded state is the last committed one: failed steps are rolled
    back and completed native steps are kept, so it is the final geometry
    that was accepted before the failing force/PES evaluation.
    """
    state = runtime.state
    record = {
        "schema_version": 1,
        "status": "history_failed",
        "history": dict(identity),
        "error": {"type": type(error).__name__, "message": str(error)},
        "state": {
            "role": "last_committed_state_before_failure",
            "elapsed_steps": int(runtime.elapsed),
            "equilibration_steps": int(neq),
            "production_steps": int(nprod),
            "production_elapsed_steps": max(0, int(runtime.elapsed) - int(neq)),
            "step": int(state.step),
            "time_au": float(state.time),
            "timestep_au": float(getattr(dynamics, "dt", float("nan"))),
            "recorded_samples": len(runtime.values),
        },
        "geometry": _pair_distance_report(state.q),
        "propagation": copy.deepcopy(dict(getattr(dynamics, "propagation_stats", {}))),
        "checkpoint": None if checkpoint is None else {
            "path": str(checkpoint), "committed_elapsed_steps": int(runtime.elapsed)},
        "partial_history_policy": (
            "This history is incomplete. The method is incomplete; completed "
            "histories must not be selected to form the final PMF."
        ),
    }
    result = getattr(error, "result", None)
    if getattr(error, "branch", None) is not None:
        record["attempted_branch"] = dict(error.branch)
    if isinstance(error, PropagationError) and result is not None:
        record["error"].update(backend=getattr(result, "backend", None),
                               completed_steps=int(getattr(result, "completed", 0)),
                               requested_steps=int(getattr(result, "requested", 0)),
                               detail=str(getattr(result, "error", "")))
    try:
        value = np.asarray(cv.value(state.q), float).reshape(-1)
        record["centroid_cv"] = value.tolist()
    except Exception as cv_error:  # report, never mask the original failure
        record["centroid_cv"] = None
        record["centroid_cv_error"] = f"{type(cv_error).__name__}: {cv_error}"
    if runtime.values:
        record["last_recorded_cv"] = np.asarray(runtime.values[-1], float).reshape(-1).tolist()
    if bias_state is not None:
        try:
            record["bias_state"] = bias_state()
        except Exception as bias_error:
            record["bias_state"] = {"error": f"{type(bias_error).__name__}: {bias_error}"}
    return record


def failure_writer(path, dynamics, cv, *, identity, neq, nprod, bias_state=None, checkpoint=None):
    """Return a ``run_biased_sampling`` failure hook writing ``path`` atomically."""
    from pathlib import Path

    from ..io import write_report_json

    def write(runtime, error):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        write_report_json(target, sampling_failure_record(
            runtime, dynamics, cv, error, identity=identity, neq=neq, nprod=nprod,
            bias_state=bias_state, checkpoint=checkpoint))

    return write


def history_failures(output):
    """Saved failure records of biased histories below ``output``."""
    import json
    from pathlib import Path

    records = []
    for path in sorted((Path(output) / "failures").glob("*.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            record = {"status": "unreadable_failure_record"}
        if record.get("status") == "superseded":
            continue
        record["file"] = str(path)
        records.append(record)
    return records


def supersede_failure(path, *, checkpoint, elapsed_steps):
    """Retain failure evidence after the same history durably completes."""
    import json
    from pathlib import Path

    from ..io import write_report_json

    target = Path(path)
    if not target.exists():
        return
    record = json.loads(target.read_text())
    if record.get("status") != "history_failed":
        return
    record["status"] = "superseded"
    record["resolution"] = {"status": "history_completed", "checkpoint": str(checkpoint),
                            "elapsed_steps": int(elapsed_steps)}
    write_report_json(target, record)


def dispatch_replicas(config, system, output, worker, executor=None, *, task_fields=None):
    """Schedule independent stream identities without changing their seeds."""
    tasks = [{"config": copy.deepcopy(config), "system": system,
              "output": str(output), "replica": replica, **(task_fields or {})}
             for replica in range(config.get("sampling", {}).get("replicas", 1))]
    return dispatch_tasks(worker, tasks, config, executor)


def dispatch_tasks(worker, tasks, config, executor=None, *, batch_trajectories=False):
    """Use the same serial/process/MPI scheduler for windows and walkers."""
    if batch_trajectories:
        return map_trajectory_tasks(worker, tasks, config, executor)
    return (executor.map(worker, tasks) if executor is not None else
            run_tasks(worker, tasks, config.get("parallel", {})))


def _fixed_step(runtime, dynamics, cv, bias, *, neq, total, interval,
                checkpoint_steps, sampling, sampler, fused):
    """Advance an unchanged bias, retaining already committed native steps."""
    state = runtime.state
    collision_clock = runtime.collision_clock
    initial_elapsed = runtime.elapsed
    thermostat_p = thermostat_rng = None
    thermostat_deadline = collision_clock.next_collision_step
    if sampler.thermostat == "andersen" and collision_clock.due(state.step):
        thermostat_p = state.p.copy()
        thermostat_rng = copy.deepcopy(dynamics.rng.bit_generator.state)
    try:
        if thermostat_p is not None:
            state.p[:] = dynamics.sample_momenta()
            collision_clock.refreshed(state.step, dynamics.rng)
        if fused:
            count = min(dynamics.block_steps, 4096, total - runtime.elapsed,
                        collision_clock.steps_until_collision(state.step),
                        checkpoint_steps - runtime.elapsed % checkpoint_steps)
            production = runtime.elapsed + np.arange(1, count + 1) - neq
            observe = (production > 0) & (production % interval == 0)
            accumulator = runtime.centroid_accumulator
            block = dynamics.propagate_block(
                state, count, bias=bias, cv=cv, observe=observe,
                check_energy=True, raise_on_error=False,
                **({"record_states": True} if accumulator is not None else {}),
            )
            runtime.elapsed += block.completed
            runtime.values.extend(block.values.copy())
            runtime.energies.extend(block.energies.tolist())
            if accumulator is not None:
                if block.q is None or len(block.q) != len(block.values):
                    raise RuntimeError("centroid path accumulation needs the recorded ring states")
                for q, value in zip(block.q, block.values):
                    accumulator.add(q, float(np.asarray(value).reshape(-1)[0]))
            if block.error is not None:
                raise PropagationError(block)
        else:
            previous = state.copy()
            previous_rng = copy.deepcopy(dynamics.rng.bit_generator.state)
            previous_operators = (dynamics.operator_state_dict()
                                  if runtime.operator_checkpoint else None)
            previous_stats = copy.deepcopy(dynamics.propagation_stats)
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
                if previous_operators is not None:
                    dynamics.load_operator_state_dict(previous_operators)
                dynamics.rng.bit_generator.state = previous_rng
                dynamics.propagation_stats = previous_stats
                raise
            if record:
                runtime.values.append(value.copy())
                runtime.energies.append(energy)
                if runtime.centroid_accumulator is not None:
                    runtime.centroid_accumulator.add(state.q, float(value[0]))
            runtime.elapsed += 1
            if (dynamics.propagation_stats["python_steps"] == previous_stats["python_steps"]
                    and dynamics.propagation_stats["native_steps"] == previous_stats["native_steps"]):
                if runtime.operator_checkpoint and getattr(dynamics, "last_step_backend", "python") == "native":
                    dynamics.propagation_stats["native_steps"] += 1
                else:
                    dynamics._record_python_steps(1)
    except Exception:
        if runtime.elapsed == initial_elapsed and thermostat_p is not None:
            state.p[:] = thermostat_p
            dynamics.rng.bit_generator.state = thermostat_rng
            collision_clock.next_collision_step = thermostat_deadline
        raise


def _history_step(runtime, dynamics, cv, update, *, neq, interval, sampling,
                  sampler, active_bias, energy_bias, fallback_reason):
    """Integrate with frozen history, observe, then commit a bias update."""
    state = runtime.state
    previous = state.copy()
    old_rng = copy.deepcopy(dynamics.rng.bit_generator.state)
    old_stats = copy.deepcopy(dynamics.propagation_stats)
    old_bath = runtime.chain.state_dict() if runtime.chain else None
    old_operators = dynamics.operator_state_dict() if runtime.operator_checkpoint else None
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
        if sampler is not None:
            sampler.step(dynamics, state, force_bias)
        elif runtime.chain is not None:
            runtime.chain.step(state, force_bias)
        elif thermostat == "pile-l":
            gamma = parse_quantity(sampling.get("gamma_centroid", "0.01 fs^-1"), "frequency")
            dynamics.sample_step(state, force_bias, gamma_centroid=gamma,
                                 method=sampling.get("integrator"))
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
        if old_operators is not None:
            dynamics.load_operator_state_dict(old_operators)
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
        if runtime.configurations is not None:
            runtime.configurations.append(state.q.copy())



def _history_block(runtime, dynamics, cv, update, *, neq, total, interval,
                   checkpoint_steps, sampling, sampler, active_bias,
                   energy_bias):
    """Run one frozen-history native block when NP7's contract is available."""
    if runtime.native_path_contract != 1:
        return False
    if runtime.chain is not None:
        return False
    thermostat = (getattr(sampler, "thermostat", None)
                  if sampler is not None else sampling.get("thermostat", "andersen"))
    thermostat_spec = getattr(sampler, "thermostat_spec", None) if sampler is not None else None
    if thermostat == "pile-l" and thermostat_spec is not None:
        return False
    if thermostat not in ("andersen", "pile-l"):
        return False
    if thermostat == "pile-l" and runtime.configurations is not None:
        return False

    state = runtime.state
    force_bias = active_bias(runtime.elapsed, update.bias) if active_bias else update.bias
    try:
        if thermostat == "andersen":
            native = getattr(getattr(dynamics.integrator, "_native", None), "MOMENTUM_REFRESH_API", 0)
            if native < 1 or runtime.collision_clock is None:
                return False
            _native_plan(dynamics, force_bias, cv, None, (), 1e-10, False, state.time)
        else:
            _sampling_native_eligibility(dynamics, force_bias, cv)
    except (NotNative, RuntimeError, ValueError, TypeError):
        return False

    remaining = total - runtime.elapsed
    if remaining <= 0:
        return False
    count = min(remaining, 4096,
                checkpoint_steps - runtime.elapsed % checkpoint_steps)
    if runtime.elapsed < neq:
        count = min(count, neq - runtime.elapsed)
    freeze = getattr(update, "options", {}).get("freeze")
    if freeze is not None:
        production = runtime.elapsed - neq
        if production < freeze:
            count = min(count, freeze - production)
    if runtime.elapsed >= neq:
        production = runtime.elapsed - neq
        next_due = ((production // update.options["pace"]) + 1) * update.options["pace"]
        if freeze is None or next_due <= freeze:
            count = min(count, next_due - production)
    if count <= 0:
        return False

    absolute_steps = state.step + np.arange(1, count + 1, dtype=np.int64)
    production = absolute_steps - neq
    observe = ((production > 0) & (production % interval == 0))
    due = np.array([production[index] > 0 and update.due(int(production[index]))
                    for index in range(count)], dtype=bool)
    observe |= due

    plan = None
    actual = count
    if thermostat == "andersen":
        plan = plan_refresh(dynamics, runtime.collision_clock, state.step, count)
        actual = int(plan.steps)
        if actual <= 0:
            return False
        observe = observe[:actual]
        due = due[:actual]

    before_state = state.copy()
    before_elapsed = runtime.elapsed
    before_rng = copy.deepcopy(dynamics.rng.bit_generator.state)
    before_stats = copy.deepcopy(dynamics.propagation_stats)
    before_deadline = (None if runtime.collision_clock is None else
                       runtime.collision_clock.next_collision_step)
    before_count = update.bias.hill_count
    value_rows = []
    try:
        if thermostat == "andersen":
            block = dynamics.propagate_block(
                state, actual, bias=force_bias, cv=cv, observe=observe,
                record_energy=False, record_states=True, backend="native",
                raise_on_error=False, refresh=plan)
        else:
            block = propagate_sampling_block(
                dynamics, state, actual, thermostat="pile-l",
                method=(sampler.integrator if sampler is not None else sampling.get("integrator")),
                gamma_centroid=(sampler.gamma_centroid if sampler is not None else
                                parse_quantity(sampling.get("gamma_centroid", "0.01 fs^-1"), "frequency")),
                bias=force_bias, cv=cv, observe=observe,
                record_energy=True, raise_on_error=False)

        observed_index = 0
        for step in block.steps:
            step = int(step)
            prod = step - neq
            value = np.asarray(block.values[observed_index], dtype=float).copy()
            if value.shape != (cv.dimension,) or not np.isfinite(value).all():
                raise ValueError("CV returned invalid observation")
            row_due = bool(update.due(prod)) if prod > 0 else False
            row_record = prod > 0 and prod % interval == 0
            frame_state = None
            if thermostat == "andersen":
                frame_state = State(block.q[observed_index], block.p[observed_index],
                                    step, float(block.times[observed_index]))
            if row_record:
                observed = float(update.bias.physical_energy(value))
                if thermostat == "andersen":
                    energy = float(dynamics.energy(
                        frame_state, energy_bias(update.bias) if energy_bias else update.bias))
                else:
                    energy = float(block.energies[observed_index])
                if not np.isfinite([observed, energy]).all():
                    raise ValueError("biased sampling observables must be finite")
                value_rows.append((value, step, float(block.times[observed_index]), observed, energy,
                                   frame_state))
            elif row_due:
                value_rows.append((value, step, float(block.times[observed_index]), None, None,
                                   frame_state))
            observed_index += 1

        runtime.elapsed += int(block.completed)
        for value, step, time, observed, energy, frame_state in value_rows:
            if observed is not None:
                runtime.values.append(value)
                runtime.clocks.append(step)
                runtime.biases.append(observed)
                runtime.energies.append(energy)
            if block.error is None and update.due(step - neq):
                deposit_state = frame_state if frame_state is not None else runtime.state
                update.apply(value, deposit_state)
        if runtime.configurations is not None and thermostat == "andersen":
            for _value, _step, _time, _observed, _energy, frame_state in value_rows:
                if _observed is not None:
                    runtime.configurations.append(frame_state.q.copy())
        if block.error is not None:
            raise PropagationError(block)
    except BaseException:
        if "block" in locals() and block.error is not None:
            raise
        runtime.state.q[:], runtime.state.p[:] = before_state.q, before_state.p
        runtime.state.step, runtime.state.time = before_state.step, before_state.time
        runtime.elapsed = before_elapsed
        dynamics.rng.bit_generator.state = before_rng
        dynamics.propagation_stats = before_stats
        if runtime.collision_clock is not None and before_deadline is not None:
            runtime.collision_clock.next_collision_step = before_deadline
        update.restore(before_count)
        saved = sum(v[3] is not None for v in value_rows)
        if saved:
            del runtime.values[-saved:]
            del runtime.clocks[-saved:]
            del runtime.biases[-saved:]
            del runtime.energies[-saved:]
            if runtime.configurations is not None:
                del runtime.configurations[-saved:]
        raise
    return True


def run_biased_sampling(runtime, dynamics, cv, bias=None, *, neq, nprod, interval,
                        checkpoint_steps, sampling, save=None, update=None,
                        sampler=None, active_bias=None, energy_bias=None,
                        fallback_reason=None, initial_save=True, final_save=True,
                        stop=None, failure=None, pair_distance_log=None):
    """Run or resume one stream up to a checkpoint/synchronization boundary.

    ``update`` has ``bias``, ``due``, ``snapshot``, ``apply`` and ``restore``;
    a fixed bias instead supplies the existing checked ``sampler``.  ``save``
    receives the current runtime and must atomically commit its state/history.
    ``stop`` is an absolute clock used by the multiple-walker block scheduler.
    ``failure(runtime, error)`` records a structured failure artifact after the
    last committed state was saved; it never replaces the original error.
    """
    total = neq + nprod if stop is None else min(stop, neq + nprod)
    if update is None and runtime.collision_clock is None:
        runtime.collision_clock = AndersenSchedule(sampling.get("andersen_interval", 10))
    fused = (update is None and sampler.thermostat == "andersen"
             and dynamics.can_fuse(bias=bias, observe=cv))
    try:
        for component in getattr(cv, "cvs", (cv,)):
            if getattr(component, "regular_branch_only", False):
                component.value(runtime.state.q)
    except Exception as error:
        if failure is not None:
            failure(runtime, error)
        raise
    if initial_save and save is not None:
        save(runtime)
    if pair_distance_log is not None:
        _record_pair_distance(pair_distance_log, runtime, runtime.elapsed)
    while runtime.elapsed < total:
        start = runtime.elapsed
        block_end = min(total, start + 1) if pair_distance_log is not None else total
        try:
            if update is None:
                _fixed_step(runtime, dynamics, cv, bias, neq=neq, total=block_end,
                            interval=interval, checkpoint_steps=checkpoint_steps,
                            sampling=sampling, sampler=sampler, fused=fused)
            else:
                handled = False
                if runtime.native_path_contract == 1:
                    handled = _history_block(
                        runtime, dynamics, cv, update, neq=neq, total=block_end,
                        interval=interval, checkpoint_steps=checkpoint_steps,
                        sampling=sampling, sampler=sampler,
                        active_bias=active_bias, energy_bias=energy_bias)
                if not handled:
                    _history_step(runtime, dynamics, cv, update, neq=neq,
                                  interval=interval, sampling=sampling, sampler=sampler,
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
            if failure is not None and isinstance(error, Exception):
                try:
                    failure(runtime, error)
                except Exception as report_error:  # keep the original failure
                    error.add_note(f"failure artifact not written: {report_error}")
            raise
        if pair_distance_log is not None:
            _record_pair_distance(pair_distance_log, runtime, start)
        if save is not None and runtime.elapsed % checkpoint_steps == 0:
            save(runtime)
    if final_save and save is not None:
        save(runtime)
    return runtime
