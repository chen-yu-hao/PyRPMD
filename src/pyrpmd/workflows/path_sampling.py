"""Input-driven path sampling with atomic, deterministic continuation.

RETIS/REPPTIS reuse the existing reversible shooting kernels. FFS explicitly
selects its forward Markov process; no thermostat is inserted into TIS paths.
The only persistent state is numeric NPZ data and finite JSON, never pickle.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np

from ..checkpoint import CheckpointError, atomic_savez
from ..constants import ATOMIC_TIME_S
from ..cv import make_cvs
from ..dynamics import Dynamics, State
from ..ffs import ForwardFluxSampler
from ..io import fingerprint, referenced_files
from ..parallel import execution_plan, task_potential
from ..pptis import ReplicaExchangePPTIS, combine_pptis_rate
from ..propagation import scientific_config
from ..retis import ReplicaExchangeTIS
from ..transition_interface import TransitionInterfaceSampler
from ..units import parse_quantity, unit_factor
from ..validation import checked_integer, finite_real
from .common import _dimension, write_json


_COMMON = {"method", "cv", "basin_a", "basin_b", "interfaces", "interface_unit", "block_size"}
_CHAIN = {"samples", "burn_in", "max_path_steps", "seeds", "external_flux"}
_RETIS = {"exchange", "minus", "minus_seed", "workers"}
_FFS = {"flux_steps", "trials_per_interface", "max_trial_steps", "initial_state",
        "dynamics", "collision_rate", "equilibration_steps"}


def _coordinate(config, system, name):
    specs = config.get("cvs", [])
    if isinstance(specs, Mapping):
        specs = [dict(value, name=key) for key, value in specs.items()]
    if not isinstance(specs, (list, tuple)) or not specs:
        raise ValueError("path_sampling requires at least one declared scalar CV")
    if not all(isinstance(spec, Mapping) for spec in specs):
        raise ValueError("cvs must contain CV tables")
    names = [spec.get("name") for spec in specs]
    if any(not isinstance(item, str) or not item for item in names) or len(set(names)) != len(names):
        raise ValueError("path_sampling CV names must be nonempty and unique")
    if name is None and len(specs) == 1:
        name = names[0]
    if not isinstance(name, str) or name not in names:
        raise ValueError("path_sampling.cv must select one declared CV by name")
    stack = make_cvs(specs, system, [name])
    cv = stack.cvs[0]
    if cv.dimension != 1 or any(period is not None for period in cv.periods):
        raise ValueError("path_sampling requires one continuous nonperiodic scalar CV")
    return cv, specs[names.index(name)], name


def _phase_point(value, config, system, label):
    if not isinstance(value, Mapping) or set(value) != {"q", "p"}:
        raise ValueError(f"{label} requires exactly q and p arrays in atomic units")
    shape = (int(config["beads"]), len(system.masses), 3)
    arrays = []
    for name in ("q", "p"):
        def contains_boolean(item):
            if isinstance(item, (bool, np.bool_)):
                return True
            return isinstance(item, (list, tuple, np.ndarray)) and any(contains_boolean(part) for part in item)

        if contains_boolean(value[name]):
            raise ValueError(f"{label}.{name} must not contain boolean values")
        raw = np.asarray(value[name])
        if raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
            raise ValueError(f"{label}.{name} must contain finite real numbers in atomic units")
        if raw.shape == shape[1:]:
            raw = np.broadcast_to(raw, shape)
        if raw.shape != shape:
            raise ValueError(f"{label}.{name} must have shape [N,3] or [P,N,3]")
        arrays.append(np.array(raw, dtype=np.float64, copy=True))
    if system.fixed_atoms:
        indices = list(system.fixed_atoms)
        if (not np.array_equal(arrays[0][:, indices], np.broadcast_to(system.geometry[indices], arrays[0][:, indices].shape))
                or np.any(arrays[1][:, indices] != 0)):
            raise ValueError(f"{label} must preserve fixed reference positions and zero fixed momenta")
    return {"q": arrays[0].tolist(), "p": arrays[1].tolist()}


def resolve_path_sampling(config, system):
    """Return idempotently normalized options; perform no MD or output writes."""
    if config.get("task") != "path_sampling":
        raise ValueError("path sampling requires task='path_sampling'")
    options = config.get("path_sampling")
    if not isinstance(options, Mapping):
        raise ValueError("path_sampling must be a table")
    result = deepcopy(dict(options))
    method = result.get("method")
    if method not in ("retis", "repptis", "ffs"):
        raise ValueError("path_sampling.method must be retis, repptis or ffs")
    allowed = _COMMON | (_FFS if method == "ffs" else _CHAIN)
    if method == "retis":
        allowed |= _RETIS
    unknown = set(result) - allowed
    if unknown:
        raise ValueError(f"Unknown path_sampling fields for {method}: " + ", ".join(sorted(unknown)))
    if system.bond_constraints:
        raise ValueError("path_sampling does not support bond constraints")
    if any(key in config for key in ("barostat", "electronic_friction", "path_integral", "active_learning")):
        raise ValueError("path_sampling does not support barostats, modified ring measures or adaptive potentials")
    if config.get("sampling"):
        raise ValueError("path_sampling uses its own budgets and explicit dynamics, not [sampling]")
    dyn = config.get("dynamics", {})
    if dyn.get("loop_backend", "auto") == "native":
        raise ValueError("path_sampling uses stepwise path termination; loop_backend='native' is unsupported")
    cv, spec, name = _coordinate(config, system, result.get("cv"))
    unit = cv.units[0]
    input_unit = result.get("interface_unit", spec.get("unit", spec.get("units", unit)))
    if isinstance(input_unit, (list, tuple)) and len(input_unit) == 1:
        input_unit = input_unit[0]
    if not isinstance(input_unit, str):
        raise ValueError("path_sampling.interface_unit must name one scalar CV unit")
    scale = unit_factor(input_unit, _dimension(unit))

    def quantity(value, label):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{label} must be a finite real quantity")
        parsed = parse_quantity(value, _dimension(unit))
        if isinstance(value, (int, float, np.integer, np.floating)):
            parsed *= scale
        return finite_real(parsed, label)

    for key in ("basin_a", "basin_b", "interfaces"):
        if key not in result:
            raise ValueError(f"path_sampling.{key} is required")
    result["basin_a"] = quantity(result["basin_a"], "path_sampling.basin_a")
    result["basin_b"] = quantity(result["basin_b"], "path_sampling.basin_b")
    raw_interfaces = result["interfaces"]
    if not isinstance(raw_interfaces, (list, tuple, np.ndarray)) or not len(raw_interfaces):
        raise ValueError("path_sampling.interfaces must be a nonempty ordered list")
    interfaces = [quantity(value, "path_sampling.interfaces") for value in raw_interfaces]
    a, b = result["basin_a"], result["basin_b"]
    if not a < b or interfaces[0] < a or interfaces[-1] >= b or np.any(np.diff(interfaces) <= 0):
        raise ValueError("path_sampling requires basin_a <= interfaces[0] < ... < basin_b")
    if method == "repptis" and interfaces[0] <= a:
        raise ValueError("REPPTIS centers require basin_a < interfaces[0]")
    result.update(cv=name, interfaces=interfaces, interface_unit=unit)
    if "block_size" in result:
        result["block_size"] = checked_integer(result["block_size"], "path_sampling.block_size")
    plan = execution_plan(config.get("parallel", {}))
    if plan.backend == "mpi" or plan.mode != "serial" or plan.bead_workers != 1 or plan.pes_ranks != 1:
        raise ValueError("path_sampling supports local serial/process execution with mode='serial' and bead_workers=1")
    if config.get("parallel", {}).get("trajectory_batch_size", 1) != 1:
        raise ValueError("path_sampling does not batch trajectory propagation")
    if method != "retis" or result.get("exchange", "neighbor") != "infinite":
        if plan.backend != "serial" or plan.trajectory_workers != 1:
            raise ValueError("parallel path workers require retis with exchange='infinite'")
    if method == "ffs":
        for key in ("flux_steps", "trials_per_interface", "initial_state", "dynamics"):
            if key not in result:
                raise ValueError(f"FFS requires explicit path_sampling.{key}")
        result["flux_steps"] = checked_integer(result["flux_steps"], "path_sampling.flux_steps")
        trials = result["trials_per_interface"]
        if isinstance(trials, (list, tuple, np.ndarray)):
            if len(trials) != len(interfaces):
                raise ValueError("trials_per_interface must have one count per interface")
            trials = [checked_integer(n, "trials_per_interface") for n in trials]
        else:
            trials = [checked_integer(trials, "trials_per_interface")] * len(interfaces)
        result["trials_per_interface"] = trials
        result["max_trial_steps"] = checked_integer(result.get("max_trial_steps", 10000), "max_trial_steps")
        result["equilibration_steps"] = checked_integer(result.get("equilibration_steps", 0), "equilibration_steps", 0)
        process = result["dynamics"]
        if process not in ("hamiltonian", "basin-andersen"):
            raise ValueError("FFS dynamics must explicitly select hamiltonian or basin-andersen")
        if process == "basin-andersen":
            if "collision_rate" not in result:
                raise ValueError("basin-andersen requires an explicit collision_rate")
            if isinstance(result["collision_rate"], (bool, np.bool_)):
                raise ValueError("collision_rate must be positive and finite")
            result["collision_rate"] = finite_real(parse_quantity(result["collision_rate"], "frequency"), "collision_rate", positive=True)
        elif "collision_rate" in result:
            raise ValueError("collision_rate requires FFS dynamics='basin-andersen'")
        result["initial_state"] = _phase_point(result["initial_state"], config, system, "initial_state")
        return result
    result["samples"] = checked_integer(result.get("samples", 1000), "path_sampling.samples")
    result["burn_in"] = checked_integer(result.get("burn_in", 100), "path_sampling.burn_in", 0)
    result["max_path_steps"] = checked_integer(result.get("max_path_steps", 10000), "max_path_steps")
    seeds = result.get("seeds")
    if not isinstance(seeds, (list, tuple)) or len(seeds) not in ({1, len(interfaces)} if method == "retis" else {len(interfaces)}):
        raise ValueError("path_sampling.seeds requires one phase point per interface (RETIS also accepts one shared seed)")
    result["seeds"] = [_phase_point(value, config, system, f"seeds[{i}]") for i, value in enumerate(seeds)]
    if method == "retis":
        result.setdefault("exchange", "neighbor")
        if result["exchange"] not in ("neighbor", "infinite"):
            raise ValueError("RETIS exchange must be neighbor or infinite")
        minus = result.setdefault("minus", True)
        if not isinstance(minus, (bool, np.bool_)):
            raise ValueError("path_sampling.minus must be boolean")
        result["minus"] = bool(minus)
        if minus:
            if result["max_path_steps"] < 2:
                raise ValueError("minus RETIS requires max_path_steps >= 2")
            if interfaces[0] != a:
                raise ValueError("minus RETIS requires interfaces[0] == basin_a")
            if "minus_seed" not in result:
                raise ValueError("minus RETIS requires a minus_seed phase point")
            result["minus_seed"] = _phase_point(result["minus_seed"], config, system, "minus_seed")
            if "external_flux" in result:
                raise ValueError("minus RETIS estimates internal flux; external_flux would be unused")
        elif "minus_seed" in result:
            raise ValueError("minus_seed requires minus=true")
        if result["exchange"] == "infinite":
            if plan.scheduling != "dynamic":
                raise ValueError("infinite RETIS requires parallel.scheduling='dynamic'")
            result["workers"] = checked_integer(result.get("workers", plan.trajectory_workers), "path_sampling.workers")
            if result["workers"] != plan.trajectory_workers:
                raise ValueError("path_sampling.workers must match parallel.trajectory_workers; the logical worker count is scientific state")
            if result["workers"] >= len(interfaces) + int(minus):
                raise ValueError("infinite RETIS needs more ensembles than workers so free ensembles remain")
        elif "workers" in result:
            raise ValueError("path_sampling.workers requires exchange='infinite'")
    if "external_flux" in result:
        flux = result["external_flux"]
        required = {"basin_a", "basin_b", "first_interface", "dt_au", "flux_au"}
        if not isinstance(flux, Mapping) or not required <= set(flux) or set(flux) - required - {"provenance"}:
            raise ValueError("external_flux requires basin_a, basin_b, first_interface, dt_au and flux_au")
        flux = dict(flux)
        expected = {"basin_a": a, "basin_b": b, "first_interface": interfaces[0]}
        for key, value in expected.items():
            flux[key] = quantity(flux[key], "external_flux." + key)
            if flux[key] != value:
                raise ValueError(f"external_flux.{key} must match the path ensemble")
        flux["dt_au"] = finite_real(flux["dt_au"], "external_flux.dt_au", positive=True)
        if flux["dt_au"] != config["dt_au"]:
            raise ValueError("external_flux.dt_au must match the dynamics timestep")
        flux["flux_au"] = finite_real(flux["flux_au"], "external_flux.flux_au")
        if flux["flux_au"] < 0 or not math.isfinite(flux["flux_au"] / ATOMIC_TIME_S):
            raise ValueError("external_flux.flux_au must be finite, nonnegative and convertible to s^-1")
        if "provenance" in flux and (not isinstance(flux["provenance"], str) or not flux["provenance"].strip()):
            raise ValueError("external_flux.provenance must be a nonempty description")
        result["external_flux"] = flux
    return result


def _state(data):
    return State(np.asarray(data["q"], dtype=float), np.asarray(data["p"], dtype=float))


def _identity(config, system):
    return {"workflow": "pyrpmd.path_sampling.v1", "physics": fingerprint(scientific_config(config), system),
            "files": referenced_files(config)}


@contextmanager
def _dynamics(config, system):
    reuse = config.get("parallel", {}).get("reuse_potentials", True)
    with task_potential(config["potential"], system, reuse=reuse) as potential:
        settings = config.get("dynamics", {})
        dynamics = Dynamics(
            potential, system.masses, config["beta"], config["beads"], config["dt_au"],
            seed=config["seed"], native=settings.get("native", config.get("native", True)),
            threads=config.get("parallel", {}).get("threads_per_rank", 1),
            loop_backend=settings.get("loop_backend", "auto"),
            integrator=settings.get("integrator", "cayley-bcb"),
            fixed_atoms=system.fixed_atoms, fixed_positions=system.geometry,
            cell=system.cell, pbc=system.pbc,
        )
        yield dynamics


def _sampler(dynamics, coordinate, config, identity):
    options = config["path_sampling"]
    params = dict(basin_a=options["basin_a"], basin_b=options["basin_b"],
                  interfaces=options["interfaces"], max_path_steps=options["max_path_steps"],
                  seed=config["seed"], identity=identity)
    if options["method"] == "retis":
        return ReplicaExchangeTIS(dynamics, coordinate, minus=options["minus"], **params)
    return ReplicaExchangePPTIS(dynamics, coordinate, **params)


def _seed_paths(sampler, options):
    points = options["seeds"]
    if len(points) == 1 and len(sampler.interfaces) > 1:
        path = sampler.generate_path(_state(points[0]), len(sampler.interfaces) - 1)
        return [path] * len(sampler.interfaces)
    return [sampler.generate_path(_state(point), i) for i, point in enumerate(points)]


def _infinite_worker(payload):
    """Importable spawn worker; all move randomness belongs to its saved seed."""
    from ..infinite_retis import execute_retis_move
    from ..system import System

    context = payload["context"]
    config = context["config"]
    system = System.from_dict(context["system"])
    coordinate, _, _ = _coordinate(config, system, config["path_sampling"]["cv"])
    with _dynamics(config, system) as dynamics:
        sampler = _sampler(dynamics, coordinate, config, context["identity"])
        return execute_retis_move(sampler, payload)


def _apply_flux(summary, options):
    if "external_flux" not in options:
        return summary
    flux = options["external_flux"]["flux_au"]
    probability = summary.get("overall_crossing_probability")
    error = summary.get("crossing_probability_standard_error_block")
    rate = None if probability is None else flux * probability
    return {**summary, "flux_au": flux, "flux_s_inv": flux / ATOMIC_TIME_S,
            "rate_estimate_au": rate, "rate_estimate_s_inv": None if rate is None else rate / ATOMIC_TIME_S,
            "rate_standard_error_block_conditional_on_flux_au": None if error is None else flux * error,
            "external_flux": options["external_flux"], "requires_independent_flux": False,
            "uncertainty_qualification": "Rate uncertainty is conditional on the supplied flux; its uncertainty is not included."}


def _empty_summary(method):
    return {"method": method, "overall_crossing_probability": None, "rate_estimate_au": None,
            "rate_estimate_s_inv": None, "rate_s_inv": None, "converged": False}


def _retis_observations(result):
    arrays = {name: np.column_stack([getattr(item, attr) for item in result.interfaces])
              for name, attr in (("crossing", "crossing"), ("accepted", "accepted"), ("path_frames", "path_frames"))}
    arrays["walker_labels"] = result.walker_labels
    if result.minus is not None:
        arrays["minus_frames"] = result.minus.path_frames
        arrays["minus_accepted"] = result.minus.accepted
    return arrays


def _run_retis(sampler, options, checkpoint, resume, budget, interval):
    if resume and checkpoint.exists():
        sampler.load_checkpoint(checkpoint)
    else:
        paths = _seed_paths(sampler, options)
        minus = sampler.generate_minus_path(_state(options["minus_seed"])) if options["minus"] else None
        sampler.initialize(paths, minus_path=minus)
        sampler.save_checkpoint(checkpoint)
    total = options["burn_in"] + options["samples"]
    if sampler._burn_in > options["burn_in"] or len(sampler._crossing) > options["samples"]:
        raise CheckpointError("RETIS checkpoint exceeds the requested sampling budget")
    stop = total if budget is None else min(total, sampler._sweeps + budget)
    try:
        while sampler._burn_in < options["burn_in"] and sampler._sweeps < stop:
            sampler._sweep()
            sampler._sweeps += 1
            sampler._burn_in += 1
            if sampler._sweeps % interval == 0:
                sampler.save_checkpoint(checkpoint)
        while sampler._sweeps < stop:
            count = min(stop - sampler._sweeps, interval - sampler._sweeps % interval)
            sampler.sample_interfaces(n_samples=count)
            sampler.save_checkpoint(checkpoint)
    except Exception:
        # Numerical failures roll back the current sweep. A control-flow
        # interrupt can land after propagation but before observation commit;
        # replay from the preceding atomic checkpoint in that case.
        sampler.save_checkpoint(checkpoint)
        raise
    sampler.save_checkpoint(checkpoint)
    if sampler._crossing:
        result = sampler.result()
        summary = _apply_flux(result.summary(block_size=options.get("block_size")), options)
        arrays = _retis_observations(result)
    else:
        summary, arrays = _empty_summary("replica_exchange_tis"), {}
    return summary, arrays, len(sampler._crossing), sampler._burn_in, sampler._sweeps, sampler._sweeps >= total


def _production_pptis(result, burn_in):
    """Discard recorded warm-up observations while preserving all diagnostics.

    Keeping warm-up in the native cumulative checkpoint lets the existing
    reader resume any completed sweep, including before production begins.
    No random draw depends on recording or slicing an observation.
    """
    interfaces = tuple(replace(item, starts=item.starts[burn_in:], ends=item.ends[burn_in:],
                               accepted=item.accepted[burn_in:], path_frames=item.path_frames[burn_in:],
                               reasons=item.reasons[burn_in:]) for item in result.interfaces)
    return replace(result, interfaces=interfaces, walker_labels=result.walker_labels[burn_in:], burn_in=burn_in)


def _run_repptis(sampler, options, checkpoint, resume, budget, interval):
    result = sampler.load_checkpoint(checkpoint) if resume and checkpoint.exists() else None
    completed = 0 if result is None else len(result.walker_labels)
    total = options["burn_in"] + options["samples"]
    if completed > total or result is not None and result.burn_in != 0:
        raise CheckpointError("REPPTIS workflow checkpoint has incompatible warm-up or sampling budget")
    stop = total if budget is None else min(total, completed + budget)
    seeds = _seed_paths(sampler, options) if result is None else None
    while completed < stop:
        count = min(stop - completed, interval - completed % interval)
        # On a numerical failure, the previous immutable result/checkpoint
        # remains authoritative; replay the uncommitted chunk after resume.
        result = sampler.sample_interfaces(seeds, n_samples=count, resume=result)
        seeds = None
        completed += count
        sampler.save_checkpoint(checkpoint, result)
    samples = max(0, completed - options["burn_in"])
    if samples:
        production = _production_pptis(result, options["burn_in"])
        summary = (combine_pptis_rate(production, options["external_flux"], block_size=options.get("block_size"))
                   if "external_flux" in options else production.summary(block_size=options.get("block_size")))
        if "external_flux" in options:
            summary["requires_independent_flux"] = False
            summary["external_flux"] = options["external_flux"]
        arrays = {name: np.column_stack([getattr(item, name) for item in production.interfaces])
                  for name in ("starts", "ends", "accepted", "path_frames")}
        arrays["walker_labels"] = production.walker_labels
    else:
        summary, arrays = _empty_summary("replica_exchange_pptis"), {}
    return summary, arrays, samples, min(completed, options["burn_in"]), completed, completed >= total


def _run_ffs(dynamics, coordinate, config, identity, checkpoint, resume, budget, interval):
    options = config["path_sampling"]
    # Reuse the same purity/reversibility guard for the deterministic MD part.
    TransitionInterfaceSampler(dynamics, coordinate, basin_a=options["basin_a"], basin_b=options["basin_b"],
                               interfaces=options["interfaces"], max_path_steps=options["max_trial_steps"])
    probability = -math.expm1(-options.get("collision_rate", 0.0) * dynamics.dt)
    sigma = np.sqrt(dynamics.masses[None, :, None] * dynamics.nbeads / dynamics.beta)

    def forward(state, rng):
        dynamics.step(state)
        value = float(np.asarray(coordinate.value(state.q)).reshape(-1)[0])
        if options["dynamics"] == "basin-andersen" and (value <= options["basin_a"] or value >= options["basin_b"]):
            if rng.random() < probability:
                state.p[:] = rng.normal(size=state.p.shape) * sigma
                if dynamics.fixed_atoms:
                    state.p[:, list(dynamics.fixed_atoms)] = 0

    sampler = ForwardFluxSampler(
        forward, coordinate, dt=dynamics.dt, basin_a=options["basin_a"], basin_b=options["basin_b"],
        interfaces=options["interfaces"], flux_steps=options["flux_steps"],
        trials_per_interface=options["trials_per_interface"], max_trial_steps=options["max_trial_steps"],
        seed=config["seed"], dynamics_id=json.dumps({**identity, "process": options["dynamics"],
                                                   "collision_rate": options.get("collision_rate"),
                                                   "law": "NVE_then_basin_Maxwell_v1"}, sort_keys=True),
        coordinate_id=json.dumps({"cv": options["cv"], "physics": identity["physics"]}, sort_keys=True),
    )
    if resume and checkpoint.exists():
        sampler.load_checkpoint(checkpoint)
    else:
        initial = _state(options["initial_state"])
        for _ in range(options["equilibration_steps"]):
            forward(initial, sampler.rng)
        sampler.initialize(initial)
        sampler.save_checkpoint(checkpoint)
    result = sampler.result()
    count = len(result.flux_events) + sum(len(stage.outcomes) for stage in result.stages)
    stop = options["flux_steps"] + sum(options["trials_per_interface"]) if budget is None else count + budget
    try:
        while result.status in ("collecting_flux", "sampling_interfaces") and count < stop:
            sampler.advance(min(interval - count % interval, stop - count))
            sampler.save_checkpoint(checkpoint)
            result = sampler.result()
            count = len(result.flux_events) + sum(len(stage.outcomes) for stage in result.stages)
    except Exception:
        # The FFS kernel rolls back ordinary failed operations. Asynchronous
        # interrupts can land between RNG draws and state commit: retain the
        # preceding atomic checkpoint and replay its uncommitted chunk instead.
        sampler.save_checkpoint(checkpoint)
        raise
    arrays = {name: getattr(result, name) for name in ("flux_values", "flux_events", "flux_a_history", "flux_unknown_history")}
    for i, stage in enumerate(result.stages):
        for name in ("outcomes", "parent_indices", "root_indices", "steps"):
            arrays[f"stage_{i}_{name}"] = getattr(stage, name)
    summary = result.summary(block_size=options.get("block_size"))
    summary["forward_dynamics"] = options["dynamics"]
    summary["sampling_support_complete"] = result.complete
    terminal = result.status in ("complete", "no_successes", "no_flux_crossings")
    return summary, arrays, sum(len(stage.outcomes) for stage in result.stages), 0, count, terminal


def _run_infinite(sampler, config, system, identity, checkpoint, resume, budget, interval):
    from ..infinite_retis import InfiniteRETIS

    options = config["path_sampling"]
    coordinator = InfiniteRETIS(
        sampler, _infinite_worker, worker_context={"config": config, "system": system.to_dict(), "identity": identity},
        parallel_config=config.get("parallel", {}), workers=options["workers"], seed=config["seed"], worker_identity=identity,
    )
    if resume and checkpoint.exists():
        coordinator.load_checkpoint(checkpoint)
    else:
        paths = _seed_paths(sampler, options)
        minus = sampler.generate_minus_path(_state(options["minus_seed"])) if options["minus"] else None
        coordinator.initialize(paths, minus_path=minus)
    total = options["samples"] + options["burn_in"]
    if coordinator.events > total or coordinator.sample_count > options["samples"]:
        raise CheckpointError("infinite RETIS checkpoint exceeds the requested budget")
    count = total - coordinator.events
    if budget is not None:
        count = min(count, budget)

    def save(event, current):
        if current.events % interval == 0 or current.events >= total:
            current.save_checkpoint(checkpoint)

    if count:
        coordinator.advance(count, burn_in_total=options["burn_in"], on_event=save)
    coordinator.save_checkpoint(checkpoint)
    summary = _apply_flux(coordinator.summary(block_size=options.get("block_size")), options)
    return (summary, coordinator.observations, coordinator.sample_count, coordinator.burn_in_done,
            coordinator.events, coordinator.events >= total)


def run_path_sampling(config, system, directory, *, max_operations=None):
    """Run or resume the configured total budget and publish auditable results.

    ``max_operations`` is an execution-only Python stopping aid, useful for
    schedulers and interruption tests. An operation is one sweep/event for TIS
    or one flux step/whole branching trial for FFS. The saved scientific budget
    is unchanged; ordinary ``pyrpmd resume`` completes it.
    """
    config = deepcopy(config)
    config["path_sampling"] = options = resolve_path_sampling(config, system)
    if max_operations is not None:
        max_operations = checked_integer(max_operations, "max_operations", 0)
    interval = checked_integer(config.get("output", {}).get("checkpoint_steps", 100), "output.checkpoint_steps")
    directory = Path(directory)
    checkpoint = directory / "checkpoints" / "path_sampling.npz"
    previous_summary = directory / "results" / "path_sampling.json"
    if config.get("resume") and not checkpoint.exists() and previous_summary.exists():
        previous = json.loads(previous_summary.read_text(encoding="utf-8"))
        if previous.get("operations_completed", 0) > 0:
            raise CheckpointError("Path-sampling checkpoint is missing from a run with recorded progress")
    identity = _identity(config, system)
    coordinate, _, _ = _coordinate(config, system, options["cv"])
    with _dynamics(config, system) as dynamics:
        if options["method"] == "ffs":
            result = _run_ffs(dynamics, coordinate, config, identity, checkpoint, bool(config.get("resume")), max_operations, interval)
        else:
            sampler = _sampler(dynamics, coordinate, config, identity)
            if options["method"] == "repptis":
                result = _run_repptis(sampler, options, checkpoint, bool(config.get("resume")), max_operations, interval)
            elif options["exchange"] == "infinite":
                result = _run_infinite(sampler, config, system, identity, checkpoint, bool(config.get("resume")), max_operations, interval)
            else:
                result = _run_retis(sampler, options, checkpoint, bool(config.get("resume")), max_operations, interval)
    summary, arrays, samples, burn, operations, complete = result
    summary = {**summary, "task": "path_sampling", "path_sampling_method": options["method"],
               "exchange": options.get("exchange"), "status": summary.get("status", "complete" if complete else "partial"),
               "completed": bool(complete), "samples": samples, "burn_in_completed": burn,
               "operations_completed": operations, "temperature_K": config["temperature_K"],
               "beta_au": config["beta"], "beads": config["beads"], "dt_au": config["dt_au"],
               "physics_fingerprint": identity["physics"], "coordinate": options["cv"], "coordinate_unit": options["interface_unit"],
               "checkpoint": "checkpoints/path_sampling.npz", "observations": "samples/path_sampling.npz",
               "rate_s_inv": None, "converged": False,
               "convergence_note": "Completed budget is not physical convergence; establish stationarity, interface support, timestep/bead/path-length convergence and independent-chain uncertainty."}
    if not complete:
        summary["status"] = "partial"
    atomic_savez(directory / "samples" / "path_sampling.npz", kind="PyRPMD.path_sampling_observations.v1",
                 metadata=np.asarray(json.dumps({"format": "PyRPMD.path_sampling_observations.v1", "identity": identity,
                                                 "operations": operations, "burn_in_completed": burn}, sort_keys=True)), **arrays)
    write_json(directory / "results" / "path_sampling.json", summary)
    return summary
