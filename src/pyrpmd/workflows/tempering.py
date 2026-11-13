"""Resumable temperature and Hamiltonian replica-exchange workflows.

One atomic ensemble checkpoint is authoritative. Replica workers read the same
committed generation, advance one walker, and atomically publish generation-
scoped stage files. The coordinator validates the complete set before applying
Metropolis exchanges and committing the next generation. This makes the serial,
process and MPI task routes use the same scientific ordering and restart state.

Coordinates, masses, momenta, energy and time are atomic units internally;
temperature ladder input/output is kelvin.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..analysis.mbar import solve_mbar
from ..bias import UmbrellaBias
from ..checkpoint import CheckpointError, atomic_savez, load_checkpoint
from ..constants import KB_AU
from ..dynamics import Dynamics, State
from ..parallel import root_call, run_tasks, task_potential
from ..potentials.base import PESResult, evaluate_checked, validate_result
from ..potentials.contraction import SplitPotential
from ..replica_exchange import (
    _static,
    exchange_pairs,
    metropolis_accept,
    round_trip_statistics,
    state_space_diffusion,
    temperature_momentum_scale,
)
from ..units import parse_quantity, time_steps
from ..validation import checked_integer, finite_real
from .common import _fingerprint, fes_method, write_json
from .sampling import middle_sampling_method

CHECKPOINT_KIND = "PyRPMD.tempering.v1"
STAGE_KIND = "PyRPMD.tempering-stage.v1"
TRAJECTORY_KIND = "PyRPMD.tempering-trajectory.v1"
ESTIMATOR_NAMES = (
    "hamiltonian_potential",
    "kinetic_primitive",
    "kinetic_centroid_virial",
    "hamiltonian_internal_energy_centroid_virial",
)

TEMPERING_FIELDS = {
    "mode",
    "temperatures",
    "temperature_min",
    "temperature_max",
    "replicas",
    "spacing",
    "lambdas",
    "target",
    "exchange_interval_steps",
    "exchange_interval",
    "pairing",
    "seed",
}


@dataclass(frozen=True)
class TemperingOptions:
    mode: str
    temperatures: np.ndarray
    betas: np.ndarray
    lambdas: np.ndarray
    target: str
    nstates: int
    exchange_interval: int
    pairing: str
    seed: int
    neq: int
    nprod: int
    total: int
    sample_interval: int
    checkpoint_interval: int
    analysis_start: int
    exchange_start: int
    thermostat: str
    sampling_method: str | None
    gamma_centroid: float | None
    andersen_interval: int
    fes_method: str | None
    cv_dimension: int
    dt: float

    def identity(self):
        return {
            "schema": 1,
            "mode": self.mode,
            "temperatures_K": self.temperatures.tolist(),
            "betas_au": self.betas.tolist(),
            "lambdas": self.lambdas.tolist(),
            "target": self.target,
            "exchange_interval_steps": self.exchange_interval,
            "pairing": self.pairing,
            "seed": self.seed,
            "analysis_start": self.analysis_start,
            "exchange_start": self.exchange_start,
            "fes_method": self.fes_method,
        }


def _temperature(value, name):
    try:
        result = parse_quantity(value, "temperature")
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive finite temperature") from error
    if result <= 0.0:
        raise ValueError(f"{name} must be a positive finite temperature")
    return float(result)


def _real_ladder(values, name, *, positive=False, nonnegative=False):
    if isinstance(values, (str, bytes, Mapping)):
        raise ValueError(f"{name} must be a list with at least two entries")
    try:
        raw = list(values)
    except TypeError as error:
        raise ValueError(f"{name} must be a list with at least two entries") from error
    if any(isinstance(value, (bool, np.bool_)) for value in raw):
        raise ValueError(f"{name} must contain finite real numbers, not booleans")
    try:
        result = np.asarray(raw, dtype=float)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"{name} must contain finite real numbers") from error
    if result.ndim != 1 or len(result) < 2 or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain at least two finite real numbers")
    if positive and np.any(result <= 0.0):
        raise ValueError(f"{name} entries must be positive")
    if nonnegative and np.any(result < 0.0):
        raise ValueError(f"{name} entries must be nonnegative")
    delta = np.diff(result)
    if not (np.all(delta > 0.0) or np.all(delta < 0.0)):
        raise ValueError(f"{name} must be strictly monotone")
    return result


def _explicit_temperatures(values):
    if isinstance(values, (str, bytes, Mapping)):
        raise ValueError("tempering.temperatures must be an explicit list")
    try:
        result = [_temperature(value, f"tempering.temperatures[{index}]")
                  for index, value in enumerate(values)]
    except TypeError as error:
        raise ValueError("tempering.temperatures must be an explicit list") from error
    return _real_ladder(result, "tempering.temperatures", positive=True)


def _temperature_ladder(table):
    generated = {"temperature_min", "temperature_max", "replicas"}
    has_generated = any(key in table for key in generated)
    if "temperatures" in table and has_generated:
        raise ValueError(
            "specify tempering.temperatures or temperature_min/temperature_max/replicas, not both"
        )
    if "temperatures" in table:
        if "spacing" in table:
            raise ValueError("tempering.spacing applies only to a generated ladder")
        return _explicit_temperatures(table["temperatures"])
    if not has_generated:
        raise ValueError(
            "temperature tempering requires temperatures or temperature_min/temperature_max/replicas"
        )
    missing = generated - set(table)
    if missing:
        raise ValueError("generated temperature ladder is missing: " + ", ".join(sorted(missing)))
    if table.get("spacing", "geometric") != "geometric":
        raise ValueError("tempering.spacing currently supports only 'geometric'")
    start = _temperature(table["temperature_min"], "tempering.temperature_min")
    stop = _temperature(table["temperature_max"], "tempering.temperature_max")
    count = checked_integer(table["replicas"], "tempering.replicas", 2)
    if start == stop:
        raise ValueError("tempering temperature endpoints must differ")
    return np.geomspace(start, stop, count)


def _exchange_interval(table, dt):
    present = [key for key in ("exchange_interval_steps", "exchange_interval") if key in table]
    if len(present) > 1:
        raise ValueError("specify only one tempering exchange interval")
    if not present:
        return 100
    if present[0] == "exchange_interval_steps":
        return checked_integer(table[present[0]], "tempering.exchange_interval_steps")
    return time_steps(table[present[0]], dt)


def _pairing(value):
    aliases = {
        "odd_even": "odd_even",
        "odd-even": "odd_even",
        "alternating": "odd_even",
        "adjacent": "adjacent",
        "sequential": "adjacent",
    }
    if not isinstance(value, str) or value not in aliases:
        raise ValueError("tempering.pairing must be odd_even or adjacent")
    return aliases[value]


def _resolve_fes(config, system):
    method = fes_method(config)
    if method == "umbrella":
        from .fes import resolve_window_plan

        cv, _, windows = resolve_window_plan(config, system)
        if len(windows) != 1:
            raise ValueError(
                "tempering umbrella FES requires exactly one base window; the tempering ladder supplies replicas"
            )
        return method, cv, windows[0]
    if method == "metadynamics":
        from .metadynamics import resolve_metadynamics

        cv, _, options = resolve_metadynamics(config, system)
        table = config.get("metadynamics", {})
        if options.get("freeze") is None:
            raise ValueError(
                "tempering metadynamics requires freeze_after_steps; exchanges start only after the bias is frozen"
            )
        if options.get("grid") is not None:
            raise ValueError("tempering metadynamics currently requires direct (non-grid) hills")
        if table.get("shared_walkers") or table.get("multiple_walkers"):
            raise ValueError("tempering owns walker exchange and cannot combine metadynamics shared walkers")
        if table.get("walls"):
            raise ValueError("tempering metadynamics does not yet combine static walls")
        return method, cv, options
    raise ValueError("tempering task='fes' supports only umbrella or frozen metadynamics")


def resolve_tempering(config, system):
    """Strictly validate and normalize a tempering request without side effects."""
    table = config.get("tempering")
    if not isinstance(table, Mapping):
        raise ValueError("tempering must be a table")
    unknown = set(table) - TEMPERING_FIELDS
    if unknown:
        raise ValueError("Unknown tempering fields: " + ", ".join(sorted(unknown)))
    task = config.get("task")
    if task not in ("pimd", "fes"):
        raise ValueError("tempering can be combined only with task='pimd' or task='fes'")
    sampling = config.get("sampling", {})
    if sampling.get("engine", "legacy") != "legacy":
        raise ValueError("tempering currently requires sampling.engine='legacy'")
    if sampling.get("replicas", 1) != 1:
        raise ValueError("sampling.replicas must be 1; tempering defines the replica count")
    if "barostat" in config:
        raise ValueError("tempering currently supports fixed-cell ensembles only")
    if "path_integral" in config:
        raise ValueError("tempering does not combine the specialized path_integral workflow")
    if "active_learning" in config:
        raise ValueError("tempering requires a fixed deterministic potential")
    if getattr(system, "bond_constraints", ()):
        raise ValueError("tempering does not yet support nonlinear bond constraints")
    if "replica_exchange" in config:
        raise ValueError("[tempering] is independent of and cannot combine the REUS [replica_exchange] table")
    if "adaptive_umbrella" in config:
        raise ValueError("tempering does not combine adaptive umbrella placement")
    parallel = config.get("parallel", {})
    if parallel.get("trajectory_batch_size", 1) != 1:
        raise ValueError("tempering exchanges individual walkers and requires trajectory_batch_size=1")
    dynamics = config.get("dynamics", {})
    if dynamics.get("loop_backend", "auto") == "native":
        raise ValueError("tempering orchestration requires dynamics.loop_backend='auto' or 'python'")
    if config.get("native", dynamics.get("native", True)) == "required":
        raise ValueError("tempering cannot guarantee a fully native propagation loop")
    dt = finite_real(config.get("dt_au"), "dt_au", positive=True)

    has_temperature = any(key in table for key in (
        "temperatures", "temperature_min", "temperature_max", "replicas", "spacing"
    ))
    has_hamiltonian = "lambdas" in table or "target" in table
    requested = table.get("mode")
    aliases = {"temperature": "temperature", "pt": "temperature",
               "hamiltonian": "hamiltonian", "hremd": "hamiltonian"}
    if requested is None:
        if has_temperature == has_hamiltonian:
            raise ValueError("tempering must select exactly one temperature or Hamiltonian ladder")
        mode = "temperature" if has_temperature else "hamiltonian"
    else:
        if requested not in aliases:
            raise ValueError("tempering.mode must be temperature/pt or hamiltonian/hremd")
        mode = aliases[requested]

    if mode == "temperature":
        if "lambdas" in table or "target" in table:
            raise ValueError("temperature tempering cannot define Hamiltonian lambdas or target")
        temperatures = _temperature_ladder(table)
        lambdas = np.ones(len(temperatures))
        target = "none"
    else:
        if has_temperature:
            raise ValueError("Hamiltonian tempering uses the job temperature and cannot define a temperature ladder")
        if "lambdas" not in table:
            raise ValueError("Hamiltonian tempering requires tempering.lambdas")
        lambdas = _real_ladder(
            table["lambdas"], "tempering.lambdas", nonnegative=True
        )
        temperatures = np.full(len(lambdas), float(config["temperature_K"]))
        target_aliases = {
            "potential": "potential", "pes": "potential",
            "bias": "bias",
            "split_slow": "split_slow", "split": "split_slow", "slow": "split_slow",
        }
        raw_target = table.get("target", "potential")
        if raw_target not in target_aliases:
            raise ValueError("tempering.target must be potential, bias or split_slow")
        target = target_aliases[raw_target]
        if target == "bias" and task != "fes":
            raise ValueError("tempering.target='bias' requires task='fes'")
        if target == "split_slow" and parallel.get("bead_workers", 1) != 1:
            raise ValueError(
                "tempering.target='split_slow' currently requires parallel.bead_workers=1"
            )
        if target == "split_slow":
            potential = config.get("potential")
            if isinstance(potential, SplitPotential):
                pass
            elif isinstance(potential, Mapping):
                kind = potential.get("provider", potential.get("type"))
                if kind != "delta" or "contracted_beads" not in potential:
                    raise ValueError(
                        "tempering.target='split_slow' requires an explicit SplitPotential "
                        "or a contracted delta provider"
                    )
                if potential.get("nbeads", config["beads"]) != config["beads"]:
                    raise ValueError("split potential nbeads must match the tempering bead count")
            else:
                raise ValueError(
                    "tempering.target='split_slow' requires an explicit SplitPotential "
                    "or a contracted delta provider"
                )

    nstates = len(temperatures)
    if parallel.get("backend") == "mpi":
        from ..parallel import execution_plan

        if execution_plan(parallel).trajectory_groups != nstates:
            raise ValueError(
                "MPI tempering requires parallel.trajectory_groups equal to the ladder size"
            )
    betas = 1.0 / (KB_AU * temperatures)
    neq = checked_integer(sampling.get("equilibration_steps", 0),
                          "sampling.equilibration_steps", 0)
    nprod = checked_integer(sampling.get("production_steps"), "sampling.production_steps")
    interval = checked_integer(
        sampling.get("sample_interval", sampling.get("interval_steps", 1)),
        "sampling.sample_interval",
    )
    checkpoint = checked_integer(
        config.get("output", {}).get("checkpoint_steps", 1000), "output.checkpoint_steps"
    )
    thermostat = sampling.get("thermostat", "pile-l")
    if thermostat not in ("pile-l", "andersen"):
        raise ValueError("tempering supports only PILE-L or fixed-schedule Andersen sampling")
    sampling_method = None
    gamma = None
    andersen = checked_integer(
        sampling.get("andersen_interval", 10), "sampling.andersen_interval"
    )
    scheme = dynamics.get("integrator", "cayley-bcb")
    if thermostat == "pile-l":
        sampling_method = middle_sampling_method(scheme, sampling.get("integrator"))
        gamma = finite_real(
            sampling.get("gamma_centroid", 1.0), "sampling.gamma_centroid", positive=True
        )
    else:
        expected = scheme + "-andersen"
        if sampling.get("integrator", expected) != expected:
            raise ValueError("Andersen tempering sampling.integrator must match the dynamics integrator")
        if sampling.get("andersen_schedule", "fixed") != "fixed":
            raise ValueError("tempering requires a fixed Andersen collision schedule")

    method = None
    cv_dimension = 0
    analysis_start = neq
    exchange_start = 0
    if task == "fes":
        method, cv, method_options = _resolve_fes(config, system)
        cv_dimension = cv.dimension
        if method == "metadynamics":
            if mode == "hamiltonian" and target == "bias":
                raise ValueError(
                    "Hamiltonian tempering of an evolving metadynamics bias is undefined; scale the PES or use umbrella FES"
                )
            analysis_start = (
                neq + method_options["freeze"] + method_options["freeze_equilibration_steps"]
            )
            exchange_start = analysis_start
            if analysis_start >= neq + nprod:
                raise ValueError("frozen metadynamics leaves no production interval for tempering analysis")
    if (neq + nprod - analysis_start) // interval < 2:
        raise ValueError("tempering analysis requires at least two saved post-equilibration samples")

    seed = checked_integer(table.get("seed", config.get("seed", 0)), "tempering.seed", 0)
    return TemperingOptions(
        mode=mode,
        temperatures=np.asarray(temperatures, dtype=float),
        betas=np.asarray(betas, dtype=float),
        lambdas=np.asarray(lambdas, dtype=float),
        target=target,
        nstates=nstates,
        exchange_interval=_exchange_interval(table, dt),
        pairing=_pairing(table.get("pairing", "odd_even")),
        seed=seed,
        neq=neq,
        nprod=nprod,
        total=neq + nprod,
        sample_interval=interval,
        checkpoint_interval=checkpoint,
        analysis_start=analysis_start,
        exchange_start=exchange_start,
        thermostat=thermostat,
        sampling_method=sampling_method,
        gamma_centroid=gamma,
        andersen_interval=andersen,
        fes_method=method,
        cv_dimension=cv_dimension,
        dt=dt,
    )


def validate_tempering(config, system):
    """Read-only validation hook for input and direct resolved callers."""
    return resolve_tempering(config, system)


def _identity(config, system, options):
    return _fingerprint(
        config, system,
        {"kind": CHECKPOINT_KIND, "tempering": options.identity()},
    )


def _checkpoint_path(output):
    return Path(output) / "checkpoints" / "tempering.npz"


def _stage_path(output, generation, walker):
    return Path(output) / "checkpoints" / f".tempering-stage-{generation:012d}-{walker:06d}.npz"


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _read_json(value, name):
    try:
        result = json.loads(str(np.asarray(value).item()))
    except (ValueError, TypeError, AttributeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid tempering checkpoint {name}") from error
    return result


def _validate_rng(state, name):
    if not isinstance(state, dict):
        raise ValueError(f"invalid tempering checkpoint {name}")
    probe = np.random.default_rng(0)
    try:
        probe.bit_generator.state = copy.deepcopy(state)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid tempering checkpoint {name}") from error
    return state


def _empty_history(options, system):
    # This helper's shapes need the caller's bead count; initialize only fields
    # whose dimensions are independent of beads here.
    return {
        "sample_steps": np.empty(0, dtype=np.int64),
        "sample_slots": np.empty((0, options.nstates), dtype=np.int64),
        "exchange_steps": np.asarray([0], dtype=np.int64),
        "assignment_history": np.arange(options.nstates, dtype=np.int64)[None, :],
    }


def _initial_history(config, system, options):
    result = _empty_history(options, system)
    r, p, n = options.nstates, config["beads"], system.natoms
    result.update({
        "sample_positions": np.empty((0, r, p, n, 3), dtype=float),
        "sample_u": np.empty((0, r, r), dtype=float),
        "sample_estimators": np.empty((0, r, r, len(ESTIMATOR_NAMES)), dtype=float),
        "sample_cv": np.empty((0, r, options.cv_dimension), dtype=float),
    })
    return result


def _write_checkpoint(path, config, system, options, data):
    atomic_savez(
        path,
        kind=CHECKPOINT_KIND,
        fingerprint=np.asarray(_identity(config, system, options)),
        elapsed=np.asarray(data["elapsed"], dtype=np.int64),
        generation=np.asarray(data["generation"], dtype=np.int64),
        q=np.asarray(data["q"], dtype=float),
        p=np.asarray(data["p"], dtype=float),
        rng=np.asarray(_json(data["rng"])),
        slot_of_walker=np.asarray(data["slot_of_walker"], dtype=np.int64),
        attempts=np.asarray(data["attempts"], dtype=np.int64),
        accepts=np.asarray(data["accepts"], dtype=np.int64),
        exchange_rng=np.asarray(_json(data["exchange_rng"])),
        sweep=np.asarray(data["sweep"], dtype=np.int64),
        propagation=np.asarray(_json(data["propagation"])),
        bias_states=np.asarray(_json(data["bias_states"])),
        sample_steps=np.asarray(data["sample_steps"], dtype=np.int64),
        sample_slots=np.asarray(data["sample_slots"], dtype=np.int64),
        sample_positions=np.asarray(data["sample_positions"], dtype=float),
        sample_u=np.asarray(data["sample_u"], dtype=float),
        sample_estimators=np.asarray(data["sample_estimators"], dtype=float),
        sample_cv=np.asarray(data["sample_cv"], dtype=float),
        exchange_steps=np.asarray(data["exchange_steps"], dtype=np.int64),
        assignment_history=np.asarray(data["assignment_history"], dtype=np.int64),
    )


def _integer_scalar(value, name, minimum=0):
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "iu":
        raise ValueError(f"tempering checkpoint has invalid {name}")
    result = int(array)
    if result < minimum:
        raise ValueError(f"tempering checkpoint has invalid {name}")
    return result


def _read_checkpoint(config, system, output, options, *, history=True):
    path = _checkpoint_path(output)
    try:
        context = load_checkpoint(path, expected_kind=CHECKPOINT_KIND, require_integrity=True)
    except (OSError, CheckpointError) as error:
        raise ValueError(f"cannot read complete tempering ensemble checkpoint: {path}") from error
    with context as saved:
        if str(saved["fingerprint"].item()) != _identity(config, system, options):
            raise ValueError("tempering checkpoint physics differs from the resolved input")
        elapsed = _integer_scalar(saved["elapsed"], "elapsed")
        generation = _integer_scalar(saved["generation"], "generation")
        if elapsed > options.total:
            raise ValueError("tempering checkpoint clock exceeds the authorized run")
        r, p, n = options.nstates, config["beads"], system.natoms
        q, momenta = saved["q"], saved["p"]
        shape = (r, p, n, 3)
        if (q.shape != shape or momenta.shape != shape
                or q.dtype.kind not in "f" or momenta.dtype.kind not in "f"
                or not np.isfinite(q).all() or not np.isfinite(momenta).all()):
            raise ValueError("tempering checkpoint is missing a valid state for every replica")
        if system.fixed_atoms:
            fixed = np.asarray(system.fixed_atoms, dtype=int)
            expected = np.broadcast_to(system.geometry[fixed], (r, p, len(fixed), 3))
            if not np.array_equal(q[:, :, fixed], expected) or np.any(momenta[:, :, fixed] != 0.0):
                raise ValueError("tempering checkpoint violates fixed-atom constraints")
        slots = saved["slot_of_walker"]
        if (slots.shape != (r,) or slots.dtype.kind not in "iu"
                or not np.array_equal(np.sort(slots), np.arange(r))):
            raise ValueError("tempering checkpoint replica-to-state mapping is not a permutation")
        attempts, accepts = saved["attempts"], saved["accepts"]
        if (attempts.shape != (r - 1,) or accepts.shape != (r - 1,)
                or attempts.dtype.kind not in "iu" or accepts.dtype.kind not in "iu"
                or np.any(attempts < 0) or np.any(accepts < 0) or np.any(accepts > attempts)):
            raise ValueError("tempering checkpoint has invalid acceptance counters")
        rng = _read_json(saved["rng"], "replica RNG states")
        if not isinstance(rng, list) or len(rng) != r:
            raise ValueError("tempering checkpoint RNG count differs from the ladder")
        rng = [_validate_rng(value, f"replica RNG {index}") for index, value in enumerate(rng)]
        exchange_rng = _validate_rng(_read_json(saved["exchange_rng"], "exchange RNG"),
                                     "exchange RNG")
        propagation = _read_json(saved["propagation"], "propagation metadata")
        if (not isinstance(propagation, list) or len(propagation) != r
                or any(not isinstance(item, dict) for item in propagation)):
            raise ValueError("tempering checkpoint propagation metadata count differs from the ladder")
        bias_states = _read_json(saved["bias_states"], "bias states")
        if not isinstance(bias_states, list) or len(bias_states) != r:
            raise ValueError("tempering checkpoint bias-state count differs from the ladder")
        result = {
            "elapsed": elapsed,
            "generation": generation,
            "q": q,
            "p": momenta,
            "rng": rng,
            "slot_of_walker": slots.astype(np.int64, copy=False),
            "attempts": attempts.astype(np.int64, copy=False),
            "accepts": accepts.astype(np.int64, copy=False),
            "exchange_rng": exchange_rng,
            "sweep": _integer_scalar(saved["sweep"], "exchange sweep"),
            "propagation": propagation,
            "bias_states": bias_states,
        }
        if not history:
            return result
        sample_steps = saved["sample_steps"]
        frames = len(sample_steps) if sample_steps.ndim == 1 else -1
        if (frames < 0 or sample_steps.dtype.kind not in "iu"
                or np.any(sample_steps <= options.analysis_start)
                or np.any(sample_steps > elapsed)
                or np.any(np.diff(sample_steps) <= 0)
                or np.any((sample_steps - options.analysis_start) % options.sample_interval != 0)):
            raise ValueError("tempering checkpoint sample clock is invalid")
        shapes = {
            "sample_slots": (frames, r),
            "sample_positions": (frames, r, p, n, 3),
            "sample_u": (frames, r, r),
            "sample_estimators": (frames, r, r, len(ESTIMATOR_NAMES)),
            "sample_cv": (frames, r, options.cv_dimension),
        }
        result["sample_steps"] = sample_steps.astype(np.int64, copy=False)
        for name, expected_shape in shapes.items():
            value = saved[name]
            if (value.shape != expected_shape or value.dtype.kind not in "iuf"
                    or not np.isfinite(value).all()):
                raise ValueError(f"tempering checkpoint has invalid {name}")
            result[name] = value
        if any(not np.array_equal(np.sort(row), np.arange(r))
               for row in result["sample_slots"]):
            raise ValueError("tempering sample assignments are not permutations")
        exchange_steps = saved["exchange_steps"]
        assignments = saved["assignment_history"]
        if (exchange_steps.ndim != 1 or exchange_steps.dtype.kind not in "iu"
                or len(exchange_steps) < 1 or exchange_steps[0] != 0
                or np.any(np.diff(exchange_steps) <= 0) or exchange_steps[-1] > elapsed
                or assignments.shape != (len(exchange_steps), r)
                or assignments.dtype.kind not in "iu"
                or any(not np.array_equal(np.sort(row), np.arange(r)) for row in assignments)):
            raise ValueError("tempering checkpoint exchange history is invalid")
        result["exchange_steps"] = exchange_steps.astype(np.int64, copy=False)
        result["assignment_history"] = assignments.astype(np.int64, copy=False)
        return result


def _initial_checkpoint(config, system, output, options):
    r, p, n = options.nstates, config["beads"], system.natoms
    q = np.broadcast_to(system.geometry, (r, p, n, 3)).copy()
    momenta = np.empty_like(q)
    rng_states = []
    for walker, beta in enumerate(options.betas):
        rng = np.random.default_rng(
            np.random.SeedSequence([int(config.get("seed", 0)), options.seed, 3701, walker])
        )
        momenta[walker] = rng.normal(size=q[walker].shape) * np.sqrt(
            system.masses[None, :, None] * p / beta
        )
        if system.fixed_atoms:
            momenta[walker, :, list(system.fixed_atoms)] = 0.0
        rng_states.append(copy.deepcopy(rng.bit_generator.state))
    exchange_rng = np.random.default_rng(
        np.random.SeedSequence([int(config.get("seed", 0)), options.seed, 3702])
    )
    data = {
        "elapsed": 0,
        "generation": 0,
        "q": q,
        "p": momenta,
        "rng": rng_states,
        "slot_of_walker": np.arange(r, dtype=np.int64),
        "attempts": np.zeros(r - 1, dtype=np.int64),
        "accepts": np.zeros(r - 1, dtype=np.int64),
        "exchange_rng": copy.deepcopy(exchange_rng.bit_generator.state),
        "sweep": 0,
        "propagation": [
            {"native_steps": 0, "python_steps": 0, "fallback_reasons": {}}
            for _ in range(r)
        ],
        "bias_states": [[] for _ in range(r)],
        **_initial_history(config, system, options),
    }
    _write_checkpoint(_checkpoint_path(output), config, system, options, data)
    return data


def _staged_files(output, generation):
    pattern = f".tempering-stage-{generation:012d}-*.npz"
    return sorted((Path(output) / "checkpoints").glob(pattern))


def _prepare_checkpoint(config, system, output, options):
    path = _checkpoint_path(output)
    resume = bool(config.get("resume", False))
    if path.exists():
        if not resume:
            raise ValueError(
                "tempering checkpoint already exists; use resume or choose a fresh output directory"
            )
        saved = _read_checkpoint(config, system, output, options, history=False)
    else:
        if resume:
            raise ValueError("tempering resume requires the complete ensemble checkpoint")
        saved = _initial_checkpoint(config, system, output, options)
    stages = _staged_files(output, saved["generation"])
    if stages and len(stages) != options.nstates:
        raise ValueError(
            "tempering resume found a partial replica generation; restore all stage files or remove the incomplete run"
        )
    return {
        "elapsed": saved["elapsed"],
        "generation": saved["generation"],
        "complete": saved["elapsed"] == options.total,
        "staged": [str(path) for path in stages],
    }


class _ScaledBias:
    def __init__(self, bias, scale):
        self.bias = bias
        self.scale = float(scale)

    def energy_force(self, q):
        energy, force = self.bias.energy_force(q)
        return self.scale * float(energy), self.scale * np.asarray(force, dtype=float)


class _ScaledPotential:
    """A borrowed state Hamiltonian; ownership remains with the task lease."""

    def __init__(self, provider, scale, target):
        self.provider = provider
        self.scale = float(scale)
        self.target = target
        metadata = dict(getattr(provider, "metadata", {}) or {})
        metadata.update(provider="tempering-scaled", stateful=False,
                        time_dependent=False, history_dependent=False)
        self.metadata = metadata
        if target == "split_slow" and not isinstance(provider, SplitPotential):
            raise ValueError("tempering.target='split_slow' requires a SplitPotential provider")

    @staticmethod
    def _scaled(result, scale):
        return PESResult(
            result.energy * scale,
            result.forces * scale,
            result.status.copy(),
            result.messages,
            None if result.virial is None else result.virial * scale,
        )

    def evaluate(self, positions):
        q = np.asarray(positions)
        if self.target == "potential":
            result = evaluate_checked(self.provider, q)
            return self._scaled(result, self.scale)
        if self.target == "split_slow":
            fast = validate_result(self.provider.evaluate_fast(q), q.shape)
            slow = validate_result(self.provider.evaluate_slow(q), q.shape)
            virial = None
            if fast.virial is not None and slow.virial is not None:
                virial = fast.virial + self.scale * slow.virial
            return PESResult(
                fast.energy + self.scale * slow.energy,
                fast.forces + self.scale * slow.forces,
                np.maximum(fast.status, slow.status),
                tuple(a if a else b for a, b in zip(fast.messages, slow.messages)),
                virial,
            )
        return self.provider.evaluate(q)


def _slot_biases(config, system, options, bias_states):
    if options.fes_method is None:
        return [None] * options.nstates, None, None
    if options.fes_method == "umbrella":
        from .fes import resolve_window_plan

        cv, _, windows = resolve_window_plan(config, system)
        window = windows[0]
        biases = [UmbrellaBias(cv, window.center, window.stiffness)
                  for _ in range(options.nstates)]
        return biases, cv, window
    from .metadynamics import _new_bias, resolve_metadynamics

    cv, _, meta = resolve_metadynamics(config, system)
    biases = [
        _new_bias(cv, meta, beta, hills=bias_states[index])
        for index, beta in enumerate(options.betas)
    ]
    return biases, cv, meta


def _effective_bias(base, scale, target):
    if base is None:
        return None
    return _ScaledBias(base, scale) if target == "bias" else base


def _provider_components(provider, q, target):
    if target == "split_slow":
        if not isinstance(provider, SplitPotential):
            raise ValueError("tempering.target='split_slow' requires a SplitPotential provider")
        fast = validate_result(provider.evaluate_fast(q), q.shape)
        slow = validate_result(provider.evaluate_slow(q), q.shape)
        return (
            float(np.sum(fast.energy)), fast.forces.copy(),
            float(np.sum(slow.energy)), slow.forces.copy(),
        )
    base = evaluate_checked(provider, q)
    return float(np.sum(base.energy)), base.forces.copy(), 0.0, np.zeros_like(q)


def evaluate_thermodynamic_states(provider, q, config, system, options, biases, cv=None):
    """Return all target reduced potentials and beta-explicit PIMD estimators.

    This function is intentionally public within the module for analytic tests.
    Bias energies follow the project convention and already contain the bead
    factor P. Forces are physical ``-dH/dq`` and are never bead-scaled here.
    """
    q = np.asarray(q, dtype=float)
    expected = (config["beads"], system.natoms, 3)
    if q.shape != expected or not np.isfinite(q).all():
        raise ValueError("tempering coordinates must be finite [P,N,3]")
    if len(biases) != options.nstates:
        raise ValueError("tempering bias count differs from the ladder")
    base_energy, base_force, slow_energy, slow_force = _provider_components(
        provider, q, options.target
    )
    p = config["beads"]
    masses = np.asarray(system.masses)[None, :, None]
    displacement = q - np.roll(q, -1, axis=0)
    spring_quadratic = float(np.sum(masses * displacement * displacement))
    offsets = q - q.mean(axis=0, keepdims=True)
    mobile = np.asarray(system.mobile_mask, dtype=bool)
    degrees = 3 * int(np.sum(mobile))
    reduced = np.empty(options.nstates, dtype=float)
    estimators = np.empty((options.nstates, len(ESTIMATOR_NAMES)), dtype=float)
    for index, (beta, scale, bias) in enumerate(zip(
            options.betas, options.lambdas, biases)):
        if bias is None:
            bias_energy = 0.0
            bias_force = np.zeros_like(q)
        else:
            bias_energy, bias_force = bias.energy_force(q)
            bias_energy = finite_real(bias_energy, "tempering bias energy")
            bias_force = np.asarray(bias_force, dtype=float)
            if bias_force.shape != q.shape or not np.isfinite(bias_force).all():
                raise ValueError("tempering bias returned invalid forces")
        if options.target == "potential":
            physical_energy = scale * base_energy + bias_energy
            force = scale * base_force + bias_force
        elif options.target == "split_slow":
            physical_energy = base_energy + scale * slow_energy + bias_energy
            force = base_force + scale * slow_force + bias_force
        elif options.target == "bias":
            physical_energy = base_energy + scale * bias_energy
            force = base_force + scale * bias_force
        else:
            physical_energy = base_energy + bias_energy
            force = base_force + bias_force
        spring = 0.5 * (p / beta) ** 2 * spring_quadratic
        reduced[index] = beta / p * (spring + physical_energy)
        primitive = degrees * p / (2.0 * beta) - spring / p
        centroid_virial = degrees / (2.0 * beta) - float(
            np.sum(offsets[:, mobile] * force[:, mobile])
        ) / (2.0 * p)
        potential = physical_energy / p
        estimators[index] = (
            potential,
            primitive,
            centroid_virial,
            potential + centroid_virial,
        )
    if not np.isfinite(reduced).all() or not np.isfinite(estimators).all():
        raise FloatingPointError("tempering reduced energies or estimators are nonfinite")
    if cv is None:
        cv_value = np.empty(0, dtype=float)
    else:
        cv_value = np.asarray(cv.value(q), dtype=float)
    if cv_value.shape != (options.cv_dimension,) or not np.isfinite(cv_value).all():
        raise ValueError("tempering CV returned an invalid value")
    return reduced, estimators, cv_value


def _dynamics(config, system, provider, beta, rng, options):
    from ..propagation import loop_options

    kwargs = dict(
        seed=rng,
        native=config.get("native", config.get("dynamics", {}).get("native", True)),
        threads=config.get("parallel", {}).get("threads_per_rank", 1),
        cell=system.cell,
        pbc=system.pbc,
        **loop_options(config),
    )
    if system.fixed_atoms:
        kwargs.update(fixed_atoms=system.fixed_atoms, fixed_positions=system.geometry)
    return Dynamics(provider, system.masses, beta, config["beads"], options.dt, **kwargs)


def _advance_one(dynamics, state, bias, options):
    if options.thermostat == "pile-l":
        dynamics.sample_step(
            state,
            bias,
            method=options.sampling_method,
            gamma_centroid=options.gamma_centroid,
        )
        dynamics._record_python_steps(1)
        return
    if state.step % options.andersen_interval == 0:
        state.p[:] = dynamics.sample_momenta()
    dynamics.propagate(state, 1, bias=bias, record_centroids=False)


def _propagate_replica(task):
    config, system = task["config"], task["system"]
    output = Path(task["output"])
    walker, stop = int(task["walker"]), int(task["stop"])
    options = resolve_tempering(config, system)
    saved = _read_checkpoint(config, system, output, options, history=False)
    if (saved["generation"] != task["generation"]
            or saved["elapsed"] != task["start"]):
        raise ValueError("tempering task generation no longer matches the ensemble checkpoint")
    slot = int(saved["slot_of_walker"][walker])
    rng = np.random.default_rng(0)
    rng.bit_generator.state = copy.deepcopy(saved["rng"][walker])
    base_biases, cv, method_options = _slot_biases(
        config, system, options, saved["bias_states"]
    )
    active_base_bias = base_biases[slot]
    active_bias = _effective_bias(active_base_bias, options.lambdas[slot], options.target)
    sample_steps, positions, reduced, estimators, cvs = [], [], [], [], []
    with task_potential(
        config["potential"], system,
        reuse=config.get("parallel", {}).get("reuse_potentials", True),
    ) as provider:
        _static(provider)
        active_provider = (
            _ScaledPotential(provider, options.lambdas[slot], options.target)
            if options.target in ("potential", "split_slow")
            else provider
        )
        dynamics = _dynamics(
            config, system, active_provider, options.betas[slot], rng, options
        )
        dynamics.propagation_stats = copy.deepcopy(saved["propagation"][walker])
        state = State(
            saved["q"][walker], saved["p"][walker], saved["elapsed"],
            saved["elapsed"] * options.dt,
        )
        while state.step < stop:
            _advance_one(dynamics, state, active_bias, options)
            if options.fes_method == "metadynamics":
                production = state.step - options.neq
                if (production > 0
                        and production % method_options["pace"] == 0
                        and production <= method_options["freeze"]):
                    active_base_bias.deposit(
                        cv.value(state.q), step=state.step, time=state.time
                    )
            if (state.step > options.analysis_start
                    and (state.step - options.analysis_start) % options.sample_interval == 0):
                values = evaluate_thermodynamic_states(
                    provider, state.q, config, system, options, base_biases, cv
                )
                sample_steps.append(state.step)
                positions.append(state.q.copy())
                reduced.append(values[0])
                estimators.append(values[1])
                cvs.append(values[2])
        exchange_u, _, _ = evaluate_thermodynamic_states(
            provider, state.q, config, system, options, base_biases, cv
        )
    bias_state = (
        active_base_bias.hills if options.fes_method == "metadynamics" else []
    )
    count = len(sample_steps)
    stage = _stage_path(output, saved["generation"], walker)
    atomic_savez(
        stage,
        kind=STAGE_KIND,
        fingerprint=np.asarray(_identity(config, system, options)),
        generation=np.asarray(saved["generation"], dtype=np.int64),
        walker=np.asarray(walker, dtype=np.int64),
        slot=np.asarray(slot, dtype=np.int64),
        start=np.asarray(saved["elapsed"], dtype=np.int64),
        stop=np.asarray(stop, dtype=np.int64),
        q=state.q,
        p=state.p,
        rng=np.asarray(_json(rng.bit_generator.state)),
        propagation=np.asarray(_json(dynamics.propagation_stats)),
        bias_state=np.asarray(_json(bias_state)),
        sample_steps=np.asarray(sample_steps, dtype=np.int64),
        sample_positions=np.asarray(positions, dtype=float).reshape(
            count, config["beads"], system.natoms, 3
        ),
        sample_u=np.asarray(reduced, dtype=float).reshape(count, options.nstates),
        sample_estimators=np.asarray(estimators, dtype=float).reshape(
            count, options.nstates, len(ESTIMATOR_NAMES)
        ),
        sample_cv=np.asarray(cvs, dtype=float).reshape(count, options.cv_dimension),
        exchange_u=exchange_u,
    )
    return {
        "walker": walker,
        "slot": slot,
        "start": saved["elapsed"],
        "stop": stop,
        "generation": saved["generation"],
        "stage": str(stage),
        "exchange_u": exchange_u.tolist(),
    }


def _stage_result_from_path(path, options):
    name = Path(path).name
    prefix = ".tempering-stage-"
    if not name.startswith(prefix) or not name.endswith(".npz"):
        raise ValueError("invalid tempering stage filename")
    cells = name[len(prefix):-4].split("-")
    if len(cells) != 2 or not all(cell.isdigit() for cell in cells):
        raise ValueError("invalid tempering stage filename")
    generation, walker = map(int, cells)
    if not 0 <= walker < options.nstates:
        raise ValueError("tempering stage walker is outside the ladder")
    return {"generation": generation, "walker": walker, "stage": str(path)}


def _read_stage(result, config, system, output, options):
    if not isinstance(result, Mapping):
        raise ValueError("tempering scheduler returned a malformed replica result")
    required = {"generation", "walker", "stage"}
    if not required.issubset(result):
        raise ValueError("tempering scheduler returned a malformed replica result")
    generation = checked_integer(result["generation"], "tempering stage generation", 0)
    walker = checked_integer(result["walker"], "tempering stage walker", 0)
    if walker >= options.nstates:
        raise ValueError("tempering stage walker is outside the ladder")
    expected = _stage_path(output, generation, walker)
    path = Path(result["stage"])
    if path != expected:
        raise ValueError("tempering worker returned an unexpected stage path")
    try:
        context = load_checkpoint(path, expected_kind=STAGE_KIND, require_integrity=True)
    except (OSError, CheckpointError) as error:
        raise ValueError(f"tempering stage is missing or damaged: {path}") from error
    with context as saved:
        if str(saved["fingerprint"].item()) != _identity(config, system, options):
            raise ValueError("tempering stage physics differs from the ensemble checkpoint")
        scalar = {
            name: _integer_scalar(saved[name], f"stage {name}")
            for name in ("generation", "walker", "slot", "start", "stop")
        }
        if scalar["generation"] != generation or scalar["walker"] != walker:
            raise ValueError("tempering stage identity differs from its filename")
        for name in ("slot", "start", "stop"):
            if name in result and int(result[name]) != scalar[name]:
                raise ValueError("tempering stage clocks differ from the worker result")
        q, momenta = saved["q"], saved["p"]
        shape = (config["beads"], system.natoms, 3)
        if (q.shape != shape or momenta.shape != shape
                or q.dtype.kind != "f" or momenta.dtype.kind != "f"
                or not np.isfinite(q).all() or not np.isfinite(momenta).all()):
            raise ValueError("tempering stage has an invalid replica state")
        steps = saved["sample_steps"]
        count = len(steps) if steps.ndim == 1 else -1
        shapes = {
            "sample_positions": (count, config["beads"], system.natoms, 3),
            "sample_u": (count, options.nstates),
            "sample_estimators": (count, options.nstates, len(ESTIMATOR_NAMES)),
            "sample_cv": (count, options.cv_dimension),
        }
        samples = {"sample_steps": steps}
        if (count < 0 or steps.dtype.kind not in "iu"
                or np.any(steps <= scalar["start"]) or np.any(steps > scalar["stop"])
                or np.any(np.diff(steps) <= 0)):
            raise ValueError("tempering stage has an invalid sample clock")
        for name, shape in shapes.items():
            value = saved[name]
            if (value.shape != shape or value.dtype.kind not in "iuf"
                    or not np.isfinite(value).all()):
                raise ValueError(f"tempering stage has invalid {name}")
            samples[name] = value
        exchange_u = saved["exchange_u"]
        if (exchange_u.shape != (options.nstates,)
                or exchange_u.dtype.kind not in "fiu"
                or not np.isfinite(exchange_u).all()):
            raise ValueError("tempering stage has invalid exchange reduced energies")
        if "exchange_u" in result:
            returned = np.asarray(result["exchange_u"], dtype=float)
            if not np.array_equal(exchange_u, returned):
                raise ValueError("tempering stage energies differ from the scheduler result")
        rng = _validate_rng(_read_json(saved["rng"], "stage RNG"), "stage RNG")
        propagation = _read_json(saved["propagation"], "stage propagation")
        if not isinstance(propagation, dict):
            raise ValueError("tempering stage propagation metadata is invalid")
        bias_state = _read_json(saved["bias_state"], "stage bias state")
        if not isinstance(bias_state, list):
            raise ValueError("tempering stage bias state is invalid")
        return {
            **scalar,
            "path": path,
            "q": q,
            "p": momenta,
            "rng": rng,
            "propagation": propagation,
            "bias_state": bias_state,
            "exchange_u": exchange_u,
            **samples,
        }


def _exchange_due(step, options):
    return (step > options.exchange_start
            and (step - options.exchange_start) % options.exchange_interval == 0)


def _commit_generation(config, system, output, options, results):
    current = _read_checkpoint(config, system, output, options, history=True)
    r = options.nstates
    if not isinstance(results, list) or len(results) != r:
        raise ValueError("tempering generation is missing one or more replica results")
    by_walker = {}
    for result in results:
        if not isinstance(result, Mapping):
            raise ValueError("tempering scheduler returned a malformed replica result")
        walker = result.get("walker")
        if walker in by_walker:
            raise ValueError("tempering scheduler returned duplicate replica identities")
        by_walker[walker] = result
    if set(by_walker) != set(range(r)):
        raise ValueError("tempering generation is missing one or more replica identities")
    stages = [
        _read_stage(by_walker[index], config, system, output, options)
        for index in range(r)
    ]
    stop = stages[0]["stop"]
    if stop <= current["elapsed"] or stop > options.total:
        raise ValueError("tempering staged generation has an invalid stop clock")
    for stage in stages:
        walker = stage["walker"]
        if (stage["generation"] != current["generation"]
                or stage["start"] != current["elapsed"]
                or stage["stop"] != stop
                or stage["slot"] != int(current["slot_of_walker"][walker])):
            raise ValueError("tempering stages do not form one synchronized ensemble generation")
    clocks = stages[0]["sample_steps"]
    if any(not np.array_equal(stage["sample_steps"], clocks) for stage in stages[1:]):
        raise ValueError("tempering replica sample clocks diverged")

    q = np.stack([stage["q"] for stage in stages])
    momenta = np.stack([stage["p"] for stage in stages])
    rng = [stage["rng"] for stage in stages]
    propagation = [stage["propagation"] for stage in stages]
    bias_states = copy.deepcopy(current["bias_states"])
    for stage in stages:
        bias_states[stage["slot"]] = stage["bias_state"]
    slots = current["slot_of_walker"].copy()
    attempts = current["attempts"].copy()
    accepts = current["accepts"].copy()
    exchange_rng = np.random.default_rng(0)
    exchange_rng.bit_generator.state = copy.deepcopy(current["exchange_rng"])
    sweep = current["sweep"]
    exchanged = False
    if _exchange_due(stop, options):
        energy = np.stack([stage["exchange_u"] for stage in stages])
        for left, right in exchange_pairs(r, options.pairing, sweep):
            walker_at_slot = np.empty(r, dtype=np.int64)
            walker_at_slot[slots] = np.arange(r, dtype=np.int64)
            first, second = int(walker_at_slot[left]), int(walker_at_slot[right])
            log_ratio = float(
                energy[first, left] + energy[second, right]
                - energy[first, right] - energy[second, left]
            )
            attempts[left] += 1
            if metropolis_accept(log_ratio, exchange_rng):
                if options.mode == "temperature":
                    momenta[first] *= temperature_momentum_scale(
                        options.betas[left], options.betas[right]
                    )
                    momenta[second] *= temperature_momentum_scale(
                        options.betas[right], options.betas[left]
                    )
                slots[first], slots[second] = right, left
                accepts[left] += 1
            exchanged = True
        sweep += 1

    count = len(clocks)
    if count:
        sample_slots = np.broadcast_to(
            current["slot_of_walker"], (count, r)
        ).copy()
        history = {
            "sample_steps": np.concatenate((current["sample_steps"], clocks)),
            "sample_slots": np.concatenate((current["sample_slots"], sample_slots)),
            "sample_positions": np.concatenate((
                current["sample_positions"],
                np.stack([stage["sample_positions"] for stage in stages], axis=1),
            )),
            "sample_u": np.concatenate((
                current["sample_u"],
                np.stack([stage["sample_u"] for stage in stages], axis=1),
            )),
            "sample_estimators": np.concatenate((
                current["sample_estimators"],
                np.stack([stage["sample_estimators"] for stage in stages], axis=1),
            )),
            "sample_cv": np.concatenate((
                current["sample_cv"],
                np.stack([stage["sample_cv"] for stage in stages], axis=1),
            )),
        }
    else:
        history = {
            name: current[name]
            for name in (
                "sample_steps", "sample_slots", "sample_positions", "sample_u",
                "sample_estimators", "sample_cv",
            )
        }
    if exchanged:
        exchange_steps = np.concatenate((current["exchange_steps"], [stop]))
        assignment_history = np.concatenate(
            (current["assignment_history"], slots[None, :]), axis=0
        )
    else:
        exchange_steps = current["exchange_steps"]
        assignment_history = current["assignment_history"]
    data = {
        "elapsed": stop,
        "generation": current["generation"] + 1,
        "q": q,
        "p": momenta,
        "rng": rng,
        "slot_of_walker": slots,
        "attempts": attempts,
        "accepts": accepts,
        "exchange_rng": copy.deepcopy(exchange_rng.bit_generator.state),
        "sweep": sweep,
        "propagation": propagation,
        "bias_states": bias_states,
        "exchange_steps": exchange_steps,
        "assignment_history": assignment_history,
        **history,
    }
    _write_checkpoint(_checkpoint_path(output), config, system, options, data)
    for stage in stages:
        try:
            stage["path"].unlink()
        except FileNotFoundError:
            pass
    return {
        "elapsed": stop,
        "generation": data["generation"],
        "complete": stop == options.total,
        "staged": [],
    }


def _recover_staged(config, system, output, options, staged):
    results = [_stage_result_from_path(path, options) for path in staged]
    # Load enough scalar metadata to reconstruct the same result object that
    # run_tasks would have returned. The commit path validates every field.
    enriched = []
    for result in results:
        stage = _read_stage(result, config, system, output, options)
        enriched.append({
            "generation": stage["generation"],
            "walker": stage["walker"],
            "slot": stage["slot"],
            "start": stage["start"],
            "stop": stage["stop"],
            "stage": str(stage["path"]),
            "exchange_u": stage["exchange_u"].tolist(),
        })
    return _commit_generation(config, system, output, options, enriched)


def _next_boundary(elapsed, options):
    candidates = [options.total]
    checkpoint = ((elapsed // options.checkpoint_interval) + 1) * options.checkpoint_interval
    candidates.append(checkpoint)
    if elapsed < options.analysis_start:
        candidates.append(options.analysis_start)
    if elapsed < options.exchange_start:
        candidates.append(options.exchange_start)
    else:
        interval = options.exchange_interval
        candidates.append(
            elapsed + (interval - (elapsed - options.exchange_start) % interval)
        )
    return min(value for value in candidates if value > elapsed)


def _ordered_samples(saved, options):
    frames, r = len(saved["sample_steps"]), options.nstates
    rows = []
    for slot in range(r):
        for frame in range(frames):
            matches = np.flatnonzero(saved["sample_slots"][frame] == slot)
            if len(matches) != 1:
                raise ValueError("tempering sample assignments do not identify one walker per state")
            rows.append((frame, int(matches[0])))
    return rows


def _mbar_analysis(saved, options):
    frames, r = len(saved["sample_steps"]), options.nstates
    if frames == 0:
        return None
    order = _ordered_samples(saved, options)
    u_nk = np.asarray([
        saved["sample_u"][frame, walker]
        for frame, walker in order
    ])
    estimator_nke = np.asarray([
        saved["sample_estimators"][frame, walker]
        for frame, walker in order
    ])
    cv_n = np.asarray([
        saved["sample_cv"][frame, walker]
        for frame, walker in order
    ])
    counts = np.full(r, frames, dtype=np.int64)
    u_kn = u_nk.T
    fits = [
        solve_mbar(
            u_kn,
            counts,
            target_u=u_kn[target],
            return_state_weights=False,
        )
        for target in range(r)
    ]
    estimates = []
    for target, fit in enumerate(fits):
        values = estimator_nke[:, target]
        row = {
            name: float(np.sum(fit.weights * values[:, index]))
            for index, name in enumerate(ESTIMATOR_NAMES)
        }
        row["weight_effective_sample_size"] = float(
            1.0 / np.sum(fit.weights * fit.weights)
        )
        if options.cv_dimension:
            row["cv_mean"] = np.sum(fit.weights[:, None] * cv_n, axis=0).tolist()
        estimates.append(row)
    reference = fits[0]
    components = reference.components.tolist()
    return {
        "solver_converged": bool(all(fit.converged for fit in fits)),
        "residuals": [float(fit.residual) for fit in fits],
        "free_energies_dimensionless": reference.free_energies.tolist(),
        "overlap": reference.overlap.tolist(),
        "components": components,
        "connected": len(set(components)) == 1,
        "counts": counts.tolist(),
        "estimators_by_state": estimates,
        "temperature_dependence": (
            "Each target state uses its own beta in the ring-spring action, "
            "primitive kinetic estimator, centroid-virial estimator and "
            "well-tempered metadynamics bias."
        ),
    }


def _direct_estimators(saved, options):
    frames, r = len(saved["sample_steps"]), options.nstates
    if frames == 0:
        return [dict.fromkeys(ESTIMATOR_NAMES) for _ in range(r)]
    result = []
    for slot in range(r):
        values = []
        for frame in range(frames):
            walker = int(np.flatnonzero(saved["sample_slots"][frame] == slot)[0])
            values.append(saved["sample_estimators"][frame, walker, slot])
        array = np.asarray(values)
        result.append({
            name: float(np.mean(array[:, index]))
            for index, name in enumerate(ESTIMATOR_NAMES)
        })
    return result


def _write_trajectories(saved, config, system, output, options):
    files = []
    frames = len(saved["sample_steps"])
    prefix = "temperature" if options.mode == "temperature" else "hamiltonian"
    for slot in range(options.nstates):
        positions, reduced, estimators, cvs, walkers = [], [], [], [], []
        for frame in range(frames):
            walker = int(np.flatnonzero(saved["sample_slots"][frame] == slot)[0])
            walkers.append(walker)
            positions.append(saved["sample_positions"][frame, walker])
            reduced.append(saved["sample_u"][frame, walker])
            estimators.append(saved["sample_estimators"][frame, walker, slot])
            cvs.append(saved["sample_cv"][frame, walker])
        positions = np.asarray(positions, dtype=float).reshape(
            frames, config["beads"], system.natoms, 3
        )
        path = Path(output) / "samples" / f"{prefix}_{slot:04d}.npz"
        metadata = {
            "format": TRAJECTORY_KIND,
            "thermodynamic_state": slot,
            "temperature_K": float(options.temperatures[slot]),
            "beta_au": float(options.betas[slot]),
            "lambda": float(options.lambdas[slot]),
            "positions": "full_ring_polymer_reorganized_by_thermodynamic_state",
            "position_unit": "bohr",
            "energy_unit": "hartree",
            "time_unit": "atomic_time",
            "estimator_names": list(ESTIMATOR_NAMES),
        }
        atomic_savez(
            path,
            kind=TRAJECTORY_KIND,
            sample_steps=saved["sample_steps"],
            times=saved["sample_steps"] * options.dt,
            walker=np.asarray(walkers, dtype=np.int64),
            positions=positions,
            centroids=positions.mean(axis=1),
            reduced_potentials=np.asarray(reduced, dtype=float).reshape(
                frames, options.nstates
            ),
            estimators=np.asarray(estimators, dtype=float).reshape(
                frames, len(ESTIMATOR_NAMES)
            ),
            cv=np.asarray(cvs, dtype=float).reshape(frames, options.cv_dimension),
            metadata=np.asarray(_json(metadata)),
        )
        files.append(str(path))
    return files


def _analyze(config, system, output, options, *, return_data=False):
    saved = _read_checkpoint(config, system, output, options, history=True)
    attempts, accepts = saved["attempts"], saved["accepts"]
    rates = np.divide(
        accepts,
        attempts,
        out=np.zeros_like(accepts, dtype=float),
        where=attempts > 0,
    )
    trajectories = _write_trajectories(saved, config, system, output, options)
    mbar = _mbar_analysis(saved, options)
    round_trips = round_trip_statistics(
        saved["assignment_history"], saved["exchange_steps"]
    )
    diffusion = state_space_diffusion(
        saved["assignment_history"], saved["exchange_steps"]
    )
    warnings = [
        "Replica-exchange completion and an MBAR solver residual do not establish bead, timestep, sampling or physical convergence.",
        "PIMD sampling clocks are not physical dynamical time.",
        "Round trips, acceptance and thermodynamic-state overlap are mixing diagnostics, not proofs of equilibrium.",
    ]
    if mbar is not None and (not mbar["solver_converged"] or not mbar["connected"]):
        warnings.append(
            "MBAR did not establish one converged connected thermodynamic-state overlap component."
        )
    result = {
        "task": config["task"],
        "method": (
            "parallel_tempering"
            if options.mode == "temperature"
            else "hamiltonian_replica_exchange"
        ),
        "status": "complete" if saved["elapsed"] == options.total else "partial",
        "converged": False,
        "replicas": options.nstates,
        "temperature_K": options.temperatures.tolist(),
        "beta_au": options.betas.tolist(),
        "lambdas": options.lambdas.tolist(),
        "hamiltonian_target": options.target,
        "pairing": options.pairing,
        "exchange_interval_steps": options.exchange_interval,
        "samples_per_state": len(saved["sample_steps"]),
        "acceptance": {
            "pairs": [[index, index + 1] for index in range(options.nstates - 1)],
            "attempts": attempts.tolist(),
            "accepts": accepts.tolist(),
            "rates": rates.tolist(),
        },
        "round_trips": round_trips,
        "state_space_diffusion": diffusion,
        "estimators_by_state_direct": _direct_estimators(saved, options),
        "mbar": mbar,
        "units": {
            "temperature": "K",
            "beta": "hartree^-1",
            "energy": "hartree",
            "position": "bohr",
            "time": "atomic_time",
        },
        "warnings": warnings,
        "files": {
            "checkpoint": str(_checkpoint_path(output)),
            "trajectories": trajectories,
            "analysis": str(Path(output) / "results" / "tempering.json"),
        },
    }
    write_json(Path(output) / "results" / "tempering.json", result)
    if return_data:
        return {"summary": result, "checkpoint": saved}
    return result


def run_tempering(config, system, output, *, return_data=False):
    """Run or resume the validated tempering ensemble."""
    options = resolve_tempering(config, system)
    status = root_call(_prepare_checkpoint, config, system, Path(output), options)
    if status["staged"]:
        status = root_call(
            _recover_staged, config, system, Path(output), options, status["staged"]
        )
    while not status["complete"]:
        stop = _next_boundary(status["elapsed"], options)
        tasks = [
            {
                "config": copy.deepcopy(config),
                "system": system,
                "output": str(output),
                "walker": walker,
                "start": status["elapsed"],
                "stop": stop,
                "generation": status["generation"],
            }
            for walker in range(options.nstates)
        ]
        results = run_tasks(_propagate_replica, tasks, config.get("parallel", {}))
        status = root_call(
            _commit_generation,
            config,
            system,
            Path(output),
            options,
            results,
        )
    return root_call(
        _analyze,
        config,
        system,
        Path(output),
        options,
        return_data=return_data,
    )


def reanalyze_tempering(config, system, output, *, return_data=False):
    """Rebuild state trajectories, diagnostics and MBAR without constructing a PES."""
    options = resolve_tempering(config, system)
    return root_call(
        _analyze,
        config,
        system,
        Path(output),
        options,
        return_data=return_data,
    )


__all__ = [
    "CHECKPOINT_KIND",
    "STAGE_KIND",
    "TRAJECTORY_KIND",
    "ESTIMATOR_NAMES",
    "TemperingOptions",
    "evaluate_thermodynamic_states",
    "resolve_tempering",
    "validate_tempering",
    "run_tempering",
    "reanalyze_tempering",
]
