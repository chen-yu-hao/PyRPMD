"""Independent umbrella trajectories and 1D/2D/3D MBAR reconstruction.

The physical umbrella W(s) is inserted as P*W in the sum-beads Hamiltonian.
Each stream owns its random generator and resumable state; scheduling does not
change stream identities. Only the coordinator writes aggregate products.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np

from pyrpmd.propagation import loop_options

from ..analysis.fes import reconstruct_fes
from ..analysis.fes_result import FESResult, write_fes_result
from ..bias import UmbrellaBias, FluxMetricBias, CompositeBias, displacement
from ..cv import make_cvs, RPMDrateCoordinate
from ..dynamics import Dynamics, State
from ..checkpoint import load_checkpoint
from ..parallel import task_potential, root_call
from ..trajectory import TrajectoryStream, _descriptor, _walk_reverse, materialize_stream
from ..trajectory_batch import task_provider, trajectory_batch_size
from ..units import parse_quantity, DIMS
from .sampling import AndersenSchedule, fes_sampler, validate_andersen_schedule
from .biased_sampling import SamplingRuntime, run_biased_sampling, dispatch_tasks
from .common import (
    _atomic_npz, _fingerprint, _dimension, checked_integer, write_json, jsonable,
    fingerprint_matches, resolve_cvs,
)


_SAMPLE_STREAM_FORMAT = "PyRPMD.fes-samples.v1"


def _production_budget(config, production_steps=None):
    """Keep an authorized execution budget separate from the physics input."""
    original = checked_integer(config["sampling"]["production_steps"],
                               "sampling.production_steps")
    if production_steps is None:
        return original
    target = checked_integer(production_steps, "production_steps")
    if target < original:
        raise ValueError("production_steps cannot reduce the original production budget")
    return target


def _window_checkpoint(task):
    return (Path(task["output"]) / "checkpoints"
            / f"window_{task['window']:05d}_replica_{task['replica']:04d}.npz")


def _sample_schema(dimension):
    return {"values": {"dtype": np.dtype("float64").str, "shape": [dimension]},
            "energies": {"dtype": np.dtype("float64").str, "shape": []}}


def _window_fingerprint(task):
    return _fingerprint(task["config"], task["system"], {
        "window": task["window"], "replica": task["replica"],
        "center": np.asarray(task["center"]), "stiffness": np.asarray(task["stiffness"]),
    })


def read_fes_sample_history(saved, *, dimension, expected_samples=None, chunk_directory=None,
                            materialize=True, include_sources=False):
    """Decode samples from an already integrity-checked checkpoint payload.

    This adapter needs no dynamics, CV factory or potential. Archived bounded
    checkpoints may point at their original immutable ``chunk_directory``;
    only the descriptor's committed prefix is followed. Callers authenticating
    physical clocks supply ``expected_samples``. Report builders can request
    exact source files without a second pass over the chunks.
    """
    dimension = checked_integer(dimension, "sample dimension")
    if expected_samples is not None:
        expected_samples = checked_integer(expected_samples, "sample count", 0)
    result = {"trajectory": None, "source_files": []}
    if "sample_stream" not in saved:
        values, energies = saved["values"], saved["energies"]
        count = len(values) if values.ndim else -1
        if expected_samples is not None and count != expected_samples:
            raise ValueError("umbrella sample count differs from its checkpoint clock")
        for name, value, shape in (("values", values, (count, dimension)),
                                    ("energies", energies, (count,))):
            if value.shape != shape or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
                raise ValueError(f"invalid umbrella checkpoint {name}")
        result.update(values=values, energies=energies, samples=count)
        return result
    if "values" in saved or "energies" in saved:
        raise ValueError("umbrella checkpoint has ambiguous sample storage")
    try:
        storage = json.loads(str(saved["sample_stream"].item()))
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError("invalid umbrella sample storage metadata") from error
    if (not isinstance(storage, dict) or set(storage) != {"format", "trajectory"}
            or storage["format"] != _SAMPLE_STREAM_FORMAT):
        raise ValueError("unsupported umbrella sample storage format")
    descriptor = _descriptor(storage["trajectory"])
    count = descriptor["frames"]
    if (descriptor["schema"] != _sample_schema(dimension)
            or (expected_samples is not None and count != expected_samples)):
        raise ValueError("umbrella sample stream shape/count differs from its checkpoint clock")
    if chunk_directory is None:
        raise ValueError("umbrella sample stream requires its original chunk directory")
    if materialize:
        for name, spec in descriptor["schema"].items():
            result[name] = np.empty((count, *spec["shape"]), dtype=spec["dtype"])
    for digest, metadata, arrays in _walk_reverse(
            chunk_directory, descriptor, fields=None if materialize else set()):
        if materialize:
            for name, value in arrays.items():
                result[name][metadata["start"]:metadata["stop"]] = value
        if include_sources:
            result["source_files"].append(str((Path(chunk_directory) / f"{digest}.npz").resolve()))
    result.update(trajectory=descriptor, samples=count)
    return result


def _read_window_checkpoint(task, cv, *, production_steps=None, materialize=False):
    """Read either historical arrays or a committed immutable sample prefix.

    The generic checkpoint reader still validates every member, including old
    unsealed NPZ payloads. Streaming checkpoints additionally validate all
    referenced chunks, not files discovered by listing a directory. No PES is
    constructed, and reading or validating a completed run never rewrites it.
    ``production_steps=None`` permits offline analysis of an extended prefix;
    execution and extension preflight pass their authorized upper bound.
    """
    config, system = task["config"], task["system"]
    checkpoint = _window_checkpoint(task)
    sampling = config["sampling"]
    neq = checked_integer(sampling["equilibration_steps"], "sampling.equilibration_steps", 0)
    interval = checked_integer(sampling.get("sample_interval", sampling.get("interval_steps", 1)),
                               "sampling.sample_interval")
    with load_checkpoint(checkpoint) as saved:
        if not fingerprint_matches(saved["fingerprint"], config, system, {
                "window": task["window"], "replica": task["replica"],
                "center": np.asarray(task["center"]), "stiffness": np.asarray(task["stiffness"]),
        }):
            raise ValueError(f"checkpoint physics differs: {checkpoint}; choose a fresh output directory")
        elapsed, step, time = saved["elapsed"], saved["step"], saved["time"]
        if (elapsed.shape != () or elapsed.dtype.kind not in "iu" or elapsed < 0
                or step.shape != () or step.dtype.kind not in "iu" or step != elapsed):
            raise ValueError(f"invalid umbrella checkpoint clock: {checkpoint}")
        elapsed = int(elapsed)
        if production_steps is not None and elapsed > neq + production_steps:
            raise ValueError(f"umbrella checkpoint exceeds authorized production budget: {checkpoint}")
        dt = sampling.get("dt_au", config["dt_au"])
        # A binary64 clock accumulates one addition per accepted step.  The
        # forward-error bound gamma(n) = n*u/(1-n*u), u=2**-53, also accounts
        # for the reference multiplication with one extra operation.  Preserve
        # the historical short-clock floor, but fail closed if the bound is
        # undefined; step counts and sample counts remain exact checks.
        roundoff = (elapsed + 1) * 2.0**-53
        if (time.shape != () or time.dtype.kind not in "fiu" or not np.isfinite(time)
                or roundoff >= 1
                or not np.isclose(time, elapsed * dt,
                                  rtol=max(1e-10, roundoff / (1 - roundoff)),
                                  atol=abs(dt) * 1e-8)):
            raise ValueError(f"invalid umbrella checkpoint time: {checkpoint}")
        state = State(saved["q"], saved["p"], int(step), float(time))
        if state.q.shape != (config["beads"], len(system.masses), 3):
            raise ValueError(f"invalid umbrella checkpoint state shape: {checkpoint}")
        fixed = np.asarray(getattr(system, "fixed_atoms", ()), dtype=int)
        if len(fixed) and (np.any(state.q[:, fixed] != system.geometry[fixed])
                           or np.any(state.p[:, fixed] != 0)):
            raise ValueError(f"umbrella checkpoint moved fixed atoms: {checkpoint}")
        rng = json.loads(str(saved["rng"]))
        np.random.default_rng(0).bit_generator.state = rng
        propagation = json.loads(str(saved["propagation"])) if "propagation" in saved else None
        if propagation is not None and not isinstance(propagation, dict):
            raise ValueError(f"invalid umbrella propagation metadata: {checkpoint}")
        count = max(0, elapsed - neq) // interval
        operator_state = json.loads(str(saved["operator_state"])) if "operator_state" in saved else None
        if operator_state is not None and not isinstance(operator_state, dict):
            raise ValueError("invalid umbrella thermostat operator state")
        if fes_sampler(config).operator_checkpoint and elapsed:
            if operator_state is None:
                raise ValueError("umbrella checkpoint is missing thermostat operator state")
            if not operator_state.get("sampling"):
                raise ValueError("umbrella checkpoint is missing thermostat auxiliary state")
        result = {"state": state, "rng": rng, "elapsed": elapsed, "propagation": propagation,
                  "operator_state": operator_state}
        collision_clock = AndersenSchedule(
            checked_integer(sampling.get("andersen_interval", 10), "sampling.andersen_interval"),
            validate_andersen_schedule(config, "fes"),
        )
        collision_clock.restore(saved, elapsed)
        result.update(collision_clock.checkpoint_fields())
        result.update(read_fes_sample_history(
            saved, dimension=cv.dimension, expected_samples=count,
            chunk_directory=checkpoint.with_suffix(".chunks"), materialize=materialize,
        ))
    return result


def _flux_coordinate(config, cv):
    """Validate an explicit rate-flux measure before sampling or reuse.

    A two-dimensional flux surface has one distinguished coordinate: its
    first scalar RPMDrate coordinate.  The second CV is a transverse
    observable, not a second independent flux factor.  Ordinary density
    surfaces must never silently acquire this weighting.
    """
    options = config.get("analysis", {})
    axis = options.get("flux_axis")
    if options.get("measure", "cv_density") != "rate_flux":
        if axis is not None:
            raise ValueError("analysis.flux_axis requires measure='rate_flux'")
        return None
    if axis is not None and (
        isinstance(axis, (bool, np.bool_))
        or not isinstance(axis, (int, np.integer))
        or axis != 0
    ):
        raise ValueError("rate_flux currently requires flux_axis=0")
    if cv.dimension == 2 and axis is None:
        raise ValueError("two-dimensional rate_flux requires explicit flux_axis=0")
    if (
        cv.dimension not in (1, 2)
        or len(cv.cvs) != cv.dimension
        or not isinstance(cv.cvs[0], RPMDrateCoordinate)
        or any(c.dimension != 1 for c in cv.cvs)
    ):
        raise ValueError("rate_flux requires the first scalar centroid RPMDrate coordinate")
    return cv.cvs[0]


def resolve_umbrellas(config, system):
    """Resolve user centers/stiffness to canonical CV units and hartree.

    A CV's optional input ``unit``/``units`` applies to numerical centers;
    explicit quantity strings carry their own units. Numerical stiffness is
    always atomic; quantity strings are dimension-checked for mixed CVs.
    """
    umbrella = config.get("umbrella", {})
    cv, selected, scales = resolve_cvs(config, system, "umbrella")
    _flux_coordinate(config, cv)
    dims = [_dimension(unit) for unit in cv.units]

    def center(value, atomic=False):
        if cv.dimension == 1 and (isinstance(value, (str, int, float, np.number))):
            value = [value]
        if len(value) != cv.dimension:
            raise ValueError("umbrella center dimension mismatch")
        return [
            parse_quantity(x, dim)
            * (scale if not atomic and isinstance(x, (int, float, np.number)) else 1)
            for x, dim, scale in zip(value, dims, scales)
        ]

    if "centers_au" in umbrella:
        centers = [center(x, True) for x in umbrella["centers_au"]]
    elif "centers" in umbrella:
        centers = [center(x) for x in umbrella["centers"]]
    elif "grid" in umbrella:
        grid = umbrella["grid"]
        if isinstance(grid, dict):
            grid = [grid]
        if len(grid) != cv.dimension:
            raise ValueError("umbrella.grid needs one axis per CV")
        axes = []
        for d, axis in enumerate(grid):
            if isinstance(axis, dict):
                lo = parse_quantity(axis["start"], dims[d]) * (
                    1 if isinstance(axis["start"], str) else scales[d]
                )
                hi = parse_quantity(axis["stop"], dims[d]) * (
                    1 if isinstance(axis["stop"], str) else scales[d]
                )
                count = checked_integer(axis["count"], "grid count")
                axes.append(np.linspace(lo, hi, count))
            else:
                axes.append(
                    [
                        parse_quantity(x, dims[d])
                        * (1 if isinstance(x, str) else scales[d])
                        for x in axis
                    ]
                )
        centers = (
            np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
            .reshape(-1, cv.dimension)
            .tolist()
        )
    elif all(k in umbrella for k in ("start", "stop", "count")) and cv.dimension == 1:
        lo = center(umbrella["start"])[0]
        hi = center(umbrella["stop"])[0]
        centers = np.linspace(
            lo, hi, checked_integer(umbrella["count"], "umbrella.count")
        )[:, None]
    else:
        raise ValueError("specify umbrella.centers or umbrella.grid")
    centers = np.asarray(centers, float)
    if (
        centers.ndim != 2
        or centers.shape[1] != cv.dimension
        or not len(centers)
        or not np.isfinite(centers).all()
    ):
        raise ValueError("invalid umbrella centers")
    raw = umbrella.get("stiffness_au", umbrella.get("stiffness", umbrella.get("kappa")))
    if raw is None:
        raise ValueError("umbrella.stiffness or kappa is required")
    if isinstance(raw, (str, int, float, np.number)):
        raw = [
            [raw if i == j else 0.0 for j in range(cv.dimension)]
            for i in range(cv.dimension)
        ]
    elif np.asarray(raw, dtype=object).ndim == 1:
        if len(raw) != cv.dimension:
            raise ValueError("diagonal stiffness dimension mismatch")
        raw = [
            [raw[i] if i == j else 0.0 for j in range(cv.dimension)]
            for i in range(cv.dimension)
        ]
    if np.shape(raw) != (cv.dimension, cv.dimension):
        raise ValueError("stiffness must be scalar, diagonal, or square matrix")
    stiffness = np.empty((cv.dimension, cv.dimension))
    for i in range(cv.dimension):
        for j in range(cv.dimension):
            dimension = tuple(
                e - a - b for e, a, b in zip(DIMS["energy"], dims[i], dims[j])
            )
            stiffness[i, j] = parse_quantity(raw[i][j], dimension)
    UmbrellaBias(cv, centers[0], stiffness)  # validates positive definiteness
    return cv, selected, centers, stiffness


def _analysis_ranges(options, cv):
    """Convert every declared range; never silently drop extra CV axes."""
    ranges = options.get("ranges")
    if ranges is None:
        return None
    if not isinstance(ranges, (list, tuple, np.ndarray)) or len(ranges) != cv.dimension:
        raise ValueError("analysis.ranges must contain one [lower, upper] pair per CV")
    converted = []
    for limits, unit in zip(ranges, cv.units):
        if not isinstance(limits, (list, tuple, np.ndarray)) or len(limits) != 2:
            raise ValueError("analysis.ranges must contain one [lower, upper] pair per CV")
        converted.append([parse_quantity(x, _dimension(unit)) for x in limits])
    return converted


def validate_fes_request(config, system):
    """Reject static window/analysis mistakes before allocating a run directory.

    This only constructs CV definitions and checks the declared recording and
    histogram settings. It neither initializes a PES nor evaluates a geometry.
    Sampling-dependent support, overlap and convergence remain analysis checks.
    """
    cv, specs, centers, stiffness = resolve_umbrellas(config, system)
    sampling = config["sampling"]
    production = checked_integer(sampling["production_steps"], "sampling.production_steps")
    interval = checked_integer(
        sampling.get("sample_interval", sampling.get("interval_steps", 1)),
        "sampling.sample_interval",
    )
    if production // interval < 2:
        raise ValueError(
            "sampling.production_steps must include at least two saved samples per umbrella; "
            "increase production_steps or reduce sample_interval"
        )
    options = config.get("analysis", {})
    bootstrap = checked_integer(
        options.get("bootstrap", options.get("bootstrap_replicates", options.get("bootstrap_samples", 20))),
        "analysis.bootstrap", 0,
    )
    if bootstrap == 1:
        raise ValueError("analysis.bootstrap must be zero or at least two")
    checked_integer(options.get("min_count", 1), "analysis.min_count")
    overlap = options.get("overlap_threshold", 1e-6)
    if (isinstance(overlap, (bool, np.bool_)) or not isinstance(overlap, (int, float, np.number))
            or np.iscomplexobj(overlap) or not np.isfinite(overlap) or overlap < 0):
        raise ValueError("analysis.overlap_threshold must be finite and nonnegative")
    bins = options.get("bins", 40)
    if np.isscalar(bins):
        bins = [bins] * cv.dimension
    if not isinstance(bins, (list, tuple, np.ndarray)) or len(bins) != cv.dimension:
        raise ValueError("analysis.bins must contain one bin count or edge array per CV")
    ranges = _analysis_ranges(options, cv)
    for axis, (spec, period) in enumerate(zip(bins, cv.periods)):
        if np.isscalar(spec):
            checked_integer(spec, "analysis.bins", 2)
            bounds = None if ranges is None else ranges[axis]
            if bounds is not None and bounds[1] <= bounds[0]:
                raise ValueError("analysis.ranges must contain increasing limits")
        else:
            raw_edges = np.asarray(spec)
            if raw_edges.dtype.kind not in "iuf":
                raise ValueError("analysis.bins edges must contain at least three increasing finite numbers")
            edges = np.asarray(raw_edges, dtype=float)
            if (edges.ndim != 1 or len(edges) < 3 or not np.isfinite(edges).all()
                    or np.any(np.diff(edges) <= 0)):
                raise ValueError("analysis.bins edges must contain at least three increasing finite numbers")
            bounds = edges[[0, -1]]
        if bounds is not None and period is not None:
            span = bounds[1] - bounds[0]
            if span > period and not np.isclose(span, period, rtol=1e-10, atol=1e-12):
                raise ValueError("a periodic FES axis may span at most one period")
    return cv, specs, centers, stiffness


def steer_to_center(q, cv, target, masses, max_steps=250, tolerance=1e-7, *, fixed_atoms=()):
    """Mass-weighted geometric continuation; no dynamics samples are taken here."""
    q = np.asarray(q, float).copy()
    invmass = 1 / np.asarray(masses)[None, :, None]
    raw_fixed = tuple(fixed_atoms)
    if any(
        isinstance(i, (bool, np.bool_))
        or not isinstance(i, (int, np.integer))
        or not 0 <= i < q.shape[1]
        for i in raw_fixed
    ) or len(set(raw_fixed)) != len(raw_fixed):
        raise ValueError("fixed_atoms must contain distinct in-range integer atom indices")
    invmass[:, list(raw_fixed), :] = 0.0
    target = np.asarray(target, float)
    for _ in range(max_steps):
        result = cv.value_and_jacobian(q)
        error = displacement(result.value, target, cv.periods)
        norm = np.linalg.norm(error)
        if norm < tolerance:
            return q
        jac = result.jacobian
        metric = np.einsum(
            "ipac,jpac,pac->ij", jac, jac, np.broadcast_to(invmass, q.shape)
        )
        multiplier = np.linalg.lstsq(metric, error, rcond=1e-12)[0]
        direction = -np.einsum("i,ipac->pac", multiplier, jac) * invmass
        if not np.any(direction):
            raise ValueError("umbrella center cannot be reached with the fixed atoms held in place")
        maxmove = np.max(np.linalg.norm(direction, axis=-1))
        if maxmove > 0.25:
            direction *= 0.25 / maxmove
        moved = False
        for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
            trial = q + scale * direction
            try:
                newnorm = np.linalg.norm(
                    displacement(cv.value(trial), target, cv.periods)
                )
            except ValueError:
                continue
            if newnorm < norm:
                q = trial
                moved = True
                break
        if not moved:
            break
    residual = float(np.linalg.norm(displacement(cv.value(q), target, cv.periods)))
    if residual > 1e-4:
        raise ValueError(
            f"could not construct umbrella initial geometry; CV residual {residual:g}"
        )
    return q


def separate_reactants_to_xi(geometry, coordinate, target, masses):
    """Construct entrance umbrellas by rigid fragment separation, not bond collapse.

    A least-norm xi projection can lower xi by collapsing the breaking bond
    instead of separating reactants. Here the internal TS geometry of each
    reactant is preserved exactly, then ordinary PES thermalization relaxes
    its internal degrees of freedom. This is initialization, not sampling.
    """
    from scipy.optimize import brentq

    geometry = np.asarray(geometry, float)
    fragments = [np.asarray(f, dtype=int) for f in coordinate.com.fragments]
    masses = np.asarray(masses, float)
    totals = np.array([masses[f].sum() for f in fragments])
    com = [np.average(geometry[f], axis=0, weights=masses[f]) for f in fragments]
    vector = com[0] - com[1]
    distance = float(np.linalg.norm(vector))
    if distance < 1e-12:
        raise ValueError(
            "cannot initialize rate entrance from coincident fragment centers"
        )
    axis = vector / distance

    def translated(delta):
        q = geometry.copy()
        q[fragments[0]] += delta * totals[1] / totals.sum() * axis
        q[fragments[1]] -= delta * totals[0] / totals.sum() * axis
        return q

    def residual(delta):
        return coordinate.centroid_value_gradient(translated(delta))[0] - target

    start = residual(0.0)
    if abs(start) < 1e-10:
        return geometry.copy()
    if start < 0:
        raise ValueError(
            "rigid entrance initialization is only for xi below the initial TS"
        )
    upper = max(1.0, coordinate.rinf - distance + 1.0)
    for _ in range(20):
        if residual(upper) <= 0:
            break
        upper *= 2
    else:
        raise ValueError(
            "could not bracket entrance umbrella by rigid fragment separation"
        )
    shift = brentq(residual, 0.0, upper, xtol=1e-12, rtol=1e-12)
    return translated(shift)


def _sample_window(task):
    config, system = task["config"], task["system"]
    cv = make_cvs(task["specs"], system)
    center = np.asarray(task["center"])
    stiffness = np.asarray(task["stiffness"])
    window, replica = task["window"], task["replica"]
    sampling = config["sampling"]
    sampler = fes_sampler(config)
    neq = checked_integer(
        sampling["equilibration_steps"], "sampling.equilibration_steps", 0
    )
    nprod = _production_budget(config, task.get("production_steps"))
    interval = checked_integer(
        sampling.get("sample_interval", sampling.get("interval_steps", 1)),
        "sampling.sample_interval",
    )
    andersen = checked_integer(
        sampling.get("andersen_interval", 10), "sampling.andersen_interval"
    )
    collision_clock = AndersenSchedule(andersen, validate_andersen_schedule(config, "fes"))
    if nprod < interval:
        raise ValueError("production_steps must include at least one saved sample")
    checkpoint_steps = checked_integer(
        config.get("output", {}).get("checkpoint_steps", 1000),
        "output.checkpoint_steps",
    )
    if min(interval, andersen, checkpoint_steps) < 1:
        raise ValueError("sampling/checkpoint intervals must be positive")
    checkpoint = _window_checkpoint(task)
    fingerprint = _window_fingerprint(task)
    saved = (_read_window_checkpoint(task, cv, production_steps=nprod)
             if checkpoint.exists() else None)
    seed = np.random.SeedSequence([int(config.get("seed", 0)), 101, window, replica])
    with task_provider(task, factory=task_potential) as provider:
        dynamics = Dynamics(
            provider,
            system.masses,
            config["beta"],
            config["beads"],
            sampling.get("dt_au", config["dt_au"]),
            seed=seed,
            native=config.get("native", config.get("dynamics", {}).get("native", True)),
            threads=config.get("parallel", {}).get("threads_per_rank", 1),
            fixed_atoms=getattr(system, "fixed_atoms", ()),
            fixed_positions=system.geometry,
            **loop_options(config),
        )
        umbrella = UmbrellaBias(cv, center, stiffness)
        bias = umbrella
        flux_coordinate = _flux_coordinate(config, cv)
        if flux_coordinate is not None:
            bias = CompositeBias(
                umbrella, FluxMetricBias(flux_coordinate, system.masses, config["beta"])
            )
        values = []
        energies = []
        elapsed = 0
        saved_elapsed = None
        descriptor = None
        store_operators = sampler.operator_checkpoint
        if saved is not None:
            state = saved["state"]
            dynamics.rng.bit_generator.state = saved["rng"]
            collision_clock.restore(saved, state.step)
            descriptor = saved["trajectory"]
            if descriptor is None:
                values = list(saved["values"])
                energies = list(saved["energies"])
            elapsed = saved["elapsed"]
            saved_elapsed = elapsed
            if saved["propagation"] is not None:
                dynamics.propagation_stats = saved["propagation"]
            if saved["operator_state"] is not None:
                dynamics.load_operator_state_dict(saved["operator_state"])
                store_operators = True
        else:
            geometry = system.geometry
            # Rigid fragment translations and equivalent-path permutations
            # assume every atom is mobile. With any fixed atoms, retain the
            # reference geometry and use the fixed-aware steering below.
            if isinstance(cv.cvs[0], RPMDrateCoordinate) and not dynamics.fixed_atoms:
                coordinate = cv.cvs[0]
                permutations = coordinate.permutations
                permutation = permutations[replica % len(permutations)]
                geometry = np.empty_like(system.geometry)
                geometry[list(permutation)] = system.geometry
                if center[0] < coordinate.centroid_value_gradient(geometry)[0] - 1e-10:
                    geometry = separate_reactants_to_xi(
                        geometry, coordinate, center[0], system.masses
                    )
            q = np.repeat(geometry[None], config["beads"], axis=0)
            q = steer_to_center(q, cv, center, system.masses, fixed_atoms=dynamics.fixed_atoms)
            state = State(q, dynamics.sample_momenta())
        del saved

        # Validate even a completed checkpoint before accepting its samples.
        dynamics._check_fixed(state.q, state.p)
        writer = None
        if elapsed < neq + nprod and (task.get("production_steps") is not None or descriptor is not None):
            # The reader has already validated an existing committed prefix.
            # A legacy history is migrated at the first successful added-state
            # checkpoint, never merely by opening a completed run.
            writer = TrajectoryStream(checkpoint.with_suffix(".chunks"),
                                      schema=_sample_schema(cv.dimension),
                                      descriptor=descriptor, validate=False)

        runtime = SamplingRuntime(state, elapsed, values=values, energies=energies,
                                  collision_clock=collision_clock,
                                  operator_checkpoint=store_operators)

        def save(current):
            nonlocal saved_elapsed, descriptor
            state, elapsed = current.state, current.elapsed
            if saved_elapsed == elapsed:
                return
            payload = dict(
                q=state.q,
                p=state.p,
                step=state.step,
                time=state.time,
                rng=json.dumps(dynamics.rng.bit_generator.state),
                elapsed=elapsed,
                fingerprint=fingerprint,
                propagation=json.dumps(dynamics.propagation_stats),
                **({"operator_state": json.dumps(dynamics.operator_state_dict(), allow_nan=False)}
                   if store_operators else {}),
                **collision_clock.checkpoint_fields(),
            )
            arrays = {"values": np.asarray(values).reshape(-1, cv.dimension),
                      "energies": np.asarray(energies)}
            if writer is not None:
                # Chunk failures abort this invocation. Never retry save with
                # an advanced state and only a partially appended observation
                # batch. The disk checkpoint still selects a complete prefix;
                # any unreferenced chunks are ignored when resuming.
                writer.extend(arrays)
                candidate = writer.prepare_commit()
                if candidate["frames"] != max(0, elapsed - neq) // interval:
                    raise ValueError("umbrella sample count differs from its checkpoint clock")
                payload["sample_stream"] = json.dumps({
                    "format": _SAMPLE_STREAM_FORMAT, "trajectory": candidate,
                })
            else:
                payload.update(arrays)
            _atomic_npz(checkpoint, **payload)
            if writer is not None:
                descriptor = candidate
                values.clear()
                energies.clear()
            saved_elapsed = elapsed

        run_biased_sampling(
            runtime, dynamics, cv, bias, neq=neq, nprod=nprod, interval=interval,
            checkpoint_steps=checkpoint_steps, sampling=sampling, sampler=sampler,
            save=save, initial_save=False,
        )
        state, elapsed = runtime.state, runtime.elapsed
        if descriptor is not None:
            # Propagation retains only observations since the previous commit;
            # reconstruction materializes the complete immutable history once.
            arrays = materialize_stream(checkpoint.with_suffix(".chunks"), descriptor)
            samples, energies = arrays["values"], arrays["energies"]
        else:
            samples = np.asarray(runtime.values).reshape(-1, cv.dimension)
        if len(samples) < 2:
            raise ValueError("at least two samples per umbrella stream required")
        return {
            "window": window,
            "replica": replica,
            "samples": samples,
            "center": center,
            "stiffness": stiffness,
            "last_q": state.q,
            "last_p": state.p,
            "energies": np.asarray(energies),
            "checkpoint": str(checkpoint),
            "native": dynamics.native,
            "propagation": dict(dynamics.propagation_stats),
        }


def _write_fes(output, data, streams, cv, config, _postprocess=None):
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
    arrays.update({f"edges_{i}": edge for i, edge in enumerate(data["edges"])})
    try:
        import h5py
    except ImportError:
        h5py = None
    if h5py is not None:
        tmp = path / "fes.h5.tmp"
        with h5py.File(tmp, "w") as h:
            h.attrs["schema_version"] = 1
            h.attrs["energy_unit"] = "hartree"
            h.attrs["cv_units"] = json.dumps(cv.units)
            h.attrs["measure"] = data["measure"]
            h.attrs["beta"] = config["beta"]
            if data["measure"] == "rate_flux":
                h.attrs["flux_axis"] = 0
            h.attrs["hamiltonian_convention"] = "sum_beads_beta_over_P"
            for name, value in arrays.items():
                h.create_dataset(name, data=value)
            groups = h.create_group("streams")
            for i, stream in enumerate(streams):
                g = groups.create_group(str(i))
                for key in ("samples", "center", "stiffness", "energies"):
                    g.create_dataset(key, data=stream[key])
                g.attrs["window"] = stream["window"]
                g.attrs["replica"] = stream["replica"]
                g.attrs["propagation"] = json.dumps(stream.get("propagation", {}))
                if "exchange_group" in stream:
                    g.attrs["exchange_group"] = stream["exchange_group"]
                    g.create_dataset("walker_ids", data=stream["walker_ids"])
        os.replace(tmp, path / "fes.h5")
    warnings = list(config.get("_warnings", []))
    if data.get("bootstrap_requested", 0):
        warnings.extend(data.get("bootstrap_block_plan", {}).get("warnings", []))
    if not data["connected"]:
        warnings.append(
            "Disconnected umbrella overlap: global FES is unidentifiable and masked."
        )
    if np.any(data["effective_samples_per_stream"] < 20):
        warnings.append(
            "Some streams have fewer than 20 estimated independent samples; increase sampling and replicas."
        )
    if np.any(data["mask"]):
        warnings.append(
            "Unsampled/insufficient-count bins are masked; no extrapolation is reported."
        )
    if data.get("bootstrap_attempted", 0) > data["bootstrap_replicates"]:
        warnings.append(
            "Some bootstrap draws failed; inspect bootstrap_diagnostics and per-bin valid counts."
        )
    if (data.get("bootstrap_requested", 0) and data["connected"]
            and np.any((~data["mask"]) & (data["bootstrap_valid_per_bin"] < 2))):
        warnings.append(
            "Some supported bins have fewer than two valid bootstrap draws; their errors are unavailable."
        )
    if (data.get("bootstrap_requested", 0) and data["connected"]
            and np.any((~data["mask"]) & (data["bootstrap_valid_per_bin"] < data["bootstrap_requested"]))):
        warnings.append(
            "Some supported bins lack complete bootstrap support; their main standard errors are unavailable. "
            "supported_draw_standard_error describes surviving draws only and is a conditional diagnostic."
        )
    summary = {
        "task": "fes",
        "status": "sampled" if data["connected"] else "disconnected_overlap",
        "dimension": cv.dimension,
        "units": cv.units,
        "energy_unit": "hartree",
        "measure": data["measure"],
        "beta": config["beta"],
        "flux_axis": 0 if data["measure"] == "rate_flux" else None,
        "estimator": "mbar",
        "windows": len(set(s["window"] for s in streams)),
        "streams": len(streams),
        "samples": sum(len(s["samples"]) for s in streams),
        "overlap_connected": data["connected"],
        "overlap_components": data["components"],
        "weight_effective_samples": data["weight_ess"],
        "effective_samples_per_stream": data["effective_samples_per_stream"],
        "bootstrap_replicates": data["bootstrap_replicates"],
        "bootstrap_requested": data.get("bootstrap_requested", 0),
        "bootstrap_attempted": data.get("bootstrap_attempted", 0),
        "bootstrap_reference_bin": data.get("bootstrap_reference_bin"),
        "standard_error_scope": data.get(
            "standard_error_scope", "complete_bootstrap_support_per_bin_fixed_reference"),
        "supported_draw_condition": data.get("supported_draw_condition"),
        "bootstrap_diagnostics": data.get("bootstrap_diagnostics", []),
        "bootstrap_block_plan": data.get("bootstrap_block_plan", {}),
        "warnings": warnings,
        "converged": False,
        "convergence_note": "Overlap and bootstrap diagnose these samples; bead, timestep and sampling convergence require separate runs.",
        "files": {"data": str(path / "fes.npz"), "table": str(path / "fes.csv")},
    }
    if h5py is not None:
        summary["files"]["hdf5"] = str(path / "fes.h5")
    if _postprocess is not None:
        # Adaptive diagnostics need the fit and streams only on root. Add
        # their provenance before serializing the authoritative FESResult so
        # fes.npz, fes.json and summary.json describe the same production.
        summary = _postprocess({"summary": summary, "data": data, "streams": streams})["summary"]
    result = FESResult.from_data(data, method="umbrella", cv_units=cv.units,
                                 metadata=summary)
    return write_fes_result(path, result, summary=summary)


def _analyze_streams(streams, config, system, output, return_data, _postprocess=None):
    # External executors return a one-shot map iterator. Analysis and exports
    # traverse these streams repeatedly, so retain every window once.
    streams = list(streams)
    if not streams:
        raise ValueError("FES analysis requires at least one sampling stream")
    cv, specs, centers, stiffness = resolve_umbrellas(config, system)
    options = config.get("analysis", {})
    if options.get("method", "mbar") not in ("mbar", "umbrella_integration"):
        raise ValueError(
            "supported FES estimator: mbar (umbrella_integration is used by rate)"
        )
    if (
        options.get("method") == "umbrella_integration"
        and config.get("task", "fes") != "rate"
    ):
        raise ValueError(
            "umbrella_integration is the 1D rate PMF estimator; generic FES uses mbar"
        )
    ranges = _analysis_ranges(options, cv)
    data = reconstruct_fes(
        [s["samples"] for s in streams],
        [s["center"] for s in streams],
        [s["stiffness"] for s in streams],
        config["beta"],
        bins=options.get("bins", 40),
        ranges=ranges,
        periods=cv.periods,
        bootstrap=checked_integer(
            options.get(
                "bootstrap",
                options.get(
                    "bootstrap_replicates", options.get("bootstrap_samples", 20)
                ),
            ),
            "analysis.bootstrap",
            0,
        ),
        seed=checked_integer(
            options.get("seed", config.get("seed", 0)), "analysis.seed", 0
        ),
        min_count=checked_integer(options.get("min_count", 1), "analysis.min_count"),
        overlap_threshold=float(options.get("overlap_threshold", 1e-6)),
        return_reweighting=return_data and config.get("task") == "surface_rate",
        bootstrap_groups=([s["exchange_group"] for s in streams]
                          if all("exchange_group" in s for s in streams) else None),
    )
    data["measure"] = options.get("measure", "cv_density")
    summary = _write_fes(output, data, streams, cv, config, _postprocess)
    result = {"summary": summary, "data": data, "streams": streams}
    return result if return_data else result["summary"]


def _make_tasks(config, system, output, *, production_steps=None):
    """Build independent windows without changing their original identity."""
    if config.get("analysis", {}).get("measure", "cv_density") not in (
        "cv_density", "rate_flux",
    ):
        raise ValueError("analysis.measure must be cv_density or rate_flux")
    if production_steps is not None:
        production_steps = _production_budget(config, production_steps)
    cv, specs, centers, stiffness = resolve_umbrellas(config, system)
    replicas = checked_integer(
        config.get("sampling", {}).get("replicas", 1), "sampling.replicas"
    )
    tasks = [
        {"config": config, "system": system, "specs": specs, "window": i,
         "replica": r, "center": center, "stiffness": stiffness, "output": str(output)}
        for i, center in enumerate(centers)
        for r in range(replicas)
    ]
    if production_steps is not None:
        for task in tasks:
            task["production_steps"] = production_steps
    return cv, tasks


def validate_fes_checkpoints(config, system, checkpoint_output, *, production_steps=None):
    """Authenticate every expected independent stream without running a PES.

    The extension coordinator supplies the latest authorized total production
    budget, checks its own completed-generation lower bound, and acquires the
    run lock before calling this read-only preflight. Missing, corrupt, foreign
    or over-budget streams fail before that coordinator authorizes new work.
    """
    from . import validate_bond_constraint_workflow
    validate_bond_constraint_workflow(system, "fes")
    validate_andersen_schedule(config, "fes")
    fes_sampler(config)
    if config.get("replica_exchange"):
        from .replica_exchange import exchange_options
        if exchange_options(config, system)["enabled"]:
            raise ValueError("production extension requires independent fixed windows")
    if config.get("adaptive_umbrella"):
        from .adaptive_umbrella import adaptive_options
        if adaptive_options(config)["enabled"]:
            raise ValueError("production extension requires independent fixed windows")
    target = _production_budget(config, production_steps)
    cv, tasks = _make_tasks(config, system, checkpoint_output, production_steps=target)
    streams = []
    for task in tasks:
        checkpoint = _window_checkpoint(task)
        if not checkpoint.is_file():
            raise ValueError(f"missing umbrella checkpoint: {checkpoint}")
        saved = _read_window_checkpoint(task, cv, production_steps=target)
        streams.append({"window": task["window"], "replica": task["replica"],
                        "elapsed": saved["elapsed"], "samples": saved["samples"],
                        "checkpoint": str(checkpoint.resolve())})
    return streams


def run_fes(config, system, output, executor=None, return_data=False, *, _postprocess=None,
            production_steps=None, checkpoint_output=None):
    """Sample and reconstruct, optionally using an authorized extended budget.

    ``production_steps`` is an internal total budget supplied by the extension
    coordinator, not a replacement for the original scientific configuration.
    ``checkpoint_output`` selects existing windows while aggregate products
    are written under ``output``. Neither option applies to adaptive or
    replica-exchange runs.
    """
    from . import validate_bond_constraint_workflow
    validate_bond_constraint_workflow(system, "fes")
    validate_andersen_schedule(config, "fes")
    fes_sampler(config)
    batch_size = trajectory_batch_size(config)
    if batch_size > 1 and (config.get("replica_exchange") or config.get("adaptive_umbrella")):
        raise ValueError("trajectory batching currently requires independent fixed umbrella windows")
    if config.get("replica_exchange"):
        from .replica_exchange import exchange_options, run_replica_exchange_fes
        if exchange_options(config, system)["enabled"]:
            if production_steps is not None or checkpoint_output is not None:
                raise ValueError("production extension requires independent fixed windows")
            return run_replica_exchange_fes(config, system, output, executor=executor,
                                            return_data=return_data)
    if config.get("adaptive_umbrella"):
        from .adaptive_umbrella import adaptive_options, run_adaptive_fes
        if adaptive_options(config)["enabled"]:
            if production_steps is not None or checkpoint_output is not None:
                raise ValueError("production extension requires independent fixed windows")
            return run_adaptive_fes(config, system, output, executor=executor, return_data=return_data)
    config = copy.deepcopy(config)
    _, tasks = _make_tasks(config, system, output if checkpoint_output is None else checkpoint_output,
                          production_steps=production_steps)
    streams = dispatch_tasks(_sample_window, tasks, config, executor, batch_trajectories=True)
    return root_call(
        _analyze_streams, streams, config, system, str(output), return_data, _postprocess
    )


def reanalyze_fes(config, system, output, return_data=False, *, _postprocess=None):
    """Reconstruct with new bins/bootstrap settings without propagating PESs.

    Completed window checkpoint arrays are the authoritative sampled data;
    changing the physical settings is rejected just as for a resumed run.
    All MPI ranks may call this; only the coordinator reads/writes files.
    """

    if config.get("replica_exchange"):
        from .replica_exchange import exchange_options, reanalyze_replica_exchange_fes
        if exchange_options(config, system)["enabled"]:
            return reanalyze_replica_exchange_fes(config, system, output, return_data=return_data)
    if config.get("adaptive_umbrella"):
        from .adaptive_umbrella import adaptive_options, reanalyze_adaptive_fes
        if adaptive_options(config)["enabled"]:
            return reanalyze_adaptive_fes(config, system, output, return_data=return_data)

    def analyze():
        cv, tasks = _make_tasks(config, system, output)
        streams = []
        for task in tasks:
            saved = _read_window_checkpoint(task, cv, materialize=True)
            streams.append({
                "window": task["window"], "replica": task["replica"],
                "center": task["center"], "stiffness": task["stiffness"],
                "samples": saved["values"], "energies": saved["energies"],
                "last_q": saved["state"].q, "last_p": saved["state"].p,
                "checkpoint": str(_window_checkpoint(task)),
            })
        return _analyze_streams(streams, config, system, str(output), return_data, _postprocess)

    return root_call(analyze)
