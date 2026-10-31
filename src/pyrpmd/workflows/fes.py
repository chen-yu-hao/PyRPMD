"""Independent umbrella trajectories and 1D/2D MBAR reconstruction.

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
from ..bias import CompositeBias, FluxMetricBias, UmbrellaBias, displacement
from ..block_propagation import PropagationError
from ..checkpoint import atomic_savez as _atomic_npz
from ..checkpoint import load_checkpoint
from ..cv import RPMDrateCoordinate, make_cvs
from ..dynamics import Dynamics, State
from ..io import fingerprint as _fingerprint
from ..io import report_jsonable as jsonable
from ..io import write_report_json as write_json
from ..parallel import root_call, run_tasks, task_potential
from ..units import DIMS, parse_quantity, unit_factor
from ..validation import checked_integer
from .sampling import fes_sampler


def _dimension(unit):
    return {
        "bohr": DIMS["length"],
        "radian": DIMS["angle"],
        "hartree": DIMS["energy"],
        "dimensionless": DIMS["dimensionless"],
    }[unit]


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
    specs = config.get("cvs", [])
    if isinstance(specs, dict):
        specs = [dict(v, name=k) for k, v in specs.items()]
    umbrella = config.get("umbrella", {})
    names = umbrella.get("cvs")
    if names is not None and isinstance(names, str):
        names = [names]
    cv = make_cvs(specs, system, names)
    selected = (
        specs
        if names is None
        else [next(s for s in specs if s.get("name") == n) for n in names]
    )
    if cv.dimension not in (1, 2):
        raise ValueError("umbrella FES supports one or two CV dimensions")
    _flux_coordinate(config, cv)
    dims = [_dimension(u) for u in cv.units]
    scales = []
    for spec, component in zip(selected, cv.cvs):
        unit = (
            spec.get("unit", spec.get("units", component.units))
            if isinstance(spec, dict)
            else component.units
        )
        if isinstance(unit, str):
            unit = [unit] * component.dimension
        if len(unit) != component.dimension:
            raise ValueError("CV input units dimension mismatch")
        scales.extend(
            [
                1.0 if u == "dimensionless" else unit_factor(u, _dimension(c))
                for u, c in zip(unit, component.units)
            ]
        )

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
    nprod = checked_integer(sampling["production_steps"], "sampling.production_steps")
    interval = checked_integer(
        sampling.get("sample_interval", sampling.get("interval_steps", 1)),
        "sampling.sample_interval",
    )
    andersen = checked_integer(
        sampling.get("andersen_interval", 10), "sampling.andersen_interval"
    )
    if nprod < interval:
        raise ValueError("production_steps must include at least one saved sample")
    checkpoint_steps = checked_integer(
        config.get("output", {}).get("checkpoint_steps", 1000),
        "output.checkpoint_steps",
    )
    if min(interval, andersen, checkpoint_steps) < 1:
        raise ValueError("sampling/checkpoint intervals must be positive")
    checkpoint = (
        Path(task["output"])
        / "checkpoints"
        / f"window_{window:05d}_replica_{replica:04d}.npz"
    )
    fingerprint = _fingerprint(
        config,
        system,
        {
            "window": window,
            "replica": replica,
            "center": center,
            "stiffness": stiffness,
        },
    )
    seed = np.random.SeedSequence([int(config.get("seed", 0)), 101, window, replica])
    with task_potential(config["potential"], system,
                        reuse=config.get("parallel", {}).get("reuse_potentials", True)) as provider:
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
        if checkpoint.exists():
            with load_checkpoint(checkpoint) as saved:
                if str(saved["fingerprint"]) != fingerprint:
                    raise ValueError(
                        f"checkpoint physics differs: {checkpoint}; choose a fresh output directory"
                    )
                state = State(
                    saved["q"], saved["p"], int(saved["step"]), float(saved["time"])
                )
                dynamics.rng.bit_generator.state = json.loads(str(saved["rng"]))
                values = list(saved["values"])
                energies = list(saved["energies"])
                elapsed = int(saved["elapsed"])
                saved_elapsed = elapsed
                if "propagation" in saved:
                    dynamics.propagation_stats = json.loads(str(saved["propagation"]))
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

        # Validate even a completed checkpoint before accepting its samples.
        dynamics._check_fixed(state.q, state.p)

        def save():
            nonlocal saved_elapsed
            if saved_elapsed == elapsed:
                return
            _atomic_npz(
                checkpoint,
                q=state.q,
                p=state.p,
                step=state.step,
                time=state.time,
                rng=json.dumps(dynamics.rng.bit_generator.state),
                values=np.asarray(values).reshape(-1, cv.dimension),
                energies=np.asarray(energies),
                elapsed=elapsed,
                fingerprint=fingerprint,
                propagation=json.dumps(dynamics.propagation_stats),
            )
            saved_elapsed = elapsed

        fused_loop = sampler.thermostat == "andersen" and dynamics.can_fuse(bias=bias, observe=cv)
        while elapsed < neq + nprod:
            initial_elapsed = elapsed
            thermostat_p = thermostat_rng = None
            if sampler.thermostat == "andersen" and state.step % andersen == 0:
                thermostat_p = state.p.copy()
                thermostat_rng = copy.deepcopy(dynamics.rng.bit_generator.state)
                state.p[:] = dynamics.sample_momenta()
            try:
                if fused_loop:
                    # Observations are selected within the block. Only Python
                    # RNG/checkpoint boundaries require returning from native.
                    count = min(dynamics.block_steps, 4096, neq+nprod-elapsed,
                                andersen-state.step % andersen,
                                checkpoint_steps-elapsed % checkpoint_steps)
                    production = elapsed + np.arange(1, count + 1) - neq
                    observe = (production > 0) & (production % interval == 0)
                    block = dynamics.propagate_block(
                        state, count, bias=bias, cv=cv, observe=observe,
                        check_energy=True, raise_on_error=False,
                    )
                    # A later failure must not discard or replay valid steps
                    # and observations that the block has already committed.
                    elapsed += block.completed
                    values.extend(block.values.copy())
                    energies.extend(block.energies.tolist())
                    if block.error is not None:
                        raise PropagationError(block)
                else:
                    # Custom/stateful Python providers retain the old callback
                    # schedule: energy is evaluated only at a sampled step.
                    previous = state.copy()
                    previous_rng = copy.deepcopy(dynamics.rng.bit_generator.state)
                    try:
                        sampler.step(dynamics, state, bias)
                        production = elapsed + 1 - neq
                        if production > 0 and production % interval == 0:
                            value = np.asarray(cv.value(state.q), dtype=float)
                            if value.shape != (cv.dimension,) or not np.isfinite(value).all():
                                raise ValueError("CV returned invalid observation")
                            energy = float(dynamics.energy(state, bias))
                            if not np.isfinite(energy):
                                raise FloatingPointError("nonfinite total ring energy")
                    except Exception:
                        state.q[:] = previous.q; state.p[:] = previous.p
                        state.step, state.time = previous.step, previous.time
                        dynamics.rng.bit_generator.state = previous_rng
                        raise
                    if production > 0 and production % interval == 0:
                        values.append(value.copy())
                        energies.append(energy)
                    elapsed += 1
                    dynamics._record_python_steps(1)
            except Exception:
                if elapsed == initial_elapsed and thermostat_p is not None:
                    # Resume will encounter this same Andersen boundary. Keep
                    # its pre-draw RNG/momenta so the redraw occurs exactly once.
                    state.p[:] = thermostat_p
                    dynamics.rng.bit_generator.state = thermostat_rng
                save()
                raise
            if elapsed % checkpoint_steps == 0:
                save()
        save()
        samples = np.asarray(values).reshape(-1, cv.dimension)
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


def _write_fes(output, data, streams, cv, config):
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
    arrays.update({f"edges_{i}": edge for i, edge in enumerate(data["edges"])})
    metadata = {
        "measure": np.asarray(data["measure"]),
        "energy_unit": np.asarray("hartree"),
        "cv_units": np.asarray(cv.units),
        "beta": np.asarray(config["beta"]),
    }
    if data["measure"] == "rate_flux":
        metadata["flux_axis"] = np.asarray(0, dtype=np.int32)
    _atomic_npz(path / "fes.npz", **arrays, **metadata)
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
    mid = [0.5 * (e[1:] + e[:-1]) for e in data["edges"]]
    mesh = np.meshgrid(*mid, indexing="ij")
    table = np.column_stack(
        [m.ravel() for m in mesh]
        + [
            data[k].ravel()
            for k in (
                "free_energy",
                "standard_error",
                "probability_mass",
                "density",
                "mask",
                "raw_count",
            )
        ]
    )
    np.savetxt(
        path / "fes.csv",
        table,
        delimiter=",",
        header=",".join(
            [f"cv{i}_{unit}" for i, unit in enumerate(cv.units)]
            + [
                "free_energy_hartree",
                "standard_error_hartree",
                "probability_mass",
                "density",
                "masked",
                "raw_count",
            ]
        ),
        comments="",
    )
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
        "standard_error_scope": "complete_bootstrap_support_per_bin_fixed_reference",
        "bootstrap_diagnostics": data.get("bootstrap_diagnostics", []),
        "bootstrap_block_plan": data.get("bootstrap_block_plan", {}),
        "warnings": warnings,
        "converged": False,
        "convergence_note": "Overlap and bootstrap diagnose these samples; bead, timestep and sampling convergence require separate runs.",
        "files": {"data": str(path / "fes.npz"), "table": str(path / "fes.csv")},
    }
    if h5py is not None:
        summary["files"]["hdf5"] = str(path / "fes.h5")
    write_json(path / "summary.json", summary)
    return summary


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
    summary = _write_fes(output, data, streams, cv, config)
    result = {"summary": summary, "data": data, "streams": streams}
    if _postprocess is not None:
        # Adaptive production diagnostics reuse this fit on root, before a
        # summary-only request discards arrays or broadcasts its result.
        result = _postprocess(result)
    return result if return_data else result["summary"]


def run_fes(config, system, output, executor=None, return_data=False, *, _postprocess=None):
    fes_sampler(config)
    if config.get("replica_exchange"):
        from .replica_exchange import exchange_options, run_replica_exchange_fes
        if exchange_options(config, system)["enabled"]:
            return run_replica_exchange_fes(config, system, output, executor=executor,
                                            return_data=return_data)
    if config.get("adaptive_umbrella"):
        from .adaptive_umbrella import adaptive_options, run_adaptive_fes
        if adaptive_options(config)["enabled"]:
            return run_adaptive_fes(config, system, output, executor=executor, return_data=return_data)
    config = copy.deepcopy(config)
    if config.get("analysis", {}).get("measure", "cv_density") not in (
        "cv_density",
        "rate_flux",
    ):
        raise ValueError("analysis.measure must be cv_density or rate_flux")
    cv, specs, centers, stiffness = resolve_umbrellas(config, system)
    replicas = checked_integer(
        config.get("sampling", {}).get("replicas", 1), "sampling.replicas"
    )
    tasks = [
        {
            "config": config,
            "system": system,
            "specs": specs,
            "window": i,
            "replica": r,
            "center": center,
            "stiffness": stiffness,
            "output": str(output),
        }
        for i, center in enumerate(centers)
        for r in range(replicas)
    ]
    streams = (
        executor.map(_sample_window, tasks)
        if executor is not None
        else run_tasks(_sample_window, tasks, config.get("parallel", {}))
    )
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
        cv, specs, centers, stiffness = resolve_umbrellas(config, system)
        replicas = int(config.get("sampling", {}).get("replicas", 1))
        streams = []
        for window, center in enumerate(centers):
            for replica in range(replicas):
                checkpoint = (
                    Path(output)
                    / "checkpoints"
                    / f"window_{window:05d}_replica_{replica:04d}.npz"
                )
                expected = _fingerprint(
                    config,
                    system,
                    {
                        "window": window,
                        "replica": replica,
                        "center": center,
                        "stiffness": stiffness,
                    },
                )
                with load_checkpoint(checkpoint) as saved:
                    if str(saved["fingerprint"]) != expected:
                        raise ValueError(f"checkpoint physics differs: {checkpoint}")
                    streams.append(
                        {
                            "window": window,
                            "replica": replica,
                            "center": center,
                            "stiffness": stiffness,
                            "samples": saved["values"],
                            "energies": saved["energies"],
                            "last_q": saved["q"],
                            "last_p": saved["p"],
                            "checkpoint": str(checkpoint),
                        }
                    )
        return _analyze_streams(streams, config, system, str(output), return_data, _postprocess)

    return root_call(analyze)
