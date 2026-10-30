"""Basin-normalized, fixed-cell RPMD site escape rates for linear centroid CVs.

The umbrella measure is ordinary CV density. For a constant mass metric z,
the constrained configuration measure differs from delta(s-s*) by a constant,
which cancels in the paired-child transmission coefficient. No gas-phase
volume, electronic weight, or distance Jacobian is part of this workflow.

Positions and all CV values remain continuous and unwrapped. A basin names
one explicit local interval, even when the physical potential is periodic.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from ..analysis.fes import (
    _block_resample,
    _histogram,
    solve_mbar,
    statistical_inefficiency,
    umbrella_reduced_potentials,
)
from ..checkpoint import atomic_savez, load_checkpoint
from ..constants import ATOMIC_TIME_S
from ..cv import Cartesian
from ..cv_periodic import FractionalCoordinate, LinearProjection
from ..dynamics import Dynamics, State
from ..parallel import make_task_potential, root_call, run_tasks
from ..propagation import loop_options
from ..units import parse_quantity, unit_factor
from .fes import (
    _dimension,
    _fingerprint,
    checked_integer,
    jsonable,
    resolve_umbrellas,
    run_fes,
    write_json,
)
from .rate import (
    _recrossing_options,
    constrained_step,
    project_momenta,
    project_positions,
)


class SiteSurface:
    """Shift a supported linear centroid CV to a zero-level constraint."""

    def __init__(self, cv, value):
        self.cv = cv
        self.value = float(value)

    def centroid_value_gradient(self, q):
        value, gradient = self.cv.centroid_value_gradient(q)
        return value - self.value, gradient


def _setup(config, system):
    if system.fixed_atoms:
        raise ValueError("site_rate requires all atoms mobile; use surface_rate for fixed layers")
    table = config.get("site_rate", {})
    if not isinstance(table, dict):
        raise ValueError("site_rate must be a table")
    if any(key in config for key in ("reaction", "rate", "collision", "outcomes")):
        raise ValueError("site_rate uses basin normalization, without gas-phase reaction/rate settings")
    if config.get("adaptive_umbrella") or config.get("metadynamics"):
        raise ValueError("site_rate currently requires fixed local umbrella windows")
    analysis = config.get("analysis", {})
    if analysis.get("measure", "cv_density") != "cv_density":
        raise ValueError("site_rate requires analysis.measure='cv_density'")
    if analysis.get("method", "mbar") != "mbar":
        raise ValueError("site_rate currently requires analysis.method='mbar'")
    if config.get("dynamics", {}).get("loop_backend") == "native":
        raise ValueError("site_rate uses checkpointed Python loops; use loop_backend='auto' or 'python'")
    if config.get("sampling", {}).get("thermostat", "andersen") != "andersen":
        raise ValueError("site_rate constrained parents currently use the Andersen thermostat")
    if "dividing_surface" in config.get("recrossing", {}):
        raise ValueError("Declare the site dividing surface only in site_rate.dividing_surface")
    stack, specs, centers, stiffness = resolve_umbrellas(config, system)
    if stack.dimension != 1 or len(stack.cvs) != 1:
        raise ValueError("site_rate requires exactly one umbrella CV dimension")
    cv = stack.cvs[0]
    if type(cv) not in (Cartesian, LinearProjection, FractionalCoordinate):
        raise ValueError("site_rate supports only Cartesian, projection or fractional constant-gradient CVs")
    if cv.estimator != "centroid" or cv.periods != (None,):
        raise ValueError("site_rate requires a continuous centroid CV and nonperiodic local umbrella bias")
    declared = config.get("cvs", [])
    if "cv" not in table and len(declared) != 1:
        raise ValueError("site_rate.cv is required when more than one CV is declared")
    name = table.get("cv", cv.name)
    if name != cv.name:
        raise ValueError("site_rate.cv must match the single selected umbrella CV")
    if "basin" not in table or "dividing_surface" not in table:
        raise ValueError("site_rate requires an explicit basin and dividing_surface")
    raw_basin = table["basin"]
    if not isinstance(raw_basin, (list, tuple, np.ndarray)) or len(raw_basin) != 2:
        raise ValueError("site_rate.basin must contain [lower, upper]")
    unit = cv.units[0]
    spec = specs[0]
    for declaration in (spec, config.get("umbrella", {})):
        if isinstance(declaration, dict) and any(key in declaration for key in ("period", "periods", "periodic", "wrap")):
            raise ValueError("site_rate does not accept periodic or wrapped CV/bias declarations")
    input_unit = spec.get("unit", spec.get("units", unit)) if isinstance(spec, dict) else unit
    if not isinstance(input_unit, str):
        if len(input_unit) != 1:
            raise ValueError("site_rate CV input units must have one entry")
        input_unit = input_unit[0]
    scale = 1.0 if input_unit == "dimensionless" else unit_factor(input_unit, _dimension(unit))

    def quantity(value):
        result = parse_quantity(value, _dimension(unit))
        return result if isinstance(value, (str, list, tuple)) else result * scale

    basin = [quantity(value) for value in raw_basin]
    surface = quantity(table["dividing_surface"])
    direction = table.get("direction", 1)
    if isinstance(direction, (bool, np.bool_)) or not isinstance(direction, (int, np.integer)) or direction not in (-1, 1):
        raise ValueError("site_rate.direction must be the integer +1 or -1")
    if not basin[1] > basin[0]:
        raise ValueError("site_rate.basin requires lower < upper")
    endpoint = basin[1] if direction == 1 else basin[0]
    if not np.isclose(surface, endpoint, rtol=1e-12, atol=1e-12 * (basin[1] - basin[0])):
        raise ValueError("site dividing surface must be the upper basin boundary for +1 or lower for -1")
    surface = endpoint
    _, gradient = cv.centroid_value_gradient(system.geometry)
    metric = float(np.sum(gradient * gradient / system.masses[:, None]))
    if not np.isfinite(metric) or metric <= 1e-28:
        raise ValueError("site CV has a singular mass metric")
    beta = float(config["beta"])
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("site_rate requires positive finite beta")
    options = _recrossing_options(config)
    if not np.isfinite(options["plateau_tolerance"]):
        raise ValueError("recrossing.plateau_tolerance must be finite")
    bootstrap = _pmf_bootstrap_count(config)
    if bootstrap == 1:
        raise ValueError("analysis.bootstrap must be zero or at least two")
    checked_integer(analysis.get("seed", config.get("seed", 0)), "analysis.seed", 0)
    checked_integer(analysis.get("min_count", 1), "analysis.min_count")
    threshold = float(analysis.get("overlap_threshold", 1e-6))
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("analysis.overlap_threshold must lie strictly between zero and one")
    bins = analysis.get("bins", 40)
    if np.isscalar(bins):
        bins = [bins]
    if len(bins) != 1:
        raise ValueError("site_rate analysis.bins requires one histogram axis")
    if np.isscalar(bins[0]):
        checked_integer(bins[0], "analysis.bins", 2)
    else:
        edges = np.asarray(bins[0], dtype=float)
        if edges.ndim != 1 or len(edges) < 3 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
            raise ValueError("analysis.bins requires finite increasing edges")
    if "ranges" in analysis:
        ranges = analysis["ranges"]
        if len(ranges) != 1 or len(ranges[0]) != 2:
            raise ValueError("site_rate analysis.ranges requires [[lower, upper]]")
        limits = [parse_quantity(value, _dimension(unit)) for value in ranges[0]]
        if limits[1] <= limits[0]:
            raise ValueError("analysis.ranges must have lower < upper")
    sampling = config["sampling"]
    checked_integer(sampling["equilibration_steps"], "sampling.equilibration_steps", 0)
    production = checked_integer(sampling["production_steps"], "sampling.production_steps")
    interval = checked_integer(sampling.get("sample_interval", sampling.get("interval_steps", 1)), "sampling.sample_interval")
    if production < interval:
        raise ValueError("production_steps must include at least one saved sample")
    checked_integer(sampling.get("replicas", 1), "sampling.replicas")
    checked_integer(sampling.get("andersen_interval", 10), "sampling.andersen_interval")
    checked_integer(config.get("output", {}).get("checkpoint_steps", 1000), "output.checkpoint_steps")
    metadata = {
        "ready": True,
        "cv": name,
        "cv_type": type(cv).__name__,
        "unit": unit,
        "estimator": "centroid",
        "measure": "cv_density",
        "basin": basin,
        "dividing_surface": surface,
        "direction": int(direction),
        "mass_metric": metric,
        "flux_prefactor_au": float(np.sqrt(metric / (2 * np.pi * beta))),
        "normalization": "density(s*) / integral_basin density(s) ds",
        "coordinate_images": "continuous unwrapped local interval",
        "constraint_measure": "constant mass metric; no position-dependent Fixman correction",
        "rate_unit": "s^-1",
        "umbrella_windows": len(centers),
    }
    return cv, metadata, options


def validate_site_rate(config, system):
    """Validate without propagating a PES; return JSON preparation metadata."""
    return jsonable(_setup(config, system)[1])


def _pmf_bootstrap_count(config):
    options = config.get("analysis", {})
    value = options.get("bootstrap", options.get("bootstrap_replicates", options.get("bootstrap_samples", 20)))
    return checked_integer(value, "analysis.bootstrap", 0)


def estimate_site_qtst(data, beta, mass_metric, basin, dividing_surface):
    """Estimate a local QTST rate from a supported, connected 1D histogram.

    Basin probability integrates the piecewise-constant *bin density* using
    exact overlap lengths, including fractional end bins. Surface density is
    the linear interpolation of adjacent bin-average densities at bin centers.
    This is a finite-bin estimator: it does not evaluate a mathematical delta
    function exactly, extrapolate tails, or fill a missing histogram bin.
    Both density and basin mass use the same MBAR normalization.
    """
    beta, metric = float(beta), float(mass_metric)
    if not np.isfinite(beta) or beta <= 0 or not np.isfinite(metric) or metric <= 0:
        raise ValueError("QTST requires positive finite beta and mass metric")
    limits = np.asarray(basin, dtype=float)
    surface = float(dividing_surface)
    if limits.shape != (2,) or not np.isfinite(limits).all() or limits[1] <= limits[0] or not np.isfinite(surface):
        raise ValueError("QTST requires a finite nonempty basin and surface")
    if data.get("measure", "cv_density") != "cv_density":
        raise ValueError("site QTST requires the ordinary CV-density measure")
    if len(data["edges"]) != 1:
        raise ValueError("site QTST requires a one-dimensional histogram")
    edge = np.asarray(data["edges"][0], dtype=float)
    density = np.asarray(data["density"], dtype=float)
    mask = np.asarray(data["mask"], dtype=bool)
    if edge.ndim != 1 or len(edge) < 3 or not np.isfinite(edge).all() or np.any(np.diff(edge) <= 0):
        raise ValueError("invalid site histogram edges")
    if density.shape != (len(edge) - 1,) or mask.shape != density.shape:
        raise ValueError("site histogram density/mask shape mismatch")
    widths = np.diff(edge)
    if "probability_mass" in data:
        mass = np.asarray(data["probability_mass"], dtype=float)
        if mass.shape != density.shape or not np.allclose(mass, density * widths, rtol=1e-10, atol=1e-15, equal_nan=True):
            raise ValueError("histogram density and probability mass have inconsistent normalization")
    result = {
        "ready": False,
        "status": "insufficient_support",
        "reasons": [],
        "k_qtst_au": None,
        "k_qtst_s": None,
        "rate_unit": "s^-1",
        "surface_density": None,
        "basin_probability": None,
        "mass_metric": metric,
        "flux_prefactor_au": float(np.sqrt(metric / (2 * np.pi * beta))),
        "basin": limits.tolist(),
        "dividing_surface": surface,
        "measure": "cv_density",
        "estimator": "finite_bin_mbar_density",
        "surface_estimator": "linear interpolation of neighboring bin-average densities at bin centers",
        "basin_estimator": "piecewise-constant density integrated over the declared basin",
    }
    if not data.get("connected", False):
        result["status"] = "disconnected_overlap"
        result["reasons"].append("umbrella_overlap_disconnected")
        return result
    if limits[0] < edge[0] or limits[1] > edge[-1]:
        result["reasons"].append("basin_not_fully_inside_histogram")
        return result
    mid = (edge[:-1] + edge[1:]) / 2
    if surface < mid[0] or surface > mid[-1]:
        result["reasons"].append("surface_not_bracketed_by_histogram_centers")
        return result
    right = int(np.searchsorted(mid, surface, side="left"))
    if right < len(mid) and surface == mid[right]:
        left = right
        fraction = 0.0
    else:
        left = right - 1
        fraction = float((surface - mid[left]) / (mid[right] - mid[left]))
    overlap = np.maximum(0, np.minimum(edge[1:], limits[1]) - np.maximum(edge[:-1], limits[0]))
    basin_bins = np.flatnonzero(overlap > 0)
    lo = min(int(basin_bins[0]), left)
    hi = max(int(basin_bins[-1]), right)
    invalid = mask | ~np.isfinite(density) | (density <= 0)
    result["support_bins"] = [lo, hi]
    result["surface_bins"] = [left, right]
    if invalid[lo:hi + 1].any():
        result["reasons"].append("masked_or_empty_bin_in_basin_surface_support")
        result["unsupported_bins"] = (lo + np.flatnonzero(invalid[lo:hi + 1])).tolist()
        return result
    population = float(np.dot(density[basin_bins], overlap[basin_bins]))
    rho = float((1 - fraction) * density[left] + fraction * density[right])
    rate = result["flux_prefactor_au"] * rho / population
    if not np.isfinite(rate) or rate <= 0 or not np.isfinite(population) or population <= 0:
        result["reasons"].append("nonfinite_or_zero_basin_flux")
        return result
    result.update(ready=True, status="ready", surface_density=rho, basin_probability=population,
                  k_qtst_au=float(rate), k_qtst_s=float(rate / ATOMIC_TIME_S))
    return result


def _estimate_pmf(sampled, config, system, output):
    cv, metadata, options = _setup(config, system)
    data = sampled["data"]
    estimate = estimate_site_qtst(data, config["beta"], metadata["mass_metric"],
                                  metadata["basin"], metadata["dividing_surface"])
    estimate["preparation"] = metadata
    estimate["warnings"] = list(sampled["summary"].get("warnings", []))
    estimate["warnings"].append("QTST uses finite histogram bins; bin-size convergence has not been established.")
    count = _pmf_bootstrap_count(config)
    draws = []
    arrays = [np.asarray(stream["samples"]) for stream in sampled["streams"]]
    blocks = np.maximum(1, np.ceil(data["statistical_inefficiency"]).astype(int))
    independent_blocks = [len(values) / block for values, block in zip(arrays, blocks)]
    if estimate["ready"] and count:
        centers = np.asarray([stream["center"] for stream in sampled["streams"]])
        stiffness = np.asarray([stream["stiffness"] for stream in sampled["streams"]])
        sizes = np.asarray([len(values) for values in arrays])
        rng = np.random.default_rng(np.random.SeedSequence([
            int(config.get("analysis", {}).get("seed", config.get("seed", 0))), 1201]))
        analysis = config.get("analysis", {})
        for _ in range(count):
            values = np.concatenate([_block_resample(a, b, rng) for a, b in zip(arrays, blocks)])
            reduced = umbrella_reduced_potentials(values, centers, stiffness, config["beta"], periods=(None,))
            mbar = solve_mbar(reduced, sizes, overlap_threshold=float(analysis.get("overlap_threshold", 1e-6)),
                              initial_free_energies=data["window_free_energies"])
            if not mbar.converged or len(set(mbar.components)) != 1:
                continue
            mass, density, raw_count, _, mask = _histogram(values, mbar.weights, data["edges"], config["beta"],
                                                          min_count=int(analysis.get("min_count", 1)))
            trial = estimate_site_qtst(
                {"edges": data["edges"], "density": density, "probability_mass": mass,
                 "mask": mask, "connected": True, "measure": "cv_density"},
                config["beta"], metadata["mass_metric"], metadata["basin"], metadata["dividing_surface"])
            if trial["ready"]:
                draws.append(trial["k_qtst_s"])
    reasons = []
    if count < 2:
        reasons.append("fewer_than_two_requested_pmf_bootstrap_draws")
    if len(draws) != count:
        reasons.append("pmf_bootstrap_draws_with_insufficient_support")
    if any(value < 2 for value in independent_blocks):
        reasons.append("fewer_than_two_correlation_blocks_in_a_pmf_stream")
    complete = estimate["ready"] and len(draws) >= 2 and not reasons
    estimate["pmf_uncertainty"] = {
        "available": complete, "reasons": reasons, "requested": count, "valid": len(draws),
        "block_lengths": blocks.tolist(), "blocks_per_stream": independent_blocks,
        "method": "independent circular correlated-block resampling within each umbrella stream",
        "standard_error_s": float(np.std(draws, ddof=1)) if complete else None,
        "percentile_95_ci_s": np.quantile(draws, [.025, .975]).tolist() if complete else None,
    }
    estimate["_bootstrap_qtst_s"] = draws
    write_json(Path(output) / "site_qtst.json", _public_pmf(estimate))
    atomic_savez(Path(output) / "site_qtst.npz", kind="site_rate_qtst",
                 edges=data["edges"][0], density=data["density"], mask=data["mask"],
                 probability_mass=data["probability_mass"], bootstrap_qtst_s=np.asarray(draws))
    tasks = []
    if estimate["ready"]:
        streams = sorted(sampled["streams"], key=lambda stream: (
            abs(float(stream["center"][0]) - metadata["dividing_surface"]),
            stream["window"], stream["replica"]))
        for chain in range(options["parent_replicas"]):
            tasks.append({"config": config, "system": system, "options": options,
                          "chain": chain, "output": str(output),
                          "initial_q": streams[chain % len(streams)]["last_q"]})
    return estimate, tasks


def _public_pmf(pmf):
    return {key: value for key, value in pmf.items() if not key.startswith("_")}


def _dynamics(config, system, provider, namespace, *identity):
    return Dynamics(
        provider, system.masses, config["beta"], config["beads"], config["dt_au"],
        seed=np.random.SeedSequence([int(config.get("seed", 0)), namespace, *identity]),
        native=config.get("native", config.get("dynamics", {}).get("native", True)),
        threads=config.get("parallel", {}).get("threads_per_rank", 1), **loop_options(config))


def _parent_task(task):
    config, system = task["config"], task["system"]
    cv, metadata, options = _setup(config, system)
    surface = SiteSurface(cv, metadata["dividing_surface"])
    chain = int(task["chain"])
    checkpoint = Path(task["output"]) / "checkpoints" / f"site_parent_{chain:04d}.npz"
    identity = _fingerprint(config, system, {
        "workflow_version": 1, "stage": "site_parent", "chain": chain,
        "surface": metadata, "options": options, "initial_q": task["initial_q"]})
    provider = make_task_potential(config["potential"], system)
    try:
        dyn = _dynamics(config, system, provider, 1211, chain)
        elapsed, snapshots = 0, []
        target = options["equilibration_steps"] + options["parent_samples"] * options["parent_interval_steps"]
        if checkpoint.exists():
            with load_checkpoint(checkpoint, expected_kind="site_rate_parent") as saved:
                if int(saved["payload_version"]) != 1:
                    raise ValueError("unsupported site parent checkpoint payload version")
                if str(saved["fingerprint"]) != identity:
                    raise ValueError(f"site parent checkpoint physics differs: {checkpoint}")
                state = State(saved["q"], saved["p"], int(saved["step"]), float(saved["time"]))
                dyn.rng.bit_generator.state = json.loads(str(saved["rng"]))
                elapsed = int(saved["elapsed"])
                snapshots = list(saved["snapshots"])
                dyn.propagation_stats = json.loads(str(saved["propagation"]))
            expected = max(0, elapsed - options["equilibration_steps"]) // options["parent_interval_steps"]
            if elapsed != state.step or not 0 <= elapsed <= target or len(snapshots) != expected:
                raise ValueError("invalid accepted prefix in site parent checkpoint")
            if not np.isclose(state.time, elapsed * dyn.dt, rtol=1e-10, atol=1e-12):
                raise ValueError("site parent checkpoint clock differs from its accepted prefix")
            for q in [state.q, *snapshots]:
                if np.shape(q) != dyn.integrator.shape or not np.isfinite(q).all() or abs(surface.centroid_value_gradient(np.mean(q, axis=0))[0]) > max(1e-8, 10 * options["constraint_tolerance"]):
                    raise ValueError("site parent checkpoint is not on its dividing surface")
            _, gradient = surface.centroid_value_gradient(state.q.mean(axis=0))
            normal_velocity = float(np.sum(gradient * state.p.mean(axis=0) / system.masses[:, None]))
            if abs(normal_velocity) > 1e-10 * max(1., np.sqrt(metadata["mass_metric"] / config["beta"])):
                raise ValueError("site parent checkpoint momentum is not tangent to the surface")
        else:
            q = np.array(task["initial_q"], dtype=float, copy=True)
            value, gradient = surface.centroid_value_gradient(q.mean(axis=0))
            # Exact linear shift avoids the nonlinear projector's step cap.
            q -= value * gradient[None] / (metadata["mass_metric"] * system.masses[None, :, None])
            q = project_positions(q, surface, system.masses, options["constraint_tolerance"])
            state = State(q, dyn.sample_momenta())
            project_momenta(state.p, state.q, surface, system.masses)

        def save():
            atomic_savez(checkpoint, kind="site_rate_parent", payload_version=1,
                         fingerprint=identity, q=state.q, p=state.p, step=state.step, time=state.time,
                         rng=json.dumps(dyn.rng.bit_generator.state), elapsed=elapsed,
                         snapshots=np.asarray(snapshots).reshape((-1,) + state.q.shape),
                         propagation=json.dumps(dyn.propagation_stats))

        interval = checked_integer(config["sampling"].get("andersen_interval", 10), "sampling.andersen_interval")
        checkpoint_steps = checked_integer(config.get("output", {}).get("checkpoint_steps", 1000), "output.checkpoint_steps")
        save()
        try:
            while elapsed < target:
                old_rng = copy.deepcopy(dyn.rng.bit_generator.state)
                candidate = state.copy()
                try:
                    if candidate.step % interval == 0:
                        candidate.p[:] = dyn.sample_momenta()
                        project_momenta(candidate.p, candidate.q, surface, system.masses)
                    constrained_step(dyn, candidate, surface, options["constraint_tolerance"])
                except BaseException:
                    dyn.rng.bit_generator.state = old_rng
                    raise
                production = elapsed + 1 - options["equilibration_steps"]
                next_snapshots = snapshots
                if production > 0 and production % options["parent_interval_steps"] == 0:
                    next_snapshots = [*snapshots, candidate.q.copy()]
                state, elapsed, snapshots = candidate, elapsed + 1, next_snapshots
                dyn.propagation_stats["python_steps"] += 1
                if elapsed % checkpoint_steps == 0:
                    save()
        except BaseException:
            save()
            raise
        save()
        return {"chain": chain, "snapshots": np.asarray(snapshots), "checkpoint": str(checkpoint),
                "propagation": dyn.propagation_stats}
    finally:
        provider.close()


def _child_task(task):
    config, system = task["config"], task["system"]
    cv, metadata, options = _setup(config, system)
    surface = SiteSurface(cv, metadata["dividing_surface"])
    chain, parent, pair = (int(task[key]) for key in ("chain", "parent", "pair"))
    nsteps = options["child_steps"]
    q0 = np.asarray(task["q"], dtype=float)
    if q0.shape != (config["beads"], system.natoms, 3) or abs(surface.centroid_value_gradient(q0.mean(axis=0))[0]) > max(1e-8, 10 * options["constraint_tolerance"]):
        raise ValueError("site child must start on the declared surface")
    checkpoint = Path(task["output"]) / "checkpoints" / f"site_child_{chain:04d}_{parent:05d}_{pair:04d}.npz"
    identity = _fingerprint(config, system, {
        "workflow_version": 1, "stage": "site_child", "chain": chain, "parent": parent,
        "pair": pair, "surface": metadata, "child_steps": nsteps, "q": q0})
    provider = make_task_potential(config["potential"], system)
    try:
        dyn = _dynamics(config, system, provider, 1307, chain, parent, pair)
        if checkpoint.exists():
            with load_checkpoint(checkpoint, expected_kind="site_rate_child") as saved:
                if int(saved["payload_version"]) != 1:
                    raise ValueError("unsupported site child checkpoint payload version")
                if str(saved["fingerprint"]) != identity:
                    raise ValueError(f"site child checkpoint physics differs: {checkpoint}")
                states = [State(saved["q"][i], saved["p"][i], int(saved["steps"][i]), float(saved["times"][i])) for i in range(2)]
                numerator = np.array(saved["numerator"], copy=True)
                velocity = float(saved["velocity"])
                initial = float(saved["initial_energy"])
                final = np.array(saved["final_energy"], copy=True)
                dyn.rng.bit_generator.state = json.loads(str(saved["rng"]))
                dyn.propagation_stats = json.loads(str(saved["propagation"]))
            if numerator.shape != (2, nsteps) or not np.isfinite(numerator).all() or final.shape != (2,):
                raise ValueError("invalid site child checkpoint records")
            if any(state.q.shape != q0.shape or state.step > nsteps for state in states):
                raise ValueError("invalid accepted prefix in site child checkpoint")
            if states[1].step and states[0].step != nsteps:
                raise ValueError("site child branches have inconsistent accepted prefixes")
            if any(not np.isclose(state.time, state.step * dyn.dt, rtol=1e-10, atol=1e-12) for state in states):
                raise ValueError("site child checkpoint clock differs from its accepted prefix")
            if any(np.any(numerator[i, state.step:] != 0) for i, state in enumerate(states)):
                raise ValueError("site child checkpoint contains unaccepted flux records")
            if any(np.isfinite(final[i]) and state.step != nsteps for i, state in enumerate(states)):
                raise ValueError("site child has an endpoint energy before trajectory completion")
            if np.isinf(final).any() or any(np.any((numerator[i] != 0) & (numerator[i] != sign * velocity)) for i, sign in enumerate((1, -1))):
                raise ValueError("site child checkpoint contains invalid energy or signed flux")
        else:
            # Unconstrained Maxwell momenta give centroid variance z/beta.
            # Projecting these momenta would erase the reactive flux.
            p = dyn.sample_momenta()
            _, gradient = surface.centroid_value_gradient(q0.mean(axis=0))
            velocity = metadata["direction"] * float(np.sum(gradient * p.mean(axis=0) / system.masses[:, None]))
            states = [State(q0, p), State(q0, -p)]
            numerator = np.zeros((2, nsteps))
            initial = dyn.energy(states[0])
            final = np.full(2, np.nan)
        if not np.isfinite(velocity) or not np.isfinite(initial):
            raise ValueError("invalid initial site child flux or energy")

        def save():
            atomic_savez(checkpoint, kind="site_rate_child", payload_version=1, fingerprint=identity,
                         q=np.asarray([state.q for state in states]), p=np.asarray([state.p for state in states]),
                         steps=[state.step for state in states], times=[state.time for state in states],
                         numerator=numerator, velocity=velocity, initial_energy=initial, final_energy=final,
                         rng=json.dumps(dyn.rng.bit_generator.state), propagation=json.dumps(dyn.propagation_stats))

        checkpoint_steps = checked_integer(config.get("output", {}).get("checkpoint_steps", 1000), "output.checkpoint_steps")
        save()
        try:
            for branch, sign in enumerate((1, -1)):
                while states[branch].step < nsteps:
                    candidate = states[branch].copy()
                    dyn.step(candidate)  # No thermostat, constraint, or umbrella.
                    value, _ = surface.centroid_value_gradient(candidate.q.mean(axis=0))
                    contribution = sign * velocity * (metadata["direction"] * value > 0)
                    numerator[branch, candidate.step - 1] = contribution
                    states[branch] = candidate
                    dyn.propagation_stats["python_steps"] += 1
                    if sum(state.step for state in states) % checkpoint_steps == 0:
                        save()
                if not np.isfinite(final[branch]):
                    final[branch] = dyn.energy(states[branch])
                save()
        except BaseException:
            save()
            raise
        endpoint = float(np.max(np.abs(final - initial)) / config["beads"])
        return {"chain": chain, "parent": parent, "pair": pair,
                "surface": metadata["dividing_surface"], "direction": metadata["direction"],
                "numerator": numerator.sum(axis=0), "denominator": abs(velocity),
                "endpoint_energy_error_hartree_per_bead": endpoint,
                "endpoint_energy_error_kbt": endpoint * config["beta"],
                "checkpoint": str(checkpoint), "propagation": dyn.propagation_stats}
    finally:
        provider.close()


def _child_tasks(parents, config, system, output):
    options = _recrossing_options(config)
    return [{"config": config, "system": system, "output": str(output),
             "chain": record["chain"], "parent": parent, "pair": pair, "q": q}
            for record in parents for parent, q in enumerate(record["snapshots"])
            for pair in range(options["child_pairs"])]


def _aggregate(children, pmf, config, output):
    options = _recrossing_options(config)
    public_pmf = _public_pmf(pmf)
    warnings = list(pmf.get("warnings", []))
    result = {
        "task": "site_rate", "rate_unit": "s^-1", "converged": False,
        "status": pmf["status"] if not pmf["ready"] else "insufficient_recrossing",
        "rate_s": None, "rate_estimate_s": None, "rate_standard_error_s": None,
        "rate_percentile_95_ci_s": None, "kappa": None, "pmf": public_pmf,
        "k_qtst_s": pmf["k_qtst_s"], "warnings": warnings,
        "convergence_note": "A finite-time plateau is not bead, timestep, cell-size or statistical convergence.",
        "files": {"pmf": str(Path(output) / "site_qtst.json"),
                  "histogram": str(Path(output) / "pmf" / "fes.csv")},
    }
    if not children:
        if not pmf["ready"]:
            warnings.append("No recrossing trajectories were launched because the full basin/surface histogram is unsupported.")
        write_json(Path(output) / "summary.json", result)
        return result
    if not pmf["ready"]:
        raise ValueError("cannot combine site children with an unsupported QTST estimate")
    grouped, identities = {}, set()
    for child in children:
        identity = tuple(child[key] for key in ("chain", "parent", "pair"))
        if identity in identities:
            raise ValueError("duplicate site child identity")
        identities.add(identity)
        if child["surface"] != pmf["dividing_surface"] or child["direction"] != pmf["preparation"]["direction"]:
            raise ValueError("site child belongs to another surface/direction")
        values = np.asarray(child["numerator"])
        if values.shape != (options["child_steps"],) or not np.isfinite(values).all() or not np.isfinite(child["denominator"]) or child["denominator"] < 0:
            raise ValueError("invalid site child flux record")
        key = identity[:2]
        entry = grouped.setdefault(key, [np.zeros(options["child_steps"]), 0.0])
        entry[0] += values
        entry[1] += child["denominator"]
    keys = sorted(grouped)
    num = np.asarray([grouped[key][0] for key in keys])
    den = np.asarray([grouped[key][1] for key in keys])
    if den.sum() <= 0:
        raise ValueError("zero initial site flux; transmission cannot be estimated")
    curve = num.sum(axis=0) / den.sum()
    tail = min(len(curve), max(2, int(np.ceil(len(curve) * options["plateau_fraction"]))))
    start, middle = len(curve) - tail, len(curve) - tail + tail // 2
    kappa = float(curve[start:].mean())
    drift = float(curve[middle:].mean() - curve[start:middle].mean())
    chain_indices = {chain: np.asarray([i for i, key in enumerate(keys) if key[0] == chain])
                     for chain in sorted({key[0] for key in keys})}
    blocks = {}
    for chain, ix in chain_indices.items():
        observable = np.divide(num[ix, start:].mean(axis=1), den[ix], out=np.zeros(len(ix)), where=den[ix] > 0)
        blocks[chain] = max(1, int(np.ceil(statistical_inefficiency(observable))))
    rng = np.random.default_rng(np.random.SeedSequence([int(config.get("seed", 0)), 1503]))
    boot_curves = []
    for _ in range(options["bootstrap"]):
        selected = np.concatenate([_block_resample(chain_indices[chain], blocks[chain], rng)
                                   for chain in rng.choice(list(chain_indices), size=len(chain_indices), replace=True)])
        if den[selected].sum() > 0:
            boot_curves.append(num[selected].sum(axis=0) / den[selected].sum())
    bootstrap = np.asarray(boot_curves).reshape((-1, len(curve)))
    recrossing_complete = len(chain_indices) >= 2 and len(bootstrap) >= 2 and len(bootstrap) == options["bootstrap"]
    kappa_draws = bootstrap[:, start:].mean(axis=1)
    curve_error = bootstrap.std(axis=0, ddof=1) if recrossing_complete else np.full_like(curve, np.nan)
    kappa_error = float(kappa_draws.std(ddof=1)) if recrossing_complete else None
    drift_error = float((bootstrap[:, middle:].mean(axis=1) - bootstrap[:, start:middle].mean(axis=1)).std(ddof=1)) if recrossing_complete else None
    threshold = max(options["plateau_tolerance"] * abs(kappa), 2 * (drift_error or 0))
    plateau = bool(tail >= 10 and abs(drift) <= threshold and kappa > 0)
    candidate = pmf["k_qtst_s"] * kappa
    complete = recrossing_complete and pmf["pmf_uncertainty"]["available"]
    rate_draws = np.array([])
    reasons = list(pmf["pmf_uncertainty"]["reasons"])
    if len(chain_indices) < 2:
        reasons.append("fewer_than_two_independent_parent_chains")
    if len(bootstrap) < 2 or len(bootstrap) != options["bootstrap"]:
        reasons.append("insufficient_recrossing_bootstrap_draws")
    if complete:
        rate_draws = kappa_draws * rng.choice(pmf["_bootstrap_qtst_s"], size=len(kappa_draws), replace=True)
    energy_error = max(child["endpoint_energy_error_kbt"] for child in children)
    if not plateau:
        warnings.append("The selected finite-time tail does not pass the positive-plateau diagnostic.")
    if energy_error > .1:
        warnings.append("A child endpoint energy error exceeds 0.1 kBT per bead; reduce the timestep.")
    if len(keys) < 20:
        warnings.append("Fewer than 20 parent configurations; statistical diagnostics are preliminary.")
    if not complete:
        warnings.append("Complete rate uncertainty is unavailable: " + "; ".join(reasons) + ".")
    times = np.arange(1, len(curve) + 1) * config["dt_au"]
    atomic_savez(Path(output) / "recrossing.npz", kind="site_rate_recrossing",
                 time_au=times, time_fs=times * ATOMIC_TIME_S / 1e-15,
                 kappa=curve, standard_error=curve_error, parent_ids=np.asarray(keys),
                 parent_numerator=num, parent_denominator=den, bootstrap_kappa=kappa_draws,
                 bootstrap_rate_s=rate_draws)
    np.savetxt(Path(output) / "recrossing.csv",
               np.column_stack([times, times * ATOMIC_TIME_S / 1e-15, curve, curve_error]),
               delimiter=",", comments="", header="time_au,time_fs,kappa,standard_error")
    result.update({
        "status": "finite_time_plateau_estimate" if plateau and energy_error <= .1 else "pilot_requires_convergence",
        "rate_s": candidate if plateau and energy_error <= .1 else None,
        "rate_estimate_s": candidate, "kappa": kappa, "kappa_standard_error": kappa_error,
        "rate_standard_error_s": float(rate_draws.std(ddof=1)) if complete else None,
        "rate_percentile_95_ci_s": np.quantile(rate_draws, [.025, .975]).tolist() if complete else None,
        "conditional_rate_standard_error_s": pmf["k_qtst_s"] * kappa_error if kappa_error is not None else None,
        "rate_uncertainty": {
            "available": complete, "reasons": reasons,
            "method": "independent PMF block-bootstrap and chain/parent-cluster bootstrap draws, fixed basin and surface",
            "conditional_quantity": "recrossing uncertainty holding QTST fixed",
            "pmf_bootstrap_valid": len(pmf["_bootstrap_qtst_s"]), "recrossing_bootstrap_valid": len(bootstrap),
            "parent_block_lengths": blocks,
        },
        "plateau_detected": plateau, "plateau_start_au": float(times[start]),
        "plateau_drift": drift, "plateau_drift_standard_error": drift_error,
        "child_time_fs": float(times[-1] * ATOMIC_TIME_S / 1e-15),
        "parent_chains": len(chain_indices), "parent_configurations": len(keys),
        "child_pairs": len(children), "child_trajectories": 2 * len(children),
        "maximum_endpoint_energy_error_kbt": energy_error,
    })
    result["files"]["recrossing"] = str(Path(output) / "recrossing.csv")
    write_json(Path(output) / "summary.json", result)
    return result


def run_site_rate(config, system, output, executor=None):
    """Sample local umbrellas, constrained parent chains, and paired children."""
    config = copy.deepcopy(config)
    root_call(validate_site_rate, config, system)
    config.setdefault("analysis", {})["measure"] = "cv_density"
    sampling_config = copy.deepcopy(config)
    # The rate bootstrap below needs the density/population ratio for each
    # correlated draw; a bootstrap of PMF ordinates alone does not supply it.
    sampling_config["analysis"]["bootstrap"] = 0
    sampled = run_fes(sampling_config, system, Path(output) / "pmf", executor=executor, return_data=True)
    pmf, tasks = root_call(_estimate_pmf, sampled, config, system, str(output))
    mapper = executor.map if executor is not None else lambda function, jobs: run_tasks(function, jobs, config.get("parallel", {}))
    parents = mapper(_parent_task, tasks)
    tasks = root_call(_child_tasks, parents, config, system, str(output))
    children = mapper(_child_task, tasks)
    return root_call(_aggregate, children, pmf, config, str(output))
