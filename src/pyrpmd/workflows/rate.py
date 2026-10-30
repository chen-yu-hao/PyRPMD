"""Bennett--Chandler bimolecular RPMD rate with explicit sampling measures.

The RPMDrate/Caracal flux-metric PMF and paired-child estimator are used,
with a newly implemented centroid Cayley-RATTLE constrained parent sampler.
The rate is k_RPMD = k_QTST * kappa; kappa is always measured, never guessed.
Reference implementations are MIT licensed; see THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations
from pyrpmd.propagation import loop_options, scientific_config

import copy
import json
from pathlib import Path
import numpy as np
from ..checkpoint import load_checkpoint

from ..analysis.fes import (
    umbrella_integration,
    _block_resample,
    _bootstrap_block_plan,
)
from ..analysis.recrossing import summarize_recrossing
from ..constants import AU_RATE_TO_CM3_S, ATOMIC_TIME_S
from ..cv import RPMDrateCoordinate
from ..dynamics import Dynamics, State
from ..input import rate_pmf_options
from ..parallel import run_tasks, root_call, make_task_potential
from ..units import parse_quantity, time_steps
from .prepare import prepare_rate
from .fes import (
    run_fes,
    write_json,
    jsonable,
    resolve_umbrellas,
    _atomic_npz,
    _fingerprint,
    checked_integer,
)

FLUX_MEASURE = "delta(xi-x) * sqrt(grad(xi) M^-1 grad(xi)/(2*pi*beta))"


def rate_coordinate(config, system):
    if config.get("task") == "surface_rate":
        from .surface import surface_definition

        return surface_definition(config, system)["coordinate"]
    reaction = config["reaction"]
    return RPMDrateCoordinate(
        reaction["fragments"],
        system.masses,
        system.geometry,
        reaction["forming_bonds"],
        reaction["breaking_bonds"],
        reaction["rinf"],
        equivalent_paths=reaction.get("equivalent_paths"),
    )


def _pmf_method(config):
    if config.get("task") == "surface_rate":
        return "mbar_surface_density"
    return rate_pmf_options(config)["method"]


def _normal(surface, q, masses):
    value, g = surface.centroid_value_gradient(np.mean(q, axis=0))
    metric = float(np.sum(g * g / np.asarray(masses)[:, None]))
    if not np.isfinite(metric) or metric <= 1e-28:
        raise ValueError("singular constrained dividing surface")
    return value, g, metric


def project_positions(q, surface, masses, tolerance=1e-10, max_iterations=100):
    """Project an initial centroid onto S=0; used only before equilibration."""
    q = np.asarray(q, float).copy()
    m = np.asarray(masses)[:, None]
    for _ in range(max_iterations):
        value, g, z = _normal(surface, q, masses)
        if abs(value) < tolerance:
            return q
        correction = -value * g / (z * m)
        maximum = np.max(np.linalg.norm(correction, axis=1))
        if maximum > 0.2:
            correction *= 0.2 / maximum
        q += correction[None]
    raise RuntimeError("initial dividing-surface projection did not converge")


def project_momenta(p, q, surface, masses):
    """Gaussian conditional momentum projection onto the centroid tangent."""
    _, g, z = _normal(surface, q, masses)
    velocity = float(np.sum(g * np.mean(p, axis=0) / np.asarray(masses)[:, None]))
    p -= velocity / z * g[None]
    return p


def constrained_step(dynamics, state, surface, tolerance=1e-10, max_iterations=50):
    """Symmetric centroid Cayley-RATTLE, transactional on PES/constraint failure.

    The old-normal multiplier is solved *inside* the position step. Because
    centroid motion is a free drift in the ring Cayley map, its corrections
    commute with all internal modes. A new-normal tangent projection follows
    the force at the constrained final position, as required by RATTLE.
    """
    dynamics._check_parameters()
    dynamics.integrator._check(state.q, state.p)
    dynamics._check_fixed(state.q, state.p)
    q = state.q.copy()
    p = state.p.copy()
    m = dynamics.masses[:, None]
    dt = dynamics.dt
    initial, g0, _ = _normal(surface, q, dynamics.masses)
    if dynamics.fixed_atoms and np.any(g0[list(dynamics.fixed_atoms)] != 0):
        raise ValueError("constrained surface gradients must exclude fixed atoms")
    if abs(initial) > max(1e-8, 10 * tolerance):
        raise ValueError("parent state is not on the dividing surface")
    dynamics.integrator.kick(p, dynamics._force(q), 0.5 * dt)
    dynamics.integrator.free_step(q, p)
    dynamics._project_fixed(q, p)
    free = q.copy()
    direction = g0 / m
    lam = 0.0
    for _ in range(max_iterations):
        value, g, _ = _normal(surface, q, dynamics.masses)
        if dynamics.fixed_atoms and np.any(g[list(dynamics.fixed_atoms)] != 0):
            raise ValueError("constrained surface gradients must exclude fixed atoms")
        if abs(value) < tolerance:
            break
        derivative = float(np.sum(g * direction))
        if abs(derivative) < 1e-25:
            raise RuntimeError("RATTLE old/new normals became orthogonal")
        lam -= value / derivative
        q[:] = free + lam * direction[None]
    else:
        raise RuntimeError(
            "Cayley-RATTLE constraint failed to converge; reduce timestep"
        )
    p += lam / dt * g0[None]
    dynamics.integrator.kick(p, dynamics._force(q), 0.5 * dt)
    project_momenta(p, q, surface, dynamics.masses)
    dynamics._project_fixed(q, p)
    if not np.isfinite(q).all() or not np.isfinite(p).all():
        raise FloatingPointError("nonfinite parent state")
    state.q[:] = q
    state.p[:] = p
    state.step += 1
    state.time += dt
    return state


def _recrossing_spec(config):
    return config.get("recrossing", config.get("rate", {}).get("recrossing", {}))


def _recrossing_options(config):
    raw = _recrossing_spec(config)
    dt = config["dt_au"]
    sampling = config["sampling"]

    def steps(name, duration, default):
        minimum = 0 if name == "equilibration_steps" else 1
        return (
            checked_integer(raw[name], "recrossing." + name, minimum)
            if name in raw
            else time_steps(raw.get(duration, default), dt, allow_zero=minimum == 0)
        )

    values = {
        "parent_replicas": checked_integer(
            raw.get("parent_replicas", 2), "recrossing.parent_replicas"
        ),
        "parent_samples": checked_integer(
            raw.get("parent_samples", sampling.get("parents", 4)),
            "recrossing.parent_samples",
        ),
        "equilibration_steps": steps("equilibration_steps", "equilibration", "20 fs"),
        "parent_interval_steps": steps(
            "parent_interval_steps", "parent_interval", "5 fs"
        ),
        "child_pairs": checked_integer(
            raw.get("child_pairs", raw.get("child_pairs_per_parent", 2)),
            "recrossing.child_pairs",
        ),
        "child_steps": steps("child_steps", "child_time", "50 fs"),
        "plateau_fraction": float(raw.get("plateau_fraction", 0.25)),
        "plateau_tolerance": float(raw.get("plateau_tolerance", 0.1)),
        "bootstrap": checked_integer(
            raw.get("bootstrap", 100), "recrossing.bootstrap", 0
        ),
        "constraint_tolerance": float(raw.get("constraint_tolerance", 1e-10)),
    }
    for key in (
        "parent_replicas",
        "parent_samples",
        "parent_interval_steps",
        "child_pairs",
        "child_steps",
    ):
        if values[key] < 1:
            raise ValueError(f"recrossing.{key} must be positive")
    if values["child_steps"] < 2:
        raise ValueError("recrossing.child_steps must be at least 2")
    if values["bootstrap"] not in (0,) and values["bootstrap"] < 2:
        raise ValueError("recrossing.bootstrap must be zero or at least 2")
    if (
        not np.isfinite(values["constraint_tolerance"])
        or values["constraint_tolerance"] <= 0
    ):
        raise ValueError("constraint tolerance must be positive")
    if (
        values["equilibration_steps"] < 0
        or not 0 < values["plateau_fraction"] <= 1
        or not np.isfinite(values["plateau_tolerance"])
        or values["plateau_tolerance"] <= 0
    ):
        raise ValueError("invalid recrossing burn-in/plateau options")
    return values


def _parent_task(task):
    config, system = task["config"], task["system"]
    options = task["options"]
    surface = rate_coordinate(config, system).dividing_surface(task["xi_star"])
    chain = int(task["chain"])
    method = _pmf_method(config)
    provider = make_task_potential(config["potential"], system)
    checkpoint = Path(task["output"]) / "checkpoints" / f"parent_{chain:04d}.npz"
    identity = _fingerprint(
        config,
        system,
        {"stage": "parent", "chain": chain, "xi": task["xi_star"], "options": options},
    )
    try:
        dyn = Dynamics(
            provider,
            system.masses,
            config["beta"],
            config["beads"],
            config["dt_au"],
            seed=np.random.SeedSequence([int(config.get("seed", 0)), 211, chain]),
            native=config.get("native", config.get("dynamics", {}).get("native", True)),
            threads=config.get("parallel", {}).get("threads_per_rank", 1),
            fixed_atoms=system.fixed_atoms,
            fixed_positions=system.geometry,
            **loop_options(config),
        )
        snapshots = []
        elapsed = 0
        if checkpoint.exists():
            with load_checkpoint(checkpoint) as saved:
                if str(saved["fingerprint"]) != identity:
                    raise ValueError(f"parent checkpoint physics differs: {checkpoint}")
                state = State(
                    saved["q"], saved["p"], int(saved["step"]), float(saved["time"])
                )
                dyn._check_fixed(state.q, state.p)
                dyn.rng.bit_generator.state = json.loads(str(saved["rng"]))
                snapshots = list(saved["snapshots"])
                for snapshot in snapshots:
                    dyn._check_fixed(snapshot, np.zeros_like(snapshot))
                elapsed = int(saved["elapsed"])
                if "propagation" in saved:
                    dyn.propagation_stats = json.loads(str(saved["propagation"]))
                elif elapsed:
                    dyn.propagation_stats["untracked_steps"] = elapsed
                if "xi_star" in saved and float(saved["xi_star"]) != task["xi_star"]:
                    raise ValueError("parent checkpoint dividing surface differs")
                if "pmf_method" in saved and str(saved["pmf_method"]) != method:
                    raise ValueError("parent checkpoint PMF method differs")
                if any(
                    abs(_normal(surface, q, system.masses)[0])
                    > max(1e-8, 10 * options["constraint_tolerance"])
                    for q in snapshots
                ):
                    raise ValueError(
                        "parent checkpoint snapshots do not lie on the requested dividing surface"
                    )
        else:
            q = project_positions(
                task["initial_q"],
                surface,
                system.masses,
                tolerance=options["constraint_tolerance"],
            )
            state = State(q, dyn.sample_momenta())
            project_momenta(state.p, state.q, surface, system.masses)

        def save():
            _atomic_npz(
                checkpoint,
                q=state.q,
                p=state.p,
                step=state.step,
                time=state.time,
                rng=json.dumps(dyn.rng.bit_generator.state),
                snapshots=np.asarray(snapshots).reshape((-1,) + state.q.shape),
                elapsed=elapsed,
                fingerprint=identity,
                xi_star=task["xi_star"],
                pmf_method=method,
                propagation=json.dumps(dyn.propagation_stats),
            )

        stats_before = copy.deepcopy(dyn.propagation_stats)
        interval = int(config["sampling"].get("andersen_interval", 10))
        checkpoint_steps = int(config.get("output", {}).get("checkpoint_steps", 1000))
        target = (
            options["equilibration_steps"]
            + options["parent_samples"] * options["parent_interval_steps"]
        )
        while elapsed < target:
            if state.step % interval == 0:
                state.p[:] = dyn.sample_momenta()
                project_momenta(state.p, state.q, surface, system.masses)
            count = min(dyn.block_steps, 4096, target - elapsed,
                        interval - state.step % interval,
                        checkpoint_steps - elapsed % checkpoint_steps)
            production = elapsed + np.arange(1, count + 1) - options["equilibration_steps"]
            observed = (production > 0) & (production % options["parent_interval_steps"] == 0)
            block = dyn.propagate_block(
                state, count, constraint=surface,
                constraint_tolerance=options["constraint_tolerance"],
                observe=observed, record_energy=False, record_states=True,
            )
            snapshots.extend(block.q)
            elapsed += block.completed
            if elapsed % checkpoint_steps == 0:
                save()
        save()
        return {
            "chain": chain,
            "snapshots": np.asarray(snapshots),
            "propagation": copy.deepcopy(dyn.propagation_stats),
            "propagation_this_call": {
                **{key: dyn.propagation_stats[key] - stats_before[key]
                   for key in ("native_steps", "python_steps")},
                "fallback_reasons": {
                    key: value - stats_before["fallback_reasons"].get(key, 0)
                    for key, value in dyn.propagation_stats["fallback_reasons"].items()
                    if value != stats_before["fallback_reasons"].get(key, 0)
                },
            },
            "checkpoint": str(checkpoint),
            "xi_star": task["xi_star"],
            "pmf_method": method,
        }
    finally:
        if hasattr(provider, "close"):
            provider.close()


def _child_task(task):
    config, system = task["config"], task["system"]
    nsteps = task["child_steps"]
    surface = rate_coordinate(config, system).dividing_surface(task["xi_star"])
    chain, parent, pair = task["chain"], task["parent"], task["pair"]
    method = _pmf_method(config)
    provider = make_task_potential(config["potential"], system)
    checkpoint = (
        Path(task["output"])
        / "checkpoints"
        / f"child_{chain:04d}_{parent:05d}_{pair:04d}.npz"
    )
    identity = _fingerprint(
        config,
        system,
        {
            "stage": "child",
            "chain": chain,
            "parent": parent,
            "pair": pair,
            "xi": task["xi_star"],
            "child_steps": nsteps,
            "q": task["q"],
        },
    )
    try:
        if checkpoint.exists():
            with load_checkpoint(checkpoint) as saved:
                if str(saved["fingerprint"]) != identity:
                    raise ValueError(f"child checkpoint physics differs: {checkpoint}")
                if "xi_star" in saved and float(saved["xi_star"]) != task["xi_star"]:
                    raise ValueError("child checkpoint dividing surface differs")
                if "pmf_method" in saved and str(saved["pmf_method"]) != method:
                    raise ValueError("child checkpoint PMF method differs")
                return {
                    "chain": chain,
                    "parent": parent,
                    "pair": pair,
                    "xi_star": task["xi_star"],
                    "pmf_method": method,
                    "numerator": saved["numerator"],
                    "propagation": json.loads(str(saved["propagation"])) if "propagation" in saved else {},
                    "denominator": float(saved["denominator"]),
                    "energy_drift": float(saved["energy_drift"]),
                    "endpoint_energy_error_hartree_per_bead": (
                        float(saved["endpoint_energy_error_hartree_per_bead"])
                        if "endpoint_energy_error_hartree_per_bead" in saved
                        else None
                    ),
                    "endpoint_energy_error_kbt": (
                        float(saved["endpoint_energy_error_kbt"])
                        if "endpoint_energy_error_kbt" in saved
                        else None
                    ),
                }
        dyn = Dynamics(
            provider,
            system.masses,
            config["beta"],
            config["beads"],
            config["dt_au"],
            seed=np.random.SeedSequence(
                [int(config.get("seed", 0)), 307, chain, parent, pair]
            ),
            native=config.get("native", config.get("dynamics", {}).get("native", True)),
            threads=config.get("parallel", {}).get("threads_per_rank", 1),
            fixed_atoms=system.fixed_atoms,
            fixed_positions=system.geometry,
            **loop_options(config),
        )
        q = np.asarray(task["q"])
        p = dyn.sample_momenta()
        dyn._check_fixed(q, p)
        constraint, g, z = _normal(surface, q, system.masses)
        tolerance = float(_recrossing_spec(config).get("constraint_tolerance", 1e-10))
        if abs(constraint) > max(1e-8, 10 * tolerance):
            raise ValueError(
                "child starting configuration is not on the requested dividing surface"
            )
        fs = np.sqrt(z / (2 * np.pi * config["beta"]))
        velocity = float(np.sum(g * np.mean(p, axis=0) / system.masses[:, None]))
        numerator = np.zeros(nsteps)
        denominator = abs(velocity) / fs
        drift = 0.0
        endpoint_error_per_bead = 0.0
        # Unconstrained Maxwell momenta are deliberately drawn AFTER constrained
        # configuration sampling. Projecting child momenta would erase flux.
        for sign in (1.0, -1.0):
            state = State(q, sign * p)
            weight = sign * velocity / fs
            initial = dyn.energy(state)
            if dyn.can_fuse(observe=surface):
                completed = 0
                while completed < nsteps:
                    count = min(dyn.block_steps, nsteps-completed)
                    block = dyn.propagate(state, count, observe=surface,
                                          sample_interval=1, record_centroids=False)
                    numerator[completed:completed+count] += weight*(block.values[:, 0] > 0)
                    completed += count
            else:
                for step in range(nsteps):
                    dyn.step(state)
                    dyn._record_python_steps(1)
                    value, _ = surface.centroid_value_gradient(state.q.mean(axis=0))
                    if value > 0:
                        numerator[step] += weight
            final = dyn.energy(state)
            endpoint_error_per_bead = max(
                endpoint_error_per_bead, abs(final - initial) / config["beads"]
            )
            drift = max(
                drift,
                abs(final - initial)
                / max(abs(initial), config["beads"] / config["beta"]),
            )
        _atomic_npz(
            checkpoint,
            numerator=numerator,
            denominator=denominator,
            energy_drift=drift,
            endpoint_energy_error_hartree_per_bead=endpoint_error_per_bead,
            endpoint_energy_error_kbt=endpoint_error_per_bead * config["beta"],
            fingerprint=identity,
            xi_star=task["xi_star"],
            pmf_method=method,
            propagation=json.dumps(dyn.propagation_stats),
        )
        return {
            "chain": chain,
            "parent": parent,
            "pair": pair,
            "xi_star": task["xi_star"],
            "pmf_method": method,
            "numerator": numerator,
            "denominator": denominator,
            "energy_drift": drift,
            "endpoint_energy_error_hartree_per_bead": endpoint_error_per_bead,
            "endpoint_energy_error_kbt": endpoint_error_per_bead * config["beta"],
            "propagation": dict(dyn.propagation_stats),
        }
    finally:
        if hasattr(provider, "close"):
            provider.close()


def _prepare(config, system, output):
    pmf_options = rate_pmf_options(config)
    prepared = prepare_rate(config, system, numerical=True)
    write_json(Path(output) / "preparation.json", prepared)
    if not prepared["ready"]:
        raise ValueError(
            "TS-only reaction preparation is ambiguous: "
            + "; ".join(prepared["diagnostics"])
            + ". See preparation.json and specify reaction.fragments/forming_bonds/breaking_bonds."
        )
    result = copy.deepcopy(config)
    result["reaction"] = prepared["reaction"]
    reaction = result["reaction"]
    result["cvs"] = [
        {
            "name": "xi",
            "kind": "rpmdrate_xi",
            "estimator": "centroid",
            "fragments": reaction["fragments"],
            "forming_bonds": reaction["forming_bonds"],
            "breaking_bonds": reaction["breaking_bonds"],
            "rinf": reaction["rinf"],
        }
    ]
    if reaction.get("equivalent_paths"):
        result["cvs"][0]["equivalent_paths"] = reaction["equivalent_paths"]
    umbrella = result.setdefault("umbrella", {})
    if not any(k in umbrella for k in ("centers", "centers_au", "grid", "start")):
        umbrella.update(start=-0.05, stop=1.05, count=23)
    if not any(k in umbrella for k in ("stiffness", "stiffness_au", "kappa")):
        umbrella["kappa"] = 50.0 / result["beta"]
    umbrella["cvs"] = ["xi"]
    if result.get("analysis", {}).get("measure", "rate_flux") != "rate_flux":
        raise ValueError(
            "standard RPMD rate sampling requires analysis.measure='rate_flux'"
        )
    result.setdefault("analysis", {}).update(measure="rate_flux")
    result["analysis"].setdefault("bins", 100)
    result["analysis"].setdefault(
        "bootstrap",
        result["analysis"].get(
            "bootstrap_replicates", result["analysis"].get("bootstrap_samples", 20)
        ),
    )
    _, _, centers, _ = resolve_umbrellas(result, system)
    _check_pmf_domain(centers[:, 0], pmf_options)
    return result


def _check_pmf_domain(centers, options):
    centers = np.asarray(centers, float)
    if centers.min() > 0 or centers.max() < 1:
        raise ValueError("rate umbrellas must cover xi=0 (reactants) through xi=1 (TS)")
    if options["xi_star"] is not None and not 0 <= options["xi_star"] <= centers.max():
        raise ValueError("dividing surface outside sampled PMF range")
    if (
        options["search_range"] is not None
        and options["search_range"][1] > centers.max()
    ):
        raise ValueError(
            "rate.pmf_search_range extends beyond the umbrella-center range"
        )


def _histogram_endpoint_support(data, value):
    """Observed histogram support, used to qualify the Gaussian UI model."""
    edges = np.asarray(data["edges"][0])
    axis = 0.5 * (edges[:-1] + edges[1:])
    mask = np.asarray(data["mask"], bool).reshape(-1)
    index = int(np.searchsorted(axis, value))
    if index < len(axis) and np.isclose(axis[index], value, rtol=0, atol=1e-14):
        bracket = [index]
    elif 0 < index < len(axis):
        bracket = [index - 1, index]
    else:
        return {
            "coordinate": value,
            "supported": False,
            "reason": "outside_bin_centers",
        }
    supported = not mask[bracket].any()
    return {
        "coordinate": value,
        "supported": bool(supported),
        "reason": None if supported else "masked_endpoint_bracket",
        "bracket_centers": axis[bracket].tolist(),
        "raw_count": np.asarray(data["raw_count"]).reshape(-1)[bracket].tolist(),
    }


def _ui_rate_pmf(sampled, config, centers, samples, stiffness, options):
    """Retain the existing Gaussian UI estimator with explicit diagnostics."""
    grid = np.linspace(0.0, centers.max(), options["points"])
    curve, derivative = umbrella_integration(
        samples, centers, stiffness, config["beta"], grid
    )
    if not np.isfinite(curve).all() or not np.isfinite(derivative).all():
        raise FloatingPointError(
            "nonfinite Gaussian UI curve; inspect the umbrella streams"
        )
    interval = options["search_range"] or [0.0, float(centers.max())]
    eligible = np.flatnonzero((grid >= interval[0]) & (grid <= interval[1]))
    if not len(eligible):
        raise ValueError("rate.pmf_search_range contains no UI grid points")
    xi = options["xi_star"]
    if xi is None:
        xi = float(grid[eligible[np.argmax(curve[eligible])]])
    barrier = float(np.interp(xi, grid, curve) - curve[0])
    connected = bool(sampled["data"]["connected"])
    gs = np.asarray(sampled["data"]["statistical_inefficiency"])
    block_plan = _bootstrap_block_plan(np.asarray([len(s) for s in samples]), gs)
    rng = np.random.default_rng(np.random.SeedSequence([options["analysis_seed"], 401]))
    barriers, curves, records, endpoints, failures = [], [], [], [], []
    if connected:
        for draw in range(options["bootstrap"]):
            record = {"draw": draw, "valid": False, "reason": None}
            draws = [
                _block_resample(s, block, rng)
                for s, block in zip(samples, block_plan["block_lengths"])
            ]
            try:
                fb, _ = umbrella_integration(
                    draws, centers, stiffness, config["beta"], grid
                )
                rv, sv = float(fb[0]), float(np.interp(xi, grid, fb))
                if not np.isfinite(fb).all() or not np.isfinite(sv - rv):
                    raise FloatingPointError("nonfinite UI bootstrap curve")
                barriers.append(sv - rv)
                curves.append(fb - rv)
                endpoints.append([rv, sv])
                record.update(
                    valid=True,
                    reference_free_energy_hartree=rv,
                    surface_free_energy_hartree=sv,
                    barrier_hartree=sv - rv,
                )
            except (ValueError, FloatingPointError) as error:
                record.update(reason="ui_fit_failed", message=str(error))
                failures.append(
                    {key: value for key, value in record.items() if key != "valid"}
                )
            records.append(record)
    barrier_error = float(np.std(barriers, ddof=1)) if len(barriers) >= 2 else None
    standard_error = (
        np.std(curves, axis=0, ddof=1)
        if len(curves) >= 2
        else np.full_like(curve, np.nan)
    )
    if block_plan["status"] == "unavailable":
        barrier_error = None
        standard_error[:] = np.nan
    empirical = {
        "reference": _histogram_endpoint_support(sampled["data"], 0.0),
        "dividing_surface": _histogram_endpoint_support(sampled["data"], xi),
    }
    warnings = [
        "Gaussian UI models each biased window by its mean and variance; inspect non-Gaussian window distributions or compare with MBAR."
    ]
    warnings.extend(block_plan["warnings"])
    if not all(point["supported"] for point in empirical.values()):
        warnings.append(
            "A Gaussian UI endpoint lacks observed histogram support; the model value is not an interpolation of measured supported bins."
        )
    if options["bootstrap"] and len(barriers) < 2:
        warnings.append(
            "Fewer than two valid UI bootstrap draws; barrier uncertainty is unavailable."
        )
    if failures:
        warnings.append(
            "Some UI bootstrap draws failed; inspect the recorded reasons and valid fraction."
        )
    if not connected:
        warnings.append(
            "Disconnected overlap does not identify the PMF barrier; no recrossing trajectories are launched."
        )
    return {
        "ready": connected,
        "status": "ready" if connected else "disconnected_overlap",
        "xi_star": xi,
        "reference": 0.0,
        "free_energy_barrier_hartree": barrier if connected else None,
        "barrier_standard_error_hartree": barrier_error,
        "barrier_percentile_95_ci_hartree": (
            np.quantile(barriers, [0.025, 0.975]).tolist()
            if len(barriers) >= 2 and block_plan["status"] != "unavailable"
            else None
        ),
        "pmf_bootstrap_barriers": barriers,
        "xi": grid,
        "free_energy": curve,
        "standard_error": standard_error,
        "derivative": derivative,
        "mask": np.full(len(grid), not connected),
        "overlap_connected": connected,
        "support": {
            "model": "Gaussian-window integration within the umbrella-center range",
            "reference": {"coordinate": 0.0, "model_supported": connected},
            "dividing_surface": {"coordinate": xi, "model_supported": connected},
            "observed_histogram": empirical,
            "search_range": interval,
            "selection": (
                "fixed"
                if options["xi_star"] is not None
                else "maximum_gaussian_ui_grid_point"
            ),
            "failure_reason": None if connected else "disconnected_overlap",
        },
        "bootstrap": {
            "requested": options["bootstrap"],
            "attempted": len(records),
            "valid": len(barriers),
            "valid_fraction": (
                len(barriers) / options["bootstrap"] if options["bootstrap"] else None
            ),
            "failures": failures,
            "failure_counts": {"ui_fit_failed": len(failures)} if failures else {},
            "draws": records,
            "surface_held_fixed": True,
            "skipped_reason": None if connected else "disconnected_overlap",
            "block_lengths": block_plan["block_lengths"],
            "block_plan": block_plan,
            "endpoint_values_hartree": np.asarray(endpoints).reshape(-1, 2),
            "endpoint_covariance_hartree_squared": (
                np.cov(np.asarray(endpoints).T, ddof=1) if len(endpoints) >= 2 else None
            ),
            "curve_valid_draws": np.full(len(grid), len(curves)),
        },
        "effective_samples_per_stream": sampled["data"]["effective_samples_per_stream"],
        "weight_ess": sampled["data"]["weight_ess"],
        "warnings": warnings,
    }


def _save_pmf_arrays(output, estimate, method):
    grid = np.asarray(estimate["xi"])
    mask = np.asarray(estimate["mask"], bool)
    curve = np.where(mask, np.nan, estimate["free_energy"])
    error = np.where(mask, np.nan, estimate["standard_error"])
    arrays = {
        "xi": grid,
        "free_energy": curve,
        "standard_error": error,
        "mask": mask,
        "standard_error_mask": mask | ~np.isfinite(error),
        "bootstrap_barriers": np.asarray(estimate["pmf_bootstrap_barriers"]),
        "method": method,
        "sampling_measure": "rate_flux",
        "xi_star": np.asarray(
            estimate["xi_star"] if estimate["xi_star"] is not None else np.nan
        ),
        "reference": 0.0,
        "ready": bool(estimate["ready"]),
    }
    for key in (
        "derivative",
        "density",
        "raw_count",
        "edges",
        "overlap",
        "components",
        "window_free_energies",
    ):
        if key in estimate:
            arrays[key] = estimate[key]
    _atomic_npz(Path(output) / "rate_pmf.npz", **arrays)
    columns = [grid, curve]
    names = ["xi", "free_energy_hartree"]
    if "derivative" in estimate:
        columns.append(np.where(mask, np.nan, estimate["derivative"]))
        names.append("derivative_hartree")
    columns.extend([error, mask.astype(int), (mask | ~np.isfinite(error)).astype(int)])
    names.extend(["standard_error_hartree", "mask", "standard_error_mask"])
    np.savetxt(
        Path(output) / "rate_pmf.csv",
        np.column_stack(columns),
        delimiter=",",
        header=",".join(names),
        comments="",
    )


def _pmf_and_parent_tasks(sampled, config, system, output, options):
    streams = sampled["streams"]
    centers = np.array([s["center"][0] for s in streams])
    samples = [s["samples"][:, 0] for s in streams]
    stiffness = np.array([s["stiffness"][0, 0] for s in streams])
    pmf_options = rate_pmf_options(config)
    _check_pmf_domain(centers, pmf_options)
    method = pmf_options["method"]
    if method == "mbar":
        from ..analysis.fes import mbar_rate_pmf

        estimate = mbar_rate_pmf(
            samples,
            centers,
            stiffness,
            config["beta"],
            xi_star=pmf_options["xi_star"],
            reference=0.0,
            bins=pmf_options["bins"],
            ranges=pmf_options["ranges"],
            search_range=pmf_options["search_range"],
            bootstrap=pmf_options["bootstrap"],
            seed=np.random.SeedSequence([pmf_options["analysis_seed"], 401]),
            min_count=pmf_options["min_count"],
            overlap_threshold=pmf_options["overlap_threshold"],
            compare_ui=pmf_options["compare_ui"],
            ui_points=pmf_options["points"],
            measure="rate_flux",
        )
    else:
        estimate = _ui_rate_pmf(
            sampled, config, centers, samples, stiffness, pmf_options
        )
    _save_pmf_arrays(output, estimate, method)
    xi = estimate["xi_star"]
    barrier = estimate["free_energy_barrier_hartree"]
    fragments = config["reaction"]["fragments"]
    ma, mb = [system.masses[list(f)].sum() for f in fragments]
    reduced = ma * mb / (ma + mb)
    rinf = config["reaction"]["rinf"]
    prefactor = 4 * np.pi * rinf * rinf / np.sqrt(2 * np.pi * config["beta"] * reduced)
    electronic = float(config["reaction"].get("electronic_weight", 1.0))
    symmetry = float(config["reaction"].get("symmetry_factor", 1.0))
    if not np.isfinite(symmetry) or symmetry <= 0:
        raise ValueError("reaction.symmetry_factor must be positive")
    qtst = (
        float(prefactor * np.exp(-config["beta"] * barrier) * electronic * symmetry)
        if estimate["ready"]
        else None
    )
    if qtst is not None and not np.isfinite(qtst):
        raise FloatingPointError("nonfinite QTST estimate; inspect PMF sampling")
    source_status = estimate["status"]
    status = (
        "ready"
        if estimate["ready"]
        else (
            "insufficient_overlap"
            if source_status == "disconnected_overlap"
            else (
                "pmf_solver_failed"
                if source_status == "mbar_not_converged"
                else "unsupported_pmf"
            )
        )
    )
    diagnostics = {
        "schema_version": 2,
        "method": method,
        "ready": bool(estimate["ready"]),
        "status": status,
        "failure_reason": None if estimate["ready"] else source_status,
        "xi_star": xi,
        "reference": 0.0,
        "free_energy_barrier_hartree": barrier,
        "barrier_standard_error_hartree": estimate["barrier_standard_error_hartree"],
        "barrier_percentile_95_ci_hartree": estimate[
            "barrier_percentile_95_ci_hartree"
        ],
        "k_qtst_au": qtst,
        "k_qtst_cm3_s": qtst * AU_RATE_TO_CM3_S if qtst is not None else None,
        "reactant_prefactor_au": prefactor,
        "electronic_weight": electronic,
        "symmetry_factor": symmetry,
        "rinf_bohr": rinf,
        "reduced_mass_au": float(reduced),
        "pmf_method": (
            "Gaussian umbrella integration; MBAR overlap checked"
            if method == "ui"
            else "Histogram MBAR with paired endpoint bootstrap"
        ),
        "measure": FLUX_MEASURE,
        "sampling_measure": "rate_flux",
        "overlap_connected": bool(estimate["overlap_connected"]),
        "pmf_bootstrap_barriers": estimate["pmf_bootstrap_barriers"],
        "warnings": list(
            dict.fromkeys(sampled["summary"]["warnings"] + estimate["warnings"])
        ),
        "intermediate_fes_bootstrap": (
            0 if method == "mbar" else pmf_options["bootstrap"]
        ),
        "intermediate_fes_purpose": "Saved umbrella samples, overlap and a preliminary histogram; the selected rate PMF backend owns barrier bootstrap.",
    }
    for key in (
        "support",
        "solver",
        "bootstrap",
        "window_moments",
        "ui_comparison",
        "effective_samples_per_stream",
        "weight_ess",
    ):
        if key in estimate:
            diagnostics[key] = estimate[key]
    diagnostics["uncertainty"] = {
        "available": estimate["barrier_standard_error_hartree"] is not None,
        "block_status": estimate.get("bootstrap", {}).get("block_plan", {}).get("status"),
        "reason": (
            None
            if estimate["barrier_standard_error_hartree"] is not None
            else (
                "bootstrap_disabled"
                if pmf_options["bootstrap"] == 0
                else (
                    "insufficient_correlated_blocks"
                    if estimate.get("bootstrap", {}).get("block_plan", {}).get("status") == "unavailable"
                    else "fewer_than_two_valid_bootstrap_draws"
                )
            )
        ),
        "surface_held_fixed": True,
        "selection_uncertainty_included": False,
    }
    diagnostics = jsonable(diagnostics)
    write_json(Path(output) / "rate_pmf.json", diagnostics)
    if not estimate["ready"]:
        return diagnostics, []
    nearest = np.argsort(abs(centers - xi))
    permutations = rate_coordinate(config, system).permutations
    tasks = []
    for chain in range(options["parent_replicas"]):
        source = streams[nearest[(chain // len(permutations)) % len(nearest)]]["last_q"]
        initial_q = np.empty_like(source)
        initial_q[:, list(permutations[chain % len(permutations)]), :] = source
        tasks.append(
            {
                "config": config,
                "system": system,
                "options": options,
                "chain": chain,
                "xi_star": xi,
                "initial_q": initial_q,
                "output": str(output),
            }
        )
    return diagnostics, tasks


def _child_tasks(parents, config, system, output, options, pmf):
    method = pmf.get("method", "ui")
    for parent in parents:
        if parent.get("xi_star", pmf["xi_star"]) != pmf["xi_star"]:
            raise ValueError("parent ensemble belongs to another dividing surface")
        if parent.get("pmf_method", method) != method:
            raise ValueError("parent ensemble belongs to another PMF method")
    return [
        {
            "config": config,
            "system": system,
            "chain": p["chain"],
            "parent": j,
            "pair": pair,
            "q": q,
            "xi_star": pmf["xi_star"],
            "child_steps": options["child_steps"],
            "output": str(output),
        }
        for p in parents
        for j, q in enumerate(p["snapshots"])
        for pair in range(options["child_pairs"])
    ]


def _aggregate(children, pmf, config, output, options):
    is_surface = config.get("task") == "surface_rate"
    task_name = "surface_rate" if is_surface else "rate"
    unit_key = "s_inv" if is_surface else "cm3_s"
    qtst_key = "k_qtst_" + unit_key
    private_pmf = ("pmf_bootstrap_barriers", "pmf_bootstrap_qtst")
    if not children:
        pmf_status = pmf.get("status", "insufficient_overlap")
        result = {
            "task": task_name,
            "status": (
                pmf_status if pmf_status != "ready" else "insufficient_recrossing"
            ),
            "converged": False,
            "rate_" + unit_key: None,
            "rate_estimate_" + unit_key: None,
            "rate_standard_error_" + unit_key: None,
            "kappa": None,
            "pmf": {k: v for k, v in pmf.items() if k not in private_pmf},
            "warnings": pmf["warnings"],
        }
        write_json(Path(output) / "summary.json", result)
        return result
    for child in children:
        if "xi_star" in child and child["xi_star"] != pmf.get("xi_star"):
            raise ValueError("recrossing child belongs to another dividing surface")
        if "pmf_method" in child and child["pmf_method"] != pmf.get("method", "ui"):
            raise ValueError("recrossing child belongs to another PMF method")
    if not pmf.get("ready", True) or pmf.get(qtst_key) is None:
        raise ValueError(
            "cannot combine recrossing trajectories with an unsupported PMF"
        )
    rng = np.random.default_rng(
        np.random.SeedSequence([int(config.get("seed", 0)), 503])
    )
    statistics = summarize_recrossing(
        children, rng=rng,
        **{key: options[key] for key in (
            "child_steps", "plateau_fraction", "plateau_tolerance", "bootstrap"
        )},
    )
    keys = statistics["parent_ids"]
    num, den = statistics["parent_numerator"], statistics["parent_denominator"]
    transmission, start = statistics["curve"], statistics["start"]
    estimate, drift = statistics["kappa"], statistics["drift"]
    boots = statistics["bootstrap_kappa"]
    stderr, drift_error = statistics["standard_error"], statistics["drift_standard_error"]
    curve_error, plateau = statistics["curve_standard_error"], statistics["plateau_detected"]
    recrossing_uncertainty = statistics["uncertainty"]
    candidate = pmf[qtst_key] * estimate
    total_boot = []
    barriers = pmf.get("pmf_bootstrap_barriers", [])
    qtst_draws = pmf.get("pmf_bootstrap_qtst", [])
    pmf_draws = qtst_draws if is_surface else barriers
    if is_surface and len(qtst_draws) >= 2 and recrossing_uncertainty["available"]:
        total_boot = [float(rng.choice(qtst_draws)) * kval for kval in boots]
    elif not is_surface and len(barriers) >= 2 and recrossing_uncertainty["available"]:
        for kval in boots:
            barrier = rng.choice(barriers)
            total_boot.append(
                pmf[qtst_key]
                * np.exp(
                    -config["beta"] * (barrier - pmf["free_energy_barrier_hartree"])
                )
                * kval
            )
    error = float(np.std(total_boot, ddof=1)) if len(total_boot) > 1 else None
    supported_draw_error = error
    conditional_error = (
        abs(pmf[qtst_key]) * stderr if stderr is not None else None
    )
    uncertainty_reasons = list(recrossing_uncertainty["reasons"])
    if len(pmf_draws) < 2:
        uncertainty_reasons.append("fewer_than_two_valid_pmf_bootstrap_draws")
    pmf_bootstrap = pmf.get("bootstrap", {})
    pmf_block_status = pmf_bootstrap.get("block_plan", {}).get("status")
    if pmf_block_status in ("limited_blocks", "unavailable"):
        uncertainty_reasons.append("pmf_bootstrap_" + pmf_block_status)
    if pmf_block_status == "unavailable":
        error = None
        supported_draw_error = None
    rejected_pmf_draws = max(0, pmf_bootstrap.get("attempted", 0) - pmf_bootstrap.get("valid", 0))
    if is_surface and rejected_pmf_draws:
        uncertainty_reasons.append("pmf_bootstrap_contains_unsupported_draws")
        # A confidence interval conditional on complete histogram support is
        # not an unconditional joint interval. Preserve it under an explicit
        # conditional label, and withhold the primary uncertainty fields.
        error = None
    times = np.arange(1, len(transmission) + 1) * config["dt_au"]
    _atomic_npz(
        Path(output) / "recrossing.npz",
        time_au=times,
        kappa=transmission,
        standard_error=curve_error,
        parent_numerator=num,
        parent_denominator=den,
        parent_ids=np.asarray(keys),
        bootstrap_kappa=np.asarray(boots),
    )
    np.savetxt(
        Path(output) / "recrossing.csv",
        np.column_stack([times, transmission, curve_error]),
        delimiter=",",
        header="time_au,kappa,standard_error",
        comments="",
    )
    warnings = list(pmf["warnings"])
    if uncertainty_reasons:
        warnings.append(
            ("Complete rate uncertainty is unavailable: " if error is None
             else "Complete rate uncertainty has limited support: ")
            + "; ".join(uncertainty_reasons)
            + ". Recrossing-only conditional uncertainty holds QTST fixed; supported-draw uncertainty is conditional on complete PMF support."
        )
    if not plateau:
        warnings.append(
            "No stable positive recrossing plateau detected in the selected tail; extend child propagation/sampling."
        )
    max_drift = max(c["energy_drift"] for c in children)
    endpoint_errors = [
        c["endpoint_energy_error_hartree_per_bead"]
        for c in children
        if c.get("endpoint_energy_error_hartree_per_bead") is not None
    ]
    endpoint_error = max(endpoint_errors) if endpoint_errors else None
    energy_diagnostics_complete = len(endpoint_errors) == len(children)
    if max_drift > 0.01:
        warnings.append(
            "Some child trajectories have >1% legacy scaled endpoint drift; this energy-zero-dependent field is informational."
        )
    if not energy_diagnostics_complete:
        warnings.append(
            "Some older child checkpoints lack energy-zero-independent errors; rate publication is withheld until those children are recomputed."
        )
    elif endpoint_error * config["beta"] > 0.1:
        warnings.append(
            "Child endpoint energy error exceeds 0.1 kBT per bead; reduce timestep before publishing a rate estimate."
        )
    if len(keys) < 20:
        warnings.append(
            "Fewer than 20 parent configurations; uncertainty and plateau checks are preliminary."
        )
    if estimate > 1 + max(0.05, 3 * (stderr or 0)):
        warnings.append(
            "Estimated transmission exceeds one beyond sampling noise; inspect PMF/surface and equilibration."
        )
    valid = (
        plateau
        and energy_diagnostics_complete
        and endpoint_error * config["beta"] <= 0.1
    )
    public_pmf = {k: v for k, v in pmf.items() if k not in private_pmf}
    result = {
        "task": task_name,
        "status": "plateau_estimate" if valid else "pilot_requires_convergence",
        "rate_" + unit_key: candidate if valid else None,
        "rate_estimate_" + unit_key: candidate,
        "rate_standard_error_" + unit_key: error,
        "rate_percentile_95_ci_" + unit_key: (
            np.quantile(total_boot, [0.025, 0.975]).tolist()
            if error is not None
            else None
        ),
        "conditional_rate_standard_error_" + unit_key: conditional_error,
        "rate_uncertainty": {
            "available": error is not None,
            "pmf_block_status": pmf_block_status,
            "reasons": uncertainty_reasons,
            "pmf_bootstrap_valid": len(pmf_draws),
            "recrossing_bootstrap_valid": len(boots),
            "conditional_quantity": "recrossing uncertainty with the QTST factor held fixed",
            "combination": "independent draws from the PMF and constrained-parent block bootstrap distributions, with the selected dividing surface fixed",
            "recrossing_scope": recrossing_uncertainty["scope"],
            "recrossing_support": recrossing_uncertainty,
            "parent_block_lengths": statistics["parent_block_lengths"],
        },
        "kappa": estimate,
        "kappa_standard_error": stderr,
        "plateau_detected": plateau,
        "plateau_start_au": float(times[start]),
        "plateau_drift": drift,
        "plateau_drift_standard_error": drift_error,
        "child_time_fs": float(times[-1] * ATOMIC_TIME_S / 1e-15),
        "parent_configurations": len(keys),
        "child_pairs": len(children),
        "child_trajectories": 2 * len(children),
        "maximum_scaled_energy_drift": max_drift,
        "maximum_endpoint_energy_error_hartree_per_bead": endpoint_error,
        "maximum_endpoint_energy_error_kbt": (
            endpoint_error * config["beta"] if endpoint_error is not None else None
        ),
        "energy_diagnostic_child_pairs": len(endpoint_errors),
        "pmf": public_pmf,
        "warnings": warnings,
        "converged": False,
        "convergence_note": (
            "A finite-time plateau does not establish bead, timestep, slab size, basin truncation or sampling convergence."
            if is_surface else "A detected time plateau is not bead, timestep, Rinf or statistical convergence."
        ),
        "files": {
            "pmf": str(Path(output) / ("surface_pmf.csv" if is_surface else "rate_pmf.csv")),
            "recrossing": str(Path(output) / "recrossing.csv"),
        },
    }
    if is_surface:
        result["rate_uncertainty"].update(
            pmf_bootstrap_requested=pmf_bootstrap.get("requested", 0),
            pmf_bootstrap_attempted=pmf_bootstrap.get("attempted", 0),
            pmf_bootstrap_rejected=rejected_pmf_draws,
            independence_assumption="Constrained-parent burn-in removes correlation with the umbrella endpoint used to initialize it; this is not established automatically.",
        )
        if rejected_pmf_draws:
            result["supported_draw_rate_standard_error_s_inv"] = supported_draw_error
            result["supported_draw_rate_percentile_95_ci_s_inv"] = (
                np.quantile(total_boot, [0.025, 0.975]).tolist()
                if supported_draw_error is not None else None
            )
            result["rate_uncertainty"]["supported_draw_condition"] = "conditional on the bootstrap draw having complete basin-to-surface support"
    write_json(Path(output) / "summary.json", result)
    return result


def run_rate(config, system, output, executor=None):
    if any(system.pbc) or system.fixed_atoms:
        raise ValueError("Gas-phase run_rate does not support periodic/fixed-atom systems; use surface_rate")
    config = root_call(_prepare, config, system, str(output))
    options = _recrossing_options(config)
    sampling_config = config
    if rate_pmf_options(config)["method"] == "mbar":
        sampling_config = copy.deepcopy(config)
        # The rate backend performs the one paired endpoint bootstrap. The
        # intermediate FES only saves samples, overlap and a preliminary curve.
        sampling_config["analysis"]["bootstrap"] = 0
    sampled = run_fes(
        sampling_config,
        system,
        Path(output) / "umbrella",
        executor=executor,
        return_data=True,
    )
    pmf, parent_tasks = root_call(
        _pmf_and_parent_tasks, sampled, config, system, str(output), options
    )

    def dispatch(function, tasks):
        return (
            executor.map(function, tasks)
            if executor is not None
            else run_tasks(function, tasks, config.get("parallel", {}))
        )

    if not parent_tasks:
        return root_call(_aggregate, [], pmf, config, str(output), options)
    parents = dispatch(_parent_task, parent_tasks)
    tasks = root_call(_child_tasks, parents, config, system, str(output), options, pmf)
    children = dispatch(_child_task, tasks)
    return root_call(_aggregate, children, pmf, config, str(output), options)
