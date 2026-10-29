"""Bennett--Chandler bimolecular RPMD rate with explicit sampling measures.

The RPMDrate/Caracal flux-metric PMF and paired-child estimator are used,
with a newly implemented centroid Cayley-RATTLE constrained parent sampler.
The rate is k_RPMD = k_QTST * kappa; kappa is always measured, never guessed.
Reference implementations are MIT licensed; see THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np

from ..analysis.fes import (
    umbrella_integration,
    statistical_inefficiency,
    _block_resample,
)
from ..constants import AU_RATE_TO_CM3_S, ATOMIC_TIME_S
from ..cv import RPMDrateCoordinate
from ..dynamics import Dynamics, State
from ..parallel import run_tasks, root_call, make_task_potential
from ..units import parse_quantity, time_steps
from .prepare import prepare_rate
from .fes import run_fes, write_json, _atomic_npz, _fingerprint, checked_integer


def rate_coordinate(config, system):
    reaction = config["reaction"]
    return RPMDrateCoordinate(
        reaction["fragments"],
        system.masses,
        system.geometry,
        reaction["forming_bonds"],
        reaction["breaking_bonds"],
        reaction["rinf"],
    )


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
    q = state.q.copy()
    p = state.p.copy()
    m = dynamics.masses[:, None]
    dt = dynamics.dt
    initial, g0, _ = _normal(surface, q, dynamics.masses)
    if abs(initial) > max(1e-8, 10 * tolerance):
        raise ValueError("parent state is not on the dividing surface")
    dynamics.integrator.kick(p, dynamics._force(q), 0.5 * dt)
    dynamics.integrator.free_step(q, p)
    free = q.copy()
    direction = g0 / m
    lam = 0.0
    for _ in range(max_iterations):
        value, g, _ = _normal(surface, q, dynamics.masses)
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
        or values["plateau_tolerance"] <= 0
    ):
        raise ValueError("invalid recrossing burn-in/plateau options")
    return values


def _parent_task(task):
    config, system = task["config"], task["system"]
    options = task["options"]
    surface = rate_coordinate(config, system).dividing_surface(task["xi_star"])
    chain = int(task["chain"])
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
        )
        snapshots = []
        elapsed = 0
        if checkpoint.exists():
            with np.load(checkpoint, allow_pickle=False) as saved:
                if str(saved["fingerprint"]) != identity:
                    raise ValueError(f"parent checkpoint physics differs: {checkpoint}")
                state = State(
                    saved["q"], saved["p"], int(saved["step"]), float(saved["time"])
                )
                dyn.rng.bit_generator.state = json.loads(str(saved["rng"]))
                snapshots = list(saved["snapshots"])
                elapsed = int(saved["elapsed"])
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
            )

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
            constrained_step(dyn, state, surface, options["constraint_tolerance"])
            elapsed += 1
            production = elapsed - options["equilibration_steps"]
            if production > 0 and production % options["parent_interval_steps"] == 0:
                snapshots.append(state.q.copy())
            if elapsed % checkpoint_steps == 0:
                save()
        save()
        return {
            "chain": chain,
            "snapshots": np.asarray(snapshots),
            "checkpoint": str(checkpoint),
        }
    finally:
        if hasattr(provider, "close"):
            provider.close()


def _child_task(task):
    config, system = task["config"], task["system"]
    nsteps = task["child_steps"]
    surface = rate_coordinate(config, system).dividing_surface(task["xi_star"])
    chain, parent, pair = task["chain"], task["parent"], task["pair"]
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
            with np.load(checkpoint, allow_pickle=False) as saved:
                if str(saved["fingerprint"]) != identity:
                    raise ValueError(f"child checkpoint physics differs: {checkpoint}")
                return {
                    "chain": chain,
                    "parent": parent,
                    "pair": pair,
                    "numerator": saved["numerator"],
                    "denominator": float(saved["denominator"]),
                    "energy_drift": float(saved["energy_drift"]),
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
        )
        q = np.asarray(task["q"])
        p = dyn.sample_momenta()
        _, g, z = _normal(surface, q, system.masses)
        fs = np.sqrt(z / (2 * np.pi * config["beta"]))
        velocity = float(np.sum(g * np.mean(p, axis=0) / system.masses[:, None]))
        numerator = np.zeros(nsteps)
        denominator = abs(velocity) / fs
        drift = 0.0
        # Unconstrained Maxwell momenta are deliberately drawn AFTER constrained
        # configuration sampling. Projecting child momenta would erase flux.
        for sign in (1.0, -1.0):
            state = State(q, sign * p)
            weight = sign * velocity / fs
            initial = dyn.energy(state)
            for step in range(nsteps):
                dyn.step(state)
                value, _ = surface.centroid_value_gradient(state.q.mean(axis=0))
                if value > 0:
                    numerator[step] += weight
            final = dyn.energy(state)
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
            fingerprint=identity,
        )
        return {
            "chain": chain,
            "parent": parent,
            "pair": pair,
            "numerator": numerator,
            "denominator": denominator,
            "energy_drift": drift,
        }
    finally:
        if hasattr(provider, "close"):
            provider.close()


def _prepare(config, system, output):
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
    return result


def _pmf_and_parent_tasks(sampled, config, system, output, options):
    streams = sampled["streams"]
    centers = np.array([s["center"][0] for s in streams])
    samples = [s["samples"][:, 0] for s in streams]
    kappa = np.array([s["stiffness"][0, 0] for s in streams])
    if centers.min() > 0 or centers.max() < 1:
        raise ValueError("rate umbrellas must cover xi=0 (reactants) through xi=1 (TS)")
    ngrid = checked_integer(
        config.get("rate", {}).get("pmf_points", 401), "rate.pmf_points", 20
    )
    grid = np.linspace(max(0.0, centers.min()), centers.max(), ngrid)
    pmf, derivative = umbrella_integration(
        samples, centers, kappa, config["beta"], grid
    )
    raw = _recrossing_spec(config)
    xi = float(
        raw.get(
            "dividing_surface",
            config.get("rate", {}).get("dividing_surface", grid[np.argmax(pmf)]),
        )
    )
    if not grid[0] <= xi <= grid[-1]:
        raise ValueError("dividing surface outside sampled PMF range")
    barrier = float(np.interp(xi, grid, pmf) - np.interp(0.0, grid, pmf))
    fragments = config["reaction"]["fragments"]
    ma, mb = [system.masses[list(f)].sum() for f in fragments]
    reduced = ma * mb / (ma + mb)
    rinf = config["reaction"]["rinf"]
    prefactor = 4 * np.pi * rinf * rinf / np.sqrt(2 * np.pi * config["beta"] * reduced)
    electronic = float(config["reaction"].get("electronic_weight", 1.0))
    symmetry = float(config["reaction"].get("symmetry_factor", 1.0))
    if not np.isfinite(symmetry) or symmetry <= 0:
        raise ValueError("reaction.symmetry_factor must be positive")
    qtst = float(prefactor * np.exp(-config["beta"] * barrier) * electronic * symmetry)
    if not np.isfinite(qtst):
        raise FloatingPointError("nonfinite QTST estimate; inspect PMF sampling")
    gs = np.asarray(sampled["data"]["statistical_inefficiency"])
    boot = []
    nboot = int(config["analysis"].get("bootstrap", 20))
    rng = np.random.default_rng(
        np.random.SeedSequence([int(config.get("seed", 0)), 401])
    )
    for _ in range(nboot):
        draws = [_block_resample(s, int(np.ceil(g)), rng) for s, g in zip(samples, gs)]
        try:
            fb, _ = umbrella_integration(draws, centers, kappa, config["beta"], grid)
        except ValueError:
            continue
        boot.append(float(np.interp(xi, grid, fb) - np.interp(0.0, grid, fb)))
    barrier_error = float(np.std(boot, ddof=1)) if len(boot) > 1 else None
    _atomic_npz(
        Path(output) / "rate_pmf.npz",
        xi=grid,
        free_energy=pmf,
        derivative=derivative,
        bootstrap_barriers=np.asarray(boot),
    )
    np.savetxt(
        Path(output) / "rate_pmf.csv",
        np.column_stack([grid, pmf, derivative]),
        delimiter=",",
        header="xi,free_energy_hartree,derivative_hartree",
        comments="",
    )
    diagnostics = {
        "xi_star": xi,
        "free_energy_barrier_hartree": barrier,
        "barrier_standard_error_hartree": barrier_error,
        "k_qtst_au": qtst,
        "k_qtst_cm3_s": qtst * AU_RATE_TO_CM3_S,
        "reactant_prefactor_au": prefactor,
        "electronic_weight": electronic,
        "symmetry_factor": symmetry,
        "rinf_bohr": rinf,
        "reduced_mass_au": float(reduced),
        "pmf_method": "Gaussian umbrella integration; MBAR overlap checked",
        "measure": "delta(xi-x) * sqrt(grad(xi) M^-1 grad(xi)/(2*pi*beta))",
        "overlap_connected": sampled["data"]["connected"],
        "pmf_bootstrap_barriers": boot,
        "warnings": sampled["summary"]["warnings"],
    }
    if not sampled["data"]["connected"]:
        diagnostics["warnings"].append(
            "QTST value is diagnostic only: disconnected windows do not identify the PMF barrier."
        )
        return diagnostics, []
    nearest = np.argsort(abs(centers - xi))
    tasks = [
        {
            "config": config,
            "system": system,
            "options": options,
            "chain": chain,
            "xi_star": xi,
            "initial_q": streams[nearest[chain % len(nearest)]]["last_q"],
            "output": str(output),
        }
        for chain in range(options["parent_replicas"])
    ]
    return diagnostics, tasks


def _child_tasks(parents, config, system, output, options, pmf):
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
    if not children:
        result = {
            "task": "rate",
            "status": "insufficient_overlap",
            "converged": False,
            "rate_cm3_s": None,
            "rate_estimate_cm3_s": None,
            "kappa": None,
            "pmf": pmf,
            "warnings": pmf["warnings"],
        }
        write_json(Path(output) / "summary.json", result)
        return result
    # A child pair is correlated and all pairs from one parent share its
    # configuration; aggregate by parent before any uncertainty estimate.
    grouped = {}
    for child in children:
        key = (child["chain"], child["parent"])
        if key not in grouped:
            grouped[key] = [np.zeros(options["child_steps"]), 0.0]
        grouped[key][0] += child["numerator"]
        grouped[key][1] += child["denominator"]
    keys = sorted(grouped)
    num = np.array([grouped[k][0] for k in keys])
    den = np.array([grouped[k][1] for k in keys])
    if den.sum() <= 0:
        raise RuntimeError(
            "zero initial reactive flux; no transmission coefficient can be estimated"
        )
    transmission = num.sum(axis=0) / den.sum()
    tail = max(2, int(np.ceil(len(transmission) * options["plateau_fraction"])))
    tail = min(tail, len(transmission))
    start = len(transmission) - tail
    middle = start + tail // 2
    estimate = float(np.mean(transmission[start:]))
    drift = (
        float(np.mean(transmission[middle:]) - np.mean(transmission[start:middle]))
        if middle > start
        else float("inf")
    )
    rng = np.random.default_rng(
        np.random.SeedSequence([int(config.get("seed", 0)), 503])
    )
    boots = []
    boot_curves = []
    chain_indices = {
        c: np.array([i for i, k in enumerate(keys) if k[0] == c]) for c, _ in keys
    }
    blocks = {}
    for chain, ix in chain_indices.items():
        observable = np.divide(
            num[ix, start:].mean(axis=1),
            den[ix],
            out=np.zeros(len(ix)),
            where=den[ix] > 0,
        )
        blocks[chain] = max(1, int(np.ceil(statistical_inefficiency(observable))))
    for _ in range(options["bootstrap"]):
        selected = []
        # Resample chains and contiguous parent blocks to preserve parent-chain
        # correlation. Multiple children from a parent remain one cluster.
        chains = list(chain_indices)
        for chain in rng.choice(chains, size=len(chains), replace=True):
            ix = chain_indices[chain]
            selected.extend(_block_resample(ix, blocks[chain], rng).tolist())
        ix = np.asarray(selected, int)
        if den[ix].sum() <= 0:
            continue
        curve = num[ix].sum(axis=0) / den[ix].sum()
        boot_curves.append(curve)
        boots.append(curve[start:].mean())
    stderr = float(np.std(boots, ddof=1)) if len(boots) > 1 else None
    drift_error = None
    if len(boot_curves) > 1:
        b = np.asarray(boot_curves)
        drift_error = float(
            np.std(b[:, middle:].mean(axis=1) - b[:, start:middle].mean(axis=1), ddof=1)
        )
        curve_error = np.std(b, axis=0, ddof=1)
    else:
        curve_error = np.full_like(transmission, np.nan)
    threshold = max(
        options["plateau_tolerance"] * abs(estimate), 2 * (drift_error or 0.0)
    )
    plateau = bool(tail >= 10 and abs(drift) <= threshold and estimate > 0)
    candidate = pmf["k_qtst_cm3_s"] * estimate
    total_boot = []
    barriers = pmf.get("pmf_bootstrap_barriers", [])
    for kval in boots:
        barrier = (
            rng.choice(barriers) if barriers else pmf["free_energy_barrier_hartree"]
        )
        total_boot.append(
            pmf["k_qtst_cm3_s"]
            * np.exp(-config["beta"] * (barrier - pmf["free_energy_barrier_hartree"]))
            * kval
        )
    error = float(np.std(total_boot, ddof=1)) if len(total_boot) > 1 else None
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
    if not plateau:
        warnings.append(
            "No stable positive recrossing plateau detected in the selected tail; extend child propagation/sampling."
        )
    max_drift = max(c["energy_drift"] for c in children)
    if max_drift > 0.01:
        warnings.append(
            "Some child trajectories have >1% scaled endpoint energy drift; decrease timestep."
        )
    if len(grouped) < 20:
        warnings.append(
            "Fewer than 20 parent configurations; uncertainty and plateau checks are preliminary."
        )
    if estimate > 1 + max(0.05, 3 * (stderr or 0)):
        warnings.append(
            "Estimated transmission exceeds one beyond sampling noise; inspect PMF/surface and equilibration."
        )
    valid = plateau and max_drift <= 0.01
    public_pmf = {k: v for k, v in pmf.items() if k != "pmf_bootstrap_barriers"}
    result = {
        "task": "rate",
        "status": "plateau_estimate" if valid else "pilot_requires_convergence",
        "rate_cm3_s": candidate if valid else None,
        "rate_estimate_cm3_s": candidate,
        "rate_standard_error_cm3_s": error,
        "kappa": estimate,
        "kappa_standard_error": stderr,
        "plateau_detected": plateau,
        "plateau_start_au": float(times[start]),
        "plateau_drift": drift,
        "plateau_drift_standard_error": drift_error,
        "child_time_fs": float(times[-1] * ATOMIC_TIME_S / 1e-15),
        "parent_configurations": len(grouped),
        "child_pairs": len(children),
        "child_trajectories": 2 * len(children),
        "maximum_scaled_energy_drift": max_drift,
        "pmf": public_pmf,
        "warnings": warnings,
        "converged": False,
        "convergence_note": "A detected time plateau is not bead, timestep, Rinf or statistical convergence.",
        "files": {
            "pmf": str(Path(output) / "rate_pmf.csv"),
            "recrossing": str(Path(output) / "recrossing.csv"),
        },
    }
    write_json(Path(output) / "summary.json", result)
    return result


def run_rate(config, system, output, executor=None):
    config = root_call(_prepare, config, system, str(output))
    options = _recrossing_options(config)
    sampled = run_fes(
        config, system, Path(output) / "umbrella", executor=executor, return_data=True
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
