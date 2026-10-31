"""Restartable constrained-parent and paired-child recrossing tasks."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from ..checkpoint import atomic_savez as _atomic_npz, load_checkpoint
from ..dynamics import Dynamics, State
from ..parallel import task_potential
from ..propagation import loop_options
from ..io import fingerprint as _fingerprint
from ..validation import checked_integer
from .rate_constraints import rate_coordinate, _normal, project_positions, project_momenta
from .rate_options import _pmf_method, _recrossing_spec, _recrossing_dynamics


def _parent_task(task):
    config, system = task["config"], task["system"]
    options = task["options"]
    surface = rate_coordinate(config, system).dividing_surface(task["xi_star"])
    chain = int(task["chain"])
    method = _pmf_method(config)
    checkpoint = Path(task["output"]) / "checkpoints" / f"parent_{chain:04d}.npz"
    identity = _fingerprint(
        config,
        system,
        {"stage": "parent", "chain": chain, "xi": task["xi_star"], "options": options},
    )
    with task_potential(config["potential"], system,
                        reuse=config.get("parallel", {}).get("reuse_potentials", True)) as provider:
        from .surface_source import source_hamiltonian, parent_selector
        hamiltonian = source_hamiltonian(provider, config, system)
        accept_parent = parent_selector(config, system)
        dyn = Dynamics(
            hamiltonian,
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
                    if not accept_parent(snapshot):
                        raise ValueError("parent checkpoint snapshot lies outside the source PMF channel domain")
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
            snapshots.extend(q for q in block.q if accept_parent(q))
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


def _child_task(task):
    config, system = task["config"], task["system"]
    nsteps = task["child_steps"]
    surface = rate_coordinate(config, system).dividing_surface(task["xi_star"])
    chain, parent, pair = task["chain"], task["parent"], task["pair"]
    method = _pmf_method(config)
    child_dynamics, friction_scale = _recrossing_dynamics(config)
    dynamics_metadata = ({"dynamics": "trpmd", "friction_scale": friction_scale,
                          "energy_conservation_applicable": False}
                         if child_dynamics == "trpmd" else {})
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
    # A complete pair needs only its checked saved result. In particular, an
    # all-cached scalar or batched replay must not initialize a PES or bead pool.
    if checkpoint.exists():
        with load_checkpoint(checkpoint) as saved:
            if str(saved["fingerprint"]) != identity:
                raise ValueError(f"child checkpoint physics differs: {checkpoint}")
            if "xi_star" in saved and float(saved["xi_star"]) != task["xi_star"]:
                raise ValueError("child checkpoint dividing surface differs")
            if "pmf_method" in saved and str(saved["pmf_method"]) != method:
                raise ValueError("child checkpoint PMF method differs")
            return {
                **dynamics_metadata,
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
    with task_potential(config["potential"], system,
                        reuse=config.get("parallel", {}).get("reuse_potentials", True)) as provider:
        from .surface_source import source_hamiltonian, parent_selector
        if not parent_selector(config, system)(task["q"]):
            raise ValueError("child initial configuration lies outside the source PMF channel domain")
        dyn = Dynamics(
            source_hamiltonian(provider, config, system),
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
            if child_dynamics == "rpmd" and dyn.can_fuse(observe=surface):
                completed = 0
                while completed < nsteps:
                    count = min(dyn.block_steps, nsteps-completed)
                    block = dyn.propagate(state, count, observe=surface,
                                          sample_interval=1, record_centroids=False)
                    numerator[completed:completed+count] += weight*(block.values[:, 0] > 0)
                    completed += count
            else:
                for step in range(nsteps):
                    if child_dynamics == "trpmd":
                        dyn.trpmd_step(state, friction_scale=friction_scale)
                    else:
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
            **dynamics_metadata,
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
            **dynamics_metadata,
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


def _dispatch_children(tasks, config, dispatch):
    batch_pairs = checked_integer(_recrossing_spec(config).get("child_batch_pairs", 1),
                                  "recrossing.child_batch_pairs")
    if batch_pairs == 1:
        return dispatch(_child_task, tasks)
    from .batched_recrossing import _child_batch_task
    batches = [tasks[start:start + batch_pairs] for start in range(0, len(tasks), batch_pairs)]
    grouped = dispatch(_child_batch_task, batches)
    return None if grouped is None else [child for batch in grouped for child in batch]
