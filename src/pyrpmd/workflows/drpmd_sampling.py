"""Collision sampling and persistent thermal or independent parent workers.

Worker services are supplied by the public workflow entry points. This keeps
worker functions picklable and preserves explicit factory/persistence hooks
without depending on the coordinator module or mutating module globals.

Collision assembly, centroid connectivity and flux estimators are adapted from
chen-yu-hao/RPMDrate, commit 9a44d6b378d8bf835d67e1b3b7f78b27bdc84a78,
rpmdrate/drpmd.py (MIT). Original RPMDrate copyright (c) 2012 Joshua W. Allen,
William H. Green and Yury V. Suleimanov; see THIRD_PARTY_NOTICES.md and LICENSE.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
import json
import math
from pathlib import Path
import time

import numpy as np

from pyrpmd.dynamics import State
from pyrpmd.parallel import execution_plan
from pyrpmd.propagation import loop_options
from pyrpmd.validation import checked_integer as _integer, finite_positive
from .drpmd_persistence import _load_npz, _mapping_arrays, _restore_rng, _rng_state

KB = 3.1668115634556e-6
_positive = partial(finite_positive, allow_bool=True)


@dataclass(frozen=True)
class WorkerServices:
    """Factories and callbacks resolved once at each worker entry point."""

    dynamics: Callable
    potential: Callable
    classifier: Callable
    launch: Callable
    trajectory: Callable
    save_arrays: Callable


def _field(system, name):
    return system[name] if isinstance(system, dict) else getattr(system, name)


def validate_fragments(fragments, natoms):
    if len(fragments) != 2 or any(len(f) == 0 for f in fragments):
        raise ValueError("dRPMD requires exactly two nonempty reactant fragments")
    flat = [i for f in fragments for i in f]
    if any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in flat):
        raise ValueError("Fragment indices must be 0-based integers")
    if sorted(flat) != list(range(natoms)):
        raise ValueError("Fragments must partition all atoms exactly once")
    return tuple(tuple(int(i) for i in f) for f in fragments)


def fragment_com(q, masses, fragment):
    ids = list(fragment)
    return (
        np.sum(np.asarray(q).mean(axis=0)[ids] * masses[ids, None], axis=0)
        / masses[ids].sum()
    )


def fragment_velocity(p, masses, fragment):
    ids = list(fragment)
    return np.asarray(p).mean(axis=0)[ids].sum(axis=0) / masses[ids].sum()


def _rotation(rng):
    quaternion = rng.normal(size=4)
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def sample_collision(
    beta, reduced_mass, bmax, separation, rng, sampling="thermal_flux", energy=None
):
    beta = _positive(beta, "beta")
    mu = _positive(reduced_mass, "reduced_mass")
    bmax = _positive(bmax, "bmax")
    separation = _positive(separation, "separation")
    if bmax >= separation:
        raise ValueError("Initial radial separation R0 must be greater than bmax")
    if sampling == "thermal_flux":
        energy = float(rng.gamma(2.0, 1.0 / beta))
    elif sampling == "maxwell":
        energy = float(rng.gamma(1.5, 1.0 / beta))
    elif sampling == "fixed_energy":
        energy = _positive(energy, "collision.energy_au")
    else:
        raise ValueError(
            "collision.sampling must be thermal_flux, maxwell or fixed_energy"
        )
    impact = float(bmax * math.sqrt(rng.random()))
    return {
        "sampling": sampling,
        "collision_energy_au": energy,
        "speed_au": math.sqrt(2.0 * energy / mu),
        "impact_parameter_au": impact,
        "azimuth_rad": float(2 * math.pi * rng.random()),
        "area_au": math.pi * bmax * bmax,
    }


def launch_collision(
    parent,
    masses,
    fragments,
    beta,
    bmax,
    separation,
    rng,
    sampling="thermal_flux",
    energy=None,
    orientation="isotropic",
):
    """Preserve thermal internal modes and replace only centroid COM motion."""
    masses = np.asarray(masses, dtype=float)
    fragments = validate_fragments(fragments, len(masses))
    if orientation not in ("isotropic", "fixed"):
        raise ValueError("collision.orientation must be isotropic or fixed")
    fm = np.array([masses[list(f)].sum() for f in fragments])
    total = fm.sum()
    mu = fm.prod() / total
    event = sample_collision(beta, mu, bmax, separation, rng, sampling, energy)
    q, p = parent.q.copy(), parent.p.copy()
    b, phi, v = event["impact_parameter_au"], event["azimuth_rad"], event["speed_au"]
    separation_vector = np.array(
        [
            b * math.cos(phi),
            b * math.sin(phi),
            math.sqrt(separation * separation - b * b),
        ]
    )
    centers = (-fm[1] / total * separation_vector, fm[0] / total * separation_vector)
    velocities = (
        np.array([0.0, 0.0, fm[1] / total * v]),
        np.array([0.0, 0.0, -fm[0] / total * v]),
    )
    for fragment, center, velocity in zip(fragments, centers, velocities):
        ids = list(fragment)
        local_q = q[:, ids] - fragment_com(q, masses, fragment)[None, None, :]
        local_p = (
            p[:, ids]
            - masses[None, ids, None]
            * fragment_velocity(p, masses, fragment)[None, None, :]
        )
        if orientation == "isotropic":
            rotation = _rotation(rng)
            local_q = local_q @ rotation.T
            local_p = local_p @ rotation.T
        q[:, ids] = local_q + center[None, None, :]
        p[:, ids] = local_p + masses[None, ids, None] * velocity[None, None, :]
    return State(q, p), event


def _production_dynamics(config, system, potential, parent_id, services):
    """Use identical propagation settings for inline and queued collisions."""
    masses = np.asarray(_field(system, "masses"), dtype=float)
    options = loop_options(config)
    native = config.get("native", config.get("dynamics", {}).get("native", True))
    threads = execution_plan(config.get("parallel")).threads
    na = config.get("nonadiabatic")
    if na:
        from pyrpmd.nonadiabatic import NRPMDDynamics

        return NRPMDDynamics(potential, masses, config["beta"], config["beads"],
                             config["dt_au"], na["nstates"], native=native,
                             threads=threads, integrator=options["integrator"])
    return services.dynamics(
        potential, masses, config["beta"], config["beads"], config["dt_au"],
        seed=np.random.SeedSequence([int(config.get("seed", 0)), 3, parent_id]),
        native=native, threads=threads, **options,
    )


def _saved_collision(task, production, classifier, services):
    """Replay one already sampled launch without advancing its thermal chain."""
    config, system = task["config"], task["system"]
    trajectory_id, parent_id = task["trajectory_id"], task["parent_id"]
    out = Path(task["output"])
    meta, arrays = _load_npz(out / "states" / f"initial-{trajectory_id:08d}.npz",
                             expected_kind="PyRPMD.drpmd.launch.v1")
    if (meta.get("physics_hash") != task["physics_hash"]
            or meta.get("trajectory_id") != trajectory_id
            or meta.get("parent_id") != parent_id):
        raise ValueError("Saved collision launch does not match this task")
    expected_shape = (config["beads"], len(_field(system, "masses")), 3)
    if any(arrays[key].shape != expected_shape or not np.isfinite(arrays[key]).all()
           for key in ("q", "p")):
        raise ValueError("Saved collision launch has invalid nuclear state")
    na = config.get("nonadiabatic")
    if na:
        from pyrpmd.nonadiabatic import NRPMDState

        for key in ("mapping_q", "mapping_p"):
            if (key not in arrays or arrays[key].shape != (config["beads"], na["nstates"])
                    or not np.isfinite(arrays[key]).all()):
                raise ValueError("Saved collision launch has invalid electronic state")
        child = NRPMDState(arrays["q"], arrays["p"], 0, 0.0,
                           arrays["mapping_q"], arrays["mapping_p"])
    else:
        child = State(arrays["q"], arrays["p"])
    record = services.trajectory(
        production, child, classifier, config, out, trajectory_id, parent_id,
        meta["launch"], task["physics_hash"],
        bool(config.get("resume", config.get("_resume", False))),
    )
    record["parent_sample_step"] = meta["parent_step"]
    return record


def _collision_task(task, services):
    """Independent collision unit for the dynamic MPI work queue."""
    potential = services.potential(task["config"]["potential"], task["system"])
    try:
        production = _production_dynamics(task["config"], task["system"],
                                         potential, task["parent_id"], services)
        return _saved_collision(task, production,
                                services.classifier(task["config"], task["system"]), services)
    finally:
        if hasattr(potential, "close"):
            potential.close()


def _parent_task(task, services):
    config, system = task["config"], task["system"]
    parent_id, ids, physics_hash = (
        task["parent_id"],
        task["trajectory_ids"],
        task["physics_hash"],
    )
    out = Path(task["output"])
    sample_only = bool(task.get("sample_only", False))
    masses = np.asarray(_field(system, "masses"), dtype=float)
    fragments = validate_fragments(_field(system, "fragments"), len(masses))
    beads, beta, dt = config["beads"], config["beta"], config["dt_au"]
    s = config.get("sampling", {})
    thermal_dt = float(s.get("dt_au", dt))
    equil = _integer(
        s.get("equilibration_steps", 1000), "sampling.equilibration_steps", 0
    )
    interval = _integer(
        s.get("interval_steps", s.get("sample_interval", 100)),
        "sampling.interval_steps",
    )
    andersen = _integer(s.get("andersen_interval", 10), "sampling.andersen_interval")
    options = loop_options(config)
    method = options["integrator"]
    if s.get("thermostat", "andersen") != "andersen" or s.get(
        "integrator", method + "-andersen"
    ) not in (method + "-andersen", method):
        raise ValueError(
            f"dRPMD parent sampling requires {method}-andersen with Andersen thermostat; "
            "sampling and production must use the same selected integrator"
        )
    potential = services.potential(config["potential"], system)
    plan = execution_plan(config.get("parallel"))
    seed = int(config.get("seed", 0))
    native = config.get("native", config.get("dynamics", {}).get("native", True))
    parent_seed = np.random.SeedSequence([seed, 1, parent_id])
    na = config.get("nonadiabatic")
    if na:
        from pyrpmd.nonadiabatic import DiagonalPotential, NRPMDState
        thermal_potential = DiagonalPotential(potential, na["initial_state"], na["nstates"])
    else:
        thermal_potential = potential
    dynamics = services.dynamics(
        thermal_potential,
        masses,
        beta,
        beads,
        thermal_dt,
        seed=parent_seed,
        native=native,
        threads=plan.threads,
        **options,
    )
    production = None if sample_only else _production_dynamics(config, system, potential, parent_id, services)
    classifier = services.classifier(config, system)
    parent_path = out / "checkpoints" / f"parent-{parent_id:06d}.npz"
    resume = bool(config.get("resume", config.get("_resume", False)))
    geometry = np.asarray(_field(system, "geometry"), dtype=float)
    parent = State(np.repeat(geometry[None], beads, axis=0), dynamics.sample_momenta())
    burn_done, next_index, pending, records = 0, 0, None, []
    if parent_path.exists():
        if not resume:
            raise FileExistsError(
                f"Parent checkpoint exists: {parent_path}; use resume explicitly"
            )
        meta, arrays = _load_npz(
            parent_path, expected_kind="PyRPMD.drpmd.parent-checkpoint.v1"
        )
        if (
            meta.get("physics_hash") != physics_hash
            or meta.get("trajectory_ids") != ids
        ):
            raise ValueError(
                "Parent checkpoint does not match this job or parent allocation"
            )
        parent = State(arrays["q"], arrays["p"], meta["step"], meta["time"])
        _restore_rng(dynamics, meta["rng"])
        burn_done, next_index, pending, records = (
            meta["burn_done"],
            meta["next_index"],
            meta["pending"],
            meta["records"],
        )
    checkpoint_steps = _integer(
        config.get("output", {}).get("checkpoint_steps", 1000), "checkpoint_steps"
    )
    thermal_seconds = 0.0

    def save_parent():
        services.save_arrays(
            parent_path,
            {
                "format": "PyRPMD.drpmd.parent-checkpoint.v1",
                "physics_hash": physics_hash,
                "parent_id": parent_id,
                "trajectory_ids": ids,
                "step": parent.step,
                "time": parent.time,
                "rng": _rng_state(dynamics),
                "burn_done": burn_done,
                "next_index": next_index,
                "pending": pending,
                "records": records,
                "hamiltonian_convention": "sum_beads_beta_over_P",
            },
            q=parent.q,
            p=parent.p,
            masses=masses,
        )

    try:
        if not sample_only:
            # Dynamic runs checkpoint the sampling clock before their queued
            # children finish. A later static resume must replay those launches
            # without resampling, including after a partially finished queue.
            complete = {record["trajectory_id"] for record in records}
            for trajectory_id in ids[:next_index]:
                if trajectory_id not in complete:
                    record = _saved_collision(dict(task, trajectory_id=trajectory_id),
                                              production, classifier, services)
                    records.append(record)
                    save_parent()
        while burn_done < equil:
            count = min(checkpoint_steps, equil - burn_done)
            start = time.perf_counter()
            dynamics.thermalize(
                parent, count, interval=andersen, fragments=fragments, resample=False
            )
            thermal_seconds += time.perf_counter() - start
            burn_done += count
            save_parent()
        for index in range(next_index, len(ids)):
            trajectory_id = ids[index]
            if pending != trajectory_id:
                start = time.perf_counter()
                dynamics.thermalize(
                    parent,
                    interval,
                    interval=andersen,
                    fragments=fragments,
                    resample=False,
                )
                thermal_seconds += time.perf_counter() - start
                if classifier is not None and hasattr(classifier, "validate_reactants"):
                    classifier.validate_reactants(parent.q)
                pending = trajectory_id
                save_parent()
            launch_rng = np.random.default_rng(
                np.random.SeedSequence([seed, 2, trajectory_id])
            )
            child, launch = services.launch(
                parent,
                masses,
                fragments,
                beta,
                config["collision"]["bmax_au"],
                config["system"]["separation_au"],
                launch_rng,
                sampling=config["collision"].get("sampling", "thermal_flux"),
                energy=config["collision"].get("energy_au"),
                orientation=config["collision"].get("orientation", "isotropic"),
            )
            if na:
                mapping_rng = np.random.default_rng(np.random.SeedSequence([seed, 4, trajectory_id]))
                child = NRPMDState.from_nuclear(child, na["nstates"], na["initial_state"], mapping_rng)
            services.save_arrays(
                out / "states" / f"initial-{trajectory_id:08d}.npz",
                {
                    "format": "PyRPMD.drpmd.launch.v1",
                    "physics_hash": physics_hash,
                    "trajectory_id": trajectory_id,
                    "parent_id": parent_id,
                    "parent_step": parent.step,
                    "parent_time_au": parent.time,
                    "launch": launch,
                    "beta": beta,
                    "beads": beads,
                    "fragments": fragments,
                    "symbols": _field(system, "symbols"),
                },
                thermal_q=parent.q,
                thermal_p=parent.p,
                q=child.q,
                p=child.p,
                **_mapping_arrays(child),
                masses=masses,
            )
            if not sample_only:
                record = services.trajectory(
                    production,
                    child,
                    classifier,
                    config,
                    out,
                    trajectory_id,
                    parent_id,
                    launch,
                    physics_hash,
                    resume,
                )
                record["parent_sample_step"] = parent.step
                records.append(record)
            next_index, pending = index + 1, None
            save_parent()
        return {
            "parent_id": parent_id,
            "records": records,
            "thermal_steps": parent.step,
            "thermal_seconds": thermal_seconds,
        }
    finally:
        if hasattr(potential, "close"):
            potential.close()


def _initial_ensemble_task(task, services):
    """Prepare independent nuclear states and reuse one PES for a work group.

    Work groups share a model instance, not a thermal parent trajectory. Every
    complete initial condition has RNG streams indexed by global trajectory ID.
    A restart reads the saved launch and checkpoint without drawing a new state.
    """
    from pyrpmd.initial_states import prepare_initial_ensemble

    config, system = task["config"], task["system"]
    out, group = Path(task["output"]), task["parent_id"]
    physics_hash = task["physics_hash"]
    resume = bool(config.get("resume", config.get("_resume", False)))
    masses = np.asarray(_field(system, "masses"), dtype=float)
    fragments = validate_fragments(_field(system, "fragments"), len(masses))
    beads, beta = config["beads"], config["beta"]
    ensemble = prepare_initial_ensemble(config["initial_ensemble"], system, beads, beta)
    labels = {"method": ensemble.method,
              "vibrational_quanta": ensemble.vibrational_quanta, "rotation": "j0"}
    plan = execution_plan(config.get("parallel"))
    seed = int(config.get("seed", 0))
    potential = services.potential(config["potential"], system)
    records = []
    try:
        production = services.dynamics(
            potential, masses, beta, beads, config["dt_au"],
            seed=np.random.SeedSequence([seed, 3, group]),
            native=config.get("native", config.get("dynamics", {}).get("native", True)),
            threads=plan.threads, **loop_options(config),
        )
        classifier = services.classifier(config, system)
        for trajectory_id in task["trajectory_ids"]:
            result_path = out / "trajectories" / f"trajectory-{trajectory_id:08d}.json"
            initial_path = out / "states" / f"initial-{trajectory_id:08d}.npz"
            checkpoint = result_path.with_suffix(".checkpoint.npz")
            if result_path.exists():
                if not resume:
                    raise FileExistsError(f"Trajectory already exists: {result_path}; use resume explicitly")
                record = json.loads(result_path.read_text())
                if (record.get("physics_hash") != physics_hash
                        or record.get("trajectory_id") != trajectory_id
                        or record.get("initial_state") != labels):
                    raise ValueError("Completed state-selected trajectory does not match this restart")
                records.append(record)
                continue
            if resume and initial_path.exists():
                metadata, arrays = _load_npz(
                    initial_path, expected_kind="PyRPMD.drpmd.launch.v1")
                if (metadata.get("physics_hash") != physics_hash
                        or metadata.get("trajectory_id") != trajectory_id
                        or metadata.get("initial_state") != labels):
                    raise ValueError("Saved state-selected launch does not match this restart")
                child, launch = State(arrays["q"], arrays["p"]), metadata["launch"]
                # Keep the original execution-group label after rescheduling.
                parent_id = metadata["parent_id"]
            else:
                if checkpoint.exists():
                    raise ValueError("State-selected trajectory checkpoint has no saved launch")
                if initial_path.exists():
                    raise FileExistsError(f"Initial state already exists: {initial_path}; use resume explicitly")
                prepared = ensemble.sample(np.random.default_rng(
                    np.random.SeedSequence([seed, 5, trajectory_id])))
                child, launch = services.launch(
                    prepared, masses, fragments, beta,
                    config["collision"]["bmax_au"], config["system"]["separation_au"],
                    np.random.default_rng(np.random.SeedSequence([seed, 2, trajectory_id])),
                    sampling="fixed_energy", energy=config["collision"]["energy_au"],
                    orientation=config["collision"].get("orientation", "isotropic"),
                )
                parent_id = group
                launch["initial_state"] = labels
                launch["initial_diagnostics"] = ensemble.diagnostics(child)
                if classifier is not None and hasattr(classifier, "validate_reactants"):
                    try:
                        classifier.validate_reactants(child.q)
                    except Exception as exc:
                        # Retain invalid preparations as failed launches. Rejecting
                        # and redrawing would change the ensemble and denominator.
                        launch["initial_validation_error"] = f"{type(exc).__name__}: {exc}"
                services.save_arrays(initial_path, {
                    "format": "PyRPMD.drpmd.launch.v1", "physics_hash": physics_hash,
                    "trajectory_id": trajectory_id, "parent_id": parent_id,
                    "initial_state": labels, "initialization_mode": "independent",
                    "rng_streams": {"state": [seed, 5, trajectory_id],
                                    "collision": [seed, 2, trajectory_id]},
                    "launch": launch, "beta": beta, "beads": beads,
                    "fragments": fragments, "symbols": _field(system, "symbols"),
                }, prepared_q=prepared.q, prepared_p=prepared.p,
                    q=child.q, p=child.p, masses=masses)
            records.append(services.trajectory(
                production, child, classifier, config, out, trajectory_id,
                parent_id, launch, physics_hash, resume,
            ))
        return {"parent_id": group, "records": records,
                "thermal_steps": 0, "thermal_seconds": 0.0,
                "initial_ensemble": ensemble.metadata}
    finally:
        if hasattr(potential, "close"):
            potential.close()
