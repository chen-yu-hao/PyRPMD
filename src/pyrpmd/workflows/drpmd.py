"""Direct-collision RPMD with thermal or independently prepared initial states.

This module keeps the established import paths and process/MPI worker entry
points. Sampling, propagation, outcome classification, restart conventions and
analysis live in focused modules; the coordinator only allocates and dispatches
work. Persistence and factory hooks are passed explicitly into worker kernels.
The array layout is [bead, atom, xyz], with physical forces equal to -grad(V).

Collision assembly, centroid connectivity and flux estimators are adapted from
chen-yu-hao/RPMDrate, commit 9a44d6b378d8bf835d67e1b3b7f78b27bdc84a78,
rpmdrate/drpmd.py (MIT). Original RPMDrate copyright (c) 2012 Joshua W. Allen,
William H. Green and Yury V. Suleimanov; see THIRD_PARTY_NOTICES.md and LICENSE.
"""

from __future__ import annotations

import copy
from functools import partial
from pathlib import Path

from pyrpmd.checkpoint import save_metadata_npz as _atomic_npz
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.io import jsonable, write_json
from pyrpmd.parallel import execution_plan, make_task_potential, run_tasks, root_call
from pyrpmd.validation import checked_integer as _integer
from . import drpmd_analysis, drpmd_sampling, drpmd_trajectory
from .drpmd_analysis import RATE_AU_TO_CM3_S, _electronic_summary, _wilson, summarize
from .drpmd_classification import (
    ConnectivityClassifier,
    PilotConnectivityClassifier,
    _CONNECTIVITY_CLASSIFIER_METHODS,
    _PILOT_CLASSIFIER_METHODS,
    _STATUSES,
    _bond_components,
    _canonical_formula,
    _centroid_bonds,
    _connectivity_pair_data,
    _formula_counts,
    _make_classifier,
    _normalize_label,
    _original_classifier_methods,
    formula,
    resolve_pilot_outcomes,
)
from .drpmd_persistence import (
    _checkpoint_clock_matches,
    _dependency_fingerprints,
    _load_npz,
    _mapping_arrays,
    _physics_hash,
    _restore_rng,
    _rng_state,
)
from .drpmd_sampling import (
    KB,
    _field,
    _positive,
    _rotation,
    fragment_com,
    fragment_velocity,
    launch_collision,
    sample_collision,
    validate_fragments,
)

_jsonable = partial(jsonable, dataclasses="asdict", callables=False, to_dict="ignore")
_atomic_json = partial(
    write_json,
    sort_keys=True,
    ensure_ascii=True,
    trailing_newline=False,
    dataclasses="asdict",
    callables=False,
    to_dict="ignore",
)

__all__ = [
    "KB", "RATE_AU_TO_CM3_S", "ConnectivityClassifier", "PilotConnectivityClassifier",
    "formula", "fragment_com", "fragment_velocity", "launch_collision",
    "resolve_pilot_outcomes", "run_drpmd", "sample_collision", "summarize",
    "validate_fragments",
]


def _worker_services():
    return drpmd_sampling.WorkerServices(
        dynamics=Dynamics,
        potential=make_task_potential,
        classifier=_make_classifier,
        launch=launch_collision,
        trajectory=_trajectory,
        save_arrays=_atomic_npz,
    )


def _trajectory(
    dynamics, state, classifier, config, out, trajectory_id, parent_id,
    launch, physics_hash, resume, *, extend_unresolved=False,
):
    if config.get("nonadiabatic"):
        if extend_unresolved:
            raise ValueError("Controlled extension is not implemented for NRPMD mapping states")
        return _nrpmd_trajectory(
            dynamics, state, classifier, config, out, trajectory_id, parent_id,
            launch, physics_hash, resume,
        )
    return drpmd_trajectory._trajectory(
        dynamics, state, classifier, config, out, trajectory_id, parent_id,
        launch, physics_hash, resume, extend_unresolved=extend_unresolved,
        save_arrays=_atomic_npz, write_metadata=_atomic_json,
    )


def _nrpmd_trajectory(
    dynamics, state, classifier, config, out, trajectory_id, parent_id,
    launch, physics_hash, resume,
):
    return drpmd_trajectory._nrpmd_trajectory(
        dynamics, state, classifier, config, out, trajectory_id, parent_id,
        launch, physics_hash, resume,
        save_arrays=_atomic_npz, write_metadata=_atomic_json,
    )


def _production_dynamics(config, system, potential, parent_id):
    return drpmd_sampling._production_dynamics(
        config, system, potential, parent_id, _worker_services(),
    )


def _saved_collision(task, production, classifier):
    return drpmd_sampling._saved_collision(
        task, production, classifier, _worker_services(),
    )


def _collision_task(task):
    return drpmd_sampling._collision_task(task, _worker_services())


def _parent_task(task):
    return drpmd_sampling._parent_task(task, _worker_services())


def _initial_ensemble_task(task):
    return drpmd_sampling._initial_ensemble_task(task, _worker_services())


def _write_summary(results, config, system, output, expected):
    return drpmd_analysis._write_summary(
        results, config, system, output, expected, write_metadata=_atomic_json,
    )


def _prepare_output(output):
    out = Path(output)
    for directory in ("checkpoints", "states", "trajectories", "results"):
        (out / directory).mkdir(parents=True, exist_ok=True)
    return str(out.resolve())


def _deferred_collision_tasks(parent_tasks, sampled):
    complete = {record["trajectory_id"] for parent in sampled for record in parent["records"]}
    return [dict(task, trajectory_id=trajectory_id)
            for task in parent_tasks for trajectory_id in task["trajectory_ids"]
            if trajectory_id not in complete]


def _write_dynamic_summary(sampled, collisions, config, system, output, expected):
    by_parent = {parent["parent_id"]: parent for parent in sampled}
    for record in collisions:
        by_parent[record["parent_id"]]["records"].append(record)
    # The coordinator alone updates ledgers after all child writers have
    # stopped. Before this point, completed trajectory files remain sufficient
    # for recovery even if a queue or the job is interrupted.
    for parent in sampled:
        parent["records"].sort(key=lambda record: record["trajectory_id"])
        path = Path(output) / "checkpoints" / f"parent-{parent['parent_id']:06d}.npz"
        meta, arrays = _load_npz(path, expected_kind="PyRPMD.drpmd.parent-checkpoint.v1")
        actual_ids = [record["trajectory_id"] for record in parent["records"]]
        if actual_ids != meta["trajectory_ids"]:
            raise RuntimeError("Dynamic collision ledger has missing or duplicate global IDs")
        meta["records"] = parent["records"]
        _atomic_npz(path, meta, **arrays)
    return _write_summary(sampled, config, system, output, expected)


def run_drpmd(config, system, output, executor=None):
    """Run N collisions from thermal parents or independent selected states.

    Each parent gets N//n collisions plus one for the first N%n parents. A parent
    equilibrates once, advances by interval_steps before every child, and is
    never overwritten by the departing collision trajectory. All ranks call this
    function in MPI mode; its final summary is broadcast to all ranks.
    """
    boundary = system if isinstance(system, dict) else system.to_dict()
    from ..geometry import validate_cell_pbc
    _, pbc = validate_cell_pbc(boundary.get("cell"), getattr(system, "pbc", boundary.get("pbc")))
    if pbc.any() or boundary.get("fixed_atoms"):
        raise ValueError("Gas-phase dRPMD does not support periodic/fixed-atom systems; use a surface workflow")
    config = copy.deepcopy(config)
    state_selected = bool(config.get("initial_ensemble"))
    if state_selected:
        if config.get("nonadiabatic"):
            raise ValueError("State-selected nuclear ensembles currently require a single adiabatic PES")
        if config["collision"].get("sampling") != "fixed_energy":
            raise ValueError("State-selected scattering requires collision.sampling='fixed_energy'")
        _positive(config["collision"].get("energy_au"), "collision.energy_au")
        from pyrpmd.initial_states import prepare_initial_ensemble
        prepare_initial_ensemble(config["initial_ensemble"], system, config["beads"], config["beta"])
    if ("nonadiabatic" in config
            or config.get("potential", {}).get("provider", config.get("potential", {}).get("type")) == "diabatic_python"):
        from pyrpmd.input import validate_nonadiabatic
        config.setdefault("task", "drpmd")
        validate_nonadiabatic(config, system=system)
        if set(config.get("outcomes", {}).get("channels", {})) & {"total_reaction", "nonreactive"}:
            raise ValueError("NRPMD product channel names total_reaction and nonreactive are reserved")
    config["outcomes"] = resolve_pilot_outcomes(
        system,
        config.get("outcomes"),
        separation_au=config["system"]["separation_au"],
        preset=config.get("preset", "pilot"),
    )
    n = _integer(config["collision"]["trajectories"], "collision.trajectories")
    _positive(config["collision"]["bmax_au"], "collision.bmax_au")
    if config["collision"].get("impact_sampling", "uniform_area") != "uniform_area":
        raise ValueError("Only uniform-area impact sampling is implemented")
    plan = execution_plan(config.get("parallel"))
    default_parents = (
        plan.trajectory_groups if plan.backend == "mpi" else plan.trajectory_workers
    )
    parents = min(
        n,
        _integer(
            config.get("sampling", {}).get("parents", default_parents),
            "sampling.parents",
        ),
    )
    physics_hash = root_call(_physics_hash, config, system, parents)
    directory = root_call(_prepare_output, output)
    tasks, start = [], 0
    for parent_id in range(parents):
        count = n // parents + (parent_id < n % parents)
        tasks.append(
            {
                "config": config,
                "system": system,
                "output": directory,
                "parent_id": parent_id,
                "trajectory_ids": list(range(start, start + count)),
                "physics_hash": physics_hash,
            }
        )
        start += count
    worker = _initial_ensemble_task if state_selected else _parent_task
    # Independent selected states already sample and propagate in one task;
    # only thermal parents need a separate collision queue after sampling.
    dynamic = (plan.backend == "mpi" and plan.scheduling == "dynamic"
               and not state_selected)
    if dynamic:
        for task in tasks:
            task["sample_only"] = True
    results = (
        executor.map(worker, tasks)
        if executor is not None
        else run_tasks(worker, tasks, config.get("parallel"))
    )
    if dynamic:
        collisions = root_call(_deferred_collision_tasks, tasks, results)
        records = (executor.map(_collision_task, collisions) if executor is not None
                   else run_tasks(_collision_task, collisions, config.get("parallel")))
        return root_call(_write_dynamic_summary, results, records, config, system, directory, n)
    return root_call(_write_summary, results, config, system, directory, n)
