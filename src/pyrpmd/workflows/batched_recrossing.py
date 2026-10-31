"""Opt-in PES batches of independent, paired-momentum recrossing children.

Checkpoint identities, RNG streams, and signed estimators match rate._child_task.
This module batches the propagation only; parent-chain sampling is unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..batched import BatchedDynamics, BatchedState
from ..dynamics import Dynamics
from ..parallel import task_potential
from ..propagation import loop_options
from . import rate


def _checkpoint(task):
    return (Path(task["output"]) / "checkpoints"
            / f"child_{task['chain']:04d}_{task['parent']:05d}_{task['pair']:04d}.npz")


def _child_batch_task(tasks):
    """Return the same ordered list of pair results as scalar child tasks.

    Each task retains its own ``SeedSequence([seed,307,chain,parent,pair])`` and
    checkpoint.  Reordering/repacking tasks therefore changes no momentum draw.
    Completed checkpoints are read with the original scalar validator.  Batch
    size is an execution setting, excluded from scientific fingerprints.
    """
    tasks = tuple(tasks)
    if not tasks:
        return []
    config, system = tasks[0]["config"], tasks[0]["system"]
    options = loop_options(config)
    if options["loop_backend"] == "native":
        raise ValueError(
            "child_batch_pairs>1 uses Python PES orchestration with a vectorized ring kernel; "
            "set dynamics.loop_backend='auto' or 'python', or use scalar children"
        )
    dynamics_name, _ = rate._recrossing_dynamics(config)
    if dynamics_name != "rpmd":
        raise ValueError("batched recrossing currently supports RPMD children only")
    common = rate._fingerprint(config, system, {
        "xi": tasks[0]["xi_star"], "child_steps": tasks[0]["child_steps"],
    })
    keys = set()
    for task in tasks:
        key = (str(_checkpoint(task)), task["chain"], task["parent"], task["pair"])
        if key in keys:
            raise ValueError("batched recrossing contains a duplicate child pair")
        keys.add(key)
        candidate = rate._fingerprint(task["config"], task["system"], {
            "xi": task["xi_star"], "child_steps": task["child_steps"],
        })
        if candidate != common or loop_options(task["config"]) != options:
            raise ValueError("batched child pairs must share the system, PES, timestep, beads and surface")
    results = [None] * len(tasks)
    pending = []
    for i, task in enumerate(tasks):
        if _checkpoint(task).exists():
            results[i] = rate._child_task(task)
        else:
            pending.append((i, task))
    if not pending:
        return results

    surface = rate.rate_coordinate(config, system).dividing_surface(tasks[0]["xi_star"])
    method = rate._pmf_method(config)
    nsteps = rate.checked_integer(tasks[0]["child_steps"], "batched child_steps")
    block_steps = rate.checked_integer(options["block_steps"], "dynamics.block_steps")
    reuse = config.get("parallel", {}).get("reuse_potentials", True)
    with task_potential(config["potential"], system, reuse=reuse) as provider:
        batch = BatchedDynamics(
            provider, system.masses, config["beta"], config["beads"], config["dt_au"],
            native=config.get("native", config.get("dynamics", {}).get("native", True)),
            threads=config.get("parallel", {}).get("threads_per_rank", 1),
            integrator=options["integrator"], fixed_atoms=system.fixed_atoms,
            fixed_positions=system.geometry,
        )
        positions, momenta, weights, denominators = [], [], [], []
        identities = []
        for _, task in pending:
            chain, parent, pair = task["chain"], task["parent"], task["pair"]
            # The scalar constructor owns the established Maxwell convention
            # and fixed-atom momentum projection. No batching-specific RNG is
            # introduced, and each pair is reproducible independent of packing.
            scalar = Dynamics(
                provider, system.masses, config["beta"], config["beads"], config["dt_au"],
                seed=np.random.SeedSequence([int(config.get("seed", 0)), 307, chain, parent, pair]),
                native=False, fixed_atoms=system.fixed_atoms, fixed_positions=system.geometry,
                integrator=options["integrator"],
            )
            q = np.asarray(task["q"])
            p = scalar.sample_momenta()
            scalar.integrator._check(np.ascontiguousarray(q, dtype=float), p)
            scalar._check_fixed(q, p)
            constraint, gradient, metric = rate._normal(surface, q, system.masses)
            tolerance = float(rate._recrossing_spec(config).get("constraint_tolerance", 1e-10))
            if abs(constraint) > max(1e-8, 10*tolerance):
                raise ValueError("child starting configuration is not on the requested dividing surface")
            flux_scale = np.sqrt(metric / (2*np.pi*config["beta"]))
            velocity = float(np.sum(gradient*p.mean(axis=0)/system.masses[:, None]))
            positions.extend((q, q))
            momenta.extend((p, -p))
            weights.append(velocity/flux_scale)
            denominators.append(abs(velocity)/flux_scale)
            identities.append(rate._fingerprint(config, system, {
                "stage": "child", "chain": chain, "parent": parent, "pair": pair,
                "xi": task["xi_star"], "child_steps": nsteps, "q": task["q"],
            }))
        state = BatchedState(np.stack(positions), np.stack(momenta))
        numerator = np.empty((len(pending), nsteps))
        initial = final = None
        completed = 0
        calls_before = batch.pes_calls
        while completed < nsteps:
            count = min(block_steps, nsteps-completed)
            block = batch.propagate_block(state, count, observe=surface, record_centroids=False)
            if initial is None:
                initial = block.initial_energy.copy()
            final = block.energies[-1]
            side = block.values[:, :, 0] > 0
            for j, weight in enumerate(weights):
                numerator[j, completed:completed+count] = weight*(
                    side[:, 2*j].astype(float) - side[:, 2*j+1].astype(float))
            completed += count
        shared_calls = batch.pes_calls-calls_before
        for j, (result_index, task) in enumerate(pending):
            selection = slice(2*j, 2*j+2)
            error = np.abs(final[selection]-initial[selection])
            endpoint_error_per_bead = float(np.max(error)/config["beads"])
            drift = float(np.max(error/np.maximum(np.abs(initial[selection]),
                                                  config["beads"]/config["beta"])))
            propagation = {
                "native_steps": 0,
                "python_steps": 2*nsteps,
                "fallback_reasons": {},
                "batch_backend": "native-ring/batched-pes" if batch.native else "numpy-ring/batched-pes",
                "batched_trajectory_steps": 2*nsteps,
                "pairs_in_shared_batch": len(pending),
                "shared_batch_pes_calls": shared_calls,
                "shared_call_accounting": "shared across these pairs; do not sum per-pair copies",
            }
            result = {
                "chain": task["chain"], "parent": task["parent"], "pair": task["pair"],
                "xi_star": task["xi_star"], "pmf_method": method,
                "numerator": numerator[j], "denominator": denominators[j],
                "energy_drift": drift,
                "endpoint_energy_error_hartree_per_bead": endpoint_error_per_bead,
                "endpoint_energy_error_kbt": endpoint_error_per_bead*config["beta"],
                "propagation": propagation,
            }
            rate._atomic_npz(
                _checkpoint(task), numerator=result["numerator"], denominator=result["denominator"],
                energy_drift=result["energy_drift"],
                endpoint_energy_error_hartree_per_bead=result["endpoint_energy_error_hartree_per_bead"],
                endpoint_energy_error_kbt=result["endpoint_energy_error_kbt"],
                fingerprint=identities[j], xi_star=task["xi_star"], pmf_method=method,
                propagation=json.dumps(propagation),
            )
            results[result_index] = result
    return results
