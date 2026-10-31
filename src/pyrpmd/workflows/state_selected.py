"""Sparse NVE checks of prepared vibrational states on the configured full PES.

This is a separated-reactant precollision stability check, not an isolated
fragment calculation: the full PES can retain interfragment interactions.
Reference-mode projection occurs only after propagation blocks return.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import json

import numpy as np

from ..block_propagation import ENERGY_PARTS, state_diagnostics
from ..constants import AU_TIME_FS
from ..dynamics import Dynamics, State
from ..initial_states import prepare_initial_ensemble
from ..input import InputError, JobSpec, load_input
from ..io import jsonable, write_json
from ..potentials import make_potential, validate_result
from ..propagation import loop_options
from ..units import time_steps


_DIAGNOSTICS = {"energy_parts": True, "ring_radii": True}


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise InputError(f"{name} must be a positive integer")
    return int(value)


def _statistics(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError("Nonfinite state-check statistics")
    std = np.std(values, axis=0, ddof=1) if len(values) > 1 else None
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": None if std is None else std.tolist(),
        "sem": None if std is None else (std / np.sqrt(len(values))).tolist(),
        "minimum": np.min(values, axis=0).tolist(),
        "maximum": np.max(values, axis=0).tolist(),
    }


def _rng_snapshot(rng):
    return json.dumps(jsonable(rng.bit_generator.state), sort_keys=True)


def _harmonic_potential(prepared, state):
    """Reference harmonic potential, zero at the supplied equilibrium geometry."""
    value = 0.0
    for reference in prepared.references:
        displacement = state.q[:, list(reference.atoms)] - reference.geometry[None]
        weighted = displacement * np.sqrt(reference.masses)[None, :, None]
        modes = reference.modes.reshape(reference.nmodes, 3 * len(reference.atoms))
        coordinates = weighted.reshape(prepared.beads, -1) @ modes.T
        value += 0.5 * np.mean(np.sum(coordinates ** 2 * reference.frequencies[None] ** 2, axis=1))
    return float(value)


def _frame(prepared, state, bead_diagnostics, hamiltonian, initial_hamiltonian, reference_energy):
    molecular = prepared.diagnostics(state)
    parts = np.asarray(bead_diagnostics["energy_parts"])
    fragments = molecular["fragments"]
    separation = np.linalg.norm(
        np.asarray(fragments[0]["centroid_com_position_au"]) - fragments[1]["centroid_com_position_au"]
    )
    return {
        "step": int(state.step),
        "time_au": float(state.time),
        "time_fs": float(state.time * AU_TIME_FS),
        "hamiltonian_per_bead_au": float(hamiltonian),
        "hamiltonian_per_bead_drift_au": float(hamiltonian - initial_hamiltonian),
        "energy_parts_au": {name: float(value) for name, value in zip(ENERGY_PARTS, parts)},
        "full_pes_primitive_relative_reference_au": float(parts[0] + parts[1] + parts[2] - parts[3] - reference_energy),
        "atom_ring_radii_au": np.asarray(bead_diagnostics["ring_radii"]).tolist(),
        "fragment_com_separation_au": float(separation),
        "fragments": fragments,
    }


def _series(samples):
    """Paired sample statistics at each common observation time."""
    series = []
    for frame_index, first in enumerate(samples[0]["frames"]):
        rows = [sample["frames"][frame_index] for sample in samples]
        if any(row["step"] != first["step"] for row in rows):
            raise RuntimeError("State-check observation grids disagree")
        record = {name: first[name] for name in ("step", "time_au", "time_fs")}
        for name in ("hamiltonian_per_bead_au", "hamiltonian_per_bead_drift_au",
                     "full_pes_primitive_relative_reference_au", "atom_ring_radii_au",
                     "fragment_com_separation_au"):
            record[name] = _statistics([row[name] for row in rows])
        record["energy_parts_au"] = {
            name: _statistics([row["energy_parts_au"][name] for row in rows]) for name in ENERGY_PARTS
        }
        record["fragments"] = []
        for fragment_index, fragment in enumerate(first["fragments"]):
            current = [row["fragments"][fragment_index] for row in rows]
            initial = [sample["frames"][0]["fragments"][fragment_index] for sample in samples]
            fragment_stats = {
                "atoms": fragment["atoms"], "vibrational_quanta": fragment["vibrational_quanta"],
            }
            for name in ("primitive_harmonic_mode_energies_au", "harmonic_mode_actions",
                         "centroid_mode_energies_au", "fragment_com_ring_spread_au"):
                fragment_stats[name] = _statistics([row[name] for row in current])
            fragment_stats["action_change_from_initial"] = _statistics([
                np.asarray(now["harmonic_mode_actions"]) - before["harmonic_mode_actions"]
                for now, before in zip(current, initial)
            ])
            record["fragments"].append(fragment_stats)
        series.append(record)
    return series


def check_state_selected(spec, *, duration="100 fs", samples=8, stride=10, output=None):
    """Run a small precollision state-survival check without a production run.

    spec is an input path, JobSpec or raw input mapping, resolved through the
    ordinary input validator. duration must be an integer number of real-time
    steps; stride is a step count. The first and last frames are always kept.
    Samples execute sequentially with the configured PES, integrator and native
    backend. They receive no collision kick, thermostat, constraint or ongoing
    fragment projection. Dynamics and initialization RNGs are checked unchanged
    during propagation.

    An explicit output path is a JSON report relative to the caller's current
    working directory, not the input or production output directory. Existing
    reports are atomically replaced. The input itself cannot be the output.
    No report is written when output is None.
    """
    samples = _positive_integer(samples, "samples")
    stride = _positive_integer(stride, "stride")
    if isinstance(spec, (str, Path)):
        spec = load_input(spec)
    elif isinstance(spec, Mapping):
        spec = JobSpec.from_mapping(spec)
    if not isinstance(spec, JobSpec):
        raise InputError("state-check requires an input path, JobSpec or input mapping")
    destination = None
    if output is not None:
        destination = Path(output).expanduser().resolve()
        if destination.suffix.lower() != ".json":
            raise InputError("state-check --output must name a JSON file")
        if destination.is_dir():
            raise InputError("state-check --output must be a file, not a directory")
        if spec.input_path is not None and destination == Path(spec.input_path).resolve():
            raise InputError("state-check output cannot replace its input")
    config, system = spec.resolve()
    if config.get("task") != "drpmd" or "initial_ensemble" not in config:
        raise InputError("state-check requires a drpmd input with qct or ne_rpmd initial_ensemble")
    if config.get("nonadiabatic"):
        raise InputError("state-check currently requires a scalar Born-Oppenheimer PES")
    prepared = prepare_initial_ensemble(config["initial_ensemble"], system, config["beads"], config["beta"])
    steps = time_steps(duration, config["dt_au"])
    options = loop_options(config)
    block_steps = min(options["block_steps"], 4096)
    seed = int(config.get("seed", 0))
    potential = make_potential(config["potential"], system)
    try:
        dynamics = Dynamics(
            potential, system.masses, config["beta"], config["beads"], config["dt_au"],
            seed=np.random.default_rng([seed, 9]),
            native=config.get("native", config.get("dynamics", {}).get("native", True)),
            threads=config.get("parallel", {}).get("threads_per_rank", 1), **options,
        )
        dynamics_rng = _rng_snapshot(dynamics.rng)
        reference_q = system.geometry[None].copy()
        reference_result = validate_result(potential.evaluate(reference_q), reference_q.shape)
        reference_energy = float(reference_result.energy[0])
        reference_forces = reference_result.forces[0]
        reference_fragment_forces = [reference_forces[list(fragment)].sum(axis=0).tolist()
                                     for fragment in system.fragments]
        trajectories = []
        for sample_id in range(samples):
            # Same initialization stream as the corresponding production
            # trajectory; the collision-orientation/kick stream is not used.
            sample_rng = np.random.default_rng([seed, 5, sample_id])
            state = prepared.sample(sample_rng)
            initialized_rng = _rng_snapshot(sample_rng)
            bead_values = state_diagnostics(dynamics, state, _DIAGNOSTICS)
            initial_hamiltonian = float(np.sum(bead_values["energy_parts"]))
            real_potential = float(bead_values["energy_parts"][0] - reference_energy)
            harmonic_potential = _harmonic_potential(prepared, state)
            initial_frame = _frame(prepared, state, bead_values, initial_hamiltonian,
                                   initial_hamiltonian, reference_energy)
            trajectory = {
                "sample_id": sample_id, "initialization_seed_sequence": [seed, 5, sample_id],
                "switch_error": {
                    "real_potential_above_reference_au": real_potential,
                    "harmonic_potential_au": harmonic_potential,
                    "real_minus_harmonic_potential_au": real_potential - harmonic_potential,
                },
                "frames": [initial_frame],
            }
            while state.step < steps:
                count = min(block_steps, steps - state.step)
                global_steps = state.step + np.arange(1, count + 1)
                observe = (global_steps % stride == 0) | (global_steps == steps)
                block = dynamics.propagate_block(
                    state, count, observe=observe, record_states=True, record_energy=True,
                    diagnostics=_DIAGNOSTICS, reuse_forces=True, check_energy=True,
                )
                # All molecular projection is outside the fused native block.
                for row, step in enumerate(block.steps):
                    frame_state = State(block.q[row], block.p[row], int(step), float(block.times[row]))
                    values = {name: array[row] for name, array in block.diagnostics.items()}
                    trajectory["frames"].append(_frame(
                        prepared, frame_state, values, block.energies[row] / prepared.beads,
                        initial_hamiltonian, reference_energy,
                    ))
            if _rng_snapshot(sample_rng) != initialized_rng or _rng_snapshot(dynamics.rng) != dynamics_rng:
                raise RuntimeError("NVE state-check unexpectedly changed an initialization or dynamics RNG")
            trajectory["maximum_absolute_observed_hamiltonian_drift_au"] = max(
                abs(frame["hamiltonian_per_bead_drift_au"]) for frame in trajectory["frames"]
            )
            trajectories.append(trajectory)
        report = {
            "schema_version": 1,
            "kind": "state_selected_precollision_stability",
            "scope": "Separated reactants on the full PES; residual interfragment interactions remain. This is not an isolated-fragment calculation.",
            "interpretation": "Mode actions are approximate harmonic diagnostics, not exact pure-state populations or final quantum-state assignments. Inspect time dependence together with observed H_P/P drift.",
            "controls": {
                "samples": samples, "steps": steps, "timestep_au": dynamics.dt,
                "duration_au": steps * dynamics.dt, "duration_fs": steps * dynamics.dt * AU_TIME_FS,
                "stride_steps": stride, "block_steps": block_steps,
                "thermostat": "none", "collision_kick": False, "propagation_rng_unchanged": True,
                "execution": "sequential_samples", "integrator": dynamics.integrator_name,
            },
            "initial_ensemble": prepared.metadata,
            "full_pes_reference": {
                "geometry_au": system.geometry.tolist(),
                "potential_energy_au": reference_energy,
                "maximum_absolute_force_au": float(np.max(np.abs(reference_forces))),
                "net_force_per_fragment_au": reference_fragment_forces,
                "switch_error_definition": "mean_beads[V_full(q)-V_full(reference_geometry)] - mean_beads[V_harmonic(q)]. The constant PES energy zero cancels; residual interactions and anharmonicity are both included.",
            },
            "initial_switch_error": {
                name: _statistics([trajectory["switch_error"][name] for trajectory in trajectories])
                for name in trajectories[0]["switch_error"]
            },
            "maximum_absolute_observed_hamiltonian_drift_au": max(
                trajectory["maximum_absolute_observed_hamiltonian_drift_au"] for trajectory in trajectories
            ),
            "time_series": _series(trajectories),
            "samples": trajectories,
            "propagation": jsonable(dynamics.propagation_stats),
            "units": {"energies": "hartree", "lengths": "bohr", "time_au": "atomic_time", "actions": "dimensionless"},
        }
    finally:
        potential.close()
    if destination is not None:
        write_json(destination, report)
    return report
