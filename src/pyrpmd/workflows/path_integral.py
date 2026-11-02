"""Restartable force-based SC, bosonic and PIGS sampling through Dynamics."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from ..checkpoint import atomic_savez, load_checkpoint
from ..dynamics import Dynamics, State
from ..pimd_methods import (SuzukiChinPotential, BosonicExchangePotential,
    GaussianTrialWavefunction, PIGSPotential)
from ..potentials import make_potential
from ..potentials.base import validate_result
from ..propagation import loop_options
from ..units import parse_quantity
from .fes import _fingerprint
from .materials import resolve_materials


KIND = "PyRPMD.path-integral.v1"


def resolve_path_integral(config, system):
    table = config.get("path_integral", {})
    if not isinstance(table, dict):
        raise ValueError("path_integral must be a table")
    method = table.get("method", "suzuki_chin")
    fields = {
        "suzuki_chin": {"method", "alpha", "fd_step"},
        "bosonic": {"method", "groups"},
        "pigs": {"method", "projection_time", "trial"},
    }
    if not isinstance(method, str) or method not in fields or set(table) - fields[method]:
        raise ValueError("Unknown path_integral method or fields for that method")
    if config.get("task") != "pimd":
        raise ValueError("path_integral sampling actions require task='pimd'")
    unsupported = {"active_learning", "electronic_friction", "nonadiabatic", "plumed"}.intersection(config)
    if unsupported:
        raise ValueError("path_integral does not support sections: " + ", ".join(sorted(unsupported)))
    if any(system.pbc) or system.fixed_atoms or config.get("barostat"):
        raise ValueError("SC/bosonic/PIGS workflow currently requires a nonperiodic all-mobile system")
    if config.get("sampling", {}).get("engine", "legacy") != "legacy":
        raise ValueError("path_integral uses the main Dynamics engine, not sampling.engine='native'")
    parallel = config.get("parallel", {})
    if (parallel.get("backend", "serial") != "serial" or parallel.get("mode", "serial") != "serial"
            or parallel.get("bead_workers", 1) != 1 or parallel.get("trajectory_workers", 1) != 1):
        raise ValueError("force-based path_integral workflow currently requires serial execution")
    if config.get("dynamics", {}).get("loop_backend") == "native":
        raise ValueError("sampling action forces require the Python loop (free steps may be native)")
    if method == "suzuki_chin" and table.get("alpha", 0.0) != 0:
        raise ValueError("SC workflow even-bead observables require alpha=0; general alpha forces are available through the Python API")
    output = config.get("output", {})
    if output.get("trajectory_storage", "memory") != "memory" or "chunk_frames" in output:
        raise ValueError("path_integral stores its full saved prefix in memory; use trajectory_storage='memory' without chunk_frames")
    if {"trajectory_stride", "diagnostics", "store_configurations"}.intersection(output):
        raise ValueError("path_integral saves all sampled configurations; trajectory_stride/diagnostics/store_configurations are unsupported")
    if {"parents", "thermostat_params"}.intersection(config.get("sampling", {})):
        raise ValueError("path_integral does not use sampling.parents or thermostat_params")
    if config.get("potential", {}).get("contracted_beads") is not None:
        raise ValueError("path_integral requires independent physical configurations, without a contracted potential")
    options = resolve_materials(config, system)
    options["storage"] = "memory"
    if options["nprod"] < options["interval"]:
        raise ValueError("production must save at least one sample")
    # Construct against a free provider to perform every action validation
    # before the actual potential or output directory is opened.
    from ..potentials.builtin import FreePotential
    _action(config, system, FreePotential())
    return method, options


def _action(config, system, provider):
    table = config["path_integral"]
    method = table.get("method", "suzuki_chin")
    args = (provider, system.masses, config["beta"], config["beads"])
    if method == "suzuki_chin":
        return SuzukiChinPotential(*args, alpha=table.get("alpha", 0.0),
            fd_step=parse_quantity(table.get("fd_step", "1e-4 bohr"), "length"))
    if method == "bosonic":
        action = BosonicExchangePotential(*args, groups=table.get("groups"))
        for group in action.groups:
            if len({system.symbols[i] for i in group}) != 1:
                raise ValueError("each boson group must contain identical elements")
        return action
    if method != "pigs":
        raise ValueError("unsupported path_integral method")
    trial = table.get("trial", {})
    if (not isinstance(trial, dict) or set(trial) - {"alpha", "center"}
            or "alpha" not in trial):
        raise ValueError("PIGS requires trial.alpha (inverse bohr squared) and optional trial.center (bohr)")
    if "projection_time" not in table:
        raise ValueError("PIGS requires an explicit imaginary projection_time")
    action = PIGSPotential(provider, system.masses, config["beads"],
        parse_quantity(table["projection_time"], "time"), GaussianTrialWavefunction(**trial))
    # Check that the trial broadcasts to the full molecular geometry.
    action.evaluate(np.repeat(system.geometry[None], config["beads"], axis=0))
    return action


def run_path_integral(config, system, output):
    method, options = resolve_path_integral(config, system)
    output = Path(output)
    summaries = []
    for replica in range(options["replicas"]):
        identity = _fingerprint(config, system, {"path_integral_replica": replica})
        checkpoint = output / "checkpoints" / f"path_integral_{replica:04d}.npz"
        physical = make_potential(config["potential"], system)
        try:
            action = _action(config, system, physical)
            dynamics = Dynamics(action, system.masses, action.beta, config["beads"], options["sample_dt"],
                seed=np.random.SeedSequence([int(config.get("seed", 0)), 2219, replica]),
                native=config.get("native", config.get("dynamics", {}).get("native", True)),
                threads=config.get("parallel", {}).get("threads_per_rank", 1),
                **loop_options(config))
            state = State(np.repeat(system.geometry[None], config["beads"], axis=0), dynamics.sample_momenta())
            positions, values, steps = [], [], []
            elapsed = 0
            total = options["neq"] + options["nprod"]
            if checkpoint.exists():
                with load_checkpoint(checkpoint, expected_kind=KIND, require_integrity=True) as saved:
                    if str(saved["fingerprint"]) != identity:
                        raise ValueError("path-integral checkpoint physics differs")
                    for key in ("elapsed", "step"):
                        scalar = np.asarray(saved[key])
                        if scalar.shape != () or scalar.dtype.kind not in "iu":
                            raise ValueError("path-integral checkpoint clock must contain integer scalars")
                    elapsed = int(saved["elapsed"])
                    if not 0 <= elapsed <= total or int(saved["step"]) != elapsed:
                        raise ValueError("invalid path-integral checkpoint clock")
                    state = State(saved["q"], saved["p"], int(saved["step"]), float(saved["time"]))
                    dynamics.integrator._check(state.q, state.p)
                    if not np.isclose(state.time, elapsed*dynamics.dt, rtol=1e-10, atol=1e-10):
                        raise ValueError("path-integral checkpoint time disagrees with step")
                    positions, values, steps = list(saved["positions"]), list(saved["potential"]), list(saved["sample_steps"])
                    expected = np.arange(options["neq"]+options["interval"], elapsed+1, options["interval"])
                    if (np.asarray(saved["sample_steps"]).dtype.kind not in "iu"
                            or not np.array_equal(steps, expected) or len(values) != len(expected)
                            or np.shape(positions) != ((len(expected), config["beads"], system.natoms, 3) if len(expected) else (0,))
                            or not np.isfinite(values).all() or not np.isfinite(positions).all()):
                        raise ValueError("path-integral checkpoint samples disagree with clock/system")
                    dynamics.rng.bit_generator.state = json.loads(str(saved["rng"]))

            def save():
                atomic_savez(checkpoint, kind=KIND, fingerprint=identity, elapsed=elapsed,
                    q=state.q, p=state.p, step=state.step, time=state.time,
                    rng=json.dumps(dynamics.rng.bit_generator.state),
                    positions=np.asarray(positions).reshape(-1, config["beads"], system.natoms, 3),
                    potential=np.asarray(values), sample_steps=np.asarray(steps, dtype=np.int64))

            save()
            while elapsed < total:
                before, rng = state.copy(), copy.deepcopy(dynamics.rng.bit_generator.state)
                try:
                    if options["thermostat"] == "pile-l":
                        dynamics.sample_step(state, gamma_centroid=options["gamma_centroid"])
                    else:
                        if state.step % options["andersen_interval"] == 0:
                            state.p[:] = dynamics.sample_momenta()
                        dynamics.step(state)
                    record = state.step > options["neq"] and (state.step-options["neq"]) % options["interval"] == 0
                    if record:
                        if method == "pigs":
                            observed = state.q[len(state.q)//2:len(state.q)//2+1]
                        elif method == "suzuki_chin":
                            observed = state.q[::2]
                        else:
                            observed = state.q
                        potential = float(validate_result(physical.evaluate(observed), observed.shape).energy.mean())
                except BaseException:
                    state = before
                    dynamics.rng.bit_generator.state = rng
                    save()
                    raise
                elapsed = state.step
                if record:
                    positions.append(state.q.copy())
                    values.append(potential)
                    steps.append(state.step)
                if elapsed % options["checkpoint_steps"] == 0:
                    save()
            save()
            summaries.append({"replica": replica, "samples": len(values),
                "potential_energy_hartree": None if not np.isfinite(values).all() else float(np.mean(values)),
                "checkpoint": str(checkpoint)})
        finally:
            physical.close()
    return {"task": "pimd", "method": method, "equilibrium_only": True,
        "trajectory_storage": "memory", "status": "complete",
        "replicas": summaries, "samples": sum(x["samples"] for x in summaries),
        "observable": ("projected_midpoint_potential" if method == "pigs" else
            "even_bead_potential_alpha_zero_only" if method == "suzuki_chin" else "symmetric_bead_potential"),
        "temperature_K": None if method == "pigs" else config["temperature_K"],
        "projection_time_au": action.projection_time if method == "pigs" else None,
        "limitations": ["potential energy only; not a total-energy or heat-capacity estimator",
            "sampling, time-step and bead/projection convergence must be established separately",
            "no real-time, pressure or permutation-resolved particle estimator"]}
