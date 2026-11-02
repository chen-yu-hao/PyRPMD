"""Registered steering workflow; endpoint work is not an unbiased FES grid."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from ..checkpoint import save_metadata_npz
from ..dynamics import Dynamics, State
from ..io import write_report_json
from ..parallel import make_task_potential, root_call
from ..potentials import validate_periodic_spec
from ..propagation import loop_options
from ..steered import SteeredBias, SteeredMD, crooks, jarzynski
from ..units import DIMS, parse_quantity
from ..validation import checked_integer, finite_real
from .biased_sampling import dispatch_tasks
from .common import _dimension, _fingerprint, resolve_cvs


STEERED_FIELDS = set("cvs centers centers_au start start_au end end_au stiffness stiffness_au "
                     "direction thermostat gamma gamma_au initial_equilibrium".split())


def validate_steered_sampling(config, sampling):
    """Check the raw frontend options before generic defaults obscure intent."""
    raw = config.get("sampling", {})
    if not isinstance(raw, Mapping):
        raise ValueError("steered sampling must be a table")
    unsupported = {"engine", "thermostat_params", "gamma_centroid", "andersen_interval",
                   "andersen_schedule", "parents"}.intersection(raw)
    if unsupported:
        raise ValueError("steered sampling does not accept " + ", ".join(sorted(unsupported)) +
                         "; configure the symmetric OU bath in [steered]")
    thermostat = config.get("steered", {}).get("thermostat", "langevin")
    if sampling.get("thermostat", thermostat) != thermostat:
        raise ValueError("sampling.thermostat conflicts with steered.thermostat")
    if sampling.get("integrator", "steered-symmetric") != "steered-symmetric":
        raise ValueError("steered sampling uses sampling.integrator='steered-symmetric'")


def _alias(table, name, *, default=None, required=False):
    if name in table and name + "_au" in table:
        raise ValueError(f"specify steered.{name} or {name}_au, not both")
    if required and name not in table and name + "_au" not in table:
        raise ValueError(f"steered.{name} is required")
    return table.get(name + "_au", table.get(name, default)), name + "_au" in table


def _point(values, cv, scales, atomic, name):
    if cv.dimension == 1 and np.isscalar(values):
        values = [values]
    if not isinstance(values, (list, tuple, np.ndarray)) or len(values) != cv.dimension:
        raise ValueError(f"steered.{name} requires one value per CV")
    result = []
    for value, unit, scale in zip(values, cv.units, scales):
        number = parse_quantity(value, _dimension(unit))
        result.append(number * (scale if not atomic and not isinstance(value, str) else 1.0))
    return result


def _stiffness(raw, cv, scales, atomic):
    values = np.asarray(raw, dtype=object)
    if values.ndim == 0:
        matrix = np.empty((cv.dimension, cv.dimension), dtype=object)
        matrix[:] = 0.0
        np.fill_diagonal(matrix, values.item())
    elif values.shape == (cv.dimension,):
        matrix = np.diag(values)
    elif values.shape == (cv.dimension, cv.dimension):
        matrix = values
    else:
        raise ValueError("steered.stiffness must be scalar, a D-vector or a D-by-D matrix")
    result = np.empty(matrix.shape)
    for i, j in np.ndindex(matrix.shape):
        dimension = tuple(e - a - b for e, a, b in
                          zip(DIMS["energy"], _dimension(cv.units[i]), _dimension(cv.units[j])))
        value = matrix[i, j]
        result[i, j] = parse_quantity(value, dimension) / (
            scales[i] * scales[j] if not atomic and not isinstance(value, str) else 1.0)
    return result


def resolve_steered(config, system):
    """Resolve the full center protocol, physical units and supported options."""
    table = config.get("steered", {})
    if not isinstance(table, Mapping) or set(table) - STEERED_FIELDS:
        raise ValueError("unknown steered fields; expected " + ", ".join(sorted(STEERED_FIELDS)))
    if not np.any(system.mobile_mask):
        raise ValueError("steered sampling requires at least one mobile atom")
    if system.bond_constraints or config.get("barostat"):
        raise ValueError("steered sampling does not support bond constraints or a barostat")
    validate_periodic_spec(config.get("potential", {}), system)
    if config.get("replica_exchange", {}).get("enabled") or config.get("adaptive_umbrella", {}).get("enabled"):
        raise ValueError("steered sampling does not support replica exchange or adaptive umbrellas")
    if config.get("umbrella") or config.get("metadynamics") or config.get("opes"):
        raise ValueError("steered sampling requires its own moving restraint without another bias")
    if config.get("dynamics", {}).get("loop_backend", "auto") == "native":
        raise ValueError("steered sampling requires loop_backend='auto' or 'python'")
    cv, selected, scales = resolve_cvs(config, system, "steered")
    sampling = config.get("sampling", {})
    production = checked_integer(sampling["production_steps"], "sampling.production_steps")
    equilibration = checked_integer(sampling.get("equilibration_steps", 0), "sampling.equilibration_steps", minimum=0)
    replicas = checked_integer(sampling.get("replicas", 1), "sampling.replicas")
    dt = finite_real(sampling.get("dt_au", config["dt_au"]), "steering timestep", positive=True)
    if "centers" in table or "centers_au" in table:
        if any(key in table for key in ("start", "start_au", "end", "end_au")):
            raise ValueError("specify steering centers or start/end, not both")
        values, atomic = _alias(table, "centers", required=True)
        if not isinstance(values, (list, tuple, np.ndarray)) or len(values) != production + 1:
            raise ValueError("steered.centers must contain production_steps+1 centers")
        centers = np.asarray([_point(row, cv, scales, atomic, "centers") for row in values])
    else:
        start, atomic_start = _alias(table, "start", required=True)
        end, atomic_end = _alias(table, "end", required=True)
        centers = np.linspace(_point(start, cv, scales, atomic_start, "start"),
                              _point(end, cv, scales, atomic_end, "end"), production + 1)
    stiffness, atomic = _alias(table, "stiffness", required=True)
    stiffness = _stiffness(stiffness, cv, scales, atomic)
    probe = SteeredBias(cv, centers, stiffness)
    probe.energy_force(np.repeat(system.geometry[None], config["beads"], axis=0))
    thermostat = table.get("thermostat", "langevin")
    if thermostat not in ("none", "langevin"):
        raise ValueError("steered.thermostat must be none or langevin")
    if sampling.get("thermostat", thermostat) != thermostat:
        raise ValueError("sampling.thermostat conflicts with steered.thermostat")
    if sampling.get("integrator", "steered-symmetric") != "steered-symmetric":
        raise ValueError("steered sampling uses sampling.integrator='steered-symmetric'")
    if any(key in sampling for key in ("thermostat_params", "gamma_centroid", "andersen_schedule")):
        raise ValueError("steered sampling owns its OU bath; set steered.gamma instead of generic thermostat controls")
    if thermostat == "none" and equilibration:
        raise ValueError("steered equilibration requires thermostat='langevin'")
    gamma, _ = _alias(table, "gamma", default="0.01 fs^-1")
    gamma = finite_real(parse_quantity(gamma, "frequency"), "steered.gamma", nonnegative=True)
    if thermostat == "langevin" and gamma <= 0:
        raise ValueError("steered Langevin gamma must be positive")
    initial_equilibrium = table.get("initial_equilibrium", False)
    if not isinstance(initial_equilibrium, bool):
        raise ValueError("steered.initial_equilibrium must be boolean")
    direction = table.get("direction", "both")
    if direction not in ("forward", "reverse", "both"):
        raise ValueError("steered.direction must be forward, reverse or both")
    analysis = config.get("analysis", {})
    if analysis.get("measure", "restrained_endpoints") != "restrained_endpoints":
        raise ValueError("steered work estimates restrained endpoint free energies; analysis.measure must be restrained_endpoints")
    method = analysis.get("method", "crooks" if direction == "both" else "jarzynski")
    if method not in ("crooks", "jarzynski") or method == "crooks" and direction != "both":
        raise ValueError("steered analysis.method must be jarzynski, or crooks with direction='both'")
    if any(key in analysis for key in ("bins", "ranges", "flux_axis", "min_count", "max_grid_points", "overlap_threshold")):
        raise ValueError("steered endpoint work analysis does not accept FES histogram or flux options")
    bootstrap_keys = [key for key in ("bootstrap", "bootstrap_replicates", "bootstrap_samples") if key in analysis]
    if len(bootstrap_keys) > 1 and len({analysis[key] for key in bootstrap_keys}) > 1:
        raise ValueError("conflicting steering bootstrap counts")
    bootstrap = checked_integer(analysis[bootstrap_keys[0]] if bootstrap_keys else 200,
                                "analysis.bootstrap", minimum=0)
    if bootstrap == 1:
        raise ValueError("steered bootstrap must be zero or at least two")
    checkpoint_steps = checked_integer(config.get("output", {}).get("checkpoint_steps", 1000), "output.checkpoint_steps")
    interval = checked_integer(sampling.get("sample_interval", sampling.get("interval_steps", 1)), "sampling.sample_interval")
    return cv, selected, {
        "centers": centers, "stiffness": stiffness, "dt": dt, "gamma": gamma,
        "thermostat": thermostat, "equilibration_steps": equilibration,
        "production_steps": production, "replicas": replicas, "direction": direction,
        "initial_equilibrium": initial_equilibrium, "method": method, "bootstrap": bootstrap,
        "analysis_seed": checked_integer(analysis.get("seed", 0), "analysis.seed", minimum=0),
        "checkpoint_steps": checkpoint_steps, "sample_interval": interval,
    }


def validate_steered(config, system):
    resolve_steered(config, system)


def _tasks(config, system, output):
    _, _, options = resolve_steered(config, system)
    directions = ("forward", "reverse") if options["direction"] == "both" else (options["direction"],)
    return [{"config": copy.deepcopy(config), "system": system, "output": str(output),
             "direction": direction, "replica": replica}
            for direction in directions for replica in range(options["replicas"])]


def _driver(task, provider):
    config, system = task["config"], task["system"]
    cv, selected, options = resolve_steered(config, system)
    identity = _fingerprint(config, system, {"steered_replica": task["replica"],
                                           "steered_direction": task["direction"]})
    dynamics = Dynamics(provider, system.masses, config["beta"], config["beads"], options["dt"],
                        seed=np.random.SeedSequence([int(config.get("seed", 0)), 711,
                                                     int(task["direction"] == "reverse"), task["replica"]]),
                        native=config.get("native", config.get("dynamics", {}).get("native", True)),
                        threads=config.get("parallel", {}).get("threads_per_rank", 1),
                        fixed_atoms=system.fixed_atoms, fixed_positions=system.geometry,
                        cell=system.cell, pbc=system.pbc if system.cell is not None else None,
                        potential_identity=identity, **loop_options(config))
    centers = options["centers"][::-1] if task["direction"] == "reverse" else options["centers"]
    bias = SteeredBias(cv, centers, options["stiffness"])
    state = State(np.repeat(system.geometry[None], config["beads"], axis=0), dynamics.sample_momenta())
    simulation = SteeredMD(dynamics, state, bias, thermostat=options["thermostat"], gamma=options["gamma"],
                           equilibration_steps=options["equilibration_steps"],
                           initial_equilibrium=options["initial_equilibrium"],
                           cv_identity=_fingerprint(config, system, {"steered_cv": selected}))
    checkpoint = Path(task["output"]) / "checkpoints" / f"steered_{task['direction']}_{task['replica']:04d}.npz"
    return simulation, checkpoint, options


def _sample(task):
    provider = make_task_potential(task["config"]["potential"], task["system"])
    try:
        simulation, checkpoint, options = _driver(task, provider)
        if checkpoint.exists():
            simulation.load(checkpoint)
        result = simulation.run(checkpoint=checkpoint, checkpoint_interval=options["checkpoint_steps"])
        return {**result, "direction": task["direction"], "replica": task["replica"],
                "fingerprint": simulation.fingerprint, "checkpoint": str(checkpoint)}
    finally:
        if callable(getattr(provider, "close", None)):
            provider.close()


class _NoEvaluation:
    """Reanalysis validates state/identity without constructing a force engine."""

    def evaluate(self, q):
        raise RuntimeError("steering reanalysis must not evaluate a potential")


def _read(task):
    simulation, checkpoint, _ = _driver(task, _NoEvaluation())
    simulation.load(checkpoint)
    return {**simulation.result(), "direction": task["direction"], "replica": task["replica"],
            "fingerprint": simulation.fingerprint, "checkpoint": str(checkpoint)}


def _analyze(streams, config, system, output, return_data):
    streams = list(streams)
    cv, _, options = resolve_steered(config, system)
    forward = np.asarray([s["total_work"][-1] for s in streams if s["direction"] == "forward" and s["complete"]])
    reverse = np.asarray([s["total_work"][-1] for s in streams if s["direction"] == "reverse" and s["complete"]])
    estimates = {}
    for name, work in (("forward", forward), ("reverse", reverse)):
        if len(work):
            estimates[name] = jarzynski(work, config["beta"], bootstrap=options["bootstrap"],
                                        seed=options["analysis_seed"]).to_dict()
            selected = [s for s in streams if s["direction"] == name and s["complete"]]
            protocol_only = jarzynski([s["protocol_work"][-1] for s in selected],
                                      config["beta"], bootstrap=0).delta_free_energy
            estimates[name]["diagnostics"].update(
                protocol_only_delta_free_energy=protocol_only,
                shadow_work_correction=estimates[name]["delta_free_energy"] - protocol_only,
                mean_shadow_work=float(np.mean([s["shadow_work"][-1] for s in selected])),
            )
    if len(forward) and len(reverse):
        estimates["crooks"] = crooks(forward, reverse, config["beta"], bootstrap=options["bootstrap"],
                                     seed=options["analysis_seed"]).to_dict()
    summary = {
        "task": "fes", "method": "steered", "status": "complete" if all(s["complete"] for s in streams) else "partial",
        "estimator": options["method"], "target": "restrained_endpoint_free_energy_difference",
        "measure": "restrained_endpoints", "energy_unit": "hartree", "beta": config["beta"],
        "dimension": cv.dimension, "units": cv.units, "direction": options["direction"],
        "replicas_per_direction": options["replicas"], "estimates": estimates,
        "sample_interval": options["sample_interval"],
        "propagation": [s["propagation"] for s in streams],
        "converged": False, "physical_converged": False,
        "initial_equilibrium_declared": options["initial_equilibrium"],
        "protocol": {"centers_au": options["centers"].tolist(), "stiffness_au": options["stiffness"].tolist(),
                     "timestep_au": options["dt"], "thermostat": options["thermostat"], "gamma_au": options["gamma"]},
        "work_convention": "work done on system; total_work=protocol_work+shadow_work; all energies H_P/P; heat excluded",
        "warnings": [
            "Endpoint restraint free energies are not an unbiased CV PMF; Hummer-Szabo reconstruction is not implemented.",
            "Equilibration uses Metropolis-corrected fixed-center dynamics; finite duration does not establish independent canonical starts.",
            "All replicas start from the supplied geometry; initial_equilibrium is only a declaration, not a validation of that ensemble.",
            "Bootstrap uncertainty is conditional on observed independent trajectories and excludes absent work tails and initialization bias.",
        ],
        "files": {"data": str(Path(output) / "steered.npz"),
                  "checkpoints": [s["checkpoint"] for s in streams]},
    }
    primary = (estimates.get("crooks") if options["method"] == "crooks" else
               estimates.get("forward", estimates.get("reverse")))
    if primary is not None:
        summary["delta_free_energy"] = primary["delta_free_energy"]
        summary["standard_error"] = primary["standard_error"]
        summary["delta_free_energy_direction"] = ("B_to_A" if options["direction"] == "reverse" else "A_to_B")
    arrays = {"forward_work": forward, "reverse_work": reverse}
    for index, stream in enumerate(streams):
        saved = np.unique(np.r_[np.arange(0, len(stream["time"]), options["sample_interval"]),
                                 len(stream["time"]) - 1])
        for name in ("time", "cv_values", "centers", "protocol_work", "shadow_work", "total_work", "heat"):
            arrays[f"stream_{index:04d}_{name}"] = stream[name][saved]
    save_metadata_npz(Path(output) / "steered.npz", {"format": "PyRPMD.steered-results.v1", "summary": summary}, **arrays)
    write_report_json(Path(output) / "summary.json", summary)
    return {"summary": summary, "data": arrays, "streams": streams} if return_data else summary


def run_steered(config, system, output, executor=None, return_data=False):
    """Run independent forward/reverse pulls with per-trajectory checkpoints."""
    validate_steered(config, system)
    streams = dispatch_tasks(_sample, _tasks(config, system, output), config, executor)
    return root_call(_analyze, streams, config, system, output, return_data)


def reanalyze_steered(config, system, output, return_data=False):
    """Reconstruct endpoint estimates without executing any force evaluations."""
    return root_call(lambda: _analyze([_read(task) for task in _tasks(config, system, output)],
                                      config, system, output, return_data))
