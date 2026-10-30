"""Fixed-cell equilibrium PIMD and unthermostatted RPMD transport streams.

Coordinates remain continuous and unwrapped for the entire ring and simulation.
The authoritative checkpoint atomically stores phase, state, RNG and every
sample; exported trajectories can therefore be regenerated after an interruption.
No thermostat or biased time is interpreted as an RPMD diffusion trajectory.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from ..checkpoint import atomic_savez, load_checkpoint
from ..constants import FS_TO_AU
from ..dynamics import Dynamics, State
from ..parallel import make_task_potential, root_call, run_tasks
from ..potentials.base import validate_result
from ..potentials.cache import StepEvaluationCache
from ..propagation import loop_options
from .fes import _fingerprint, checked_integer, write_json


CHECKPOINT_KIND = "PyRPMD.materials.v1"
OBSERVABLE_NAMES = (
    "potential_energy",
    "kinetic_primitive",
    "kinetic_centroid_virial",
    "internal_energy_centroid_virial",
    "ring_polymer_hamiltonian",
)
DIFFUSION_FIELDS = {
    "atom_indices", "axes", "reference_atoms", "fit_range", "max_lag",
    "bootstrap", "seed", "block_length",
}


def _indices(value, size, name, *, default=None):
    if value is None:
        value = default
    raw = np.asarray(value)
    if (raw.ndim != 1 or raw.dtype.kind not in "iu" or not len(raw)
            or np.any(raw < 0) or np.any(raw >= size)
            or len(set(raw.tolist())) != len(raw)):
        raise ValueError(f"{name} must contain distinct 0-based indices in [0,{size})")
    return raw.astype(int).tolist()


def resolve_materials(config, system):
    """Validate the scientific choices also for direct resolved-workflow calls."""
    if system.fixed_atoms:
        raise ValueError("pimd/diffusion require all atoms mobile; fixed layers are supported by fes/surface_rate")
    task = config.get("task")
    if task not in ("pimd", "diffusion"):
        raise ValueError("materials task must be pimd or diffusion")
    if config.get("umbrella") or config.get("metadynamics") or config.get("reaction"):
        raise ValueError("pimd/diffusion workflows require an unbiased materials system")
    sampling = config.get("sampling", {})
    if not isinstance(sampling, Mapping):
        raise ValueError("sampling must be a table")
    options = {
        "neq": checked_integer(sampling.get("equilibration_steps", 100), "sampling.equilibration_steps", 0),
        "nprod": checked_integer(sampling.get("production_steps", 1000), "sampling.production_steps"),
        "interval": checked_integer(sampling.get("sample_interval", sampling.get("save_every", 10)), "sampling.sample_interval"),
        "replicas": checked_integer(sampling.get("replicas", 4 if task == "diffusion" else 1), "sampling.replicas"),
        "checkpoint_steps": checked_integer(config.get("output", {}).get("checkpoint_steps", 1000), "output.checkpoint_steps"),
        "andersen_interval": checked_integer(sampling.get("andersen_interval", 10), "sampling.andersen_interval"),
    }
    loop = loop_options(config)
    thermostat = sampling.get("thermostat", "andersen" if loop["integrator"] == "velocity-verlet" else "pile-l")
    if thermostat not in ("pile-l", "andersen"):
        raise ValueError("materials sampling.thermostat must be pile-l or andersen")
    if thermostat == "pile-l" and loop["integrator"] == "velocity-verlet":
        raise ValueError("PILE-L needs Cayley BCOCB or normal-mode BAOAB; Verlet uses Andersen")
    if thermostat == "pile-l" and loop["loop_backend"] == "native":
        raise ValueError("PILE-L sampling needs loop_backend='auto' or 'python'")
    expected_sampler = ({"cayley-bcb": "bcocb", "normal-mode-bab": "baoab"}[loop["integrator"]]
                        if thermostat == "pile-l" else loop["integrator"] + "-andersen")
    if sampling.get("integrator", expected_sampler) != expected_sampler:
        raise ValueError(f"materials sampling.integrator must be {expected_sampler} for the selected method")
    gamma = sampling.get("gamma_centroid", 1.0 / (100.0 * FS_TO_AU))
    if not isinstance(gamma, (int, float, np.number)) or isinstance(gamma, (bool, np.bool_)) or not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("sampling.gamma_centroid must be positive in inverse atomic time")
    dt = float(config["dt_au"])
    sample_dt = float(sampling.get("dt_au", dt))
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(sample_dt) or sample_dt <= 0:
        raise ValueError("materials timesteps must be positive and finite")
    options.update(thermostat=thermostat, gamma_centroid=float(gamma), sample_dt=sample_dt,
                   production_dt=sample_dt if task == "pimd" else dt)
    if options["nprod"] // options["interval"] < (3 if task == "diffusion" else 1):
        raise ValueError("production_steps must contain at least three sample intervals for diffusion, one for pimd")
    table = config.get("diffusion", {})
    if not isinstance(table, Mapping) or set(table) - DIFFUSION_FIELDS:
        raise ValueError("Unknown diffusion fields or non-table diffusion options")
    if task == "diffusion":
        default = [i for i, symbol in enumerate(system.symbols) if symbol in ("H", "D", "T")]
        options["atom_indices"] = _indices(table.get("atom_indices"), system.natoms,
                                          "diffusion.atom_indices", default=default)
        options["axes"] = _indices(table.get("axes"), 3, "diffusion.axes", default=[0, 1, 2])
        options["reference_atoms"] = None
        if table.get("reference_atoms") is not None:
            ref = _indices(table["reference_atoms"], system.natoms, "diffusion.reference_atoms")
            if set(ref) & set(options["atom_indices"]):
                raise ValueError("diffusion.reference_atoms must not include the measured atoms")
            options["reference_atoms"] = ref
        for key, minimum in (("max_lag", 1), ("bootstrap", 0), ("seed", 0), ("block_length", 2)):
            if key in table and (table[key] is not None or key in ("bootstrap", "seed")):
                checked_integer(table[key], "diffusion." + key, minimum)
        if table.get("bootstrap") == 1:
            raise ValueError("diffusion.bootstrap must be zero or at least two")
        # These limits depend only on the declared recording schedule. Fail
        # before constructing a provider or allocating an output directory,
        # rather than discovering invalid analysis choices after production.
        frames = options["nprod"] // options["interval"] + 1
        max_lag = table.get("max_lag")
        if max_lag is None:
            max_lag = max(1, (frames - 1) // 4)
        if max_lag >= frames:
            raise ValueError("diffusion.max_lag must be smaller than the planned saved frame count")
        if table.get("block_length") is not None and table["block_length"] < 4 * max_lag:
            raise ValueError("diffusion.block_length must be at least four times max_lag")
        if table.get("fit_range") is not None:
            bounds = table["fit_range"]
            if not isinstance(bounds, (tuple, list, np.ndarray)) or len(bounds) != 2:
                raise ValueError("diffusion.fit_range needs [start, stop] in atomic time")
            saved_dt = options["interval"] * options["production_dt"]
            last_lag = max_lag * saved_dt
            try:
                if any(isinstance(v, (bool, np.bool_)) for v in bounds):
                    raise ValueError
                lower = max(1, max_lag // 2) * saved_dt if bounds[0] is None else float(bounds[0])
                upper = last_lag if bounds[1] is None else float(bounds[1])
            except (ValueError, TypeError) as exc:
                raise ValueError("diffusion.fit_range needs two real time bounds or None") from exc
            if not np.isfinite([lower, upper]).all() or lower < 0:
                raise ValueError("diffusion.fit_range must be finite and nonnegative")
            if upper <= lower:
                raise ValueError("diffusion.fit_range must be increasing")
            if upper > last_lag * (1 + 1e-12) or lower > last_lag:
                raise ValueError("diffusion.fit_range extends beyond the available lag time")
    return options


def validate_materials(config, system):
    resolve_materials(config, system)


def quantum_observables(state, potential, masses, beta):
    """Physical quantum estimators, distinct from the diagnostic H_P.

    For the sum-beads Hamiltonian sampled at beta/P, K_primitive =
    3NP/(2 beta) - H_spring/P and K_cv = 3N/(2 beta)
    - sum((q-q_centroid)*F)/(2P). No fictitious-momentum kinetic energy
    enters the quantum internal energy estimator.
    """
    q, p = state.q, state.p
    result = validate_result(potential.evaluate(q), q.shape)
    beads, natoms, _ = q.shape
    offsets = q - q.mean(axis=0, keepdims=True)
    dq = q - np.roll(q, -1, axis=0)
    spring = 0.5 * (beads / beta) ** 2 * np.sum(np.asarray(masses)[None, :, None] * dq * dq)
    potential_energy = float(np.mean(result.energy))
    primitive = 3.0 * natoms * beads / (2.0 * beta) - spring / beads
    virial = 3.0 * natoms / (2.0 * beta) - np.sum(offsets * result.forces) / (2.0 * beads)
    hamiltonian = np.sum(result.energy) + spring + np.sum(p * p / (2.0 * np.asarray(masses)[None, :, None]))
    values = np.array([potential_energy, primitive, virial, potential_energy + virial, hamiltonian], dtype=float)
    radii = np.mean(np.sum(offsets * offsets, axis=-1), axis=0)
    if not np.isfinite(values).all() or not np.isfinite(radii).all():
        raise FloatingPointError("nonfinite materials observable")
    return values, radii


def _identity(config, system, replica):
    # Every atom is saved, so changing a transport selection or fit is strictly
    # offline analysis and does not alter the sampled Hamiltonian or RNG stream.
    physics = copy.deepcopy(config)
    physics.pop("diffusion", None)
    return _fingerprint(physics, system, {"materials_replica": replica})


def _checkpoint_path(output, replica):
    return Path(output) / "checkpoints" / f"materials_{replica:04d}.npz"


def _trajectory_path(output, replica):
    return Path(output) / "samples" / f"materials_{replica:04d}.npz"


def _expected_steps(elapsed, options):
    if elapsed < options["neq"]:
        return np.empty(0, dtype=np.int64)
    return options["neq"] + np.arange((elapsed - options["neq"]) // options["interval"] + 1, dtype=np.int64) * options["interval"]


def _read_stream(config, system, output, replica, options=None):
    options = resolve_materials(config, system) if options is None else options
    checkpoint = _checkpoint_path(output, replica)
    with load_checkpoint(checkpoint, expected_kind=CHECKPOINT_KIND, require_integrity=True) as saved:
        if str(saved["fingerprint"]) != _identity(config, system, replica):
            raise ValueError(f"materials checkpoint physics differs: {checkpoint}")
        elapsed = int(saved["elapsed"])
        total = options["neq"] + options["nprod"]
        if not 0 <= elapsed <= total or int(saved["step"]) != elapsed:
            raise ValueError("materials checkpoint has an invalid phase clock")
        expected_time = min(elapsed, options["neq"]) * options["sample_dt"] + max(0, elapsed - options["neq"]) * options["production_dt"]
        if not np.isclose(float(saved["time"]), expected_time, rtol=1e-10, atol=1e-10):
            raise ValueError("materials checkpoint time disagrees with its phase clock")
        clocks = saved["sample_steps"].copy()
        expected = _expected_steps(elapsed, options)
        # A failure evaluating the initial frame can retain the initialized
        # state/RNG without claiming that the frame has been observed.
        initial_pending = elapsed == options["neq"] == 0 and clocks.shape == (0,)
        if not initial_pending and not np.array_equal(clocks, expected):
            raise ValueError("materials checkpoint samples disagree with its phase clock")
        count = len(clocks)
        positions, velocities = saved["positions"].copy(), saved["velocities"].copy()
        observable, radii = saved["observables"].copy(), saved["radii_gyration_squared"].copy()
        if (positions.shape != (count, system.natoms, 3) or velocities.shape != positions.shape
                or observable.shape != (count, len(OBSERVABLE_NAMES))
                or radii.shape != (count, system.natoms)
                or any(not np.isfinite(a).all() for a in (positions, velocities, observable, radii))):
            raise ValueError("materials checkpoint has invalid observable arrays")
        state = State(saved["q"], saved["p"], int(saved["step"]), float(saved["time"]))
        if state.q.shape != (config["beads"], system.natoms, 3):
            raise ValueError("materials checkpoint state shape disagrees with system")
        propagation = json.loads(str(saved["propagation"]))
        return {
            "replica": replica, "state": state, "elapsed": elapsed,
            "rng": json.loads(str(saved["rng"])), "sample_steps": clocks,
            "times": (clocks - options["neq"]) * options["production_dt"],
            "positions": positions, "velocities": velocities, "observables": observable,
            "radii_gyration_squared": radii, "propagation": propagation,
            "checkpoint": str(checkpoint), "complete": elapsed == total,
        }


def _sample_replica(task):
    config, system, output, replica = task["config"], task["system"], task["output"], task["replica"]
    options = resolve_materials(config, system)
    checkpoint = _checkpoint_path(output, replica)
    provider = StepEvaluationCache(make_task_potential(config["potential"], system))
    try:
        rng = np.random.default_rng(np.random.SeedSequence([int(config.get("seed", 0)), 1301, replica]))
        kwargs = dict(seed=rng, native=config.get("native", config.get("dynamics", {}).get("native", True)),
                      threads=config.get("parallel", {}).get("threads_per_rank", 1), **loop_options(config))
        sampling = Dynamics(provider, system.masses, config["beta"], config["beads"], options["sample_dt"], **kwargs)
        production = sampling if config["task"] == "pimd" else Dynamics(
            provider, system.masses, config["beta"], config["beads"], options["production_dt"], **kwargs)
        elapsed = 0
        positions, velocities, observable, radii, clocks = [], [], [], [], []
        last_saved = None
        if checkpoint.exists():
            saved = _read_stream(config, system, output, replica, options)
            state, elapsed = saved["state"], saved["elapsed"]
            rng.bit_generator.state = saved["rng"]
            sampling.propagation_stats = copy.deepcopy(saved["propagation"]["sampling"])
            if production is not sampling:
                production.propagation_stats = copy.deepcopy(saved["propagation"]["production"])
            if saved["complete"]:
                return saved
            positions, velocities = list(saved["positions"]), list(saved["velocities"])
            observable, radii, clocks = list(saved["observables"]), list(saved["radii_gyration_squared"]), list(saved["sample_steps"])
            last_saved = (elapsed, len(clocks))
        else:
            state = State(np.repeat(system.geometry[None], config["beads"], axis=0), sampling.sample_momenta())

        def save():
            nonlocal last_saved
            # A failed transaction restores state, RNG and statistics together.
            # Outside it, only accepting a step or the pending initial sample
            # changes the payload. Do not rewrite an already committed prefix.
            version = (elapsed, len(clocks))
            if version == last_saved:
                return
            atomic_savez(checkpoint, kind=CHECKPOINT_KIND,
                         fingerprint=_identity(config, system, replica), elapsed=elapsed,
                         q=state.q, p=state.p, step=state.step, time=state.time,
                         rng=json.dumps(rng.bit_generator.state),
                         positions=np.asarray(positions).reshape(-1, system.natoms, 3),
                         velocities=np.asarray(velocities).reshape(-1, system.natoms, 3),
                         observables=np.asarray(observable).reshape(-1, len(OBSERVABLE_NAMES)),
                         radii_gyration_squared=np.asarray(radii).reshape(-1, system.natoms),
                         sample_steps=np.asarray(clocks, dtype=np.int64),
                         propagation=json.dumps({"sampling": sampling.propagation_stats,
                                                 "production": production.propagation_stats if production is not sampling else None}))
            last_saved = version

        def observe():
            values, radius = quantum_observables(state, provider, system.masses, config["beta"])
            return state.q.mean(axis=0), state.p.mean(axis=0) / system.masses[:, None], values, radius

        def append_record(record):
            pos, vel, values, radius = record
            positions.append(pos.copy()); velocities.append(vel.copy())
            observable.append(values.copy()); radii.append(radius.copy()); clocks.append(state.step)

        if elapsed == options["neq"] == 0 and not clocks:
            try:
                append_record(observe())
            except BaseException:
                save()
                raise
        save()
        total = options["neq"] + options["nprod"]
        while elapsed < total:
            is_sampling = config["task"] == "pimd" or elapsed < options["neq"]
            dynamics = sampling if is_sampling else production
            old_state = state.copy()
            old_rng = copy.deepcopy(rng.bit_generator.state)
            old_stats = copy.deepcopy(dynamics.propagation_stats)
            try:
                # Energy, physical-force and observable consumers share the
                # endpoint evaluation only inside this transaction. Never
                # carry a PES cache through a failure, restart or phase change.
                with provider.step():
                    if is_sampling and options["thermostat"] == "pile-l":
                        dynamics.sample_step(state, gamma_centroid=options["gamma_centroid"])
                        dynamics.propagation_stats["python_steps"] += 1
                        reasons = dynamics.propagation_stats["fallback_reasons"]
                        reason = "PILE-L sampling uses Python/NumPy"
                        reasons[reason] = reasons.get(reason, 0) + 1
                    else:
                        if is_sampling and state.step % options["andersen_interval"] == 0:
                            state.p[:] = dynamics.sample_momenta()
                        # Preserve the accepted prefix and observation boundary
                        # independently of execution block sizing.
                        dynamics.propagate(state, 1, record_centroids=False)
                    prospective = elapsed + 1
                    record = prospective >= options["neq"] and (prospective - options["neq"]) % options["interval"] == 0
                    observation = observe() if record else None
            except BaseException:
                state = old_state
                rng.bit_generator.state = old_rng
                dynamics.propagation_stats = old_stats
                save()
                raise
            elapsed = prospective
            if record:
                append_record(observation)
            if elapsed % options["checkpoint_steps"] == 0:
                save()
        save()
        stream = _read_stream(config, system, output, replica, options)
        return stream
    finally:
        if hasattr(provider, "close"):
            provider.close()


def _write_trajectory(stream, config, system, output):
    atomic_savez(_trajectory_path(output, stream["replica"]), kind="PyRPMD.materials-trajectory.v1",
                 positions=stream["positions"], velocities=stream["velocities"], times=stream["times"],
                 sample_steps=stream["sample_steps"], observables=stream["observables"],
                 radii_gyration_squared=stream["radii_gyration_squared"],
                 metadata=json.dumps({"task": config["task"], "positions": "unwrapped_ring_polymer_centroids",
                                      "position_unit": "bohr", "velocity_unit": "bohr/atomic_time",
                                      "time_unit": "atomic_time", "energy_unit": "hartree",
                                      "observable_names": OBSERVABLE_NAMES, "system": system.to_dict()}))


def _analyze(streams, config, system, output, return_data=False):
    options = resolve_materials(config, system)
    if not streams or any(len(s["times"]) < 2 for s in streams):
        raise ValueError("materials analysis requires saved production samples")
    for stream in streams:
        _write_trajectory(stream, config, system, output)
    result = {
        "task": config["task"], "status": "complete" if all(s["complete"] for s in streams) else "partial",
        "physical_converged": False, "converged": False,
        "replicas": len(streams), "samples_per_replica": [len(s["times"]) for s in streams],
        "phase": "equilibrium_sampling" if config["task"] == "pimd" else "unthermostatted_rpmd_production",
        "sampling_thermostat": options["thermostat"], "integrator": loop_options(config)["integrator"],
        "coordinate_convention": "continuous_unwrapped_ring_polymer_centroids",
        "cell": system.to_dict().get("cell"), "pbc": system.to_dict().get("pbc", [False, False, False]),
        "propagation": [s["propagation"] for s in streams],
        "warnings": ["Bead count, timestep, equilibration, system size and statistical convergence have not been certified."],
        "files": {"checkpoints": [s["checkpoint"] for s in streams],
                  "trajectories": [str(_trajectory_path(output, s["replica"])) for s in streams]},
    }
    if options["neq"] == 0:
        result["warnings"].append("No equilibration was requested; the collapsed-ring initial geometry is not an equilibrium configurational sample.")
    if config["task"] == "pimd":
        from ..analysis.diagnostics import correlated_mean_diagnostics

        means = np.array([s["observables"].mean(axis=0) for s in streams])
        spread = np.std(means, axis=0, ddof=1) / np.sqrt(len(streams)) if len(streams) > 1 else None
        result["energy_unit"] = "hartree"
        result["estimators"] = {name: {"mean": float(means[:, i].mean()),
                                       "standard_error_between_replicas": None if spread is None else float(spread[i])}
                                for i, name in enumerate(OBSERVABLE_NAMES)}
        result["radius_of_gyration_squared_bohr2"] = np.mean(
            [s["radii_gyration_squared"].mean(axis=0) for s in streams], axis=0).tolist()
        result["sampling_diagnostics"] = [
            {"replica": s["replica"], "observables": {
                name: correlated_mean_diagnostics(s["observables"][:, i])
                for i, name in enumerate(OBSERVABLE_NAMES)
            }} for s in streams
        ]
        result["warnings"].append("PIMD sampling times are not physical dynamical times; H_P is reported only as an integration diagnostic.")
    else:
        from ..analysis.diffusion import analyze_diffusion
        from ..analysis.diagnostics import nve_energy_diagnostics

        times = streams[0]["times"]
        if any(not np.array_equal(s["times"], times) for s in streams):
            raise ValueError("diffusion replicas need the same completed sample clock for aggregation")
        result["nve_energy_diagnostics"] = nve_energy_diagnostics(
            [s["observables"][:, OBSERVABLE_NAMES.index("ring_polymer_hamiltonian")] for s in streams],
            times, beads=config["beads"], beta=config["beta"],
        )
        if not result["nve_energy_diagnostics"]["within_tolerance"]:
            result["warnings"].append(
                "Saved production frames exceed 0.1 kBT per bead in H_P error; "
                "check the timestep independently of the MSD fit."
            )
        positions = np.array([s["positions"] for s in streams])
        velocities = np.array([s["velocities"] for s in streams])
        table = config.get("diffusion", {})
        params = {k: table[k] for k in ("fit_range", "max_lag", "bootstrap", "seed", "block_length") if k in table}
        params.setdefault("seed", int(config.get("seed", 0)))
        params.update(atom_indices=options["atom_indices"], axes=options["axes"])
        refs = options["reference_atoms"]
        if refs is not None:
            params.update(reference_positions=positions[:, :, refs], reference_velocities=velocities[:, :, refs],
                          reference_masses=system.masses[refs])
        diffusion = analyze_diffusion(positions, times, velocities=velocities, **params)
        result["diffusion"] = diffusion
        result["analysis_status"] = diffusion["status"]
        result["diffusion_m2_s"] = diffusion["diffusion_m2_s"]
        result["diffusion_tensor_m2_s"] = diffusion["diffusion_tensor_m2_s"]
        result["files"]["diffusion"] = str(Path(output) / "results" / "diffusion.json")
        write_json(result["files"]["diffusion"], diffusion)
    write_json(Path(output) / "summary.json", result)
    return {"summary": result, "streams": streams} if return_data else result


def run_materials(config, system, output, executor=None, return_data=False):
    """Execute independent replica streams via serial/process/MPI task groups."""
    options = resolve_materials(config, system)
    tasks = [{"config": copy.deepcopy(config), "system": system, "output": str(output), "replica": replica}
             for replica in range(options["replicas"])]
    streams = executor.map(_sample_replica, tasks) if executor is not None else run_tasks(
        _sample_replica, tasks, config.get("parallel", {}))
    return root_call(_analyze, streams, config, system, str(output), return_data)


def reanalyze_diffusion(config, system, output, return_data=False):
    """Analyze saved unwrapped production trajectories; never evaluate a PES."""
    if config.get("task") != "diffusion":
        raise ValueError("reanalyze_diffusion requires a diffusion run")
    def analyze():
        options = resolve_materials(config, system)
        streams = [_read_stream(config, system, output, replica, options) for replica in range(options["replicas"])]
        return _analyze(streams, config, system, output, return_data)
    return root_call(analyze)
