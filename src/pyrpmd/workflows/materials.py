"""Equilibrium PIMD (fixed cell or BZP NPT) and fixed-cell RPMD transport.

Coordinates remain continuous and unwrapped for the entire ring and simulation.
The authoritative checkpoint atomically stores phase, state, RNG and a durable
trajectory prefix. New runs stream immutable chunks with bounded staging memory;
legacy checkpoints containing every sample remain readable and can be migrated.
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
from ..io import fingerprint as _fingerprint
from ..io import write_report_json as write_json
from ..parallel import make_task_potential, root_call, run_tasks
from ..potentials.base import validate_result
from ..potentials.cache import StepEvaluationCache
from ..propagation import loop_options
from ..trajectory import TrajectoryStream, iter_chunks, materialize_stream
from ..units import parse_quantity
from ..validation import checked_integer

CHECKPOINT_KIND = "PyRPMD.materials.v1"
STREAM_CHECKPOINT_KIND = "PyRPMD.materials-stream.v1"
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
    if config.get("sampling", {}).get("engine", "legacy") == "native":
        from .native_pimd import resolve_native_pimd
        return resolve_native_pimd(config, system)
    if config.get("sampling", {}).get("engine", "legacy") != "legacy":
        raise ValueError("sampling.engine must be legacy or native")
    from ..electronic_friction import validate_electronic_friction
    friction_spec = validate_electronic_friction(config, system)
    if not np.any(system.mobile_mask):
        raise ValueError("pimd/diffusion require at least one mobile atom")
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
    if system.fixed_atoms and loop["loop_backend"] == "native":
        raise ValueError("fixed atoms require loop_backend='auto' or 'python'")
    output = config.get("output", {})
    storage = output.get("trajectory_storage", "stream")
    if storage not in ("stream", "memory"):
        raise ValueError("output.trajectory_storage must be stream or memory")
    options.update(storage=storage, chunk_frames=checked_integer(
        output.get("chunk_frames", 256), "output.chunk_frames"))
    if options["chunk_frames"] > 65536:
        raise ValueError("output.chunk_frames must be at most 65536")
    options["barostat"] = None
    if "barostat" in config:
        table = config["barostat"]
        if not isinstance(table, Mapping) or set(table) - {"method", "pressure", "time_constant", "friction"}:
            raise ValueError("Unknown barostat fields or non-table barostat options")
        if task != "pimd":
            raise ValueError("barostat requires task='pimd'; RPMD diffusion uses fixed-cell production")
        if table.get("method", "bzp") != "bzp":
            raise ValueError("barostat.method must be bzp")
        if system.cell is None or not all(system.pbc) or system.fixed_atoms:
            raise ValueError("BZP requires a fully periodic 3D cell with all atoms mobile")
        if loop["loop_backend"] == "native":
            raise ValueError("BZP requires loop_backend='auto' or 'python'")
        parallel = config.get("parallel", {})
        if (parallel.get("backend") == "mpi" or parallel.get("mode", "serial") != "serial"
                or parallel.get("bead_workers", 1) != 1):
            raise ValueError("BZP supports serial PES evaluation with optional trajectory workers; bead/MPI/GPU execution is not supported")
        if "pressure" not in table:
            raise ValueError("barostat.pressure is required")
        pressure = parse_quantity(table["pressure"], "pressure")
        tau = parse_quantity(table.get("time_constant", "1000 fs"), "time")
        friction = None if "friction" not in table else parse_quantity(table["friction"], "frequency")
        if tau <= 0 or (friction is not None and friction < 0):
            raise ValueError("BZP time_constant must be positive and friction nonnegative")
        options["barostat"] = dict(pressure_au=pressure, tau_au=tau, friction_au=friction)
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
                   electronic_friction=friction_spec,
                   production_dt=sample_dt if task == "pimd" else dt)
    if options["nprod"] // options["interval"] < (3 if task == "diffusion" else 1):
        raise ValueError("production_steps must contain at least three sample intervals for diffusion, one for pimd")
    table = config.get("diffusion", {})
    if not isinstance(table, Mapping) or set(table) - DIFFUSION_FIELDS:
        raise ValueError("Unknown diffusion fields or non-table diffusion options")
    if task == "diffusion":
        default = [i for i, symbol in enumerate(system.symbols)
                   if symbol in ("H", "D", "T") and system.mobile_mask[i]]
        options["atom_indices"] = _indices(table.get("atom_indices"), system.natoms,
                                          "diffusion.atom_indices", default=default)
        if set(options["atom_indices"]) & set(system.fixed_atoms):
            raise ValueError("diffusion.atom_indices must select mobile atoms")
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


def quantum_observables(state, potential, masses, beta, mobile_mask=None):
    """Physical quantum estimators, distinct from the diagnostic H_P.

    For the sum-beads Hamiltonian sampled at beta/P, K_primitive =
    3N_mobile*P/(2 beta) - H_spring/P and K_cv = 3N_mobile/(2 beta)
    - sum((q-q_centroid)*F)/(2P). No fictitious-momentum kinetic energy
    enters the quantum internal energy estimator.
    """
    q, p = state.q, state.p
    result = validate_result(potential.evaluate(q), q.shape)
    beads, natoms, _ = q.shape
    mobile = np.ones(natoms, dtype=bool) if mobile_mask is None else np.asarray(mobile_mask)
    if mobile.shape != (natoms,) or mobile.dtype.kind != "b" or not mobile.any():
        raise ValueError("mobile_mask must select at least one atom")
    if (not np.array_equal(q[:, ~mobile], np.broadcast_to(q[:1, ~mobile], q[:, ~mobile].shape))
            or np.any(p[:, ~mobile] != 0)):
        raise ValueError("fixed atoms must have identical bead positions and zero momenta")
    nmobile = int(mobile.sum())
    offsets = q - q.mean(axis=0, keepdims=True)
    dq = q - np.roll(q, -1, axis=0)
    spring = 0.5 * (beads / beta) ** 2 * np.sum(np.asarray(masses)[None, mobile, None] * dq[:, mobile] ** 2)
    potential_energy = float(np.mean(result.energy))
    primitive = 3.0 * nmobile * beads / (2.0 * beta) - spring / beads
    virial = 3.0 * nmobile / (2.0 * beta) - np.sum(offsets[:, mobile] * result.forces[:, mobile]) / (2.0 * beads)
    hamiltonian = np.sum(result.energy) + spring + np.sum(p * p / (2.0 * np.asarray(masses)[None, :, None]))
    values = np.array([potential_energy, primitive, virial, potential_energy + virial, hamiltonian], dtype=float)
    radii = np.mean(np.sum(offsets * offsets, axis=-1), axis=0)
    radii[~mobile] = 0.0
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


def _stream_directory(output, replica):
    return Path(output) / "samples" / f"materials_{replica:04d}"


def _trajectory_schema(system, options):
    schema = {
        "sample_steps": {"dtype": "<i8", "shape": []},
        "positions": {"dtype": "<f8", "shape": [system.natoms, 3]},
        "velocities": {"dtype": "<f8", "shape": [system.natoms, 3]},
        "observables": {"dtype": "<f8", "shape": [len(OBSERVABLE_NAMES)]},
        "radii_gyration_squared": {"dtype": "<f8", "shape": [system.natoms]},
    }
    if options["barostat"] is not None:
        schema.update(cells={"dtype": "<f8", "shape": [3, 3]},
                      pressures={"dtype": "<f8", "shape": []},
                      volumes={"dtype": "<f8", "shape": []})
    return schema


def _check_fixed_state(state, system):
    if system.fixed_atoms:
        fixed = list(system.fixed_atoms)
        if (not np.array_equal(state.q[:, fixed], np.broadcast_to(
                system.geometry[fixed], state.q[:, fixed].shape))
                or np.any(state.p[:, fixed] != 0)):
            raise ValueError("fixed atoms must have reference positions on every bead and zero momenta")


def _expected_steps(elapsed, options):
    if elapsed < options["neq"]:
        return np.empty(0, dtype=np.int64)
    return options["neq"] + np.arange((elapsed - options["neq"]) // options["interval"] + 1, dtype=np.int64) * options["interval"]


def _read_stream(config, system, output, replica, options=None, *, materialize=True):
    """Validate a committed prefix; trajectory chunks are never discovered by glob."""
    options = resolve_materials(config, system) if options is None else options
    checkpoint = _checkpoint_path(output, replica)
    schema = _trajectory_schema(system, options)
    with load_checkpoint(checkpoint, require_integrity=True) as saved:
        kind = saved.envelope.get("kind")
        if kind not in (CHECKPOINT_KIND, STREAM_CHECKPOINT_KIND):
            raise ValueError("unsupported materials checkpoint kind")
        if str(saved["fingerprint"]) != _identity(config, system, replica):
            raise ValueError(f"materials checkpoint physics differs: {checkpoint}")
        elapsed = int(saved["elapsed"])
        total = options["neq"] + options["nprod"]
        if not 0 <= elapsed <= total or int(saved["step"]) != elapsed:
            raise ValueError("materials checkpoint has an invalid phase clock")
        expected_time = (min(elapsed, options["neq"]) * options["sample_dt"]
                         + max(0, elapsed - options["neq"]) * options["production_dt"])
        if not np.isclose(float(saved["time"]), expected_time, rtol=1e-10, atol=1e-10):
            raise ValueError("materials checkpoint time disagrees with its phase clock")
        state = State(saved["q"], saved["p"], int(saved["step"]), float(saved["time"]))
        if state.q.shape != (config["beads"], system.natoms, 3):
            raise ValueError("materials checkpoint state shape disagrees with system")
        _check_fixed_state(state, system)
        result = {
            "replica": replica, "state": state, "elapsed": elapsed,
            "rng": json.loads(str(saved["rng"])),
            "propagation": json.loads(str(saved["propagation"])),
            "checkpoint": str(checkpoint), "complete": elapsed == total,
            "storage": "stream" if kind == STREAM_CHECKPOINT_KIND else "memory",
            "barostat": json.loads(str(saved["barostat"])) if "barostat" in saved else None,
        }
        if (result["barostat"] is None) != (options["barostat"] is None):
            raise ValueError("materials checkpoint is missing or has unexpected barostat state")
        if result["barostat"] is not None:
            from ..barostat import validate_bzp_snapshot
            result["barostat"] = validate_bzp_snapshot(result["barostat"], system.cell)
        if kind == STREAM_CHECKPOINT_KIND:
            descriptor = json.loads(str(saved["trajectory"]))
            if not isinstance(descriptor, dict) or descriptor.get("schema") != schema:
                raise ValueError("materials trajectory schema disagrees with system")
            result.update(trajectory=descriptor, trajectory_directory=str(_stream_directory(output, replica)))
            chunks = iter_chunks(result["trajectory_directory"], descriptor)
            count = descriptor["frames"]
        else:
            arrays = {name: saved[name] for name in schema}
            count = len(arrays["sample_steps"])
            chunks = iter([arrays])
            result.update(arrays)
        expected_count = 0 if elapsed < options["neq"] else (elapsed - options["neq"]) // options["interval"] + 1
        # An initial observable failure can save state/RNG before frame zero.
        initial_pending = elapsed == options["neq"] == 0 and count == 0
        if count != expected_count and not initial_pending:
            raise ValueError("materials checkpoint samples disagree with its phase clock")
        cursor = 0
        last_cell = None
        last_step = None
        for arrays in chunks:
            length = len(arrays["sample_steps"])
            for name, spec in schema.items():
                value = arrays[name]
                if value.shape != (length, *spec["shape"]) or not np.isfinite(value).all():
                    raise ValueError("materials checkpoint has invalid observable arrays")
            expected = options["neq"] + np.arange(cursor, cursor + length, dtype=np.int64) * options["interval"]
            if not np.array_equal(arrays["sample_steps"], expected):
                raise ValueError("materials checkpoint samples disagree with its phase clock")
            if system.fixed_atoms:
                fixed = list(system.fixed_atoms)
                # Centroid summation can round a repeated reference by a few ulps.
                if (not np.allclose(arrays["positions"][:, fixed], system.geometry[fixed], rtol=0, atol=1e-12)
                        or np.any(arrays["velocities"][:, fixed] != 0)
                        or np.any(arrays["radii_gyration_squared"][:, fixed] != 0)):
                    raise ValueError("materials trajectory violates fixed-atom constraints")
            if "volumes" in arrays and (np.any(arrays["volumes"] <= 0)
                    or not np.allclose(np.abs(np.linalg.det(arrays["cells"])), arrays["volumes"], rtol=1e-12)):
                raise ValueError("materials NPT trajectory has invalid cell volumes")
            if "cells" in arrays and length:
                scales = np.einsum("fij,ij->f", arrays["cells"], system.cell) / np.sum(system.cell**2)
                if (np.any(scales <= 0) or not np.allclose(arrays["cells"],
                        scales[:, None, None] * system.cell, rtol=1e-11, atol=1e-12)):
                    raise ValueError("materials NPT trajectory cell is not an isotropic reference scaling")
                last_cell, last_step = arrays["cells"][-1], int(arrays["sample_steps"][-1])
            cursor += length
        if cursor != count:
            raise ValueError("materials trajectory frame count differs from its checkpoint")
        if (last_step == elapsed and last_cell is not None
                and not np.array_equal(last_cell, np.asarray(result["barostat"]["cell"]))):
            raise ValueError("materials checkpoint cell disagrees with its last saved frame")
        result["frames"] = count
    if materialize and result["storage"] == "stream":
        result.update(materialize_stream(result["trajectory_directory"], result["trajectory"]))
    if "sample_steps" in result:
        result["times"] = (result["sample_steps"] - options["neq"]) * options["production_dt"]
    return result


def _sample_replica(task):
    config, system, output, replica = task["config"], task["system"], task["output"], task["replica"]
    options = resolve_materials(config, system)
    materialize = task.get("materialize", True)
    checkpoint = _checkpoint_path(output, replica)
    schema = _trajectory_schema(system, options)
    saved = _read_stream(config, system, output, replica, options, materialize=False) if checkpoint.exists() else None
    if saved is not None and saved["complete"]:
        if saved["storage"] == options["storage"]:
            return _read_stream(config, system, output, replica, options, materialize=materialize) if materialize else saved
        # A completed storage migration must work without the original PES,
        # calculator process or electronic-structure driver being available.
        payload = dict(fingerprint=_identity(config, system, replica), elapsed=saved["elapsed"],
                       q=saved["state"].q, p=saved["state"].p,
                       step=saved["state"].step, time=saved["state"].time,
                       rng=json.dumps(saved["rng"]), propagation=json.dumps(saved["propagation"]))
        if saved["barostat"] is not None:
            payload["barostat"] = json.dumps(saved["barostat"])
        if options["storage"] == "stream":
            writer = TrajectoryStream(_stream_directory(output, replica), schema=schema,
                                      chunk_frames=options["chunk_frames"])
            writer.extend({name: saved[name] for name in schema})
            payload["trajectory"] = json.dumps(writer.prepare_commit())
            kind = STREAM_CHECKPOINT_KIND
        else:
            payload.update(materialize_stream(saved["trajectory_directory"], saved["trajectory"]))
            kind = CHECKPOINT_KIND
        atomic_savez(checkpoint, kind=kind, **payload)
        return _read_stream(config, system, output, replica, options, materialize=materialize)
    provider = StepEvaluationCache(make_task_potential(config["potential"], system))
    friction_provider = None
    try:
        rng = np.random.default_rng(np.random.SeedSequence([int(config.get("seed", 0)), 1301, replica]))
        kwargs = dict(seed=rng, native=config.get("native", config.get("dynamics", {}).get("native", True)),
                      threads=config.get("parallel", {}).get("threads_per_rank", 1), **loop_options(config))
        if system.fixed_atoms:
            kwargs.update(fixed_atoms=system.fixed_atoms, fixed_positions=system.geometry)
        sampling = Dynamics(provider, system.masses, config["beta"], config["beads"], options["sample_dt"], **kwargs)
        if options["electronic_friction"] is not None:
            from ..electronic_friction import (
                ElectronicFrictionDynamics,
                make_electronic_friction,
            )
            friction_provider = make_electronic_friction(config, system)
            production = ElectronicFrictionDynamics(
                provider, system.masses, config["beta"], config["beads"], options["production_dt"],
                friction_provider, **kwargs)
        else:
            production = sampling if config["task"] == "pimd" else Dynamics(
                provider, system.masses, config["beta"], config["beads"], options["production_dt"], **kwargs)
        piston = None
        if options["barostat"] is not None:
            from ..barostat import BZPBarostat
            piston = BZPBarostat(provider, system, config["beta"], config["beads"], **options["barostat"])
        elapsed = 0
        records = {name: [] for name in schema}
        writer = None
        if options["storage"] == "stream":
            writer = TrajectoryStream(_stream_directory(output, replica), schema=schema,
                chunk_frames=options["chunk_frames"],
                descriptor=saved["trajectory"] if saved is not None and saved["storage"] == "stream" else None,
                validate=False)
        last_saved = None
        if saved is not None:
            state, elapsed = saved["state"], saved["elapsed"]
            rng.bit_generator.state = saved["rng"]
            sampling.propagation_stats = copy.deepcopy(saved["propagation"]["sampling"])
            if production is not sampling:
                production.propagation_stats = copy.deepcopy(saved["propagation"]["production"])
            if piston is not None:
                piston.restore(saved["barostat"])
            if saved["storage"] == "memory":
                if writer is not None:
                    writer.extend({name: saved[name] for name in schema})
                else:
                    records = {name: list(saved[name]) for name in schema}
            elif writer is None:
                arrays = materialize_stream(saved["trajectory_directory"], saved["trajectory"])
                records = {name: list(arrays[name]) for name in schema}
                del arrays
            if saved["storage"] == options["storage"]:
                last_saved = (elapsed, saved["frames"])
            del saved
        else:
            state = State(np.repeat(system.geometry[None], config["beads"], axis=0), sampling.sample_momenta())

        def frame_count():
            return writer.frames if writer is not None else len(records["sample_steps"])

        def save():
            nonlocal last_saved
            version = (elapsed, frame_count())
            if version == last_saved:
                return
            arrays = dict(fingerprint=_identity(config, system, replica), elapsed=elapsed,
                          q=state.q, p=state.p, step=state.step, time=state.time,
                          rng=json.dumps(rng.bit_generator.state),
                          propagation=json.dumps({"sampling": sampling.propagation_stats,
                              "production": production.propagation_stats if production is not sampling else None}))
            if piston is not None:
                arrays["barostat"] = json.dumps(piston.snapshot())
            if writer is not None:
                # Chunks are durable before the atomic state+prefix commit.
                # A crash here leaves harmless orphan chunks, never extra frames.
                arrays["trajectory"] = json.dumps(writer.prepare_commit())
                kind = STREAM_CHECKPOINT_KIND
            else:
                arrays.update({name: np.asarray(records[name], dtype=spec["dtype"]).reshape(-1, *spec["shape"])
                               for name, spec in schema.items()})
                kind = CHECKPOINT_KIND
            atomic_savez(checkpoint, kind=kind, **arrays)
            last_saved = version

        def observe():
            args = (state, provider, system.masses, config["beta"])
            values, radius = (quantum_observables(*args, mobile_mask=system.mobile_mask)
                              if system.fixed_atoms else quantum_observables(*args))
            record = dict(sample_steps=state.step, positions=state.q.mean(axis=0),
                          velocities=state.p.mean(axis=0) / system.masses[:, None],
                          observables=values, radii_gyration_squared=radius)
            if system.fixed_atoms:
                record["positions"][list(system.fixed_atoms)] = system.geometry[list(system.fixed_atoms)]
            if piston is not None:
                record.update(cells=piston.cell, pressures=piston.pressure(state), volumes=piston.volume)
            return record

        def append_record(record):
            if writer is not None:
                writer.append(record)
            else:
                for name, value in record.items():
                    records[name].append(np.array(value, copy=True))

        if elapsed == options["neq"] == 0 and frame_count() == 0:
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
            old_piston = piston.snapshot() if piston is not None else None
            try:
                with provider.step():
                    if piston is not None:
                        piston.step(state, dynamics, thermostat=options["thermostat"],
                                    gamma_centroid=options["gamma_centroid"],
                                    andersen_interval=options["andersen_interval"])
                    elif is_sampling and options["thermostat"] == "pile-l":
                        dynamics.sample_step(state, gamma_centroid=options["gamma_centroid"])
                        dynamics._record_python_steps(1)
                        reasons = dynamics.propagation_stats["fallback_reasons"]
                        reason = "PILE-L sampling uses Python/NumPy"
                        reasons[reason] = reasons.get(reason, 0) + 1
                    else:
                        if is_sampling and state.step % options["andersen_interval"] == 0:
                            state.p[:] = dynamics.sample_momenta()
                        dynamics.propagate(state, 1, record_centroids=False)
                    prospective = elapsed + 1
                    record = prospective >= options["neq"] and (prospective - options["neq"]) % options["interval"] == 0
                    observation = observe() if record else None
                if record:
                    append_record(observation)
            except BaseException:
                state = old_state
                rng.bit_generator.state = old_rng
                dynamics.propagation_stats = old_stats
                if piston is not None:
                    piston.restore(old_piston)
                save()
                raise
            elapsed = prospective
            if elapsed % options["checkpoint_steps"] == 0:
                save()
        save()
        return _read_stream(config, system, output, replica, options, materialize=materialize)
    finally:
        try:
            if callable(getattr(friction_provider, "close", None)):
                friction_provider.close()
        finally:
            if hasattr(provider, "close"):
                provider.close()


def _write_trajectory(stream, config, system, output):
    options = resolve_materials(config, system)
    metadata = {"task": config["task"], "positions": "unwrapped_ring_polymer_centroids",
                "position_unit": "bohr", "velocity_unit": "bohr/atomic_time",
                "time_unit": "atomic_time", "energy_unit": "hartree",
                "pressure_unit": "hartree/bohr^3", "cell_unit": "bohr",
                "equilibration_steps": options["neq"],
                "sample_interval_steps": options["interval"],
                "production_timestep_au": options["production_dt"],
                "beads": config["beads"], "beta_au": config["beta"],
                "observable_names": OBSERVABLE_NAMES, "system": system.to_dict()}
    if stream["storage"] == "stream":
        path = _stream_directory(output, stream["replica"]) / "manifest.json"
        # This is a derived export, not the authoritative simulation commit.
        write_json(path, {"format": "PyRPMD.materials-trajectory-stream.v1",
                          "trajectory": stream["trajectory"], "metadata": metadata})
        return path
    atomic_savez(_trajectory_path(output, stream["replica"]), kind="PyRPMD.materials-trajectory.v1",
                 positions=stream["positions"], velocities=stream["velocities"], times=stream["times"],
                 sample_steps=stream["sample_steps"], observables=stream["observables"],
                 radii_gyration_squared=stream["radii_gyration_squared"],
                 **{name: stream[name] for name in ("cells", "pressures", "volumes") if name in stream},
                 metadata=json.dumps(metadata))
    return _trajectory_path(output, stream["replica"])


def _analysis_stream(stream, config, system, options, *, return_data):
    """Load scalar diagnostics and only transport-selected coordinate histories.

    Propagation and checkpointing are bounded by chunk_frames. Offline FFT
    diagnostics still scale with the number of saved scalar/selected-atom
    samples; they never require every substrate coordinate in stream mode.
    Explicit return_data=True requests the complete in-memory trajectory.
    """
    if return_data:
        if "positions" not in stream:
            stream.update(materialize_stream(stream["trajectory_directory"], stream["trajectory"]))
    elif stream["storage"] == "stream":
        fields = ["sample_steps", "observables"]
        fields += [name for name in ("pressures", "volumes") if name in stream["trajectory"]["schema"]]
        stream.update(materialize_stream(stream["trajectory_directory"], stream["trajectory"], fields=fields))
        radius_sum = np.zeros(system.natoms)
        atoms = (sorted(set(options["atom_indices"]) | set(options["reference_atoms"] or []))
                 if config["task"] == "diffusion" else None)
        fields = ["radii_gyration_squared"] + (["positions", "velocities"] if atoms is not None else [])
        if atoms is not None:
            stream["positions"] = np.empty((stream["frames"], len(atoms), 3))
            stream["velocities"] = np.empty_like(stream["positions"])
            stream["_analysis_atoms"] = atoms
        cursor = 0
        for chunk in iter_chunks(stream["trajectory_directory"], stream["trajectory"], fields=fields):
            length = len(chunk["radii_gyration_squared"])
            radius_sum += chunk["radii_gyration_squared"].sum(axis=0)
            if atoms is not None:
                for name in ("positions", "velocities"):
                    stream[name][cursor:cursor + length] = chunk[name][:, atoms]
            cursor += length
        stream["radii_gyration_squared"] = (radius_sum / max(1, stream["frames"]))[None]
    elif config["task"] == "diffusion":
        atoms = sorted(set(options["atom_indices"]) | set(options["reference_atoms"] or []))
        for name in ("positions", "velocities"):
            stream[name] = stream[name][:, atoms]
        stream["_analysis_atoms"] = atoms
    stream["times"] = (stream["sample_steps"] - options["neq"]) * options["production_dt"]
    return stream


def _analyze(streams, config, system, output, return_data=False):
    options = resolve_materials(config, system)
    if not streams or any(s["frames"] < 2 for s in streams):
        raise ValueError("materials analysis requires saved production samples")
    trajectories = [_write_trajectory(s, config, system, output) for s in streams]
    streams = [_analysis_stream(s, config, system, options, return_data=return_data) for s in streams]
    result = {
        "task": config["task"], "status": "complete" if all(s["complete"] for s in streams) else "partial",
        "physical_converged": False, "converged": False,
        "replicas": len(streams), "samples_per_replica": [len(s["times"]) for s in streams],
        "phase": ("centroid_ef_rpmd_production" if options["electronic_friction"] is not None else
                  "equilibrium_sampling" if config["task"] == "pimd" else "unthermostatted_rpmd_production"),
        "sampling_thermostat": options["thermostat"], "integrator": loop_options(config)["integrator"],
        "coordinate_convention": "continuous_unwrapped_ring_polymer_centroids",
        "cell": system.to_dict().get("cell"), "pbc": system.to_dict().get("pbc", [False, False, False]),
        "ensemble": "npt" if options["barostat"] is not None else "fixed_cell",
        "fixed_atoms": list(system.fixed_atoms), "mobile_atoms": int(system.mobile_mask.sum()),
        "trajectory_storage": [s["storage"] for s in streams],
        "propagation": [s["propagation"] for s in streams],
        "warnings": ["Bead count, timestep, equilibration, system size and statistical convergence have not been certified."],
        "files": {"checkpoints": [s["checkpoint"] for s in streams],
                  "trajectories": [str(path) for path in trajectories]},
    }
    if options["neq"] == 0:
        result["warnings"].append("No equilibration was requested; the collapsed-ring initial geometry is not an equilibrium configurational sample.")
    if system.fixed_atoms:
        result["warnings"].append("Fixed atoms are external coordinates; quantum kinetic-energy estimators count mobile atoms only.")
    if options["barostat"] is not None:
        from ..analysis.diagnostics import correlated_mean_diagnostics
        from ..constants import BOHR_M, HARTREE_J
        result["barostat"] = {
            "method": "bzp", **options["barostat"],
            "target_pressure_bar": options["barostat"]["pressure_au"] * HARTREE_J / BOHR_M**3 / 1e5,
            "final_cells_bohr": [s["barostat"]["cell"] for s in streams],
            "mean_volume_bohr3": float(np.mean([s["volumes"].mean() for s in streams])),
            "mean_pressure_hartree_per_bohr3": float(np.mean([s["pressures"].mean() for s in streams])),
            "volume_bohr3": [{"mean": float(s["volumes"].mean()),
                               **correlated_mean_diagnostics(s["volumes"])} for s in streams],
            "pressure_hartree_per_bohr3": [{"mean": float(s["pressures"].mean()),
                                            **correlated_mean_diagnostics(s["pressures"])} for s in streams],
        }
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
        from ..analysis.diagnostics import nve_energy_diagnostics
        from ..analysis.diffusion import analyze_diffusion

        times = streams[0]["times"]
        if any(not np.array_equal(s["times"], times) for s in streams):
            raise ValueError("diffusion replicas need the same completed sample clock for aggregation")
        if options["electronic_friction"] is None:
            result["nve_energy_diagnostics"] = nve_energy_diagnostics(
                [s["observables"][:, OBSERVABLE_NAMES.index("ring_polymer_hamiltonian")] for s in streams],
                times, beads=config["beads"], beta=config["beta"],
            )
            if not result["nve_energy_diagnostics"]["within_tolerance"]:
                result["warnings"].append(
                    "Saved production frames exceed 0.1 kBT per bead in H_P error; "
                    "check the timestep independently of the MSD fit."
                )
        else:
            result["electronic_friction"] = {
                "method": "markovian_centroid_ef_rpmd", "bath_temperature_K": config.get("temperature_K"),
                "tensor_convention": "physical_eta_mass_per_time", "friction_average": "all_beads",
                "noise_convention": "shared_bead_increment_centroid_covariance_M_over_beta",
                "internal_mode_thermostat": False, "non_markovian_memory": False,
            }
            result["nve_energy_diagnostics"] = None
            result["warnings"].append(
                "EF-RPMD exchanges energy with the electronic bath; H_P is not conserved. "
                "Centroid friction omits bath-induced changes to the imaginary-time distribution."
            )
        positions = np.array([s["positions"] for s in streams])
        velocities = np.array([s["velocities"] for s in streams])
        table = config.get("diffusion", {})
        params = {k: table[k] for k in ("fit_range", "max_lag", "bootstrap", "seed", "block_length") if k in table}
        params.setdefault("seed", int(config.get("seed", 0)))
        analysis_atoms = streams[0].get("_analysis_atoms", list(range(system.natoms)))
        params.update(atom_indices=[analysis_atoms.index(i) for i in options["atom_indices"]], axes=options["axes"])
        refs = options["reference_atoms"]
        if refs is not None:
            ref_indices = [analysis_atoms.index(i) for i in refs]
            params.update(reference_positions=positions[:, :, ref_indices], reference_velocities=velocities[:, :, ref_indices],
                          reference_masses=system.masses[refs])
        diffusion = analyze_diffusion(positions, times, velocities=velocities, **params)
        diffusion["input"]["atoms"] = system.natoms
        diffusion["input"]["atom_indices"] = options["atom_indices"]
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
    if config.get("sampling", {}).get("engine", "legacy") == "native":
        from .native_pimd import run_native_pimd
        return run_native_pimd(config, system, output, executor, return_data)
    options = resolve_materials(config, system)
    tasks = [{"config": copy.deepcopy(config), "system": system, "output": str(output), "replica": replica,
              "materialize": False}
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
        streams = [_read_stream(config, system, output, replica, options, materialize=False)
                   for replica in range(options["replicas"])]
        return _analyze(streams, config, system, output, return_data)
    return root_call(analyze)
