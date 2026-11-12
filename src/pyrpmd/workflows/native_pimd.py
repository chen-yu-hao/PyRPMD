"""Opt-in native PIMD and diffusion workflow with immutable trajectory records.

The journal is the commit point. A crash before publication can leave an
unreferenced transaction directory, but cannot advance the accepted trajectory.
Restart reads only committed records. Python runs at checkpoint/phase boundaries,
never at the observation cadence. Diffusion is an explicit E2 route: native
PILE-L equilibrates, then a thermostat-free native NVE engine records centroids.
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from ..checkpoint import atomic_savez
from ..io import file_sha256, jsonable, referenced_files, write_json
from ..native_ensemble import (
    _COLUMNS,
    _THERMOSTATS,
    NativeEnsemble,
    NativePeriodicPES,
    ensemble_from_config,
)
from ..parallel import make_task_potential, root_call, run_tasks
from ..propagation import loop_options
from .common import _fingerprint, checked_integer
from .pimd import OBSERVABLE_NAMES as _NAMES

_SCHEMA = "pyrpmd.native-pimd-journal.v1"


# Thermostats with one bath for the whole system. With P>1 the stiff, nearly
# harmonic internal ring-polymer modes exchange energy too slowly through a
# single global bath; analytic harmonic and quartic controls show biased
# estimators (see docs/native-pimd-workflow.md, "Thermostat choice").
GLOBAL_BATH_THERMOSTATS = ("nhc", "svr")


def global_thermostat_warning(thermostat, beads):
    """Warning text for a single global bath on a ring polymer, else None."""
    if thermostat not in GLOBAL_BATH_THERMOSTATS or int(beads) <= 1:
        return None
    return (f"sampling.thermostat='{thermostat}' is one global bath for all {int(beads)} beads; "
            "it does not thermalize the ring-polymer internal modes and can bias PIMD "
            "estimators far beyond their error bars. Use 'pile-l' or 'pile-g' for P>1.")


def sampling_splitting(integrator, thermostat, barostat="none", *, middle=None):
    """Name the full map reported by the shared native executor."""
    if barostat != "none":
        return "bzp-" + middle if middle is not None else "obabo"
    if thermostat == "nhc":
        return "obcbo" if integrator == "cayley-bcb" else "obabo"
    return "bcocb" if integrator == "cayley-bcb" else "baoab"


def mixed_sampling_method(config):
    """New independent middle choice; None retains the historical native map."""
    s = config.get("sampling", {})
    scheme = loop_options(config)["integrator"]
    if s.get("thermostat", "pile-l") != "pile-l":
        return None
    from .sampling import middle_sampling_method
    selected = middle_sampling_method(scheme, s.get("integrator"))
    return selected if selected != middle_sampling_method(scheme) else None


def resolve_native_pimd(config, system):
    """Preflight the opt-in route; the native constructor checks all matrices."""
    from ..trajectory_batch import trajectory_batch_size
    if trajectory_batch_size(config) > 1:
        raise ValueError("native PIMD does not support trajectory PES batching; use sampling.engine='legacy'")
    if system.bond_constraints:
        raise ValueError("native PIMD does not implement bond constraints; use legacy velocity-verlet")
    task = config.get("task")
    if task not in ("pimd", "diffusion"):
        raise ValueError("sampling.engine='native' requires task='pimd' or task='diffusion'")
    s = config.get("sampling", {})
    loop = loop_options(config)
    if loop["integrator"] not in ("cayley-bcb", "normal-mode-bab"):
        raise ValueError("native PIMD requires Cayley or normal-mode integration")
    if loop["loop_backend"] == "python" or config.get("native") is False or config.get("dynamics", {}).get("native") is False:
        raise ValueError("native PIMD cannot disable the native engine")
    mode = s.get("thermostat", "pile-l")
    if mode not in _THERMOSTATS or mode in ("none", "trpmd"):
        raise ValueError("equilibrium native PIMD requires a sampling thermostat, not none/TRPMD")
    expected = (sampling_splitting(loop["integrator"], mode, config.get("barostat", {}).get("mode", "none"))
                if mode == "nhc" else "bcocb" if loop["integrator"] == "cayley-bcb" else "baoab")
    if mode == "pile-l":
        from .sampling import middle_sampling_method
        middle_sampling_method(loop["integrator"], s.get("integrator"))
    elif s.get("integrator", expected) != expected:
        raise ValueError(f"native PIMD sampling.integrator must be {expected}")
    if any(config.get(key) for key in ("umbrella", "metadynamics", "reaction", "electronic_friction")):
        raise ValueError("native PIMD is unbiased; reaction/bias settings are unsupported")
    if task == "diffusion" and mode != "pile-l":
        raise ValueError("native diffusion currently supports only PILE-L equilibration")
    parallel = config.get("parallel", {})
    if (parallel.get("mode", "serial") != "serial" or parallel.get("bead_workers", 1) != 1
            or parallel.get("pes_ranks", 1) != 1 or parallel.get("threads_per_rank", 1) != 1):
        raise ValueError("native PIMD supports trajectory parallelism only; use one rank/thread per trajectory")
    dt = float(config["dt_au"])
    if not np.isfinite(dt) or dt <= 0 or float(s.get("dt_au", dt)) != dt:
        raise ValueError("native PIMD sampling and dynamics need the same positive timestep")
    options = {
        "neq": checked_integer(s.get("equilibration_steps", 100), "equilibration_steps", 0),
        "nprod": checked_integer(s.get("production_steps", 1000), "production_steps"),
        "interval": checked_integer(s.get("sample_interval", 10), "sample_interval"),
        "replicas": checked_integer(s.get("replicas", 1), "replicas"),
        "checkpoint_steps": checked_integer(config.get("output", {}).get("checkpoint_steps", 1000), "checkpoint_steps"),
        "thermostat": mode, "sample_dt": dt, "production_dt": dt,
    }
    minimum_frames = 3 if task == "diffusion" else 2
    if options["nprod"] // options["interval"] < minimum_frames:
        raise ValueError("native diffusion production needs at least three saved frames" if task == "diffusion"
                         else "native PIMD production needs at least two saved frames")
    b = config.get("barostat", {})
    native_barostat_fields = {"mode", "pressure", "tau", "gamma", "compressibility", "cell_mask",
                              "reference_cell", "reference_cell_unit", "stress"}
    if "barostat" in config and ("mode" not in b or set(b) - native_barostat_fields):
        raise ValueError("native PIMD barostat requires explicit mode and native pressure/tau/gamma controls; legacy method/time_constant/friction controls use sampling.engine='legacy'")
    if b.get("mode", "none") not in ("none", "bzp", "mttk", "mtk", "rgb", "scr", "isotropic", "flexible", "anisotropic", "stochastic-rescaling"):
        raise ValueError("unsupported native PIMD barostat; select Suzuki-Chin through the legacy path_integral workflow")
    if b.get("mode", "none") != "none":
        if task == "diffusion":
            raise ValueError("native diffusion uses a fixed cell; NPT is available on task='pimd'")
        if not all(system.pbc) or system.fixed_atoms:
            raise ValueError("native NPT requires 3D PBC and all atoms mobile")
        if mode == "piglet":
            if b.get("mode") not in ("bzp", "isotropic"):
                raise ValueError("PIGLET NPT supports only the isotropic BZP barostat")
            from ..native_ensemble import _validate_piglet_npt
            _validate_piglet_npt(s.get("thermostat_params", {}).get("gle", {}), config["beads"])
    if mixed_sampling_method(config) is not None:
        # Workflow units have already been resolved by _engine; only the mode
        # is needed to reject unvalidated flexible-cell middle compositions.
        bmode = {"isotropic": "bzp"}.get(b.get("mode", "none"), b.get("mode", "none"))
        if bmode not in ("none", "bzp"):
            raise ValueError("mixed middle NPT requires isotropic BZP; flexible/SCR maps retain their legacy splitting")
    spec = config.get("potential", {})
    kind = spec.get("provider", spec.get("type", "builtin"))
    if kind not in ("builtin", "free", "harmonic", "periodic_cosine", "cabi", "native", "rpmdrate", "f2py", "native_periodic"):
        raise ValueError("native PIMD requires a compiled PES or supported builtin; Python/ASE/Torch/socket callbacks are not native")
    if any(system.pbc) and kind in ("cabi", "native", "rpmdrate", "f2py"):
        raise ValueError("molecular native PES bindings do not carry cell/virial data")
    if kind in ("builtin", "free", "harmonic", "periodic_cosine"):
        # Constructing an analytic builtin has no external resource side effect
        # and validates all its parameters using the existing provider contract.
        from ..potentials import make_potential
        _native_potential(make_potential(spec, system))
    if task == "diffusion":
        # Reuse the materials validation contract without selecting its
        # legacy numerical backend. This validates atom/axis/lag choices
        # before provider construction or output creation. NativeEnsemble
        # records only steps after sample_start, so unlike the legacy
        # engine there is no saved handoff frame at production time zero.
        from .materials import _resolve_legacy_materials
        legacy = copy.deepcopy(config)
        legacy.setdefault("sampling", {})["engine"] = "legacy"
        legacy.setdefault("dynamics", {})["loop_backend"] = "auto"
        diffusion = _resolve_legacy_materials(legacy, system, handoff_frame=False)
        options.update({key: diffusion[key] for key in (
            "atom_indices", "axes", "reference_atoms", "production_dt",
        )})
        options["diffusion"] = {key: config.get("diffusion", {})[key]
                                 for key in ("fit_range", "max_lag", "bootstrap", "seed", "block_length")
                                 if key in config.get("diffusion", {})}
    return options


def _native_potential(provider):
    from ..potentials.builtin import (
        FreePotential,
        HarmonicPotential,
        PeriodicCosinePotential,
    )
    if type(provider) is FreePotential:
        return {"kind": "free"}
    if type(provider) is HarmonicPotential:
        if provider.k.ndim != 0 or np.any(provider.center != 0):
            raise ValueError("native harmonic mapping requires scalar k and zero center")
        return {"kind": "harmonic", "k": float(provider.k)}
    if type(provider) is PeriodicCosinePotential:
        if provider.atoms is not None or np.any(provider.repeats != 1) or provider.confinement != 0:
            raise ValueError("native cosine mapping requires all atoms, unit repeats and no confinement")
        return {"kind": "periodic_cosine", "heights": provider.height.tolist()}
    if isinstance(provider, NativePeriodicPES) or callable(getattr(provider, "native_binding", None)):
        return provider
    raise ValueError("potential has no callback-free native binding")


def _engine(config, system, replica, provider, *, thermostat=None):
    s = config["sampling"]
    t = copy.deepcopy(s.get("thermostat_params", {}))
    if "mode" in t:
        raise ValueError("set thermostat mode in sampling.thermostat, not thermostat_params")
    t["mode"] = s.get("thermostat", "pile-l") if thermostat is None else thermostat
    if thermostat == "none":
        t = {"mode": "none"}
    if thermostat != "none" and "gamma_centroid" in s:
        if "gamma_centroid" in t:
            raise ValueError("gamma_centroid was specified twice")
        t["gamma_centroid"] = s["gamma_centroid"]
    if t["mode"] == "pile-l":
        t.setdefault("gamma_centroid", "0.01 fs^-1")
    if t["mode"] == "andersen":
        t.setdefault("interval", s.get("andersen_interval", 10))
    # Independent replica RNG streams, stable against process count and chunks.
    seed = int(np.random.SeedSequence([int(config.get("seed", 0)), 1303, replica]).generate_state(1, dtype=np.uint64)[0])
    potential = _native_potential(provider)
    ring_integrator = loop_options(config)["integrator"]
    spec = dict(positions=system.geometry.tolist(), masses=system.masses.tolist(),
                beta=config["beta"], beads=config["beads"], timestep=config["dt_au"],
                integrator=ring_integrator, thermostat=t,
                barostat=copy.deepcopy(config.get("barostat", {"mode": "none"})),
                seed=seed, fixed_atoms=list(system.fixed_atoms), pbc=list(map(bool, system.pbc)),
                potential=potential if isinstance(potential, dict) else {"kind": "free"})
    if thermostat is None and mixed_sampling_method(config) is not None:
        spec["sampling_integrator"] = mixed_sampling_method(config)
    if system.cell is not None:
        spec["cell"] = system.cell.tolist()
    identity = _fingerprint(config, system, {"potential_files": referenced_files(config)})
    return ensemble_from_config(spec, potential_override=None if isinstance(potential, dict) else potential,
                                potential_identity=identity)


def _checksum(value):
    clean = {k: v for k, v in value.items() if k != "checksum"}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _journal(path, identity, options, natoms):
    value = json.loads(path.read_text())
    if (value.get("schema") != _SCHEMA or value.get("identity") != identity
            or value.get("checksum") != _checksum(value)):
        raise ValueError("native PIMD journal identity/checksum differs")
    end = 0
    for i, entry in enumerate(value["blocks"]):
        name = entry["directory"]
        if Path(name).name != name or not name.startswith("chunk-"):
            raise ValueError("invalid native PIMD block path")
        if entry["start"] != end or entry["end"] < end or (i > 0 and entry["end"] == end):
            raise ValueError("native PIMD journal has a discontinuous clock")
        end = entry["end"]
        directory = path.parent / name
        if file_sha256(directory/"state.npz") != entry["state_sha256"]:
            raise ValueError("native PIMD checkpoint checksum differs")
        if entry["samples_sha256"] is not None:
            sample_path = directory/"samples.npy"
            if file_sha256(sample_path) != entry["samples_sha256"]:
                raise ValueError("native PIMD trajectory checksum differs")
            a = max(0, entry["start"] - options["neq"]) // options["interval"]
            b = max(0, end - options["neq"]) // options["interval"]
            data = np.load(sample_path, mmap_mode="r" if b > a else None, allow_pickle=False)
            expected = options["neq"] + np.arange(a+1, b+1)*options["interval"]
            if data.shape != (b-a, len(_COLUMNS)+7*natoms) or not np.array_equal(data[:, 0], expected):
                raise ValueError("native PIMD saved records disagree with their clock/shape")
        elif end > options["neq"] and (end-options["neq"])//options["interval"] > max(0, entry["start"]-options["neq"])//options["interval"]:
            raise ValueError("native PIMD journal omits production samples")
    if not value["blocks"] or end > options["neq"]+options["nprod"]:
        raise ValueError("invalid native PIMD journal duration")
    return value


def _commit(path, journal, engine, start, steps, options, *, checkpoint_engine=None):
    directory = Path(tempfile.mkdtemp(prefix="chunk-", dir=path.parent))
    before = engine.state()
    checkpoint_before = None if checkpoint_engine is None else checkpoint_engine.state()
    published = False
    try:
        production = start >= options["neq"] and steps > 0
        if steps:
            engine.run(steps, stride=options["interval"] if production else 2**64-1,
                       sample_start=options["neq"] if production else 0,
                       record_centroids=production, record_moments=production,
                       trajectory=directory/"samples.npy" if production else None)
        if checkpoint_engine is not None:
            current = engine.state()
            checkpoint_engine.set_state(current["q"], current["p"], step=current["step"])
            checkpoint_engine.save(directory/"state.npz")
        else:
            engine.save(directory/"state.npz")
        entry = {"start": start, "end": start+steps, "directory": directory.name,
                 "state_sha256": file_sha256(directory/"state.npz"),
                 "samples_sha256": file_sha256(directory/"samples.npy") if production else None}
        changed = {**journal, "blocks": journal["blocks"] + [entry]}
        changed["checksum"] = _checksum(changed)
        write_json(path, changed)  # only this atomic replace commits the block
        published = True
        return changed
    finally:
        if not published:
            engine.restore(before)
            if checkpoint_engine is not None and checkpoint_before is not None:
                checkpoint_engine.restore(checkpoint_before)
            shutil.rmtree(directory)


def _sample_diffusion(task):
    """Run native PILE-L equilibration followed by native NVE production."""
    config, system, output, replica = task["config"], task["system"], Path(task["output"]), task["replica"]
    options = resolve_native_pimd(config, system)
    path = output/"samples"/f"native_pimd_{replica:04d}"/"journal.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = _fingerprint(config, system, {"native_pimd_replica": replica, "files": referenced_files(config)})
    total = options["neq"] + options["nprod"]
    journal = _journal(path, identity, options, system.natoms) if path.exists() else None
    if journal is not None and journal["blocks"][-1]["end"] >= total:
        return _replica_summary(path, journal, options, system, replica, task="diffusion")
    provider = make_task_potential(config["potential"], system)
    sampling_engine = production_engine = None
    try:
        elapsed = 0 if journal is None else journal["blocks"][-1]["end"]
        if elapsed < options["neq"]:
            sampling_engine = _engine(config, system, replica, provider)
            if journal is None:
                journal = _commit(path, {"schema": _SCHEMA, "identity": identity, "blocks": []},
                                  sampling_engine, 0, 0, options)
                elapsed = 0
            else:
                sampling_engine.load(path.parent/journal["blocks"][-1]["directory"]/"state.npz")
            if sampling_engine.state()["step"] != elapsed:
                raise ValueError("native diffusion checkpoint and journal clocks differ")
            while elapsed < options["neq"]:
                count = min(options["checkpoint_steps"], options["neq"] - elapsed)
                checkpoint_engine = None
                if elapsed + count == options["neq"]:
                    production_engine = _engine(config, system, replica, provider, thermostat="none")
                    checkpoint_engine = production_engine
                journal = _commit(path, journal, sampling_engine, elapsed, count, options,
                                  checkpoint_engine=checkpoint_engine)
                elapsed += count
        else:
            production_engine = _engine(config, system, replica, provider, thermostat="none")
            if journal is None:
                journal = _commit(path, {"schema": _SCHEMA, "identity": identity, "blocks": []},
                                  production_engine, 0, 0, options)
                elapsed = 0
            else:
                production_engine.load(path.parent/journal["blocks"][-1]["directory"]/"state.npz")
            if production_engine.state()["step"] != elapsed:
                raise ValueError("native diffusion checkpoint and journal clocks differ")
        if elapsed >= options["neq"]:
            if production_engine is None:
                production_engine = _engine(config, system, replica, provider, thermostat="none")
                production_engine.load(path.parent/journal["blocks"][-1]["directory"]/"state.npz")
            while elapsed < total:
                count = min(options["checkpoint_steps"], total - elapsed)
                journal = _commit(path, journal, production_engine, elapsed, count, options)
                elapsed += count
    finally:
        if callable(getattr(provider, "close", None)):
            provider.close()
    return _replica_summary(path, journal, options, system, replica, task="diffusion")


def _sample(task):
    config, system, output, replica = task["config"], task["system"], Path(task["output"]), task["replica"]
    if config.get("task") == "diffusion":
        return _sample_diffusion(task)
    options = resolve_native_pimd(config, system)
    path = output/"samples"/f"native_pimd_{replica:04d}"/"journal.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = _fingerprint(config, system, {"native_pimd_replica": replica, "files": referenced_files(config)})
    total = options["neq"] + options["nprod"]
    journal = _journal(path, identity, options, system.natoms) if path.exists() else None
    if journal is None or journal["blocks"][-1]["end"] < total:
        provider = make_task_potential(config["potential"], system)
        try:
            engine = _engine(config, system, replica, provider)
            if journal is None:
                journal = _commit(path, {"schema": _SCHEMA, "identity": identity, "blocks": []}, engine, 0, 0, options)
            else:
                engine.load(path.parent/journal["blocks"][-1]["directory"]/"state.npz")
            elapsed = journal["blocks"][-1]["end"]
            if engine.state()["step"] != elapsed:
                raise ValueError("native PIMD checkpoint and journal clocks differ")
            while elapsed < total:
                boundary = options["neq"] if elapsed < options["neq"] else total
                count = min(options["checkpoint_steps"], boundary-elapsed)
                journal = _commit(path, journal, engine, elapsed, count, options)
                elapsed += count
        finally:
            if callable(getattr(provider, "close", None)):
                provider.close()
    return _replica_summary(path, journal, options, system, replica)


def _replica_summary(path, journal, options, system, replica, *, task="pimd"):
    from ..analysis.diagnostics import correlated_mean_diagnostics
    # Only scalar diagnostic series are assembled. Coordinates/velocities and
    # radii remain memory-mapped per chunk; MPI returns summaries, not trajectories.
    scalars, frames, count = [], [], 0
    radii_sum = np.zeros(system.natoms)
    for entry in journal["blocks"]:
        if entry["samples_sha256"] is None:
            continue
        has_frames = max(0, entry["end"]-options["neq"])//options["interval"] > max(0, entry["start"]-options["neq"])//options["interval"]
        data = np.load(path.parent/entry["directory"]/"samples.npy", mmap_mode="r" if has_frames else None, allow_pickle=False)
        if not len(data):
            continue
        scalars.append(np.array(data[:, [2, 3, 4, 5, 6, 7]]))
        if task == "diffusion":
            frames.append(np.array(data))
        radii_sum += data[:, len(_COLUMNS)+6*system.natoms:].sum(axis=0)
        count += len(data)
    values = np.ascontiguousarray(np.concatenate(scalars))
    obs = np.ascontiguousarray(np.column_stack((values[:, :3], values[:, 0]+values[:, 2], values[:, 3])))
    if not np.isfinite(obs).all() or not np.isfinite(radii_sum).all():
        raise ValueError("nonfinite native PIMD observables")
    result = {"replica": replica, "frames": count, "means": obs.mean(axis=0).tolist(),
              "radii": (radii_sum/count).tolist(), "journal": str(path),
              "checkpoint": str(path.parent/journal["blocks"][-1]["directory"]/"state.npz"),
              "native_blocks": len(journal["blocks"])-1,
              "diagnostics": {name: correlated_mean_diagnostics(obs[:, i]) for i, name in enumerate(_NAMES)}}
    if task == "diffusion":
        data = np.ascontiguousarray(np.concatenate(frames))
        natoms = system.natoms
        offset = len(_COLUMNS)
        result.update(
            task="diffusion",
            sample_steps=data[:, 0].astype(np.int64),
            times=(data[:, 0] - options["neq"]) * options["production_dt"],
            positions=data[:, offset:offset + 3 * natoms].reshape(-1, natoms, 3),
            velocities=data[:, offset + 3 * natoms:offset + 6 * natoms].reshape(-1, natoms, 3),
            radii_gyration_squared=data[:, offset + 6 * natoms:].copy(),
            observables=obs,
        )
    if system.cell is not None:
        result["volume_mean_bohr3"] = float(values[:, 4].mean())
        result["pressure_mean_au"] = float(values[:, 5].mean())
    return result


def _diffusion_summary(streams, config, system, output, return_data=False):
    from ..analysis.diagnostics import nve_energy_diagnostics
    from ..analysis.diffusion import analyze_diffusion

    times = streams[0]["times"]
    if any(not np.array_equal(s["times"], times) for s in streams):
        raise ValueError("native diffusion replicas need the same completed sample clock")
    output = Path(output)
    trajectories = []
    for stream in streams:
        path = output/"samples"/f"native_pimd_{stream['replica']:04d}"/"trajectory.npz"
        metadata = json.dumps({
            "task": "diffusion", "positions": "unwrapped_ring_polymer_centroids",
            "position_unit": "bohr", "velocity_unit": "bohr/atomic_time",
            "time_unit": "atomic_time", "energy_unit": "hartree",
            "pressure_unit": "hartree/bohr^3", "cell_unit": "bohr",
            "equilibration_steps": config["sampling"]["equilibration_steps"],
            "sample_interval_steps": config["sampling"]["sample_interval"],
            "production_timestep_au": config["dt_au"],
            "beads": config["beads"], "beta_au": config["beta"],
            "observable_names": list(_NAMES),
            "system": system.to_dict(),
        })
        atomic_savez(path, kind="PyRPMD.materials-trajectory.v1",
                     positions=stream["positions"], velocities=stream["velocities"],
                     times=stream["times"], sample_steps=stream["sample_steps"],
                     observables=stream["observables"],
                     radii_gyration_squared=stream["radii_gyration_squared"],
                     metadata=metadata)
        trajectories.append(str(path))
    result = {
        "task": "diffusion", "status": "complete",
        "converged": False, "replicas": len(streams),
        "samples_per_replica": [s["frames"] for s in streams],
        "phase": "unthermostatted_rpmd_production", "backend": "native-cxx17",
        "python_step_callbacks": 0, "sampling_thermostat": "pile-l",
        "integrator": loop_options(config)["integrator"],
        "coordinate_convention": "continuous_unwrapped_ring_polymer_centroids",
        "cell": system.to_dict().get("cell"), "pbc": system.to_dict().get("pbc", [False, False, False]),
        "ensemble": "fixed_cell", "fixed_atoms": list(system.fixed_atoms),
        "mobile_atoms": int(system.mobile_mask.sum()), "trajectory_storage": ["native_pimd_journal"] * len(streams),
        "propagation": [{"native_steps": config["sampling"]["equilibration_steps"] + config["sampling"]["production_steps"],
                         "python_steps": 0, "native_blocks": s["native_blocks"]} for s in streams],
        "files": {"journals": [s["journal"] for s in streams],
                  "checkpoints": [s["checkpoint"] for s in streams], "trajectories": trajectories},
        "warnings": ["Native diffusion is an opt-in E2 contract with a new thermostat/RNG map.",
                      "Bead count, timestep, system size and statistical convergence have not been certified."],
    }
    diagnostics = nve_energy_diagnostics(
        [s["observables"][:, 4] for s in streams], times,
        beads=config["beads"], beta=config["beta"],
    )
    result["nve_energy_diagnostics"] = diagnostics
    if not diagnostics["within_tolerance"]:
        result["warnings"].append("Saved production frames exceed the NVE energy diagnostic tolerance.")
    table = config.get("diffusion", {})
    params = {key: table[key] for key in ("fit_range", "max_lag", "bootstrap", "seed", "block_length") if key in table}
    params.setdefault("seed", int(config.get("seed", 0)))
    selected_atoms = options_atom_indices(config, system)
    params.update(atom_indices=selected_atoms,
                  axes=config.get("diffusion", {}).get("axes", [0, 1, 2]))
    refs = config.get("diffusion", {}).get("reference_atoms")
    if refs is not None:
        params.update(reference_positions=np.array([s["positions"][:, refs] for s in streams]),
                      reference_velocities=np.array([s["velocities"][:, refs] for s in streams]),
                      reference_masses=system.masses[refs])
    diffusion = analyze_diffusion(np.array([s["positions"] for s in streams]), times,
                                 velocities=np.array([s["velocities"] for s in streams]), **params)
    diffusion["input"]["atoms"] = system.natoms
    diffusion["input"]["atom_indices"] = options_atom_indices(config, system)
    result["diffusion"] = diffusion
    result["analysis_status"] = diffusion["status"]
    result["diffusion_m2_s"] = diffusion["diffusion_m2_s"]
    result["diffusion_tensor_m2_s"] = diffusion["diffusion_tensor_m2_s"]
    result["files"]["diffusion"] = str(output/"results"/"diffusion.json")
    write_json(output/"results"/"diffusion.json", diffusion)
    write_json(output/"summary.json", result)
    return {"summary": result, "streams": streams} if return_data else result


def options_atom_indices(config, system):
    table = config.get("diffusion", {})
    if "atom_indices" in table:
        return [int(i) for i in table["atom_indices"]]
    return [i for i, symbol in enumerate(system.symbols)
            if symbol in ("H", "D", "T") and system.mobile_mask[i]]


def _summary(streams, config, system, output, return_data=False):
    if config.get("task") == "diffusion":
        return _diffusion_summary(streams, config, system, output, return_data)
    means = np.asarray([s["means"] for s in streams])
    errors = np.std(means, axis=0, ddof=1)/np.sqrt(len(streams)) if len(streams)>1 else None
    bmode = config.get("barostat", {}).get("mode", "none")
    summary = {"task": "pimd", "status": "complete", "converged": False,
        "phase": "equilibrium_sampling", "backend": "native-cxx17", "python_step_callbacks": 0,
        "sampling_thermostat": config["sampling"].get("thermostat", "pile-l"), "barostat": bmode,
        "sampling_splitting": sampling_splitting(
            ({"bcocb": "cayley-bcb", "baoab": "normal-mode-bab"}.get(config["sampling"].get("integrator"),
              loop_options(config)["integrator"]) if config["sampling"].get("thermostat", "pile-l") == "pile-l"
             else loop_options(config)["integrator"]),
            config["sampling"].get("thermostat", "pile-l"), bmode, middle=mixed_sampling_method(config)).upper(),
        "integrator": loop_options(config)["integrator"], "replicas": len(streams),
        "samples_per_replica": [s["frames"] for s in streams], "energy_unit": "hartree",
        "estimators": {name: {"mean": float(means[:, i].mean()), "standard_error_between_replicas":
                       None if errors is None else float(errors[i])} for i, name in enumerate(_NAMES)},
        "radius_of_gyration_squared_bohr2": np.mean([s["radii"] for s in streams], axis=0).tolist(),
        "sampling_diagnostics": [{"replica": s["replica"], "observables": s["diagnostics"]} for s in streams],
        "files": {"journals": [s["journal"] for s in streams], "checkpoints": [s["checkpoint"] for s in streams]},
        "propagation": [{"native_steps": config["sampling"]["equilibration_steps"]+config["sampling"]["production_steps"],
                         "python_steps": 0, "native_blocks": s["native_blocks"]} for s in streams],
        "record_layout": {"scalar_columns": _COLUMNS, "tail": "centroids[N,3], velocities[N,3], radius_squared[N]"},
        "warnings": ["Bead, timestep, equilibration, system-size and statistical convergence have not been certified.",
                     "PIMD sampling times are not physical dynamics. H_P excludes piston/thermostat heat and is not an NPT conserved quantity.",
                     "Propagation memory excludes trajectory history; offline correlation diagnostics retain scalar time series."]}
    bath_warning = global_thermostat_warning(config["sampling"].get("thermostat", "pile-l"), config["beads"])
    if bath_warning is not None:
        summary["warnings"].append(bath_warning)
    if config["sampling"]["equilibration_steps"] == 0:
        summary["warnings"].append("No equilibration: collapsed-ring initial geometry is not an equilibrium sample.")
    if system.cell is not None:
        summary.update(initial_cell_bohr=system.cell.tolist(), pbc=list(map(bool,system.pbc)),
                       volume_mean_bohr3=float(np.mean([s["volume_mean_bohr3"] for s in streams])),
                       pressure_mean_au=float(np.mean([s["pressure_mean_au"] for s in streams])),
                       pressure_estimator={
                           "name": "instantaneous_centroid_momentum_virial",
                           "sample_column": "pressure", "summary_field": "pressure_mean_au",
                           "role": "piston_feedback" if bmode != "none" else "fixed_cell_diagnostic",
                           "uses_configured_temperature_kinetic_term": False,
                           "configured_temperature_cv_mean_equivalence":
                               "justified by canonical centroid kinetic moments conditional on coordinates and volume; not assumed for PIGLET",
                       })
    write_json(Path(output)/"summary.json", summary)
    return {"summary": summary, "streams": streams} if return_data else summary


def run_native_pimd(config, system, output, executor=None, return_data=False):
    from .sampling import validate_andersen_schedule
    validate_andersen_schedule(config, "native_pimd")
    options = resolve_native_pimd(config, system)
    tasks = [dict(config=copy.deepcopy(config), system=system, output=str(output), replica=r) for r in range(options["replicas"])]
    streams = executor.map(_sample, tasks) if executor is not None else run_tasks(_sample, tasks, config.get("parallel", {}))
    return root_call(_summary, list(streams) if streams is not None else streams, config, system, str(output), return_data)


def _reanalyze_native_diffusion(config, system, output, return_data=False):
    """Reanalyze committed native diffusion journals without constructing a PES."""
    if config.get("task") != "diffusion":
        raise ValueError("native diffusion reanalysis requires task='diffusion'")
    options = resolve_native_pimd(config, system)
    streams = []
    for replica in range(options["replicas"]):
        path = Path(output)/"samples"/f"native_pimd_{replica:04d}"/"journal.json"
        if not path.exists():
            raise ValueError(f"native diffusion journal is missing: {path}")
        identity = _fingerprint(config, system, {"native_pimd_replica": replica, "files": referenced_files(config)})
        journal = _journal(path, identity, options, system.natoms)
        streams.append(_replica_summary(path, journal, options, system, replica, task="diffusion"))
    return _diffusion_summary(streams, config, system, output, return_data)


def _check_native_pimd(config, system, provider):
    """Check the actual native map/PES without propagating or creating output."""
    from ..potentials.base import PESResult, validate_result

    engine = None
    try:
        engine = _engine(config, system, 0, provider)
        initial = engine.state()
        checked = validate_result(PESResult(
            initial["energy"], initial["force"], virial=initial["virial"],
        ), (config["beads"], system.natoms, 3))
        if any(not np.isfinite(initial[key]).all()
               for key in ("q", "p", "aux", "cell", "cell_p")):
            raise FloatingPointError("nonfinite native PIMD initial state")
        return {
            "engine": "native", "backend": "native-cxx17",
            "energy_hartree": checked.energy.tolist(),
            "status": checked.status.tolist(),
            "initial_virial_hartree": checked.virial.tolist(),
            "step": initial["step"],
            "force_evaluations": initial["force_evaluations"],
        }
    finally:
        # Release the native handle before the caller closes its provider.
        engine = None
