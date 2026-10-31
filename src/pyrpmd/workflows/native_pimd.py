"""Opt-in PIMD workflow: native trajectory blocks, immutable streamed records.

The journal is the commit point. A crash before publication can leave an
unreferenced transaction directory, but cannot advance the accepted trajectory.
Restart reads only committed records. Python runs at checkpoint/phase boundaries,
never at the observation cadence. This is not a diffusion/rate propagator.
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from ..io import file_sha256, jsonable, referenced_files, write_json
from ..io import fingerprint as _fingerprint
from ..native_ensemble import (
    _COLUMNS,
    _THERMOSTATS,
    NativeEnsemble,
    NativePeriodicPES,
    ensemble_from_config,
)
from ..parallel import make_task_potential, root_call, run_tasks
from ..propagation import loop_options
from ..validation import checked_integer

_SCHEMA = "pyrpmd.native-pimd-journal.v1"
_NAMES = ("potential_energy", "kinetic_primitive", "kinetic_centroid_virial",
          "internal_energy_centroid_virial", "ring_polymer_hamiltonian")


def resolve_native_pimd(config, system):
    """Preflight the opt-in route; the native constructor checks all matrices."""
    if config.get("task") != "pimd":
        raise ValueError("sampling.engine='native' currently requires task='pimd'")
    s = config.get("sampling", {})
    loop = loop_options(config)
    if loop["integrator"] not in ("cayley-bcb", "normal-mode-bab"):
        raise ValueError("native PIMD requires Cayley or normal-mode integration")
    if loop["loop_backend"] == "python" or config.get("native") is False or config.get("dynamics", {}).get("native") is False:
        raise ValueError("native PIMD cannot disable the native engine")
    mode = s.get("thermostat", "pile-l")
    if mode not in _THERMOSTATS or mode in ("none", "trpmd"):
        raise ValueError("equilibrium native PIMD requires a sampling thermostat, not none/TRPMD")
    expected = "bcocb" if loop["integrator"] == "cayley-bcb" else "baoab"
    if s.get("integrator", expected) != expected:
        raise ValueError(f"native PIMD sampling.integrator must be {expected}")
    if any(config.get(key) for key in ("umbrella", "metadynamics", "reaction", "electronic_friction")):
        raise ValueError("native PIMD is unbiased; reaction/bias settings are unsupported")
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
    if options["nprod"] // options["interval"] < 2:
        raise ValueError("native PIMD production needs at least two saved frames")
    b = config.get("barostat", {})
    native_barostat_fields = {"mode", "pressure", "tau", "gamma", "compressibility", "cell_mask",
                              "reference_cell", "reference_cell_unit", "stress"}
    if "barostat" in config and ("mode" not in b or set(b) - native_barostat_fields):
        raise ValueError("native PIMD barostat requires explicit mode and native pressure/tau/gamma controls; legacy method/time_constant/friction controls use sampling.engine='legacy'")
    if b.get("mode", "none") not in ("none", "bzp", "mttk", "mtk", "rgb", "scr", "isotropic", "flexible", "anisotropic", "stochastic-rescaling"):
        raise ValueError("unsupported native PIMD barostat; Suzuki-Chin action is not implemented")
    if b.get("mode", "none") != "none":
        if not all(system.pbc) or system.fixed_atoms:
            raise ValueError("native NPT requires 3D PBC and all atoms mobile")
        if mode == "piglet":
            raise ValueError("PIGLET NPT is not validated")
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


def _engine(config, system, replica, provider):
    s = config["sampling"]
    t = copy.deepcopy(s.get("thermostat_params", {}))
    if "mode" in t:
        raise ValueError("set thermostat mode in sampling.thermostat, not thermostat_params")
    t["mode"] = s.get("thermostat", "pile-l")
    if "gamma_centroid" in s:
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
    spec = dict(positions=system.geometry.tolist(), masses=system.masses.tolist(),
                beta=config["beta"], beads=config["beads"], timestep=config["dt_au"],
                integrator=loop_options(config)["integrator"], thermostat=t,
                barostat=copy.deepcopy(config.get("barostat", {"mode": "none"})),
                seed=seed, fixed_atoms=list(system.fixed_atoms), pbc=list(map(bool, system.pbc)),
                potential=potential if isinstance(potential, dict) else {"kind": "free"})
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


def _commit(path, journal, engine, start, steps, options):
    directory = Path(tempfile.mkdtemp(prefix="chunk-", dir=path.parent))
    before = engine.state()
    published = False
    try:
        production = start >= options["neq"] and steps > 0
        if steps:
            engine.run(steps, stride=options["interval"] if production else 2**64-1,
                       sample_start=options["neq"] if production else 0,
                       record_centroids=production, record_moments=production,
                       trajectory=directory/"samples.npy" if production else None)
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
            shutil.rmtree(directory)


def _sample(task):
    config, system, output, replica = task["config"], task["system"], Path(task["output"]), task["replica"]
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


def _replica_summary(path, journal, options, system, replica):
    from ..analysis.diagnostics import correlated_mean_diagnostics
    # Only scalar diagnostic series are assembled. Coordinates/velocities and
    # radii remain memory-mapped per chunk; MPI returns summaries, not trajectories.
    scalars, count = [], 0
    radii_sum = np.zeros(system.natoms)
    for entry in journal["blocks"]:
        if entry["samples_sha256"] is None:
            continue
        has_frames = max(0, entry["end"]-options["neq"])//options["interval"] > max(0, entry["start"]-options["neq"])//options["interval"]
        data = np.load(path.parent/entry["directory"]/"samples.npy", mmap_mode="r" if has_frames else None, allow_pickle=False)
        if not len(data):
            continue
        scalars.append(np.array(data[:, [2, 3, 4, 5, 6, 7]]))
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
    if system.cell is not None:
        result["volume_mean_bohr3"] = float(values[:, 4].mean())
        result["pressure_mean_au"] = float(values[:, 5].mean())
    return result


def _summary(streams, config, system, output, return_data=False):
    means = np.asarray([s["means"] for s in streams])
    errors = np.std(means, axis=0, ddof=1)/np.sqrt(len(streams)) if len(streams)>1 else None
    bmode = config.get("barostat", {}).get("mode", "none")
    summary = {"task": "pimd", "status": "complete", "physical_converged": False, "converged": False,
        "phase": "equilibrium_sampling", "backend": "native-cxx17", "python_step_callbacks": 0,
        "sampling_thermostat": config["sampling"].get("thermostat", "pile-l"), "barostat": bmode,
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
    if config["sampling"]["equilibration_steps"] == 0:
        summary["warnings"].append("No equilibration: collapsed-ring initial geometry is not an equilibrium sample.")
    if system.cell is not None:
        summary.update(initial_cell_bohr=system.cell.tolist(), pbc=list(map(bool,system.pbc)),
                       volume_mean_bohr3=float(np.mean([s["volume_mean_bohr3"] for s in streams])),
                       pressure_mean_au=float(np.mean([s["pressure_mean_au"] for s in streams])))
    write_json(Path(output)/"summary.json", summary)
    return {"summary": summary, "streams": streams} if return_data else summary


def run_native_pimd(config, system, output, executor=None, return_data=False):
    options = resolve_native_pimd(config, system)
    tasks = [dict(config=copy.deepcopy(config), system=system, output=str(output), replica=r) for r in range(options["replicas"])]
    streams = executor.map(_sample, tasks) if executor is not None else run_tasks(_sample, tasks, config.get("parallel", {}))
    return root_call(_summary, list(streams) if streams is not None else streams, config, system, str(output), return_data)
