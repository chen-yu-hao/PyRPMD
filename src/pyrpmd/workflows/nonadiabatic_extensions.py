"""Restartable ensembles for explicitly selected spin and hopping methods.

MASH active populations and amplitude/W-spin density estimators have separate
schemas.  Every checkpoint owns a complete trajectory, its electronic gauge,
all three random streams, and the observations committed with that state.
"""
from __future__ import annotations

import copy
import csv
import io
import json
from pathlib import Path

import numpy as np

from ..advanced_surface_hopping import RPSHBAState, RPSHDMState
from ..checkpoint import atomic_savez, load_checkpoint, CheckpointError
from ..io import atomic_write_text, fingerprint, jsonable, write_json
from ..nonadiabatic import validate_diabatic_result
from ..parallel import execution_plan, run_tasks
from ..propagation import scientific_config
from ..rp_mash import RPMASHState
from ..spin_mapping import SMNRPMDState, W_RADIUS
from ..surface_hopping import RPSHState
from ..system import System
from .nonadiabatic import _streams


_KIND = "PyRPMD.extended_nonadiabatic_trajectory.v1"
_RESULT_KIND = "PyRPMD.extended_nonadiabatic_ensemble.v1"
_CLASSES = {"sm-nrpmd": SMNRPMDState, "rp-mash": RPMASHState,
            "rpsh-ca": RPSHState, "rpsh-ba": RPSHBAState,
            "rpsh-dm": RPSHDMState, "soc-rpsh": RPSHState}
_ROLES = ("nuclear", "electronic", "dynamics")


def _driver(config, system, rng):
    from ..nonadiabatic_registry import make_dynamics
    driver = make_dynamics(config, system, rng)
    if config["method"] not in _CLASSES:
        raise ValueError("unsupported extended nonadiabatic method")
    if config["method"] != "sm-nrpmd" and driver.rng is not rng:
        driver.potential.close()
        raise ValueError("dynamics factory must preserve the supplied trajectory RNG object")
    return driver


def _history_names(method):
    result = ["steps", "population", "energy", "norm", "centroid_q", "centroid_p"]
    if method != "rp-mash":
        result.append("density")
    if method != "sm-nrpmd":
        result.append("active")
    if method not in ("sm-nrpmd", "rp-mash"):
        result.append("amplitude")
    return tuple(result)


def _state_names(method):
    if method == "sm-nrpmd":
        return ("spin",)
    if method == "rp-mash":
        return ("spin", "active_surface", "sampling")
    if method == "rpsh-ba":
        return ("coefficients", "active_surface", "basis_vectors")
    return ("coefficients", "active_surface")


def _initialize(config, system, dynamics, streams):
    from ..nonadiabatic_registry import initialize_state
    return initialize_state(config, system, dynamics, streams[0], streams[1])


def _execution_metadata(config):
    options = config["dynamics"]
    return {"electronic_backend": options.get("electronic_backend", "numpy"),
            "electronic_device": options.get("electronic_device", "cpu")}


def _observe(dynamics, state, method):
    dynamics._check(state)
    if isinstance(state, SMNRPMDState):
        norm = np.sum(state.spin**2, axis=-1)
    elif isinstance(state, RPMASHState):
        if state.sampling != "population":
            raise ValueError("MASH ensemble workflow requires population importance sampling")
        norm = np.array([state.spin @ state.spin])
    else:
        norm = np.atleast_1d(np.sum(np.abs(state.coefficients)**2, axis=-1))
    record = {"energy": np.float64(dynamics.energy(state)), "norm": np.asarray(norm, dtype=np.float64),
              "centroid_q": state.q.mean(axis=0), "centroid_p": state.p.mean(axis=0)}
    if method == "rp-mash":
        record["population"] = np.asarray(dynamics.populations(state), dtype=np.float64)
    else:
        rho = np.asarray(dynamics.density_matrix(state), dtype=np.complex128)
        record.update(density=rho, population=rho.diagonal().real.copy())
    if method != "sm-nrpmd":
        record["active"] = np.int64(state.active_surface)
    if method not in ("sm-nrpmd", "rp-mash"):
        record["amplitude"] = np.asarray(dynamics.populations(state), dtype=np.float64)
    if not all(np.isfinite(value).all() for value in record.values()):
        raise FloatingPointError("nonfinite extended nonadiabatic observation")
    return record


def _append(history, dynamics, state, method):
    record = _observe(dynamics, state, method)
    history["steps"].append(np.int64(state.step))
    for key, value in record.items():
        history[key].append(np.array(value, copy=True))


def _save(path, config, identity, index, state, streams, history):
    metadata = {"format": _KIND, "physics_fingerprint": identity, "trajectory": int(index),
                "method": config["method"], "state_class": type(state).__name__,
                **_execution_metadata(config),
                "rngs": {role: copy.deepcopy(rng.bit_generator.state) for role, rng in zip(_ROLES, streams)}}
    arrays = {"metadata": np.asarray(json.dumps(jsonable(metadata), sort_keys=True, allow_nan=False)),
              "q": state.q, "p": state.p, "step": np.asarray(state.step, dtype=np.int64),
              "time": np.asarray(state.time, dtype=np.float64),
              **{"history_"+key: np.asarray(value) for key, value in history.items()}}
    for key in _state_names(config["method"]):
        value = getattr(state, key)
        arrays[key] = np.asarray(value, dtype=np.int64) if key == "active_surface" else np.asarray(value)
    atomic_savez(path, kind=_KIND, **arrays)


def _typed(value, shape, dtype, label):
    if value.shape != shape or value.dtype != np.dtype(dtype) or not np.isfinite(value).all():
        raise CheckpointError(f"invalid checkpoint {label} shape, dtype or finite values")


def _load(path, config, identity, index, dynamics, streams):
    method = config["method"]
    with load_checkpoint(path, expected_kind=_KIND, require_integrity=True) as saved:
        data = {key: saved[key] for key in saved.files}
    inventory = ({"metadata", "q", "p", "step", "time"} | set(_state_names(method))
                 | {"history_"+name for name in _history_names(method)})
    if set(data) != inventory:
        raise CheckpointError("extended nonadiabatic checkpoint payload inventory mismatch")
    raw = data["metadata"]
    if raw.shape != () or raw.dtype.kind != "U":
        raise CheckpointError("invalid checkpoint metadata dtype")
    try:
        metadata = json.loads(str(raw.item()))
    except (TypeError, ValueError) as error:
        raise CheckpointError("invalid checkpoint metadata JSON") from error
    if (not isinstance(metadata, dict)
            or set(metadata) != {"format", "physics_fingerprint", "trajectory", "method", "state_class", "rngs",
                                 "electronic_backend", "electronic_device"}
            or metadata["format"] != _KIND or metadata["physics_fingerprint"] != identity
            or type(metadata["trajectory"]) is not int or metadata["trajectory"] != index
            or metadata["method"] != method or metadata["state_class"] != _CLASSES[method].__name__):
        raise CheckpointError("extended nonadiabatic checkpoint identity mismatch")
    if (metadata["electronic_backend"] not in ("numpy", "native", "torch")
            or not isinstance(metadata["electronic_device"], str) or not metadata["electronic_device"]):
        raise CheckpointError("invalid checkpoint electronic execution metadata")
    _typed(data["step"], (), "int64", "step")
    _typed(data["time"], (), "float64", "time")
    step, time = int(data["step"]), float(data["time"])
    target, stride = config["ensemble"]["steps"], config["ensemble"]["sample_stride"]
    if not 0 <= step <= target or not np.isclose(time, step*config["dt_au"], rtol=2e-11, atol=1e-12):
        raise CheckpointError("invalid checkpoint clock")
    beads, atoms, states = config["beads"], len(dynamics.masses), dynamics.nstates
    shape = (beads, atoms, 3)
    for key in ("q", "p"):
        _typed(data[key], shape, "float64", key)
    if method in ("sm-nrpmd", "rp-mash"):
        _typed(data["spin"], (beads, 3) if method == "sm-nrpmd" else (3,), "float64", "spin")
    else:
        _typed(data["coefficients"], (beads, states) if method == "rpsh-dm" else (states,), "complex128", "coefficients")
    if method != "sm-nrpmd":
        _typed(data["active_surface"], (), "int64", "active_surface")
        if not 0 <= int(data["active_surface"]) < states:
            raise CheckpointError("checkpoint active surface is out of range")
    if method == "rpsh-ba":
        _typed(data["basis_vectors"], (beads, states, states), "complex128", "basis_vectors")
    if method == "rp-mash" and (data["sampling"].shape != () or data["sampling"].dtype.kind != "U"
                                  or str(data["sampling"].item()) != "population"):
        raise CheckpointError("checkpoint MASH sampling must be population")
    common = (data["q"], data["p"], step, time)
    try:
        if method == "sm-nrpmd":
            state = SMNRPMDState(*common, data["spin"])
        elif method == "rp-mash":
            state = RPMASHState(*common, data["spin"], int(data["active_surface"]), str(data["sampling"].item()))
        elif method == "rpsh-ba":
            state = RPSHBAState(*common, data["coefficients"], int(data["active_surface"]), data["basis_vectors"])
        else:
            state = _CLASSES[method](*common, data["coefficients"], int(data["active_surface"]))
        dynamics._check(state)
        if method == "rpsh-ba":
            # Unitarity alone does not establish that a saved frame belongs to
            # the current q.  Check the ordered bare PES eigenbasis, including
            # frames at checkpoints between observation times.
            result = validate_diabatic_result(dynamics.potential.evaluate(state.q), shape, states)
            frame = state.basis_vectors
            transformed = frame.conj().swapaxes(-1, -2) @ result.matrix @ frame
            expected = np.linalg.eigvalsh(result.matrix)[..., :, None] * np.eye(states)
            if not np.allclose(transformed, expected, rtol=2e-11, atol=1e-8):
                raise ValueError("BA transported basis does not diagonalize the current matrix")
    except Exception as error:
        raise CheckpointError("invalid extended nonadiabatic checkpoint state") from error
    expected_steps = np.arange(0, step+1, stride, dtype=np.int64)
    if step == target and target % stride:
        expected_steps = np.append(expected_steps, np.int64(target))
    frames = len(expected_steps)
    norm_count = beads if method in ("sm-nrpmd", "rpsh-dm") else 1
    shapes = {"steps": (frames,), "population": (frames, states), "energy": (frames,),
              "norm": (frames, norm_count), "centroid_q": (frames, atoms, 3), "centroid_p": (frames, atoms, 3)}
    if method != "rp-mash":
        shapes["density"] = (frames, states, states)
    if method != "sm-nrpmd":
        shapes["active"] = (frames,)
    if method not in ("sm-nrpmd", "rp-mash"):
        shapes["amplitude"] = (frames, states)
    history = {}
    for key, expected_shape in shapes.items():
        value = data["history_"+key]
        dtype = "int64" if key in ("steps", "active") else "complex128" if key == "density" else "float64"
        _typed(value, expected_shape, dtype, "history_"+key)
        history[key] = [row.copy() for row in value]
    if not np.array_equal(data["history_steps"], expected_steps):
        raise CheckpointError("checkpoint observation schedule mismatch")
    if not np.allclose(data["history_norm"], 1., rtol=0., atol=1e-8):
        raise CheckpointError("checkpoint electronic norm history is invalid")
    population = data["history_population"]
    if not np.allclose(population.sum(axis=-1), 1., rtol=0., atol=1e-8):
        raise CheckpointError("checkpoint population must sum to one")
    if method != "rp-mash":
        rho = data["history_density"]
        if (not np.allclose(rho, rho.conj().swapaxes(-1, -2), rtol=0., atol=1e-8)
                or not np.allclose(np.trace(rho, axis1=-2, axis2=-1), 1., rtol=0., atol=1e-8)):
            raise CheckpointError("checkpoint density must be Hermitian with unit trace")
        if not np.allclose(population, rho.diagonal(axis1=-2, axis2=-1).real, rtol=0., atol=1e-8):
            raise CheckpointError("checkpoint population/density mismatch")
        eigenvalues = np.linalg.eigvalsh(rho)
        lower = 0.5-W_RADIUS if method == "sm-nrpmd" else 0.
        upper = 0.5+W_RADIUS if method == "sm-nrpmd" else 1.
        if np.any(eigenvalues < lower-1e-8) or np.any(eigenvalues > upper+1e-8):
            raise CheckpointError("checkpoint density eigenvalues violate the method's estimator bounds")
        if method in ("rpsh-ca", "soc-rpsh") and not np.allclose(rho@rho, rho, rtol=0., atol=1e-8):
            raise CheckpointError("checkpoint CA density must be a pure amplitude density")
    if method != "sm-nrpmd":
        active = data["history_active"]
        if np.any(active < 0) or np.any(active >= states):
            raise CheckpointError("checkpoint active surface history is out of range")
        if method == "rp-mash":
            if not np.array_equal(population, np.eye(states)[active]):
                raise CheckpointError("checkpoint MASH population must equal active-surface indicators")
        else:
            amplitude = data["history_amplitude"]
            if (np.any(amplitude < -1e-8) or np.any(amplitude > 1.+1e-8)
                    or not np.allclose(amplitude.sum(axis=-1), 1., rtol=0., atol=1e-8)):
                raise CheckpointError("checkpoint adiabatic amplitude population is invalid")
    if int(expected_steps[-1]) == state.step:
        try:
            observed = _observe(dynamics, state, method)
        except Exception as error:
            raise CheckpointError("checkpoint final observation cannot be re-evaluated") from error
        for key, value in observed.items():
            previous = data["history_"+key][-1]
            equal = np.array_equal(value, previous) if key == "active" else np.allclose(value, previous, rtol=2e-11, atol=1e-8)
            if not equal:
                raise CheckpointError(f"checkpoint state/final observation mismatch for {key}")
    # Decode every random stream on temporary generators.  Nothing is restored
    # to the live factory RNG before all state/history/phase checks have passed.
    if not isinstance(metadata["rngs"], dict) or set(metadata["rngs"]) != set(_ROLES):
        raise CheckpointError("checkpoint random stream inventory mismatch")
    restored = []
    try:
        for role in _ROLES:
            temporary = np.random.default_rng()
            temporary.bit_generator.state = metadata["rngs"][role]
            restored.append(copy.deepcopy(temporary.bit_generator.state))
    except (TypeError, ValueError, KeyError) as error:
        raise CheckpointError("invalid checkpoint random generator state") from error
    for rng, value in zip(streams, restored):
        rng.bit_generator.state = value
    return state, history


def _trajectory(task):
    config, system_data, directory, index = task
    system = System.from_dict(system_data)
    streams = _streams(config, index)
    dynamics = None
    state = None
    path = Path(directory)/"checkpoints"/f"nonadiabatic_{index:06d}.npz"
    failure = Path(directory)/"logs"/f"nonadiabatic_{index:06d}_failure.json"
    try:
        dynamics = _driver(config, system, streams[2])
        identity = fingerprint(scientific_config(config), system)
        if path.exists():
            if not config.get("resume", False):
                raise ValueError("existing trajectory requires resume")
            state, history = _load(path, config, identity, index, dynamics, streams)
        else:
            state = _initialize(config, system, dynamics, streams)
            history = {key: [] for key in _history_names(config["method"])}
            _append(history, dynamics, state, config["method"])
            _save(path, config, identity, index, state, streams, history)
        target = config["ensemble"]["steps"]
        stride, block = config["ensemble"]["sample_stride"], config["dynamics"]["block_steps"]
        while state.step < target:
            dynamics.step(state)
            if state.step % stride == 0 or state.step == target:
                _append(history, dynamics, state, config["method"])
            if state.step % block == 0 or state.step == target:
                _save(path, config, identity, index, state, streams, history)
        if failure.exists():
            failure.unlink()
        return str(path)
    except BaseException as error:
        write_json(failure, {"trajectory": int(index), "last_in_memory_step": None if state is None else int(state.step),
                             "error_type": type(error).__name__, "error": str(error), "checkpoint": str(path),
                             "resume_from_last_complete_checkpoint": True})
        raise
    finally:
        if dynamics is not None:
            dynamics.potential.close()


def _sem(values):
    return None if len(values) < 2 else np.std(values, axis=0, ddof=1)/np.sqrt(len(values))


def _estimator_metadata(method):
    mash = method == "rp-mash"
    result = {"population_basis": "centroid_adiabatic" if mash else "fixed_diabatic",
              "population_estimator": "active_surface_indicator" if mash else "signed_w_spin" if method == "sm-nrpmd" else "amplitude_density",
              "density_basis": None if mash else "fixed_diabatic", "coherence_available": not mash}
    if method != "sm-nrpmd":
        result["active_surface_basis"] = "common_bead_adiabatic" if method == "rpsh-ba" else "centroid_adiabatic"
    if method not in ("sm-nrpmd", "rp-mash"):
        result["amplitude_population_basis"] = "common_bead_adiabatic" if method == "rpsh-ba" else "centroid_adiabatic"
    if mash:
        result["electronic_initial_sampling"] = "adiabatic_hemisphere_importance_abs_Sz"
    return result


def _write_results(directory, config, identity, histories):
    method = config["method"]
    stack = lambda key: np.asarray([value[key] for value in histories])
    steps = np.asarray(histories[0]["steps"], dtype=np.int64)
    population, energy, norm = stack("population"), stack("energy"), stack("norm")
    statistics = _estimator_metadata(method)
    reason = "fewer_than_two_trajectories" if len(histories) < 2 else None
    metadata = {"format": _RESULT_KIND, "method": method, "physics_fingerprint": identity,
                "independent_trajectories": True, "statistics_axis": "trajectory",
                "time_unit": "atomic", "energy_unit": "hartree", "physical_converged": False,
                "sem_unavailable_reason": reason, **statistics, **_execution_metadata(config)}
    arrays = {"metadata": np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)),
              "time": steps*config["dt_au"], "steps": steps, "populations": population,
              "population_mean": population.mean(axis=0), "energy": energy, "energy_drift": energy-energy[:, :1],
              "electronic_norm": norm, "norm_drift": norm-norm[:, :1],
              "centroid_q": stack("centroid_q"), "centroid_p": stack("centroid_p")}
    population_sem = _sem(population)
    if population_sem is not None:
        arrays["population_sem"] = population_sem
    if method != "rp-mash":
        density = stack("density")
        arrays.update(diabatic_density=density, density_mean=density.mean(axis=0))
        if len(histories) > 1:
            arrays.update(density_sem_real=_sem(density.real), density_sem_imag=_sem(density.imag))
    if method != "sm-nrpmd":
        active = stack("active").astype(np.int64)
        indicators = np.eye(population.shape[-1])[active]
        arrays.update(active_surfaces=active, active_surface_counts=np.sum(indicators, axis=0).astype(np.int64),
                      active_surface_fractions=np.mean(indicators, axis=0))
        if len(histories) > 1:
            arrays["active_surface_sem"] = _sem(indicators)
    if method not in ("sm-nrpmd", "rp-mash"):
        amplitude = stack("amplitude")
        arrays.update(adiabatic_amplitude_populations=amplitude,
                      adiabatic_amplitude_population_mean=amplitude.mean(axis=0))
        if len(histories) > 1:
            arrays["adiabatic_amplitude_population_sem"] = _sem(amplitude)
    atomic_savez(directory/"results"/"nonadiabatic.npz", kind=_RESULT_KIND, **arrays)
    text = io.StringIO()
    writer = csv.writer(text)
    nstates = population.shape[-1]
    fields = [("population", arrays["population_mean"])]
    if population_sem is not None:
        fields.append(("population_sem", population_sem))
    if method != "sm-nrpmd":
        fields.append(("active_surface_fraction", arrays["active_surface_fractions"]))
    if method not in ("sm-nrpmd", "rp-mash"):
        fields.append(("adiabatic_amplitude_population", arrays["adiabatic_amplitude_population_mean"]))
    writer.writerow(["time_au"] + [f"{name}_{i}" for name, values in fields for i in range(nstates)])
    for frame, time in enumerate(arrays["time"]):
        writer.writerow([time]+[float(values[frame, i]) for _, values in fields for i in range(nstates)])
    atomic_write_text(directory/"results"/"populations.csv", text.getvalue())
    return arrays, metadata


def prepare_extended_nonadiabatic(config, system, numerical=False):
    streams = _streams(config, 0)
    dynamics = _driver(config, system, streams[2])
    try:
        state = _initialize(config, system, dynamics, streams)
        record = _observe(dynamics, state, config["method"])
        result = {"method": config["method"], "nstates": dynamics.nstates, "model": config["model"],
                  "initialization": config["initial"]["kind"], "state_shape": list(state.q.shape),
                  "electronic_state_class": type(state).__name__, "integrator": dynamics.integrator_name,
                  "initial_energy_hartree": float(record["energy"]), "physical_converged": False,
                  **_estimator_metadata(config["method"]), **_execution_metadata(config)}
        if numerical:
            from ..potentials.diabatic import check_diabatic_potential
            result["matrix_gradient_check"] = check_diabatic_potential(dynamics.potential, state.q, dynamics.nstates)
        return result
    finally:
        dynamics.potential.close()


def run_extended_nonadiabatic(config, system, directory):
    plan = execution_plan(config["parallel"])
    if plan.backend not in ("serial", "process") or plan.mode != "serial" or plan.bead_workers != 1:
        raise ValueError("extended nonadiabatic ensembles support serial or trajectory-process execution")
    if config["dynamics"].get("loop_backend", "auto") not in ("auto", "python"):
        raise ValueError("extended nonadiabatic methods require loop_backend='auto' or 'python'")
    tasks = [(config, system.to_dict(), str(directory), index) for index in range(config["ensemble"]["trajectories"])]
    run_tasks(_trajectory, tasks, config["parallel"])
    return analyze_extended_nonadiabatic(config, system, directory)


def analyze_extended_nonadiabatic(config, system, directory):
    directory = Path(directory)
    identity = fingerprint(scientific_config(config), system)
    histories = []
    for index in range(config["ensemble"]["trajectories"]):
        streams = _streams(config, index)
        dynamics = _driver(config, system, streams[2])
        try:
            state, history = _load(directory/"checkpoints"/f"nonadiabatic_{index:06d}.npz", config, identity,
                                   index, dynamics, streams)
            if state.step != config["ensemble"]["steps"]:
                raise CheckpointError(f"trajectory {index} is incomplete; resume before ensemble reduction")
            histories.append(history)
        finally:
            dynamics.potential.close()
    arrays, metadata = _write_results(directory, config, identity, histories)
    model = config["model"].get("model", config["model"].get("name", config["model"].get("schema")))
    return {"task": "nonadiabatic", "method": config["method"], "model": model,
            "trajectories": len(histories), "steps": config["ensemble"]["steps"], "frames": len(arrays["steps"]),
            **_estimator_metadata(config["method"]), "estimator": metadata["population_estimator"],
            **_execution_metadata(config),
            "final_populations": arrays["population_mean"][-1].tolist(),
            "final_population_sem": None if "population_sem" not in arrays else arrays["population_sem"][-1].tolist(),
            "sem_unavailable_reason": metadata["sem_unavailable_reason"],
            "uncertainty": "pointwise SEM across independent trajectories; excludes systematic method and discretization errors",
            "max_abs_energy_drift_at_saved_frames_hartree": float(np.max(np.abs(arrays["energy_drift"]))),
            "max_electronic_norm_drift_at_saved_frames": float(np.max(np.abs(arrays["norm_drift"]))),
            "final_active_surface_fractions": None if "active_surface_fractions" not in arrays else arrays["active_surface_fractions"][-1].tolist(),
            "physical_converged": False, "completed": True, "initialization": config["initial"],
            "physics_fingerprint": identity, "arrays": str(directory/"results"/"nonadiabatic.npz"),
            "time_series": str(directory/"results"/"populations.csv")}
