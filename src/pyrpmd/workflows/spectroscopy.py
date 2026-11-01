"""Serial spectroscopy trajectories with property-model and phase provenance.

Saved traces are bead averages of property evaluations, never the property of
the average geometry. Independent replicas retain separate time axes. Atomic
checkpoints contain the physical state, RNG, phase clock, and the exact trace
prefix, allowing restart without duplicate frames or re-equilibration.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

import numpy as np

from ..analysis.spectroscopy import (relative_ir_spectrum, relative_raman_spectrum,
                                     relative_sfg_spectrum)
from ..checkpoint import atomic_savez, load_checkpoint
from ..cmd import PACMDDynamics, _integer, _scalar
from ..dynamics import Dynamics, State
from ..potentials import make_potential
from ..potentials.python import import_object
from ..properties import CallablePropertyModel, TorchScriptPropertyModel, validate_property_result
from ..units import parse_quantity
from ..io import referenced_files
from .fes import _fingerprint, write_json


SPECTROSCOPY_FIELDS = {
    "method", "target_frequency", "friction_scale", "spectra", "sfg_component",
    "max_lag", "window", "sfg_window", "normalize", "laser_wavenumber_cm1",
}
PROPERTY_FIELDS = {
    "provider", "factory", "params", "model", "device", "dtype",
    "position_unit", "dipole_unit", "polarizability_unit",
}
KIND = "PyRPMD.spectroscopy.v1"


def resolve_spectroscopy(config, system):
    """Validate a resolved JobSpec before building a potential or writing data."""
    table = config.get("spectroscopy", {})
    if not isinstance(table, Mapping) or set(table) - SPECTROSCOPY_FIELDS:
        raise ValueError("unknown spectroscopy fields or non-table spectroscopy options")
    props = config.get("properties")
    if not isinstance(props, Mapping) or set(props) - PROPERTY_FIELDS:
        raise ValueError("properties must be a table containing supported model fields")
    provider = props.get("provider")
    if provider == "python":
        if not isinstance(props.get("factory"), str) or ":" not in props["factory"]:
            raise ValueError("Python properties require factory='module:object'")
        if set(props) - {"provider", "factory", "params"}:
            raise ValueError("Python properties use atomic units; only factory/params are supported")
        if not isinstance(props.get("params", {}), Mapping):
            raise ValueError("properties.params must be a table")
    elif provider == "torchscript":
        if not isinstance(props.get("model"), str) or not props["model"]:
            raise ValueError("TorchScript properties require a model file")
        if "factory" in props or "params" in props:
            raise ValueError("TorchScript properties use the standard tensor dictionary contract")
        for key, default, choices in (
            ("dtype", "float64", ("float32", "float64")),
            ("position_unit", "bohr", ("bohr", "angstrom")),
            ("dipole_unit", "au", ("au", "debye")),
            ("polarizability_unit", "au", ("au", "angstrom3")),
        ):
            if props.get(key, default) not in choices:
                raise ValueError(f"properties.{key} must be one of {choices}")
        if not isinstance(props.get("device", "cpu"), str) or not props.get("device", "cpu"):
            raise ValueError("properties.device must be a nonempty device string")
    else:
        raise ValueError("properties.provider must be python or torchscript")
    method = table.get("method", "pa-cmd")
    if method not in ("pa-cmd", "rpmd", "trpmd", "f-cmd"):
        raise ValueError("spectroscopy.method must be pa-cmd, rpmd, trpmd or f-cmd; QCMD/h-CMD are not implemented")
    if method == "f-cmd" and config["beads"] != 1:
        raise ValueError("f-CMD requires beads=1 on a fitted centroid PMF; do not quantize it again")
    if method != "pa-cmd" and "target_frequency" in table:
        raise ValueError("target_frequency applies only to PA-CMD")
    if method not in ("pa-cmd", "trpmd") and "friction_scale" in table:
        raise ValueError("friction_scale applies only to PA-CMD or TRPMD internal modes")
    if system.fixed_atoms:
        raise ValueError("spectroscopy currently requires all atoms mobile")
    for key in ("barostat", "umbrella", "metadynamics", "opes", "plumed", "electronic_friction"):
        if config.get(key):
            raise ValueError(f"spectroscopy requires unbiased fixed-cell production; {key} is not supported")
    parallel = config.get("parallel", {})
    if (parallel.get("backend", "serial") != "serial" or parallel.get("mode", "serial") != "serial"
            or parallel.get("trajectory_workers", 1) != 1 or parallel.get("bead_workers", 1) != 1
            or parallel.get("threads_per_rank", 1) != 1 or parallel.get("pes_ranks", 1) != 1):
        raise ValueError("spectroscopy reference workflow currently supports serial execution")
    loop = config.get("dynamics", {})
    if (loop.get("loop_backend", "auto") == "native" or loop.get("native") == "required"
            or config.get("native") == "required"):
        raise ValueError("spectroscopy uses the NumPy reference engine; required native execution is unsupported")
    if loop.get("integrator", "normal-mode-bab") != "normal-mode-bab":
        raise ValueError("spectroscopy reference workflow requires dynamics.integrator='normal-mode-bab'")
    sampling = config.get("sampling", {})
    if sampling.get("engine", "legacy") != "legacy":
        raise ValueError("spectroscopy uses the NumPy reference engine")
    if sampling.get("thermostat", "pile-l") != "pile-l":
        raise ValueError("spectroscopy equilibration uses a normal-mode Langevin thermostat")
    if sampling.get("integrator", "baoab") != "baoab":
        raise ValueError("spectroscopy equilibration requires sampling.integrator='baoab'")
    options = {
        "method": method,
        "equilibration_steps": _integer(sampling.get("equilibration_steps", 100), "equilibration_steps", 0),
        "production_steps": _integer(sampling.get("production_steps", 1000), "production_steps"),
        "sample_interval": _integer(sampling.get("sample_interval", 10), "sample_interval"),
        "replicas": _integer(sampling.get("replicas", 1), "replicas"),
        "dt": _scalar(config["dt_au"], "dt_au"),
        "sampling_dt": _scalar(sampling.get("dt_au", config["dt_au"]), "sampling.dt_au"),
        "gamma_centroid": _scalar(sampling.get("gamma_centroid", .01), "gamma_centroid"),
        "friction_scale": _scalar(table.get("friction_scale", 1.), "friction_scale", positive=False),
        "checkpoint_steps": _integer(config.get("output", {}).get("checkpoint_steps", 1000), "checkpoint_steps"),
    }
    if method == "pa-cmd":
        # Wavenumber is an energy unit; hbar=1 converts it to angular frequency.
        value = table.get("target_frequency", "14000 cm^-1")
        try:
            frequency = parse_quantity(value, "energy")
        except ValueError:
            frequency = parse_quantity(value, "frequency")
        options["target_frequency"] = _scalar(frequency, "target_frequency")
    spectra = table.get("spectra", ["ir"])
    if (not isinstance(spectra, (list, tuple)) or not spectra or any(not isinstance(name, str) for name in spectra)
            or len(set(spectra)) != len(spectra)
            or any(name not in ("ir", "raman", "sfg") for name in spectra)):
        raise ValueError("spectra must be a nonempty unique list of ir, raman and/or sfg")
    options["spectra"] = list(spectra)
    frames = options["production_steps"] // options["sample_interval"] + 1
    if frames < (5 if "sfg" in spectra else 3):
        raise ValueError("production must save at least five frames for SFG, three for IR/Raman")
    options["required_properties"] = sorted({
        key for name in spectra for key in ({"ir": ("dipole",), "raman": ("polarizability",),
                                             "sfg": ("dipole", "polarizability")}[name])})
    options["window"] = table.get("window", "hann")
    options["sfg_window"] = table.get("sfg_window", "cosine")
    if options["window"] not in ("hann", "boxcar") or options["sfg_window"] not in ("cosine", "boxcar"):
        raise ValueError("window must be hann/boxcar and sfg_window must be cosine/boxcar")
    options["normalize"] = table.get("normalize", True)
    if not isinstance(options["normalize"], bool):
        raise ValueError("spectroscopy.normalize must be boolean")
    options["max_lag"] = table.get("max_lag")
    if options["max_lag"] is not None:
        lag = _integer(options["max_lag"], "max_lag", 2)
        if lag >= frames:
            raise ValueError("spectroscopy.max_lag must be smaller than saved production frames")
    from ..analysis.spectroscopy import _polarization_component
    component = _polarization_component(table.get("sfg_component", "xxz"))
    options["sfg_component"] = "".join("xyz"[i] for i in component)
    options["laser_wavenumber_cm1"] = table.get("laser_wavenumber_cm1")
    if options["laser_wavenumber_cm1"] is not None:
        _scalar(options["laser_wavenumber_cm1"], "laser_wavenumber_cm1")
    return options


def _make_properties(spec, system):
    if spec["provider"] == "torchscript":
        return TorchScriptPropertyModel(spec["model"], system.atomic_numbers,
                                        cell=system.cell, pbc=system.pbc,
                                        **{key: spec[key] for key in PROPERTY_FIELDS
                                           if key in spec and key not in ("provider", "model")})
    obj = import_object(spec["factory"])(**spec.get("params", {}))
    try:
        binder = getattr(obj, "bind_system", None)
        if callable(binder):
            binder(system)
        evaluator = getattr(obj, "evaluate", obj)
        if not callable(evaluator):
            raise ValueError("property factory must return a callable or evaluate provider")
        return CallablePropertyModel(obj)
    except BaseException:
        close = getattr(obj, "close", None)
        if callable(close):
            close()
        raise


def _model_identity(spec):
    path = spec.get("model")
    if path is None and spec.get("factory", "").split(":")[0].endswith(".py"):
        path = spec["factory"].rsplit(":", 1)[0]
    return None if path is None else hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _engines(config, system, potential, options, rng):
    shared = dict(potential=potential, masses=system.masses, beta=config["beta"],
                  nbeads=config["beads"], seed=rng)
    if options["method"] == "pa-cmd":
        shared.update(target_frequency=options["target_frequency"], friction_scale=options["friction_scale"])
        return (PACMDDynamics(dt=options["sampling_dt"], **shared),
                PACMDDynamics(dt=options["dt"], **shared))
    if options["method"] == "f-cmd":
        metadata = getattr(potential, "metadata", {})
        tag = metadata.get("centroid_pmf_beta_au")
        if tag is None or not np.isclose(_scalar(tag, "centroid_pmf_beta_au"), config["beta"], rtol=1e-10, atol=0):
            raise ValueError("f-CMD requires a provider declaring matching metadata['centroid_pmf_beta_au']")
    shared.update(native=False, integrator="normal-mode-bab", loop_backend="python")
    return Dynamics(dt=options["sampling_dt"], **shared), Dynamics(dt=options["dt"], **shared)


def prepare_spectroscopy(config, system, *, potential=None):
    """Describe the run and optionally check its property/centroid-PMF contract.

    A structural check never imports a user model. The numerical check uses
    the already-open PES owned by the caller and closes only the temporary
    property provider; it performs no propagation and writes no trajectories.
    """
    options = resolve_spectroscopy(config, system)
    result = {
        "method": options["method"],
        "spectra": options["spectra"],
        "required_properties": options["required_properties"],
        "property_provider": config["properties"]["provider"],
        "property_estimator": "bead_mean_of_property_surface",
        "production_centroid_thermostat": False,
        "replicas": options["replicas"],
        "saved_frames_per_replica": options["production_steps"] // options["sample_interval"] + 1,
        "sample_timestep_au": options["dt"] * options["sample_interval"],
        "physical_converged": False,
    }
    if options["method"] == "pa-cmd":
        result["target_angular_frequency_au"] = options["target_frequency"]
    if potential is not None:
        # Constructor checks also enforce the fitted-PMF temperature for f-CMD.
        _engines(config, system, potential, options, np.random.default_rng(0))
        properties = _make_properties(config["properties"], system)
        try:
            q = np.repeat(system.geometry[None], config["beads"], axis=0)
            checked = validate_property_result(
                properties.evaluate(q), config["beads"],
                require=options["required_properties"],
            )
            result["property_check"] = {
                "input_shape": list(q.shape),
                "output_shapes": {key: list(getattr(checked, key).shape)
                                  for key in options["required_properties"]},
                "units": {key: "e*bohr" if key == "dipole" else "atomic_polarizability"
                          for key in options["required_properties"]},
            }
        finally:
            properties.close()
    return result


def _checkpoint(output, replica):
    return Path(output) / "checkpoints" / f"spectroscopy_{replica:04d}.npz"


def _read_checkpoint(path, config, system, options, identity):
    with load_checkpoint(path, expected_kind=KIND, require_integrity=True) as saved:
        if str(saved["fingerprint"]) != identity:
            raise ValueError("spectroscopy checkpoint physics or property-model content changed")
        raw_step = np.asarray(saved["step"])
        if raw_step.ndim:
            raise ValueError("spectroscopy checkpoint step must be a scalar")
        step = _integer(raw_step.item(), "checkpoint step", 0)
        total = options["equilibration_steps"] + options["production_steps"]
        if not 0 <= step <= total:
            raise ValueError("spectroscopy checkpoint phase clock is out of range")
        expected_time = (min(step, options["equilibration_steps"]) * options["sampling_dt"]
                         + max(0, step - options["equilibration_steps"]) * options["dt"])
        if not np.isclose(float(saved["time"]), expected_time, rtol=2e-12, atol=2e-12):
            raise ValueError("spectroscopy checkpoint time differs from phase clock")
        state = State(saved["q"], saved["p"], step, float(saved["time"]))
        if state.q.shape != (config["beads"], system.natoms, 3):
            raise ValueError("spectroscopy checkpoint geometry shape changed")
        completed = step - options["equilibration_steps"]
        expected_steps = (np.arange(completed // options["sample_interval"] + 1) * options["sample_interval"]
                          if completed >= 0 else np.empty(0, dtype=int))
        if (np.asarray(saved["sample_steps"]).dtype.kind not in "iu"
                or not np.array_equal(saved["sample_steps"], expected_steps)):
            raise ValueError("spectroscopy checkpoint has a missing or duplicated trace frame")
        traces = {}
        for key in options["required_properties"]:
            shape = (len(expected_steps), 3) if key == "dipole" else (len(expected_steps), 3, 3)
            value = np.asarray(saved[key])
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError("invalid saved spectroscopy property trace")
            traces[key] = [row.copy() for row in value]
        return state, json.loads(str(saved["rng"])), expected_steps.tolist(), traces


def _sample_replica(config, system, output, replica, options):
    identity = _fingerprint(config, system, {"spectroscopy_replica": replica,
                                           "property_model_sha256": _model_identity(config["properties"]),
                                           "source_files": referenced_files(config)})
    path = _checkpoint(output, replica)
    rng = np.random.default_rng(np.random.SeedSequence([int(config.get("seed", 0)), 2503, replica]))
    traces = {key: [] for key in options["required_properties"]}
    sample_steps = []
    state = None
    if path.exists():
        state, rng_state, sample_steps, traces = _read_checkpoint(path, config, system, options, identity)
        rng.bit_generator.state = rng_state
    total = options["equilibration_steps"] + options["production_steps"]
    if state is not None and state.step == total:
        return {key: np.asarray(value) for key, value in traces.items()}
    potential = make_potential(config["potential"], system)
    properties = None
    try:
        properties = _make_properties(config["properties"], system)
        sampler, production = _engines(config, system, potential, options, rng)
        if state is None:
            q = np.broadcast_to(system.geometry, (config["beads"], system.natoms, 3)).copy()
            state = State(q, sampler.sample_momenta())
        # Fail property contracts before starting a potentially long trajectory.
        validate_property_result(properties.evaluate(state.q), config["beads"],
                                 require=options["required_properties"])

        def record():
            values = validate_property_result(properties.evaluate(state.q), config["beads"],
                                              require=options["required_properties"])
            sample_steps.append(state.step - options["equilibration_steps"])
            for key in traces:
                traces[key].append(getattr(values, key).mean(axis=0))

        def save():
            arrays = {key: np.asarray(value).reshape(-1, *((3,) if key == "dipole" else (3, 3)))
                      for key, value in traces.items()}
            atomic_savez(path, kind=KIND, fingerprint=identity, q=state.q, p=state.p,
                         step=state.step, time=state.time, rng=json.dumps(rng.bit_generator.state),
                         sample_steps=np.asarray(sample_steps, dtype=np.int64), **arrays)

        if state.step == options["equilibration_steps"] and not sample_steps:
            record()
        while state.step < total:
            if state.step < options["equilibration_steps"]:
                sampler.sample_step(state, gamma_centroid=options["gamma_centroid"])
            elif options["method"] == "trpmd":
                production.trpmd_step(state, friction_scale=options["friction_scale"])
            else:
                production.step(state)
            production_step = state.step - options["equilibration_steps"]
            if production_step >= 0 and production_step % options["sample_interval"] == 0:
                record()
            if state.step % options["checkpoint_steps"] == 0 or state.step == total:
                save()
        return {key: np.asarray(value) for key, value in traces.items()}
    finally:
        potential.close()
        close = getattr(properties, "close", None)
        if callable(close):
            close()


def _analyze(config, system, output, options, streams):
    results = Path(output) / "results"
    results.mkdir(parents=True, exist_ok=True)
    traces = {key: np.stack([s[key] for s in streams]) for key in options["required_properties"]}
    dt = options["dt"] * options["sample_interval"]
    data_path = results / "spectroscopy-traces.npz"
    atomic_savez(data_path, kind="PyRPMD.spectroscopy-traces.v1", **traces,
                 sample_interval_au=dt, metadata=json.dumps({
                     "estimator": "mean of per-bead property surface values",
                     "method": options["method"], "axes": "replica,time,property_components"}))
    spectra = {}
    common = dict(normalize=options["normalize"])
    if "ir" in options["spectra"]:
        spectra["ir"] = relative_ir_spectrum(traces["dipole"][:, :, None, :], dt,
                                              beta=config["beta"], window=options["window"], **common)
    if "raman" in options["spectra"]:
        spectra["raman"] = relative_raman_spectrum(traces["polarizability"][:, :, None, :, :], dt,
                                                    beta=config["beta"], window=options["window"],
                                                    laser_wavenumber_cm1=options["laser_wavenumber_cm1"], **common)
    if "sfg" in options["spectra"]:
        spectra["sfg"] = relative_sfg_spectrum(traces["dipole"][:, :, None, :],
                                                traces["polarizability"][:, :, None, :, :], dt,
                                                beta=config["beta"], component=options["sfg_component"],
                                                max_lag=options["max_lag"], window=options["sfg_window"], **common)
    files = {"traces": str(data_path), "checkpoints": [str(_checkpoint(output, r)) for r in range(len(streams))]}
    metadata = {}
    for name, result in spectra.items():
        result["dynamics_method"] = options["method"]
        if name == "ir":
            result["estimator"] = "beta*omega^2 times approximate Kubo dipole spectrum"
        arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
        scalars = {key: value for key, value in result.items() if not isinstance(value, np.ndarray)}
        target = results / f"{name}-spectrum.npz"
        atomic_savez(target, kind=f"PyRPMD.{name}-spectrum.v1", metadata=json.dumps(scalars), **arrays)
        files[name] = str(target)
        metadata[name] = scalars
    summary = {
        "task": "spectroscopy", "status": "complete", "method": options["method"],
        "physical_converged": False, "converged": False,
        "replicas": len(streams), "saved_frames_per_replica": len(next(iter(streams[0].values()))),
        "sample_interval_au": dt, "beta_au": config["beta"], "beads": config["beads"],
        "equilibration_steps": options["equilibration_steps"],
        "production_steps": options["production_steps"],
        "production_centroid_thermostat": False, "absolute_intensity": False,
        "property_estimator": "bead-averaged total-system properties in laboratory axes",
        "spectra": metadata, "files": files,
        "warnings": ["Timestep, bead, trajectory length and equilibrium sampling convergence are not certified.",
                     "Spectral intensities omit electromagnetic, system-size and experimental optical prefactors."],
    }
    if options["method"] == "pa-cmd":
        summary["target_angular_frequency_au"] = options["target_frequency"]
        summary["warnings"].append("PA-CMD requires independent adiabaticity and internal-thermostat convergence checks.")
    if options["method"] == "f-cmd":
        summary["warnings"].append(
            "f-CMD properties are evaluated on the centroid: nonlinear surfaces require a validated "
            "conditional centroid-property model, or introduce an additional observable approximation."
        )
    if options["equilibration_steps"] == 0:
        summary["warnings"].append("No equilibration requested; the collapsed initial ring is not an equilibrium sample.")
    write_json(results / "summary.json", summary)
    return summary


def run_spectroscopy(config, system, output):
    """Run/resume serial independent spectroscopy replicas and save spectra."""
    options = resolve_spectroscopy(config, system)
    streams = [_sample_replica(config, system, output, replica, options)
               for replica in range(options["replicas"])]
    return _analyze(config, system, output, options, streams)


def reanalyze_spectroscopy(config, system, output):
    """Reanalyze already saved averaged properties without loading PES/ML code."""
    options = resolve_spectroscopy(config, system)
    path = Path(output) / "results" / "spectroscopy-traces.npz"
    with load_checkpoint(path, expected_kind="PyRPMD.spectroscopy-traces.v1", require_integrity=True) as data:
        if float(data["sample_interval_au"]) != options["dt"] * options["sample_interval"]:
            raise ValueError("saved spectroscopy sampling interval differs from configuration")
        arrays = {key: np.asarray(data[key]) for key in options["required_properties"]}
        expected_frames = options["production_steps"] // options["sample_interval"] + 1
        for key, value in arrays.items():
            tail = (3,) if key == "dipole" else (3, 3)
            if value.shape != (options["replicas"], expected_frames, *tail) or not np.isfinite(value).all():
                raise ValueError("saved spectroscopy property shapes or values are invalid")
        streams = [{key: value[r].copy() for key, value in arrays.items()} for r in range(options["replicas"])]
    return _analyze(config, system, output, options, streams)
