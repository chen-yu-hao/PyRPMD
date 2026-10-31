"""Normalize thermodynamics, propagation, sampling and execution controls."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np

from .constants import KB_AU
from .input_paths import _path
from .input_schema import InputError, _integer, _positive
from .units import parse_quantity, time_steps


@dataclass
class RuntimeOptions:
    """Resolved thermodynamic and integration values used during normalization."""

    preset: str
    notes: list[str]
    temperature: float
    beta: float
    beads: int
    seed: int
    dynamics: dict
    dt: float
    integrator: str


def _steps(table, key, duration_key, default, dt, *, allow_zero=False):
    # Do not evaluate a duration which the caller has replaced with exact steps.
    if key in table:
        return _integer(table[key], key, 0 if allow_zero else 1)
    return time_steps(table.get(duration_key, default), dt, allow_zero=allow_zero)



def resolve_dynamics(d, *, selected, initial_ensemble):
    preset = d.get("preset", "pilot")
    if preset not in ("pilot", "production"):
        raise InputError("preset must be pilot or production")
    notes = []
    if preset == "production":
        required = ("beads", "dynamics") if selected else ("temperature", "beads", "dynamics", "sampling")
        for k in required:
            if k not in d:
                raise InputError(f"production requires explicit {k}")
    if selected:
        # beta controls the NE-RPMD reference rings. At P=1 it is only a
        # required positive integrator parameter and does not thermalize QCT.
        beta = initial_ensemble.get("beta", 1.0)
        temperature = _positive(1 / KB_AU / beta, "auxiliary inverse-beta scale")
        notes.append("State-selected initial ensemble: temperature_K is an internal inverse-beta compatibility scale, not a thermal temperature")
        if initial_ensemble["method"] == "qct" and "beta" not in initial_ensemble:
            notes.append("QCT P=1 uses beta=1 hartree^-1 as an unused integrator compatibility parameter")
    else:
        if "temperature" not in d:
            notes.append("Pilot default temperature: 300 K")
        temperature = _positive(
            parse_quantity(d.get("temperature", "300 K"), "temperature"), "temperature"
        )
        beta = 1 / (KB_AU * temperature)
    beads = d.get("beads", 1 if selected and initial_ensemble["method"] == "qct" else 32)
    if beads == "auto":
        if preset == "production":
            raise InputError(
                "Resolve auto beads with convergence checks before production"
            )
        beads = 32
        notes.append(
            "Auto bead pilot uses P=32; bead convergence has not been established"
        )
    beads = _integer(beads, "beads")
    if selected and initial_ensemble["method"] == "qct" and beads != 1:
        raise InputError("qct initial ensembles require beads=1")
    if selected and initial_ensemble["method"] == "ne_rpmd" and beads < 2:
        raise InputError("ne_rpmd initial ensembles require beads>=2; use qct for a classical quantum-number ensemble")
    seed = _integer(d.get("seed", 0), "seed", 0)
    dyn = dict(d.get("dynamics", {}))
    if dyn.get("loop_backend", "auto") not in ("auto", "native", "python"):
        raise InputError("dynamics.loop_backend must be auto/native/python")
    block_steps = _integer(dyn.get("block_steps", 256), "dynamics.block_steps")
    if block_steps > 65536:
        raise InputError("dynamics.block_steps must be <= 65536")
    dt = _positive(
        parse_quantity(dyn.get("timestep", "0.1 fs"), "time"), "dynamics.timestep"
    )
    from .dynamics import normalize_integrator

    try:
        integrator = normalize_integrator(dyn.get("integrator", "cayley-bcb"))
    except ValueError as error:
        raise InputError(str(error)) from error
    # Preserve the identity of legacy inputs with an implicit default.
    if "integrator" in dyn:
        d["dynamics"]["integrator"] = integrator
    return RuntimeOptions(preset, notes, temperature, beta, beads, seed, dyn, dt, integrator)


def resolve_sampling(d, runtime, *, selected, materials, native_pimd):
    task = d["task"]
    dt, integrator, dyn, notes = runtime.dt, runtime.integrator, runtime.dynamics, runtime.notes
    sampling = deepcopy(d.get("sampling", {}))
    if selected:
        if set(sampling) - {"parents"}:
            raise InputError("State-selected initial_ensemble allows only sampling.parents; thermal equilibration and thermostat settings do not apply")
    else:
        sampling_dt = _positive(
            parse_quantity(sampling.get("timestep", dt), "time"), "sampling.timestep"
        )
        sampling["dt_au"] = sampling_dt
        default_thermostat = "pile-l" if materials and integrator != "velocity-verlet" else "andersen"
        sampling["thermostat"] = sampling.get("thermostat", default_thermostat)
        if not isinstance(sampling["thermostat"], str):
            raise InputError("sampling.thermostat must be a name such as 'andersen', 'pile-l' or 'nose-hoover-chain'")
        sampler = {"cayley-bcb": "bcocb", "normal-mode-bab": "baoab"}.get(integrator)
        expected_sampler = sampler if native_pimd or sampling["thermostat"] == "pile-l" else integrator + "-" + sampling["thermostat"]
        sampling["integrator"] = sampling.get("integrator", expected_sampler)
        if native_pimd:
            from .native_ensemble import _THERMOSTATS
            if sampling["thermostat"] not in _THERMOSTATS or sampling["thermostat"] in ("none", "trpmd"):
                raise InputError("native PIMD requires an equilibrium sampling thermostat")
            if sampler is None or sampling["integrator"] != sampler:
                raise InputError("Native PIMD requires matching BCOCB/BAOAB sampling")
            if "gamma_centroid" in sampling:
                if sampling["thermostat"] != "pile-l":
                    raise InputError("sampling.gamma_centroid requires PILE-L")
                sampling["gamma_centroid"] = _positive(parse_quantity(sampling["gamma_centroid"], "frequency"), "gamma_centroid")
        elif task == "metadynamics" and sampling["thermostat"] == "nose-hoover-chain":
            from .thermostats import nhc_options
            try:
                nhc_options(sampling)
            except (ValueError, TypeError) as error:
                raise InputError(str(error)) from error
            if sampling["integrator"] != expected_sampler:
                raise InputError("NHC sampling.integrator must match dynamics.integrator + '-nose-hoover-chain'")
            if "gamma_centroid" in sampling:
                raise InputError("sampling.gamma_centroid requires PILE-L sampling")
        elif (materials or task in ("fes", "metadynamics")) and sampling["thermostat"] == "pile-l":
            if sampler is None or sampling["integrator"] != sampler:
                raise InputError("PILE-L requires BCOCB with Cayley or BAOAB with normal-mode dynamics")
            sampling["gamma_centroid"] = _positive(
                parse_quantity(sampling.get("gamma_centroid", "0.01 fs^-1"), "frequency"),
                "sampling.gamma_centroid",
            )
            if dyn.get("loop_backend", "auto") == "native":
                raise InputError("PILE-L sampling requires loop_backend='auto' or 'python'; its deterministic substeps may still be native")
        elif sampling["integrator"] != integrator + "-andersen" or sampling["thermostat"] != "andersen":
            raise InputError(
                "Workflow sampling requires sampling.integrator='"
                + integrator + "-andersen' and thermostat='andersen'; "
                "sampling and production use the same selected ring-polymer integrator"
            )
        elif "gamma_centroid" in sampling:
            raise InputError("sampling.gamma_centroid requires PILE-L sampling")
        if not math.isclose(sampling_dt, dt, rel_tol=1e-14):
            raise InputError(
                "Workflow sampling and production time steps must match"
            )
        sampling["equilibration_steps"] = _steps(
            sampling,
            "equilibration_steps",
            "equilibration",
            "100 fs",
            dt,
            allow_zero=True,
        )
        adaptive_table = d.get("adaptive_umbrella", {})
        enhanced_sampling = task == "metadynamics" or (
            isinstance(adaptive_table, Mapping)
            and adaptive_table.get("enabled", bool(adaptive_table))
        )
        default_production = "1000 fs" if enhanced_sampling else "200 fs"
        if enhanced_sampling and not any(k in sampling for k in ("production_steps", "production")):
            notes.append("Enhanced-sampling default production: 1000 fs; inspect production ESS and CV coverage before increasing the budget or interpreting the FES.")
        sampling["production_steps"] = _steps(
            sampling, "production_steps", "production", default_production, dt
        )
        interval = sampling.get("interval_steps", sampling.get("sample_interval"))
        if interval is None:
            interval = time_steps(
                sampling.get("interval", sampling.get("save_interval", "1 fs")), dt
            )
        sampling["interval_steps"] = sampling["sample_interval"] = _integer(
            interval, "sampling.interval_steps"
        )
        sampling["andersen_interval"] = _integer(
            sampling.get(
                "andersen_interval",
                sampling.get("thermostat_params", {}).get("interval_steps", 10),
            ),
            "sampling.andersen_interval",
        )
        sampling["replicas"] = _integer(
            sampling.get("replicas", 4 if task == "diffusion" else 1), "sampling.replicas"
        )
    if "parents" in sampling:
        sampling["parents"] = _integer(sampling["parents"], "sampling.parents")
    return sampling


def resolve_parallel(d, potential):
    parallel = deepcopy(d.get("parallel", {}))
    if (
        "trajectory_workers" in parallel
        and "trajectory_groups" in parallel
        and parallel["trajectory_workers"] != parallel["trajectory_groups"]
    ):
        raise InputError("trajectory_workers and trajectory_groups disagree")
    # 'mode' names per-trajectory work; backend names task transport.
    parallel.setdefault("bead_workers", 1)
    parallel.setdefault(
        "mode",
        (
            "gpu"
            if str(potential.get("device", "")).startswith("cuda")
            else ("bead" if parallel["bead_workers"] > 1 else "serial")
        ),
    )
    default_workers = 1
    if parallel["mode"] == "gpu":
        devices = parallel.get("devices", [0])
        if isinstance(devices, (str, int)):
            devices = [devices]
        default_workers = len(devices) * _integer(
            parallel.get("trajectories_per_gpu", 1), "parallel.trajectories_per_gpu"
        )
    parallel.setdefault(
        "trajectory_workers",
        parallel.get("trajectory_groups", parallel.get("workers", default_workers)),
    )
    parallel.setdefault(
        "backend", "process" if parallel["trajectory_workers"] > 1 else "serial"
    )
    parallel.setdefault("threads_per_rank", 1)
    parallel.setdefault("pes_ranks", 1)
    parallel.setdefault("scheduling", "static")
    parallel.setdefault("coordinator_ranks", 1 if parallel["scheduling"] == "dynamic" else 0)
    for k in (
        "trajectory_workers",
        "bead_workers",
        "threads_per_rank",
        "pes_ranks",
    ):
        parallel[k] = _integer(parallel[k], f"parallel.{k}")
    from .parallel import execution_plan

    try:
        execution_plan(parallel)
    except ValueError as e:
        raise InputError(str(e)) from e
    return parallel


def resolve_output(d, base, dt, *, selected):
    task = d["task"]
    output = deepcopy(d.get("output", {}))
    output["directory"] = _path(base, output.get("directory", f"runs/{task}"))
    output["checkpoint_steps"] = _steps(
        output, "checkpoint_steps", "checkpoint_interval", "10 fs", dt
    )
    if "trajectory_stride" in output:
        output["trajectory_stride"] = _integer(
            output["trajectory_stride"], "output.trajectory_stride"
        )
    if selected or "diagnostics" in output:
        if task != "drpmd":
            raise InputError("output.diagnostics currently requires task='drpmd'")
        diagnostics = output.get("diagnostics", {})
        if not isinstance(diagnostics, Mapping) or set(diagnostics) - {"energy_parts", "ring_radii"}:
            raise InputError("output.diagnostics must be a table containing only energy_parts and ring_radii")
        if any(not isinstance(value, (bool, np.bool_)) for value in diagnostics.values()):
            raise InputError("output.diagnostics flags must be boolean")
        output["diagnostics"] = {
            name: bool(diagnostics.get(name, selected))
            for name in ("energy_parts", "ring_radii")
        }
    return output
