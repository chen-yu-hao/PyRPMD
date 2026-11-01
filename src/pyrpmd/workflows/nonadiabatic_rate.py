"""Input-driven SEO thermal NRPMD electronic population-rate workflow.

This uses the existing signed thermal sampler and analytical population-flux
estimator. Focused excited-state trajectories are deliberately not substituted
for a thermal ensemble. The workflow retains raw signed contributions and
withholds an automatic physical-rate claim.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
import json
from pathlib import Path
import tomllib

import numpy as np

from ..checkpoint import atomic_savez
from ..constants import ATOMIC_TIME_S
from ..dynamics import CayleyIntegrator
from ..io import file_sha256, jsonable, write_json
from ..nonadiabatic import _positive_integer
from ..potentials import make_potential
from ..potentials.python import import_object
from ..thermal_nrpmd import ThermalNRPMDDynamics, ThermalNRPMDSampler, thermal_population_rate


def _fields(value, allowed, label):
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    unknown = set(value)-set(allowed)
    if unknown:
        raise ValueError(f"unknown {label} fields: {sorted(unknown)}")
    return dict(value)


def _positive(value, label):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.number)) or np.iscomplexobj(value)):
        raise ValueError(f"{label} must be a positive finite number in atomic units")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be a positive finite number in atomic units")
    return value


def _nonnegative_integer(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(value)


def resolve_nonadiabatic_rate_config(mapping, base_directory="."):
    """Resolve a compact, strict AU-only thermal-rate input without running PES code.

    Top-level keys: task, masses, q, beta, beads, nstates, seed, potential,
    sampling, dynamics, rate. ``q`` may be [N,3] (replicated to beads) or
    [P,N,3]. Relative Python factory files are resolved against the input.
    """
    config = _fields(mapping, ("task", "masses", "q", "beta", "beads", "nstates", "seed",
                               "potential", "sampling", "dynamics", "rate"), "nonadiabatic rate")
    if config.get("task", "nonadiabatic_rate") != "nonadiabatic_rate":
        raise ValueError("task must be nonadiabatic_rate")
    missing = {"masses", "q", "beta", "nstates", "potential"}-set(config)
    if missing:
        raise ValueError(f"missing nonadiabatic rate fields: {sorted(missing)}")
    nstates = _positive_integer(config["nstates"], "nstates")
    if nstates < 2:
        raise ValueError("an electronic transition needs at least two states")
    beads = _positive_integer(config.get("beads", 1), "beads")
    beta = _positive(config["beta"], "beta")
    q = np.asarray(config["q"])
    if q.dtype.kind not in "fiu" or not np.isfinite(q).all():
        raise ValueError("q must contain finite real atomic-unit coordinates")
    if q.ndim == 2:
        q = np.repeat(q[None], beads, axis=0)
    checker = CayleyIntegrator(config["masses"], beta, beads, 0., native=False)
    checker._check(np.ascontiguousarray(q, dtype=float), np.zeros_like(q, dtype=float))
    seed = _nonnegative_integer(config.get("seed", 0), "seed")
    potential = _fields(config["potential"], ("provider", "factory", "params", "units", "derivative"), "potential")
    if potential.get("provider", "diabatic_python") != "diabatic_python":
        raise ValueError("thermal electronic rates require a real diabatic_python potential")
    potential["provider"] = "diabatic_python"
    if not isinstance(potential.get("factory"), str) or ":" not in potential["factory"]:
        raise ValueError("potential.factory must be module:factory or file.py:factory")
    module, name = potential["factory"].rsplit(":", 1)
    if module.endswith(".py"):
        path = (Path(base_directory)/module).resolve()
        if not path.is_file():
            raise ValueError(f"potential factory file does not exist: {path}")
        potential["factory"] = str(path)+":"+name
    if "params" in potential and not isinstance(potential["params"], dict):
        raise ValueError("potential.params must be a mapping")
    if potential.get("derivative", "gradient") != "gradient":
        raise ValueError("diabatic matrices require positive gradients")
    sample = _fields(config.get("sampling", {}),
                     ("samples", "burn_in", "stride", "nuclear_step", "centroid_step"), "sampling")
    sampling = {"samples": _positive_integer(sample.get("samples", 100), "sampling.samples"),
                "burn_in": _nonnegative_integer(sample.get("burn_in", 100), "sampling.burn_in"),
                "stride": _positive_integer(sample.get("stride", 10), "sampling.stride"),
                "nuclear_step": _positive(sample.get("nuclear_step", .1), "sampling.nuclear_step"),
                "centroid_step": _positive(sample.get("centroid_step", .1), "sampling.centroid_step")}
    dyn = _fields(config.get("dynamics", {}), ("dt", "steps", "stride", "integrator", "native"), "dynamics")
    dynamics = {"dt": _positive(dyn.get("dt", .1), "dynamics.dt"),
                "steps": _positive_integer(dyn.get("steps", 100), "dynamics.steps"),
                "stride": _positive_integer(dyn.get("stride", 1), "dynamics.stride"),
                "integrator": dyn.get("integrator", "normal-mode-bab"), "native": dyn.get("native", False)}
    if not isinstance(dynamics["native"], (bool, np.bool_)):
        raise ValueError("dynamics.native must be boolean")
    CayleyIntegrator(checker.masses, beta, beads, dynamics["dt"], native=False, scheme=dynamics["integrator"])
    rate = _fields(config.get("rate", {}), ("reactant_state", "product_state", "plateau_start", "plateau_stop",
                                            "block_size", "relative_tolerance", "min_average_sign"), "rate")
    rate.setdefault("reactant_state", 0)
    rate.setdefault("product_state", 1)
    for key in ("reactant_state", "product_state"):
        rate[key] = _nonnegative_integer(rate[key], "rate."+key)
        if rate[key] >= nstates:
            raise ValueError(f"rate.{key} is outside the electronic state range")
    if rate["reactant_state"] == rate["product_state"]:
        raise ValueError("reactant_state and product_state must differ")
    if ("plateau_start" in rate) != ("plateau_stop" in rate):
        raise ValueError("rate requires both plateau_start and plateau_stop")
    if "plateau_start" in rate:
        start, stop = rate["plateau_start"], rate["plateau_stop"]
        if (any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, float)) for x in (start, stop))
                or not np.isfinite(start+stop) or not 0 <= start < stop <= dynamics["steps"]*dynamics["dt"]):
            raise ValueError("plateau interval must lie within propagation times")
        times = np.arange(0, dynamics["steps"]+1, dynamics["stride"])*dynamics["dt"]
        if times[-1] != dynamics["steps"]*dynamics["dt"]:
            times = np.append(times, dynamics["steps"]*dynamics["dt"])
        if np.count_nonzero((times >= start) & (times <= stop)) < 4:
            raise ValueError("plateau interval needs at least four recorded times")
    if "block_size" in rate:
        rate["block_size"] = _positive_integer(rate["block_size"], "rate.block_size")
    rate["relative_tolerance"] = _positive(rate.get("relative_tolerance", .1), "rate.relative_tolerance")
    rate["min_average_sign"] = _positive(rate.get("min_average_sign", .05), "rate.min_average_sign")
    if rate["min_average_sign"] > 1:
        raise ValueError("rate.min_average_sign must not exceed one")
    return {"task": "nonadiabatic_rate", "masses": checker.masses.tolist(), "q": q.tolist(),
            "beta": beta, "beads": beads, "nstates": nstates, "seed": seed,
            "potential": potential, "sampling": sampling, "dynamics": dynamics, "rate": rate}


def load_nonadiabatic_rate_input(path):
    """Read TOML/.inp, JSON, or a trusted Python build() input."""
    path = Path(path).resolve()
    if path.suffix == ".py":
        mapping = import_object(str(path)+":build")()
    elif path.suffix == ".json":
        mapping = json.loads(path.read_text())
    else:
        mapping = tomllib.loads(path.read_text())
    return resolve_nonadiabatic_rate_config(mapping, path.parent)


def run_nonadiabatic_rate(input_or_mapping, output):
    """Sample thermal parents, propagate the rate correlation and save evidence.

    A fresh output directory is required; there is no implicit restart or
    overwrite. ``summary['physical_converged']`` is always false: sign, block,
    bead, step-size and plateau studies remain separate scientific evidence.
    """
    config = (resolve_nonadiabatic_rate_config(copy.deepcopy(input_or_mapping))
              if isinstance(input_or_mapping, Mapping) else load_nonadiabatic_rate_input(input_or_mapping))
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("nonadiabatic rate output must be a new or empty directory")
    provenance = {"config": config}
    module = config["potential"]["factory"].rsplit(":", 1)[0]
    if module.endswith(".py"):
        provenance["factory_sha256"] = file_sha256(module)
    identity = hashlib.sha256(json.dumps(jsonable(provenance), sort_keys=True, allow_nan=False).encode()).hexdigest()
    potential = make_potential(config["potential"])
    try:
        sampler = ThermalNRPMDSampler(potential, config["masses"], config["beta"], config["q"],
                                      config["nstates"], seed=config["seed"],
                                      nuclear_step=config["sampling"]["nuclear_step"],
                                      centroid_step=config["sampling"]["centroid_step"])
        output.mkdir(parents=True, exist_ok=True)
        write_json(output/"manifest.json", {"schema": "PyRPMD.nonadiabatic-rate.v1", "status": "running",
                                           "physics_fingerprint": identity, **provenance})
        ensemble = sampler.sample(config["sampling"]["samples"], burn_in=config["sampling"]["burn_in"],
                                  stride=config["sampling"]["stride"])
        parents = {key: np.stack([getattr(state, key) for state in ensemble.states])
                   for key in ("q", "p", "mapping_q", "mapping_p")}
        atomic_savez(output/"thermal-parents.npz", kind="PyRPMD.thermal-rate-parents.v1", **parents,
                     signs=ensemble.signs, electronic_populations=ensemble.electronic_populations,
                     sweeps=ensemble.sweeps, masses=ensemble.masses, beta=np.asarray(ensemble.beta))
        dynamics = ThermalNRPMDDynamics(potential, config["masses"], config["beta"], config["beads"],
                                        config["dynamics"]["dt"], config["nstates"],
                                        native=config["dynamics"]["native"],
                                        integrator=config["dynamics"]["integrator"])
        rate = config["rate"]
        correlation = thermal_population_rate(
            ensemble, dynamics, rate["reactant_state"], rate["product_state"],
            steps=config["dynamics"]["steps"], stride=config["dynamics"]["stride"],
            min_average_sign=rate["min_average_sign"])
        atomic_savez(output/"population-rate.npz", kind="PyRPMD.thermal-population-rate.v1",
                     time=correlation.time, numerator=correlation.numerator,
                     reactant_weight=correlation.reactant_weight, signs=correlation.signs,
                     population_correlation=correlation.metadata["population_correlation_signed_samples"])
        summary = correlation.summarize(start=rate.get("plateau_start"), stop=rate.get("plateau_stop"),
                                        block_size=rate.get("block_size"), relative_tolerance=rate["relative_tolerance"])
        candidate = summary.get("candidate_rate")
        summary.update(task="nonadiabatic_rate", method="seo_thermal_nrpmd", physics_fingerprint=identity,
                       sampling=ensemble.sampling_diagnostics,
                       candidate_rate_s_inv=candidate/ATOMIC_TIME_S if candidate is not None else None,
                       formal_rate_s_inv=None,
                       limitations=["finite-time estimator, not a converged chemical rate",
                                    "single-chain sampling; supplied blocks must resolve correlation",
                                    "real diabatic electronic states; no thermal QD extension"])
        write_json(output/"results"/"summary.json", summary)
        write_json(output/"manifest.json", {"schema": "PyRPMD.nonadiabatic-rate.v1", "status": "complete",
                                           "physics_fingerprint": identity, **provenance})
        return summary
    finally:
        potential.close()
