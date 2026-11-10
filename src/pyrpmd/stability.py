"""Public workflow maturity, independently of a particular run's convergence.

This module performs no imports of workflows or optional numerical backends.
Descriptions define the scope of each label; method overrides can only raise
the caution level of the containing task.
"""
from __future__ import annotations

from copy import deepcopy

LEVELS = ("stable", "beta", "research")

TASK_STABILITY = {
    "rate": {"level": "stable", "scope": "Single-surface Bennett-Chandler rates with supported PMFs and explicit recrossing gates."},
    "drpmd": {"level": "stable", "scope": "Thermal single-surface scattering with explicit unresolved and failed outcomes."},
    "fes": {"level": "stable", "scope": "Umbrella FES and independent frozen-bias metadynamics; method overrides apply."},
    "network": {"level": "beta", "scope": "Gas-phase reaction networks assembled from individually checked rate channels."},
    "metadynamics": {"level": "stable", "scope": "Deprecated alias for fes.method=metadynamics; independent histories."},
    "opes": {"level": "beta", "scope": "Deprecated alias for fes.method=opes; adaptive density estimation."},
    "pcet": {"level": "beta", "scope": "Documented model PCET rates; system-specific parameter validation is required."},
    "pimd": {"level": "stable", "scope": "Distinguishable-particle primitive-action equilibrium PIMD."},
    "diffusion": {"level": "stable", "scope": "Unwrapped centroid transport with physical dynamics and diffusive-regime diagnostics."},
    "site_rate": {"level": "stable", "scope": "Basin-normalized single-site escape with PMF and recrossing diagnostics."},
    "surface_rate": {"level": "beta", "scope": "Surface and slab rates with explicit normalization and fixed-atom limits."},
    "nonadiabatic": {"level": "beta", "scope": "Explicit model and ensemble nonadiabatic dynamics; research variants are separately labeled."},
    "nonadiabatic_rate": {"level": "research", "scope": "Signed thermal NRPMD rate estimates with unresolved sign and convergence limitations."},
    "spectroscopy": {"level": "stable", "scope": "RPMD/TRPMD correlation functions and relative spectra with explicit property surfaces."},
    "plumed": {"level": "beta", "scope": "External PLUMED bias integration; backend availability and restart contracts vary."},
    "path_sampling": {"level": "research", "scope": "Reactive-path ensembles and interface sampling with method-specific acceptance criteria."},
}

METHOD_STABILITY = {
    "fes:opes": "beta", "fes:abf": "beta", "fes:eabf": "beta",
    "fes:steered": "beta", "fes:ves": "beta",
    "spectroscopy:pa-cmd": "beta", "spectroscopy:f-cmd": "beta",
    "spectroscopy:f-qcmd": "research",
    "nonadiabatic:rp-iesh": "research",
    "nonadiabatic:rp-mash-ba": "research",
    "nonadiabatic:multistate-mash": "research",
    "nonadiabatic:adaptive-nrpmd": "research",
    "method:seo_thermal_nrpmd": "research",
    "method:subspace-correction": "research",
    "method:subspace_correction": "research",
    "path_integral:suzuki-chin": "beta", "path_integral:suzuki_chin": "beta",
    "path_integral:bosonic": "beta", "path_integral:pigs": "beta",
    "path_integral:fermionic": "research",
    "feature:shared_walkers": "beta", "feature:nonadiabatic_scattering": "research",
    "feature:subspace_correction": "research", "feature:state_selected_scattering": "beta",
}


def stability_registry():
    """Return a JSON-ready snapshot that callers cannot mutate in place."""
    return {"schema_version": 1, "levels": list(LEVELS),
            "tasks": deepcopy(TASK_STABILITY), "method_overrides": dict(METHOD_STABILITY)}


def classify_stability(config):
    """Classify public or resolved settings without changing their identity."""
    task = config.get("task")
    if task not in TASK_STABILITY:
        raise ValueError(f"No stability classification for task {task!r}")
    result = {"task": task, **TASK_STABILITY[task], "overrides": []}
    family = "fes" if task in ("fes", "metadynamics", "opes") else task
    method = config.get("method")
    if family in ("fes", "spectroscopy"):
        method = config.get(family, {}).get("method", method)
    if task in ("metadynamics", "opes"):
        method = task
    keys = [f"{family}:{method}", f"method:{method}"]
    action = config.get("path_integral", {}).get("method")
    if action is not None:
        keys.append("path_integral:" + str(action))
    for bias in ("metadynamics", "opes"):
        table = config.get(bias, config.get("fes", {}).get(bias, {}))
        if (table.get("shared_walkers") or table.get("multiple_walkers")
                or table.get("walkers_mode", "serial") != "serial"):
            keys.append("feature:shared_walkers")
    if config.get("subspace_correction"):
        keys.append("feature:subspace_correction")
    if task == "drpmd" and config.get("nonadiabatic"):
        keys.append("feature:nonadiabatic_scattering")
    if task == "drpmd" and config.get("initial_ensemble"):
        keys.append("feature:state_selected_scattering")
    for key in keys:
        level = METHOD_STABILITY.get(key)
        if level is not None:
            result["overrides"].append({"selector": key, "level": level})
            if LEVELS.index(level) > LEVELS.index(result["level"]):
                result["level"] = level
    return result
