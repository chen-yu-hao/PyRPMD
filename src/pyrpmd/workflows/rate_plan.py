"""A concise, read-only description of the two rate-calculation stages."""

from __future__ import annotations

from ..constants import ATOMIC_TIME_S
from ..input import rate_pmf_options


def rate_pmf_plan(config):
    """Expose effective clocks/operators before any PES or trajectories run."""
    options = rate_pmf_options(config)
    sampler = options.get("sampler", "umbrella")
    recrossing = config.get("recrossing", config.get("rate", {}).get("recrossing", {}))
    parent = recrossing.get("parent_thermostat", config.get("sampling", {}))
    plan = {
        "sampler": sampler,
        "estimator": "saved_surface" if options["source"] is not None else options["method"],
        "measure": "rate_flux",
        "recrossing": {
            "integrator": config.get("dynamics", {}).get("integrator", "cayley-bcb"),
            "timestep_fs": config["dt_au"] * ATOMIC_TIME_S / 1e-15,
            "parent_thermostat": parent.get("thermostat", "andersen"),
            "child_dynamics": recrossing.get("dynamics", "rpmd"),
        },
    }
    if options["source"] is not None:
        source = options["source"]
        plan["source_directory"] = source if isinstance(source, str) else source["directory"]
        plan["sampling"] = None
        plan["barrier_uncertainty"] = "unavailable: saved pointwise errors lack paired endpoint covariance"
        return plan
    stage = config.get("rate", {}).get("pmf_stage", {}).get("config", config)
    sampling = stage["sampling"]
    dynamics = stage.get("dynamics", {})
    dt_fs = stage["dt_au"] * ATOMIC_TIME_S / 1e-15
    production = sampling["production_steps"]
    interval = sampling["interval_steps"]
    phases = [{"name": "initial_equilibration", "steps": sampling["equilibration_steps"]}]
    discarded = 0
    if sampler in ("metadynamics", "opes"):
        bias = stage[sampler]
        training, relaxation = bias["freeze_after_steps"], bias["freeze_equilibration_steps"]
        phases.extend([
            {"name": "bias_training", "steps": training},
            {"name": "frozen_bias_equilibration", "steps": relaxation},
        ])
        discarded = training + relaxation
        plan["barrier_uncertainty"] = "unavailable: correlated frozen-bias endpoint bootstrap is not implemented"
    else:
        plan["barrier_uncertainty"] = "paired block bootstrap when requested and fully supported"
    phases.append({"name": "pmf_production", "steps": production - discarded})
    for phase in phases:
        phase["duration_fs"] = phase["steps"] * dt_fs
    plan["sampling"] = {
        "integrator": dynamics.get("integrator", "cayley-bcb"),
        "thermostat": sampling.get("thermostat", "andersen"),
        "splitting": sampling.get("integrator"),
        "timestep_fs": dt_fs,
        "replicas": sampling.get("replicas", 1),
        "sample_interval_steps": interval,
        "retained_samples_per_replica": production // interval - discarded // interval,
        "phases": phases,
    }
    return plan
