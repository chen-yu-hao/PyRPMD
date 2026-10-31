"""Checked recrossing options and PMF method selection."""

from __future__ import annotations

import numpy as np

from ..input import rate_pmf_options
from ..parallel import execution_plan
from ..propagation import loop_options
from ..units import time_steps
from ..validation import checked_integer


def _pmf_method(config):
    if config.get("task") == "surface_rate":
        return "projected_surface_density" if config.get("surface", {}).get("free_energy") else "mbar_surface_density"
    return rate_pmf_options(config)["method"]


def _recrossing_spec(config):
    return config.get("recrossing", config.get("rate", {}).get("recrossing", {}))


def _recrossing_dynamics(config):
    """Keep friction changes in scientific fingerprints, batching out of them."""
    raw = _recrossing_spec(config)
    method = raw.get("dynamics", "rpmd")
    if method not in ("rpmd", "trpmd"):
        raise ValueError("recrossing.dynamics must be 'rpmd' or 'trpmd'")
    scale = raw.get("friction_scale", 1.0)
    if isinstance(scale, (bool, np.bool_)) or not np.isscalar(scale):
        raise ValueError("recrossing.friction_scale must be finite and nonnegative")
    scale = float(scale)
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("recrossing.friction_scale must be finite and nonnegative")
    if method == "rpmd" and "friction_scale" in raw:
        raise ValueError("recrossing.friction_scale requires dynamics='trpmd'")
    batch = checked_integer(raw.get("child_batch_pairs", 1), "recrossing.child_batch_pairs")
    if batch > 1 and execution_plan(config.get("parallel", {})).bead_workers > 1:
        raise ValueError(
            "recrossing.child_batch_pairs>1 requires parallel.bead_workers=1; "
            "process/MPI bead proxies do not declare independent-frame batching. "
            "Use trajectory_workers/trajectory_groups, or child_batch_pairs=1."
        )
    if batch > 1 and loop_options(config)["loop_backend"] == "native":
        raise ValueError("Batched recrossing requires loop_backend='auto' or 'python'")
    if method == "trpmd" and batch > 1:
        raise ValueError("TRPMD recrossing currently requires child_batch_pairs=1")
    if method == "trpmd" and config.get("dynamics", {}).get("loop_backend") == "native":
        raise ValueError("TRPMD child propagation requires loop_backend='auto' or 'python'")
    if method == "trpmd" and loop_options(config)["integrator"] == "velocity-verlet":
        raise ValueError("TRPMD requires cayley-bcb or normal-mode-bab integration")
    return method, scale


def _recrossing_options(config):
    raw = _recrossing_spec(config)
    _recrossing_dynamics(config)
    dt = config["dt_au"]
    sampling = config["sampling"]

    def steps(name, duration, default):
        minimum = 0 if name == "equilibration_steps" else 1
        return (
            checked_integer(raw[name], "recrossing." + name, minimum)
            if name in raw
            else time_steps(raw.get(duration, default), dt, allow_zero=minimum == 0)
        )

    values = {
        "parent_replicas": checked_integer(
            raw.get("parent_replicas", 2), "recrossing.parent_replicas"
        ),
        "parent_samples": checked_integer(
            raw.get("parent_samples", sampling.get("parents", 4)),
            "recrossing.parent_samples",
        ),
        "equilibration_steps": steps("equilibration_steps", "equilibration", "20 fs"),
        "parent_interval_steps": steps(
            "parent_interval_steps", "parent_interval", "5 fs"
        ),
        "child_pairs": checked_integer(
            raw.get("child_pairs", raw.get("child_pairs_per_parent", 2)),
            "recrossing.child_pairs",
        ),
        "child_steps": steps("child_steps", "child_time", "50 fs"),
        "plateau_fraction": float(raw.get("plateau_fraction", 0.25)),
        "plateau_tolerance": float(raw.get("plateau_tolerance", 0.1)),
        "bootstrap": checked_integer(
            raw.get("bootstrap", 100), "recrossing.bootstrap", 0
        ),
        "constraint_tolerance": float(raw.get("constraint_tolerance", 1e-10)),
    }
    for key in (
        "parent_replicas",
        "parent_samples",
        "parent_interval_steps",
        "child_pairs",
        "child_steps",
    ):
        if values[key] < 1:
            raise ValueError(f"recrossing.{key} must be positive")
    if values["child_steps"] < 2:
        raise ValueError("recrossing.child_steps must be at least 2")
    if values["bootstrap"] not in (0,) and values["bootstrap"] < 2:
        raise ValueError("recrossing.bootstrap must be zero or at least 2")
    if (
        not np.isfinite(values["constraint_tolerance"])
        or values["constraint_tolerance"] <= 0
    ):
        raise ValueError("constraint tolerance must be positive")
    if (
        values["equilibration_steps"] < 0
        or not 0 < values["plateau_fraction"] <= 1
        or not np.isfinite(values["plateau_tolerance"])
        or values["plateau_tolerance"] <= 0
    ):
        raise ValueError("invalid recrossing burn-in/plateau options")
    return values
