"""Distributed OPES workflow helpers."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np

from . import _opes_serial as _serial
for _name, _value in vars(_serial).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

from ..analysis.fes_result import FESResult, write_fes_result
from ..checkpoint import load_checkpoint
from ..opes import SharedOPESBias
from ..walker_exchange import (
    AsyncWalkerStore, allgather_centers, ensure_manifest, make_record, mpi_world,
    read_committed_records, validate_record)

MPI_KIND = "PyRPMD.shared-opes.mpi.v1"
ASYNC_KIND = "PyRPMD.shared-opes.async-walker.v1"

def _mode(config):
    return config.get("opes", {}).get("walkers_mode", "serial")


def _new_shared_bias(cv, options, beta, walkers, state=None):
    return SharedOPESBias(
        cv, options["sigma"], beta, walkers=walkers,
        barrier=options["barrier"], bias_factor=options["bias_factor"],
        mode=options.get("bias_mode", "metad"), state=state)


def _validate_shared_options(config, options):
    walkers = config.get("sampling", {}).get("replicas", 1)
    if walkers < 2:
        raise ValueError("distributed OPES requires sampling.replicas >= 2")
    if options.get("adaptive_sigma"):
        raise ValueError("distributed OPES requires fixed sigma")
    if options.get("freeze") is not None or options.get("freeze_equilibration_steps"):
        raise ValueError("distributed OPES does not support frozen-bias production")
    if options.get("method") != "kde" or options.get("measure") != "cv_density":
        raise ValueError("distributed OPES currently supports KDE cv_density analysis")
    return walkers


def _shared_fingerprint(config, system, mode, **extra):
    return _fingerprint(
        checkpoint_cv_config(config), system,
        {"opes_shared": 1, "walkers_mode": mode, **extra})


def _analyze_shared(bias, samples, config, system, output, options, *, mode,
                    checkpoint_files, complete, return_data):
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 3 or samples.shape[2] != bias.cv.dimension or not samples.size:
        raise ValueError("shared OPES analysis needs finite [samples,walkers,D] observations")
    if not np.isfinite(samples).all():
        raise ValueError("shared OPES samples must be finite")
    cv = bias.cv
    pooled = samples.reshape(-1, cv.dimension).copy()
    bounds = options["ranges"]
    if bounds is None:
        bounds = []
        for index, period in enumerate(cv.periods):
            lo, hi = pooled[:, index].min(), pooled[:, index].max()
            width = max(float(hi - lo), options["sigma"][index])
            bounds.append([-period / 2, period / 2] if period is not None
                          else [lo - 0.05 * width, hi + 0.05 * width])
    edges = [np.linspace(lo, hi, count + 1)
             for (lo, hi), count in zip(bounds, options["bins"])]
    for index, period in enumerate(cv.periods):
        if period is not None:
            pooled[:, index] = ((pooled[:, index] - bounds[index][0]) % period
                                + bounds[index][0])
    grid = np.stack(np.meshgrid(*[
        0.5 * (edge[1:] + edge[:-1]) for edge in edges], indexing="ij"), axis=-1)
    counts = np.histogramdd(pooled, bins=edges)[0].astype(int)
    mask = counts < options["min_count"]
    density_factor = bias.bias_factor if bias.mode == "explore" else 1.0
    raw = -density_factor * bias.log_probability(grid) / config["beta"]
    mask |= ~np.isfinite(raw)
    free = np.full_like(raw, np.nan)
    reference = None
    if (~mask).any():
        reference = np.unravel_index(np.argmin(np.where(mask, np.inf, raw)), raw.shape)
        free[~mask] = raw[~mask] - raw[reference]
    nan = np.full_like(free, np.nan)
    data = {"free_energy": free,
            "free_energy_per_replica": free[None].copy(),
            "replica_spread": nan.copy(), "standard_error": nan.copy(),
            "mask": mask, "mask_per_replica": mask[None].copy(),
            "raw_count": counts, "raw_count_per_replica": counts[None].copy(),
            "edges": edges}
    output = Path(output)
    warnings = list(config.get("_warnings", [])) + [
        "Shared OPES is uncompressed: every committed kernel is retained and normalization is rebuilt from the complete deterministic event order.",
        "One shared history couples walkers; walker spread is not an independent-replica uncertainty estimate.",
        "Finite-time KDE smoothing, CV adequacy, equilibration, bead and timestep convergence remain unverified."]
    if mode == "async-files":
        warnings.append(
            "Asynchronous kernel weights retain the bias visible to their writer at publication; the merged KDE is eventually consistent, not trajectory-wise identical to mpi-sync.")
    summary = {"task": "opes", "status": (
            "finite_time_estimate" if complete else "partial_finite_time_estimate"),
        "method": "opes", "algorithm": bias.method, "estimator": "kde",
        "measure": "cv_density", "energy_unit": "hartree",
        "dimension": cv.dimension, "units": cv.units,
        "walkers": samples.shape[1], "walkers_mode": mode,
        "independent_histories": 1, "samples": int(samples.shape[0] * samples.shape[1]),
        "kernels": len(bias), "log_z": bias.log_z,
        "weight_effective_sample_size": bias.effective_sample_size,
        "compression": "none", "common_support": bool((~mask).any()),
        "reference_bin": None if reference is None else list(reference),
        "converged": False, "physical_converged": False,
        "sample_bias_convention": (
            "physical shared V before the probability update; H_P contains P*V; "
            "synchronous log weights use the common pre-commit bias" if mode == "mpi-sync"
            else "physical locally visible V before publication; H_P contains P*V"),
        "warnings": warnings,
        "files": {"data": str(output / "fes.npz"),
                  "table": str(output / "fes.csv"),
                  "checkpoints": [str(path) for path in checkpoint_files]}}
    result = FESResult.from_data(data, method="opes", cv_units=cv.units,
                                 metadata=summary)
    summary = write_fes_result(output, result, summary)
    return {"summary": summary, "data": data, "bias": bias} if return_data else summary


