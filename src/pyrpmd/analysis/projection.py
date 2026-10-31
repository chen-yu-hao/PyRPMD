"""Boltzmann marginalization of gridded CV-density free energies.

Integrate density, not F and not a minimum-energy path. Bin widths supply the
CV measure; no extra radial Jacobian belongs to a histogram density PMF.
Masked regions are unknown, never zero probability. A partially supported
marginal is withheld by default, including for channel-conditioned domains.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def marginalize_fes(free_energy, edges, beta, *, axes=0, mask=None, selection=None):
    """Integrate out other axes, optionally inside a Boolean channel domain.

    ``edges`` holds D arrays of bin boundaries in canonical CV units.
    ``selection`` has the full FES shape and declares the integration domain
    (e.g. ``max(r1,r2) >= cutoff`` excludes CO2). This is a conditional density
    projection, not the published historical operation of deleting hills.
    ``coverage`` is supported/domain CV volume, not a probability estimate.
    """
    free = np.asarray(free_energy, float)
    if free.ndim < 1 or free.ndim > 3 or not np.isscalar(beta) or isinstance(beta, (bool, np.bool_)) or not np.isfinite(beta) or beta <= 0:
        raise ValueError("projection needs a 1D--3D FES and positive finite beta")
    if isinstance(axes, (int, np.integer)) and not isinstance(axes, (bool, np.bool_)):
        axes = (int(axes),)
    if (not isinstance(axes, (tuple, list)) or not axes
            or any(isinstance(a, (bool, np.bool_)) or not isinstance(a, (int, np.integer)) or not 0 <= a < free.ndim for a in axes)
            or len(set(axes)) != len(axes)):
        raise ValueError("axes must be distinct valid retained dimensions")
    axes = tuple(int(a) for a in axes)
    if len(edges) != free.ndim:
        raise ValueError("one edge array is required per FES dimension")
    bounds = [np.asarray(e, float) for e in edges]
    if any(e.shape != (n + 1,) or not np.isfinite(e).all() or np.any(np.diff(e) <= 0)
           for e, n in zip(bounds, free.shape)):
        raise ValueError("edges must be finite, strictly increasing and match the FES")
    invalid = ~np.isfinite(free)
    if mask is not None:
        supplied = np.asarray(mask)
        if supplied.shape != free.shape or supplied.dtype.kind != "b":
            raise ValueError("mask must be a Boolean array with the FES shape")
        invalid |= supplied
    domain = np.ones(free.shape, bool) if selection is None else np.asarray(selection)
    if domain.shape != free.shape or domain.dtype.kind != "b":
        raise ValueError("selection must be a Boolean array with the FES shape")
    eliminated = tuple(a for a in range(free.ndim) if a not in axes)
    valid = domain & ~invalid
    log_volume = np.zeros(free.shape)
    for a in eliminated:
        shape = [1] * free.ndim
        shape[a] = free.shape[a]
        log_volume += np.log(np.diff(bounds[a])).reshape(shape)
    volume = np.exp(log_volume)
    total_volume = np.sum(np.where(domain, volume, 0.0), axis=eliminated)
    sampled_volume = np.sum(np.where(valid, volume, 0.0), axis=eliminated)
    coverage = np.divide(sampled_volume, total_volume, out=np.zeros_like(sampled_volume), where=total_volume > 0)
    missing = np.any(domain & invalid, axis=eliminated) | (total_volume == 0)
    # Subtract a common finite gauge before exponentiation/log-sum-exp.
    anchor = float(np.min(free[valid])) if np.any(valid) else 0.0
    logs = np.full(free.shape, -np.inf)
    logs[valid] = -beta * (free[valid] - anchor) + log_volume[valid]
    projected = -logsumexp(logs, axis=eliminated) / beta
    projected[missing] = np.nan
    if np.any(~missing):
        projected -= np.min(projected[~missing])
    remaining = sorted(axes)
    permutation = [remaining.index(a) for a in axes]
    return {"free_energy": projected.transpose(permutation), "mask": missing.transpose(permutation),
            "coverage": coverage.transpose(permutation), "edges": [bounds[a].copy() for a in axes],
            "axes": axes, "measure": "cv_density", "method": "boltzmann_marginal",
            "conditioned": selection is not None, "converged": False}


def project_run(directory, *, axes, output, selection_file=None):
    """Project a saved FES without running dynamics; persist source provenance."""
    import json
    from pathlib import Path
    from ..checkpoint import load_checkpoint, atomic_savez
    from ..io import file_sha256, write_json

    source = Path(directory)
    config = json.loads((source / "input" / "resolved.json").read_text())
    summary = json.loads((source / "summary.json").read_text())
    if summary.get("measure", summary.get("sampling_measure", "cv_density")) != "cv_density":
        raise ValueError("projection requires a CV-density FES")
    selection = None if selection_file is None else np.load(selection_file, allow_pickle=False)
    with load_checkpoint(source / "fes.npz") as data:
        free = data["free_energy"]
        edges = [data[f"edges_{i}"] for i in range(free.ndim)]
        result = marginalize_fes(free, edges, config["beta"], axes=axes, mask=data["mask"], selection=selection)
    axes = result["axes"]
    target = Path(output)
    if target.exists() and any(target.iterdir()):
        raise ValueError("projection output must be a new or empty directory")
    target.mkdir(parents=True, exist_ok=True)
    atomic_savez(target / "pmf.npz", kind="arrays", free_energy=result["free_energy"],
                 mask=result["mask"], coverage=result["coverage"],
                 **{f"edges_{i}": e for i, e in enumerate(result["edges"])})
    coordinates = np.meshgrid(*[0.5 * (e[1:] + e[:-1]) for e in result["edges"]], indexing="ij")
    np.savetxt(target / "pmf.csv", np.column_stack([c.ravel() for c in coordinates] +
               [result[k].ravel() for k in ("free_energy", "mask", "coverage")]), delimiter=",", comments="",
               header=",".join([f"cv{a}_au" for a in axes] + ["free_energy_hartree", "masked", "coverage"]))
    public = {"task": "fes_projection", "method": result["method"], "measure": "cv_density",
              "source": str(source.resolve()), "source_sha256": file_sha256(source / "fes.npz"),
              "axes": list(axes), "beta": config["beta"], "conditioned": result["conditioned"],
              "selection_sha256": None if selection_file is None else file_sha256(selection_file),
              "supported_bins": int((~result["mask"]).sum()), "total_bins": result["mask"].size,
              "converged": False,
              "warnings": ["Incomplete integration slices are masked; projection does not establish FES or rate convergence."]}
    write_json(target / "summary.json", public)
    return public
