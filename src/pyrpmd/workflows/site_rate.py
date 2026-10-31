"""Basin-normalized RPMD escape rates for differentiable scalar centroid CVs.

The umbrella measure is ordinary CV density. Constrained RATTLE parents have
the positional measure delta(s-s*) sqrt(z) exp(-beta_P H_P). The inverse metric
weight z**(-1/2) converts their child fluxes to the conditional delta measure;
the QTST surface speed is 1 / (sqrt(2*pi*beta) <z**(-1/2)>_constrained).
See Sprik and Ciccotti, JCP 109, 7737 (1998), doi:10.1063/1.477419.
No gas-phase volume, electronic weight, or extra distance Jacobian is used.

Positions and all CV values remain continuous and unwrapped. A basin names
one explicit local interval, even when the physical potential is periodic.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from ..analysis.fes import (
    _block_resample,
    _histogram,
    solve_mbar,
    umbrella_reduced_potentials,
)
from ..analysis.recrossing import summarize_recrossing
from ..checkpoint import atomic_savez, load_checkpoint
from ..constants import ATOMIC_TIME_S
from ..cv import Cartesian
from ..cv_periodic import FractionalCoordinate, LinearProjection
from ..dynamics import Dynamics, State
from ..parallel import task_potential, root_call, run_tasks
from ..propagation import loop_options
from ..units import parse_quantity, unit_factor
from .fes import resolve_umbrellas, run_fes
from .common import _dimension, _fingerprint, checked_integer, jsonable, write_json

from .rate import (
    _recrossing_options,
    _recrossing_dynamics,
    _recrossing_spec,
    constrained_step,
    project_momenta,
    project_positions,
)


_CONSTANT_CV_METHODS = {
    cls: {name: getattr(cls, name, None) for name in (
        "value", "value_and_jacobian", "centroid_value_gradient",
        "centroid_value_gradient_hessian", "_affine_gradient",
    )}
    for cls in (Cartesian, LinearProjection, FractionalCoordinate)
}


class SiteSurface:
    """Shift a scalar differentiable centroid CV to a zero-level constraint."""

    def __init__(self, cv, value):
        self.cv = cv
        self.value = float(value)

    def centroid_value_gradient(self, q):
        value, gradient = _centroid_value_gradient(self.cv, q)
        return value - self.value, gradient


def _centroid_value_gradient(cv, q):
    """Evaluate the centroid contract, including a scalar Python CV plugin."""
    q = np.asarray(q, dtype=float)
    method = getattr(cv, "centroid_value_gradient", None)
    if callable(method):
        value, gradient = method(q)
    else:
        # A plugin declaring estimator='centroid' supplies its centroid
        # derivative through the same full-ring protocol at one bead.
        result = cv.value_and_jacobian(q[None])
        value, gradient = result.value[0], result.jacobian[0, 0]
    gradient = np.asarray(gradient, dtype=float)
    if (np.ndim(value) != 0 or not np.isfinite(value)
            or gradient.shape != q.shape or not np.isfinite(gradient).all()):
        raise ValueError("site CV must supply a finite scalar value and Cartesian centroid gradient")
    return float(value), gradient


def _mass_metric(cv, q, masses):
    _, gradient = _centroid_value_gradient(cv, q)
    metric = float(np.sum(gradient * gradient / np.asarray(masses)[:, None]))
    if not np.isfinite(metric) or metric <= 1e-28:
        raise ValueError("site CV has a singular mass metric")
    return metric


def _validate_centroid_contract(cv, geometry, beads):
    """Reject inconsistent plugin values/Jacobians before either sampler runs."""
    count = max(2, int(beads))
    offsets = np.arange(count, dtype=float) - (count - 1) / 2
    direction = np.sin(np.arange(geometry.size) + 1).reshape(geometry.shape)
    probe = geometry[None] + 1e-4 * offsets[:, None, None] * direction[None] / count
    value, gradient = _centroid_value_gradient(cv, probe.mean(axis=0))
    result = cv.value_and_jacobian(probe)
    if (np.shape(result.value) != (1,) or np.shape(result.jacobian) != (1,) + probe.shape
            or not np.allclose(result.value, [value], rtol=1e-9, atol=1e-10)
            or not np.allclose(result.jacobian, gradient[None, None] / count, rtol=1e-8, atol=1e-10)):
        raise ValueError("site CV must obey its declared centroid value/Jacobian contract")


def _setup(config, system):
    if _recrossing_dynamics(config)[0] != "rpmd":
        raise ValueError("site_rate does not yet support TRPMD recrossing")
    if _recrossing_spec(config).get("child_batch_pairs", 1) != 1:
        raise ValueError("site_rate does not yet support batched recrossing checkpoints")
    if system.fixed_atoms:
        raise ValueError("site_rate requires all atoms mobile; use surface_rate for fixed layers")
    table = config.get("site_rate", {})
    if not isinstance(table, dict):
        raise ValueError("site_rate must be a table")
    if any(key in config for key in ("reaction", "rate", "collision", "outcomes")):
        raise ValueError("site_rate uses basin normalization, without gas-phase reaction/rate settings")
    if config.get("adaptive_umbrella") or config.get("metadynamics"):
        raise ValueError("site_rate currently requires fixed local umbrella windows")
    analysis = config.get("analysis", {})
    if analysis.get("measure", "cv_density") != "cv_density":
        raise ValueError("site_rate requires analysis.measure='cv_density'")
    if analysis.get("method", "mbar") != "mbar":
        raise ValueError("site_rate currently requires analysis.method='mbar'")
    if config.get("dynamics", {}).get("loop_backend") == "native":
        raise ValueError("site_rate uses checkpointed Python loops; use loop_backend='auto' or 'python'")
    if config.get("sampling", {}).get("thermostat", "andersen") != "andersen":
        raise ValueError("site_rate constrained parents currently use the Andersen thermostat")
    if "dividing_surface" in config.get("recrossing", {}):
        raise ValueError("Declare the site dividing surface only in site_rate.dividing_surface")
    stack, specs, centers, stiffness = resolve_umbrellas(config, system)
    if stack.dimension != 1 or len(stack.cvs) != 1:
        raise ValueError("site_rate requires exactly one umbrella CV dimension")
    cv = stack.cvs[0]
    builtin_affine = type(cv) in _CONSTANT_CV_METHODS
    constant_metric = builtin_affine and all(
        getattr(type(cv), name, None) is original
        for name, original in _CONSTANT_CV_METHODS[type(cv)].items()
    )
    if builtin_affine and any(name in cv.__dict__ for name in (
        "value", "value_and_jacobian", "centroid_value_gradient",
        "centroid_value_gradient_hessian", "_affine_gradient",
    )):
        raise ValueError("site_rate does not accept overridden CV methods on built-ins; use a custom centroid CV")
    if cv.estimator != "centroid" or cv.periods != (None,):
        raise ValueError("site_rate requires a continuous centroid CV and nonperiodic local umbrella bias")
    declared = config.get("cvs", [])
    if "cv" not in table and len(declared) != 1:
        raise ValueError("site_rate.cv is required when more than one CV is declared")
    name = table.get("cv", cv.name)
    if name != cv.name:
        raise ValueError("site_rate.cv must match the single selected umbrella CV")
    if "basin" not in table or "dividing_surface" not in table:
        raise ValueError("site_rate requires an explicit basin and dividing_surface")
    raw_basin = table["basin"]
    if not isinstance(raw_basin, (list, tuple, np.ndarray)) or len(raw_basin) != 2:
        raise ValueError("site_rate.basin must contain [lower, upper]")
    unit = cv.units[0]
    spec = specs[0]
    for declaration in (spec, config.get("umbrella", {})):
        if isinstance(declaration, dict) and any(key in declaration for key in ("period", "periods", "periodic", "wrap")):
            raise ValueError("site_rate does not accept periodic or wrapped CV/bias declarations")
    input_unit = spec.get("unit", spec.get("units", unit)) if isinstance(spec, dict) else unit
    if not isinstance(input_unit, str):
        if len(input_unit) != 1:
            raise ValueError("site_rate CV input units must have one entry")
        input_unit = input_unit[0]
    scale = 1.0 if input_unit == "dimensionless" else unit_factor(input_unit, _dimension(unit))

    def quantity(value):
        result = parse_quantity(value, _dimension(unit))
        return result if isinstance(value, (str, list, tuple)) else result * scale

    basin = [quantity(value) for value in raw_basin]
    surface = quantity(table["dividing_surface"])
    direction = table.get("direction", 1)
    if isinstance(direction, (bool, np.bool_)) or not isinstance(direction, (int, np.integer)) or direction not in (-1, 1):
        raise ValueError("site_rate.direction must be the integer +1 or -1")
    if not basin[1] > basin[0]:
        raise ValueError("site_rate.basin requires lower < upper")
    endpoint = basin[1] if direction == 1 else basin[0]
    if not np.isclose(surface, endpoint, rtol=1e-12, atol=1e-12 * (basin[1] - basin[0])):
        raise ValueError("site dividing surface must be the upper basin boundary for +1 or lower for -1")
    surface = endpoint
    _validate_centroid_contract(cv, system.geometry, config["beads"])
    metric = _mass_metric(cv, system.geometry, system.masses)
    beta = float(config["beta"])
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("site_rate requires positive finite beta")
    options = _recrossing_options(config)
    bootstrap = _pmf_bootstrap_count(config)
    if bootstrap == 1:
        raise ValueError("analysis.bootstrap must be zero or at least two")
    checked_integer(analysis.get("seed", config.get("seed", 0)), "analysis.seed", 0)
    checked_integer(analysis.get("min_count", 1), "analysis.min_count")
    threshold = float(analysis.get("overlap_threshold", 1e-6))
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("analysis.overlap_threshold must lie strictly between zero and one")
    bins = analysis.get("bins", 40)
    if np.isscalar(bins):
        bins = [bins]
    if len(bins) != 1:
        raise ValueError("site_rate analysis.bins requires one histogram axis")
    if np.isscalar(bins[0]):
        checked_integer(bins[0], "analysis.bins", 2)
    else:
        edges = np.asarray(bins[0], dtype=float)
        if edges.ndim != 1 or len(edges) < 3 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
            raise ValueError("analysis.bins requires finite increasing edges")
    if "ranges" in analysis:
        ranges = analysis["ranges"]
        if len(ranges) != 1 or len(ranges[0]) != 2:
            raise ValueError("site_rate analysis.ranges requires [[lower, upper]]")
        limits = [parse_quantity(value, _dimension(unit)) for value in ranges[0]]
        if limits[1] <= limits[0]:
            raise ValueError("analysis.ranges must have lower < upper")
    sampling = config["sampling"]
    checked_integer(sampling["equilibration_steps"], "sampling.equilibration_steps", 0)
    production = checked_integer(sampling["production_steps"], "sampling.production_steps")
    interval = checked_integer(sampling.get("sample_interval", sampling.get("interval_steps", 1)), "sampling.sample_interval")
    if production < interval:
        raise ValueError("production_steps must include at least one saved sample")
    checked_integer(sampling.get("replicas", 1), "sampling.replicas")
    checked_integer(sampling.get("andersen_interval", 10), "sampling.andersen_interval")
    checked_integer(config.get("output", {}).get("checkpoint_steps", 1000), "output.checkpoint_steps")
    metadata = {
        "ready": True,
        "cv": name,
        "cv_type": type(cv).__name__,
        "unit": unit,
        "estimator": "centroid",
        "measure": "cv_density",
        "basin": basin,
        "dividing_surface": surface,
        "direction": int(direction),
        "mass_metric": metric if constant_metric else None,
        "flux_prefactor_au": float(np.sqrt(metric / (2 * np.pi * beta))) if constant_metric else None,
        "normalization": "density(s*) / integral_basin density(s) ds",
        "coordinate_images": "continuous unwrapped local interval",
        "constraint_measure": (
            "constant mass metric; no position-dependent Fixman correction" if constant_metric
            else "delta(s-s*) sqrt(z); child fluxes reweighted by z^(-1/2)"
        ),
        "rate_unit": "s^-1",
        "umbrella_windows": len(centers),
    }
    # Keep the historical metadata/fingerprints unchanged for the established
    # linear-coordinate route. Nonlinear checkpoints have a distinct identity.
    if not constant_metric:
        metadata.update(metric_estimator="constrained_parent_inverse_sqrt",
                        reference_mass_metric=metric)
    return cv, metadata, options


def validate_site_rate(config, system):
    """Validate without propagating a PES; return JSON preparation metadata."""
    return jsonable(_setup(config, system)[1])


def _pmf_bootstrap_count(config):
    options = config.get("analysis", {})
    value = options.get("bootstrap", options.get("bootstrap_replicates", options.get("bootstrap_samples", 20)))
    return checked_integer(value, "analysis.bootstrap", 0)


def estimate_site_qtst(data, beta, mass_metric, basin, dividing_surface, *, constrained_mass_metrics=None):
    """Estimate a local QTST rate from a supported, connected 1D histogram.

    Basin probability integrates the piecewise-constant *bin density* using
    exact overlap lengths, including fractional end bins. Surface density is
    the linear interpolation of adjacent bin-average densities at bin centers.
    This is a finite-bin estimator: it does not evaluate a mathematical delta
    function exactly, extrapolate tails, or fill a missing histogram bin.
    Both density and basin mass use the same MBAR normalization.

    For a nonlinear coordinate pass ``mass_metric=None`` and the per-parent
    ``constrained_mass_metrics``. These must come from the canonical RATTLE
    surface ensemble, not unconstrained or biased umbrella configurations.
    """
    beta = float(beta)
    if constrained_mass_metrics is not None:
        if mass_metric is not None:
            raise ValueError("specify a constant mass metric or constrained metric samples, not both")
        metrics = np.asarray(constrained_mass_metrics, dtype=float)
        if metrics.ndim != 1 or not metrics.size or not np.isfinite(metrics).all() or np.any(metrics <= 1e-28):
            raise ValueError("constrained mass metrics must be a nonempty positive finite vector")
        inverse_mean = float(np.mean(1 / np.sqrt(metrics)))
        metric = 1 / inverse_mean**2
    elif mass_metric is not None:
        metric = float(mass_metric)
    else:
        raise ValueError("QTST requires a constant mass metric or constrained metric samples")
    if not np.isfinite(beta) or beta <= 0 or not np.isfinite(metric) or metric <= 0:
        raise ValueError("QTST requires positive finite beta and mass metric")
    limits = np.asarray(basin, dtype=float)
    surface = float(dividing_surface)
    if limits.shape != (2,) or not np.isfinite(limits).all() or limits[1] <= limits[0] or not np.isfinite(surface):
        raise ValueError("QTST requires a finite nonempty basin and surface")
    if data.get("measure", "cv_density") != "cv_density":
        raise ValueError("site QTST requires the ordinary CV-density measure")
    if len(data["edges"]) != 1:
        raise ValueError("site QTST requires a one-dimensional histogram")
    edge = np.asarray(data["edges"][0], dtype=float)
    density = np.asarray(data["density"], dtype=float)
    mask = np.asarray(data["mask"], dtype=bool)
    if edge.ndim != 1 or len(edge) < 3 or not np.isfinite(edge).all() or np.any(np.diff(edge) <= 0):
        raise ValueError("invalid site histogram edges")
    if density.shape != (len(edge) - 1,) or mask.shape != density.shape:
        raise ValueError("site histogram density/mask shape mismatch")
    widths = np.diff(edge)
    if "probability_mass" in data:
        mass = np.asarray(data["probability_mass"], dtype=float)
        if mass.shape != density.shape or not np.allclose(mass, density * widths, rtol=1e-10, atol=1e-15, equal_nan=True):
            raise ValueError("histogram density and probability mass have inconsistent normalization")
    result = {
        "ready": False,
        "status": "insufficient_support",
        "reasons": [],
        "k_qtst_au": None,
        "k_qtst_s": None,
        "rate_unit": "s^-1",
        "surface_density": None,
        "basin_probability": None,
        "mass_metric": metric,
        "flux_prefactor_au": float(np.sqrt(metric / (2 * np.pi * beta))),
        "basin": limits.tolist(),
        "dividing_surface": surface,
        "measure": "cv_density",
        "estimator": "finite_bin_mbar_density",
        "surface_estimator": "linear interpolation of neighboring bin-average densities at bin centers",
        "basin_estimator": "piecewise-constant density integrated over the declared basin",
    }
    if constrained_mass_metrics is not None:
        result.update(mass_metric=None, effective_surface_mass_metric=metric,
                      surface_inverse_sqrt_metric_mean=inverse_mean,
                      metric_estimator="constrained_parent_inverse_sqrt")
    if not data.get("connected", False):
        result["status"] = "disconnected_overlap"
        result["reasons"].append("umbrella_overlap_disconnected")
        return result
    if limits[0] < edge[0] or limits[1] > edge[-1]:
        result["reasons"].append("basin_not_fully_inside_histogram")
        return result
    mid = (edge[:-1] + edge[1:]) / 2
    if surface < mid[0] or surface > mid[-1]:
        result["reasons"].append("surface_not_bracketed_by_histogram_centers")
        return result
    right = int(np.searchsorted(mid, surface, side="left"))
    if right < len(mid) and surface == mid[right]:
        left = right
        fraction = 0.0
    else:
        left = right - 1
        fraction = float((surface - mid[left]) / (mid[right] - mid[left]))
    overlap = np.maximum(0, np.minimum(edge[1:], limits[1]) - np.maximum(edge[:-1], limits[0]))
    basin_bins = np.flatnonzero(overlap > 0)
    lo = min(int(basin_bins[0]), left)
    hi = max(int(basin_bins[-1]), right)
    invalid = mask | ~np.isfinite(density) | (density <= 0)
    result["support_bins"] = [lo, hi]
    result["surface_bins"] = [left, right]
    if invalid[lo:hi + 1].any():
        result["reasons"].append("masked_or_empty_bin_in_basin_surface_support")
        result["unsupported_bins"] = (lo + np.flatnonzero(invalid[lo:hi + 1])).tolist()
        return result
    population = float(np.dot(density[basin_bins], overlap[basin_bins]))
    rho = float((1 - fraction) * density[left] + fraction * density[right])
    rate = result["flux_prefactor_au"] * rho / population
    if not np.isfinite(rate) or rate <= 0 or not np.isfinite(population) or population <= 0:
        result["reasons"].append("nonfinite_or_zero_basin_flux")
        return result
    result.update(ready=True, status="ready", surface_density=rho, basin_probability=population,
                  k_qtst_au=float(rate), k_qtst_s=float(rate / ATOMIC_TIME_S))
    return result


def _estimate_pmf(sampled, config, system, output):
    cv, metadata, options = _setup(config, system)
    data = sampled["data"]
    general = metadata["mass_metric"] is None
    metric = 1.0 if general else metadata["mass_metric"]
    estimate = estimate_site_qtst(data, config["beta"], metric,
                                  metadata["basin"], metadata["dividing_surface"])
    estimate["preparation"] = metadata
    estimate["warnings"] = list(sampled["summary"].get("warnings", []))
    estimate["warnings"].append("QTST uses finite histogram bins; bin-size convergence has not been established.")
    count = _pmf_bootstrap_count(config)
    draws = []
    draw_records = []
    arrays = [np.asarray(stream["samples"]) for stream in sampled["streams"]]
    blocks = np.maximum(1, np.ceil(data["statistical_inefficiency"]).astype(int))
    independent_blocks = [len(values) / block for values, block in zip(arrays, blocks)]
    if estimate["ready"] and count:
        centers = np.asarray([stream["center"] for stream in sampled["streams"]])
        stiffness = np.asarray([stream["stiffness"] for stream in sampled["streams"]])
        sizes = np.asarray([len(values) for values in arrays])
        rng = np.random.default_rng(np.random.SeedSequence([
            int(config.get("analysis", {}).get("seed", config.get("seed", 0))), 1201]))
        analysis = config.get("analysis", {})
        values = np.concatenate(arrays)
        reduced = umbrella_reduced_potentials(values, centers, stiffness, config["beta"], periods=(None,))
        offsets = np.r_[0, np.cumsum(sizes)]
        for draw in range(count):
            # The circular-block draw order is unchanged. Reusing columns of
            # the common bias matrix avoids rebuilding every umbrella energy.
            indices = np.concatenate([
                _block_resample(np.arange(n), block, rng) + offset
                for n, block, offset in zip(sizes, blocks, offsets[:-1])
            ])
            record = {"draw": draw, "valid": False, "reason": None}
            mbar = solve_mbar(reduced[:, indices], sizes, overlap_threshold=float(analysis.get("overlap_threshold", 1e-6)),
                              return_state_weights=False,
                              initial_free_energies=data["window_free_energies"])
            if not mbar.converged:
                record["reason"] = "mbar_not_converged"
            elif len(set(mbar.components)) != 1:
                record["reason"] = "disconnected_overlap"
            else:
                mass, density, raw_count, _, mask = _histogram(
                    values[indices], mbar.weights, data["edges"], config["beta"],
                    min_count=int(analysis.get("min_count", 1)))
                trial = estimate_site_qtst(
                    {"edges": data["edges"], "density": density, "probability_mass": mass,
                     "mask": mask, "connected": True, "measure": "cv_density"},
                    config["beta"], metric, metadata["basin"], metadata["dividing_surface"])
                if trial["ready"]:
                    draws.append(trial["k_qtst_s"])
                    record.update(valid=True, k_qtst_s=trial["k_qtst_s"])
                else:
                    record["reason"] = trial["reasons"][0]
            draw_records.append(record)
    failures = [record for record in draw_records if not record["valid"]]
    failure_counts = {}
    for record in failures:
        failure_counts[record["reason"]] = failure_counts.get(record["reason"], 0) + 1
    reasons = []
    if count < 2:
        reasons.append("fewer_than_two_requested_pmf_bootstrap_draws")
    if len(draws) != count:
        reasons.append("pmf_bootstrap_draws_with_insufficient_support")
    if any(value < 2 for value in independent_blocks):
        reasons.append("fewer_than_two_correlation_blocks_in_a_pmf_stream")
    complete = estimate["ready"] and len(draws) >= 2 and not reasons
    estimate["pmf_uncertainty"] = {
        "available": complete, "reasons": reasons, "requested": count, "valid": len(draws),
        "attempted": len(draw_records), "draws": draw_records,
        "failures": failures, "failure_counts": failure_counts,
        "block_lengths": blocks.tolist(), "blocks_per_stream": independent_blocks,
        "method": "independent circular correlated-block resampling within each umbrella stream",
        "standard_error_s": float(np.std(draws, ddof=1)) if complete else None,
        "percentile_95_ci_s": np.quantile(draws, [.025, .975]).tolist() if complete else None,
    }
    estimate["_bootstrap_qtst_s"] = draws
    if general:
        # The histogram determines rho(s*) / P(basin), but the nonlinear
        # surface speed is unknown until constrained parents are available.
        speed = float(np.sqrt(1 / (2 * np.pi * config["beta"])))
        estimate["_bootstrap_density_ratio"] = np.asarray(draws) * ATOMIC_TIME_S / speed
        estimate["_density_uncertainty_available"] = complete
        estimate["mass_metric"] = None
        estimate["flux_prefactor_au"] = None
        estimate["k_qtst_au"] = estimate["k_qtst_s"] = None
        estimate["surface_metric_status"] = "awaiting_constrained_parents" if estimate["ready"] else "histogram_unsupported"
        estimate["metric_estimator"] = "constrained_parent_inverse_sqrt"
        estimate["pmf_uncertainty"].update(available=False, standard_error_s=None, percentile_95_ci_s=None)
        for record in draw_records:
            if "k_qtst_s" in record:
                record["density_ratio"] = record.pop("k_qtst_s") * ATOMIC_TIME_S / speed
        estimate["_bootstrap_qtst_s"] = []
    write_json(Path(output) / "site_qtst.json", _public_pmf(estimate))
    atomic_savez(Path(output) / "site_qtst.npz", kind="site_rate_qtst",
                 edges=data["edges"][0], density=data["density"], mask=data["mask"],
                 probability_mass=data["probability_mass"], bootstrap_qtst_s=np.asarray(estimate["_bootstrap_qtst_s"]))
    tasks = []
    if estimate["ready"]:
        streams = sorted(sampled["streams"], key=lambda stream: (
            abs(float(stream["center"][0]) - metadata["dividing_surface"]),
            stream["window"], stream["replica"]))
        for chain in range(options["parent_replicas"]):
            tasks.append({"config": config, "system": system, "options": options,
                          "chain": chain, "output": str(output),
                          "initial_q": streams[chain % len(streams)]["last_q"]})
    return estimate, tasks


def _public_pmf(pmf):
    return {key: value for key, value in pmf.items() if not key.startswith("_")}


def _dynamics(config, system, provider, namespace, *identity):
    return Dynamics(
        provider, system.masses, config["beta"], config["beads"], config["dt_au"],
        seed=np.random.SeedSequence([int(config.get("seed", 0)), namespace, *identity]),
        native=config.get("native", config.get("dynamics", {}).get("native", True)),
        threads=config.get("parallel", {}).get("threads_per_rank", 1), **loop_options(config))


def _parent_task(task):
    config, system = task["config"], task["system"]
    cv, metadata, options = _setup(config, system)
    surface = SiteSurface(cv, metadata["dividing_surface"])
    chain = int(task["chain"])
    checkpoint = Path(task["output"]) / "checkpoints" / f"site_parent_{chain:04d}.npz"
    identity = _fingerprint(config, system, {
        "workflow_version": 1, "stage": "site_parent", "chain": chain,
        "surface": metadata, "options": options, "initial_q": task["initial_q"]})
    with task_potential(config["potential"], system,
                        reuse=config.get("parallel", {}).get("reuse_potentials", True)) as provider:
        dyn = _dynamics(config, system, provider, 1211, chain)
        elapsed, snapshots = 0, []
        target = options["equilibration_steps"] + options["parent_samples"] * options["parent_interval_steps"]
        if checkpoint.exists():
            with load_checkpoint(checkpoint, expected_kind="site_rate_parent") as saved:
                if int(saved["payload_version"]) != 1:
                    raise ValueError("unsupported site parent checkpoint payload version")
                if str(saved["fingerprint"]) != identity:
                    raise ValueError(f"site parent checkpoint physics differs: {checkpoint}")
                state = State(saved["q"], saved["p"], int(saved["step"]), float(saved["time"]))
                dyn.rng.bit_generator.state = json.loads(str(saved["rng"]))
                elapsed = int(saved["elapsed"])
                snapshots = list(saved["snapshots"])
                dyn.propagation_stats = json.loads(str(saved["propagation"]))
            expected = max(0, elapsed - options["equilibration_steps"]) // options["parent_interval_steps"]
            if elapsed != state.step or not 0 <= elapsed <= target or len(snapshots) != expected:
                raise ValueError("invalid accepted prefix in site parent checkpoint")
            if not np.isclose(state.time, elapsed * dyn.dt, rtol=1e-10, atol=1e-12):
                raise ValueError("site parent checkpoint clock differs from its accepted prefix")
            for q in [state.q, *snapshots]:
                if np.shape(q) != dyn.integrator.shape or not np.isfinite(q).all() or abs(surface.centroid_value_gradient(np.mean(q, axis=0))[0]) > max(1e-8, 10 * options["constraint_tolerance"]):
                    raise ValueError("site parent checkpoint is not on its dividing surface")
            _, gradient = surface.centroid_value_gradient(state.q.mean(axis=0))
            normal_velocity = float(np.sum(gradient * state.p.mean(axis=0) / system.masses[:, None]))
            metric = _mass_metric(cv, state.q.mean(axis=0), system.masses)
            if abs(normal_velocity) > 1e-10 * max(1., np.sqrt(metric / config["beta"])):
                raise ValueError("site parent checkpoint momentum is not tangent to the surface")
        else:
            q = np.array(task["initial_q"], dtype=float, copy=True)
            value, gradient = surface.centroid_value_gradient(q.mean(axis=0))
            if metadata["mass_metric"] is not None:
                # Exact linear shift avoids the nonlinear projector's cap.
                q -= value * gradient[None] / (metadata["mass_metric"] * system.masses[None, :, None])
            q = project_positions(q, surface, system.masses, options["constraint_tolerance"])
            state = State(q, dyn.sample_momenta())
            project_momenta(state.p, state.q, surface, system.masses)

        def save():
            atomic_savez(checkpoint, kind="site_rate_parent", payload_version=1,
                         fingerprint=identity, q=state.q, p=state.p, step=state.step, time=state.time,
                         rng=json.dumps(dyn.rng.bit_generator.state), elapsed=elapsed,
                         snapshots=np.asarray(snapshots).reshape((-1,) + state.q.shape),
                         propagation=json.dumps(dyn.propagation_stats))

        interval = checked_integer(config["sampling"].get("andersen_interval", 10), "sampling.andersen_interval")
        checkpoint_steps = checked_integer(config.get("output", {}).get("checkpoint_steps", 1000), "output.checkpoint_steps")
        save()
        try:
            while elapsed < target:
                old_rng = copy.deepcopy(dyn.rng.bit_generator.state)
                candidate = state.copy()
                try:
                    if candidate.step % interval == 0:
                        candidate.p[:] = dyn.sample_momenta()
                        project_momenta(candidate.p, candidate.q, surface, system.masses)
                    constrained_step(dyn, candidate, surface, options["constraint_tolerance"])
                except BaseException:
                    dyn.rng.bit_generator.state = old_rng
                    raise
                production = elapsed + 1 - options["equilibration_steps"]
                next_snapshots = snapshots
                if production > 0 and production % options["parent_interval_steps"] == 0:
                    next_snapshots = [*snapshots, candidate.q.copy()]
                state, elapsed, snapshots = candidate, elapsed + 1, next_snapshots
                dyn.propagation_stats["python_steps"] += 1
                if elapsed % checkpoint_steps == 0:
                    save()
        except BaseException:
            save()
            raise
        save()
        return {"chain": chain, "snapshots": np.asarray(snapshots), "checkpoint": str(checkpoint),
                "propagation": dyn.propagation_stats}


def _child_task(task):
    config, system = task["config"], task["system"]
    cv, metadata, options = _setup(config, system)
    surface = SiteSurface(cv, metadata["dividing_surface"])
    chain, parent, pair = (int(task[key]) for key in ("chain", "parent", "pair"))
    nsteps = options["child_steps"]
    q0 = np.asarray(task["q"], dtype=float)
    if q0.shape != (config["beads"], system.natoms, 3) or not np.isfinite(q0).all() or abs(surface.centroid_value_gradient(q0.mean(axis=0))[0]) > max(1e-8, 10 * options["constraint_tolerance"]):
        raise ValueError("site child must start on the declared surface")
    metric = _mass_metric(cv, q0.mean(axis=0), system.masses)
    weight = 1.0 if metadata["mass_metric"] is not None else 1 / np.sqrt(metric)
    checkpoint = Path(task["output"]) / "checkpoints" / f"site_child_{chain:04d}_{parent:05d}_{pair:04d}.npz"
    identity = _fingerprint(config, system, {
        "workflow_version": 1, "stage": "site_child", "chain": chain, "parent": parent,
        "pair": pair, "surface": metadata, "child_steps": nsteps, "q": q0})
    with task_potential(config["potential"], system,
                        reuse=config.get("parallel", {}).get("reuse_potentials", True)) as provider:
        dyn = _dynamics(config, system, provider, 1307, chain, parent, pair)
        if checkpoint.exists():
            with load_checkpoint(checkpoint, expected_kind="site_rate_child") as saved:
                if int(saved["payload_version"]) != 1:
                    raise ValueError("unsupported site child checkpoint payload version")
                if str(saved["fingerprint"]) != identity:
                    raise ValueError(f"site child checkpoint physics differs: {checkpoint}")
                states = [State(saved["q"][i], saved["p"][i], int(saved["steps"][i]), float(saved["times"][i])) for i in range(2)]
                numerator = np.array(saved["numerator"], copy=True)
                velocity = float(saved["velocity"])
                initial = float(saved["initial_energy"])
                final = np.array(saved["final_energy"], copy=True)
                dyn.rng.bit_generator.state = json.loads(str(saved["rng"]))
                dyn.propagation_stats = json.loads(str(saved["propagation"]))
            if numerator.shape != (2, nsteps) or not np.isfinite(numerator).all() or final.shape != (2,):
                raise ValueError("invalid site child checkpoint records")
            if any(state.q.shape != q0.shape or state.step > nsteps for state in states):
                raise ValueError("invalid accepted prefix in site child checkpoint")
            if states[1].step and states[0].step != nsteps:
                raise ValueError("site child branches have inconsistent accepted prefixes")
            if any(not np.isclose(state.time, state.step * dyn.dt, rtol=1e-10, atol=1e-12) for state in states):
                raise ValueError("site child checkpoint clock differs from its accepted prefix")
            if any(np.any(numerator[i, state.step:] != 0) for i, state in enumerate(states)):
                raise ValueError("site child checkpoint contains unaccepted flux records")
            if any(np.isfinite(final[i]) and state.step != nsteps for i, state in enumerate(states)):
                raise ValueError("site child has an endpoint energy before trajectory completion")
            if np.isinf(final).any() or any(np.any((numerator[i] != 0) & (numerator[i] != sign * velocity)) for i, sign in enumerate((1, -1))):
                raise ValueError("site child checkpoint contains invalid energy or signed flux")
        else:
            # Unconstrained Maxwell momenta give centroid variance z/beta.
            # Projecting these momenta would erase the reactive flux.
            p = dyn.sample_momenta()
            _, gradient = surface.centroid_value_gradient(q0.mean(axis=0))
            velocity = metadata["direction"] * float(np.sum(gradient * p.mean(axis=0) / system.masses[:, None]))
            states = [State(q0, p), State(q0, -p)]
            numerator = np.zeros((2, nsteps))
            initial = dyn.energy(states[0])
            final = np.full(2, np.nan)
        if not np.isfinite(velocity) or not np.isfinite(initial):
            raise ValueError("invalid initial site child flux or energy")

        def save():
            atomic_savez(checkpoint, kind="site_rate_child", payload_version=1, fingerprint=identity,
                         q=np.asarray([state.q for state in states]), p=np.asarray([state.p for state in states]),
                         steps=[state.step for state in states], times=[state.time for state in states],
                         numerator=numerator, velocity=velocity, initial_energy=initial, final_energy=final,
                         rng=json.dumps(dyn.rng.bit_generator.state), propagation=json.dumps(dyn.propagation_stats))

        checkpoint_steps = checked_integer(config.get("output", {}).get("checkpoint_steps", 1000), "output.checkpoint_steps")
        save()
        try:
            for branch, sign in enumerate((1, -1)):
                while states[branch].step < nsteps:
                    candidate = states[branch].copy()
                    dyn.step(candidate)  # No thermostat, constraint, or umbrella.
                    value, _ = surface.centroid_value_gradient(candidate.q.mean(axis=0))
                    contribution = sign * velocity * (metadata["direction"] * value > 0)
                    numerator[branch, candidate.step - 1] = contribution
                    states[branch] = candidate
                    dyn.propagation_stats["python_steps"] += 1
                    if sum(state.step for state in states) % checkpoint_steps == 0:
                        save()
                if not np.isfinite(final[branch]):
                    final[branch] = dyn.energy(states[branch])
                save()
        except BaseException:
            save()
            raise
        endpoint = float(np.max(np.abs(final - initial)) / config["beads"])
        result = {"chain": chain, "parent": parent, "pair": pair,
                "surface": metadata["dividing_surface"], "direction": metadata["direction"],
                "numerator": weight * numerator.sum(axis=0), "denominator": weight * abs(velocity),
                "endpoint_energy_error_hartree_per_bead": endpoint,
                "endpoint_energy_error_kbt": endpoint * config["beta"],
                "checkpoint": str(checkpoint), "propagation": dyn.propagation_stats}
        if metadata["mass_metric"] is None:
            result.update(mass_metric=metric, configuration_weight=float(weight))
        return result


def _child_tasks(parents, config, system, output):
    options = _recrossing_options(config)
    return [{"config": config, "system": system, "output": str(output),
             "chain": record["chain"], "parent": parent, "pair": pair, "q": q}
            for record in parents for parent, q in enumerate(record["snapshots"])
            for pair in range(options["child_pairs"])]


def _metric_reweighted_children(children):
    """Validate inverse-metric records and give every parent equal mass."""
    weights, counts = {}, {}
    for child in children:
        key = (child["chain"], child["parent"])
        metric = float(child.get("mass_metric", np.nan))
        weight = float(child.get("configuration_weight", np.nan))
        if (not np.isfinite(metric) or metric <= 1e-28 or not np.isfinite(weight)
                or not np.isclose(weight, 1 / np.sqrt(metric), rtol=1e-12, atol=0)):
            raise ValueError("invalid nonlinear site child mass metric or inverse-metric weight")
        if key in weights and not np.isclose(weight, weights[key], rtol=1e-12, atol=0):
            raise ValueError("children of one parent have inconsistent mass metrics")
        weights[key] = weight
        counts[key] = counts.get(key, 0) + 1
    normalized = []
    for child in children:
        count = counts[(child["chain"], child["parent"])]
        normalized.append({**child, "numerator": np.asarray(child["numerator"]) / count,
                           "denominator": child["denominator"] / count})
    return normalized, weights


def _complete_surface_qtst(pmf, weights, beta):
    """Combine ordinary CV density with the constrained surface measure."""
    pmf = copy.deepcopy(pmf)
    values = np.asarray(list(weights.values()), dtype=float)
    inverse_mean = float(values.mean())
    speed = 1 / (np.sqrt(2 * np.pi * beta) * inverse_mean)
    rate = speed * pmf["surface_density"] / pmf["basin_probability"]
    pmf.update(flux_prefactor_au=float(speed), k_qtst_au=float(rate),
               k_qtst_s=float(rate / ATOMIC_TIME_S), surface_metric_status="estimated",
               effective_surface_mass_metric=float(1 / inverse_mean**2),
               surface_inverse_sqrt_metric_mean=inverse_mean)
    pmf["surface_metric"] = {
        "parent_configurations": len(values), "inverse_sqrt_mean": inverse_mean,
        "inverse_sqrt_minimum": float(values.min()), "inverse_sqrt_maximum": float(values.max()),
        "reweighting_effective_samples": float(values.sum()**2 / np.dot(values, values)),
        "ensemble": "canonical constrained RATTLE parents",
        "uncertainty": "joint chain/block resampling with transmission coefficient",
    }
    conditional_draws = np.asarray(pmf["_bootstrap_density_ratio"]) * speed / ATOMIC_TIME_S
    pmf["_bootstrap_qtst_s"] = conditional_draws.tolist()
    density_available = pmf["_density_uncertainty_available"]
    pmf["pmf_uncertainty"].update(
        available=density_available,
        conditional_density_standard_error_s=float(conditional_draws.std(ddof=1)) if density_available else None,
    )
    return pmf


def _aggregate(children, pmf, config, output):
    options = _recrossing_options(config)
    general = pmf.get("metric_estimator") == "constrained_parent_inverse_sqrt"
    metric_weights = {}
    if general and children and pmf["ready"]:
        children, metric_weights = _metric_reweighted_children(children)
        pmf = _complete_surface_qtst(pmf, metric_weights, config["beta"])
    public_pmf = _public_pmf(pmf)
    warnings = list(pmf.get("warnings", []))
    result = {
        "task": "site_rate", "rate_unit": "s^-1", "converged": False,
        "status": pmf["status"] if not pmf["ready"] else "insufficient_recrossing",
        "rate_s": None, "rate_estimate_s": None, "rate_standard_error_s": None,
        "rate_percentile_95_ci_s": None, "kappa": None, "pmf": public_pmf,
        "k_qtst_s": pmf["k_qtst_s"], "warnings": warnings,
        "convergence_note": "A finite-time plateau is not bead, timestep, cell-size or statistical convergence.",
        "files": {"pmf": str(Path(output) / "site_qtst.json"),
                  "histogram": str(Path(output) / "pmf" / "fes.csv")},
    }
    if not children:
        if not pmf["ready"]:
            warnings.append("No recrossing trajectories were launched because the full basin/surface histogram is unsupported.")
        write_json(Path(output) / "summary.json", result)
        return result
    if not pmf["ready"]:
        raise ValueError("cannot combine site children with an unsupported QTST estimate")
    for child in children:
        if child["surface"] != pmf["dividing_surface"] or child["direction"] != pmf["preparation"]["direction"]:
            raise ValueError("site child belongs to another surface/direction")
    rng = np.random.default_rng(np.random.SeedSequence([int(config.get("seed", 0)), 1503]))
    statistics = summarize_recrossing(
        children, rng=rng,
        **({"extra_parent_observables": metric_weights, "return_bootstrap_indices": True} if general else {}),
        **{key: options[key] for key in (
            "child_steps", "plateau_fraction", "plateau_tolerance", "bootstrap"
        )},
    )
    keys = statistics["parent_ids"]
    num, den = statistics["parent_numerator"], statistics["parent_denominator"]
    curve, start = statistics["curve"], statistics["start"]
    kappa, drift = statistics["kappa"], statistics["drift"]
    blocks = statistics["parent_block_lengths"]
    bootstrap, kappa_draws = statistics["bootstrap_curves"], statistics["bootstrap_kappa"]
    recrossing_uncertainty = statistics["uncertainty"]
    # Keep the site's conservative primary policy. A long single chain can
    # still provide an explicitly conditional block error below.
    independent_chains = recrossing_uncertainty["independent_parent_chains"]
    recrossing_complete = recrossing_uncertainty["available"] and independent_chains >= 2
    curve_error = statistics["curve_standard_error"] if recrossing_complete else np.full_like(curve, np.nan)
    kappa_error = statistics["standard_error"] if recrossing_complete else None
    drift_error = statistics["drift_standard_error"] if recrossing_complete else None
    plateau = statistics["plateau_detected"]
    candidate = pmf["k_qtst_s"] * kappa
    complete = recrossing_complete and pmf["pmf_uncertainty"]["available"]
    rate_draws = np.array([])
    qtst_draws = np.array([])
    reasons = list(pmf["pmf_uncertainty"]["reasons"]) + recrossing_uncertainty["reasons"]
    if independent_chains < 2:
        reasons.append("fewer_than_two_independent_parent_chains")
    if complete:
        if general:
            # The same constrained configuration controls both the QTST
            # metric and kappa. Independent draws of these two factors would
            # destroy their covariance and misstate the rate uncertainty.
            parent_weights = np.asarray([metric_weights[key] for key in keys])
            inverse_means = np.asarray([parent_weights[ix].mean()
                                       for ix in statistics["bootstrap_parent_indices"]])
            density_draws = rng.choice(pmf["_bootstrap_density_ratio"], size=len(kappa_draws), replace=True)
            qtst_draws = density_draws / (np.sqrt(2 * np.pi * config["beta"]) * inverse_means * ATOMIC_TIME_S)
            rate_draws = kappa_draws * qtst_draws
            pmf["pmf_uncertainty"].update(
                standard_error_s=float(qtst_draws.std(ddof=1)),
                percentile_95_ci_s=np.quantile(qtst_draws, [.025, .975]).tolist(),
                method="independent umbrella blocks and joint constrained-parent chain/block metric resampling",
            )
        else:
            rate_draws = kappa_draws * rng.choice(pmf["_bootstrap_qtst_s"], size=len(kappa_draws), replace=True)
    if general:
        pmf["pmf_uncertainty"]["available"] = complete
        pmf["pmf_uncertainty"]["surface_metric_reasons"] = (
            recrossing_uncertainty["reasons"]
            + (["fewer_than_two_independent_parent_chains"] if independent_chains < 2 else [])
        )
        if not complete:
            pmf["pmf_uncertainty"].update(standard_error_s=None, percentile_95_ci_s=None)
        result["pmf"] = _public_pmf(pmf)
        write_json(Path(output) / "site_qtst.json", result["pmf"])
        histogram_path = Path(output) / "site_qtst.npz"
        if histogram_path.exists():
            with load_checkpoint(histogram_path, expected_kind="site_rate_qtst") as saved:
                histogram = {name: np.array(saved[name], copy=True)
                             for name in ("edges", "density", "mask", "probability_mass")}
            atomic_savez(histogram_path, kind="site_rate_qtst", **histogram,
                         bootstrap_qtst_s=qtst_draws,
                         metric_parent_ids=np.asarray(keys),
                         parent_inverse_sqrt_metric=np.asarray([metric_weights[key] for key in keys]))
    energy_error = max(child["endpoint_energy_error_kbt"] for child in children)
    if not plateau:
        warnings.append("The selected finite-time tail does not pass the positive-plateau diagnostic.")
    if energy_error > .1:
        warnings.append("A child endpoint energy error exceeds 0.1 kBT per bead; reduce the timestep.")
    if len(keys) < 20:
        warnings.append("Fewer than 20 parent configurations; statistical diagnostics are preliminary.")
    if not complete:
        warnings.append("Complete rate uncertainty is unavailable: " + "; ".join(reasons) + ".")
    times = np.arange(1, len(curve) + 1) * config["dt_au"]
    atomic_savez(Path(output) / "recrossing.npz", kind="site_rate_recrossing",
                 time_au=times, time_fs=times * ATOMIC_TIME_S / 1e-15,
                 kappa=curve, standard_error=curve_error, parent_ids=np.asarray(keys),
                 parent_numerator=num, parent_denominator=den, bootstrap_kappa=kappa_draws,
                 bootstrap_rate_s=rate_draws)
    np.savetxt(Path(output) / "recrossing.csv",
               np.column_stack([times, times * ATOMIC_TIME_S / 1e-15, curve, curve_error]),
               delimiter=",", comments="", header="time_au,time_fs,kappa,standard_error")
    result.update({
        "status": "finite_time_plateau_estimate" if plateau and energy_error <= .1 else "pilot_requires_convergence",
        "rate_s": candidate if plateau and energy_error <= .1 else None,
        "rate_estimate_s": candidate, "kappa": kappa, "kappa_standard_error": kappa_error,
        "rate_standard_error_s": float(rate_draws.std(ddof=1)) if complete else None,
        "rate_percentile_95_ci_s": np.quantile(rate_draws, [.025, .975]).tolist() if complete else None,
        "conditional_kappa_standard_error": statistics["standard_error"],
        "conditional_rate_standard_error_s": (
            pmf["k_qtst_s"] * statistics["standard_error"]
            if statistics["standard_error"] is not None else None
        ),
        "rate_uncertainty": {
            "available": complete, "reasons": reasons,
            "method": (
                "independent PMF blocks with jointly resampled parent metric and transmission, fixed basin and surface"
                if general else "independent PMF block-bootstrap and chain/parent-cluster bootstrap draws, fixed basin and surface"
            ),
            "conditional_quantity": "recrossing uncertainty holding QTST fixed",
            "pmf_bootstrap_valid": len(pmf["_bootstrap_qtst_s"]), "recrossing_bootstrap_valid": len(bootstrap),
            "parent_block_lengths": blocks,
            "recrossing_scope": recrossing_uncertainty["scope"],
            "recrossing_support": recrossing_uncertainty,
        },
        "plateau_detected": plateau, "plateau_start_au": float(times[start]),
        "plateau_diagnostics": statistics["plateau_diagnostics"],
        "plateau_drift": drift, "plateau_drift_standard_error": drift_error,
        "child_time_fs": float(times[-1] * ATOMIC_TIME_S / 1e-15),
        "parent_chains": statistics["parent_chains"], "parent_configurations": len(keys),
        "child_pairs": len(children), "child_trajectories": 2 * len(children),
        "maximum_endpoint_energy_error_kbt": energy_error,
    })
    result["files"]["recrossing"] = str(Path(output) / "recrossing.csv")
    write_json(Path(output) / "summary.json", result)
    return result


def run_site_rate(config, system, output, executor=None):
    """Sample local umbrellas, constrained parent chains, and paired children."""
    config = copy.deepcopy(config)
    root_call(validate_site_rate, config, system)
    config.setdefault("analysis", {})["measure"] = "cv_density"
    sampling_config = copy.deepcopy(config)
    # The rate bootstrap below needs the density/population ratio for each
    # correlated draw; a bootstrap of PMF ordinates alone does not supply it.
    sampling_config["analysis"]["bootstrap"] = 0
    sampled = run_fes(sampling_config, system, Path(output) / "pmf", executor=executor, return_data=True)
    pmf, tasks = root_call(_estimate_pmf, sampled, config, system, str(output))
    mapper = executor.map if executor is not None else lambda function, jobs: run_tasks(function, jobs, config.get("parallel", {}))
    parents = mapper(_parent_task, tasks)
    tasks = root_call(_child_tasks, parents, config, system, str(output))
    children = mapper(_child_task, tasks)
    return root_call(_aggregate, children, pmf, config, str(output))
