"""Resumable independent-history OPES and optional frozen-bias production."""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Mapping
import numpy as np

from ..checkpoint import load_checkpoint
from ..dynamics import Dynamics, State
from ..opes import OPESBias, normalized_weights
from ..parallel import make_task_potential, root_call
from ..propagation import loop_options
from ..units import parse_quantity
from .common import (_atomic_npz, _fingerprint, _dimension, checked_integer,
                     fingerprint_matches, resolve_cvs)
from .biased_sampling import SamplingRuntime, run_biased_sampling, dispatch_replicas
from ..analysis.fes_result import FESResult, write_fes_result, estimate_fes_uncertainty
from .sampling import fes_sampler, checkpoint_cv_config


OPES_FIELDS = {"cvs", "sigma", "sigma_au", "barrier", "barrier_au", "bias_factor", "pace",
               "freeze_after_steps", "freeze_equilibration_steps"}


def _stream_fingerprint(config, system, replica):
    return _fingerprint(checkpoint_cv_config(config), system, {"opes_replica": replica})


def resolve_opes(config, system):
    """Resolve units and reject unsupported combinations before loading a PES."""
    if any(system.pbc) or system.fixed_atoms:
        raise ValueError("OPES workflow currently requires a nonperiodic system with all atoms mobile")
    table = config.get("opes", {})
    if not isinstance(table, Mapping):
        raise ValueError("opes must be a table")
    if set(table) - OPES_FIELDS:
        raise ValueError("Unknown opes fields: " + ", ".join(sorted(set(table) - OPES_FIELDS)))
    if config.get("umbrella") or config.get("metadynamics") or config.get("plumed") or config.get("replica_exchange", {}).get("enabled"):
        raise ValueError("OPES workflow does not combine another sampling bias or replica exchange")
    checkpoint_cv_config(config)  # Validate direct object identity before loading a potential.
    cv, selected, scales = resolve_cvs(config, system, "opes")
    if "sigma" in table and "sigma_au" in table:
        raise ValueError("specify opes.sigma or sigma_au, not both")
    widths = table.get("sigma_au", table.get("sigma"))
    if cv.dimension == 1 and isinstance(widths, (str, int, float, np.number)):
        widths = [widths]
    if not isinstance(widths, (list, tuple, np.ndarray)) or len(widths) != cv.dimension:
        raise ValueError("opes.sigma needs one width per CV")
    sigma = [parse_quantity(value, _dimension(unit)) *
             (scale if "sigma_au" not in table and isinstance(value, (int, float, np.number)) else 1.0)
             for value, unit, scale in zip(widths, cv.units, scales)]
    if "barrier" in table and "barrier_au" in table:
        raise ValueError("specify opes.barrier or barrier_au, not both")
    if "barrier" not in table and "barrier_au" not in table:
        raise ValueError("opes.barrier is required")
    barrier = parse_quantity(table.get("barrier_au", table.get("barrier")), "energy")
    bias = OPESBias(cv, sigma, config["beta"], barrier=barrier, bias_factor=table.get("bias_factor", 10))
    sampling = config.get("sampling", {})
    neq = checked_integer(sampling.get("equilibration_steps", 0), "sampling.equilibration_steps", 0)
    nprod = checked_integer(sampling["production_steps"], "sampling.production_steps")
    interval = checked_integer(sampling.get("sample_interval", sampling.get("interval_steps", 1)), "sampling.sample_interval")
    pace = checked_integer(table.get("pace", 100), "opes.pace")
    freeze = checked_integer(table["freeze_after_steps"], "opes.freeze_after_steps") if "freeze_after_steps" in table else None
    relax = checked_integer(table.get("freeze_equilibration_steps", 0), "opes.freeze_equilibration_steps", 0)
    if freeze is not None and (freeze % pace or freeze < pace or freeze > nprod):
        raise ValueError("opes.freeze_after_steps must be a multiple of pace within production_steps")
    if relax and freeze is None:
        raise ValueError("opes.freeze_equilibration_steps requires freeze_after_steps")
    if min(nprod // interval, nprod // pace) < 1:
        raise ValueError("OPES production must contain at least one sample and one probability update")
    checked_integer(sampling.get("andersen_interval", 10), "sampling.andersen_interval")
    checked_integer(sampling.get("replicas", 1), "sampling.replicas")
    checked_integer(config.get("output", {}).get("checkpoint_steps", 1000), "output.checkpoint_steps")
    if sampling.get("thermostat", "andersen") != "andersen":
        raise ValueError("OPES workflow currently requires the Andersen thermostat")
    fes_sampler(config)  # Reuse the checked Andersen/integrator contract.
    dt = sampling.get("dt_au", config["dt_au"])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("OPES timestep must be positive and finite")
    if config.get("dynamics", {}).get("loop_backend", "auto") == "native":
        raise ValueError("OPES requires loop_backend='auto' or 'python'; native free-ring substeps remain available")
    analysis = config.get("analysis", {})
    method = analysis.get("method", "kde")
    if method not in ("kde", "reweighted_histogram"):
        raise ValueError("OPES analysis.method must be kde or reweighted_histogram")
    if analysis.get("measure", "cv_density") != "cv_density":
        raise ValueError("OPES estimates CV-density free energies, not rate_flux")
    for key in ("bootstrap", "bootstrap_replicates", "bootstrap_samples"):
        if checked_integer(analysis.get(key, 0), "analysis." + key, 0) != 0:
            raise ValueError("OPES bootstrap is not implemented; use independent histories and convergence studies")
    if method == "reweighted_histogram" and (freeze is None or (nprod // interval) * interval <= freeze + relax):
        raise ValueError("reweighted_histogram requires saved frozen-bias samples after freeze_after_steps + freeze_equilibration_steps")
    bins = analysis.get("bins", 40)
    if np.isscalar(bins):
        bins = [bins] * cv.dimension
    if len(bins) != cv.dimension:
        raise ValueError("analysis.bins must contain one count per OPES CV")
    bins = [checked_integer(b, "analysis.bins", 2) for b in bins]
    if np.prod(bins, dtype=float) > 1_000_000:
        raise ValueError("OPES analysis grid exceeds one million points")
    ranges = None
    if "ranges" in analysis:
        raw = analysis["ranges"]
        if not isinstance(raw, (list, tuple, np.ndarray)) or len(raw) != cv.dimension:
            raise ValueError("analysis.ranges needs one pair per CV")
        ranges = []
        for pair, unit, period in zip(raw, cv.units, cv.periods):
            if not isinstance(pair, (list, tuple, np.ndarray)) or len(pair) != 2:
                raise ValueError("analysis.ranges entries must contain [lower,upper]")
            lo, hi = [parse_quantity(value, _dimension(unit)) for value in pair]
            if not hi > lo or (period is not None and not np.isclose(hi - lo, period, rtol=1e-10, atol=1e-12)):
                raise ValueError("analysis.ranges must increase and periodic ranges must span one period")
            ranges.append([lo, hi])
    return cv, selected, {"sigma": sigma, "barrier": barrier, "bias_factor": bias.bias_factor,
        "pace": pace, "neq": neq, "nprod": nprod, "interval": interval, "freeze": freeze,
        "freeze_equilibration_steps": relax, "method": method, "bins": bins, "ranges": ranges,
        "min_count": checked_integer(analysis.get("min_count", 1), "analysis.min_count")}


def validate_opes(config, system):
    resolve_opes(config, system)


def _new_bias(cv, options, beta, state=None):
    return OPESBias(cv, options["sigma"], beta, barrier=options["barrier"],
                    bias_factor=options["bias_factor"], state=state)


class _OPESUpdate:
    """Probability adaptation/freeze policy; propagation is shared."""

    def __init__(self, bias, cv, options, beta):
        self.bias, self.cv, self.options, self.beta = bias, cv, options, beta

    def due(self, production):
        return (production % self.options["pace"] == 0
                and (self.options["freeze"] is None or production <= self.options["freeze"]))

    def snapshot(self):
        return self.bias.state_dict()

    def apply(self, value, state):
        self.bias.update(value, step=state.step, time=state.time)

    def restore(self, snapshot):
        self.bias = _new_bias(self.cv, self.options, self.beta, snapshot)


def _checkpoint_path(output, replica):
    return Path(output) / "checkpoints" / f"opes_{replica:04d}.npz"


def _checkpoint_clock(data, key):
    value = np.asarray(data[key])
    if (value.shape != () or value.dtype.kind not in "iu" or value < 0
            or value > np.iinfo(np.int64).max):
        raise ValueError("OPES checkpoint " + key + " must be a nonnegative integer scalar clock")
    return int(value)


def _read_stream(config, system, output, replica, cv, options):
    checkpoint = _checkpoint_path(output, replica)
    with load_checkpoint(checkpoint) as data:
        if not fingerprint_matches(data["fingerprint"], checkpoint_cv_config(config), system, {"opes_replica": replica}):
            raise ValueError(f"checkpoint physics differs: {checkpoint}")
        elapsed = _checkpoint_clock(data, "elapsed")
        if elapsed > options["neq"] + options["nprod"] or _checkpoint_clock(data, "step") != elapsed:
            raise ValueError("OPES checkpoint has an invalid sampling clock")
        state = State(data["q"], data["p"], elapsed, float(data["time"]))
        if state.q.shape != (config["beads"], len(system.masses), 3):
            raise ValueError("OPES checkpoint state shape differs from the system")
        expected_time = elapsed * config["sampling"].get("dt_au", config["dt_au"])
        if not np.isclose(state.time, expected_time, rtol=1e-10, atol=1e-12):
            raise ValueError("OPES checkpoint time disagrees with its step")
        bias_state = json.loads(str(data["bias_state"]))
        bias = _new_bias(cv, options, config["beta"], bias_state)
        adaptation = min(max(0, elapsed - options["neq"]), options["freeze"] or options["nprod"])
        updates = adaptation // options["pace"]
        expected_steps = options["neq"] + np.arange(1, updates + 1) * options["pace"]
        if len(bias) != updates or not np.array_equal(bias_state["steps"], expected_steps):
            raise ValueError("OPES checkpoint kernel history disagrees with its update clock")
        if not np.allclose(bias_state["times"], expected_steps * config["sampling"].get("dt_au", config["dt_au"]), rtol=1e-10, atol=1e-12):
            raise ValueError("OPES kernel times disagree with their update clock")
        nsamples = max(0, elapsed - options["neq"]) // options["interval"]
        samples, clocks = data["samples"].copy(), data["sample_steps"].copy()
        expected_clocks = options["neq"] + np.arange(1, nsamples + 1) * options["interval"]
        if samples.shape != (nsamples, cv.dimension) or not np.isfinite(samples).all() or not np.array_equal(clocks, expected_clocks):
            raise ValueError("OPES checkpoint samples disagree with their sampling clock")
        observable = {key: data[key].copy() for key in ("sample_bias", "sample_energies")}
        if any(value.shape != (nsamples,) or not np.isfinite(value).all() for value in observable.values()):
            raise ValueError("OPES checkpoint has invalid observables")
        rng = json.loads(str(data["rng"]))
        # Validate saved RNG before constructing an expensive potential.
        np.random.default_rng().bit_generator.state = rng
        return {"replica": replica, "state": state, "elapsed": elapsed, "rng": rng,
                "samples": samples, "sample_steps": clocks, **observable,
                "bias_state": bias_state, "propagation": json.loads(str(data["propagation"])),
                "checkpoint": str(checkpoint), "complete": elapsed == options["neq"] + options["nprod"]}


def _sample_replica(task):
    config, system, output, replica = task["config"], task["system"], task["output"], task["replica"]
    cv, _, options = resolve_opes(config, system)
    checkpoint = _checkpoint_path(output, replica)
    saved = _read_stream(config, system, output, replica, cv, options) if checkpoint.exists() else None
    if saved is not None and saved["complete"]:
        return saved
    fingerprint = _stream_fingerprint(config, system, replica)
    provider = make_task_potential(config["potential"], system)
    try:
        dynamics = Dynamics(provider, system.masses, config["beta"], config["beads"],
            config["sampling"].get("dt_au", config["dt_au"]),
            seed=np.random.SeedSequence([int(config.get("seed", 0)), 811, replica]),
            native=config.get("native", config.get("dynamics", {}).get("native", True)),
            threads=config.get("parallel", {}).get("threads_per_rank", 1), **loop_options(config))
        bias = _new_bias(cv, options, config["beta"], None if saved is None else saved["bias_state"])
        samples, clocks, biases, energies = [], [], [], []
        elapsed = 0
        if saved is None:
            state = State(np.repeat(system.geometry[None], config["beads"], axis=0), dynamics.sample_momenta())
        else:
            state, elapsed = saved["state"], saved["elapsed"]
            dynamics.rng.bit_generator.state = saved["rng"]
            dynamics.propagation_stats = saved["propagation"]
            samples, clocks = list(saved["samples"]), list(saved["sample_steps"])
            biases, energies = list(saved["sample_bias"]), list(saved["sample_energies"])

        runtime = SamplingRuntime(state, elapsed, samples, clocks, biases, energies)
        update = _OPESUpdate(bias, cv, options, config["beta"])

        def save(current):
            _atomic_npz(checkpoint, fingerprint=fingerprint, elapsed=current.elapsed,
                q=current.state.q, p=current.state.p, step=current.state.step, time=current.state.time,
                rng=json.dumps(dynamics.rng.bit_generator.state),
                bias_state=json.dumps(update.bias.state_dict(), allow_nan=False),
                samples=np.asarray(current.values).reshape(-1, cv.dimension),
                sample_steps=np.asarray(current.clocks, dtype=int), sample_bias=np.asarray(current.biases),
                sample_energies=np.asarray(current.energies), propagation=json.dumps(dynamics.propagation_stats))

        run_biased_sampling(runtime, dynamics, cv, neq=options["neq"], nprod=options["nprod"],
                            interval=options["interval"], sampling=config["sampling"],
                            checkpoint_steps=config.get("output", {}).get("checkpoint_steps", 1000),
                            update=update, save=save,
                            fallback_reason="OPES probability updates require the Python workflow")
        return _read_stream(config, system, output, replica, cv, options)
    finally:
        if hasattr(provider, "close"):
            provider.close()


def _analyze(streams, config, system, output, return_data):
    streams = list(streams)  # executor.map may return a one-shot iterator.
    if not streams:
        raise ValueError("OPES analysis requires at least one history")
    cv, _, options = resolve_opes(config, system)
    biases = [_new_bias(cv, options, config["beta"], stream["bias_state"]) for stream in streams]
    arrays, log_weights = [], []
    for stream in streams:
        selected = np.ones(len(stream["samples"]), dtype=bool)
        if options["method"] == "reweighted_histogram":
            selected = stream["sample_steps"] > options["neq"] + options["freeze"] + options["freeze_equilibration_steps"]
        samples = stream["samples"][selected].copy()
        if not len(samples):
            raise ValueError("OPES analysis has no samples in the selected production stage")
        arrays.append(samples)
        log_weights.append(config["beta"] * stream["sample_bias"][selected])
    bounds = options["ranges"]
    if bounds is None:
        combined = np.concatenate(arrays)
        bounds = []
        for j, period in enumerate(cv.periods):
            lo, hi = combined[:, j].min(), combined[:, j].max()
            width = max(float(hi - lo), options["sigma"][j])
            bounds.append([-period / 2, period / 2] if period is not None else [lo - 0.05 * width, hi + 0.05 * width])
    edges = [np.linspace(lo, hi, n + 1) for (lo, hi), n in zip(bounds, options["bins"])]
    for samples in arrays:
        for j, period in enumerate(cv.periods):
            if period is not None:
                samples[:, j] = (samples[:, j] - bounds[j][0]) % period + bounds[j][0]
    grid = np.stack(np.meshgrid(*[0.5 * (edge[1:] + edge[:-1]) for edge in edges], indexing="ij"), axis=-1)
    counts = np.asarray([np.histogramdd(values, bins=edges)[0].astype(int) for values in arrays])
    mask = counts < options["min_count"]
    raw, ess = [], []
    for index, bias in enumerate(biases):
        if options["method"] == "kde":
            raw.append(-bias.log_probability(grid) / config["beta"])
            ess.append(bias.effective_sample_size)
        else:
            weights = normalized_weights(log_weights[index])
            mass = np.histogramdd(arrays[index], bins=edges, weights=weights)[0]
            volume = np.prod(np.meshgrid(*[np.diff(edge) for edge in edges], indexing="ij"), axis=0)
            with np.errstate(divide="ignore"):
                raw.append(-np.log(mass / volume) / config["beta"])
            ess.append(float(1.0 / np.sum(weights**2)))
    raw = np.asarray(raw)
    mask |= ~np.isfinite(raw)
    common = ~np.any(mask, axis=0)
    free = np.full(grid.shape[:-1], np.nan)
    spread = np.full_like(free, np.nan)
    reference = None
    if common.any():
        reference = np.unravel_index(np.argmin(np.where(common, raw.mean(axis=0), np.inf)), common.shape)
        shifted = raw - raw[(slice(None),) + reference].reshape((-1,) + (1,) * cv.dimension)
        free[common] = shifted[:, common].mean(axis=0)
        spread = estimate_fes_uncertainty(shifted, kind="replica_spread", mask=~common)
    else:
        shifted = raw.copy()
        for i in range(len(streams)):
            if (~mask[i]).any():
                shifted[i] -= np.min(raw[i][~mask[i]])
    shifted[mask] = np.nan
    data = {"free_energy": free, "free_energy_per_replica": shifted,
            "replica_spread": spread, "standard_error": np.full_like(free, np.nan),
            "mask": ~common, "mask_per_replica": mask, "raw_count": counts.sum(axis=0),
            "raw_count_per_replica": counts, "edges": edges}
    output = Path(output)
    warnings = list(config.get("_warnings", [])) + [
        "Fixed-bandwidth, uncompressed OPES: cost grows with kernel count; no PLUMED numerical parity is claimed.",
        "No reaction rate follows from biased trajectory time. CV adequacy, equilibrium, bead and timestep convergence remain unverified.",
        "Weight-only effective sample size ignores autocorrelation. Replica spread is descriptive, not a standard error."]
    warnings.append("The KDE free energy retains finite-bandwidth smoothing and adaptation error."
        if options["method"] == "kde" else
        "Only frozen-bias samples after the requested relaxation are reweighted; adequate equilibration is the user's responsibility.")
    if not common.any():
        warnings.append("Independent histories have no common sampled bin; their aggregate FES is masked.")
    summary = {"task": "opes", "beta": config["beta"], "status": "finite_time_estimate" if all(stream.get("complete", True) for stream in streams) else "partial_finite_time_estimate", "method": "opes", "algorithm": OPESBias.method,
        "estimator": options["method"], "measure": "cv_density", "energy_unit": "hartree",
        "dimension": cv.dimension, "units": cv.units, "replicas": len(streams),
        "samples": sum(len(values) for values in arrays), "kernels_per_replica": [len(bias) for bias in biases],
        "log_z_per_replica": [bias.log_z for bias in biases], "weight_effective_sample_size": ess,
        "freeze_after_steps": options["freeze"], "freeze_equilibration_steps": options["freeze_equilibration_steps"],
        "common_support": bool(common.any()), "reference_bin": None if reference is None else list(reference),
        "converged": False, "propagation": [stream["propagation"] for stream in streams], "warnings": warnings,
        "sample_bias_convention": "physical V before the probability update; H_P contains P*V; log weights are beta*V",
        "files": {"data": str(output / "fes.npz"), "table": str(output / "fes.csv"),
                  "checkpoints": [stream["checkpoint"] for stream in streams]}}
    result = FESResult.from_data(data, method="opes", cv_units=cv.units, metadata=summary)
    summary = write_fes_result(output, result, summary)
    return {"summary": summary, "data": data, "streams": streams} if return_data else summary


def run_opes(config, system, output, executor=None, return_data=False):
    """Run independent OPES histories through the existing task scheduler."""
    validate_opes(config, system)
    streams = dispatch_replicas(config, system, output, _sample_replica, executor)
    return root_call(_analyze, streams, config, system, str(output), return_data)


def reanalyze_opes(config, system, output, return_data=False):
    """Recompute saved OPES estimates without constructing a potential."""
    def analyze():
        cv, _, options = resolve_opes(config, system)
        streams = [_read_stream(config, system, output, replica, cv, options)
                   for replica in range(config.get("sampling", {}).get("replicas", 1))]
        return _analyze(streams, config, system, output, return_data)
    return root_call(analyze)
