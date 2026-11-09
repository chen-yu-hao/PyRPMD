"""Resumable independent-history OPES and optional frozen-bias production."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from ..warm_start import initial_ring

from ..checkpoint import load_checkpoint
from ..bias import CompositeBias, FluxMetricBias
from ..cv import RPMDrateCoordinate
from ..dynamics import Dynamics, State
from ..opes import OPESBias
from ..analysis.frozen_bias import frozen_samples, weighted_histogram
from ..parallel import make_task_potential, root_call
from ..propagation import loop_options
from ..units import parse_quantity
from .common import (_atomic_npz, _fingerprint, _dimension, checked_integer,
                     fingerprint_matches, resolve_cvs)
from .biased_sampling import SamplingRuntime, run_biased_sampling, dispatch_replicas, failure_writer
from ..analysis.fes_result import FESResult, write_fes_result, estimate_fes_uncertainty
from .sampling import (fes_sampler, checkpoint_cv_config, read_operator_checkpoint,
                       validate_andersen_schedule)


OPES_FIELDS = {"cvs", "sigma", "sigma_au", "barrier", "barrier_au", "bias_factor", "pace",
               "freeze_after_steps", "freeze_equilibration_steps", "mode", "adaptive_sigma", "sigma_min", "sigma_min_au", "walkers_mode"}


def _stream_fingerprint(config, system, replica):
    """Adapt OPES CV identity and replica keys to the shared stream schema."""
    return _fingerprint(checkpoint_cv_config(config), system, {"opes_replica": replica})


def resolve_opes(config, system):
    """Resolve units and reject unsupported combinations before loading a PES."""
    validate_andersen_schedule(config, "opes")
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
    if "sigma_min" in table and "sigma_min_au" in table:
        raise ValueError("specify opes.sigma_min or sigma_min_au, not both")
    minimum = table.get("sigma_min_au", table.get("sigma_min"))
    if minimum is not None:
        if cv.dimension == 1 and isinstance(minimum, (str, int, float, np.number)):
            minimum = [minimum]
        if not isinstance(minimum, (list, tuple, np.ndarray)) or len(minimum) != cv.dimension:
            raise ValueError("opes.sigma_min needs one width per CV")
        minimum = [parse_quantity(value, _dimension(unit)) *
                   (scale if "sigma_min_au" not in table and isinstance(value, (int, float, np.number)) else 1.0)
                   for value, unit, scale in zip(minimum, cv.units, scales)]
    bias = OPESBias(cv, sigma, config["beta"], barrier=barrier, bias_factor=table.get("bias_factor", 10),
                    mode=table.get("mode", "metad"), adaptive_sigma=table.get("adaptive_sigma", False), sigma_min=minimum)
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
    fes_sampler(config)
    dt = sampling.get("dt_au", config["dt_au"])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("OPES timestep must be positive and finite")
    if config.get("dynamics", {}).get("loop_backend", "auto") == "native":
        raise ValueError("OPES requires loop_backend='auto' or 'python'; native free-ring substeps remain available")
    analysis = config.get("analysis", {})
    method = analysis.get("method", "kde")
    if method not in ("kde", "reweighted_histogram"):
        raise ValueError("OPES analysis.method must be kde or reweighted_histogram")
    measure = analysis.get("measure", "cv_density")
    if measure not in ("cv_density", "rate_flux"):
        raise ValueError("OPES analysis.measure must be cv_density or rate_flux")
    if measure == "rate_flux":
        if method != "reweighted_histogram":
            raise ValueError("rate_flux OPES requires frozen-bias reweighted_histogram, not a KDE estimate")
        if (cv.dimension != 1 or len(cv.cvs) != 1
                or not isinstance(cv.cvs[0], RPMDrateCoordinate)):
            raise ValueError("rate_flux OPES requires one centroid RPMDrate coordinate")
        FluxMetricBias(cv.cvs[0], system.masses, config["beta"])
        if config.get("adaptive_umbrella", {}).get("enabled"):
            raise ValueError("rate_flux OPES does not combine adaptive umbrellas")
    for key in ("bootstrap", "bootstrap_replicates", "bootstrap_samples"):
        if checked_integer(analysis.get(key, 0), "analysis." + key, 0) != 0:
            raise ValueError("OPES bootstrap is not implemented; use independent histories and convergence studies")
    if method == "reweighted_histogram" and (freeze is None or (nprod // interval) * interval <= freeze + relax):
        raise ValueError("reweighted_histogram requires saved frozen-bias samples after freeze_after_steps + freeze_equilibration_steps")
    if measure == "rate_flux" and nprod // interval - (freeze + relax) // interval < 2:
        raise ValueError("rate_flux reweighted_histogram requires at least two saved frozen-bias samples")
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
    options = {"sigma": sigma, "barrier": barrier, "bias_factor": bias.bias_factor,
        "bias_mode": bias.mode, "adaptive_sigma": bias.adaptive_sigma, "sigma_min": minimum,
        "pace": pace, "neq": neq, "nprod": nprod, "interval": interval, "freeze": freeze,
        "freeze_equilibration_steps": relax, "method": method, "measure": measure, "bins": bins, "ranges": ranges,
        "min_count": checked_integer(analysis.get("min_count", 1), "analysis.min_count")}
    mode = table.get("walkers_mode", "serial")
    if mode not in ("serial", "mpi-sync", "async-files"):
        raise ValueError("opes.walkers_mode must be serial, mpi-sync or async-files")
    if mode != "serial":
        from ._distributed_opes_common import _validate_shared_options
        _validate_shared_options(config, options)
    return cv, selected, options


def validate_opes(config, system):
    resolve_opes(config, system)


def _new_bias(cv, options, beta, state=None):
    return OPESBias(cv, options["sigma"], beta, barrier=options["barrier"],
                    bias_factor=options["bias_factor"], state=state, mode=options.get("bias_mode", "metad"),
                    adaptive_sigma=options.get("adaptive_sigma", False), sigma_min=options.get("sigma_min"))


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


def _resume_initialization_schema(config, system, output):
    """Choose an unambiguous initial convention for never-started replicas.

    Checkpointed streams always retain their own version. A partial legacy
    job also needs its missing streams to use the old initialization if the
    complete job is to match its uninterrupted pre-upgrade trajectory.
    """
    versions, missing = set(), False
    for replica in range(config.get("sampling", {}).get("replicas", 1)):
        path = _checkpoint_path(output, replica)
        if not path.exists():
            missing = True
            continue
        with load_checkpoint(path) as data:
            if not fingerprint_matches(data["fingerprint"], checkpoint_cv_config(config),
                                       system, {"opes_replica": replica}):
                raise ValueError(f"checkpoint physics differs: {path}")
            state = json.loads(str(data["bias_state"]))
            version = state.get("schema_version") if isinstance(state, dict) else None
            if isinstance(version, bool) or not isinstance(version, int) or version not in (1, 2):
                raise ValueError(f"invalid OPES initialization schema: {path}")
            versions.add(version)
    if missing and len(versions) > 1:
        raise ValueError("cannot initialize missing OPES replicas from mixed schema-1/schema-2 histories; existing checkpoints remain available for reanalysis")
    return 1 if versions == {1} else 2


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
        operators = read_operator_checkpoint(data, fes_sampler(config), elapsed, "OPES")
        return {"replica": replica, "state": state, "elapsed": elapsed, "rng": rng,
                **({"operator_state": operators} if operators is not None else {}),
                "samples": samples, "sample_steps": clocks, **observable,
                "bias_state": bias_state, "propagation": json.loads(str(data["propagation"])),
                "checkpoint": str(checkpoint), "complete": elapsed == options["neq"] + options["nprod"]}


def _opes_bias_summary(bias):
    state = bias.state_dict()
    steps = state.get("steps", [])
    return {"kind": "opes", "method": state.get("method"), "kernel_count": len(state.get("centers", [])),
            "last_kernel_step": steps[-1] if len(steps) else None, "log_z": state.get("log_z"),
            "barrier": state.get("barrier"), "bias_factor": state.get("bias_factor")}


def _sample_replica(task):
    config, system, output, replica = task["config"], task["system"], task["output"], task["replica"]
    cv, _, options = resolve_opes(config, system)
    sampler = fes_sampler(config)
    store_operators = sampler.operator_checkpoint
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
        initial_state = None if saved is None else saved["bias_state"]
        if saved is None and task.get("initialization_schema", 2) == 1:
            initial_state = _new_bias(cv, options, config["beta"]).state_dict()
            initial_state["schema_version"] = 1
            initial_state.pop("initial_bias", None)
        bias = _new_bias(cv, options, config["beta"], initial_state)
        samples, clocks, biases, energies = [], [], [], []
        elapsed = 0
        if saved is None:
            state = State(initial_ring(config, system), dynamics.sample_momenta())
        else:
            state, elapsed = saved["state"], saved["elapsed"]
            if "operator_state" in saved:
                dynamics.load_operator_state_dict(saved["operator_state"])
                store_operators = True
            dynamics.rng.bit_generator.state = saved["rng"]
            dynamics.propagation_stats = saved["propagation"]
            samples, clocks = list(saved["samples"]), list(saved["sample_steps"])
            biases, energies = list(saved["sample_bias"]), list(saved["sample_energies"])

        runtime = SamplingRuntime(state, elapsed, samples, clocks, biases, energies,
                                  operator_checkpoint=store_operators)
        update = _OPESUpdate(bias, cv, options, config["beta"])
        metric = (FluxMetricBias(cv.cvs[0], system.masses, config["beta"])
                  if options["measure"] == "rate_flux" else None)

        def total_bias(current):
            # An OPES rollback reconstructs update.bias. Always compose with
            # the current object rather than retaining a pre-rollback bias.
            return current if metric is None else CompositeBias(current, metric)

        def save(current):
            operators = ({"operator_state": json.dumps(dynamics.operator_state_dict(), allow_nan=False)}
                         if store_operators else {})
            _atomic_npz(checkpoint, fingerprint=fingerprint, elapsed=current.elapsed,
                q=current.state.q, p=current.state.p, step=current.state.step, time=current.state.time,
                rng=json.dumps(dynamics.rng.bit_generator.state),
                bias_state=json.dumps(update.bias.state_dict(), allow_nan=False),
                samples=np.asarray(current.values).reshape(-1, cv.dimension),
                sample_steps=np.asarray(current.clocks, dtype=int), sample_bias=np.asarray(current.biases),
                sample_energies=np.asarray(current.energies), propagation=json.dumps(dynamics.propagation_stats),
                **operators)

        run_biased_sampling(runtime, dynamics, cv, neq=options["neq"], nprod=options["nprod"],
                            interval=options["interval"], sampling=config["sampling"],
                            checkpoint_steps=config.get("output", {}).get("checkpoint_steps", 1000),
                            update=update, sampler=sampler, save=save,
                            active_bias=lambda elapsed, current: total_bias(current),
                            energy_bias=total_bias,
                            fallback_reason="OPES probability updates require the Python workflow",
                            failure=failure_writer(
                                Path(output) / "failures" / f"opes_{replica:04d}.json",
                                dynamics, cv, identity={"method": "opes", "replica": replica},
                                neq=options["neq"], nprod=options["nprod"], checkpoint=checkpoint,
                                bias_state=lambda: _opes_bias_summary(update.bias)))
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
    for index, stream in enumerate(streams):
        if options["method"] == "reweighted_histogram":
            samples, logs = frozen_samples(
                stream, config["beta"], equilibration_steps=options["neq"],
                freeze_after_steps=options["freeze"],
                freeze_equilibration_steps=options["freeze_equilibration_steps"],
                final_bias=biases[index],
            )
        else:
            samples = stream["samples"].copy()
            logs = config["beta"] * stream["sample_bias"]
        if not len(samples):
            raise ValueError("OPES analysis has no samples in the selected production stage")
        arrays.append(samples)
        log_weights.append(logs)
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
            density_factor = bias.bias_factor if bias.mode == "explore" else 1.0
            raw.append(-density_factor * bias.log_probability(grid) / config["beta"])
            ess.append(bias.effective_sample_size)
        else:
            free, effective = weighted_histogram(arrays[index], log_weights[index], edges, config["beta"])
            raw.append(free)
            ess.append(effective)
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
        "Uncompressed OPES: cost grows with kernel count; no PLUMED numerical parity is claimed.",
        "No reaction rate follows from biased trajectory time. CV adequacy, equilibrium, bead and timestep convergence remain unverified.",
        "Weight-only effective sample size ignores autocorrelation. Replica spread is descriptive, not a standard error."]
    warnings.append("The KDE free energy retains finite-bandwidth smoothing and adaptation error."
        if options["method"] == "kde" else
        "Only frozen-bias samples after the requested relaxation are reweighted; adequate equilibration is the user's responsibility.")
    if options["adaptive_sigma"]:
        warnings.append("Adaptive widths use online observed fluctuations and Scott/ESS scaling at deposition events, not PLUMED's diffusion-window rule.")
    state_versions = [bias.state_dict()["schema_version"] for bias in biases]
    initial_biases = [bias.state_dict().get("initial_bias", 0.0) for bias in biases]
    if 1 in state_versions:
        warnings.append("Legacy OPES schema-1 histories retain their zero empty-history bias and original first-kernel weight; they have not been converted to schema-2 initialization.")
    if options["bias_mode"] == "explore" and options["method"] == "kde":
        warnings.append("Explore KDE reports -gamma/beta*log(sampled density); it is a free-energy estimate only after convergence to the well-tempered target. Frozen-bias reweighting is preferable for quantitative analysis.")
    if not common.any():
        warnings.append("Independent histories have no common sampled bin; their aggregate FES is masked.")
    summary = {"task": "opes", "beta": config["beta"], "status": "finite_time_estimate" if all(stream.get("complete", True) for stream in streams) else "partial_finite_time_estimate", "method": "opes", "algorithm": biases[0].method,
        "estimator": options["method"], "measure": options["measure"], "energy_unit": "hartree",
        "dimension": cv.dimension, "units": cv.units, "replicas": len(streams),
        "samples": sum(len(values) for values in arrays), "kernels_per_replica": [len(bias) for bias in biases],
        "log_z_per_replica": [bias.log_z for bias in biases], "weight_effective_sample_size": ess,
        "bias_state_versions": state_versions, "initial_empty_bias_per_replica": initial_biases,
        "freeze_after_steps": options["freeze"], "freeze_equilibration_steps": options["freeze_equilibration_steps"],
        "common_support": bool(common.any()), "reference_bin": None if reference is None else list(reference),
        "converged": False, "propagation": [stream["propagation"] for stream in streams], "warnings": warnings,
        "sample_bias_convention": "physical V before the probability update; H_P contains P*V; log weights are beta*V",
        "files": {"data": str(output / "fes.npz"), "table": str(output / "fes.csv"),
                  "checkpoints": [stream["checkpoint"] for stream in streams]}}
    if options["method"] == "reweighted_histogram":
        summary.update(reweighted=True, retained_samples_per_replica=[len(values) for values in arrays])
    if options["measure"] == "rate_flux":
        summary["sample_bias_convention"] = (
            "physical V before the probability update; H_P contains P*V - P*log(flux_metric)/beta; "
            "frozen log weights are beta*V and preserve the rate_flux measure")
        warnings.append("RPMDrate xi can have a pole outside the reaction corridor; fixed-bias relaxation and stationary coverage require system-specific checks.")
    result = FESResult.from_data(data, method="opes", cv_units=cv.units, metadata=summary)
    summary = write_fes_result(output, result, summary)
    return {"summary": summary, "data": data, "streams": streams} if return_data else summary


class _DistributedFESSampler:
    """Preserve the serial history-dependent Andersen refresh clock."""

    def __init__(self, delegate, interval):
        self._delegate = delegate
        self._interval = interval

    def __getattr__(self, name):
        return getattr(self._delegate, name)

    def step(self, dynamics, state, bias):
        if (self._delegate.thermostat == "andersen"
                and state.step % self._interval == 0):
            state.p[:] = dynamics.sample_momenta()
        return self._delegate.step(dynamics, state, bias)


def _distributed_fes_sampler(config):
    return _DistributedFESSampler(
        fes_sampler(config),
        config.get("sampling", {}).get("andersen_interval", 10),
    )


def run_opes(config, system, output, executor=None, return_data=False):
    """Run independent histories or explicitly distributed shared walkers."""
    selected = config.get("opes", {}).get("walkers_mode", "serial")
    if selected != "serial":
        if executor is not None:
            raise ValueError(
                "distributed OPES owns walker scheduling and does not accept an executor")
        from . import validate_bond_constraint_workflow
        validate_bond_constraint_workflow(system, "opes")
        validate_opes(config, system)
        if selected == "mpi-sync":
            from ._distributed_opes_mpi import _run_mpi
            return _run_mpi(config, system, output, return_data=return_data)
        from ._distributed_opes_async import _run_async
        return _run_async(config, system, output, return_data=return_data)
    from . import validate_bond_constraint_workflow
    validate_bond_constraint_workflow(system, "opes")
    from .sampling import validate_andersen_schedule
    validate_andersen_schedule(config, "opes")
    validate_opes(config, system)
    schema = root_call(_resume_initialization_schema, config, system, output)
    streams = dispatch_replicas(config, system, output, _sample_replica, executor,
                                task_fields={"initialization_schema": schema})
    return root_call(_analyze, streams, config, system, str(output), return_data)


def reanalyze_opes(config, system, output, return_data=False):
    """Recompute saved OPES estimates without constructing a potential."""
    selected = config.get("opes", {}).get("walkers_mode", "serial")
    if selected != "serial":
        cv, _, options = resolve_opes(config, system)
        if selected == "mpi-sync":
            from ._distributed_opes_mpi import reanalyze_mpi
            return reanalyze_mpi(config, system, output, cv, options, return_data=return_data)
        from ._distributed_opes_async_io import _collect_async
        return _collect_async(config, system, output, cv, options, return_data=return_data)
    def analyze():
        cv, _, options = resolve_opes(config, system)
        streams = [_read_stream(config, system, output, replica, cv, options)
                   for replica in range(config.get("sampling", {}).get("replicas", 1))]
        return _analyze(streams, config, system, output, return_data)
    return root_call(analyze)
