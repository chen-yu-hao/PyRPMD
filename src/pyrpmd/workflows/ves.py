"""Independent VES histories using the shared transactional FES sampler.

The optimizer observes every production step, including partial update blocks.
Both that state and the matching well-tempered target are checkpointed alongside
the MD state, random generator and thermostat auxiliaries.  Frozen production
can be reconstructed with the same equilibrium reweighting used by OPES/MetaD.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from ..analysis.fes_result import FESResult, estimate_fes_uncertainty, write_fes_result
from ..analysis.frozen_bias import frozen_bias_fes
from ..checkpoint import atomic_savez, load_checkpoint
from ..dynamics import Dynamics, State
from ..parallel import make_task_potential, root_call
from ..propagation import loop_options
from ..units import DIMS, parse_quantity
from ..validation import checked_integer, finite_positive
from ..ves import VESBias
from .biased_sampling import SamplingRuntime, dispatch_replicas, run_biased_sampling
from .common import _dimension, _fingerprint, fes_method, resolve_cvs
from .sampling import checkpoint_cv_config, fes_sampler, validate_andersen_schedule


VES_FIELDS = set(
    "cvs ranges ranges_au orders target bias_factor learning_rate learning_rate_au "
    "pace target_update_interval quadrature_order wall_stiffness wall_stiffness_au "
    "freeze_after_steps freeze_equilibration_steps".split()
)
_FALLBACK_REASON = "VES optimization uses the Python workflow"


def _alias(table, name, default=None, *, required=False):
    if name in table and name + "_au" in table:
        raise ValueError(f"specify ves.{name} or {name}_au, not both")
    if required and name not in table and name + "_au" not in table:
        raise ValueError(f"ves.{name} is required")
    return table.get(name + "_au", table.get(name, default)), name + "_au" in table


def _quantity(value, dimension, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite physical quantity, not a boolean")
    result = parse_quantity(value, dimension)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _ranges(raw, cv, scales, atomic, label):
    if not isinstance(raw, (list, tuple, np.ndarray)) or len(raw) != cv.dimension:
        raise ValueError(f"{label} needs one [lower, upper] pair per CV")
    result = []
    for pair, unit, scale in zip(raw, cv.units, scales):
        if not isinstance(pair, (list, tuple, np.ndarray)) or len(pair) != 2:
            raise ValueError(f"{label} entries must be [lower, upper]")
        values = [_quantity(value, _dimension(unit), label) *
                  (scale if not atomic and not isinstance(value, str) else 1.0)
                  for value in pair]
        if not values[1] > values[0]:
            raise ValueError(f"{label} must be increasing")
        result.append(values)
    return np.asarray(result)


def resolve_ves(config, system):
    """Validate the complete protocol before allocating output or a PES."""
    if fes_method(config) != "ves":
        raise ValueError("VES workflow requires fes.method='ves'")
    validate_andersen_schedule(config, "ves")
    if any(system.pbc) or system.fixed_atoms or system.bond_constraints:
        raise ValueError("VES requires a nonperiodic, unconstrained system with all atoms mobile")
    if any(config.get(key) for key in ("barostat", "path_integral", "replica_exchange", "adaptive_umbrella")):
        raise ValueError("VES does not combine barostats, alternate path integrals or replica exchange/adaptive umbrellas")
    if config.get("dynamics", {}).get("loop_backend") == "native":
        raise ValueError("VES updates require loop_backend='auto' or 'python'")
    if any(key in config.get("output", {}) for key in
           ("trajectory_stride", "trajectory_storage", "chunk_frames", "store_configurations", "diagnostics")):
        raise ValueError("VES exports CV traces via sampling.sample_interval; trajectory output controls are unsupported")
    for name in ("umbrella", "metadynamics", "opes", "abf", "eabf", "steered", "plumed",
                 "rate", "recrossing", "collision", "outcomes"):
        if config.get(name):
            raise ValueError("VES does not combine " + name + " controls")
    if set(config.get("reaction", {})) - {"electronic_weight"}:
        raise ValueError("VES does not use reaction controls")
    table = config.get("ves", {})
    if not isinstance(table, Mapping):
        raise ValueError("ves must be a table")
    unknown = set(table) - VES_FIELDS
    if unknown:
        raise ValueError("Unknown ves fields: " + ", ".join(sorted(unknown)))
    checkpoint_cv_config(config)
    cv, selected, scales = resolve_cvs(config, system, "ves")
    raw, atomic = _alias(table, "ranges", required=True)
    bounds = _ranges(raw, cv, scales, atomic, "ves.ranges")
    raw, atomic = _alias(table, "wall_stiffness", 1.0)
    if np.isscalar(raw):
        raw = [raw] * cv.dimension
    if not isinstance(raw, (list, tuple, np.ndarray)) or len(raw) != cv.dimension:
        raise ValueError("ves.wall_stiffness needs one value per CV")
    wall = []
    for value, unit, scale in zip(raw, cv.units, scales):
        dimension = tuple(e - 2*c for e, c in zip(DIMS["energy"], _dimension(unit)))
        converted = _quantity(value, dimension, "ves.wall_stiffness")
        wall.append(converted / (scale**2 if not atomic and not isinstance(value, str) else 1.0))
    learning_rate, _ = _alias(table, "learning_rate")
    if learning_rate is not None:
        learning_rate = _quantity(learning_rate, "energy", "ves.learning_rate")
    kwargs = {
        "orders": table.get("orders", 4), "target": table.get("target", "uniform"),
        "bias_factor": table.get("bias_factor", 10.0), "learning_rate": learning_rate,
        "pace": table.get("pace", 100), "target_update_interval": table.get("target_update_interval", 100),
        "quadrature_order": table.get("quadrature_order", 32), "wall_stiffness": wall,
    }
    # The core owns basis allocation, centroid, target and optimizer checks.
    bias = VESBias(cv, config["beta"], bounds, **kwargs)
    initial = np.repeat(system.geometry[None], config["beads"], axis=0)
    bias.energy_force(initial)
    # A tensor target requires a regular D-dimensional CV density. Normalize
    # Jacobian rows so the rank decision is independent of each CV's unit.
    jacobian = np.asarray(cv.value_and_jacobian(initial).jacobian, float)
    gradient = jacobian.sum(axis=1).reshape(cv.dimension, -1)
    scale = np.max(np.abs(gradient), axis=1)
    if np.any(scale == 0) or not np.isfinite(scale).all():
        raise ValueError("VES centroid CV Jacobian must have full rank for a regular density")
    normalized = gradient / scale[:, None]
    eigenvalues = np.linalg.eigvalsh(normalized @ normalized.T)
    if eigenvalues[0] <= 1e-12 * eigenvalues[-1]:
        raise ValueError("VES centroid CV Jacobian must have full rank for a regular density")
    sampling = config.get("sampling", {})
    unsupported = {"parents", "engine", "device", "normal_mode_layout"}.intersection(sampling)
    if unsupported:
        raise ValueError("VES does not use sampling." + ", sampling.".join(sorted(unsupported)))
    if sampling.get("thermostat", "andersen") == "andersen" and sampling.get("thermostat_params"):
        raise ValueError("VES Andersen sampling uses andersen_interval, not thermostat_params")
    if sampling.get("thermostat", "andersen") != "andersen" and sampling.get("thermostat") != "pile-l" and "gamma_centroid" in sampling:
        raise ValueError("sampling.gamma_centroid requires PILE-L sampling")
    fes_sampler(config)
    options = {
        "ranges": bounds, "bias_kwargs": kwargs,
        "neq": checked_integer(sampling.get("equilibration_steps", 0), "sampling.equilibration_steps", 0),
        "nprod": checked_integer(sampling.get("production_steps"), "sampling.production_steps"),
        "interval": checked_integer(sampling.get("sample_interval", sampling.get("interval_steps", 1)), "sampling.sample_interval"),
        "dt": finite_positive(sampling.get("dt_au", config["dt_au"]), "VES timestep"),
        "checkpoint_steps": checked_integer(config.get("output", {}).get("checkpoint_steps", 1000), "output.checkpoint_steps"),
        "freeze": checked_integer(table["freeze_after_steps"], "ves.freeze_after_steps") if "freeze_after_steps" in table else None,
        "relax": checked_integer(table.get("freeze_equilibration_steps", 0), "ves.freeze_equilibration_steps", 0),
    }
    checked_integer(sampling.get("replicas", 1), "sampling.replicas")
    checked_integer(sampling.get("andersen_interval", 10), "sampling.andersen_interval")
    if min(options["nprod"] // options["interval"], options["nprod"] // kwargs["pace"]) < 1:
        raise ValueError("VES production must contain at least one sample and one optimizer block")
    freeze = options["freeze"]
    if freeze is not None and (freeze % kwargs["pace"] or not kwargs["pace"] <= freeze <= options["nprod"]):
        raise ValueError("ves.freeze_after_steps must be a multiple of pace within production_steps")
    if options["relax"] and freeze is None:
        raise ValueError("ves.freeze_equilibration_steps requires freeze_after_steps")
    analysis = config.get("analysis", {})
    method = analysis.get("method", "bias_estimate")
    if method not in ("bias_estimate", "reweighted_histogram"):
        raise ValueError("VES analysis.method must be bias_estimate or reweighted_histogram")
    if analysis.get("measure", "cv_density") != "cv_density":
        raise ValueError("VES requires analysis.measure='cv_density'")
    for name in ("bootstrap", "bootstrap_replicates", "bootstrap_samples"):
        if checked_integer(analysis.get(name, 0), "analysis." + name, 0):
            raise ValueError("VES bootstrap is not implemented; use independent replicas")
    if any(name in analysis for name in ("flux_axis", "seed", "overlap_threshold", "max_grid_points")):
        raise ValueError("VES does not use analysis.flux_axis/seed/overlap_threshold/max_grid_points")
    if method == "reweighted_histogram" and (
            freeze is None or (options["nprod"] // options["interval"]) * options["interval"] <= freeze + options["relax"]):
        raise ValueError("reweighted_histogram requires samples after ves.freeze_after_steps + freeze_equilibration_steps")
    bins = analysis.get("bins", 40)
    if np.isscalar(bins):
        bins = [bins] * cv.dimension
    if not isinstance(bins, (list, tuple, np.ndarray)) or len(bins) != cv.dimension:
        raise ValueError("VES analysis.bins needs one count per CV")
    bins = [checked_integer(value, "analysis.bins", 2) for value in bins]
    if np.prod(bins, dtype=float) > 1_000_000:
        raise ValueError("VES analysis grid exceeds one million points")
    ranges = bounds.copy()
    if "ranges" in analysis:
        # Shared analysis.ranges uses canonical CV units, as OPES and MetaD.
        ranges = _ranges(analysis["ranges"], cv, scales, True, "analysis.ranges")
        if np.any(ranges[:, 0] < bounds[:, 0]) or np.any(ranges[:, 1] > bounds[:, 1]):
            raise ValueError("VES analysis.ranges must lie within ves.ranges")
    options.update(method=method, bins=bins, analysis_ranges=ranges,
                   min_count=checked_integer(analysis.get("min_count", 1), "analysis.min_count"))
    return cv, selected, options


def validate_ves(config, system):
    resolve_ves(config, system)


def _new_bias(cv, options, beta, state=None):
    return VESBias(cv, beta, options["ranges"], state=state, **options["bias_kwargs"])


class _VESUpdate:
    def __init__(self, bias, cv, options, beta):
        self.bias, self.cv, self.options, self.beta = bias, cv, options, beta

    def due(self, production):
        # Even observations between coefficient updates mutate partial moments.
        return self.options["freeze"] is None or production <= self.options["freeze"]

    def snapshot(self):
        return self.bias.state_dict()

    def apply(self, value, state):
        self.bias.observe(value)

    def restore(self, snapshot):
        self.bias = _new_bias(self.cv, self.options, self.beta, snapshot)


def _identity(config, system, replica):
    return _fingerprint(checkpoint_cv_config(config), system, {"ves_schema": 1, "replica": replica})


def _checkpoint(output, replica):
    return Path(output) / "checkpoints" / f"ves_{replica:04d}.npz"


def _clock(data, name):
    raw = np.asarray(data[name])
    if raw.shape != () or raw.dtype.kind not in "iu" or raw < 0 or raw > np.iinfo(np.int64).max:
        raise ValueError("VES checkpoint " + name + " must be a nonnegative integer scalar clock")
    return int(raw)


def _validate_checkpoint_candidate(current, bias, propagation, options, dimension):
    """Reject a partially committed shared-loop step before replacing storage.

    The shared history loop commits bookkeeping after its numerical rollback
    block. A signal can interrupt that commit between clocks, diagnostics and
    trace appends. Such an in-memory state must never replace a durable prefix.
    """
    elapsed = checked_integer(current.elapsed, "VES checkpoint elapsed", 0)
    if (current.state.step != elapsed or elapsed > options["neq"] + options["nprod"]
            or not np.isclose(current.state.time, elapsed * options["dt"], rtol=1e-10, atol=1e-12)):
        raise ValueError("state clock and elapsed steps disagree")
    production = max(0, elapsed - options["neq"])
    count = production // options["interval"]
    if any(len(trace) != count for trace in
           (current.values, current.clocks, current.biases, current.energies)):
        raise ValueError("sample trace lengths disagree with elapsed steps")
    expected = options["neq"] + np.arange(1, count + 1) * options["interval"]
    if not np.array_equal(current.clocks, expected):
        raise ValueError("sample clocks disagree with elapsed steps")
    values = np.asarray(current.values)
    if count and (values.shape != (count, dimension) or not np.isfinite(values).all()):
        raise ValueError("sample values are incomplete or nonfinite")
    for trace in (current.biases, current.energies):
        value = np.asarray(trace)
        if value.shape != (count,) or not np.isfinite(value).all():
            raise ValueError("sample observables are incomplete or nonfinite")
    adaptation = min(production, options["freeze"] or options["nprod"])
    pace = options["bias_kwargs"]["pace"]
    if (bias.observations != adaptation or bias.updates != adaptation // pace
            or bias.pending_count != adaptation % pace):
        raise ValueError("optimizer clock disagrees with elapsed steps")
    steps = sum(checked_integer(propagation[name], "VES " + name, 0)
                for name in ("python_steps", "native_steps"))
    reasons = propagation["fallback_reasons"]
    if (steps != elapsed or not isinstance(reasons, Mapping)
            or checked_integer(reasons.get(_FALLBACK_REASON, 0), "VES workflow steps", 0) != elapsed):
        raise ValueError("propagation bookkeeping disagrees with elapsed steps")


def _validate_operators(config, system, options, elapsed, saved, rng):
    """Admit complete baths and the actual selected sampler before PES setup."""
    if (not isinstance(saved, dict) or set(saved) != {"version", "rng", "composition", "sampling"}
            or type(saved["version"]) is not int or saved["version"] != 1
            or not isinstance(saved["composition"], dict) or not isinstance(saved["sampling"], dict)):
        raise ValueError("invalid VES propagation operator checkpoint")
    if saved["rng"] != rng:
        raise ValueError("VES checkpoint RNG and propagation operator state disagree")
    candidate = Dynamics(None, system.masses, config["beta"], config["beads"],
                         options["dt"], native=False, **loop_options(config))
    if saved.get("composition") != candidate.operator_state_dict()["composition"]:
        raise ValueError("VES checkpoint base propagation operator state differs")
    sampler = fes_sampler(config)
    if elapsed and sampler.thermostat_spec is not None:
        method = sampler.integrator if sampler.thermostat == "pile-l" else None
        expected = {"cayley-bcb": "bcocb", "normal-mode-bab": "baoab"}.get(candidate.integrator_name)
        candidate._sample_composition(sampler.thermostat_spec, None if method == expected else method)
        keys = set(candidate.operator_state_dict()["sampling"])
        if not saved.get("sampling"):
            raise ValueError("VES checkpoint is missing thermostat auxiliary state")
        if set(saved["sampling"]) != keys:
            raise ValueError("VES checkpoint thermostat identity differs")
        state = saved["sampling"][next(iter(keys))]
        duration = options["dt"] * (0.5 if sampler.thermostat == "nhc" or candidate.integrator_name == "velocity-verlet" else 1.0)
        if (not isinstance(state, dict) or not isinstance(state.get("baths"), dict)
                or set(state["baths"]) != {float(duration).hex()}):
            raise ValueError("VES checkpoint is missing thermostat auxiliary state for the current duration")
        if state.get("native") is not None or state.get("native_seed") is not None:
            raise ValueError("VES checkpoint contains an unexpected native executor")
    elif saved.get("sampling"):
        raise ValueError("VES checkpoint contains an unexpected sampling operator")
    try:
        candidate.load_operator_state_dict(saved)
    except (KeyError, TypeError) as error:
        raise ValueError("invalid VES propagation operator checkpoint") from error


def _read_stream(config, system, output, replica, cv, options):
    checkpoint = _checkpoint(output, replica)
    with load_checkpoint(checkpoint, expected_kind="ves_stream", require_integrity=True) as data:
        if str(data["fingerprint"]) != _identity(config, system, replica):
            raise ValueError("VES checkpoint physics differs")
        elapsed = _clock(data, "elapsed")
        if elapsed != _clock(data, "step") or elapsed > options["neq"] + options["nprod"]:
            raise ValueError("VES checkpoint has an invalid sampling clock")
        time = np.asarray(data["time"])
        if time.shape != () or time.dtype.kind not in "fiu" or not np.isfinite(time) or time < 0:
            raise ValueError("VES checkpoint time must be a nonnegative finite scalar")
        state = State(data["q"], data["p"], elapsed, float(time))
        if state.q.shape != (config["beads"], len(system.masses), 3) or not np.isclose(
                state.time, elapsed * options["dt"], rtol=1e-10, atol=1e-12):
            raise ValueError("VES checkpoint state shape or time differs")
        saved_bias = json.loads(str(data["bias_state"]))
        bias = _new_bias(cv, options, config["beta"], saved_bias)
        production = max(0, elapsed - options["neq"])
        adaptation = min(production, options["freeze"] or options["nprod"])
        pace = options["bias_kwargs"]["pace"]
        if (bias.observations != adaptation or bias.updates != adaptation // pace
                or bias.pending_count != adaptation % pace):
            raise ValueError("VES checkpoint optimizer clock disagrees with sampling")
        samples, clocks = data["samples"].copy(), data["sample_steps"].copy()
        count = production // options["interval"]
        if (samples.shape != (count, cv.dimension) or not np.isfinite(samples).all()
                or clocks.dtype.kind not in "iu" or not np.array_equal(
                    clocks, options["neq"] + np.arange(1, count + 1) * options["interval"])):
            raise ValueError("VES checkpoint samples disagree with the sampling clock")
        observables = {key: data[key].copy() for key in ("sample_bias", "sample_energies")}
        if any(value.shape != (count,) or not np.isfinite(value).all() for value in observables.values()):
            raise ValueError("VES checkpoint has invalid observables")
        rng = json.loads(str(data["rng"]))
        np.random.default_rng().bit_generator.state = rng
        operators = json.loads(str(data["operator_state"]))
        _validate_operators(config, system, options, elapsed, operators, rng)
        propagation = json.loads(str(data["propagation"]))
        if not isinstance(propagation, dict):
            raise ValueError("VES checkpoint propagation diagnostics must be a table")
    return {"replica": replica, "state": state, "elapsed": elapsed, "rng": rng,
            "operator_state": operators, "bias_state": saved_bias,
            "samples": samples, "sample_steps": clocks, **observables,
            "propagation": propagation, "checkpoint": str(checkpoint),
            "complete": elapsed == options["neq"] + options["nprod"]}


def _sample_replica(task):
    config, system, output, replica = task["config"], task["system"], task["output"], task["replica"]
    cv, _, options = resolve_ves(config, system)
    checkpoint = _checkpoint(output, replica)
    saved = _read_stream(config, system, output, replica, cv, options) if checkpoint.exists() else None
    if saved is not None and saved["complete"]:
        return saved
    identity = _identity(config, system, replica)
    provider = make_task_potential(config["potential"], system)
    try:
        dynamics = Dynamics(provider, system.masses, config["beta"], config["beads"], options["dt"],
            seed=np.random.SeedSequence([int(config.get("seed", 0)), 2414, replica]),
            native=config.get("native", config.get("dynamics", {}).get("native", True)),
            threads=config.get("parallel", {}).get("threads_per_rank", 1), **loop_options(config))
        bias = _new_bias(cv, options, config["beta"], None if saved is None else saved["bias_state"])
        if saved is None:
            state = State(np.repeat(system.geometry[None], config["beads"], axis=0), dynamics.sample_momenta())
            runtime = SamplingRuntime(state, operator_checkpoint=True)
        else:
            dynamics.load_operator_state_dict(saved["operator_state"])
            dynamics.propagation_stats = saved["propagation"]
            runtime = SamplingRuntime(saved["state"], saved["elapsed"], list(saved["samples"]),
                list(saved["sample_steps"]), list(saved["sample_bias"]), list(saved["sample_energies"]),
                operator_checkpoint=True)
        update = _VESUpdate(bias, cv, options, config["beta"])

        def save(current):
            # Capture the caller's exception before validation creates one of
            # its own. Preserve the existing checkpoint when an interrupt
            # exposes an incomplete bookkeeping commit, and let the shared
            # runner re-raise the original exception unchanged.
            unwinding = sys.exception() is not None
            try:
                _validate_checkpoint_candidate(current, update.bias, dynamics.propagation_stats,
                                               options, cv.dimension)
                payload = dict(fingerprint=identity, elapsed=current.elapsed,
                    q=current.state.q, p=current.state.p, step=current.state.step, time=current.state.time,
                    rng=json.dumps(dynamics.rng.bit_generator.state, allow_nan=False),
                    operator_state=json.dumps(dynamics.operator_state_dict(), allow_nan=False),
                    bias_state=json.dumps(update.bias.state_dict(), allow_nan=False),
                    samples=np.asarray(current.values).reshape(-1, cv.dimension),
                    sample_steps=np.asarray(current.clocks, dtype=np.int64),
                    sample_bias=np.asarray(current.biases), sample_energies=np.asarray(current.energies),
                    propagation=json.dumps(dynamics.propagation_stats, allow_nan=False))
            except (ValueError, TypeError, KeyError) as error:
                if unwinding:
                    return
                raise ValueError("inconsistent VES checkpoint candidate: " + str(error)) from error
            atomic_savez(checkpoint, kind="ves_stream", **payload)

        run_biased_sampling(runtime, dynamics, cv, neq=options["neq"], nprod=options["nprod"],
            interval=options["interval"], checkpoint_steps=options["checkpoint_steps"],
            sampling=config["sampling"], sampler=fes_sampler(config), update=update, save=save,
            fallback_reason=_FALLBACK_REASON)
        return _read_stream(config, system, output, replica, cv, options)
    finally:
        if hasattr(provider, "close"):
            provider.close()


def _bias_estimate(streams, biases, cv, options):
    edges = [np.linspace(lo, hi, n + 1) for (lo, hi), n in zip(options["analysis_ranges"], options["bins"])]
    grid = np.stack(np.meshgrid(*[(edge[1:] + edge[:-1]) * 0.5 for edge in edges], indexing="ij"), axis=-1)
    counts = np.asarray([np.histogramdd(stream["samples"], bins=edges)[0].astype(np.int64) for stream in streams])
    raw = np.empty((len(biases),) + grid.shape[:-1])
    points = grid.reshape(-1, cv.dimension)
    for bias, surface in zip(biases, raw):
        # A legal million-point grid and 255-term expansion must not allocate
        # grid-by-basis intermediates of several gigabytes during analysis.
        chunk = max(1, 65536 // bias.basis.size)
        flattened = surface.reshape(-1)
        for start in range(0, len(points), chunk):
            flattened[start:start + chunk] = bias.free_energy(points[start:start + chunk])
    mask = (counts < options["min_count"]) | ~np.isfinite(raw)
    common = ~np.any(mask, axis=0)
    free, spread = np.full(grid.shape[:-1], np.nan), np.full(grid.shape[:-1], np.nan)
    reference = None
    shifted = raw.copy()
    if common.any():
        reference = np.unravel_index(np.argmin(np.where(common, raw.mean(axis=0), np.inf)), common.shape)
        shifted -= raw[(slice(None),) + reference].reshape((-1,) + (1,) * cv.dimension)
        free[common] = shifted[:, common].mean(axis=0)
        spread = estimate_fes_uncertainty(shifted, kind="replica_spread", mask=~common)
    else:
        for index in range(len(streams)):
            if (~mask[index]).any():
                shifted[index] -= np.min(raw[index][~mask[index]])
    shifted[mask] = np.nan
    return {"edges": edges, "free_energy": free, "free_energy_per_replica": shifted,
            "replica_spread": spread, "standard_error": np.full_like(free, np.nan),
            "mask": ~common, "mask_per_replica": mask,
            "raw_count": counts.sum(axis=0), "raw_count_per_replica": counts,
            "common_support": bool(common.any()),
            "reference_bin": None if reference is None else list(reference)}


def _analyze(streams, config, system, output, return_data):
    streams = list(streams)
    if not streams:
        raise ValueError("VES analysis requires at least one independent history")
    cv, _, options = resolve_ves(config, system)
    biases = [_new_bias(cv, options, config["beta"], stream["bias_state"]) for stream in streams]
    if options["method"] == "reweighted_histogram":
        data = frozen_bias_fes(streams, config["beta"], bins=options["bins"],
            ranges=options["analysis_ranges"], periods=cv.periods, widths=np.diff(options["ranges"], axis=1)[:, 0],
            equilibration_steps=options["neq"], freeze_after_steps=options["freeze"],
            freeze_equilibration_steps=options["relax"], min_count=options["min_count"], final_biases=biases)
    else:
        data = _bias_estimate(streams, biases, cv, options)
    data["coefficients_per_replica"] = np.asarray([bias.coefficients for bias in biases])
    data["target_coefficients_per_replica"] = np.asarray([bias.target_coefficients for bias in biases])
    data["reconstruction_target_coefficients_per_replica"] = np.asarray([bias.reconstruction_target for bias in biases])
    warnings = list(config.get("_warnings", [])) + [
        "Finite-time VES estimate: basis truncation, optimizer/target convergence, CV adequacy, equilibration, bead and timestep convergence remain unverified.",
        "Replica spread describes independent estimates; it is not a calibrated standard error. No reaction rate follows from biased trajectory time.",
        "Optimizer moments are conditional on the declared CV domain; outside observations are excluded and counted, while soft walls remain active.",
    ]
    warnings.append(
        "Only frozen-bias samples after the requested relaxation are reweighted. Weight-only effective sample size ignores autocorrelation."
        if options["method"] == "reweighted_histogram" else
        "Bias reconstruction uses -V-log(p_target)/beta with its matching target. It is a variational estimate, not an unbiased histogram or a proof of convergence."
    )
    if not data["common_support"]:
        warnings.append("Independent histories have no common sampled bin; their aggregate FES is masked.")
    summary = {
        "task": "fes", "method": "ves", "algorithm": "linear_legendre_variational_enhanced_sampling",
        "estimator": options["method"], "measure": "cv_density", "beta": config["beta"],
        "dimension": cv.dimension, "units": list(cv.units), "energy_unit": "hartree",
        "status": "finite_time_estimate" if all(stream["complete"] for stream in streams) else "partial_finite_time_estimate",
        "converged": False, "physical_converged": False,
        "target": options["bias_kwargs"]["target"], "bias_factor": options["bias_kwargs"]["bias_factor"],
        "basis": "tensor_product_legendre_without_constant", "orders": list(biases[0].basis.orders),
        "ranges_au": options["ranges"].tolist(), "replicas": len(streams),
        "samples": sum(data.get("retained_samples_per_replica", [len(stream["samples"]) for stream in streams])),
        "updates_per_replica": [bias.updates for bias in biases],
        "target_updates_per_replica": [bias.target_updates for bias in biases],
        "observations_per_replica": [bias.observations for bias in biases],
        "outside_observations_per_replica": [bias.outside for bias in biases],
        "pending_observations_per_replica": [bias.pending_count for bias in biases],
        "freeze_after_steps": options["freeze"], "freeze_equilibration_steps": options["relax"],
        "common_support": data["common_support"], "reference_bin": data["reference_bin"],
        "sample_bias_convention": "physical V including walls before the optimizer update; H_P contains P*V; frozen log weights are beta*V",
        "propagation": [stream["propagation"] for stream in streams], "warnings": warnings,
        "files": {"checkpoints": [stream["checkpoint"] for stream in streams]},
    }
    if options["method"] == "reweighted_histogram":
        summary.update(reweighted=True, retained_samples_per_replica=data["retained_samples_per_replica"],
                       weight_effective_sample_size=data["weight_effective_sample_size"])
    result = FESResult.from_data(data, method="ves", cv_units=cv.units, metadata=summary)
    summary = write_fes_result(output, result, summary)
    return {"summary": summary, "data": data, "streams": streams} if return_data else summary


def run_ves(config, system, output, executor=None, return_data=False):
    """Run or resume independent VES replicas through the FES scheduler."""
    validate_ves(config, system)
    streams = dispatch_replicas(config, system, output, _sample_replica, executor)
    return root_call(_analyze, streams, config, system, output, return_data)


def reanalyze_ves(config, system, output, return_data=False):
    """Reconstruct saved VES histories without constructing a potential."""
    def analyze():
        cv, _, options = resolve_ves(config, system)
        streams = [_read_stream(config, system, output, replica, cv, options)
                   for replica in range(config.get("sampling", {}).get("replicas", 1))]
        return _analyze(streams, config, system, output, return_data)
    return root_call(analyze)
