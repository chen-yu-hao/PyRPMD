"""Independent, exactly resumable projected ABF/eABF centroid histories."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from ..abf import ABFBias, ExtendedABF, MeanForceEstimator, validate_centroid_cv
from ..analysis.fes_result import FESResult, write_fes_result
from ..checkpoint import atomic_savez, load_checkpoint
from ..dynamics import Dynamics, State
from ..parallel import make_task_potential, root_call
from ..potentials.base import evaluate_checked
from ..propagation import loop_options
from ..units import DIMS, parse_quantity
from ..validation import checked_integer, finite_positive
from .biased_sampling import dispatch_replicas
from .common import _dimension, _fingerprint, fes_method, resolve_cvs
from .sampling import checkpoint_cv_config, fes_sampler


_COMMON = {"cvs", "ranges", "ranges_au", "bins", "full_samples", "pace", "wall_stiffness", "wall_stiffness_au"}
ABF_FIELDS = _COMMON | {"derivative", "fd_step"}
EABF_FIELDS = _COMMON | {"coupling", "coupling_au", "extended_mass", "extended_mass_au", "friction", "friction_au"}


def _pick(table, name, default=None, *, required=False):
    if name in table and name + "_au" in table:
        raise ValueError(f"specify {name} or {name}_au, not both")
    if required and name not in table and name + "_au" not in table:
        raise ValueError(f"{name} is required")
    return table.get(name + "_au", table.get(name, default)), name + "_au" in table


def _vector_parameter(table, name, cv, scales, dimensions, *, default=None, required=False, power=0):
    raw, atomic = _pick(table, name, default, required=required)
    if isinstance(raw, (str, int, float, np.number)):
        raw = [raw] * cv.dimension
    if not isinstance(raw, (list, tuple, np.ndarray)) or len(raw) != cv.dimension:
        raise ValueError(f"{name} needs one value per CV")
    values = []
    for value, dimension, scale in zip(raw, dimensions, scales):
        converted = parse_quantity(value, dimension)
        if not atomic and isinstance(value, (int, float, np.number)):
            converted *= scale ** power
        values.append(finite_positive(converted, name))
    return np.asarray(values)


def resolve_abf(config, system):
    method = fes_method(config)
    if method not in ("abf", "eabf"):
        raise ValueError("ABF workflow requires fes.method='abf' or 'eabf'")
    if any(system.pbc) or system.fixed_atoms or system.bond_constraints:
        raise ValueError("ABF currently requires a nonperiodic, unconstrained system with all atoms mobile")
    if config.get("barostat") or config.get("path_integral") or config.get("replica_exchange"):
        raise ValueError("ABF does not combine barostats, alternate path integrals or replica exchange")
    if config.get("dynamics", {}).get("loop_backend") == "native":
        raise ValueError("ABF history updates require loop_backend='auto' or 'python'")
    if "trajectory_stride" in config.get("output", {}):
        raise ValueError("ABF exports CV traces via sampling.sample_interval; output.trajectory_stride is unsupported")
    for other in ("umbrella", "metadynamics", "opes", "plumed", "steered", "rate", "recrossing"):
        if config.get(other):
            raise ValueError("ABF does not combine " + other + " controls")
    if set(config.get("reaction", {})) - {"electronic_weight", "electronic", "electronic_weight_model"}:
        raise ValueError("ABF does not use reaction controls")
    if config.get("eabf" if method == "abf" else "abf"):
        raise ValueError("select exactly one ABF method")
    table = config.get(method, {})
    allowed = ABF_FIELDS if method == "abf" else EABF_FIELDS
    if not isinstance(table, Mapping) or set(table) - allowed:
        raise ValueError(f"invalid {method} fields")
    checkpoint_cv_config(config)
    cv, selected, scales = resolve_cvs(config, system, method)
    validate_centroid_cv(cv)
    ranges, atomic = _pick(table, "ranges", required=True)
    if not isinstance(ranges, (list, tuple, np.ndarray)) or len(ranges) != cv.dimension:
        raise ValueError("ABF ranges need one [lower, upper] pair per CV")
    bounds = []
    for pair, unit, scale in zip(ranges, cv.units, scales):
        if not isinstance(pair, (list, tuple, np.ndarray)) or len(pair) != 2:
            raise ValueError("ABF ranges need [lower, upper] pairs")
        values = [parse_quantity(x, _dimension(unit)) *
                  (scale if not atomic and isinstance(x, (int, float, np.number)) else 1) for x in pair]
        if values[0] >= values[1]:
            raise ValueError("ABF ranges must increase")
        bounds.append(values)
    bins = table.get("bins", 40)
    if np.isscalar(bins):
        bins = [bins] * cv.dimension
    if len(bins) != cv.dimension:
        raise ValueError("ABF bins need one count per CV")
    bins = [checked_integer(b, "ABF bins", 3 if method == "eabf" else 2) for b in bins]
    stiffness_dims = [tuple(e - 2 * c for e, c in zip(DIMS["energy"], _dimension(unit))) for unit in cv.units]
    wall = _vector_parameter(table, "wall_stiffness", cv, scales, stiffness_dims, default=1.0, power=-2)
    options = {"method": method, "edges": [np.linspace(lo, hi, n + 1) for (lo, hi), n in zip(bounds, bins)],
               "full_samples": checked_integer(table.get("full_samples", 200), "full_samples"),
               "pace": checked_integer(table.get("pace", 100), "ABF pace"), "wall_stiffness": wall}
    if method == "abf":
        options.update(derivative=table.get("derivative", "analytic"), fd_step=table.get("fd_step", 1e-5))
        MeanForceEstimator(cv, system.masses, config["beta"], derivative=options["derivative"], fd_step=options["fd_step"])
    else:
        options["coupling"] = _vector_parameter(table, "coupling", cv, scales, stiffness_dims, required=True, power=-2)
        # Auxiliary kinetic energy is p_lambda^2/(2*mu), so mu has
        # dimensions energy*time^2/CV^2 (ordinary mass for a length CV).
        mass_dims = [DIMS["mass"] if unit == "bohr" else
                     tuple(e + 2*t - 2*c for e, t, c in zip(DIMS["energy"], DIMS["time"], _dimension(unit)))
                     for unit in cv.units]
        options["mass"] = _vector_parameter(table, "extended_mass", cv, scales, mass_dims, required=True, power=-2)
        raw, _ = _pick(table, "friction", "0.01 fs^-1")
        options["friction"] = finite_positive(parse_quantity(raw, "frequency"), "friction")
    sampling = config.get("sampling", {})
    if "parents" in sampling or "engine" in sampling or (sampling.get("thermostat", "andersen") == "andersen"
                                  and sampling.get("thermostat_params")):
        raise ValueError("ABF does not use sampling.parents/engine or Andersen thermostat_params; set andersen_interval")
    options.update(neq=checked_integer(sampling.get("equilibration_steps", 0), "equilibration_steps", 0),
                   nprod=checked_integer(sampling.get("production_steps"), "production_steps"),
                   interval=checked_integer(sampling.get("sample_interval", sampling.get("interval_steps", 1)), "sample_interval"),
                   dt=finite_positive(sampling.get("dt_au", config["dt_au"]), "timestep"),
                   checkpoint_steps=checked_integer(config.get("output", {}).get("checkpoint_steps", 1000), "checkpoint_steps"))
    checked_integer(sampling.get("replicas", 1), "replicas")
    checked_integer(sampling.get("andersen_interval", 10), "andersen_interval")
    fes_sampler(config)
    analysis = config.get("analysis", {})
    expected = "mean_force" if method == "abf" else "czar"
    if analysis.get("method", expected) != expected or analysis.get("measure", "cv_density") != "cv_density":
        raise ValueError(f"{method} requires analysis.method='{expected}' and measure='cv_density'")
    for name in ("bootstrap", "bootstrap_replicates", "bootstrap_samples"):
        if checked_integer(analysis.get(name, 0), "analysis." + name, 0):
            raise ValueError("ABF bootstrap is not implemented; use independent replicas")
    if any(key in analysis for key in ("ranges", "bins", "flux_axis", "overlap_threshold", "max_grid_points", "seed")):
        raise ValueError("ABF analysis uses the sampling grid; put bins/ranges in the method table")
    options["min_count"] = checked_integer(analysis.get("min_count", 1), "analysis.min_count")
    # Validate all allocation/bias controls before constructing a PES.
    initial = np.repeat(system.geometry[None], config["beads"], axis=0)
    _new_bias(cv, options, config["beta"], initial).energy_force(initial)
    return cv, selected, options


def validate_abf(config, system):
    resolve_abf(config, system)


def _new_bias(cv, options, beta, q):
    shared = dict(full_samples=options["full_samples"], wall_stiffness=options["wall_stiffness"])
    if options["method"] == "abf":
        return ABFBias(cv, options["edges"], beta, **shared)
    return ExtendedABF(cv, options["edges"], beta, cv.value(q), coupling=options["coupling"],
                       mass=options["mass"], friction=options["friction"], **shared)


def _identity(config, system, replica):
    return _fingerprint(checkpoint_cv_config(config), system, {"abf_schema": 1, "replica": replica})


def _checkpoint(output, method, replica):
    return Path(output) / "checkpoints" / f"{method}_{replica:04d}.npz"


def _validate_operator_checkpoint(config, system, options, elapsed, operators):
    """Validate the active bath before even a completed stream is accepted.

    A lazily initialized composition can restore an empty bath, which is valid
    only before its first step.  Bind the actual sampler without a PES to check
    both its key and the required duration, then reuse the operator loaders for
    the Python-chain or native-bath identity, auxiliaries and checksums.
    """
    if (not isinstance(operators, dict) or set(operators) != {"version", "rng", "composition", "sampling"}
            or type(operators["version"]) is not int or operators["version"] != 1
            or not isinstance(operators["composition"], dict) or not isinstance(operators["sampling"], dict)):
        raise ValueError("invalid ABF propagation operator checkpoint")
    candidate = Dynamics(None, system.masses, config["beta"], config["beads"], options["dt"],
                         native=False, **loop_options(config))
    # Every ABF/eABF step carries a Python bias, so a full native executor is
    # never selected.  Andersen and legacy PILE draw from the main RNG and do
    # not create persistent bath helpers in either composition.
    if operators["composition"] != candidate.operator_state_dict()["composition"]:
        raise ValueError("ABF checkpoint base propagation operator state differs")
    sampler = fes_sampler(config)
    if elapsed == 0 or sampler.thermostat_spec is None:
        if operators["sampling"]:
            raise ValueError("ABF checkpoint contains an unexpected sampling operator")
    else:
        method = sampler.integrator if sampler.thermostat == "pile-l" else None
        expected = {"cayley-bcb": "bcocb", "normal-mode-bab": "baoab"}.get(candidate.integrator_name)
        composition_method = None if method == expected else method
        candidate._sample_composition(sampler.thermostat_spec, composition_method)
        keys = set(candidate.operator_state_dict()["sampling"])
        if not operators["sampling"]:
            raise ValueError("ABF checkpoint is missing thermostat auxiliary state")
        if set(operators["sampling"]) != keys:
            raise ValueError("ABF checkpoint sampling operator key differs from the current sampler")
        saved = operators["sampling"][next(iter(keys))]
        if not isinstance(saved, dict):
            raise ValueError("invalid ABF sampling operator checkpoint")
        half = sampler.thermostat == "nhc" or candidate.integrator_name == "velocity-verlet"
        duration = options["dt"] * (0.5 if half else 1.0)
        if not isinstance(saved.get("baths"), dict) or set(saved["baths"]) != {float(duration).hex()}:
            raise ValueError("ABF checkpoint is missing thermostat auxiliary state for the current duration")
        if saved.get("native") is not None or saved.get("native_seed") is not None:
            raise ValueError("ABF checkpoint contains an unexpected native executor")
    try:
        candidate.load_operator_state_dict(operators)
    except (KeyError, TypeError) as exc:
        raise ValueError("invalid ABF thermostat auxiliary state") from exc


def _read(config, system, output, replica, cv, options):
    path = _checkpoint(output, options["method"], replica)
    with load_checkpoint(path, expected_kind="abf_stream", require_integrity=True) as data:
        if str(data["fingerprint"]) != _identity(config, system, replica):
            raise ValueError("ABF checkpoint physics differs")
        raw = np.asarray(data["step"])
        if raw.shape != () or raw.dtype.kind not in "iu":
            raise ValueError("invalid ABF checkpoint clock")
        elapsed = int(raw)
        if not 0 <= elapsed <= options["neq"] + options["nprod"]:
            raise ValueError("ABF checkpoint exceeds the requested sampling clock")
        state = State(data["q"], data["p"], elapsed, float(data["time"]))
        if state.q.shape != (config["beads"], len(system.masses), 3) or not np.isclose(
                state.time, elapsed * options["dt"], rtol=1e-10, atol=1e-12):
            raise ValueError("ABF checkpoint state shape or time differs")
        bias_state = json.loads(str(data["bias_state"]))
        bias = _new_bias(cv, options, config["beta"], state.q)
        bias.load_state_dict(bias_state)
        adaptive = bias.bias if isinstance(bias, ExtendedABF) else bias
        production = max(0, elapsed - options["neq"])
        if adaptive.observations != production or adaptive.refreshes != production // options["pace"]:
            raise ValueError("ABF checkpoint adaptation clock differs")
        operators = json.loads(str(data["operators"]))
        _validate_operator_checkpoint(config, system, options, elapsed, operators)
        samples, clocks = data["samples"].copy(), data["sample_steps"].copy()
        count = production // options["interval"]
        if (samples.shape != (count, cv.dimension) or not np.isfinite(samples).all()
                or clocks.dtype.kind not in "iu" or not np.array_equal(
                    clocks, options["neq"] + np.arange(1, count + 1) * options["interval"])):
            raise ValueError("ABF checkpoint samples disagree with the clock")
    return {"state": state, "bias": bias, "operators": operators, "samples": samples,
            "sample_steps": clocks, "checkpoint": str(path),
            "complete": elapsed == options["neq"] + options["nprod"]}


def _grid_snapshot(grid, value):
    index = grid.index(value)
    return (grid, index, grid.outside, None if index is None else
            (grid.count[index], grid.mean[index].copy(), grid.m2[index].copy()))


def _sample(task):
    config, system, output, replica = (task[k] for k in ("config", "system", "output", "replica"))
    cv, _, options = resolve_abf(config, system)
    path = _checkpoint(output, options["method"], replica)
    saved = _read(config, system, output, replica, cv, options) if path.exists() else None
    if saved is not None and saved["complete"]:
        return saved
    provider = make_task_potential(config["potential"], system)
    try:
        dynamics = Dynamics(provider, system.masses, config["beta"], config["beads"], options["dt"],
                            seed=np.random.SeedSequence([int(config.get("seed", 0)), 1117, replica]),
                            native=config.get("native", config.get("dynamics", {}).get("native", True)),
                            threads=config.get("parallel", {}).get("threads_per_rank", 1), **loop_options(config))
        sampler = fes_sampler(config)
        if saved is None:
            state = State(np.repeat(system.geometry[None], config["beads"], axis=0), dynamics.sample_momenta())
            bias = _new_bias(cv, options, config["beta"], state.q)
            if isinstance(bias, ExtendedABF):
                bias.momentum = dynamics.rng.normal(size=cv.dimension) * np.sqrt(bias.mass / config["beta"])
            samples, clocks = [], []
        else:
            state, bias = saved["state"], saved["bias"]
            dynamics.load_operator_state_dict(saved["operators"])
            samples, clocks = list(saved["samples"]), list(saved["sample_steps"])
        estimator = None if isinstance(bias, ExtendedABF) else MeanForceEstimator(
            cv, system.masses, config["beta"], derivative=options["derivative"], fd_step=options["fd_step"])
        adaptive = bias.bias if isinstance(bias, ExtendedABF) else bias

        def save():
            atomic_savez(path, kind="abf_stream", fingerprint=_identity(config, system, replica),
                         q=state.q, p=state.p, step=state.step, time=state.time,
                         operators=json.dumps(dynamics.operator_state_dict(), allow_nan=False),
                         bias_state=json.dumps(bias.state_dict(), allow_nan=False),
                         samples=np.asarray(samples).reshape(-1, cv.dimension), sample_steps=np.asarray(clocks, dtype=np.int64))

        save()
        while state.step < options["neq"] + options["nprod"]:
            previous, operators = state.copy(), dynamics.operator_state_dict()
            old_aux = ((bias.position.copy(), bias.momentum.copy()) if isinstance(bias, ExtendedABF) else None)
            old_surface = adaptive.surface
            old_counts = (adaptive.observations, adaptive.refreshes)
            touched = []
            previous_samples = len(samples)
            try:
                if sampler.thermostat == "andersen" and state.step % config["sampling"].get("andersen_interval", 10) == 0:
                    state.p[:] = dynamics.sample_momenta()
                if isinstance(bias, ExtendedABF):
                    bias.advance(dynamics, state, sampler.step)
                else:
                    sampler.step(dynamics, state, bias)
                production = state.step - options["neq"]
                if production > 0:
                    if isinstance(bias, ExtendedABF):
                        value = np.asarray(cv.value(state.q))
                        touched = [_grid_snapshot(bias.bias.grid, bias.position), _grid_snapshot(bias.physical_grid, value)]
                        bias.observe(state.q)
                    else:
                        physical = evaluate_checked(provider, state.q)
                        value, gradient = estimator.evaluate(state.q, physical.forces)
                        touched = [_grid_snapshot(bias.grid, value)]
                        bias.observe(value, gradient)
                    if production % options["pace"] == 0:
                        adaptive.refresh()
                    if production % options["interval"] == 0:
                        samples.append(value.copy())
                        clocks.append(state.step)
            except BaseException:
                state = previous
                dynamics.load_operator_state_dict(operators)
                if old_aux is not None:
                    bias.position, bias.momentum = old_aux
                adaptive.surface = old_surface
                adaptive.observations, adaptive.refreshes = old_counts
                for grid, index, outside, stats in touched:
                    grid.outside = outside
                    if index is not None:
                        grid.count[index], grid.mean[index], grid.m2[index] = stats
                del samples[previous_samples:]
                del clocks[previous_samples:]
                save()
                raise
            if state.step % options["checkpoint_steps"] == 0:
                save()
        save()
    finally:
        if hasattr(provider, "close"):
            provider.close()
    return _read(config, system, output, replica, cv, options)


def _analyze(streams, config, system, output, return_data):
    streams = list(streams)
    cv, _, options = resolve_abf(config, system)
    surfaces = [s["bias"].fes_result(min_count=options["min_count"]) for s in streams]
    common = np.logical_and.reduce([~s.mask for s in surfaces])
    values = np.stack([s.free_energy for s in surfaces])
    free = np.full(common.shape, np.nan)
    error = free.copy()
    reference = None
    if common.any():
        reference = tuple(np.argwhere(common)[0])
        values -= values[(slice(None),) + reference].reshape((-1,) + (1,) * cv.dimension)
        free[common] = values[:, common].mean(axis=0)
        if len(values) > 1:
            error[common] = values[:, common].std(axis=0, ddof=1)
    summary = {"task": "fes", "method": options["method"], "beta": config["beta"], "measure": "cv_density",
               "status": "finite_time_estimate" if all(s["complete"] for s in streams) else "partial_finite_time_estimate",
               "replicas": len(streams), "samples": sum(s["bias"].bias.observations if isinstance(s["bias"], ExtendedABF)
                                                          else s["bias"].observations for s in streams),
               "converged": False, "physical_converged": False,
               "estimator": "czar" if options["method"] == "eabf" else "projected_mean_force",
               "algorithm": "extended_system_projected_abf" if options["method"] == "eabf" else "projected_abf",
               "reference_bin": None if reference is None else list(reference), "common_support": bool(common.any()),
               "projection_per_replica": [s.metadata["projection"] for s in surfaces],
               "warnings": ["Adaptive histories, finite timestep and finite grid resolution require independent convergence studies.",
                            "Replica spread is descriptive, not a calibrated standard error; disconnected support is masked."],
               "files": {"checkpoints": [s["checkpoint"] for s in streams]}}
    if options["method"] == "eabf":
        summary["warnings"].append("CZAR masks boundary bins and incomplete derivative stencils; auxiliary-coordinate PMF is a different, smoothed free energy.")
    extras = {"replica_free_energy": values, "raw_count": sum(s.extras["counts"] for s in surfaces),
              "counts_per_replica": np.stack([s.extras["counts"] for s in surfaces]),
              "mean_force_per_replica": np.stack([s.extras["mean_force"] for s in surfaces])}
    result = FESResult(tuple(options["edges"]), free, error, ~common, cv.units, options["method"],
                       "replica_spread", metadata=summary, extras=extras)
    summary = write_fes_result(output, result, summary)
    return {"summary": summary, "data": result.arrays(), "streams": streams} if return_data else summary


def run_abf(config, system, output, executor=None, return_data=False):
    validate_abf(config, system)
    streams = dispatch_replicas(config, system, output, _sample, executor)
    return root_call(_analyze, streams, config, system, output, return_data)


def reanalyze_abf(config, system, output, return_data=False):
    def analyze():
        cv, _, options = resolve_abf(config, system)
        streams = [_read(config, system, output, replica, cv, options)
                   for replica in range(config.get("sampling", {}).get("replicas", 1))]
        return _analyze(streams, config, system, output, return_data)
    return root_call(analyze)
