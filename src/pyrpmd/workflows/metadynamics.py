"""Resumable standard and well-tempered metadynamics, with independent walkers.

Each replica owns its history; this is not a shared-bias multiple-walker
algorithm. Every deposition is after a completed integration step, the two
force evaluations within a step see the same history, and checkpoint state,
RNG, sample clock and hills are committed in one atomic NPZ. HILLS.csv is a
regenerable view, never the authority for resuming a trajectory.
"""

from __future__ import annotations
import copy
import csv
import json
from pathlib import Path
from collections.abc import Mapping
import numpy as np

from ..analysis.metadynamics import bias_fes
from ..bias import MetadynamicsBias
from ..cv import make_cvs, RPMDrateCoordinate
from ..dynamics import Dynamics, State
from ..checkpoint import load_checkpoint
from ..parallel import run_tasks, make_task_potential, root_call
from ..propagation import loop_options
from ..units import parse_quantity, unit_factor
from .fes import _atomic_npz, _fingerprint, _dimension, checked_integer, write_json


METADYNAMICS_FIELDS = {"cvs", "mode", "sigma", "sigma_au", "height", "height_au", "pace", "bias_factor"}


def resolve_metadynamics(config, system):
    """Validate algorithm options and convert widths/height to canonical units."""
    if any(system.pbc) or system.fixed_atoms:
        raise ValueError(
            "Metadynamics does not yet support periodic systems or fixed atoms; "
            "use the surface FES umbrella workflow"
        )
    table = config.get("metadynamics", {})
    if not isinstance(table, Mapping):
        raise ValueError("metadynamics must be a table")
    extra = set(table) - METADYNAMICS_FIELDS
    if extra:
        raise ValueError("Unknown metadynamics fields: " + ", ".join(sorted(extra)))
    specs = config.get("cvs", [])
    if isinstance(specs, dict):
        specs = [dict(v, name=k) for k, v in specs.items()]
    names = table.get("cvs")
    if isinstance(names, str):
        names = [names]
    cv = make_cvs(specs, system, names)
    selected = specs if names is None else [next(s for s in specs if s.get("name") == name) for name in names]
    if cv.dimension not in (1, 2):
        raise ValueError("metadynamics supports one or two CV dimensions")
    scales = []
    for spec, component in zip(selected, cv.cvs):
        units = spec.get("unit", spec.get("units", component.units)) if isinstance(spec, dict) else component.units
        if isinstance(units, str):
            units = [units] * component.dimension
        if len(units) != component.dimension:
            raise ValueError("CV input units dimension mismatch")
        scales.extend([1.0 if u == "dimensionless" else unit_factor(u, _dimension(c))
                       for u, c in zip(units, component.units)])
    if "sigma" in table and "sigma_au" in table:
        raise ValueError("specify metadynamics.sigma or sigma_au, not both")
    widths = table.get("sigma_au", table.get("sigma"))
    if widths is None:
        raise ValueError("metadynamics.sigma is required")
    if cv.dimension == 1 and isinstance(widths, (str, int, float, np.number)):
        widths = [widths]
    if not isinstance(widths, (list, tuple, np.ndarray)) or len(widths) != cv.dimension:
        raise ValueError("metadynamics.sigma needs one width per CV")
    sigma = [parse_quantity(v, _dimension(unit)) *
             (scale if "sigma_au" not in table and isinstance(v, (int, float, np.number)) else 1.0)
             for v, unit, scale in zip(widths, cv.units, scales)]
    if "height" in table and "height_au" in table:
        raise ValueError("specify metadynamics.height or height_au, not both")
    if "height" not in table and "height_au" not in table:
        raise ValueError("metadynamics.height is required")
    height = parse_quantity(table.get("height_au", table.get("height")), "energy")
    options = {
        "sigma": sigma,
        "height": height,
        "mode": table.get("mode", "well_tempered"),
        "bias_factor": table.get("bias_factor"),
        "pace": checked_integer(table.get("pace", 100), "metadynamics.pace"),
    }
    # Constructor validation is shared with the direct Python API.
    bias = MetadynamicsBias(cv, sigma, height, config["beta"], mode=options["mode"],
                            bias_factor=options["bias_factor"])
    options["bias_factor"] = bias.bias_factor
    if config.get("umbrella"):
        raise ValueError("the metadynamics workflow does not combine an umbrella bias")
    if config.get("dynamics", {}).get("loop_backend", "auto") == "native":
        raise ValueError("history-dependent metadynamics bias requires loop_backend='auto' or 'python'")
    analysis = config.get("analysis", {})
    if analysis.get("method", "bias_estimate") != "bias_estimate":
        raise ValueError("metadynamics analysis.method must be bias_estimate; evolving hills are not fixed-umbrella MBAR data")
    if analysis.get("measure", "cv_density") not in ("cv_density", "equilibrium"):
        raise ValueError("metadynamics estimates CV-density free energies, not rate_flux")
    for key in ("bootstrap", "bootstrap_replicates", "bootstrap_samples"):
        if checked_integer(analysis.get(key, 0), "analysis." + key, 0) != 0:
            raise ValueError("bootstrap of time-dependent metadynamics histories is not implemented; use independent replicas")
    sampling = config.get("sampling", {})
    neq = checked_integer(sampling.get("equilibration_steps", 0), "sampling.equilibration_steps", 0)
    nprod = checked_integer(sampling["production_steps"], "sampling.production_steps")
    interval = checked_integer(sampling.get("sample_interval", sampling.get("interval_steps", 1)), "sampling.sample_interval")
    checked_integer(sampling.get("andersen_interval", 10), "sampling.andersen_interval")
    checked_integer(sampling.get("replicas", 1), "sampling.replicas")
    checked_integer(config.get("output", {}).get("checkpoint_steps", 1000), "output.checkpoint_steps")
    if sampling.get("thermostat", "andersen") != "andersen":
        raise ValueError("metadynamics workflow sampling currently requires the Andersen thermostat")
    if not np.isfinite(sampling.get("dt_au", config["dt_au"])) or sampling.get("dt_au", config["dt_au"]) <= 0:
        raise ValueError("metadynamics timestep must be positive and finite")
    if min(nprod // interval, nprod // options["pace"]) < 1:
        raise ValueError("production_steps must include at least one sample and one hill deposition")
    bins = analysis.get("bins", 40)
    if np.isscalar(bins):
        bins = [bins] * cv.dimension
    if len(bins) != cv.dimension:
        raise ValueError("analysis.bins must contain one count per metadynamics CV")
    options["bins"] = [checked_integer(b, "analysis.bins", 2) for b in bins]
    if np.prod(options["bins"], dtype=float) > 1_000_000:
        raise ValueError("metadynamics analysis grid exceeds one million points")
    options["ranges"] = None
    if "ranges" in analysis:
        bounds = analysis["ranges"]
        if not isinstance(bounds, (list, tuple, np.ndarray)) or len(bounds) != cv.dimension:
            raise ValueError("analysis.ranges must contain one pair per CV")
        ranges = []
        for pair, unit, period in zip(bounds, cv.units, cv.periods):
            if len(pair) != 2:
                raise ValueError("analysis.ranges entries must contain [lower,upper]")
            lo, hi = [parse_quantity(v, _dimension(unit)) for v in pair]
            if not hi > lo:
                raise ValueError("analysis.ranges must be increasing")
            if period is not None and not np.isclose(hi - lo, period, rtol=1e-10, atol=1e-12):
                raise ValueError("a periodic metadynamics analysis range must span exactly one period")
            ranges.append([lo, hi])
        options["ranges"] = ranges
    options.update(neq=neq, nprod=nprod, interval=interval,
                   min_count=checked_integer(analysis.get("min_count", 1), "analysis.min_count"))
    return cv, selected, options


def validate_metadynamics(config, system):
    """Read-only early validation for both input frontends and the workflow."""
    resolve_metadynamics(config, system)


def _new_bias(cv, options, beta, hills=None):
    return MetadynamicsBias(cv, options["sigma"], options["height"], beta,
                           mode=options["mode"], bias_factor=options["bias_factor"], hills=hills)


def _hills_path(output, replica):
    return Path(output) / "replicas" / f"{replica:04d}" / "HILLS.csv"


def _write_hills(path, bias):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "time_au"] + [f"cv{j}_{u}" for j, u in enumerate(bias.cv.units)] +
                        [f"sigma{j}_{u}" for j, u in enumerate(bias.cv.units)] +
                        ["height_hartree", "bias_before_hartree"])
        for hill in bias.hills:
            writer.writerow([hill["step"], hill["time"]] + hill["center"] +
                            bias.sigma.tolist() + [hill["height"], hill["bias_before"]])
    temp.replace(path)


def _checkpoint_path(output, replica):
    return Path(output) / "checkpoints" / f"metadynamics_{replica:04d}.npz"


def _read_stream(config, system, output, replica, cv, options):
    checkpoint = _checkpoint_path(output, replica)
    expected = _fingerprint(config, system, {"metadynamics_replica": replica})
    with load_checkpoint(checkpoint) as data:
        if str(data["fingerprint"]) != expected:
            raise ValueError(f"checkpoint physics differs: {checkpoint}")
        elapsed = int(data["elapsed"])
        if elapsed < 0 or elapsed > options["neq"] + options["nprod"] or int(data["step"]) != elapsed:
            raise ValueError("metadynamics checkpoint has an invalid sampling clock")
        hills = json.loads(str(data["hills"]))
        bias = _new_bias(cv, options, config["beta"], hills)
        expected_hills = max(0, elapsed - options["neq"]) // options["pace"]
        if len(hills) != expected_hills or any(
            h["step"] != options["neq"] + (i + 1) * options["pace"] for i, h in enumerate(hills)
        ):
            raise ValueError("metadynamics checkpoint hill history disagrees with its deposition clock")
        samples = data["samples"].copy()
        clocks = data["sample_steps"].copy()
        expected_samples = max(0, elapsed - options["neq"]) // options["interval"]
        if samples.shape != (expected_samples, cv.dimension) or not np.array_equal(
            clocks, options["neq"] + np.arange(1, expected_samples + 1) * options["interval"]
        ) or not np.isfinite(samples).all():
            raise ValueError("metadynamics checkpoint samples disagree with their sampling clock")
        result = {"replica": replica, "state": State(data["q"], data["p"], int(data["step"]), float(data["time"])),
                  "elapsed": elapsed, "rng": json.loads(str(data["rng"])),
                  "samples": samples, "sample_steps": clocks,
                  "sample_bias": data["sample_bias"].copy(), "sample_energies": data["sample_energies"].copy(),
                  "hills": bias.hills, "propagation": json.loads(str(data["propagation"])),
                  "checkpoint": str(checkpoint), "complete": elapsed == options["neq"] + options["nprod"]}
    if result["sample_bias"].shape != (expected_samples,) or result["sample_energies"].shape != (expected_samples,):
        raise ValueError("metadynamics checkpoint observable shapes disagree")
    return result


def _sample_replica(task):
    config, system, output, replica = task["config"], task["system"], task["output"], task["replica"]
    cv, _, options = resolve_metadynamics(config, system)
    fingerprint = _fingerprint(config, system, {"metadynamics_replica": replica})
    checkpoint = _checkpoint_path(output, replica)
    provider = make_task_potential(config["potential"], system)
    try:
        dynamics = Dynamics(provider, system.masses, config["beta"], config["beads"],
                            config["sampling"].get("dt_au", config["dt_au"]),
                            seed=np.random.SeedSequence([int(config.get("seed", 0)), 709, replica]),
                            native=config.get("native", config.get("dynamics", {}).get("native", True)),
                            threads=config.get("parallel", {}).get("threads_per_rank", 1), **loop_options(config))
        bias = _new_bias(cv, options, config["beta"])
        samples, sample_steps, sample_bias, energies = [], [], [], []
        elapsed = 0
        if checkpoint.exists():
            saved = _read_stream(config, system, output, replica, cv, options)
            state = saved["state"]
            dynamics.rng.bit_generator.state = saved["rng"]
            dynamics.propagation_stats = saved["propagation"]
            bias = _new_bias(cv, options, config["beta"], saved["hills"])
            samples, sample_steps = list(saved["samples"]), list(saved["sample_steps"])
            sample_bias, energies = list(saved["sample_bias"]), list(saved["sample_energies"])
            elapsed = saved["elapsed"]
        else:
            state = State(np.repeat(system.geometry[None], config["beads"], axis=0), dynamics.sample_momenta())

        def save():
            _atomic_npz(checkpoint, fingerprint=fingerprint, elapsed=elapsed,
                        q=state.q, p=state.p, step=state.step, time=state.time,
                        rng=json.dumps(dynamics.rng.bit_generator.state),
                        hills=json.dumps(bias.hills, allow_nan=False),
                        samples=np.asarray(samples).reshape(-1, cv.dimension),
                        sample_steps=np.asarray(sample_steps, dtype=int),
                        sample_bias=np.asarray(sample_bias), sample_energies=np.asarray(energies),
                        propagation=json.dumps(dynamics.propagation_stats))
            _write_hills(_hills_path(output, replica), bias)

        # Reconstruct a stale/missing HILLS view even when the saved run is done.
        save()
        interval = config["sampling"].get("andersen_interval", 10)
        checkpoint_steps = config.get("output", {}).get("checkpoint_steps", 1000)
        total = options["neq"] + options["nprod"]
        while elapsed < total:
            previous = state.copy()
            old_rng = copy.deepcopy(dynamics.rng.bit_generator.state)
            nhills = bias.hill_count
            try:
                if state.step % interval == 0:
                    state.p[:] = dynamics.sample_momenta()
                # No propagation block can straddle a deposition. Both kicks
                # use this unchanged history; the new hill affects the next step.
                dynamics.step(state, bias)
                production = elapsed + 1 - options["neq"]
                record = production > 0 and production % options["interval"] == 0
                deposit = production > 0 and production % options["pace"] == 0
                value = cv.value(state.q) if record or deposit else None
                observed_bias = float(bias.physical_energy(value)) if record else None
                energy = float(dynamics.energy(state, bias)) if record else None
                if deposit:
                    bias.deposit(value, step=state.step, time=state.time)
            except BaseException:
                state = previous
                dynamics.rng.bit_generator.state = old_rng
                bias.truncate_hills(nhills)
                save()
                raise
            elapsed += 1
            dynamics.propagation_stats["python_steps"] += 1
            reason = "history-dependent metadynamics bias"
            reasons = dynamics.propagation_stats.setdefault("fallback_reasons", {})
            reasons[reason] = reasons.get(reason, 0) + 1
            if record:
                samples.append(value.copy())
                sample_steps.append(state.step)
                sample_bias.append(observed_bias)
                energies.append(energy)
            if elapsed % checkpoint_steps == 0:
                save()
        save()
        return _read_stream(config, system, output, replica, cv, options)
    finally:
        if hasattr(provider, "close"):
            provider.close()


def _analyze(streams, config, system, output, return_data):
    cv, _, options = resolve_metadynamics(config, system)
    biases = [_new_bias(cv, options, config["beta"], s["hills"]) for s in streams]
    data = bias_fes(biases, [s["samples"] for s in streams], bins=options["bins"],
                    ranges=options["ranges"], min_count=options["min_count"])
    output = Path(output)
    arrays = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
    arrays.update({f"edges_{j}": v for j, v in enumerate(data["edges"])})
    _atomic_npz(output / "fes.npz", **arrays)
    mesh = np.meshgrid(*[0.5 * (e[1:] + e[:-1]) for e in data["edges"]], indexing="ij")
    table = np.column_stack([m.ravel() for m in mesh] + [data[k].ravel() for k in
                             ("free_energy", "replica_spread", "mask", "raw_count")])
    np.savetxt(output / "fes.csv", table, delimiter=",", comments="",
               header=",".join([f"cv{j}_{u}" for j, u in enumerate(cv.units)] +
                               ["bias_estimated_free_energy_hartree", "replica_spread_hartree", "masked", "raw_count"]))
    warnings = list(config.get("_warnings", [])) + [
        "Finite-time bias estimate only: CV sufficiency, deposition width/pace, sampling, bead and timestep convergence remain unverified.",
        "Biased trajectory times do not provide reaction rates; no infrequent-metadynamics rate estimator or time-dependent reweighting is implemented.",
        "Unsampled bins are masked. Replica spread is descriptive, not a calibrated free-energy error bar.",
    ]
    if options["mode"] == "standard":
        warnings.append("Constant-height metadynamics continues depositing hills; -V can oscillate about the free energy and has finite-width/deposition error.")
    if not data["common_support"]:
        warnings.append("No common sampled bin between all replica histories; their relative gauges cannot be aligned and aggregate FES is masked.")
    coverage = [float(np.mean(~mask)) for mask in data["mask_per_replica"]]
    common_coverage = float(np.mean(~data["mask"]))
    in_range = [float(count.sum() / len(s["samples"]))
                for count, s in zip(data["raw_count_per_replica"], streams)]
    if options["ranges"] is not None and common_coverage < 0.5:
        warnings.append("Less than half of the requested FES grid has common sampled support; inspect CV coverage and increase exploration before interpreting the bias estimate.")
    if options["ranges"] is not None and min(in_range) < 0.5:
        warnings.append("A replica spends more than half its saved samples outside analysis.ranges. Analysis ranges only select output bins; they do not confine dynamics.")
    if any(isinstance(component, RPMDrateCoordinate) for component in cv.cvs):
        warnings.append("RPMDrate xi can have a pole outside the reaction corridor and is unsuitable for unconstrained global exploration. This cv_density estimate is not the rate_flux PMF used for RPMD rates.")
    summary = {
        "task": "metadynamics", "status": "bias_estimate" if all(s["complete"] for s in streams) else "partial_bias_estimate",
        "estimator": "bias_estimate", "mode": options["mode"], "bias_factor": options["bias_factor"],
        "fes_formula": "F=-V+C" if options["mode"] == "standard" else "F=-gamma/(gamma-1)*V+C",
        "fes_factor": data["fes_factor"], "measure": "cv_density", "energy_unit": "hartree",
        "dimension": cv.dimension, "units": cv.units, "replicas": len(streams),
        "hills_per_replica": [len(s["hills"]) for s in streams],
        "samples": sum(len(s["samples"]) for s in streams), "common_support": data["common_support"],
        "reference_bin": data["reference_bin"], "converged": False,
        "sampling_diagnostics": {
            "supported_grid_fraction_per_replica": coverage,
            "common_supported_grid_fraction": common_coverage,
            "samples_in_grid_fraction_per_replica": in_range,
            "sampled_cv_min_per_replica": [np.min(s["samples"], axis=0).tolist() for s in streams],
            "sampled_cv_max_per_replica": [np.max(s["samples"], axis=0).tolist() for s in streams],
            "deposited_height_per_replica": [sum(h["height"] for h in s["hills"]) for s in streams],
            "last_hill_height_per_replica": [s["hills"][-1]["height"] if s["hills"] else None for s in streams],
            "coverage_note": "Grid coverage is not equilibration or FES convergence; automatically chosen ranges describe only the visited domain.",
        },
        "propagation": [s["propagation"] for s in streams], "warnings": warnings,
        "sample_bias_convention": "physical V before deposition at the sampled step; energies include P*V",
        "files": {"data": str(output / "fes.npz"), "table": str(output / "fes.csv"),
                  "hills": [str(_hills_path(output, s["replica"])) for s in streams],
                  "checkpoints": [s["checkpoint"] for s in streams]},
    }
    write_json(output / "summary.json", summary)
    if return_data:
        return {"summary": summary, "data": data, "streams": streams}
    return summary


def run_metadynamics(config, system, output, executor=None, return_data=False):
    """Run independent histories and export their finite-time bias estimates."""
    validate_metadynamics(config, system)
    replicas = config.get("sampling", {}).get("replicas", 1)
    tasks = [{"config": copy.deepcopy(config), "system": system, "output": str(output), "replica": r}
             for r in range(replicas)]
    streams = (executor.map(_sample_replica, tasks) if executor is not None else
               run_tasks(_sample_replica, tasks, config.get("parallel", {})))
    return root_call(_analyze, streams, config, system, str(output), return_data)


def reanalyze_metadynamics(config, system, output, return_data=False):
    """Recompute the bias estimate from authoritative saved histories only."""
    def analyze():
        cv, _, options = resolve_metadynamics(config, system)
        streams = [_read_stream(config, system, output, r, cv, options)
                   for r in range(config.get("sampling", {}).get("replicas", 1))]
        for stream in streams:
            _write_hills(_hills_path(output, stream["replica"]), _new_bias(cv, options, config["beta"], stream["hills"]))
        return _analyze(streams, config, system, output, return_data)
    return root_call(analyze)
