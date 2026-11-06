"""Resumable metadynamics with independent replicas or synchronized walkers.

Independent replicas own their histories; shared walkers synchronize hills at
block boundaries and atomically checkpoint one global history. Every deposition is after a completed integration step, the two
force evaluations within a step see the same history, and checkpoint state,
RNG, sample clock and hills are committed in one atomic NPZ. HILLS.csv is a
regenerable view, never the authority for resuming a trajectory.
"""

from __future__ import annotations

import copy
import csv
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from ..analysis.metadynamics import bias_fes
from ..analysis.frozen_bias import frozen_bias_fes
from ..bias import CompositeBias, FluxMetricBias, GridMetadynamicsBias, MetadynamicsBias, WallBias
from ..checkpoint import load_checkpoint
from ..cv import RPMDrateCoordinate, make_cvs
from ..dynamics import Dynamics, State
from ..warm_start import initial_ring
from ..parallel import make_task_potential, root_call
from ..potentials import validate_periodic_spec
from ..propagation import loop_options
from ..units import parse_quantity
from ..thermostats import NoseHooverChain, nhc_options
from .common import (_atomic_npz, _fingerprint, _dimension, checked_integer,
                     fingerprint_matches, resolve_cvs)
from .sampling import (fes_sampler, legacy_metadynamics_sampling, middle_sampling_method,
                       read_operator_checkpoint, validate_andersen_schedule)
from .biased_sampling import SamplingRuntime, run_biased_sampling, dispatch_replicas, dispatch_tasks
from ..analysis.fes_result import FESResult, write_fes_result


METADYNAMICS_FIELDS = {"cvs", "mode", "sigma", "sigma_au", "height", "height_au", "pace", "bias_factor", "walls", "walls_during_equilibration", "grid", "shared_walkers", "multiple_walkers", "block_steps", "freeze_after_steps", "freeze_equilibration_steps"}


def resolve_walls(config, system):
    """Build scalar length-CV walls; bare bounds/kappa use atomic units."""
    result = []
    specs = config.get("cvs", [])
    if isinstance(specs, dict):
        specs = [dict(v, name=k) for k, v in specs.items()]
    raw = config.get("metadynamics", {}).get("walls", [])
    if not isinstance(raw, (list, tuple)):
        raise ValueError("metadynamics.walls must be an array of tables")
    for wall in raw:
        if not isinstance(wall, Mapping) or set(wall) - {"cv", "lower", "upper", "kappa"}:
            raise ValueError("wall accepts cv, lower, upper, kappa")
        if not isinstance(wall.get("cv"), str) or "kappa" not in wall:
            raise ValueError("wall requires a named cv and kappa")
        cv = make_cvs(specs, system, [wall["cv"]])
        if cv.units != ("bohr",):
            raise ValueError("workflow walls currently require scalar length CVs")
        result.append(WallBias(cv, parse_quantity(wall["kappa"], "spring"),
                               **{k: parse_quantity(wall[k], "length") for k in ("lower", "upper") if k in wall}))
    return CompositeBias(*result) if result else None


def _grid_options(table, cv, scales, sigma):
    if "grid" not in table:
        return None
    grid = table["grid"]
    if not isinstance(grid, Mapping) or set(grid) - {"ranges", "ranges_au", "bins"}:
        raise ValueError("metadynamics.grid must contain only ranges (or ranges_au) and bins")
    if ("ranges" in grid) == ("ranges_au" in grid):
        raise ValueError("metadynamics.grid requires exactly one of ranges and ranges_au")
    bounds = grid.get("ranges_au", grid.get("ranges"))
    if not isinstance(bounds, (list, tuple, np.ndarray)) or len(bounds) != cv.dimension:
        raise ValueError("metadynamics.grid.ranges needs one [lower,upper] pair per CV")
    ranges = []
    for pair, unit, scale in zip(bounds, cv.units, scales):
        if not isinstance(pair, (list, tuple, np.ndarray)) or len(pair) != 2:
            raise ValueError("metadynamics.grid.ranges needs one [lower,upper] pair per CV")
        ranges.append([parse_quantity(v, _dimension(unit)) *
                       (scale if "ranges_au" not in grid and isinstance(v, (int, float, np.number)) else 1.0)
                       for v in pair])
    # All numerical and periodic-domain validation is shared with the API.
    probe = GridMetadynamicsBias(cv, sigma, 1.0, 1.0, mode="standard",
                                grid_ranges=ranges, grid_bins=grid.get("bins"))
    return {"ranges": probe.grid_ranges.tolist(), "bins": probe.grid_bins.tolist()}

def resolve_metadynamics(config, system):
    """Validate algorithm options and convert widths/height to canonical units."""
    if not np.any(system.mobile_mask):
        raise ValueError("metadynamics requires at least one mobile atom")
    validate_periodic_spec(config.get("potential", {}), system)
    validate_andersen_schedule(config, "metadynamics")
    table = config.get("metadynamics", {})
    if not isinstance(table, Mapping):
        raise ValueError("metadynamics must be a table")
    extra = set(table) - METADYNAMICS_FIELDS
    if extra:
        raise ValueError("Unknown metadynamics fields: " + ", ".join(sorted(extra)))
    if not isinstance(table.get("walls_during_equilibration", True), bool):
        raise ValueError("metadynamics.walls_during_equilibration must be boolean")
    cv, selected, scales = resolve_cvs(config, system, "metadynamics")
    resolve_walls(config, system)
    if not isinstance(table.get("multiple_walkers", False), bool):
        raise ValueError("metadynamics.multiple_walkers must be boolean")
    if table.get("multiple_walkers", False) and table.get("shared_walkers", False):
        raise ValueError("metadynamics.shared_walkers and multiple_walkers are distinct modes; select only one")
    if "block_steps" in table and not table.get("multiple_walkers", False):
        raise ValueError("metadynamics.block_steps requires multiple_walkers=true")
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
    options["grid"] = _grid_options(table, cv, scales, sigma)
    if options["grid"] is not None:
        initial_bias = _new_bias(cv, options, config["beta"])
        initial_bias.physical_energy(cv.value(np.repeat(system.geometry[None], config["beads"], axis=0)))
    if config.get("umbrella"):
        raise ValueError("the metadynamics workflow does not combine an umbrella bias")
    if config.get("dynamics", {}).get("loop_backend", "auto") == "native":
        raise ValueError("history-dependent metadynamics bias requires loop_backend='auto' or 'python'")
    analysis = config.get("analysis", {})
    method = analysis.get("method", "bias_estimate")
    if method not in ("bias_estimate", "reweighted_histogram"):
        raise ValueError("metadynamics analysis.method must be bias_estimate or reweighted_histogram; evolving hills are not fixed-umbrella MBAR data")
    measure = analysis.get("measure", "cv_density")
    if measure not in ("cv_density", "equilibrium", "rate_flux"):
        raise ValueError("metadynamics analysis.measure must be cv_density or rate_flux")
    if measure == "rate_flux":
        if method != "reweighted_histogram":
            raise ValueError("rate_flux metadynamics requires frozen-bias reweighted_histogram, not a raw bias estimate")
        if (cv.dimension != 1 or len(cv.cvs) != 1
                or not isinstance(cv.cvs[0], RPMDrateCoordinate)):
            raise ValueError("rate_flux metadynamics requires one centroid RPMDrate coordinate")
        if any(system.pbc) or system.fixed_atoms:
            raise ValueError("rate_flux metadynamics requires a nonperiodic system with all atoms mobile")
        if table.get("walls") or table.get("shared_walkers") or table.get("multiple_walkers"):
            raise ValueError("rate_flux metadynamics does not combine static walls or shared walkers")
        FluxMetricBias(cv.cvs[0], system.masses, config["beta"])
    if config.get("replica_exchange", {}).get("enabled") or config.get("adaptive_umbrella", {}).get("enabled"):
        raise ValueError("metadynamics does not combine replica exchange or adaptive umbrellas")
    for key in ("bootstrap", "bootstrap_replicates", "bootstrap_samples"):
        if checked_integer(analysis.get(key, 0), "analysis." + key, 0) != 0:
            raise ValueError("metadynamics bootstrap is not implemented; frozen-bias replica spread is not a standard error")
    sampling = config.get("sampling", {})
    neq = checked_integer(sampling.get("equilibration_steps", 0), "sampling.equilibration_steps", 0)
    nprod = checked_integer(sampling["production_steps"], "sampling.production_steps")
    interval = checked_integer(sampling.get("sample_interval", sampling.get("interval_steps", 1)), "sampling.sample_interval")
    freeze = (checked_integer(table["freeze_after_steps"], "metadynamics.freeze_after_steps")
              if "freeze_after_steps" in table else None)
    relax = checked_integer(table.get("freeze_equilibration_steps", 0), "metadynamics.freeze_equilibration_steps", 0)
    if freeze is not None and (freeze % options["pace"] or freeze < options["pace"] or freeze > nprod):
        raise ValueError("metadynamics.freeze_after_steps must be a multiple of pace within production_steps")
    if relax and freeze is None:
        raise ValueError("metadynamics.freeze_equilibration_steps requires freeze_after_steps")
    if freeze is not None and (table.get("shared_walkers") or table.get("multiple_walkers")):
        raise ValueError("frozen-bias metadynamics currently requires independent histories, not shared walkers")
    if method == "reweighted_histogram":
        if freeze is None or nprod // interval - (freeze + relax) // interval < 2:
            raise ValueError("reweighted_histogram requires at least two saved frozen-bias samples after freeze_after_steps + freeze_equilibration_steps")
    options.update(method=method, measure="rate_flux" if measure == "rate_flux" else "cv_density",
                   freeze=freeze, freeze_equilibration_steps=relax)
    checked_integer(sampling.get("andersen_interval", 10), "sampling.andersen_interval")
    checked_integer(sampling.get("replicas", 1), "sampling.replicas")
    checked_integer(config.get("output", {}).get("checkpoint_steps", 1000), "output.checkpoint_steps")
    thermostat = sampling.get("thermostat", "andersen")
    if not legacy_metadynamics_sampling(sampling):
        fes_sampler(config)
    if thermostat == "nose-hoover-chain":
        nhc_options(sampling)
    if thermostat == "pile-l" and legacy_metadynamics_sampling(sampling):
        if loop_options(config)["integrator"] == "velocity-verlet":
            raise ValueError("PILE-L requires Cayley or normal-mode dynamics")
        middle_sampling_method(loop_options(config)["integrator"], sampling.get("integrator"))
        gamma = parse_quantity(sampling.get("gamma_centroid", "0.01 fs^-1"), "frequency")
        if gamma <= 0:
            raise ValueError("sampling.gamma_centroid must be positive")
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
    options["max_grid_points"] = checked_integer(analysis.get("max_grid_points", 10_000_000), "analysis.max_grid_points")
    if np.prod(options["bins"], dtype=float) > options["max_grid_points"]:
        raise ValueError("metadynamics analysis grid exceeds analysis.max_grid_points")
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
    if options["grid"] is not None:
        if options["ranges"] is None:
            options["ranges"] = options["grid"]["ranges"]
        else:
            for bounds, domain, period in zip(options["ranges"], options["grid"]["ranges"], cv.periods):
                if period is None and (bounds[0] < domain[0] or bounds[1] > domain[1]):
                    raise ValueError("metadynamics analysis.ranges must lie inside the bias grid")
    store_configurations = config.get("output", {}).get("store_configurations", False)
    if not isinstance(store_configurations, bool):
        raise ValueError("output.store_configurations must be boolean")
    options["multiple_walkers"] = table.get("multiple_walkers", False)
    options["block_steps"] = checked_integer(table.get("block_steps", options["pace"]), "metadynamics.block_steps")
    options.update(store_configurations=store_configurations, neq=neq, nprod=nprod, interval=interval,
                   min_count=checked_integer(analysis.get("min_count", 1), "analysis.min_count"))
    if not isinstance(table.get("shared_walkers", False), bool):
        raise ValueError("metadynamics.shared_walkers must be boolean")
    from ..warm_start import initial_hills
    options["initial_hills"] = initial_hills(config)
    if options["initial_hills"] is not None:
        if (table.get("shared_walkers") or options["multiple_walkers"]
                or options["grid"] is not None):
            raise ValueError("an imported initial bias requires independent direct metadynamics histories")
        for hill in options["initial_hills"]:
            if len(hill["center"]) != cv.dimension:
                raise ValueError("initial bias hill dimension differs from the metadynamics CVs")
    if table.get("shared_walkers", False):
        from .shared_metadynamics import validate_shared
        validate_shared(config, options)
    return cv, selected, options


def validate_metadynamics(config, system):
    """Read-only early validation for both input frontends and the workflow."""
    resolve_metadynamics(config, system)


def _new_bias(cv, options, beta, hills=None, grid_state=None):
    common = dict(mode=options["mode"], bias_factor=options["bias_factor"], hills=hills)
    if options.get("initial_hills"):
        if options["grid"] is not None:
            raise ValueError("an imported initial bias is not supported with grid metadynamics")
        return MetadynamicsBias(cv, options["sigma"], options["height"], beta,
                                initial_hills=options["initial_hills"], **common)
    if options["grid"] is not None:
        return GridMetadynamicsBias(cv, options["sigma"], options["height"], beta,
                                    grid_ranges=options["grid"]["ranges"],
                                    grid_bins=options["grid"]["bins"], grid_state=grid_state, **common)
    if grid_state is not None:
        raise ValueError("direct metadynamics checkpoint contains an unexpected grid")
    return MetadynamicsBias(cv, options["sigma"], options["height"], beta, **common)


class _MetadynamicsUpdate:
    """The algorithm-specific hill transaction used by the common sampler."""

    def __init__(self, bias, options, walker=None):
        self.bias, self.options, self.walker = bias, options, walker

    def due(self, production):
        return (production % self.options["pace"] == 0
                and (self.options.get("freeze") is None or production <= self.options["freeze"]))

    def snapshot(self):
        return self.bias.hill_count

    def apply(self, value, state):
        kwargs = {} if self.walker is None else {"walker": self.walker}
        self.bias.deposit(value, step=state.step, time=state.time, **kwargs)

    def restore(self, count):
        self.bias.truncate_history(count)


def _grid_payload(bias):
    if not isinstance(bias, GridMetadynamicsBias):
        return {}
    grid = bias.grid_state()
    return {"grid_values": grid["values"],
            "grid_metadata": json.dumps(grid["metadata"], allow_nan=False)}


def _hills_path(output, replica):
    return Path(output) / "replicas" / f"{replica:04d}" / "HILLS.csv"


def _write_hills(path, bias):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        shared = any("walker" in hill for hill in bias.hills)
        writer.writerow(["step", "time_au"] + [f"cv{j}_{u}" for j, u in enumerate(bias.cv.units)] +
                        [f"sigma{j}_{u}" for j, u in enumerate(bias.cv.units)] +
                        ["height_hartree", "bias_before_hartree"] + (["walker"] if shared else []))
        for hill in bias.hills:
            writer.writerow([hill["step"], hill["time"]] + hill["center"] +
                            bias.sigma.tolist() + [hill["height"], hill["bias_before"]] +
                            ([hill["walker"]] if shared else []))
    temp.replace(path)


def _checkpoint_path(output, replica):
    return Path(output) / "checkpoints" / f"metadynamics_{replica:04d}.npz"


def _shared_checkpoint_path(output):
    return Path(output) / "checkpoints" / "metadynamics_walkers.npz"


def _write_shared_checkpoint(config, system, output, streams, bias):
    """Commit the synchronized ensemble and exactly one global hill history."""
    path = _shared_checkpoint_path(output)
    elapsed = streams[0]["elapsed"]
    observations = {}
    if any("operator_state" in stream for stream in streams):
        if not all(isinstance(stream.get("operator_state"), dict) for stream in streams):
            raise ValueError("shared metadynamics requires operator state for every walker")
        observations["operator_states"] = json.dumps(
            [stream["operator_state"] for stream in streams], allow_nan=False)
    if config.get("output", {}).get("store_configurations", False):
        observations["sample_configurations"] = np.asarray(
            [stream["sample_configurations"] for stream in streams])
    _atomic_npz(path,
        fingerprint=_fingerprint(config, system, {"metadynamics_walkers": len(streams)}),
        elapsed=elapsed, step=elapsed,
        q=np.asarray([stream["state"].q for stream in streams]),
        p=np.asarray([stream["state"].p for stream in streams]),
        time=np.asarray([stream["state"].time for stream in streams]),
        rng=json.dumps([stream["rng"] for stream in streams]),
        thermostat_state=json.dumps([stream["thermostat_state"] for stream in streams]),
        propagation=json.dumps([stream["propagation"] for stream in streams]),
        samples=np.asarray([stream["samples"] for stream in streams]),
        sample_steps=streams[0]["sample_steps"],
        sample_bias=np.asarray([stream["sample_bias"] for stream in streams]),
        sample_energies=np.asarray([stream["sample_energies"] for stream in streams]),
        hills=json.dumps(bias.hills, allow_nan=False), **_grid_payload(bias), **observations)
    _write_hills(Path(output) / "HILLS.csv", bias)


def _read_shared_checkpoint(config, system, output, cv, options):
    path = _shared_checkpoint_path(output)
    replicas = config.get("sampling", {}).get("replicas", 1)
    with load_checkpoint(path) as data:
        if not fingerprint_matches(data["fingerprint"], config, system,
                                   {"metadynamics_walkers": replicas}):
            raise ValueError(f"checkpoint physics differs: {path}")
        for key in ("elapsed", "step"):
            clock = np.asarray(data[key])
            if clock.shape != () or clock.dtype.kind not in "iu" or clock < 0:
                raise ValueError("shared metadynamics checkpoint requires integer scalar clocks")
        elapsed = int(data["elapsed"])
        total = options["neq"] + options["nprod"]
        if elapsed > total or int(data["step"]) != elapsed:
            raise ValueError("shared metadynamics checkpoint has an invalid sampling clock")
        if elapsed != total and elapsed % options["block_steps"]:
            raise ValueError("shared metadynamics checkpoint must end at a block boundary")
        hills = json.loads(str(data["hills"]))
        updates = max(0, elapsed - options["neq"]) // options["pace"]
        expected = [(options["neq"] + (event + 1) * options["pace"], walker)
                    for event in range(updates) for walker in range(replicas)]
        if [(hill.get("step"), hill.get("walker")) for hill in hills] != expected:
            raise ValueError("shared metadynamics hill history disagrees with walker deposition clocks")
        dt = config["sampling"].get("dt_au", config["dt_au"])
        if any(not np.isclose(hill["time"], hill["step"] * dt, rtol=1e-10, atol=1e-10)
               for hill in hills):
            raise ValueError("shared metadynamics hill times disagree with deposition clocks")
        cached_grid = None
        if "grid_values" in data or "grid_metadata" in data:
            if ("grid_values" not in data or "grid_metadata" not in data
                    or data.envelope["integrity"] != "verified"):
                raise ValueError("metadynamics grid checkpoint requires a complete integrity-verified cache")
            cached_grid = {"values": data["grid_values"], "metadata": json.loads(str(data["grid_metadata"]))}
        bias = _new_bias(cv, options, config["beta"], hills, cached_grid)
        count = max(0, elapsed - options["neq"]) // options["interval"]
        clocks = data["sample_steps"].copy()
        if not np.array_equal(clocks, options["neq"] + np.arange(1, count + 1) * options["interval"]):
            raise ValueError("shared metadynamics samples disagree with the sampling clock")
        shapes = {"q": (replicas, config["beads"], system.natoms, 3),
                  "p": (replicas, config["beads"], system.natoms, 3), "time": (replicas,),
                  "samples": (replicas, count, cv.dimension),
                  "sample_bias": (replicas, count), "sample_energies": (replicas, count)}
        stored = "sample_configurations" in data.files
        if stored != options["store_configurations"]:
            raise ValueError("checkpoint configuration history differs from output.store_configurations; "
                             "restore the original option or start a new run")
        if stored:
            shapes["sample_configurations"] = (replicas, count, config["beads"], system.natoms, 3)
        arrays = {key: data[key].copy() for key in shapes}
        if any(array.shape != shapes[key] or not np.isfinite(array).all()
               for key, array in arrays.items()):
            raise ValueError("shared metadynamics checkpoint has invalid state or observable arrays")
        if not np.allclose(arrays["time"], elapsed * dt, rtol=1e-10, atol=1e-10):
            raise ValueError("shared metadynamics checkpoint time disagrees with its sampling clock")
        rngs = json.loads(str(data["rng"]))
        baths = json.loads(str(data["thermostat_state"]))
        stats = json.loads(str(data["propagation"]))
        if any(not isinstance(items, list) or len(items) != replicas for items in (rngs, baths, stats)):
            raise ValueError("shared metadynamics checkpoint walker metadata count differs")
        operators = json.loads(str(data["operator_states"])) if "operator_states" in data else None
        if operators is not None and (not isinstance(operators, list) or len(operators) != replicas):
            raise ValueError("shared metadynamics checkpoint operator state count differs")
        sampler = None if legacy_metadynamics_sampling(config["sampling"]) else fes_sampler(config)
        streams = []
        for walker in range(replicas):
            operator = read_operator_checkpoint(
                {} if operators is None else {"operator_state": json.dumps(operators[walker])},
                sampler, elapsed, "shared metadynamics",
            )
            np.random.default_rng().bit_generator.state = rngs[walker]
            state = State(arrays["q"][walker], arrays["p"][walker], elapsed, arrays["time"][walker])
            if system.fixed_atoms:
                indices = np.asarray(system.fixed_atoms)
                expected_positions = np.broadcast_to(system.geometry[indices], state.q[:, indices].shape)
                if not np.array_equal(state.q[:, indices], expected_positions) or np.any(state.p[:, indices] != 0):
                    raise ValueError("fixed atoms must have reference positions on every bead and zero momenta")
            if isinstance(bias, GridMetadynamicsBias):
                bias._coordinates(cv.value(state.q))
                bias._coordinates(arrays["samples"][walker])
            streams.append({"replica": walker, "state": state, "elapsed": elapsed,
                "rng": rngs[walker], "thermostat_state": baths[walker],
                "samples": arrays["samples"][walker], "sample_steps": clocks.copy(),
                "sample_bias": arrays["sample_bias"][walker],
                "sample_energies": arrays["sample_energies"][walker],
                "propagation": stats[walker], "checkpoint": str(path), "complete": elapsed == total})
            if operator is not None:
                streams[-1]["operator_state"] = operator
            if stored:
                streams[-1]["sample_configurations"] = arrays["sample_configurations"][walker]
    return {"streams": streams, "hills": bias.hills,
            "grid_state": bias.grid_state() if isinstance(bias, GridMetadynamicsBias) else None,
            "elapsed": elapsed}


def _run_multiple_walkers(config, system, output, executor, return_data):
    """Synchronize local deposits after each complete block, in stable order.

    Each block starts from the last global history. Within the block each
    walker sees its own new hills; well-tempered heights are never recomputed
    retrospectively. Failed blocks leave the previous ensemble checkpoint
    untouched, so rerunning restores every walker and its RNG together.
    """
    cv, _, options = resolve_metadynamics(config, system)
    replicas = config.get("sampling", {}).get("replicas", 1)

    def prepare():
        if _shared_checkpoint_path(output).exists():
            return _read_shared_checkpoint(config, system, output, cv, options)
        if any((Path(output) / "checkpoints").glob("metadynamics_[0-9]*.npz")):
            raise ValueError("independent metadynamics checkpoints cannot become shared walkers; choose a fresh output directory")
        return {"streams": [None] * replicas, "hills": [], "grid_state": None, "elapsed": 0}

    ensemble = root_call(prepare)
    total = options["neq"] + options["nprod"]
    while ensemble["elapsed"] < total:
        stop = min(total, ensemble["elapsed"] + options["block_steps"])
        tasks = [{"config": copy.deepcopy(config), "system": system, "output": str(output),
                  "replica": walker, "shared": True, "stop": stop,
                  "saved": ensemble["streams"][walker], "hills": ensemble["hills"],
                  "grid_state": ensemble["grid_state"]} for walker in range(replicas)]
        streams = dispatch_tasks(_sample_replica, tasks, config, executor)

        def synchronize():
            completed = sorted(list(streams), key=lambda stream: stream["replica"])
            if [stream["replica"] for stream in completed] != list(range(replicas)):
                raise ValueError("shared metadynamics scheduler returned inconsistent walker identities")
            if any(stream["elapsed"] != stop for stream in completed):
                raise ValueError("shared metadynamics walkers did not reach the synchronization boundary")
            previous_count = len(ensemble["hills"])
            deposits = [hill for stream in completed for hill in stream["hills"][previous_count:]]
            deposits.sort(key=lambda hill: (hill["step"], hill["walker"]))
            bias = _new_bias(cv, options, config["beta"], ensemble["hills"] + deposits)
            for stream in completed:
                del stream["hills"]
            _write_shared_checkpoint(config, system, output, completed, bias)
            return {"streams": completed, "hills": bias.hills,
                    "grid_state": bias.grid_state() if isinstance(bias, GridMetadynamicsBias) else None,
                    "elapsed": stop}

        ensemble = root_call(synchronize)
    return root_call(_analyze_shared, ensemble, config, system, output, return_data)


def _analyze_shared(ensemble, config, system, output, return_data):
    streams = ensemble["streams"]
    # A global bias is one statistical history, regardless of walker count.
    pooled = {**streams[0], "samples": np.concatenate([s["samples"] for s in streams]),
              "hills": ensemble["hills"], "grid_state": ensemble["grid_state"],
              "shared_walkers": streams}
    result = _analyze([pooled], config, system, output, True)
    if return_data:
        result["streams"] = streams
        result["hills"] = ensemble["hills"]
        return result
    return result["summary"]


def _read_stream(config, system, output, replica, cv, options):
    checkpoint = _checkpoint_path(output, replica)
    with load_checkpoint(checkpoint) as data:
        if not fingerprint_matches(data["fingerprint"], config, system, {"metadynamics_replica": replica}):
            raise ValueError(f"checkpoint physics differs: {checkpoint}")
        elapsed = int(data["elapsed"])
        if elapsed < 0 or elapsed > options["neq"] + options["nprod"] or int(data["step"]) != elapsed:
            raise ValueError("metadynamics checkpoint has an invalid sampling clock")
        hills = json.loads(str(data["hills"]))
        cached_grid = None
        if "grid_values" in data or "grid_metadata" in data:
            if "grid_values" not in data or "grid_metadata" not in data or data.envelope["integrity"] != "verified":
                raise ValueError("metadynamics grid checkpoint requires a complete integrity-verified cache")
            cached_grid = {"values": data["grid_values"], "metadata": json.loads(str(data["grid_metadata"]))}
        bias = _new_bias(cv, options, config["beta"], hills, cached_grid)
        adaptation = max(0, elapsed - options["neq"])
        if options.get("freeze") is not None:
            adaptation = min(adaptation, options["freeze"])
        expected_hills = adaptation // options["pace"]
        if len(hills) != expected_hills or any(
            h["step"] != options["neq"] + (i + 1) * options["pace"] for i, h in enumerate(hills)
        ):
            raise ValueError("metadynamics checkpoint hill history disagrees with its deposition clock")
        sampling_dt = config["sampling"].get("dt_au", config["dt_au"])
        if any(not np.isclose(h["time"], h["step"] * sampling_dt, rtol=1e-10, atol=1e-10)
               for h in hills):
            raise ValueError("metadynamics checkpoint hill time disagrees with its deposition clock")
        samples = data["samples"].copy()
        clocks = data["sample_steps"].copy()
        expected_samples = max(0, elapsed - options["neq"]) // options["interval"]
        if samples.shape != (expected_samples, cv.dimension) or not np.array_equal(
            clocks, options["neq"] + np.arange(1, expected_samples + 1) * options["interval"]
        ) or not np.isfinite(samples).all():
            raise ValueError("metadynamics checkpoint samples disagree with their sampling clock")
        configurations = {}
        stored = "sample_configurations" in data.files
        if stored != options["store_configurations"]:
            raise ValueError("checkpoint configuration history differs from output.store_configurations; "
                             "restore the original option or start a new run")
        if stored:
            positions = data["sample_configurations"].copy()
            if (positions.shape != (expected_samples, config["beads"], len(system.masses), 3)
                    or not np.isfinite(positions).all()):
                raise ValueError("metadynamics checkpoint configurations disagree with the sampling clock")
            configurations["sample_configurations"] = positions
        result = {**configurations, "replica": replica, "state": State(data["q"], data["p"], int(data["step"]), float(data["time"])),
                  "elapsed": elapsed, "rng": json.loads(str(data["rng"])),
                  "thermostat_state": json.loads(str(data["thermostat_state"])) if "thermostat_state" in data else None,
                  "samples": samples, "sample_steps": clocks,
                  "sample_bias": data["sample_bias"].copy(), "sample_energies": data["sample_energies"].copy(),
                  "hills": bias.hills, "propagation": json.loads(str(data["propagation"])),
                  "checkpoint": str(checkpoint), "complete": elapsed == options["neq"] + options["nprod"]}
        sampler = None if legacy_metadynamics_sampling(config["sampling"]) else fes_sampler(config)
        operators = read_operator_checkpoint(data, sampler, elapsed, "metadynamics")
        if operators is not None:
            result["operator_state"] = operators
        if isinstance(bias, GridMetadynamicsBias):
            result["grid_state"] = bias.grid_state()
    if result["sample_bias"].shape != (expected_samples,) or result["sample_energies"].shape != (expected_samples,):
        raise ValueError("metadynamics checkpoint observable shapes disagree")
    state = result["state"]
    if state.q.shape != (config["beads"], system.natoms, 3):
        raise ValueError("metadynamics checkpoint state shape disagrees with system")
    expected_time = elapsed * config["sampling"].get("dt_au", config["dt_au"])
    if not np.isclose(state.time, expected_time, rtol=1e-10, atol=1e-10):
        raise ValueError("metadynamics checkpoint time disagrees with its sampling clock")
    if system.fixed_atoms:
        indices = np.asarray(system.fixed_atoms)
        expected_positions = np.broadcast_to(system.geometry[indices], state.q[:, indices].shape)
        if not np.array_equal(state.q[:, indices], expected_positions) or np.any(state.p[:, indices] != 0.0):
            raise ValueError("fixed atoms must have reference positions on every bead and zero momenta")
    if not np.isfinite(result["sample_bias"]).all() or not np.isfinite(result["sample_energies"]).all():
        raise ValueError("metadynamics checkpoint observables must be finite")
    if isinstance(bias, GridMetadynamicsBias):
        # Also validate completed restart states; no next force evaluation is
        # guaranteed to occur when only regenerating output files.
        bias._coordinates(cv.value(state.q))
        bias._coordinates(result["samples"])
    return result


def _sample_replica(task):
    config, system, output, replica = task["config"], task["system"], task["output"], task["replica"]
    cv, _, options = resolve_metadynamics(config, system)
    fingerprint = _fingerprint(config, system, {"metadynamics_replica": replica})
    shared = task.get("shared", False)
    checkpoint = _shared_checkpoint_path(output) if shared else _checkpoint_path(output, replica)
    provider = make_task_potential(config["potential"], system)
    try:
        dynamics = Dynamics(provider, system.masses, config["beta"], config["beads"],
                            config["sampling"].get("dt_au", config["dt_au"]),
                            seed=np.random.SeedSequence([int(config.get("seed", 0)), 709, replica]),
                            native=config.get("native", config.get("dynamics", {}).get("native", True)),
                            threads=config.get("parallel", {}).get("threads_per_rank", 1),
                            fixed_atoms=system.fixed_atoms, fixed_positions=system.geometry, **loop_options(config))
        thermostat = config["sampling"].get("thermostat", "andersen")
        sampler = None if legacy_metadynamics_sampling(config["sampling"]) else fes_sampler(config)
        store_operators = sampler is not None and sampler.operator_checkpoint
        chain = NoseHooverChain(dynamics, **nhc_options(config["sampling"])) if thermostat == "nose-hoover-chain" else None
        walls = resolve_walls(config, system)
        bias = _new_bias(cv, options, config["beta"])
        samples, sample_steps, sample_bias, energies = [], [], [], []
        configurations = []
        elapsed = 0
        saved = task.get("saved") if shared else (_read_stream(config, system, output, replica, cv, options) if checkpoint.exists() else None)
        if saved is not None:
            state = saved["state"]
            dynamics.integrator._check(state.q, state.p)
            dynamics._check_fixed(state.q, state.p)
            if chain is not None:
                chain.load_state_dict(saved["thermostat_state"])
            if "operator_state" in saved:
                dynamics.load_operator_state_dict(saved["operator_state"])
                store_operators = True
            dynamics.rng.bit_generator.state = saved["rng"]
            dynamics.propagation_stats = saved["propagation"]
            bias = _new_bias(cv, options, config["beta"], task["hills"] if shared else saved["hills"],
                             task.get("grid_state") if shared else saved.get("grid_state"))
            samples, sample_steps = list(saved["samples"]), list(saved["sample_steps"])
            sample_bias, energies = list(saved["sample_bias"]), list(saved["sample_energies"])
            elapsed = saved["elapsed"]
            if options["store_configurations"]:
                configurations = list(saved["sample_configurations"])
        else:
            state = State(initial_ring(config, system), dynamics.sample_momenta())
        metric = (FluxMetricBias(cv.cvs[0], system.masses, config["beta"])
                  if options["measure"] == "rate_flux" else None)

        def total_bias(current, include_walls=True):
            active_walls = walls if include_walls else None
            return (current if active_walls is None and metric is None
                    else CompositeBias(current, active_walls, metric))

        runtime = SamplingRuntime(state, elapsed, samples, sample_steps, sample_bias, energies, chain,
                                  configurations if options["store_configurations"] else None,
                                  operator_checkpoint=store_operators)
        update = _MetadynamicsUpdate(bias, options, replica if shared else None)

        def save(current):
            grid_payload = _grid_payload(update.bias)
            observations = {}
            if store_operators:
                observations["operator_state"] = json.dumps(dynamics.operator_state_dict(), allow_nan=False)
            if current.configurations is not None:
                observations["sample_configurations"] = np.asarray(current.configurations).reshape(
                    -1, config["beads"], system.natoms, 3)
            _atomic_npz(checkpoint, fingerprint=fingerprint, elapsed=current.elapsed,
                        q=current.state.q, p=current.state.p, step=current.state.step, time=current.state.time,
                        rng=json.dumps(dynamics.rng.bit_generator.state),
                        thermostat_state=json.dumps(current.chain.state_dict() if current.chain else None),
                        hills=json.dumps(update.bias.hills, allow_nan=False),
                        samples=np.asarray(current.values).reshape(-1, cv.dimension),
                        sample_steps=np.asarray(current.clocks, dtype=int),
                        sample_bias=np.asarray(current.biases), sample_energies=np.asarray(current.energies),
                        propagation=json.dumps(dynamics.propagation_stats), **grid_payload, **observations)
            _write_hills(_hills_path(output, replica), update.bias)

        run_biased_sampling(runtime, dynamics, cv, neq=options["neq"], nprod=options["nprod"],
                            interval=options["interval"], sampling=config["sampling"],
                            checkpoint_steps=config.get("output", {}).get("checkpoint_steps", 1000),
                            update=update, sampler=sampler,
                            save=None if shared else save, stop=task.get("stop"),
                            active_bias=lambda elapsed, current: total_bias(
                                current, elapsed >= options["neq"] or config["metadynamics"].get("walls_during_equilibration", True)),
                            energy_bias=total_bias,
                            fallback_reason="history-dependent metadynamics bias")
        if shared:
            observations = {}
            if store_operators:
                observations["operator_state"] = dynamics.operator_state_dict()
            if runtime.configurations is not None:
                observations["sample_configurations"] = np.asarray(runtime.configurations).reshape(
                    -1, config["beads"], system.natoms, 3)
            return {**observations, "replica": replica, "state": runtime.state, "elapsed": runtime.elapsed,
                    "rng": dynamics.rng.bit_generator.state,
                    "thermostat_state": runtime.chain.state_dict() if runtime.chain else None,
                    "samples": np.asarray(runtime.values).reshape(-1, cv.dimension),
                    "sample_steps": np.asarray(runtime.clocks, dtype=int),
                    "sample_bias": np.asarray(runtime.biases), "sample_energies": np.asarray(runtime.energies),
                    "hills": update.bias.hills, "propagation": dynamics.propagation_stats,
                    "checkpoint": str(checkpoint), "complete": runtime.elapsed == options["neq"] + options["nprod"]}
        return _read_stream(config, system, output, replica, cv, options)
    finally:
        if hasattr(provider, "close"):
            provider.close()


def _analyze(streams, config, system, output, return_data):
    # External executor.map results may be consumed only once, while bias
    # reconstruction, diagnostics and export all need the same replicas.
    streams = list(streams)
    if not streams:
        raise ValueError("metadynamics analysis requires at least one replica")
    cv, _, options = resolve_metadynamics(config, system)
    biases = [_new_bias(cv, options, config["beta"], s["hills"], s.get("grid_state")) for s in streams]
    reweighted = options["method"] == "reweighted_histogram"
    if reweighted:
        data = frozen_bias_fes(
            streams, config["beta"], bins=options["bins"], ranges=options["ranges"],
            periods=cv.periods, widths=options["sigma"],
            equilibration_steps=options["neq"], freeze_after_steps=options["freeze"],
            freeze_equilibration_steps=options["freeze_equilibration_steps"],
            min_count=options["min_count"], final_biases=biases,
        )
        cutoff = options["neq"] + options["freeze"] + options["freeze_equilibration_steps"]
        observed = [s["samples"][s["sample_steps"] > cutoff] for s in streams]
    else:
        data = bias_fes(biases, [s["samples"] for s in streams], bins=options["bins"],
                        ranges=options["ranges"], min_count=options["min_count"], max_grid_points=options["max_grid_points"])
        observed = [s["samples"] for s in streams]
    output = Path(output)
    warnings = list(config.get("_warnings", [])) + [
        ("Only fixed-bias production after adaptation and relaxation is reweighted; equilibrium, CV, bead and timestep convergence remain unverified."
         if reweighted else "Finite-time bias estimate only: CV sufficiency, deposition width/pace, sampling, bead and timestep convergence remain unverified."),
        "Biased trajectory times do not provide reaction rates; no infrequent-metadynamics rate estimator or time-dependent reweighting is implemented.",
        "Unsampled bins are masked. Replica spread is descriptive, not a calibrated free-energy error bar.",
    ]
    if options["mode"] == "standard" and not reweighted:
        warnings.append("Constant-height metadynamics continues depositing hills; -V can oscillate about the free energy and has finite-width/deposition error.")
    if options["grid"] is not None:
        warnings.append("The applied bias uses Hermite grid interpolation. Refine grid bins independently of FES analysis bins to assess interpolation error.")
    if not data["common_support"]:
        warnings.append("No common sampled bin between all replica histories; their relative gauges cannot be aligned and aggregate FES is masked.")
    coverage = [float(np.mean(~mask)) for mask in data["mask_per_replica"]]
    common_coverage = float(np.mean(~data["mask"]))
    in_range = [float(count.sum() / len(values))
                for count, values in zip(data["raw_count_per_replica"], observed)]
    if options["ranges"] is not None and common_coverage < 0.5:
        warnings.append("Less than half of the requested FES grid has common sampled support; inspect CV coverage and increase exploration before interpreting the bias estimate.")
    if options["ranges"] is not None and min(in_range) < 0.5:
        warnings.append("A replica spends more than half its saved samples outside analysis.ranges. Analysis ranges only select output bins; they do not confine dynamics.")
    if any(isinstance(component, RPMDrateCoordinate) for component in cv.cvs):
        warnings.append("RPMDrate xi can have a pole outside the reaction corridor and is unsuitable for unconstrained global exploration. "
                        + ("The rate_flux metric is sampled explicitly; sufficient stationary coverage must still be established."
                           if options["measure"] == "rate_flux" else "This cv_density estimate is not the rate_flux PMF used for RPMD rates."))
    summary = {
        "task": "metadynamics", "beta": config["beta"], "status": options["method"] if all(s["complete"] for s in streams) else "partial_" + options["method"],
        "estimator": options["method"], "mode": options["mode"], "bias_factor": options["bias_factor"],
        "bias_backend": "grid" if options["grid"] is not None else "direct",
        "bias_grid": options["grid"], "fixed_atoms": list(system.fixed_atoms),
        "fes_formula": ("F=-log(reweighted bin mass / bin volume)/beta+C" if reweighted
                        else "F=-V+C" if options["mode"] == "standard" else "F=-gamma/(gamma-1)*V+C"),
        "fes_factor": data.get("fes_factor"), "measure": options["measure"], "energy_unit": "hartree",
        "dimension": cv.dimension, "units": cv.units, "replicas": len(streams),
        "thermostat": config["sampling"].get("thermostat", "andersen"),
        "fixed_atoms": list(system.fixed_atoms), "pbc": list(system.pbc),
        "wall_ensemble": bool(config.get("metadynamics", {}).get("walls")),
        "hills_per_replica": [len(s["hills"]) for s in streams],
        "initial_bias_hills": len(options["initial_hills"] or ()),
        "samples": sum(len(values) for values in observed), "common_support": data["common_support"],
        "reference_bin": data["reference_bin"], "converged": False,
        "sampling_diagnostics": {
            "supported_grid_fraction_per_replica": coverage,
            "common_supported_grid_fraction": common_coverage,
            "samples_in_grid_fraction_per_replica": in_range,
            "sampled_cv_min_per_replica": [np.min(values, axis=0).tolist() for values in observed],
            "sampled_cv_max_per_replica": [np.max(values, axis=0).tolist() for values in observed],
            "deposited_height_per_replica": [sum(h["height"] for h in s["hills"]) for s in streams],
            "last_hill_height_per_replica": [s["hills"][-1]["height"] if s["hills"] else None for s in streams],
            "coverage_note": "Grid coverage is not equilibration or FES convergence; automatically chosen ranges describe only the visited domain.",
        },
        "propagation": [s["propagation"] for s in streams], "warnings": warnings,
        "stored_configurations": options["store_configurations"],
        "sample_bias_convention": "physical MetaD V before deposition; energies include P*(V+walls); FES includes the wall-defined ensemble",
        "files": {"data": str(output / "fes.npz"), "table": str(output / "fes.csv"),
                  "hills": [str(_hills_path(output, s["replica"])) for s in streams],
                  "checkpoints": [s["checkpoint"] for s in streams]},
    }
    if options["freeze"] is not None:
        summary.update(freeze_after_steps=options["freeze"],
                       freeze_equilibration_steps=options["freeze_equilibration_steps"])
    if reweighted:
        summary.update(reweighted=True,
                       retained_samples_per_replica=data["retained_samples_per_replica"],
                       weight_effective_sample_size=data["weight_effective_sample_size"],
                       uncertainty_note="Replica spread and weight-only ESS do not account for autocorrelation; PMF standard error is unavailable.",
                       sample_bias_convention=("physical frozen V; H_P contains P*V"
                           + (" - P*log(flux_metric)/beta" if options["measure"] == "rate_flux" else " plus any static walls")
                           + "; log weights are beta*V and preserve the declared target measure"))
    if options["multiple_walkers"]:
        walkers = streams[0]["shared_walkers"]
        summary.update(multiple_walkers=True, walkers=len(walkers), independent_histories=1,
                       synchronization_steps=options["block_steps"], global_hills=len(streams[0]["hills"]),
                       propagation=[walker["propagation"] for walker in walkers],
                       uncertainty_note="Shared-bias walkers are correlated; replica spread requires independent ensembles and is unavailable for this one global history.")
        summary["warnings"].append(summary["uncertainty_note"])
        summary["files"]["hills"] = [str(output / "HILLS.csv")]
    result = FESResult.from_data(data, method="metadynamics", cv_units=cv.units, metadata=summary)
    summary = write_fes_result(output, result, summary)
    if return_data:
        return {"summary": summary, "data": data, "streams": streams}
    return summary


def run_metadynamics(config, system, output, executor=None, return_data=False):
    """Run independent histories or either supported shared-walker ensemble."""
    from . import validate_bond_constraint_workflow
    validate_bond_constraint_workflow(system, "metadynamics")
    validate_andersen_schedule(config, "metadynamics")
    validate_metadynamics(config, system)
    if config.get("metadynamics", {}).get("shared_walkers", False):
        if executor is not None:
            raise ValueError("shared walkers use a synchronous coordinator, without an external executor")
        from .shared_metadynamics import run_shared_metadynamics
        return run_shared_metadynamics(config, system, output, return_data=return_data)
    if config.get("metadynamics", {}).get("multiple_walkers", False):
        return _run_multiple_walkers(config, system, output, executor, return_data)
    if _shared_checkpoint_path(output).exists():
        raise ValueError("shared metadynamics checkpoints cannot become independent replicas; choose a fresh output directory")
    streams = dispatch_replicas(config, system, output, _sample_replica, executor)
    return root_call(_analyze, streams, config, system, str(output), return_data)


def reanalyze_metadynamics(config, system, output, return_data=False):
    """Recompute the bias estimate from authoritative saved histories only."""
    if config.get("metadynamics", {}).get("shared_walkers", False):
        from .shared_metadynamics import reanalyze_shared_metadynamics
        return reanalyze_shared_metadynamics(config, system, output, return_data=return_data)
    def analyze():
        cv, _, options = resolve_metadynamics(config, system)
        if options["multiple_walkers"]:
            ensemble = _read_shared_checkpoint(config, system, output, cv, options)
            _write_hills(Path(output) / "HILLS.csv",
                         _new_bias(cv, options, config["beta"], ensemble["hills"], ensemble["grid_state"]))
            return _analyze_shared(ensemble, config, system, output, return_data)
        streams = [_read_stream(config, system, output, r, cv, options)
                   for r in range(config.get("sampling", {}).get("replicas", 1))]
        for stream in streams:
            _write_hills(_hills_path(output, stream["replica"]),
                         _new_bias(cv, options, config["beta"], stream["hills"], stream.get("grid_state")))
        return _analyze(streams, config, system, output, return_data)
    return root_call(analyze)
