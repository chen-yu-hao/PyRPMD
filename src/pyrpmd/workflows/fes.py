"""Independent umbrella trajectories and 1D/2D MBAR reconstruction.

The physical umbrella W(s) is inserted as P*W in the sum-beads Hamiltonian.
Each stream owns its random generator and resumable state; scheduling does not
change stream identities. Only the coordinator writes aggregate products.
"""

from __future__ import annotations

from ..propagation import execution_options, scientific_config

import copy
import hashlib
import json
import os
from pathlib import Path
import numpy as np

from ..analysis.fes import reconstruct_fes
from ..bias import UmbrellaBias, FluxMetricBias, CompositeBias, displacement
from ..cv import make_cvs, RPMDrateCoordinate
from ..dynamics import Dynamics, State
from ..parallel import run_tasks, make_task_potential, root_call
from ..units import parse_quantity, unit_factor, DIMS


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(jsonable(data), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temp.replace(path)


def _fingerprint(config, system, extras):
    # Analysis settings, output path and worker count do not change a stream.
    physics = {
        k: v
        for k, v in scientific_config(config).items()
        if k
        not in (
            "parallel",
            "output",
            "analysis",
            "resume",
            "_resume",
            "_input_path",
            "_base_directory",
            "_warnings",
        )
    }
    value = {
        "schema": 1,
        "physics": physics,
        "sampling_measure": config.get("analysis", {}).get("measure", "cv_density"),
        "system": system.to_dict(),
        "stream": extras,
    }
    return hashlib.sha256(
        json.dumps(
            jsonable(value), sort_keys=True, separators=(",", ":"), default=repr
        ).encode()
    ).hexdigest()


def _atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _dimension(unit):
    return {
        "bohr": DIMS["length"],
        "radian": DIMS["angle"],
        "hartree": DIMS["energy"],
        "dimensionless": DIMS["dimensionless"],
    }[unit]


def checked_integer(value, name, minimum=1):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def resolve_umbrellas(config, system):
    """Resolve user centers/stiffness to canonical CV units and hartree.

    A CV's optional input ``unit``/``units`` applies to numerical centers;
    explicit quantity strings carry their own units. Numerical stiffness is
    always atomic; quantity strings are dimension-checked for mixed CVs.
    """
    specs = config.get("cvs", [])
    if isinstance(specs, dict):
        specs = [dict(v, name=k) for k, v in specs.items()]
    umbrella = config.get("umbrella", {})
    names = umbrella.get("cvs")
    if names is not None and isinstance(names, str):
        names = [names]
    cv = make_cvs(specs, system, names)
    selected = (
        specs
        if names is None
        else [next(s for s in specs if s.get("name") == n) for n in names]
    )
    if cv.dimension not in (1, 2):
        raise ValueError("umbrella FES supports one or two CV dimensions")
    dims = [_dimension(u) for u in cv.units]
    scales = []
    for spec, component in zip(selected, cv.cvs):
        unit = (
            spec.get("unit", spec.get("units", component.units))
            if isinstance(spec, dict)
            else component.units
        )
        if isinstance(unit, str):
            unit = [unit] * component.dimension
        if len(unit) != component.dimension:
            raise ValueError("CV input units dimension mismatch")
        scales.extend(
            [
                1.0 if u == "dimensionless" else unit_factor(u, _dimension(c))
                for u, c in zip(unit, component.units)
            ]
        )

    def center(value, atomic=False):
        if cv.dimension == 1 and (isinstance(value, (str, int, float, np.number))):
            value = [value]
        if len(value) != cv.dimension:
            raise ValueError("umbrella center dimension mismatch")
        return [
            parse_quantity(x, dim)
            * (scale if not atomic and isinstance(x, (int, float, np.number)) else 1)
            for x, dim, scale in zip(value, dims, scales)
        ]

    if "centers_au" in umbrella:
        centers = [center(x, True) for x in umbrella["centers_au"]]
    elif "centers" in umbrella:
        centers = [center(x) for x in umbrella["centers"]]
    elif "grid" in umbrella:
        grid = umbrella["grid"]
        if isinstance(grid, dict):
            grid = [grid]
        if len(grid) != cv.dimension:
            raise ValueError("umbrella.grid needs one axis per CV")
        axes = []
        for d, axis in enumerate(grid):
            if isinstance(axis, dict):
                lo = parse_quantity(axis["start"], dims[d]) * (
                    1 if isinstance(axis["start"], str) else scales[d]
                )
                hi = parse_quantity(axis["stop"], dims[d]) * (
                    1 if isinstance(axis["stop"], str) else scales[d]
                )
                count = checked_integer(axis["count"], "grid count")
                axes.append(np.linspace(lo, hi, count))
            else:
                axes.append(
                    [
                        parse_quantity(x, dims[d])
                        * (1 if isinstance(x, str) else scales[d])
                        for x in axis
                    ]
                )
        centers = (
            np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
            .reshape(-1, cv.dimension)
            .tolist()
        )
    elif all(k in umbrella for k in ("start", "stop", "count")) and cv.dimension == 1:
        lo = center(umbrella["start"])[0]
        hi = center(umbrella["stop"])[0]
        centers = np.linspace(
            lo, hi, checked_integer(umbrella["count"], "umbrella.count")
        )[:, None]
    else:
        raise ValueError("specify umbrella.centers or umbrella.grid")
    centers = np.asarray(centers, float)
    if (
        centers.ndim != 2
        or centers.shape[1] != cv.dimension
        or not len(centers)
        or not np.isfinite(centers).all()
    ):
        raise ValueError("invalid umbrella centers")
    raw = umbrella.get("stiffness_au", umbrella.get("stiffness", umbrella.get("kappa")))
    if raw is None:
        raise ValueError("umbrella.stiffness or kappa is required")
    if isinstance(raw, (str, int, float, np.number)):
        raw = [
            [raw if i == j else 0.0 for j in range(cv.dimension)]
            for i in range(cv.dimension)
        ]
    elif np.asarray(raw, dtype=object).ndim == 1:
        if len(raw) != cv.dimension:
            raise ValueError("diagonal stiffness dimension mismatch")
        raw = [
            [raw[i] if i == j else 0.0 for j in range(cv.dimension)]
            for i in range(cv.dimension)
        ]
    if np.shape(raw) != (cv.dimension, cv.dimension):
        raise ValueError("stiffness must be scalar, diagonal, or square matrix")
    stiffness = np.empty((cv.dimension, cv.dimension))
    for i in range(cv.dimension):
        for j in range(cv.dimension):
            dimension = tuple(
                e - a - b for e, a, b in zip(DIMS["energy"], dims[i], dims[j])
            )
            stiffness[i, j] = parse_quantity(raw[i][j], dimension)
    UmbrellaBias(cv, centers[0], stiffness)  # validates positive definiteness
    return cv, selected, centers, stiffness


def steer_to_center(q, cv, target, masses, max_steps=250, tolerance=1e-7):
    """Mass-weighted geometric continuation; no dynamics samples are taken here."""
    q = np.asarray(q, float).copy()
    invmass = 1 / np.asarray(masses)[None, :, None]
    target = np.asarray(target, float)
    for _ in range(max_steps):
        result = cv.value_and_jacobian(q)
        error = displacement(result.value, target, cv.periods)
        norm = np.linalg.norm(error)
        if norm < tolerance:
            return q
        jac = result.jacobian
        metric = np.einsum(
            "ipac,jpac,pac->ij", jac, jac, np.broadcast_to(invmass, q.shape)
        )
        multiplier = np.linalg.lstsq(metric, error, rcond=1e-12)[0]
        direction = -np.einsum("i,ipac->pac", multiplier, jac) * invmass
        maxmove = np.max(np.linalg.norm(direction, axis=-1))
        if maxmove > 0.25:
            direction *= 0.25 / maxmove
        moved = False
        for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
            trial = q + scale * direction
            try:
                newnorm = np.linalg.norm(
                    displacement(cv.value(trial), target, cv.periods)
                )
            except ValueError:
                continue
            if newnorm < norm:
                q = trial
                moved = True
                break
        if not moved:
            break
    residual = float(np.linalg.norm(displacement(cv.value(q), target, cv.periods)))
    if residual > 1e-4:
        raise ValueError(
            f"could not construct umbrella initial geometry; CV residual {residual:g}"
        )
    return q


def separate_reactants_to_xi(geometry, coordinate, target, masses):
    """Construct entrance umbrellas by rigid fragment separation, not bond collapse.

    A least-norm xi projection can lower xi by collapsing the breaking bond
    instead of separating reactants. Here the internal TS geometry of each
    reactant is preserved exactly, then ordinary PES thermalization relaxes
    its internal degrees of freedom. This is initialization, not sampling.
    """
    from scipy.optimize import brentq

    geometry = np.asarray(geometry, float)
    fragments = [np.asarray(f, dtype=int) for f in coordinate.com.fragments]
    masses = np.asarray(masses, float)
    totals = np.array([masses[f].sum() for f in fragments])
    com = [np.average(geometry[f], axis=0, weights=masses[f]) for f in fragments]
    vector = com[0] - com[1]
    distance = float(np.linalg.norm(vector))
    if distance < 1e-12:
        raise ValueError(
            "cannot initialize rate entrance from coincident fragment centers"
        )
    axis = vector / distance

    def translated(delta):
        q = geometry.copy()
        q[fragments[0]] += delta * totals[1] / totals.sum() * axis
        q[fragments[1]] -= delta * totals[0] / totals.sum() * axis
        return q

    def residual(delta):
        return coordinate.centroid_value_gradient(translated(delta))[0] - target

    start = residual(0.0)
    if abs(start) < 1e-10:
        return geometry.copy()
    if start < 0:
        raise ValueError(
            "rigid entrance initialization is only for xi below the initial TS"
        )
    upper = max(1.0, coordinate.rinf - distance + 1.0)
    for _ in range(20):
        if residual(upper) <= 0:
            break
        upper *= 2
    else:
        raise ValueError(
            "could not bracket entrance umbrella by rigid fragment separation"
        )
    shift = brentq(residual, 0.0, upper, xtol=1e-12, rtol=1e-12)
    return translated(shift)


def _sample_window(task):
    config, system = task["config"], task["system"]
    cv = make_cvs(task["specs"], system)
    center = np.asarray(task["center"])
    stiffness = np.asarray(task["stiffness"])
    window, replica = task["window"], task["replica"]
    sampling = config["sampling"]
    neq = checked_integer(
        sampling["equilibration_steps"], "sampling.equilibration_steps", 0
    )
    nprod = checked_integer(sampling["production_steps"], "sampling.production_steps")
    interval = checked_integer(
        sampling.get("sample_interval", sampling.get("interval_steps", 1)),
        "sampling.sample_interval",
    )
    andersen = checked_integer(
        sampling.get("andersen_interval", 10), "sampling.andersen_interval"
    )
    if nprod < interval:
        raise ValueError("production_steps must include at least one saved sample")
    checkpoint_steps = checked_integer(
        config.get("output", {}).get("checkpoint_steps", 1000),
        "output.checkpoint_steps",
    )
    if min(interval, andersen, checkpoint_steps) < 1:
        raise ValueError("sampling/checkpoint intervals must be positive")
    checkpoint = (
        Path(task["output"])
        / "checkpoints"
        / f"window_{window:05d}_replica_{replica:04d}.npz"
    )
    fingerprint = _fingerprint(
        config,
        system,
        {
            "window": window,
            "replica": replica,
            "center": center,
            "stiffness": stiffness,
        },
    )
    seed = np.random.SeedSequence([int(config.get("seed", 0)), 101, window, replica])
    provider = make_task_potential(config["potential"], system)
    try:
        dynamics = Dynamics(
            provider,
            system.masses,
            config["beta"],
            config["beads"],
            sampling.get("dt_au", config["dt_au"]),
            seed=seed,
            native=config.get("native", config.get("dynamics", {}).get("native", True)),
            threads=config.get("parallel", {}).get("threads_per_rank", 1),
            **execution_options(config),
        )
        umbrella = UmbrellaBias(cv, center, stiffness)
        bias = umbrella
        if config.get("analysis", {}).get("measure") == "rate_flux":
            if len(cv.cvs) != 1 or not isinstance(cv.cvs[0], RPMDrateCoordinate):
                raise ValueError(
                    "rate_flux measure requires the centroid RPMDrate coordinate"
                )
            bias = CompositeBias(
                umbrella, FluxMetricBias(cv.cvs[0], system.masses, config["beta"])
            )
        values = []
        energies = []
        elapsed = 0
        if checkpoint.exists():
            with np.load(checkpoint, allow_pickle=False) as saved:
                if str(saved["fingerprint"]) != fingerprint:
                    raise ValueError(
                        f"checkpoint physics differs: {checkpoint}; choose a fresh output directory"
                    )
                state = State(
                    saved["q"], saved["p"], int(saved["step"]), float(saved["time"])
                )
                dynamics.rng.bit_generator.state = json.loads(str(saved["rng"]))
                values = list(saved["values"])
                energies = list(saved["energies"])
                elapsed = int(saved["elapsed"])
        else:
            geometry = system.geometry
            if len(cv.cvs) == 1 and isinstance(cv.cvs[0], RPMDrateCoordinate):
                coordinate = cv.cvs[0]
                permutations = coordinate.permutations
                permutation = permutations[replica % len(permutations)]
                geometry = np.empty_like(system.geometry)
                geometry[list(permutation)] = system.geometry
                if center[0] < coordinate.centroid_value_gradient(geometry)[0] - 1e-10:
                    geometry = separate_reactants_to_xi(
                        geometry, coordinate, center[0], system.masses
                    )
            q = np.repeat(geometry[None], config["beads"], axis=0)
            q = steer_to_center(q, cv, center, system.masses)
            state = State(q, dynamics.sample_momenta())

        def save():
            _atomic_npz(
                checkpoint,
                q=state.q,
                p=state.p,
                step=state.step,
                time=state.time,
                rng=json.dumps(dynamics.rng.bit_generator.state),
                values=np.asarray(values).reshape(-1, cv.dimension),
                energies=np.asarray(energies),
                elapsed=elapsed,
                fingerprint=fingerprint,
            )

        while elapsed < neq + nprod:
            if state.step % andersen == 0:
                state.p[:] = dynamics.sample_momenta()
            count = min(dynamics.block_steps, neq + nprod - elapsed,
                        andersen - state.step % andersen,
                        checkpoint_steps - elapsed % checkpoint_steps)
            production = elapsed + np.arange(1, count + 1) - neq
            observed = (production > 0) & (production % interval == 0)
            block = dynamics.propagate_block(state, count, bias=bias, cv=cv, observe=observed)
            values.extend(block.values)
            energies.extend(block.energies)
            elapsed += block.completed
            if elapsed % checkpoint_steps == 0:
                save()
        save()
        samples = np.asarray(values).reshape(-1, cv.dimension)
        if len(samples) < 2:
            raise ValueError("at least two samples per umbrella stream required")
        return {
            "window": window,
            "replica": replica,
            "samples": samples,
            "center": center,
            "stiffness": stiffness,
            "last_q": state.q,
            "last_p": state.p,
            "energies": np.asarray(energies),
            "checkpoint": str(checkpoint),
            "native": dynamics.native,
            "propagation_this_call": dynamics.loop_stats,
        }
    finally:
        if hasattr(provider, "close"):
            provider.close()


def _write_fes(output, data, streams, cv, config):
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
    arrays.update({f"edges_{i}": edge for i, edge in enumerate(data["edges"])})
    _atomic_npz(path / "fes.npz", **arrays)
    try:
        import h5py
    except ImportError:
        h5py = None
    if h5py is not None:
        tmp = path / "fes.h5.tmp"
        with h5py.File(tmp, "w") as h:
            h.attrs["schema_version"] = 1
            h.attrs["energy_unit"] = "hartree"
            h.attrs["cv_units"] = json.dumps(cv.units)
            h.attrs["measure"] = data["measure"]
            h.attrs["hamiltonian_convention"] = "sum_beads_beta_over_P"
            for name, value in arrays.items():
                h.create_dataset(name, data=value)
            groups = h.create_group("streams")
            for i, stream in enumerate(streams):
                g = groups.create_group(str(i))
                for key in ("samples", "center", "stiffness", "energies"):
                    g.create_dataset(key, data=stream[key])
                g.attrs["window"] = stream["window"]
                g.attrs["replica"] = stream["replica"]
                g.attrs["propagation_this_call"] = json.dumps(stream.get("propagation_this_call", {}))
        os.replace(tmp, path / "fes.h5")
    mid = [0.5 * (e[1:] + e[:-1]) for e in data["edges"]]
    mesh = np.meshgrid(*mid, indexing="ij")
    table = np.column_stack(
        [m.ravel() for m in mesh]
        + [
            data[k].ravel()
            for k in (
                "free_energy",
                "standard_error",
                "probability_mass",
                "density",
                "mask",
                "raw_count",
            )
        ]
    )
    np.savetxt(
        path / "fes.csv",
        table,
        delimiter=",",
        header=",".join(
            [f"cv{i}_{unit}" for i, unit in enumerate(cv.units)]
            + [
                "free_energy_hartree",
                "standard_error_hartree",
                "probability_mass",
                "density",
                "masked",
                "raw_count",
            ]
        ),
        comments="",
    )
    warnings = list(config.get("_warnings", []))
    if not data["connected"]:
        warnings.append(
            "Disconnected umbrella overlap: global FES is unidentifiable and masked."
        )
    if np.any(data["effective_samples_per_stream"] < 20):
        warnings.append(
            "Some streams have fewer than 20 estimated independent samples; increase sampling and replicas."
        )
    if np.any(data["mask"]):
        warnings.append(
            "Unsampled/insufficient-count bins are masked; no extrapolation is reported."
        )
    summary = {
        "task": "fes",
        "status": "sampled" if data["connected"] else "disconnected_overlap",
        "dimension": cv.dimension,
        "units": cv.units,
        "energy_unit": "hartree",
        "measure": data["measure"],
        "estimator": "mbar",
        "windows": len(set(s["window"] for s in streams)),
        "streams": len(streams),
        "propagation_this_call": [s.get("propagation_this_call", {}) for s in streams],
        "samples": sum(len(s["samples"]) for s in streams),
        "overlap_connected": data["connected"],
        "overlap_components": data["components"],
        "weight_effective_samples": data["weight_ess"],
        "effective_samples_per_stream": data["effective_samples_per_stream"],
        "bootstrap_replicates": data["bootstrap_replicates"],
        "warnings": warnings,
        "converged": False,
        "convergence_note": "Overlap and bootstrap diagnose these samples; bead, timestep and sampling convergence require separate runs.",
        "files": {"data": str(path / "fes.npz"), "table": str(path / "fes.csv")},
    }
    if h5py is not None:
        summary["files"]["hdf5"] = str(path / "fes.h5")
    write_json(path / "summary.json", summary)
    return summary


def _analyze_streams(streams, config, system, output, return_data):
    cv, specs, centers, stiffness = resolve_umbrellas(config, system)
    options = config.get("analysis", {})
    if options.get("method", "mbar") not in ("mbar", "umbrella_integration"):
        raise ValueError(
            "supported FES estimator: mbar (umbrella_integration is used by rate)"
        )
    if (
        options.get("method") == "umbrella_integration"
        and config.get("task", "fes") != "rate"
    ):
        raise ValueError(
            "umbrella_integration is the 1D rate PMF estimator; generic FES uses mbar"
        )
    ranges = options.get("ranges")
    if ranges is not None:
        ranges = [
            [parse_quantity(x, _dimension(unit)) for x in limits]
            for limits, unit in zip(ranges, cv.units)
        ]
    data = reconstruct_fes(
        [s["samples"] for s in streams],
        [s["center"] for s in streams],
        [s["stiffness"] for s in streams],
        config["beta"],
        bins=options.get("bins", 40),
        ranges=ranges,
        periods=cv.periods,
        bootstrap=checked_integer(
            options.get(
                "bootstrap",
                options.get(
                    "bootstrap_replicates", options.get("bootstrap_samples", 20)
                ),
            ),
            "analysis.bootstrap",
            0,
        ),
        seed=checked_integer(
            options.get("seed", config.get("seed", 0)), "analysis.seed", 0
        ),
        min_count=checked_integer(options.get("min_count", 1), "analysis.min_count"),
        overlap_threshold=float(options.get("overlap_threshold", 1e-6)),
    )
    data["measure"] = options.get("measure", "cv_density")
    summary = _write_fes(output, data, streams, cv, config)
    if return_data:
        return {"summary": summary, "data": data, "streams": streams}
    return summary


def run_fes(config, system, output, executor=None, return_data=False):
    config = copy.deepcopy(config)
    if config.get("analysis", {}).get("measure", "cv_density") not in (
        "cv_density",
        "rate_flux",
    ):
        raise ValueError("analysis.measure must be cv_density or rate_flux")
    cv, specs, centers, stiffness = resolve_umbrellas(config, system)
    replicas = checked_integer(
        config.get("sampling", {}).get("replicas", 1), "sampling.replicas"
    )
    tasks = [
        {
            "config": config,
            "system": system,
            "specs": specs,
            "window": i,
            "replica": r,
            "center": center,
            "stiffness": stiffness,
            "output": str(output),
        }
        for i, center in enumerate(centers)
        for r in range(replicas)
    ]
    streams = (
        executor.map(_sample_window, tasks)
        if executor is not None
        else run_tasks(_sample_window, tasks, config.get("parallel", {}))
    )
    return root_call(
        _analyze_streams, streams, config, system, str(output), return_data
    )


def reanalyze_fes(config, system, output, return_data=False):
    """Reconstruct with new bins/bootstrap settings without propagating PESs.

    Completed window checkpoint arrays are the authoritative sampled data;
    changing the physical settings is rejected just as for a resumed run.
    All MPI ranks may call this; only the coordinator reads/writes files.
    """

    def analyze():
        cv, specs, centers, stiffness = resolve_umbrellas(config, system)
        replicas = int(config.get("sampling", {}).get("replicas", 1))
        streams = []
        for window, center in enumerate(centers):
            for replica in range(replicas):
                checkpoint = (
                    Path(output)
                    / "checkpoints"
                    / f"window_{window:05d}_replica_{replica:04d}.npz"
                )
                expected = _fingerprint(
                    config,
                    system,
                    {
                        "window": window,
                        "replica": replica,
                        "center": center,
                        "stiffness": stiffness,
                    },
                )
                with np.load(checkpoint, allow_pickle=False) as saved:
                    if str(saved["fingerprint"]) != expected:
                        raise ValueError(f"checkpoint physics differs: {checkpoint}")
                    streams.append(
                        {
                            "window": window,
                            "replica": replica,
                            "center": center,
                            "stiffness": stiffness,
                            "samples": saved["values"],
                            "energies": saved["energies"],
                            "last_q": saved["q"],
                            "last_p": saved["p"],
                            "checkpoint": str(checkpoint),
                        }
                    )
        return _analyze_streams(streams, config, system, str(output), return_data)

    return root_call(analyze)
