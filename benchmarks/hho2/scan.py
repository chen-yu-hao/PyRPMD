"""Reproducible HHOO centroid umbrella scan, requiring an external reactive PES.

No electronic model is supplied or substituted. ``plan`` is executable without
a PES; ``run`` requires a source manifest. Results are finite-confinement,
single-electronic-surface, finite-sampling estimates. A masked-grid connection
is a projected equilibrium diagnostic, not a reaction branching ratio.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from scipy.optimize import minimize

from pyrpmd.analysis.fes import reconstruct_fes
from pyrpmd.bias import UmbrellaBias
from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.constants import (
    AMU_TO_AU, ANGSTROM_TO_AU, BOHR_ANGSTROM, FS_TO_AU,
    KB_AU, KCAL_MOL_TO_AU, AU_ENERGY_KCAL_MOL,
)
from pyrpmd.cv import CVStack, Distance
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.potentials import Potential, PESResult, check_potential, make_potential
from pyrpmd.potentials.base import validate_result
from pyrpmd.system import System
from pyrpmd.workflows.fes import steer_to_center, write_json

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmarks.hho2.channels import classify_centroid, minimax_path

SEED_NAMES = ("reactant", "exchange", "H2_O2", "OH_OH", "H2O_O")
ENTRANCE_IDENTITIES = ("reactant_H0+H1O2", "exchanged_H1+H0O2")
MASSES = np.array([1.00782503223] * 2 + [15.99491461957] * 2) * AMU_TO_AU


def seed_geometry(name):
    """Unoptimized Å templates; labels retain H0 as the incident H atom."""
    angle = np.deg2rad(104.5)
    seeds = {
        "reactant": [[-2.8, 0.5, 0.3], [-0.33, 0.91, 0.0], [0., 0., 0.], [1.33, 0., 0.]],
        "H2_O2": [[0., 2.8, 0.], [0.741, 2.8, 0.], [0., 0., 0.], [1.208, 0., 0.]],
        "OH_OH": [[-0.97, 0., 0.], [3.97, 0.2, 0.], [0., 0., 0.], [3., 0.2, 0.]],
        "H2O_O": [[0.958, 0., 0.], [0.958*np.cos(angle), 0.958*np.sin(angle), 0.], [0., 0., 0.], [0., 0., 3.]],
    }
    seeds["exchange"] = np.asarray(seeds["reactant"])[[1, 0, 2, 3]].tolist()
    if name not in seeds:
        raise ValueError(f"unknown initial geometry family {name}")
    geometry = np.asarray(seeds[name], float) * ANGSTROM_TO_AU
    return geometry - np.average(geometry, axis=0, weights=MASSES)


def plan(profile="pilot", *, temperature=1500., beads=1, dt_fs=0.1, seed=20261030,
         wall_angstrom=6.0, repeats=1):
    profiles = {
        "smoke": (3, 3, 5., 20., 8, 0),
        "pilot": (11, 10, 200., 1000., 30, 20),
        "extended": (19, 18, 500., 5000., 50, 100),
    }
    if profile not in profiles:
        raise ValueError("profile must be smoke, pilot, or extended")
    if (not np.isfinite([temperature, dt_fs, wall_angstrom]).all()
            or min(temperature, dt_fs) <= 0 or wall_angstrom < 5.):
        raise ValueError("temperature/dt must be positive; confinement must be >=5 Å")
    for name, value, minimum in (("beads", beads, 1), ("repeats", repeats, 1), ("seed", seed, 0)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    nx, ny, equil, prod, bins, bootstrap = profiles[profile]
    axes = [np.linspace(.65, 4.5, nx), np.linspace(1.0, 4.5, ny)]
    centers = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 2)
    interval = max(1, round(.5 / dt_fs))
    neq, nprod = round(equil / dt_fs), round(prod / dt_fs)
    if neq < 1 or nprod < 4*interval:
        raise ValueError("timestep is too large for the selected sampling profile")
    streams = int(len(centers) * len(SEED_NAMES) * repeats)
    samples_per_stream = int(nprod // interval)
    total_samples = streams * samples_per_stream
    return {
        "schema": "pyrpmd.hho2.scan.v1", "profile": profile,
        "physical_converged": False, "pes_status": "external_model_required",
        "atomic_order": ["H0_incoming", "H1_in_HO2", "O0", "O1"],
        "temperature_K": float(temperature), "beads": beads, "dt_fs": float(dt_fs),
        "beta": 1 / (KB_AU * temperature), "seed": int(seed),
        "coordinate": "centroid distance", "cv_atoms": [[0, 1], [2, 3]],
        "cv_names": ["r_HH", "r_OO"],
        "centers_angstrom": centers.tolist(),
        "stiffness_kcal_mol_angstrom2": [40., 40.],
        "wall": {"pair_distance_angstrom": float(wall_angstrom),
                 "stiffness_kcal_mol_angstrom2": 20.,
                 "retained_in_target_pmf": True, "applied_to": "every bead, all six pairs"},
        "seed_families": list(SEED_NAMES), "repeats_per_family": repeats,
        "equilibration_steps": neq, "production_steps": nprod,
        "sample_interval": interval, "andersen_interval": max(1, round(5. / dt_fs)),
        "checkpoint_steps": 1000,
        "bins": [bins, bins], "ranges_angstrom": [[.5, 4.75], [.8, 4.75]],
        "bootstrap": bootstrap, "min_count": 10, "min_weighted_bin_ess": 10.,
        "streams": streams,
        "planned_samples_per_stream": samples_per_stream,
        "planned_total_samples": total_samples,
        "dense_reduced_potential_bytes_lower_bound": 8 * streams * total_samples,
        "memory_estimate_scope": "Only one float64 [streams,total_samples] reduced-potential matrix before identical-window combination; actual peak memory is higher (additional MBAR arrays, bootstrap, coordinates, and file loading).",
        "aggregate_bead_steps": int(len(centers) * len(SEED_NAMES) * repeats * (neq+nprod) * beads),
        "interpretation": "P=1 is classical; any P>1 requires bead-number convergence. No automatic rate or branching estimate.",
    }


class ConfinedPotential(Potential):
    """The same translation/rotation/permutation-invariant wall in every window."""
    def __init__(self, potential, wall_angstrom=6., k_kcal_mol_angstrom2=20.):
        if not np.isfinite([wall_angstrom, k_kcal_mol_angstrom2]).all() or min(wall_angstrom, k_kcal_mol_angstrom2) <= 0:
            raise ValueError("confinement radius and stiffness must be finite and positive")
        self.potential = potential
        self.radius = wall_angstrom * ANGSTROM_TO_AU
        self.k = k_kcal_mol_angstrom2 * KCAL_MOL_TO_AU / ANGSTROM_TO_AU**2
        self.metadata = dict(getattr(potential, "metadata", {}), confinement="common_pair_wall")

    def wall(self, q):
        q = np.asarray(q, float)
        if q.ndim != 3 or q.shape[1:] != (4, 3) or not np.isfinite(q).all():
            raise ValueError("HHOO coordinates must be finite [P,4,3]")
        energies, forces = np.zeros(len(q)), np.zeros_like(q)
        for i in range(4):
            for j in range(i+1, 4):
                vector = q[:, i] - q[:, j]
                r = np.linalg.norm(vector, axis=1)
                delta = np.maximum(r - self.radius, 0.)
                energies += .5 * self.k * delta**2
                scalar = np.divide(-self.k * delta, r, out=np.zeros_like(r), where=r>0)
                pairforce = scalar[:, None] * vector
                forces[:, i] += pairforce
                forces[:, j] -= pairforce
        return energies, forces

    def evaluate(self, q):
        result = validate_result(self.potential.evaluate(q), np.shape(q))
        e, f = self.wall(q)
        return PESResult(result.energy + e, result.forces + f)

    def close(self):
        if callable(getattr(self.potential, "close", None)):
            self.potential.close()


def load_model(path):
    """Validate explicit model identity and hash every declared source artifact."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError("H+HO2 reactive PES manifest missing; supply --pes-config. No surrogate is substituted.")
    source_bytes = path.read_bytes()
    model = json.loads(source_bytes)
    if not isinstance(model, dict):
        raise ValueError("PES manifest must be a JSON object")
    if (model.get("atomic_numbers") != [1, 1, 8, 8]
            or type(model.get("multiplicity")) is not int
            or model["multiplicity"] not in (1, 3)):
        raise ValueError("PES manifest must declare HHOO atomic order and singlet or triplet multiplicity")
    if not model.get("reference") or not isinstance(model.get("potential"), dict):
        raise ValueError("PES manifest needs a reference and a PyRPMD potential specification")
    if not isinstance(model.get("artifacts"), list) or not model["artifacts"]:
        raise ValueError("PES manifest must pin source/model artifacts with path and SHA256")
    pinned = set()
    for artifact in model["artifacts"]:
        if (not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str)
                or not artifact["path"] or not isinstance(artifact.get("sha256"), str)
                or re.fullmatch(r"[0-9a-fA-F]{64}", artifact["sha256"]) is None):
            raise ValueError("each PES artifact needs a path and a 64-digit SHA256")
        source = (path.parent / artifact["path"]).resolve()
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != artifact["sha256"].lower():
            raise ValueError(f"missing or checksum-mismatched PES artifact: {source}")
        pinned.add(source)
        artifact["path"] = str(source)
        artifact["sha256"] = artifact["sha256"].lower()
    # Native libraries/factory paths are resolved against the manifest.
    potential = model["potential"]
    provider = potential.get("provider")
    if provider not in ("python", "rpmdrate", "f2py", "cabi", "native", "ase", "torch"):
        raise ValueError("reactive PES must use an explicit external provider; built-in teaching potentials are not H+HO2 models")
    # Pin the entry actually used by this provider, not an unrelated optional
    # manifest/library field that the factory would ignore.
    if provider in ("python", "ase"):
        runtime = potential.get("factory", potential.get("function"))
        runtime = runtime.rsplit(":", 1)[0] if isinstance(runtime, str) and ":" in runtime else None
    elif provider in ("cabi", "native"):
        runtime = potential.get("library")
    elif provider in ("rpmdrate", "f2py"):
        runtime = potential.get("module", potential.get("library"))
    else:
        runtime = potential.get("model")
        if runtime is None and isinstance(potential.get("factory"), str):
            runtime = potential["factory"].rsplit(":", 1)[0]
    if not isinstance(runtime, str) or (path.parent / runtime).resolve() not in pinned:
        raise ValueError("the PES provider's runtime entry file must be listed in artifacts with its SHA256")
    entry_files = []
    for key in ("library", "manifest"):
        if key in potential:
            entry = (path.parent / potential[key]).resolve()
            potential[key] = str(entry)
            entry_files.append(entry)
    for key in ("factory", "function"):
        if key in potential and isinstance(potential[key], str) and ":" in potential[key]:
            filename, name = potential[key].rsplit(":", 1)
            if filename.endswith((".py", ".so")):
                entry = (path.parent / filename).resolve()
                potential[key] = str(entry) + ":" + name
                entry_files.append(entry)
    for key in ("module", "model"):
        if key in potential and isinstance(potential[key], str):
            entry = (path.parent / potential[key]).resolve()
            if entry.is_file():
                potential[key] = str(entry)
                entry_files.append(entry)
    if not entry_files:
        raise ValueError("reactive PES requires a pinned local factory, function, module, or library entry point")
    if any(entry not in pinned for entry in entry_files):
        raise ValueError("every PES runtime entry file must be listed in artifacts with its SHA256")

    def resolve_parameters(value):
        if isinstance(value, dict):
            return {k: resolve_parameters(v) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve_parameters(v) for v in value]
        if isinstance(value, str):
            candidate = (path.parent / value).resolve()
            if candidate in pinned:
                return str(candidate)
        return value

    if "params" in potential:
        potential["params"] = resolve_parameters(potential["params"])
    model["manifest_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    return model


def _fingerprint(settings, model, window, family, repeat):
    return hashlib.sha256(json.dumps(
        [settings, model, window, family, repeat], sort_keys=True, allow_nan=False
    ).encode()).hexdigest()


def _stream_path(output, window, family, repeat):
    return Path(output) / "streams" / f"w{window:04d}_{family}_{repeat:03d}.npz"


def _checked_stream(path, settings, model, window, family, repeat, *, complete=False):
    """Check scientific identity, clocks and saved observations before reuse."""
    with load_checkpoint(path, expected_kind="arrays", require_integrity=True) as saved:
        data = {key: saved[key].copy() for key in saved.files}
    if str(data["fingerprint"]) != _fingerprint(settings, model, window, family, repeat):
        raise ValueError(f"run identity changed for {path}; use a new output directory")
    if (int(data["window"]), str(data["family"]), int(data["repeat"])) != (window, family, repeat):
        raise ValueError(f"stream identity does not match its planned path: {path}")
    total = settings["equilibration_steps"] + settings["production_steps"]
    elapsed = int(data["elapsed"])
    if (elapsed != data["elapsed"] or not 0 <= elapsed <= total
            or int(data["step"]) != elapsed
            or not np.isclose(float(data["time"]), elapsed * settings["dt_fs"] * FS_TO_AU,
                              rtol=1e-9, atol=1e-10)):
        raise ValueError(f"inconsistent stream checkpoint clock: {path}")
    if complete and elapsed != total:
        raise ValueError(f"incomplete stream: {path}")
    n = max(0, elapsed - settings["equilibration_steps"]) // settings["sample_interval"]
    shapes = {"q": (settings["beads"], 4, 3), "p": (settings["beads"], 4, 3),
              "samples": (n, 2), "centroids": (n, 4, 3),
              "physical_energies": (n,), "wall_energies": (n,), "ring_radii": (n,)}
    if any(data[key].shape != shape or not np.isfinite(data[key]).all()
           for key, shape in shapes.items()):
        raise ValueError(f"inconsistent stream state or observation arrays: {path}")
    center = np.asarray(settings["centers_angstrom"][window]) * ANGSTROM_TO_AU
    stiffness = np.diag(settings["stiffness_kcal_mol_angstrom2"]) * KCAL_MOL_TO_AU / ANGSTROM_TO_AU**2
    if not np.array_equal(data["center"], center) or not np.array_equal(data["stiffness"], stiffness):
        raise ValueError(f"stream umbrella differs from its planned Hamiltonian: {path}")
    q = data["centroids"]
    observed = np.column_stack((np.linalg.norm(q[:, 0]-q[:, 1], axis=1),
                                np.linalg.norm(q[:, 2]-q[:, 3], axis=1)))
    if not np.allclose(data["samples"], observed, rtol=1e-11, atol=1e-11):
        raise ValueError(f"saved CVs disagree with the centroid geometries: {path}")
    return data


def _stream(task):
    settings, model, output, window, family, repeat = task
    geometry = seed_geometry(family)
    system = System(("H", "H", "O", "O"), MASSES, geometry, multiplicity=model["multiplicity"])
    potential = ConfinedPotential(make_potential(model["potential"], system),
                                  settings["wall"]["pair_distance_angstrom"],
                                  settings["wall"]["stiffness_kcal_mol_angstrom2"])
    cv = CVStack([Distance([0, 1]), Distance([2, 3])])
    center = np.asarray(settings["centers_angstrom"][window]) * ANGSTROM_TO_AU
    stiffness = np.diag(settings["stiffness_kcal_mol_angstrom2"]) * KCAL_MOL_TO_AU / ANGSTROM_TO_AU**2
    bias = UmbrellaBias(cv, center, stiffness)
    path = _stream_path(output, window, family, repeat)
    path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = _fingerprint(settings, model, window, family, repeat)
    dynamics = Dynamics(potential, MASSES, settings["beta"], settings["beads"], settings["dt_fs"]*FS_TO_AU,
                        seed=np.random.SeedSequence([settings["seed"], window, SEED_NAMES.index(family), repeat]),
                        loop_backend="python", integrator="cayley-bcb")
    records = {k: [] for k in ("samples", "centroids", "physical_energies", "wall_energies", "ring_radii")}
    elapsed = 0
    try:
        if path.exists():
            saved = _checked_stream(path, settings, model, window, family, repeat)
            state = State(saved["q"], saved["p"], int(saved["step"]), float(saved["time"]))
            dynamics.rng.bit_generator.state = json.loads(str(saved["rng"]))
            elapsed = int(saved["elapsed"])
            records = {k: list(saved[k]) for k in records}
        else:
            geometry = steer_to_center(geometry[None], cv, center, MASSES)[0]
            def objective(flat):
                q = flat.reshape(1, 4, 3)
                result = potential.evaluate(q)
                e, f = bias.energy_force(q)
                return float(result.energy[0]+e), -(result.forces+f).reshape(-1)
            relaxed = minimize(objective, geometry.reshape(-1), jac=True, method="L-BFGS-B", options={"maxiter":200})
            if not np.isfinite(relaxed.x).all() or not np.isfinite(relaxed.fun):
                raise ValueError("nonfinite window initialization")
            state = State(np.repeat(relaxed.x.reshape(1, 4, 3), settings["beads"], axis=0), dynamics.sample_momenta())

        def save():
            arrays = {k:np.asarray(v) for k,v in records.items()}
            arrays["samples"] = arrays["samples"].reshape(-1, 2)
            arrays["centroids"] = arrays["centroids"].reshape(-1, 4, 3)
            atomic_savez(path, kind="arrays", fingerprint=fingerprint, center=center, stiffness=stiffness,
                         window=window, family=family, repeat=repeat, elapsed=elapsed,
                         q=state.q, p=state.p, step=state.step, time=state.time,
                         rng=json.dumps(dynamics.rng.bit_generator.state),
                         **arrays)

        total = settings["equilibration_steps"] + settings["production_steps"]
        while elapsed < total:
            if state.step % settings["andersen_interval"] == 0:
                state.p[:] = dynamics.sample_momenta()
            dynamics.step(state, bias)
            elapsed += 1
            production = elapsed - settings["equilibration_steps"]
            if production > 0 and production % settings["sample_interval"] == 0:
                centroid = state.q.mean(0)
                wall_e, _ = potential.wall(state.q)
                records["samples"].append(cv.value(state.q))
                records["centroids"].append(centroid.copy())
                # Keep the underlying PES and retained wall energies separately.
                records["physical_energies"].append(float((potential.evaluate(state.q).energy-wall_e).mean()))
                records["wall_energies"].append(float(wall_e.mean()))
                records["ring_radii"].append(float(np.sqrt(np.mean((state.q-centroid)**2))))
            if elapsed % settings["checkpoint_steps"] == 0:
                save()
        save()
        return str(path)
    finally:
        potential.close()


def _projected_basin_seeds(channels, fractions, free_energy, supported):
    """Group chemically identical entrances only for projected basin selection.

    r_HH and r_OO are unchanged by swapping the two equal-mass H atoms.
    At symmetric equilibrium the two entrance identity fractions are equal,
    so neither can satisfy an 80% threshold even for a pure entrance basin.
    Keep the identity-resolved fractions outside this helper for diagnostics;
    their sum defines H+HO2 here, not a resolved exchange reaction path.
    """
    grouped = {}
    for channel, fraction in zip(channels, fractions):
        if "ambiguous" in channel or "intermediate" in channel:
            continue
        name = "H+HO2" if channel in ENTRANCE_IDENTITIES else str(channel)
        if name not in grouped:
            grouped[name] = np.zeros_like(free_energy, dtype=float)
        grouped[name] += fraction
    basins = {}
    for name, fraction in grouped.items():
        good = supported & np.isfinite(free_energy) & (fraction >= .8)
        if np.any(good):
            index = np.nanargmin(np.where(good, free_energy, np.nan))
            basins[name] = tuple(int(i) for i in np.unravel_index(index, free_energy.shape))
    return basins, grouped


def analyze(output, *, bootstrap=None):
    output = Path(output)
    settings = json.loads((output / "plan.json").read_text())
    model = json.loads((output / "model.json").read_text())
    paths = sorted((output / "streams").glob("w*.npz"))
    identities = [(w, f, r) for w in range(len(settings["centers_angstrom"]))
                  for f in settings["seed_families"] for r in range(settings["repeats_per_family"])]
    expected = {_stream_path(output, *identity): identity for identity in identities}
    if (len(expected) != settings["streams"] or set(paths) != set(expected)
            or len(identities) != len(expected)):
        raise ValueError("not all planned streams are present; refusing to infer missing channel connections")
    samples, centers, stiffness, geometries = [], [], [], []
    for path in paths:
        data = _checked_stream(path, settings, model, *expected[path], complete=True)
        samples.append(data["samples"]); centers.append(data["center"])
        stiffness.append(data["stiffness"]); geometries.extend(data["centroids"])
    result = reconstruct_fes(samples, centers, stiffness, settings["beta"], bins=settings["bins"],
                             ranges=np.array(settings["ranges_angstrom"])*ANGSTROM_TO_AU,
                             bootstrap=settings["bootstrap"] if bootstrap is None else bootstrap,
                             min_count=settings["min_count"], seed=settings["seed"])
    x, edges, weights = np.concatenate(samples), result["edges"], result["weights"]
    # Importance-weight ESS is a second support check; it does not correct time correlation.
    mass = np.histogramdd(x, edges, weights=weights)[0]
    square = np.histogramdd(x, edges, weights=weights**2)[0]
    ess = np.divide(mass**2, square, out=np.zeros_like(mass), where=square>0)
    supported = ~result["mask"] & (ess >= settings["min_weighted_bin_ess"])
    free_energy = np.where(supported, result["free_energy"] * AU_ENERGY_KCAL_MOL, np.nan)
    labels = np.asarray([classify_centroid(q)["channel"] for q in geometries])
    stream_diagnostics = []
    start = 0
    for path, sample in zip(paths, samples):
        local = labels[start:start+len(sample)]
        names, counts = np.unique(local, return_counts=True)
        window, family, repeat = expected[path]
        stream_diagnostics.append({
            "window": window, "initial_family": family, "repeat": repeat,
            "samples": len(local), "channel_counts": dict(zip(names, counts.tolist())),
            "consecutive_centroid_label_changes": int(np.count_nonzero(local[1:] != local[:-1])),
        })
        start += len(sample)
    channels = sorted(set(labels))
    probability = np.array([np.histogramdd(x[labels==label], edges, weights=weights[labels==label])[0]
                            for label in channels])
    fractions = np.divide(probability, mass[None], out=np.zeros_like(probability), where=mass[None]>0)
    fractions[:, ~supported] = np.nan
    # Basin paths use chemical families; original H-identity labels are saved.
    basins, basin_fractions = _projected_basin_seeds(channels, fractions, free_energy, supported)
    connections = []
    for i, (left, start) in enumerate(basins.items()):
        for right, goal in list(basins.items())[i+1:]:
            connections.append({"from":left, "to":right, **minimax_path(free_energy, start, goal, mask=~supported)})
    np.savez_compressed(output / "fes.npz", free_energy_kcal_mol=free_energy, mask=~supported,
                        standard_error_kcal_mol=np.where(supported,result["standard_error"]*AU_ENERGY_KCAL_MOL,np.nan),
                        x_edges_angstrom=edges[0]*BOHR_ANGSTROM, y_edges_angstrom=edges[1]*BOHR_ANGSTROM,
                        weighted_bin_ess=ess, channel_names=np.asarray(channels), channel_fractions=fractions,
                        basin_channel_names=np.asarray(list(basin_fractions)),
                        basin_channel_fractions=np.asarray(list(basin_fractions.values())).reshape((-1,) + free_energy.shape),
                        overlap=result["overlap"], components=result["components"])
    summary = {"schema":"pyrpmd.hho2.analysis.v1", "physical_converged":False,
               "temperature_K":settings["temperature_K"], "beads":settings["beads"],
               "model_reference": model["reference"], "multiplicity": model["multiplicity"],
               "model_manifest_sha256": model["manifest_sha256"],
               "samples":len(x), "streams":len(paths), "overlap_connected":result["connected"],
               "supported_bins":int(supported.sum()), "total_bins":int(supported.size),
               "seed_families":settings["seed_families"], "basins":basins, "projected_connections":connections,
               "basin_channel_groups": {name: list(ENTRANCE_IDENTITIES) if name == "H+HO2" else [name]
                                        for name in basin_fractions},
               "basin_minimum_conditional_fraction": .8,
               "per_stream_channel_diagnostics": stream_diagnostics,
               "label_change_interpretation": "Consecutive stored centroid labels, including ambiguous labels; these changes are not resolved kinetic transitions or proof of mixing.",
               "bootstrap_block_plan":result.get("bootstrap_block_plan"),
               "bootstrap_diagnostics":result.get("bootstrap_diagnostics"),
               "limitations":["Finite common confinement is retained in the target PMF.",
                              "The (r_HH,r_OO) projection is invariant under H-atom exchange: entrance identity labels are retained, but their sum H+HO2 defines one basin and no exchange pathway is resolved.",
                              "Centroid geometric labels and projected connectivity do not establish high-dimensional kinetic connectivity.",
                              "Independent seed, timestep, bead-number, sampling-length, and wall-radius convergence remain required.",
                              "No rate constants or product branching ratios are inferred."]}
    write_json(output / "summary.json", summary)
    render(output)
    return summary


def render(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    with np.load(output / "fes.npz", allow_pickle=False) as data:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
        x, y = data["x_edges_angstrom"], data["y_edges_angstrom"]
        for ax, array, title in zip(axes, [data["free_energy_kcal_mol"], data["standard_error_kcal_mol"], data["weighted_bin_ess"]],
                                   ["Finite-sampling centroid PMF / kcal mol$^{-1}$", "Block-bootstrap SE / kcal mol$^{-1}$", "Importance-weight bin ESS (no time correction)"]):
            im = ax.pcolormesh(x, y, np.where(data["mask"],np.nan,array).T, shading="flat", cmap="viridis")
            ax.set(xlabel=r"$r_{HH}$ / Å", ylabel=r"$r_{OO}$ / Å", title=title)
            fig.colorbar(im, ax=ax)
        fig.savefig(output / "pmf.png", dpi=170)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "check-pes", "run", "analyze"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pes-config", type=Path)
    parser.add_argument("--profile", choices=["smoke", "pilot", "extended"], default="pilot")
    parser.add_argument("--temperature", type=float, default=1500.)
    parser.add_argument("--beads", type=int, default=1)
    parser.add_argument("--dt-fs", type=float, default=.1)
    parser.add_argument("--wall-angstrom", type=float, default=6.)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "analyze":
        print(json.dumps(analyze(args.output), default=lambda x:np.asarray(x).tolist(), indent=2)); return
    settings = plan(args.profile, temperature=args.temperature, beads=args.beads, dt_fs=args.dt_fs,
                    seed=args.seed, wall_angstrom=args.wall_angstrom, repeats=args.repeats)
    if args.command == "plan":
        plan_path = args.output / "plan.json"
        if plan_path.exists() and json.loads(plan_path.read_text()) != settings:
            raise ValueError("output plan differs; use a new run directory")
        write_json(plan_path, settings)
        print(json.dumps({k:settings[k] for k in ("pes_status","streams","aggregate_bead_steps",
                          "planned_total_samples","dense_reduced_potential_bytes_lower_bound",
                          "memory_estimate_scope","temperature_K","beads")},indent=2)); return
    if args.pes_config is None:
        parser.error("--pes-config is required: no validated H+HO2 reactive PES is bundled")
    if args.workers < 1:
        parser.error("--workers must be positive")
    model = load_model(args.pes_config)
    geometry = seed_geometry("reactant")
    system = System(("H","H","O","O"), MASSES, geometry, multiplicity=model["multiplicity"])
    with make_potential(model["potential"], system) as potential:
        points = np.array([seed_geometry(name) for name in SEED_NAMES])
        force_check = check_potential(potential, points)
        if force_check["max_absolute_force_error"] > 1e-5:
            raise ValueError(f"PES force finite-difference check failed: {force_check}")
        energies = validate_result(potential.evaluate(points), points.shape).energy
    if args.command == "check-pes":
        write_json(args.output / "pes-check.json", {"model":model, "force_check":force_check,
                   "unoptimized_seed_energies_hartree":dict(zip(SEED_NAMES,energies)),
                   "scope":"Adapter force consistency only; requires author reference energies and scientific PES validation."}); return
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != settings:
        raise ValueError("output plan differs; use a new run directory")
    model_path = args.output / "model.json"
    if model_path.exists() and json.loads(model_path.read_text()) != model:
        raise ValueError("output model identity differs; use a new run directory")
    write_json(plan_path, settings); write_json(model_path, model)
    tasks = [(settings,model,str(args.output),w,f,r) for w in range(len(settings["centers_angstrom"]))
             for f in settings["seed_families"] for r in range(settings["repeats_per_family"])]
    if args.workers == 1:
        for task in tasks: print(_stream(task), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for path in pool.map(_stream,tasks): print(path,flush=True)
    analyze(args.output)


if __name__ == "__main__":
    main()
