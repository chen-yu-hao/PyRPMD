"""Actual H + BrCl multichannel centroid-PMF regression on a pinned LEPS PES.

Sampling uses the native ring-polymer/PES/bias loop. The common all-pair wall
is retained in the target Hamiltonian. Channel labels and four-neighbor paths
are geometric equilibrium projections, not scattering branching fractions.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyrpmd.analysis.fes import reconstruct_fes, statistical_inefficiency
from pyrpmd.bias import UmbrellaBias
from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.constants import (
    AMU_TO_AU, ANGSTROM_TO_AU, AU_ENERGY_KCAL_MOL, BOHR_ANGSTROM,
    FS_TO_AU, KB_AU, KCAL_MOL_TO_AU,
)
from pyrpmd.cv import CVStack, Distance
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.workflows.fes import write_json
from benchmarks.brhcl.channels import (
    CHANNELS, BondThresholds, classify_centroids, minimax_path, select_channel_basins,
)
from benchmarks.brhcl.pes import BrHClPotential, DEFAULT_LIBRARY

MASSES_AMU = np.array([78.9183376, 1.00782503223, 34.968852682])
MASSES = MASSES_AMU * AMU_TO_AU
FAMILIES = ("compact", "open")


def plan(*, profile="pilot", temperature=2000., beads=1, dt_fs=.1,
         seed=20261030, wall_angstrom=6., equilibration_fs=None,
         production_fs=None, bootstrap=None, orientation_mc_trials=32,
         save_interval_fs=1.):
    profiles = {"smoke": (3, 5., 20., 8, 0), "pilot": (11, 500., 1000., 32, 12)}
    if profile not in profiles:
        raise ValueError("profile must be smoke or pilot")
    side, equil, prod, bins, boot = profiles[profile]
    equil = equil if equilibration_fs is None else equilibration_fs
    prod = prod if production_fs is None else production_fs
    boot = boot if bootstrap is None else bootstrap
    if not np.isfinite([temperature, dt_fs, wall_angstrom, equil, prod, save_interval_fs]).all() or min(temperature, dt_fs, prod, save_interval_fs) <= 0 or equil < 0 or wall_angstrom < 5:
        raise ValueError("invalid temperature, timestep, durations or common wall")
    for name, value, lower in (("beads", beads, 1), ("seed", seed, 0), ("bootstrap", boot, 0), ("orientation_mc_trials", orientation_mc_trials, 0)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < lower:
            raise ValueError(f"{name} must be integer >= {lower}")
    if boot == 1:
        raise ValueError("bootstrap must be 0 or >=2")
    interval, andersen = round(save_interval_fs / dt_fs), round(10. / dt_fs)
    neq, nprod = round(equil / dt_fs), round(prod / dt_fs)
    if (interval < 1 or nprod < 4 * interval or neq % interval or nprod % interval
            or andersen % interval or not np.isclose(interval * dt_fs, save_interval_fs)
            or not np.isclose(andersen * dt_fs, 10.)
            or not np.isclose(neq * dt_fs, equil) or not np.isclose(nprod * dt_fs, prod)):
        raise ValueError("timestep and durations must align to save and 10 fs Andersen intervals")
    axes = [np.linspace(1.15, 4.95, side), np.linspace(1.0, 4.8, side)]
    centers = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 2)
    streams = len(centers) * len(FAMILIES)
    return {
        "schema": "pyrpmd.brhcl.scan.v1", "profile": profile,
        "model": "BrHCl_LEPS1", "atomic_order": ["Br", "H", "Cl"],
        "isotopes": [79, 1, 35], "masses_amu": MASSES_AMU.tolist(), "multiplicity": 2,
        "temperature_K": float(temperature), "beta": 1 / (KB_AU * temperature),
        "beads": int(beads), "dt_fs": float(dt_fs), "seed": int(seed),
        "cv_names": ["r_HBr", "r_HCl"], "cv_pairs": [[0, 1], [1, 2]],
        "centers_angstrom": centers.tolist(), "families": list(FAMILIES),
        "stiffness_kcal_mol_angstrom2": 80.,
        "wall_angstrom": float(wall_angstrom), "wall_k_kcal_mol_angstrom2": 20.,
        "equilibration_steps": int(neq), "production_steps": int(nprod),
        "sample_interval": interval, "andersen_interval": andersen,
        "checkpoint_steps": max(andersen, round(100. / dt_fs)),
        "bins": [bins, bins], "ranges_angstrom": [[.95, 5.25], [.85, 5.1]],
        "bootstrap": int(boot), "min_count": 20, "min_weighted_bin_ess": 20.,
        "streams": streams, "samples_per_stream": nprod // interval,
        "planned_total_samples": streams * (nprod // interval),
        "single_dense_matrix_lower_bound_bytes": 8 * streams**2 * (nprod // interval),
        "aggregate_bead_steps": streams * (neq + nprod) * int(beads),
        "integrator": "cayley-bcb", "loop_backend": "native", "physical_converged": False,
        "orientation_mc_trials": int(orientation_mc_trials),
        "sampler": ("native RPMD propagation plus invariant-measure orientation Metropolis at Andersen refresh; equilibrium samples only"
                    if orientation_mc_trials else "native RPMD propagation with Andersen momentum refresh; equilibrium samples"),
    }


def initial_geometry(center_angstrom, family):
    """Two transverse starting families at exactly the requested distances."""
    x, y = np.asarray(center_angstrom, float)
    if family not in FAMILIES or min(x, y) <= 0:
        raise ValueError("invalid starting family or distances")
    lo, hi = abs(x-y) + .02, x+y - .02
    z = np.clip(2.136 if family == "compact" else .9*(x+y), lo, hi)
    cosine = np.clip((x*x+y*y-z*z)/(2*x*y), -1., 1.)
    q = np.array([[x, 0., 0.], [0., 0., 0.], [y*cosine, y*np.sqrt(1-cosine*cosine), 0.]])
    q -= np.average(q, axis=0, weights=MASSES)
    return q * ANGSTROM_TO_AU


def _fingerprint(settings, library_hash, window, family):
    return hashlib.sha256(json.dumps([settings, library_hash, window, family], sort_keys=True).encode()).hexdigest()


def _path(output, window, family):
    return Path(output) / "streams" / f"w{window:04d}_{family}.npz"


def checked_stream(path, settings, library_hash, window, family, *, complete=False):
    with load_checkpoint(path, expected_kind="arrays", require_integrity=True) as saved:
        data = {key: saved[key].copy() for key in saved}
    required = {"fingerprint", "center", "stiffness", "window", "family", "q", "p",
                "step", "time", "elapsed", "rng", "propagation", "samples", "centroids",
                "ring_radii", "initial_minimized_energy", "maximum_block_energy_drift_kbt"}
    if not required.issubset(data):
        raise ValueError("checkpoint metadata is incomplete")

    def integer(name):
        value = data[name]
        if value.shape != () or value.dtype.kind not in "iu":
            raise ValueError(f"checkpoint {name} must be a scalar integer")
        return int(value)

    def real(name):
        value = data[name]
        if value.shape != () or value.dtype.kind not in "iuf" or not np.isfinite(value):
            raise ValueError(f"checkpoint {name} must be a finite real scalar")
        return float(value)

    def string(name):
        value = data[name]
        if value.shape != () or value.dtype.kind != "U":
            raise ValueError(f"checkpoint {name} must be a scalar string")
        return str(value)

    if string("fingerprint") != _fingerprint(settings, library_hash, window, family):
        raise ValueError("checkpoint scientific identity differs")
    center = np.asarray(settings["centers_angstrom"][window]) * ANGSTROM_TO_AU
    stiffness = np.eye(2) * settings["stiffness_kcal_mol_angstrom2"] * KCAL_MOL_TO_AU / ANGSTROM_TO_AU**2
    if (integer("window") != window or string("family") != family
            or not np.array_equal(data["center"], center)
            or not np.array_equal(data["stiffness"], stiffness)):
        raise ValueError("checkpoint window, family or umbrella metadata differs")
    elapsed = integer("elapsed")
    total = settings["equilibration_steps"] + settings["production_steps"]
    expected_n = max(0, elapsed-settings["equilibration_steps"]) // settings["sample_interval"]
    if not 0 <= elapsed <= total or integer("step") != elapsed or not np.isclose(real("time"), elapsed*settings["dt_fs"]*FS_TO_AU, rtol=1e-10, atol=1e-9):
        raise ValueError("checkpoint clock is inconsistent")
    if complete and elapsed != total:
        raise ValueError("stream is incomplete")
    try:
        propagation = json.loads(string("propagation"))
        rng = json.loads(string("rng"))
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint propagation or RNG metadata is invalid") from error
    if (not isinstance(propagation, dict)
            or type(propagation.get("native_steps")) is not int
            or type(propagation.get("python_steps")) is not int
            or propagation["native_steps"] != elapsed or propagation["python_steps"] != 0
            or propagation.get("fallback_reasons") != {}):
        raise ValueError("checkpoint native propagation counts are inconsistent")
    if not isinstance(rng, dict):
        raise ValueError("checkpoint RNG metadata is invalid")
    trials = settings.get("orientation_mc_trials", 0)
    expected_attempts = trials * ((elapsed + settings["andersen_interval"] - 1) // settings["andersen_interval"])
    if trials or "orientation_mc" in data:
        if "orientation_mc" not in data:
            raise ValueError("checkpoint orientation MC counts are missing")
        try:
            counts = json.loads(string("orientation_mc"))
        except (TypeError, ValueError) as error:
            raise ValueError("checkpoint orientation MC counts are invalid") from error
        if (not isinstance(counts, dict)
                or any(type(counts.get(key)) is not int or counts[key] < 0
                       for key in ("attempted", "accepted", "invalid_domain"))
                or counts["attempted"] != expected_attempts
                or counts["accepted"] + counts["invalid_domain"] > counts["attempted"]):
            raise ValueError("checkpoint orientation MC counts are inconsistent")
    if data["q"].shape != (settings["beads"], 3, 3) or data["p"].shape != data["q"].shape or data["samples"].shape != (expected_n, 2) or data["centroids"].shape != (expected_n, 3, 3):
        raise ValueError("checkpoint state or sample shapes differ")
    for key in ("q", "p", "samples", "centroids", "ring_radii"):
        if data[key].dtype.kind not in "iuf" or not np.isfinite(data[key]).all():
            raise ValueError("nonfinite checkpoint state")
    if data["ring_radii"].ndim != 1 or np.any(data["ring_radii"] < 0):
        raise ValueError("checkpoint ring-radius diagnostics are invalid")
    real("initial_minimized_energy")
    if real("maximum_block_energy_drift_kbt") < 0:
        raise ValueError("checkpoint energy-drift diagnostic is negative")
    distances = np.linalg.norm(data["centroids"][:, [0, 1]] - data["centroids"][:, [1, 2]], axis=2)
    if not np.allclose(distances, data["samples"], rtol=1e-10, atol=1e-10):
        raise ValueError("saved CVs disagree with centroid geometry")
    return data


def sample_stream(task):
    settings, library, library_hash, output, window, family = task
    if hashlib.sha256(Path(library).read_bytes()).hexdigest() != library_hash:
        raise ValueError("PES library changed during the run")
    potential = BrHClPotential(library=library, wall_angstrom=settings["wall_angstrom"],
                             wall_k_kcal_mol_angstrom2=settings["wall_k_kcal_mol_angstrom2"])
    cv = CVStack([Distance([0, 1]), Distance([1, 2])])
    center = np.asarray(settings["centers_angstrom"][window]) * ANGSTROM_TO_AU
    stiffness = np.eye(2) * settings["stiffness_kcal_mol_angstrom2"] * KCAL_MOL_TO_AU / ANGSTROM_TO_AU**2
    bias = UmbrellaBias(cv, center, stiffness)
    dynamics = Dynamics(potential, MASSES, settings["beta"], settings["beads"], settings["dt_fs"]*FS_TO_AU,
                        seed=np.random.SeedSequence([settings["seed"], window, FAMILIES.index(family)]),
                        native=True, threads=1, integrator="cayley-bcb", loop_backend="native")
    path = _path(output, window, family)
    path.parent.mkdir(parents=True, exist_ok=True)
    elapsed, values, centroids, radii, max_drift, initial_energy = 0, [], [], [], 0., None
    mc_counts = {key:0 for key in ("attempted", "accepted", "invalid_domain")}
    try:
        if path.exists():
            saved = checked_stream(path, settings, library_hash, window, family)
            state = State(saved["q"], saved["p"], int(saved["step"]), float(saved["time"]))
            dynamics.rng.bit_generator.state = json.loads(str(saved["rng"]))
            elapsed = int(saved["elapsed"])
            values, centroids, radii = list(saved["samples"]), list(saved["centroids"]), list(saved["ring_radii"])
            max_drift = float(saved["maximum_block_energy_drift_kbt"])
            initial_energy = float(saved["initial_minimized_energy"])
            dynamics.propagation_stats = json.loads(str(saved["propagation"]))
            if "orientation_mc" in saved:
                mc_counts = json.loads(str(saved["orientation_mc"]))
        else:
            q = initial_geometry(settings["centers_angstrom"][window], family)
            def objective(flat):
                geom = flat.reshape(1, 3, 3)
                result = potential.evaluate(geom)
                energy, force = bias.energy_force(geom)
                return float(result.energy[0]+energy), -(result.forces+force).ravel()
            relaxed = minimize(objective, q.ravel(), jac=True, method="L-BFGS-B", options={"maxiter":300})
            if not np.isfinite(relaxed.fun) or not np.isfinite(relaxed.x).all():
                raise ValueError("nonfinite window initialization")
            initial_energy = float(relaxed.fun)
            state = State(np.repeat(relaxed.x.reshape(1, 3, 3), settings["beads"], axis=0), dynamics.sample_momenta())

        def save():
            atomic_savez(path, kind="arrays", fingerprint=_fingerprint(settings, library_hash, window, family),
                         center=center, stiffness=stiffness, window=window, family=family,
                         q=state.q, p=state.p, step=state.step, time=state.time, elapsed=elapsed,
                         rng=json.dumps(dynamics.rng.bit_generator.state),
                         propagation=json.dumps(dynamics.propagation_stats),
                         orientation_mc=json.dumps(mc_counts),
                         samples=np.asarray(values).reshape(-1, 2), centroids=np.asarray(centroids).reshape(-1, 3, 3),
                         ring_radii=np.asarray(radii), initial_minimized_energy=initial_energy,
                         maximum_block_energy_drift_kbt=max_drift)

        total = settings["equilibration_steps"] + settings["production_steps"]
        while elapsed < total:
            if state.step % settings["andersen_interval"] == 0:
                if settings.get("orientation_mc_trials", 0):
                    from benchmarks.brhcl.sampling import orientational_metropolis
                    moved, diagnostics = orientational_metropolis(
                        potential, state.q, settings["beta"], dynamics.rng,
                        trials=settings["orientation_mc_trials"], atom=2, anchor=1)
                    state.q[:] = moved
                    for key in mc_counts:
                        mc_counts[key] += int(diagnostics[key])
                state.p[:] = dynamics.sample_momenta()
            count = min(total-elapsed, settings["andersen_interval"]-state.step % settings["andersen_interval"],
                        settings["checkpoint_steps"]-elapsed % settings["checkpoint_steps"])
            if elapsed < settings["equilibration_steps"]:
                count = min(count, settings["equilibration_steps"]-elapsed)
            block = dynamics.propagate(state, count, bias=bias, observe=cv,
                                       sample_interval=settings["sample_interval"], record_centroids=True)
            if block.backend != "native":
                raise RuntimeError("benchmark requires the fused native propagation loop")
            elapsed += count
            keep = (block.steps > settings["equilibration_steps"]) & ((block.steps-settings["equilibration_steps"]) % settings["sample_interval"] == 0)
            values.extend(block.values[keep]); centroids.extend(block.centroids[keep])
            if elapsed > settings["equilibration_steps"]:
                radii.append(float(np.sqrt(np.mean((state.q-state.q.mean(0))**2))))
                max_drift = max(max_drift, settings["beta"]*max(abs(block.minimum_energy-block.initial_energy),
                    abs(block.maximum_energy-block.initial_energy))/settings["beads"])
            if elapsed % settings["checkpoint_steps"] == 0:
                save()
        save()
        return {"window":window, "family":family, "samples":len(values), "native_steps":dynamics.propagation_stats["native_steps"]}
    finally:
        potential.close()


def run(output, settings, *, library=DEFAULT_LIBRARY, workers=4):
    output, library = Path(output), Path(library).resolve()
    if not library.is_file():
        raise FileNotFoundError("build the source-pinned BrHCl PES before sampling")
    library_hash = hashlib.sha256(library.read_bytes()).hexdigest()
    if (output / "plan.json").exists() and json.loads((output/"plan.json").read_text()) != settings:
        raise ValueError("output contains a different scientific plan; choose another directory")
    if (output / "model.json").exists() and json.loads((output/"model.json").read_text())["library_sha256"] != library_hash:
        raise ValueError("output uses a different PES library")
    potential = BrHClPotential(library=library)
    try:
        model = {"library":str(library), "library_sha256":library_hash, "metadata":potential.metadata}
    finally:
        potential.close()
    write_json(output/"plan.json", settings); write_json(output/"model.json", model)
    tasks = [(settings, str(library), library_hash, str(output), w, family)
             for w in range(len(settings["centers_angstrom"])) for family in FAMILIES]
    start = time.perf_counter()
    if workers == 1:
        for task in tasks:
            sample_stream(task)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            futures = [pool.submit(sample_stream, task) for task in tasks]
            for n, future in enumerate(as_completed(futures), 1):
                future.result()
                if n % max(1, len(tasks)//10) == 0 or n == len(tasks):
                    print(json.dumps({"completed_streams":n,"total_streams":len(tasks),"seconds":time.perf_counter()-start}), flush=True)
    write_json(output/"sampling.json", {"seconds":time.perf_counter()-start,"streams":len(tasks),"workers":workers,
                                       "note":"Includes checkpoint reuse on resume; not necessarily fresh sampling time."})


def _fit(settings, streams, *, bootstrap):
    transverse = []
    for stream in streams:
        q = stream["centroids"]
        a, b = q[:,0]-q[:,1], q[:,2]-q[:,1]
        z = np.linalg.norm(a-b,axis=1)
        cosine = np.clip(np.sum(a*b,axis=1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)),-1.,1.)
        transverse.append(np.column_stack((z,cosine)))
    result = reconstruct_fes([s["samples"] for s in streams], [s["center"] for s in streams],
        [s["stiffness"] for s in streams], settings["beta"], bins=settings["bins"],
        ranges=np.asarray(settings["ranges_angstrom"])*ANGSTROM_TO_AU, bootstrap=bootstrap,
        min_count=settings["min_count"], seed=settings["seed"]+901,
        correlation_observables=transverse)
    samples = np.concatenate([s["samples"] for s in streams])
    square = np.histogramdd(samples, result["edges"], weights=result["weights"]**2)[0]
    mass = result["probability_mass"]
    ess = np.divide(mass**2, square, out=np.zeros_like(mass), where=square>0)
    result["supported"] = (~result["mask"]) & (ess>=settings["min_weighted_bin_ess"])
    result["bin_weight_ess"] = ess
    return result


def analyze(output, *, bootstrap=None, reference=False, reference_cache=None):
    output = Path(output)
    settings = json.loads((output/"plan.json").read_text())
    model = json.loads((output/"model.json").read_text())
    identities = [(w, f) for w in range(len(settings["centers_angstrom"])) for f in FAMILIES]
    expected = {_path(output, w, f) for w,f in identities}
    if set((output/"streams").glob("*.npz")) != expected:
        raise ValueError("planned stream collection is incomplete or contains foreign files")
    streams = [checked_stream(_path(output,w,f),settings,model["library_sha256"],w,f,complete=True) for w,f in identities]
    result = _fit(settings, streams, bootstrap=settings["bootstrap"] if bootstrap is None else bootstrap)
    x = np.concatenate([s["samples"] for s in streams])
    geometry = np.concatenate([s["centroids"] for s in streams])
    labels = classify_centroids(geometry)
    mass, weights, supported = result["probability_mass"], result["weights"], result["supported"]
    fractions = np.asarray([np.histogramdd(x[labels==name], result["edges"], weights=weights[labels==name])[0] for name in CHANNELS])
    np.divide(fractions, mass[None], out=fractions, where=mass[None]>0)
    fractions[:,~supported] = np.nan
    energy = np.where(supported, result["free_energy"]*AU_ENERGY_KCAL_MOL, np.nan)
    basins = select_channel_basins(CHANNELS, fractions, energy, supported)
    connections=[]
    for i,(left,start) in enumerate(basins.items()):
        for right,goal in list(basins.items())[i+1:]:
            connections.append({"from":left,"to":right,**minimax_path(energy,start,goal,mask=~supported)})
    diagnostics=[]
    offset=0
    for (window,family),s in zip(identities,streams):
        local=labels[offset:offset+len(s["samples"])]; offset+=len(local)
        z=np.linalg.norm(s["centroids"][:,0]-s["centroids"][:,2],axis=1)
        diagnostics.append({"window":window,"family":family,"samples":len(local),
            "channel_counts":{name:int(np.count_nonzero(local==name)) for name in CHANNELS},
            "adjacent_label_changes":int(np.count_nonzero(local[1:]!=local[:-1])),
            "third_distance_statistical_inefficiency":float(statistical_inefficiency(z)),
            "cv_statistical_inefficiency":float(statistical_inefficiency(s["samples"])),
            "native_steps":json.loads(str(s["propagation"]))["native_steps"],
            "orientation_mc":json.loads(str(s["orientation_mc"])) if "orientation_mc" in s else {},
            "maximum_block_energy_drift_kbt":float(s["maximum_block_energy_drift_kbt"])})
    arrays={"free_energy_kcal_mol":energy,"mask":~supported,"raw_count":result["raw_count"],
            "standard_error_kcal_mol":np.where(supported,result["standard_error"]*AU_ENERGY_KCAL_MOL,np.nan),
            "bootstrap_valid_per_bin":result["bootstrap_valid_per_bin"],
            "weighted_bin_ess":result["bin_weight_ess"],"x_edges_angstrom":result["edges"][0]*BOHR_ANGSTROM,
            "y_edges_angstrom":result["edges"][1]*BOHR_ANGSTROM,"channel_names":np.asarray(CHANNELS),
            "channel_fractions":fractions,"overlap":result["overlap"],"components":result["components"]}
    comparison={}
    for family in FAMILIES:
        selected=[s for identity,s in zip(identities,streams) if identity[1]==family]
        part=_fit(settings, selected, bootstrap=0)
        arrays[f"free_energy_{family}_kcal_mol"]=np.where(part["supported"],part["free_energy"]*AU_ENERGY_KCAL_MOL,np.nan)
        comparison[family]={"connected":part["connected"],"supported_bins":int(part["supported"].sum())}
    common=np.isfinite(arrays["free_energy_compact_kcal_mol"])&np.isfinite(arrays["free_energy_open_kcal_mol"])
    if common.any():
        delta=(arrays["free_energy_compact_kcal_mol"]-arrays["free_energy_open_kcal_mol"])[common]
        delta-=delta.mean()
        comparison["gauge_aligned_rms_kbt"]=float(np.sqrt(np.mean(delta**2))*settings["beta"]/AU_ENERGY_KCAL_MOL)
        comparison["common_supported_bins"]=int(common.sum())
    threshold_checks=[]
    original=asdict(BondThresholds())
    for scale in (.95,1.05):
        threshold=BondThresholds(**{k:v*(1 if k=="minimum_distance" else scale) for k,v in original.items()})
        other=classify_centroids(geometry,thresholds=threshold)
        p=np.asarray([np.histogramdd(x[other==name],result["edges"],weights=weights[other==name])[0] for name in CHANNELS])
        np.divide(p,mass[None],out=p,where=mass[None]>0);p[:,~supported]=np.nan
        selected=select_channel_basins(CHANNELS,p,energy,supported)
        threshold_checks.append({"scale":scale,"basins":selected,"raw_label_agreement":float(np.mean(other==labels))})
    summary={"schema":"pyrpmd.brhcl.analysis.v1","model":"BrHCl_LEPS1","physical_converged":False,
             "temperature_K":settings["temperature_K"],"beads":settings["beads"],"timestep_fs":settings["dt_fs"],
             "streams":len(streams),"samples":len(x),"overlap_connected":result["connected"],
             "supported_bins":int(supported.sum()),"total_bins":supported.size,
             "effective_samples_minimum":float(result["effective_samples_per_stream"].min()),
             "effective_samples_median":float(np.median(result["effective_samples_per_stream"])),
             "basins":basins,"projected_connections":connections,"initial_family_comparison":comparison,
             "threshold_sensitivity":threshold_checks,"per_stream":diagnostics,
             "bootstrap_block_plan":result["bootstrap_block_plan"],
             "bootstrap_requested":result["bootstrap_requested"],
             "bootstrap_replicates":result["bootstrap_replicates"],
             "bootstrap_diagnostics":result["bootstrap_diagnostics"],
             "bootstrap_correlation_observables":"max g over r_HBr, r_HCl, r_BrCl and cos(Br-H-Cl); each complete stream is resampled in contiguous circular blocks",
             "maximum_third_to_cv_inefficiency_ratio":float(max(
                 d["third_distance_statistical_inefficiency"]/d["cv_statistical_inefficiency"]
                 for d in diagnostics)),
             "maximum_block_energy_drift_kbt":max(d["maximum_block_energy_drift_kbt"] for d in diagnostics),
             "ring_radius_median_bohr":float(np.median(np.concatenate([s["ring_radii"] for s in streams]))),
             "common_wall_retained":True,"channel_thresholds_angstrom":original,
             "orientation_mc_trials_per_refresh":settings.get("orientation_mc_trials",0),
             "sampler_interpretation":"Equilibrium ring-polymer samples; orientation Metropolis moves preclude interpreting the stored sequence as real-time reaction dynamics.",
             "limitations":["Historical single-surface semiempirical LEPS model, not quantitatively validated modern reaction chemistry.",
                "Finite common confinement is retained; no infinite-volume thermodynamic normalization.",
                "Centroid labels and supported 2D paths do not establish scattering branching fractions or rate constants.",
                "Adjacent label changes include ambiguous configurations and are not committed reaction events.",
                "Weight ESS is not time-correlation corrected. CV correlations do not establish hidden-channel mixing.",
                "No automatic claim of bead, timestep, wall or statistical convergence."]}
    if reference or reference_cache is not None:
        if settings["beads"]!=1:
            raise ValueError("classical quadrature is only a P=1 reference")
        if hashlib.sha256(Path(model["library"]).read_bytes()).hexdigest()!=model["library_sha256"]:
            raise ValueError("classical reference PES no longer matches the sampled model")
        if reference_cache is not None:
            from benchmarks.brhcl.reference import load_reference_cache
            exact=load_reference_cache(reference_cache, settings=settings,
                                       library_sha256=model["library_sha256"], edges=result["edges"])
        else:
            from benchmarks.brhcl.reference import classical_pmf
            potential=BrHClPotential(library=model["library"])
            try:
                exact=classical_pmf(potential,*result["edges"],settings["beta"],wall_angstrom=settings["wall_angstrom"],
                                    wall_k_kcal_mol_angstrom2=settings["wall_k_kcal_mol_angstrom2"])
            finally:
                potential.close()
        arrays["classical_reference_kcal_mol"]=exact["free_energy"]*AU_ENERGY_KCAL_MOL
        arrays["quadrature_error_kbt"]=exact["quadrature_error_kbt"]
        arrays["quadrature_converged"]=exact["quadrature_converged"]
        mask=supported&exact["quadrature_converged"]
        reference_summary={k:v for k,v in exact.items() if not isinstance(v,np.ndarray)}
        reference_summary["comparison_bins"]=int(mask.sum())
        if mask.any():
            delta=(result["free_energy"]-exact["free_energy"])[mask]*settings["beta"]
            offset=float(delta.mean());delta-=offset
            reference_summary.update({"gauge_aligned_rms_kbt":float(np.sqrt(np.mean(delta**2))),
                "median_absolute_error_kbt":float(np.median(np.abs(delta))),"maximum_absolute_error_kbt":float(np.max(np.abs(delta))),
                "gauge_offset_kbt":offset})
            arrays["classical_reference_aligned_kcal_mol"]=(exact["free_energy"]+offset/settings["beta"])*AU_ENERGY_KCAL_MOL
        summary["classical_reference"]=reference_summary
        write_json(output/"reference-summary.json",reference_summary)
    atomic_savez(output/"fes.npz",kind="arrays",**arrays)
    write_json(output/"summary.json",summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("plan","run","analyze"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--profile",choices=("smoke","pilot"),default="pilot")
    parser.add_argument("--temperature",type=float,default=2000.)
    parser.add_argument("--beads",type=int,default=1)
    parser.add_argument("--dt-fs",type=float,default=.1)
    parser.add_argument("--seed",type=int,default=20261030)
    parser.add_argument("--wall-angstrom",type=float,default=6.)
    parser.add_argument("--equilibration-fs",type=float)
    parser.add_argument("--production-fs",type=float)
    parser.add_argument("--bootstrap",type=int)
    parser.add_argument("--orientation-mc-trials",type=int,default=32)
    parser.add_argument("--save-interval-fs",type=float,default=1.)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--library",type=Path,default=DEFAULT_LIBRARY)
    parser.add_argument("--reference",action="store_true")
    parser.add_argument("--reference-cache",type=Path)
    args=parser.parse_args()
    if args.command=="analyze":
        summary=analyze(args.output,bootstrap=args.bootstrap,reference=args.reference,
                        reference_cache=args.reference_cache)
        print(json.dumps({k:v for k,v in summary.items() if k not in ("per_stream","bootstrap_block_plan")},default=str,indent=2))
        return
    settings=plan(profile=args.profile,temperature=args.temperature,beads=args.beads,dt_fs=args.dt_fs,
                  seed=args.seed,wall_angstrom=args.wall_angstrom,equilibration_fs=args.equilibration_fs,
                  production_fs=args.production_fs,bootstrap=args.bootstrap,
                  orientation_mc_trials=args.orientation_mc_trials,save_interval_fs=args.save_interval_fs)
    if args.command=="plan":
        if (args.output/"plan.json").exists() and json.loads((args.output/"plan.json").read_text())!=settings:
            raise ValueError("cannot overwrite a different existing plan")
        write_json(args.output/"plan.json",settings)
        print(json.dumps({k:settings[k] for k in ("streams","planned_total_samples","single_dense_matrix_lower_bound_bytes","aggregate_bead_steps","beads","temperature_K")},indent=2))
    else:
        if args.workers<1:
            raise ValueError("workers must be positive")
        run(args.output,settings,library=args.library,workers=args.workers)


if __name__=="__main__":
    main()
