"""Reproducible multichain study of the analytic constant-potential NRPMD model.

The full preset spans voltage and bead count, learns a path separately for each
parameter point, and launches independent production chains on its frozen
windows. Uncertainty resamples original chain blocks and refits MBAR. Matched
parents are propagated at three timesteps for exactly the same physical time.
This is an analytical-model convergence study, not an electrode rate prediction.

Run ``python examples/electrochemistry/adaptive_nrpmd_study.py --plan`` first.
Completed parameter points can be resumed with ``--resume``; changing either
the configuration or relevant implementation source invalidates that cache.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

import adaptive_constant_potential as example
import pyrpmd.adaptive_nrpmd as adaptive_module
import pyrpmd.thermal_nrpmd as thermal_module
from pyrpmd.adaptive_nrpmd import AdaptivePathOptions, reweight_thermal_umbrellas, run_adaptive_nrpmd_path
from pyrpmd.analysis.fes import solve_mbar, umbrella_reduced_potentials
from pyrpmd.bias import UmbrellaBias
from pyrpmd.cv import Cartesian, CVStack
from pyrpmd.electrochemistry import (
    AdiabaticElectrodePotential, ConstantPotentialDiabaticPotential, ElectronReservoir,
)
from pyrpmd.thermal_nrpmd import ThermalNRPMDDynamics, ThermalNRPMDSampler


OBSERVABLE_NAMES = (
    "x", "y", "x2_bead", "y2_bead", "xy_bead", "x2_centroid", "y2_centroid",
    "xy_centroid", "population_0", "population_1", "electron_number_bead",
    "radius_gyration_squared_xy",
)


@dataclass(frozen=True)
class StudyConfig:
    potentials_V: tuple = (-0.6, 0.0, 0.6)
    beads: tuple = (1, 2, 4)
    production_chains: int = 3
    images: int = 5
    pilot_iterations: int = 3
    pilot_samples: int = 96
    pilot_burn_in: int = 96
    pilot_stride: int = 2
    production_samples: int = 512
    production_burn_in: int = 256
    production_stride: int = 2
    bootstrap_replicates: int = 64
    bootstrap_block_multiplier: float = 2.0
    dynamics_parents: int = 16
    dynamics_dts: tuple = (0.02, 0.01, 0.005)
    dynamics_duration: float = 2.0
    min_average_sign: float = 0.1
    min_effective_samples: float = 20.0
    overlap_threshold: float = 0.01
    beta: float = 3.0
    mass: float = 1.5
    stiffness: float = 5.0
    seed: int = 20261101

    def __post_init__(self):
        for name in ("production_chains", "images", "pilot_iterations", "pilot_samples",
                     "pilot_stride", "production_samples", "production_stride",
                     "bootstrap_replicates", "dynamics_parents"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("pilot_burn_in", "production_burn_in", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.images < 3:
            raise ValueError("at least three path images are required")
        for name in ("bootstrap_block_multiplier", "dynamics_duration", "min_average_sign",
                     "min_effective_samples", "overlap_threshold", "beta", "mass", "stiffness"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.min_average_sign > 1 or self.overlap_threshold >= 1:
            raise ValueError("invalid sign or overlap threshold")
        voltage = tuple(float(v) for v in self.potentials_V)
        if not voltage or not np.isfinite(voltage).all() or len(set(voltage)) != len(voltage):
            raise ValueError("potentials_V must contain unique finite voltages")
        beads = tuple(self.beads)
        if (not beads or len(set(beads)) != len(beads)
                or any(isinstance(p, bool) or not isinstance(p, (int, np.integer)) or p < 1 for p in beads)):
            raise ValueError("beads must contain unique positive integers")
        dts = tuple(float(dt) for dt in self.dynamics_dts)
        if (len(dts) < 2 or not np.isfinite(dts).all() or min(dts) <= 0
                or any(a <= b for a, b in zip(dts, dts[1:]))):
            raise ValueError("dynamics_dts must be strictly decreasing positive timesteps")
        for dt in dts:
            for value in (self.dynamics_duration/dt, dts[0]/dt):
                if not np.isclose(value, round(value), atol=1e-10, rtol=0):
                    raise ValueError("timesteps must divide duration and the common observation interval")
        object.__setattr__(self, "potentials_V", voltage)
        object.__setattr__(self, "beads", beads)
        object.__setattr__(self, "dynamics_dts", dts)


def preset_config(name):
    if name == "full":
        return StudyConfig()
    if name == "smoke":
        return StudyConfig(
            potentials_V=(0.0,), beads=(1,), production_chains=2,
            pilot_iterations=1, pilot_samples=24, pilot_burn_in=16,
            production_samples=32, production_burn_in=24, production_stride=1,
            bootstrap_replicates=8, dynamics_parents=4,
            dynamics_dts=(0.02, 0.01), dynamics_duration=0.1,
            min_effective_samples=2.0,
        )
    raise ValueError("preset must be smoke or full")


def _jsonable(value):
    return example._jsonable(value)


def _canonical(value):
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def source_identity():
    """Hash the sources that define this study, its model and sampling measure."""
    root = Path(adaptive_module.__file__).resolve().parents[2]
    paths = [Path(__file__).resolve(), Path(example.__file__).resolve()]
    paths.extend(root / "src/pyrpmd" / p for p in (
        "adaptive_nrpmd.py", "thermal_nrpmd.py", "nonadiabatic.py", "dynamics.py",
        "bias.py", "cv.py", "analysis/fes.py", "electrochemistry/nonadiabatic.py",
        "electrochemistry/potentials.py", "electrochemistry/reservoir.py",
    ))
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return {"sha256": hashlib.sha256(_canonical(hashes).encode()).hexdigest(), "files": hashes}


def plan(config):
    if not isinstance(config, StudyConfig):
        raise TypeError("config must be StudyConfig")
    points = len(config.potentials_V)*len(config.beads)
    pilot = config.pilot_iterations*config.images*(config.pilot_burn_in + config.pilot_samples*config.pilot_stride)
    production = config.production_chains*config.images*(config.production_burn_in + config.production_samples*config.production_stride)
    return {"config": asdict(config), "parameter_points": points,
            "production_streams_per_point": config.production_chains*config.images,
            "production_samples_per_point": config.production_chains*config.images*config.production_samples,
            "maximum_pilot_sweeps_per_point": pilot,
            "production_sweeps_per_point": production,
            "bootstrap_scope": "independent source chains and circular source blocks; MBAR refitted; conditional on frozen path",
            "dynamics_scope": "identical complete initial states at each timestep; same physical final time",
            "source_identity": source_identity(), "physical_converged": False}


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def make_model(potential_V):
    reservoir = ElectronReservoir(
        potential_V, 0.0, 0.1, "analytic-model-reference; no SHE/RHE calibration",
    )
    electrode = AdiabaticElectrodePotential(
        example.CommonChargingModel(), reservoir, electron_bounds=(-5., 5.),
        chemical_potential_tolerance_au=1e-10,
    )
    potential = ConstantPotentialDiabaticPotential(example.CurvedTwoStateModel(), electrode)
    cv = CVStack([Cartesian(0, 0, name="x"), Cartesian(0, 1, name="y")])
    return potential, reservoir, cv


def source_observables(ensemble, mu):
    q = np.array([state.q for state in ensemble.states])[:, :, 0, :2]
    centroid = q.mean(axis=1)
    bead_second = (q*q).mean(axis=1)
    spread = ((q-centroid[:, None])**2).sum(axis=2).mean(axis=1)
    return np.column_stack((
        centroid, bead_second, (q[:, :, 0]*q[:, :, 1]).mean(axis=1),
        centroid**2, centroid[:, 0]*centroid[:, 1], ensemble.electronic_populations,
        0.3*centroid[:, 0]+0.08*centroid[:, 1]+1.2*mu, spread,
    ))


def source_block_bootstrap(cv_values, signs, observables, centers, stiffness, beta,
                           chain_ids, window_ids, *, replicates, seed, block_lengths,
                           min_average_sign=0.1, overlap_threshold=0.01,
                           minimum_valid_fraction=1.0):
    """Hierarchical chain/circular-block resampling, refitting MBAR every draw.

    Each original (chain, window) remains an individual time series. Chains are
    independently resampled within each umbrella, and circular blocks are drawn
    within the chosen chain. Confidence intervals require all requested draws
    to resolve positive signed normalization and at least eight blocks/stream.
    They are conditional on the selected finite block lengths and frozen path.
    """
    values, signs, observables = map(np.asarray, (cv_values, signs, observables))
    chain_ids, window_ids = map(np.asarray, (chain_ids, window_ids))
    centers = np.asarray(centers, dtype=float)
    if (values.ndim != 2 or not len(values) or signs.shape != (len(values),) or observables.ndim != 2
            or len(observables) != len(values) or chain_ids.shape != signs.shape
            or window_ids.shape != signs.shape or not np.all(np.isin(signs, (-1., 1.)))
            or centers.ndim != 2 or not len(centers) or centers.shape[1] != values.shape[1]
            or not all(np.isfinite(x).all() for x in (values, observables, centers))):
        raise ValueError("invalid source bootstrap arrays")
    if (not np.issubdtype(chain_ids.dtype, np.integer)
            or not np.issubdtype(window_ids.dtype, np.integer)
            or np.any(chain_ids < 0) or np.any(window_ids < 0)
            or not np.array_equal(np.unique(window_ids), np.arange(len(centers)))):
        raise ValueError("source chain/window IDs must be nonnegative integers covering exactly the umbrella windows")
    if (not np.isfinite(min_average_sign) or not 0 < min_average_sign <= 1
            or not np.isfinite(overlap_threshold) or not 0 < overlap_threshold < 1):
        raise ValueError("invalid bootstrap sign or overlap threshold")
    if isinstance(replicates, bool) or not isinstance(replicates, (int, np.integer)) or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    if minimum_valid_fraction != 1.0:
        raise ValueError("confidence intervals require every bootstrap replicate; no failure filtering")
    windows = np.arange(len(centers))
    chains = np.unique(chain_ids)
    streams = {(int(c), int(w)): np.flatnonzero((chain_ids == c) & (window_ids == w))
               for c in chains for w in windows}
    if any(len(ids) == 0 for ids in streams.values()):
        raise ValueError("each independent chain must contain every umbrella window")
    lengths = {}
    for key, ids in streams.items():
        if key not in block_lengths:
            raise ValueError("block_lengths must define every source chain/window stream")
        value = block_lengths[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 1 <= value <= len(ids):
            raise ValueError("block lengths must be integer values within each source stream")
        lengths[key] = int(value)
    if len({len(ids) for ids in streams.values()}) != 1:
        raise ValueError("this example bootstrap requires equal source-stream sample counts")
    count = len(next(iter(streams.values())))
    u = umbrella_reduced_potentials(values, centers, stiffness, beta)
    counts = np.full(len(centers), count*len(chains), dtype=int)
    rng = np.random.default_rng(seed)
    estimates, sign_draws, failures = [], [], []
    for draw in range(replicates):
        indices = []
        for window in windows:
            for chain in rng.choice(chains, size=len(chains), replace=True):
                ids = streams[int(chain), int(window)]
                length = lengths[int(chain), int(window)]
                starts = rng.integers(0, count, size=math.ceil(count/length))
                local = ((starts[:, None]+np.arange(length)) % count).reshape(-1)[:count]
                indices.extend(ids[local])
        indices = np.asarray(indices, dtype=int)
        fit = solve_mbar(u[:, indices], counts, overlap_threshold=overlap_threshold,
                         return_state_weights=False)
        mean_sign = float(fit.weights @ signs[indices])
        if not fit.converged or len(set(fit.components)) != 1 or mean_sign < min_average_sign:
            failures.append({"replicate": draw, "mbar_converged": bool(fit.converged),
                             "overlap_connected": len(set(fit.components)) == 1,
                             "average_sign": mean_sign})
            continue
        estimates.append((fit.weights*signs[indices]) @ observables[indices]/mean_sign)
        sign_draws.append(mean_sign)
    estimates = np.asarray(estimates, dtype=float).reshape(-1, observables.shape[1])
    complete_blocks = [len(streams[key])//length for key, length in lengths.items()]
    reasons = []
    if len(chains) < 2:
        reasons.append("fewer_than_two_independent_source_chains")
    if replicates < 32:
        reasons.append("fewer_than_32_bootstrap_replicates")
    if min(complete_blocks) < 8:
        reasons.append("fewer_than_eight_complete_blocks_in_a_source_stream")
    if failures:
        reasons.append("bootstrap_contains_unresolved_reweighting_or_signed_normalization")
    available = not reasons
    return {"method": "hierarchical_source_chain_circular_block_bootstrap_with_MBAR_refit",
            "scope": "conditional_on_frozen_path_and_selected_block_lengths",
            "requested_replicates": replicates, "valid_replicates": len(estimates),
            "failure_records": failures, "unavailable_reasons": reasons,
            "confidence_intervals_available": available,
            "confidence_level": 0.95,
            "block_lengths": [{"chain": c, "window": w, "length": length}
                              for (c, w), length in sorted(lengths.items())],
            "minimum_complete_blocks_per_stream": min(complete_blocks),
            "standard_error": np.std(estimates, axis=0, ddof=1).tolist() if available else None,
            "percentile_interval": np.quantile(estimates, [0.025, 0.975], axis=0).tolist() if available else None,
            "valid_draw_estimates_diagnostic": estimates.tolist(),
            "valid_draw_average_signs_diagnostic": sign_draws,
            "physical_converged": False}


def matched_timestep_dynamics(parents, potential, masses, beta, beads, *, dts,
                              duration, min_average_sign=0.1):
    """Propagate identical full parents at each dt, with synchronized output."""
    dts = tuple(float(dt) for dt in dts)
    if (len(dts) < 2 or not np.isfinite(dts).all() or min(dts) <= 0
            or not np.isfinite(duration) or duration <= 0
            or not np.isfinite(min_average_sign) or not 0 < min_average_sign <= 1):
        raise ValueError("matched timesteps and duration must be finite and positive with a valid sign threshold")
    coarse = dts[0]
    nframes = int(round(duration/coarse))+1
    if (any(a <= b for a, b in zip(dts, dts[1:]))
            or any(not np.isclose(coarse/dt, round(coarse/dt), atol=1e-10, rtol=0) for dt in dts)
            or not np.isclose(duration/coarse, nframes-1, atol=1e-10, rtol=0)):
        raise ValueError("matched timesteps must divide the common interval and duration")
    n = len(parents.states)
    if not n:
        raise ValueError("matched dynamics requires at least one complete parent")
    arrays = {name: np.empty((len(dts), n, nframes, *getattr(parents.states[0], name).shape))
              for name in ("q", "p", "mapping_q", "mapping_p")}
    energies = np.empty((len(dts), n, nframes))
    for j, dt in enumerate(dts):
        dynamics = ThermalNRPMDDynamics(potential, masses, beta, beads, dt, 2, native=False)
        stride = int(round(coarse/dt))
        for i, parent in enumerate(parents.states):
            state = parent.copy()
            for frame in range(nframes):
                if frame:
                    for _ in range(stride):
                        dynamics.step(state)
                for name, values in arrays.items():
                    values[j, i, frame] = getattr(state, name)
                energies[j, i, frame] = dynamics.energy(state)
    signs = parents.signs
    mean_sign = float(signs.mean())
    resolved = mean_sign >= min_average_sign
    centroids = arrays["q"].mean(axis=3)[:, :, :, 0, :2]
    populations = (0.5*(arrays["mapping_q"]**2+arrays["mapping_p"]**2-1.)).mean(axis=3)
    signed_cv = np.einsum("n,dntk->dtk", signs, centroids)/signs.sum() if resolved else None
    signed_pop = np.einsum("n,dntk->dtk", signs, populations)/signs.sum() if resolved else None
    errors = []
    for j, dt in enumerate(dts[:-1]):
        metrics = {"dt": dt, "reference_dt": dts[-1]}
        for name, values in arrays.items():
            delta = (values[j]-values[-1]).reshape(n, nframes, -1)
            metrics[name+"_rms_error_by_time"] = np.sqrt(np.mean(delta**2, axis=(0, 2))).tolist()
            metrics[name+"_maximum_absolute_error"] = float(np.max(abs(delta)))
        metrics["signed_centroid_max_error"] = float(np.max(abs(signed_cv[j]-signed_cv[-1]))) if resolved else None
        metrics["signed_population_max_error"] = float(np.max(abs(signed_pop[j]-signed_pop[-1]))) if resolved else None
        errors.append(metrics)
    drift = np.max(abs(energies-energies[:, :, :1]), axis=(1, 2))
    ratios = {name: (errors[0][name+"_maximum_absolute_error"]/errors[1][name+"_maximum_absolute_error"]
                     if len(errors) >= 2 and errors[1][name+"_maximum_absolute_error"] > 0 else None)
              for name in arrays}
    report = {"available": True, "parents": n, "parent_average_sign": mean_sign,
              "signed_observables_available": resolved,
              "signed_observables_unavailable_reasons": [] if resolved else ["resampled_average_sign_below_threshold"],
              "dts": dts, "duration": duration, "time": (np.arange(nframes)*coarse).tolist(),
              "same_complete_parents": True, "bias_applied": False, "thermostat_applied": False,
              "source_ancestry": parents.sampling_diagnostics,
              "maximum_absolute_hamiltonian_drift": drift.tolist(),
              "signed_centroid_mean": None if signed_cv is None else signed_cv.tolist(),
              "signed_MMST_population_mean": None if signed_pop is None else signed_pop.tolist(),
              "errors_against_finest_timestep": errors,
              "coarse_to_intermediate_error_ratios": ratios,
              "uncertainty_scope": "paired_numerical_error_conditional_on_same_sampled_parents",
              "rate_constant": None, "physical_converged": False}
    arrays.update(energies=energies, signs=signs, time=np.arange(nframes)*coarse, dts=np.asarray(dts))
    return report, arrays


def _point_id(voltage, beads):
    # repr(float) round-trips uniquely; decimal rounding can alias distinct cases.
    voltage = float(voltage)
    sign = "m" if voltage < 0 else "p"
    token = repr(abs(voltage)).replace("+", "p").replace("-", "m").replace(".", "_")
    return f"U{sign}{token}_P{beads}"


def run_case(config_dict, potential_V, beads, output_directory, *, resume=False):
    config = StudyConfig(**config_dict)
    output = Path(output_directory)/_point_id(potential_V, beads)
    output.mkdir(parents=True, exist_ok=True)
    identity = {"config": asdict(config), "potential_V": float(potential_V), "beads": int(beads),
                "source_identity": source_identity()}
    fingerprint = hashlib.sha256(_canonical(identity).encode()).hexdigest()
    report_path, source_path, dynamics_path = output/"report.json", output/"source.npz", output/"dynamics.npz"
    if report_path.exists():
        saved = json.loads(report_path.read_text())
        if not resume:
            raise ValueError(f"completed case exists: {output}; use --resume for exact identity reuse")
        if saved.get("fingerprint") != fingerprint:
            raise ValueError(f"resume configuration/source identity mismatch: {output}")
        for name, expected in saved["artifact_sha256"].items():
            path = output/name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"resume artifact missing or changed: {path}")
        return saved
    with threadpool_limits(limits=1):
        start = time.perf_counter()
        potential, reservoir, cv = make_model(potential_V)
        point_seed = np.random.SeedSequence([config.seed, config.potentials_V.index(float(potential_V)), config.beads.index(beads)])
        children = point_seed.spawn(3+config.production_chains*config.images)
        images = np.zeros((config.images, beads, 1, 3))
        images[:, :, 0, 0] = np.linspace(-1., 1., config.images)[:, None]
        # The integrated driver's small first production block is discarded;
        # all study evidence comes from independently initialized chains below.
        options = AdaptivePathOptions(
            max_iterations=config.pilot_iterations, pilot_samples=config.pilot_samples,
            pilot_burn_in=config.pilot_burn_in, pilot_stride=config.pilot_stride,
            production_samples=1, production_burn_in=0, production_stride=1,
            relaxation=0.5, max_step=0.15, path_tolerance=0.02,
            min_average_sign=config.min_average_sign, min_effective_samples=1.,
            overlap_threshold=config.overlap_threshold, nuclear_step=0.25, centroid_step=0.2,
        )
        path = run_adaptive_nrpmd_path(potential, [config.mass], config.beta, images, 2, cv,
                                      config.stiffness, options=options, cv_scales=[1., 1.],
                                      seed=int(children[0].generate_state(1)[0]))
        path_seconds = time.perf_counter()-start
        centers = path.centers
        windows, chains, window_ids = [], [], []
        for chain in range(config.production_chains):
            for window, center in enumerate(centers):
                q = np.zeros((beads, 1, 3))
                q[:, 0, :2] = center
                sampler = ThermalNRPMDSampler(
                    potential, [config.mass], config.beta, q, 2,
                    seed=children[3+chain*config.images+window],
                    nuclear_step=0.25, centroid_step=0.2,
                    bias=UmbrellaBias(cv, center, config.stiffness),
                )
                sampled = sampler.sample(config.production_samples, burn_in=config.production_burn_in,
                                          stride=config.production_stride)
                windows.append(sampled)
                chains.extend([chain]*len(sampled.states))
                window_ids.extend([window]*len(sampled.states))
        chain_ids, window_ids = np.asarray(chains), np.asarray(window_ids)
        weighted = reweight_thermal_umbrellas(
            windows, cv, np.tile(centers, (config.production_chains, 1)), config.stiffness,
            min_average_sign=config.min_average_sign, min_effective_samples=config.min_effective_samples,
            overlap_threshold=config.overlap_threshold,
        )
        source = weighted.ensemble
        observable = source_observables(source, reservoir.chemical_potential_au)
        per_chain = []
        for chain in range(config.production_chains):
            local = reweight_thermal_umbrellas(
                windows[chain*config.images:(chain+1)*config.images], cv, centers, config.stiffness,
                min_average_sign=config.min_average_sign, min_effective_samples=config.min_effective_samples,
                overlap_threshold=config.overlap_threshold,
            )
            per_chain.append({"chain": chain, "reweighting": local.summary(),
                              "observables": (dict(zip(OBSERVABLE_NAMES, local.expectation(
                                  source_observables(local.ensemble, reservoir.chemical_potential_au)))))
                              if not local.unavailable_reasons else None})
        inefficiencies = weighted.diagnostics["window_statistical_inefficiency_diagnostics"]
        block_lengths = {(c, w): min(config.production_samples, max(1, int(math.ceil(
            config.bootstrap_block_multiplier*inefficiencies[c*config.images+w]))))
                         for c in range(config.production_chains) for w in range(config.images)}
        bootstrap = source_block_bootstrap(
            weighted.cv_values, source.signs, observable, centers,
            np.eye(cv.dimension)*config.stiffness, config.beta, chain_ids, window_ids,
            replicates=config.bootstrap_replicates, seed=children[1], block_lengths=block_lengths,
            min_average_sign=config.min_average_sign, overlap_threshold=config.overlap_threshold,
        )
        sampling_seconds = time.perf_counter()-start-path_seconds
        estimates = dict(zip(OBSERVABLE_NAMES, weighted.expectation(observable))) if not weighted.unavailable_reasons else None
        raw = {name: np.array([getattr(state, name) for state in source.states])
               for name in ("q", "p", "mapping_q", "mapping_p")}
        raw.update(signs=source.signs, electronic_populations=source.electronic_populations,
                   sweeps=source.sweeps, cv_values=weighted.cv_values, weights=weighted.weights,
                   chain_ids=chain_ids, window_ids=window_ids, source_stream_ids=weighted.window_indices,
                   source_sample_indices=weighted.sample_indices, centers=centers,
                   observables=observable, observable_names=np.asarray(OBSERVABLE_NAMES),
                   metadata_json=np.asarray(_canonical(identity)))
        _write_npz(source_path, **raw)
        report = {"schema_version": 1, "point_id": output.name, "fingerprint": fingerprint,
                  **identity, "model_scope": "analytic_two_state_curved_valley_with_common_BO_charging",
                  "reservoir_chemical_potential_au": reservoir.chemical_potential_au,
                  "path": path.summary(), "path_history": path.history,
                  "discarded_integrated_driver_production_samples": config.images,
                  "production_independent_chains_per_window": config.production_chains,
                  "production_stream_diagnostics": [
                      {"chain": i//config.images, "window": i % config.images,
                       **sampled.sampling_diagnostics}
                      for i, sampled in enumerate(windows)],
                  "production_reweighting": weighted.summary(), "observable_names": OBSERVABLE_NAMES,
                  "observables": estimates, "independent_chain_results": per_chain,
                  "bootstrap": bootstrap, "units": {"length": "bohr", "energy": "hartree", "time": "atomic_time", "voltage": "volt"},
                  "sampling_time_is_physical_time": False, "physical_converged": False,
                  "rate_constant": None,
                  "artifacts": {"source": "source.npz"}}
        if not weighted.unavailable_reasons:
            parents = weighted.resample_parents(config.dynamics_parents, seed=children[2])
            dynamic_report, dynamic_arrays = matched_timestep_dynamics(
                parents, potential, [config.mass], config.beta, beads,
                dts=config.dynamics_dts, duration=config.dynamics_duration,
                min_average_sign=config.min_average_sign,
            )
            report["dynamics"] = dynamic_report
            report["artifacts"]["dynamics"] = "dynamics.npz"
            _write_npz(dynamics_path, **dynamic_arrays,
                       source_flat_indices=np.asarray(parents.sampling_diagnostics["source_flat_indices"]))
        else:
            report["dynamics"] = {"available": False, "unavailable_reasons": weighted.unavailable_reasons,
                                  "physical_converged": False}
        q = raw["q"].reshape(-1, 1, 3)
        charge = potential.solve(q)
        report["charge"] = {"configuration_count": len(q),
                            "maximum_chemical_potential_residual_au": float(np.max(abs(charge.chemical_potential_residuals_au))),
                            "minimum_electron_number": float(charge.electron_numbers.min()),
                            "maximum_electron_number": float(charge.electron_numbers.max())}
        report["timing_seconds"] = {"path": path_seconds, "sampling_and_bootstrap": sampling_seconds,
                                    "total": time.perf_counter()-start}
        report["artifact_sha256"] = {name: hashlib.sha256((output/name).read_bytes()).hexdigest()
                                     for name in report["artifacts"].values()}
        _write_json(report_path, report)
        return _jsonable(report)


def run_study(config, output, workers=1, resume=False):
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    study_plan = plan(config)
    manifest = output/"plan.json"
    if manifest.exists():
        if not resume or _canonical(json.loads(manifest.read_text())) != _canonical(study_plan):
            raise ValueError("existing study requires --resume with identical plan and source hashes")
    else:
        _write_json(manifest, study_plan)
    tasks = [(v, p) for v in config.potentials_V for p in config.beads]
    reports = []
    if workers == 1:
        for v, p in tasks:
            reports.append(run_case(asdict(config), v, p, output, resume=resume))
            print(f"completed {_point_id(v,p)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_case, asdict(config), v, p, str(output), resume=resume): (v,p)
                       for v,p in tasks}
            for future in as_completed(futures):
                reports.append(future.result())
                v, p = futures[future]
                print(f"completed {_point_id(v,p)}", flush=True)
    reports.sort(key=lambda r: tasks.index((r["potential_V"],r["beads"])))
    summary = {"schema_version": 1, "plan": study_plan,
               "cases": [{key: r[key] for key in (
                   "point_id", "potential_V", "beads", "observables", "production_reweighting",
                   "bootstrap", "independent_chain_results", "charge", "timing_seconds")}
                         | {"report": r["point_id"]+"/report.json",
                            "dynamics_available": r["dynamics"]["available"],
                            "path_stopping_reason": r["path"]["stopping_reason"]}
                         for r in reports],
               "physical_converged": False, "rate_constant": None,
               "reference_comparison": "Run adaptive_nrpmd_reference.py for independent finite-bead grid reference."}
    _write_json(output/"summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("smoke", "full"), default="full")
    parser.add_argument("--config", type=Path, help="JSON field overrides on the selected preset")
    parser.add_argument("--plan", action="store_true", help="print budgets without running")
    parser.add_argument("--output", type=Path, default=Path("runs/adaptive-nrpmd-study"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    values = asdict(preset_config(args.preset))
    if args.config:
        supplied = json.loads(args.config.read_text())
        if not isinstance(supplied, dict) or set(supplied)-{f.name for f in fields(StudyConfig)}:
            parser.error("configuration must be an object with known StudyConfig fields")
        values.update(supplied)
    config = StudyConfig(**values)
    if args.plan:
        print(json.dumps(_jsonable(plan(config)), indent=2, allow_nan=False))
        return None
    return run_study(config, args.output, args.workers, args.resume)


if __name__ == "__main__":
    main()
