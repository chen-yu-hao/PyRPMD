"""Native RPMD children from an independent, finite-P surface-flux ensemble.

This benchmark does not change the production integrator, PES or coordinate.
All registered parents are retained. Ordinary Maxwell momenta are antithetic;
the configuration source already contains the flux metric, so both numerator
and denominator divide by that metric. Independent surface chains, not their
children, are the uncertainty units.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmarks.h3.classical_tst import MASS_AU
from benchmarks.h3.pes import BKMP2Potential
from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.constants import FS_TO_AU, KB_AU
from pyrpmd.cv import RPMDrateCoordinate
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.io import write_json

EVIDENCE = ROOT / "docs/validation/literature-convergence-2026-11-01"
PARENT_PROTOCOL = EVIDENCE / "surface-partition/primary-protocol.json"
BETA = 1. / (KB_AU * 300.)
MASS = np.full(3, MASS_AU)
PAIRS = 2
CHUNK_PARENTS = 8


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def source_identity():
    paths = [Path(__file__), ROOT / "benchmarks/h3/pes.py",
             ROOT / "benchmarks/h3/classical_tst.py"]
    paths += [ROOT / "src/pyrpmd" / name for name in
              ("dynamics.py", "propagation.py", "cv.py", "constants.py", "checkpoint.py", "io.py")]
    result = {str(path.relative_to(ROOT)): sha(path) for path in paths}
    native = Path(importlib.import_module("pyrpmd._native").__file__)
    result["loaded_native:" + native.name] = sha(native)
    return result


def surface_coordinate():
    r = 1.7570000251
    coordinate = RPMDrateCoordinate(
        [[0], [1, 2]], MASS, np.array([[-r, 0., 0.], [0., 0., 0.], [r, 0., 0.]]),
        [[0, 1]], [[1, 2]], 30., equivalent_paths=[[0, 2, 1]])
    return coordinate.dividing_surface(1.)


def minimum_pair(q):
    q = np.asarray(q, float)
    return float(min(np.linalg.norm(q[..., i, :] - q[..., j, :], axis=-1).min()
                     for i, j in ((0, 1), (0, 2), (1, 2))))


def assign_channel(q, seed, parent_index):
    """The physical equivalent channel uses an independent, reproducible stream."""
    channel = int(np.random.default_rng(np.random.SeedSequence(
        [int(seed), 95321, int(parent_index)])).integers(2))
    return np.asarray(q, float)[:, [0, 2, 1] if channel else [0, 1, 2]].copy(), channel


def maxwell_momenta(seed, parent_index, pair, beads, beta=BETA, masses=MASS):
    """No timestep, job order, chunk size or restart count enters the RNG key."""
    rng = np.random.default_rng(np.random.SeedSequence(
        [int(seed), 173017, int(parent_index), int(pair)]))
    return rng.normal(size=(beads, 3, 3)) * np.sqrt(np.asarray(masses)[None, :, None] * beads / beta)


def flux_velocity(q, p, *, surface=None, beta=BETA, masses=MASS):
    surface = surface_coordinate() if surface is None else surface
    value, gradient = surface.centroid_value_gradient(np.asarray(q).mean(axis=0))
    if abs(value) > 1e-10:
        raise ValueError("parent centroid is not on the registered dividing surface")
    metric = np.sum(gradient * gradient / np.asarray(masses)[:, None])
    fs = float(np.sqrt(metric / (2. * np.pi * beta)))
    velocity = float(np.sum(gradient * np.asarray(p).mean(axis=0) / np.asarray(masses)[:, None]))
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("invalid surface flux metric")
    return fs, velocity


def paired_flux(velocity, fs, surface_values):
    values = np.asarray(surface_values)
    if values.shape[0] != 2 or not np.isfinite(values).all():
        raise ValueError("finite plus/minus surface observations required")
    return velocity / fs * ((values[0] > 0).astype(float) - (values[1] > 0)), abs(velocity) / fs


def design(library):
    parents = json.loads(PARENT_PROTOCOL.read_text())
    with BKMP2Potential(library) as potential:
        model = potential.metadata
    return {
        "schema": "pyrpmd.h3.quantum-surface-recrossing.v1",
        "parent_protocol_sha256": sha(PARENT_PROTOCOL), "seeds": parents["ts_seeds"],
        "beads": parents["beads"], "temperature_K": 300., "beta": BETA,
        "mass_au": MASS_AU, "parents_per_seed": parents["TS_snapshots_per_seed"],
        "parent_completed_sweeps": parents["TS_snapshot_completed_sweeps"],
        "momentum_pairs_per_parent": PAIRS, "children_per_pair": 2,
        "initial_momentum_pair_indices": [0, 1], "precision_extension_pair_indices": [2, 3],
        "timestep_fs": [.1, .05], "primary_timestep_fs": .05,
        "initial_horizon_fs": 200, "all_child_extension_horizon_fs": 400,
        "surface_observation_interval_fs": 1, "minimum_pair_observation_interval_fs": 1,
        "minimum_pair_scope": "1-fs boundary observations; not an every-physical-step minimum",
        "energy_scope": "full ring Hamiltonian extrema at every physical step; errors scaled by beta/P",
        "integrator": "unmodified PyRPMD cayley-bcb, required fused native loop, no thermostat",
        "configuration_source": "static finite-P delta(s1)*fs(q) flux measure; total COM modes marginalized",
        "momentum_source": "independent ordinary Maxwell p_j variance=m*P/beta, paired plus/minus",
        "COM_convention": "parent COM coordinates fixed to zero; free COM momenta retained and decoupled from relative observables",
        "flux_estimator": "sum[(v/fs)*(H(s_plus)-H(s_minus))] / sum[abs(v)/fs]",
        "expected_denominator": 2.,
        "uncertainty_unit": "eight independent parent chains; combine each chain BAR log-Z and log-kappa before its rate CI",
        "plateau_windows_fs": [[80, 100], [180, 200], [380, 400]],
        "plateau_equivalence_absolute_margin": .01,
        "timestep_equivalence_absolute_margin": .01,
        "equivalence_rule": "paired-chain 95% Student-t interval wholly within the stated +/- margin",
        "precision_gate_relative_kappa_95_halfwidth": .03,
        "extension_rule": "extend every registered child to 400 fs if 200-fs time plateau gate fails; no selective extension",
        "precision_extension_rule": "If all-chain time/timestep equivalence passes but kappa relative 95% halfwidth exceeds 3%, append the same pair indices 2 and 3 to every parent, seed and timestep in a separately retained batch. Reevaluate gates on all four pairs; no extension based on closeness to a paper point.",
        "failure_rule": "retain all children; any sampled domain crossing is reported; PES guard/integration failure stops the job with saved failure state",
        "static_domain_minimum_pair_bohr": parents["minimum_bead_pair_bohr"],
        "source_sha256": source_identity(), "model": model,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
    }


def register(output, library):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "protocol.json"
    expected = design(library)
    if path.exists():
        actual = json.loads(path.read_text())
        actual.pop("created_before_primary_children_utc")
        if actual != expected:
            raise ValueError("recrossing protocol or source changed; preserve the registered campaign")
    else:
        write_json(path, dict(expected, created_before_primary_children_utc=datetime.now(timezone.utc).isoformat()))
    return expected


def expected_parent_settings(parent_protocol, seed):
    """Bind the entire registered source, domain and Monte Carlo protocol."""
    p = parent_protocol
    return dict(schema="pyrpmd.h3.surface-partition.v1", phase="production", surface="ts",
                beads=p["beads"], seed=seed, warmup=p["warmup_sweeps"], production=p["production_sweeps"],
                lambdas=p["lambda_grid"], temperature_K=p["temperature_K"], beta=BETA, mass_au=MASS_AU,
                minimum_bead_pair_bohr=p["minimum_bead_pair_bohr"],
                maximum_centroid_bond_bohr=p["maximum_centroid_bond_bohr"],
                radial_order=p["reference_radial_order"], angular_order=p["reference_angular_order"],
                domain_samples=p["domain_samples_per_independent_seed"], radius_step=p["radius_step"],
                ratio_step=p["ratio_step"], alpha_R=p["alpha_R"], alpha_r=p["alpha_r_by_surface"]["ts"],
                swap_interval=p["swap_interval"], checkpoint_interval=p["checkpoint_interval"],
                sample_interval=p["sample_interval"], diagnostic_interval=p["diagnostic_interval"],
                snapshot_interval=p["TS_snapshot_interval"], **p["source_identity"],
                endpoint_snapshot_source="static equilibrium Monte Carlo at lambda=1; normalized total COM analytically marginalized",
                endpoint_snapshot_channel="r01=r12<=r02; equivalent channel and Maxwell momenta assigned only by separate child RNG",
                target="physical bead domain and continuous centroid shape; normalized Jacobi free-ring prior",
                reference="centroid PES at max(r, minimum); full centroid support (0, maximum)",
                sampling_is_real_time_dynamics=False, uncertainty_unit="independent seed chain")


def load_parents(directory, protocol, *, calibration=False, count=None):
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text())
    settings = json.loads((directory / "settings.json").read_text())
    if not report["complete"] or not report["source_unchanged"] or report["surface"] != "ts":
        raise ValueError("complete, unchanged TS parent chain required")
    checkpoint = directory / "checkpoint.npz"
    if sha(checkpoint) != report["checkpoint_sha256"]:
        raise ValueError("parent checkpoint/report checksum mismatch")
    with load_checkpoint(checkpoint, expected_kind="h3.surface-partition", require_integrity=True) as saved:
        expected_fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
        if str(saved["fingerprint"].item()) != expected_fingerprint:
            raise ValueError("parent checkpoint is not bound to these complete settings")
        if str(saved["domain_sha256"].item()) != sha(directory / "reference-domain.json"):
            raise ValueError("parent domain normalization differs from the checkpoint")
        if int(saved["completed"]) != settings["warmup"] + settings["production"]:
            raise ValueError("parent checkpoint production is incomplete")
        q = saved["endpoint_q"]
        completed = saved["endpoint_completed_sweeps"]
    if q.ndim != 4 or q.shape[1:] != (protocol["beads"], 3, 3) or not np.isfinite(q).all():
        raise ValueError("invalid parent geometry array")
    if not calibration:
        parent_protocol = json.loads(PARENT_PROTOCOL.read_text())
        checks = (report["seed"] in protocol["seeds"], report["phase"] == "production",
                  report["beads"] == protocol["beads"],
                  settings == expected_parent_settings(parent_protocol, report["seed"]),
                  len(q) == protocol["parents_per_seed"],
                  np.array_equal(completed, protocol["parent_completed_sweeps"]))
        if not all(checks) or count is not None:
            raise ValueError("parent chain differs from the registered primary protocol")
    else:
        if report["phase"] != "calibration":
            raise ValueError("calibration must use explicitly excluded calibration parents")
        if count is not None:
            if not 0 < count <= len(q):
                raise ValueError("invalid calibration parent count")
            q, completed = q[:count], completed[:count]
    if (settings["beta"] != protocol["beta"] or settings["mass_au"] != protocol["mass_au"]
            or settings["model"] != protocol["model"] or settings["beads"] != protocol["beads"]):
        raise ValueError("parent physical Hamiltonian differs from the child Hamiltonian")
    for parent in q:
        flux_velocity(parent, np.zeros_like(parent))
    if minimum_pair(q) < protocol["static_domain_minimum_pair_bohr"] - 1e-12:
        raise ValueError("initial parent outside the declared static domain")
    if np.max(abs(q.mean(axis=2))) > 1e-10:
        raise ValueError("unexpected COM convention in parent source")
    identity = {"checkpoint_sha256": sha(checkpoint), "settings_sha256": sha(directory / "settings.json"),
                "report_sha256": sha(directory / "report.json"), "seed": report["seed"],
                "BAR_log_TS_quantum_classical_ratio": report["analysis"]["log_quantum_classical_ratio"]}
    return q, completed, identity


def read_chunk(path, expected_fingerprint, indices):
    with load_checkpoint(path, expected_kind="h3.quantum-surface-children", require_integrity=True) as saved:
        if str(saved["fingerprint"].item()) != expected_fingerprint:
            raise ValueError("child chunk identity differs")
        if not np.array_equal(saved["parent_index"], indices):
            raise ValueError("child chunk parent inventory differs")
        return {key: saved[key] for key in saved.files}


def run_directory(output, seed, timestep, horizon, pair_batch=0):
    base = Path(output) if pair_batch == 0 else Path(output) / f"extra-pair-batch-{pair_batch}"
    return base / f"seed-{seed}" / f"dt-{timestep:g}" / f"horizon-{horizon}"


def run_seed(output, library, seed, timestep, *, horizon=200, pair_batch=0, calibration_parent=None,
             calibration_count=None, stop_after_chunks=None):
    output = Path(output)
    calibration = calibration_parent is not None
    protocol = design(library) if calibration else register(output, library)
    if pair_batch not in (0, 1):
        raise ValueError("only the registered initial and precision-extension pair batches are allowed")
    if timestep not in protocol["timestep_fs"]:
        raise ValueError("unregistered child timestep")
    if calibration:
        if not isinstance(horizon, int) or not 1 <= horizon <= 400:
            raise ValueError("integer calibration horizon in [1,400] required")
        parent_directory = Path(calibration_parent)
    else:
        if seed not in protocol["seeds"] or horizon not in (200, 400):
            raise ValueError("unregistered seed or horizon")
        parent_directory = PARENT_PROTOCOL.parent / "primary/ts" / f"seed-{seed}"
    parents, sweeps, parent_identity = load_parents(parent_directory, protocol, calibration=calibration,
                                                   count=calibration_count)
    if seed != parent_identity["seed"]:
        raise ValueError("child seed must identify its original parent chain")
    before = source_identity()
    run = run_directory(output, seed, timestep, horizon, pair_batch)
    run.mkdir(parents=True, exist_ok=True)
    start_horizon = 200 if not calibration and horizon == 400 else 0
    previous = run.parent / "horizon-200"
    settings = {"schema": protocol["schema"], "phase": "excluded calibration" if calibration else "primary",
                "seed": seed, "beads": protocol["beads"], "parents": len(parents),
                "timestep_fs": timestep, "horizon_fs": horizon, "start_horizon_fs": start_horizon,
                "pair_batch": pair_batch, "momentum_pair_indices": list(range(PAIRS*pair_batch, PAIRS*(pair_batch+1))),
                "protocol_sha256": fingerprint(protocol), "parent_identity": parent_identity,
                "source_sha256": before, "chunk_parents": CHUNK_PARENTS}
    identity = fingerprint(settings)
    settings_path = run / "settings.json"
    if settings_path.exists():
        if json.loads(settings_path.read_text()) != settings:
            raise ValueError("child run settings changed")
    else:
        write_json(settings_path, settings)
    prior_identity = None
    prior_hashes = {}
    if start_horizon:
        prior_settings = dict(settings, horizon_fs=200, start_horizon_fs=0)
        if json.loads((previous / "settings.json").read_text()) != prior_settings:
            raise ValueError("extension is not the same registered child ensemble")
        prior_identity = fingerprint(prior_settings)
        prior_report = json.loads((previous / "report.json").read_text())
        if not prior_report["complete"]:
            raise ValueError("extend only a complete 200-fs child cohort")
        if prior_report["settings_sha256"] != sha(previous / "settings.json"):
            raise ValueError("predecessor report/settings checksum mismatch")
        prior_hashes = {item["file"]: item["sha256"] for item in prior_report["chunks"]}
        expected_names = {f"chunk-{left:04d}.npz" for left in range(0, len(parents), CHUNK_PARENTS)}
        if set(prior_hashes) != expected_names:
            raise ValueError("predecessor report has an incomplete child inventory")
        for name, expected_hash in prior_hashes.items():
            if sha(previous / name) != expected_hash:
                raise ValueError("predecessor chunk differs from its completed report")
    times = np.arange(1, horizon + 1, dtype=float)
    block_steps = round(1. / timestep)
    if abs(block_steps * timestep - 1.) > 1e-12:
        raise ValueError("timestep must exactly divide the 1-fs observation interval")
    started = time.perf_counter()
    surface = surface_coordinate()
    new_chunks = 0
    with BKMP2Potential(library) as potential:
        dynamics = Dynamics(potential, MASS, BETA, protocol["beads"], timestep * FS_TO_AU,
                            native="required", loop_backend="native")
        if not dynamics.can_fuse(observe=surface):
            raise ValueError("native RPMD child path unavailable")
        for left in range(0, len(parents), CHUNK_PARENTS):
            indices = np.arange(left, min(left + CHUNK_PARENTS, len(parents)))
            path = run / f"chunk-{left:04d}.npz"
            if path.exists():
                saved = read_chunk(path, identity, indices)
                if start_horizon and str(saved["previous_chunk_sha256"].item()) != prior_hashes[path.name]:
                    raise ValueError("existing extension used a different predecessor chunk")
                continue
            if stop_after_chunks is not None and new_chunks >= stop_after_chunks:
                break
            old = read_chunk(previous / path.name, prior_identity, indices) if start_horizon else None
            shape = (len(indices), PAIRS, 2)
            surfaces = np.empty(shape + (horizon,))
            minima = np.empty(shape + (horizon + 1,))
            extrema = np.empty(shape + (3,))
            initial_p = np.empty((len(indices), PAIRS, protocol["beads"], 3, 3))
            final_q = np.empty(shape + (protocol["beads"], 3, 3))
            final_p = np.empty_like(final_q)
            final_step = np.empty(shape, dtype=np.int64)
            final_time = np.empty(shape)
            velocities = np.empty((len(indices), PAIRS))
            fluxes = np.empty(len(indices))
            channels = np.empty(len(indices), dtype=np.int64)
            numerical_before = dict(dynamics.propagation_stats)
            for local, index in enumerate(indices):
                q, channels[local] = assign_channel(parents[index], seed, int(index))
                for pair in range(PAIRS):
                    p = maxwell_momenta(seed, int(index), PAIRS * pair_batch + pair, protocol["beads"])
                    initial_p[local, pair] = p
                    fluxes[local], velocities[local, pair] = flux_velocity(q, p, surface=surface)
                    if old is not None and not np.array_equal(p, old["initial_p"][local, pair]):
                        raise ValueError("extension momentum replay differs")
                    for side, sign in enumerate((1., -1.)):
                        if old is None:
                            state = State(q, sign * p)
                            initial_energy = dynamics.energy(state)
                            extrema[local, pair, side] = [initial_energy, initial_energy, initial_energy]
                            minima[local, pair, side, 0] = minimum_pair(q)
                        else:
                            state = State(old["final_q"][local, pair, side], old["final_p"][local, pair, side],
                                          step=int(old["final_step"][local, pair, side]),
                                          time=float(old["final_time_au"][local, pair, side]))
                            extrema[local, pair, side] = old["energy_extrema"][local, pair, side]
                            minima[local, pair, side, :start_horizon + 1] = old["minimum_pair_at_1fs"][local, pair, side]
                            surfaces[local, pair, side, :start_horizon] = old["surface_values"][local, pair, side]
                        for frame in range(start_horizon, horizon):
                            try:
                                result = dynamics.propagate(state, block_steps, observe=surface,
                                                            sample_interval=block_steps, record_centroids=False)
                            except BaseException as error:
                                atomic_savez(run / "failed-child-state.npz", kind="h3.failed-quantum-child",
                                             q=state.q, p=state.p, parent_index=index, pair=pair, side=side,
                                             last_observed_boundary_fs=frame, state_step=state.step,
                                             state_time_au=state.time, fingerprint=identity)
                                write_json(run / "failure.json", {"parent_index": int(index), "pair": pair,
                                           "side": side, "last_observed_boundary_fs": frame,
                                           "saved_state_step": state.step, "saved_state_time_au": state.time,
                                           "error": repr(error),
                                           "policy": "No child removed; the registered job is incomplete."})
                                raise
                            if result.backend != "native":
                                raise ValueError("unexpected Python child propagation")
                            surfaces[local, pair, side, frame] = result.values[-1, 0]
                            minima[local, pair, side, frame + 1] = minimum_pair(state.q)
                            extrema[local, pair, side, 1] = min(extrema[local, pair, side, 1], result.minimum_energy)
                            extrema[local, pair, side, 2] = max(extrema[local, pair, side, 2], result.maximum_energy)
                        final_q[local, pair, side], final_p[local, pair, side] = state.q, state.p
                        final_step[local, pair, side], final_time[local, pair, side] = state.step, state.time
            numerator = velocities[..., None] / fluxes[:, None, None] * (
                (surfaces[:, :, 0] > 0).astype(float) - (surfaces[:, :, 1] > 0))
            denominator = abs(velocities) / fluxes[:, None]
            native_steps = dynamics.propagation_stats["native_steps"] - numerical_before["native_steps"]
            expected_steps = len(indices) * PAIRS * 2 * (horizon - start_horizon) * block_steps
            if native_steps != expected_steps or dynamics.propagation_stats["python_steps"]:
                raise ValueError("child native step accounting differs")
            if source_identity() != before:
                raise ValueError("child source changed before checkpoint commit")
            atomic_savez(path, kind="h3.quantum-surface-children", fingerprint=identity,
                         parent_index=indices, parent_completed_sweeps=sweeps[indices], channel=channels,
                         momentum_pair_index=np.arange(PAIRS*pair_batch, PAIRS*(pair_batch+1)),
                         time_fs=times, surface_values=surfaces, minimum_pair_at_1fs=minima,
                         energy_extrema=extrema, initial_p=initial_p, final_q=final_q, final_p=final_p,
                         final_step=final_step, final_time_au=final_time,
                         flux_metric=fluxes, velocity=velocities, numerator=numerator, denominator=denominator,
                         native_steps_this_stage=native_steps, previous_chunk_sha256=sha(previous / path.name) if old is not None else "")
            new_chunks += 1
            print(json.dumps({"seed": seed, "dt_fs": timestep, "horizon_fs": horizon,
                              "pair_batch": pair_batch,
                              "parents_completed": int(indices[-1]) + 1,
                              "elapsed_this_call_seconds": time.perf_counter() - started}), flush=True)
    return summarize(run, settings, identity, parents, elapsed=time.perf_counter() - started)


def summarize(run, settings, identity, parents, *, elapsed=0.):
    run = Path(run)
    numerator = np.zeros(settings["horizon_fs"])
    denominator = 0.
    errors, minimum, crossed = [], np.inf, 0
    count, native_steps, chunks = 0, 0, []
    for left in range(0, len(parents), CHUNK_PARENTS):
        indices = np.arange(left, min(left + CHUNK_PARENTS, len(parents)))
        path = run / f"chunk-{left:04d}.npz"
        if not path.exists():
            break
        saved = read_chunk(path, identity, indices)
        numerator += saved["numerator"].sum(axis=(0, 1))
        denominator += float(saved["denominator"].sum())
        extrema = saved["energy_extrema"]
        error = np.maximum(abs(extrema[..., 1] - extrema[..., 0]), abs(extrema[..., 2] - extrema[..., 0]))
        errors.extend((error * BETA / settings["beads"]).ravel().tolist())
        minimum = min(minimum, float(saved["minimum_pair_at_1fs"].min()))
        crossed += int(np.sum(saved["minimum_pair_at_1fs"].min(axis=-1) < .5))
        native_steps += int(saved["native_steps_this_stage"])
        count += len(indices)
        chunks.append({"file": path.name, "sha256": sha(path)})
    times = np.arange(1, settings["horizon_fs"] + 1, dtype=float)
    kappa = numerator / denominator if denominator else None
    plateaus = {f"{lo}-{hi}": float(kappa[(times >= lo) & (times <= hi)].mean())
                for lo, hi in ((80, 100), (180, 200), (380, 400))
                if kappa is not None and hi <= settings["horizon_fs"]}
    report = {"schema": settings["schema"], "phase": settings["phase"], "seed": settings["seed"],
              "timestep_fs": settings["timestep_fs"], "horizon_fs": settings["horizon_fs"],
              "pair_batch": settings["pair_batch"], "momentum_pair_indices": settings["momentum_pair_indices"],
              "complete": count == len(parents), "parents_completed": count, "pairs_per_parent": PAIRS,
              "plateaus": plateaus, "mean_denominator": denominator / (count * PAIRS) if count else None,
              "expected_mean_denominator": 2., "native_steps_this_stage": native_steps,
              "unique_native_steps_all_stages": count * PAIRS * 2 * round(settings["horizon_fs"] / settings["timestep_fs"]),
              "python_steps": 0, "maximum_energy_error_kbt_per_bead": max(errors) if errors else None,
              "energy_error_quantiles_kbt_per_bead": np.quantile(errors, [0., .5, .95, .99, 1.]) if errors else None,
              "minimum_pair_at_1fs_bohr": minimum if count else None,
              "children_with_sampled_pair_below_static_domain": crossed,
              "minimum_pair_scope": "Only initial and 1-fs boundaries; every-step PES guard and full-H extrema are separate checks.",
              "source_unchanged": source_identity() == settings["source_sha256"],
              "settings_sha256": sha(run / "settings.json"), "chunks": chunks,
              "parent_identity": settings["parent_identity"], "elapsed_this_call_seconds": elapsed,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "physical_converged": False, "literature_reproduced": False}
    write_json(run / "report.json", report)
    if kappa is not None:
        np.savetxt(run / "kappa.csv", np.column_stack((times, kappa)), delimiter=",",
                   header="time_fs,kappa", comments="")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--timestep", type=float, default=.05)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--pair-batch", type=int, choices=(0, 1), default=0)
    parser.add_argument("--calibration-parent", type=Path)
    parser.add_argument("--calibration-count", type=int)
    parser.add_argument("--stop-after-chunks", type=int)
    args = parser.parse_args()
    if args.register_only:
        register(args.output, args.library)
        return
    if args.seed is None:
        parser.error("--seed is required when launching children")
    report = run_seed(args.output, args.library, args.seed, args.timestep, horizon=args.horizon,
                      pair_batch=args.pair_batch,
                      calibration_parent=args.calibration_parent, calibration_count=args.calibration_count,
                      stop_after_chunks=args.stop_after_chunks)
    print(json.dumps({key: report[key] for key in
                      ("seed", "complete", "parents_completed", "plateaus", "maximum_energy_error_kbt_per_bead",
                       "minimum_pair_at_1fs_bohr", "elapsed_this_call_seconds")}), flush=True)


if __name__ == "__main__":
    main()
