"""Separate P=1 sampler with an exact incoming-coordinate independence move.

The frozen original mixing sampler is imported for its physical target,
Cartesian moves, rotations and initialization; none of its files or saved
runs is modified. This version adds the independently tested radial volume
Jacobian. It is a benchmark equilibrium sampler, not real-time dynamics.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.h3.classical_sampling_campaign import file_hash
from benchmarks.h3.classical_sampling_xi_proposal import propose
from benchmarks.h3.mixing_sampling import initial_paths, remaining_action, random_rotations, provenance as original_provenance
from benchmarks.h3.classical_tst import MASS_AU
from benchmarks.h3.pes import BKMP2Potential
from pyrpmd.constants import KB_AU
from pyrpmd.checkpoint import CheckpointError, atomic_savez, load_checkpoint
from pyrpmd.io import write_json

CHECKPOINT_KIND = "pyrpmd.h3.classical-xi-checkpoint.v1"


def source_identity(library):
    result = original_provenance(library)
    for name in ("classical_sampling_v2.py", "classical_sampling_xi_proposal.py", "classical_sampling.py"):
        path = ROOT/"benchmarks/h3"/name
        result[str(path.relative_to(ROOT))] = file_hash(path)
    result["src/pyrpmd/checkpoint.py"] = file_hash(ROOT/"src/pyrpmd/checkpoint.py")
    return result


def settings_fingerprint(settings):
    encoded = json.dumps(settings, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def restored_state(path, settings):
    """Check the complete configuration and payload before admitting a restart."""
    fields = {"R", "r", "completed", "samples", "diagnostics", "accepted", "proposed",
              "swaps", "lineage", "rng", "settings_fingerprint"}
    with load_checkpoint(path, expected_kind=CHECKPOINT_KIND, require_integrity=True) as saved:
        if set(saved.files) != fields:
            raise CheckpointError("unexpected classical checkpoint payload fields")
        state = {key: saved[key] for key in fields}
    if str(state["settings_fingerprint"]) != settings_fingerprint(settings):
        raise CheckpointError("classical checkpoint settings fingerprint mismatch")
    completed = state["completed"]
    if (completed.shape != () or completed.dtype.kind not in "iu" or int(completed) < 0
            or int(completed) > settings["warmup"]+settings["production"]):
        raise CheckpointError("invalid classical checkpoint completed count")
    n = len(settings["centers"])
    samples = max(0, int(completed)-settings["warmup"])
    interval = settings["diagnostic_interval"]
    shapes = {"R": (n, 1, 3), "r": (n, 1, 3), "samples": (samples, n),
              "diagnostics": ((samples+interval-1)//interval, n, 7),
              "accepted": (5, n), "proposed": (5, n), "swaps": (2, n-1), "lineage": (n,)}
    for name, shape in shapes.items():
        if state[name].shape != shape:
            raise CheckpointError(f"classical checkpoint shape mismatch: {name}")
        if name in ("R", "r", "samples", "diagnostics") and state[name].dtype != np.dtype(float):
            raise CheckpointError(f"classical checkpoint dtype mismatch: {name}")
        if not np.isfinite(state[name]).all():
            raise CheckpointError(f"nonfinite classical checkpoint array: {name}")
    for name in ("accepted", "proposed", "swaps", "lineage"):
        if state[name].dtype.kind not in "iu" or np.any(state[name] < 0):
            raise CheckpointError(f"invalid classical checkpoint counters: {name}")
    if (np.any(state["accepted"] > state["proposed"])
            or np.any(state["swaps"][1] > state["swaps"][0])
            or not np.array_equal(np.sort(state["lineage"]), np.arange(n))):
        raise CheckpointError("inconsistent classical checkpoint counters or lineages")
    if state["rng"].shape != () or state["settings_fingerprint"].shape != ():
        raise CheckpointError("invalid classical checkpoint metadata shape")
    return state


def run(output, library, *, seed, centers, stiffness, xi_moves=1, warmup=2000,
        production=20000, resume=False, stop_after=None, checkpoint_interval=5000,
        diagnostic_interval=100, centroid_R=.45, centroid_r=.085,
        rotation_angle=.45, swap_interval=10, center_limit=.82,
        xi_bounds=(-.2, .9), minimum_pair=.5, maximum_centroid_bond=8.):
    """Run a new, source-bound P=1 chain; resume requires identical settings."""
    output = Path(output).resolve()
    centers = np.asarray(centers, float)
    stiffness = np.broadcast_to(np.asarray(stiffness, float), centers.shape).copy()
    if (centers.ndim != 1 or len(centers) < 2 or not np.isfinite(centers).all()
            or not np.isfinite(stiffness).all() or np.any(stiffness <= 0)
            or xi_moves not in (0, 1, 3) or warmup < 0 or production < 1
            or min(checkpoint_interval, diagnostic_interval, swap_interval) < 1
            or not .4 <= minimum_pair <= .7 or not 6 <= maximum_centroid_bond <= 12
            or (stop_after is not None and stop_after < 1)):
        raise ValueError("invalid fixed P=1 sampler configuration")
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError("new sampler requires a new output folder")
    output.mkdir(parents=True, exist_ok=True)
    beta = 1/(KB_AU*300.)
    settings = dict(schema="pyrpmd.h3.classical-xi-independence.v2", beads=1, seed=seed,
        warmup=warmup, production=production, centers=centers.tolist(), stiffness=stiffness.tolist(),
        xi_moves=xi_moves, center_limit=center_limit, xi_bounds=list(xi_bounds),
        centroid_R=centroid_R, centroid_r=centroid_r, rotation_angle=rotation_angle,
        swap_interval=swap_interval, checkpoint_interval=checkpoint_interval,
        diagnostic_interval=diagnostic_interval, temperature_K=300., beta=beta,
        mass_au=MASS_AU, rinf_bohr=30., minimum_bead_pair_bohr=minimum_pair,
        maximum_centroid_breaking_bond_bohr=maximum_centroid_bond,
        positive_coordinate_denominator=True, source_sha256=source_identity(library),
        sampling_is_real_time_dynamics=False,
        move_names=["R_centroid", "r_centroid", "R_rotation", "r_rotation", "xi_independence"])
    settings_path = output/"settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError("settings or source changed; use a new independent run")
    write_json(settings_path, settings)
    rng = np.random.default_rng(seed)
    count = len(centers)
    R, r = initial_paths(centers, 1)
    completed = 0
    samples = np.empty((production, count))
    diagnostics = np.empty(((production+diagnostic_interval-1)//diagnostic_interval, count, 7))
    accepted = np.zeros((5, count), dtype=np.int64)
    proposed = np.zeros_like(accepted)
    swaps = np.zeros((2, count-1), dtype=np.int64)
    lineage = np.arange(count)
    checkpoint = output/"checkpoint.npz"
    if resume and checkpoint.exists():
        old = restored_state(checkpoint, settings)
        R, r = old["R"], old["r"]
        completed = int(old["completed"])
        samples[:len(old["samples"])] = old["samples"]
        diagnostics[:len(old["diagnostics"])] = old["diagnostics"]
        accepted, proposed, swaps = old["accepted"], old["proposed"], old["swaps"]
        lineage = old["lineage"]
        rng.bit_generator.state = json.loads(str(old["rng"]))
        del old
    provider = BKMP2Potential(library)

    def evaluate(Rtrial, rtrial):
        return remaining_action(provider, Rtrial, rtrial, beta, centers, stiffness,
            minimum_pair=minimum_pair, maximum_centroid_bond=maximum_centroid_bond)

    action, xi, base, energy, geometry = evaluate(R, r)
    if not np.isfinite(action).all():
        provider.close()
        raise ValueError("invalid initial or restart geometry")
    started = time.perf_counter()
    started_cpu = time.process_time()

    def save():
        if settings["source_sha256"] != source_identity(library):
            raise RuntimeError("source changed during sampling; refusing checkpoint overwrite")
        n = max(0, completed-warmup)
        nd = (n+diagnostic_interval-1)//diagnostic_interval
        atomic_savez(checkpoint, kind=CHECKPOINT_KIND,
            settings_fingerprint=np.asarray(settings_fingerprint(settings)),
            R=R, r=r, completed=completed, samples=samples[:n],
            diagnostics=diagnostics[:nd], accepted=accepted, proposed=proposed,
            swaps=swaps, lineage=lineage, rng=np.array(json.dumps(rng.bit_generator.state)))
        progress = dict(seed=seed, xi_moves=xi_moves, completed_sweeps=completed,
            production_sweeps=n, elapsed_seconds=time.perf_counter()-started,
            physical_converged=False)
        write_json(output/"progress.json", progress)
        print(json.dumps(progress), flush=True)

    def apply(candidate_R, candidate_r, values, accept):
        new_action, new_xi, new_base, new_energy, new_geometry = values
        R[accept] = candidate_R[accept]
        r[accept] = candidate_r[accept]
        for old, new in ((action, new_action), (xi, new_xi), (base, new_base),
                         (energy, new_energy), (geometry, new_geometry)):
            old[accept] = new[accept]

    try:
        limit = warmup+production if stop_after is None else min(warmup+production, completed+stop_after)
        while completed < limit:
            for move in range(4):
                candidate_R, candidate_r = R, r
                if move == 0:
                    candidate_R = R+rng.normal(scale=centroid_R, size=(count, 1, 3))
                elif move == 1:
                    candidate_r = r+rng.normal(scale=centroid_r, size=(count, 1, 3))
                else:
                    rotation = random_rotations(rng, count, angle=rotation_angle)
                    if move == 2:
                        candidate_R = np.einsum("nij,npj->npi", rotation, R)
                    else:
                        candidate_r = np.einsum("nij,npj->npi", rotation, r)
                values = evaluate(candidate_R, candidate_r)
                accept = np.log(rng.random(count)) < action-values[0]
                proposed[move] += 1
                accepted[move] += accept
                apply(candidate_R, candidate_r, values, accept)
            for _ in range(xi_moves):
                candidate_R, log_jacobian_ratio, eligible, old_xi, wanted_xi = propose(
                    rng, R, r, centers, stiffness, beta, center_limit=center_limit, xi_bounds=xi_bounds)
                values = evaluate(candidate_R, r)
                # The Gaussian proposal density cancels the umbrella term.
                # The Cartesian radial volume factor does not cancel.
                accept = (eligible & np.isfinite(values[0]) &
                          (np.log(rng.random(count)) < base-values[2]+log_jacobian_ratio))
                proposed[4] += centers <= center_limit
                accepted[4] += accept
                apply(candidate_R, r, values, accept)
            if (completed+1) % swap_interval == 0:
                parity = int(rng.integers(2))
                left = np.arange(parity, count-1, 2)
                right = left+1
                old_bias = .5*beta*(stiffness[left]*(xi[left]-centers[left])**2
                                      +stiffness[right]*(xi[right]-centers[right])**2)
                new_bias = .5*beta*(stiffness[left]*(xi[right]-centers[left])**2
                                      +stiffness[right]*(xi[left]-centers[right])**2)
                accept = np.log(rng.random(len(left))) < old_bias-new_bias
                swaps[0, left] += 1
                swaps[1, left] += accept
                a, b = left[accept], right[accept]
                for array in (R, r, xi, base, energy, geometry, lineage):
                    array[a], array[b] = array[b].copy(), array[a].copy()
                action[:] = base+.5*beta*stiffness*(xi-centers)**2
            completed += 1
            if completed > warmup:
                index = completed-warmup-1
                samples[index] = xi
                if index % diagnostic_interval == 0:
                    diagnostics[index//diagnostic_interval] = np.column_stack((geometry, energy,
                                                                                np.zeros(count), lineage))
            if completed % checkpoint_interval == 0:
                save()
        save()
    finally:
        provider.close()
    report = dict(schema=settings["schema"], completed=completed == warmup+production,
        completed_sweeps=completed, production_sweeps=max(0, completed-warmup), seed=seed,
        xi_moves=xi_moves, beads=1, elapsed_seconds=time.perf_counter()-started,
        cpu_elapsed_seconds=time.process_time()-started_cpu,
        created_utc=datetime.now(timezone.utc).isoformat(),
        source_unchanged=settings["source_sha256"] == source_identity(library),
        checkpoint_sha256=file_hash(checkpoint), physical_converged=False, literature_reproduced=False,
        acceptance_per_window=accepted/np.maximum(proposed, 1),
        xi_acceptance_active_windows=float(accepted[4].sum()/max(1, proposed[4].sum())),
        diagnostics_columns=["R_centroid_bohr", "r_centroid_bohr", "signed_cosine", "minimum_bead_pair_bohr",
                             "mean_energy_hartree", "r_ring_radius_squared_bohr2", "lineage"])
    write_json(output/"run-report.json", report)
    return report


def _job(job):
    return run(**job)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    schedule = json.loads(args.schedule.read_text())
    if protocol["sampler_source_sha256"] != file_hash(Path(__file__)):
        raise ValueError("registered sampler source differs from this executable")
    if protocol["proposal_source_sha256"] != file_hash(ROOT/"benchmarks/h3/classical_sampling_xi_proposal.py"):
        raise ValueError("registered proposal source differs from the imported version")
    if protocol["schedule_sha256"] != file_hash(args.schedule):
        raise ValueError("registered umbrella schedule differs from the supplied file")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"protocol.json").write_bytes(args.protocol.read_bytes())
    (args.output/"schedule.json").write_bytes(args.schedule.read_bytes())
    jobs = [dict(output=args.output/f"xi-{row['xi_moves']}"/f"seed-{seed}", library=args.library,
                 seed=seed, centers=schedule["centers"], stiffness=schedule["stiffness"],
                 xi_moves=row["xi_moves"], warmup=protocol["warmup_sweeps"], production=protocol["production_sweeps"])
            for row in protocol["candidates"] for seed in row["seeds"]]
    if args.workers == 1:
        for job in jobs:
            _job(job)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(_job, jobs))
