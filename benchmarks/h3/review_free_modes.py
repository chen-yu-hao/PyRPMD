"""Independent review of periodic Andersen refresh and free-ring mixing.

This exercises the current Dynamics implementation on independent free particles.
It diagnoses a sampling mechanism, not a quantitative H3 rate error. No production
integrator or thermostat is changed. Runtime is restricted to one CPU/thread.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
if hasattr(os, "sched_getaffinity"):
    _allowed = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {max(_allowed)})

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyrpmd.constants import AU_TIME_FS, KB_AU
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.potentials.builtin import FreePotential

MASS = 1837.1526474396187
BEADS = 128
TEMPERATURE = 300.0
DT_FS = 0.1
BETA = 1.0 / (KB_AU * TEMPERATURE)


def free_ring_draw(rng, masses, beads=BEADS, beta=BETA):
    """Condition Gaussian link increments on closure, then fix the centroid."""
    masses = np.asarray(masses, float)
    increments = rng.normal(size=(beads, len(masses), 3))
    increments *= np.sqrt(beta / (beads * masses))[None, :, None]
    increments -= increments.mean(axis=0, keepdims=True)
    positions = np.cumsum(increments, axis=0)
    positions -= positions.mean(axis=0, keepdims=True)
    return positions


def spectrum():
    k = np.arange(BEADS // 2 + 1)
    omega = 2 * BEADS / BETA * np.sin(np.pi * k / BEADS)
    phase_per_step = 2 * np.arctan(0.5 * DT_FS / AU_TIME_FS * omega)
    attenuation = np.cos(50 * phase_per_step) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        variance_relaxation_fs = -5.0 / np.log(attenuation)
    return omega, phase_per_step, attenuation, variance_relaxation_fs


def standardized_modes(q, masses):
    omega = spectrum()[0]
    transformed = np.fft.rfft(q, axis=0, norm="ortho")
    # E[|q_hat_k|^2] = P / (beta*m*omega_k^2), also for the Nyquist mode.
    # The unconstrained translational centroid is excluded from this check.
    result = transformed[1:] * np.sqrt(BETA * np.asarray(masses)[None, :, None] / BEADS)
    result *= omega[1:, None, None]
    return result


def measure(q, masses):
    modes = standardized_modes(q, masses)
    ratios = np.mean(np.abs(modes) ** 2, axis=(1, 2))
    return ratios, modes


def simulate(method, atoms, targets_fs, seed):
    masses = np.full(atoms, MASS)
    dyn = Dynamics(FreePotential(), masses, BETA, BEADS, DT_FS / AU_TIME_FS,
                   seed=seed, native="required", threads=1)
    state = State(np.zeros((BEADS, atoms, 3)), dyn.sample_momenta())
    targets = {round(t / DT_FS): t for t in targets_fs}
    schedule_rng = np.random.default_rng(seed + 9000)
    phase = spectrum()[1][1:]
    last_collision = 0
    next_collision = 0
    missing_variance = np.ones(BEADS // 2)
    snapshots, raw = [], {}
    collision_intervals = []
    start = time.perf_counter()
    for step in range(max(targets)):
        if method != "pile-l" and step == next_collision:
            if step:
                missing_variance *= np.cos((step - last_collision) * phase) ** 2
            last_collision = step
            state.p[:] = dyn.sample_momenta()
            interval = 50 if method == "fixed-andersen" else int(schedule_rng.integers(25, 76))
            collision_intervals.append(interval)
            next_collision = step + interval
        if method == "pile-l":
            dyn.sample_step(state, gamma_centroid=0.01 * AU_TIME_FS, method="bcocb")
        else:
            dyn.step(state)
        if state.step in targets:
            ratios, modes = measure(state.q, masses)
            expected = (None if method == "pile-l" else
                        1 - missing_variance * np.cos((state.step - last_collision) * phase) ** 2)
            snapshots.append({"time_fs": targets[state.step],
                              "mode_variance_ratio": ratios.tolist(),
                              "conditional_free_map_prediction": None if expected is None else expected.tolist()})
            raw[f"{method}_{state.step}"] = modes
            print(json.dumps({"method": method, "time_fs": targets[state.step],
                              "mode37_variance_ratio": float(ratios[36]),
                              "mode13_variance_ratio": float(ratios[12]),
                              "elapsed_seconds": time.perf_counter() - start}), flush=True)
    return {"method": method, "seed": seed, "independent_particles": atoms,
            "independent_cartesian_components": 3 * atoms,
            "snapshot_statistics": snapshots, "collision_intervals_steps": collision_intervals,
            "elapsed_seconds": time.perf_counter() - start, "native_integrator": dyn.native}, raw


def pcn_check(seed=91425, particles=1024, alpha=0.35):
    rng = np.random.default_rng(seed)
    masses = np.tile([MASS, 2 * MASS], particles // 2)
    q = free_ring_draw(rng, masses)
    noise = free_ring_draw(rng, masses)
    proposal = np.sqrt(1 - alpha * alpha) * q + alpha * noise
    before, raw_before = measure(q, masses)
    after, raw_after = measure(proposal, masses)
    increments = np.roll(q, -1, axis=0) - q
    expected_increment_variance = BETA / (BEADS * masses) * (1 - 1 / BEADS)
    observed = np.mean(increments * increments, axis=(0, 2))
    return {"seed": seed, "independent_particles": particles, "alpha": alpha,
            "mass_pattern": "alternating 1H and twice the 1H mass",
            "maximum_centroid_residual_bohr": float(np.max(np.abs(proposal.mean(axis=0)))),
            "mean_link_variance_ratio": float(np.mean(observed / expected_increment_variance)),
            "mode_variance_before": before.tolist(), "mode_variance_after": after.tolist(),
            "scope": "Gaussian prior stationarity only; no H3 acceptance or rate calculation"}, {
                "pcn_prior_before": raw_before, "pcn_prior_after": raw_after}


def run(output, atoms=32):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    omega, phase, attenuation, relaxation = spectrum()
    rows = [{"mode": k, "omega_au": float(omega[k]),
             "phase_per_collision_over_pi": float(50 * phase[k] / np.pi),
             "variance_attenuation_per_collision": float(attenuation[k]),
             "variance_relaxation_fs": float(relaxation[k]),
             "collapsed_start_variance_ratio_after_500fs": float(1 - attenuation[k] ** 100),
             "collapsed_start_variance_ratio_after_1500fs": float(1 - attenuation[k] ** 300)}
            for k in range(1, BEADS // 2 + 1)]
    simulations, raw = [], {}
    for method, targets, seed in (("fixed-andersen", [500., 1500.], 91420),
                                  ("random-interval-andersen", [500., 1500.], 91421),
                                  ("pile-l", [500.], 91422)):
        result, observations = simulate(method, atoms, targets, seed)
        simulations.append(result)
        raw.update(observations)
    pcn, observations = pcn_check()
    raw.update(observations)
    np.savez_compressed(output / "free-mode-observations.npz", **raw)
    angular = []
    for label, inertia in (("relative orbital direction at R=30 bohr", 2*MASS/3*30**2),
                           ("free H2 orientation at r=1.40147 bohr", MASS*1.4014719755**2/2)):
        diffusion_per_fs = (5/AU_TIME_FS) / (2*BETA*inertia) / AU_TIME_FS
        angular.append({"observable": label, "inertia_au": inertia,
                        "diffusion_per_fs": diffusion_per_fs,
                        "P2_relaxation_fs": 1/(6*diffusion_per_fs)})
    sources = ["src/pyrpmd/dynamics.py", "src/pyrpmd/workflows/fes.py",
               "src/pyrpmd/bias.py", "benchmarks/h3/review_free_modes.py"]
    report = {"schema": "pyrpmd.h3.review-free-modes.v1",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "temperature_K": TEMPERATURE, "beads": BEADS, "timestep_fs": DT_FS,
              "mass_au": MASS, "mode_relaxation": rows,
              "simulations": simulations, "pcn_prior_check": pcn,
              "free_rotor_small_step_estimates": angular,
              "source_sha256": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
              "raw_file": "free-mode-observations.npz",
              "raw_sha256": hashlib.sha256((output/"free-mode-observations.npz").read_bytes()).hexdigest(),
              "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "scope": "Exact free-mode mechanism and actual production-Dynamics control; no quantitative attribution of the H3 rate discrepancy.",
              "caveats": ["Physical H3 forces can shift or couple free-mode frequencies.",
                           "Slow orbital orientation alone does not establish slow H2-relative orientation.",
                           "No production source or sampling protocol is modified."]}
    (output/"free-mode-review.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--particles", type=int, default=32)
    args = parser.parse_args()
    run(args.output, args.particles)
