"""Reproducible H + H2 state-selected scattering on the real BKMP2 surface.

This benchmark explicitly implements the *2021 triatomic* preparation in
Marjollet & Welsch, IJQC 121 e26447, DOI 10.1002/qua.26447.  It does not
change the general PyRPMD initial-ensemble defaults.  The QCT control uses
the same harmonic reference, not the external QCT data cited by that paper.

All integration steps, PES evaluations and routine outcome checks run in
the C++ propagation loop. Python schedules independent trajectories/blocks.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
from scipy.stats import beta as beta_distribution

from pyrpmd.constants import AMU_TO_AU, EV_TO_AU, FS_TO_AU, WAVENUMBER_TO_AU, BOHR_ANGSTROM
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.initial_states import prepare_initial_ensemble
from pyrpmd.workflows.drpmd import _rotation
from benchmarks.h3.drpmd import H3ExchangeClassifier
from benchmarks.h3.pes import BKMP2Potential

ROOT = Path(__file__).resolve().parents[2]
MASS = 1.00782503223 * AMU_TO_AU
BOND = 1.4014719755049045
OMEGA = 4407.0 * WAVENUMBER_TO_AU
_WORKER = None


def paper_beta(total_energy_ev, v):
    """2021 Eqs. 26/27 for m_A < m_BC; retain the *printed* minus sign."""
    total = total_energy_ev * EV_TO_AU
    collision = total - (v + 0.5) * OMEGA
    if collision <= 0:
        raise ValueError("Total energy must exceed the harmonic internal energy")
    beta0 = (2.0 / np.pi) * 4.5 / (collision + v * OMEGA)
    fraction = 0.01 * total
    if not 0 < fraction < OMEGA / 2:
        raise ValueError("Energy outside the positive printed beta-minus regime")
    lower = np.log((OMEGA - fraction) / fraction) / OMEGA
    return float(min(1500.0, max(beta0, lower)))


@dataclass(frozen=True)
class Config:
    method: str = "ne_rpmd"
    v: int = 0
    total_energy_ev: float = 0.90
    beads: int = 32
    samples: int = 2048
    seed: int = 20261031
    id_offset: int = 0
    bmin: float = 0.0
    bmax: float = 3.5
    dt_fs: float = 0.02
    time_cap_fs: float = 250.0
    extended_cap_fs: float = 1000.0
    initial_axial_bohr: float = 10.0
    exit_bohr: float = 10.0
    bond_cutoff_bohr: float = 2.5
    persistence_fs: float = 1.0
    check_fs: float = 0.2
    dmin_bohr: float = 1.078
    integrator: str = "normal-mode-bab"
    beta_override: float | None = None

    @property
    def beta(self):
        value = (paper_beta(self.total_energy_ev, self.v)
                 if self.beta_override is None else self.beta_override)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Auxiliary beta must be finite and positive")
        return float(value)

    @property
    def collision_energy(self):
        return self.total_energy_ev * EV_TO_AU - (self.v + 0.5) * OMEGA

    @property
    def label(self):
        return f"{self.method}-v{self.v}-Et{self.total_energy_ev:.3f}-P{self.beads}"


def setup(config):
    system = {"symbols": ["H"] * 3, "masses": [MASS] * 3,
              "fragments": [[0], [1, 2]],
              "geometry": [[0, 0, -12], [0, 0, -BOND / 2], [0, 0, BOND / 2]]}
    section = {"method": config.method,
               "fragments": [{}, {"j": 0, "vibrational_quanta": [config.v],
               "frequencies": ["4407 cm^-1"],
               "modes": [[0, 0, -1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)]]}]}
    if config.method == "ne_rpmd":
        section["beta"] = config.beta
    prepared = prepare_initial_ensemble(section, system, config.beads, config.beta)
    potential = BKMP2Potential()
    dynamics = Dynamics(potential, np.array(system["masses"]), config.beta,
                        config.beads, config.dt_fs * FS_TO_AU, native="required",
                        loop_backend="native", threads=1, block_steps=2048,
                        integrator=config.integrator)
    classifier = H3ExchangeClassifier(system, config.bond_cutoff_bohr, config.exit_bohr)
    return config, prepared, dynamics, classifier


def _init_worker(config):
    global _WORKER
    _WORKER = setup(config)


def prepare_paper_state(config, prepared, trajectory_id):
    """Condition the 2021 harmonic ensemble, then launch along laboratory z.

    BC has vibrational ring modes only. A has free internal ring modes along
    the incident direction only. Rejected centroid bonds are counted and
    resampled *before launch*, exactly as the paper's explicit conditioning.
    """
    sequence = np.random.SeedSequence([config.seed, config.v,
               round(config.total_energy_ev * 1_000_000),
               1 if config.method == "qct" else 2, config.beads, trajectory_id])
    initial_rng, launch_rng = map(np.random.default_rng, sequence.spawn(2))
    for attempts in range(1, 10001):
        state = prepared.sample(initial_rng)
        q, p = state.q, state.p
        # Equal H masses: remove the beadwise molecular COM, not vibration.
        q[:, 1:] -= q[:, 1:].mean(axis=1)[:, None, :]
        p[:, 1:] -= p[:, 1:].mean(axis=1)[:, None, :]
        q[:, 0] -= q[:, 0].mean(axis=0)
        p[:, 0] -= p[:, 0].mean(axis=0)
        q[:, 0, :2] = 0.0
        p[:, 0, :2] = 0.0
        bond = float(np.linalg.norm(q[:, 2].mean(axis=0) - q[:, 1].mean(axis=0)))
        if config.method == "qct" or bond >= config.dmin_bohr:
            break
    else:
        raise RuntimeError("Initial ensemble conditioning exhausted 10000 draws")
    rotation = _rotation(launch_rng)
    q[:, 1:] = q[:, 1:] @ rotation.T
    p[:, 1:] = p[:, 1:] @ rotation.T
    radii = np.sqrt(np.mean(np.sum((q - q.mean(axis=0)) ** 2, axis=2), axis=0))
    axial = config.initial_axial_bohr + radii[0] + max(radii[1:])
    b = math.sqrt(config.bmin ** 2 + (config.bmax ** 2 - config.bmin ** 2) * launch_rng.random())
    phi = launch_rng.uniform(0, 2 * np.pi)
    separation = np.array([b * np.cos(phi), b * np.sin(phi), axial])
    q[:, 0] -= 2 / 3 * separation
    q[:, 1:] += 1 / 3 * separation
    speed = math.sqrt(2 * config.collision_energy / (2 * MASS / 3))
    p[:, 0, 2] += MASS * 2 / 3 * speed
    p[:, 1:, 2] -= MASS / 3 * speed
    # Conservation/launch invariants apply to every trajectory, not just a test.
    if not np.allclose(p.mean(axis=0).sum(axis=0), 0, atol=2e-11):
        raise AssertionError("Nonzero launch centroid total momentum")
    relative_velocity = p[:, 0].mean(axis=0) / MASS - p[:, 1:].mean(axis=(0, 1)) / MASS
    if not np.isclose(MASS / 3 * relative_velocity @ relative_velocity,
                      config.collision_energy, rtol=2e-12, atol=1e-14):
        raise AssertionError("Incorrect launch collision energy")
    return state, {"preparation_draws": attempts, "initial_bond_bohr": bond,
                   "b_bohr": b, "initial_axial_bohr": float(axial),
                   "initial_rg_bohr": radii.tolist()}


def trajectory(trajectory_id):
    config, prepared, dynamics, classifier = _WORKER
    started = time.perf_counter()
    row = {"id": trajectory_id, "status": "failed", "channel": None}
    before = dict(dynamics.propagation_stats)
    try:
        state, launch = prepare_paper_state(config, prepared, trajectory_id)
        row.update(launch)
        initial = dynamics.energy(state) / config.beads
        maximum_drift = 0.0
        candidate = since = None
        status, channel, reason = "unresolved", None, "time_budget"
        cap = round(config.time_cap_fs / config.dt_fs)
        final_cap = round(config.extended_cap_fs / config.dt_fs)
        check = max(1, round(config.check_fs / config.dt_fs))
        persistence = round(config.persistence_fs / config.dt_fs)
        descriptor = classifier.native_outcome_descriptor()
        crossed_cap = False
        while state.step < final_cap and status == "unresolved":
            target = cap if state.step < cap else final_cap
            if state.step >= cap:
                crossed_cap = True
            count = min(2048, target - state.step)
            observe = (np.arange(state.step + 1, state.step + count + 1) % check == 0)
            observe[-1] = True
            block = dynamics.propagate_block(
                state, count, observe=observe, record_states=False,
                raise_on_error=False, reuse_forces=True, check_energy=True,
                outcome={"descriptor": descriptor, "check_interval": check,
                         "persistence_steps": persistence, "start_step": state.step,
                         "candidate": candidate, "candidate_since": since})
            if block.backend != "native":
                raise AssertionError("A trajectory left the native propagation path")
            if len(block.energies):
                maximum_drift = max(maximum_drift, float(np.max(np.abs(block.energies / config.beads - initial))))
            if block.error:
                status, reason = "failed", str(block.error)
                break
            event = block.outcome
            candidate, since = event["candidate"], event["candidate_since"]
            if event["resolved"] is not None:
                status, channel = event["resolved"]
                reason = "stable_separated_outgoing"
            elif event.get("pending_check"):
                label = classifier(state)
                label = None if label is None else (label["status"], label["channel"])
                if label != candidate:
                    candidate, since = label, state.step if label else None
                if label is not None and state.step - since >= persistence:
                    status, channel = label
                    reason = "stable_separated_outgoing"
        row.update(status=status, channel=channel, reason=reason,
                   time_fs=float(state.time / FS_TO_AU), steps=int(state.step),
                   extended_beyond_initial_cap=crossed_cap,
                   max_observed_energy_drift_ev=maximum_drift / EV_TO_AU,
                   initial_ring_hamiltonian_per_bead_ev=initial / EV_TO_AU,
                   final_diagnostics=classifier.diagnostics(state))
        if status in ("unresolved", "failed"):
            row["endpoint_q_bohr"] = state.q.tolist()
            row["endpoint_p_au"] = state.p.tolist()
            row["candidate"] = candidate
            row["candidate_since"] = since
    except Exception as exc:
        row.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
    row["native_steps"] = dynamics.propagation_stats["native_steps"] - before["native_steps"]
    row["python_steps"] = dynamics.propagation_stats["python_steps"] - before["python_steps"]
    row["wall_seconds"] = time.perf_counter() - started
    return row


def clopper_pearson(k, n, alpha=0.05):
    return [0.0 if k == 0 else float(beta_distribution.ppf(alpha / 2, k, n - k + 1)),
            1.0 if k == n else float(beta_distribution.ppf(1 - alpha / 2, k + 1, n - k))]


def summarize(config, rows):
    n = len(rows)
    counts = {s: sum(r["status"] == s for r in rows) for s in ("product", "nonreactive", "unresolved", "failed")}
    k = counts["product"]
    unknown = counts["unresolved"] + counts["failed"]
    area = np.pi * (config.bmax ** 2 - config.bmin ** 2)
    ci = clopper_pearson(k, n)
    drifts = [r["max_observed_energy_drift_ev"] for r in rows if "max_observed_energy_drift_ev" in r]
    times = [r["time_fs"] for r in rows if "time_fs" in r]
    draws = sum(r.get("preparation_draws", 0) for r in rows)
    return {"config": asdict(config), "beta_au": config.beta,
            "collision_energy_ev": config.collision_energy / EV_TO_AU,
            "omega_au": OMEGA, "omega_cm-1": 4407.0,
            "counts": counts, "launched_attempts": n,
            "preparation_draws": draws, "preparation_rejections": draws - sum("preparation_draws" in r for r in rows),
            "area_bohr2": area, "sigma_bohr2": area * k / n,
            "sigma_angstrom2": area * k / n * BOHR_ANGSTROM ** 2,
            "sigma_confirmed_ci95_bohr2": [area * x for x in ci],
            "sigma_unknown_bounds_bohr2": [area * k / n, area * (k + unknown) / n],
            "sigma_ci95_including_unknown_bohr2": [area * ci[0], area * clopper_pearson(k + unknown, n)[1]],
            "max_observed_energy_drift_ev": max(drifts, default=None),
            "energy_drift_quantiles_ev": np.quantile(drifts, [0.5, 0.95, 0.99]).tolist() if drifts else [],
            "flight_time_quantiles_fs": np.quantile(times, [0.5, 0.9, 0.99, 1]).tolist() if times else [],
            "extended_trajectories": sum(r.get("extended_beyond_initial_cap", False) for r in rows),
            "native_steps": sum(r["native_steps"] for r in rows),
            "python_steps": sum(r["python_steps"] for r in rows),
            "trajectory_worker_seconds": sum(r["wall_seconds"] for r in rows)}


def run_case(config, workers, output):
    script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ledger = output / "trajectories.jsonl"
    if ledger.exists():
        raise FileExistsError(f"Refusing to overwrite completed/partial trajectory ledger: {ledger}")
    started = time.perf_counter()
    rows = []
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    ids = range(config.id_offset, config.id_offset + config.samples)
    with ledger.open("x") as stream:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_init_worker, initargs=(config,)) as pool:
            futures = [pool.submit(trajectory, i) for i in ids]
            for future in as_completed(futures):
                row = future.result()
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                rows.append(row)
                if len(rows) % max(32, config.samples // 8) == 0:
                    stream.flush()
                    print(json.dumps({"case": config.label, "complete": len(rows),
                                      "total": config.samples, "seconds": round(time.perf_counter() - started, 2)}), flush=True)
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != script_sha256:
        raise RuntimeError("Driver source changed during this run; trajectory ledger retained without a final summary")
    report = summarize(config, rows)
    report["elapsed_seconds"] = time.perf_counter() - started
    report["workers"] = workers
    report["provenance"] = {
        "paper_doi": "10.1002/qua.26447", "paper_figures": "5B (v=0), 6B (v=1)",
        "method": "2021 triatomic harmonic preparation; explicit NE-RPMD centroid bond conditioning",
        "qct_control": "unconditioned harmonic action initialization; not the paper's externally sourced QCT curve",
        "mass_amu": 1.00782503223, "bond_bohr": BOND,
        "python": platform.python_version(), "numpy": np.__version__,
        "seed_rule": "SeedSequence([seed,v,round(Etot_eV*1e6),method_code,P,trajectory_id]).spawn(2)",
        "script_sha256": script_sha256,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "pes": json.loads((ROOT / "benchmarks/h3/build/libbkmp2.build.json").read_text()),
        "unknown_rule": "Unresolved and failed attempts remain in denominator; identified bounds reported",
        "ci": "Two-sided exact Clopper-Pearson binomial 95%; does not include systematic errors",
        "energy_drift": "H_P/P deviation observed every check_fs and at block endpoints; not primitive energy",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("counts", "sigma_bohr2", "sigma_confirmed_ci95_bohr2", "elapsed_seconds", "python_steps")}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["ne_rpmd", "qct"], default="ne_rpmd")
    parser.add_argument("--v", type=int, choices=[0, 1], default=0)
    parser.add_argument("--etot", type=float, nargs="+", required=True)
    parser.add_argument("--beads", type=int, default=32)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--id-offset", type=int, default=0)
    parser.add_argument("--bmin", type=float, default=0.0)
    parser.add_argument("--bmax", type=float, default=3.5)
    parser.add_argument("--dt-fs", type=float, default=0.02)
    parser.add_argument("--time-cap-fs", type=float, default=250.0)
    parser.add_argument("--extended-cap-fs", type=float, default=1000.0)
    parser.add_argument("--initial-axial-bohr", type=float, default=10.0)
    parser.add_argument("--exit-bohr", type=float, default=10.0)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--integrator", choices=["normal-mode-bab", "cayley-bcb", "velocity-verlet"], default="normal-mode-bab")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.method == "qct":
        args.beads = 1
    if not 0 <= args.bmin < args.bmax or args.samples < 1 or args.workers < 1:
        parser.error("Require 0 <= bmin < bmax, samples >= 1, workers >= 1")
    if not 0 < args.dt_fs <= args.time_cap_fs <= args.extended_cap_fs:
        parser.error("Require 0 < dt <= time cap <= extended cap")
    if args.method == "ne_rpmd" and args.beads < 2:
        parser.error("NE-RPMD requires at least two beads")
    for energy in args.etot:
        config = Config(method=args.method, v=args.v, total_energy_ev=energy,
                        beads=args.beads, samples=args.samples, seed=args.seed,
                        id_offset=args.id_offset, bmin=args.bmin, bmax=args.bmax,
                        dt_fs=args.dt_fs, time_cap_fs=args.time_cap_fs,
                        extended_cap_fs=args.extended_cap_fs,
                        initial_axial_bohr=args.initial_axial_bohr, exit_bohr=args.exit_bohr,
                        integrator=args.integrator, beta_override=args.beta)
        run_case(config, args.workers, Path(args.output) / config.label)


if __name__ == "__main__":
    main()
