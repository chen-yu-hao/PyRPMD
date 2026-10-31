#!/usr/bin/env python3
"""Reproducible RPMD/TRPMD harmonic spectrum and centroid invariance check.

Run from the repository root, for example:
  PYTHONPATH=src python benchmarks/trpmd/harmonic_spectrum.py --output /tmp/trpmd.json

This is an analytic mechanical/estimator benchmark, not a molecular PES or
published reaction-rate reproduction. All numerical parameters use atomic units.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from pyrpmd.analysis.spectroscopy import (
    centroid_velocities, velocity_autocorrelation, relative_ir_spectrum,
)
from pyrpmd.dynamics import Dynamics, State, native_capabilities
from pyrpmd.potentials import HarmonicPotential


def run(replicas=4, steps=2048, beads=8, dt=.03):
    mass, beta, omega = 1.5, 4., .7
    rng = np.random.default_rng(3071)
    eye = np.eye(beads)
    laplacian = 2 * eye - np.roll(eye, 1, axis=0) - np.roll(eye, -1, axis=0)
    values, vectors = np.linalg.eigh(laplacian)
    frequencies = np.sqrt(omega ** 2 + (beads / beta) ** 2 * np.maximum(values, 0))
    initials = []
    for _ in range(replicas):
        qk = rng.normal(size=(beads, 1, 3)) * np.sqrt(
            beads / (beta * mass * frequencies[:, None, None] ** 2)
        )
        pk = rng.normal(size=(beads, 1, 3)) * np.sqrt(mass * beads / beta)
        initials.append(State(np.einsum("bk,kac->bac", vectors, qk),
                              np.einsum("bk,kac->bac", vectors, pk)))
    records, reference = [], None
    for friction in (0., .5, 1.):
        start = time.perf_counter()
        momenta, dipoles = [], []
        for replica, initial in enumerate(initials):
            state = initial.copy()
            dynamics = Dynamics(
                HarmonicPotential(mass * omega ** 2), [mass], beta, beads, dt,
                seed=1000 + replica, integrator="normal-mode-bab",
            )
            p_record = np.empty((steps, beads, 1, 3))
            mu_record = np.empty((steps, beads, 3))
            for step in range(steps):
                dynamics.trpmd_step(state, friction_scale=friction)
                p_record[step] = state.p
                # The test dipole is linear, mu(q)=q for this one-atom model.
                mu_record[step] = state.q[:, 0]
            momenta.append(p_record)
            dipoles.append(mu_record)
        momenta = np.asarray(momenta)
        velocities = centroid_velocities(momenta, [mass])
        if reference is None:
            reference = velocities.copy()
        vacf = velocity_autocorrelation(momenta, [mass], dt, max_lag=steps // 4)
        ir = relative_ir_spectrum(np.asarray(dipoles), dt, beta=beta, window="hann")
        peak = int(np.argmax(ir["relative_ir"]))
        observed = float(ir["angular_frequency_au"][peak])
        centroid_error = float(np.max(np.abs(velocities - reference)))
        resolution = 2 * np.pi * float(ir["resolution_au"])
        records.append({
            "friction_scale": friction,
            "elapsed_seconds": time.perf_counter() - start,
            "peak_angular_frequency_au": observed,
            "exact_angular_frequency_au": omega,
            "angular_frequency_resolution_au": resolution,
            "peak_within_one_record_bin": abs(observed - omega) <= resolution,
            "max_centroid_velocity_difference_from_rpmd": centroid_error,
            "centroid_matches_rpmd": centroid_error < 2e-10,
            "vacf_zero_lag": float(vacf["correlation"][0]),
            "exact_ensemble_vacf_zero_lag": 3 / (beta * mass),
            "vacf_zero_lag_replica_standard_error": float(vacf["standard_error"][0]) if replicas > 1 else None,
        })
    return {
        "scope": "analytic harmonic mechanics and finite-record spectra; not physical-rate validation",
        "equilibrium_initialization": "independent exact Gaussian ring-polymer harmonic Gibbs samples",
        "beads": beads, "replicas": replicas, "steps": steps, "dt_au": dt,
        "mass_au": mass, "beta_au": beta, "native": native_capabilities(),
        "results": records,
        "passed": all(x["peak_within_one_record_bin"] and x["centroid_matches_rpmd"] for x in records),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--beads", type=int, default=8)
    args = parser.parse_args()
    if args.replicas < 1 or args.steps < 128 or args.beads < 1:
        parser.error("replicas and beads must be positive; steps must be >=128")
    result = run(args.replicas, args.steps, args.beads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
