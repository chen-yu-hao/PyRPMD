"""Independent continuous-time harmonic check of the published PIGLET fixture.

Solves the augmented coordinate/momentum/auxiliary Lyapunov equation directly;
no PyRPMD thermostat, propagator or estimator implementation enters the solve.
This checks matrix units and the fitted harmonic quantum response. It does not
prove anharmonic NPT, finite-timestep or sampling convergence.

Run: PYTHONPATH=src python benchmarks/condensed_phase/piglet_reference.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import solve_continuous_lyapunov

if __package__:
    from .models import HERE, piglet_parameters
else:
    from models import HERE, piglet_parameters


def harmonic_reference(beta=5.0, beta_omega_values=(0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 20.0)):
    """One Cartesian harmonic oscillator, P=4, mass-scaled q and p.

    For each real ring normal mode, the linear SDE is dX=-D X dt+B dW,
    with X=(q,p,s). Its stationary covariance solves D S+S D.T=B B.T.
    The momenta use theta=P/beta, so the input dimensionless covariance C
    enters B B.T as theta*(A C+C A.T). The oscillator restoring frequency
    includes the free-ring frequency; its position variance follows from the
    augmented solve, not from assuming the free-particle input covariance C.
    """
    scalar = np.asarray(beta)
    if scalar.dtype.kind not in "fiu" or scalar.shape != () or not np.isfinite(scalar) or scalar <= 0:
        raise ValueError("beta must be positive and finite")
    beta = float(beta)
    band = np.asarray(beta_omega_values)
    if (band.dtype.kind not in "fiu" or band.ndim != 1 or not len(band)
            or not np.isfinite(band).all() or np.any(band <= 0)):
        raise ValueError("beta_omega_values must be a positive finite vector")
    spec = piglet_parameters(beta)
    beads = spec["beads"]
    theta = beads / beta
    # Real mode order: centroid, cosine/sine pairs, then Nyquist for even P.
    labels = [0] + [k for k in range(1, (beads + 1) // 2) for _ in range(2)]
    if beads % 2 == 0:
        labels.append(beads // 2)
    ring_squared = (2 * beads / beta * np.sin(np.pi * np.asarray(labels) / beads))**2
    cases = []
    for dimensionless_frequency in band:
        frequency = float(dimensionless_frequency / beta)
        position_variances, residuals = [], []
        for mode, spring in enumerate(ring_squared):
            a = spec["A"][mode]
            c = theta * spec["C"][mode]
            size = len(a) + 1
            drift = np.zeros((size, size))
            diffusion = np.zeros_like(drift)
            drift[0, 1] = -1
            drift[1, 0] = frequency**2 + spring
            drift[1:, 1:] = a
            diffusion[1:, 1:] = a @ c + c @ a.T
            if np.min(np.real(np.linalg.eigvals(drift))) <= 0:
                raise ValueError("augmented harmonic GLE drift is unstable")
            covariance = solve_continuous_lyapunov(drift, diffusion)
            covariance = 0.5 * (covariance + covariance.T)
            if covariance[0, 0] <= 0 or not np.isfinite(covariance).all():
                raise ValueError("unphysical harmonic stationary covariance")
            residual = drift @ covariance + covariance @ drift.T - diffusion
            residuals.append(float(np.linalg.norm(residual) / max(1.0, np.linalg.norm(diffusion))))
            position_variances.append(float(covariance[0, 0]))
        exact = frequency / (4 * math.tanh(float(dimensionless_frequency) / 2))
        potential = 0.5 * frequency**2 * np.mean(position_variances)
        kinetic = 0.5 / beta + 0.5 * frequency**2 * np.sum(position_variances[1:]) / beads
        canonical = theta / (frequency**2 + ring_squared)
        primitive = 0.5 * frequency**2 * canonical.mean()
        cases.append({
            "beta_omega": float(dimensionless_frequency),
            "exact_potential_and_kinetic_hartree": exact,
            "piglet_potential_hartree": float(potential),
            "piglet_centroid_virial_kinetic_hartree": float(kinetic),
            "potential_relative_error": float(potential / exact - 1),
            "kinetic_relative_error": float(kinetic / exact - 1),
            "canonical_primitive_same_beads_relative_error": float(primitive / exact - 1),
            "maximum_normalized_lyapunov_residual": max(residuals),
        })
    source = json.loads((HERE / "piglet_4_750K.json").read_text())
    return {
        "schema": "PyRPMD.piglet-harmonic-reference.v1",
        "beta": beta, "beads": beads,
        "source_blob_sha": source["source_blob_sha"], "source_url": source["source_url"],
        "method": "continuous augmented harmonic GLE Lyapunov solve",
        "scope": "matrix units and harmonic fit on listed frequencies; no anharmonic/NPT/timestep/sampling convergence claim",
        "maximum_absolute_potential_relative_error": max(abs(row["potential_relative_error"]) for row in cases),
        "maximum_absolute_kinetic_relative_error": max(abs(row["kinetic_relative_error"]) for row in cases),
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(harmonic_reference(args.beta), indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)


if __name__ == "__main__":
    main()
