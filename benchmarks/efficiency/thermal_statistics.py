#!/usr/bin/env python3
"""Harmonic finite-P energies and PILE-L mode-momentum statistics.

Synthetic oscillator validation; not a chemical or thermodynamic convergence
claim. Block standard errors account for temporal correlations. Normal-mode
momentum distribution tests use independent atomic coordinates in one frame.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from scipy.stats import kstest
from pyrpmd.native_ensemble import NativeEnsemble


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rng", choices=["mt19937_64", "philox-v1"], default="philox-v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    atoms, beads, beta, dt = 128, 8, 4., .01
    thermostat = {"mode": "pile-l", "gamma_centroid": 1.}
    if args.rng != "mt19937_64":
        thermostat["rng"] = args.rng
    engine = NativeEnsemble(np.zeros((beads, atoms, 3)), np.ones(atoms), beta, dt,
                             thermostat=thermostat, seed=1749, threads=args.threads,
                             potential={"kind": "harmonic", "k": 1.})
    engine.run(4000, stride=4000)
    samples = engine.run(40000, stride=10)["samples"]
    omega = 2*beads/beta*np.sin(np.pi*np.arange(beads)/beads)
    target = np.sum(1/(1+omega**2))/(2*beta)
    energies = {}
    for name, column in [("potential", 2), ("kinetic_centroid_virial", 4)]:
        values = samples[:, column]/(3*atoms)
        blocks = np.array_split(values, 40)
        error = np.std([block.mean() for block in blocks], ddof=1)/np.sqrt(len(blocks))
        energies[name] = {"mean_per_dof": float(values.mean()), "standard_error": float(error),
                          "finite_P_target_per_dof": float(target),
                          "accepted": bool(abs(values.mean()-target) < 6*error+.002*target)}
    # An independent larger ensemble supplies independent coordinates for KS.
    atoms = 2048
    independent = NativeEnsemble(np.zeros((beads, atoms, 3)), np.ones(atoms), beta, dt,
                                 thermostat=thermostat, seed=9049, threads=args.threads,
                                 potential={"kind": "harmonic", "k": 1.})
    independent.run(2000, stride=2000)
    momenta = independent.state()["p"]
    fft = np.fft.rfft(momenta, axis=0, norm="ortho")
    modes = [fft[0].real]
    for k in range(1, beads//2):
        modes.extend([np.sqrt(2)*fft[k].real, -np.sqrt(2)*fft[k].imag])
    modes.append(fft[-1].real)
    results = []
    for k, mode in enumerate(modes):
        values = mode.ravel()/np.sqrt(beads/beta)
        test = kstest(values, "norm")
        ratio = float(np.mean(values**2))
        results.append({"mode": k, "temperature_ratio": ratio, "ks_pvalue": float(test.pvalue),
                        "accepted": bool(test.pvalue > .001/beads and abs(ratio-1)<6*np.sqrt(2/len(values)))})
    output = {"rng": args.rng, "beads": beads, "beta_au": beta, "dt_au": dt,
              "energies": energies, "normal_modes": results,
              "accepted": all(row["accepted"] for row in energies.values()) and all(row["accepted"] for row in results),
              }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps(output))
    if not output["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
