"""Exact quantum vs RPMD rate of collinear H + H2 (BKMP2) across the crossover temperature.

    python benchmarks/h3/build_pes.py           # once
    python benchmarks/h3_subspace/run.py quantum --temperature 200 --spacing 2.0
    python benchmarks/h3_subspace/run.py rpmd --temperature 200 --beads 64 --seed 1

Each call prints one JSON record.  The model is the whole collinear system (no
projection), so C = k_QM / k_RPMD is the RPMD method error itself.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from collinear import CS, CT, MASS, Collinear, to_st  # noqa: E402

from pyrpmd.analysis.subspace_model import model_from_function  # noqa: E402
from pyrpmd.analysis.subspace_quantum import exact_quantum_rate  # noqa: E402
from pyrpmd.analysis.subspace_rpmd import ring_polymer_rate  # noqa: E402

S_MAX = 110.0          # box half-length along s
T_RANGE = (70.0, 290.0)
REFERENCE = -63.0      # reactant slice: r1 - r2 = 3.6 bohr, V within 2e-5 Eh of the asymptote
ABSORB_FROM = 75.0     # |s| where the absorber starts
BARRIER = 0.015347     # classical barrier above H + H2, Eh
OMEGA_BARRIER = 1510.7  # cm^-1, imaginary frequency at the saddle
KB = 3.166811563e-6     # Hartree / K
TC = OMEGA_BARRIER * 1.4387769 / (2 * np.pi)  # crossover temperature, K


def model(spacing):
    potential = Collinear()
    axes = [np.arange(-S_MAX, S_MAX + 1e-9, spacing), np.arange(T_RANGE[0], T_RANGE[1] + 1e-9, spacing)]
    return model_from_function(potential, axes, gradient=potential.gradient, masses=1.0,
                               dividing_surface=0.0, reactant_reference=REFERENCE,
                               provenance={"system": "collinear H + H2, BKMP2", "spacing": spacing})


def h2_ground_energy(spacing=0.01):
    """Collinear H2 vibrational ground level (r2 = 12 bohr), relative to the classical asymptote."""
    potential = Collinear()
    r = np.arange(0.8, 4.0, spacing)
    s, t = to_st(r, 12.0)
    v = potential(np.stack([s, t], -1))
    n = len(r)
    i = np.arange(n)[:, None] - np.arange(n)[None, :]
    kinetic = np.where(i == 0, np.pi**2 / 3, 2.0 * (-1.0) ** i / np.where(i == 0, 1, i) ** 2)
    kinetic /= 2 * (MASS / 2) * spacing**2
    return float(np.linalg.eigvalsh(kinetic + np.diag(v))[0])


def absorber(scale):
    width = S_MAX - ABSORB_FROM
    def eps(points):
        x = np.clip((np.abs(points[..., 0]) - ABSORB_FROM) / width, 0.0, None)
        return scale * x**2
    return eps


def quantum(args):
    beta = 1.0 / (KB * args.temperature)
    sub = model(args.spacing)
    e0 = h2_ground_energy()
    width = S_MAX - ABSORB_FROM
    strength = args.absorber * 30.0 * np.sqrt(2.0 / (np.pi * beta)) / width
    start = time.time()
    result = exact_quantum_rate(
        sub, beta, method="sylvester", dense_limit=20000, absorber=absorber(strength), normalization="slice",
        energy_floor=e0 - args.floor_kt / beta,
        truncation_energy=e0 + max(args.truncation, BARRIER + 25.0 / beta))
    return {"stage": "quantum", "temperature": args.temperature, "spacing": args.spacing,
            "absorber_scale": args.absorber, "floor_kt": args.floor_kt, "h2_ground": e0,
            "k": result.rate, "diagnostics": result.diagnostics, "seconds": time.time() - start}


def rpmd(args):
    beta = 1.0 / (KB * args.temperature)
    sub = model(args.spacing)
    start = time.time()
    r = ring_polymer_rate(sub, beta, args.beads, windows=args.windows, walkers=args.walkers,
                          sweeps=args.sweeps, trajectories=args.trajectories, seed=args.seed)
    return {"stage": "rpmd", "temperature": args.temperature, "beads": args.beads, "seed": args.seed,
            "k": r.rate, "standard_error": r.rate_standard_error, "qtst": r.qtst_rate,
            "kappa": r.transmission, "kappa_error": r.transmission_standard_error,
            "diagnostics": r.diagnostics, "seconds": time.time() - start}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["quantum", "rpmd", "info"])
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--spacing", type=float, default=2.0)
    p.add_argument("--absorber", type=float, default=1.0)
    p.add_argument("--floor-kt", type=float, default=1.0)
    p.add_argument("--truncation", type=float, default=0.04)
    p.add_argument("--beads", type=int, default=32)
    p.add_argument("--windows", type=int, default=48)
    p.add_argument("--walkers", type=int, default=32)
    p.add_argument("--sweeps", type=int, default=400)
    p.add_argument("--trajectories", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.stage == "info":
        out = {"crossover_temperature": TC, "barrier": BARRIER, "h2_ground": h2_ground_energy(),
               "s_per_bohr": 1 / CS, "t_per_bohr": 1 / CT}
    else:
        out = (quantum if args.stage == "quantum" else rpmd)(args)
    print(json.dumps(out, default=float))


if __name__ == "__main__":
    main()
