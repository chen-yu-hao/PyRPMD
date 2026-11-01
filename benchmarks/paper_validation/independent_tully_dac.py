#!/usr/bin/env python3
"""Independently reproduce Liu et al. (JCP 161, 024102, 2024) DAC SOFT data.

Atomic units; Tully II Eq. (69-71), mass=2000, x0=-9, initial diabatic 0,
Gaussian sigma_x=sigma_p=1/sqrt(2), momentum p0=5 or 15.
No PyRPMD propagator or potential is imported in this reference program.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = {5: ("dac_p5_liu2024_reference.csv", 6000),
        15: ("dac_p15_liu2024_reference.csv", 2800)}
MASS = 2000.0


def load_reference(momentum):
    filename, end = DATA[momentum]
    path = HERE / filename
    with path.open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(line for line in stream if not line.startswith("#")))
    times = np.array([float(row["time"]) for row in records])
    state0 = np.array([float(row["sigma_0_0_real"]) for row in records])
    state1 = np.array([float(row["sigma_1_1_real"]) for row in records])
    if not np.array_equal(times, np.arange(0, end + 1, 40, dtype=float)):
        raise ValueError("published data window must consist of 40-au samples")
    if (not np.isfinite(state0 + state1).all()
            or np.max(np.abs(state0 + state1 - 1)) > 2e-7
            or min(state0.min(), state1.min()) < -2e-7):
        raise ValueError("published electronic population is not normalized")
    return path, times, state1


def propagate(momentum, *, points=1024, dt=2., limit=2800):
    steps = round(limit / dt)
    if (abs(steps * dt - limit) > 1e-9
            or abs(40 / dt - round(40 / dt)) > 1e-9):
        raise ValueError("dt must divide the horizon and 40-au observation stride")
    x = np.linspace(-15., 15., points, endpoint=False)
    dx = x[1] - x[0]
    matrix = np.zeros((points, 2, 2))
    matrix[:, 1, 1] = -0.1 * np.exp(-0.28 * x*x) + 0.05
    matrix[:, 0, 1] = matrix[:, 1, 0] = 0.015 * np.exp(-0.06 * x*x)
    energies, vectors = np.linalg.eigh(matrix)
    half = np.einsum("nsi,ni,nti->nst", vectors,
                     np.exp(-0.5j * dt * energies), vectors)
    k = 2 * np.pi * np.fft.fftfreq(points, d=dx)
    kinetic = np.exp(-0.5j * dt * k*k / MASS)
    wave = np.zeros((points, 2), dtype=complex)
    wave[:, 0] = np.pi**(-0.25) * np.exp(-0.5 * (x + 9)**2
                                          + 1j * momentum * (x + 9))
    wave /= np.sqrt(dx * np.sum(abs(wave)**2))
    values, norms = [], []
    stride = round(40 / dt)
    for step in range(steps + 1):
        if step % stride == 0:
            probability = np.abs(wave)**2
            values.append(float(dx * probability[:, 1].sum()))
            norms.append(float(dx * probability.sum()))
        if step == steps:
            break
        wave = np.einsum("nij,nj->ni", half, wave)
        wave = np.fft.ifft(np.fft.fft(wave, axis=0) * kinetic[:, None], axis=0)
        wave = np.einsum("nij,nj->ni", half, wave)
    return np.asarray(values), np.asarray(norms)


def calculate(momentum):
    path, times, reference = load_reference(momentum)
    _, limit = DATA[momentum]
    records, diagnostics = {}, {}
    for points, dt in ((1024, 2.), (1024, 1.), (2048, 2.), (2048, 1.)):
        key = f"grid{points}_dt{dt:g}"
        computed, norm = propagate(momentum, points=points, dt=dt, limit=limit)
        diff = computed - reference
        records[key] = computed.tolist()
        diagnostics[key] = {
            "max_abs_vs_published": float(np.max(abs(diff))),
            "rmse_vs_published": float(np.sqrt(np.mean(diff**2))),
            "max_norm_error": float(np.max(abs(norm - 1.)))}
    difference = np.asarray(records["grid2048_dt1"]) - np.asarray(records["grid1024_dt2"])
    grid_difference = np.asarray(records["grid2048_dt1"]) - np.asarray(records["grid1024_dt1"])
    return {
        "momentum": momentum, "reference_file": path.name,
        "reference_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "times_au": times.tolist(), "published_population1": reference.tolist(),
        "quantum_population1": records, "metrics": diagnostics,
        "refinement": {
            "max_abs_grid_and_timestep_difference": float(max(abs(difference))),
            "max_abs_grid_only_difference": float(max(abs(grid_difference)))},
        "pass_original_published_propagator_reproduction":
            diagnostics["grid1024_dt2"]["max_abs_vs_published"] < 3e-8,
        "pass_all_norms": all(v["max_norm_error"] < 1e-9
                              for v in diagnostics.values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("p5", "p15", "both"), default="both")
    parser.add_argument("--output", type=Path, default=HERE / "results" / "dac_quantum.json")
    args = parser.parse_args()
    momenta = (5, 15) if args.case == "both" else (int(args.case[1:]),)
    payload = {
        "format": "PyRPMD.paper.Tully.DAC.quantum.v1",
        "publication": "Liu et al., J. Chem. Phys. 161, 024102 (2024), DOI:10.1063/5.0208708",
        "upstream_commit": "3951a0a19008fb545ad17f0d2961d934633f7422",
        "initial_state": {"diabatic": 0, "x_bohr": -9.,
                          "sigma_x_bohr": 2**-0.5, "sigma_p_au": 2**-0.5,
                          "mass_me": MASS},
        "potential_hartree": {"v00": 0, "v11": "-0.10 exp(-0.28 x^2) + 0.05",
                              "v01": "0.015 exp(-0.06 x^2)"},
        "periodic_box_bohr": [-15, 15], "python": platform.python_version(),
        "numpy": np.__version__, "cases": {str(p): calculate(p) for p in momenta}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    overview = {
        p: {"max_abs_reference_error": c["metrics"]["grid1024_dt2"]["max_abs_vs_published"],
            "grid_refinement": c["refinement"],
            "passed": c["pass_original_published_propagator_reproduction"]}
        for p, c in payload["cases"].items()}
    print(json.dumps(overview, indent=2), flush=True)
    if not all(c["pass_original_published_propagator_reproduction"]
               and c["pass_all_norms"] for c in payload["cases"].values()):
        raise SystemExit("independent quantum reproduction failed")


if __name__ == "__main__":
    main()
