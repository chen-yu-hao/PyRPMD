"""Analytic references and independent-replica checks for the tutorial tier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import i0e

from pyrpmd.constants import ATOMIC_TIME_S, FS_TO_AU, KB_AU, WAVENUMBER_TO_AU

MASS = 1837.15264744
BETA = 1 / (300 * KB_AU)


def harmonic_reference(k=.02, mass=MASS, beta=BETA, beads=16):
    omega2 = k / mass
    ring = 2 * beads / beta * np.sin(np.pi * np.arange(beads) / beads)
    variance = 3 / (beta * mass) * np.sum(1 / (omega2 + ring**2))
    radius = variance - 3 / (beta * k)
    return {"potential_energy": .5 * k * variance,
            "kinetic_centroid_virial": 1.5 / beta + .5 * k * radius,
            "radius_of_gyration_squared_bohr2": radius}


def site_rate_reference(height=.006, length=4., mass=MASS, beta=BETA):
    # Integral over one basin: L exp(-z) I0(z). i0e avoids overflow.
    z = .5 * beta * height
    return np.exp(-2 * z) / (length * i0e(z) * np.sqrt(2 * np.pi * beta * mass) * ATOMIC_TIME_S)


def diffusion_reference(gamma_fs=.02, mass=MASS, beta=BETA):
    return 1 / (beta * mass * (gamma_fs / FS_TO_AU))


def spectroscopy_reference(k=.05, mass=MASS):
    return np.sqrt(k / mass) / WAVENUMBER_TO_AU


def check(name, measured, reference, standard_error, *, resolution=0., max_relative_error=None):
    valid = (measured is not None and standard_error is not None
             and np.isfinite([measured, reference, standard_error, resolution]).all()
             and standard_error >= 0 and resolution >= 0)
    difference = None if not valid else float(measured - reference)
    z = None if not valid or standard_error == 0 else difference / standard_error
    precise = valid and (max_relative_error is None or standard_error <= abs(reference) * max_relative_error)
    return {"observable": name, "measured": measured, "reference": reference,
            "standard_error": standard_error, "deviation_in_standard_errors": z,
            "resolution_bound": resolution, "difference": difference,
            "precision_target_relative_sem": max_relative_error,
            "passed": bool(precise and abs(difference) <= 3 * standard_error + resolution)}


def replica_check(name, values, reference, **options):
    values = np.asarray(values, float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("at least two finite independent replica estimates are required")
    return check(name, float(values.mean()), reference,
                 float(values.std(ddof=1) / np.sqrt(len(values))), **options)


def compare(name, run):
    run = Path(run)
    summary = json.loads((run / "results/summary.json").read_text())
    rows = []
    if name == "harmonic_pimd":
        if summary["status"] != "complete":
            raise ValueError("PIMD run is incomplete")
        reference = harmonic_reference()
        for key in ("potential_energy", "kinetic_centroid_virial"):
            observed = summary["estimators"][key]
            rows.append(check(key, observed["mean"], reference[key],
                              observed["standard_error_between_replicas"], max_relative_error=.03))
        # For this harmonic model K_cv=3/(2 beta)+(k/2) Rg^2, including each saved frame.
        rows.append(check("radius_of_gyration_squared_bohr2", summary["radius_of_gyration_squared_bohr2"][0],
                          reference["radius_of_gyration_squared_bohr2"],
                          summary["estimators"]["kinetic_centroid_virial"]["standard_error_between_replicas"] * 2 / .02,
                          max_relative_error=.03))
    elif name == "periodic_site_rate":
        rows.append(check("directed_rate_s", summary.get("rate_s"), site_rate_reference(),
                          summary.get("rate_standard_error_s"), max_relative_error=.15))
        if not summary.get("plateau_detected") or not summary.get("rate_uncertainty", {}).get("available"):
            rows[0]["passed"] = False
    elif name == "free_diffusion":
        data = json.loads((run / "results/diffusion.json").read_text())
        rows.append(check("diffusion_au", data["diffusion_au"], diffusion_reference(),
                          data["uncertainty"]["standard_error_au"], max_relative_error=.15))
    elif name == "double_well_metad":
        if summary["status"].startswith("partial"):
            raise ValueError("metadynamics run is incomplete")
        with np.load(run / "fes.npz", allow_pickle=False) as saved:
            grid = .5 * (saved["edges_0"][:-1] + saved["edges_0"][1:])
            curves = saved["free_energy_per_replica"]
            mask = saved["mask_per_replica"]
            points = [-1., 0., 1.]
            if any(np.any(mask[:, np.searchsorted(grid, point) - 1:np.searchsorted(grid, point) + 1]) for point in points):
                raise ValueError("barrier or either well lacks support in a replica")
            barriers = [np.interp(0., grid, curve) - .5 * (np.interp(-1., grid, curve) + np.interp(1., grid, curve)) for curve in curves]
            # Exact bin-averaged Boltzmann density, matching the histogram estimator.
            from scipy.integrate import quad
            edges = saved["edges_0"]
            density = [quad(lambda x: np.exp(-BETA * .004 * (x*x - 1)**2), a, b)[0] / (b-a)
                       for a, b in zip(edges[:-1], edges[1:])]
            free = -np.log(density) / BETA
            target = np.interp(0., grid, free) - .5 * (np.interp(-1., grid, free) + np.interp(1., grid, free))
            rows.append(replica_check("histogram_barrier_hartree", barriers, target, max_relative_error=.15))
            rows[0]["continuous_barrier_hartree"] = .004
    elif name == "harmonic_spectroscopy":
        if summary["status"] != "complete":
            raise ValueError("spectroscopy run is incomplete")
        with np.load(run / "results/ir-spectrum.npz", allow_pickle=False) as saved:
            grid = saved["wavenumber_cm1"]
            intensities = saved["per_replica"] * saved["angular_frequency_au"][None]**2
            peaks = grid[np.argmax(intensities, axis=1)]
            # The predetermined bound is half an FFT bin. No padding or fitted tolerance.
            resolution = float(np.diff(grid)[0] / 2)
            rows.append(replica_check("ir_peak_cm1", peaks, spectroscopy_reference(), resolution=resolution))
            rows[0]["peak_grid_spacing_cm1"] = 2 * resolution
    else:
        raise ValueError("unknown tutorial: " + name)
    return {"tutorial": name, "passed": all(row["passed"] for row in rows), "checks": rows,
            "scope": "Analytic model agreement at the specified finite bead count and budget; production summary is unchanged."}


def main(name):
    parser = argparse.ArgumentParser(description="Compare this tutorial to its analytic reference (3 SEM).")
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    report = compare(name, args.run)
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if report["passed"] else 1)
