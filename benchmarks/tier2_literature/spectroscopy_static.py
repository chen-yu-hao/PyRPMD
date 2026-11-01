#!/usr/bin/env python3
"""Independent finite-P static reference for the Cao--Voth quartic benchmark.

Run spectroscopy.py first, then this script with the same --output directory.
This quadrature transfer matrix represents the actual primitive finite-P
configuration measure, without any MD stepper or force-matching machinery.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import eigh
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits


def finite_bead_covariance(beads, *, beta=5., points=401, extent=8., field=.01, states=16):
    """Centroid and ordinary bead variance from a symmetric transfer matrix.

    T_f(x,y)=dx*(2*pi*tau)^(-1/2)*exp[-(x-y)^2/(2*tau)] *
             exp[-tau*(V(x)+V(y))/2 + tau*f*(x+y)/2].
    Z(f)=Tr(T_f^P), and Var(q_centroid)=d²logZ/df²/beta².
    The second derivative uses symmetric fields and Richardson extrapolation.
    Returning two extrapolations exposes residual field-size error.
    """
    x = np.linspace(-extent, extent, points)
    dx, tau = x[1] - x[0], beta / beads
    physical = .5 * x * x + .1 * x ** 4
    kinetic = dx / np.sqrt(2 * np.pi * tau) * np.exp(-(x[:, None] - x[None, :]) ** 2 / (2 * tau))

    def partition(force):
        if beads == 1:
            # Tr(T) is its diagonal sum. At P=1 the narrow Boltzmann factor
            # makes high transfer eigenvalues numerically unresolved, and no
            # eigendecomposition is needed for this exactly classical case.
            log_weights = -beta * (physical - force * x)
            log_z = logsumexp(log_weights)
            bead_q2 = float(np.exp(log_weights - log_z) @ (x * x))
            return float(log_z + np.log(dx / np.sqrt(2 * np.pi * beta))), bead_q2
        endpoints = np.exp(-.5 * tau * (physical - force * x))
        transfer = endpoints[:, None] * kinetic * endpoints[None]
        eigenvalues, eigenvectors = eigh(transfer, subset_by_index=(points - states, points - 1))
        if np.any(eigenvalues <= 0):
            raise RuntimeError("retained transfer eigenvalues must be positive")
        log_weights = beads * np.log(eigenvalues)
        log_z = logsumexp(log_weights)
        normalized = np.exp(log_weights - log_z)
        bead_q2 = float(normalized @ (eigenvectors ** 2).T @ (x * x))
        return float(log_z), bead_q2

    zero, q2 = partition(0.)
    estimates = []
    symmetry = []
    for h in (field, field / 2, field / 4):
        plus, _ = partition(h)
        minus, _ = partition(-h)
        estimates.append((plus - 2 * zero + minus) / (h * h * beta * beta))
        symmetry.append(abs(plus - minus))
    first = (4 * estimates[1] - estimates[0]) / 3
    second = (4 * estimates[2] - estimates[1]) / 3
    return {"beads": beads, "grid_points": points, "extent": extent,
            "retained_transfer_states": None if beads == 1 else states,
            "centroid_variance": second, "bead_variance": q2,
            "field_extrapolation_change": abs(second - first), "field_symmetry_error": max(symmetry),
            "log_partition": zero}


def run(output):
    output = Path(output)
    with (output / "exact-basis-convergence.csv").open() as stream:
        reference = list(csv.DictReader(stream))[-1]
    c0, q2 = float(reference["c0_kubo"]), float(reference["q2_ordinary"])
    start = time.perf_counter()
    rows = []
    for beads in (1, 8, 16, 32, 64, 128):
        for points in (301, 501):
            result = finite_bead_covariance(beads, points=points)
            result.update(exact_centroid_variance=c0, exact_bead_variance=q2,
                          centroid_relative_error=result["centroid_variance"] / c0 - 1,
                          bead_relative_error=result["bead_variance"] / q2 - 1)
            rows.append(result)
    with (output / "finite-bead-static-reference.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    selected = [row for row in rows if row["grid_points"] == 501 and row["beads"] > 1]
    fine = [row for row in selected if row["beads"] >= 16]
    exponent = float(np.polyfit(np.log([row["beads"] for row in fine]),
                               np.log([abs(row["centroid_relative_error"]) for row in fine]), 1)[0])
    report = {"status": "complete", "method": "symmetric primitive transfer-matrix quadrature",
        "physical_units": "same reduced atomic units as the Cao-Voth model", "beta": 5.,
        "potential": "q^2/2 + 0.1*q^4", "field": .01,
        "centroid_bead_error_log_log_slope_P16_through_P128": exponent,
        "max_grid_change_centroid_variance": max(abs(rows[i]["centroid_variance"] - rows[i + 1]["centroid_variance"]) for i in range(0, len(rows), 2)),
        "max_field_extrapolation_change": max(row["field_extrapolation_change"] for row in rows),
        "runtime_seconds": time.perf_counter() - start,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "note": "This independently evaluates the finite-P measure; it is not another PA-CMD trajectory."}
    (output / "finite-bead-static-reference.json").write_text(json.dumps(report, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    p = np.asarray([row["beads"] for row in selected])
    centroid = np.abs([row["centroid_relative_error"] for row in selected])
    bead = np.abs([row["bead_relative_error"] for row in selected])
    ax.loglog(p, centroid, "o-", label="Centroid variance / exact Kubo C(0)")
    ax.loglog(p, bead, "s-", label="Bead variance / exact ordinary ⟨q²⟩")
    ax.loglog(p, centroid[-1] * (p[-1] / p) ** 2, ":", color="gray", label="P⁻² reference slope")
    ax.set(xlabel="Beads", ylabel="Absolute relative discretization error",
           title="Cao–Voth quartic, β=5: independent transfer-matrix reference")
    ax.legend(fontsize=9)
    fig.savefig(output / "bead-discretization.png", dpi=180)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output)
