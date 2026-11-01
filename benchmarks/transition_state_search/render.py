"""Render measured TS-search and rate evidence, without rerunning dynamics.

    python benchmarks/transition_state_search/render.py \
      docs/validation/transition-state-search-2026-11-01

Requires the optional ``plots`` dependency. Figures are written next to the
JSON evidence as PNG and PDF, with the same limits and failure labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmarks.transition_state_search.analytic import MullerBrown


def save(figure, directory, name):
    for extension in ("png", "pdf"):
        metadata = {"CreationDate": None, "ModDate": None} if extension == "pdf" else None
        figure.savefig(directory / f"{name}.{extension}", dpi=180, metadata=metadata)
    plt.close(figure)


def saddle_figure(directory):
    analytic = json.loads((directory / "analytic.json").read_text())
    h3 = json.loads((directory / "h3.json").read_text())
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.7), layout="constrained")
    ax = axes[0]
    surface = MullerBrown()
    x, y = np.meshgrid(np.linspace(-1.6, 1.1, 220), np.linspace(-0.2, 2.0, 220))
    dx, dy = x[..., None] - surface.x0, y[..., None] - surface.y0
    energy = np.sum(surface.amplitude * np.exp(surface.a * dx**2 +
                    surface.b * dx * dy + surface.c * dy**2), axis=-1)
    ax.contour(x, y, energy, levels=np.arange(-140, 21, 20),
               colors="#b4bbc5", linewidths=0.6)
    for start, end, color, label in (
        ("SP1", "SP3", "#007f88", "Upper barrier: 13 images"),
        ("SP3", "SP2", "#bb6c12", "Lower barrier: 13 images"),
    ):
        case = next(c for c in analytic["neb"] if c["start"] == start and c["end"] == end
                    and c["images"] == 13 and c["tolerance"] == 1e-6
                    and c["initial_path"] == "linear")
        path = np.asarray(case["path"])[:, 0, :2]
        ax.plot(*path.T, "o-", color=color, markersize=3, linewidth=1.5, label=label)
    failed = next(c for c in analytic["neb"] if c["start"] == "SP3" and c["end"] == "SP2"
                  and c["images"] == 5 and c["tolerance"] == 1e-6)
    path = np.asarray(failed["path"])[:, 0, :2]
    ax.plot(*path.T, "x--", color="#ae325e", linewidth=1.4,
            label="5 images: unconverged at 1,000 steps")
    for point in analytic["references"]:
        if point["surface"] != "muller_brown":
            continue
        px, py = point["xy"]
        ax.plot(px, py, "*" if point["index"] == 1 else "o", color="#172b4d",
                markersize=9 if point["index"] == 1 else 5, zorder=5)
        ax.annotate(point["id"], (px, py), xytext=(6, 6), textcoords="offset points", fontsize=8)
    ax.set(xlabel="Model x", ylabel="Model y", xlim=(-1.6, 1.1), ylim=(-0.2, 2.0),
           title="Müller–Brown paths: successes and retained failure")
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.95)

    ax = axes[1]
    reference = h3["independent_reference"]
    for key, xkey, label, color, marker in (
        ("energy_curvature_convergence", "energy_difference_step_bohr",
         "Independent energy curvature", "#007f88", "o"),
        ("force_hessian_convergence", "force_difference_step_bohr",
         "Force-difference Hessian", "#bb6c12", "s"),
    ):
        points = reference[key]
        ax.semilogx([p[xkey] for p in points],
                    [p["imaginary_frequency_magnitude_cm_minus1"] for p in points],
                    marker=marker, color=color, label=label, markersize=4)
    ax.invert_xaxis()
    ax.grid(alpha=0.2)
    ax.set(xlabel="Finite-difference displacement / bohr",
           ylabel="Imaginary frequency magnitude / cm$^{-1}$",
           title="BKMP2 H₃: independent numerical frequency check")
    ax.legend(fontsize=8, loc="upper right")
    ax.text(0.34, 0.64, "Computed references; no published\n"
            "frequency value is asserted.\n\n"
            "Saddle geometry and barrier agree\n"
            "at the paper's printed precision.",
            transform=ax.transAxes, fontsize=8)
    save(figure, directory, "saddles")


def rate_figure(directory):
    path = directory / "rates.json"
    if not path.exists():
        return
    evidence = json.loads(path.read_text())
    cases = [c for c in evidence["cases"] if c.get("comparison", {}).get("rate_ratio") is not None]
    if not cases:
        return
    figure, ax = plt.subplots(figsize=(8.8, 4.7), layout="constrained")
    positions = np.arange(len(cases))
    ax.scatter(positions-0.08, [c["comparison"]["qtst_ratio"] for c in cases],
               label="QTST / corresponding paper QTST", marker="o", color="#007f88", s=45)
    ax.scatter(positions+0.08, [c["comparison"]["rate_ratio"] for c in cases],
               label="Diagnostic rate / corresponding paper rate", marker="s", color="#bb6c12", s=40)
    ax.axhline(1, color="#172b4d", linewidth=1, linestyle="--", label="Published central value")
    ax.set_yscale("log")
    ax.set_xticks(positions, [c["case"] for c in cases], rotation=12)
    ax.set(ylabel="Computed / published (log scale)",
           title="300 K H + H₂: finite-sampling comparisons")
    ax.grid(axis="y", which="both", alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    figure.supxlabel("P = 1 compares with the classical row; P = 128 with the RPMD row.\n"
                      "Central values only; seed and time-half disagreement preclude a convergence claim.",
                      fontsize=8)
    save(figure, directory, "rates")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_directory", type=Path)
    args = parser.parse_args()
    with plt.rc_context({"font.size": 9, "axes.titlesize": 10,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"}):
        saddle_figure(args.evidence_directory)
        rate_figure(args.evidence_directory)
