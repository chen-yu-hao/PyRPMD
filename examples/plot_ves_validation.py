"""Plot saved T24 stochastic VES evidence without running any simulation.

The input is the JSON report from ``examples/ves_validation.py``, including
the exact bin edges for each reference case. The figure displays all four
dimension/target combinations, preserves their saved support and shared
reference bin, and uses a common color scale for the two 2D residual maps.

Example::

    python examples/plot_ves_validation.py \
        docs/validation/t24-ves/reference.json \
        --output docs/validation/t24-ves/stochastic_validation.png \
        --pdf docs/validation/t24-ves/stochastic_validation.pdf

VES bands are one standard error of the independent replica mean, as recorded
by the validation harness. MBAR error bars are the saved bootstrap standard
errors. Neither is a statement of physical convergence. This script requires
Matplotlib only for plotting; it does not import or execute the VES sampler.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np


_ARRAYS = (
    "reference_free_energy", "ves_mean", "ves_sem", "mbar_free_energy", "mbar_sem"
)
_TARGETS = ("uniform", "well_tempered")
_LABELS = {"uniform": "uniform", "well_tempered": "well-tempered"}
_VES = "#007f79"
_MBAR = "#c96a19"
_EXACT = "#263442"
_EXCLUDED = "#e1e5e9"


def read_cases(path):
    """Validate saved grids, uncertainty and masks before producing a figure."""
    report = json.loads(Path(path).read_text())
    cases = {}
    for item in report["reference_cases"]:
        dimension, target = item["dimension"], item["target"]
        if dimension not in (1, 2) or target not in _TARGETS:
            raise ValueError("expected one- and two-dimensional uniform/well_tempered cases")
        key = (dimension, target)
        if key in cases:
            raise ValueError("duplicate dimension/target reference case")
        if item["beta"] != 1.0:
            raise ValueError("this figure documents beta=1 toy-model validation only")
        edges = [np.asarray(edge, dtype=float) for edge in item["edges"]]
        if len(edges) != dimension or any(
            edge.ndim != 1 or len(edge) < 3 or not np.isfinite(edge).all()
            or np.any(np.diff(edge) <= 0) for edge in edges
        ):
            raise ValueError("reference case has invalid bin edges")
        shape = tuple(len(edge) - 1 for edge in edges)
        arrays = {name: np.asarray(item[name], dtype=float) for name in _ARRAYS}
        if any(array.shape != shape for array in arrays.values()):
            raise ValueError("reference arrays disagree with saved bin edges")
        support = np.asarray(item["support"])
        if support.dtype.kind != "b" or support.shape != shape or not support.any():
            raise ValueError("reference case requires a boolean, nonempty support mask")
        if any(not np.isfinite(array[support]).all() for array in arrays.values()):
            raise ValueError("supported values and errors must be finite")
        if any(np.any(arrays[name][support] < 0) for name in ("ves_sem", "mbar_sem")):
            raise ValueError("standard errors must be nonnegative")
        reference = tuple(item["reference_bin"])
        if len(reference) != dimension or any(
            type(index) is not int or not 0 <= index < count
            for index, count in zip(reference, shape)
        ):
            raise ValueError("invalid shared gauge reference")
        if not all(np.isfinite(array[reference]) for array in arrays.values()):
            raise ValueError("shared gauge reference must have finite saved values")
        residual = arrays["ves_mean"] - arrays["reference_free_energy"]
        rmse = float(np.sqrt(np.mean(residual[support] ** 2)))
        if not np.isclose(rmse, item["ves_quadrature_rmse_hartree"], rtol=1e-11, atol=1e-14):
            raise ValueError("saved RMSE disagrees with the plotted residual/support")
        cases[key] = {**item, **arrays, "edges": edges, "support": support,
                      "reference_bin": reference, "residual": residual, "rmse": rmse}
    if set(cases) != {(dimension, target) for dimension in (1, 2) for target in _TARGETS}:
        raise ValueError("the figure requires all four saved reference cases")
    return cases


def _style_axis(axis):
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        axis.spines[side].set_color("#aeb6bf")
    axis.tick_params(colors="#354150", length=3)
    axis.set_axisbelow(True)


def make_figure(cases):
    """Return the figure; all values, errors and support come from saved data."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5, "axes.labelsize": 11,
        "axes.titlesize": 11.5, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    figure = plt.figure(figsize=(11.3, 9.0))
    grid = figure.add_gridspec(2, 2, left=.09, right=.89, bottom=.19, top=.805,
                              hspace=.52, wspace=.28, height_ratios=(1., 1.12))
    axes = np.asarray([[figure.add_subplot(grid[row, col]) for col in range(2)]
                       for row in range(2)])
    figure.suptitle("VES: independent stochastic validation", x=.09, y=.972,
                     ha="left", fontsize=19, weight="semibold", color=_EXACT)
    replicas = {case["replicas"] for case in cases.values()}
    count = str(next(iter(replicas))) if len(replicas) == 1 else "independent"
    figure.text(.09, .934,
                rf"$\beta = 1\;\mathrm{{hartree}}^{{-1}}$ toy models"
                f"  |  {count} replica optimizations per case  |  bin-integrated quadrature",
                color="#52606f", fontsize=11)

    handles = [
        Line2D([], [], color=_EXACT, lw=1.7, label="Quadrature"),
        Line2D([], [], color=_VES, marker="o", markersize=3.5, lw=1.6,
               label="VES mean ± 1 replica SEM"),
        Line2D([], [], color=_MBAR, marker="s", markerfacecolor="white",
               markersize=4.3, lw=0, label="Umbrella/MBAR ± 1 bootstrap SE"),
    ]
    figure.legend(handles=handles, loc="upper left", bbox_to_anchor=(.083, .906),
                  frameon=False, ncols=3, handlelength=2.0, columnspacing=1.6,
                  fontsize=10)

    for col, target in enumerate(_TARGETS):
        case = cases[(1, target)]
        axis = axes[0, col]
        edge = case["edges"][0]
        centers = (edge[:-1] + edge[1:]) / 2
        support = case["support"]
        reference = case["reference_bin"][0]
        for index in np.flatnonzero(~support):
            if index != reference:
                axis.axvspan(edge[index], edge[index + 1], facecolor=_EXCLUDED,
                             hatch="///", edgecolor="#c0c7cf", linewidth=0, zorder=0)
        axis.plot(centers, case["reference_free_energy"], color=_EXACT, lw=1.7, zorder=2)
        mean = np.where(support, case["ves_mean"], np.nan)
        sem = case["ves_sem"]
        axis.fill_between(centers, mean - sem, mean + sem, color=_VES, alpha=.22,
                          linewidth=0, zorder=3)
        axis.plot(centers, mean, color=_VES, marker="o", ms=3.1, lw=1.3, zorder=4)
        axis.errorbar(centers[support], case["mbar_free_energy"][support],
                      yerr=case["mbar_sem"][support], fmt="s", ms=3.6,
                      color=_MBAR, markerfacecolor="white", markeredgewidth=.9,
                      capsize=2, elinewidth=.9, zorder=5)
        axis.plot(centers[reference], case["reference_free_energy"][reference],
                  marker="x", color=_EXACT, markersize=6, markeredgewidth=1.2, zorder=6)
        axis.set_xlim(edge[0], edge[-1])
        axis.set_xlabel(r"$x$ (bohr)")
        axis.set_ylabel(r"$F - F_{\rm ref}$ (hartree)")
        axis.grid(axis="y", color="#e6eaee", linewidth=.7)
        axis.set_title(f"{'ab'[col]}  1D · {_LABELS[target]}\n"
                       f"VES−quadrature RMSE: {case['rmse']:.4f} hartree",
                       loc="left", pad=11)
        _style_axis(axis)

    maximum = max(float(np.max(np.abs(cases[(2, target)]["residual"]
                                     [cases[(2, target)]["support"]])))
                  for target in _TARGETS)
    if maximum == 0:
        maximum = np.finfo(float).eps
    norm = TwoSlopeNorm(vmin=-maximum, vcenter=0., vmax=maximum)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(_EXCLUDED)
    mesh = None
    for col, target in enumerate(_TARGETS):
        case = cases[(2, target)]
        axis = axes[1, col]
        xedge, yedge = case["edges"]
        residual = np.ma.array(case["residual"], mask=~case["support"])
        mesh = axis.pcolormesh(xedge, yedge, residual.T, cmap=cmap, norm=norm,
                              shading="flat", rasterized=True)
        reference = case["reference_bin"]
        axis.plot((xedge[reference[0]] + xedge[reference[0] + 1]) / 2,
                  (yedge[reference[1]] + yedge[reference[1] + 1]) / 2,
                  marker="x", color=_EXACT, markersize=6, markeredgewidth=1.2)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(r"$x$ (bohr)")
        axis.set_ylabel(r"$y$ (bohr)")
        axis.set_title(f"{'cd'[col]}  2D · {_LABELS[target]}\n"
                       f"VES−quadrature RMSE: {case['rmse']:.4f} hartree",
                       loc="left", pad=11)
        _style_axis(axis)

    position = axes[1, 1].get_position()
    color_axis = figure.add_axes([.918, position.y0, .017, position.height])
    colorbar = figure.colorbar(mesh, cax=color_axis)
    colorbar.set_label(r"$F_{\rm VES} - F_{\rm quadrature}$ (hartree)", fontsize=10.5)
    colorbar.ax.tick_params(labelsize=9)
    colorbar.outline.set_edgecolor("#aeb6bf")

    excluded = sum(int(np.sum(~case["support"])) - int(not case["support"][case["reference_bin"]])
                   for case in cases.values())
    support_note = ("All nonreference bins are in the saved comparison support."
                    if excluded == 0 else "Gray/hatching retains excluded comparison support.")
    figure.text(.09, .127,
                "× Shared gauge reference: finite, but excluded from error statistics. " + support_note,
                fontsize=9.1, color="#52606f")
    figure.text(.09, .090,
                "Independent continuous IID draws train each current bias; these panels do not test MD mixing.\n"
                "Uncertainty bars describe the saved validation estimators. Physical convergence remains unverified.",
                fontsize=9.5, color="#52606f", linespacing=1.6)
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pdf", type=Path)
    args = parser.parse_args()
    cases = read_cases(args.report)
    figure = make_figure(cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, metadata={
        "Title": "VES stochastic validation against quadrature and umbrella/MBAR",
        "Description": "Saved stochastic evidence; physical convergence unverified.",
    })
    print(args.output)
    if args.pdf is not None:
        args.pdf.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.pdf, metadata={
            "Title": "VES stochastic validation against quadrature and umbrella/MBAR",
            "Subject": "Toy-model consistency checks; physical convergence unverified",
        })
        print(args.pdf)
    plt.close(figure)


if __name__ == "__main__":
    main()
