"""Plot the completed Richardson--Thoss Fig. 1(c) comparison without rerunning it.

Inputs are the completed benchmark summary and the independently extracted
published CSV. Outputs contain only numerical data plotted by this script;
the original copyrighted PDF or figure image is never embedded or copied.

Example::

    python examples/nonadiabatic/plot_published_nrpmd_benchmark.py \
        docs/validation/published-nrpmd-2013-2026-11-01/summary.json

The default destination is a ``figures`` subdirectory beside the summary.
The SVG and PNG are self-contained; the adjacent manifest records their exact
input hashes and the availability of each reported statistical interval.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


OBSERVABLES = ("Cxx", "C11")
BEADS = (1, 4, 8)
PAPER_COLORS = {1: "#245da8", 4: "#178149", 8: "#d43735"}
RUN_COLOR = "#7653a6"
REASONS = {
    "source_average_sign_below_threshold": "Source average sign below threshold",
    "source_average_sign_not_resolved_against_bootstrap_error": "Average sign not resolved",
    "bootstrap_contains_unresolved_signed_normalization": "Unresolved sign in bootstrap draws",
    "fewer_than_eight_complete_blocks_in_a_source_chain": "Fewer than 8 blocks in a source chain",
    "fewer_than_two_independent_chains": "Fewer than 2 independent source chains",
    "fewer_than_32_bootstrap_replicates": "Fewer than 32 bootstrap draws",
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _finite_array(value, shape, name):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"invalid {name}: expected finite shape {shape}")
    return array


def _timeline(value, name):
    time = np.asarray(value, dtype=float)
    if time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError(f"invalid increasing time array for {name}")
    return time


def load_inputs(summary_path, published_path):
    """Validate supplied curves and keep bootstrap and readout errors separate."""
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    if summary.get("observable_order") != list(OBSERVABLES):
        raise ValueError("summary must order the observables as Cxx, C11")
    if summary.get("paper", {}).get("doi") != "10.1063/1.4816124":
        raise ValueError("summary is not the Richardson--Thoss 2013 benchmark")
    results = {result["beads"]: result for result in summary["results"]}
    if any(beads not in results for beads in BEADS):
        raise ValueError("completed summary must include P1, P4 and P8")
    curves = {}
    for beads in BEADS:
        result = results[beads]
        time = _timeline(result["time"], f"P{beads}")
        statistics = result["statistics"]
        estimate = (None if statistics["estimate"] is None else
                    _finite_array(statistics["estimate"], (len(time), 2), "NRPMD estimate"))
        interval = None
        if statistics["confidence_intervals_available"]:
            if estimate is None or not np.isclose(statistics["confidence_level"], .95):
                raise ValueError("available confidence intervals must describe a 95% estimate")
            interval = _finite_array(statistics["percentile_interval"], (2, len(time), 2), "95% interval")
            if np.any(interval[0] > interval[1]):
                raise ValueError("confidence interval limits are reversed")
        curves[beads] = {"time": time, "estimate": estimate, "interval": interval,
                         "statistics": statistics}
    reference = summary["exact_quantum_reference"]
    exact_time = _timeline(reference["time"], "independent DVR")
    exact = {name: _finite_array(reference[name], (len(exact_time),), name) for name in OBSERVABLES}
    with Path(published_path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    published = {}
    for name in OBSERVABLES:
        for series in ("dvr", "P1", "P4", "P8"):
            selected = [row for row in rows if row["observable"] == name and row["series"] == series]
            if not selected:
                raise ValueError(f"missing published {name} {series} data")
            selected.sort(key=lambda row: float(row["time"]))
            time = _timeline([row["time"] for row in selected], f"published {name} {series}")
            value = _finite_array([row["value"] for row in selected], time.shape, "published values")
            error = _finite_array([row["absolute_uncertainty"] for row in selected], time.shape, "graphic-readout bounds")
            if np.any(error < 0):
                raise ValueError("graphic-readout bounds cannot be negative")
            published[name, series] = (time, value, error)
    published_hash = _sha256(published_path)
    expected_hash = summary.get("published_figure_comparison", {}).get("source_csv_sha256")
    if expected_hash is not None and expected_hash != published_hash:
        raise ValueError("published CSV does not match the benchmark comparison's source hash")
    provenance_path = Path(published_path).with_suffix(".provenance.json")
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance["csv_sha256"] != published_hash:
            raise ValueError("published CSV does not match its extraction provenance")
    else:
        provenance = None
    return summary, curves, exact_time, exact, published, provenance


def _missing_interval_text(statistics):
    headline = ("NRPMD mean and 95% interval unavailable" if statistics["estimate"] is None
                else "95% interval unavailable")
    reasons = [REASONS.get(reason, reason.replace("_", " "))
               for reason in statistics["unavailable_reasons"]]
    return "\n".join([headline]+reasons)


def make_figure(summary_path, published_path, output_directory):
    summary_path, published_path = Path(summary_path), Path(published_path)
    summary, curves, exact_time, exact, published, provenance = load_inputs(summary_path, published_path)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": "#25323f", "text.color": "#25323f",
        "xtick.color": "#516171", "ytick.color": "#516171",
        "axes.edgecolor": "#aeb8c2", "axes.linewidth": .8,
        "svg.fonttype": "none", "svg.hashsalt": "pyrpmd-published-benchmark-2013",
    })
    figure, axes = plt.subplots(2, 3, figsize=(14.4, 9.4), sharex=True, sharey="row")
    figure.subplots_adjust(left=.075, right=.985, bottom=.245, top=.765, hspace=.15, wspace=.10)
    figure.suptitle("NRPMD against the published Fig. 1(c) benchmark", x=.075, y=.972,
                     ha="left", fontsize=20, fontweight="bold")
    figure.text(.075, .933,
        r"Richardson & Thoss, J. Chem. Phys. 139, 031102 (2013)  |  $m=\hbar=\omega=\kappa=\beta=\Delta=1$, $\alpha=0$",
        ha="left", fontsize=11, color="#516171")
    handles = [
        Line2D([], [], color=RUN_COLOR, linewidth=2, label="This calculation (NRPMD)"),
        Patch(facecolor=RUN_COLOR, alpha=.18, edgecolor="none", label="95% source-block bootstrap interval"),
        Line2D([], [], color="#404a55", linewidth=1.7, linestyle="--", label="Published NRPMD (panel color)"),
        Line2D([], [], color="black", marker="o", markersize=4, linestyle="none", label="Published DVR markers"),
        Line2D([], [], color="#9aa3ad", linewidth=1.2, label="Independent sinc-DVR"),
        Line2D([], [], color="#77828d", marker="|", markersize=9, linestyle="none", label="Published graphic-readout bound"),
    ]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .901),
                  ncol=3, frameon=False, fontsize=10, columnspacing=2.2, handlelength=2.7)
    extents = {name: [float(exact[name].min()), float(exact[name].max())] for name in OBSERVABLES}
    statuses = []
    for column, beads in enumerate(BEADS):
        curve = curves[beads]
        statistics = curve["statistics"]
        available = curve["interval"] is not None
        status = "95% interval available" if available else "95% interval unavailable"
        axes[0, column].set_title(
            f"P = {beads}    |    average sign = {statistics['average_sign']:.3f}\n{status}",
            fontsize=11, pad=10, color="#25323f" if available else "#8a561b")
        statuses.append({"beads": beads, "average_sign": statistics["average_sign"],
                         "estimate_available": curve["estimate"] is not None,
                         "confidence_intervals_available": available,
                         "unavailable_reasons": statistics["unavailable_reasons"]})
        for row, name in enumerate(OBSERVABLES):
            ax = axes[row, column]
            ax.grid(axis="y", color="#e7ebef", linewidth=.7, zorder=0)
            ax.axhline(0., color="#ccd3da", linewidth=.8, zorder=1)
            ax.plot(exact_time, exact[name], color="#9aa3ad", linewidth=1.2, zorder=2)
            paper_time, paper_value, paper_bound = published[name, f"P{beads}"]
            ax.errorbar(paper_time, paper_value, yerr=paper_bound, fmt="none",
                        ecolor=PAPER_COLORS[beads], elinewidth=.65, alpha=.45, zorder=3)
            ax.plot(paper_time, paper_value, color=PAPER_COLORS[beads], linestyle="--",
                    linewidth=1.8, zorder=5, dash_capstyle="round")
            dvr_time, dvr_value, dvr_bound = published[name, "dvr"]
            ax.errorbar(dvr_time, dvr_value, yerr=dvr_bound, fmt="o", color="black",
                        markersize=3.2, markeredgewidth=0, ecolor="#77828d",
                        elinewidth=.65, capsize=0, zorder=6)
            extents[name].extend((float(np.min(paper_value-paper_bound)),
                                  float(np.max(paper_value+paper_bound)),
                                  float(np.min(dvr_value-dvr_bound)),
                                  float(np.max(dvr_value+dvr_bound))))
            if available:
                lo, hi = curve["interval"][:, :, row]
                ax.fill_between(curve["time"], lo, hi, color=RUN_COLOR, alpha=.18,
                                linewidth=0, zorder=2)
                extents[name].extend((float(lo.min()), float(hi.max())))
            if curve["estimate"] is not None:
                value = curve["estimate"][:, row]
                ax.plot(curve["time"], value, color=RUN_COLOR, linewidth=2.0, zorder=4)
                extents[name].extend((float(value.min()), float(value.max())))
            if not available:
                ax.text(.035, .035, _missing_interval_text(statistics),
                        transform=ax.transAxes, ha="left", va="bottom", fontsize=8.3,
                        linespacing=1.35, color="#80501b",
                        bbox={"facecolor": "white", "alpha": .94,
                              "edgecolor": "#e1d7c7", "boxstyle": "round,pad=.45"}, zorder=10)
            ax.set_xlim(float(exact_time[0]), float(exact_time[-1]))
            ax.set_xticks(np.arange(0., exact_time[-1]+1e-10, 1.))
            if row == 1:
                ax.set_xlabel(r"Time $t$ ($\omega^{-1}$)", labelpad=7)
            if column == 0:
                ax.set_ylabel(r"$\widetilde C_{xx}(t)$" if row == 0 else r"$\widetilde C_{11}(t)$",
                              fontsize=13, labelpad=9)
    for row, name in enumerate(OBSERVABLES):
        low, high = min(extents[name]), max(extents[name])
        margin = .07*(high-low)
        axes[row, 0].set_ylim(low-margin, high+margin)
    config = summary["config"]
    budget = (f"{config['chains']} independent source chains × {config['samples']} parent trajectories per bead count; "
              f"dt = {config['dt']:g}; {config['bootstrap_replicates']} bootstrap draws; "
              f"potential split: {summary['potential_split'].replace('_', ' ')}.")
    figure.text(.075, .184, budget, fontsize=10.3, ha="left", fontweight="bold")
    caption = (
        "Shading: pointwise 95% percentile intervals from source-chain / circular-block joint-ratio bootstrap; signs and their shared normalization are retained.\n"
        "Intervals are conditional on source mixing and selected block lengths. A failed gate withholds the band; it is never replaced by a zero-width interval.\n"
        "Dashed curves and black markers use the original published vertices / marker centers. Their vertical bars are conservative graphic-readout bounds,\n"
        "which are separate from Monte Carlo uncertainty. Gray curves are an independently recomputed spectral Kubo reference.\n"
        "This is a benchmark of the paper's nonadiabatic model. It does not establish physical convergence or validate the electrochemical reservoir or adaptive path."
    )
    figure.text(.075, .148, caption, fontsize=9.4, ha="left", va="top", linespacing=1.45, color="#516171")
    stem = output/"published_nrpmd_comparison"
    figure.savefig(stem.with_suffix(".svg"), metadata={"Date": None}, facecolor="white")
    svg_path = stem.with_suffix(".svg")
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines())+"\n",
                        encoding="utf-8")
    figure.savefig(stem.with_suffix(".png"), dpi=180, metadata={"Software": "PyRPMD benchmark plotting"}, facecolor="white")
    plt.close(figure)
    manifest = {
        "schema_version": 1, "title": "NRPMD against the published Fig. 1(c) benchmark",
        "inputs": {"summary": str(summary_path), "summary_sha256": _sha256(summary_path),
                   "published_csv": str(published_path), "published_csv_sha256": _sha256(published_path)},
        "plot_script_sha256": _sha256(__file__), "matplotlib_version": matplotlib.__version__,
        "panels": {"columns_beads": BEADS, "rows_observables": OBSERVABLES}, "statuses": statuses,
        "interval_scope": "pointwise_95_percent_source_chain_circular_block_joint_ratio_bootstrap_conditional_on_source_mixing_and_block_lengths",
        "published_error_bar_scope": "conservative_graphic_readout_bound_not_author_sampling_error",
        "published_extraction_provenance": provenance,
        "original_pdf_or_image_embedded": False, 
        "artifacts": {suffix: {"file": stem.with_suffix(suffix).name,
                               "sha256": _sha256(stem.with_suffix(suffix))} for suffix in (".svg", ".png")},
    }
    (output/"plot_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--published-data", type=Path, default=Path(__file__).with_name("data")/"richardson_thoss_2013_fig1c.csv")
    parser.add_argument("--output", type=Path, help="defaults to figures/ beside the summary")
    arguments = parser.parse_args(argv)
    output = arguments.output or arguments.summary.parent/"figures"
    manifest = make_figure(arguments.summary, arguments.published_data, output)
    print(json.dumps({"output": str(output), "statuses": manifest["statuses"]}, indent=2))


if __name__ == "__main__":
    main()
