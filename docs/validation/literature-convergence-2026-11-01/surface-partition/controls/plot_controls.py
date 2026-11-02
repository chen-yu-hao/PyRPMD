"""Render completed, registered QTST controls without changing inference.

The input is the fixed control runner's complete comparison.json. This script
does not estimate a rate, select seeds, modify a margin or change any data.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    directory = args.directory
    comparison_path = directory / "comparison.json"
    protocol_path = directory / "protocol.json"
    comparison = json.loads(comparison_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    if not comparison["complete"] or comparison["protocol_sha256"] != sha(protocol_path):
        raise ValueError("a complete comparison with the registered protocol is required")
    cases = comparison["controls"]
    if any(not case["complete"] or len(case["completed_seed_results"]) != len(case["seeds"])
           for case in cases):
        raise ValueError("every registered control seed must be present")
    margin = protocol["equivalence_log_margin"]
    limits = [100 * math.expm1(-margin), 100 * math.expm1(margin)]
    rows = []
    for case in cases:
        result = case["relative_QTST_comparison"]
        if result["equivalence_log_margin"] != margin:
            raise ValueError("control equivalence margin differs from the protocol")
        rows.append({
            "case": case["name"], "beads": case["beads"],
            "minimum_pair_bohr": case["minimum"], "independent_seeds": len(case["seeds"]),
            "relative_difference_percent": 100 * result["relative_point_difference"],
            "CI95_low_percent": 100 * result["CI95_relative_difference"][0],
            "CI95_high_percent": 100 * result["CI95_relative_difference"][1],
            "CI95_within_registered_margin": result["CI95_wholly_within_margin"],
            "sampling_flags_passed": case["seed_sampling_flags_passed"],
        })
    data_path = directory / "comparison-plot-data.csv"
    with data_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.spines.left": False})
    figure, axis = plt.subplots(figsize=(10.2, 4.6))
    figure.subplots_adjust(left=.22, right=.73, bottom=.28, top=.75)
    axis.axvspan(*limits, color="#E8F2F4", zorder=0)
    axis.axvline(0, color="#6D747A", linewidth=1, linestyle="--", zorder=1)
    for bound in limits:
        axis.axvline(bound, color="#A7BCC4", linewidth=.8, zorder=1)
    for index, row in enumerate(rows):
        point = row["relative_difference_percent"]
        low, high = row["CI95_low_percent"], row["CI95_high_percent"]
        color = "#087F8C" if row["CI95_within_registered_margin"] else "#B76A15"
        axis.errorbar(point, index, xerr=[[point-low], [high-point]], color=color,
                      fmt="o", markersize=6, capsize=4, linewidth=1.6, zorder=3)
        label = f"{point:+.2f}%  [{low:+.2f}, {high:+.2f}]"
        if not row["sampling_flags_passed"]:
            label += " *"
        axis.text(1.035, index, label, transform=axis.get_yaxis_transform(),
                  va="center", fontsize=10, color="#26343B")
    labels = [f"P = {row['beads']}, minimum {row['minimum_pair_bohr']:g} bohr"
              for row in rows]
    axis.set_yticks(range(len(rows)), labels)
    axis.tick_params(axis="y", length=0, pad=10)
    axis.set_ylim(len(rows)-.5, -.5)
    left = min(limits[0], *(row["CI95_low_percent"] for row in rows))
    right = max(limits[1], *(row["CI95_high_percent"] for row in rows))
    padding = max(.7, .06*(right-left))
    axis.set_xlim(left-padding, right+padding)
    axis.grid(axis="x", color="#E5EAEC", linewidth=.7)
    axis.set_axisbelow(True)
    axis.set_xlabel("QTST change relative to P = 128, minimum 0.5 bohr (%)", labelpad=10)
    figure.text(.03, .94, "Independent equilibrium controls", fontsize=17, weight="bold", color="#20343D")
    figure.text(.03, .865, "H + H₂ on BKMP2, 300 K · Four new seeds per control versus eight fixed baseline seeds",
                fontsize=10.5, color="#465963")
    axis.text(1.035, 1.08, "Point [95% interval]", transform=axis.transAxes,
              fontsize=10, color="#465963")
    handles = [Line2D([0], [0], marker="o", color="#087F8C", label="Interval inside registered margin"),
               Line2D([0], [0], marker="o", color="#B76A15", label="Interval not inside registered margin")]
    figure.legend(handles=handles, loc="lower left", bbox_to_anchor=(.215, .10), frameon=False,
                  ncol=2, fontsize=9)
    figure.text(.03, .065,
                f"95% Welch intervals across independent seed chains. Shaded margin: {limits[0]:.2f}% to {limits[1]:+.2f}%.",
                fontsize=9, color="#465963")
    note = "Matching exact H₂ normalization is used for each control. These checks concern equilibrium QTST, not recrossing."
    if not all(row["sampling_flags_passed"] for row in rows):
        note += "  * Sampling flag failed."
    figure.text(.03, .025, note, fontsize=9, color="#465963")
    outputs = [data_path]
    for suffix in ("png", "svg"):
        path = directory / f"comparison.{suffix}"
        figure.savefig(path, dpi=180, facecolor="white")
        outputs.append(path)
    plt.close(figure)
    metadata = {
        "schema": "pyrpmd.h3.equilibrium-control-figure.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_sha256": sha(comparison_path), "protocol_sha256": sha(protocol_path),
        "plot_source_sha256": sha(__file__), "statistical_analysis_changed": False,
        "outputs": {path.name: {"sha256": sha(path), "bytes": path.stat().st_size} for path in outputs},
    }
    (directory / "figure-metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")


if __name__ == "__main__":
    main()
