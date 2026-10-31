"""Render the saved, finite-budget H3 comparison without rerunning dynamics."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
CASES = [
    ("fixed96-short", "96 windows / 200 fs", "#8a969e", ":"),
    ("adaptive-short", "Adaptive / 200 fs", "#df8f26", ":"),
    ("fixed96-production", "96 windows / 1000 fs", "#2075ae", "-"),
    ("adaptive-production", "Adaptive / 1000 fs", "#43a47c", "--"),
    ("adaptive-matched", "Adaptive / matched total steps", "#824b9d", "-"),
]
HARTREE_KCAL_MOL = 627.5094740631


def main():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    labels = []
    for i, (name, label, color, style) in enumerate(CASES):
        data = json.loads((HERE / "evidence" / f"{name}.json").read_text())
        x = np.asarray(data["xi"])
        y = np.asarray(data["free_energy_hartree"], float) * HARTREE_KCAL_MOL
        y[np.asarray(data["mask"], bool)] = np.nan
        axes[0].plot(x, y, label=label, color=color, ls=style, lw=1.6)
        axes[1].errorbar(data["barrier_hartree"] * HARTREE_KCAL_MOL, i,
                         xerr=data["barrier_standard_error_hartree"] * HARTREE_KCAL_MOL,
                         fmt="o", color=color, capsize=4)
        labels.append(label)
    axes[0].axvline(1.00925, color="#444444", lw=.8, ls=":")
    axes[0].set(xlabel="RPMDrate coordinate xi", ylabel="Rate-flux PMF (kcal/mol)", xlim=(-.02, 1.1))
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    axes[1].set(yticks=range(len(labels)), yticklabels=labels,
                xlabel="Barrier at xi = 1.00925 (kcal/mol)")
    axes[1].invert_yaxis()
    for ax in axes:
        ax.grid(alpha=.18)
    fig.suptitle("H + H2 / BKMP2: 1500 K, 16 beads, 0.1 fs", y=.99)
    fig.text(.5, .015, "Finite-budget diagnostic; error bars are block-bootstrap SD (10 or 20 draws), not physical convergence.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, .97))
    fig.savefig(HERE / "h3-comparison.png", dpi=180)
    fig.savefig(HERE / "h3-comparison.pdf")


if __name__ == "__main__":
    main()
