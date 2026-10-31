"""Plot the isolated-H2 numerical reference and finite-bead harmonic estimator.

Run after ``benchmarks/h3/h2_reference.py``::

    python benchmarks/h3/plot_h2_reference.py

The left panel uses the verified BKMP2 radial potential, extrapolated energy
levels, and normalized finite-grid reduced radial wavefunctions.  The right
panel evaluates the canonical primitive harmonic physical-energy estimator;
it is not the conserved ring-polymer Hamiltonian H_P/P.  Neither panel reports
reactive-scattering cross sections.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "runs/state_selected_reproduction/h2_reference"
OUTPUT = ROOT / "runs/state_selected_reproduction/figures/h2_reference"
COLORS = ("#007C91", "#C06424", "#7950A2")


def finite_bead_ratio(beads, beta, omega):
    """<E_primitive>/(omega/2) for a canonical harmonic ring, hbar=1."""
    beads = np.asarray(beads, dtype=float)
    x = beta * omega / (2 * beads)
    return 1 / (np.tanh(beads * np.arcsinh(x)) * np.sqrt(1 + x * x))


def make_figure(reference=REFERENCE, output=OUTPUT):
    reference, output = Path(reference), Path(output)
    report = json.loads(reference.with_suffix(".json").read_text())
    if not report["passed"]:
        raise ValueError("Reference convergence gate must pass before plotting")
    with np.load(reference.with_suffix(".npz"), allow_pickle=False) as archive:
        r = archive["r_bohr"].copy()
        potential = archive["potential_relative_hartree"].copy()
        probability = archive["probability_density_dr"].copy()
        levels_au = archive["reference_energies_hartree"].copy()
    rows = report["levels"]
    if len(rows) < 3 or [row["v"] for row in rows[:3]] != [0, 1, 2]:
        raise ValueError("The figure requires v=0,1,2 reference states")
    if not np.allclose(levels_au[:3], [row["energy_above_well_minimum_hartree"] for row in rows[:3]],
                       rtol=0, atol=1e-14):
        raise ValueError("JSON and NPZ level energies disagree")
    if not np.all(np.diff(r) > 0) or not np.allclose(
        (r[1] - r[0]) * probability[:3].sum(axis=1), 1, rtol=0, atol=1e-10
    ):
        raise ValueError("Saved radial probabilities are not normalized on a regular increasing grid")
    # Recover the same unit conversion and comparison frequency used by the
    # reference file, keeping this plot script independent of core imports.
    hartree_to_ev = rows[0]["energy_above_well_minimum_eV"] / rows[0]["energy_above_well_minimum_hartree"]
    harmonic = report["harmonic_reference"]
    wavenumber = harmonic["comparison_harmonic_wavenumber_cm_minus1"]
    omega = harmonic["omega_au"] * wavenumber / harmonic["wavenumber_cm_minus1"]
    levels_ev = levels_au[:3] * hartree_to_ev
    potential_ev = potential * hartree_to_ev

    style = {
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "mathtext.fontset": "dejavusans", "axes.titlesize": 12,
        "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#4F5660", "axes.linewidth": 0.8,
        "xtick.color": "#363C43", "ytick.color": "#363C43",
        "text.color": "#20262E", "axes.labelcolor": "#20262E",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }
    with plt.rc_context(style):
        fig, (left, right) = plt.subplots(
            1, 2, figsize=(11.4, 5.35), gridspec_kw={"width_ratios": [1.13, 1.0]}
        )
        fig.subplots_adjust(left=0.075, right=0.975, bottom=0.205, top=0.775, wspace=0.27)
        fig.suptitle("Isolated H$_2$: bound-state and ring-polymer diagnostics",
                     x=0.075, y=0.973, ha="left", fontsize=16, fontweight="semibold")
        fig.text(0.075, 0.91,
                 r"BKMP2 diatomic asymptote  |  $j=0$  |  $\mu=918.5763\,m_e$",
                 fontsize=11, color="#59616C")

        for axis in (left, right):
            axis.set_axisbelow(True)
            axis.grid(axis="y", color="#E1E5E8", linewidth=0.65)
            axis.tick_params(direction="out", length=3.5, width=0.7)

        # Panel A: each density is offset by its own energy. Its height is a
        # display scale, not an additional physical energy or density unit.
        left.set_title("A  BKMP2 vibrational levels", loc="left", pad=23, fontweight="semibold")
        left.text(0, 1.025, r"Solid levels: numerical $j=0$ eigenvalues", transform=left.transAxes,
                  fontsize=9.5, color="#59616C")
        density_height_ev = 0.12
        for v, (energy, color) in enumerate(zip(levels_ev, COLORS)):
            scaled = density_height_ev * probability[v] / probability[v].max()
            left.hlines(energy, 0.76, 3.65, color=color, linewidth=0.9, alpha=0.72, zorder=2)
            left.fill_between(r, energy, energy + scaled, color=color, alpha=0.22, linewidth=0, zorder=3)
            left.plot(r, energy + scaled, color=color, linewidth=1.2, zorder=4)
            left.text(2.60, energy - (0.071 if v < 2 else 0.075),
                      rf"$v={v}$   {energy:.4f} eV", color=color, fontsize=10,
                      bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0}, zorder=7)

        left.plot(r, potential_ev, color="#232A34", linewidth=1.8, zorder=5)
        for v in (0, 1):
            harmonic_energy = (v + 0.5) * omega * hartree_to_ev
            left.hlines(harmonic_energy, 2.62, 3.65, color="#6D747D", linewidth=1.2,
                        linestyle=(0, (4, 2.5)), zorder=6)
            left.text(2.60, harmonic_energy + 0.054,
                      f"Harmonic {harmonic_energy:.4f} eV", color="#5B626C", fontsize=9.1,
                      bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0}, zorder=7)
        left.set_xlim(0.76, 3.75)
        left.set_ylim(-0.035, 1.52)
        left.set_xticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
        left.set_yticks(np.arange(0.0, 1.51, 0.3))
        left.set_xlabel(r"H–H distance $r$ (bohr)", labelpad=8)
        left.set_ylabel(r"Energy above the well minimum (eV)", labelpad=8)
        left.legend(handles=[Line2D([0], [0], color="#232A34", linewidth=1.8,
                                    label="BKMP2 potential")],
                    loc="upper right", frameon=False, fontsize=9.4,
                    handlelength=1.8, borderaxespad=0.7)

        # Panel B: canonical harmonic ensemble at each beta, normalized by
        # the harmonic ground-state energy. Curves include finite-temperature
        # contributions; finite-P values can undershoot the ground-state ratio.
        right.set_title("B  Finite-bead harmonic energy", loc="left", pad=23, fontweight="semibold")
        right.text(0, 1.025, rf"Primitive estimator; $\omega={wavenumber:.0f}$ cm$^{{-1}}$",
                   transform=right.transAxes, fontsize=9.5, color="#59616C")
        beads = np.geomspace(8, 256, 500)
        right.axvline(32, color="#848B95", linewidth=1.0, linestyle=(0, (1.5, 2.5)), zorder=1)
        right.text(32, 1.0475, "$P=32$", ha="center", va="bottom", fontsize=10, color="#4E5662")
        legend_handles = []
        for beta, color in zip((200.0, 300.0, 400.0), COLORS):
            ratio = finite_bead_ratio(beads, beta, omega)
            continuum = 1 / np.tanh(beta * omega / 2)
            at32 = float(finite_bead_ratio(32, beta, omega))
            right.plot(beads, ratio, color=color, linewidth=1.9, zorder=3)
            right.axhline(continuum, color=color, linewidth=1.05, linestyle=(0, (4, 3)), alpha=0.70, zorder=2)
            right.plot(32, at32, marker="o", markersize=5.5, color=color,
                       markeredgecolor="white", markeredgewidth=0.8, zorder=5)
            right.annotate(f"{at32:.4f}", xy=(32, at32), xytext=(9, -2 if beta == 200 else -5),
                           textcoords="offset points", color=color, fontsize=9.3,
                           bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.7}, zorder=6)
            legend_handles.append(Line2D([0], [0], color=color, linewidth=2,
                                         label=rf"$\beta={beta:.0f}\ E_h^{{-1}}$"))
        right.set_xscale("log", base=2)
        right.set_xlim(8, 256)
        right.set_xticks([8, 16, 32, 64, 128, 256], labels=["8", "16", "32", "64", "128", "256"])
        right.set_ylim(0.885, 1.055)
        right.set_yticks([0.90, 0.94, 0.98, 1.00, 1.04])
        right.set_xlabel("Number of beads, $P$", labelpad=8)
        right.set_ylabel(r"$\langle E_{\rm prim}\rangle\,/\,(\hbar\omega/2)$", labelpad=8)
        legend_handles.append(Line2D([0], [0], color="#737B84", linewidth=1.2,
                                     linestyle=(0, (4, 3)), label=r"Dashed: $P\rightarrow\infty$"))
        right.legend(handles=legend_handles, loc="lower right", frameon=False,
                     fontsize=9.6, handlelength=2.4, labelspacing=0.55,
                     borderpad=0.5, borderaxespad=0.65)

        fig.text(0.075, 0.095,
                 r"A: Shaded $|u_v(r)|^2$ is normalized in $dr$ and scaled to 0.12 eV height for display.",
                 fontsize=9.2, color="#59616C")
        fig.text(0.075, 0.052,
                 r"Isolated-molecule reference; no reactive cross sections.  B shows a physical-energy estimator, not $H_P/P$.",
                 fontsize=9.2, color="#59616C")

        output.parent.mkdir(parents=True, exist_ok=True)
        description = (
            "Isolated H2 on verified BKMP2. Left: Richardson-extrapolated j=0 energies and finite-grid "
            "radial densities; harmonic reference 4407 cm^-1. Right: canonical primitive harmonic "
            "energy divided by hbar*omega/2, beta=200,300,400 Hartree^-1. Not reactive cross sections. "
            "Reference analysis SHA256=" + report["provenance"]["analysis_source_sha256"]
        )
        fig.savefig(output.with_suffix(".png"), dpi=300,
                    metadata={"Title": "Isolated H2 reference diagnostics", "Description": description})
        fig.savefig(output.with_suffix(".pdf"),
                    metadata={"Title": "Isolated H2 reference diagnostics", "Author": "PyRPMD benchmark",
                              "Subject": description, "Creator": "benchmarks/h3/plot_h2_reference.py"})
        plt.close(fig)
    return output.with_suffix(".png"), output.with_suffix(".pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=REFERENCE,
                        help="Path prefix of the existing reference .json and .npz")
    parser.add_argument("--output", type=Path, default=OUTPUT,
                        help="Output path prefix for .png and .pdf")
    args = parser.parse_args()
    for path in make_figure(args.reference, args.output):
        print(path)


if __name__ == "__main__":
    main()
