"""Draw the manuscript figures from the records in paper/data/.

Usage: python paper/scripts/make_figures.py
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAPER = Path(__file__).resolve().parents[1]
DATA, FIG = PAPER / "data", PAPER / "figures"
FIG.mkdir(exist_ok=True)

# Fixed categorical order; every series also has its own marker or line style.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.2,
    "lines.markersize": 4, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})


def load(name):
    return json.loads((DATA / name).read_text())


def tag(ax, letter):
    ax.text(-0.2, 1.04, f"({letter})", transform=ax.transAxes, fontweight="bold", va="bottom")


def figure_equilibrium():
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.0))
    (a, b), (c, d) = axes

    scan = load("beadscan.json")
    beads = np.array([r["beads"] for r in scan["rows"]])
    mha = 1e3
    a.axhline(scan["exact_quantum_kinetic"] * mha, color=MUTED, lw=0.8, ls="--")
    a.text(1.0, scan["exact_quantum_kinetic"] * mha + 0.03, "exact quantum", color=MUTED, fontsize=7)
    a.axhline(scan["classical_kinetic"] * mha, color=MUTED, lw=0.8, ls=":")
    a.text(64, scan["classical_kinetic"] * mha + 0.03, "classical", color=MUTED, fontsize=7, ha="right")
    a.plot(beads, [r["finite_p_reference"] * mha for r in scan["rows"]], color=BLUE, lw=1.2,
           label="analytic finite-$P$")
    a.errorbar(beads, [r["kinetic_cv"] * mha for r in scan["rows"]],
               yerr=[3 * r["kinetic_cv_sem"] * mha for r in scan["rows"]], fmt="o", color=ORANGE,
               mec="white", mew=0.6, capsize=2, label="PyRPMD ($\\pm3\\sigma$)")
    a.set_xscale("log", base=2)
    a.set_xticks(beads, [str(n) for n in beads])
    a.set_xlabel("number of beads $P$")
    a.set_ylabel("$\\langle K_{\\mathrm{cv}}\\rangle$ / m$E_\\mathrm{h}$")
    a.legend(loc="center right")
    tag(a, "a")

    trace = load("trace_convergence.json")["rows"]
    p = np.array([r["beads"] for r in trace])
    b.loglog(p, [r["primitive_absolute_logZ_error"] for r in trace], "o-", color=BLUE, mec="white", mew=0.6,
             label="primitive")
    b.loglog(p, [r["TI_absolute_logZ_error"] for r in trace], "s-", color=ORANGE, mec="white", mew=0.6,
             label="Takahashi–Imada")
    b.loglog(p[2:], 0.04 * (p[2:] / 16.0) ** -2 * 2.2, color=MUTED, lw=0.7, ls=":")
    b.text(70, 6e-3, "$P^{-2}$", color=MUTED, fontsize=7)
    b.loglog(p[2:], 3.3e-4 * (p[2:] / 16.0) ** -4 * 0.35, color=MUTED, lw=0.7, ls=":")
    b.text(20, 1.2e-6, "$P^{-4}$", color=MUTED, fontsize=7)
    b.set_xscale("log", base=2)
    b.set_xticks(p, [str(n) for n in p])
    b.set_xlabel("number of beads $P$")
    b.set_ylabel("$|\\ln Z_P-\\ln Z|$ per mode")
    b.legend(loc="lower left")
    tag(b, "b")

    fes = load("metad_fes.json")
    x = np.array(fes["x_bohr"])
    mean = np.array(fes["mean_kbt"], dtype=float)
    sem = np.array(fes["sem_kbt"], dtype=float)
    c.fill_between(x, mean - 3 * sem, mean + 3 * sem, color=ORANGE, alpha=0.25, lw=0)
    c.plot(x, fes["exact_kbt"], color=BLUE, label="exact Boltzmann")
    c.plot(x[::3], mean[::3], "o", color=ORANGE, ms=2.6, mec="white", mew=0.4,
           label="WT-MetaD, 4 replicas ($\\pm3\\sigma$)")
    c.set_xlabel("$x$ / $a_0$")
    c.set_ylabel("$F(x)$ / $k_\\mathrm{B}T$")
    c.set_ylim(-0.6, 6.5)
    c.legend(loc="upper center")
    tag(c, "c")

    spec = load("spectrum.json")
    ref = load("tutorials.json")["harmonic_spectroscopy"]["checks"][0]["reference"]
    d.axvline(ref, color=BLUE, lw=1.0, label="$\\sqrt{k/m}$")
    d.plot(spec["wavenumber_cm1"], spec["intensity"], color=ORANGE, lw=1.0, label="TRPMD")
    d.set_xlim(1090, 1200)
    d.set_xlabel("wavenumber / cm$^{-1}$")
    d.set_ylabel("relative IR intensity")
    d.legend(loc="upper right")
    tag(d, "d")

    fig.tight_layout(w_pad=2.0, h_pad=1.6)
    fig.savefig(FIG / "equilibrium.pdf")
    plt.close(fig)


def figure_integrators():
    rows = load("integrators.json")
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    style = {"cayley-bcb": ("o-", BLUE, "Cayley BCB"), "normal-mode-bab": ("s--", ORANGE, "normal-mode BAB"),
             "velocity-verlet": ("^-", AQUA, "velocity Verlet")}
    for name, (fmt, color, label) in style.items():
        sel = [r for r in rows if r["integrator"] == name and r["stable"]]
        ax.loglog([r["dt_fs"] for r in sel], [r["max_relative_energy_error"] for r in sel], fmt,
                  color=color, mec="white", mew=0.6, label=label)
    dt = np.array([0.0125, 0.4])
    ax.loglog(dt, 2.2e-7 * (dt / 0.0125) ** 2, color=MUTED, lw=0.7, ls=":")
    ax.text(0.16, 1.6e-5, "$\\Delta t^{2}$", color=MUTED, fontsize=7)
    ticks = [0.0125, 0.025, 0.05, 0.1, 0.2, 0.4]
    ax.set_xticks(ticks, [str(v) for v in ticks])
    ax.minorticks_off()
    ax.set_xlabel("time step $\\Delta t$ / fs")
    ax.set_ylabel("max $|H_P(t)-H_P(0)|/H_P(0)$")
    ax.legend(loc="upper left")
    fig.savefig(FIG / "integrators.pdf")
    plt.close(fig)


def figure_rates():
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.4))
    a, b, c = axes
    ex = load("exchange.json")
    kbt = ex["kbt_hartree"]
    a.plot(ex["pmf"]["xi"], np.array(ex["pmf"]["free_energy_hartree"]) / kbt, color=BLUE)
    a.set_xlabel("reaction coordinate $\\xi$")
    a.set_ylabel("$W(\\xi)$ / $k_\\mathrm{B}T$")
    a.set_xlim(0, 1.05)
    tag(a, "a")

    rec = ex["recrossing"]
    t, k, e = (np.array(rec[key]) for key in ("time_fs", "kappa", "standard_error"))
    b.fill_between(t, k - 2 * e, k + 2 * e, color=BLUE, alpha=0.25, lw=0)
    b.plot(t, k, color=BLUE, label="D + H$_2$ model, $P=4$")
    with open(DATA / "h3_bkmp2_kappa_curve.csv") as handle:
        h3 = np.array([[float(v) for v in row] for row in list(csv.reader(handle))[1:]])
    b.fill_between(h3[:, 0], h3[:, 2], h3[:, 3], color=ORANGE, alpha=0.25, lw=0)
    b.plot(h3[:, 0], h3[:, 1], color=ORANGE, ls="--", label="H + H$_2$ BKMP2, $P=128$")
    b.set_xlabel("$t$ / fs")
    b.set_ylabel("$\\kappa(t)$")
    b.set_xlim(0, 100)
    b.set_ylim(0.7, 1.02)
    b.legend(loc="center right")
    tag(b, "b")

    s = ex["summary"]
    labels = ["dRPMD", "BC, UI", "BC, MBAR"]
    values = [s["drpmd"], s["bc_ui"], s["bc_mbar"]]
    y = np.arange(3)[::-1]
    for yi, v, color, marker in zip(y, values, (BLUE, ORANGE, AQUA), ("o", "s", "^")):
        c.errorbar(v["rate_cm3_s"] * 1e11, yi, xerr=2 * v["sem_cm3_s"] * 1e11, fmt=marker, color=color,
                   mec="white", mew=0.6, ms=6, capsize=3)
        c.text(v["rate_cm3_s"] * 1e11, yi + 0.22, f"{v['rate_cm3_s'] * 1e11:.2f}", ha="center", fontsize=7)
    c.set_yticks(y, labels)
    c.set_ylim(-0.6, 2.7)
    c.set_xlabel("$k$ / $10^{-11}$ cm$^3$ molecule$^{-1}$ s$^{-1}$")
    c.tick_params(axis="y", length=0)
    tag(c, "c")
    fig.tight_layout(w_pad=1.6)
    fig.savefig(FIG / "rates.pdf")
    plt.close(fig)


def figure_performance():
    data = load("throughput.json")
    rows = [r for r in data["rows"] if r["atoms"] == 3]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
    a, b = axes
    style = {"fused": ("o-", BLUE, "fused C++ block"), "substep": ("s--", ORANGE, "Python loop, C++ substeps"),
             "numpy": ("^-", AQUA, "NumPy reference")}
    for path, (fmt, color, label) in style.items():
        sel = [r for r in rows if r["path"] == path]
        a.loglog([r["beads"] for r in sel], [r["seconds_per_step_median"] * 1e6 for r in sel], fmt,
                 color=color, mec="white", mew=0.6, label=label)
    a.set_xscale("log", base=2)
    beads = sorted({r["beads"] for r in rows})
    a.set_xticks(beads, [str(n) for n in beads])
    a.set_xlabel("number of beads $P$ ($N=3$)")
    a.set_ylabel("wall time per step / $\\mu$s")
    a.legend(loc="center left", bbox_to_anchor=(0.0, 0.52))
    tag(a, "a")

    par = load("parallel.json")
    workers = [r["workers"] for r in par["rows"]]
    speed = [par["rows"][0]["wall_seconds"] / r["wall_seconds"] for r in par["rows"]]
    b.plot(workers, workers, color=MUTED, lw=0.7, ls=":")
    b.text(workers[-1] - 0.1, workers[-1] - 0.45, "ideal", color=MUTED, fontsize=7, ha="right")
    b.plot(workers, speed, "o-", color=BLUE, mec="white", mew=0.6)
    for w, s in zip(workers, speed):
        b.text(w, s + 0.18, f"{s:.2f}", ha="center", fontsize=7)
    b.set_xticks(workers)
    b.set_xlabel("trajectory worker processes")
    b.set_ylabel("speed-up")
    b.set_ylim(0.5, workers[-1] + 0.6)
    tag(b, "b")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIG / "performance.pdf")
    plt.close(fig)


if __name__ == "__main__":
    figure_equilibrium()
    figure_integrators()
    figure_rates()
    figure_performance()
    print("figures written to", FIG)
