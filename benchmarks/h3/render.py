"""Plot saved H3 benchmark data, without resampling or filling empty bins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyrpmd.constants import AU_ENERGY_KCAL_MOL, AU_TIME_FS


def render(directory):
    directory = Path(directory)
    report_file = directory / "benchmark.json"
    report = json.loads(report_file.read_text())
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 180,
        }
    )
    settings = report.get("settings", report)
    subtitle = f"BKMP2 | {settings['temperature_K']:g} K | P={settings['beads']} | finite-sampling validation"
    figures = []
    pmf = [
        path
        for path in directory.rglob("rate_pmf.npz")
        if "resume-evidence" not in path.relative_to(directory).parts
    ]
    if pmf:
        data = np.load(pmf[0], allow_pickle=False)
        recrossing = np.load(pmf[0].parent / "recrossing.npz", allow_pickle=False)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
        ui = data["free_energy"] - np.interp(0.0, data["xi"], data["free_energy"])
        axes[0].plot(
            data["xi"],
            ui * AU_ENERGY_KCAL_MOL,
            color="#246BCE",
            lw=2,
            label="Gaussian UI",
        )
        comparison_file = directory / "pmf-diagnostics.json"
        if comparison_file.is_file():
            comparison = json.loads(comparison_file.read_text())
            for entry, color, style in zip(
                comparison["per_bins"],
                ("#D89540", "#CB4B37", "#85577E"),
                (":", "-", "--"),
            ):
                curve = entry["curve"]
                energy = np.array(curve["free_energy_hartree"], dtype=float)
                energy[np.asarray(curve["mask"], dtype=bool)] = np.nan
                axes[0].plot(
                    curve["xi"],
                    energy * AU_ENERGY_KCAL_MOL,
                    color=color,
                    ls=style,
                    lw=1.3,
                    label=f"MBAR ({entry['bins']} bins)",
                )
            axes[0].axvline(
                comparison["fixed_xi_star"], color="#777777", lw=0.9, ls="--"
            )
        axes[0].set_xlim(0, float(data["xi"][-1]))
        axes[0].legend(fontsize=8, loc="best")
        axes[0].set(
            xlabel=r"Reaction coordinate $\xi$",
            ylabel="Flux PMF / kcal mol$^{-1}$",
            title="Umbrella PMF: estimator comparison",
        )
        t = recrossing["time_au"] * AU_TIME_FS
        kappa = recrossing["kappa"]
        error = recrossing["standard_error"]
        axes[1].plot(t, kappa, color="#17866B", lw=1.7)
        axes[1].fill_between(
            t,
            kappa - error,
            kappa + error,
            color="#17866B",
            alpha=0.2,
            label="1 bootstrap standard error",
        )
        axes[1].set(
            xlabel="Time / fs",
            ylabel=r"Transmission $\kappa(t)$",
            title="Unthermostatted paired children",
        )
        axes[1].legend(fontsize=8, loc="best")
        for ax in axes:
            ax.grid(alpha=0.18)
        fig.suptitle(subtitle, fontsize=12)
        figures.append((fig, "pmf-recrossing"))
    elif report.get("task") == "fes2d":
        data = np.load(directory / "run" / "fes.npz", allow_pickle=False)
        energy = np.ma.array(
            data["free_energy"] * AU_ENERGY_KCAL_MOL, mask=data["mask"]
        )
        error = np.ma.array(
            data["standard_error"] * AU_ENERGY_KCAL_MOL, mask=data["mask"]
        )
        fig, axes = plt.subplots(1, 2, figsize=(10.7, 4.6), layout="constrained")
        for ax, values, title in zip(
            axes,
            (energy, error),
            ("2D centroid free energy", "Block-bootstrap uncertainty"),
        ):
            mesh = ax.pcolormesh(
                data["edges_0"],
                data["edges_1"],
                values.T,
                shading="flat",
                cmap="viridis",
            )
            fig.colorbar(mesh, ax=ax, label="kcal mol$^{-1}$")
            ax.set(xlabel=r"$r_{01}$ / bohr", ylabel=r"$r_{12}$ / bohr", title=title)
            ax.set_aspect("equal", adjustable="box")
            ax.set_facecolor("#ECEFF2")
        fig.suptitle(
            subtitle + "\nGray cells: insufficient sampling; no extrapolation",
            fontsize=11,
        )
        figures.append((fig, "fes2d"))
    elif report.get("task") == "drpmd":
        summary = json.loads(next(directory.rglob("results/summary.json")).read_text())
        records = summary["records"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), layout="constrained")
        colors = {
            "product": "#C74B43",
            "nonreactive": "#246BCE",
            "unresolved": "#9B8A54",
            "failed": "#303030",
        }
        for status in colors:
            subset = [r for r in records if r["status"] == status]
            if subset:
                axes[0].scatter(
                    [r["launch"]["impact_parameter_au"] for r in subset],
                    [
                        r["launch"]["collision_energy_au"] * AU_ENERGY_KCAL_MOL
                        for r in subset
                    ],
                    s=22,
                    alpha=0.75,
                    color=colors[status],
                    label=f"{status}: {len(subset)}",
                )
        axes[0].set(
            xlabel="Impact parameter / bohr",
            ylabel="Collision energy / kcal mol$^{-1}$",
            title="Thermal-flux collision samples",
        )
        axes[0].legend(fontsize=8)
        errors = [
            r["max_energy_drift_per_bead"] * AU_ENERGY_KCAL_MOL
            for r in records
            if r["max_energy_drift_per_bead"] is not None
        ]
        axes[1].hist(errors, bins=24, color="#17866B", alpha=0.9)
        axes[1].set(
            xlabel="Maximum energy error per bead / kcal mol$^{-1}$",
            ylabel="Trajectories",
            title="NVE energy diagnostics",
        )
        reactive = next((r for r in records if r["status"] == "product"), None)
        if reactive is not None:
            trajectory = next(
                directory.rglob(f"trajectory-{reactive['trajectory_id']:08d}.npz")
            )
            with np.load(trajectory, allow_pickle=False) as data:
                centroid = data["centroid"]
                times = data["time_au"] * AU_TIME_FS
            for pair, color in zip(
                ((1, 2), (0, 1), (0, 2)), ("#246BCE", "#C74B43", "#9B8A54")
            ):
                distance = np.linalg.norm(
                    centroid[:, pair[0]] - centroid[:, pair[1]], axis=1
                )
                axes[2].plot(
                    times, distance, color=color, lw=1.5, label=f"r{pair[0]}{pair[1]}"
                )
            axes[2].set(
                xlabel="Time / fs",
                ylabel="Centroid distance / bohr",
                title=f"Exchange trajectory {reactive['trajectory_id']}",
            )
            axes[2].legend(fontsize=8)
        else:
            axes[2].text(
                0.5,
                0.5,
                "No exchange observed in this sample",
                ha="center",
                va="center",
                transform=axes[2].transAxes,
            )
            axes[2].set_axis_off()
        for ax in axes:
            ax.grid(alpha=0.18)
        fig.suptitle(subtitle, fontsize=12)
        figures.append((fig, "drpmd"))
    else:
        raise ValueError(f"No recognized benchmark task in {report_file}")
    destination = directory / "figures"
    destination.mkdir(exist_ok=True)
    for fig, name in figures:
        for extension in ("png", "pdf"):
            path = destination / f"{name}.{extension}"
            fig.savefig(path)
            print(path)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    render(parser.parse_args().directory)
