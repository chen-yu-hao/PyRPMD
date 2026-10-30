"""Render measured PMF validation evidence; no chemistry is synthesized."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(data, output):
    data, output = Path(data), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    one = json.loads((data / "1d-accuracy.json").read_text())
    two = json.loads((data / "2d-accuracy.json").read_text())
    archive = data / "plot-data.json"
    archived = json.loads(archive.read_text()) if archive.is_file() else None
    if archived is not None:
        grids = {k: np.asarray(v, dtype=bool if k == "mask" else float)
                 for k, v in archived["analytic_2d"].items()}
    else:
        with np.load(data / "2d-nonsquare-grids.npz") as raw:
            grids = {k: raw[k] for k in raw.files}
    beta = two["beta"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    fig.suptitle("Independent coupled-Gaussian PMF validation | analytic test model", fontsize=15)
    arrays = [grids["exact_free_energy_aligned"]*beta, grids["free_energy"]*beta,
              (grids["free_energy"]-grids["exact_free_energy_aligned"])*beta]
    for i, (ax, field, title) in enumerate(zip(axes, arrays, ["Exact bin-integrated reference", "PyRPMD MBAR reconstruction", "Reconstruction minus reference"])):
        kw = {"cmap":"viridis", "vmin":0, "vmax":6} if i<2 else {"cmap":"RdBu_r", "vmin":-.2,"vmax":.2}
        im = ax.pcolormesh(grids["x_edges"],grids["y_edges"],np.where(grids["mask"],np.nan,field).T,shading="flat",**kw)
        ax.set(xlabel="x / test CV unit", ylabel="y / test CV unit",title=title)
        fig.colorbar(im,ax=ax,label="Free energy / kBT")
    fig.savefig(output / "analytic-2d-validation.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    periodic = two["periodic_regression"]
    values = [periodic["baseline_histogram_frames"],periodic["fixed_histogram_frames"]]
    axes[0].bar(["Before fix", "After fix"],values,color=["#B56A52","#287E8C"])
    axes[0].axhline(periodic["input_frames"],color="#333333",ls="--",lw=1)
    axes[0].set(title="Equivalent periodic angles retained",ylabel="Frames in histogram",ylim=(0,900))
    for i,v in enumerate(values): axes[0].text(i,v+15,str(v),ha="center")
    ar = one["correlated_ui_bootstrap_vs_ar1_covariance"]
    ratios = [ar["legacy"]["ratio_to_analytic"], ar["revised"]["ratio_to_analytic"]]
    axes[1].bar(["Old blocks: g", "New blocks: 5g"],ratios,color=["#B56A52","#287E8C"])
    axes[1].axhline(1,color="#333333",ls="--",lw=1,label="AR(1) covariance oracle")
    axes[1].set(title="Correlated PMF uncertainty",ylabel="Estimated SE / analytic SE",ylim=(0,1.18))
    for i,v in enumerate(ratios): axes[1].text(i,v+.025,f"{v:.3f}",ha="center")
    axes[1].legend(loc="upper left",fontsize=8)
    fig.savefig(output / "pmf-regression-fixes.png",dpi=160)
    plt.close(fig)

    if archived is not None:
        report = archived["h3_estimator_comparison"]
        g = {k: np.asarray(v, dtype=bool if k == "mask" else float)
             for k, v in archived["h3_2d"].items()}
    else:
        report = json.loads((data / "h3-rate" / "pmf-diagnostics.json").read_text())
        with np.load(data / "h3-fes2d" / "run" / "fes.npz") as raw:
            g = {k:raw[k] for k in raw.files}
    fig, axes = plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    fig.suptitle("Real H + H2 / BKMP2 | 1500 K, P=16 | finite sampling, not converged",fontsize=14)
    energy = g["free_energy"]*627.509474
    im=axes[0].pcolormesh(g["edges_0"],g["edges_1"],np.where(g["mask"],np.nan,energy).T,
                         shading="flat",cmap="viridis")
    axes[0].set(xlabel="Centroid r01 / bohr",ylabel="Centroid r12 / bohr",title="2D distance-density PMF (30,000 frames)")
    fig.colorbar(im,ax=axes[0],label="PMF / kcal mol$^{-1}$")
    ui=report["ui"]["curve"]
    axes[1].plot(ui["xi"],np.array(ui["free_energy_hartree"])*627.509474,label="Gaussian UI",lw=2)
    for entry in report["per_bins"]:
        curve=entry["curve"]
        values=np.array([np.nan if x is None else x for x in curve["free_energy_hartree"]])*627.509474
        axes[1].plot(curve["xi"],values,label=f"MBAR, {entry['bins']} bins",lw=1.1)
    axes[1].axvline(report["fixed_xi_star"],ls="--",color="#666666",lw=1)
    axes[1].set(xlabel="RPMDrate coordinate xi",ylabel="PMF / kcal mol$^{-1}$",title="Same samples, fixed dividing surface",xlim=(0,1.12),ylim=(-2,24))
    axes[1].legend(fontsize=8)
    fig.savefig(output / "h3-real-pmf-validation.png",dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    render(args.data,args.output)
