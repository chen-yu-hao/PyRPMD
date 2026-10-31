"""Create numerical comparison evidence and figures from completed BrHCl scans.

All surfaces remain masked where the actual sampling support is insufficient.
Free-energy comparisons remove one additive constant on their common support.
The finite rectangle and retained common wall are part of the reported target.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.brhcl.reference import load_reference_cache
from benchmarks.brhcl.channels import minimax_path
from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.constants import AU_ENERGY_KCAL_MOL, ANGSTROM_TO_AU
from pyrpmd.workflows.fes import write_json

CASES = ("p1", "p8", "p1-mixed", "p8-mixed", "p1-final", "p8-final",
         "p16-final", "p8-halfdt-final")


def load_case(folder):
    folder = Path(folder)
    with load_checkpoint(folder/"fes.npz", expected_kind="arrays", require_integrity=True) as saved:
        arrays = {key:saved[key].copy() for key in saved}
    return {"arrays":arrays, **{name:json.loads((folder/f"{name}.json").read_text())
                               for name in ("plan", "model", "summary", "sampling")}}


def aligned_difference(left, right, beta, *, mask=None):
    valid = np.isfinite(left) & np.isfinite(right)
    if mask is not None:
        valid &= mask
    difference = np.full_like(left, np.nan)
    if not valid.any():
        return difference, {"comparison_bins":0}
    delta = (left-right)[valid] * beta / AU_ENERGY_KCAL_MOL
    offset = float(delta.mean())
    delta -= offset
    difference[valid] = delta
    return difference, {"comparison_bins":int(valid.sum()), "gauge_offset_kbt":offset,
                        "gauge_aligned_rms_kbt":float(np.sqrt(np.mean(delta**2))),
                        "median_absolute_error_kbt":float(np.median(np.abs(delta))),
                        "p90_absolute_error_kbt":float(np.quantile(np.abs(delta), .9)),
                        "maximum_absolute_error_kbt":float(np.max(np.abs(delta)))}


def _compatible(left, right):
    for key in ("model", "atomic_order", "masses_amu", "temperature_K", "beta",
                "cv_names", "wall_angstrom", "wall_k_kcal_mol_angstrom2"):
        if left["plan"][key] != right["plan"][key]:
            raise ValueError(f"comparison targets differ in {key}")
    if left["model"]["library_sha256"] != right["model"]["library_sha256"]:
        raise ValueError("comparison PES libraries differ")
    for key in ("x_edges_angstrom", "y_edges_angstrom"):
        if not np.array_equal(left["arrays"][key], right["arrays"][key]):
            raise ValueError("comparison grids differ")


def build_report(run_root, output):
    run_root, output = Path(run_root), Path(output)
    evidence, figures = output/"evidence", output/"figures"
    evidence.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    cases = {name:load_case(run_root/name) for name in CASES}
    baseline = cases["p1-final"]
    for item in cases.values():
        _compatible(item, baseline)
    settings = baseline["plan"]
    # Reconstruct exactly the bin edges passed to MBAR (not a rounded A->bohr
    # round trip) before checking the independently computed reference cache.
    bounds = np.asarray(settings["ranges_angstrom"])*ANGSTROM_TO_AU
    edges = [np.linspace(*axis, bins+1) for axis,bins in zip(bounds,settings["bins"])]
    exact = load_reference_cache(run_root/"reference32.npz", settings=settings,
                                 library_sha256=baseline["model"]["library_sha256"], edges=edges)
    reference = exact["free_energy"]*AU_ENERGY_KCAL_MOL
    arrays = {"reference_kcal_mol":reference,
              "reference_converged":exact["quadrature_converged"],
              "reference_error_kbt":exact["quadrature_error_kbt"],
              "x_edges_angstrom":baseline["arrays"]["x_edges_angstrom"],
              "y_edges_angstrom":baseline["arrays"]["y_edges_angstrom"]}
    comparisons, summaries = {}, {}
    for name, item in cases.items():
        summary, plan = item["summary"], item["plan"]
        compact = {key:value for key,value in summary.items()
                   if key not in ("per_stream", "bootstrap_diagnostics")}
        compact["production_fs"] = plan["production_steps"]*plan["dt_fs"]
        compact["equilibration_fs"] = plan["equilibration_steps"]*plan["dt_fs"]
        compact["save_interval_fs"] = plan["sample_interval"]*plan["dt_fs"]
        compact["last_invocation_sampling_seconds"] = item["sampling"]["seconds"]
        compact["sampling_time_note"] = item["sampling"]["note"]
        trials = [stream.get("orientation_mc",{}) for stream in summary["per_stream"]]
        attempted = sum(move.get("attempted",0) for move in trials)
        accepted = sum(move.get("accepted",0) for move in trials)
        compact["mc_total_attempted"] = attempted
        compact["mc_total_accepted"] = accepted
        compact["mc_acceptance_fraction"] = accepted/attempted if attempted else None
        compact["third_distance_g_median"] = float(np.median([
            stream["third_distance_statistical_inefficiency"] for stream in summary["per_stream"]]))
        standard_error = item["arrays"]["standard_error_kcal_mol"]
        finite = np.isfinite(standard_error)
        compact["finite_standard_error_bins"] = int(finite.sum())
        compact["median_standard_error_kbt"] = (float(np.median(standard_error[finite]))
                                                  *plan["beta"]/AU_ENERGY_KCAL_MOL if finite.any() else None)
        summaries[name] = compact
        f = item["arrays"]["free_energy_kcal_mol"]
        arrays[f"{name}_free_energy_kcal_mol"] = f
        arrays[f"{name}_weighted_bin_ess"] = item["arrays"]["weighted_bin_ess"]
        if plan["beads"] == 1:
            difference, metric = aligned_difference(f, reference, plan["beta"],
                                                     mask=exact["quadrature_converged"])
            arrays[f"{name}_reference_difference_kbt"] = difference
            comparisons[f"{name}_vs_classical_reference"] = metric
        shutil.copyfile(run_root/name/"plan.json", evidence/f"{name}-plan.json")
        shutil.copyfile(run_root/name/"model.json", evidence/f"{name}-model.json")
        shutil.copyfile(run_root/name/"summary.json", evidence/f"{name}-summary.json")
        shutil.copyfile(run_root/name/"fes.npz", evidence/f"{name}-fes.npz")
    for left,right,label in (("p8-final","p1-final","p8_minus_p1"),
                             ("p16-final","p8-final","p16_minus_p8"),
                             ("p8-halfdt-final","p8-final","halfdt_minus_p8")):
        difference, metric = aligned_difference(cases[left]["arrays"]["free_energy_kcal_mol"],
            cases[right]["arrays"]["free_energy_kcal_mol"], settings["beta"])
        arrays[f"{label}_difference_kbt"] = difference
        comparisons[label] = metric
    matched = (np.isfinite(cases["p1"]["arrays"]["free_energy_kcal_mol"])
               & np.isfinite(baseline["arrays"]["free_energy_kcal_mol"])
               & exact["quadrature_converged"])
    for name in ("p1", "p1-final"):
        delta, metric = aligned_difference(cases[name]["arrays"]["free_energy_kcal_mol"],
                                           reference, settings["beta"], mask=matched)
        arrays[f"{name}_matched_reference_difference_kbt"] = delta
        comparisons[f"{name}_vs_reference_on_matched_support"] = metric
    reference_connections = {}
    basins = baseline["summary"]["basins"]
    for convention, mask in (("same_sampling_support",baseline["arrays"]["mask"]),
                              ("all_converged_reference_bins",~exact["quadrature_converged"])):
        connections = []
        for i,(left,start) in enumerate(basins.items()):
            for right,goal in list(basins.items())[i+1:]:
                connections.append({"from":left,"to":right,
                                    **minimax_path(reference,start,goal,mask=mask)})
        reference_connections[convention] = connections
    quantum = cases["p8-final"]
    restricted_connections = []
    for connection in quantum["summary"]["projected_connections"]:
        excluded = np.zeros_like(quantum["arrays"]["mask"])
        excluded_names = []
        for k,name in enumerate(quantum["arrays"]["channel_names"][:3]):
            if name not in (connection["from"],connection["to"]):
                excluded |= quantum["arrays"]["channel_fractions"][k] >= .8
                excluded_names.append(str(name))
        restricted_connections.append({"from":connection["from"],"to":connection["to"],
            "excluded_resolved_channels":excluded_names,
            **minimax_path(quantum["arrays"]["free_energy_kcal_mol"],
                quantum["summary"]["basins"][connection["from"]],
                quantum["summary"]["basins"][connection["to"]],
                mask=quantum["arrays"]["mask"]|excluded)})
    for name in ("pes-verification.json", "native-smoke.json", "pes-tests.json",
                 "sampling-verification.json", "reference32.json", "reference32.npz",
                 "fes-extra-correlation-verification.json", "sampling-engine.json", "integration-review.json",
                 "cabi-batch-before.json", "cabi-batch-after.json", "integration-verification.json",
                 "pytest-full.txt", "pytest-full.xml"):
        source = run_root/name
        if source.exists():
            shutil.copyfile(source, evidence/name)
    for source in sorted((run_root/"reference-validation").glob("*.json")):
        shutil.copyfile(source,evidence/f"reference-{source.name}")
    result = {"schema":"pyrpmd.brhcl.validation.v1", "physical_converged":False,
              "reference_converged_bins":int(exact["quadrature_converged"].sum()),
              "reference_total_bins":int(exact["quadrature_converged"].size),
              "reference_max_successive_order_error_kbt":float(exact["quadrature_error_kbt"].max()),
              "cases":summaries, "comparisons":comparisons,
              "classical_reference_connections_at_sampled_seed_bins":reference_connections,
              "p8_paths_excluding_other_resolved_channels":restricted_connections,
              "comparison_convention":"Uniform-bin RMS over finite common support after subtracting the mean free-energy difference. Different bead/dt comparisons also contain sampling noise.",
              "scope":"Historical BrHCl_LEPS1; T=2000 K; finite common 6 A all-pair soft wall; equilibrium centroid-density PMF. No rates or branching fractions."}
    write_json(evidence/"comparisons.json", result)
    atomic_savez(evidence/"plot-data.npz", kind="arrays", **arrays)
    draw_figures(arrays, cases, comparisons, figures)
    return result


def draw_figures(arrays, cases, comparisons, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patheffects

    plt.rcParams.update({"font.size":10, "axes.titlesize":11,
                         "axes.labelsize":11, "savefig.facecolor":"white"})
    xedge, yedge = arrays["x_edges_angstrom"], arrays["y_edges_angstrom"]
    xc, yc = (xedge[1:]+xedge[:-1])/2, (yedge[1:]+yedge[:-1])/2
    energy_cmap = plt.colormaps["viridis"].copy()
    energy_cmap.set_bad("#e5e7eb")
    error_cmap = plt.colormaps["RdBu_r"].copy()
    error_cmap.set_bad("#e5e7eb")

    def panel(axis, values, title, *, error=False, limit=1):
        image = axis.pcolormesh(xedge,yedge,values.T,shading="flat",
                               cmap=error_cmap if error else energy_cmap,
                               vmin=-limit if error else 0, vmax=limit if error else 75,
                               rasterized=True)
        axis.set(xlabel=r"$r_{\mathrm{HBr}}$ ($\AA$)",ylabel=r"$r_{\mathrm{HCl}}$ ($\AA$)",title=title)
        axis.set_aspect("equal")
        axis.set_xticks([1,2,3,4,5]); axis.set_yticks([1,2,3,4,5])
        return image

    fig,axes = plt.subplots(1,3,figsize=(14.5,4.9),constrained_layout=True)
    panel(axes[0],arrays["reference_kcal_mol"],"P=1: independent configurational integral")
    panel(axes[1],arrays["p1-final_free_energy_kcal_mol"],"P=1: sampled centroid PMF")
    image = panel(axes[2],arrays["p8-final_free_energy_kcal_mol"],"P=8: sampled centroid PMF")
    summary = cases["p8-final"]["summary"]
    names = {"H+BrCl":"R", "HCl+Br":"A", "HBr+Cl":"B"}
    for name,index in summary["basins"].items():
        i,j = index
        axes[2].plot(xc[i],yc[j],"o",ms=6,color="white",mec="black",zorder=5)
        offset = (-14 if i>len(xc)/2 else 8, -16 if j>len(yc)/2 else 8)
        label = axes[2].annotate(names[name],(xc[i],yc[j]),xytext=offset,
                                 textcoords="offset points",weight="bold",color="white")
        label.set_path_effects([patheffects.withStroke(linewidth=2,foreground="black")])
    for connection in summary["projected_connections"]:
        if connection["from"]=="HCl+Br" and connection["to"]=="HBr+Cl" and connection["connected"]:
            route = np.asarray(connection["path"])
            axes[2].plot(xc[route[:,0]],yc[route[:,1]],"--",lw=1.7,color="white",
                         path_effects=[patheffects.withStroke(linewidth=2.5,foreground="#222222")])
    fig.colorbar(image,ax=axes,label="Relative PMF (kcal mol$^{-1}$)",shrink=.82,extend="max")
    fig.suptitle("H + BrCl  →  HCl + Br / HBr + Cl\nBrHCl_LEPS1 model · 2000 K · common 6 Å soft wall",fontsize=14)
    fig.supxlabel("Gray: insufficient sampling support. R: H+BrCl; A: HCl+Br; B: HBr+Cl. Dashed line: a supported projected A–B path, not a dynamical trajectory.",fontsize=9)
    fig.savefig(output/"brhcl-2d-pmf.png",dpi=200)
    fig.savefig(output/"brhcl-2d-pmf.pdf")
    plt.close(fig)

    fig,axes = plt.subplots(2,2,figsize=(10.5,10),constrained_layout=True)
    definitions = [
        ("p1_matched_reference_difference_kbt","p1_vs_reference_on_matched_support","Plain MD: P=1 minus reference",4.),
        ("p1-final_matched_reference_difference_kbt","p1-final_vs_reference_on_matched_support","Long MD + orientation MC: P=1 minus reference",1.),
        ("p16_minus_p8_difference_kbt","p16_minus_p8","Bead sensitivity: P=16 minus P=8",1.),
        ("halfdt_minus_p8_difference_kbt","halfdt_minus_p8","Timestep sensitivity: 0.05 fs minus 0.10 fs",1.),
    ]
    for axis,(key,metric,title,limit) in zip(axes.flat,definitions):
        rms = comparisons[metric]["gauge_aligned_rms_kbt"]
        image = panel(axis,arrays[key],f"{title}\nRMS = {rms:.3f} $k_BT$",error=True,limit=limit)
        fig.colorbar(image,ax=axis,label="Aligned difference ($k_BT$)",extend="both",shrink=.88)
    fig.suptitle("Independent accuracy and sensitivity checks\nGray bins are excluded, not interpolated",fontsize=14)
    fig.supxlabel("The upper panels use identical supported bins. Each comparison removes one additive constant. Bead and timestep differences include statistical noise.",fontsize=9)
    fig.savefig(output/"brhcl-validation.png",dpi=200)
    fig.savefig(output/"brhcl-validation.pdf")
    plt.close(fig)

    fig,axes = plt.subplots(1,2,figsize=(10.5,4.2),constrained_layout=True)
    stages = [("p1","p8"),("p1-mixed","p8-mixed"),("p1-final","p8-final")]
    pos = np.arange(3)
    for series,(label,color,shift) in enumerate((("P=1","#2166ac",-.16),("P=8","#b2182b",.16))):
        vals = [cases[pair[series]]["summary"]["initial_family_comparison"]["gauge_aligned_rms_kbt"] for pair in stages]
        axes[0].bar(pos+shift,vals,width=.3,label=label,color=color)
        for x,y in zip(pos+shift,vals):
            axes[0].text(x,y+.08,f"{y:.2f}",ha="center",fontsize=9)
    axes[0].set(xticks=pos,xticklabels=["Plain MD\n1 ps / stream","MD + MC32\n1 ps / stream","MD + MC64\n10 ps / stream"],
                ylabel="Compact / open PMF RMS ($k_BT$)",title="Independent starting-family agreement")
    axes[0].legend(frameon=False)
    for label,color,run in (("P=1","#2166ac","p1-final"),("P=8","#b2182b","p8-final")):
        blocks = cases[run]["summary"]["bootstrap_block_plan"]["blocks_per_stream"]
        axes[1].hist(blocks,bins=np.linspace(0,200,21),alpha=.55,label=label,color=color)
    axes[1].axvline(20,color="#333333",ls="--",lw=1)
    axes[1].set(xlabel="Conservative 5g blocks per stream",ylabel="Number of streams",
                title="Correlation check includes the hidden distance")
    axes[1].legend(frameon=False)
    fig.suptitle("Window overlap alone did not detect slow hidden-coordinate mixing",fontsize=13)
    fig.supxlabel("Family RMS uses each run's common compact/open support; reported block counts do not establish complete mixing.",fontsize=9)
    fig.savefig(output/"brhcl-mixing.png",dpi=200)
    plt.close(fig)

    from matplotlib.colors import BoundaryNorm, ListedColormap
    fig,axes = plt.subplots(1,2,figsize=(12,5.5),constrained_layout=True)
    data = cases["p8-final"]["arrays"]
    conditional = np.where(np.isfinite(data["channel_fractions"]), data["channel_fractions"], -1.)
    classes = conditional.argmax(axis=0)
    classes[conditional.max(axis=0)<.8] = 3
    colors = ["#e9c46a", "#2878b5", "#c95151", "#ece4f2"]
    cmap = ListedColormap(colors); cmap.set_bad("#b8bcc2")
    mapped = np.ma.array(classes,mask=data["mask"])
    image = axes[0].pcolormesh(xedge,yedge,mapped.T,cmap=cmap,
                               norm=BoundaryNorm(np.arange(5)-.5,4),shading="flat")
    axes[0].set(xlabel=r"$r_{\mathrm{HBr}}$ ($\AA$)",ylabel=r"$r_{\mathrm{HCl}}$ ($\AA$)",
                title="P=8: conditional channel identity (at least 80%)")
    axes[0].set_aspect("equal")
    bar = fig.colorbar(image,ax=axes[0],ticks=np.arange(4),orientation="horizontal",shrink=.95)
    bar.ax.set_xticklabels(["R: H+BrCl","A: HCl+Br","B: HBr+Cl","Mixed / intermediate"],fontsize=8)
    positions = {"H+BrCl":(.5,.85),"HCl+Br":(.12,.17),"HBr+Cl":(.88,.17)}
    for connection in summary["projected_connections"]:
        if not connection["connected"]:
            continue
        a,b = np.array(positions[connection["from"]]),np.array(positions[connection["to"]])
        axes[1].plot(*np.stack((a,b)).T,color="#525964",lw=2)
        middle = (a+b)/2
        axes[1].text(*middle,f"{connection['bottleneck']:.1f}",ha="center",va="center",fontsize=11,
                     bbox={"facecolor":"white","edgecolor":"none","pad":4})
    for name,index in summary["basins"].items():
        color = colors[list(positions).index(name)]
        foreground = "#222222" if name=="H+BrCl" else "white"
        axes[0].plot(xc[index[0]],yc[index[1]],"o",color="black",ms=5)
        axes[1].text(*positions[name],f"{names[name]}: {name}\nF = {data['free_energy_kcal_mol'][tuple(index)]:.1f}",
                     ha="center",va="center",color=foreground,fontsize=11,
                     bbox={"boxstyle":"round,pad=.6","facecolor":color,"edgecolor":"#555555"})
    axes[1].set(xlim=(-.1,1.1),ylim=(-.05,1.05),title="Supported four-neighbor connectivity")
    axes[1].set_axis_off()
    axes[1].text(.5,-.02,"Nodes: seed-bin PMF. Edges: minimax path maximum F.\nSame arbitrary zero; kcal mol$^{-1}$. Edges are not directional barriers.",
                 ha="center",va="center",fontsize=10)
    fig.suptitle("Three chemical arrangements are resolved in the sampled 2D projection",fontsize=14)
    fig.supxlabel("Gray: unsupported. Black dots: selected seed bins. Graph edges show projected reachability, not a transition state, reaction rate, or branching fraction.",fontsize=9)
    fig.savefig(output/"brhcl-channels.png",dpi=200)
    fig.savefig(output/"brhcl-channels.pdf")
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    report=build_report(args.runs,args.output)
    print(json.dumps(report["comparisons"],indent=2))
