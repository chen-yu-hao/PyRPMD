"""Plot checked-in scientific validation numbers; does not run trajectories."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    args=parser.parse_args()
    harmonic=json.loads((args.directory/"harmonic_summary.json").read_text())["results"]
    lj=json.loads((args.directory/"lj_summary.json").read_text())["results"]
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                         "axes.grid":True,"grid.alpha":.18,"savefig.dpi":180})
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout="constrained")
    exact=harmonic["quantum_reference"]
    cases=harmonic["gaussian_estimator_cases"]
    sample_counts={c["samples"] for c in cases}
    sample_label=(f"{next(iter(sample_counts)):,} Gaussian draws" if len(sample_counts)==1 else "Gaussian samples")
    p=np.array([c["beads"] for c in cases])
    crystal_atoms=harmonic["parameters"]["atoms"]
    for ax,key,title in ((axes[0,0],"energy",f"{crystal_atoms}-atom harmonic crystal: SC energy"),
                         (axes[0,1],"heat_capacity_kB",f"{crystal_atoms}-atom harmonic crystal: SC heat capacity")):
        reference=np.array([c["finite_bead_reference"][key] for c in cases])
        production=np.array([c["deterministic_production_moments"][key] for c in cases])
        sample_key = "internal_energy_hartree" if key=="energy" else "heat_capacity_over_kB"
        error_key = "internal_energy_standard_error_hartree" if key=="energy" else "heat_capacity_standard_error_over_kB"
        sampled=np.array([c["production_estimators"][sample_key] for c in cases])
        errors=np.array([c["production_estimators"][error_key] for c in cases])
        ax.plot(p,reference,"-",color="#276b9e",label="Gaussian partition reference")
        ax.plot(p,production,"o",fillstyle="none",markersize=7,color="#d56c38",label="Production estimator, exact moments")
        ax.errorbar(p*1.04,sampled,yerr=1.96*errors,fmt="x",color="#73588e",capsize=3,
                    label=sample_label+": +/-1.96 SE")
        ax.axhline(exact[key],color="#365d47",ls="--",label="Continuum quantum limit")
        ax.set(xscale="log",title=title,xlabel="Beads P",ylabel="E (model units)" if key=="energy" else r"$C_V/k_\mathrm{B}$")
        ax.set_xticks(p,labels=p)
        ax.legend(fontsize=8)
    ax=axes[1,0]
    atoms=lj["parameters"]["atoms"]
    for method,marker,color in (("sc","o","#276b9e"),("primitive","s","#d56c38")):
        cases=sorted([c for c in lj["cases"] if c["external_pressure"] is None and c["dt"]==.04 and c["method"]==method],key=lambda c:c["beads"])
        x=[c["beads"] for c in cases]
        means=np.array([c["statistics"]["internal_energy_hartree"]["mean"] for c in cases])/atoms
        limits=np.array([c["statistics"]["internal_energy_hartree"]["confidence_interval_95"] for c in cases]).T/atoms
        ax.errorbar(x,means,yerr=np.array([means-limits[0],limits[1]-means]),fmt=marker+"-",color=color,capsize=3,label=method.upper())
    ax.set(xscale="log",title=f"{atoms}-atom FCC solid: bead refinement",xlabel="Beads P",ylabel=r"$E/(N\varepsilon)$")
    ax.set_xticks([4,8,16,32,64],labels=[4,8,16,32,64]); ax.legend(fontsize=8)
    ax=axes[1,1]
    cases=[c for c in lj["cases"] if c["external_pressure"] is not None and c["dt"]==.04]
    labels=[f"{('RPC' if c['method']=='rpc8' else c['method'].upper())}\nP={c['beads']}"+
            ("\nPc=4" if c["method"]=="rpc" else "\nPc=8" if c["method"]=="rpc8" else "") for c in cases]
    means=np.array([c["statistics"]["volume"]["mean"] for c in cases])
    limits=np.array([c["statistics"]["volume"]["confidence_interval_95"] for c in cases]).T
    for index,case in enumerate(cases):
        is_piglet=case["method"]=="piglet"
        ax.errorbar(index,means[index],yerr=[[means[index]-limits[0,index]],[limits[1,index]-means[index]]],
                    fmt="x" if is_piglet else "o",color="#b23838" if is_piglet else "#365d47",capsize=3)
    ax.set_xticks(np.arange(len(cases)),labels=labels,fontsize=8)
    ax.set(title=f"{atoms}-atom FCC solid: isotropic NPT at p*=1",ylabel=r"$V/\sigma^3$")
    replicas={c["statistics"]["volume"]["independent_replicas"] for c in lj["cases"]}
    replica_label=str(next(iter(replicas))) if len(replicas)==1 else "independent"
    fig.suptitle("Condensed-phase NQE validation: model solids\nBottom panels: 95% Student intervals across "+replica_label+" replicas; not convergence bounds\nPIGLET P4 (red cross): configurational-pressure check failed; see report",fontsize=12,fontweight="bold")
    fig.savefig(args.directory/"validation.png")
    fig.savefig(args.directory/"validation.pdf")


if __name__=="__main__":
    main()
