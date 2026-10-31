"""Summarize actual runs, rebin saved weights and render scientific figures."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyrpmd.workflows.fes import write_json
from ogre_benchmark import oracle, BETA, KJ

LABELS = {"fixed":"Fixed umbrella", "adaptive":"Adaptive umbrella", "unbiased":"Unbiased",
          "standard":"Standard MetaD", "well_tempered":"WT-MetaD"}
METHODS = list(LABELS)


def rebin(case, model):
    """Histogram the same normalized MBAR weights; no new sampling or fitting."""
    case=Path(case)
    with h5py.File(case/"run/fes.h5","r") as h:
        values=np.concatenate([h["streams"][k]["samples"][:] for k in sorted(h["streams"],key=int)])
        weights=h["weights"][:]
        old_edges=[h["edges_0"][:],h["edges_1"][:]]
    with np.load(case/"run/fes.npz",allow_pickle=False) as original:
        # Independently verify stream ordering before reusing weights.
        check=np.histogramdd(values,bins=old_edges,weights=weights)[0]
        assert np.max(abs(check-original["probability_mass"]))<1e-14
        assert np.array_equal(np.histogramdd(values,bins=old_edges)[0],original["raw_count"])
    bins=(50,100) if model=="double_well" else (240,240)
    edges=[np.linspace(e[0],e[-1],n+1) for e,n in zip(old_edges,bins)]
    mass=np.histogramdd(values,bins=edges,weights=weights)[0]
    count=np.histogramdd(values,bins=edges)[0]
    density=mass/(np.diff(edges[0])[:,None]*np.diff(edges[1])[None,:])
    good=(count>=5)&(density>0)&np.isfinite(density)
    f=np.full(good.shape,np.nan);f[good]=-np.log(density[good])*KJ/BETA
    exact,point=oracle(edges,model,order=10)
    residual=f-exact;residual-=np.mean(residual[good])
    paper=(f-np.min(f[good]))-(point-np.min(point[good]))
    result={"case":case.name,"model":model,"bins":list(bins),
            "bin_widths":[float(e[1]-e[0]) for e in edges],"supported_bins":int(good.sum()),
            "total_bins":good.size,"coverage":float(good.mean()),
            "bin_integral_mean_gauge_rmse_kjmol":float(np.sqrt(np.mean(residual[good]**2))),
            "paper_point_min_gauge_rmse_kjmol":float(np.sqrt(np.mean(paper[good]**2))),
            "note":"Same saved samples and MBAR weights; verified coarse histogram equality; missing bins excluded and counted. Grid widths follow the published source scripts, not an assertion of identical trajectories or estimator."}
    write_json(case/"paper-grid.json",result)
    np.savez_compressed(case/"paper-grid.npz",free_energy_kjmol=f,reference_kjmol=exact,
                        residual_kjmol=residual,mask=~good,edges_0=edges[0],edges_1=edges[1])
    return result


def range_stats(rows,key):
    values=np.asarray([r[key] for r in rows],float)
    return {"mean":float(values.mean()),"min":float(values.min()),"max":float(values.max()),
            "sd":float(values.std(ddof=1)) if len(values)>1 else None}


def collect(base):
    base=Path(base)
    harmonic=[json.loads(p.read_text()) for p in sorted((base/"harmonic").glob("*/comparison.json"))]
    ogre=[json.loads(p.read_text()) for p in sorted((base/"ogre").glob("*/comparison.json"))]
    failures=[]
    for p in sorted((base/"ogre").glob("*/failure.json")):
        plan=json.loads((p.parent/"run/adaptive-plan.json").read_text())
        failures.append({"case":p.parent.name,"status":plan["status"],"failure_reason":plan["failure_reason"],
            "pilot_rounds":len(plan["rounds"]),"production_started":False,
            "min_effective_samples_final":min(plan["rounds"][-1]["effective_samples_per_window"]),
            "overlap_target_met":plan["rounds"][-1]["overlap_connected_at_target"],
            "target_effective_samples":plan["options"]["min_effective_samples"],
            "integration_steps":sum(len(r["centers"])*(r["pilot"]["equilibration_steps"]+r["pilot"]["production_steps"]) for r in plan["rounds"])})
    aggregate=[]
    for beads in (1,8):
        for method in METHODS:
            rows=[r for r in harmonic if r["beads"]==beads and r["method"]==method]
            if rows:aggregate.append({"model":"harmonic","beads":beads,"method":method,"scale":1,"seeds":len(rows),
                "rmse":range_stats(rows,"rmse_kbt"),"error_unit":"kBT", "coverage":range_stats(rows,"roi_coverage"),
                "integration_steps":range_stats(rows,"integration_steps"),"windows":range_stats(rows,"windows")})
    for model in ("double_well","ackley"):
        for scale in (1,5):
            for method in METHODS:
                rows=[r for r in ogre if r["model"]==model and r["method"]==method and r["scale"]==scale]
                if rows:aggregate.append({"model":model,"beads":1,"method":method,"scale":scale,"seeds":len(rows),
                    "rmse":range_stats(rows,"rmse_kjmol"),"error_unit":"kJ/mol", "coverage":range_stats(rows,"coverage"),
                    "paper_point_min_gauge_rmse":range_stats(rows,"paper_point_min_gauge_rmse_kjmol"),
                    "integration_steps":range_stats(rows,"integration_steps"),"windows":range_stats(rows,"windows")})
    paper_grid=[]
    for p in sorted((base/"ogre").glob("*/comparison.json")):
        row=json.loads(p.read_text())
        if (row["model"]=="ackley" and row["method"]=="fixed") or (row["model"]=="double_well" and row["scale"]==5):
            paper_grid.append(rebin(p.parent,row["model"]))
    suites=ET.parse(base/"full-tests.xml").getroot()
    test_summary=[dict(e.attrib) for e in suites.iter("testsuite")]
    return {"source_commit":"6fde29d2039c01921f00393bf17830b939af131b",
            "completed_fes_runs":len(harmonic)+len(ogre),"incomplete_adaptive_runs":len(failures),
            "harmonic":harmonic,"ogre":ogre,"failures":failures,"aggregate":aggregate,"paper_grid":paper_grid,
            "tests":test_summary,"total_fes_integration_steps":sum(r["integration_steps"] for r in harmonic+ogre+failures),
            "limitations":["Different sampling budgets; no equal-cost efficiency ranking.",
              "Three independent seeds give observed variation, not a calibrated confidence interval.",
              "OGRe model tests are P=1, matching the classical analytic reference; no comparison of a quantum centroid FES to bare V.",
              "No molecular RPMD rate or activated nonharmonic quantum FES has been reproduced here.",
              "Full source suite skips unavailable BKMP2, EAM and PyTorch cases; skips are not passes.",
              "Ackley has a common exterior confining wall outside |CV|=8; the [-6,6]^2 local reference is unchanged."]}


def plots(base,out,summary):
    base=Path(base);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout="constrained")
    for beads,shift,color in [(1,-.17,"#277da1"),(8,.17,"#e76f51")]:
        rows=[next(r for r in summary["aggregate"] if r["model"]=="harmonic" and r["beads"]==beads and r["method"]==m) for m in METHODS]
        for ax,key in zip(axes,["rmse","coverage"]):
            values=np.array([r[key]["mean"] for r in rows]);low=[r[key]["min"] for r in rows];high=[r[key]["max"] for r in rows]
            ax.bar(np.arange(5)+shift,values,.32,color=color,label=f"P={beads}")
            ax.errorbar(np.arange(5)+shift,values,yerr=[values-low,high-values],fmt="none",ecolor="#222",capsize=3)
            ax.set_xticks(range(5),[LABELS[m] for m in METHODS],rotation=20,ha="right")
    axes[0].set_ylabel("RMSE / kBT on supported bins");axes[1].set_ylabel("Coverage of F <= 4 kBT region")
    axes[1].set_ylim(0,1.05);axes[0].legend()
    fig.suptitle("2D harmonic centroid FES | three seeds, whiskers = observed range\nDifferent run budgets; missing bins retained")
    fig.savefig(out/"harmonic-methods.png",dpi=180);fig.savefig(out/"harmonic-methods.pdf");plt.close(fig)

    cases=[("fixed",5),("adaptive",5),("unbiased",1),("standard",1),("well_tempered",1)]
    fig,axes=plt.subplots(2,6,figsize=(17,6.4),layout="constrained",sharex=True,sharey=True)
    first=np.load(base/"ogre/double_well-fixed-s4101-n5/comparison.npz")
    reference=first["reference_kjmol"]-first["reference_kjmol"].min()
    edge=[first["edges_0"],first["edges_1"]]
    im=axes[0,0].pcolormesh(*edge,reference.T,cmap="viridis",vmin=0,vmax=11,shading="flat")
    axes[0,0].set_title("Exact bin integral");axes[1,0].axis("off")
    for j,(method,scale) in enumerate(cases,1):
        path=base/f"ogre/double_well-{method}-s4101-n{scale}"
        a=np.load(path/"comparison.npz");d=json.loads((path/"comparison.json").read_text())
        values=a["reference_kjmol"]-first["reference_kjmol"].min()+a["residual_kjmol"]
        values=np.ma.masked_where(a["mask"],values)
        error=np.ma.masked_where(a["mask"],a["residual_kjmol"])
        axes[0,j].pcolormesh(*edge,values.T,cmap="viridis",vmin=0,vmax=11,shading="flat")
        diff=axes[1,j].pcolormesh(*edge,error.T,cmap="RdBu_r",vmin=-3,vmax=3,shading="flat")
        for ax in axes[:,j]:ax.set_facecolor("#dedede")
        axes[0,j].set_title(f"{LABELS[method]}\n{d['integration_steps']/1e6:.2f} M steps")
        axes[1,j].set_title(f"RMSE {d['rmse_kjmol']:.2f}; coverage {100*d['coverage']:.1f}%",fontsize=9)
    for ax in axes[-1,1:]:ax.set_xlabel("CV x")
    axes[0,0].set_ylabel("CV y");axes[1,1].set_ylabel("CV y")
    fig.colorbar(im,ax=axes[0,:],label="Free energy / kJ mol^-1",shrink=.75)
    fig.colorbar(diff,ax=axes[1,1:],label="Error / kJ mol^-1",shrink=.75)
    fig.suptitle("OGRe double-well model, 300 K, P=1 | seed 4101\nGray = unsupported; budgets differ; MetaD is a finite-time point estimate")
    fig.savefig(out/"double-well-comparison.png",dpi=170);fig.savefig(out/"double-well-comparison.pdf");plt.close(fig)

    a=np.load(base/"ogre/ackley-fixed-s4101-n1/paper-grid.npz")
    edge=[a["edges_0"],a["edges_1"]]
    ref=a["reference_kjmol"]-a["reference_kjmol"].min()
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout="constrained")
    for ax,data,title in zip(axes,[ref,ref+a["residual_kjmol"],a["residual_kjmol"]],["Exact bin integral","Fixed umbrella, 169 windows","Error; gray = unsupported"]):
        if ax is not axes[0]:data=np.ma.masked_where(a["mask"],data)
        last=ax is axes[2]
        im=ax.pcolormesh(*edge,data.T,shading="flat",cmap="RdBu_r" if last else "viridis",vmin=-4 if last else 0,vmax=4 if last else 15)
        ax.set_facecolor("#dedede");ax.set_title(title);ax.set_xlabel("CV x");ax.set_ylabel("CV y");ax.set_aspect("equal")
        fig.colorbar(im,ax=ax,label="kJ/mol",shrink=.85)
    fig.suptitle("Ackley model | published-source bin width 0.05, same saved samples\nAdaptive pilots failed ESS target in all three seeds; no adaptive production FES")
    fig.savefig(out/"ackley-paper-grid.png",dpi=180);fig.savefig(out/"ackley-paper-grid.pdf");plt.close(fig)

    q=json.loads((base/"harmonic/harmonic-quantum.json").read_text())["rows"]
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    p=np.array([r["beads"] for r in q]);axes[0].plot(p,[r["finite_P_exact_bead_second_moment"] for r in q],"-",label="Finite-P exact")
    axes[0].errorbar(p,[r["bead_second_moment"] for r in q],yerr=[r["bead_batch_se"] for r in q],fmt="o",label="Actual sampling, batch SE")
    axes[0].axhline(q[0]["infinite_P_exact_bead_second_moment"],ls="--",color="gray",label="Quantum limit")
    axes[0].axhline(.25,ls=":",color="black",label="Exact centroid variance")
    axes[0].set_xlabel("Beads P");axes[0].set_ylabel("Position second moment / bohr^2");axes[0].legend(fontsize=8)
    for dt,color in [(.04,"#277da1"),(.02,"#e76f51")]:
        values=[next(z for z in r["deterministic_stationary_checks"] if z["dt"]==dt)["relative_error_vs_finite_P"] for r in q]
        axes[1].plot(p,values,"o-",label=f"dt={dt}",color=color)
    axes[1].set_yscale("log");axes[1].set_xlabel("Beads P");axes[1].set_ylabel("Stationary variance relative discretization error");axes[1].legend()
    fig.suptitle("Quantum harmonic validation | beta*hbar*omega = 4\nFull bead distribution checked separately from centroid FES")
    fig.savefig(out/"quantum-checks.png",dpi=180);fig.savefig(out/"quantum-checks.pdf");plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--base",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();s=collect(a.base);a.output.mkdir(parents=True,exist_ok=True)
    write_json(a.output/"summary.json",s)
    rows=[]
    for r in s["aggregate"]:
        row={k:v for k,v in r.items() if not isinstance(v,dict)}
        for key,value in r.items():
            if isinstance(value,dict):
                for stat,n in value.items():row[key+"_"+stat]=n
        rows.append(row)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (a.output/"method-summary.csv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    plots(a.base,a.output,s)
    print(json.dumps({"completed":s["completed_fes_runs"],"incomplete":s["incomplete_adaptive_runs"],"integration_steps":s["total_fes_integration_steps"],"output":str(a.output)}))


if __name__=="__main__":main()
