"""Audit all optimization attempts and plot explicitly matched comparisons.

The common-support score is recalculated after intersecting masks and fitting
each surface's additive constant on that same intersection. Coarse and paper
grids remain separate. No new dynamics or oracle-guided window placement is
performed here. Partial inventories can be collected; use --require-complete
for the final figures' three-seed comparisons.
"""
from __future__ import annotations
import argparse
import csv
from functools import lru_cache
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ogre_benchmark import oracle

SEEDS = (4101, 4102, 4103)
COLORS = ("#236b8e", "#d16f32", "#6f5393")
GRIDS = ("coarse", "paper")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def csv_rows(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        writer=csv.DictWriter(f, fieldnames=keys)
        writer.writeheader();writer.writerows(rows)


def stats(values):
    v=np.asarray(values, float)
    return dict(mean=float(v.mean()), min=float(v.min()), max=float(v.max()),
                sd=float(v.std(ddof=1)) if len(v)>1 else None)


@lru_cache(maxsize=128)
def surface(case_path, grid):
    path=Path(case_path)
    file=path/("comparison.npz" if grid=="coarse" else "paper-grid.npz")
    with np.load(file, allow_pickle=False) as z:
        data={k:z[k].copy() for k in z.files}
    report=json.loads((path/"comparison.json").read_text())
    if "point_reference_kjmol" not in data:
        _, data["point_reference_kjmol"]=oracle([data["edges_0"],data["edges_1"]],report["model"],point=True)
    data["good"]=(~data["mask"]) & np.isfinite(data["free_energy_kjmol"]) & np.isfinite(data["reference_kjmol"])
    return data


def score(data, good):
    if not np.any(good):
        raise ValueError("No shared support to compare")
    f=data["free_energy_kjmol"]
    ref=data["reference_kjmol"]
    residual=f-ref
    offset=float(residual[good].mean())
    point=data["point_reference_kjmol"]
    paper=(f[good]-f[good].min())-(point[good]-point[good].min())
    return dict(rmse_kjmol=float(np.sqrt(np.mean((residual[good]-offset)**2))),
                additive_offset_kjmol=offset,
                point_min_gauge_rmse_kjmol=float(np.sqrt(np.mean(paper**2))))


def collect(base, optimized):
    cases=[]; failures=[]; pending=[]
    for source, root in (("baseline",base),("optimized",optimized)):
        for path in sorted(root.iterdir()):
            if not path.is_dir() or not (path/"input-resolved.json").exists():
                continue
            if not (path/"comparison.json").exists():
                entry=dict(source=source,case=path.name,path=str(path))
                if (path/"failure.json").exists():
                    entry["failure"]=json.loads((path/"failure.json").read_text())
                    plan_path=path/"run/adaptive-plan.json"
                    if plan_path.exists():
                        plan=json.loads(plan_path.read_text())
                        entry.update(plan_status=plan.get("status"),
                                     integration_steps=sum(len(r["centers"])*(r["pilot"]["production_steps"]+r["pilot"]["equilibration_steps"]) for r in plan["rounds"]))
                    failures.append(entry)
                else:
                    pending.append(entry)
                continue
            d=json.loads((path/"comparison.json").read_text())
            cfg=json.loads((path/"input-resolved.json").read_text())["config"]
            label=d.get("label",f"{d['method']}-n{d.get('scale',1)}")
            case=dict(source=source,case=path.name,path=str(path),label=label,model=d["model"],
                      method=d["method"],seed=d["seed"],integration_steps=d["integration_steps"],
                      pilot_steps=d.get("pilot_steps",0),windows=d["windows"],samples=d["samples"],
                      dt_au=cfg["dt_au"],andersen_interval=cfg["sampling"]["andersen_interval"],
                      production_steps_per_window=cfg["sampling"]["production_steps"],
                      equilibration_steps_per_window=cfg["sampling"]["equilibration_steps"],
                      physical_convergence_claimed=False,grids={})
            for grid in GRIDS:
                if grid=="paper" and not (path/"paper-grid.npz").exists():
                    continue
                a=surface(str(path),grid)
                case["grids"][grid]=dict(bin_widths=[float(a[f"edges_{j}"][1]-a[f"edges_{j}"][0]) for j in (0,1)],
                                         bins=list(a["good"].shape),coverage=float(a["good"].mean()),
                                         supported_bins=int(a["good"].sum()),total_bins=int(a["good"].size),
                                         **score(a,a["good"]))
            case["original_report"]=d
            cases.append(case)
    return cases,failures,pending


def compare(before, after, experiment, grid):
    a=surface(before["path"],grid);b=surface(after["path"],grid)
    for key in ("edges_0","edges_1"):
        np.testing.assert_array_equal(a[key],b[key])
    np.testing.assert_allclose(a["reference_kjmol"],b["reference_kjmol"],atol=1e-12,rtol=0)
    common=a["good"] & b["good"]
    sa=score(a,common);sb=score(b,common)
    return dict(experiment=experiment,model=before["model"],grid=grid,seed=before["seed"],
                before=before["case"],after=after["case"],before_path=before["path"],after_path=after["path"],
                integration_steps_before=before["integration_steps"],integration_steps_after=after["integration_steps"],
                common_coverage=float(common.mean()),common_supported_bins=int(common.sum()),
                coverage_before=float(a["good"].mean()),coverage_after=float(b["good"].mean()),
                rmse_own_before=score(a,a["good"])["rmse_kjmol"],rmse_own_after=score(b,b["good"])["rmse_kjmol"],
                rmse_common_before=sa["rmse_kjmol"],rmse_common_after=sb["rmse_kjmol"],
                offset_common_before=sa["additive_offset_kjmol"],offset_common_after=sb["additive_offset_kjmol"],
                paper_point_min_common_before=sa["point_min_gauge_rmse_kjmol"],
                paper_point_min_common_after=sb["point_min_gauge_rmse_kjmol"])


def match_comparisons(cases, require_complete):
    by={(c["source"],c["model"],c["label"],c["seed"]):c for c in cases}
    pairs=[]
    specs=[("double_well_equal_budget","double_well","optimized","legacy-budget4000000","coverage-budget4000000"),
           ("ackley_fixed_thermostat_equal_budget","ackley","baseline","fixed-n1","thermostat500"),
           ("ackley_adaptive_different_budget","ackley","baseline","fixed-n1","coverage-tuned"),
           ("well_tempered_height_same_steps","double_well","baseline","well_tempered-n1","well_tempered-height04")]
    missing=[]
    for experiment,model,origin,old,new in specs:
        for seed in SEEDS:
            before=by.get((origin,model,old,seed));after=by.get(("optimized",model,new,seed))
            if before is None or after is None:
                missing.append(f"{experiment}: seed {seed}");continue
            for grid in GRIDS:
                if grid in before["grids"] and grid in after["grids"]:
                    pairs.append(compare(before,after,experiment,grid))
    # Retain every MetaD parameter attempt, including worse results and controls.
    for after in cases:
        if after["source"]=="optimized" and after["method"] in ("standard","well_tempered") and after["label"]!="well_tempered-height04":
            before=by[("baseline",after["model"],after["method"]+"-n1",after["seed"])]
            pairs.append(compare(before,after,"metadynamics_scan:"+after["label"],"coarse"))
    if require_complete and missing:
        raise RuntimeError("Incomplete required comparisons: "+"; ".join(missing))
    return pairs,missing


def aggregates(cases,pairs):
    own=[];paired=[]
    for key in sorted({(c["source"],c["model"],c["label"],g) for c in cases for g in c["grids"]}):
        origin,model,label,grid=key
        group=[c for c in cases if (c["source"],c["model"],c["label"])==key[:3] and grid in c["grids"]]
        row=dict(source=origin,model=model,label=label,grid=grid,seeds=len(group))
        for name in ("coverage","rmse_kjmol","point_min_gauge_rmse_kjmol"):
            row[name]=stats([c["grids"][grid][name] for c in group])
        row["integration_steps"]=stats([c["integration_steps"] for c in group])
        own.append(row)
    for experiment,grid in sorted({(p["experiment"],p["grid"]) for p in pairs}):
        group=[p for p in pairs if (p["experiment"],p["grid"])==(experiment,grid)]
        row=dict(experiment=experiment,grid=grid,seeds=len(group))
        for name in ("rmse_common_before","rmse_common_after","coverage_before","coverage_after","common_coverage",
                     "rmse_own_before","rmse_own_after","paper_point_min_common_before","paper_point_min_common_after",
                     "integration_steps_before","integration_steps_after"):
            row[name]=stats([p[name] for p in group])
        paired.append(row)
    return own,paired


def savefig(fig,out,name):
    fig.savefig(out/(name+".png"),dpi=190)
    fig.savefig(out/(name+".pdf"))
    plt.close(fig)


def overview(pairs,out):
    specs=[("double_well_equal_budget","Double well | bins 0.04","≈4.00M steps each\nCoverage refinement + thermostat"),
           ("ackley_fixed_thermostat_equal_budget","Ackley, fixed | bins 0.1","8.45M steps each\nThermostat setting only"),
           ("ackley_adaptive_different_budget","Ackley, adaptive | bins 0.1","8.45M → 11.368M steps\nSettings, windows and time differ"),
           ("well_tempered_height_same_steps","WT-MetaD | grid 0.04","0.21M steps each\nHill-height setting only")]
    fig,axes=plt.subplots(2,4,figsize=(16,7),layout="constrained")
    for col,(experiment,title,cost) in enumerate(specs):
        group=sorted([r for r in pairs if r["experiment"]==experiment and r["grid"]=="coarse"],key=lambda r:r["seed"])
        if len(group)!=3:
            plt.close(fig);return
        for row,(key,scale) in enumerate((("rmse_common_",1),("coverage_",100))):
            for j,p in enumerate(group):
                axes[row,col].plot([0,1],scale*np.array([p[key+"before"],p[key+"after"]]),"o-",color=COLORS[j],label=f"Seed {p['seed']}")
            axes[row,col].set_xticks([0,1],["Before","After"])
            axes[row,col].set_xlim(-.25,1.25)
            axes[row,col].grid(axis="y",alpha=.18)
        axes[0,col].set_title(title+"\n"+cost,fontsize=10)
        axes[0,col].set_ylabel("Shared-support RMSE / kJ mol$^{-1}$")
        axes[1,col].set_ylabel("Full-domain coverage / %")
        axes[1,col].set_ylim(0,104)
        before=np.mean([r["rmse_common_before"] for r in group]);after=np.mean([r["rmse_common_after"] for r in group])
        axes[0,col].text(.04,.02,f"Mean {before:.3f} → {after:.3f}",transform=axes[0,col].transAxes,fontsize=9)
        before=np.mean([r["coverage_before"] for r in group]);after=np.mean([r["coverage_after"] for r in group])
        axes[1,col].text(.04,.02,f"Mean {before:.1%} → {after:.1%}",transform=axes[1,col].transAxes,fontsize=9)
    axes[0,0].legend(fontsize=8)
    fig.suptitle("Optimization results | three paired seeds | P=1, 300 K\nErrors use each before/after pair's common support; missing-domain coverage is always shown",fontsize=14)
    savefig(fig,out,"optimization-overview")


def paired_figure(pairs,out,experiment,title,labels,name,grids=GRIDS):
    fig,axes=plt.subplots(len(grids),2,figsize=(11,4*len(grids)),layout="constrained",squeeze=False)
    for row,grid in enumerate(grids):
        group=sorted([p for p in pairs if p["experiment"]==experiment and p["grid"]==grid],key=lambda x:x["seed"])
        if not group:
            plt.close(fig);return
        for j,p in enumerate(group):
            axes[row,0].plot([0,1],[p["rmse_common_before"],p["rmse_common_after"]],"o-",color=COLORS[j],label=f"Seed {p['seed']}")
            axes[row,1].plot([0,1],100*np.array([p["coverage_before"],p["coverage_after"]]),"o-",color=COLORS[j])
        unit="0.04" if group[0]["model"]=="double_well" else "0.1"
        if grid=="paper":unit="0.02" if group[0]["model"]=="double_well" else "0.05"
        axes[row,0].set_title(f"Bin width {unit}: matched-support error")
        axes[row,0].set_ylabel("RMSE / kJ mol$^{-1}$")
        axes[row,1].set_title(f"Bin width {unit}: full-domain coverage")
        axes[row,1].set_ylabel("Supported bins / %")
        axes[row,1].set_ylim(0,103)
        for ax in axes[row]:
            ax.set_xticks([0,1],labels)
            ax.set_xlim(-.25,1.25)
            ax.grid(axis="y",alpha=.18)
        axes[row,0].text(.02,.02,f"Shared support: {100*min(p['common_coverage'] for p in group):.1f}–{100*max(p['common_coverage'] for p in group):.1f}%",transform=axes[row,0].transAxes,fontsize=9)
        axes[row,1].text(.02,.03,f"Mean coverage: {100*np.mean([p['coverage_before'] for p in group]):.2f}% → {100*np.mean([p['coverage_after'] for p in group]):.2f}%",transform=axes[row,1].transAxes,fontsize=9)
    axes[0,0].legend(fontsize=9)
    fig.suptitle(title+"\nP=1, 300 K; per-seed common masks and least-squares additive constants",fontsize=13)
    savefig(fig,out,name)


def residual_figure(pairs,out):
    selected=[p for p in pairs if p["experiment"]=="double_well_equal_budget" and p["seed"]==4101]
    if len(selected)!=2:return
    panels=[]
    for grid in GRIDS:
        p=next(x for x in selected if x["grid"]==grid)
        for side in ("before","after"):
            a=surface(p[side+"_path"],grid)
            residual=a["free_energy_kjmol"]-a["reference_kjmol"]-p["offset_common_"+side]
            panels.append((grid,side,a,residual,p))
    vmax=max(1.,np.ceil(max(float(np.max(abs(r[a["good"]]))) for _,_,a,r,_ in panels)*2)/2)
    fig,axes=plt.subplots(2,2,figsize=(10,8),layout="constrained",sharex=True,sharey=True)
    for ax,(grid,side,a,r,p) in zip(axes.ravel(),panels):
        im=ax.pcolormesh(a["edges_0"],a["edges_1"],np.ma.masked_where(~a["good"],r).T,
                         shading="flat",cmap="RdBu_r",vmin=-vmax,vmax=vmax)
        ax.set_facecolor("#cfcfcf");ax.set_xlabel("CV x");ax.set_ylabel("CV y")
        label="Legacy planner" if side=="before" else "Coverage refinement + thermostat tuning"
        width="0.04" if grid=="coarse" else "0.02"
        ax.set_title(f"{label} | bins {width}\nShared RMSE {p['rmse_common_'+side]:.3f}; coverage {100*p['coverage_'+side]:.1f}%",fontsize=10)
    fig.colorbar(im,ax=axes,label="Estimated F − exact bin-integrated F / kJ mol$^{-1}$",shrink=.85)
    fig.suptitle("Double-well residuals | seed 4101 | ≈4 million steps each\nOne color scale; gray = unsupported; offsets fitted on common support",fontsize=13)
    savefig(fig,out,"double_well_residuals")


def ackley_figure(cases,out):
    groups=[("baseline","fixed-n1","Fixed, original\n8.45M steps; dt 0.1 fs"),
            ("optimized","thermostat500","Fixed, thermostat tuned\n8.45M steps; dt 0.1 fs"),
            ("optimized","coverage-tuned","Adaptive, tuned settings\n11.368M steps; dt 0.5 fs")]
    by={(c["source"],c["label"],c["seed"]):c for c in cases if c["model"]=="ackley"}
    if any((o,l,s) not in by for o,l,_ in groups for s in SEEDS):return []
    records=[]
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout="constrained")
    for row,grid in enumerate(GRIDS):
        for j,seed in enumerate(SEEDS):
            casegroup=[by[o,l,seed] for o,l,_ in groups]
            arrays=[surface(c["path"],grid) for c in casegroup]
            for a in arrays[1:]:
                for key in ("edges_0","edges_1"):np.testing.assert_array_equal(arrays[0][key],a[key])
            common=np.logical_and.reduce([a["good"] for a in arrays])
            errors=[];coverage=[]
            for c,a in zip(casegroup,arrays):
                result=score(a,common);errors.append(result["rmse_kjmol"]);coverage.append(100*a["good"].mean())
                records.append(dict(grid=grid,seed=seed,case=c["case"],label=c["label"],common_coverage=float(common.mean()),integration_steps=c["integration_steps"],**result))
            axes[row,0].plot(range(3),errors,"o-",color=COLORS[j],label=f"Seed {seed}")
            axes[row,1].plot(range(3),coverage,"o-",color=COLORS[j])
        width="0.1" if grid=="coarse" else "0.05"
        axes[row,0].set(title=f"Bins {width}: same support across all three methods",ylabel="RMSE / kJ mol$^{-1}$")
        axes[row,1].set(title=f"Bins {width}: coverage over full domain",ylabel="Supported bins / %",ylim=(0,103))
        for ax in axes[row]:
            ax.set_xticks(range(3),[g[2] for g in groups],fontsize=8)
            ax.grid(axis="y",alpha=.18)
        support=[r["common_coverage"] for r in records if r["grid"]==grid]
        axes[row,0].text(.02,.02,f"Shared support: {100*min(support):.1f}–{100*max(support):.1f}%",transform=axes[row,0].transAxes,fontsize=9)
    axes[0,0].legend(fontsize=9)
    fig.suptitle("Ackley: distinguish thermostat tuning from a larger adaptive calculation\nOnly the two fixed-window cases have matched step counts and physical time",fontsize=13)
    savefig(fig,out,"ackley_comparison")
    return records


def meta_scan_figure(pairs,performance,out):
    rows=[p for p in pairs if p["seed"]==4101 and (p["experiment"].startswith("metadynamics_scan:") or p["experiment"]=="well_tempered_height_same_steps")]
    if not rows:return
    rows=sorted(rows,key=lambda p:p["after"])
    readable={"double_well-standard-height04-s4101":"Standard MetaD\nheight 0.4",
              "double_well-well_tempered-andersen500-s4101":"WT: collision\n50 fs",
              "double_well-well_tempered-andersen500_height04-s4101":"WT: 50 fs\n+ height 0.4",
              "double_well-well_tempered-dt04-s4101":"WT: dt 0.4 fs",
              "double_well-well_tempered-height04-s4101":"WT: height 0.4"}
    labels=[readable.get(p["after"],p["after"].replace("double_well-","").replace("-s4101","")) for p in rows]
    fig,axes=plt.subplots(1,3,figsize=(16,4.5),layout="constrained")
    x=np.arange(len(rows));w=.36
    axes[0].bar(x-w/2,[p["rmse_common_before"] for p in rows],w,color="#718390",label="Original on this pair's mask")
    axes[0].bar(x+w/2,[p["rmse_common_after"] for p in rows],w,color="#2e958c",label="Parameter trial")
    axes[0].set(title="Each trial vs original on their common support",ylabel="RMSE / kJ mol$^{-1}$")
    axes[1].bar(x-w/2,100*np.array([p["coverage_before"] for p in rows]),w,color="#718390")
    axes[1].bar(x+w/2,100*np.array([p["coverage_after"] for p in rows]),w,color="#2e958c")
    axes[1].set(title="Coverage loss remains visible",ylabel="Supported bins / %",ylim=(0,103))
    for ax in axes[:2]:ax.set_xticks(x,labels,rotation=20,ha="right",fontsize=8)
    axes[0].legend(fontsize=8)
    if performance:
        axes[2].bar(range(len(performance)),[p["speedup"] for p in performance],color="#416a86")
        axes[2].set_xticks(range(len(performance)),[str(p["hills"]) for p in performance])
        axes[2].set(title="History-array cache: identical bias/gradient",xlabel="Number of hills",ylabel="Bias-evaluation microbenchmark speedup")
    fig.suptitle("MetaD: parameter trials include negative results | seed 4101 | heights in kJ/mol\nCache speedup is a component benchmark; tuning and cache correctness are separate",fontsize=13)
    savefig(fig,out,"metadynamics_scan_and_cache")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline",type=Path,default=Path("/workspace/work/rpmd-fes-review/ogre"))
    p.add_argument("--optimized",type=Path,default=Path("/workspace/work/rpmd-fes-optimization/production"))
    p.add_argument("--output",type=Path,default=Path("/workspace/work/rpmd-fes-optimization/report"))
    p.add_argument("--settings",type=Path,default=Path("/workspace/work/rpmd-fes-optimization/settings"))
    p.add_argument("--meta",type=Path,default=Path("/workspace/work/metadynamics-opt"))
    p.add_argument("--require-complete",action="store_true")
    p.add_argument("--collect-only",action="store_true")
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    cases,failures,pending=collect(args.baseline,args.optimized)
    pairs,missing=match_comparisons(cases,args.require_complete)
    own,paired=aggregates(cases,pairs)
    performance=json.loads((args.meta/"performance.json").read_text()) if (args.meta/"performance.json").exists() else []
    result=dict(cases=cases,failures=failures,pending=pending,missing_required=missing,
                aggregate=own,paired_comparisons=pairs,paired_aggregate=paired,metadynamics_cache_performance=performance,
                limitations=["All completed optimization attempts are retained; absence of a FES is not a successful result.",
                    "Double-well comparison combines coverage refinement and thermostat tuning, not an isolated algorithm effect.",
                    "Three seeds show observed variation, not a calibrated confidence interval.",
                    "Same seeds and extended trajectories can share prefixes; variants are paired rather than independent replicates.",
                    "Common-support errors refit additive constants on the same mask. Coverage of the entire requested domain is reported separately.",
                    "Coarse grids and source-paper grid widths are reported separately; using paper widths does not reproduce the paper protocol.",
                    "All OGRe model FES comparisons here use P=1. No nonharmonic quantum-centroid or molecular rate convergence is claimed.",
                    "No claim of achieving published OGRe accuracy or reproducing its adaptive algorithm is made."])
    flat=[]
    for c in cases:
        for grid,d in c["grids"].items():
            flat.append({**{k:v for k,v in c.items() if k not in ("grids","original_report")},"grid":grid,
                         **{k:v for k,v in d.items() if k not in ("bins","bin_widths")},
                         "bin_width_x":d["bin_widths"][0],"bin_width_y":d["bin_widths"][1]})
    csv_rows(args.output/"all_cases.csv",flat)
    csv_rows(args.output/"paired_comparisons.csv",pairs)
    for filename in ("settings_summary.json","settings_summary.csv","settings_scan.png","settings_scan.pdf"):
        if (args.settings/filename).exists():shutil.copy2(args.settings/filename,args.output/filename)
    for filename in ("performance.json","common_support.csv","common_support.json","history_diagnosis.json"):
        if (args.meta/filename).exists():shutil.copy2(args.meta/filename,args.output/("metadynamics_"+filename))
    if not args.collect_only:
        plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
        overview(pairs,args.output)
        paired_figure(pairs,args.output,"double_well_equal_budget",
                      "Double well: coverage refinement + thermostat tuning | ≈4M steps per case, pilots included",
                      ["Legacy adaptive\nplanner","Coverage refinement\n+ tuned thermostat"],"double_well_equal_budget")
        residual_figure(pairs,args.output)
        result["ackley_three_way_common_support"]=ackley_figure(cases,args.output)
        csv_rows(args.output/"ackley_three_way_common_support.csv",result["ackley_three_way_common_support"])
        paired_figure(pairs,args.output,"well_tempered_height_same_steps",
                      "WT-MetaD: hill-height tuning at 210k steps | finite-time error and coverage",
                      ["Original height 0.1 kJ/mol","Hill height 0.4 kJ/mol"],"well_tempered_height_comparison",grids=("coarse",))
        meta_scan_figure(pairs,performance,args.output)
    write_json(args.output/"optimization_summary.json",result)
    print(json.dumps(dict(cases=len(cases),optimized_cases=sum(c["source"]=="optimized" for c in cases),
                          failures=len(failures),pending=len(pending),pairs=len(pairs),missing=missing,output=str(args.output))))


if __name__=="__main__":main()
