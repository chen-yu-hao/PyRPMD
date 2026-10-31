"""Run PyRPMD on the published OGRe 2D model potentials at 300 K.

This reproduces model equilibrium distributions, not the OGRe algorithm.
Classical P=1 is required for comparison with V(x,y). A nonlinear quantum
centroid free energy must NOT be compared to the bare potential.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from numpy.polynomial.legendre import leggauss

from pyrpmd.constants import KB_AU, AU_TIME_FS
from pyrpmd.system import System
from pyrpmd.potentials.cabi import CABIPotential
from pyrpmd.workflows.fes import run_fes, write_json
from pyrpmd.workflows.metadynamics import run_metadynamics

ROOT = Path(__file__).resolve().parents[2]
KJ = 2625.4996394799
BETA = 1 / (KB_AU * 300)


def potential(x, y, model):
    if model == "double_well":
        return ((x*x+y*y)**2 - 10*np.exp(-30*(x-.2)**2-3*(y-.4)**2)
                - 10*np.exp(-30*(x+.2)**2-3*(y+.4)**2))
    return -20*np.exp(-.2*np.sqrt((x*x+y*y)/2)) - np.exp((np.cos(2*np.pi*x)+np.cos(2*np.pi*y))/2)+np.e+20


def build(output):
    output = Path(output);output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("ogre_models.c")
    report = {}
    for model in ("double_well", "ackley"):
        target = output / (model + ".so")
        cmd = ["cc", "-shared", "-fPIC", "-O2", "-std=c11", "-I", str(ROOT / "native/include"),
               str(source), "-o", str(target), "-lm"]
        if model == "ackley":cmd.append("-DACKLEY")
        subprocess.run(cmd, check=True, capture_output=True)
        rng = np.random.default_rng(29381)
        q = rng.uniform(-1, 1, (50, 1, 3));q[:, 0, 2]=0
        pes = CABIPotential(target);result = pes.evaluate(q)
        expected = potential(q[:, 0, 0], q[:, 0, 1], model) / KJ
        fd = np.zeros_like(q)
        for j in range(3):
            plus=q.copy();minus=q.copy();plus[:,:,j]+=1e-6;minus[:,:,j]-=1e-6
            fd[:,:,j]=-(pes.evaluate(plus).energy-pes.evaluate(minus).energy)[:,None]/2e-6
        energy_error=float(np.max(abs(result.energy-expected)))
        force_error=float(np.max(abs(result.forces-fd)))
        assert energy_error < 1e-14 and force_error < 1e-10
        report[model]={"command":cmd,"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
                       "library_sha256":hashlib.sha256(target.read_bytes()).hexdigest(),
                       "energy_error_hartree":energy_error,"finite_difference_force_error_au":force_error}
        pes.close()
    write_json(output / "checks.json",report)


def config(model, method, seed, library, scale=1):
    double = model == "double_well"
    ranges = [[-.5,.5],[-1,1]] if double else [[-6,6],[-6,6]]
    c={"task":"fes", "beta":BETA, "beads":1, "dt_au":.1/AU_TIME_FS,
       "seed":seed,"native":"required", "potential":{"provider":"cabi","library":str(library)},
       "dynamics":{"loop_backend":"auto","block_steps":256},
       "sampling":{"equilibration_steps":10000,"production_steps":40000*scale,
                   "sample_interval":20,"andersen_interval":100,"replicas":1},
       "cvs":[{"name":a,"kind":"cartesian","atom":0,"axis":a,"estimator":"centroid"} for a in ["x","y"]],
       "analysis":{"bins":[25,50] if double else [120,120],"ranges":ranges,"bootstrap":0,"min_count":5},
       "output":{"checkpoint_steps":1000000}}
    if method in ("fixed","adaptive"):
        if double:
            axes=[np.linspace(-.5,.5,5 if method=="fixed" else 3),np.linspace(-1,1,9 if method=="fixed" else 5)]
            k=200/KJ
        else:
            axes=[np.linspace(-6,6,13 if method=="fixed" else 7)]*2
            k=5/KJ
        c["umbrella"]={"centers":[[x,y] for x in axes[0] for y in axes[1]],"stiffness":k}
        if method=="adaptive":
            c["adaptive_umbrella"]={"pilot_steps":5000,"pilot_equilibration_steps":2000,
                "max_rounds":5,"max_windows":192,"target_overlap":.015,"min_effective_samples":30.0}
    elif method=="unbiased":
        if not double:raise ValueError("Ackley unbiased benchmark not configured")
        c["umbrella"]={"centers":[[0,0]],"stiffness":1e-12}
        c["sampling"]["production_steps"]=200000*scale
    else:
        if not double:raise ValueError("Ackley metadynamics benchmark not configured")
        c["task"]="metadynamics";c["sampling"]["production_steps"]=200000*scale
        c["metadynamics"]={"mode":method,"sigma":[.07,.14],"height":.1/KJ,"pace":200}
        if method=="well_tempered":c["metadynamics"]["bias_factor"]=5.0
    return c


def oracle(edges, model, point=False, order=10):
    centers=[(a[:-1]+a[1:])/2 for a in edges]
    x,y=np.meshgrid(*centers,indexing="ij")
    point_f=potential(x,y,model)
    if point:return point_f,point_f
    nodes,weights=leggauss(order)
    qx=centers[0][:,None]+np.diff(edges[0])[:,None]*nodes/2
    qy=centers[1][:,None]+np.diff(edges[1])[:,None]*nodes/2
    u=potential(qx[:,None,:,None],qy[None,:,None,:],model)
    density=np.einsum("a,b,ijab->ij",weights,weights,np.exp(-BETA*u/KJ))/4
    return -np.log(density)*KJ/BETA,point_f


def run(model,method,seed,base,scale):
    base=Path(base).resolve();out=base/f"{model}-{method}-s{seed}-n{scale}";out.mkdir(parents=True,exist_ok=True)
    library=base/"build"/(model+".so")
    c=config(model,method,seed,library,scale)
    system=System(("H",),np.array([1837.152647]),np.zeros((1,3)))
    write_json(out/"input-resolved.json",{"config":c,"system":system.to_dict()})
    start=time.monotonic()
    try:
        result=(run_metadynamics if c["task"]=="metadynamics" else run_fes)(c,system,out/"run",return_data=True)
    except RuntimeError as e:
        write_json(out/"failure.json",{"error":str(e),"wall_seconds_this_call":time.monotonic()-start})
        raise
    wall=time.monotonic()-start
    data=result["data"];is_meta=c["task"]=="metadynamics"
    exact,point=oracle(data["edges"],model,point=is_meta)
    fine,_=oracle(data["edges"],model,point=is_meta,order=20)
    good=~data["mask"] & np.isfinite(data["free_energy"])
    f=data["free_energy"]*KJ
    residual=f-exact;residual-=residual[good].mean()
    paper_residual=(f-f[good].min())-(point-point[good].min())
    windows=result["summary"].get("windows",1)
    pilot_steps=0;rounds=0
    if (out/"run/adaptive-plan.json").exists():
        plan=json.loads((out/"run/adaptive-plan.json").read_text());rounds=len(plan["rounds"])
        pilot_steps=sum(len(r["centers"])*(r["pilot"]["equilibration_steps"]+r["pilot"]["production_steps"]) for r in plan["rounds"])
    report={"model":model,"method":method,"seed":seed,"scale":scale,"beads":1,"temperature_kelvin":300,
            "paper_doi":"10.1021/acs.jctc.3c01028", "wall_seconds_this_call":wall,
            "windows":windows,"samples":result["summary"]["samples"],"pilot_steps":pilot_steps,"adaptive_rounds":rounds,
            "integration_steps":windows*(c["sampling"]["equilibration_steps"]+c["sampling"]["production_steps"])+pilot_steps,
            "supported_bins":int(good.sum()),"total_bins":good.size,"coverage":float(good.mean()),
            "rmse_kjmol":float(np.sqrt(np.mean(residual[good]**2))),
            "max_abs_error_kjmol":float(np.max(abs(residual[good]))),
            "paper_point_min_gauge_rmse_kjmol":float(np.sqrt(np.mean(paper_residual[good]**2))),
            "quadrature_order_change_max_kjmol":float(np.max(abs(exact-fine))),
            "overlap_connected":result["summary"].get("overlap_connected"),
            "definition":"Bin-integrated density for umbrellas; point bias estimate for meta; least-squares additive gauge. Paper-style point/minimum gauge also reported.",
            "physical_convergence_claimed":False,"pilot_samples_used_in_production":0}
    if is_meta:report["hills"]=len(result["streams"][0]["hills"])
    if model=="double_well":
        both=good & good[::-1,::-1]
        report["inversion_symmetry_rmse_kjmol"]=float(np.sqrt(np.mean((f-f[::-1,::-1])[both]**2)))
    write_json(out/"comparison.json",report)
    np.savez_compressed(out/"comparison.npz",free_energy_kjmol=f,reference_kjmol=exact,
                        point_reference_kjmol=point,residual_kjmol=residual,paper_residual_kjmol=paper_residual,
                        mask=data["mask"],edges_0=data["edges"][0],edges_1=data["edges"][1])
    print(json.dumps(report),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["build","run"]);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--model",choices=["double_well","ackley"],default="double_well")
    p.add_argument("--method",choices=["fixed","adaptive","unbiased","standard","well_tempered"],default="fixed")
    p.add_argument("--seed",type=int,default=4101);p.add_argument("--scale",type=int,default=1)
    a=p.parse_args()
    if a.action=="build":build(a.output.resolve()/"build")
    else:run(a.model,a.method,a.seed,a.output,a.scale)


if __name__=="__main__":main()
