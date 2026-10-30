"""Verify production native loops on real BKMP2; emit timings AND numerical gates.

This is an implementation-equivalence study, not a converged rate calculation.
Both paths have identical masses, state, BCB map, timestep and requested samples.
Native final-force energies avoid a redundant PES call when recording energy.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmarks.h3.pes import BKMP2Potential
from benchmarks.h3.drpmd import job_mapping
from benchmarks.h3.rpmd import configuration
from benchmarks.loop_native.benchmark import initial,MASS_H,BETA,DT_AU
from pyrpmd.dynamics import Dynamics,State
from pyrpmd.cv import Distance,CVStack,RPMDrateCoordinate
from pyrpmd.bias import UmbrellaBias,FluxMetricBias,CompositeBias
from pyrpmd.workflows.rate import constrained_step,project_positions,project_momenta,rate_coordinate,_parent_task,_child_task,_recrossing_options,_prepare
from pyrpmd.workflows.fes import run_fes
from pyrpmd.workflows import drpmd
from pyrpmd.input import JobSpec


def compare(a,b,*,atol=2e-8,rtol=2e-8):
    aa,bb=np.asarray(a),np.asarray(b)
    np.testing.assert_allclose(aa,bb,atol=atol,rtol=rtol)
    return float(np.max(abs(aa-bb))) if aa.size else 0.0


def definitions(beads,kind):
    q,p=initial(beads)
    cv=RPMDrateCoordinate([[0],[1,2]],MASS_H,
            np.array([[-1.75,0,0],[0,0,0],[1.75,0,0]]),[[0,1]],[[1,2]],10.,equivalent_paths=[[0,2,1]])
    dyn=Dynamics(BKMP2Potential(),MASS_H,BETA,beads,DT_AU)
    bias=constraint=observation=None
    energy=kind!="nve"
    if kind=="umbrella2d":
        observation=CVStack([Distance([0,1]),Distance([1,2])])
        bias=UmbrellaBias(observation,[8.,1.401472],[[.01,.001],[.001,.02]])
    elif kind=="rate_flux":
        observation=cv;bias=CompositeBias(UmbrellaBias(cv,[.2],.2),FluxMetricBias(cv,MASS_H,BETA))
    elif kind in ("parent_constraint","recrossing"):
        observation=cv.dividing_surface(1.00925)
        q=np.repeat(cv.reference[None],beads,axis=0)
        q=project_positions(q,observation,MASS_H)
        if kind=="parent_constraint":
            constraint=observation;project_momenta(p,q,observation,MASS_H)
        else:
            energy=False
    return dyn,State(q,p),bias,constraint,observation,energy


def execute(beads,kind,backend,steps):
    dyn,s,bias,constraint,cv,energy=definitions(beads,kind)
    observed=kind!="nve"
    rows=[];energies=[]
    start=time.perf_counter()
    if backend=="native":
        remaining=steps
        while remaining:
            n=min(128,remaining)
            block=dyn.propagate_block(s,n,bias=bias,constraint=constraint,cv=cv,
                    observe=np.ones(n,dtype=np.uint8) if observed else np.zeros(n,dtype=np.uint8),
                    record_energy=energy,backend="native")
            rows.extend(block.values);energies.extend(block.energies);remaining-=n
    else:
        # Original public Dynamics.step + the same requested observations, not
        # the new fallback wrapper with its additional block-record machinery.
        for _ in range(steps):
            if constraint is not None:constrained_step(dyn,s,constraint)
            else:dyn.step(s,bias)
            if observed:
                rows.append(cv.value(s.q))
                energies.append(dyn.energy(s,bias) if energy else np.nan)
    elapsed=time.perf_counter()-start
    return elapsed,s,np.asarray(rows),np.asarray(energies),dyn.loop_stats


def core_study(steps,repeats):
    output=[];rng=np.random.default_rng(77491)
    for beads in (16,32):
        for kind in ("nve","umbrella2d","rate_flux","parent_constraint","recrossing"):
            correctness={}
            ref=execute(beads,kind,"python",24)
            got=execute(beads,kind,"native",24)
            correctness["max_q_abs_error_bohr"]=compare(got[1].q,ref[1].q,atol=5e-10)
            correctness["max_p_abs_error_au"]=compare(got[1].p,ref[1].p,atol=5e-9)
            correctness["max_cv_abs_error"]=compare(got[2],ref[2],atol=2e-10)
            finite=np.isfinite(ref[3])
            correctness["max_energy_abs_error_hartree"]=compare(got[3][finite],ref[3][finite],atol=2e-10)
            timings={"python":[],"native":[]}
            for repeat in range(repeats):
                for backend in rng.permutation(["python","native"]):
                    duration,final,values,energies,stats=execute(beads,kind,str(backend),steps)
                    if backend=="native":assert stats["native_steps"]==steps
                    timings[backend].append(duration)
            medians={k:statistics.median(v) for k,v in timings.items()}
            row={"beads":beads,"case":kind,"steps":steps,"repeats":repeats,
                 "seconds":timings,"steps_per_second":{k:steps/v for k,v in medians.items()},
                 "speedup":medians["python"]/medians["native"],"checks":correctness,
                 "same_observation_cadence":True,"native_steps_confirmed":steps,
                 "energy_observation_reuses_final_force":kind in ("umbrella2d","rate_flux","parent_constraint")}
            output.append(row);print(json.dumps(row),flush=True)
    return output


def workflow_study(root):
    root.mkdir(parents=True,exist_ok=True);result={}
    # Real 2D FES and rate-flux umbrella streams use the actual workflow executor.
    for stage in ("fes2d","rate"):
        configs={};outputs={};timing={}
        for backend in ("python","native"):
            out=root/(stage+"-"+backend);out.mkdir()
            config,system=configuration(stage,out,workers=1)
            config["dynamics"].update(loop_backend=backend,block_steps=31)
            config["sampling"].update(equilibration_steps=40,production_steps=160,replicas=1,
                                       sample_interval=4,andersen_interval=13)
            config["output"]["checkpoint_steps"]=37;config["analysis"].update(bootstrap=0,min_count=1)
            if stage=="fes2d":
                config["umbrella"]["grid"]=[{"start":1.65,"stop":1.85,"count":2}]*2
                config["analysis"]["bins"]=[8,8]
            else:
                config["umbrella"].update(start=0.,stop=1.1,count=12)
                config["analysis"]["bins"]=20
                config=_prepare(config,system,out)
            start=time.perf_counter();sampled=run_fes(config,system,out/"umbrella",return_data=True)
            timing[backend]=time.perf_counter()-start;outputs[backend]=sampled;configs[backend]=(config,system)
            if backend=="native":
                assert sum(s["propagation_this_call"]["native_steps"] for s in sampled["streams"])==len(sampled["streams"])*200
        errors=[]
        for a,b in zip(outputs["python"]["streams"],outputs["native"]["streams"]):
            errors.append({"samples":compare(a["samples"],b["samples"]),"energies":compare(a["energies"],b["energies"]),
                           "q":compare(a["last_q"],b["last_q"]),"p":compare(a["last_p"],b["last_p"])})
        result[stage]={"streams":len(errors),"saved_samples":outputs["native"]["summary"]["samples"],"seconds":timing,"errors":errors,"converged":False}
        # Exact completed restart, with a different native block length.
        config,system=configs["native"];config["dynamics"]["block_steps"]=7
        restored=run_fes(config,system,root/(stage+"-native")/"umbrella",return_data=True)
        for old,new in zip(outputs["native"]["streams"],restored["streams"]):
            np.testing.assert_array_equal(old["samples"],new["samples"])
            np.testing.assert_array_equal(old["energies"],new["energies"])
        result[stage]["exact_completed_restart"]=True
        if stage=="rate":
            # PMF estimator is unchanged; compare UI on the same saved streams.
            from pyrpmd.analysis.fes import umbrella_integration
            curves=[]
            for backend in ("python","native"):
                streams=outputs[backend]["streams"]
                curve=umbrella_integration([s["samples"][:,0] for s in streams],
                      np.array([s["center"][0] for s in streams]),np.array([s["stiffness"][0,0] for s in streams]),config["beta"],np.linspace(0,1.1,100))
                curves.append(curve)
            result[stage]["pmf_comparison"]=compare(curves[0][0],curves[1][0])
            result[stage]["pmf_derivative_comparison"]=compare(curves[0][1],curves[1][1])
    # Actual constrained parent and paired recrossing child tasks on BKMP2.
    parents={};children={};durations={}
    for backend in ("python","native"):
        out=root/("recrossing-"+backend);out.mkdir()
        config,system=configuration("rate",out,workers=1)
        config["dynamics"].update(loop_backend=backend,block_steps=19)
        config["recrossing"].update(equilibration_steps=17,parent_interval_steps=11,parent_samples=3,child_steps=160)
        config["output"]["checkpoint_steps"]=13
        config=_prepare(config,system,out)
        options=_recrossing_options(config)
        initial_q=np.repeat(system.geometry[None],config["beads"],axis=0)
        t=time.perf_counter()
        parents[backend]=_parent_task({"config":config,"system":system,"options":options,"xi_star":1.00925,"chain":0,"output":str(out),"initial_q":initial_q})
        if backend=="native":assert parents[backend]["propagation_this_call"]["native_steps"]>0
        children[backend]=[]
        for i,q in enumerate(parents[backend]["snapshots"]):
            child=_child_task({"config":config,"system":system,"chain":0,"parent":i,"pair":0,"q":q,"xi_star":1.00925,"child_steps":160,"output":str(out)})
            if backend=="native":assert child["propagation_this_call"]["native_steps"]==320
            children[backend].append(child)
        durations[backend]=time.perf_counter()-t
    parent_error=compare(parents["native"]["snapshots"],parents["python"]["snapshots"])
    for a,b in zip(children["native"],children["python"]):
        compare(a["numerator"],b["numerator"]);compare(a["denominator"],b["denominator"])
    result["recrossing"]={"parent_configurations":3,"child_trajectories":6,"child_steps":160,"max_parent_error":parent_error,"seconds":durations,"all_flux_series_match":True}
    # Actual dRPMD parent/launch/classifier/checkpoint path.
    mappings={};results={};duration={}
    for backend in ("python","native"):
        mapping=job_mapping(trajectories=8,workers=1,parents=2,beads=16,equilibration_fs=5.,sample_interval_fs=2.,max_time_fs=200.)
        mapping["dynamics"].update(loop_backend=backend,block_steps=31)
        mapping["output"].update(checkpoint_steps=37,trajectory_stride=7)
        mapping["output"].pop("checkpoint_interval",None)
        config,system=JobSpec.from_mapping(mapping).resolve();mappings[backend]=(config,system)
        start=time.perf_counter();results[backend]=drpmd.run_drpmd(config,system,root/("drpmd-"+backend));duration[backend]=time.perf_counter()-start
    assert results["python"]["counts"]==results["native"]["counts"]
    d_errors=[]
    for i,(a,b) in enumerate(zip(results["python"]["records"],results["native"]["records"])):
        assert (a["status"],a["channel"],a["steps"])==(b["status"],b["channel"],b["steps"])
        name=f"trajectory-{i:08d}.npz"
        with np.load(root/"drpmd-python"/"trajectories"/name) as x,np.load(root/"drpmd-native"/"trajectories"/name) as y:
            d_errors.append({k:compare(x[k],y[k],atol=2e-6) for k in ("final_q","final_p","steps","centroid","energy_per_bead")})
    assert sum(x["collision"]["native_steps"] for x in results["native"]["propagation_this_call"].values())>0
    config,system=mappings["native"];config=copy.deepcopy(config);config["resume"]=True
    restored=drpmd.run_drpmd(config,system,root/"drpmd-native")
    assert restored["records"]==results["native"]["records"]
    # Interrupted child restart: a fresh run is interrupted just after a durable
    # checkpoint. Resume with a different block length and compare all arrays.
    save=drpmd._atomic_npz;interrupted=[False]
    def save_then_interrupt(path,metadata,**arrays):
        save(path,metadata,**arrays)
        if metadata.get("format")=="PyRPMD.drpmd.trajectory-checkpoint.v1" and metadata.get("done",0)>0 and not interrupted[0]:
            interrupted[0]=True;raise KeyboardInterrupt("intentional test interruption")
    fresh=copy.deepcopy(config);fresh.pop("resume",None)
    try:
        with patch.object(drpmd,"_atomic_npz",side_effect=save_then_interrupt):
            drpmd.run_drpmd(fresh,system,root/"drpmd-interrupted")
    except KeyboardInterrupt:pass
    assert interrupted[0]
    fresh["resume"]=True;fresh["dynamics"]["block_steps"]=7
    drpmd.run_drpmd(fresh,system,root/"drpmd-interrupted")
    for i in range(8):
        name=f"trajectory-{i:08d}.npz"
        with np.load(root/"drpmd-native"/"trajectories"/name) as x,np.load(root/"drpmd-interrupted"/"trajectories"/name) as y:
            for k in ("final_q","final_p","steps","centroid","energy_per_bead"):
                np.testing.assert_array_equal(x[k],y[k])
    result["drpmd"]={"trajectories":8,"counts":results["native"]["counts"],"seconds":duration,"errors":d_errors,
                      "exact_completed_restart":True,"exact_interrupted_restart_with_different_block_size":True,
                      "native_steps":sum(x["collision"]["native_steps"] for x in results["native"]["propagation_this_call"].values())}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--steps",type=int,default=400);parser.add_argument("--repeats",type=int,default=5)
    parser.add_argument("--skip-workflows",action="store_true");args=parser.parse_args()
    if args.steps<1 or args.repeats<3:parser.error("positive steps and at least three repeats required")
    if hasattr(os,"sched_getaffinity"):
        available=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{available[0]})
    report={"schema":"pyrpmd.native-production-study.v1","passed":False,"physical_convergence":False,
            "environment":{"python":sys.version,"platform":platform.platform(),"affinity":sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None},
            "source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/"native/cayley.cpp",ROOT/"native/propagation.h",ROOT/"src/pyrpmd/propagation.py",Path(__file__)]}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        report["core"]=core_study(args.steps,args.repeats)
        if not args.skip_workflows:report["workflows"]=workflow_study(args.output.parent/"workflows")
        report["passed"]=True
    finally:
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"passed":report["passed"],"output":str(args.output)}),flush=True)

if __name__=="__main__":main()
