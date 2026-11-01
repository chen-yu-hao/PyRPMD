#!/usr/bin/env python3
"""Tully SAC p=10 independent published quantum reference versus PyRPMD.

Reference: Liu et al., J. Chem. Phys. 161, 024102 (2024), DOI 10.1063/5.0208708.
This does NOT use fitted trajectory data or assume a semiclassical method must agree.
Units: atomic. Nuclear Wigner Gaussian of sigma_x^2=0.5, sigma_p^2=0.5.
Diabatic initial state 0 at q=-9, p=10, mass=2000.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path
import numpy as np
import scipy
from pyrpmd.dynamics import State
from pyrpmd.nonadiabatic import NRPMDDynamics
from pyrpmd.spin_mapping import SMNRPMDDynamics
from pyrpmd.surface_hopping import RPSHDynamics
from pyrpmd.potentials.nonadiabatic_models import TullyModelOne

ROOT=Path(__file__).resolve().parent
REFERENCE=ROOT/"sac_p10_liu2024_reference.csv"
MASS=2000.0
REFERENCE_TIMES=np.array([0.,400.,800.,1200.,1600.,2000.,2400.,2800.])
def initialize_parent(rng):
    q=np.zeros((1,1,3));p=np.zeros_like(q)
    q[0,0,0]=-9.+rng.normal(scale=np.sqrt(.5))
    p[0,0,0]=10.+rng.normal(scale=np.sqrt(.5))
    return State(q,p)

def sample(method, i, dt, times, seed):
    # Independent streams; the same nuclear parents for each method at fixed i.
    entropy=np.random.SeedSequence([seed,i])
    pseed,eseed,dseed=entropy.spawn(3)
    parent=initialize_parent(np.random.default_rng(pseed))
    pes=TullyModelOne()
    if method=="nrpmd":
        driver=NRPMDDynamics(pes,[MASS],beta=100.,nbeads=1,dt=dt,nstates=2,native=False)
        state=driver.initialize(parent,0,np.random.default_rng(eseed)) if hasattr(driver,"initialize") else None
        if state is None:
            from pyrpmd.nonadiabatic import NRPMDState
            state=NRPMDState.from_nuclear(parent,2,0,np.random.default_rng(eseed))
        obs=lambda s: float(driver.populations(s)[1])
    elif method=="sm-nrpmd":
        driver=SMNRPMDDynamics(pes,[MASS],beta=100.,nbeads=1,dt=dt,native=False)
        state=driver.initialize(parent,initial_state=0,rng=np.random.default_rng(eseed))
        obs=lambda s: float(driver.populations(s)[1])
    elif method=="rpsh":
        driver=RPSHDynamics(pes,[MASS],beta=100.,nbeads=1,dt=dt,nstates=2,
                             native=False,seed=np.random.default_rng(dseed),rescaling="ring")
        state=driver.initialize(parent,active_surface=0)
        # The FSSH surface is an adiabatic index, not a diabatic population.
        # Project the ACTIVE adiabatic surface onto diabatic projector |1><1|.
        def obs(s):
            result=pes.evaluate(s.q.mean(axis=0,keepdims=True))
            _,U=np.linalg.eigh(result.matrix[0])
            return float(abs(U[1,s.active_surface])**2)
    else: raise ValueError(method)
    out=[]
    frame=0
    for step in range(int(round(times[-1]/dt))+1):
        t=step*dt
        if frame<len(times) and abs(t-times[frame])<1e-8:
            out.append(obs(state))
            frame+=1
        if step*dt<times[-1]-.5*dt:driver.step(state)
    if len(out)!=len(times):raise ValueError("observation clock mismatch")
    return out

def run(method,n,dt,seed,times):
    samples=np.array([sample(method,i,dt,times,seed) for i in range(n)])
    return {"mean":samples.mean(axis=0).tolist(),"sem":(samples.std(axis=0,ddof=1)/np.sqrt(n)).tolist(),
            "trajectories":samples.tolist(),"n":n,"dt":dt}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--n",type=int,default=48)
    p.add_argument("--dt",type=float,default=8.)
    p.add_argument("--seed",type=int,default=731)
    p.add_argument("--methods",nargs="+",default=["nrpmd","sm-nrpmd","rpsh"])
    p.add_argument("--output",type=Path,default=ROOT/"results"/"sac_p10_pilot.json")
    a=p.parse_args()
    if a.n<2 or a.dt<=0 or any(abs(t/a.dt-round(t/a.dt))>1e-8 for t in REFERENCE_TIMES):
        raise ValueError("n>=2; dt must divide every observation time")
    with REFERENCE.open() as stream:reader=list(csv.DictReader(row for row in stream if not row.startswith("#")))
    times=np.array([float(row["time"]) for row in reader])
    ref=np.array([float(row["sigma_1_1_real"]) for row in reader])
    ref_at=np.interp(REFERENCE_TIMES,times,ref)
    data={"schema":"PyRPMD.paper_validation.Tully_SAC_p10.v1",
          "reference":{"paper":"Liu et al. JCP 161 024102 (2024), DOI:10.1063/5.0208708",
                       "upstream":"xiangsunlab/benchmark-data pinned at 3951a0a19008fb545ad17f0d2961d934633f7422",
                       "file_sha256":hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
                       "times":REFERENCE_TIMES.tolist(),"population1":ref_at.tolist()},
          "physics":{"model":"Tully simple avoided crossing 1990","a":.01,"b":1.6,"c":.005,"d":1.,
                     "mass":MASS,"q0":-9.,"p0":10.,"var_q":.5,"var_p":.5,
                     "initial_diabatic_state":0,"beads":1,
                     "nuclear_initial":"Gaussian Wigner, sampled independently by trajectory"},
          "runtime":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__},
          "methods":{}}
    for method in a.methods:
        print(f"Running {method}: {a.n} trajectories, dt={a.dt} au",flush=True)
        result=run(method,a.n,a.dt,a.seed,REFERENCE_TIMES)
        result["difference"]=list(np.array(result["mean"])-ref_at)
        result["rmse"]=float(np.sqrt(np.mean(np.square(result["difference"]))))
        result["max_abs_error"]=float(np.max(np.abs(result["difference"])))
        result["within_2sem"]=(np.abs(result["difference"])<=2*np.array(result["sem"])).tolist()
        result["observable"]=("diabatic MMST W signed population" if method=="sm-nrpmd" else
            "diabatic MMST mapping signed population" if method=="nrpmd" else
            "active adiabatic-surface projected diabatic population (not amplitude population)")
        data["methods"][method]=result
        print(method,"RMSE=",result["rmse"],"P1(2400)=",result["mean"][-2],
              "reference=",ref_at[-2],flush=True)
    data["claims"]={"independent_quantum_reference_available":True,
                    "paper_level_agreement_certified":False,
                    "reason":"Finite-trajectory pilot, one bead and one timestep; convergence and physical-method bias unquantified."}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
    print("RESULT_JSON",a.output)
if __name__=="__main__":main()
