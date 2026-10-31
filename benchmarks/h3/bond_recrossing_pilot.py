"""Exploratory *actual* BKMP2 bond-diagonal RPMD signed recrossing pilot.

This computes a finite-time side correlation, not an asymptotic chemical
product committor or a normalized bimolecular rate. The third H-H bond and
canonical conditional surface weighting are not reconstructed here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from pyrpmd.cv import BondDifference
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.parallel import make_task_potential
from pyrpmd.propagation import loop_options
from pyrpmd.workflows.rate import _normal, project_positions, project_momenta, constrained_step
from rate_postprocess import configurations

def take_nearest_checkpoint(root, count):
    records=[]
    for path in sorted(Path(root).glob("checkpoints/window_*.npz")):
        with np.load(path,allow_pickle=False) as d:
            q=d["q"].copy()
        c=q.mean(axis=0)
        a=np.linalg.norm(c[0]-c[1]);b=np.linalg.norm(c[1]-c[2])
        if 1.4 <= a <= 3.1 and 1.4 <= b <= 3.1:
            records.append((abs(a-b),str(path),q))
    if len(records)<count:
        raise ValueError("insufficient archived starting states near bond surface")
    records.sort(key=lambda r:r[0])
    return records[:count]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--fes-run",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--parents-per-chain",type=int,default=4)
    p.add_argument("--child-fs",type=float,default=40.0)
    p.add_argument("--seed",type=int,default=27012026)
    a=p.parse_args()
    if a.parents_per_chain<1 or not 0<a.child_fs<=200:
        raise ValueError("invalid parent count / child time")
    a.output.mkdir(parents=True,exist_ok=True)
    _,cfg,system=configurations(a.output,profile="pilot",workers=1)
    if cfg["beads"] != 16 or cfg["temperature_K"] != 1500:
        raise ValueError("unexpected H3 physical parameters")
    surface=BondDifference(bonds=[[0,1],[1,2]])
    picks=take_nearest_checkpoint(a.fes_run,2)
    beta=cfg["beta"];dt=cfg["dt_au"]
    nsteps=max(2,int(round(a.child_fs / (dt*0.02418884326505))))
    nparent=2*a.parents_per_chain
    numerator=np.zeros((nparent,nsteps,1))
    denominator=np.zeros(nparent)
    parent_ids=np.zeros((nparent,2),int)
    diagnostics=[]
    provider=make_task_potential(cfg["potential"],system)
    try:
        for chain,(_,checkpoint,q0) in enumerate(picks):
            rngseed=np.random.SeedSequence([a.seed,chain,3])
            dyn=Dynamics(provider,system.masses,beta,16,dt,seed=rngseed,
                         native=True,threads=1,**loop_options(cfg))
            q=project_positions(q0,surface,system.masses)
            state=State(q,dyn.sample_momenta())
            project_momenta(state.p,state.q,surface,system.masses)
            interval=max(1,int(round(5.0/(dt*0.02418884326505))))
            equil=max(1,int(round(50.0/(dt*0.02418884326505))))
            total=equil+a.parents_per_chain*interval
            for step in range(total):
                if step%50==0:
                    state.p[:]=dyn.sample_momenta()
                    project_momenta(state.p,state.q,surface,system.masses)
                constrained_step(dyn,state,surface)
                if step+1>equil and (step+1-equil)%interval==0:
                    parent=(step+1-equil)//interval-1
                    idx=chain*a.parents_per_chain+parent
                    parent_ids[idx]=[chain,parent]
                    snapshot=state.q.copy()
                    # Fresh Gaussian child momenta, not projected onto the tangent.
                    rng=np.random.SeedSequence([a.seed,chain,parent,19])
                    childdyn=Dynamics(provider,system.masses,beta,16,dt,seed=rng,
                                      native=True,threads=1,**loop_options(cfg))
                    pchild=childdyn.sample_momenta()
                    residual,g,metric=_normal(surface,snapshot,system.masses)
                    assert abs(residual)<1e-8
                    velocity=float(np.sum(g*pchild.mean(axis=0)/system.masses[:,None]))
                    fs=np.sqrt(metric/(2*np.pi*beta))
                    denominator[idx]=abs(velocity)/fs
                    energy_error=0.
                    for sign in (1.,-1.):
                        child=State(snapshot,sign*pchild)
                        initial=childdyn.energy(child)
                        for k in range(nsteps):
                            childdyn.step(child)
                            value,_=surface.centroid_value_gradient(child.q.mean(axis=0))
                            numerator[idx,k,0]+=sign*velocity/fs*(value>0)
                        energy_error=max(energy_error,abs(childdyn.energy(child)-initial)*beta/16)
                    diagnostics.append({"chain":chain,"parent":parent,"energy_error_kbt":energy_error,
                                        "starting_checkpoint":checkpoint,"normal_speed_au":float(fs)})
    finally:
        if hasattr(provider,"close"):provider.close()
    t=np.arange(1,nsteps+1)*dt
    flux=numerator.sum(axis=(0,2))/denominator.sum()
    head=int(0.75*len(flux))
    estimate=float(flux[head:].mean())
    np.savez_compressed(a.output/"bond_recrossing_pilot.npz",time_au=t,parent_numerator=numerator,
                        parent_denominator=denominator,parent_ids=parent_ids,channel_names=np.array(["bond_diagonal_side"]),
                        estimator="signed_flux_side",surface_kind="r01_minus_r12",
                        interpretation="FINITE-TIME SIDE CORRELATION ONLY; not product commitment")
    report={"schema":"pyrpmd.h3.bond-2d-recrossing-pilot.v1","temperature_K":1500,
            "beads":16,"child_time_fs":a.child_fs,"parent_configurations":nparent,
            "child_trajectories":2*nparent,"kappa_side_candidate":estimate,
            "kappa_side_curve_plateau_start_fs":a.child_fs*.75,
            "kappa_physical_converged":False,"normalized_2d_rate_cm3_s":None,
            "status":"diagnostic_only__no_physical_entrance_bridge_or_complete_products",
            "maximum_endpoint_energy_error_kbt":max(d["energy_error_kbt"] for d in diagnostics),
            "parents":diagnostics}
    (a.output/"bond_recrossing_pilot.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="parents"},indent=2))

if __name__=="__main__":main()
