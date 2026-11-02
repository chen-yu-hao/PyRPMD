"""Diagnose PIGLET pressure feedback versus centroid-virial observables.

The principal solid campaign found a pressure discrepancy. This separate
diagnostic retains both estimators and the actual centroid kinetic ratio;
it never replaces the original results. Two timesteps use equal physical
equilibration/production durations and the original four predefined seeds.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path

import numpy as np

from models import LennardJonesSolid, build_lj, fcc, piglet_parameters
from validate import primitive_observables, save_csv, interval, numerical_source_identity
from pyrpmd.barostat import BZPBarostat
from pyrpmd.dynamics import Dynamics, State


def run(options):
    output, dt, seed = options
    source_identity=numerical_source_identity()
    source_identity["piglet_pressure_diagnostic.py"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    q, cell = fcc()
    masses=np.full(len(q),100.)
    beta=5.; beads=4; pressure=1.
    physical=LennardJonesSolid(build_lj(output/"build"),len(q),cell)
    piston=BZPBarostat(physical,{"cell":cell,"pbc":[True]*3,"masses":masses},beta,beads,pressure,5.)
    dynamics=Dynamics(physical,masses,beta,beads,dt,seed=seed,native=True,
        integrator="normal-mode-bab",loop_backend="python",thermostat={"mode":"piglet","gle":piglet_parameters(beta)},
        barostat=piston,cell=cell,pbc=[True]*3)
    state=State(np.repeat(q[None],beads,axis=0),dynamics.sample_momenta())
    burn=round(120./dt); production=round(480./dt); stride=round(.8/dt)
    rows=[]
    for step in range(burn+production):
        dynamics.step(state)
        if step>=burn and (step-burn+1)%stride==0:
            volume=piston.volume
            observed=primitive_observables(physical,state.q,beta,volume)
            piston_pressure=piston.pressure(state)
            centroid_p=state.p.mean(axis=0)
            twice_kinetic=float(np.sum(centroid_p**2/masses[:,None]))
            ratio=beta*twice_kinetic/(3*len(q))
            delta=(twice_kinetic-3*len(q)/beta)/(3*volume)
            rows.append({"step":float(step+1),"time":state.time,"volume":volume,
                "configurational_cv_pressure":observed["pressure_au"],
                "piston_feedback_pressure":piston_pressure,
                "centroid_kinetic_ratio":ratio,
                "pressure_difference_from_centroid_kinetic":delta,
                "pressure_identity_residual":piston_pressure-observed["pressure_au"]-delta,
                "internal_energy_hartree":observed["internal_energy_hartree"]})
    current=numerical_source_identity()
    current["piglet_pressure_diagnostic.py"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if current!=source_identity:
        raise RuntimeError("numerical source changed during diagnostic")
    path=output/f"piglet_pressure_dt{dt}_seed{seed}.csv"
    digest=save_csv(path,rows)
    keys=list(rows[0])[2:]
    result={"dt":dt,"seed":seed,"burn_steps":burn,"production_steps":production,
            "burn_time":120.,"production_time":480.,"sample_time_interval":.8,
            "raw":path.name,"sha256":digest,"source_sha256_before_execution":source_identity,
            "means":{key:float(np.mean([r[key] for r in rows])) for key in keys},
            "half_means":{key:[float(np.mean(a)) for a in np.array_split([r[key] for r in rows],2)]
                          for key in ("volume","configurational_cv_pressure","piston_feedback_pressure","centroid_kinetic_ratio")}}
    print(json.dumps({"event":"piglet_pressure_diagnostic","dt":dt,"seed":seed,**result["means"]}),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--jobs",type=int,default=2)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    build_lj(args.output/"build")
    options=[(args.output,dt,seed) for dt in (.04,.02) for seed in range(82719,82723)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
        rows=list(pool.map(run,options))
    groups=[]
    for dt in (.04,.02):
        selected=[row for row in rows if row["dt"]==dt]
        groups.append({"dt":dt,"replicas":selected,
            "statistics":{key:interval([row["means"][key] for row in selected]) for key in selected[0]["means"]}})
    result={"schema":"PyRPMD.piglet-pressure-diagnostic.v1",
            "reason":"Observed low-noise pressure differs from external pressure in main anharmonic solid campaign",
            "target_pressure":1.,"canonical_centroid_kinetic_ratio":1.,
            "equal_physical_duration":True,"groups":groups,
            "interpretation":"Feedback uses actual centroid kinetic energy; replacing it by its canonical expectation is a separate diagnostic in a noncanonical PIGLET state."}
    (args.output/"piglet_pressure_diagnostic.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
