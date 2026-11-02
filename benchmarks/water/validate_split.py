#!/usr/bin/env python3
"""Source-size force errors and bounded MTS/RPC throughput, without D claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from benchmarks.water.campaign import atomic_json,sha256
from benchmarks.water.qtip4pf import make_system,O_MASS_DA,H_MASS_DA
from benchmarks.water.split_reference import ContractedSlowPotential,GroupEvaluator,SplitMTS,intramolecular,split_system
from pyrpmd.constants import AMU_TO_AU,KCAL_MOL_TO_AU
from pyrpmd.dynamics import State


def rms(values):
    return float(np.sqrt(np.mean(np.asarray(values)**2)))


def validate(snapshot,output,*,lj_backend="nonbonded"):
    import openmm
    identity={name:sha256(ROOT/name) for name in (
        "benchmarks/water/validate_split.py","benchmarks/water/split_reference.py",
        "benchmarks/water/qtip4pf.py","benchmarks/water/native_reference.py",
        "src/pyrpmd/dynamics.py","src/pyrpmd/constants.py","native/cayley.cpp","native/thermal.cpp")}
    with np.load(snapshot,allow_pickle=False) as saved:
        q,p,box=saved["q"],saved["p"],float(saved["box_angstrom"])
    beads,atoms,_=q.shape;n=atoms//3
    if (beads,n)!=(32,216):raise ValueError("this recorded audit expects the source-size P32 snapshot")
    system,metadata=split_system(molecules=n,box_angstrom=box,lj_cutoff_angstrom=9,
                                electrostatic_cutoff_angstrom=5,ewald_tolerance=1e-5,lj_backend=lj_backend)
    grouped=GroupEvaluator(system,platform="CPU",threads=1)
    baseline=GroupEvaluator(make_system(molecules=n,box_angstrom=box,cutoff_angstrom=9,ewald_tolerance=1e-5),platform="CPU",threads=1)
    try:
        legacy_e,legacy_f=baseline.evaluate(q)
        full_e,full_f=grouped.evaluate(q)
        fast_e,fast_f=intramolecular(q)
        short_e,short_f=grouped.evaluate(q,groups=(1<<1)|(1<<3))
        full_slow=full_f-fast_f
        report={"schema":"pyrpmd.water-split-mts-audit.v1","external_experimental_reference":True,
                "physical_converged":False,"literature_reproduced":False,"openmm":openmm.__version__,
                "platform":"CPU","threads":1,"molecules":n,"beads":beads,
                "snapshot_sha256":sha256(snapshot),"source_sha256":identity,"split_model":metadata,
                "full_electrostatic_split_comparison":{
                    "reference":"audited original 9 A direct cutoff / PME tolerance 1e-5",
                    "candidate":"5 A electrostatic direct / 9 A LJ / PME tolerance 1e-5; no bead contraction",
                    "energy_mean_difference_kcal_mol_water":float(np.mean(full_e-legacy_e)/(n*KCAL_MOL_TO_AU)),
                    "force_relative_RMS_difference":rms(full_f-legacy_f)/rms(legacy_f),
                    "force_max_difference_hartree_bohr":float(np.max(abs(full_f-legacy_f)))},
                "contraction_errors":[],"integration_and_timing":[],
                "limitations":["One short-warmup P32 configuration is a numerical probe, not an equilibrium ensemble.",
                               "RPC is an approximate Hamiltonian; energy conservation alone does not validate its diffusion.",
                               "The experimental screened-Coulomb split is not claimed to reproduce the original paper's exact RPC implementation.",
                               "No GPU or additional cloud compute is used; wall times are affected by shared CPU scheduling."]}
        for copies in (1,3,5,9,17,32):
            slow=ContractedSlowPotential(grouped,beads,copies)
            qc=np.einsum("kp,pac->kac",slow.transform,q)
            long_e,long_f=grouped.evaluate(qc,groups=1<<2)
            mapped=beads/copies*np.einsum("kp,kac->pac",slow.transform,long_f)
            approximated_f=short_f+mapped+fast_f
            approximated_e=short_e.sum()+beads/copies*long_e.sum()+fast_e.sum()
            difference=approximated_f-full_f
            report["contraction_errors"].append({"long_range_copies":copies,
                "energy_difference_kcal_mol_water":float((approximated_e-full_e.sum())/(beads*n*KCAL_MOL_TO_AU)),
                "total_force_relative_RMS_error":rms(difference)/rms(full_f),
                "nonbonded_force_relative_RMS_error":rms(difference)/rms(full_slow),
                "centroid_nonbonded_force_relative_RMS_error":rms(difference.mean(axis=0))/rms(full_slow.mean(axis=0)),
                "force_max_error_hartree_bohr":float(np.max(abs(difference)))})
        masses=np.tile([O_MASS_DA,H_MASS_DA,H_MASS_DA],n)*AMU_TO_AU
        for copies,outer,inner,duration_fs in ((32,.5,.1,5),(32,.25,.05,5),(1,.5,.1,10),(5,.5,.1,10)):
            slow=ContractedSlowPotential(grouped,beads,copies)
            integrator=SplitMTS(slow,masses,298,beads,outer,inner)
            state=State(q.copy(),p.copy())
            before=integrator.energy(state);delta=[]
            calls_before=grouped.calls
            wall,cpu=time.perf_counter(),time.process_time()
            steps=round(duration_fs/outer)
            for _ in range(steps):
                integrator.step(state)
                delta.append((integrator.energy(state)-before)/(beads*n*KCAL_MOL_TO_AU))
            elapsed=time.perf_counter()-wall;cpu_elapsed=time.process_time()-cpu
            row={"long_range_copies":copies,"outer_dt_fs":outer,"inner_dt_fs":inner,
                 "duration_fs":duration_fs,"outer_steps":steps,"wall_seconds":elapsed,"process_cpu_seconds":cpu_elapsed,
                 "effective_CPU_equivalents":cpu_elapsed/elapsed,"wall_seconds_per_outer_step":elapsed/steps,
                 "process_cpu_seconds_per_outer_step":cpu_elapsed/steps,
                 "max_abs_energy_change_kcal_mol_water":float(max(abs(np.asarray(delta)))),
                 "energy_scope":"the actually propagated contracted/uncontracted Hamiltonian divided by P",
                 "conditional_ps_per_wall_hour":steps*outer/1000/elapsed*3600,
                 "force_calls_in_timed_steps":grouped.calls-calls_before}
            # A deterministic NVE map should also return to q after reversing p.
            state.p*=-1
            for _ in range(steps):integrator.step(state)
            row["forward_backward_max_q_error_bohr"]=float(np.max(abs(state.q-q)))
            row["forward_backward_max_p_error_au"]=float(np.max(abs(state.p+p)))
            report["integration_and_timing"].append(row)
            print(json.dumps(row),flush=True)
        if identity!={name:sha256(ROOT/name) for name in identity}:
            raise RuntimeError("source changed during split numerical audit")
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        atomic_json(output/"split-validation.json",report)
        return report
    finally:
        grouped.close();baseline.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot",type=Path,default=ROOT/"docs/validation/literature-audit-2026-11-01/water/p32-warmup/endpoint.npz")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--lj-backend",choices=("nonbonded","custom"),default="nonbonded")
    args=parser.parse_args()
    validate(args.snapshot,args.output,lj_backend=args.lj_backend)


if __name__=="__main__":main()
