#!/usr/bin/env python3
"""Numerical/cost checks for experimental native water MTS and RPC paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from benchmarks.water.campaign import atomic_json,sha256
from benchmarks.water.native_optimizations import DelayedSlowKickMTS,NativeSplitRPMD,openmm_frequency_beta
from benchmarks.water.native_reference import Config,create_engine,get_state,sample_momenta,set_state
from benchmarks.water.qtip4pf import initial_grid
from benchmarks.water.split_reference import ContractedSlowPotential,GroupEvaluator,SplitMTS
from pyrpmd.constants import FS_TO_AU,KCAL_MOL_TO_AU
from pyrpmd.dynamics import Dynamics,State


def raw_ring_state(engine):
    q,v=[],[]
    for bead in range(engine.beads):
        state=engine.integrator.getState(bead,getPositions=True,getVelocities=True)
        q.append(np.asarray(state.getPositions(asNumpy=True)/engine.unit.nanometer).copy())
        v.append(np.asarray(state.getVelocities(asNumpy=True)/(engine.unit.nanometer/engine.unit.picosecond)).copy())
    return np.asarray(q),np.asarray(v)


def restore_raw_ring(engine,q,v):
    for bead in range(engine.beads):
        engine.integrator.setPositions(bead,q[bead])
        engine.integrator.setVelocities(bead,v[bead])


def checkpoint_probe():
    grid,box=initial_grid(cells_per_axis=2,density_g_cm3=.997,seed=32)
    rows=[]
    for platform in ("Reference","CPU"):
        for sampling in (False,True):
            options=dict(molecules=8,box_angstrom=box,beads=8,long_copies=1,dt_fs=.1,
                lj_cutoff_angstrom=2.5,electrostatic_cutoff_angstrom=2.,platform=platform,sampling=sampling)
            engine=NativeSplitRPMD(**options)
            rng=np.random.default_rng(31)
            q=np.repeat(grid[None],8,axis=0)+rng.normal(0,.02,(8,24,3))
            p=np.stack([sample_momenta(rng,engine.masses,298*8,remove_com=False)[0] for _ in range(8)])
            try:
                engine.set_state(q,p);engine.integrator.step(3)
                raw_q,raw_v=raw_ring_state(engine)
                physical_before=engine.get_state()
                blob=engine.context.createCheckpoint()
                engine.integrator.step(2)
                continued=engine.get_state()
                engine.context.loadCheckpoint(blob)
                context_only=engine.get_state()
                restore_raw_ring(engine,raw_q,raw_v)
                restored_q,restored_v=raw_ring_state(engine)
                engine.integrator.step(2)
                replay=engine.get_state()
                fresh=NativeSplitRPMD(**options)
                try:
                    fresh.context.loadCheckpoint(blob)
                    restore_raw_ring(fresh,raw_q,raw_v)
                    fresh_raw=raw_ring_state(fresh)
                    fresh.integrator.step(2)
                    fresh_result=fresh.get_state()
                finally:fresh.close()
                rows.append({"platform":platform,"sampling":sampling,"beads":8,"long_copies":1,
                    "context_checkpoint_bytes":len(blob),
                    "context_only_max_q_restore_error_bohr":float(np.max(abs(context_only[0]-physical_before[0]))),
                    "context_only_restores_ring":bool(np.array_equal(context_only[0],physical_before[0])),
                    "explicit_raw_q_restored_bitwise":bool(np.array_equal(raw_q,restored_q)),
                    "explicit_raw_v_restored_bitwise":bool(np.array_equal(raw_v,restored_v)),
                    "combined_checkpoint_continuation_max_q_error_bohr":float(np.max(abs(replay[0]-continued[0]))),
                    "combined_checkpoint_continuation_max_p_error_au":float(np.max(abs(replay[1]-continued[1]))),
                    "fresh_context_raw_q_restored_bitwise":bool(np.array_equal(raw_q,fresh_raw[0])),
                    "fresh_context_raw_v_restored_bitwise":bool(np.array_equal(raw_v,fresh_raw[1])),
                    "fresh_context_continuation_max_q_error_bohr":float(np.max(abs(fresh_result[0]-continued[0]))),
                    "fresh_context_continuation_max_p_error_au":float(np.max(abs(fresh_result[1]-continued[1]))),
                    "note":"Context blob plus explicit all-copy raw q/v; stochastic continuation tests actual RNG restoration, rather than inferring it from a successful API call."})
            finally:engine.close()
    return rows


def source_size_timing():
    rows=[]
    path=ROOT/"docs/validation/literature-audit-2026-11-01/water/classical-pilot/endpoint.npz"
    with np.load(path,allow_pickle=False) as data:q,p,box=data["q"],data["p"],float(data["box_angstrom"])
    for outer,inner in ((.25,None),(.5,.1),(1.,.1)):
        if inner is None:
            system,integrator,context,masses=create_engine(Config(),box)
            set_state(context,q,p,masses)
            get=lambda:get_state(context,masses,energy=True)
        else:
            engine=DelayedSlowKickMTS(molecules=216,box_angstrom=box,outer_dt_fs=outer,inner_dt_fs=inner,
                                      cutoff_angstrom=9.,platform="CPU",threads=1)
            engine.set_state(q,p);integrator=engine.integrator
            get=lambda:engine.get_state(energy=True)
        try:
            steps=round(100/outer)
            if inner is None:
                initial=get();before=initial[2]+float(np.sum(initial[1]**2/(2*masses[None,:,None])))
            else:before=get()[2]
            wall,cpu=time.perf_counter(),time.process_time();integrator.step(steps)
            wall,cpu=time.perf_counter()-wall,time.process_time()-cpu
            final=get()
            after=final[2] if inner is not None else final[2]+float(np.sum(final[1]**2/(2*masses[None,:,None])))
            rows.append({"method":"native Verlet" if inner is None else "delayed-slow-kick native MTS",
                "beads":1,"outer_dt_fs":outer,"inner_dt_fs":inner,"duration_fs":100,
                "steps":steps,"wall_seconds":wall,"process_cpu_seconds":cpu,"effective_CPU_equivalents":cpu/wall,
                "wall_seconds_per_step":wall/steps,"process_cpu_seconds_per_step":cpu/steps,
                "conditional_ps_per_wall_hour":.1/wall*3600,
                "endpoint_energy_change_kcal_mol_water":(after-before)/(216*KCAL_MOL_TO_AU)})
        finally:
            if inner is None:del context,integrator
            else:engine.close()
        print(json.dumps(rows[-1]),flush=True)
    path=ROOT/"docs/validation/literature-audit-2026-11-01/water/p32-warmup/endpoint.npz"
    with np.load(path,allow_pickle=False) as data:q,p,box=data["q"],data["p"],float(data["box_angstrom"])
    for copies,dt in ((32,.5),(1,.5),(1,.25),(1,.1),(5,.5),(5,.25)):
        engine=NativeSplitRPMD(molecules=216,box_angstrom=box,beads=32,long_copies=copies,
            dt_fs=dt,platform="CPU",threads=1,lj_backend="custom")
        grouped=GroupEvaluator(engine.system,platform="CPU",threads=1)
        hamiltonian=SplitMTS(ContractedSlowPotential(grouped,32,copies),engine.masses,298,32,dt,dt)
        hamiltonian.fast=Dynamics(hamiltonian.fast.potential,engine.masses,openmm_frequency_beta(298),
                                 32,dt*FS_TO_AU,native=True,threads=1,integrator="normal-mode-bab")
        try:
            engine.set_state(q,p)
            before=hamiltonian.energy(State(q.copy(),p.copy()))
            steps=round(10/dt)
            wall,cpu=time.perf_counter(),time.process_time();engine.integrator.step(steps)
            wall,cpu=time.perf_counter()-wall,time.process_time()-cpu
            qq,pp=engine.get_state()
            after=hamiltonian.energy(State(qq,pp))
            rows.append({"method":"native RPMD RPC single-time-step BAB","beads":32,"long_copies":copies,
                "dt_fs":dt,"duration_fs":10,"steps":steps,"wall_seconds":wall,"process_cpu_seconds":cpu,
                "effective_CPU_equivalents":cpu/wall,"wall_seconds_per_step":wall/steps,
                "process_cpu_seconds_per_step":cpu/steps,"conditional_ps_per_wall_hour":.01/wall*3600,
                "endpoint_energy_change_kcal_mol_water":(after-before)/(32*216*KCAL_MOL_TO_AU),
                "energy_scope":"actually propagated contracted Hamiltonian with the OpenMM spring-frequency convention, divided by P",
                "model":engine.model})
        finally:engine.close();grouped.close()
        print(json.dumps(rows[-1]),flush=True)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--source-size-timing",action="store_true")
    args=parser.parse_args()
    names=("benchmarks/water/validate_native_optimizations.py","benchmarks/water/native_optimizations.py",
           "benchmarks/water/split_reference.py","benchmarks/water/native_reference.py",
           "benchmarks/water/qtip4pf.py","src/pyrpmd/dynamics.py","src/pyrpmd/constants.py")
    identity={name:sha256(ROOT/name) for name in names}
    report={"schema":"pyrpmd.native-water-optimization-audit.v1","source_sha256":identity,
            "checkpoint_probes":checkpoint_probe(),"source_size_timing":[],
            "literature_reproduced":False,
            "limitations":["External experimental reference; registered native-Verlet campaign unchanged.",
                "Short single-configuration energy changes and timings do not establish timestep/RPC diffusion convergence.",
                "Shared CPU scheduler changes wall throughput; process CPU time is also reported."]}
    if args.source_size_timing:report["source_size_timing"]=source_size_timing()
    if identity!={name:sha256(ROOT/name) for name in names}:raise RuntimeError("source changed during audit")
    atomic_json(args.output,report)
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
