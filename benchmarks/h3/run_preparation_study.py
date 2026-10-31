"""Fixed follow-up workload: initial-state controls, not curve fitting."""
from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.h3.state_preparation_study import LaunchConfig, PreparationSpec, run


def jobs(group):
    points=((0,.55),(1,.90),(1,1.10))
    if group in ("fast","all"):
        for v,energy in points:
            yield "collisions",f"ne-direct-none-v{v}-Et{energy:.3f}",LaunchConfig(
                v=v,total_energy_ev=energy,samples=1024,id_offset=10000,bmax=3.5 if v==0 else 4.),PreparationSpec(conditioning="none"),False
        for convention in ("fixed_collision","fixed_total"):
            for v,energy in points:
                yield "collisions",f"qct-action-{convention}-v{v}-Et{energy:.3f}",LaunchConfig(
                    method="qct",beads=1,v=v,total_energy_ev=energy,samples=4096,
                    id_offset=20000,bmax=3.5 if v==0 else 4.),PreparationSpec(
                    kind="action_qct",conditioning="none",energy_convention=convention),False
    if group in ("switch","all"):
        for v,energy in points:
            yield "collisions",f"ne-switch200-none-v{v}-Et{energy:.3f}",LaunchConfig(
                v=v,total_energy_ev=energy,samples=1024,id_offset=10000,bmax=3.5 if v==0 else 4.),PreparationSpec(
                kind="switch",conditioning="none",switch_time_fs=200.),False


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group",choices=["fast","switch","audit","all"],default="all")
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--output",type=Path,default=Path("runs/state_preparation_study"))
    args=parser.parse_args()
    for scope,label,config,spec,audit in jobs(args.group):
        run(config,spec,args.output/scope/label,workers=args.workers,audit=audit)
    if args.group in ("audit", "all"):
        # Reduce native every-step hold integrals before thinning the saved
        # trace. The same compact wrapper was used for the published audit.
        from benchmarks.h3.preparation_audit import run_cases
        run_cases(args.output/"isolated_audit", workers=args.workers)


if __name__=="__main__":
    main()
