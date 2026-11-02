"""Reproduce the rejected, too-fast P32 FCC-solid piston protocol.

The accepted reference changes tau=5 to tau=10 with the original four seeds.
No core force or barostat equation is changed, and no failing production
trajectory is included in an ensemble average.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from models import LennardJonesSolid, build_lj, fcc
from validate import numerical_source_identity
from pyrpmd.barostat import BZPBarostat
from pyrpmd.dynamics import Dynamics, State


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    q,cell=fcc(); masses=np.full(len(q),100.)
    potential=LennardJonesSolid(build_lj(args.output/"build"),len(q),cell)
    piston=BZPBarostat(potential,{"masses":masses,"cell":cell,"pbc":[True]*3},5.,32,1.,5.)
    dynamics=Dynamics(potential,masses,5.,32,.04,seed=82719,native=True,
        integrator="normal-mode-bab",loop_backend="python",thermostat="pile-l",
        barostat=piston,cell=cell,pbc=[True]*3)
    state=State(np.repeat(q[None],32,axis=0),dynamics.sample_momenta())
    data={"schema":"PyRPMD.condensed-rejected-protocol.v1",
          "model":"FCC32 switched LJ, beta5, mass100, initialdensity1",
          "beads":32,"dt":.04,"barostat_tau":5.,"external_pressure":1.,"seed":82719,
          "original_attempt":"First replica failed during early equilibration; later replicas were not attempted at tau=5.",
          "accepted_protocol_change":"barostat_tau=10; original four seeds and all other parameters retained",
          "source_sha256_at_replay":numerical_source_identity()}
    try:
        for _ in range(400):
            dynamics.step(state)
    except Exception as exc:
        data.update(status="rejected_before_production",error_type=type(exc).__name__,error=str(exc),
                    completed_steps=state.step,completed_time=state.time,
                    restored_cell_after_rejection=piston.cell.tolist())
    else:
        raise RuntimeError("the documented rejected protocol did not reproduce within 400 steps")
    (args.output/"rejected_protocol.json").write_text(json.dumps(data,indent=2)+"\n")
    print(json.dumps({k:v for k,v in data.items() if not k.startswith("source")},indent=2))


if __name__=="__main__":
    main()
