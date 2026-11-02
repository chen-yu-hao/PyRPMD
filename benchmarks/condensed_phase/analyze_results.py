"""Independent-replica comparisons for the saved condensed-phase campaign."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from validate import interval


def key(case):
    return (case["method"],case["beads"],case["dt"],case["external_pressure"])


def label(case):
    ensemble="NVT" if case["external_pressure"] is None else "NPT"
    return f"{case['method'].upper()} P={case['beads']} dt={case['dt']} {ensemble}"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    args=parser.parse_args()
    cases=json.loads((args.directory/"lj_summary.json").read_text())["results"]["cases"]
    by_key={key(c):c for c in cases}
    specs=[(("primitive",64,.04,None),("primitive",32,.04,None),"primitive bead refinement"),
           (("sc",16,.04,None),("sc",8,.04,None),"SC bead refinement 8 to 16"),
           (("sc",32,.04,None),("sc",16,.04,None),"SC bead refinement 16 to 32"),
           (("sc",32,.04,None),("primitive",64,.04,None),"independent discretizations at largest P"),
           (("sc",8,.02,None),("sc",8,.04,None),"SC NVT timestep refinement; equal steps"),
           (("sc",8,.02,1.),("sc",8,.04,1.),"SC NPT timestep refinement; equal steps"),
           (("piglet",4,.04,1.),("primitive",32,.04,1.),"PIGLET versus full-ring NPT reference"),
           (("rpc",16,.04,1.),("primitive",16,.04,1.),"RPC contraction at matched full bead count"),
           (("rpc8",16,.04,1.),("primitive",16,.04,1.),"RPC Pc=8 at matched full bead count"),
           (("rpc8",16,.04,1.),("rpc",16,.04,1.),"RPC contraction refinement Pc=4 to Pc=8"),
           (("sc",16,.04,1.),("primitive",32,.04,1.),"SC versus primitive NPT reference")]
    comparisons=[]
    for akey,bkey,purpose in specs:
        a,b=by_key[akey],by_key[bkey]
        ar={row["seed"]:row for row in a["replicas"]}
        br={row["seed"]:row for row in b["replicas"]}
        if set(ar)!=set(br):
            raise ValueError("paired comparisons require identical predefined seed sets")
        differences={field:interval([ar[seed]["means"][field]-br[seed]["means"][field] for seed in sorted(ar)])
                     for field in ("internal_energy_hartree","volume","pressure_au")}
        comparisons.append({"purpose":purpose,"difference":label(a)+" minus "+label(b),
                            "statistics":differences})
    drifts=[]
    for case in cases:
        replica_differences={field:[] for field in ("internal_energy_hartree","volume","pressure_au")}
        for row in case["replicas"]:
            data=np.genfromtxt(args.directory/row["raw"],delimiter=",",names=True)
            for field in replica_differences:
                halves=np.array_split(data[field],2)
                replica_differences[field].append(float(halves[1].mean()-halves[0].mean()))
        drifts.append({"case":label(case),"second_half_minus_first_half":{field:interval(values)
                       for field,values in replica_differences.items()}})
    output={"schema":"PyRPMD.condensed-phase-comparisons.v1",
            "statistics":"95% Student intervals across four independent paired seed differences; not systematic-error bounds",
            "comparisons":comparisons,"drift_diagnostics":drifts}
    (args.directory/"comparisons.json").write_text(json.dumps(output,indent=2)+"\n")
    lines=["# Executed model-solid validation", "", "All energies below are total-cell values in reduced units.",
           "Intervals are 95% Student intervals across four independent trajectories; they do not include bead, timestep, cell-size or model error.", "",
           "## Fixed-volume FCC solid", "", "| Method | P | dt | E, 95% interval | Pressure, 95% interval |", "| --- | ---: | ---: | ---: | ---: |"]
    def fmt(c,field):
        r=c["statistics"][field]; lo,hi=r["confidence_interval_95"]
        return f"{r['mean']:.6f} [{lo:.6f}, {hi:.6f}]"
    for c in cases:
        if c["external_pressure"] is None:
            lines.append(f"| {c['method'].upper()} | {c['beads']} | {c['dt']} | {fmt(c,'internal_energy_hartree')} | {fmt(c,'pressure_au')} |")
    lines.extend(["","## Isotropic NPT FCC solid (external p=1)","",
                  "The pressure column uses the canonical expected centroid kinetic term. Under PIGLET it must be read alongside the separate actual-piston-pressure diagnostic.","",
                  "| Method | P | dt | Volume, 95% interval | E, 95% interval | Configurational-CV pressure, 95% interval |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for c in cases:
        if c["external_pressure"] is not None:
            method=("RPC Pc=8" if c['method']=="rpc8" else c['method'].upper()+(" Pc=4" if c['method']=='rpc' else ""))
            lines.append(f"| {method} | {c['beads']} | {c['dt']} | {fmt(c,'volume')} | {fmt(c,'internal_energy_hartree')} | {fmt(c,'pressure_au')} |")
    lines.extend(["","## Pairwise comparison details","", "Differences and drift diagnostics are retained in [comparisons.json](comparisons.json).",
                  "The original seed set was retained in all comparisons. The additional SC16/32 and matched-P RPC cases resolve observed bead/contraction uncertainty; no trajectories were selected by outcome.","",
                  "![Convergence and pressure sampling](validation.png)","",
                  "See [the scientific guide](../../../docs/condensed-phase-nqe.md) for models, equations, parameter provenance and scope."])
    (args.directory/"RESULTS.md").write_text("\n".join(lines)+"\n")


if __name__=="__main__":
    main()
