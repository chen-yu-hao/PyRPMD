"""Independent-replica comparisons for the saved condensed-phase campaign."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from validate import interval


def key(case):
    return (case["method"],case["beads"],case["dt"],case["external_pressure"])


def label(case):
    ensemble="NVT" if case["external_pressure"] is None else "NPT"
    method={"rpc":"RPC Pc=4","rpc8":"RPC Pc=8"}.get(case["method"],case["method"].upper())
    tau="" if ensemble=="NVT" else f" tau={case.get('barostat_tau',5.)}"
    return f"{method} P={case['beads']} dt={case['dt']} {ensemble}{tau}"


def pressure_diagnostic(directory):
    """Postprocess retained pressure identities; no new dynamics or fitting."""
    source=directory/"piglet_pressure_diagnostic.json"
    if not source.exists():
        return None
    result=json.loads(source.read_text())
    for group in result["groups"]:
        balance=[]
        canonical_balance=[]
        max_residual=0.
        for row in group["replicas"]:
            raw=directory/row["raw"]
            if hashlib.sha256(raw.read_bytes()).hexdigest()!=row["sha256"]:
                raise ValueError(f"diagnostic raw checksum mismatch: {raw}")
            data=np.genfromtxt(raw,delimiter=",",names=True)
            balance.append(float(np.mean(data["volume"]*(data["piston_feedback_pressure"]-1.)+1/5.)))
            canonical_balance.append(float(np.mean(data["volume"]*(data["configurational_cv_pressure"]-1.)+1/5.)))
            max_residual=max(max_residual,float(np.max(np.abs(data["pressure_identity_residual"]))))
        group["statistics"]["volume_weighted_piston_balance"]=interval(balance)
        group["statistics"]["volume_weighted_canonical_substitution_balance"]=interval(canonical_balance)
        group["maximum_absolute_pressure_identity_residual"]=max_residual
    result["balance_definition"]="mean(V*(p_feedback-p_ext)+1/beta), beta=5 and p_ext=1; stationary BZP target zero"
    result["assessment"]=("The P=4 harmonic calibration fails this anharmonic solid pressure/transferability check: "
        "canonical-substituted pressure and centroid kinetic ratio exclude their targets at both timesteps. "
        "This establishes an estimator-substitution discrepancy; it does not isolate the microscopic cause "
        "among noncanonical anharmonic energy transfer, finite beads and remaining sampling effects.")
    return result


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
    diagnostic=pressure_diagnostic(args.directory)
    output={"schema":"PyRPMD.condensed-phase-comparisons.v1",
            "statistics":"95% Student intervals across four independent paired seed differences; not systematic-error bounds",
            "comparisons":comparisons,"drift_diagnostics":drifts,
            "piglet_pressure_diagnostic":diagnostic}
    (args.directory/"comparisons.json").write_text(json.dumps(output,indent=2)+"\n")
    lines=["# Executed model-solid validation", "", "All energies below are total-cell values in reduced units.",
           "Complete raw CSVs are supplied as an accompanying review archive and are reproducible with the driver. Public summaries retain the raw filenames and SHA256 checksums; the CSVs are not part of the public source distribution.",
           "Intervals are 95% Student intervals across four independent trajectories; they do not include bead, timestep, cell-size or model error.", "",
           "## Fixed-volume FCC solid", "", "| Method | P | dt | E, 95% interval | Pressure, 95% interval |", "| --- | ---: | ---: | ---: | ---: |"]
    def fmt(c,field):
        r=c["statistics"][field]; lo,hi=r["confidence_interval_95"]
        return f"{r['mean']:.6f} [{lo:.6f}, {hi:.6f}]"
    comparison_by_purpose={row["purpose"]:row for row in comparisons}
    drift_by_case={row["case"]:{"statistics":row["second_half_minus_first_half"]} for row in drifts}
    for c in cases:
        if c["external_pressure"] is None:
            lines.append(f"| {c['method'].upper()} | {c['beads']} | {c['dt']} | {fmt(c,'internal_energy_hartree')} | {fmt(c,'pressure_au')} |")
    lines.extend(["", "The SC P=16 to P=32 total-energy shift is "+fmt(comparison_by_purpose["SC bead refinement 16 to 32"],"internal_energy_hartree")+
                  "; the primitive P=32 to P=64 shift is "+fmt(comparison_by_purpose["primitive bead refinement"],"internal_energy_hartree")+". Both remain material relative to sampling error.", "",
                  "The SC P=32 minus primitive P=64 energy difference is "+fmt(comparison_by_purpose["independent discretizations at largest P"],"internal_energy_hartree")+
                  ". Its interval includes zero, but overlap between finite-P calculations does not establish the infinite-P limit. These results demonstrate refinement and quantify its remaining uncertainty."])
    lines.extend(["","## Isotropic NPT FCC solid (external p=1)","",
                  "The pressure column uses the canonical expected centroid kinetic term. Under PIGLET it must be read alongside the separate actual-piston-pressure diagnostic.","",
                  "RPC energy and pressure are observables of the contracted Hamiltonian. No full-P physical-energy or virial correction is applied to the contracted samples.", "",
                  "| Method | P | dt | Piston tau | Volume, 95% interval | E, 95% interval | Configurational-CV pressure, 95% interval |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for c in cases:
        if c["external_pressure"] is not None:
            method=("RPC Pc=8" if c['method']=="rpc8" else c['method'].upper()+(" Pc=4" if c['method']=='rpc' else ""))
            lines.append(f"| {method} | {c['beads']} | {c['dt']} | {c.get('barostat_tau',5.)} | {fmt(c,'volume')} | {fmt(c,'internal_energy_hartree')} | {fmt(c,'pressure_au')} |")
    lines.extend(["", "At matched P=16, the RPC Pc=4 minus full-ring energy difference is "+fmt(comparison_by_purpose["RPC contraction at matched full bead count"],"internal_energy_hartree")+
                  "; Pc=8 reduces it to "+fmt(comparison_by_purpose["RPC Pc=8 at matched full bead count"],"internal_energy_hartree")+
                  ". The virial implements the contracted action correctly, while this repulsive/attractive splitting remains strongly unconverged at both contraction levels.", "",
                  "SC P=16 minus primitive P=32 NPT energy differs by "+fmt(comparison_by_purpose["SC versus primitive NPT reference"],"internal_energy_hartree")+
                  ". This compares two finite-P approximations with distinct piston times; it does not establish a converged equation of state.", "",
                  "The SC timestep comparisons retain equal step counts, so dt=0.02 has half the physical burn-in and production duration of dt=0.04. The targeted PIGLET diagnostic below separately holds physical duration fixed.", "",
                  "## Sampled SC heat capacities", "",
                  "These canonical fluctuation estimates use independent trajectory estimates for their intervals. Refined-protocol Cv/Cp remains statistically unresolved at these lengths; a negative lower confidence bound is uncertainty, not a negative physical heat capacity. Correlation and finite-length bias require longer production sampling. Deterministic harmonic Gaussian moments provide the separate Cv estimator-correctness check; the NPT volume quadrature supplies the finite-P Cp reference.", "",
                  "| Ensemble | P | dt | Quantity | Estimate, 95% interval |",
                  "| --- | ---: | ---: | --- | ---: |"])
    for c in cases:
        if c["method"]=="sc":
            ensemble="NVT" if c["external_pressure"] is None else "NPT"
            lines.append(f"| {ensemble} | {c['beads']} | {c['dt']} | {c['heat_capacity_kind']}/kB | {fmt(c,'heat_capacity_over_kB')} |")
    if diagnostic is not None:
        lines.extend(["","## PIGLET pressure transferability check: failed for this P=4 fit", "",
                      "The added diagnostic retains the original seeds and uses equal physical equilibration time 120 and production time 480 at both timesteps. It does not replace the main campaign.", "",
                      "The canonical centroid bath covariance is a required input condition, not evidence that the coupled anharmonic system has canonical centroid momenta. Here R=beta*sum(|mean_bead(p)|^2/m)/(3N) exceeds one.", "",
                      "| dt | Configurational-CV pressure | Actual piston pressure | Centroid kinetic ratio R | Volume-weighted piston balance |",
                      "| ---: | ---: | ---: | ---: | ---: |"])
        for group in diagnostic["groups"]:
            lines.append(f"| {group['dt']} | {fmt(group,'configurational_cv_pressure')} | {fmt(group,'piston_feedback_pressure')} | {fmt(group,'centroid_kinetic_ratio')} | {fmt(group,'volume_weighted_piston_balance')} |")
        lines.extend(["", "Every interval in this table is a 95% Student interval across four independent trajectories. The exact per-frame pressure difference is [sum(|pbar|^2/m)-3N/beta]/(3V). The feedback stationarity balance is V*(p_feedback-p_ext)+1/beta, whose target is zero; it does not enforce an unbiased physical equation of state.", "",
                      "Both timesteps fail the canonical-substituted pressure and centroid kinetic target. The algebraic discrepancy is resolved, while its physical origin and a transferable parameterization remain unresolved. Harmonic calibration accuracy therefore does not establish anharmonic PIGLET-NPT accuracy.", "",
                      "Full diagnostics, source identities and raw-file hashes: [piglet_pressure_diagnostic.json](piglet_pressure_diagnostic.json); derived volume-weighted balances: [comparisons.json](comparisons.json)."])
    if (args.directory/"rejected_protocol.json").exists():
        lines.extend(["", "## Retained failures and execution history", "",
                      "The primitive P=32 NPT reference with piston tau=5 failed during equilibration: its step-38 cell proposal crossed the LJ model's cell-side > cutoff validity bound. It contributed no production frames. The replacement used tau=10 with the same four seeds, timestep, bead count and run lengths. [Failure replay and source identity](rejected_protocol.json).", "",
                      "A separate source-identity guard interrupted two trajectories after report-only source line shifts confused live-function source lookup; a third scheduled trajectory had not yet started. Completed CSVs were retained under exact hash checks; the three missing trajectories used unchanged protocols after switching to name-based AST lookup. [Execution record](execution_interruption.json)."])
    lines.extend(["","## Pairwise comparison details","", "Differences and drift diagnostics are retained in [comparisons.json](comparisons.json).",
                  "The original seed set was retained in all comparisons. The additional SC16/32 and matched-P RPC cases quantify observed bead/contraction uncertainty; no trajectories were selected by outcome.","",
                  "The SC P=8, dt=0.04 NPT energy second-half minus first-half shift is "+fmt(drift_by_case[label(by_key[("sc",8,.04,1.)])],"internal_energy_hartree")+
                  ". This is a retained pointwise drift diagnostic, not proof of stationarity for other cases or an independently corrected estimate.", "",
                  "![Convergence and pressure sampling](validation.png)","",
                  "Gaussian sample markers are shifted horizontally by 4% for visibility; the simulations use the bead counts stated in the table. [PDF figure](validation.pdf).", "",
                  "## Reanalyzing the accompanying archive", "",
                  "Extract the private validation archive into the repository checkout containing the public summaries and sidecars. The archive paths start with benchmarks/condensed_phase/results, with no additional enclosing directory. From the repository root:", "",
                  "```bash", "unzip /path/to/PyRPMD-condensed-phase-validation-20261102.zip",
                  "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src",
                  "NQE_RESULTS=benchmarks/condensed_phase/results",
                  "python benchmarks/condensed_phase/analyze_results.py \"$NQE_RESULTS\"",
                  "python benchmarks/condensed_phase/plot_validation.py \"$NQE_RESULTS\"", "```", "",
                  "This uses the supplied summaries and raw series without running dynamics. The scientific guide gives complete commands to regenerate the trajectories. The public repository contains no raw-CSV download links.", "",
                  "See [the scientific guide](../../../docs/condensed-phase-nqe.md) for models, equations, parameter provenance and scope."])
    (args.directory/"RESULTS.md").write_text("\n".join(lines)+"\n")


if __name__=="__main__":
    main()
