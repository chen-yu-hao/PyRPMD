"""Join the executed protocols and record explicit first-campaign provenance.

Only for the 2026-11-02 review campaign. Its initial trajectories began
before the source-sidecar feature existed. Their source identities are marked
retrospective, never claimed as before-execution snapshots. Subsequent normal
driver runs capture their identities before propagation without this script.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from models import HarmonicCrystal
from validate import (gaussian_estimator_moments, npt_crystal_reference,
    numerical_source_identity, save_provenance, trajectory_identity, interval)
from pyrpmd.suzuki_chin import suzuki_chin_thermodynamics


def write_integrity_manifest(directory):
    root=Path(__file__).resolve().parents[2]
    analysis=[*Path(__file__).resolve().parent.glob("*.py"),root/"docs/condensed-phase-nqe.md"]
    manifest={"schema":"PyRPMD.condensed-campaign-integrity.v1",
              "sealed_utc":datetime.now(timezone.utc).isoformat(),
              "source_identity_at_seal":numerical_source_identity(),
              "analysis_source_sha256_at_seal":{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(analysis)},
              "first_campaign_source_capture":"mixed; inspect per-trajectory provenance and summary notes",
              "data_distribution":"All 76 raw observation CSVs are public in the Git repository under benchmarks/condensed_phase/results, alongside source, summaries, trajectory provenance sidecars and figures. The accompanying review archive remains available. Python source distributions exclude raw CSVs to control package size.",
              "files":{str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(directory.iterdir()) if p.is_file() and p.name!="integrity_manifest.json"}}
    (directory/"integrity_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--partial",action="store_true",help="seal only completed protocols and label combined summary incomplete")
    parser.add_argument("--manifest-only",action="store_true",help="refresh final integrity hashes without recomputing summaries")
    args=parser.parse_args()
    directory=args.directory
    if args.manifest_only:
        write_integrity_manifest(directory)
        return
    references=[]
    for index in range(15):
        path=directory/f"lj_case_{index}_summary.json"
        if args.partial and not path.exists():
            continue
        data=json.loads(path.read_text())
        options=argparse.Namespace(**data["run_options"])
        case=data["results"]["cases"][0]
        case.setdefault("barostat_tau",None if case["external_pressure"] is None else 5.)
        for replica,row in enumerate(case["replicas"]):
            raw=directory/row["raw"]
            digest=hashlib.sha256(raw.read_bytes()).hexdigest()
            if digest!=row["sha256"]:
                raise ValueError(f"raw checksum mismatch: {raw}")
            identity=trajectory_identity(options,case["method"],case["beads"],case["dt"],case["external_pressure"],replica)
            sidecar=raw.with_suffix(".provenance.json")
            if not sidecar.exists():
                save_provenance(raw,identity,recording=(
                    "retrospective first-campaign adoption: source hashes recorded at seal time; "
                    "no start-of-process source snapshot is asserted. Numerical model and "
                    "propagation functions were reviewed as unchanged; runner scheduling/reporting "
                    "and provenance features were edited during execution."))
            metadata=json.loads(sidecar.read_text())
            if metadata["identity"]!=identity or metadata["raw_sha256"]!=digest:
                raise ValueError(f"trajectory source/protocol/content provenance differs: {raw}")
            row["provenance"]=sidecar.name
            values=np.genfromtxt(raw,delimiter=",",names=True)
            keys=("internal_energy_hartree","volume","pressure_au")
            row["half_means"]={key:[float(np.mean(a)) for a in np.array_split(values[key],2)] for key in keys}
        if case["method"]=="sc":
            case["heat_capacity_kind"]="Cv" if case["external_pressure"] is None else "Cp"
            case["statistics"]["heat_capacity_over_kB"]=interval(
                [row["thermodynamics"]["heat_capacity_over_kB"] for row in case["replicas"]])
        references.append(case)
    combined=json.loads((directory/"lj_case_0_summary.json").read_text())
    combined["run_options"].update(protocol_index=None,jobs=4,resume=True)
    combined["results"]["cases"]=references
    combined["status"]="complete" if len(references)==15 else "partial"
    combined["completed_protocols"]=len(references)
    combined["expected_protocols"]=15
    combined["provenance_note"]=("Original ten protocols plus five targeted bead/RPC refinements. "
        "Initial completed replicas have explicitly retrospective source-sidecar adoption. "
        "Refinement trajectories captured source identities before execution and checked after it.")
    (directory/"lj_summary.json").write_text(json.dumps(combined,indent=2,allow_nan=False)+"\n")

    path=directory/"harmonic_summary.json"
    harmonic=json.loads(path.read_text())
    model=HarmonicCrystal(bulk=.2)
    for case in harmonic["results"]["gaussian_estimator_cases"]:
        case["deterministic_production_moments"]=gaussian_estimator_moments(model,8.,case["beads"])
    pressure=model.quantum_reference(8.)["pressure"]
    for case in harmonic["results"]["npt"]:
        case["reference"]=npt_crystal_reference(model,8.,pressure,case["beads"])
        cp=[]
        for row in case["replicas"]:
            raw=directory/row["raw"]
            if hashlib.sha256(raw.read_bytes()).hexdigest()!=row["sha256"]:
                raise ValueError(f"raw checksum mismatch: {raw}")
            data=np.genfromtxt(raw,delimiter=",",names=True)
            columns={key:data[key] for key in data.dtype.names}
            thermo=suzuki_chin_thermodynamics(columns,8.,volumes=data["volume"],pressure_au=pressure,block_size=40)
            row["thermodynamics"]=thermo
            cp.append(thermo["heat_capacity_over_kB"])
        case["statistics"]["heat_capacity_Cp_kB"]=interval(cp)
    harmonic["provenance_note"]=("First-campaign harmonic execution; source identity recorded "
        "retrospectively at seal time. Gaussian moments and NPT Cp postprocessing rerun on "
        "the retained data with the sealed analysis source.")
    path.write_text(json.dumps(harmonic,indent=2,allow_nan=False)+"\n")
    write_integrity_manifest(directory)


if __name__=="__main__":
    main()
