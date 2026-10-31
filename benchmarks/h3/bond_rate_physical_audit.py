"""Audit a *physical* 2D vs independent 1D H3 RPMD rate comparison.

Usage:
 python benchmarks/h3/bond_rate_physical_audit.py \
     --fes2d h3-4x.zip --rate1d h3-rate.zip --output audit.json

A bounded 2D PMF does NOT define free bimolecular reactant normalization.
Report unavailable quantities as null; never silently use an artificial
umbrella-box volume or an xi-surface recrossing factor for another surface.
"""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import zipfile
import numpy as np


def audit(fes2d_zip, rate1d_zip):
    with zipfile.ZipFile(fes2d_zip) as f2, zipfile.ZipFile(rate1d_zip) as f1:
        diag=json.loads(f2.read("validation/h3-bond-extended/diagnostics.json"))
        summary=json.loads(f2.read("validation/h3-bond-extended/run/summary.json"))
        resolved=json.loads(f2.read("validation/h3-bond-extended/run/input/resolved.json"))
        f=np.load(io.BytesIO(f2.read("validation/h3-bond-extended/run/fes.npz")),allow_pickle=False)
        bench=json.loads(f1.read("rate/benchmark.json"))
        one=json.loads(f1.read("rate/run/summary.json"))
        offline=json.loads(f1.read("rate/offline-rate.json"))
        if summary["measure"] != "cv_density":
            raise ValueError("2D PMF must be sampled as cv_density, not rate_flux")
        if diag["beads"] != bench["beads"] or diag["temperature_K"] != bench["temperature_K"]:
            raise ValueError("rate and PMF require matching beads and temperature")
        x=np.asarray(f["edges_0"],float);y=np.asarray(f["edges_1"],float)
        F=np.asarray(f["free_energy"],float)
        masked=np.asarray(f["mask"],bool)|~np.isfinite(F)
        beta=float(resolved["beta"])
        supported=~masked
        common=supported & supported.T if (F.shape[0]==F.shape[1] and np.allclose(x,y)) else None
        if common is None:
            raise ValueError("expected common H3 bond-distance grid for symmetry audit")
        delta=beta*(F-F.T)
        c=(x[:-1]+x[1:])/2
        focused=(c>=1.4)&(c<=3.1)
        local=common&focused[:,None]&focused[None,:]
        return {
          "schema":"pyrpmd.h3.bond-rate-physical-audit.v1",
          "temperature_K":diag["temperature_K"],"beads":diag["beads"],
          "two_d":{
            "coordinates":["r01","r12"],"measure":"cv_density",
            "samples":int(summary["samples"]),
            "supported_bins":int(supported.sum()),
            "mirror_rms_kBT":float(np.sqrt(np.mean(delta[common]**2))),
            "important_region_mirror_rms_kBT":float(np.sqrt(np.mean(delta[local]**2))),
            "qtst_cm3_molecule_s":None,"kappa":None,
            "rate_cm3_molecule_s":None,"converged":False,
            "missing_inputs":[
              "complete same-ensemble asymptotic entrance flux bridge",
              "canonical conditional normal-speed average on r01=r12",
              "signed RPMD recrossing launched on exactly r01=r12",
              "explicit third-bond and both exit-channel product commitments"
            ],
            "finite_histogram_reactant_volume_valid":False
          },
          "one_d_independent":{
            "qtst_cm3_molecule_s":float(offline["static"]["k_tst_cm3_s"]),
            "kappa":float(one["kappa"]),
            "candidate_rate_cm3_molecule_s":float(one["rate_estimate_cm3_s"]),
            "standard_error_cm3_molecule_s":float(one["rate_standard_error_cm3_s"]),
            "converged":bool(one["converged"]),
            "surface_definition":bench["surface_definition"],
            "parent_configurations":int(one["parent_configurations"]),
            "child_trajectories":int(one["child_trajectories"])
          },
          "two_d_vs_one_d_relative_rate_error":None,
          "scientific_note":"The 1D xi surface includes both equivalent exits; the diagonal r01=r12 is a different surface and cannot borrow its kappa. Do not infer a quantitative 2D rate from a finite umbrella PMF and an uncalibrated entrance.",
        }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fes2d",required=True)
    parser.add_argument("--rate1d",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    report=audit(args.fes2d,args.rate1d)
    target=Path(args.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
