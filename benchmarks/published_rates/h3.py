"""Run a source-matched H3/BKMP2 pilot against the published 300 K RPMD row.

This creates an auditable comparison manifest, never a convergence assertion.
The short profile tests the real physical workflow; use independent seed,
bead/timestep/Rinf and sampling series before validating a production rate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from pyrpmd.constants import AU_TIME_FS
from pyrpmd.input import JobSpec
from pyrpmd.potentials.python import import_object
from pyrpmd.runner import run_resolved
from pyrpmd.workflows.fes import write_json

HERE = Path(__file__).resolve().parent
H3 = HERE.parent / "h3"


def configuration(output, library, *, profile="pilot", seed=831072, workers=2):
    if profile not in ("smoke", "pilot", "extended"):
        raise ValueError("profile must be smoke, pilot or extended")
    source = JobSpec.from_file(H3 / "rate.inp")
    data = source.data
    data.update(temperature="300 K", beads=128, seed=seed)
    data["potential"]["params"] = {"library": str(Path(library).resolve())}
    data["reaction"]["rinf"] = "30 bohr"
    # CPC 2013 Table 1: 111 windows at spacing .01 and k=30 hartree.
    # Match physical parameters, with an independently chosen sampling protocol:
    # this driver uses Andersen/Cayley rather than the paper's GLE/Verlet.
    # See umbrella-normalization.md for the source and initialization differences.
    data["umbrella"].update(start=-.05, stop=1.05, count=111,
                            kappa="30 hartree")
    profiles = {
        "smoke": ("5 fs", "10 fs", 2, 1, "10 fs", "2 fs", "5 fs"),
        "pilot": ("50 fs", "100 fs", 16, 2, "200 fs", "10 fs", "80 fs"),
        "extended": ("1 ps", "2 ps", 128, 4, "1 ps", "20 fs", "150 fs"),
    }
    equil, production, parents, pairs, parent_equil, parent_interval, child_time = profiles[profile]
    data["sampling"].update(equilibration=equil, production=production,
                            interval="0.5 fs", replicas=2)
    data["recrossing"].update(parent_replicas=2, parent_samples=parents,
                              child_pairs=pairs, equilibration=parent_equil,
                              parent_interval=parent_interval, child_time=child_time,
                              bootstrap=40 if profile != "extended" else 200)
    data["analysis"]["bootstrap"] = 10 if profile != "extended" else 40
    data["parallel"] = {"trajectory_workers": workers, "threads_per_rank": 1,
                         "backend": "process" if workers > 1 else "serial"}
    data["output"]["directory"] = str(Path(output).resolve() / "run")
    return source.resolve()


def run(output, library, *, profile="pilot", seed=831072, workers=2, resume=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config, system = configuration(output, library, profile=profile, seed=seed, workers=workers)
    provider = import_object(str(H3 / "pes.py") + ":build")(library=str(Path(library).resolve()))
    metadata = dict(provider.metadata)
    provider.close()
    write_json(output / "settings.json", {"config": config, "system": system.to_dict(),
                                          "pes_provenance": metadata})
    started = time.perf_counter()
    print(json.dumps({"event": "start", "profile": profile, "temperature_K": 300,
                      "beads": 128, "rinf_bohr": 30, "windows": 111, "seed": seed}), flush=True)
    result = run_resolved(config, system, resume=resume)
    elapsed = time.perf_counter() - started
    candidate = result.get("rate_estimate_cm3_s", result.get("rate_cm3_s"))
    support = result.get("rate_uncertainty", {})
    manifest = {
        "case_id": "h_h2_bkmp2_rpmd_300k",
        "reaction": "H + H2 -> H2 + H", "pes": "BKMP2", "method": "RPMD",
        "temperature_K": 300.0, "beads": 128, "timestep_fs": config["dt_au"] * AU_TIME_FS,
        "rinf_bohr": 30.0, "isotope_masses_au": system.masses.tolist(),
        "nuclear_spin_convention": "distinguishable_nuclei",
        "electronic_convention": "single_adiabatic_surface", "electronic_weight": 1.0,
        "symmetry_convention": "equivalent_product_channels_included", "symmetry_factor": 1.0,
        "rate_unit": "cm3 molecule-1 s-1", "rate_cm3_molecule_s": candidate,
        "qtst_rate_cm3_molecule_s": result.get("pmf", {}).get("k_qtst_cm3_s"),
        "transmission_coefficient": result.get("kappa"),
        "convergence": {"beads": False, "timestep": False, "rinf": False,
                        "sampling": False, "plateau": bool(result.get("plateau_detected", False))},
        "uncertainty": {"includes_pmf": bool(support.get("available", False)),
                        "includes_recrossing": bool(support.get("available", False)),
                        "standard_error_cm3_molecule_s": result.get("rate_standard_error_cm3_s"),
                        "independent_parent_chains": 2,
                        "method": support.get("combination", "unavailable")},
        "profile": profile, "seed": seed, "elapsed_seconds": elapsed,
        "workflow_status": result.get("status"), "physical_converged": False,
        "provenance": {"source": str(output / "run" / "summary.json"),
                       "settings": str(output / "settings.json"),
                       "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "pes": metadata},
        "notes": ["Source-matched temperature, PES, bead count, timestep, Rinf and channel convention.",
                  "Independent Andersen/Cayley sampling protocol, differing from the paper's GLE/Verlet protocol.",
                  "Diagnostic budget; no bead/time-step/Rinf or statistical convergence assertion.",
                  "A numerical ratio against the published rate is not a physical-validation pass."],
    }
    write_json(output / "calculation.json", manifest)
    print(json.dumps({"event": "finished", "elapsed_seconds": elapsed,
                      "rate_estimate_cm3_molecule_s": candidate, "status": result.get("status"),
                      "converged": False}), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--library", default=str(H3 / "build" / "libbkmp2.so"))
    parser.add_argument("--profile", choices=("smoke", "pilot", "extended"), default="pilot")
    parser.add_argument("--seed", type=int, default=831072)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.output, args.library, profile=args.profile, seed=args.seed,
        workers=args.workers, resume=args.resume)
