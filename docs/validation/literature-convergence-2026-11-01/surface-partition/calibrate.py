"""Fixed, excluded calibration for the independent finite-P partition method.

Run from the repository root with one BLAS/OpenMP thread. Counts, seeds and
diagnostic decision rules are written before any physical-PES sampling.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from benchmarks.h3.classical_tst import MASS_AU
from benchmarks.h3.pes import BKMP2Potential
from benchmarks.h3.surface_partition import (
    _provenance, estimate_reference_domain, importance_diagnostic,
    reference_quadrature, run,
)
from pyrpmd.constants import KB_AU
from pyrpmd.io import write_json


def main():
    output = Path(__file__).resolve().parent
    if hasattr(os, "sched_getaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {allowed[-1]})
    library = ROOT/"benchmarks/h3/build/libbkmp2.so"
    provider = BKMP2Potential(library)
    identity = _provenance(provider)
    beta = 1/(KB_AU*300.)
    protocol = {
        "schema": "pyrpmd.h3.surface-partition-calibration.v1",
        "phase": "excluded calibration; no primary inference",
        "created_before_sampling_utc": datetime.now(timezone.utc).isoformat(),
        "beads": 128, "temperature_K": 300., "minimum_bead_pair_bohr": .5,
        "maximum_centroid_bond_bohr": 8.,
        "surfaces": ["ts", "reactant"], "chain_seeds": [319031, 319033],
        "importance_seeds": [319037, 319039], "importance_draws_per_surface": 8192,
        "domain_refinement_seeds": [319041, 319043], "domain_draws": 16384,
        "quadrature_orders_radial_angular": [[128, 64], [192, 96], [256, 128]],
        "warmup_sweeps": 1000, "production_sweeps": 5000,
        "lambda_grid": (np.linspace(0., 1., 25)**2).tolist(),
        "proposal": {"radius_step": .035, "ratio_step": .03, "alpha_R": .35, "alpha_r": .30},
        "swap_interval": 5, "sample_interval": 1, "diagnostic_interval": 10,
        "checkpoint_interval": 1000,
        "calibration_decisions_use_only": [
            "BAR/exchange overlap and lambda endpoint traversals",
            "proposal acceptance, autocorrelation ESS and first/second-half drift",
            "quadrature refinement and binomial domain-normalization precision",
            "importance ESS, largest normalized weight, and tail concentration",
            "walltime and memory, never closeness to a literature rate"],
        "working_diagnostic_flags": {"BAR_overlap_below": .2, "exchange_acceptance_below": .15,
            "energy_ESS_below": 100, "absolute_half_log_difference_above": .2,
            "largest_importance_weight_above": .05},
        "maximum_CPU_affinity_count": 1,
        "source_identity": identity,
        "calibration_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    protocol_path = output/"calibration-protocol.json"
    if protocol_path.exists():
        raise ValueError("calibration protocol already exists; preserve it and use an explicit new version")
    write_json(protocol_path, protocol)
    started = time.perf_counter()
    summaries = {}
    for index, surface in enumerate(protocol["surfaces"]):
        print(f"Starting excluded {surface} calibration", flush=True)
        quadratures = [reference_quadrature(provider, surface, beta, radial_order=radial,
                                            angular_order=angular)
                       for radial, angular in protocol["quadrature_orders_radial_angular"]]
        quadrature_rows = [reference.report() for reference in quadratures]
        for row in quadrature_rows:
            row["classical_log_difference_from_finest"] = (
                row["log_classical_shape_integral"]-quadratures[-1].log_classical)
        write_json(output/f"{surface}-reference-quadrature.json", quadrature_rows)
        reference = quadratures[1]
        importance, arrays = importance_diagnostic(provider, reference, beads=128, mass=MASS_AU,
            samples=8192, seed=protocol["importance_seeds"][index])
        raw = output/f"{surface}-importance.npz"
        np.savez_compressed(raw, **arrays)
        importance["raw_sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
        importance["source_identity"] = identity
        write_json(output/f"{surface}-importance.json", importance)
        print(json.dumps({"surface": surface, "importance_log_ratio": importance["log_quantum_classical_ratio"],
                          "importance_ESS": importance["effective_importance_samples"],
                          "importance_max_weight": importance["largest_normalized_weight"]}), flush=True)
        refined = estimate_reference_domain(quadratures[-1], beads=128, mass=MASS_AU,
            samples=16384, seed=protocol["domain_refinement_seeds"][index])
        write_json(output/f"{surface}-refined-reference-domain.json", refined)
        report = run(output/f"calibration-{surface}", library, surface=surface, beads=128,
            seed=protocol["chain_seeds"][index], warmup=1000, production=5000,
            domain_samples=16384, checkpoint_interval=1000, phase="calibration")
        analysis = report["analysis"]
        summaries[surface] = {
            "bar_log_ratio": analysis["log_quantum_classical_ratio"],
            "TI_simpson_log_ratio": analysis["TI_simpson_log_ratio"],
            "first_half_log_ratio": analysis["first_half_log_ratio"],
            "second_half_log_ratio": analysis["second_half_log_ratio"],
            "minimum_BAR_overlap": analysis["minimum_BAR_overlap"],
            "minimum_energy_ESS": min(value for value in analysis["diagnostic_energy_effective_samples"] if value is not None),
            "importance": {key: importance[key] for key in ("log_quantum_classical_ratio", "effective_importance_samples", "largest_normalized_weight")},
            "total_endpoint_traversals": int(np.sum(report["endpoint_traversals_per_walker"])),
            "minimum_exchange_acceptance": float(np.min(report["swap_accepts"]/np.maximum(report["swap_attempts"], 1))),
            "elapsed_chain_seconds": report["elapsed_this_call_seconds"],
        }
        print(json.dumps({"surface": surface, **summaries[surface]}), flush=True)
    result = {"phase": protocol["phase"], "summaries": summaries,
              "elapsed_seconds": time.perf_counter()-started,
              "peak_RSS_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "source_unchanged": identity == _provenance(provider),
              "source_identity": identity, "physical_converged": False,
              "literature_reproduced": False}
    write_json(output/"calibration-summary.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
