"""Excluded 500-sweep mixing check after overlap-based proposal changes."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from benchmarks.h3.surface_partition import _provenance, run
from benchmarks.h3.pes import BKMP2Potential
from pyrpmd.io import write_json


def main():
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    output = Path(__file__).resolve().parent
    library = ROOT/"benchmarks/h3/build/libbkmp2.so"
    provider = BKMP2Potential(library)
    protocol = {"phase": "excluded efficiency calibration", "created_before_sampling_utc": datetime.now(timezone.utc).isoformat(),
                "basis": "Original 9-node coarsened BAR overlap 0.889 TS / 0.953 reactant; high proposal acceptance",
                "beads": 128, "warmup": 500, "production": 500,
                "lambdas": (np.linspace(0., 1., 9)**2).tolist(), "radius_step": .08,
                "ratio_step": .03, "alpha_R": .7, "alpha_r_by_surface": {"ts": .7, "reactant": .45},
                "seed_by_surface": {"ts": 319051, "reactant": 319053}, "swap_interval": 1,
                "domain_samples": 4096, "TS_snapshot_interval": 100, "checkpoint_interval": 250,
                "source_identity": _provenance(provider),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "decisions_ignore": "Closeness to published rate or any pilot point estimate"}
    path = output/"efficiency-protocol.json"
    if path.exists():
        raise ValueError("preserve the existing efficiency protocol and outputs")
    write_json(path, protocol)
    summaries = {}
    for surface in ("ts", "reactant"):
        result = run(output/f"efficiency-{surface}", library, surface=surface, beads=128,
                     seed=protocol["seed_by_surface"][surface], warmup=500, production=500,
                     lambdas=np.asarray(protocol["lambdas"]), radius_step=.08, ratio_step=.03,
                     alpha_R=.7, alpha_r=protocol["alpha_r_by_surface"][surface],
                     swap_interval=1, domain_samples=4096, snapshot_interval=100 if surface == "ts" else 0,
                     checkpoint_interval=250, phase="calibration")
        summaries[surface] = {"minimum_BAR_overlap": result["analysis"]["minimum_BAR_overlap"],
                              "minimum_energy_ESS": min(x for x in result["analysis"]["diagnostic_energy_effective_samples"] if x is not None),
                              "acceptance_at_lambda1": result["acceptance_per_lambda"][:, -1],
                              "total_endpoint_traversals": int(np.sum(result["endpoint_traversals_per_walker"])),
                              "endpoint_snapshot_count": result["endpoint_snapshot_count"],
                              "elapsed_seconds": result["elapsed_this_call_seconds"]}
        print(json.dumps({"surface": surface, **summaries[surface]}, default=lambda x: x.tolist()), flush=True)
    write_json(output/"efficiency-summary.json", {"phase": protocol["phase"], "summaries": summaries,
               "source_unchanged": protocol["source_identity"] == _provenance(provider),
               "physical_converged": False, "literature_reproduced": False})


if __name__ == "__main__":
    main()
