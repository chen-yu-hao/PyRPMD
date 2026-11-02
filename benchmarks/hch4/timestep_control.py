"""Follow up the retained HCH4 short-NVE failure at a smaller time step."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from benchmarks.hch4.reproduction_input import PROTOCOL, resolve_pilot
from benchmarks.hch4.verify import paper_geometries
from benchmarks.hch4.verify_native import initial_state
from pyrpmd.constants import FS_TO_AU
from pyrpmd.dynamics import Dynamics
from pyrpmd.potentials import make_potential


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    started = time.process_time()
    original = json.loads(args.probe.read_text())
    _, config, system = resolve_pilot(args.library)
    potential = make_potential(config["potential"], system)
    reactant, _ = paper_geometries(30.0)
    beta = config["beta"]
    beads = 128
    state0 = initial_state(reactant, system.masses, beta, beads,
                           PROTOCOL["campaign"]["calibration_seed_excluded_from_primary"] + 10 + beads)
    rows = [row for row in original["nve_checks"] if row["beads"] == beads]
    for dt in (0.025,):
        dynamics = Dynamics(potential, system.masses, beta, beads, dt * FS_TO_AU,
                            native="required", threads=1, loop_backend="native")
        state = state0.copy()
        e0 = dynamics.energy(state) / beads
        result = dynamics.propagate_block(state, round(20 / dt), backend="native")
        energies = np.r_[e0, result.energies / beads]
        rows.append({"beads": beads, "dt_fs": dt, "duration_fs": 20,
                     "energy_span_per_bead_hartree": float(np.ptp(energies)),
                     "energy_end_minus_start_per_bead_hartree": float(energies[-1] - e0),
                     "scope": "Identical non-equilibrium start to the original probe"})
    report = {"schema": "pyrpmd.hch4.time-step-followup.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "original_probe_sha256": hashlib.sha256(args.probe.read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "native_probe_source_sha256": hashlib.sha256(Path(__file__).with_name("verify_native.py").read_bytes()).hexdigest(),
              "rows": rows,
              "successive_span_ratios": [rows[i + 1]["energy_span_per_bead_hartree"] / rows[i]["energy_span_per_bead_hartree"]
                                        for i in range(len(rows) - 1)],
              "CPU_seconds": time.process_time() - started,
              "original_gate_unchanged": original["gates"]["P128_short_NVE"],
              "rate_validated": False,
              "interpretation": "A diagnostic discretization refinement. It neither erases the paper-step failure nor establishes thermal or rate convergence."}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
