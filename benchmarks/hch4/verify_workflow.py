"""Three-step real umbrella/checkpoint smoke test, without rate analysis."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from benchmarks.hch4.reproduction_input import resolve_pilot
from pyrpmd.workflows.fes import _sample_window, resolve_umbrellas
from pyrpmd.workflows.rate import _prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a fresh workflow-smoke output directory")
    args.output.mkdir(parents=True)
    started = time.process_time()
    records = []
    for beads in (1, 128):
        _, config, system = resolve_pilot(args.library, beads=beads, output=args.output)
        config["sampling"].update(equilibration_steps=0, production_steps=3,
                                   sample_interval=1, interval_steps=1)
        prepared = _prepare(config, system, str(args.output / f"P{beads}-preparation"))
        _, specs, centers, stiffness = resolve_umbrellas(prepared, system)
        for window in (5, 105):  # Table centers xi=0 and xi=1.
            outputs = []
            for backend in ("native", "python"):
                working = copy.deepcopy(prepared)
                working["dynamics"]["loop_backend"] = backend
                path = args.output / f"P{beads}-window-{window}-{backend}"
                task = {"config": working, "system": system, "specs": specs,
                        "center": centers[window], "stiffness": stiffness,
                        "window": window, "replica": 0, "output": str(path)}
                first = _sample_window(task)
                checkpoint = Path(first["checkpoint"])
                digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                restarted = _sample_window(task)
                for key in ("samples", "last_q", "last_p", "energies"):
                    np.testing.assert_array_equal(first[key], restarted[key])
                assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
                outputs.append(first)
            a, b = outputs
            q_error = float(np.max(abs(a["last_q"] - b["last_q"])))
            p_error = float(np.max(abs(a["last_p"] - b["last_p"])))
            e_error = float(np.max(abs(a["energies"] - b["energies"]))) / beads
            native_steps = a["propagation"]["native_steps"]
            passed = bool(q_error < 2e-10 and p_error < 2e-9 and e_error < 2e-10
                          and native_steps == 3 and a["propagation"]["python_steps"] == 0)
            records.append({"beads": beads, "window": window, "center": centers[window].tolist(),
                            "steps": 3, "samples": len(a["samples"]),
                            "native_propagation": a["propagation"],
                            "python_propagation": b["propagation"],
                            "position_error_bohr": q_error, "momentum_error_au": p_error,
                            "energy_per_bead_error_hartree": e_error,
                            "completed_restart_arrays_and_file_exact": True, "passed": passed})
    root = Path(__file__).resolve().parents[2]
    paths = ["benchmarks/hch4/verify_workflow.py", "benchmarks/hch4/reproduction_input.py",
             "benchmarks/hch4/reproduction_protocol.json", "benchmarks/hch4/pes.py",
             "src/pyrpmd/workflows/rate.py", "src/pyrpmd/workflows/fes.py",
             "src/pyrpmd/workflows/sampling.py", "src/pyrpmd/checkpoint.py"]
    result = {"schema": "pyrpmd.hch4.workflow-smoke.v1", "passed": all(r["passed"] for r in records),
              "scope": "Three steps only: standard rate preparation, umbrella task, native/Python execution and exact completed-checkpoint reload. No equilibrium, PMF or rate analysis.",
              "rate_validated": False, "records": records, "CPU_seconds": time.process_time() - started,
              "source_files": {str(p): hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths},
              "files": {str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(args.output.rglob("*")) if p.is_file()}}
    (args.output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": result["passed"], "CPU_seconds": result["CPU_seconds"], "records": records}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
