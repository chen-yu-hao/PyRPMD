"""Compare the same C++ engine with/without normal-mode fusion.

The unfused path retains the previous C-O-C maps and RNG ordering. Both runs
stay in C++; this measures transform removal, not Python call batching or a
realistic expensive-PES speedup. Numerical equality is tolerance-based because
removing inverse/forward FFT pairs removes their roundoff as well.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path
import numpy as np
from pyrpmd.native_ensemble import NativeEnsemble


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("native-mode-fusion.json"))
    args = parser.parse_args()
    if args.steps < 1 or args.repeats < 1:
        parser.error("steps and repeats must be positive")
    cases = []
    for beads in (1, 8, 32, 31):
        seconds = {"unfused": [], "fused": []}
        states = {}
        for repeat in range(args.repeats):
            # Alternate ordering to reduce systematic cache/clock drift.
            for fused in ((False, True) if repeat % 2 == 0 else (True, False)):
                label = "fused" if fused else "unfused"
                engine = NativeEnsemble(np.zeros((beads, 2, 3)), [1., 2.], 2., .025,
                    potential={"kind": "harmonic", "k": .7}, thermostat="pile-l", seed=731,
                    fuse_modes=fused)
                start = time.perf_counter()
                result = engine.run(args.steps, stride=args.steps)
                seconds[label].append(time.perf_counter()-start)
                states[label] = engine.state()
                assert result["force_evaluations"] == args.steps+1
        deviations = {}
        for key in ("q", "p"):
            np.testing.assert_allclose(states["fused"][key], states["unfused"][key], atol=1e-8, rtol=1e-8)
            deviations[key] = float(np.max(np.abs(states["fused"][key]-states["unfused"][key])))
        assert states["fused"]["rng"] == states["unfused"]["rng"]
        median = {key: statistics.median(value) for key, value in seconds.items()}
        cases.append(dict(beads=beads, atoms=2, steps=args.steps, repeats=args.repeats,
                          seconds=seconds, median_seconds=median,
                          speedup=median["unfused"]/median["fused"], max_absolute_difference=deviations,
                          rng_identical=True, force_evaluations=args.steps+1))
    root = Path(__file__).resolve().parents[2]
    source = {p: hashlib.sha256((root/p).read_bytes()).hexdigest()
              for p in ("native/thermal.hpp", "native/thermal.cpp", "src/pyrpmd/native_ensemble.py")}
    report = dict(purpose="Mode-fusion microbenchmark on a cheap harmonic PES, not a chemistry speedup",
                  platform=platform.platform(), python=sys.version, numpy=np.__version__,
                  source_sha256=source, cases=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"cases": [{"beads": c["beads"], "speedup": c["speedup"],
                                 "max_absolute_difference": c["max_absolute_difference"]} for c in cases]}, indent=2))


if __name__ == "__main__":
    main()
