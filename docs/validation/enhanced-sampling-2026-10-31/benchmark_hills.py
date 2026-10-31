"""Reproduce exact-sum hill-kernel timing against the audited parent commit."""
import json
from pathlib import Path
import statistics
import subprocess
import time
import types

import numpy as np
from pyrpmd.bias import MetadynamicsBias
from pyrpmd.cv import Cartesian, CVStack


def main():
    legacy = types.ModuleType("bias_before_sampling_audit")
    source = subprocess.check_output(
        ["git", "show", "347facf:src/pyrpmd/bias.py"], text=True)
    exec(compile(source, "<audited-parent-bias>", "exec"), legacy.__dict__)
    cv = CVStack([Cartesian(0, 0), Cartesian(0, 1)])
    rng = np.random.default_rng(11)
    records = []
    for count in (100, 1000, 5000):
        hills = [{"center": center.tolist(), "height": .01, "step": i + 1,
                  "time": float(i + 1), "bias_before": 0.}
                 for i, center in enumerate(rng.normal(size=(count, 2)))]
        biases = [cls(cv, [.3, .5], .01, 1., mode="standard", hills=hills)
                  for cls in (legacy.MetadynamicsBias, MetadynamicsBias)]
        for points in (np.array([.1, -.2]), rng.normal(size=(13, 2))):
            for before, after in zip(biases[0].physical_energy_gradient(points),
                                     biases[1].physical_energy_gradient(points)):
                np.testing.assert_array_equal(before, after)
        timings = []
        for bias in biases:
            repeats = []
            for _ in range(5):
                started = time.process_time()
                for _ in range(200):
                    bias.physical_energy_gradient(np.array([.1, -.2]))
                repeats.append(time.process_time() - started)
            timings.append(statistics.median(repeats))
        records.append({"hills": count, "calls": 200, "repeats": 5,
                        "median_before_cpu_seconds": timings[0],
                        "median_after_cpu_seconds": timings[1],
                        "speedup": timings[0] / timings[1],
                        "identical_energy_and_gradient": True})
    target = Path(__file__).with_name("evidence") / "hill-kernel-timing.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
