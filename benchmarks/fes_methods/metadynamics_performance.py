"""Compare exact hill evaluation with a saved pre-optimization bias.py.

Example: git show 6fde29d:src/pyrpmd/bias.py > /tmp/bias_before.py
Then pass --before-source /tmp/bias_before.py --output performance.json.
CPU time measures the bias kernel only, not complete trajectory throughput.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import time
import timeit

import numpy as np
from pyrpmd.bias import MetadynamicsBias
from pyrpmd.cv import CVStack, Cartesian


def load_before(path):
    spec = importlib.util.spec_from_file_location("bias_before", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MetadynamicsBias


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calls", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--hills", type=int, nargs="+", default=[100, 1000, 5000])
    args = parser.parse_args()
    before_class = load_before(args.before_source)
    cv = CVStack([Cartesian(0, 0), Cartesian(0, 1)])
    rng = np.random.default_rng(481)
    records = []
    for count in args.hills:
        hills = [dict(center=rng.normal(size=2).tolist(), height=.03, step=i+1,
                      time=(i+1)*.1, bias_before=0.) for i in range(count)]
        before = before_class(cv, [.07, .14], .03, 1., hills=hills)
        after = MetadynamicsBias(cv, [.07, .14], .03, 1., hills=hills)
        points = rng.normal(size=(50, 2))
        for a, b in zip(before.physical_energy_gradient(points), after.physical_energy_gradient(points)):
            np.testing.assert_array_equal(a, b)
        q = np.array([.13, -.21])
        timings = [float(np.median(timeit.Timer(lambda: bias.physical_energy_gradient(q),
                    timer=time.process_time).repeat(repeat=args.repeats, number=args.calls)))
                   for bias in (before, after)]
        records.append(dict(hills=count, calls=args.calls, clock="process_time",
                            before_seconds=timings[0], after_seconds=timings[1],
                            speedup=timings[0]/timings[1], array_equal=True,
                            source_before_sha256=hashlib.sha256(args.before_source.read_bytes()).hexdigest(),
                            source_after_sha256=hashlib.sha256(Path(inspect.getfile(MetadynamicsBias)).read_bytes()).hexdigest()))
        print(json.dumps(records[-1]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2)+"\n")


if __name__ == "__main__":
    main()
