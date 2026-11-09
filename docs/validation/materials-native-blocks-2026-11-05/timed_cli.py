"""Run the formal ``pyrpmd run`` CLI entry in-process with propagation timers.

Usage: python timed_cli.py <report.json> <cli args...>
The source tree under test is selected by PYTHONPATH. Timers only wrap the
public propagation methods and native entry points; nothing is altered.
"""
import json
import sys
import time

t_start = time.perf_counter()
import numpy as np  # noqa: E402
import pyrpmd  # noqa: E402
import pyrpmd._native as native  # noqa: E402
from pyrpmd.dynamics import Dynamics  # noqa: E402
from pyrpmd import cli  # noqa: E402

report = {"pyrpmd": pyrpmd.__file__, "propagate": [], "propagate_block": [],
          "run_block": [], "advance": []}


def wrap_method(name):
    original = getattr(Dynamics, name)

    def timed(self, state, steps, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            return original(self, state, steps, *args, **kwargs)
        finally:
            report[name].append((int(steps), time.perf_counter() - t0))
    setattr(Dynamics, name, timed)


def wrap_native(name, steps_of):
    original = getattr(native, name)

    def timed(*args):
        t0 = time.perf_counter()
        try:
            return original(*args)
        finally:
            report[name].append((steps_of(args), time.perf_counter() - t0))
    setattr(native, name, timed)


wrap_method("propagate")
wrap_method("propagate_block")
wrap_native("run_block", lambda a: int(a[4]))
wrap_native("advance", lambda a: int(len(a[5])))

t_cli = time.perf_counter()
code = cli.main(sys.argv[2:])
t_end = time.perf_counter()


def summary(rows):
    lengths = [n for n, _ in rows]
    return {"calls": len(rows), "steps": int(sum(lengths)),
            "seconds": float(sum(t for _, t in rows)),
            "lengths": {str(k): lengths.count(k) for k in sorted(set(lengths))}}


out = {"pyrpmd": report["pyrpmd"], "exit": code,
       "import_seconds": t_cli - t_start, "cli_seconds": t_end - t_cli,
       **{k: summary(report[k]) for k in ("propagate", "propagate_block", "run_block", "advance")}}
with open(sys.argv[1], "w") as handle:
    json.dump(out, handle, indent=1)
