#!/usr/bin/env python3
"""Write/replay exact legacy checkpoints using the currently imported checkout.

Run --write in main@801953d, then --compare in the candidate with identical
compiler flags and Python dependencies. Reference NPZs are local artifacts;
only the JSON verification report belongs in the repository.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pyrpmd.native_ensemble import NativeEnsemble


def cases():
    for beads, atoms in [(1, 3), (7, 3), (32, 3), (32, 64)]:
        for mode in ("andersen", "langevin", "pile-l", "pile-g", "svr", "ffl", "nhc", "trpmd", "gle", "nm-gle", "nm-gle-g", "piglet"):
            bath = {"mode": mode}
            if "gle" in mode or mode == "piglet":
                bath["gle"] = {"A": [[1., .2], [-.2, 1.]]}
            if mode == "piglet":
                bath["gle"].update(A=[bath["gle"]["A"]]*beads, C=np.eye(2).tolist(),
                                   beads=beads, beta=2., provenance="Synthetic test covariance")
            yield f"{mode}-p{beads}-n{atoms}", dict(
                q=np.full((beads, atoms, 3), .1), masses=np.linspace(1., 2., atoms),
                beta=2., dt=.01, potential={"kind": "harmonic", "k": .5}, thermostat=bath)
    for mode in ("pile-l", "pile-g", "nhc", "gle", "piglet"):
        bath = {"mode": mode}
        if mode == "gle":
            bath["gle"] = {"A": [[1., .2], [-.2, 1.]]}
        if mode == "piglet":
            bath["gle"] = {"A": [[[1., .2], [-.2, 1.]]]*32, "C": np.eye(2).tolist(),
                           "beads": 32, "beta": 2., "provenance": "Synthetic test covariance"}
        yield mode+"-large", dict(q=np.full((32, 512, 3), .1), masses=np.ones(512),
                                 beta=2., dt=.01, potential={"kind": "harmonic", "k": .5}, thermostat=bath)
    yield "bzp-large", dict(q=np.full((32, 512, 3), .1), masses=np.ones(512), beta=2., dt=.01,
                           potential={"kind": "free"}, cell=np.eye(3)*10, thermostat="pile-l",
                           barostat={"mode": "bzp", "pressure": .2565, "tau": 10.})
    yield "pile-l-bzp", dict(q=np.full((8, 9, 3), .1), masses=np.ones(9), beta=2., dt=.01,
                             potential={"kind": "periodic_cosine", "heights": [.1, .2, .3]},
                             cell=np.eye(3)*3, thermostat="pile-l",
                             barostat={"mode": "bzp", "pressure": .1, "tau": 2.})

    yield "middle-bzp", dict(q=np.full((32, 64, 3), .1), masses=np.ones(64), beta=2., dt=.01,
                             potential={"kind": "free"}, cell=np.eye(3)*5, thermostat="pile-l",
                             barostat={"mode": "bzp", "pressure": .1, "tau": 2.})
    for mode in ("mttk", "rgb", "scr"):
        barostat = {"mode": mode, "tau": 2.}
        if mode == "rgb":
            barostat.update(stress=(np.eye(3)*.1).tolist(), reference_cell=(np.eye(3)*3).tolist())
        else:
            barostat["pressure"] = .1
        if mode == "scr":
            barostat["compressibility"] = .1
        yield mode, dict(q=np.full((1 if mode == "scr" else 7, 2, 3), .1), masses=[1., 2.],
                         beta=2., dt=.01, potential={"kind": "free"}, cell=np.eye(3)*3,
                         thermostat="pile-l", barostat=barostat)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--write", type=Path)
    actions.add_argument("--compare", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    directory = args.write or args.compare
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    configurations = [(name+"-"+integrator, {**options, "integrator": integrator,
                       **({"sampling_integrator": "bcocb" if integrator == "cayley-bcb" else "baoab"}
                          if name == "middle-bzp" else {})})
                      for name, options in cases()
                      for integrator in ("cayley-bcb", "normal-mode-bab")]
    for name, options in configurations:
        if args.threads != 1:
            options["threads"] = args.threads
        engine = NativeEnsemble(**options, seed=49)
        engine.run(5)
        checkpoint = directory/(name+".npz")
        expected_file = directory/(name+"-expected.npz")
        if args.write:
            engine.save(checkpoint)
        else:
            # Restore into a different initial seed: saved state owns the RNG.
            engine = NativeEnsemble(**options, seed=901)
            engine.load(checkpoint)
        engine.run(11)
        state = engine.state()
        if args.write:
            engine.save(expected_file)
        expected = NativeEnsemble(**options, seed=0)
        expected.load(expected_file)
        wanted = expected.state()
        same = all(np.array_equal(state[key], wanted[key]) for key in state)
        rows.append({"case": name, "bitwise_equal": same,
                     "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
    args.output.write_text(json.dumps({"cases": rows, "accepted": sum(r["bitwise_equal"] for r in rows),
                                       "expected": len(rows)}, indent=2)+"\n")
    if not all(row["bitwise_equal"] for row in rows):
        raise SystemExit("Legacy replay differs: "+str([row["case"] for row in rows if not row["bitwise_equal"]]))
    print(f"{len(rows)}/{len(rows)} legacy checkpoints replayed bitwise")


if __name__ == "__main__":
    main()
