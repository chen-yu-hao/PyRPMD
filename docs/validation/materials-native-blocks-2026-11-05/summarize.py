import json, glob, statistics as st, subprocess, sys
S = sys.argv[1]
rows = {}
for label in ("before", "after"):
    for b in (32, 128):
        runs = [f for f in sorted(glob.glob(f"{S}/bench/results/{label}/p{b}/run*/timing.json")) if "warmup" not in f]
        d = [json.load(open(f)) for f in runs]
        prop = [x["propagate"]["seconds"] + x["propagate_block"]["seconds"] for x in d]
        nat = [x["run_block"]["seconds"] + x["advance"]["seconds"] for x in d]
        cli = [x["cli_seconds"] for x in d]
        wall = [x["process_wall_seconds"] for x in d]
        calls = {k: d[0][k]["lengths"] for k in ("propagate", "propagate_block", "run_block", "advance") if d[0][k]["calls"]}
        f = lambda v: f"{st.median(v):.4f} (min {min(v):.4f}, max {max(v):.4f}, n={len(v)})"
        rows[(label, b)] = dict(prop=st.median(prop), cli=st.median(cli), wall=st.median(wall))
        print(f"{label:6s} P={b:3d} propagation {f(prop)} native {f(nat)} cli {f(cli)} process {f(wall)} calls {calls}")
for b in (32, 128):
    B, A = rows[("before", b)], rows[("after", b)]
    print(f"P={b}: propagation speedup {B['prop']/A['prop']:.2f}x, cli {B['cli']/A['cli']:.2f}x, process {B['wall']/A['wall']:.2f}x")
