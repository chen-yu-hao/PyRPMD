#!/usr/bin/env bash
# Usage: run_bench.sh <label: before|after> <src dir> <beads: 32|128> <repeat index>
# Each run writes a fresh output directory; nothing is overwritten.
set -euo pipefail
S=/tmp/claude-0/-home-user-PyRPMD/11ebb923-4d4d-5a22-b9c2-baa1472b76ea/scratchpad
label=$1; src=$2; beads=$3; rep=$4
out=$S/bench/results/$label/p$beads/run$rep
if [ -e "$out" ]; then echo "refusing to overwrite $out" >&2; exit 1; fi
mkdir -p "$out"
export PYTHONPATH=$src:/home/user/PyRPMD OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
t0=$(date +%s.%N)
python $S/bench/timed_cli.py "$out/timing.json" run $S/bench/inputs/hch4-nve-p$beads.inp --output "$out/run" > "$out/stdout.txt" 2> "$out/stderr.txt"
t1=$(date +%s.%N)
python - "$out/timing.json" "$t0" "$t1" <<'EOF'
import json, sys
path, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
data = json.load(open(path)); data["process_wall_seconds"] = t1 - t0
json.dump(data, open(path, "w"), indent=1)
EOF
