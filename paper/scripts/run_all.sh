#!/usr/bin/env bash
# Regenerate every calculation quoted in paper/pyrpmd-cpc.tex.
# Usage (from the repository root, with PyRPMD installed):
#   bash paper/scripts/run_all.sh WORK_DIRECTORY
# WORK_DIRECTORY must be new or empty. Wall time: about one hour on four cores.
set -euo pipefail
WORK=$(mkdir -p "$1" && cd "$1" && pwd)
REPO=$(cd "$(dirname "$0")/../.." && pwd)
RUNS=$WORK/runs; DATA=$WORK/data; EX=$WORK/exchange
mkdir -p "$RUNS" "$DATA" "$EX/build"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"

# 1. Analytic tutorials (Table 2, Fig. 2c,d).
for t in harmonic_pimd periodic_site_rate double_well_metad free_diffusion harmonic_spectroscopy; do
  python -m pyrpmd run examples/tutorials/$t/input.inp --output "$RUNS/$t" > "$WORK/$t.log" &
done; wait

# 2. Bead-number scan of the harmonic oscillator (Fig. 2a).
for P in 1 2 4 8 32 64; do
  sed "s/^beads = 16/beads = $P/" examples/tutorials/harmonic_pimd/input.inp > "$WORK/bead_p$P.inp"
done
(cd "$WORK" && printf '%s\n' 1 2 4 8 32 64 | xargs -P 4 -I{} sh -c \
  "python -m pyrpmd run bead_p{}.inp --output $RUNS/bead_p{} > bead_p{}.log")
ln -sfn "$RUNS/harmonic_pimd" "$RUNS/bead_p16"

# 3. Measurement scripts (Fig. 2b, Fig. 3, Fig. 5a).
python examples/quantum_statistics/harmonic_validation.py > "$DATA/quantum_statistics.json"
python paper/scripts/integrators.py > "$DATA/integrators.json"
python paper/scripts/throughput.py > "$DATA/throughput.json"

# 4. Model exchange reaction: dRPMD and Bennett-Chandler (Table 3, Fig. 4).
cp examples/tutorials/exchange_rate/* "$EX/" 2>/dev/null || true
${CC:-cc} -O3 -std=c11 -fPIC -shared -I native/include "$EX/exchange.c" -lm -o "$EX/build/libexchange.so"
cd "$EX"
for s in 1 2 3 4; do sed "s/^seed = 170165/seed = 17016$((5+s))0/" drpmd.inp > drpmd_s$s.inp; done
for s in 1 2 3; do sed "s/^seed = 170167/seed = 17016${s}7/" rate.inp > rate_s$s.inp; done
sed 's/^estimator = "ui"/estimator = "mbar"/' rate.inp > rate_mbar.inp
for s in 1 2 3; do sed 's/^estimator = "ui"/estimator = "mbar"/' rate_s$s.inp > rate_mbar_s$s.inp; done
python -m pyrpmd run drpmd.inp --output "$RUNS/exchange_drpmd" > drpmd.log
printf '%s\n' 1 2 3 4 | xargs -P 4 -I{} sh -c "python -m pyrpmd run drpmd_s{}.inp --output $RUNS/exchange_drpmd_s{} > drpmd_s{}.log"
python -m pyrpmd run rate.inp --output "$RUNS/exchange_rate_ui" > rate.log &
printf '%s\n' 1 2 3 | xargs -P 3 -I{} sh -c "python -m pyrpmd run rate_s{}.inp --output $RUNS/exchange_rate_ui_s{} > rate_s{}.log"; wait
python -m pyrpmd run rate_mbar.inp --output "$RUNS/exchange_rate_mbar" > rate_mbar.log &
printf '%s\n' 1 2 3 | xargs -P 3 -I{} sh -c "python -m pyrpmd run rate_mbar_s{}.inp --output $RUNS/exchange_rate_mbar_s{} > rate_mbar_s{}.log"; wait

# 5. Worker-count scaling of dRPMD (Fig. 5b); run on an otherwise idle machine.
python "$REPO/paper/scripts/parallel.py" "$EX" "$RUNS" > "$DATA/parallel.json"

# 6. Collect and plot.
cd "$REPO"
python paper/scripts/collect.py "$RUNS" "$DATA"
python paper/scripts/make_figures.py
