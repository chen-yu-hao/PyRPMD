---
orphan: true
---
# Native ensembles: local validation, 2026-10-31

Base: `76dd6ad24adc6c244f775857987cd2f0e119845f`.

The C++17 `_thermal` extension was compiled locally with GCC and run on Python
3.13.5. The local checkout was a source subset, not a full repository checkout.
The existing `pyrpmd.dynamics` module was unavailable there; its eight comparison
cases were skipped locally and are retained for the full-repository CI.

Release build: **114 passed, 8 skipped** (14.65 seconds). Undefined-behavior
sanitizer: **88 passed, 8 skipped, 26 deselected**. Statistical long runs and the
thread timing check were excluded from that focused sanitizer run; no sanitizer
diagnostic was emitted. The unit-aware NPT CLI executed 10,000 steps in one
native call, with 10,001 potential evaluations and 1,000 streamed frames;
`physical_converged=false` remains explicit.

This is a summary of those observed local results, not a raw console-log archive.
Existing repository regressions, installed-wheel checks, and real-system
workflows were not run in that source-subset environment. Remote Actions status
must be checked separately. Commands to reproduce:

```bash
python -m pip install -e '.[test]'
python -m pytest -q tests/test_native_ensembles.py
# For a build made with -fsanitize=undefined -fno-sanitize-recover=all:
python -m pytest -q tests/test_native_ensembles.py \
  -k 'not harmonic_finite and not ideal_gas and not gil'
python benchmarks/efficiency/native_ensembles.py --output native-benchmark.json
```

## Call-boundary benchmark

The retained driver is `benchmarks/efficiency/native_ensembles.py`; rounded
observed medians are in `benchmarks/efficiency/native-ensembles-summary.json`.
Each case uses two atoms, a cheap harmonic model, PILE-L, 20,000 steps and three
repeats. The baseline is **the same new native engine called one step at a time**,
not the old entire Python workflow or a costly PES.

| Beads | One C++ block, median seconds | One Python call per step | Ratio |
| --- | ---: | ---: | ---: |
| 1 | 0.00631 | 0.07798 | 12.37 |
| 8 | 0.08931 | 0.16654 | 1.86 |
| 32 | 0.39605 | 0.48946 | 1.24 |

Every case has an identical final state digest and 20,001 force evaluations in
both executions. These ratios isolate boundary/boxing overhead; they are not
promised speedups for ML, DFT, real chemistry or existing PyRPMD workflows.

Harmonic and ideal-gas tests compare numerical distributions to analytic finite-
bead/statistical controls. Flexible-cell tests exercise independent step algebra
and constrained ideal-gas volume fluctuations; they are not a full production
crystal/NPT validation. PIGLET tests use a canonical test matrix, not a calibrated
bead-count reduction. Suzuki-Chin NPT, slab/fixed-atom NPT and legacy-workflow
migration are explicitly outside this change.
