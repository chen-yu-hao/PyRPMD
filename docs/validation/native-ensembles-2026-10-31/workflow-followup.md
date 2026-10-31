---
orphan: true
---
# Native PIMD workflow and normal-mode fusion follow-up

## Changes

The `task="pimd"` workflow now accepts explicit `sampling.engine="native"` and
runs the C++ ensemble engine through the standard input/runner/resume interfaces.
Native observations and immutable streamed chunks remove Python calls at the
save cadence. Python is entered only at equilibration/checkpoint boundaries.
The existing serial/process/MPI task executor schedules replicas, with one
thread/rank per trajectory; this is not native bead-MPI or GPU force execution.

A checksummed atomic journal is the commit point for state and sample chunks.
Injected checkpoint/publication errors roll back state/RNG, and a completed
resume reanalyzes records without constructing or evaluating a PES. The direct
API now also rolls back after final-output publication errors and preserves
other writers' files. Native periodic PES preflight is available through the
standard factory, while compiled callbacks remain direct during trajectories.

Normal-mode NVT baths share one transform visit across free/bath/free maps:
four transforms per step instead of ten. The old unfused map remains selectable
for numerical comparisons. RNG draw ordering is unchanged; removing transforms
changes floating-point rounding, so the rounding path is fingerprinted.

## Tests actually run

Environment: Linux x86_64, Python 3.13.5, GCC 14.2.0, NumPy 2.3.5. Build:
`python setup.py build_ext --inplace`. Test environment uses `PYTHONPATH=src`,
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, `OMP_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`.

```text
python -m pytest -q tests/test_native_ensembles.py \
  tests/test_native_ensemble_portability.py tests/test_native_pimd_followup.py \
  --disable-warnings
213 passed in 20.87s

python -m pytest -q --ignore=tests/test_hho2_channels.py \
  --ignore=tests/test_hho2_scan.py --disable-warnings
1711 passed, 21 skipped, 4 warnings, 153 subtests passed in 160.43s
```

The 213 tests are included in the 1711; these counts must not be added.
The local source was the downloaded source distribution at parent `a4e28098`
with this follow-up applied. That distribution omits the pre-existing
`benchmarks/hho2/channels.py` and `scan.py`, so their two test modules were
excluded explicitly. Skips cover optional ASE, real BKMP2/EAM potentials and an
independent PMF reference solver. This is not an unqualified full-checkout or
real-system validation. No local MPI validation is claimed for the new workflow.

The new 89 tests include 56 fused/unfused method/bead/integrator comparisons,
native moments/offset clocks, publication rollback, concurrent file preservation,
nine thermostatted workflows, four NPT modes, serial/process replica equality,
interrupted recovery with changed checkpoint cadence, corruption checks, fixed
NVT atoms, and explicit unsupported-option rejection. A compiled cell-dependent
PES test counts exactly 38 native evaluations for 37 integration steps.
A save-every-step run with 7 equilibration + 1000 production steps makes exactly
two `NativeEnsemble.run` calls, not 1007 Python step calls.

Both standard example commands completed successfully:

```bash
python -m pyrpmd run examples/ensembles/pimd_native.inp --output /mnt/data/native-standard-nvt
python -m pyrpmd run examples/ensembles/pimd_native_npt.inp --output /mnt/data/native-standard-npt
```

These are harmonic/ideal-gas controls, not production material calculations.
Raw pytest outputs are retained beside this report.

## Performance evidence

Command: `python benchmarks/efficiency/native_mode_fusion.py --steps 5000
--repeats 3 --output /mnt/data/native-mode-fusion.json`.
Two atoms, cheap harmonic PES, alternating fused/unfused order, median elapsed
time. Both paths use one C++ call. This measures redundant transform removal,
not Python batching or a speedup for expensive electronic-structure potentials.
Short-run timing is noisy; these are measured cases, not performance guarantees.

| Beads | Unfused median (s) | Fused median (s) | Ratio |
| --- | ---: | ---: | ---: |
| 1 | 0.001852 | 0.001705 | 1.09 |
| 8 | 0.026563 | 0.016421 | 1.62 |
| 32 | 0.120663 | 0.071522 | 1.69 |
| 31 | 1.802510 | 0.744319 | 2.42 |

All paired RNG states are identical. Maximum q/p differences are at most
2.71e-13 in these runs. Odd non-power-of-two rings still use the existing direct
DFT; their absolute runtime is much worse than the power-of-two FFT case despite
a larger relative improvement. The JSON includes individual timings, platform,
source SHA256 hashes and deviations.

## CI and remaining scope

All seven PR workflows passed on parent `a4e280988bbfdb618f074d9458235fe4750f6357`
before this follow-up. Those results do not certify the follow-up commit; inspect
its own Actions checks. The PR remains unmerged.

This follow-up integrates PIMD, not diffusion/rate/FES/metadynamics. Surface NPT,
Suzuki-Chin action, calibrated PIGLET parameter sets, native socket/GPU force
backends, and real-material accuracy validation remain out of scope. Particle
H_P is not a conserved NPT enthalpy. Propagation avoids coordinate history in
memory, but offline correlation diagnostics retain scalar time series; historical
checkpoint files consume disk. Only one writer may own each replica directory.
