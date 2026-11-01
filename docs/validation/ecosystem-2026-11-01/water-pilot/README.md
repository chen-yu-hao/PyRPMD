# q-TIP4P/F periodic integration evidence

This directory retains the actual model, trajectory/dipoles, diagnostics and
report from the 2026-11-01 pilot. It is a software integration test with a real
water potential. **No physical liquid result is claimed.**

Reproduce in a new directory from the checkout:

```bash
python benchmarks/water/pilot.py --output runs/water-pilot \
  --beads 4 --equilibration-steps 64 --production-steps 256 --seed 541
python benchmarks/water/replay.py runs/water-pilot
```

Replay this retained record without running OpenMM:

```bash
python benchmarks/water/replay.py docs/validation/ecosystem-2026-11-01/water-pilot
```

## What ran

- Python 3.12.14 on Linux x86_64; NumPy/SciPy/OpenMM versions are in `report.json`.
- OpenMM Reference platform, q-TIP4P/F with virtual M sites, periodic PME and
  no constraints. The model is stored in `model.xml`.
- Eight molecules, 298 K, 0.997 g/cm³, P=4, 0.05 fs timestep, 64 sampling and
  256 NVE production steps. Production spans 12.8 fs.
- The integrator and forces ran on CPU. No GPU speed or correctness claim is made.

## Actual checks

| Check | Observed value | Predeclared bound |
| --- | ---: | ---: |
| Independent cluster energy error | 2.35e-12 hartree | 2e-10 hartree |
| All-coordinate independent finite-difference force error | 7.41e-11 hartree/bohr | 2e-9 hartree/bohr |
| Maximum NVE ring-H energy drift per bead | 4.20e-6 hartree | 1e-5 hartree |

All saved arrays are finite. Hash verification and estimator replay pass.
`diffusion-diagnostic.json` reports `ballistic_or_drifting`; an accepted liquid
diffusion coefficient is unavailable. `spectrum-diagnostic.npz` is an unresolved
short-record RPMD diagnostic, not the original paper's PA-CMD liquid spectrum.

The grid initialization, tiny box, short run, four beads, one replica and
2.5 Å cutoff do not match or converge the paper's liquid calculation. The report
therefore retains `physical_converged=false`, `literature_reproduced=false` and
`liquid_diffusion_angstrom2_ps=null` even though software checks pass.

## Provenance and artifacts

`report.json` records the actual base Git revision, dirty-working-tree state,
source hashes at run start and end, model parameters, full command, dependency
versions, artifact sizes/hashes and runtime. It does not invent a final commit
hash before the changes are committed. A later source change need not invalidate
the retained old result; compare its recorded hashes with the version being
reviewed and rerun when evaluating a changed implementation.

`trajectory.npz` stores atomic-unit time, unwrapped molecular centroid positions
and velocities, per-bead dipoles, the ring-polymer Hamiltonian, box, initial
physical geometry and final physical q/p. These are enough to recompute the
reported diagnostic estimators. They are not a full restart checkpoint with
thermostat/RNG history. No external model download is required.

The scientific source and source-verified table values are recorded separately
in `benchmarks/water/references.json`; none of those literature numbers are
presented as a value measured in this pilot.
