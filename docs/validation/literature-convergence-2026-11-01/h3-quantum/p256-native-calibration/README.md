# Proposed excluded P=256 implementation calibration

This is an unexecuted proposal for one fixed P=256 parent, seed 825601 / saved index 0 / completed sweep 2100. The independent calibration momentum is fixed by `SeedSequence([20261101, 256, 902731])`; primary momentum keys are untouched. Both step sizes, 0.1 and 0.05 fs, start from the same copied state.

Each step size runs a 4 fs fused-native path, the corresponding public `Dynamics.step` path, and a native 2 fs + saved state + recreated Dynamics/PES + 2 fs path. The budget is 360 total map steps, one CPU, 60 whole-process CPU seconds and 120 wall seconds. No rate, transmission, plateau or dynamical bead-convergence statistic will be computed.

The map tolerances are inherited unchanged from the real-BKMP2 native propagation harness: rtol 2e-8, with q/p/surface/energy atols of 5e-10 bohr, 5e-9 atomic momentum units, 2e-10 bohr and 2e-10 Hartree. Native save/recreate/reload comparisons must be bitwise exact, as in the earlier excluded P128 restart check. The Python reference shares the native free-ring map and model; it checks fused-loop equivalence, not independent physics.

[proposal.json](proposal.json) binds the exact parent envelope, original frozen sources, native library, BKMP2 library/build, prior checks, once-only policy and resource limits. Preparing this proposal performed no PES evaluation or propagation. Root adoption and a reviewed additive runner remain required before execution.
