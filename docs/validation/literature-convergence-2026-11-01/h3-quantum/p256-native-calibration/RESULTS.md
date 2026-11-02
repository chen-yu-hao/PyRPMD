# Excluded P=256 map and restart calibration: passed

The adopted, independently reviewed calibration completed once with exit code 0. It used the predetermined seed-825601 parent at saved index 0, the separate calibration momentum key `[20261101, 256, 902731]`, and both registered step sizes without substitution or retry. The [original proposal](proposal.json), its initial unexecuted status and the initial review candidate remain preserved.

The actual work was **360 map steps**: 240 through the fused native propagation entry point and 120 through public `Dynamics.step`, all with P=256. At each step size, the continuous native trajectory, Python loop and native 2 fs + save/recreate/reload + 2 fs path used identical copied initial coordinates and momenta. No thermostat, bias or constraints were introduced.

| Step size | Horizon | Maximum q difference | Maximum p difference | Maximum surface/clock difference | Maximum energy difference | Native restart |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 0.1 fs | 4 fs | 0 | 0 | 0 | 9.6633812e-13 Ha | Every recorded array bitwise identical |
| 0.05 fs | 4 fs | 0 | 0 | 0 | 7.9580786e-13 Ha | Every recorded array bitwise identical |

The differences compare the public Python loop with native fusion. The Python loop uses the same native free-ring map and PES, so this establishes implementation equivalence at the actual P=256 memory/layout size, not an independent physics oracle. All initial and subsequent trace entries were finite. The inherited tolerances were unchanged. Native save/recreate/reload matched q, p, surface, energy, step and time bit for bit, and no Python fallback occurred in native paths.

The short native energy ranges were 0.006863372406 Ha and 0.001382733944 Ha for 0.1 and 0.05 fs respectively, using the sum-over-beads ring-polymer Hamiltonian and one-femtosecond boundary observations. These are recorded diagnostics; this calibration does not impose a new energy-drift threshold or establish physical timestep convergence.

## Provenance and resources

[result/report.json](result/report.json) is a byte-exact copy of the original report, SHA256 `ea075cd029804fa4dbc691ad90db3ffc075deaa22471d51573d50b6b356abb53`. It records `all_checks_passed: true`, exact step accounting, actual imported paths and unchanged source/input guards. The frozen executed driver is `benchmarks/h3/p256_native_calibration.py`, SHA256 `9eb0bfc88b210d6abf432b91f8ee54274ec3f0d94de8c573433d76400b5220c8`.

The numerical process used CPU 6 with all registered thread counts set to one. Its actual full child-process CPU usage, including finalization, was **0.737406 seconds**; the supervising call took **0.776089 seconds** of wall time. The report's earlier in-process measurement was 0.701116 CPU seconds and 0.703121 wall seconds, with peak RSS **67,640 KiB**. The 60-CPU-second hard limit and 120-second wall limit were installed; neither was approached. [execution.json](execution.json) preserves the observed exit and complete output-log hash.

The focused synthetic gate passed **16 tests** on the original Python 3.12 runtime. Independent review caught an initial-frame finiteness omission before any PES evaluation: matching NaNs could otherwise pass NumPy's default comparison. The resolved candidate checks every trace entry and uses `equal_nan=False`. Both the initial 15-test candidate and the corrected 16-test candidate, source snapshots and review are retained. These tests are separate from the earlier cross-version suite.

The initial-state ledger was committed before any PES call. [raw-data-index.json](raw-data-index.json) identifies the nine raw checkpoint/trace NPZ files and their local archive. The archive was extracted into a new directory, every member hashed and every raw checkpoint integrity envelope reloaded. No external model source, model binary, native binary or repository source is included. The original report and all bound runtime outputs remained unchanged during packaging.

This calibration contains no transmission coefficient, rate, plateau estimate or dynamical bead-convergence claim. Its independent momentum stream and outputs are excluded from the primary child statistics.
