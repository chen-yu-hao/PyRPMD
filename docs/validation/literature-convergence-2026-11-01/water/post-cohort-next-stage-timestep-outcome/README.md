# Addendum after the complete paired timestep control

**The newly completed control does not satisfy the sealed proposals' conditional admission. Neither the additional preparation nor the prospective long-lag production study is adopted or launched.** This is an additive outcome record; the earlier design snapshot and every failed result remain unchanged.

The [complete control bundle](../timestep-paired-control/completed-original-cohort/README.md) preserves all eight seeds, both timesteps and all 16 branches. The root wrapper exited normally after 64 ps and 384,000 integration steps. Its energy checks pass, while both fixed diffusion equivalence tests remain `equivalence_unresolved`:

| Fine minus coarse estimate | Mean, Å²/ps | Nominal 95% interval, Å²/ps | Whole interval inside ±0.01? |
| --- | ---: | --- | --- |
| VACF integral through 2 ps | −0.01412453 | [−0.04126837, 0.01301931] | No |
| MSD fit over 1–2 ps | −0.02483984 | [−0.04919207, −0.00048761] | No |

The MSD interval is nominally below zero but crosses the negative equivalence boundary. The unchanged protocol therefore does not classify it as a proven whole-interval difference beyond the margin. The VACF interval includes zero and also extends beyond the equivalence band. Neither finding establishes that the finer 0.125 fs step is converged. These are nominal eight-pair, seven-degree-of-freedom intervals conditional on the original short trajectories and their preparations.

## What remains unadopted

The [sealed design snapshot](../post-cohort-next-stage-proposals/README.md) binds all eight original 100 ps checkpoints, the frozen preparation-only runner and coordinator, fixed additional 400 ps preparation, unchanged scalar thresholds and fixed late-window diagnostics. It separately proposes 20 × 20 ps NVE segments per chain, with longer diffusion lags and a finite resource ceiling. That snapshot is preserved byte-for-byte, including its independent design review.

Its prospective admission requires a complete successful paired control, normal process completion and released writer leases. Normal completion and the subsequent read-only acquisition/release of all 12 relevant leases are now documented. **The required numerical-control pass is false.** This addendum does not replace that rule with successful energy checks, a favorable individual chain, a smaller mean difference, or a wider tolerance.

Additional preparation might provide useful equilibrium diagnostics under the existing thermostat. It cannot by itself remove timestep uncertainty, and it cannot be assumed to eliminate the transient MSD curvature responsible for the original failed 0.5–1 ps versus 1–2 ps comparison. The original potential-energy stationarity failures for seeds 1102, 1105 and 1106 also remain in the record. The original formal literature comparison stays null.

A later explicit preparation-only decision could be considered with these limitations retained. No such decision, resource assignment, coordinator registration, changed timestep or new production launch is made here. The prospective long-lag design requires its own implementation, review, fully bound starting states and explicit adoption; its tentative 0.25 fs setting has not been validated by this control.

## Evidence and provenance

[outcome.json](outcome.json) binds the exact sealed proposal inventory, complete control report, normal root exit, post-exit lease proof and actual extracted-data replay. The latter reproduced every scientific and failed-gate field, plus all 33 compact curve arrays, while prohibiting original-directory fallback reads. Its only report difference is the explicitly relocated registration's provenance hash.

The first offline archiving attempt's interpreter error is retained with its original log and source snapshot. It performed no numerical trajectory read or physical work. The corrected offline attempt completed successfully without modifying any existing scientific source, data, threshold or result.

Root has separately requested a read-only feasibility comparison between the new coarse branches and the corresponding saved original first NVE segments. That is a distinct future evidence operation. Its outcome cannot be presumed here, and no all-segment reuse study is adopted by this addendum.
