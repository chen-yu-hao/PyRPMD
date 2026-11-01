# Complete eight-chain paired timestep control

**All 16 fixed branches completed; both diffusion equivalence tests remain unresolved.** The energy checks passed. The control does not clear the original preparation failure, establish a converged diffusion coefficient, or support a paper-agreement claim.

The [unchanged control report](analysis/report.json) has SHA-256 `6d657d7dc11d98dd78ac8a2659a9907c8b94eedba0aa013dcd2d3d5db54aa370`. Every seed and branch remains present, including the original preparations that failed their potential-energy stationarity screen. The scope is the classical **external OpenMM reference**, with the frozen q-TIP4P/F implementation and finite 216-molecule cell. This is not a quantum calculation or validation of PyRPMD production dynamics.

## Fixed experiment and normal completion

The [protocol](../protocol.json), [root adoption](../original-cohort-adoption/root-adoption.json) and [actual registration](control-registration.json) were fixed before execution. Seeds 1101–1108 each use their exact original 100 ps pre-NVE checkpoint. One pending COM-projected Maxwell draw is saved once, then used by both new contexts. Each branch has no thermostat and no additional momentum draw.

| Quantity | Registered value |
| --- | --- |
| Chains / branches | Eight paired chains /16 complete branches |
| Timesteps |0.25 fs and 0.125 fs |
| Duration |4 ps per branch;64 ps total |
| Integration steps |16,000 coarse and32,000 fine per seed;384,000 total |
| Recording |801 coincident observations at 0, 5,…,4000 fs per branch |
| Molecular observables |All 216 unwrapped mass-weighted COM positions and coincident canonical velocities |
| Primary estimates |VACF integral through 2 ps; MSD linear slope over 1–2 ps |
| Statistical unit |Eight whole paired-chain differences; nominal Student-t interval with seven degrees of freedom |
| Equivalence rule |The entire 95% interval must lie inside ±0.01Å²/ps |

The root wrapper's [actual terminal observation](root-execution/root-exec-observation.json) records session 91496 exiting 0. Its [completion record](root-execution/completion.json) reports every command reaped, one 1540.609-second control invocation and 1546.269 seconds for registration, execution and analysis together. These are elapsed times, not an invented total CPU-time measurement.

After that observed exit, the offline evidence process acquired all 11 original writer locks plus `control.lock` nonblocking, held them only for the immutable snapshot, then closed every handle. The [lease proof](replay/lease-proof.json) records acquisition and release; it authorizes no later simulation.

## Results under the unchanged rules

Differences below are **fine minus coarse**. Units are Å²/ps.

| Estimate | Mean difference | Standard error | Nominal 95% interval | Fixed equivalence result |
| --- | ---: | ---: | --- | --- |
| VACF through 2 ps |−0.0141245323 |0.0114791349 |[−0.0412683731,0.0130193084] |Unresolved |
| MSD over 1–2 ps |−0.0248398419 |0.0102985619 |[−0.0491920712,−0.0004876125] |Unresolved |

The VACF interval includes zero and extends beyond the equivalence band. The MSD interval is nominally below zero, but crosses the negative equivalence boundary. Under the fixed classifier, neither interval is wholly outside the equivalence band, so both retain `equivalence_unresolved`, rather than being relabelled a proven difference exceeding the margin. Neither result establishes that 0.125 fs is itself converged.

[paired-chain-estimates.csv](paired-chain-estimates.csv) retains every branch estimate and all eight paired differences. No segment, seed, estimator or confidence interval was selected after seeing the values. Individual 4 ps records are too short and the original preparations insufficiently stationary to turn these comparisons into physical diffusion measurements.

All 16 energy envelopes and linear drift checks passed their existing limits of 0.01kcal/mol/water and 0.001kcal/mol/water/ps. Smaller energy errors at the finer step do not substitute for a successful transport equivalence test. The original failed 0.5–1 ps versus1–2 ps comparison and the failed preparation-U screens remain unchanged in the [original cohort](../../original-fixed-cohort/README.md).

## Independent checks and actual archive replay

The offline operation checked the complete registration with the frozen loader, each branch's full chunk/model checksums, and every stored zero-step canonical state against the same saved initial draw. It separately recomputed the eight paired means, sample standard errors and nominal df=7 intervals with `math.fsum`; those agree with the recorded analysis to rounding accuracy. See [independent checks](replay/independent-checks.json).

It then wrote a raw-only archive and **actually extracted** every member into a new directory. A new registration changes only paths to the extracted control, original-state subset and reference metadata. Every archived input byte remains untouched. The [relocation record](replay/relocation.json) makes this explicit; the relocated registration has its own hash and is not presented as the original registration.

The original frozen `paired_timestep.analyze` was called on that extracted data. An audit hook rejected all reads beneath the original control, original raw-data and root execution directories. All 96 numerical input files came from the extracted payload:80 branch checkpoints,8 shared initial states and8 exact original 100 ps checkpoints. The [read audit](replay/read-audit.json) records zero original-directory fallback attempts.

The resulting report matches **every scientific, statistical and failed-gate field**. Its only differing provenance field is `registration_sha256`, because the registration's paths were explicitly relocated. All 33 compact curve arrays have identical names, shapes, dtypes and bytes; the complete curve container also has the same SHA-256, `5b66d8e9e3dc4c964044b390bc743d7d3121fd492777dd37241d52d1b072eedb`. The [replay report](replay/replay-report.json) records 8.716 CPU seconds in the main process,0.112 CPU seconds in short read-only Git helpers,8.898 wall seconds and170,596 KiB peak RSS on CPU5 with one numerical thread. No Context, PES call, RNG draw or integration step occurred.

The first offline launch used a mistakenly resolved virtual-environment executable and failed the frozen environment check because the system interpreter lacked OpenMM package metadata. The [failure receipt](attempt-1-failure.json), original source snapshot and log are retained. It created no archive and read no numerical trajectory. The explicit second attempt corrected the launcher, used a new output directory and reduced its main CPU cap; it changed no scientific source, data or rule. [Observed exit 0](observed-exit-v2.json) closes that successful attempt. These are operational replay attempts, not repeated physical experiments.

## Raw archive and exact reconstruction

Archive: `pyrpmd-water-paired-timestep-2026-11-01-raw.tar.gz`

Size: **203,386,405 bytes**; **202 members**.

SHA-256: **`3629eb0f07319354aeacdf6d1af9327566277c7083246e35a4ff03ed271959bf`**.

The [raw-data index](raw-data-index.json) and [full member mapping](replay/archive-members.json) identify each source/member path, size and digest. The archive contains all control chunks/initial states, models, manifests, original selected100 ps checkpoints, source-run metadata, root execution records, the fixed protocol, README, source references and checksums. It contains no Python/Fortran source, compiled/PES library, source patch or third-party download. No upload is claimed.

`original-selection/` is deliberately a provenance subset sufficient for the control analysis; it is not the complete200 ps original campaign. That full campaign remains in its separately verified archive, SHA-256 `b3dc0faf5ddb84d06182ebf788edc73dfa1a0ec674069f01e61eaa62918d21af`.

[source-identities.json](source-identities.json) gives exact Git commits and hashes for all ten frozen imported source files and the original environment. In particular, the original diffusion helper must be reconstructed at its specified historical hash instead of silently taking a newer version. The raw archive supplies all numerical inputs. The path-only relocation and audit guard in [archive_and_replay.py](archive_and_replay.py) document the successful replay procedure; its once-only destination must be a fresh directory for any separately authorized reproduction.

## Consequences for future work

The separately sealed [next-stage proposals](../../post-cohort-next-stage-proposals/README.md) are still unadopted. Their conditional transport admission is not satisfied by this result. Additional preparation could investigate equilibrium under the frozen thermostat, but cannot by itself remove timestep uncertainty or the short-lag MSD curvature already observed. Any further timestep, preparation or long-lag production study needs its own explicit design, source checks, complete-cohort rules and resource decision. Nothing in this evidence operation launches or adopts such a study.
