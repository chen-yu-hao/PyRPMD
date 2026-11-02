# Completed excluded H + CH₄ efficiency pilot: saved-data replay

The fixed pilot completed normally and its saved calculation was reproduced without new potential evaluations, random draws or dynamics. **The aggregate proposal-efficiency screen failed.** The failure is retained; this bundle does not report a physical reaction rate, a quantum result or literature agreement.

This is an **independent-calculation replay by the producer's author**, with a separate read-only review requested from another agent. The replay uses a standalone implementation of the calculations and never imports the producer, its analyzer, PyRPMD or a potential library. Authorship independence must not be inferred from agreement between the two implementations.

## Complete execution and fixed scope

Root adopted this one excluded experiment before execution. The actual run ended with exit zero in session `70968`, and the supervisor reported that its owned worker was reaped. Its original report SHA-256 is `5162a97ad5e2fd981a039898020c3798e6b0ab957cd56e5ff3153c18aa0d58fc`.

All **800 deterministic calibration geometries, 3,072 main target draws and 128 paired entry-control draws** remain in their original ledgers. Of these 4,000 staged geometries, 3,672 required and received a successful PES evaluation. The other 328 draws retained exact geometric zero weights: 303 were outside the outer methane-radius domain, 18 belonged to an inactive maximum channel and 7 violated the outer pair-distance domain. They were neither replaced nor removed from the unconditional averages.

The primary target retains C–H distances from 1.4 to 4.5 bohr and a minimum all-atom pair distance of 0.7 bohr. Separate inner/outer domain records are diagnostic. The reference separation is 30 bohr; a preselected 16 entry draws per seed are paired with the same shapes and directions at 60 bohr. All 128 entry draws per seed remain in the primary denominator.

**Both surfaces, ξ = 0.999 and 1.007, are classical P = 1 calculations.** Evaluating the latter coordinate does not make it a P = 128 result. Each surface sums its four active-channel chart integrals once; the entry integral uses one sphere. The same global rotation measure cancels between numerator and denominator. No extra factor of four or physical-rate conversion is applied here.

The eight fixed fresh diagnostic seeds are `2026112201` through `2026112208`. Each complete seed vector, including both surfaces and the paired entry controls, is the uncertainty unit conditional on the frozen proposal. The proposal calibration is separate from those fresh draws.

## The failed efficiency checks

The following are the original registered thresholds and observed values, also recovered by the replay. The pooled ESS threshold is 128; the maximum normalized weight threshold is 0.025; the largest 1% of draws may carry at most 0.20 of the observed weight. Leaving out any whole seed may change the pooled log integral by at most 0.25.

| Target | Empirical ESS / 1,024 draws | Maximum normalized weight | Weight in largest 11 draws | Pooled efficiency screen |
| --- | ---: | ---: | ---: | --- |
| ξ = 0.999 | 336.1714 | 0.0205369 | 0.0924471 | Pass |
| ξ = 1.007 | 165.4859 | **0.0451470** | 0.1612604 | **Fail: maximum weight** |
| Entry sphere | 704.0164 | 0.00502933 | 0.0401074 | Pass |

The pooled ξ = 1.007 ESS exceeds its minimum; it is **not** an ESS-threshold failure. Its channel 2 independently fails both per-channel checks: ESS **27.5309 < 32** and maximum normalized weight **0.157433 > 0.10**, among 256 unconditional draws. The other channel screens pass. The aggregate screen consequently remains false. No threshold, seed, proposal or draw budget was changed after observing these results.

Observed outer-shell mass is very small for these draws, but its empirical ESS is approximately one. Those observations cannot establish an upper bound on unsampled tails or justify removing the finite-domain qualification. A passing efficiency screen would not by itself establish that qualification either.

## Independent calculation and uncertainty

[replay_saved.py](replay_saved.py) reconstructs the molecular coordinates and solves the surface radius with Brent's method. It computes the flux metric by summing the squared **Cartesian gradients divided by the atomic masses**, instead of calling the producer's compact metric formula. It independently evaluates the complete uniform/Gaussian/Student mixture in the correct sphere measure using precision matrices, including the stereographic Jacobian and both reflection components.

All 800 calibration objectives are recovered from the saved call energies and reconstructed measures. The replay reconstructs the central finite-difference Hessians, scaled-coordinate transformations, single Newton shifts, covariance matrices, ordered unit-Mahalanobis probe axes and reflection checks. It uses no new PES value or fitted parameter. The maximum probe residuals remain 0.1113, 0.4979 and 0.02137 for the two surfaces and entry model; the original calibration gates all pass.

An additive [calibration-input audit](calibration-input-audit/audit.json) also derives the initial shapes and scales from the pinned geometry/design, verifies all 711 exact tag-to-stencil displacements and all 89 ordered followup labels, and binds the physics/geometry hashes to the actual execution registration. Its largest coordinate difference is `2.44e-15` bohr; mass, temperature, reference bond difference and scale values match exactly. This closes the input-placement check without changing the already executed replay or archive bytes.

Every main and control log weight is rebuilt, including zeros. The four equally sized channel averages are summed once. Central ratios use the ratio of pooled integral means. The nominal log-ratio error uses the eight paired seed influences `A_i / mean(A) - B_i / mean(B)` and a Student-t factor with seven degrees of freedom. This preserves the observed pairing covariance. It is a small-sample conditional Monte Carlo approximation, **not a physical-rate confidence interval, a finite-domain error estimate or a bound on unseen weights**.

The two dimensionless log-ratio diagnostics are approximately −31.23075 and −30.57746. Their paired difference is approximately 0.65329. These are finite-domain P1 flux-ratio diagnostics and are not converted to rates. The matched entry-only 60/30-bohr log change after removing each sphere's area factor is approximately `6.18e-9`; it does not establish convergence of the complete transition-state surface or a rate with respect to reference separation.

The [replay protocol](replay-protocol.json) declares floating-point reconstruction tolerances before the calculation. They apply only to comparing two numerical implementations; every physical and efficiency predicate remains unchanged. Actual maximum absolute differences were much smaller:

| Reconstructed quantity group | Maximum absolute difference |
| --- | ---: |
| Coordinates and geometric measures | 3.55 × 10⁻¹⁵ |
| Calibration matrices, probes and summaries | 1.08 × 10⁻¹¹ |
| Full mixture log densities and log weights | 5.68 × 10⁻¹⁴ |
| Reported analysis values | 1.71 × 10⁻¹² |

The [complete replay result](actual-replay-attempt-1/replay-result.json) retains the recomputed analysis, comparison counts, input identities and failed gates. The [execution receipt](actual-replay-attempt-1/execution.json) records 2.40 CPU seconds and 2.42 wall seconds on one CPU. All 16 original pilot files and 129 bound external sources matched before and after the read-only calculation. A preceding [14-case analytic gate](initial-synthetic-gate/gate.json) checks Cartesian coarea measures, channel counting, mixture normalization constants, unconditional zeros and paired seed-vector uncertainty. It is not a new physical benchmark.

The actual pilot worker used 5.554369 CPU seconds as measured after reaping; the supervisor's elapsed interval was 7.520177 seconds. Its nominal whole-process CPU caps were 2 seconds for preparation, 3 for supervision and 55 for the worker. The preparation and supervisor partial CPU observations are retained as partial observations, not added to manufacture an exact whole-invocation total. Saved-data reconstruction and archive verification are separate evidence operations, with no additional physical sampling.

## Exact-byte raw archive and reconstruction

The local archive has **36 explicit members**: all 16 actual pilot files, 7 root execution documents/logs, 9 fixed reference metadata documents and 4 calculation-replay documents/logs. [raw-archive-members.json](raw-archive-members.json) records every original path, member path, size and SHA-256. It excludes Python/Fortran source, native or potential libraries, unrelated campaigns and prior pilot data. Those dependencies and the earlier excluded-pilot history remain identified by the frozen source/design records. No upload is claimed.

Archive filename: `pyrpmd-hch4-efficiency-pilot-2026-11-01-v1.tar.gz`

Archive size: **1,136,325 bytes**

Archive SHA-256: **`9a259292ec15c17171e999256ffa96c0aec6b7885e28ad40c78904ab8742cd3b`**

The packaging verification rehashed every original file, archive member and fresh extracted file. It then ran the same standalone saved-data calculator on the extracted data and reference metadata. Its recomputed analysis, calibration summaries, comparison counts/errors, staged counts, saved PES count and zero-draw counts matched the first replay exactly. The [archive execution receipt](archive-gate/execution.json) records this separate 3.12 CPU-second operation. Exact scientific equality is documented in the [fresh-extraction verification](fresh-extraction-verification.json).

Given the source version bound in the manifest, the existing compatible NumPy/SciPy environment and an extracted archive, the saved-data calculation can be reproduced with:

```bash
python replay_saved.py \
  --data /path/to/extracted/data \
  --physics /path/to/extracted/reference/physics-coarea-proposal.json \
  --design /path/to/extracted/reference/efficiency-design.json \
  --output /path/to/new-replay-result.json
```

The output must be new. Reconstructing saved calculations needs no PES library; reproducing the original physical evaluations separately requires the pinned external model and its provenance. The archive neither substitutes for that model validation nor authorizes new sampling.
