# Subspace path sources (2026-11-10)

This note records the validation scope for issue #157, which adds selectable
one-dimensional path sources to `pyrpmd subspace-correct`:

```toml
[path]
source = "centroid"       # centroid | bead_weighted | instanton | least_action
```

The implementation is based on `main` after PR #159 so that the converged
truncated-DVR diagnostics and the existing subspace-transfer records remain
available. Every source uses the same `assemble_reaction_path`
continuation, the same PES, and one projected `H_S` for both halves of
`C_S = k_QM(H_S) / k_RPMD(H_S)`. The report written by the pipeline retains the
source and its diagnostics under `path`, and every generated report continues
to set `physical_converged = false` until the grid, truncation, bead, and
sampling checks have been completed.

## What is compared

The historical centroid path is the comparison baseline. `bead_weighted`
uses the largest transverse eigenvector of the sampled bead covariance
`Sigma(xi)` and shifts each locked-window centroid by one bead RMS
`sqrt(lambda_max)`, with its sign selected from the local curvature.
`instanton` attempts a folded thermal-instanton half-ring below the crossover
temperature and falls back to the centroid path at or above `T_c` (or when the
optimizer does not return a converged stationary orbit). `least_action` scans
the one-parameter family between the sampled minimum-energy path and its
endpoint chord and records the selected action and all scan points.

| Quantity | Before this change | After this change | Validation status |
| --- | --- | --- | --- |
| Path source | Centroid mean path only | `centroid`, `bead_weighted`, `instanton`, `least_action` | Unit coverage added for selection and diagnostics |
| Bead covariance | Used for curvature-times-width diagnostics | Also provides the `bead_weighted` displacement | RMS and transverse principal-mode checks are covered |
| Thermal instanton | Not used by the subspace pipeline | Folded half-ring below `T_c`; centroid fallback at/above `T_c` | Fallback behavior is unit-tested; production orbit still needs a bead/optimizer study |
| Least-action path | Not available | Recorded alpha scan with minimum-action selection | Scan bookkeeping is unit-tested; production rates remain pending |
| Normal-cut Hessian | Instability count only | Count, grid indices, minimum curvature, and `normal_hessian_positive` | Unstable-point diagnostic is unit-tested |
| Exact quantum rate | Existing truncated-DVR route | Same route for every selected `H_S` | Keep the six-hardest-transverse-quanta wall check before reporting a converged rate |

The path-source unit tests do not replace the model-family benchmark. Reuse
`benchmarks/subspace_theory/transfer.py --family curvature` with
`h=0, 0.3, 0.6, 0.9`, both symmetric and asymmetric barriers, and
`T=600, 300, 200 K`. The read-only workflow
`.github/workflows/subspace-transfer.yml` prints the aggregate `TABLE` rows in
its aggregate job. No new path-source point is marked physically converged in
this note until those rows and the truncation ladder have been recorded.

## Transfer-error baseline

The existing curvature-family transfer records give the small-curvature
reference used to judge a path source. For the symmetric barrier,

\[
\Delta = \ln C(H) - \ln C(H_S) \simeq 0.15(\kappa\rho)^2,
\]

and for the asymmetric barrier,

\[
\Delta \simeq 0.11(\kappa\rho)^2.
\]

The path source changes the sampled path and therefore the measured
`kappa*rho`; it does not change the definition of `C_S` or permit a tolerance
relaxation. The before/after comparison must report both the measured
`kappa*rho` and the resulting `Delta` for each temperature and barrier. Near or
above the crossover, the instanton source is expected to reproduce the
centroid result exactly when the fallback is selected. The bead-weighted and
least-action sources are geometric approximations and must be compared with
the centroid result rather than assumed to be equivalent.

The high-temperature behavior of each source is therefore:

| Source | Behavior for `T > T_c` | Degenerates to centroid? |
| --- | --- | --- |
| `centroid` | Historical path is used | Yes, by definition |
| `bead_weighted` | Covariance shift is still applied when bead moments are available | No, unless the sampled RMS shift is zero |
| `instanton` | The optimizer is skipped and the centroid core is returned with a fallback reason | Yes |
| `least_action` | The alpha scan is still performed on the sampled MEP/chord family | No, unless its selected alpha is zero or the family is degenerate |

## H + H2 at 200 K baseline

The saved three-dimensional BKMP2 run in
`docs/validation/2026-11-09-subspace-transfer/h3-200K/` provides the current
reference:

| Quantity | Existing result |
| --- | ---: |
| Full-system RPMD rate | `7.38e-19 +/- 2.2e-19 cm^3 molecule^-1 s^-1` |
| `C_S(1D)` from the centroid path | `2.06 +/- 0.04` |
| `C_S(2D)` reference correction | `2.66 +/- 0.20` |
| 2D-corrected rate | `1.97e-18 +/- 0.61e-18 cm^3 molecule^-1 s^-1` |
| User-supplied QD reference | `2.01e-18 cm^3 molecule^-1 s^-1` |
| `ln(C_S(2D)/C_S(1D))` | `+0.26 +/- 0.09` |

The numerical before/after table is intentionally explicit about what has and
has not been run in this checkout:

| Path source | Before (`C_S(1D)`) | After (`C_S(1D)`) | Interpretation |
| --- | ---: | ---: | --- |
| `centroid` | `2.06 +/- 0.04` | `2.06 +/- 0.04` expected regression | Historical baseline |
| `bead_weighted` | Not available | Pending restored-run calculation | Compare with the 2D reference after DVR convergence |
| `instanton` | Not available | Pending restored-run calculation | At `T >= T_c`, report the centroid fallback value |
| `least_action` | Not available | Pending restored-run calculation | Retain the complete alpha scan and selected action |
| 2D reference | `2.66 +/- 0.20` | `2.66 +/- 0.20` | Existing comparison target |

The new path sources are intended to reduce the one-dimensional deficit toward
the 2D correction. A new `C_S(1D)` value for `bead_weighted`, `instanton`, or
`least_action` must be generated from the restored
`data/subspace-runs-2026-11` inputs with the same PES and the same `H_S` in the
quantum and RPMD halves. This checkout does not claim those production values
are converged; they are therefore deliberately absent from the numerical
baseline table above.

For every new 200 K result, record the exact-DVR grid, spacing, wall energy,
retained-point count, and `truncation_margin` diagnostic. The wall must remain
at least about six quanta of the hardest transverse vibration above the valley
floor, including the dividing-surface line. A result that has not passed this
check stays `physical_converged = false` even if its statistical error is small.

## Temperature and applicability limits

* At `T >= T_c`, the `instanton` source records `fallback = "centroid"` and
  `fallback_reason = "temperature_at_or_above_crossover"`; it does not call the
  instanton optimizer. This is the required high-temperature degeneration to
  the historical result.
* At `T < T_c`, an instanton is a local stationary-orbit calculation. A failed
  optimizer, a collapsed orbit, an unresolved phase mode, or a path with more
  than one physical negative mode is recorded and falls back to the centroid
  core. This diagnostic is not evidence that the centroid path is physically
  converged.
* `bead_weighted` is the inexpensive local covariance approximation. It can
  fail when locked windows do not contain a reliable covariance, when the
  principal direction is tangent or dominated by rigid-body motion, or when
  the shifted path leaves the PES domain. The pipeline records the covariance
  range and the applied mass-weighted displacement.
* `least_action` is a one-parameter WKB/LAT-style scan. It is a path-selection
  diagnostic, not a replacement for an exact multidimensional quantum rate.
  The alpha scan must be retained so a shallow or non-unique minimum is visible.
* Leaving the MEP does not remove the potential penalty: `project_path_subspace`
  evaluates the complete PES on the selected path and adds the normal-cut
  zero-point (or `free_energy`) term. Non-positive normal Hessians are retained
  in the provenance as unstable grid indices and are not silently labelled
  converged.

The accepted use range is consequently the region where the source-specific
path is smooth, the normal Hessian is positive on the retained grid, the
curvature-times-width transfer criterion is small enough for a one-dimensional
correction, and the exact-DVR truncation ladder is stable. Outside that range,
the report should retain the diagnostics and use the 2D correction or a direct
multidimensional quantum calculation.
