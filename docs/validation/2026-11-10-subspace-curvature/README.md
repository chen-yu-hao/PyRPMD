# SCT curvature mass for one-dimensional path subspaces

Date: 2026-11-10. This validation records the implementation for issue #155,
which extends `project_path_subspace` with `curvature="small"`. The baseline
transfer data are the converged records in
[`2026-11-09-subspace-transfer`](../2026-11-09-subspace-transfer/README.md).

## What changed

The path is first parameterized by mass-weighted arc length `sigma`. At every
requested path point, the curvature normal is projected through the normal-cut
Hessian to obtain `omega_perp`, with `tbar = sqrt(hbar / omega_perp)` and
`abar = kappa * tbar`. The SCT metric is

`mu_eff / mu = exp(-2 abar - abar**2 + (d tbar / d sigma)**2)`.

For `abar > 1`, the value used in the metric is clipped to one. The model
provenance reports `max_abar`, `curvature_truncated_fraction`, the pointwise
`curvature_mass_ratio`, and the sigma-to-tau arrays. The transformed coordinate
`d tau = sqrt(mu_eff / mu) d sigma` is resampled on a uniform tau grid with
unit mass. The exact DVR and RPMD calculations therefore receive the same
transformed `SubspaceModel` and the same potential. `curvature="none"` is an
identity path and retains the previous grid and values.

## Unit-test checks

The new tests in `tests/test_subspace_correction.py` cover:

- a straight valley: `curvature="small"` and `"none"` have bitwise-identical
  axes and values;
- a circular valley: the pointwise effective mass agrees with
  `exp(-2 a - a^2)`, where `a = 1 / (R sqrt(omega))`, and the tau span equals
  the trapezoidal integral of `sqrt(mu_eff / mu)`;
- `abar > 1`: a `RuntimeWarning` is emitted and the clipped-point fraction is
  recorded; invalid curvature modes are rejected.

For the circular test (`R = 10`, `omega = 0.2`), the analytic ratio is
`0.6082` and the transformed length is `0.7799` times the sigma length. The
large-curvature test reports the clipping fraction rather than treating the
point as physically converged.

## Transfer benchmark: before and after

The Actions matrix now runs both `curvature_mass="none"` and
`curvature_mass="small"` for the curvature family. The existing converged
baseline is reproduced below. The SCT columns are filled by the aggregate
workflow artifact after the `issue-155-curvature` run; until that run is
available they are explicitly marked pending rather than inferred from the
old model.

| barrier | h (bohr) | kappa rho (200 K) | Delta 600 K, baseline | Delta 600 K, SCT | Delta 300 K, baseline | Delta 300 K, SCT | Delta 200 K, baseline | Delta 200 K, SCT |
| --- | ---: | ---: | ---: | --- | ---: | --- | ---: | --- |
| symmetric | 0.0 | 0.00 | +0.008 | pending | −0.015 | pending | +0.004 | pending |
| symmetric | 0.3 | 0.40 | +0.022 | pending | +0.024 | pending | +0.028 | pending |
| symmetric | 0.6 | 0.79 | +0.078 | pending | +0.111 | pending | +0.068 | pending |
| symmetric | 0.9 | 1.19 | +0.163 | pending | +0.156 | pending | +0.162 | pending |
| asymmetric | 0.0 | 0.00 | −0.004 | pending | +0.004 | pending | +0.062 | pending |
| asymmetric | 0.3 | 0.40 | +0.030 | pending | +0.015 | pending | +0.065 | pending |
| asymmetric | 0.6 | 0.79 | +0.062 | pending | +0.079 | pending | +0.061 | pending |
| asymmetric | 0.9 | 1.19 | +0.129 | pending | +0.124 | pending | +0.066 | pending |

The baseline values are the zero-point projection from
`curvature_points.json`; the quoted fit is
`Delta = 0.148 (kappa rho)^2` for symmetric barriers and
`0.112 (kappa rho)^2` for asymmetric barriers. The SCT validity diagnostic is
`abar`, not the bead-width `kappa rho`: `abar > 1` is the large-curvature
region and must be reported as clipped. The empirical pre-SCT transfer study
already shows that `kappa rho ~ 0.6` is the boundary for about 5% transfer
error, while `kappa rho = 1.19` is outside that regime.

The exact calculations retain the existing convergence requirements: the DVR
wall is at least about six stiff transverse quanta above the valley floor,
and the report must include `truncation_margin`, grid spacing and retained
points. No result in this validation is marked `physical_converged`.

## H + H2 at 200 K

The converged PR #159 baseline is:

| quantity | baseline |
| --- | ---: |
| `C_S(1D)` | `2.06 +/- 0.04` |
| `C_S(2D)` | `2.66 +/- 0.20` |
| corrected rate using 2D | `1.97e-18 +/- 0.61e-18 cm^3 s^-1` |
| QD reference | `2.01e-18 cm^3 s^-1` |

The reproducible SCT input is
[`correct_200K_curvature.toml`](../../../benchmarks/h3_subspace/correct_200K_curvature.toml).
It restores the raw run from `data/subspace-runs-2026-11` and writes a fresh
`subspace-sct-1d` directory with `reuse = false`, so the one-dimensional
`C_S` is recomputed from the transformed model. In this checkout the BKMP2
shared library required by `benchmarks/h3/pes.py` is not present, so that
recalculation could not be executed locally; the baseline above is retained
without claiming it is the SCT result. The Actions/validation artifact should
replace the pending value with the new `C_S(1D)` and its provenance, including
`max_abar`, `curvature_truncated_fraction`, DVR `truncation_margin`, grid
spacing and `physical_converged = false`.

## Reproduction

```bash
# restore the raw run as described by data/subspace-runs-2026-11/README.md
pyrpmd subspace-correct benchmarks/h3_subspace/correct_200K_curvature.toml
python tools/check_workflow_policy.py
PYTHONPATH=src python -m pytest -q tests/test_subspace_correction.py tests/test_subspace_transfer.py
```
