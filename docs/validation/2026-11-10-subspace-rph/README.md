# Curved-path zero-point projections (issue #156)

Date: 2026-11-10. This note records the implementation and validation status
of the three `project_path_subspace` zero-point choices. It is diagnostic; no
rate in this directory is marked `physical_converged`.

## Change under test

`zero_point="harmonic"` is the pre-issue normal-cut projected Hessian. The
new `"rph"` choice forms the mass-weighted Miller--Handy--Adams reaction-path
force constants and removes the local force--curvature connection before
diagonalisation. In a normal basis (N), with (P=NN^T),

\[
 K_{\rm RPH}=N^TKN-\tfrac12\{(N^Tg)(N^T\kappa)^T
                    +(N^T\kappa)(N^Tg)^T\},
\]

where (g=\nabla V/\sqrt m) and \(\kappa=d\hat t/d\xi\). In codimension one
this is (K_{nn}-(g\cdot n)(\kappa\cdot n)), the term specified by the
Miller--Handy--Adams expansion. `"dvr"` diagonalises each local normal cut
with a Colbert--Miller sinc-DVR (81 points and eight oscillator lengths by
default); it takes the cut ground level, or the cut thermal free energy for
`environment="free_energy"`.

## Unit and analytic checks

The new `tests/test_subspace_model_zero_point.py` has seven tests. A straight
separable harmonic tube gives the same (V_S) for all three choices. The DVR
normal-cut zero-point term agrees with \(\hbar\omega/2\) to below (10^{-6})
(the direct check is (2.8\times10^{-16}) in the local run). A circular
analytic path with radial force (F=0.10), radius (R=5), and physical
frequency (0.20) gives

| projection | normal zero-point | RPH correction |
| --- | ---: | ---: |
| harmonic | 0.07071 | 0 |
| rph | 0.10000 | (F/R=0.02000) |

This check isolates the gradient-times-curvature term and does not assert that
an RPH one-dimensional model contains corner-cutting tunnelling.

## Curvature-family rate matrix

The read-only workflow now carries `zero_point` in the matrix and runs all
three choices for the curvature family (`h=0, 0.3, 0.6, 0.9`, symmetric and
asymmetric barriers, 600/300/200 K). The branch filter includes
`issue-156-rph-dvr`; `tools/check_workflow_policy.py` reports `workflow policy:
OK`. The aggregate job prints `zero_point`, (C_S), and
\(\Delta=\ln C(H)-\ln C_S\) for every row.

The existing Actions run (harmonic projection, 600 K, symmetric barrier) is
the before curve:

| h | kappa*rho | ln C_S(1D) | Delta |
| ---: | ---: | ---: | ---: |
| 0.0 | 0.00 | +0.102 | +0.008 |
| 0.3 | 0.40 | +0.094 | +0.022 |
| 0.6 | 0.79 | +0.045 | +0.078 |
| 0.9 | 1.19 | −0.016 | +0.163 |

The corresponding full-system (C(H)) values are 1.116, 1.123, 1.130, and
1.159. Thus the historical normal-cut model loses 0.12 in (ln C_S) while
(ln C(H)) rises by only 0.04.

The three curves are intentionally produced by the workflow rather than
substituted with a low-statistics local run. A local pilot at 600 K, using a
coarse 2.0 arc-length grid and 8 walkers × 40 sweeps × 64 trajectories, was
used only as a smoke test; it is not a converged rate and is not used to claim
the target. Its (h=0.9), spacing-1.0 values were

| projection | C_S(1D) | ln C_S | Delta at kappa*rho=1.19 using C(H)=1.159 |
| --- | ---: | ---: | ---: |
| harmonic | 0.973 | −0.027 | +0.172 |
| rph | 0.976 | −0.024 | +0.172 |
| dvr | 1.009 | +0.009 | +0.139 |

The pilot confirms that the DVR removes part of the shoulder-induced trend,
but it does not satisfy the convergence gate. The Actions artifacts are the
required source for the final \(\ln C_S\)-versus-(h) and
\(\Delta(\kappa\rho)\) comparison; statistical errors, spacing ladders, and
the energy truncation margin must be read from those `TABLE` rows before
accepting a physical result.

## Exact-rate and truncation checks

The exact solver is unchanged and still reports `truncation_margin`. For the
2D transfer matrix the existing accepted wall is 0.07 Eh above the barrier plus
the largest transverse zero-point term (about six stiff-mode quanta); the
0.10-Eh check changes the two 200 K curvature points by at most the previously
recorded 0.002 in \(\Delta\). The new normal-cut DVR has explicit grid
controls (`dvr_points`, `dvr_half_width`, `dvr_spacing`) in model provenance.
The harmonic analytic test was repeated with 81 points and eight oscillator
lengths; changing the cut grid is a required check before a rate is called
converged.

## H + H₂, 200 K

The restored data branch `data/subspace-runs-2026-11` contains the raw run and
the previous calculation reports (C_S(1D)=2.06\pm0.04) and
\(C_S(2D)=2.66\pm0.20\), giving (1.97\times10^{-18}) cm³/s against the
2.01×10⁻¹⁸ reference. Recomputing this saved run in the current container was
blocked before the projection stage because the provenance points to the
checksum-pinned BKMP2 plugin (`benchmarks/h3/pes.py`) and GNU Fortran is not
installed. No replacement number is presented as converged.

The RPH/DVR choice changes only the one-dimensional normal environment term;
the curvature-width test and the 2D curvature-normal plane remain separate.
They can be combined multiplicatively only after the nested (C_S(2D)/C_S(1D))
sensitivity and the DVR/Hessian grid checks are rerun on the same saved PES.

## Applicability and failure modes

The RPH correction is a local harmonic connection correction. It does not
restore a transverse coordinate when \(\kappa\rho\gtrsim1\), and it can leave
negative normal curvatures (recorded in provenance). The DVR option handles
anharmonic normal cuts but still uses independent local normal modes; strong
mode mixing or a path that exits the PES domain requires a higher-dimensional
subspace. All rate outputs and this report retain `physical_converged=false`
until the spacing, cut grid, absorber, truncation margin, bead number, and RPMD
sampling gates pass.
