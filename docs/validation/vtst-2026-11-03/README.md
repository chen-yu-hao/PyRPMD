# T31 VTST/SCT validation record — 2026-11-03

## Scope

This record covers deterministic algorithm checks performed while implementing
`pyrpmd.vtst`. It is not a production H + H2/BKMP2 validation and does not claim
agreement with POLYRATE. `physical_converged` remains false.

## Environment

- Linux 6.18.57 x86_64, KVM
- AMD EPYC 9V74 virtual CPU allocation, 5 visible single-threaded cores
- Python 3.13.5
- NumPy 2.3.5
- SciPy 1.17.0
- `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`

The isolated harness used the task implementation plus minimal package stubs
because the full repository could not be cloned in this execution environment.
The pull-request CI is the authoritative full-package result.

## Unit and independent-reference checks

Command:

```text
PYTHONPATH=/mnt/data/t31_work/src:/mnt/data/t31_work \
  python -m pytest -q /mnt/data/t31_work/tests/test_vtst.py \
  -k "not reactant_partition_matches_low_temperature_harmonic_state"
```

Observed output:

```text
.....                                                                    [100%]
5 passed, 1 deselected in 3.61s
```

The isolated checks cover:

- bidirectional mass-scaled MEP integration on an analytic separable saddle;
- Cartesian projection and a symmetric CVT free-energy profile;
- the exact numerical limit SCT = ZCT for a straight path;
- a sampled Eckart barrier whose WKB action is compared with an independent
  continuous SciPy quadrature; and
- workflow publication, scientific-fingerprint sensitivity, and CLI dispatch.

The full-repository test additionally compares the reactant translational,
rotational, vibrational, and total partition logs directly with
`pyrpmd.low_temperature.HarmonicState`; that module was not duplicated into the
isolated harness.

## MEP-step convergence on the analytic separable path

The same analytic potential, endpoint tolerance, reactant partition, 500 K
rate, and 32-point thermal quadrature were used. Only the requested maximum MEP
step changed.

| MEP step (bohr) | Path points | CVT rate (s^-1) | Relative change from previous | MEP time (s) | CVT/ZCT/SCT time (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.08 | 45 | 1.2722044453e9 | — | 0.0202 | 1.0403 |
| 0.04 | 61 | 1.2722044525e9 | 5.66e-9 | 0.0214 | 1.0885 |
| 0.02 | 99 | 1.2722044463e9 | 4.84e-9 | 0.0410 | 1.6422 |

Both path sides met the analytic force criterion at every resolution. This is a
numerical regression and step-convergence check for the implementation, not a
physical convergence result.

## Straight-path tunnelling limit

For a synthetic collinear H3 path with constant tangent and locally constructed
positive transverse Hessians, the largest evaluated curvature was below
`1e-11 bohr^-1`. Every SCT effective-mass ratio was exactly `1.0` in binary
floating-point, and ZCT/SCT thermal coefficients and corrected rates were equal
for 300 K and 500 K.

## Missing production validation

A verified 200-1000 K H + H2/BKMP2 comparison table was not available in the
repository inputs and no external POLYRATE executable or independently checked
published table was used. No agreement claim is made. That campaign remains a
separate high-cost validation item; this PR supplies the code path, transparent
approximations, and reproducible small-system references needed to run it.
