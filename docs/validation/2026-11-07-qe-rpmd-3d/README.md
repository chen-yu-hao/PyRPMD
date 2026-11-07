# Three-dimensional quantum-correction stress test (2026-11-07)

This record exercises the response-level quantum-correction workflow on a
coupled three-dimensional harmonic oscillator. It is deliberately a stress
test for the transfer approximation, rather than a claim of a general exact
three-dimensional rate correction.

## Model and reference calculations

The Cartesian force-constant matrix is

```text
[[1.00, 0.12, 0.05],
 [0.12, 1.40, 0.08],
 [0.05, 0.08, 1.80]]
```

with `beta=2`. Its normal-mode frequencies are

```text
[0.9827921209975772, 1.188372998825422, 1.3497737819974716]
```

The quantum reference uses a finite spectral Kubo sum in an `8^3=512` state
normal-mode Fock basis. The estimated omitted thermal weight is
`1.5414705423211217e-07`. The RPMD reference is the analytic classical
Gaussian correlation for quadratic observables. The full observable is
`x^2+y^2+z^2`; the transferred correction uses the intermediate subspace
`x^2`.

The calculation uses 601 points on `[0,30]`, 180 frequencies on `[0.03,3.5]`,
and Fourier--Laplace damping `0.08`.

Run it with:

```bash
PYTHONPATH=src python examples/three_d_quantum_correction.py
```

## Results

```text
C_Q,sub(0)    = 0.520704194033574
C_RPMD,sub(0) = 0.5115176460770439
C_Q,full(0)   = 0.9632595493732827
C_RPMD,full(0)= 0.937286538167585

inverse_defect_rms                  = 0.4324382273517201
relative_inverse_defect             = 0.18867639129385422
negative_real_fraction_full         = 0.3409804291032106
negative_real_fraction_corrected    = 0.30320354130400035
corrected_condition_number_max      = 1.0
corrected_effective_rank_min        = 1
```

Transferring the `x^2` quantum-minus-RPMD defect to the full
`x^2+y^2+z^2` response gives a relative RMS difference of
`1.0554854999667558` against the independent full-system quantum response.
This large error is expected for an intermediate subspace and is recorded as a
boundary diagnostic for the approximation.

The algebraic full-subspace check uses the full observable in all three inputs.
It gives a maximum absolute response error of
`1.9860273225978185e-15`, while the inverse defect itself remains nonzero
(`0.22193679762120636`). The exact limit comes from cancellation in the same
response space, not from the defect being zero.

The result is intentionally marked `physical_converged=False`. The finite Fock
truncation, time window, frequency grid and response regularization must still
be varied before interpreting the intermediate-subspace result physically.
