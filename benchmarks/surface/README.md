# Surface accuracy reference

This benchmark computes the finite-bead centroid QTST for the shipped
separable cosine teaching potential without MD, umbrellas, MBAR or histogram
reconstruction. It also checks the package's basin integration against the
classical analytic limit and rejects a persistent oscillatory transmission
curve as a plateau.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python benchmarks/surface/accuracy.py \
  --output runs/surface-accuracy/reference.json
```

NumPy and SciPy are the only additional numerical machinery; both are package
dependencies. Four independently scrambled Sobol sequences are compared at
2^14 and 2^17 points for P=1,2,4,8,16,32,64. P=1 uses the analytic expression.
The output's scramble standard deviation measures numerical integration
variation, not a physical rate uncertainty or a rigorous confidence interval.

For x-direction potential `A/2 * (1-cos(2*pi*x/L))`, split the bead coordinates
into a fixed centroid and internal free-ring Gaussian modes. If `lambda` is a
nonzero eigenvalue of the ring Laplacian, its normal-coordinate variance is
`beta/(m*P*lambda)`. Define `a=beta*A/2` and the bead averages
`C=mean(cos(2*pi*delta/L))`, `S=mean(sin(2*pi*delta/L))`. Analytically integrating
the centroid over the full basin `[-L/2,L/2]` gives

```text
rho(L/2) / integral_basin rho(s) ds
  = E[exp(-a*C)] / (L * E[I0(a*sqrt(C*C+S*S))])

k_QTST = sqrt(1/(2*pi*beta*m)) * density_ratio / atomic_time
```

The transverse and vertical factors cancel because this teaching potential
is separable and its substrate is rigid. The reference assumes the same
primitive finite-P, unwrapped, zero-winding ring ensemble as the example.
It is not an exact quantum dynamical rate or a validation of real catalytic
chemistry. It does not converge the transmission coefficient.

The original pilot audit and the reporting changes are recorded in
[the validation note](../../docs/validation/surface-accuracy-2026-10-31/README.md).
