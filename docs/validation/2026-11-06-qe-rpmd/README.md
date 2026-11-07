# QE-RPMD validation record (2026-11-06)

This branch integrates the first reproducible QE-RPMD components into PyRPMD:

- `pyrpmd.analysis.dvr_heom.DVRSystemHEOM` propagates a one-dimensional
  system coordinate on a uniform DVR grid for an arbitrary diagonal
  potential, diagonal coupling `f(s)`, optional counterterm, and a finite
  multi-exponential bath correlation `C(t) = sum_k c_k exp(-nu_k t)`.
- `pyrpmd.analysis.spectral_density` implements the fixed convention
  `J(w) = beta*w*integral C_K(t) cos(w*t) dt`, reconstructs the complex
  quantum correlation, and fits it to a deterministic finite exponential
  expansion while reporting absolute, relative, and RMS residuals.
- The new tests compare the generic DVR equations against the existing sparse
  `SpinBosonHEOM` implementation and exercise the analytic harmonic/Drude
  Kubo fixture.

## Reproducibility settings

The exact-regression test uses

- `epsilon=0.4`, `delta=0.8`, `lambda=0.12`, `cutoff=1.3`,
  `beta=1.7`, `coupling_scale=0.65`;
- Matsubara terms `K=1`, hierarchy depth `2`, no terminator;
- 31 output points on `[0, 3]`, `DOP853`, `rtol=2e-11`, `atol=2e-13`;
- a fixed Hermitian initial density
  `[[0.7, 0.15j], [-0.15j, 0.3]]`.

The harmonic recovery fixture uses the analytic Kubo correlation
`C_K(t)=2*lambda/beta*exp(-cutoff*t)`, 10,001 samples on `[0,100]`, and
1,200 frequencies on `[0.02,8]`. The measured maximum relative spectral-density
error is required to be below 1%.

Run the focused checks with:

```bash
python -m pytest tests/test_heom.py tests/test_qe_rpmd_exact.py
```

## Scope and limits

This is the P1/P2 integration and exact-interface regression step. It does not
claim that the B1 double-well benchmark, B2 anharmonic-bath error decomposition,
or P5 diagnostic correlations have been completed. Those require explicit
discrete-bath reference calculations and separate convergence tables for DVR
grid, path-bead count, hierarchy depth, exponential count, and trajectory
statistics.

The generic solver intentionally keeps a dense hierarchy and is intended for
small validation systems. Production-scale HEOM should continue to use the
existing sparse `SpinBosonHEOM` implementation.
