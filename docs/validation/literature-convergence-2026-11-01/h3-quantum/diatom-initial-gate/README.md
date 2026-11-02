# Initial independent diatom check

The first execution is retained here. Its combined `numerical_converged` flag
was false because moving the radial hard wall from 0.4 to 0.6 bohr changed the
log partition ratio by 5.95e-7, larger than an initial 1e-7 threshold. The
256-to-384 grid change was 5.72e-13, and the wall change to the physical
0.5-bohr bead cutoff was 1.32e-9. No potential, mass, temperature, eigenvalue,
or numerical result was adjusted. The current report separates convergence
for a fixed Hamiltonian from a stress test that changes its hard wall. The
original all-wall criterion remains reported and remains false.

The first analytic harmonic test also incorrectly demanded spectral accuracy
for every angular momentum at a finite sine grid. A radial wavefunction with
odd J does not have a smooth odd continuation at zero, so the centrifugal
endpoint gives algebraic error (J=1 is dominant at approximately h^3). The
current test compares against the same exact oscillator spectrum and checks
this refinement explicitly. Even-J levels retain the strict absolute test.
The H2 calculation has a positive radial lower bound and converges spectrally.
