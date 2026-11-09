# AQSRPMD / PyRPMD: three 3D **anharmonic** benchmark upgrades

Run everything locally with `bash reproduce.sh` (NumPy, SciPy, Numba,
Matplotlib required; use `OPENBLAS_NUM_THREADS=1`). The script regenerates the
raw numerical data and an independently viewable HTML visual report.

All models use dimensionless units hbar=m_i=k_B=1. Each has an explicit 3D
potential, analytic force, actual quantum calculation, and benchmark checks.

1. **3D coupled Morse**: D=(8,7,6), a=(.65,.70,.75), gamma=(.12,.08,.06),
   beta=6. An independent 3D sinc-DVR Hamiltonian obtains the quantum ground
   energy and x-response. PIMD with an exact finite-P Metropolis target estimates
   centroid-virial energy. RPMD real-time centroid autocorrelation produces a
   separate response reference. The x-only exact vs RPMD defect is transferred
   to the 3D RPMD response.
2. **3D nonlinear Fermi**: V=2x²+y²/2+0.35xy²+0.0203125y⁴
   +1.3²(b-.35x)²/2, beta=6. Exact 3D Fock spectral Kubo x-response and
   a 2D (x,y) reference are compared with independently sampled 3D/2D RPMD
   correlation trajectories. The x y² term is genuinely nonharmonic and
   generates a doublet near 1.96/2.19.
3. **3D curved Eckart**: V=.45 sech²(q)+1.1²[x-.8 tanh(4q)]²/2
   +.015[x-.8 tanh(4q)]⁴+1.6²(z-.12x)²/2. The exact **pure-state** 3D
   split-operator wavepacket has quantum x/z degrees of freedom. It is compared
   against a 2D quantum restriction and a 1D bath-frozen **quantum** control.
   Separately, a centroid-constrained 3D imaginary-time instanton action is
   optimized at beta=35 and compared across bead counts.

**Important**: For cases 1 and 2, the reported percentage errors concern the
Fourier-Laplace **position response**, not a chemical reaction rate. Case 3
reports finite-time *wavepacket-integrated transmission* and an instanton
*action*, not a thermal RPMD reaction rate. None of these benchmark calculations
establishes an exact general 3D hybrid RPMD rate formula. The limited 1D/2D
quantum correction remains a tested **approximation**.

See `three_anharmonic_validation_report.html` for the visual study and
`results/*.json`/`*.npz` for raw results. The main statistical weaknesses are
finite P, bootstrap uncertainty in nonlinear response inversion, Laplace
broadening dependence, and nonzero unabsorbed wavepacket norm.