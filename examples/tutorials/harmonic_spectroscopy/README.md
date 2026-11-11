# Harmonic TRPMD infrared frequency

The model has k=0.05 hartree/bohr² and mass 1837.15264744 electron masses.
Its exact angular frequency is √(k/m), giving **1144.976858 cm⁻¹**.
TRPMD thermostats only internal ring modes during production. For a harmonic
well and a linear dipole, the centroid frequency is the exact reference;
the borrowed toy property module is not a molecular IR intensity model.

Four replicas each equilibrate for 500 fs and record 4 ps of dynamics at
0.2 fs cadence. The saved spectrum uses a Hann window. compare.py reports
replica peak SEM and the deterministic half-bin rounding bound separately.
It accepts |peak−reference|≤3 SEM+Δν/2, with Δν taken directly from the saved
FFT grid, never fitted to the observed discrepancy. Zero padding is not an
improvement in physical frequency resolution. Identical replica peak bins
can give SEM=0; the separately reported grid bound then remains relevant.

Run from the repository root with an existing PyRPMD installation:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pyrpmd run examples/tutorials/harmonic_spectroscopy/input.inp \
  --output runs/tutorials/harmonic_spectroscopy
python examples/tutorials/harmonic_spectroscopy/compare.py runs/tutorials/harmonic_spectroscopy
```

Use a fresh output directory. The comparison prints JSON and exits nonzero on
a failed criterion; it never changes the production summary or its convergence
flags. These are analytic model tutorials with explicit sampling budgets.
The budget targets a few minutes on one CPU core, but **wall time and statistical
acceptance have not yet been measured for this input**. The scheduled CI records
wall time, logs and comparison artifacts. A passed 3-SEM check is model agreement,
not a general bead/timestep convergence certification; occasional statistical
rejections must remain visible rather than trigger automatic seed retries.
