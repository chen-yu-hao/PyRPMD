# Classical directed periodic escape

One H-mass particle moves in V(x)=0.006[1−cos(2πx/L)]/2 hartree,
L=4 bohr, at 300 K with P=1. The directed classical TST rate is
k=exp(−βh)/[L exp(−βh/2) I₀(βh/2) √(2πβm)] in atomic units,
or **2.271294731×10¹⁰ s⁻¹**. This is one exit direction; there is no factor of two.
For this separable 1D barrier, forward classical trajectories have κ=1 over
the short recrossing observation before crossing the next dividing surface.

The tutorial uses 13 windows, four replicas, 4 ps production per window,
96 parent configurations and 192 child pairs. Its reported full rate SEM must
include PMF and recrossing uncertainty. A missing supported PMF, plateau, or
full uncertainty fails comparison. Acceptance requires |k−k_ref|≤3 SEM and
SEM/k_ref≤15%; a conditional κ-only error cannot replace the full rate error.

Run from the repository root with an existing PyRPMD installation:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pyrpmd run examples/tutorials/periodic_site_rate/input.inp \
  --output runs/tutorials/periodic_site_rate
python examples/tutorials/periodic_site_rate/compare.py runs/tutorials/periodic_site_rate
```

Use a fresh output directory. The comparison prints JSON and exits nonzero on
a failed criterion; it never changes the production summary or its convergence
flags. These are analytic model tutorials with explicit sampling budgets.
The budget targets a few minutes on one CPU core, but **wall time and statistical
acceptance have not yet been measured for this input**. The scheduled CI records
wall time, logs and comparison artifacts. A passed 3-SEM check is model agreement,
not a general bead/timestep convergence certification; occasional statistical
rejections must remain visible rather than trigger automatic seed retries.
