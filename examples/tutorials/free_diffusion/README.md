# Free centroid diffusion with friction

A free H-mass ring polymer with P=4 at 300 K experiences centroid electronic
friction γ=0.02 fs⁻¹. Its exact long-time diffusion coefficient is
D=kBT/(mγ)=**0.001068939530 bohr²/atomic_time**. This is Langevin centroid
dynamics, not diffusion inferred from thermostatted PIMD sampling time.

Eight independent trajectories equilibrate for 500 fs and run for 10 ps.
The 400–1000 fs MSD fit begins eight friction times after the ballistic regime;
the exact finite-time MSD is 2D[t−(1−exp(−γt))/γ] per Cartesian dimension.
The comparison requires the workflow to accept a diffusive regime and its
replica SEM to satisfy |D−D_ref|≤3 SEM and SEM/D_ref≤15%. Many overlapping
time origins are not counted as independent replicas.

Run from the repository root with an existing PyRPMD installation:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pyrpmd run examples/tutorials/free_diffusion/input.inp \
  --output runs/tutorials/free_diffusion
python examples/tutorials/free_diffusion/compare.py runs/tutorials/free_diffusion
```

Use a fresh output directory. The comparison prints JSON and exits nonzero on
a failed criterion; it never changes the production summary or its convergence
flags. These are analytic model tutorials with explicit sampling budgets.
The budget targets a few minutes on one CPU core, but **wall time and statistical
acceptance have not yet been measured for this input**. The scheduled CI records
wall time, logs and comparison artifacts. A passed 3-SEM check is model agreement,
not a general bead/timestep convergence certification; occasional statistical
rejections must remain visible rather than trigger automatic seed retries.
