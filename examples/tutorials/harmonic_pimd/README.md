# Finite-bead harmonic PIMD

One H-mass particle in V(q)=0.02 |q|²/2 hartree, at 300 K with P=16.
The exact primitive finite-P references are ⟨V⟩=K_cv=0.002618793760 hartree
and Rg²=0.1193728556 bohr². With ω²=k/m and ω_j=2P sin(πj/P)/β,
⟨|q|²⟩=3 Σ_j[βm(ω²+ω_j²)]⁻¹ and Rg²=⟨|q|²⟩−3/(βk).
The infinite-P quantum oscillator value is a different reference.

Four independent replicas each equilibrate for 2 ps and sample for 40 ps using
native PILE-L. The comparison uses the actual SEM across replica means; Rg²
uses the exact per-frame harmonic identity K_cv=3/(2β)+(k/2)Rg² to propagate
its SEM. Each error must be within 3 SEM, and each SEM must be below 3% of the
reference. No fixed error bar is invented before running.

Run from the repository root with an existing PyRPMD installation:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pyrpmd run examples/tutorials/harmonic_pimd/input.inp \
  --output runs/tutorials/harmonic_pimd
python examples/tutorials/harmonic_pimd/compare.py runs/tutorials/harmonic_pimd
```

Use a fresh output directory. The comparison prints JSON and exits nonzero on
a failed criterion; it never changes the production summary or its convergence
flags. These are analytic model tutorials with explicit sampling budgets.
The budget targets a few minutes on one CPU core, but **wall time and statistical
acceptance have not yet been measured for this input**. The scheduled CI records
wall time, logs and comparison artifacts. A passed 3-SEM check is model agreement,
not a general bead/timestep convergence certification; occasional statistical
rejections must remain visible rather than trigger automatic seed retries.
