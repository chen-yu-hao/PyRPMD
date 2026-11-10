# Frozen well-tempered metadynamics

The P=1 potential is V(x)=0.004(x²−1)² hartree, with independent transverse
harmonic confinement. The exact continuous barrier is 0.004 hartree,
or 4.21033 kBT at 300 K. Four independent well-tempered bias histories deposit
for 10 ps, freeze, relax for 2 ps, then retain 18 ps of production each.
Only samples after the freeze and relaxation enter reweighted histograms.

The comparison evaluates F(0)−[F(−1)+F(1)]/2 for every replica. Its reference
integrates exp(−βV) over the exact saved bins before taking −log/β, so finite
bin averaging is explicit rather than mistaken for sampling error. All
three locations must be supported in every history. Acceptance requires
agreement within three SEM across the four barriers and SEM below 15% of the
reference. Replica standard deviation is divided by √4; it is not called SEM
without that conversion.

Run from the repository root with an existing PyRPMD installation:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pyrpmd run examples/tutorials/double_well_metad/input.inp \
  --output runs/tutorials/double_well_metad
python examples/tutorials/double_well_metad/compare.py runs/tutorials/double_well_metad
```

Use a fresh output directory. The comparison prints JSON and exits nonzero on
a failed criterion; it never changes the production summary or its convergence
flags. These are analytic model tutorials with explicit sampling budgets.
The budget targets a few minutes on one CPU core, but **wall time and statistical
acceptance have not yet been measured for this input**. The scheduled CI records
wall time, logs and comparison artifacts. A passed 3-SEM check is model agreement,
not a general bead/timestep convergence certification; occasional statistical
rejections must remain visible rather than trigger automatic seed retries.
