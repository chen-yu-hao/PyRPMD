# Reactive D + H₂ exchange on an analytic teaching PES

This example closes the gap between free-scattering mechanics examples and a
reactive dRPMD calculation. The isotope labels allow the built-in connectivity
classifier to distinguish D + H₂ from DH + H. The PES is the analytic function
in `examples/rate/toy_exchange.py`, with depth **0.15 hartree**, coupling
**0.36 hartree**, bending **0.1 hartree**, α=1.2 bohr⁻¹ and r₀=1.2 bohr. It is
not a fitted physical H₃ surface. `exchange.c` implements the same energy and
its analytic Cartesian gradient through the existing C ABI.

Both methods use **1500 K, P=4, dt=0.1 fs and R∞=9 bohr**. dRPMD samples incident
energy from Gamma(2,kBT), impact parameter uniformly in area to 4.5 bohr, and
isotropic orientations. The 32 independent parent chains each equilibrate for
1 ps and advance 20 fs between collision launches. The 4096-trajectory budget
targets a few minutes on one core; its measured time and statistical acceptance
must be recorded when run. Check parent correlation, bmax, entrance/exit
separation, time limit, bead count and timestep before quantitative use.

Build with the existing C compiler and the repository's ABI header, then run:

```sh
python examples/tutorials/exchange_rate/build.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pyrpmd run examples/tutorials/exchange_rate/drpmd.inp \
  --output runs/tutorials/exchange-drpmd
python examples/tutorials/exchange_rate/compare.py runs/tutorials/exchange-drpmd
```

`drpmd-short.inp` keeps the same model and parent equilibration but uses 8 parents
and 256 collisions. Its `--smoke` comparison requires at least one reactive
collision and no failed/unresolved collisions. It demonstrates execution;
it does not assert a converged rate. A deterministic test also launches a
collinear above-barrier collision and checks actual DH formation, outward
separation and native step counters.

`rate.inp` is the longer Bennett–Chandler counterpart. It uses two independent
chains per window, 1 ps equilibration and 10 ps production. The k/T force
constants are 0.0016 and 0.0032, giving thermal ξ widths about 0.0445 and 0.0315,
with center spacings 0.02 and 0.01. It fixes ξ‡=1 and uses 1000 constrained
parents with 10 child trajectories each, propagated for 100 fs. This is a
reference-budget input, not a promise of a few-minute run. To compare fresh
calculations, run it in a new directory and pass `--bc-run RUN` to `compare.py`.
An MBAR comparison can use a fresh copy with `rate.pmf.estimator="mbar"`.

The following values are **reported in [issue #170](https://github.com/chen-yu-hao/PyRPMD/issues/170)**,
not new results from these inputs:

| Method | Rate / cm³ molecule⁻¹ s⁻¹ |
| --- | --- |
| dRPMD, 20000 trajectories / 211 reactive | (1.055 ± 0.069) × 10⁻¹¹ |
| MBAR BC, same soft-window samples | (1.061 ± 0.090) × 10⁻¹¹ |
| UI BC, ten times stiffer windows | (1.142 ± 0.051) × 10⁻¹¹ |

Without `--bc-run`, the comparator explicitly labels the last row as its
historical reference. Quantitative comparison requires at least 20 reactions,
8 independent parents, complete outcomes and a non-null dRPMD rate. It reads
the trajectory table and resamples entire parent chains to estimate the SEM;
it never uses an IID interval as a parent-chain error estimate. A supplied BC
run must have a non-null final rate and full PMF plus recrossing uncertainty.
Acceptance is agreement within three combined standard errors. Insufficient
sampling fails visibly; it never retries seeds or changes result files.
