# BKMP2 H + H2 rate calculation

These inputs select the **published BKMP2 Born–Oppenheimer H3 PES**, not the
analytic `toy_exchange.py` teaching potential. They reuse the verified model in
`benchmarks/h3/pes.py` and the checksum-pinned 1996 Boothroyd–Keogh–Martin–Peterson
Fortran source. See [`../../benchmarks/h3/SOURCE.md`](../../benchmarks/h3/SOURCE.md)
for provenance, citation, energy zero, compiler flags, and source licensing.

From the root of a PyRPMD checkout:

```bash
python -m pip install -e '.[test]'
python benchmarks/h3/build_pes.py             # needs GNU Fortran
python benchmarks/h3/pes.py --check --output runs/bkmp2-pes-check.json
pyrpmd validate examples/rate/bkmp2_pilot.inp --check-pes
pyrpmd run examples/rate/bkmp2_pilot.inp --output runs/bkmp2-pilot
pyrpmd analyze runs/bkmp2-pilot
pyrpmd validate examples/rate/bkmp2_production.inp --check-pes
pyrpmd run examples/rate/bkmp2_production.inp --output runs/bkmp2-production
pyrpmd analyze runs/bkmp2-production
```

No BKMP2 source or binary is redistributed here. The build downloads an
exactly pinned upstream file and refuses a checksum mismatch; for offline
builds use `python benchmarks/h3/build_pes.py --source /path/to/h3_bkmp2.f`
with the identical source bytes.

## Physical / statistical choices

- H3 doublet ground-state PES, three neutral 1H atoms, 1.00782503223 u each;
  geometry is the verified symmetric collinear first-order saddle at
  1.7570000251 bohr adjacent distances, with coordinates explicitly in bohr.
- Both H0 + H1H2 -> H0H1 + H2 and H0 + H1H2 -> H0H2 + H1 channels are included
  via `equivalent_paths`; the symmetry factor stays **1.0**, with **no second
  factor of 2**. All barriers and rates use a consistent electronic surface.
- Entrance separation `Rinf=10 bohr` and reaction coordinate xi from the
  benchmark rate workflow; 231 windows span -0.05 to 1.10 with 0.005 spacing.
  The 150-hartree spring at 1500 K follows `kforce=0.1*T` in the RPMDrate H + H2
  example (beta*k≈31577, sigma_xi≈0.00563). The polymer-bead factor is applied
  internally. This increases work substantially and does not establish convergence.
- PMF uses MBAR; recrossing uses unbiased, unthermostatted paired RPMD children.
  The dividing surface xi* is automatically selected from the maximum of the
  supported MBAR PMF (discrete histogram bin center). Its location must be
  checked for sensitivity to binning, sampling noise and PMF search support.
- `bkmp2_pilot.inp` is a computational smoke test, **not** a publishable rate.
  `bkmp2_production.inp` increases P, reduces timestep, extends PMF and
  recrossing sampling, and increases independent chains/bootstrap; despite
  `preset="production"` it must not be called converged without further checks.

## Mandatory scientific convergence checks

1. Confirm PES oracle energies, gradients, the sole negative saddle mode,
   translation/rotation/permutation invariance and the build SHA using
   `benchmarks/h3/pes.py --check` before propagation.
2. Compare P=16/32/64 and dt=0.1/0.05/0.025 fs while holding the rest fixed;
   confirm the unbiased child energy drift is negligible and estimates are
   stable to statistical error. Do not combine different settings by resume.
3. Check adjacent-window overlap connectivity, per-window ESS, window-by-window
   equilibration, block length and PMF/MBAR vs flux-weighted UI consistency.
   Revisit width/stiffness if some windows have no overlap. The automatic
   xi* is the largest supported PMF bin center, not a continuous optimum. Establish stability
   in sampling length and at independent random seeds.
4. Check `Rinf=10` versus larger separation (e.g. 12 or 14 bohr) and PMF
   entrance normalization; verify the PMF maximum and xi* choice consistently.
5. Increase constrained parent count and +/- child pairs/duration until
   kappa(t) has a plateau and its parent-cluster confidence interval and rate
   are stable. Check endpoint energy drift and the full rate's uncertainty.

If any physical, time-plateau, energy-drift or PMF-support gate fails, treat the
output as diagnostic rather than a converged thermal rate. See
`benchmarks/h3/README.md`, `benchmarks/h3/RATE_POSTPROCESS.md` and
`docs/rate-input-v2.md` for the validated implementation and conventions.

The original `research.inp`, `pilot.inp`, `minimal.inp`, etc. are intentionally
left as analytic *teaching and regression* cases, because the repository tests
assert their synthetic saddle and sampling budgets. Do not use them to report
BKMP2 rates. `general_coordinate.inp` is a separate, non-H3 site-rate test.
