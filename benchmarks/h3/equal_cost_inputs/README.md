# BKMP2 H + H₂: equal MD-step fixed vs adaptive **full RPMD rates**

**Status: runnable experiment definition; NOT a completed numerical result.**
This campaign compares 96 fixed umbrella windows with an adaptive scheme
starting from six windows. Both use the true checksum-pinned BKMP2 surface,
the RPMDrate centroid `xi` coordinate and both symmetry-equivalent exchange
branches. It does not use `toy_exchange.py`.

## Run from the repository root

```bash
python -m pip install -e '.[test]' matplotlib
python benchmarks/h3/build_pes.py
python benchmarks/h3/pes.py --check --output runs/h3-bkmp2-pes-check.json

pyrpmd validate benchmarks/h3/equal_cost_inputs/fixed96.inp --check-pes
pyrpmd validate benchmarks/h3/equal_cost_inputs/adaptive.inp --check-pes

pyrpmd run benchmarks/h3/equal_cost_inputs/fixed96.inp \
  --output runs/h3-equal-fixed96
pyrpmd run benchmarks/h3/equal_cost_inputs/adaptive.inp \
  --output runs/h3-equal-adaptive

python benchmarks/h3/equal_cost_compare.py \
  --fixed runs/h3-equal-fixed96 \
  --adaptive runs/h3-equal-adaptive \
  --output runs/h3-equal-comparison
```

The analyzer refuses missing or unsupported PMFs rather than replacing them
with assumed values. It produces `comparison.json`, `comparison.csv`,
`pmf_mbar_ui.png`, and `mbar_minus_ui.png`. Rate estimates are only reported
if their own recrossing child trajectories ran successfully; the final
`accepted_rate_cm3_molecule_s` is null unless the quality gates pass.

## Strictly specified cost accounting

- T=1500 K, P=16, dt=0.1 fs, Rinf=10 bohr; identical H masses,
  spin surface, flux metric, Andersen sampling and equivalent pathways.
- Both cover xi from -0.05 through 1.10, K=400 kBT, two replicas, and 120
  MBAR bins. The fixed run uses 96 evenly spaced window centers.
- **Fixed PMF step budget:** 96 windows x 2 streams x
  (3000 equilibration + 10000 production steps) = **2,496,000**.
- **Adaptive PMF step budget:** the same 2,496,000 **including all pilot
  equilibration and pilot production**. After the adaptive plan freezes,
  `total_integration_steps` resolves the per-stream frozen production length
  from (total - all discarded pilots - frozen equilibration)/frozen streams.
  The saved sample stride can leave a small explicitly reported unspent
  remainder (must be <= 0.1%). If the pilot consumes the target, the run fails.
  The production RNG is independent of discarded pilot samples.
- **Recrossing budgets:** the same 2 constrained parent chains, 32
  configurations/chain, 2 plus-minus child pairs/configuration, 80 fs child
  duration, 0.1 fs integrator, and 100 parent-cluster bootstrap attempts.
  Nominally 10,400 parent steps + 204,800 child steps = **215,200** per method,
  subject to real constraint retries and failed children. Audit actual
  trajectories and runtime rather than assuming perfect equality.
- This equality targets **nominal MD integration steps**; adaptive placement,
  MBAR fits, and constraint retries may incur unequal wall time or PES force
  calls. It is not a wall-clock benchmark.

## Estimator protocol, xi* and statistical comparisons

Both calculate histogram MBAR as the **primary** rate-flux PMF and Gaussian
umbrella integration (UI) on the **same frozen production trajectories**.
Within each method, xi* is automatically selected from the supported MBAR PMF
maximum in the *predeclared* search interval `[0.75, 1.08]`. UI is evaluated
at that same xi*, so its barrier difference is paired to MBAR without shifting
the rate dividing surface to make either estimator look better.

The two sampling methods can select **different xi***; each constrained
recrossing calculation then uses its own MBAR-selected surface and identical
trajectory budgets. Compare: (1) xi* locations, (2) two estimators' barriers
and paired MBAR-minus-UI bootstrap errors, (3) overlap graph and MBAR residual,
(4) per-stream ESS, (5) kappa plateau and NVE endpoint drift, and (6) the
finite-time and (if reported) quality-gated RPMD rate. Bootstrap holds each
selected surface fixed; it does **not** account for xi* selection uncertainty.

**No numerical results are included.** Do not substitute previous fixed-xi*
`PMF-CONVERGENCE` results for this auto-xi* matched-cost experiment. Before
publication, repeat with independent seeds, more beads, smaller dt, longer
sampling and recrossing propagation, and increased Rinf. A flat or
multi-modal PMF near the maximum requires a separate dividing-surface
stability analysis.
