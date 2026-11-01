# Adaptive constant-potential thermal NRPMD validation

This record covers the new Python integration on base commit
`2f907ba96d22cb134198715e398b1d9f8d83d929`. It is an analytical-model and targeted
regression check, not validation of a real electrode, chemical rate, or current.

## Regression checks

The [machine-readable validation record](validation.json) records **439 passed,
3 skipped, 0 failures/errors**. The three skips require an optional compiled
nuclear kernel. Reference Python propagation was exercised. The test selection
covers the existing NRPMD, thermal NRPMD, complex-matrix and electrochemistry
implementations together with all four new test files (57 new cases).
It is not the full repository test suite or a native/MPI/GPU validation.

The new independent checks include:

- finite-difference matrix gradients after electronic minimization, explicit
  voltage sign and unchanged fixed-geometry electronic gaps/couplings;
- arbitrary thermal mapping norms and the one-state ordinary-RPMD limit;
- exact `beta*W` umbrella factors, harmonic normal-mode curvatures and sampled
  centroid/mapping statistics;
- unbiased Gaussian moments and window offsets from analytically generated
  umbrella samples, and signed ratios against a synthetic analytical reference;
- disconnected-overlap/sign failures, unchanged endpoint centers, repeatable
  adaptation, fresh-production sample counts, full-state resampling ancestry,
  and refusal of inappropriate rate error bars after resampling.

The command is saved in `validation.json`. Dependencies used here were Python
3.12.14, NumPy 2.5.4, SciPy 1.17.0, pytest 9.2.0 and h5py 3.16.0. Dependencies
were installed outside the repository; no dependency lockfile was changed.

## Complete two-bead example

[two-bead-example.json](two-bead-example.json) is the complete measured report
from:

```bash
python examples/electrochemistry/adaptive_constant_potential.py \
  --potential-V -0.3 --beads 2 --seed 20261101 \
  --output runs/adaptive-electrode.json
```

Five straight initial images underwent four adaptive rounds, followed by 256
fresh production samples per frozen window. The middle center moved from
`(0,0)` to approximately `(0.0033353,0.2744467)` bohr. Endpoint centers stayed
fixed. The stopping reason is `adaptation_budget_exhausted`; the record does
not claim a converged path.

The measured MBAR average sign was `0.8107909`, and the correlation-adjusted
positive-measure ESS diagnostic was `76.3581`. The overlap graph was connected
and the MBAR equations met their residual criterion. These diagnostics do not
replace source-chain bootstrap, independent chains or quantum bead convergence.

The same constant-potential matrix propagated 32 resampled parents for 0.6
atomic time, without umbrella forces or thermostat. The maximum absolute
sum-bead propagation-Hamiltonian drift was `4.2434e-4` hartree at `dt=0.01`.
Electronic-stationarity residuals were below `1e-14` hartree. These are short
integration checks; this record does not infer a rate from their duration.

The cheap `--quick --beads 1` test exercises the classical-limit full pipeline.
The intentionally smaller two-bead `--quick` budget can return unresolved signs
or insufficient ESS; in that case the example reports the reason and launches
no dynamics. Guards are not relaxed to force a successful demonstration.

## Scientific scope

The model is `V_mu=V_bare+min_N(E_charge-mu*N)*I`: common BO charging with one
electronic optimum per nuclear bead. It affects nuclear forces but leaves gaps
and couplings fixed at a given nuclear geometry. This is not state-dependent
Faradaic reservoir exchange or shared-N grand-canonical sampling.

The path is a finite-restraint string in selected scaled CVs. The initial signed
SEO measure is retained through positive-weight MBAR and unbiased real-time
NRPMD. All reports retain `physical_converged=false`. Details and references
are in the [implementation guide](../../adaptive-nrpmd.md).
