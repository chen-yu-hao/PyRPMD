# How to converge a result

Start with a model whose result is independently known. The
[analytic tutorials](../examples/tutorials/) provide fixed inputs and comparison
scripts for harmonic PIMD, periodic site escape, a double-well free energy,
free diffusion and harmonic spectra. Their reports distinguish a measured
uncertainty from discretization error and from an unfinished calculation.

Choose the observable and sampling ensemble first: equilibrium PIMD times are
not physical transport times, a CV density is not a rate-flux PMF, and a fixed
collision energy yields a cross section rather than a thermal rate. Record the
PES version, atom order, mass/isotope convention, temperature, bead count and
normalization in the input. Use a fresh run when these physical choices change.

## A practical convergence study

1. Check the potential's units and analytic forces, then choose a stable timestep.
   Compare an observable and energy drift at dt and dt/2 from matched initial
   conditions. Repeat at P and 2P where quantum discretization matters.
2. Equilibrate independent replicas and inspect stationarity. Compare early and
   late production blocks and estimate autocorrelation times. Longer correlated
   histories do not supply the same information as independent replicas.
3. Increase production length, the number of independent replicas, and any
   observable-specific extent until changes are small against a predefined
   accuracy goal. Report statistical and systematic changes separately.
4. Preserve the complete set of histories, including failures and unresolved
   trajectories. A confidence interval conditioned on surviving histories is
   different from an uncertainty interval for the requested ensemble.

## Checks for each observable

| Observable | Evidence to inspect |
| --- | --- |
| Free energy / PMF | Connected overlap, per-stream and endpoint ESS, sampled support, bin/grid refinement, independent replicas, and equilibrium stationarity. Frozen-bias estimators use only production after freeze and relaxation. |
| RPMD or site rate | An accepted PMF plus a transmission plateau, parent-chain uncertainty, child energy conservation, longer child times, and consistent basin/asymptotic normalization. |
| Direct scattering | Failed and unresolved fractions, outward separated products, larger impact cutoff and final separation, parent-state equilibration, and the correct collision-energy weights. |
| PIMD energies | Finite-P reference or bead refinement, independent replica means, virial/primitive estimator consistency, equilibration and time-step checks. |
| Diffusion | Unwrapped centroids, a diffusive MSD interval after the ballistic regime, fit-window variation, long independent replicas and consistent drift treatment. |
| Spectra | Physical production dynamics, property-surface identity, longer records, window dependence, independent replica estimates and finite frequency resolution. |

The [PMF comparison tool](../benchmarks/h3/rate_pmf_comparison.py) compares
MBAR, histogram WHAM and Gaussian UI on the same saved samples. Agreement
between estimators does not replace independent sampling. The
[MetaD/OPES stability ladder](../benchmarks/h3/stability_ladder.py) tests domain
survival under increasing bias and dt/2 before a separate longer run.

## Interpret the reported status

`QUEUED` and `RUNNING` describe execution. Completed files and a zero exit status
establish that the requested stage finished. `ready` describes a stage's
documented gates; `converged` and its diagnostics address the requested
scientific criteria. Missing uncertainties remain missing, and a withheld rate
must remain null. A `stable` workflow label describes software scope and
compatibility; it does not override any of these calculation-specific results.

For a tutorial's fixed analytic model, compare the deviation in standard errors
and inspect its precision target. Three-SEM tests can occasionally reject a
correct stochastic calculation. Keep that event visible, investigate its
diagnostics, and plan independent additional sampling instead of retrying
seeds until a comparison happens to pass.
