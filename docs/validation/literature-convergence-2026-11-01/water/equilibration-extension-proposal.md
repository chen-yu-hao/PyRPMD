# Proposed fixed equilibration extension

**Prepared for review; not adopted or launched.** The current eight-seed experiment continues unchanged. This proposal is a separate next-stage design to consider only after the complete first cohort has been retained and assessed. Its machine-readable details are in [equilibration-extension-proposal.json](equilibration-extension-proposal.json).

## Why a further check may be needed

The source paper reports a 100 ps preparation, but elapsed time alone is not evidence that a particular realization has reached equilibrium. Our eight chains share a different realized initial geometry, and the trajectories have finite, correlated fluctuations. Matching the paper's duration does not replace checking the actual observables. The reference is Habershon, Markland and Manolopoulos, *J. Chem. Phys.* **131**, 024501 (2009), [DOI 10.1063/1.3167790](https://doi.org/10.1063/1.3167790), [author manuscript](https://arxiv.org/pdf/1011.1047).

In the first full preparation audit, seed 1102's mean potential energy changed by **−0.15744 kcal/mol/water** between 50–75 ps and 75–100 ps, exceeding the predeclared **0.05** threshold. Seed 1101 changed by −0.00901 kcal/mol/water. Temperature, OH distance and HOH angle differences passed their original numerical thresholds for both chains. Temperature stationarity therefore did not resolve the potential-energy question. The failed criterion is evidence that this preparation has not demonstrated the requested stationarity; by itself, it does not prove whether the cause is residual relaxation or unusually strong correlated equilibrium fluctuations.

No original chain or segment will be discarded, restarted with a preferable seed, or relabeled as equilibrated because its diffusion estimate looks favorable.

## Fixed starting states and budget

The proposed extension branches **all eight original seed chains** from their exact **100 ps preparation checkpoints**, before any NVE momentum draw. It does not use selected production endpoints. The selector is absolute step 400000 with `active_segment = -1`; the filename alone is insufficient. Before any extension starts, a separate receipt must record all eight checkpoint hashes, their model/configuration identities and their saved external RNG states. Some of those checkpoints do not yet exist.

Each chain receives exactly **400 ps additional preparation**, reaching **500 ps total preparation**. There is no early pass, moving-window stopping rule, or automatic increase selected from a diffusion estimate. Every chain completes the same budget before a production decision. The original Hamiltonian, 0.25 fs native Verlet map, 298 K temperature, fixed cell, full PME settings, isotope masses, and one-thread CPU configuration remain unchanged.

The original periodic global Maxwell refresh also remains unchanged. At the branch point, absolute step 400000, the extension must perform exactly one normal preparation draw with `remove_com=False` using the saved RNG. It must not perform the original NVE COM-projected draw. Subsequent preparation draws occur at the original absolute multiples of 400 steps, corresponding to 100 fs. Raw engine state and the frozen CPU checkpoint validation contract are retained; no bitwise future CPU-trajectory guarantee is implied.

The extension needs a separate branch manifest and output directory. Editing an original manifest's configuration to evade its resume checks is explicitly excluded. A future preparation-only runner must stop before any NVE draw at 500 ps.

## Fixed assessment

The primary stationarity windows are **300–400 ps** and **400–500 ps** of absolute preparation time. Earlier 100–200 and 200–300 ps windows are diagnostic only and cannot trigger early production or replace a failed final window.

Every chain is assessed against the original absolute half-mean thresholds: 5 K in kinetic temperature, 0.05 kcal/mol/water in potential energy, 0.001 Å in mean OH distance, and 0.1 degrees in mean HOH angle. All eight individual results remain visible. An equally weighted summary of the eight chain-level differences receives a Student-t 95% interval with seven degrees of freedom; cancellation in that average cannot erase an individual-chain failure.

Fixed 5, 10 and 20 ps block estimates and autocorrelation diagnostics assess the uncertainty of those comparisons. They do not choose endpoints. If slow correlations remain unresolved, the stage does not establish eligibility for a converged production claim.

Structural checks compare the same two final 100 ps windows. Oxygen RDFs use fixed 0.02 Å bins, 0.5 ps sampling, cubic minimum-image distances and an explicitly recorded finite-pair normalization. The proposed numerical criteria are a maximum absolute RDF difference of 0.1 over the fixed 2.3–6.0 Å range, and an absolute coordination-number difference no larger than 0.1 at the fixed 3.5 Å radius. These are proposed diagnostic tolerances, not claimed literature error bars. The radius is not fitted to a newly observed minimum. Full curves, block uncertainties, across-seed variation, and OH/HOH distributions from the full-state checkpoints are retained.

## Production remains a separate decision

Any subsequent production cohort must be separately labeled and registered only after all eight extensions finish and the fixed preparation/structural checks are assessed. A concrete candidate is the same 25 × 4 ps NVE design for each seed, using the continued RNG and one declared COM-projected momentum draw at each segment boundary.

The first cohort remains available, including its failed preparation diagnostic. The extended and original cohorts share seed histories and must not be counted as sixteen independent chains. They must not be silently pooled or substituted for one another. Time-step sensitivity and transport-estimator equivalence remain separate questions; a longer preparation alone does not establish them or quantum-water agreement.

Before adopting this proposal, a new runner requires small Reference continuous-versus-split tests at the exact refresh boundary, an explicit saved-RNG/one-draw check, the existing CPU raw-state and force-cache validation, and a preparation-only terminal-phase test. None of those requirements authorizes modifying the running original experiment.

## Measured cost

The [completed original native call](scheduling-v2/migration-odd-natural-exit.json) advanced seed 1101 from 118 to 185 ps: 268000 committed steps used **1367.64 CPU seconds** and **1808.69 logged wall seconds**. This corresponds to **0.005103 CPU seconds per step** and **0.006749 wall seconds per step** for that actual call. CPU time includes startup and restart validation. This is a measurement from the unchanged original trajectory, not a separate performance-only run.

| Proposed work | Steps across eight seeds | CPU hours at that measurement | Wall hours with four full CPU equivalents | Wall hours if four workers each retain the observed allocation |
|---|---:|---:|---:|---:|
| Additional 400 ps preparation per seed | 12800000 | 18.14 | 4.54 | 6.00 |
| Optional later 100 ps NVE per seed | 3200000 | 4.54 | 1.13 | 1.50 |

These are conditional planning estimates. The measured call was NVE; preparation has a different observation cadence and momentum-refresh overhead. Shared-host contention can change wall time, and the numbers are not guaranteed upper bounds. The additional preparation and optional production have **not been launched**.
