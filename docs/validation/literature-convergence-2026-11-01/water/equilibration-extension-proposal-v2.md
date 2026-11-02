# Proposed preparation extension: revision 2, not adopted or launched

This revision records the response to the independent review of the proposed
RDF maximum-bin gate. It supersedes the proposal's **unadopted structural-gate
design**, while preserving the original proposal bytes and all physical settings.
The machine-readable policy is in
[equilibration-extension-proposal-v2.json](equilibration-extension-proposal-v2.json).

The previous [JSON](equilibration-extension-proposal.json) remains at SHA256
`a5a9c85a2e2093839f3fd688516d65adef825e162b21351708fe5fa547acb1d6`, and its
[prose companion](equilibration-extension-proposal.md) remains at
`6bed2bb46e1de2ba0cc084a2fe7a472f9504bb6e1f908177a7e11514dc3c974f`.
No actual liquid RDF or diffusion values were inspected to choose this revision.

## The fixed experiment stays the same

The complete original eight-chain cohort must first be retained and assessed.
If the later preparation extension is adopted, all eight original seeds branch
from their exact original 100 ps checkpoints, before any NVE momentum draw.
Each receives a fixed additional 400 ps of preparation, reaching 500 ps total.
Every seed completes the whole budget before assessment; no early pass, chosen
late window, seed replacement, or diffusion-driven stopping is allowed.

The frozen full-PME q-TIP4P/F model, classical masses, density, 0.25 fs time step,
single-thread CPU platform, and existing periodic 100 fs Maxwell refresh remain
unchanged. At step 400000 the continuation performs exactly one pending
preparation draw with `remove_com=False` from the saved NumPy RNG. Subsequent
refreshes stay on the original absolute 400-step grid. The separate runner
stops at 500 ps before drawing any NVE momentum. It never changes the old
manifests or reuses their output directories.

The primary stationarity comparison remains the fixed 300–400 ps and 400–500 ps
windows, with every seed's absolute mean-difference threshold unchanged:

| Observable | Original per-seed threshold retained |
| --- | ---: |
| Momentum temperature | 5 K |
| Potential energy per water | 0.05 kcal/mol |
| Mean OH length | 0.001 Å |
| Mean HOH angle | 0.1° |

Fixed 5/10/20 ps blocks and autocorrelation diagnostics accompany the individual
results and equal-weight eight-chain summaries. Earlier fixed windows remain
diagnostic. Individual failures cannot cancel in an average. Unresolved slow
correlation or insufficient precision does not establish equilibrium.

## What changed in the proposed RDF assessment

The [analytic noise audit](rdf-noise-audit/README.md) shows why 185 separate
absolute-bin comparisons cannot be treated as one simple tolerance check.
Under an explicitly excluded uniform independent-Poisson fixture, the proposed
sample count gives a 41.99% chance of a false exceedance in one chain and a
98.72% chance across eight chains when bins and chains are also independent.
Those assumptions are not calibrated to water, and the probabilities are not
claimed to predict liquid-water behavior.

The maximum `|Δg|` and the proposed reference value 0.1 therefore remain
**descriptive only**. Full per-seed RDF curves, binning, fixed windows, and
coordination values are retained. A mandatory RDF-equivalence criterion remains
unset until a justified simultaneous-uncertainty policy is specified and
independently reviewed in advance. A wide interval containing zero is not an
equivalence result. The separate coordination threshold of 0.1 also remains
unadopted and requires fixed-block uncertainty review.

This revision does not weaken or reclassify the original failed preparation
energy gate. The existing cohort and every production segment are retained
separately. Any later production cohort requires its own decision after all
fixed preparation checks; it is never started automatically by the preparation
runner. Two cohorts sharing seed histories are not 16 independent chains.

## Execution readiness and unresolved adoption requirements

The separate preparation-only runner and its nine small Reference/CPU boundary
tests are ready for independent review. They cover the pending refresh at the
branch point, split restarts on and between refresh boundaries, the saved RNG,
the absence of an NVE draw, raw-state/force-cache validation, and model/source
guards. Passing these numerical tests does not demonstrate liquid equilibration.

Before any long launch, the review must settle the RDF uncertainty design,
confirm the complete original cohort assessment, and seal the exact original
100 ps checkpoint/model/configuration hashes for **all eight** seeds. The generic
single-chain runner does not make that cohort-level scientific decision.
No real 400 ps extension configuration has been launched for this revision.

The previous measured runtime estimate remains unchanged: the fixed all-seed
extension requires 12.8 million additional steps, approximately 18.14 CPU-hours
using the measured original native call, or about 6.0 wall-hours with four workers
at the observed allocation. This is a conditional estimate, not a wall-time
guarantee; the measurement came from NVE and the preparation cadence differs.
