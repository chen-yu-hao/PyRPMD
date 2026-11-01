# Literature convergence follow-up: 2026-11-01

**The complete classical H + H₂ reference rate at 300 K agrees with the
published RPMDrate result within the paper's stated precision.** Quantum
H + H₂ and liquid-water transport are separate, ongoing convergence studies.
This record preserves the earlier failed estimates and distinguishes an
independent equilibrium reference from the production MD umbrella workflow.

The preceding [literature audit](../literature-audit-2026-11-01/README.md)
is unchanged. Its large rate discrepancies, short water trajectories and
software-test results remain historical observations, not overwritten runs.

## Results established in this snapshot

| Quantity | New result | Comparison and scope |
| --- | ---: | --- |
| Classical H + H₂ TST | 1.5531187868 × 10⁻¹⁸ cm³ molecule⁻¹ s⁻¹ | Independent deterministic surface integral; paper 1.56 × 10⁻¹⁸ |
| Classical transmission κ | 0.99503424, 95% CI [0.99383298, 0.99623550] | Four independent source ensembles and unchanged native PyRPMD children; paper 0.994 |
| Complete classical H + H₂ rate | 1.54540638 × 10⁻¹⁸, 95% CI [1.54354068, 1.54727207] × 10⁻¹⁸ cm³ molecule⁻¹ s⁻¹ | −0.30% from the paper's rounded 1.55 × 10⁻¹⁸; within its approximately 5% stated sampling precision |
| Separate P=1 umbrella MC, 262,144 sweeps per chain | 1.5721439 × 10⁻¹⁸, 95% CI [1.3966727, 1.7708574] × 10⁻¹⁸ | Point accuracy passes; the predeclared 1% precision criterion still fails |
| H₂ finite-P quantum/classical partition ratio, P=128 | 0.000694194285900783 | Independent radial transfer matrix; radial-grid log change 1.03 × 10⁻¹¹ |
| H₂ continuum quantum/classical ratio | 0.000685451742716482 | Independent radial Schrödinger/DVR calculation; the finite-P difference is retained |
| Stochastic P=128 H₂ normalization control | 0.0006879725, 95% CI [0.0006792694, 0.0006967870] | Four independent chains; exact finite-P value is inside the interval |
| Independent P=128 H + H₂ QTST | 2.55248792 × 10⁻¹⁶, 95% CI [2.51679142, 2.58869070] × 10⁻¹⁶ cm³ molecule⁻¹ s⁻¹ | Eight independent surface-partition chains and matching exact finite-P reactant denominator; +1.69% from paper QTST; complete quantum dynamics still running |

The classical paper point is slightly outside the much narrower new rate
interval. Agreement here means agreement within the paper's approximately
5% precision and rounding, not a claim that the published point is inside
our interval. These intervals measure sampling uncertainty; they do not
include uncertainty in the PES or the physical approximation.

The literature reference is Y. V. Suleimanov, J. W. Allen and W. H. Green,
*RPMDrate: Bimolecular chemical reaction rates from ring polymer molecular
dynamics*, Comput. Phys. Commun. 184, 833–840 (2013),
[doi:10.1016/j.cpc.2012.10.017](https://doi.org/10.1016/j.cpc.2012.10.017),
Table 2. The H + H₂ entry uses the ξ‡=1.000 dividing surface. Its P=128,
300 K targets are QTST=2.51 × 10⁻¹⁶, κ=0.776 and full rate=1.95 × 10⁻¹⁶
cm³ molecule⁻¹ s⁻¹.

## Evidence by question

- [Classical source, quadrature, WHAM and complete rate](h3-classical/README.md).
  This includes all fixed-budget stages, exact prefix replays, independent
  surface moments, native dynamics, timestep checks and the final plateau.
- [Fixed-interval Andersen collision diagnosis and correction](review/README.md).
  The optional geometric clock removes the demonstrated resonant mode
  under-sampling while retaining the exact legacy default and restart state.
- [Quantum equilibrium methods and deterministic H₂ normalization](h3-quantum/README.md).
  Finite-P transfer matrices and the continuum DVR answer different questions;
  the P=128 comparison uses its matching finite-P normalization.
- [Independent finite-P surface partition campaign](surface-partition/README.md).
  Nine λ states connect a known reference to the physical surface distribution.
  BAR, independent-chain uncertainty and parent-source snapshots are explicit.
- [Quantum child protocol](h3-quantum/recrossing-protocol.json) and
  [analysis specification](h3-quantum/recrossing-analysis-prespecification.json).
  These bind all parents, both timesteps, observation windows, precision-only
  continuation and the joint partition/transmission uncertainty.

Raw checkpoint arrays stay in separately checksummed data archives. The
repository keeps the executed benchmark sources, protocols, report tables,
test receipts, figures and raw-data indexes. External BKMP2 source and compiled
libraries are fetched/built under their recorded provenance and are not
redistributed as PyRPMD code.

## Why the previous rate checks failed

Three concrete issues have been isolated without changing a rate prefactor,
temperature, mass or potential to match the paper.

First, a fixed Andersen collision interval can repeatedly refresh momenta
near a free-ring normal-mode recurrence. In the actual P=128 production
umbrella path, a mode initialized collapsed retained only about 10.6% of
its equilibrium variance after 500 fs with the fixed schedule; the geometric
clock produced about 98.5% in the same controlled test. The older real H₃
reactant endpoints also show this mode's variance near 30%, consistent with
the independently derived relaxation timescale. This is evidence of a real
sampling defect; it is not a complete numerical decomposition of the old
reaction-rate error.

Second, Gaussian umbrella integration is not exact for non-Gaussian window
distributions. The independently integrated classical density gives a 2.52%
asymptotic bias for the original strong umbrellas. Uniformly soft umbrellas
near the barrier are bimodal and can have almost no barrier support. The
new mixed-stiffness schedule is derived from exact overlap/support integrals,
and its histogram WHAM estimator passes a separate deterministic oracle.

Third, apparently reasonable individual-window means and small coordinate
Rhat values do not guarantee a precise integrated barrier. The original
classical ladder reaches a point estimate only 0.78% above the paper after
262,144 sweeps per chain, yet its independent-seed interval is still much
wider than its declared precision target. That failed gate remains visible.

## Quantum real-time campaign

All eight fixed-budget surface-partition chains are complete. Their
[quantum transition-state factor](h3-quantum/independent-qtst-comparison.json)
is 1.69% above the paper. The paper QTST point lies slightly below our new
95% interval, while the difference remains within the paper's approximately
5% reported precision. A quantum transition-state factor is not a complete
RPMD rate: it still needs its own transmission coefficient and independent
bead/domain refinement checks.

The registered finite-P surface sampler supplies eight independent chains
with 200 equilibrium parent geometries each. All 1,600 parents are retained.
Each starts two ordinary Maxwell momentum pairs, p and −p, with separate
keyed random streams. Identical parent configurations and momenta are used
at 0.1 fs and 0.05 fs; the latter is the primary step. The initial real-time
horizon is 200 fs. A failed plateau comparison triggers the predeclared
all-child 400-fs extension, and inadequate κ precision can trigger the
separately retained second momentum-pair batch for every parent.

The source already includes the configuration flux metric, fs(q), so each
signed child contribution divides by fs. The centroid velocity uses physical
masses with no extra factor of P. Unit and analytical tests check this
normalization. Every physical step checks the full ring Hamiltonian; the
minimum bead-pair distance is recorded at 1-fs observation boundaries.
No child leaving a diagnostic domain is silently removed.

An excluded calibration on the real P=128 BKMP2 system verifies that a
continuous 400-fs trajectory and a 200-fs checkpoint followed by another
200 fs, with recreated native and PES contexts, give bitwise-equal q, p,
surface/energy observations and clocks at both step sizes. This validates
the continuation mechanism; it does not establish a transmission plateau.

For the complete rate, the same independent chain supplies both its BAR
partition estimate and its child parents. The analysis therefore combines
log QTST and log κ within each chain before constructing the eight-chain
Student-t interval. It does not count individual children as independent
replicas or add the two uncertainties under a false independence assumption.
QTST, κ and their product are each compared with the paper, so compensating
component errors cannot produce an unqualified pass.

## Water and other requested systems

The water campaign retains 216 flexible q-TIP4P/F molecules, 298 K, the
declared density and full PME. Eight fixed independent seeds are scheduled
for 100 ps preparation plus 25 × 4 ps NVE segments each. No converged new
diffusion coefficient is claimed in this snapshot. CPU checkpoint restoration
and momentum conventions are audited separately from physical equilibration.
Short native MTS/RPC checks establish mathematical map/force agreement only;
they do not establish equivalent long-time diffusion.

The water reference is S. Habershon, T. E. Markland and D. E. Manolopoulos,
J. Chem. Phys. 131, 024501 (2009),
[doi:10.1063/1.3167790](https://doi.org/10.1063/1.3167790),
[author manuscript](https://arxiv.org/abs/1011.1047). Its diffusion comparison
uses RPMD; its infrared comparison uses PA-CMD. Rigid constraints or a
different spectroscopy method would define a different calculation.

The methane, OH, F, isotope, Pd and Ni targets remain separately identified
in the [benchmark inventory](../../../benchmarks/literature/README.md).
The completed H₃ reference and provider force checks do not validate those
unrun reaction or materials campaigns.

## Reproducibility and interpretation

Executed source hashes, exact model bytes, fixed seeds, statistical units and
protocol timing are recorded beside each result. Excluded calibrations and
failed gates remain available. Where an early script was not saved byte for
byte, the record says so and distinguishes a later numerical replay from
an exact historical source replay.

Software-test passes demonstrate the tested numerical contracts. They do
not replace physical convergence, and previous full-suite results are not
silently reassigned to newer source revisions. All changes remain reviewable
in the draft [PR #62](https://github.com/chen-yu-hao/PyRPMD/pull/62).
