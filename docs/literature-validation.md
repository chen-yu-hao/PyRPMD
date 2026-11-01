# Literature benchmarks and validation status

PyRPMD distinguishes a working implementation, a short physical-model pilot,
a source-verified published datum and a converged reproduction. These are
different pieces of evidence. The Tier 3 changes add interoperability, model
checks and auditable reproduction protocols; they do not turn incomplete
sampling into validated rate constants.

## Current case inventory

| Requested family | Available evidence and software | What prevents a reproduction claim |
| --- | --- | --- |
| H + CH₄ | Pinned RPMDrate PES-2008; 15.0143 kcal/mol barrier versus paper 15.01; force/native/umbrella checks and complete 768-draw reference pilot | Converged thermal-rate sampling; the short P128 0.1 fs NVE gate fails and the reference pilot has severe weight concentration and unresolved tails |
| OH + H₂ | Author tables checked, with corrected/uncorrected conventions separated; 300 K values agree across the two papers | Exact NN1 surface/weights, unresolved 1000 K table disagreement, and a full converged rate campaign |
| F + H₂ | Original paper and erratum identified; separately pinned author SW1/HSW sources pass bounded static energy, SW1 gradient and author-point checks | Corrected numeric reference stays null; match PES/electronic conventions and validate dynamics before rate sampling |
| H + H₂ | Pinned BKMP2; classical reference and complete four-pair P128 quantum rate agree within paper precision; quantum rate +0.916%, whole nominal 95% interval inside the paper's ±5% band, all eleven fixed-P primary criteria pass | Supplementary parent-half equivalence still fails; coarse P64 fails equilibrium equivalence; independent umbrella precision and dynamic P256 convergence remain separate |
| D/Mu + H₂; D + HMu | Primary-source index distinguishes the H₂ isotope series from deep-tunnelling muonium exchange; unresolved table versions stay unpromoted | Exact source-specific masses, implementation bytes, supplementary protocols and light-mass convergence |
| H on Ni | Pinned author-cited EAM4 Ni(100)/H routines; active gradients, invariances, fixed/mobile partitions and cutoff checks pass through a static adapter | Original diffusion targets and full paper protocol remain unresolved; no matched diffusion or mixed-bead dynamics has run; earlier NiAlH pilot remains a separate model |
| H in Pd | Published bulk/nanocluster study identified | Match potential, loading, geometry and transport estimator; run the campaign |
| q-TIP4P/F diffusion | Flexible force field, full PME, independent force checks, eight-seed OpenMM reference campaign running, completed velocity-recording check and prepared P32 reference jobs | Complete reference equilibrium/production gates, then quantum production and long-time comparison with PyRPMD's own transport workflow |
| q-TIP4P/F IR | Bead dipoles and replayable RPMD spectrum diagnostic | Original paper uses PA-CMD; source curves, long liquid records and method-aware comparison are still required |

The machine-readable inventory and a local evidence-bundle auditor are in
[`benchmarks/literature`](https://github.com/chen-yu-hao/PyRPMD/tree/main/benchmarks/literature).
The existing
[published-rate comparator](https://github.com/chen-yu-hao/PyRPMD/tree/main/benchmarks/published_rates)
retains responsibility for reaction/PES/units/spin/symmetry matching. The new
auditor verifies supplied file hashes and performs numerical equivalence checks
on independent-replica estimates. It never declares a scientific reproduction
on its own.

## A reviewable reproduction record

For each case, keep the exact reference source/table or digitized curve,
the PES sources and parameter checksums, isotope masses, boundary/ensemble
definitions, all resolved inputs, software/compiler/device versions, seeds,
raw observations, intermediate PMF/correlation curves, uncertainty calculations
and the source revision. Checkpoint files and replay scripts make failures as
reviewable as successful runs. Establish the comparison policy before looking
at the result.

For a rate, converge beads, timestep, reactant separation, PMF overlap/sampling
and the recrossing plateau. Include both PMF and recrossing uncertainty and
retain correlation between children from the same parent. Verify equivalent
channels and electronic/nuclear-spin factors rather than adjusting the final
number to match a paper. Compare exact quantum dynamics separately from RPMD
implementation references.

For diffusion, establish equilibrium, a diffusive time interval, stable
MSD/VACF estimates and independent-replica uncertainty. Unwrap molecular or
particle centroids consistently, account for lattice/COM drift, and match
dimensionality, density, loading, cell/slab size and fixed-layer conventions.
A few-femtosecond trajectory can test data flow and energy conservation but
cannot provide a macroscopic diffusion coefficient.

For water IR, retain the flexible potential and evaluate dipoles on each bead.
Converge sampling cadence, record length, window/broadening, bead count,
timestep and any TRPMD friction. Compare whole bands and normalization, not
one visually pleasing peak. Relative IR lacks the absolute optical prefactor.

## New water pilot

The [`benchmarks/water`](https://github.com/chen-yu-hao/PyRPMD/tree/main/benchmarks/water)
factory implements the published q-TIP4P/F equations with physical O,H,H atoms
and virtual charge sites. Its NumPy cluster oracle independently checks the
OpenMM force expression and virtual-site force transfer. Tests also check
isotope substitution, neutral dipoles, molecular images under periodic PME,
and rejection of invalid configurations. The small runtime pilot is expressly
an implementation check.

The original [water paper](https://doi.org/10.1063/1.3167790), available as an
[author manuscript](https://arxiv.org/abs/1011.1047), supplies RPMD diffusion
targets in Tables III–IV and PA-CMD spectra in Figure 5. The latter is an
essential method distinction: a TRPMD spectrum may be compared scientifically
with PA-CMD, but agreement is not a test that the same algorithm was reproduced.
Rigidifying q-TIP4P/F changes the potential model and removes its stretch IR.

The running classical water cohort retains its original 100 ps preparation
and 25 production segments for every seed. A separate all-eight continuation
coordinator has passed 35 orchestration fixtures, including real process-signal
and lease tests. It can extend every exact pre-production checkpoint by the
same 400 ps after a complete original assessment and a recorded adoption.
It has not yet been adopted or run. Neither this software gate nor a later
preparation extension can replace failed equilibrium checks with a diffusion
agreement claim.

## Author-cited nickel model

The [materials source audit](https://github.com/chen-yu-hao/PyRPMD/blob/feat/literature-convergence-20261101/docs/validation/literature-convergence-2026-11-01/other-reactions/materials-source-index/README.md)
identifies Cambridge OPTIM's `nimetpot.f90`, whose header credits Yury
Suleimanov and explicitly cites the 2012 Ni(100) paper. An isolated interface
builds the unchanged single-bead routines and checks their original eV/Å
conventions. Ten collected tests cover active-coordinate gradients, geometric
invariances, minimum-image boundaries, cutoff behavior and fixed-Ni semantics.
The maximum finite-difference discrepancy over the three tested spacings is
3.83 × 10⁻⁸ eV/Å. This is static implementation evidence; the original paper's
diffusion values, complete simulation protocol and original executable identity
remain unverified. The public OPTIM slab example is recorded as an example,
not substituted for those missing paper settings.

## Classical reference and complete quantum comparison

The 2026-11-01 follow-up obtains a complete classical 300 K H + H₂ rate of
**1.54540638 × 10⁻¹⁸ cm³ molecule⁻¹ s⁻¹**, with an independent-seed 95% interval
[1.54354068, 1.54727207] × 10⁻¹⁸. It combines deterministic surface quadrature
with independent surface-flux parents and unchanged native PyRPMD children.
The paper's rounded 1.55 × 10⁻¹⁸ value differs by 0.30%, within its approximately
5% stated precision; the published point is slightly outside the narrower
new interval. This reference does not by itself validate production umbrella
sampling or quantum RPMD.

The follow-up also identifies fixed-interval Andersen mode under-sampling
and supplies the separately tested [geometric collision clock](andersen-sampling.md).
A finite-P radial transfer matrix and a continuum radial Schrödinger calculation
independently check the quantum reactant normalization. The registered P=128
transmission campaign retains eight independent parent chains and paired
timesteps; its complete rate interval preserves correlation between each
chain's partition and child estimates. Its complete initial 200 fs cohort
gives κ=0.77307729 and **1.97196035 × 10⁻¹⁶ cm³ molecule⁻¹ s⁻¹**,
with a nominal 95% interval [1.91141982, 2.03441839] × 10⁻¹⁶. This is
1.126% above the paper; the full interval lies inside its approximate ±5%
band. The separately registered κ precision test still fails, with a 3.272%
half-width against a 3.000% target, so every parent receives the predeclared
additional momentum pairs. Earlier failed estimates remain intact.

The complete registered four-pair extension now gives **κ=0.77121022**,
95% interval [0.75333936, 0.78908108], and
**k=1.96785637 × 10⁻¹⁶ cm³ molecule⁻¹ s⁻¹**, nominal 95% interval
[1.93293861, 2.00340490] × 10⁻¹⁶. The paper difference is +0.916%;
the κ half-width is 2.317%, below the unchanged 3.000% limit. All eleven
fixed-P primary criteria pass. Independent reconstruction of all 800 raw
chunks, followed by a second replay from actual archive extraction, agrees
exactly. The supplementary parent-time-half equivalence check still fails:
its interval [−0.02310, +0.03626] extends beyond the fixed ±0.03 margin.
The separate P256 experiment has all eight parent chains complete, with
independent verification and exact archive-extraction replay. Its excluded
360-step native/Python/restart implementation check passes, and the complete
eight-chain child campaign has started. Its full dynamic bead comparison is
not yet complete.

The independent eight-ladder umbrella estimate at 30,000 production sweeps
is 11.583% above the paper and fails its 3% standard-error target. It advances
to the registered 100,000-sweep prefix without dropping chains or changing
parameters. Agreement of the independent surface reference cannot replace
that sampler's own convergence checks. Equilibrium bead refinements also do
not establish full dynamical bead convergence.
Detailed protocols, tables and limitations are available in the
[convergence record under review](https://github.com/chen-yu-hao/PyRPMD/blob/feat/literature-convergence-20261101/docs/validation/literature-convergence-2026-11-01/README.md).

## Recommended campaign order

1. Complete the quantum H + H₂ convergence record with the pinned BKMP2
   surface and separately establish production-umbrella precision. Reuse that procedure for D/Mu,
   with source-verified isotope targets and extra light-mass convergence.
2. Establish water force-model agreement first, then source-sized liquid
   equilibrium and classical diffusion, then RPMD isotope diffusion. Treat IR
   as a separate method-aware study.
3. Resolve methane timestep and sampling-cost controls using the now-pinned
   PES-2008, and obtain the matched OH/F surfaces and Ni/Pd material models
   before allocating expensive production. Distinct surfaces should have
   distinct case identities and comparisons.
4. Have another researcher reproduce at least one complete record from the
   archived inputs and model bytes before describing the suite as validated.

These priorities are implementation and scientific judgments, not claims that
all benchmark runs are complete. See [Publication readiness](publication-readiness.md)
for turning completed evidence into a software or methods paper.
