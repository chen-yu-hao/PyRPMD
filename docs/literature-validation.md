# Literature benchmarks and validation status

PyRPMD distinguishes a working implementation, a short physical-model pilot,
a source-verified published datum and a converged reproduction. These are
different pieces of evidence. The Tier 3 changes add interoperability, model
checks and auditable reproduction protocols; they do not turn incomplete
sampling into validated rate constants.

## Current case inventory

| Requested family | Available evidence and software | What prevents a reproduction claim |
| --- | --- | --- |
| H + CH₄ | Pinned RPMDrate PES-2008 source/build; 15.0143 kcal/mol barrier versus paper 15.01; force, native-map and actual umbrella/checkpoint checks | Converged thermal-rate sampling; the short P128 0.1 fs NVE probe fails its fixed energy gate |
| OH + H₂ | Source-verified NN1/YZCL2 entries with corrected/uncorrected conventions separated | Matched surface and a full converged rate campaign |
| F + H₂ | Original paper and erratum identified; missing numeric reference stays null | Extract corrected source data and match PES/electronic conventions |
| H + H₂ | Pinned BKMP2; classical reference agrees within paper precision; initial P128 quantum rate +1.126% with whole nominal 95% interval inside the paper's ±5% band; equilibrium P128→256 and domain checks | Complete predeclared extra momentum sampling: κ half-width 3.272% exceeds 3.000%; coarse P64 fails equilibrium equivalence; independent umbrella precision and dynamic bead convergence remain separate |
| D/Mu + H₂; D + HMu | Primary-source index distinguishes the H₂ isotope series from deep-tunnelling muonium exchange; unresolved table versions stay unpromoted | Exact source-specific masses, implementation bytes, supplementary protocols and light-mass convergence |
| H on Ni | Real EAM/ASE integration pilot with H/D and bead cases | Existing EAM/slab pilot is not the exact model of the 2012 Ni(100) study |
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

## Classical reference and initial complete quantum comparison

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
