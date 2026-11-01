# Nonadiabatic validation against the published Tully model

This experiment uses the actual public `AdiabaticRPSHDynamics` and
`MASHDynamics` propagators on Tully's simple avoided crossing Hamiltonian.
The reference probabilities are **independent quantum calculations of that
published model**, not numerical points digitized from a paper. This is a
single-bead (`P=1`) scattering test; it does not establish quantum nuclear
accuracy of the ring-polymer extensions or validate a thermal rate constant.

The completed numerical results and confidence intervals are recorded in
`channel_probabilities.csv` and `summary.json`. The accompanying report gives
the interpretation, including deviations and the limits of finite ensembles.

## Published problem and source mapping

The Hamiltonian is Eq. (21) of [Tully, *J. Chem. Phys.* **93**, 1061–1071
(1990), DOI 10.1063/1.459170](https://doi.org/10.1063/1.459170):

\[
V_{11}(x)=A\,\operatorname{sgn}(x)(1-e^{-B|x|}),\quad
V_{22}(x)=-V_{11}(x),\quad
V_{12}(x)=V_{21}(x)=C e^{-D x^2}.
\]

All quantities are in atomic units: `A=0.01`, `B=1.6`, `C=0.005`, `D=1`,
and nuclear mass `m=2000`. Tully's Section IV A prepares incidence from the
left on the lower adiabatic surface. Figure 4 reports lower transmission,
lower reflection and upper transmission. The original paper used 2,000
trajectories per classical point; the present study uses 512 per main point.
The [published paper PDF](https://xuv.scs.illinois.edu/chem540/GroupProjects/Benke_Hammer_SourcePaper1.pdf)
provides the equations and figure definitions used here.

The three incident momenta are `8.8`, `15` and `25`. The asymptotic incoming
total energy is `p_in**2/(2*m)-A`. Trajectories start at `x=-8`; their initial
momenta are corrected by the very small difference between the finite-boundary
potential and its asymptote, so their conserved target energy is the specified
asymptotic energy. An outgoing trajectory is classified at `x>=8, p>0` or
`x<=-8, p<0`. The upper asymptotic channel opens at `sqrt(80)≈8.94427`, so
`p=8.8` tests the closed upper channel and reflection through temporary upper
surface excursions.

FSSH starts with a pure lower-state electronic amplitude, uses NAC-directed
momentum rescaling, retains momentum at frustrated hops, and applies no
decoherence correction. MASH samples the lower-population hemisphere with
density proportional to the absolute longitudinal spin; its outgoing
population estimator is the active-surface indicator. It reverses the NAC
momentum and reflects the longitudinal spin after a frustrated hop. Both use
four electronic substeps per nuclear step and exact electronic overlaps from
an analytic adiabatic provider. The provider supplies energies, positive
gradients, derivative couplings, and geometry-to-geometry overlaps through
the same callback interface intended for external electronic-structure codes.
It is an analytic test provider, not an ab initio calculation.

MASH's theoretical source is [Mannouch and Richardson, *J. Chem. Phys.*
**158**, 104111 (2023), DOI 10.1063/5.0139734](https://doi.org/10.1063/5.0139734).
The integrator distinction is documented in [Geuther, Asnaashari and Richardson,
*J. Chem. Theory Comput.* **21**, 2179–2188 (2025),
DOI 10.1021/acs.jctc.4c01684](https://pmc.ncbi.nlm.nih.gov/articles/PMC11912208/):
Sections 2.1 and 2.3 explain impulse/reflection and phase-aligned overlap
propagation; Section 3 distinguishes endpoint algorithms from reversible and
event-resolved algorithms. **The PyRPMD implementation tested here switches
at the completed nuclear-step endpoint. It is not the event-resolved integrator
from that paper.** Timestep refinement measures its error rather than silently
assuming equivalence to that more accurate algorithm.

`sources.json` and the manifest provide a machine-readable source mapping.

## Ensembles, uncertainty and timestep tests

For each method and momentum, the main calculation uses `dt=4` and four
independent blocks of 128 trajectories, with master seeds `731921`, `182331`,
`904577` and `411953`. Each trajectory has its own `SeedSequence`, combining
the master seed, incident momentum and trajectory index. This avoids a
trajectory's random-number consumption affecting subsequent trajectories.

The first block is repeated at `dt=8` and `dt=2`. Thus each method/momentum has
512 main trajectories plus 128 coarse and 128 fine trajectories; the full
experiment has 4,608 trajectories. MASH keeps exactly the same sampled initial
spin across timestep refinements. FSSH also reuses each trajectory seed, but
different step counts change the subsequent random decisions; its pairing is
a variance diagnostic, not a claim that stochastic paths should coincide.

Channel probabilities have two-sided 95% Wilson binomial intervals. These
are pointwise sampling intervals, not simultaneous confidence bands and not
estimates of systematic method error. Zero observed events have a nonzero
upper confidence bound. The four seed blocks are retained individually;
Pearson homogeneity statistics are only treated as applicable when expected
cell counts are at least five. Absence of a detected seed effect does not
prove convergence.

The timestep report compares the same 128 trajectory identities, records
changed channel outcomes with their own Wilson intervals, and measures
maximum absolute energy drift along every trajectory. Identical outcomes
in 128 pairs do not establish zero timestep error. Energy conservation also
cannot by itself establish the convergence order of electronic populations
or nuclear trajectories across surface switches. A maximum duration of
12 nominal flight times is imposed; unfinished trajectories are retained in
the explicit `unresolved` category and are never discarded or redistributed.
Conservative asymptotic probability bounds assign each unresolved outcome
to every possible channel in turn. A secondary, predeclared follow-up of
the censored paths uses 48 nominal flight times, records its result separately,
and leaves the main 12-flight-time classifications and denominators intact.

## Independent quantum reference

`quantum_scattering` solves the stationary, two-component Schrödinger equation
in the diabatic basis. A central second-difference kinetic operator on
`[-12,12]` is connected to analytically eliminated constant-potential leads.
Outgoing propagating waves and exponentially decaying closed channels are
imposed through the leads' exact discrete Bloch factors. Transmission and
reflection are normalized using the corresponding discrete probability
currents, including the outgoing-to-incident velocity factor.

This reference shares the **Hamiltonian** with the trajectory calculation.
It does not share a nuclear trajectory approximation, electronic hopping
rule, or time integrator. The controls are:

- A free incoming diabatic channel transmits with unit probability and no
  reflection, including the boundary conditions.
- With zero diabatic coupling and a closed right channel, all incident flux
  reflects.
- The coupled solver conserves incoming flux and shows second-order mesh
  convergence when the spacing is halved.
- Production references use spacings `0.005`, `0.0025` and `0.00125`; a
  separate `[-16,16]` calculation tests the finite boundary.

The smallest-spacing result is the comparison reference; the change from the
previous spacing is reported separately. It is many orders of magnitude
larger than the linear-solve residual, and that discretization change must
not be replaced by the smaller flux-conservation error as an accuracy claim.

### The paper's finite wavepacket width is a different comparison

Tully's Eq. (20) uses a Gaussian wavepacket with width typically `sigma_x=20/p`.
This corresponds to momentum standard deviation `sigma_p=p/20`, whereas
the paper's trajectories and this study's trajectories use fixed incident
momentum. The paper-width quantum result is therefore a momentum average,
not the same number as a monochromatic quantum result. The distinction
matters near the upper-channel threshold and narrow scattering resonances.

The main script's 17-node Hermite estimates are preserved in
`quantum_paper_width_average.csv` as **preliminary quadrature estimates**.
They are not used as acceptance targets. At `p=8.8`, a separate convergence
study records 33-, 65- and 129-node estimates and demonstrates that a small
Hermite rule is insufficient. `nonadiabatic_analysis.py --packet-convergence`
instead uses adaptive vector quadrature split explicitly at `sqrt(80)`,
at two spatial spacings and requested quadrature tolerances. Its results
are in `paper_width_reference_convergence.json`. This is still a
paper-inspired width calculation, not digitization of Fig. 4 or proof that
every individual paper point used exactly the same width.

## Direct-data transport and implementation checks

`transport_validation.json` follows a Tully trajectory through changing
adiabatic bases and arbitrary eigenvector sign changes. QD-NRPMD is compared
to the preexisting global-diabatic MInt implementation with identical physical
initial mapping variables. This is a representation-equivalence check;
focused signed mapping estimators are not reinterpreted as scattering counts
or thermal rates. Separate full-scattering sign-flip comparisons verify that
RPSH and MASH populations and endpoints do not depend on provider eigenvector
sign choices along the trajectory.

To make thousands of public-engine trajectories practical, the two-state
matrix logarithm and Hermitian exponential now use exact SU(2)/Pauli algebra.
They are regression-tested against SciPy's general matrix functions across
complex phases, tiny rotations, near-right-angle rotations and degenerate
Hamiltonians. This changes the evaluation cost; it does not add a classical
or electronic approximation. Larger electronic spaces retain the existing
general matrix routines.

## Reproduction and artifacts

From the repository root, with NumPy, SciPy, Matplotlib and a compiled native
backend available:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
python benchmarks/tier2_literature/nonadiabatic.py \
  --output docs/validation/tier2-literature-2026-11-01/nonadiabatic --workers 4

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
python benchmarks/tier2_literature/nonadiabatic_analysis.py \
  --output docs/validation/tier2-literature-2026-11-01/nonadiabatic \
  --packet-convergence --extend-unresolved
```

Each job writes an atomic checkpoint after every completed trajectory. Rerunning
with identical code and settings resumes the saved prefix; changed source
identity is rejected. The manifest records source SHA256 values, settings,
native capabilities and software versions. The raw `jobs/*.json` also record
the sampled initial spin/amplitude and seed entropy for each trajectory.

| Artifact | Meaning |
| --- | --- |
| `jobs/*.json`, `trajectories.csv` | Raw trajectory outcomes and numerical diagnostics |
| `seed_blocks.csv`, `seed_independence.json` | Independent seed block counts and diagnostics |
| `channel_probabilities.csv`, `summary.json` | Probabilities, Wilson intervals and monochromatic quantum differences |
| `paired_timestep_comparison.csv`, `timestep_diagnostics.*` | Finite-step differences and energy drift |
| `censoring_bounds.json`, `tail_followup/*.json` | Conservative unresolved-outcome bounds and separately labeled tail follow-up |
| `quantum_reference_convergence.csv` | Spatial-grid and boundary convergence |
| `quantum_paper_width_average.csv`, `packet_quadrature_*.json` | Preliminary Gaussian quadrature estimates, not converged comparison targets |
| `paper_width_reference_convergence.json` | Adaptive near-threshold Gaussian integration and mesh check |
| `transport_validation.json` | QC overlap/gauge and QD/global-diabatic equivalence |
| `scattering_channels.png`, `scattering_channels.svg` | Three-point main-ensemble comparison with uncertainty bars |
| `manifest.json`, `sources.json` | Reproducibility and primary-source mapping |

This benchmark cannot establish production accuracy for arbitrary molecules,
multistate MASH, complex spin-orbit couplings, `P>1` MASH/RPSH, decoherence,
long-time electronic coherence, or the separate thermal nonadiabatic-rate
workflow. Those require their own model-specific convergence evidence.
