# Initial vibration-selected QCT and NE-RPMD scattering

PyRPMD can prepare independent, approximately vibration-selected initial
conditions for direct, fixed-collision-energy scattering. The two preparation
methods are `qct` (one bead) and `ne_rpmd` (a harmonic ring-polymer ensemble plus
centroid vibrational excitation). This is an experimental initial-state method.
The reported observable is an initial-state-selected **chemical-channel integral
cross section**, subject to the usual dynamical and sampling convergence checks.

The implementation does not assign final rovibrational quantum states or compute
an exact quantum state-to-state S-matrix. Its `j=0` setting enforces zero
**centroid angular momentum** of each incoming fragment; it is not a complete
quantum rotational-state sampler. Arbitrary rotational excitation, quantum
coherences, rotational ring modes, and final-state binning are outside this
version's scope.

This nuclear initial-ensemble option is separate from electronic
`nonadiabatic.method = "nrpmd"`. The two options cannot be combined. Omitting
`initial_ensemble` retains the existing thermal-parent dRPMD workflow.

## Method and conventions

| Preparation | Beads | Initial vibration | Collision ensemble |
| --- | --- | --- | --- |
| Existing thermal parent | One or more | Equilibrated at the specified temperature | Existing thermal or fixed-energy options |
| `qct` | Exactly one | Random harmonic phases with mode energies approximately `(v + 1/2) * omega` | Fixed energy, independent preparations |
| `ne_rpmd` | At least two | Cold harmonic rings plus a centroid displacement for the additional excitation `v * omega` | Fixed energy, independent preparations |

All formulas below use atomic units, including `hbar = 1`. Molecular normal
modes and ring-polymer Fourier modes are different coordinates. If a molecular
mode has frequency $\omega_a$, a ring mode has spring frequency $\omega_k$, and
there are $P$ beads, the harmonic reference draws use

$$
\operatorname{var}(Q_{ka}) =
\frac{P}{\beta(\omega_a^2+\omega_k^2)},\qquad
\operatorname{var}(\Pi_{ka}) = \frac{P}{\beta}.
$$

The orthonormal ring coordinate $Q_{0a}$ equals $\sqrt{P}$ times the bead-mean
molecular coordinate. NE-RPMD adds a random-phase centroid displacement with
excitation energy $v_a\omega_a$. It does **not** add a second half-quantum on
top of the ground-state ring fluctuations. QCT instead starts with the harmonic
action assignment $E_a=(v_a+1/2)\omega_a$ on a single bead. Removing residual
centroid rigid rotation can change these initial harmonic mode energies, so
inspect the sampled diagnostics as well as the requested quantum numbers.

Each fragment, including an atom, retains noncentroid free-ring center-of-mass
fluctuations in the NE preparation. Its centroid translation is fixed until
collision launch. Molecular rotational ring modes are omitted. Coordinates and
momenta are rotated together at launch, preserving their relative orientation,
and the specified collision energy supplies the relative centroid motion.

The harmonic references describe the isolated fragments near their equilibrium
geometries. Propagation uses the full configured PES. Anharmonicity, mode
coupling, large vibrational excitation, the projection of rigid motion, and the
change from a harmonic reference to the full PES can all make the prepared
ensemble differ from a stationary rovibrational state. A quantum number in the
input is therefore a preparation label, not proof of exact quantum-state purity.

### Beta is an ensemble parameter here

For `ne_rpmd`, `initial_ensemble.beta` is explicit and also sets the spring
frequencies used in propagation. A numerical value is in inverse hartree; for
example, `"600 hartree^-1"` is a unit-bearing value. The formal conversion
$T_\beta=1/(k_B\beta)$ does not make this fixed-energy, vibration-selected
ensemble a thermal gas at $T_\beta$.

Do not specify top-level `temperature` for either selected method. NE-RPMD
requires an explicit `initial_ensemble.beta`. QCT may omit beta; its internal
P=1 compatibility value does not affect propagation and is not a temperature.
The choice of beta is part of the approximation and needs its own sensitivity
study together with bead convergence. Large beta reduces harmonic thermal
excitation but can require more beads. A fixed choice of 32 beads is not a
convergence result. This implementation does not automatically select beta
from the collision energy or reproduce every beta prescription in the papers.

The literature develops the nonequilibrium RPMD framework [1], applies
QCT-like ring-polymer preparations to state-selective gas-phase scattering
[2,3], and extends the approach to polyatomic reactants [4]. The harmonic
initialization used here is based on the construction discussed in [4], with
the implementation restrictions described above. Related molecule-surface work
[5] demonstrates another nonequilibrium application; moving-surface scattering
is not implemented by this initial-ensemble option.

## Configuration

State-selected preparation requires `task = "drpmd"`,
`collision.sampling = "fixed_energy"`, a positive `collision.energy`, and
`collision.impact_sampling = "uniform_area"`. The system must contain exactly
two nonempty, nonperiodic reactant fragments with all atoms mobile. Thermal
parent burn-in, Andersen collisions, and thermal collision-energy sampling
must not be requested for this branch.

The following are configuration fragments to add to a complete dRPMD input.
The reference files and numerical choices must be supplied and validated for
the actual potential. The first incoming fragment is an atom, and the second
is a diatom with one vibrational mode:

```toml
schema_version = 1
task = "drpmd"
beads = 32
seed = 17

[initial_ensemble]
method = "ne_rpmd"
beta = "600 hartree^-1"

[[initial_ensemble.fragments]]
# Atomic fragment: no vibrational reference or quantum numbers.

[[initial_ensemble.fragments]]
j = 0
vibrational_quanta = [1]
hessian_file = "diatom_hessian.npy"
hessian_unit = "hartree/bohr^2"

[collision]
sampling = "fixed_energy"
energy = "0.5 eV"
impact_sampling = "uniform_area"
orientation = "isotropic"
bmax = "3 angstrom"
trajectories = 1000
max_time = "20 ps"
```

These beta, bead-count, energy, cutoff and time values illustrate the syntax;
they are not recommended converged settings for a particular reaction. Set
`system.reactants` and `system.separation`, the potential, timestep, classifier,
output and parallel settings as in a complete dRPMD run. The two
`initial_ensemble.fragments` tables follow `system.fragments`/reactant order.
For a molecular fragment, omitted `vibrational_quanta` means zero in every
vibrational mode. Only `j = 0` is accepted.

For the QCT comparison, use `method = "qct"` and top-level `beads = 1`, with
the same fragment references, excitation labels and collision setup. One-bead
thermal sampling is not substituted for QCT harmonic action sampling.

### Supplying harmonic references

Provide one of the following for each molecular fragment:

- A Cartesian `hessian` array or `hessian_file`, with shape
  `[3*Nfragment, 3*Nfragment]`. The default unit is `hartree/bohr^2`; a
  `hessian_unit` can specify another supported spring-constant unit.
- `frequencies` together with `modes` or `modes_file`. Frequencies can be
  unit-bearing strings such as `"4400 cm^-1"`, or numerical atomic-unit values.
  A common `frequency_unit` can be supplied for numerical frequencies.

Files may be `.npy` arrays without pickles or whitespace-delimited numerical
text. File contents are included in the normalized configuration, so changing
a reference is a physical input change, not merely a path change.

Explicit molecular mode rows have shape `[Nv, Nfragment, 3]` or
`[Nv, 3*Nfragment]`. They must be **orthonormal mass-weighted eigenvectors**,
orthogonal to translations and rotations for the actual isotope masses.
Cartesian displacement eigenvectors from another program may use a different
normalization and cannot be copied without checking the convention. Wavenumbers
in `cm^-1` denote spectroscopic energies $hc\tilde\nu$, consistent with
$\hbar\omega$; they are not unconverted angular frequencies.

The expected number of modes is $3N-5$ for a linear molecule and $3N-6$ for a
nonlinear molecule; an atom has no vibrational modes. Hessian preparation
projects out rigid motion, applies the current masses and diagonalizes the
vibrational block. It rejects missing positive curvature, including unstable
or zero-frequency vibrational modes. Hessian mode order is increasing
frequency. In a degenerate or nearly degenerate subspace, the chosen mode basis
matters for a mode-specific excitation; supply explicit modes when that choice
needs to be controlled.

The reference geometry must be an isolated-fragment minimum on the intended
PES. This is a physical prerequisite, not something that Hessian symmetry and
positive eigenvalues alone can establish. Changing isotope masses requires
consistent normal modes; the Hessian route applies the new masses during
preparation.

## Independent sampling and reproducibility

The state-selected branch prepares each incoming trajectory independently.
Its random stream is derived from the master seed and trajectory identity;
workers do not evolve a shared thermal parent between selected-state launches.
The existing `parents` setting may partition work, but in this branch it is
not a source of thermal-chain dependence. Sampling does not use a production
thermostat, and propagation does not reassign the requested excitation after
the collision starts.

The low-level preparation API is:

```python
import numpy as np
from pyrpmd.initial_states import prepare_initial_ensemble

prepared = prepare_initial_ensemble(
    initial_ensemble, system, beads=32, beta=600.0
)
state = prepared.sample(np.random.default_rng(17))
initial_diagnostics = prepared.diagnostics(state)
reference_metadata = prepared.metadata
```

`system` contains the equilibrium geometry, fragment atom indices and masses
in PyRPMD's atomic-unit convention. The returned `State` has ordinary
`q` and `p` arrays with shape `[P, N, 3]`; it can be passed through the existing
collision-launch and propagation interfaces. `sample()` does not perform an
Andersen burn-in, collision kick, thermostat step or trajectory propagation.

## Check precollision state stability

Use the full-PES state check to examine how a selected ensemble evolves before
it is interpreted as an incoming state:

```bash
pyrpmd state-check selected.inp --time '100 fs' --samples 8 --stride 10 --output state-check.json
```

This check launches no incoming collision velocity. It propagates separated
reactants on the configured full PES and records mode diagnostics over time,
the change from the harmonic reference to the full PES, and fragment separation
diagnostics. Finite separation can still allow the reactants to interact. With
`system.reactants`, vary `system.separation` to establish that the test
approximates isolated fragments. With an inline or whole-system XYZ geometry,
move the fragments in that geometry: changing the launch-separation setting
alone does not move the reference used by this check. Inspect the reported
`fragment_com_separation_au`.
Eight samples and 100 fs are illustrative command arguments, not a sufficient
validation standard for every system.

Compare ensembles at preparation and after representative free-flight times.
Inspect mode-energy/action distributions, centroid angular momenta, fragment
COM motion and ring spreads. Test the relevant incoming flight time as well as
longer times when necessary. An isolated-state change can precede any collision
and bias an apparent excitation effect. Timestep, beta, bead count, reference
geometry and harmonic-mode choices need separate checks.

The initial harmonic-to-full-PES switching error is
`mean(V_full(q)) - V_full(reference) - V_harmonic(q)`. Subtracting the full-PES
reference cancels its arbitrary energy zero. This diagnostic concerns the
initial potential mismatch; it is not a comparison of two unaligned total
energy zeros. The report contains per-sample frames and ensemble statistics
for mode actions, paired changes from each sample's initial action, energy
parts, ring radii and observed Hamiltonian drift. Mode projections use a
mass-weighted Eckart alignment and remain local harmonic diagnostics; strong
rotation, large deformation or dissociation invalidates a quantum-population
interpretation. Choose the observation stride to keep the report manageable:
the checker retains samples and observation frames in memory.

`--output` is a JSON file relative to the invoking working directory. The
command writes no production run directory. Omitting it returns/prints the
report only. The Python entry point is
`pyrpmd.check_state_selected(spec, duration="100 fs", samples=8, stride=10)`.

### Conserved ring energy and physical vibrational energy

The dynamics conserve the extended ring Hamiltonian

$$
H_P = \sum_{s=1}^{P}\left[
  \sum_i \frac{p_{si}^2}{2m_i}+V(q_s)
  +\frac12\sum_i m_i\omega_P^2\lvert q_{si}-q_{s+1,i}\rvert^2
\right].
$$

For multiple beads, `H_P/P` is not the physical vibrational energy of a
molecule. It includes fictitious ring degrees of freedom and their spring
energy. Conserving it checks numerical propagation, not preservation of a
particular quantum state. Do not rescale all momenta to force `H_P/P` to equal
$\sum_a(v_a+1/2)\omega_a$.

State-selected collision traces enable the following observations by default:

```toml
[output.diagnostics]
energy_parts = true
ring_radii = true
```

Each trajectory NPZ contains `energy_parts[frame,4]` in the column order
`potential`, `centroid_kinetic`, `internal_kinetic`, `spring` (hartree), whose
sum equals unbiased `H_P/P`. `internal_kinetic` refers to internal bead motion,
not molecular vibration. `ring_radii[frame,atom]` is the root-mean-square bead
distance from that atom's centroid, in bohr. The same arrays and frame clock
are saved in checkpoints. The native block computes them from its current
q/p and already evaluated potential, without additional PES calls or Python
observation callbacks. Changing observation settings mid-trace is rejected
when a checkpoint must be continued, avoiding incomplete histories.

`prepared.diagnostics()` reports aligned harmonic mode projections, centroid
mode energies, angular momenta and ring spreads. Its primitive harmonic
mode-energy estimator contains a negative spring contribution. Individual
estimates and the derived actions can fluctuate, be nonintegral, or even be
negative. They are ensemble diagnostics; rounding them does not provide valid
final quantum-state assignments. The sampler metadata distinguishes the
requested harmonic energy, expected finite-bead mode energy before the
`j=0` projection, and residual harmonic thermal excitation.

## Fixed-energy cross sections and missing outcomes

For equal statistical weights and $b=b_{\max}\sqrt{U}$, with $U$ uniform on
$[0,1]$, the sampled disk has area $A=\pi b_{\max}^2$. If $N_c$ of all $N$
launched trajectories resolve into chemical channel $c$, its observed
contribution is

$$
\widehat\sigma_c = A\frac{N_c}{N}.
$$

Let $N_U$ count trajectories still unresolved at their time limit and $N_F$
count numerical failures. Both stay in the denominator. Conservative
finite-sample assignment bounds are

$$
\sigma_{c,\mathrm{observed}}=A\frac{N_c}{N},\qquad
\sigma_{c,\mathrm{possible}}=A\frac{N_c+N_U+N_F}{N}.
$$

These are missing-outcome bounds, **not statistical confidence intervals**.
They apply only inside the sampled impact-parameter disk and assume the
resolved classifications are correct. They do not bound reactions beyond an
unconverged `bmax`. Channel upper bounds are componentwise: the same missing
trajectory cannot ultimately populate every mutually exclusive channel.

`unresolved` represents a censored, otherwise valid propagation. `failed`
represents a numerical or execution problem and is reported separately; it is
not an observed physical survival time. Treating either class as nonreactive,
or dropping it from the denominator, can bias the result. This implementation
withholds the final `cross_section_bohr2` while unknown outcomes remain or the
classification/cutoff validation gates are not satisfied. Candidate counts are
still visible. Classification gates are caller-supplied assertions; the
analysis helper does not validate the classifier automatically.

The standalone analysis entry point is:

```python
from math import pi
from pyrpmd.analysis.scattering import summarize_fixed_energy_scattering

summary = summarize_fixed_energy_scattering(
    records,
    area_bohr2=pi * bmax_au**2,
    initialization_mode="independent",
    collision_energy_au=energy_au,
    channels=["channel_a", "channel_b"],
    classifier_validated=True,
    cutoff_validated=True,
)
```

Each product record needs a nonempty channel name. The helper rejects mixed
collision energies, non-area-uniform sampling, unequal trajectory weights,
invalid statuses and duplicate trajectory identities. An explicit equal
positive weight vector is allowed; its common factor cancels. Importance
sampling, signed electronic populations and variable-weight estimators are
not implemented by this helper.

Results include `total_reaction`, `channels` and `initial_state_groups`. State
groups use a canonical JSON descriptor containing the preparation method,
vibrational quantum numbers and rotation convention. Each group uses its own
trajectory denominator and lists its fraction of the supplied sample. If
different states are pooled, the aggregate is only the mixture implied by
those sample fractions; it is not automatically a physical beam-state mixture
or thermal average.

### Statistical intervals

`iid_wilson95_bohr2` is available only when the caller declares independent
complete initial-condition draws, the outcomes are complete, the
classification/cutoff gates are satisfied, and the estimate concerns one
initial-state ensemble. Independent random orientations on a correlated
thermal parent do not qualify. A zero-event independent sample has a positive
upper Wilson limit, not evidence for an exactly zero cross section.

For `initialization_mode="parent_chain"`, the helper resamples entire parent
clusters and recalculates the ratio of event counts to launched trajectories.
This preserves dependence among children of the same parent and handles
unequal parent sizes. The default requirement of eight independent parent
clusters is a cautious software threshold, not a universal guarantee of
bootstrap accuracy. Assess between-chain convergence and increase the number
of independent chains when needed.

Insufficient parent counts, missing parent identities, incomplete outcomes,
unvalidated gates, or absent between-parent variation produce explicit
unavailability reasons. In particular, a zero-width empirical bootstrap from
zero observed events is not presented as a useful confidence interval.
Unknown initialization dependence also withholds intervals. A pooled mixture
of different prescribed states has no pooled IID interval; per-state
intervals remain available when eligible. Channel intervals are marginal,
not simultaneous confidence regions for all branching fractions.

## Time limits and continuation

Choose a maximum trajectory time by observing resolution and cross-section
convergence for the actual reaction, collision energy and prepared state.
Increasing the time limit can resolve long-lived complexes, but does not by
itself establish that long-time ring-polymer dynamics faithfully represent a
physical complex lifetime. A small energy drift alone cannot make that claim.

Ordinary resume restores an interrupted run. To increase the allowed duration
of completed `unresolved` trajectories, use the explicit continuation command:

```bash
pyrpmd extend runs/selected --max-time '1 ns'
```

`--max-time` is the new **total** time limit measured from each trajectory's
original launch, not an additional duration. The original resolved input,
manifest and thermal-parent checkpoints stay unchanged. The separate
`results/extensions.json` ledger records the authorized budgets. Repeating the
same limit resumes an interrupted extension or leaves already completed work
intact. Ordinary `pyrpmd resume` also continues a pending authorized extension;
a completed extension remains visible when an ordinary resume rebuilds the
summary. Product, nonreactive and failed trajectories are retained.

Continuation preserves the current full-bead positions and momenta and the
workflow's candidate-channel and persistence clock. The endpoint force is
recomputed; production has no thermostat and performs no new random momentum
draw. Arbitrary private history inside a custom classifier object has no
automatic checkpoint protocol, so a classifier must be restart-compatible.
Reinitializing the rings, drawing new velocities,
thermostatting a trapped complex or projecting it back onto the requested
incoming mode would create a different ensemble. Numerical failures require
diagnosis and are not silently converted into additional censored trajectories.

Converge the exit separation and persistence criteria along with trajectory
length. A distant, outward-moving fragment geometry is useful evidence of an
outgoing channel, but attractive long-range interactions can permit recapture.
The existing formula-based classifier also cannot distinguish every atom
exchange or structural isomer. Use identity-aware channel criteria when the
scientific observable requires them.

## References and scope of the evidence

1. R. Welsch et al., *Non-equilibrium dynamics from RPMD and CMD*,
   **J. Chem. Phys. 145**, 204118 (2016).
   [DOI: 10.1063/1.4967958](https://doi.org/10.1063/1.4967958).
   Nonequilibrium theoretical foundation and specified preparation protocols;
   it does not establish arbitrary excited-state sampling as exact.
2. A. Marjollet and R. Welsch, *Nuclear quantum effects in state-selective
   scattering from ring polymer molecular dynamics*, **J. Chem. Phys. 152**,
   194113 (2020).
   [DOI: 10.1063/5.0004179](https://doi.org/10.1063/5.0004179).
   Initial-state scattering benchmarks and the QCT/RPMD preparation idea.
3. A. Marjollet and R. Welsch, *State-selective cross sections from ring polymer
   molecular dynamics*, **Int. J. Quantum Chem. 121**, e26447 (2021; first
   published online in 2020).
   [DOI: 10.1002/qua.26447](https://doi.org/10.1002/qua.26447).
   Detailed methodology, beta choices and bead-convergence studies for
   atom-diatom reactions.
4. A. Marjollet, L. Inhester and R. Welsch, *Initial state-selected scattering
   for the reactions H + CH4/CHD3 and F + CHD3 employing ring polymer molecular
   dynamics*, **J. Chem. Phys. 156**, 044101 (2022).
   [DOI: 10.1063/5.0076216](https://doi.org/10.1063/5.0076216).
   Polyatomic harmonic preparation and ground/excited incoming vibrational
   states; the main methodological reference for this implementation.
5. N. Gerrits, B. Jackson and A. Bogaerts, *Accurate Reaction Probabilities for
   Translational Energies on Both Sides of the Barrier of Dissociative
   Chemisorption on Metal Surfaces*, **J. Phys. Chem. Lett. 15**, 2566–2572
   (2024).
   [DOI: 10.1021/acs.jpclett.3c03408](https://doi.org/10.1021/acs.jpclett.3c03408).
   Related nonequilibrium molecule-surface application, not a benchmark of
   the present gas-phase implementation.

Published agreement for those systems does not validate an arbitrary PES,
reference geometry, excited mode, beta or bead count. Report the preparation
convention and convergence checks with the resulting cross sections, and keep
any comparison with exact quantum dynamics or experiment specific to the
reaction and observable tested.
