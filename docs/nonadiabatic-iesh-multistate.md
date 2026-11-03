# Classical and ring-polymer IESH, and multistate MASH

`task="nonadiabatic"` accepts `method="iesh"`, `method="rp-iesh"` and
`method="multistate-mash"`. Classical IESH and multistate MASH require **one
bead**. RP-IESH accepts a nuclear ring polymer and uses a shared electronic
determinant at its centroid. These methods require **real Hamiltonians and
nondegenerate adiabatic levels**. They use separate states and estimators;
their names are not aliases for the existing two-state `rp-mash` workflow. All
coordinates, matrix elements, gradients, momenta and model parameters below use
atomic units. Explicit unit strings remain available for masses, temperature,
time and the existing quantity fields.

| Method | Electronic state | Nuclear potential | Reported electronic population |
| --- | --- | --- | --- |
| `iesh` | Orthonormal orbitals of one spinless Slater determinant | Bare nuclear energy plus the sum of occupied adiabatic orbital energies | One-particle density; populations sum to the electron count |
| `rp-iesh` | One spinless Slater determinant shared by all beads; centroid electronic propagation | Ring springs and beadwise bare/occupied-orbital energies | One-particle density; adiabatic populations use the centroid basis and sum to the electron count |
| `multistate-mash` | One normalized complex mapping vector | Adiabatic surface with largest mapping population | Runeson–Manolopoulos signed observable estimator; populations sum to one |
| Existing `rp-mash` | Two-state centroid spin | CA-RP-MASH surface for the whole ring | Adiabatic active-state indicator with its own initial importance sampling |

Completion of a workflow produces `completed=true` and
`physical_converged=false`. Numerical identities, finite-model reference
comparisons and Monte Carlo uncertainty do not certify physical convergence.

## Run and resume

```bash
pyrpmd validate examples/nonadiabatic/iesh.inp --check-pes
pyrpmd run examples/nonadiabatic/iesh.inp --output runs/iesh
pyrpmd resume runs/iesh
pyrpmd analyze runs/iesh --recompute

pyrpmd validate examples/nonadiabatic/rp_iesh.inp --check-pes
pyrpmd run examples/nonadiabatic/rp_iesh.inp --output runs/rp-iesh
pyrpmd resume runs/rp-iesh
pyrpmd analyze runs/rp-iesh --recompute

pyrpmd validate examples/nonadiabatic/multistate_mash.inp --check-pes
pyrpmd run examples/nonadiabatic/multistate_mash.inp --output runs/multistate-mash
pyrpmd resume runs/multistate-mash
```

These small inputs demonstrate the complete public workflow. They intentionally
use `native=false`; IESH can also request the compiled nuclear free-step kernel.
Neither electronic solver is a native full-trajectory or GPU implementation.
Trajectory-level process workers are supported, with the same logical random
streams as serial execution. `dynamics.block_steps` controls checkpoint cadence.

## IESH definition and metal model

The implementation follows [Shenvi, Roy and Tully (2009)](https://doi.org/10.1063/1.3125436)
and the implementation analysis of
[Gardner et al. (2023)](https://doi.org/10.1063/5.0137137),
[arXiv:2212.00518v2](https://arxiv.org/abs/2212.00518v2).
One electron occupies each selected orbital; no spin-degeneracy factor is added.
The nuclei move on

$$
E_s(q,p)=\sum_a\frac{p_a^2}{2m_a}+U_0(q)+\sum_{i\in s}\epsilon_i(q).
$$

`NewnsAndersonModel` supplies an impurity level and a discretized metallic band.
For the selected Cartesian coordinate $x=q_{\mathrm{coordinate}}-\mathrm{center}$,

$$
h_{dd}=\epsilon_0+a x,\quad h_{kk}=\epsilon_k,\quad
h_{dk}=h_{kd}=\sqrt{\frac{\Gamma(x)w_k}{2\pi}},\quad
\Gamma(x)=\Gamma_0e^{b x},\quad U_0(x)=\tfrac12 Kx^2.
$$

The following parameters are part of the immutable, versioned model identity.

| Input parameter | Meaning |
| --- | --- |
| `nband` | Number of metal quadrature states; total orbital count is `nband+1` |
| `band_min`, `band_max` | Finite band edges in hartree |
| `chemical_potential` | Band quadrature splitting point, strictly inside the band |
| `discretization` | `gauss-legendre`, split at the chemical potential, or uniform `midpoint` |
| `level_energy`, `level_slope` | Impurity energy $\epsilon_0$ and derivative $a$ |
| `hybridization`, `hybridization_slope` | $\Gamma_0\ge0$ and logarithmic derivative $b$ |
| `bare_force_constant` | Bare nuclear force constant $K\ge0$, in hartree/bohr squared |
| `center`, `coordinate`, `natoms` | Reference position, flattened Cartesian index and atom count |

`evaluate(q)` returns a **one-electron** `DiabaticResult`. The scalar
`bare_potential(q)` returns `PESResult(energy, forces)` and is added once per
nuclear configuration, independently of the electron count. For `rp-iesh`
this means once per bead. Its forces follow the ordinary $-\nabla U_0$
contract. The CLI restricts this model to the IESH methods, so another method
cannot silently omit its separate bare potential. Both `iesh` and `rp-iesh`
accept this model. A custom IESH provider must also describe a one-electron
matrix; a molecular many-electron state matrix
does not acquire an orbital interpretation merely by choosing this method.

`dynamics.nelectrons` sets a fixed electron count (default
`max(model.nstates // 2, 1)`). `initial.occupied` is an ordered list of distinct
zero-based adiabatic orbital indices. Its default fills the lowest
`nelectrons` levels. The band `chemical_potential` does **not** automatically
sample a Fermi–Dirac distribution or choose the particle number. The `beta`
input controls the requested nuclear preparation; it does not thermalize the
electronic determinant. The implemented workflow has no electronic thermostat.

### Determinant current and momentum change

Electronic orbitals are stored as complex128 columns $C$ in a fixed diabatic
basis and evolved with the same unitary midpoint propagator. In the current
adiabatic basis, let $S=C_{s,:}$. Replacing occupied row $r$ by an empty orbital
$j$ gives the determinant ratio

$$
R_{jr}=\frac{\det S_{r\to j}}{\det S}=(CS^{-1})_{jr}.
$$

With $d_{ij}=\langle i|\nabla j\rangle$ and $D_{ij}=v\cdot d_{ij}$, the
positive outgoing current for $s_r\to j$ is
$2\operatorname{Re}(R_{jr})D_{s_rj}$. The implementation obtains all ratios
from one linear solve, preserves the occupied-row ordering and excludes
Pauli-forbidden targets. It does not form exponentially small source
determinants. Singular or numerically unresolved conditional overlaps fail
explicitly.

At $P=1$, velocity Verlet advances the nuclei. Electronic midpoint substeps integrate
the current along the nuclear drift, and at most one selected hop is applied
at the nuclear-step endpoint. Momentum changes along the NAC with the physical
inverse-mass metric. An accepted hop exactly balances the orbital energy gap
with kinetic energy, up to roundoff. An energetically frustrated upward hop
keeps both momenta and the occupied determinant, as in the original IESH
prescription. A zero NAC cannot cause an artificial accepted hop.

`electronic_substeps` defaults to 4. `max_hop_probability` is in $(0,1]$,
default 1; the example uses 0.2. Exceeding this bound raises an error and rolls
back state and RNG. Probabilities are never normalized to conceal an oversized
timestep. Hopping probabilities must also be **small and timestep-converged**:
increasing electronic substeps alone does not refine the endpoint hop timing.
`gap_tolerance` rejects unresolved adiabatic levels rather than regularizing NACs.

### IESH and electronic friction

The independent example `iesh_friction_reference.py` compares IESH with the
existing `ElectronicFrictionDynamics` at P=1. Both share the same finite-band
Hamiltonian and bare nuclear potential. The filled-ground-determinant surface
supplies the EF-RPMD force and the mechanical-energy baseline for both methods;
IESH follows its current occupied determinant after each hop. For a
constant $\Gamma$ and a cold spinless bath, its continuum reference uses

$$
A(\epsilon;x)=\frac{\Gamma/2}{\pi[(\epsilon-h_{dd})^2+(\Gamma/2)^2]},\qquad
\eta(x)=\pi\left(\frac{dh_{dd}}{dx}\right)^2 A(\mu;x)^2.
$$

This is the zero-temperature specialization of Gardner et al., Eqs. 33–34.
$\eta$ is a physical friction tensor component (mass/time), not an inverse-time
rate. The OU reference has an explicitly reported large finite `beta`.

The comparison checks mechanical energy lost into electronic excitation.
Slow nuclei, adequate band width and level resolution, and times before bath
recurrences are needed. Sending $\Gamma$ to zero at fixed nuclear velocity is
not the electronic-friction limit; the paper itself finds discrepancies in
that regime. Finite electronic levels quantize IESH energy exchange, so
individual trajectories need not resemble smooth friction trajectories.

## Centroid ring-polymer IESH (T25)

`method="rp-iesh"` implements the centroid RPIESH construction described in
[Gardner's 2023 doctoral thesis](https://wrap.warwick.ac.uk/id/eprint/185255/),
Section VI.1.1, Eqs. VI.1–VI.2, together with the centroid approximation in
Section II.5.2 and the IESH determinant currents cited above. The direct
RP-IESH provenance is an archived doctoral thesis. The method is a research
approximation; the thesis reports appreciable errors in some desorption
regimes. No quantum detailed-balance or general physical-convergence guarantee
is implied.

### State and propagation

The Python classes are `pyrpmd.rp_iesh.RPIESHState` and `RPIESHDynamics`.
Positions and momenta have shape `(P, natoms, 3)`. The complex128 orbital
matrix still has shape `(nstates, nelectrons)` and the ordered occupation
vector has length `nelectrons`: neither has an independent bead axis. All
beads follow the same occupied adiabatic-orbital indices. Electronic
propagation, determinant currents, candidate selection and the NAC direction
are evaluated at the centroid. Adiabatic electronic populations also use
that centroid basis; the fixed-diabatic density is the shared projector
$CC^\dagger$ and has trace `nelectrons`.

For an occupied index set $s$, define the bead potential

$$
W_s(q_b)=U_0(q_b)+\sum_{i\in s}\epsilon_i(q_b).
$$

Away from hopping events, the nuclear forces correspond to

$$
H_P=\sum_{b=1}^P\left[
\sum_a\frac{p_{a,b}^2}{2m_a}
+\sum_a\frac{m_a\omega_P^2}{2}(q_{a,b}-q_{a,b-1})^2
+W_s(q_b)\right],\qquad \omega_P=P/\beta,
$$

with cyclic bead indices and $\hbar=1$ in atomic units. The force returned by
the potential provider remains an ordinary physical force; it is not divided
by $P$. In the fixed diabatic basis the common orbital matrix obeys
$i\dot C=h(\bar q)C$, where $\bar q=P^{-1}\sum_bq_b$.

`rp-iesh` uses the IESH controls `nelectrons`, `electronic_substeps`,
`max_hop_probability` and `gap_tolerance`. Its hopping convention is fixed:
centroid energy gap, uniform bead momentum shift, and rejected frustrated
hops with unchanged momenta. There is no additional decoherence prescription,
independent-bead electronic determinant, complex-SOC treatment, or selectable
full-ring-energy rescaling. Unsupported options fail explicitly. Near-degenerate
adiabatic levels and unresolved conditional determinant overlaps retain the
IESH failure policy.

### What a hop conserves

The distinction between centroid and full-ring energy is observable even if
the propagation timestep is very small. For a single-electron hop $i\to j$,
write

$$
\Delta_c=\epsilon_j(\bar q)-\epsilon_i(\bar q),\qquad
\Delta_b=\epsilon_j(q_b)-\epsilon_i(q_b).
$$

The accepted hop changes every bead's momentum by the same vector,
$p_b'=p_b+\Delta p$, with $\Delta p$ along the centroid NAC. Its magnitude
uses the physical inverse-mass metric to satisfy

$$
\sum_a\frac{(\bar p_a+\Delta p_a)^2-\bar p_a^2}{2m_a}
=-\Delta_c,\qquad \bar p=P^{-1}\sum_bp_b.
$$

There is no additional division of $\Delta p$ by $P$. The internal-mode
momenta are unchanged by this common shift. Summing its kinetic-energy change
over beads gives $-P\Delta_c$; the spring and bare potentials do not change
instantaneously. Therefore direct subtraction of the pre-hop and post-hop
Hamiltonians gives

$$
\boxed{\Delta H_P^{\rm hop}=\sum_{b=1}^P\Delta_b-P\Delta_c.}
$$

This algebraic diagnostic is generally nonzero for a spread-out ring on
nonlinear adiabatic surfaces. It vanishes for $P=1$, coincident beads and
identical bead/centroid gaps. Replacing $\Delta_c$ by the bead-average gap
would define a different hopping rule.

`RPIESHHopResult.centroid_energy_error` reports the residual in the centroid
hop balance; its `energy_error` attribute has the same meaning.
`ring_energy_change` reports the full-ring jump. Neither field certifies
conservation along the surrounding finite-timestep nuclear trajectory.
The RP state, checkpoint and observed result arrays additionally retain
`hop_energy_change`, the cumulative signed full-ring change from accepted
hops, and `hop_energy_change_abs`, the sum of its absolute increments.
`energy_drift_minus_hop_changes` subtracts those signed increments from the
change in the reported full-ring energy. It separates the explicit hopping
contribution from the remaining numerical drift; it does not repair the
underlying approximation or enforce a conservation law.

Result metadata identifies this choice using
`method_variant="gardner-2023-centroid"`, `method_status="research"`,
`electronic_geometry="centroid"`,
`hop_rescaling="centroid_gap_uniform_bead_momentum_shift"`,
`frustrated_hops="keep"`,
`energy_estimator="ring_polymer_hamiltonian_sum"`, and
`energy_conservation="centroid_hop_balance_only"`.
The full-ring energy is a sum over beads. Dividing it by the bead count for
display does not turn it into the centroid Hamiltonian.

### Preparation, identity and restart

The checked-in `rp_iesh.inp` uses eight beads, four trajectories and a short
propagation to demonstrate validation, output and restart. Its
`harmonic_ring` preparation samples an independent harmonic reference in the
selected nuclear coordinate. The determinant is then initialized from the
specified occupied adiabatic orbitals at each sampled centroid. This is not
the coupled electron–nuclear thermal state. The fixed electronic occupation
and particle count do not become a Fermi–Dirac sample when `beta` is changed.

The `rp-iesh` method name, bead count, spring temperature, preparation,
scientific controls and fixed hopping convention participate in the new
method's identity. Existing `iesh` identities and checkpoint layouts are
preserved. An RP checkpoint records the common electronic matrix and
occupations, all bead coordinates and momenta, both accumulated hop-energy
diagnostics and the existing random streams. Restart does not reinterpret a
classical IESH checkpoint as a ring-polymer state.

### Validation scope

Validation separates independent references and exact limiting cases from
accuracy claims about approximate coupled dynamics:

- The $P=1$ regression compares matched initial states and random streams
  against classical IESH, including hopping and continuation.
- Finite differences check the beadwise occupied-state force and centroid
  overlap derivative. Direct before/after energy evaluation checks the
  full-ring hop-jump formula independently of its cumulative accounting.
- A small fixed-electron-number Fock-space Hamiltonian can be propagated
  with the one-dimensional Fourier solver. Its creation/annihilation signs
  and one-particle density provide an independent many-electron reference.
  A coordinate-independent electronic Hamiltonian supplies an exactly
  separable propagation limit; coupled finite-bath comparisons additionally
  contain RP-IESH approximation error and trajectory sampling error.
- Interrupted-run tests compare states, observations and RNG streams with
  uninterrupted propagation and validate corrupted-checkpoint rejection.

These checks support implementation correctness within their tested limits.
Finite-bead, finite-band and finite-ensemble results retain
`physical_converged=false`. Quantitative metal-surface predictions require
separate timestep, bead-count, band-discretization, initialization and
statistical studies. A bosonic spin-boson HEOM result is not automatically a
reference for the fermionic electronic bath of a Newns–Anderson model.

### Executed T25 operational check: 2026-11-03

The checked-in eight-bead example passed `validate --check-pes`, `run`,
completed-run `resume` and `analyze --recompute`. Both replay commands
preserved all 27 result arrays exactly and retained the physics fingerprint.
The four trajectories produced nine observations each. The largest
matrix-gradient finite-difference error was $1.19\times10^{-13}$
hartree/bohr, and the maximum full-ring energy change at saved frames was
$6.63\times10^{-11}$ hartree. The cumulative hop-energy diagnostics were
zero for this short example, so it does not exercise nonzero jump accounting.
OpenMP and OpenBLAS each used one thread. This operational check keeps
`physical_converged=false`; independent references and dedicated hop tests
are needed to assess the remaining dynamics.

### Executed finite-bath/Fourier comparison: 2026-11-03

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python \
  examples/nonadiabatic/rp_iesh_reference.py --trajectories 128
```

This independent reference lifts a three-orbital Newns–Anderson Hamiltonian
into its two-electron Fock sector using explicit creation/annihilation signs,
then propagates the nuclear coordinate with the Fourier solver. The reported
populations are the diagonal of the one-particle density and sum to two.
The Fock-basis occupation probabilities are a different observable.

The run used 128 trajectories, eight beads, seed 20261103, nuclear mass 1000,
`beta=80`, harmonic preparation frequency 0.02, RP timestep 0.2 and duration
20, all in atomic units. The initial electronic state was a fixed-diabatic
Slater determinant occupying orbitals 0 and 1, factorized from the harmonic
nuclear thermal state. The Python reference samples initial active adiabatic
occupations from the squared Slater minors at each sampled centroid. This
preparation is different from the specified adiabatic occupation in the CLI
input example above.

The final refined Fourier calculation used 384 grid points on $[-3,3)$ bohr
and timestep 0.025 a.u. Its harmonic thermal-state truncation omitted
$3.11\times10^{-13}$ probability. Separate timestep, grid and box refinements
changed the final density matrix by at most $5.19\times10^{-9}$,
$3.28\times10^{-13}$ and $3.48\times10^{-13}$, respectively.

| Fixed-diabatic orbital | Fourier population | RP-IESH population | RP-IESH pointwise SEM | Difference / SEM |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.802580081 | 0.803248711 | 0.000813479 | 0.822 |
| 1 | 0.987495972 | 0.987477084 | 0.000009219 | -2.049 |
| 2 | 0.209923947 | 0.209274205 | 0.000815501 | -0.797 |

The largest population discrepancy is $6.69\times10^{-4}$. The
middle-orbital difference is approximately 2.05 sampling SEMs. SEM measures
independent-trajectory sampling uncertainty only; it excludes finite-bead,
finite-timestep and approximate-method errors. The full complex densities are
retained in the reference record, and the population table does not establish
coherence accuracy. This finite isolated bath is not a metallic continuum.

The ensemble contained one accepted and one frustrated hop. The accepted
hop had a full-ring energy change of $8.84\times10^{-4}$ hartree while the
maximum centroid hop-balance residual was $2.78\times10^{-17}$ hartree.
After subtracting cumulative hop changes, the largest remaining absolute
final-minus-initial energy change among trajectories was
$2.17\times10^{-6}$ hartree. These values illustrate why
centroid balance and full-ring energy changes are reported separately.

The interrupted-run regression additionally exercises a nonzero hop-energy
accumulator between observations: a stochastic hop occurs at step 28, the
checkpoint is saved at step 33, and the last saved observation is step 26.
Restart must preserve the newer accumulator even though the observation
history still contains zero accumulated hop energy. The resumed checkpoint
payload and result arrays match uninterrupted propagation bit for bit.

[Full two-electron/Fourier result and model parameters](validation/t25-rp-iesh/fock-fourier-reference.json).
The result retains `physical_converged=false`; RP timestep, bead-count and
ensemble convergence have not been established by this short comparison.

### Single-thread step timings: 2026-11-03

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python \
  examples/nonadiabatic/rp_iesh_performance.py --steps 100 --repeats 5 --warmups 10
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python \
  examples/nonadiabatic/rp_iesh_performance.py --native --steps 100 --repeats 5 --warmups 10
```

The fixed model has 12 bath orbitals, 13 total one-electron states and six
electrons. Timings cover the complete validated propagation step, including
bead forces, electronic substeps and hop decisions. Initialization,
observations, checkpointing and output are excluded. Each repeat restores
the same initial state and RNG. `--native` requires the compiled nuclear map;
the electronic algebra remains NumPy in both cases.

The environment was Linux x86_64 with nine logical CPUs visible, Python
3.12.14, NumPy 2.3.5 and OpenBLAS 0.3.30. The host CPU model was not exposed.
The nuclear kernel and every detected BLAS/OpenMP pool used one thread.
Other validation jobs in this session were stopped during the measurements;
shared infrastructure can still affect wall-clock timings.

| Beads | Python nuclear map, median ms/step | Native nuclear map, median ms/step | Python repeat range | Native repeat range |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 11.306 | 10.040 | 10.463–12.723 | 9.103–12.413 |
| 8 | 14.714 | 11.576 | 11.777–16.300 | 11.492–15.764 |
| 32 | 22.808 | 27.990 | 20.158–31.994 | 13.606–43.566 |

Ranges are also in ms/step. The repeat spread, especially at 32 beads,
does not support a general speedup claim for the native nuclear map. These
small CPU cases do not measure GPU throughput or a converged large metallic
bath. Complete parameters, repeats and thread-pool records are retained in
the [Python timing record](validation/t25-rp-iesh/performance-python.json)
and [native timing record](validation/t25-rp-iesh/performance-native.json).

### Primary references for the RP-IESH extension

- J. J. Gardner, *Simulation of nonadiabatic molecular dynamics at metal
  surfaces*, PhD thesis, University of Warwick (2023), Sections II.5.2 and
  VI.1.1, [archived thesis](https://wrap.warwick.ac.uk/id/eprint/185255/).
- N. Shenvi, S. Roy and J. C. Tully, *Nonadiabatic dynamics at metal surfaces:
  independent-electron surface hopping*, J. Chem. Phys. **130**, 174107
  (2009), [DOI: 10.1063/1.3125436](https://doi.org/10.1063/1.3125436).
- J. Gardner, D. Corken, S. M. Janke, S. Habershon and R. J. Maurer,
  *Efficient implementation and performance analysis of the independent
  electron surface hopping method for dynamics at metal surfaces*,
  J. Chem. Phys. **158**, 064101 (2023),
  [DOI: 10.1063/5.0137137](https://doi.org/10.1063/5.0137137).
- P. Shushkov, R. Li and J. C. Tully, *Ring polymer molecular dynamics with
  surface hopping*, J. Chem. Phys. **137**, 22A549 (2012),
  [DOI: 10.1063/1.4766449](https://doi.org/10.1063/1.4766449).
- F. A. Shakib and P. Huo, *Ring Polymer Surface Hopping: Incorporating
  Nuclear Quantum Effects into Nonadiabatic Molecular Dynamics Simulations*,
  J. Phys. Chem. Lett. **8**, 3073–3080 (2017),
  [DOI: 10.1021/acs.jpclett.7b01343](https://doi.org/10.1021/acs.jpclett.7b01343).

## Original 2023 multistate MASH

`MultistateMASHDynamics` implements
[Runeson and Manolopoulos, JCP 159, 094115 (2023)](https://doi.org/10.1063/5.0158147),
using the verified [arXiv:2305.08835v3](https://arxiv.org/abs/2305.08835v3)
formulation. This is the original multistate version, not unSMASH or a later
initialization variant. The active surface is the largest adiabatic
$|c_i|^2$. At a surface boundary the impulse uses the **full multistate**
coherence/NAC direction of Appendix E, including other electronic states.
Using only the direct source–target NAC is generally incorrect for three or
more states.

The fixed-diabatic amplitudes obey $\dot c=-iV(q)c$. A DOP853 solver and
sign-change event roots locate population boundaries. Accepted transitions
conserve the active-surface Hamiltonian. A frustrated transition reflects the
mass-weighted boundary-normal momentum, preserves its tangent components, and
leaves the electronic amplitudes continuous.

### Initial distribution and observable weights

For a requested pure electronic initial state, sample the complex unit sphere
uniformly conditional on that state's population being largest. The code uses
permutation symmetry to generate this measure without an $N$-fold rejection
cost. The public CLI prepares an **adiabatic** `initial.active_surface`.
The Python initializer additionally supports `initial_state` in
`basis="diabatic"`, and custom normalized coefficients.

For $N$ electronic states define $H_N=\sum_{k=1}^N 1/k$ and

$$
\alpha_N=\frac{N-1}{H_N-1},\qquad
\rho_{\mathrm{map}}=\alpha_Ncc^\dagger+\frac{1-\alpha_N}{N}I.
$$

This unit-trace matrix supplies populations, coherences and linear electronic
observables in the chosen basis. Individual samples can have negative
populations; they are retained without clipping. `active_populations()` is a
separate surface histogram and is not substituted for this estimator.

The continuous equations reduce to two-state MASH at $N=2$. The 2023 sampling
and observable prescription still differs from the original
Mannouch–Richardson formulation implemented by `MASHDynamics`. The regression
tests compare the **same mapping initial conditions** while reducing the
older endpoint integrator's timestep. Matching two independently chosen
default sampling distributions would test different observables. Existing
`MASHDynamics` and `RPMASHDynamics` retain their behavior.

### Integration and explicit restrictions

- `native=false` and `integrator="adaptive-dop853"` are required. The
  electronic and nuclear ODEs run on CPU in double precision.
- `dt` is the interval advanced by one `step()`. The CLI observes every
  `sample_stride` steps, including the initial and final states.
  `rtol` (default $10^{-9}$),
  `atol` ($10^{-11}$), and `max_step` (`dt/4`) control the adaptive solve.
  A relative tolerance below SciPy's representable bound is rejected rather
  than silently increased.
- Sign-change detection can miss an even number of crossings within an
  excessively long internal step. Check `max_step` convergence as well as ODE
  tolerances; adaptive integration is not proof of resolving every possible
  crossing.
- `max_events` defaults to 64 per `dt` propagation step. Exceeding it, an ambiguous
  simultaneous triple boundary, a degenerate energy level or a nonfinite
  direction fails without publishing a partial state.
- No P>1 or complex-SOC extension is inferred. The original 2023 method is not
  size consistent when uncoupled spectator states are added.

## Checkpoint, identity and output contracts

Every new scientific option and the complete model identity participates in
the existing `scientific_config` fingerprint. The existing eight methods'
resolved identities are protected by pre-change hash regressions. Worker
placement and checkpoint cadence retain the existing execution-only semantics.

The dedicated trajectory checkpoint uses the existing atomic, integrity-checked
`atomic_savez` implementation. It contains nuclear state, fixed-diabatic
electronic coefficients, occupied orbitals or active surface, MASH sampling
mode, all three random streams, and committed observations. No hidden adiabatic
phase or previous-coordinate gauge is needed. Each failed step is
transactional. Restart validates array precision, field inventory, fingerprint,
preparation, clocks, RNGs, density algebra and occupation constraints before
restoring any random stream. It also checks state/observation consistency
whenever an observation is saved at the checkpoint's current step.

The new result schema `PyRPMD.multistate_nonadiabatic_ensemble.v1` keeps:

- `populations`, `diabatic_density` and their means in the fixed diabatic basis;
- weighted or one-particle `adiabatic_populations`, with explicitly named
  estimator metadata;
- `active_electronic_states` and separate `active_population_mean/counts`;
- energy, electronic norms, centroid coordinates and momenta per trajectory;
- independent-trajectory pointwise SEM, including real and imaginary
  coherence errors, when at least two trajectories are available.

IESH normalization is `nelectrons`; MASH normalization is one. With one
trajectory SEM is unavailable, not zero. The ordinary `results/populations.csv`
and `results/nonadiabatic.npz` files remain the workflow output locations.

## Reproducible independent comparisons

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python \
  examples/nonadiabatic/multistate_mash_reference.py --trajectories 128
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python \
  examples/nonadiabatic/iesh_friction_reference.py --trajectories 32
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python \
  examples/nonadiabatic/iesh_performance.py
```

The MASH reference compares moving Gaussian nuclear packets on Tully I and a
coupled three-state LVC model with the independent `SplitOperator1D` Fourier
solver. The same pure diabatic state and positive Gaussian Wigner nuclear
distribution are used. Time and grid refinements of the Fourier calculation
are reported separately from trajectory SEM and approximate-method error.
Additional fast regressions compare both models' heavy-nucleus limit against
Fourier dynamics, with electronic/spatial quadrature instead of stochastic
acceptance thresholds.

The IESH kernel benchmark compares one simultaneous linear solve against
separate explicit Slater determinants. Its speedup applies only to that
operation; finite-band eigensystems and electronic/nuclear propagation remain
part of the whole-trajectory cost.

## Executed finite-model checks: 2026-11-02

The reference commands above were executed on an AMD EPYC 9V74 virtual machine
with 9 visible CPUs, Python 3.12.14 and NumPy 2.3.5. OpenMP and OpenBLAS each
used one thread. Other project tasks shared the machine, so elapsed timings
include scheduling contention. The exact model parameters, seeds and limits
are retained in the JSON records below.

### Moving-packet MASH / Fourier comparison

Both models used 128 independent trajectories, seed 20261102, nuclear mass 200,
position mean -3 and standard deviation 0.6, mean momentum 5 and Wigner momentum
standard deviation $1/1.2$. The electronic initial state was diabatic state 0.
The duration was 240 a.u.; MASH used `dt=4`, `max_step=4`, `rtol=1e-9`,
`atol=1e-11`. The final Fourier reference had 1024 grid points on $[-20,20)$
and timestep 0.5 a.u.

| Model / state | Fourier population | MASH population | MASH pointwise SEM |
| --- | ---: | ---: | ---: |
| Tully I / 0 | 0.85830732 | 0.92927875 | 0.03767301 |
| Tully I / 1 | 0.14169268 | 0.07072125 | 0.03767301 |
| Three-state LVC / 0 | 0.58697318 | 0.53137142 | 0.05186512 |
| Three-state LVC / 1 | 0.36790382 | 0.37616771 | 0.05137989 |
| Three-state LVC / 2 | 0.04512300 | 0.09246087 | 0.03914979 |

The largest absolute differences are 0.07097143 and 0.05560175 respectively;
every displayed discrepancy is below two corresponding sampling SEMs. This
finite ensemble cannot separate method bias from sampling error. In particular,
the Tully initial sample mean is `[1.07476108, -0.07476108]`; those signed
finite-sample fluctuations were retained, not clipped or renormalized.

Halving the Fourier timestep changed final populations by at most
$8.12\times10^{-7}$ (Tully) and $2.58\times10^{-7}$ (LVC); doubling the grid
changed them by $3.02\times10^{-8}$ and $1.28\times10^{-13}$.
The maximum final MASH energy errors were $1.04\times10^{-9}$ and
$7.25\times10^{-11}$ hartree. The ensembles contained 122 and 192 population
boundary events. These checks do not establish large-ensemble, all-step-size
or physical convergence.

[Full MASH/Fourier output](validation/t12-nonadiabatic-2026-11-02/mash-reference.json).

### Cold finite-band IESH / electronic-friction comparison

The comparison used 63 midpoint bath states plus one impurity, 32 electrons,
band edges $\pm0.25$ hartree, $\Gamma=0.05$ hartree, impurity slope
0.02 hartree/bohr, and no bare harmonic restoring force. All 32 trajectories
started at $x=-1$, $p_x=20$ with mass 2000, `beta=1e8`, timestep 1 and two
electronic substeps, and ran for 200 a.u.

| Observable | Mean (hartree) | Independent-trajectory SEM (hartree) |
| --- | ---: | ---: |
| IESH mechanical-energy loss relative to the ground surface | 0.00184354 | 0.00087276 |
| P=1 EF-RPMD loss on that ground surface | 0.00266015 | 0.00000144 |
| IESH final electronic excitation energy | 0.00184353 | 0.00087276 |

Both methods show energy dissipation of the same order, with a difference below
one IESH SEM. Seven hops were accepted; none was energetically frustrated.
The maximum absolute final IESH total-energy drift across trajectories was
$6.95\times10^{-9}$ hartree, much smaller than the measured loss, and the largest step hopping probability was
0.04514. The uniform-band recurrence estimate is 791.7 a.u. and the initial
dimensionless rate $|v\,dh/dx|/\Gamma^2$ is 0.08.

This is a qualitative finite-band check with few hopping events, not a
converged continuum-friction result. More trajectories, bath resolution,
band-width and timestep studies remain necessary for quantitative conclusions.

[Full IESH/friction output](validation/t12-nonadiabatic-2026-11-02/iesh-friction-reference.json).

### Performance scope

For 64 electronic states, 32 electrons and all 1024 allowed single excitations,
the 15-repeat median time was **141.96 microseconds** for the single linear
solve and **33.789 milliseconds** for separate explicit determinants, a
**238.02-fold kernel speedup**. The maximum absolute ratio difference was
$6.82\times10^{-14}$. Both paths used the same orbitals and included their
reported validation work; this is not a claimed full-trajectory speedup.

The moving-packet experiments measured 33.09 ms (Tully) and 36.68 ms (LVC)
per MASH outer propagation step. The 64-orbital metal calculation measured
55.03 ms per IESH step and 6.22 ms per P=1 EF-RPMD step. These single-thread
CPU reference timings come from a contended shared machine and compare
different algorithms, not equivalent computational workloads.

[Kernel timing record](validation/t12-nonadiabatic-2026-11-02/iesh-kernel-performance.json)
and [environment](validation/t12-nonadiabatic-2026-11-02/environment.json).

### Regression and public-workflow checks

The implementation started from main commit `8cc402f`. Before editing, the
existing MASH, RP-MASH, nonadiabatic workflow/input/checkpoint and electronic
friction regression group passed **301 tests in 25.96 s**. The exact baseline
command and result are retained with the validation records.

The focused checks below are overlapping groups, not additive test counts.
Each command used `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` and the Python
interpreter recorded in `environment.json`.

| Test selection | Result |
| --- | --- |
| `tests/test_iesh.py` | 49 passed in 2.77 s |
| `tests/test_multistate_mash.py tests/test_mash.py tests/test_rp_mash.py` | 58 passed in 39.25 s |
| `tests/test_nonadiabatic_t12_workflow.py tests/test_nonadiabatic_extension_input.py` | 103 passed in 19.44 s |
| `tests/test_multistate_mash_fourier.py` | 2 passed in 6.30 s |

Both checked-in input examples passed `pyrpmd validate --check-pes`, `run` and
`resume`. Completed-run resume preserved populations and physics fingerprints
exactly. The largest matrix-gradient finite-difference errors were
$3.21\times10^{-14}$ for IESH and $5.94\times10^{-14}$ for multistate MASH.
The workflow regressions additionally interrupt unfinished runs and compare
resumed histories and random streams bit for bit, compare serial and process
workers, reject corrupted checkpoints, and protect all eight legacy method
identities.

[Baseline output](validation/t12-nonadiabatic-2026-11-02/baseline-pytest.log),
[workflow output](validation/t12-nonadiabatic-2026-11-02/workflow-pytest.log),
[Fourier regression output](validation/t12-nonadiabatic-2026-11-02/fourier-pytest.log),
and [CLI verification](validation/t12-nonadiabatic-2026-11-02/cli-verification.json).

The full regression environment must allow local IPC sockets: the restrictive
execution sandbox denies `socket(AF_INET, SOCK_STREAM)` and caused twelve
existing i-PI tests to fail with `PermissionError`. The identical i-PI file
passed all **19 tests in 4.43 s** with that restriction lifted; no i-PI code
was changed. See the [i-PI environment check](validation/t12-nonadiabatic-2026-11-02/ipi-pytest.log).

The final full-suite command was:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ../t12-venv/bin/python -m pytest -q
```

It exited successfully with:

```text
8247 passed, 225 skipped, 51 warnings, 486 subtests passed in 1573.77s (0:26:13)
```

The [complete pytest output](validation/t12-nonadiabatic-2026-11-02/pytest.log)
retains the warning and optional-dependency/opt-in skip reasons. No numerical
result is declared physically converged by this software regression result.

Documentation is checked with the strict command:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ../t12-venv/bin/python \
  -m sphinx -W --keep-going -b html docs docs/_build/html
```
