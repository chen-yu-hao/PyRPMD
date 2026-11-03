# Classical IESH and multistate MASH

`task="nonadiabatic"` accepts `method="iesh"` and
`method="multistate-mash"`. Both currently require **one bead, real Hamiltonians
and nondegenerate adiabatic levels**. They use separate states and estimators;
neither name is an alias for the existing two-state `rp-mash` workflow. All
coordinates, matrix elements, gradients, momenta and model parameters below use
atomic units. Explicit unit strings remain available for masses, temperature,
time and the existing quantity fields.

| Method | Electronic state | Nuclear potential | Reported electronic population |
| --- | --- | --- | --- |
| `iesh` | Orthonormal orbitals of one spinless Slater determinant | Bare nuclear energy plus the sum of occupied adiabatic orbital energies | One-particle density; populations sum to the electron count |
| `multistate-mash` | One normalized complex mapping vector | Adiabatic surface with largest mapping population | Runeson–Manolopoulos signed observable estimator; populations sum to one |
| Existing `rp-mash` | Two-state centroid spin | CA-RP-MASH surface for the whole ring | Adiabatic active-state indicator with its own initial importance sampling |
| `rp-mash-ba` | One common two-state adiabatic spin | BA-RP-MASH surface for the whole ring; bead-averaged electronic evolution | Adiabatic active-state indicator with population importance sampling |

Completion of a workflow produces `completed=true` and
`physical_converged=false`. Numerical identities, finite-model reference
comparisons and Monte Carlo uncertainty do not certify physical convergence.

## Run and resume

```bash
pyrpmd validate examples/nonadiabatic/iesh.inp --check-pes
pyrpmd run examples/nonadiabatic/iesh.inp --output runs/iesh
pyrpmd resume runs/iesh
pyrpmd analyze runs/iesh --recompute

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
`bare_potential(q)` returns `PESResult(energy, forces)` and is added once,
independently of the electron count. Its forces follow the ordinary
$-\nabla U_0$ contract. The CLI restricts this model to IESH, so another method
cannot silently omit its separate bare potential. A custom IESH provider must
also describe a one-electron matrix; a molecular many-electron state matrix
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

Velocity Verlet advances the nuclei. Electronic midpoint substeps integrate
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

## T25: two-state bead-approximation RP-MASH

The optional T25 extension selects **(b), bead-approximation RP-MASH**. Use
`method="rp-mash-ba"` or the Python
`pyrpmd.rp_mash_ba.RPMASHBADynamics` API for this two-state method.
It follows the BA equations of
[Cao, Suarez and Kretchmer (2025 preprint)](https://doi.org/10.26434/chemrxiv-2025-xps2g),
Eqs. (34)–(43). The spin precession averages each bead's gap and velocity–NAC
contraction. An equator event switches all beads simultaneously and balances
the full ring energy gap by an impulse along the extended bead-space NAC.

This is a distinct approximation from the existing centroid `rp-mash`, with
its own method identity and adiabatic spin checkpoint representation. The
classical `multistate-mash` method above still requires one bead; this addition
does not infer a multistate ring-polymer generalization. It also does not
change IESH or add a TDDFT provider. See the
[two-state spin guide](nonadiabatic-spin-methods.md#ba-rp-mash) for the equations,
population preparation, restart behavior, numerical controls and explicit
limitations.

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
