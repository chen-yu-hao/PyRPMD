# H + H2 initial-preparation study: protocol and reproducibility

This follow-up examines the sensitivity of BKMP2 H + H2(v = 0, 1; j = 0)
scattering to the initial molecular preparation. It adds full-potential
action-quantized QCT, removes the published centroid-bond cutoff in a separate
NE-RPMD control, and tests an experimental isolated-molecule potential switch.
The calculation matrix was fixed before comparing the new cross sections.
No parameter is chosen to fit the published curves.

Read this document with the [original reproduction protocol](STATE_SELECTED_REPRODUCTION.md),
the [PES provenance](SOURCE.md), and the machine-readable records produced by
the commands below. This document defines the experiment; results belong in
the generated analysis. A completed software validation or an isolated
preparation audit does not establish reproduction of a scattering curve.

## What is being compared

The original reference is Marjollet and Welsch,
[IJQC 121, e26447 (2021)](https://doi.org/10.1002/qua.26447), following their
[JCP 152, 194113 (2020)](https://doi.org/10.1063/5.0004179) method paper.
The existing benchmark uses their triatomic preparation convention, with
explicit implementation choices recorded in the original protocol.

Three checks made before this follow-up delimit its purpose:

- The harmonic sampler's coordinate and momentum covariances agree with the
  normal-mode preparation in Marjollet, Inhester and Welsch,
  [JCP 156, 044101 (2022), Eqs. (3)–(7)](https://doi.org/10.1063/5.0076216),
  after accounting for the orthonormal, mass-weighted coordinate convention.
- The physical half-kick / exact free-ring step / physical half-kick propagator
  is consistent with the deterministic normal-mode splitting described by
  Ceriotti et al., [JCP 133, 124104 (2010), Sec. II A](https://doi.org/10.1063/1.3489925).
  This does not establish identity with every detail of another authors'
  implementation. No thermostat is used in the present propagation.
- The centroid-bond lower cutoff of 1.078 bohr is documented in the 2021
  article. Removing it changes a preparation assumption; it does not repair
  a demonstrated sampler or integrator error.

The questions are therefore whether a full-PES classical orbit changes the
QCT control, how the cutoff changes the accepted ensemble, and how a finite
potential-switch time changes the subsequent ensemble and scattering.
Smooth switching of a multi-bead chain is an experimental diagnostic here.
It is not an established correction to NE-RPMD or a guarantee of a quantum
vibrational pure state.

## Fixed workload and existing controls

Every variant is evaluated at the three nominal labels
`(v, Etot/eV) = (0, 0.55), (1, 0.90), (1, 1.10)`.
The H mass is 1.00782503223 u, the harmonic reference bond is
1.4014719755049045 bohr, and its frequency is 4407 cm^-1.
The impact-parameter area has `bmin = 0`, with `bmax = 3.5 bohr` for v = 0
and `4.0 bohr` for v = 1. These are the previously specified cutoffs;
this follow-up does not establish a new impact-parameter convergence result.

| New collision variant | Preparation | N per point | P | Trajectory IDs |
| --- | --- | ---: | ---: | --- |
| `ne-direct-none` | Harmonic NE ensemble, no centroid cutoff, instantaneous use of full PES | 1024 | 32 | 10000–11023 |
| `ne-switch200-none` | Same uncut draw, isolated BC switch for 200 fs, then collision | 1024 | 32 | 10000–11023 |
| `qct-action-fixed_collision` | Full-PES j = 0 action orbit, original collision energy | 4096 | 1 | 20000–24095 |
| `qct-action-fixed_total` | Same action orbit, collision energy adjusted to the stated total label | 4096 | 1 | 20000–24095 |

This is **30,720 new collision attempts**: 6,144 NE-RPMD attempts and 24,576
action-QCT attempts. The collision group uses a 0.02 fs timestep. The switch
collision cases start scattering immediately after the 200 fs preparation;
they do not include the audit's additional 100 fs hold.

Existing controls come from `runs/state_selected_reproduction/main/`.
The paper-conditioned NE control has **4096 original trajectories per point**.
For paired comparisons with the new NE cases, use precisely its first **1024
IDs, 10000–11023**. The analyzer plots this matched subset and retains the
full 4096 result as a separately labeled CSV entry. It does not rescale the
4096 counts or treat the subset as an independent second sample. The original
harmonic-prepared QCT control already has the same 4096 IDs as the new action
QCT cases. Reusing these controls adds no new collision attempts.

The independent isolated-diatom audit has four variants at all three points:

| Audit variant | Centroid cutoff | Switch time | NVE hold | Independent attempts per point |
| --- | --- | ---: | ---: | ---: |
| `direct_paper` | 1.078 bohr | 0 fs, sudden | 100 fs | 512 |
| `direct_none` | None | 0 fs, sudden | 100 fs | 512 |
| `switch50_none` | None | 50 fs | 100 fs | 512 |
| `switch200_none` | None | 200 fs | 100 fs | 512 |

These **6,144 isolated audit attempts** use P = 32 and IDs 910000–910511,
with the same ID range reused across controlled variants. They are not
collision samples and never enter a cross-section denominator. A 50 fs
switch is included in the isolated audit only, not in the collision matrix.
Small smoke runs have separate directories and are not pooled into either
formal workload.

## Full-PES, j = 0 action-QCT preparation

`diatomic_action.py` evaluates the isolated H2 asymptote of the real BKMP2
surface with both spectator-H distances fixed at 99 bohr. Write
`U(r) = Vfull(r) - Vmin`, and use the H2 reduced mass `mu = mH/2`.
For a bound orbit between its two turning points, the implemented action is

\[
J(E)=\frac{1}{2\pi}\oint p_r\,dr
    =\frac{1}{\pi}\int_{r_-}^{r_+}
      \sqrt{2\mu[E-U(r)]}\,dr = v+\tfrac12,
\]

in atomic units, with hbar = 1. There is no centrifugal term because j = 0.
This is leading-order EBK/WKB action quantization. Its energies are not exact
solutions of the radial Schrodinger equation, and its phase-space samples
are classical trajectories, not a quantum phase-space distribution.
The time-phase construction follows the diatomic QCT procedure described in
[the 2021 paper, Sec. 2.1](https://doi.org/10.1002/qua.26447), and in
[Karplus, Porter and Sharma, JCP 43, 3259 (1965)](https://doi.org/10.1063/1.1697301).

One uniform random phase selects **time divided by period** along the orbit.
It does not select a uniform bond length: the classical molecule spends more
time near a turning point. The implementation regularizes the turning-point
integrals, refines action and period quadrature, and builds a separately
refined inverse time map. Momentum is evaluated from the actual energy shell,
`p_r = +/-sqrt(2*mu*(E-U(r)))`, with the sign given by the direction of motion.
Unsupported domains and unconverged quadrature fail explicitly.

The original harmonic QCT phase and the new time phase use the same random
draw. Their mapping reproduces the original Cartesian phase point exactly
when the target is harmonic. On the anharmonic surface it supplies a defined
common-random-number comparison, not equality of the two molecular states.

### Collision energy and total-energy labels

Let `Elabel` denote the nominal `--etot` label, let
`Eharm(v) = (v+1/2)*Omega`, and let `Eaction(v)` be the full-PES action energy
above Vmin. The two action-QCT variants answer different questions:

| Convention | Actual collision energy | Actual asymptotic internal + collision energy |
| --- | --- | --- |
| `fixed_collision` | `Elabel - Eharm(v)` | `Elabel - Eharm(v) + Eaction(v)` |
| `fixed_total` | `Elabel - Eaction(v)` | `Elabel` |

Both use the same full-PES action orbit. The second increases the collision
energy by `Eharm - Eaction` for these two levels. The RNG identity continues
to use the nominal label, so changing the collision-energy convention does
not change the phase, orientation or impact-parameter draw.
"Asymptotic total" excludes the small residual interaction at finite launch
separation; the initial full collision Hamiltonian is recorded separately.

Useful energy-definition checks, in eV above the BKMP2 well minimum, are:

| v | 4407 cm^-1 harmonic label | Full-PES EBK action energy | Independent radial quantum reference |
| ---: | ---: | ---: | ---: |
| 0 | 0.2731991812 | 0.2691926995 | 0.2701867127 |
| 1 | 0.8195975437 | 0.7851776048 | 0.7860954928 |

The action values are recorded in each action case's `provenance.json` and
trajectory metadata. The quantum values come from the existing
`runs/state_selected_reproduction/h2_reference.json`, with grid/box checks
and Richardson extrapolation performed by `h2_reference.py`. These three
columns intentionally differ. None is a fitted scattering parameter, and
the numerical quantum reference does not turn an EBK orbit into a quantum
eigenstate.

## NE ensemble, cutoff, and pairing

The NE variants retain the original 2021 beta rule, including the printed
minus-sign convention documented in the original protocol. Beta is a fixed
auxiliary ensemble parameter in Hartree^-1 for each point; it is not an actual
collision or gas temperature. P and beta remain fixed throughout preparation,
hold and collision. All NE comparisons keep the original collision energy;
they do not rescale momenta to force an internal-energy diagnostic to match a
chosen target.

For orthonormal, mass-weighted ring coordinates of a harmonic molecular mode,
the unexcited preparation has

\[
\operatorname{Var}(Q_k)=\frac{P}{\beta(\Omega^2+\omega_k^2)},\qquad
\operatorname{Var}(\Pi_k)=\frac{P}{\beta},\qquad
\omega_k=\frac{2P}{\beta}\sin\frac{\pi k}{P}.
\]

A random-phase centroid displacement and momentum add nominal excitation
`v*Omega`. The finite-P harmonic ensemble already supplies its ground-state
contribution; a second `(1/2)*Omega` kick is not added. Conditioning can change
the accepted mean and phase distribution. These covariance checks validate
the specified harmonic preparation, not the transfer of that ensemble to an
anharmonic potential.

The benchmark removes the molecular beadwise center-of-mass motion and keeps
only the incident-axis internal ring motion of atom A, as specified in the
original triatomic driver. The `paper` condition rejects a draw if the
centroid H–H distance is below 1.078 bohr and redraws the **whole initial
state** before launch. `none` removes this acceptance condition. A short
individual bead bond is not equivalent to a short centroid bond.

All variants retain seed 20261031 and the original trajectory-specific RNG
identity. Initial sampling and launch randomness use separate child streams.
Consequently, removing the cutoff keeps the launch rotation and sampled
impact parameter for a given ID, but can change both the molecular state and
the incoming A ring on IDs that required a redraw. The axial launch distance,
`10 + Rg(A) + max(Rg(B), Rg(C))` bohr, can change as a consequence. Thus this
contrast tests the complete unconditioned versus conditioned ensemble; it
does not hold every initial A coordinate or separation fixed.

Within the **same** conditioning rule, the direct-versus-switch comparison
starts from the same sampled state. The switch keeps the incoming A ring,
orientation, beadwise BC center of mass, launch momentum, impact parameter and
the pre-switch axial separation. It does not recalculate that separation from
the final BC radius. This is an isolated BC preparation experiment followed
by the original collision launch, not an extra 200 fs of approach dynamics.
The analyzer verifies exact ID sets and serialized launch invariants; full
A coordinates/momenta and orientation are not stored for every successful
collision, so their preservation also rests on the frozen source and tests.

## Isolated BC switching and energy accounting

`diatomic_switch.py` loads a separate, content-addressed C++ helper. It uses
the existing BKMP2 radial C binding and existing free-ring Fourier routine;
the production propagation kernel is unchanged. Each native call performs
the complete switch and optional hold, with no Python callback at each step.
The allowed construction is a collinear j = 0 diatom with relative coordinates
`r_b` and canonical momenta `p_b = mu*dr_b/dt`. Molecular COM motion is absent
from this internal problem. The incoming atom is held unchanged while BC is
prepared, then the updated internal phase point is inserted back into its
original launch geometry.

The two potentials share the same absolute energy reference:

\[
V_h(r)=V_{\min}+\tfrac12\mu\Omega^2(r-r_{\rm eq})^2,\qquad
V_\lambda=(1-\lambda)V_h+\lambda V_f.
\]

For the recorded BKMP2 build, `Vmin = -0.17449577089730772 hartree` at the
supplied equilibrium bond. The 99-versus-150 bohr spectator-distance checks
give energy differences of approximately `3.173e-12 hartree` and bond-gradient
differences below `3e-17 hartree/bohr` at the five recorded test bonds. The
metadata retain all individual checks. This is a quantified asymptotic
approximation, not a claim that a finite spectator distance is mathematically
infinite.

For switch duration tau and `s = t/tau` in `[0,1]`,

\[
\lambda(t)=10s^3-15s^4+6s^5,\qquad
\dot\lambda(t)=30s^2(1-s)^2/\tau.
\]

The schedule is fixed at 0 before the switch and 1 after it. Each physical
force kick uses its actual endpoint lambda. Native BAB propagation uses an
exact free-ring middle step, with no thermostat, phase resampling, clipping,
reflection or post-switch energy adjustment. Switch and hold durations and
trace stride must be integer multiples of the preparation timestep, 0.02 fs.

Define bead averages

\[
K=\frac1P\sum_b\frac{p_b^2}{2\mu},\qquad
S=\frac1P\sum_b\frac{\mu(P/\beta)^2}{2}(r_b-r_{b+1})^2.
\]

The driven Hamiltonian is `Hlambda/P = K + mean(Vlambda) + S`.
Its change during a ramp includes physical switching work. The helper records

\[
w(t)=\int_0^t \dot\lambda(t')\,
       \overline{V_f-V_h}(t')\,dt',\qquad
\epsilon(t)=H_\lambda(t)/P-H_h(0)/P-w(t).
\]

Here `w` is already work **per bead**. The implementation integrates it using
the trapezoidal rule on every native force endpoint. The residual epsilon
contains both splitting and work-quadrature error; the uncorrected change in
H/P during the switch must not be called energy drift. For a sudden switch,
the t = 0 row has lambda = 1 and `w(0) = mean(Vf-Vh)` at unchanged coordinates
and momenta. During the subsequent full-PES hold, work is constant and NVE
drift is additionally measured relative to the switch endpoint.

The primitive internal diagnostic is a separate quantity:

\[
E_{\rm primitive}^{\rm full}=K+\overline{V_f}-V_{\min}-S.
\]

It uses the full target potential even during the ramp and can fluctuate
strongly or take negative trajectory values. It is neither the driven
Hamiltonian nor a state-population measurement. Its evolution need not be
flat when H/P is conserved. Native trace potentials and H/P include Vmin;
the primitive diagnostic and centroid harmonic energy have their respective
well constants removed. JSON metadata record this distinction and units.

The full PES is evaluated and checked even at lambda = 0. A domain failure
is reported, not silently discarded because the initial force is harmonic.
The helper retains the last accepted internal phase point in its failure
result; a failed preparation remains a failed attempt in the study ledger.

## Audit reduction and statistical unit

The compact audit wrapper is `preparation_audit.py`; its actual output root is
`runs/state_preparation_study/isolated_audit/`.
It computes three distinct time summaries for each successful trajectory:

1. `hold`: the full 100 fs hold, using the native trapezoidal integral at every
   accepted timestep. This is the primary time average.
2. `hold_trace_check`: the same interval integrated from the 0.2 fs trace,
   providing an independent output-stride check.
3. `final_50fs`: the final 50 fs, integrated from the 0.2 fs trace, as a separate
   late-time diagnostic.

Only after these reductions is the serialized trace thinned to 5 fs plus
the exact initial, switch and final endpoints. Each trajectory keeps a compact
`columns`/`units`/`records` representation. The manifest documents the energy
zeros, trace policy, wrapper hash and completion status.

An independent trajectory ID contributes one equally weighted time mean.
Ensemble SEM is the sample standard deviation of these trajectory means
(`ddof=1`) divided by `sqrt(n)`; it is undefined and stored as null for n < 2.
Correlated time points do not increase the number of independent samples.
Means are conditional on successful preparation/reduction, with all failures
still included in attempted/prepared/failed counts. Initial and final single
frames are reported separately and do not replace a hold average.

## Outcomes, unresolved trajectories, and uncertainty

The collision classifier and time budgets remain those of the original
benchmark: an initial 250 fs budget, continuation of the same phase point to
1000 fs if needed, 0.2 fs checks, and 1 fs persistence of a separated, outgoing
2+1 channel. A single centroid bond must be shorter than 2.5 bohr and the other
atom at least 10 bohr from the diatom COM with outward motion. The original
bond is nonreactive; either exchanged bond is reactive. Both exchange channels
are counted directly, without an extra factor of two.

Launched attempts are recorded as `product`, `nonreactive`, `unresolved`, or
`failed`. A failure is not a physical nonreaction; reaching the time cap is
not a numerical failure. Both remain in the denominator. The completed
direct-uncut v = 1 cases already contain three unresolved endpoints each,
six in total. Their large, separating three-fragment centroid geometries
are diagnostics, not grounds to retrospectively change the 2+1 classifier.
They remain unresolved in all estimates and comparisons.

For sampled area `A = pi*(bmax^2-bmin^2)`, N attempts, R confirmed reactions
and U unresolved-plus-failed attempts, the identified cross-section interval is

\[
\sigma\in A\,[R/N,(R+U)/N].
\]

The analyzer also provides exact two-sided 95% Clopper–Pearson intervals,
including a conservative envelope with the lower endpoint for R and upper
endpoint for R+U. A confirmed-reaction estimate and its interval are labeled
separately from the uncertainty over missing outcomes. All displayed cross
sections are converted from bohr^2 to Angstrom^2.

Paired contrasts are `variant - reference` on exactly equal ID sets. Definite
and possible gains and losses in reactivity are retained for every pair.
Each discordance probability receives a two-sided 97.5% Clopper–Pearson
interval; subtracting the intervals gives at least 95% simultaneous coverage
for that contrast by Bonferroni. Unknown pairs expand the definite/possible
counts and identification interval; they are not dropped or set to zero.
When any pair is unknown, the analyzer omits a single empirical delta point
estimate. The intervals are per contrast; no family-wise adjustment across
all twelve different contrasts is claimed.

The literature overlay uses the existing digitized 2021 NE-RPMD markers and
explicit linear interpolation at the requested energy labels. Its band is
a digitization tolerance, not a statistical confidence interval. Interpolation
model error and the published sampling uncertainty remain unquantified.
The QCT comparison contains the present calculations, not digitized external
QCT data. Neither agreement at a few points nor a shift after switching
establishes a general state-preparation method.

## Reproduce and inspect the study

Run from the repository root with the native PyRPMD extension built, a C++17
compiler, and GNU Fortran. The standalone switch currently uses a POSIX build
environment. NumPy and SciPy are runtime dependencies; pytest runs the tests
and Matplotlib produces the analysis figures. The commands below use fresh
output directories: existing case ledgers are refused rather than overwritten.
Choose worker counts to fit available CPU resources, especially when running
collision and audit groups concurrently.

```bash
python -m pip install -e '.[test]' matplotlib
export PYTHONPATH=src:.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python benchmarks/h3/build_pes.py
python benchmarks/h3/pes.py --check

python -m pytest -q tests/test_initial_states.py \
  tests/test_h3_diatomic_action.py tests/test_h3_diatomic_switch.py \
  tests/test_h3_preparation_study.py tests/test_h3_preparation_audit.py \
  tests/test_h3_preparation_analysis.py tests/test_drpmd_propagation_resume.py

python -u -m benchmarks.h3.run_preparation_study \
  --group fast --workers 8 --output runs/state_preparation_study
python -u -m benchmarks.h3.run_preparation_study \
  --group switch --workers 8 --output runs/state_preparation_study
python -u -m benchmarks.h3.preparation_audit \
  --output runs/state_preparation_study/isolated_audit --workers 4

python -m benchmarks.h3.analyze_preparation_study \
  --baseline runs/state_selected_reproduction \
  --study runs/state_preparation_study \
  --output runs/state_preparation_study/analysis
```

These commands run the collision groups and the compact audit separately.
For a fresh output root, `python -m benchmarks.h3.run_preparation_study
--group all --workers 8` runs the same twelve collision cases followed by the
compact audit. `--group audit` selects only that compact audit; it does not
serialize the unreduced dense traces.
The audit plan can be inspected without initializing the PES or writing files:

```bash
python -m benchmarks.h3.preparation_audit --dry-run
```

The analyzer requires the original baseline's complete raw ledgers,
configurations and final summaries, its `h2_reference.json`, and its
`literature_reference/` CSVs. Published aggregate cross sections alone cannot
reconstruct the matched 1024-ID subset. Reproduce or retain the baseline using
the original protocol before running this analysis. Only completed cases
with a final summary and their exact declared ID sets enter the analysis;
in-flight cases are explicitly listed as pending. The analyzer discovers
the compact cases under `isolated_audit/` and does not include smoke runs.

Each collision case records `config.json`, `preparation.json`,
`provenance.json`, `trajectories.jsonl`, and a final `summary.json`.
The audit adds root `audit_manifest.json` and per-case
`audit_time_averages.json`. Source and binary identities include the underlying
PES, production kernel, Python preparation helper, native switch source,
Fourier header, compiled switch library, compiler flags and wrapper.
Keep these fixed while a run is active. The native switch is built before
worker launch, and sources are checked again after the cases finish.

The repository also contains a separate checkpoint-clock validation fix:
the permitted roundoff in an accumulated `time += dt` clock now depends on
the integer step count, while the step count itself remains checked exactly.
This avoids rejecting valid long checkpoints. It changes checkpoint validation
tolerance; the numerical propagation map and initial-ensemble definitions
remain the same. Its regression test accepts a million-step accumulated clock
and rejects a full-timestep error. This benchmark's continuous collision
driver uses its recorded native blocks and does not obtain a different
preparation from that validation change.

The analysis writes:

- `preparation_analysis.json`: completion status, pending cases, estimates,
  diagnostics and hashes of input records.
- `preparation_cross_sections.csv` and `preparation_paired_comparisons.csv`:
  clearly labeled sample sizes, actual collision-energy conventions, outcome
  counts, identification bounds and confidence intervals.
- `audit_statistics.csv` and `internal_energy_references.csv`: trajectory-mean
  audit statistics and separate harmonic/action/quantum energy references.
- `audit_paired_time_means.csv`: differences of native hold means between
  preparation variants at the same IDs, including 200 fs versus 50 fs ramps.
- `preparation_numerics.csv`: native work-balance residuals, hold drift and
  observed bond extrema, separate from the statistical uncertainty.
- `unresolved_endpoint_diagnostics.json`: saved endpoint geometry and velocity
  checks with the original status explicitly unchanged.
- `preparation_report.html`, plus PNG/SVG figures for scattering and, when
  audits are available, hold-energy diagnostics.

The optional Chinese PDF uses ReportLab and a CJK TrueType font:

```bash
python benchmarks/h3/render_preparation_report.py
```

After the exact calculation/analysis sources have been committed, the archive
builder can create a checked data bundle with all six baseline cases needed
for the paired comparisons, all twelve new collision cases and all twelve
isolated audits. It verifies the supplied commit against the current source
bytes, all completed ledgers and their recorded analysis hashes; it refuses
incomplete data and existing output files. The commit must exist in the local
Git object database (fetch a connector-published commit first if necessary).

```bash
python benchmarks/h3/package_preparation_study.py \
  --code-commit <exact-40-character-source-commit> \
  --output /absolute/output/pyrpmd_state_preparation_data.zip
```

The bundle retains the historical baseline's separate source commit. It
contains no third-party PES source/binary or publisher PDF. Its internal
`SHA256SUMS` checks data and report members; external member and ZIP checksums
also verify the manifest and the complete archive.

### Validation scope

The action module is checked against analytic harmonic and Morse results,
independent action quadrature, direct Hamilton-equation integration,
time-uniform orbit moments, energy-shell preservation, and explicit domain
failures. The native switch has independent P = 1/5/32/64 harmonic normal-mode
oracles, a time-dependent DOP853 work reference, second-order timestep checks,
a P = 1 adiabatic-action example, sudden-jump accounting, and real-BKMP2
short-run/domain checks. Before this workload, its targeted suite passed
12 tests plus 15 subtests. Preparation tests check COM/collision invariants,
phase pairing and the energy-convention distinction; audit/analysis tests
exercise window reductions, missing outcomes and interval bookkeeping.

These checks establish that the declared numerical construction is implemented
consistently. They do not establish P = 32 convergence for every new ensemble,
long-time quantum-state stability, arbitrary rotational-state preparation,
product-state assignment, or the accuracy of smooth-switched NE-RPMD
cross sections. The fixed 50/200 fs audit durations are sensitivity points,
not proof of an adiabatic limit. Interpret the completed collision and
isolated-audit results together, including their unknown counts and preparation
work, before drawing a physical conclusion.

## Primary references

1. A. I. Boothroyd, W. J. Keogh, P. G. Martin and M. R. Peterson,
   *A refined H3 potential energy surface*, J. Chem. Phys. **104**, 7139–7152
   (1996), [10.1063/1.471430](https://doi.org/10.1063/1.471430).
2. M. Karplus, R. N. Porter and R. D. Sharma, *Exchange Reactions with
   Activation Energy. I. Simple Barrier Potential for (H, H2)*,
   J. Chem. Phys. **43**, 3259 (1965),
   [10.1063/1.1697301](https://doi.org/10.1063/1.1697301).
3. M. Ceriotti, M. Parrinello, T. E. Markland and D. E. Manolopoulos,
   *Efficient stochastic thermostatting of path integral molecular dynamics*,
   J. Chem. Phys. **133**, 124104 (2010),
   [10.1063/1.3489925](https://doi.org/10.1063/1.3489925).
   The deterministic free-ring splitting is the relevant part here.
4. A. Marjollet and R. Welsch, *Nuclear quantum effects in state-selective
   scattering from ring polymer molecular dynamics*, J. Chem. Phys. **152**,
   194113 (2020), [10.1063/5.0004179](https://doi.org/10.1063/5.0004179).
5. A. Marjollet and R. Welsch, *State-selective cross sections from ring polymer
   molecular dynamics*, Int. J. Quantum Chem. **121**, e26447 (2021),
   [10.1002/qua.26447](https://doi.org/10.1002/qua.26447).
6. A. Marjollet, L. Inhester and R. Welsch, *Initial state-selected scattering
   for the reactions H + CH4/CHD3 and F + CHD3 employing ring polymer molecular
   dynamics*, J. Chem. Phys. **156**, 044101 (2022),
   [10.1063/5.0076216](https://doi.org/10.1063/5.0076216).
