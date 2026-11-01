# Independent quantum H + H2 convergence work

This work follows the nonstationary trajectories retained in the
[earlier audit](../../literature-audit-2026-11-01/README.md). The goal is a
statistically and numerically controlled comparison with Table 2 of
[Suleimanov, Allen and Green, CPC 184 (2013), DOI 10.1016/j.cpc.2012.10.017](https://doi.org/10.1016/j.cpc.2012.10.017).
At 300 K that table reports P128 QTST = 2.51e-16, transmission coefficient
0.776, and full RPMD rate = 1.95e-16 cm³ molecule⁻¹ s⁻¹. The paper's
approximately 5% sampling-precision statement is not a supplied modern
confidence interval.

No potential parameter, isotope mass, temperature or rate prefactor is fitted
to these values. Calibration runs, failed initializations, independent seeds
and all completed prefixes remain distinguishable.

## Registered umbrella campaign

[protocol.json](protocol.json) was written before primary rate estimates.
Its SHA256 is
`5834471583ae8e513050dc1c22e065459e1cfa6f139cc5490fcb4eded9fe711a`.
The primary run uses 128 beads, 300 K, Rinf = 30 bohr, equal neutral-1H masses
1837.1526474396187 in atomic units, and the same pinned BKMP2 potential as
the independent classical quadrature. Eight independent ladder seeds are
211017, 211019, 211021, 211023, 211027, 211031, 211033 and 211039.

Each ladder has 5,000 warmup sweeps and registered production prefixes of
30,000, 100,000 and 300,000 sweeps. Every sweep contains global centroid,
orientation and Gaussian-prior path moves, followed at fixed intervals by
adjacent umbrella exchanges. A sweep is a Monte Carlo unit, not physical MD
time. The initial 600-sweep calibration seed 211013 is excluded from all
primary estimates.

The 48-window stiffness/center schedule comes from independent classical
overlap calculations, not from optimizing agreement with the quantum paper.
It uses soft incoming windows and dense strong windows around the barrier.
The complete
[schedule](../h3-classical/umbrella-schedule.json) retains its generation
criterion and SHA256. The same schedule is used by every seed.

### Equilibrium measure

For equal atomic masses m, define Jacobi vectors

\[
 R=q_0-(q_1+q_2)/2,\qquad r=q_2-q_1,
 \qquad \mu_R=2m/3,\quad\mu_r=m/2.
\]

The beadwise total center of mass is a decoupled free coordinate and is
integrated out analytically. A zero-centroid periodic free ring has action

\[
 S_\mu(x)=\frac{\mu P}{2\beta}\sum_j|x_{j+1}-x_j|^2.
\]

Projected independent Gaussian increments of variance beta/(mu P), followed
by cumulative summation and centroid subtraction, draw exactly this prior.
The pCN move is

\[
 x'=\bar x+\sqrt{1-\alpha^2}(x-\bar x)+\alpha\eta.
\]

It is reversible with respect to the Gaussian spring prior, so its Metropolis
ratio contains only the remaining action

\[
 \Phi=\beta\overline V+\beta W(\xi)-\log f_s,
 \qquad
 f_s=\sqrt{\frac{\nabla\xi^T M^{-1}\nabla\xi}{2\pi\beta}}.
\]

Cartesian centroid shifts and rigid Jacobi rotations preserve their stated
volume and spring measure. Exchanges include each window's actual, possibly
different, stiffness. The derivation follows the Gaussian-prior pCN
construction of [Cotter et al., Statistical Science 28 (2013),
DOI 10.1214/13-STS421](https://arxiv.org/abs/1202.0709), applied here to the
explicit Jacobi free-ring measure. Independent tests check Fourier covariance,
the mass/spring identity, proposal-density cancellation, production CV/flux
agreement and exact interrupted replay.

This is an alternative equilibrium sampler used by the benchmark. Real-time
recrossing must still use the physical, unthermostatted PyRPMD equations.

### Explicit physical domain

The old BKMP2 extrapolation has unphysical compact wells. All bead pair
distances must be at least 0.5 bohr, the breaking-bond centroid is bounded above
by 8 bohr, and the coordinate denominator must be positive. The latter keeps
global moves in the incoming reaction-coordinate component instead of jumping
through its pole. No lower bound is imposed on the centroid bond itself:
allowed extended bead paths can have a smaller centroid distance.

The registered lower-cutoff checks are 0.4 and 0.6 bohr. Classical cutoff
insensitivity does not automatically establish quantum cutoff insensitivity.
All sixteen independent equilibrium control chains have now completed.
Relative to P128 / 0.5 bohr, the P256 QTST change is −0.563%, with a 95%
interval [−2.802%, +1.728%]; the 0.4 and 0.6 bohr checks also individually
pass the registered ±log(1.05) equivalence margin. P64 fails equivalence,
so the original aggregate control gate remains false. Full values and
matching finite-P/domain denominators are in the
[control report](../surface-partition/controls/comparison.json).
These equilibrium checks do not establish bead/domain convergence of the
real-time transmission coefficient.

## Free-energy estimation and uncertainty

The primary analysis is histogram WHAM with bin widths 0.001, 0.0005 and
0.00025 in xi. It makes no Gaussian assumption about a window's biased
distribution. All CV observations contribute; a histogram range that drops
even one sample is rejected. Solver success additionally requires its actual
self-consistency residual and a connected overlap graph.

The rate uses the fixed geometrical surface xi = 1 and a quadratic local fit
to log density at xi = 0 and xi = 1. The primary fit halfwidth is 0.004, with
0.002, 0.003 and 0.006 controls. This does not select the noisiest apparent
barrier bin. The independent classical density oracle verifies the WHAM
equations and measures the finite-bin/local-fit bias separately; see
[wham-oracle.json](../h3-classical/wham-oracle.json).

Complete independently seeded exchange ladders determine statistical
uncertainty. Splitting a ladder in time keeps all its exchanged windows
together. Windows, beads, individual observations and bootstrap replicates
are not independent experimental replicas. The registered precision target
is 3% relative standard error, with temporal-half and independent-seed
consistency checks. A systematic discrepancy after convergence calls for
diagnosis; sampling must not stop selectively when a fluctuating result
happens to cross the paper value.

Method reference: [Kumar et al., J. Comput. Chem. 13 (1992),
DOI 10.1002/jcc.540130812](https://doi.org/10.1002/jcc.540130812).

## Independent reactant normalization

Two deterministic calculations provide a check of the H2 reactant quantum
normalization without any umbrella data.

1. [diatom_partition.py](../../../../benchmarks/h3/diatom_partition.py) solves
   the radial Schrodinger equation for each J using a sine DVR, and sums
   `(2J+1) exp(-beta E_vJ)`. The classical denominator is
   `(mu/(2 pi beta))^(3/2) 4 pi integral r² exp(-beta V) dr`. Analytic 3D
   oscillator tests independently verify the radial spectrum and all factors.
2. [diatom_transfer.py](../../../../benchmarks/h3/diatom_transfer.py) expands
   the exact free 3D heat kernel into partial waves, inserts primitive potential
   factors, and evaluates the finite-P trace by radial quadrature. This checks
   the same bead discretization used by primitive path-integral sampling.

For reduced radial measure dr the free kernel is

\[
 K_J(r,r';\tau)=\frac{\mu\sqrt{rr'}}{\tau}
 I_{J+1/2}\!\left(\frac{\mu rr'}{\tau}\right)
 \exp\!\left[-\frac{\mu(r^2+r'^2)}{2\tau}\right].
\]

On quadrature nodes the symmetric transfer matrix is
`A_ij = sqrt(w_i w_j) exp[-tau(V_i+V_j)/2] K_J(r_i,r_j;tau)`.
With tau = beta/P, the partition function is
`sum_J (2J+1) trace(A_J^P)`. Scaled Bessel functions avoid overflow. Tests
recover the J=0 half-line image kernel, the Cartesian free diagonal, the
classical P1 partition, and the exact finite-P 3D harmonic result. Small
negative transfer eigenvalues are checked against roundoff and their possible
trace contribution is bounded, rather than silently accepting an indefinite
matrix.

The computed Q_quantum/Q_classical ratios at 300 K are:

| Calculation | Ratio |
| --- | ---: |
| Continuum radial spectrum | 0.00068545174271648 |
| Primitive P128 transfer trace | 0.00069419428590078 |

The finite-P128 log difference is 0.0126737742 (about 1.275% in the ratio).
The transfer calculation's radial quadrature log change across its prescribed
256/384/512 checks is 1.03e-11. A continuum answer must therefore not be used
as an exactly matching P128 normalization. Full outputs and refinement tables
are in [diatom-partition.json](diatom-partition.json) and
[diatom-transfer.json](diatom-transfer.json).

All J have their ordinary 2J+1 multiplicity, without nuclear-spin or exchange
weights. The 99-bohr isolated spectator limit and beadwise radial walls are
explicit: they are not identical to a finite-Rinf H3 integral with a centroid
upper bound. Upper-box checks quantify the remote radial tail. The initial
DVR gate and its overly strict harmonic endpoint test are retained with an
explanation in [diatom-initial-gate](diatom-initial-gate/README.md); no computed
physical value was adjusted when the convergence diagnostics were clarified.

These normalizations alone do not reproduce a reaction rate.

## Independent surface partition and native recrossing

The separate [surface-partition campaign](../surface-partition/README.md)
now has all eight fixed-budget TS chains complete. Combining their BAR
surface corrections with the independently converged finite-P H₂
normalization and classical surface integral gives
**QTST=2.5524879152e-16 cm³ molecule⁻¹ s⁻¹** at P=128 and 300 K.
The eight-chain 95% interval is [2.5167914228e-16, 2.5886907028e-16].
The point is 1.69% above the paper's 2.51e-16, within its approximately 5%
reported precision. The paper's rounded point is slightly below this new
interval; no claim that it lies inside the interval is made.
The earlier [QTST comparison record](independent-qtst-comparison.json)
retains this distinction and did not yet validate a full RPMD rate.
The [complete initial child cohort](recrossing-stage-200-pairs-2/README.md)
now gives κ=0.77307729 and a full rate 1.97196035e-16, with nominal
95% interval [1.91141982e-16, 2.03441839e-16]. The point is 1.126% above
the paper and the entire interval lies inside its approximate ±5% band.
The separate κ precision gate still fails: its relative 95% half-width
is 3.272%, above the registered 3.000% target. The immutable all-parent
extra-momentum decision was therefore activated; this first failure is retained.

The [complete four-pair extension](recrossing-stage-200-pairs-4/README.md)
now gives κ=0.77121022, 95% CI [0.75333936, 0.78908108], and a full
rate 1.96785637e-16, nominal 95% CI [1.93293861e-16, 2.00340490e-16].
The paper difference is +0.916% and the κ half-width is 2.317%.
All eleven fixed-P primary criteria pass without changing thresholds,
parents, seeds, model, normalization or analysis windows. The supplementary
parent-time-half interval [−0.02310, +0.03626] still fails equivalence
within ±0.03. The unchanged analyzer keeps that failure explicit.

An [independent raw replay](recrossing-stage-200-pairs-4/independent-review/README.md)
reconstructs all 800 chunks, keyed momenta, signed fluxes and joint-chain
statistics. A [base-plus-increment archive extraction](recrossing-stage-200-pairs-4/raw-archive/README.md)
reproduces every raw identity and numerical summary exactly. Dynamic bead
convergence remains a separate [fixed eight-chain P256 design](p256-fixed-eight-chain-design/README.md).
All eight parent chains are complete and their [archive](p256-parent-stage/README.md)
passes independent verification and actual extraction replay. The
[child/analysis tool](p256-child-tool/README.md) uses that full parent cohort;
physical children require the separate coordinator and resource checks.
The full dynamic bead comparison is not yet complete. Before launch,
the [excluded native implementation calibration](p256-native-calibration/RESULTS.md)
completed its fixed 360 steps: q, p, surface values and clocks agree exactly
between native and Python propagation at both timesteps; energy differences
are below 1e-12 hartree, and the native restart traces are bitwise identical.
This short check is excluded from every primary rate estimate.

The [actual child launch record](p256-child-launch/README.md) records adoption
and the two admitted workers. All eight parent chains, four momentum pairs
per parent, both timesteps and the fixed 200 fs horizon remain required before
the frozen analyzer can report the independent P128/P256 comparison.

The [registered native child protocol](recrossing-protocol.json) uses every
one of the 1,600 saved equilibrium parents, two keyed Maxwell pairs per
parent, and identical initial states at 0.1 and 0.05 fs. Both p and −p are
propagated without a thermostat. The primary horizon is 200 fs; a failed
plateau comparison triggers the all-child 400-fs continuation. A separate,
predeclared extra momentum batch can improve precision without selecting
geometries, seeds or favorable rate points. Before either conditional
extension, an immutable decision receipt must reference the complete
eight-chain report and the exact registered gate that triggered it; a
partial-cohort diagnostic cannot trigger a continuation.

The source already includes fs(q), so child flux weights divide by fs.
The full-rate analysis combines the partition and κ within each original
chain before constructing its Student-t interval. It retains their covariance
and never treats individual children as independent parent ensembles.
The [analysis specification](recrossing-analysis-prespecification.json)
records its timing and the additional numerical-stability/component checks.

The [independent method review](../review/pr68-software-review/quantum-method-review.json)
confirms the antithetic flux normalization, physical mass/bead factors,
paired comparisons and covariance calculation without inspecting a primary
rate. The registered central rate is a geometric mean over chain estimates;
finite-chain ratio and log-transformation bias are not separately corrected.
The Student-t intervals are nominal, approximate sampling intervals, not
sequentially adjusted guarantees after conditional extensions. Passing the
separate point-accuracy and precision gates is not identical to demonstrating
that the entire interval lies inside the paper's approximately 5% band; that
stronger condition is retained as its own diagnostic.

An excluded real-PES continuation check verifies bitwise-equal q, p,
observations and clocks for continuous 400-fs propagation versus a 200-fs
save/recreate/reload plus 200 fs at both timesteps. See the
[native replay report](recrossing-native-replay/native-restart-validation.json)
and [pair-batch regression](recrossing-native-replay/pair-batch-replay.json).
Minimum bead-pair distances in production children are observations at
1-fs boundaries, not a claimed every-step distance minimum. Full-H energy
extrema and the PES hard guard are evaluated at every physical step.

The umbrella cohort analyzer initially saved its complete numerical report
but failed while printing a NumPy boolean in the console summary. The
[retained fix receipt](analysis-stdout-fix/receipt.json) verifies that all
scientific fields from the two-chain diagnostic are exactly unchanged after
the console-only conversion. That incomplete two-chain result cannot pass
the eight-chain convergence criterion.

The [complete 30,000-sweep umbrella cohort](umbrella-stage-30000/README.md)
now includes all eight ladders. Its QTST is 2.80072503e-16, with nominal
95% interval [2.33507586e-16, 3.35923162e-16]. The primary log standard
error is 0.0768976 versus the registered 0.03 target; precision and paper
point agreement both fail. All other registered equilibrium checks pass.
The next registered 100,000-sweep prefix has been selected for every seed
on the basis of the precision failure. Its exact first-stage checkpoint,
RNG and sample prefixes are archived before continuation. A small window
Rhat (maximum 1.0005765) does not establish precise integrated free energy.

The [complete 100,000-sweep prefix](umbrella-stage-100000/complete-cohort-analysis/README.md)
now retains all eight ladders and gives QTST=2.587396804e-16, nominal
95% interval [2.294132956e-16, 2.918149187e-16], +3.084% from the paper.
Seven registered criteria pass. Statistical precision alone still fails:
the primary log standard error is 0.05087375, above the fixed 0.03 target.
This is a QTST estimate; no recrossing factor or full rate is inferred.

The [complete archive and extracted-data replay](umbrella-stage-100000/raw-archive-v2/README.md)
preserve all checkpoint, RNG and sample-prefix states and reproduce every
scientific field and saved WHAM array exactly. The original campaign's
unexplained lease reappearance is explicitly retained in a separate reviewed
read-only admission. No further cleanup or original-directory writer was
admitted. Any future registered 300,000 prefix must use a new isolated
verified restore and a separate root adoption; it cannot be selected by
closeness to the paper.

## Reproduction and retained initialization failures

Build the pinned external BKMP2 library using the repository's documented
provenance workflow. Then, from the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python benchmarks/h3/quantum_sampling_campaign.py \
  --library /absolute/path/to/libbkmp2.so \
  --output /new/path/to/quantum-campaign \
  --protocol docs/validation/literature-convergence-2026-11-01/h3-quantum/protocol.json \
  --production-stage 30000 --workers 2
```

The protocol verifies source and schedule hashes. Subsequent prescribed
prefixes continue the exact saved geometry/RNG state and verify every earlier
CV prefix hash. A new scientific target or sampler version needs its own
explicit protocol and output directory.

The initial driver created a log inside a new sampling directory before the
sampler's freshness guard. That initialization was rejected before any
sampling step. The log placement was corrected, and all empty/zero-step
attempts remain recorded by [initialization-failure.json](initialization-failure.json)
and the preserved initial driver. The active campaign uses a fresh output
directory, rather than deleting any ambiguous prior run.

The production Andersen fix is separately reviewed in
[the sampling-correction evidence](../review/README.md). Its successful tests
do not replace the convergence checks of this Monte Carlo benchmark.
