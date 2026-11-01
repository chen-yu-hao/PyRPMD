# Independent finite-bead dividing-surface partition check

This is an independent equilibrium calculation of the H3 centroid transition
surface and isolated-H2 reactant quantum/classical partition ratios. It is
benchmark tooling. It does not propagate a real-time trajectory or compute a
recrossing coefficient, and an equilibrium partition result alone is not a
complete RPMD rate reproduction.

The calculation uses the same author-verified BKMP2 potential and declared
physical domain as the contemporaneous umbrella campaign, but it does not
integrate a sampled umbrella PMF. It instead connects a normalized free-ring
reference to the physical potential through a fixed sequence of Hamiltonians.
No published rate is used in its energy, normalization, proposals, estimators,
or calibration decisions.

## Independent measure and normalization

On one active centroid surface the neighboring distances satisfy
`r01 = r12 = r`, with `r02 = r*t`, `1 <= t <= 2`. After integrating global
rotation, the Cartesian coarea and positive-flux measure is proportional to

\[
  r^4 t\sqrt{2+t^2}\,dr\,dt.
\]

The isolated-diatom centroid measure is `r² dr`. Constant rotational, channel,
mass and thermal-wavelength factors cancel within each quantum/classical
ratio. The Jacobi coordinates have reduced masses `2m/3` and `m/2`; their
zero-centroid spring priors are normalized Gaussian distributions. The free
total-COM modes integrate out. There is no extra factor of the bead count.
The tests derive the TS metric directly from Cartesian bond gradients and
recover the separate deterministic classical-TST quadrature.

The common target domain requires every bead pair to be at least 0.5 bohr
apart and the centroid neighboring bond to be at most 8 bohr. A quantum
centroid may be below 0.5 bohr while all of its individual bead pairs satisfy
the physical cutoff. Therefore, restricting the auxiliary reference centroid
to the classical radial interval would omit target support.

The implemented reference has full support `0 < r < 8` and uses
`Vref(r,t) = Vcent(max(r,0.5),t)`. The plateau below 0.5 bohr is an auxiliary
reference only; the physical endpoint remains the average potential at the
beads. Its radial contribution is integrated analytically as `a⁵/5` on the TS
and `a³/3` for the reactant, including its probability in log form even when
ordinary floating-point exponentiation underflows.

The bridging Hamiltonian is

\[
 U_\lambda=(1-\lambda)V_{\rm ref}+\lambda\bar V_{\rm beads}.
\]

An independent draw of reference centroids and exact free-ring paths estimates
the probability `pD` of the common physical bead domain at lambda zero. The
reported normalization is

\[
 \log\frac{Z_Q}{Z_C}
 =\log\frac{Z_{\rm ref}}{Z_C}+\log p_D
  -\sum_i\Delta f_{i,i+1}.
\]

Neither the reference/classical ratio nor the domain probability is silently
set to one. The one-bead normalization reduces analytically to one, including
for synthetic references whose low-radius plateau has appreciable mass.

## Sampling and statistical interpretation

Centroid radius and angle proposals are symmetric in their stated
coordinates and retain the shape Jacobian in their Metropolis probability.
The noncentroid proposals are
`eta_new = sqrt(1-alpha²)*eta_old + alpha*eta_free`. Their proposal-density
ratio exactly cancels the normalized Gaussian spring-prior ratio. This is the
Gaussian-reference-preserving pCN construction described by
[Cotter et al., Statistical Science 28, 424–446 (2013)](https://arxiv.org/abs/1202.0709).
The remaining acceptance uses the physical potential, shape density and
declared domain. Adjacent replicas exchange lambda states with the complete
joint-density ratio.

Adjacent-lambda free energies use the two-ensemble acceptance-ratio estimator
of [Bennett, J. Comput. Phys. 22, 245–268 (1976)](https://www.sciencedirect.com/science/article/pii/0021999176900784).
The implementation and tests retain the signs and sample-count term explicitly.
Trapezoid and Simpson thermodynamic integration are cross-checks, not the
primary estimator. A fixed lambda grid avoids choosing a quadrature after
seeing a favorable rate.

Independent seed chains are the uncertainty units. Saved sweeps, lambda
replicas, and first/second-half estimates are correlated and are not counted
as independent experiments. The per-lambda energy ESS, overlap, exchange
acceptance, endpoint traversals and half differences are mixing diagnostics.

Checkpoints commit only complete sweeps. They retain all ring coordinates,
replica identities, RNG state, observations and normalization identity. Exact
restart tests include unaligned sample/checkpoint intervals and a deliberate
interruption partway through a sweep. Resuming under changed parameters,
source hashes or runtime versions is rejected.

## Excluded calibration and evidence

`calibrate.py` writes `calibration-protocol.json` before any physical sampling.
The protocol fixes P=128, 300 K, a 25-point squared-uniform lambda grid,
1000 warmup and 5000 production sweeps, independent seeds, quadrature orders,
and 8192 independent reference importance draws per surface. These pilots
are excluded from subsequent primary inference. Decisions about a primary
protocol use mixing, overlap and computational cost, never agreement with a
published point.

The direct-importance pilot estimates the same reference-to-target integral
without a Markov chain, retaining zero weights for paths outside the domain.
Its ESS, maximum normalized weight and tail concentration are diagnostics of
whether that estimator is usable; an empirical ESS cannot rule out an unseen
rare tail. Its point estimate does not establish convergence.

The independent H2 finite-bead transfer-matrix result used to validate the
reactant normalization is in
`../h3-quantum/diatom-transfer.json`. At P=128 its log quantum/classical ratio
is **−7.27275868577072**. The continuum DVR value is a separate limit; the
P=128 primitive discretization changes the log ratio by approximately
0.01267. The transfer calculation bounds every bead's radius above by 8 bohr,
whereas this sampler bounds the centroid radius; its 6/8/10-bohr box checks
quantify the remote-tail sensitivity.

`ci-status.json` records a single current-head GitHub Actions query. It is a
timestamped observation of queued/pending jobs, not a claim that remote CI
passed.

### Calibration results and resulting fixed primary protocol

The initial calibration used one seed per surface. Its summaries are in
`calibration-summary.json`; neither result is a primary estimate or a
seed-based uncertainty interval.

| Excluded P=128 calibration | TS | H2 reactant |
| --- | ---: | ---: |
| BAR log quantum/classical ratio | −2.09338657 | −7.32293657 |
| First-half log ratio | −2.14875999 | −7.30160724 |
| Second-half log ratio | −2.03761070 | −7.34430258 |
| Minimum adjacent BAR overlap | 0.98054 | 0.99245 |
| Minimum per-lambda energy ESS | 156.37 | 323.76 |
| Endpoint traversals | 42 | 44 |
| Direct-importance ESS out of 8192 | 52.21 | 93.57 |
| Largest single importance weight | 10.83% | 6.38% |

Both importance pilots have concentrated weights and remain excluded. The
reactant BAR pilot is 0.05018 below the deterministic finite-P log ratio,
about 4.89% below it on the ordinary ratio scale. This single-chain diagnostic
does not establish agreement or a systematic discrepancy. The primary
normalization control uses four independently seeded chains.

Coarsening the existing calibration states to nine squared-uniform lambda
nodes retained minimum overlap 0.889 on the TS and 0.953 for the reactant.
The high proposal acceptance motivated larger pCN and centroid-radius moves.
A separate excluded 500-production-sweep check used these changes. It gave
TS endpoint radius/angle/R-pCN/r-pCN acceptance of
0.590/0.499/0.479/0.456, minimum overlap 0.870, and 100 endpoint traversals.
Its minimum energy ESS was only 11.8, so its role was a short implementation
and mixing check, not a converged partition estimate.

`primary-protocol.json` was written before primary sampling. It registers:

- Eight TS seeds: 721301, 721303, 721307, 721309, 721313, 721317, 721321,
  721323; four reactant control seeds: 721401, 721403, 721407, 721409.
- P=128, 300 K, 2000 discarded warmup sweeps and 20000 production sweeps
  per seed; nine squared-uniform lambda nodes and adjacent exchange every
  sweep.
- Radius proposal width 0.08 bohr, angle-ratio width 0.03, TS pCN amplitudes
  0.7/0.7, and reactant pCN amplitude 0.45.
- Independent reference-domain experiments with 131072 paths per seed;
  unchanged 192/96 centroid quadrature, checked separately against 256/128.
- All TS lambda-one Cartesian configurations at every 100th production
  sweep, giving exactly 200 registered parents per seed. The snapshot arrays
  are `endpoint_q`, shape `(200,128,3,3)` in bohr, and
  `endpoint_completed_sweeps`, values `2100,2200,...,22000` including warmup.
  They consume no sampler RNG and retain beadwise zero total COM because
  the free total-COM factor was integrated out analytically.
- Equal-weight independent-seed log estimates and two-sided Student-t 95%
  intervals. Per-seed half differences in the log ratio, endpoint centroid
  radius and endpoint mean bead energy remain explicit temporal diagnostics.

The recrossing calculation is separate. Parent configurations are from a
static equilibrium Monte Carlo ensemble, not from a real-time parent
trajectory. Equivalent reaction channels and Maxwell momenta must use
independent child RNG streams. Since the configuration ensemble already
contains the flux metric factor, ordinary-Maxwell paired children retain
the existing `v_xi/fs` numerator and `abs(v_xi)/fs` denominator weights.
Partition and recrossing estimates from the same seed must be paired before
estimating a combined rate interval; they must not be treated as independent
sources of uncertainty.

### Software gate and executed source preservation

`test-gate.json`, `tests.log`, and `tests.xml` record **34 passing tests**,
including exact one-bead normalization, the finite-P harmonic determinant,
the independent real BKMP2 classical integral, BAR signs, full-support
reference behavior, checkpoint interruption/replay, complete endpoint-parent
prefixes, and zero change to RNG or sampled states when snapshots are enabled.
The NumPy floor regression removes `np.trapezoid` and verifies that analysis
continues through the supported SciPy integration API.

The first physical calibration executed sampler SHA256
`64b0f0cd6b2382db139afd69e1308f01628b6761a9a8f9561d9bf088ff67cf6f`.
Its exact source bytes and tests are preserved as
`surface_partition.calibration-v1.py.txt` and
`test_h3_surface_partition.calibration-v1.py.txt`. The subsequent changes use
`scipy.integrate.trapezoid` for NumPy 1.26 compatibility and add optional
deterministic endpoint snapshots. The initial physical calibration was not
rerun to replace its results.

The primary sampler SHA256 is
`23d2805270a82c91eff7b9a70f89621a356d7625aeaf76246f00896147b7a5db`.
The frozen primary protocol SHA256 is
`4657032eb8e10d59061a52a45adba6cac9d4d0d87213461af11e5589de49c99c`.
Checkpoint settings also bind the potential binary, relevant source files,
Python, NumPy, SciPy and platform identity. Raw NPZ checkpoints and importance
observations are retained locally with checksums; the JSON/text evidence does
not claim that those binary files have been published.

## Completed primary equilibrium experiment

All eight TS chains and all four reactant normalization-control chains reached
their registered counts. `primary-summary.json` contains the per-seed values,
the independent-seed intervals, and temporal diagnostics. The eight completed
TS directories contain all **1600 registered endpoint parents**. Their
checkpoint, settings, report and reference-domain files are immutable; the
separate child calculation binds their exact hashes.

| Quantity | Registered result | 95% interval across independent seeds |
| --- | ---: | ---: |
| TS log quantum/classical ratio, eight seeds | −2.17078499 | [−2.18486868, −2.15670130] |
| TS quantum/classical ratio | 0.1140880 | [0.1124925, 0.1157062] |
| H2 log quantum/classical ratio, four control seeds | −7.28176172 | [−7.29449269, −7.26903076] |
| H2 quantum/classical ratio, stochastic control | 0.0006879725 | [0.0006792694, 0.0006967870] |

The independently converged finite-P H2 value, 0.0006941943, lies inside the
stochastic control interval. The control point is 0.896% lower. This supports
the normalization at the reported statistical precision; it does not assert
exact equivalence or replace the deterministic reactant denominator.
`reactant-control.json` retains this comparison and each control's hashes.

The minimum TS per-lambda energy ESS over all seeds is 778.2, and the minimum
BAR overlap is 0.8938. The paired second-minus-first-half log difference is
0.00475 with interval [−0.03269, 0.04219]. The corresponding endpoint-radius
difference is −0.00264 bohr with interval [−0.00561, 0.000324], and the mean
bead-energy difference is −0.0000361 hartree with interval
[−0.0001595, 0.0000873]. These quantify the residual temporal uncertainty;
the saved sweeps are not treated as independent replicas.

Combining the TS ratio with the independent classical TST integral
1.553118786789885 × 10⁻¹⁸ cm³ molecule⁻¹ s⁻¹ and the deterministic finite-P H2
normalization gives a **fixed-surface QTST factor** of
2.55249 × 10⁻¹⁶ cm³ molecule⁻¹ s⁻¹, with seed interval
[2.51679, 2.58869] × 10⁻¹⁶. This is 1.69% above the 2.51 × 10⁻¹⁶ value in
Table 2 of [Suleimanov, Allen and Green (2013)](https://doi.org/10.1016/j.cpc.2012.10.017).
The manuscript reports approximately 5% sampling precision; that statement is
not interpreted as a separately reconstructed 95% confidence interval. Our
new interval does not include the paper's rounded QTST point, and the
comparison retains that distinction. The complete RPMD rate also requires
the independently propagated recrossing coefficient.

## Registered bead and domain controls

After the full primary cohort completed, `controls/protocol.json` registered
four separate equilibrium experiments: P=64 and P=256 at the 0.5-bohr cutoff,
and P=128 at 0.4 and 0.6 bohr. Each uses four new seeds with 2000 warmup and
20000 production sweeps, unchanged nine-node lambda and proposal settings,
and no endpoint snapshots. No primary parent file is extended or modified.

Each control uses the exact H2 transfer-matrix result for its own bead count
and radial domain. Comparisons also retain the domain-dependent classical TS
and reactant integrals; comparing quantum/classical correction factors alone
would omit a possible change in classical-domain normalization. The ratio of
equilibrium QTST factors is compared with the frozen P=128 primary result
using an independent-group Welch interval. The declared equivalence criterion
requires the entire 95% log-ratio interval to lie within ±log(1.05). An interval
too wide to meet that criterion remains inconclusive; it is not adjusted to
make the literature point agree.

The controls protocol SHA256 is
`0b3cbbff1aa90f8ec0f9c77d0d5c68786f0f368e26d8b93bc6c35604b736eb56`.
`control-test-gate.json` records four additional checks of the domain-volume
normalization, Welch uncertainty, full-interval equivalence rule and matching
finite-P/domain oracle selection. These are equilibrium controls only:
they do not establish bead convergence of the real-time recrossing factor.

## Reproduce in a separate directory

Build the documented BKMP2 library first, as described in
[`benchmarks/h3/README.md`](../../../../benchmarks/h3/README.md), and install
the package and its numerical dependencies. The default library location is
`benchmarks/h3/build/libbkmp2.so`; both commands also accept `--library`.
Run from the repository root and use a new sibling directory so the original
primary files remain immutable:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python benchmarks/h3/surface_partition_campaign.py \
  --output docs/validation/literature-convergence-2026-11-01/surface-partition-replay

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python benchmarks/h3/surface_partition_controls.py \
  --output docs/validation/literature-convergence-2026-11-01/surface-partition-replay/controls
```

The control runner expects a completed primary summary in its parent
directory and the independently generated `h3-quantum/diatom-transfer.json`
beside that primary directory. Both runners register their settings before
sampling and use at most two single-thread workers. Reinvoking the same
command resumes an interrupted campaign only when its complete identity
still matches; completed seed files are checked and left intact. The
`--register-only` option records the protocol without starting a chain.
Runtime and binary hashes are part of the identity, so a reproduction on a
different installation gets a separate protocol and cannot silently resume
the original campaign.

The supplied exact-reactant reference also binds the BKMP2 binary identity.
A differently compiled library requires a matching reference generated with
`benchmarks/h3/diatom_transfer.py` in a separate reproduction checkout;
the controls deliberately reject a mismatched reference. Preserve this
experiment's original reference and parent files when doing that rebuild.

`primary-manifest.json` binds all completed primary sources, text evidence
and raw checkpoint files. The checkpoint hashes support integrity checks;
they are not a claim of cross-platform bitwise-identical stochastic results.
