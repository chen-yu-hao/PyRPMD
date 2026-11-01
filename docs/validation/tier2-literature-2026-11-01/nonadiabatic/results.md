# Quantitative results: Tully model I

Actual public-engine trajectories: **4,608**; unresolved trajectories: **1**. Main points contain 512 trajectories each, with four independent seed blocks.

These are independent quantum calculations and PyRPMD ensembles for the published Hamiltonian; no numerical points were digitized from Tully's Fig. 4. See [README.md](README.md) and [sources.json](sources.json) for equations, source locations and limitations.

## Main probabilities at Δt = 4 a.u.

Parentheses give pointwise 95% Wilson intervals for classification by the fixed time cutoff. All trajectories, including unresolved outcomes, remain in the denominator. The quantum reference is monochromatic and asymptotic; conservative assignments of unresolved outcomes are provided separately in `censoring_bounds.json`.

| Method | Momentum | Lower transmission | Upper transmission | Lower reflection |
| --- | ---: | --- | --- | --- |
| Independent quantum | 8.8 | 0.932748 | 0.000000 | 0.067252 |
| Independent quantum | 15 | 0.677012 | 0.322988 | 0.000000 |
| Independent quantum | 25 | 0.376322 | 0.623678 | 0.000000 |
| FSSH (`P=1` RPSH) | 8.8 | 0.9199 (0.8932–0.9404) | 0.0000 (0.0000–0.0074) | 0.0801 (0.0596–0.1068) |
| FSSH (`P=1` RPSH) | 15 | 0.6406 (0.5982–0.6810) | 0.3594 (0.3190–0.4018) | 0.0000 (0.0000–0.0074) |
| FSSH (`P=1` RPSH) | 25 | 0.3574 (0.3171–0.3999) | 0.6426 (0.6001–0.6829) | 0.0000 (0.0000–0.0074) |
| Endpoint MASH | 8.8 | 0.9473 (0.9244–0.9635) | 0.0000 (0.0000–0.0074) | 0.0508 (0.0349–0.0734) |
| Endpoint MASH | 15 | 0.6445 (0.6021–0.6848) | 0.3555 (0.3152–0.3979) | 0.0000 (0.0000–0.0074) |
| Endpoint MASH | 25 | 0.3555 (0.3152–0.3979) | 0.6445 (0.6021–0.6848) | 0.0000 (0.0000–0.0074) |

All four outgoing channels, including small upper-state reflection, are retained in `channel_probabilities.csv`. Zero observed counts do not imply a zero true probability.

### Finite-duration censoring

| Method | Momentum | Unresolved /512 at Δt=4 |
| --- | ---: | ---: |
| mash | 8.8 | 1 |
| mash | 15 | 0 |
| mash | 25 | 0 |
| rpsh | 8.8 | 0 |
| rpsh | 15 | 0 |
| rpsh | 25 | 0 |

Each unresolved path could eventually reach any channel. The empirical asymptotic probability therefore lies between `count/N` and `(count+unresolved)/N` before accounting for sampling error. `censoring_bounds.json` also combines these worst-case assignments with Wilson intervals. Any secondary 48-flight-time tail follow-up is recorded separately and does not replace the predeclared 12-flight-time outcome in this main table.

## Reference discretization

| Momentum | Max channel change, h=0.0025→0.00125 | Max channel change, L=12→16 at h=0.0025 | Max flux defect |
| ---: | ---: | ---: | ---: |
| 8.8 | 1.42e-05 | 4.28e-12 | 9.85e-12 |
| 15 | 4.77e-05 | 4.55e-14 | 2e-13 |
| 25 | 0.000175 | 3.43e-14 | 1.43e-13 |

Flux conservation is a solver check; the larger mesh change is the relevant discretization diagnostic.

## Independent seed blocks

The selected channel is lower reflection below the upper-state threshold and upper transmission above it. Each entry is the event count out of 128. A Pearson p-value is only printed when every expected cell count is at least five; this is a descriptive homogeneity check, not a claim that finite sampling has converged.

| Method | Momentum | Channel | Four block counts | Pearson p-value |
| --- | ---: | --- | --- | ---: |
| rpsh | 8.8 | reflected_lower | [9, 6, 10, 16] | 0.1331 |
| rpsh | 15 | transmitted_upper | [44, 46, 51, 43] | 0.7316 |
| rpsh | 25 | transmitted_upper | [88, 76, 80, 85] | 0.4100 |
| mash | 8.8 | reflected_lower | [5, 8, 7, 6] | 0.8470 |
| mash | 15 | transmitted_upper | [39, 45, 47, 51] | 0.4650 |
| mash | 25 | transmitted_upper | [84, 85, 74, 87] | 0.3281 |

## Timestep refinement

The following comparisons retain the same 128 trajectory identities. FSSH random-number sequences diverge across timesteps, so channel-change fractions are not convergence rates of deterministic paths. MASH uses the same initial spin in each pair. Mean maximum energy drift is measured along each whole trajectory, in hartree; the table compares the same subset at both timesteps.

| Method | Momentum | Δt coarse→fine | Changed channels /128 | 95% change-fraction interval | Mean max energy drift coarse→fine |
| --- | ---: | --- | ---: | --- | --- |
| rpsh | 8.8 | 8→4 | 15 | 0.0723–0.1844 | 3.22e-06→7.51e-07 |
| rpsh | 8.8 | 4→2 | 13 | 0.0603–0.1660 | 7.51e-07→1.93e-07 |
| rpsh | 15 | 8→4 | 59 | 0.3770–0.5472 | 1.17e-05→2.81e-06 |
| rpsh | 15 | 4→2 | 57 | 0.3621–0.5318 | 2.81e-06→7.46e-07 |
| rpsh | 25 | 8→4 | 57 | 0.3621–0.5318 | 3.36e-05→8.48e-06 |
| rpsh | 25 | 4→2 | 67 | 0.4375–0.6080 | 8.48e-06→2.11e-06 |
| mash | 8.8 | 8→4 | 3 | 0.0080–0.0666 | 3.17e-06→7.91e-07 |
| mash | 8.8 | 4→2 | 1 | 0.0014–0.0429 | 7.91e-07→1.99e-07 |
| mash | 15 | 8→4 | 0 | 0.0000–0.0291 | 1.13e-05→2.81e-06 |
| mash | 15 | 4→2 | 0 | 0.0000–0.0291 | 2.81e-06→7.04e-07 |
| mash | 25 | 8→4 | 0 | 0.0000–0.0291 | 3.57e-05→8.85e-06 |
| mash | 25 | 4→2 | 0 | 0.0000–0.0291 | 8.85e-06→2.25e-06 |

Per-channel fine-minus-coarse probability differences and paired standard errors are in `paired_timestep_comparison.csv`. A zero observed change in 128 pairs still has a nonzero upper confidence bound.

**The FSSH refinement ensembles do not establish population convergence at the few-percent level.** At p=25, the upper-transmission estimate changes by −0.08594 from Δt=4 to Δt=2, with paired standard error 0.06374 (a normal-approximation 95% uncertainty of about ±0.125). That difference is not resolved by 128 paths. For MASH, the same initial spins give one changed classification at p=8.8 and no changes at the two higher momenta; zero observed changes still permits a change probability up to 2.91% under the Wilson bound. Roughly fourfold reductions in energy drift on halving Δt are a useful numerical check, but cannot establish the convergence order of populations across hops.

## Interpretation of the paper comparison

All 24 main outgoing-channel pointwise 95% intervals contain the monochromatic quantum reference at the three momenta. These are dependent channel comparisons within six ensembles, not 24 independent proofs of accuracy. In particular, upper transmission at p=15 exceeds the reference by 0.03639 for FSSH and 0.03248 for MASH; the sampling intervals are too broad to resolve errors of a few percentage points. The results provide a bounded statistical consistency check, not exact agreement or evidence that one method outperforms the other.

At p=8.8 the separately converged paper-inspired Gaussian-width reference has lower reflection 0.03528 and upper transmission 0.04874. The monochromatic comparison instead has reflection 0.06725 and a closed upper channel. Substituting the Gaussian reference would change the initial distribution and therefore the question being tested; it is not used to improve apparent method agreement.

## Scope of the result

Statistical compatibility at a few model points is useful validation, but it does not establish exact quantum dynamics or reproduce every point in a published figure. Endpoint MASH remains a first-order-at-hops discretization requiring refinement; it is not the reversible event-resolved algorithm. Nuclear tunneling, long-time coherence and arbitrary molecular dynamics are not established here. The single-bead results do not validate thermal NRPMD rates or multi-bead RPSH/MASH accuracy.
