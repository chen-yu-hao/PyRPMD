# Numerical audit and literature comparison — 1 November 2026

This is a fresh execution audit for
[PyRPMD PR #62](https://github.com/chen-yu-hao/PyRPMD/pull/62), including real
BKMP2, q-TIP4P/F and Ni–H EAM calculations. Source hashes, settings, raw-data
checksums, and unsuccessful runs are retained alongside successful checks.

**The independent classical H + H₂ transition-state integral agrees with the
published value to 0.44%. The sampled H + H₂ thermal RPMD rate and the water
diffusion coefficient have not passed physical convergence checks.** Software
regression success and local force accuracy do not turn those unfinished
physical validations into literature reproductions.

## Evidence map

| Area | What this audit establishes | Detailed evidence |
| --- | --- | --- |
| Software and distribution | Installed Python 3.13/3.14 numerical coverage; real native/optional-engine regression; diagnosed portability defects | [Software audit](software/README.md) |
| H + H₂ / BKMP2 | Author PES oracle, derivatives, independent classical TST, fresh quantum/classical sampling controls and time-stability failures | [H3 evidence](h3/README.md) |
| q-TIP4P/F water | Independent force checks, electrostatics and timestep controls, a 216-molecule classical pilot, short P=8/16/32 controls | [Water evidence](water/README.md) |
| Ni–H EAM | Independent ASE/LAMMPS energies, forces, periodicity and strain virials using one verified potential table | [Materials audit](materials/README.md) |

The earlier [ecosystem implementation audit](../ecosystem-2026-11-01/README.md)
remains a separate historical record. Its archived H3 results and tiny water
smoke trajectory are not substituted for the fresh calculations in this audit.

The integrated Python 3.12 run completed with **4,527 passed, 97 skipped,
365 subtests passed and no failures**, in 739.88 seconds. All 450 source files
recorded before that run had identical hashes afterward. The log retains 932
deprecation warnings, predominantly from ASE with NumPy 2.5, plus the report
renderer. [Full results and provenance](software/full-suite.json),
[raw pytest log](software/full-suite.log) and [JUnit](software/full-suite.xml)
are retained. Separately installed native wheels under Python 3.13 and 3.14
each passed 980 selected tests plus 64 subtests; these are targeted numerical
runs, not a claim of the full suite on those interpreters.

The later reference-label correction in the water analyzer passed 24 focused
tests. The H3 audit orchestration scripts were added after the full-suite
snapshot; their actual executions, exact original-versus-portable script hashes,
and full-data replays are recorded separately. Seven focused tests verify that
the audit UI cannot silently attach the 300 K ordinary-H reference to another
temperature or isotope. These later checks are not added to the full-suite
count. See [orchestration provenance](h3/orchestration-provenance.json) and
[complete CV replay](h3/extended-pmf-replay.json).

Cross-platform execution exposed three distinct infrastructure/portability
problems: a read-only file descriptor passed to `fsync` on Windows, a legacy
oversubscription setting ignored by Open MPI 5, and a hosted runner's
unsupported RDMA device selected by MPICH/UCX before LAMMPS initialization.
The fixes use a writable synchronization handle, explicit MPI placement, and
TCP/shared-memory transports for the single-host interoperability job.
Each has an observed failure record and a focused verification. Pending
remote jobs remain pending in the [software report](software/README.md).

## H + H₂: isolate the potential, the integral, and the sampling

The comparison source is Suleimanov, Allen and Green, *Computer Physics
Communications* **184**, 833–840 (2013),
[DOI 10.1016/j.cpc.2012.10.017](https://doi.org/10.1016/j.cpc.2012.10.017).
The [MIT manuscript record](https://dspace.mit.edu/entities/publication/a4555c62-26a0-4500-8953-5dd1905583cc)
identifies the source; the
[author-uploaded full manuscript](https://www.researchgate.net/publication/279840192_RPMDrate_Bimolecular_chemical_reaction_rates_from_ring_polymer_molecular_dynamics)
provides searchable Tables 1–2. The relevant 300 K reference values, all in
cm³ molecule⁻¹ s⁻¹, are:

| Quantity | Classical | RPMD, 128 beads |
| --- | ---: | ---: |
| Transition-state factor | 1.56 × 10⁻¹⁸ | 2.51 × 10⁻¹⁶ |
| Transmission coefficient κ | 0.994 | 0.776 |
| Full thermal rate | 1.55 × 10⁻¹⁸ | 1.95 × 10⁻¹⁶ |

These columns cannot be interchanged. A TST/QTST integral is compared with its
corresponding static factor; it is not compared with the recrossing-corrected
rate as though the observables were identical. The source describes roughly
5% sampling precision, without supplying a confidence interval to use as a
modern statistical acceptance test.

### Independent classical integral

[classical-tst-quadrature.json](h3/classical-tst-quadrature.json) records
**1.553118786789885 × 10⁻¹⁸**, a relative difference of **−0.4411%** from the
classical TST reference. This calculation uses the verified BKMP2 PES directly
in interatomic-distance coordinates. It bypasses the production reaction CV,
umbrella sampling, PMF estimator and trajectory propagator. The measure, mass
metric and two equivalent product arrangements are derived explicitly in
`benchmarks/h3/classical_tst.py`; no prefactor is fitted to the paper.

Changing the Gauss quadrature from 192 to 256 nodes changes the answer by
1.12 × 10⁻¹² relatively. Declared physical-channel integration-domain checks
change it by 2.10 × 10⁻¹³. These are numerical quadrature checks, not estimates
of the physical PES error. The legacy surface has an unphysical compact-triangle
extrapolation outside the audited reaction domain. The report states that
domain explicitly and does not claim an unrestricted global partition integral.

### Sampling controls

Fresh short and standard pilots reproduce the large rate deficit previously
seen in the archived pilot. Extending the biased trajectories changes the
estimate substantially. In the first extended P=128 campaign, the full-sample
QTST estimate is about 6.4344 × 10⁻¹⁶, but the first and second halves give
1.682 × 10⁻¹⁶ and 2.420 × 10⁻¹⁵: a factor of about 14.4. The P=1 control also
drifts, from 3.412 × 10⁻¹⁹ to 1.033 × 10⁻¹⁸ between production halves.

These observations directly establish a failure of sampling stability. They
are consistent with inadequate equilibration/sampling in both classical and
quantum calculations; they do not prove that every part of the rate algorithm
is correct. A plateau in a short recrossing calculation cannot validate a
nonstationary PMF, because the full rate depends exponentially on the PMF barrier.

The long controls use 111 windows, two streams per window, 0.5 ps equilibration,
1 ps production, a 0.1 fs step, 300 K, R∞=30 bohr and k=30 hartree. Table 1 of
the paper uses 200 trajectories per window, 20 ps equilibration and 100 ps
production per trajectory. The present Andersen/Cayley protocol also differs
from the paper's H3 GLE/Verlet protocol. Matching the PES, temperature, beads,
window spacing and force constant does not imply an identical sampling protocol.

The original full workflows for two long controls were terminated with exit
137 during large intermediate MBAR analyses. Every one of their 222 sampling
checkpoints was complete. A [resource snapshot](software/resources.json)
records an 8 GiB shared limit and cumulative OOM-kill events; it cannot assign
an event to a particular process. Recovery uses the unchanged full CV samples,
with the analysis stages and any altered analysis settings reported explicitly.

### Recovered full-rate candidates

All three long controls were analyzed on their complete 444,000 saved CV
values. The existing unique-state MBAR path checks the overlap and PMF on
those same values, followed by newly propagated parent/child trajectories.
No short-pilot transmission coefficient is reused. Each control supplies
64 parent configurations and 512 child trajectories.

| Calculation | TST/QTST factor | κ | Full-rate candidate | Appropriate published full rate |
| --- | ---: | ---: | ---: | ---: |
| Classical P=1 | 5.87482 × 10⁻¹⁹ | 0.997470 | 5.85996 × 10⁻¹⁹ | 1.55 × 10⁻¹⁸ |
| Quantum P=128, seed A | 6.43440 × 10⁻¹⁶ | 0.774671 | 4.98455 × 10⁻¹⁶ | 1.95 × 10⁻¹⁶ |
| Quantum P=128, seed B | 1.33407 × 10⁻¹⁶ | 0.715314 | 9.54280 × 10⁻¹⁷ | 1.95 × 10⁻¹⁶ |

Rate units are cm³ molecule⁻¹ s⁻¹. **Every row remains unconverged.** The two
quantum seeds differ by a factor of 5.22 in the final candidate. Seed B's
production halves drift in the opposite direction from seed A, reinforcing
the failure of stability. At fixed dividing surfaces, full-sample MBAR and
Gaussian UI barriers differ by about 0.00012–0.00014 hartree (roughly 14–16%
in the static rate factor). Estimator agreement at that level does not remove
the much larger time/seed sensitivity.

## Water: match the observable and the dynamics

The source is Habershon, Markland and Manolopoulos, *J. Chem. Phys.* **131**,
024501 (2009), [DOI 10.1063/1.3167790](https://doi.org/10.1063/1.3167790),
[author manuscript](https://arxiv.org/pdf/1011.1047). The detailed
[source audit](water/source-audit.json) distinguishes printed parameters,
later author-attributed Fortran parameters, selected isotope masses, and
unspecified details of the original implementation.

| Observable | Published reference | Status of this audit |
| --- | ---: | --- |
| Classical H₂O self-diffusion, 216 molecules | 0.199 ± 0.002 Å²/ps | Short classical pilot; no accepted diffusion estimate |
| RPMD H₂O self-diffusion, 32 beads | 0.221 ± 0.001 Å²/ps | Short quantum controls; not reproduced |
| RPMD D₂O self-diffusion | 0.172 ± 0.001 Å²/ps | Not simulated in this audit |
| Quantum mean O–H bond length | 0.978 ± 0.001 Å | Short P=32 structural diagnostic; not an equilibrium validation |
| Infrared spectrum | PA-CMD correlation function | No PA-CMD reproduction |

The uncertainties above are the paper's reported standard errors. The
classical, quantum, heavy-water and infinite-size-extrapolated results have
different meanings and are not pooled into a single comparison target.

The fresh pilot uses the source-sized 216-molecule box at 298 K and
0.997 g/cm³, with the printed q-TIP4P/F parameters, 9 Å cutoff and full PME.
It includes 20 ps of sampling followed by two consecutive 4 ps NVE segments
with refreshed Maxwell momenta. The original protocol has 100 ps equilibration
and 20 batches of 100 such production segments, or 8 ns aggregate production.
Adjacent segments, overlapping time origins, and molecules within a box are
not treated as independent replicas for a validated error bar.

Force verification includes an independent cluster oracle and the original
author-attributed Fortran implementation with matched parameters. PME is checked
against tight direct Ewald, platform precision is measured, and halved-timestep
energy diagnostics are recorded. Full per-bead PME, the single-step integrator,
and PILE-L sampling differ from the original contraction/MTS/Andersen protocol.

### Measured classical transport diagnostics

The complete 112,000-step run took 1,639.64 seconds under the recorded shared
CPU load. The last 5 ps of equilibration averaged 297.384 K, an O–H distance
of 0.963728 Å and an H–O–H angle of 104.7998°. The largest NVE energy changes
were 0.001090 and 0.001258 kcal/mol per water in the two segments.

| Four-ps segment | MSD slope estimate, 1–2 ps (Å²/ps) | VACF integral through 2 ps (Å²/ps) | Log–log MSD slope |
| --- | ---: | ---: | ---: |
| 1 | 0.117181 | 0.153852 | 0.5944 |
| 2 | 0.193284 | 0.184710 | 0.7742 |

The corresponding classical target is 0.199 ± 0.002 Å²/ps. The first segment
fails the long-time MSD check; the second only meets the software's candidate
heuristic. Changing the MSD fitting interval from 0.1–0.5 ps to 0.5–1 ps to
1–2 ps changes the segment-average raw estimate from 0.3585 to 0.1530 to
0.1552 Å²/ps. **No validated diffusion value or standard error is emitted.**
Temperature control and local structural agreement have therefore not been
mistaken for converged liquid transport. See the
[full classical analysis](water/classical-pilot/report.json) and
[diagnostic figure](water/classical-pilot/diagnostics.png).

The 100-fs P=32 warmup gives an O–H distance of 0.977889 Å and H–O–H angle
of 104.7387° in its last 50 fs, close to the paper's equilibrium means. Its
20-fs NVE MSD has a log slope of 1.9858 and is explicitly classified as
ballistic. This short local-response result is not bead or structural convergence.

A post-review reference-label correction also ensures that classical D₂O
does not receive the quantum D₂O diffusion target. Its separate 24-test water
regression is recorded in the water evidence; the simulations above are all
H₂O and their dynamics and results are unchanged.

Ordinary RPMD high-frequency dipole spectra are not the PA-CMD spectra used
by this paper. Likewise q-TIP4P/F is flexible: adding bond constraints would
change the model under comparison. The short quantum trajectories are therefore
reported as structural/integration checks, with ballistic MSD explicitly rejected
as a diffusion measurement.

## The remaining requested literature cases

H + CH₄ and OH + H₂ have verified numerical references in the existing
[published-rate registry](../../../benchmarks/published_rates/references.json).
This audit does not claim new production calculations on their exact reactive
PESs. F + H₂, Mu/D + H₂, and Pd/Ni diffusion also require the correct model,
observable, conventions and source corrections before a reproducibility claim.
The [literature benchmark guide](../../literature-validation.md) describes these
evidence requirements. The [reaction source audit](h3/other-reaction-references.md)
also records the OH electronic-state conventions and source-table discrepancy,
and distinguishes the F + H₂ scientific erratum from its separate publisher's
note; these unresolved choices are not treated as numerical passes.

The Mu + H₂ reference
[10.1021/jz201702q](https://doi.org/10.1021/jz201702q) specifically emphasizes
zero-point energy with a minor tunnelling contribution. A distinct published
deep-tunnelling stress test is **D + HMu → DMu + H**,
[10.1021/jz502216g](https://doi.org/10.1021/jz502216g). Keeping these identities
separate avoids attaching the wrong scientific claim to a benchmark label.

## Reproducibility and interpretation

The per-area reports give the exact commands, seeds, dependency versions,
model hashes and analysis inputs. Compact numerical evidence is committed
here. Complete generated raw datasets are delivered separately with checksums;
third-party PES source/binary files with unestablished redistribution terms
are excluded. External models are obtained from the pinned source identified
by each benchmark and verified before use.

The [H3 raw-data index](h3/raw-data-index.json) and
[water raw-data index](water/raw-data-index.json) identify the complete
datasets and their checksums. The water [summary table](water/summary.csv)
and H3 [summary table](h3/compact-results.csv) expose units, reference values
and convergence status explicitly. Analysis code is versioned in this
repository; the separate archives contain the generated experimental data
and source references.

The next physical acceptance gates are independent-seed and within-stream
stationarity, bead/timestep/geometry convergence, and uncertainty based on
independent sampling units. For a reaction rate, both the PMF and recrossing
factor need support. For diffusion, the long-time regime, estimator consistency,
equilibration and finite-size convention need support. Until those checks pass,
the audit records diagnostics and discrepancies rather than a validated rate or
diffusion coefficient.
