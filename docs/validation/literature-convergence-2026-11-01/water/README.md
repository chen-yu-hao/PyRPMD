# q-TIP4P/F: continuing toward transport convergence

This campaign continues the [initial water audit](../../literature-audit-2026-11-01/water/README.md). The new classical trajectories use **OpenMM's native integrator as an external reference**. They are not PyRPMD production trajectories. Their purpose is to separate force-field, integrator, sampling, and transport-estimator errors before claiming agreement with the paper.

**The registered stage is still running. No diffusion agreement or quantum-water convergence is claimed.** The first two seed chains began from the archived 28 ps classical geometry and have now each committed a full new 100 ps preparation and entered fixed production. Short numerical experiments below do not substitute for that preparation or for the registered production trajectories.

## Fixed classical experiment

The primary literature reference is Habershon, Markland, and Manolopoulos, *J. Chem. Phys.* **131**, 024501 (2009), [DOI 10.1063/1.3167790](https://doi.org/10.1063/1.3167790), [author manuscript](https://arxiv.org/pdf/1011.1047). Table IV reports classical H2O diffusion of **0.199 ± 0.002 Å²/ps**. Its quoted uncertainty is a standard error, obtained from the paper's repeated batches. The quantum H2O and D2O values are different targets and do not apply to this classical experiment.

The complete original registration is [protocol.json](protocol.json). Its SHA256 is `288059c14c1b0467c322806fcf41765b69df0ed50af436941e7f0cad3fbcf631`. It was written before the new production data existed and is retained unchanged.

| Item | Registered experiment |
|---|---|
| Molecules / beads | 216 H2O / P = 1 |
| Thermodynamic state | 298 K; density 0.997 g/cm³; fixed cubic cell |
| Force field | Printed Table I q-TIP4P/F parameters; declared NIST isotope masses and virtual M site |
| Nonbonded interactions | 9 Å cutoff; full PME with requested tolerance 1e−5; no dispersion tail correction |
| Constraints | None; flexible OH stretches and HOH angle |
| Independent chains | Eight, seeds 1101–1108 |
| Preparation | 100 ps per chain; global canonical Maxwell momentum refresh every 100 fs |
| Production | 25 consecutive 4 ps NVE segments per chain; fresh Maxwell momenta at each segment boundary; one removal of total COM momentum |
| Time step / integrator | 0.25 fs; native OpenMM Verlet with synchronized physical velocities |
| Observations | Every 5 fs in NVE; every 100 fs during preparation |
| Total registered NVE | 800 ps; the paper used 8000 ps |
| Uncertainty unit | Independent seed-chain mean; not an individual molecule, time origin, or adjacent segment |
| Stopping | Finish the fixed stage, or retain a checkpoint at a declared compute budget; no reference-value selection |

All eight chains share an initial geometry, so separate random seeds alone do not prove adequate equilibration. The last 50 ps of each preparation are checked in two 25 ps windows for temperature, potential energy, OH distance, and HOH angle. Five-ps block statistics and oxygen RDFs provide additional diagnostics. Production-half stability, the diffusive time regime, and time-step sensitivity still require assessment.

The paper's printed parameters, the author's distributed source constants, and the OpenMM implementation are distinguished in the earlier audit. Small rounding/unit-constant differences remain explicitly documented. “Same q-TIP4P/F parameterization” is not a claim that every source-executable convention is bitwise identical.

The declared masses are O-16 = 15.99491461957 Da and H-1 = 1.00782503223 Da. The article does not give its exact numerical isotope masses, translational-velocity weighting, velocity-recording cadence, or LJ tail convention. These implementation choices are explicit rather than inferred to match an unavailable original input deck. The predeclared classical cadence control below now checks the 5 fs recording interval against a 0.25 fs record for one fixed segment.

This stable evidence snapshot contains completed numerical checks and the fixed scientific protocol. Preparation/production progress records are updated separately while the registered trajectories continue. No incomplete-chain result here is presented as a literature comparison.

### Estimators and uncertainty

The primary estimator is the molecular COM velocity-autocorrelation integral divided by three, integrated to 2 ps. The secondary estimator is the molecular COM MSD slope divided by six, fitted from 1 to 2 ps. Every completed registered segment contributes equally to its chain mean, including segments whose individual diagnostic time window looks poor. An incomplete final segment is reported as partial and is not silently treated as a complete 4 ps record.

The eight independent chain means receive equal weight. Standard errors and 95% Student-t intervals use seven degrees of freedom after all chains finish. The analyzer never counts overlapping origins, molecules, or neighboring 4 ps segments as independent replicas.

The original paired-difference screen uses `max(0.01 Å²/ps, 95% CI halfwidth)` as a compatibility threshold. **Passing that screen is a failure to detect a difference, not an equivalence result.** A [supplementary preregistration](paired-equivalence-preregistration.json), recorded before any new NVE segment existed, adds the stricter requirement that the entire paired-chain 95% CI lie inside **[−0.01, +0.01] Å²/ps**. It covers the prespecified VACF cutoffs, MSD fit windows, and MSD–VACF comparison. A broad interval cannot satisfy that equivalence criterion.

The fixed stage targets a primary 95% CI halfwidth no larger than 0.02 Å²/ps. That is substantially looser than the paper's reported uncertainty and must not be represented as reproduction of its precision. No literature comparison is automatically issued from a partially completed set of chains. Even passing all automated checks leaves the time-step and structural/transport reviews open.

### Preregistered velocity-recording control

The [cadence registration](velocity-cadence-preregistration.json) selected seed 1101, segment 0 before that chain produced any NVE data. It requires the exact checkpoint at the end of 100 ps preparation, before the segment's Maxwell momentum draw. The subsequently available checkpoint and its SHA256 are recorded in [velocity-cadence-selection.json](velocity-cadence-selection.json). The control does not change the main campaign.

[velocity_cadence.py](../../../../benchmarks/water/velocity_cadence.py) restores the binary checkpoint, redundant raw engine state, and external RNG, draws the one registered COM-projected Maxwell momentum, and replays 4 ps with the same 0.25 fs Verlet map. It records molecular COM velocities in float64 at every step and checkpoints each 1 ps. Every 0.5, 1, 2.5, and 5 fs record is a subsample of this same dense trajectory. The predeclared numerical criterion is that each VACF integral to 2 ps differs from the dense result by at most 0.001 Å²/ps.

The comparison includes both the coarser lag quadrature and the reduced set of time origins. A separate diagnostic subsamples the dense-origin VACF only in lag, which isolates quadrature error. The FFT sums component blocks instead of allocating a full complex time-by-molecule workspace; the VACF remains uncentered and is divided by the actual origin count and molecule count. No individual molecular/time-origin independence is assumed.

Seven [focused regressions](cadence-control-pytest.xml) cover direct-dot versus block-FFT correlations, constant velocity, NumPy 1.26 compatibility, fixed checkpoint selection, and a real Reference-platform replay/restart comparison with the native segment's original RNG phase and map. The source-sized 4 ps replay has now completed: [report](velocity-cadence/report.json), [derived VACF curves](velocity-cadence/curves.npz).

| Recorded interval | Integral difference from the 0.25 fs record, Å²/ps | Dense-origin lag-quadrature difference, Å²/ps |
|---:|---:|---:|
| 0.5 fs | −3.63021e−6 | −2.59648e−10 |
| 1 fs | −1.08165e−5 | −1.29028e−9 |
| 2.5 fs | −3.17780e−5 | −8.14736e−9 |
| 5 fs | −6.45709e−5 | −2.59374e−8 |

Every interval passes the preregistered 0.001 Å²/ps numerical margin on this predetermined segment. The small difference mainly comes from subsampling the available time origins. The replay took about 325 wall seconds and 159 process CPU seconds on the shared host. This result assesses recording numerics; it does not supply a new independent diffusion estimate, select a favorable trajectory, or certify a literature result. CPU force-cache roundoff also precludes a bitwise future-trajectory replay claim. The raw dense velocity/checkpoint chunks remain in the separate dataset, while the compact curves contain only one-dimensional lag and VACF arrays.

## Native integrator and checkpoint verification

### Coincident physical velocities

OpenMM's built-in Verlet stores a half-step velocity. The reference runner explicitly converts between that value and the velocity at the position's time:

`v_engine = v_physical − dt × F(q)/(2m)` on input; the inverse relation is used for observations and canonical momenta.

At identical physical q and p, Reference-platform native Verlet agrees with the independent PyRPMD P=1 normal-mode BAB map. A zero-friction LangevinMiddle control gives the same synchronized state. The recorded 216-water, 1 ps numerical check is [native-verlet-numerics.json](native-verlet-numerics.json):

| Time step | Maximum sampled energy change, kcal/mol/water |
|---:|---:|
| 0.5 fs | 0.00256503 |
| 0.25 fs | 0.000552304 |
| 0.125 fs | 0.000300661 |

These short trajectories test numerical energy behavior. They do not establish the time-step error of a diffusion coefficient.

Plain CustomIntegrator velocity-Verlet was rejected after an independent probe captured its actual last-kick force. On this runtime its intermediate virtual M-site positions were stale, even though a fresh endpoint force query appeared correct. The force error was about 2.51e−5 Hartree/bohr and the one-step momentum defect about 5.18e−5 atomic units. The registered classical campaign uses built-in Verlet. [native-validation.json](native-validation.json) and the [official-source audit](openmm-source-audit.json) retain the evidence.

### Explicit checkpoint-contract revision at 47 ps

The original runner incorrectly required the reconstructed canonical momentum to be bitwise identical after a CPU checkpoint reload. Both initial chains stopped safely at their first attempted continuation from 47 ps. No integration occurred after the failed check and no committed data were replaced.

The [read-only checkpoint audit](checkpoint-audit.json) found:

- Physical positions were bitwise identical after restoring the historical checkpoint.
- A separately measured before/after round trip preserved **all particle positions, including virtual sites, and all raw half-step velocities bitwise**.
- Recomputed CPU forces changed by about 2.06e−7 to 2.24e−7 in relative RMS, consistent with force-cache reconstruction at finite precision.
- The largest canonical momentum discrepancy was 1.81e−7 atomic units. The measured force difference explains that discrepancy through the exact half-kick conversion to within 3.7e−15 atomic units.
- A 20-step comparison after restart differed by at most 4.13e−8 bohr in position and 1.43e−6 atomic units in momentum. **CPU continuation is not claimed to replay a later trajectory bitwise.**

[Checkpoint contract v2](checkpoint-contract-v2.json) now stores redundant raw engine positions, half-step velocities, and forces. Positions and raw velocities must match exactly. The canonical-momentum discrepancy must equal the measured force-cache conversion, with residual at most 2e−13 atomic units. Independent force bounds also apply: CPU component absolute tolerance 0.005 kJ/mol/nm, relative tolerance 2e−6, and relative RMS difference at most 1e−6. Algebraic consistency alone cannot admit a changed force model.

The migration accepts only the exact historical source SHA plus unchanged force-field, constants, helper, configuration, environment, and input-checkpoint identities. The original manifest bytes and every old trajectory chunk remain available. Each migrated manifest records both source versions, the boundary step, checksums, and measured restoration diagnostics. Historical chunks did not contain redundant raw velocity/force arrays; the old force is explicitly labeled **inferred**, and independent raw-state identity is not retroactively claimed for those files.

Both original chains advanced from 47 to 48 ps through that explicit migration and then continued from new v2 checkpoints. No seed, physical setting, preparation duration, production length, or integration map changed. The [source patch](checkpoint-contract-v2.patch) permits reconstruction of the original runner in a separate audit checkout. Thirteen focused tests passed, including a real 216-water CPU checkpoint, altered-state rejection, migration restrictions, and preservation of historical bytes; see [checkpoint-v2-pytest.xml](checkpoint-v2-pytest.xml).

One recovery limitation remains explicit: interruption between writing the historical-manifest backup and publishing the migrated manifest can leave an orphan backup. The migration then stops instead of overwriting that evidence. The two actual migrations completed successfully.

## Quantum and MTS acceleration experiments

All experiments in this section are external numerical references. They do not modify the fixed classical protocol and do not demonstrate a converged quantum diffusion coefficient.

### Native RPMD convention

An independent native OpenMM RPMD / PyRPMD BAB comparison identified OpenMM 8.6's source-code value `hbar = 1.054571628e−34` SI. It makes the free-ring frequencies approximately `1.0000001798324383` times the modern-constant convention used by PyRPMD. A deterministic control that matches only this frequency convention, while keeping the same physical q and p, reduces the 20-step P=8 residual to about 1.08e−10 bohr and 6.68e−9 atomic momentum units. The control does not change a production sampling temperature. See [native-rpmd-frequency-control.json](native-rpmd-frequency-control.json).

### Explicit screened-Coulomb split and bead contraction

The experimental split retains the 9 Å oxygen LJ cutoff and uses a 5 Å real-space electrostatic cutoff. The smooth PME part carries its intramolecular Ewald exclusion corrections when contracted. The full uncontracted sum is checked against the original audited potential. The short/smooth split is not claimed to be the exact source implementation used in the original paper.

The source-size P=32 single-configuration probe gives approximately:

| Smooth-force copies K | Nonbonded-force RMS error | Centroid nonbonded-force RMS error |
|---:|---:|---:|
| 1 | 9.96% | 0.722% |
| 3 | 7.28% | 0.307% |
| 5 | 5.88% | 0.184% |
| 9 | 4.49% | 0.101% |
| 17 | 2.96% | 0.0455% |
| 32 | Numerical identity | Numerical identity |

The input ring received only the earlier short warm-up. These are configuration-specific error diagnostics, not equilibrium error bars. A small centroid-force error cannot by itself certify a diffusion coefficient. K and bead-count convergence still require independent ensemble/property comparisons.

The uncontracted 5 Å/9 Å split differs from the original 9 Å PME model by about 1.17e−5 relative RMS force at requested PME tolerance 1e−5. It preserves the same intended Ewald model within the measured numerical approximations. In the explicit rRESPA reference, halving both outer and inner steps lowers the short energy defect by approximately 4.7-fold, and momentum reversal returns the configuration to around 1e−13 bohr.

[split-validation.json](split-validation.json) reports both wall time and process CPU time. CPU scheduling is shared, so wall-time fluctuations are not interpreted as algorithmic speedups. Replacing an oxygen-only LJ interaction group with a second native, charge-free NonbondedForce preserved energy/forces in a nonzero-LJ test but did not improve the measured CPU cost.

The **5 Å direct Ewald cutoff above is not the paper's 5 Å Coulomb splitting length**. The original refined RPC paper, [Markland and Manolopoulos, CPL 464, 256–261 (2008)](https://doi.org/10.1016/j.cplett.2008.09.019), describes a full Ewald calculation on the centroid plus a short correction on each bead. The author's [lecture, slides 41–44](https://cscamm.umd.edu/programs/qcp10/presentations/Manolopoulos.pdf), separately identifies this short/long separation. The screened-cutoff experiment remains labeled as its own approximation.

### Compact polynomial separation under investigation

An openly available, author-uploaded primary follow-up supplies an explicit compact splitting function: Fanourgakis, Markland, and Manolopoulos, *J. Chem. Phys.* **131**, 094102 (2009), [DOI 10.1063/1.3216520](https://doi.org/10.1063/1.3216520), [author-uploaded paper](https://www.researchgate.net/publication/26798566_A_fast_path_integral_method_for_polarizable_force_fields), Eq. 35. For `x = r/sigma <= 1`, it is `f(x) = 7x/4 − 21x^5/4 + 7x^6 − 5x^7/2`; beyond that interval `f = 1`. The short potential is `(1−f)/r`, and it and its first three derivatives vanish at the boundary. The long potential is smooth near zero.

The independent [compact_split.py](../../../../benchmarks/water/compact_split.py) factory retains the original full Coulomb PME model and subtracts this compact potential in the slow group, while adding the identical compact potential to the full-bead short group. Thus the uncontracted sum retains the original force field, and the 5 Å splitting length no longer changes the 9 Å PME real-space cutoff. All intramolecular nonbonded exclusions remain consistent. The polynomial, nonzero-LJ full energy/force identity, and contracted-energy gradients have small Reference tests. This polynomial has **not** been established as the identical function used in the 2008 implementation; source-sized errors, timing, ensemble and transport checks remain open. No long quantum trajectory uses it yet.

### Delayed slow-kick native MTS

A separate classical CustomIntegrator construction stores the final slow half-kick as a velocity offset and applies a full slow kick at the start of the next outer step. Inner fast velocity-Verlet substeps depend only on physical intramolecular coordinates. The virtual sites are refreshed at the outer boundary before another slow-force evaluation. This avoids the intermediate virtual-site problem of the rejected ordinary custom VV map.

Its Reference q/p map and reversal agree with an independent explicit rRESPA implementation. Native RPMD contraction at K=1, 3, and the full P=8 ring also agrees with the independently constructed contracted Hamiltonian and BAB map after isolating the known hbar convention. See [native-optimizations-pytest.xml](native-optimizations-pytest.xml). These tests do not authorize replacing the registered classical baseline or using a larger time step without transport checks.

The [216-molecule timing record](native-optimization-timing.json) reports the following bounded controls on one CPU thread. Classical trajectories span 100 fs; quantum trajectories span 10 fs. The numbers below include both process CPU time and the actual contended wall time, and are not long-run throughput guarantees.

| External map | Steps, fs | CPU seconds per step | Wall seconds per step |
|---|---:|---:|---:|
| Classical native Verlet | 0.25 | 0.00598 | 0.02561 |
| Classical delayed-kick MTS | 0.5 / 0.1 | 0.00732 | 0.02363 |
| Classical delayed-kick MTS | 1.0 / 0.1 | 0.00888 | 0.02768 |
| P32 screened-cutoff native RPMD, K32 | 0.5 | 0.49519 | 1.75796 |
| P32 screened-cutoff native RPMD, K1 | 0.5 | 0.12849 | 0.46740 |
| P32 screened-cutoff native RPMD, K5 | 0.5 | 0.18882 | 0.52177 |
| P32 screened-cutoff native RPMD, K1 | 0.25 | 0.13855 | 0.48346 |

Even K1 at 0.5 fs would require about 7.14 CPU hours, or 26 wall hours at its measured allocation, for 100 ps. Extending that measured cost to the paper's 8000 ps production would require about 571 CPU hours or 87 days at that particular wall throughput, before preparation or error controls. The 0.5 fs single-step quantum map has not been certified for diffusion; these cost estimates are conditional, not a proposal to skip time-step or RPC convergence. The registered classical Verlet stage continues unchanged while a more suitable compact separation is investigated.

### Native RPMD checkpoint limitation

A direct probe found that ordinary `Context.createCheckpoint()` / `loadCheckpoint()` can both return successfully while **not restoring RPMD's individual ring copies**. Reference and CPU retained the advanced copies: after two further steps, restoration errors were about 0.0424 bohr and 1.756 atomic momentum units. The [probe record](native-rpmd-checkpoint-probe.json) prevents an API-success assumption from being mistaken for a valid restart.

A quantum runner must explicitly preserve and restore every copy's raw position and velocity and independently test stochastic continuation. [validate_native_optimizations.py](../../../../benchmarks/water/validate_native_optimizations.py) checks the combined raw-copy/binary-checkpoint approach. In the recorded small-system tests, the combined approach restored raw q/v exactly and reproduced both NVE and PILE continuation in fresh Reference and CPU contexts, with zero observed q/p discrepancy; see [native-optimization-checkpoints.json](native-optimization-checkpoints.json). Two further regression cases cover fresh-context NVE and stochastic restoration. These tests do not assume that a particular upstream limitation remains unfixed: they test the explicit restoration procedure. A long quantum campaign still needs a declared phase/RNG contract and source-size checks before launch.

### Prepared full-PME quantum jobs

A separate [full-PME runner and compute template](quantum-template/README.md) now makes the next quantum experiment concrete without launching it. It specifies 216 H2O, P32, full 9 Å PME on every bead, a conservative single 0.1 fs step, twenty independently seeded 100 ps preparations, and 100 fixed 4 ps NVE segments per chain. The resulting 8000 ps production matches the paper's production duration while avoiding an unvalidated contraction approximation.

This new runner uses the validated **geometric** external Andersen clock with mean 100 fs, saves its absolute deadline and RNG state, and removes only the total ring COM momentum at NVE initialization. Every-copy state restoration, phase transitions and an independent full-Hamiltonian map/energy comparison passed [twelve small tests](quantum-runner-pytest.xml). OpenMM 8.6.1 does not provide an RPMDIntegrator XML serialization proxy, so the verified reconstruction contract uses `integrator.json` alongside standard `model.xml`. The [focused receipt](quantum-runner-focused-gate.json) records the actual gates and remaining limits.

The CUDA double example is explicitly **untested**. No GPU resource or long CPU quantum production was started. Selected-device correctness, source-sized throughput/memory, bead/time-step controls and liquid equilibration remain necessary before that template can become a validated transport campaign. The existing classical protocol is unchanged.

## Running and inspecting the registered stage

Run commands from the repository root with the recorded Python/OpenMM environment. The baseline driver is [native_reference.py](../../../../benchmarks/water/native_reference.py), the streaming analyzer is [analyze_native.py](../../../../benchmarks/water/analyze_native.py), and the bounded queue driver is [native_campaign_schedule.py](../../../../benchmarks/water/native_campaign_schedule.py).

For a new chain:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENMM_CPU_THREADS=1 \
python benchmarks/water/native_reference.py \
  --config docs/validation/literature-convergence-2026-11-01/water/configs/classical-seed-1101.json \
  --output /your/raw-water-data/runs/classical-1101 --wall-seconds 1800
```

For an existing current-format chain, use `--resume` with its output directory. Historical v1 checkpoints additionally require the explicit `--migrate-checkpoint-contract` flag and the exact eligible source identity. The flag is not used again after migration.

The POSIX scheduler runs a fixed odd or even seed queue, pins the [registered source/protocol/configuration identities](native-queue-registration.json), and records committed step/segment and chunk-prefix hashes around every invocation. Three lightweight regressions verify stopping on child errors, refusing self-consistent replacement of a committed chunk, and requiring serialized-engine identity records. Run at most one odd queue and one even queue concurrently, only after manual invocations for those same chains have stopped:

```bash
python benchmarks/water/native_campaign_schedule.py \
  --protocol docs/validation/literature-convergence-2026-11-01/water/protocol.json \
  --output /your/raw-water-data --queue odd \
  --wall-seconds 7200 --invocation-seconds 1800
```

Use `--queue even` for the other worker. A child error stops its queue; it never replaces a seed, discards a segment, changes a platform/thread count, or adapts settings to a diffusion value.

The analyzer verifies raw chunk/model hashes and streams one NVE segment at a time. It validates the registered source and configuration, handles checkpoint metadata independently of observation arrays, and rejects missing, duplicate, nonuniform, nonfinite, or wrongly shaped records. The actual raw chunks, engine XML, RNG/checkpoint records, manifests, and queue logs belong to the separate data archive. Git retains compact reports, curves, tests, replay code, and source/hash references.

### Frozen sources and portable scheduler tests

[code-freeze.json](code-freeze.json) records the 21 original benchmark/test files, compiled native hashes, original checkout revision, and four completed focused gates totaling 64 passing tests. Running simulations retain that recorded source identity even when the follow-up code is integrated into a newer checkout.

An isolated-checkout review identified that two scheduler unit tests depended on the live campaign registration and historical initial checkpoint. A [separate portability patch](scheduler-tests-portability.patch) replaces that dependency with a complete temporary source/configuration/protocol/initial-state/hash fixture. The real scheduler registration, manifest and prefix checks remain active, and subprocess failures/history replacements are still exercised. [Three independent tests pass](scheduler-tests-portability-pytest.xml); the [old/new source and patch receipt](scheduler-tests-portability.json) preserves provenance. The original frozen test file and all running scheduler/physics/protocol files were left unchanged; the patch applies to the integration checkout.
