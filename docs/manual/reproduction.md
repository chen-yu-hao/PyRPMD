# Paper systems and reproduction checklist

**Scope of this checklist.** The source-pinned plans below retain their original
audit boundaries. Subsequent integrations change several implementation gaps.
Use the [current method catalog](references.md), [capabilities](capabilities.md)
and [next-release notes](../releases/unreleased.md) for current support; a newer
implementation does not by itself complete a paper reproduction.

**First finish the H + H2/BKMP2 convergence record and, in parallel, the Richardson–Thoss 2013 two-state
model; then study q-TIP4P/F water equilibrium, diffusion and spectra.** These campaigns test rates,
nonadiabatic dynamics and condensed-phase nuclear quantum effects. All 111 inventory entries
have a task below; their primary papers and code paths are in the [source catalog](references.md).

## Task types and priorities

An original-paper system is the model actually used in that paper. An existing entry point may instead
be a substitute regression or a short workflow demonstration. A **paper reproduction** must match the PES,
parameters, initial distribution, dynamics, estimator and processing; a **substitute regression** tests
implementation; a **new application** needs a separately justified model/reference. Label these separately.
This is a documentation/source audit and calculation plan, not a new production-simulation result.
Existing evidence is recorded in the frozen
[literature-validation inventory](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/literature-validation.md) and
[campaign registry](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/literature/campaign.json).

| Priority | Meaning | Deliverable |
| --- | --- | --- |
| P0 | Analytic/numerical controls and interface checks | Reference values, errors and dt/grid/bead checks |
| P1 | Recommended early paper-model or central-method campaign | Target figure/table, independent-seed intervals and convergence record |
| P2 | Needs data, substantial sampling or method completion | Resolve the stated gaps and verify intermediate quantities first |
| P3 | Complex composition or substantially incomplete source workflow | Complete model/sampler/estimator support before registering production |

These priorities are recommendations, not paper claims or runtime estimates. Mixed labels such as P0/P2
mean a control experiment followed by the more demanding physical-system campaign.

## Eight campaigns to start with

| Order | System and methods | Compute | Gate |
| --- | --- | --- | --- |
| 1 | Harmonic/quartic controls; PIMD, integrators, baths, estimators | Position variance, energy, heat capacity, momentum distribution; P/dt/autocorrelation | Agreement with analytic or independent references |
| 2 | H + H2 / BKMP2; RPMD rate | PMF, k_QTST, kappa(t), k(T), H/D KIE | Complete parent-time, independent-umbrella precision and dynamic P256 checks |
| 3 | Richardson–Thoss 2013 coupled harmonic model | Fig.1(c) position/electronic correlations, average sign and ESS | lowest_adiabatic splitting; bead, dt and sample convergence |
| 4 | Tully I–III; FSSH/MASH/RPSH-CA/BA/DM | Four scattering channels, population, coherence, purity and active-state fractions | Identical physical incoming distribution and correct asymptotic channels |
| 5 | q-TIP4P/F water; PIMD, RPMD, PA-CMD/TRPMD | RDF, energy, D from MSD/VACF, dipole correlation and IR | Separate classical/quantum and equilibration/real-time dynamics; source Fig.5 is PA-CMD |
| 6 | OGRe double-well/Ackley and OPES Müller models | FES, barriers, RMSE, coverage, ESS and total force evaluations | Include pilot cost; separate OPES variants and compare fixed history with PLUMED |
| 7 | Eckart and bound symmetric double wells; instantons | k(T), path, action, fluctuation determinant and splitting | Bead/Hessian convergence; molecular rotation/translation treatment first |
| 8 | 2008 PCET model then 2019 Au/TEAH+ | k(U,T), forward/reverse rates, H/D KIE and vibronic contributions | Validate local model first; add paper distance averaging/double-layer terms |

Sources: [RPMD rates](references.md#rpm-rate), [thermal NRPMD](references.md#na-thermal-nrpmd),
[Tully](references.md#na-tully-models), [water](references.md#water09), [sampling](references.md#samp-adaptive),
[instantons](references.md#rpi), [PCET](references.md#pcet).

## Outstanding conditions in existing evidence

H + H2 has fixed-P=128 agreement within the paper precision in the four-pair extension. The supplementary
parent-half equivalence check still fails; dynamic P=256 production is incomplete; independent umbrella
precision remains a separate condition. A matching final rate alone does not finish the convergence study.
The eight-seed classical OpenMM q-TIP4P/F reference is complete, but preparation-energy checks, the
short-lag MSD screen and timestep-equivalence intervals still leave issues. Quantum and PyRPMD production
transport are unrun. H + CH4, OH + H2, F + H2 and Ni-surface campaigns have distinct readiness states;
consult the frozen inventory rather than labeling them all reproduced.

## Required record for every calculation

1. Paper DOI/version, target figure/table and errata; reference numbers, digitization uncertainty or independent reference method.
2. PES/model bytes and SHA256, software commit/build/device, units, masses, electronic/nuclear states, boundary and symmetry conventions.
3. Temperature, beads, dt, initialization/equilibration, independent seeds, windows/CVs, parent/child durations and output interval.
4. The same estimator and normalization as the paper. An arbitrary FES additive constant may be removed; a fitted scale cannot conceal disagreement.
5. Independent-chain or correlated-block uncertainty, ESS, unresolved trajectories and average sign. Shared walkers/exchange slots are correlated.
6. Predetermined refinement ranges and equivalence margins. Separate physical-model, statistical and discretization errors; no universal 5% pass rule applies.

Archive failed/unresolved runs and original inputs. Choose gates for the actual observable before looking
at agreement with the paper, rather than tuning a threshold to obtain a pass.

## Existing starting commands

Install the package and run from the checkout root. These budgets are starting points, not convergence
guarantees. External PES/PLUMED/electronic-structure dependencies follow their guides. Entry points were
checked in this audit; new production simulations were not run.

```bash
python examples/nonadiabatic/published_nrpmd_benchmark.py --plan
python benchmarks/fes_methods/ogre_benchmark.py build --output runs/ogre
python benchmarks/fes_methods/ogre_benchmark.py run --model double_well --method adaptive --seed 4101 --output runs/ogre
python benchmarks/tier2_literature/sampling.py reference --output runs/tier2-sampling
python benchmarks/tier2_literature/sampling.py opes --output runs/tier2-sampling
python examples/sampling/committor.py --shots 64 --beads 4 --output runs/committor-example
```

For the paper-specific NRPMD production run retain `--potential-split lowest_adiabatic` and the supplied
paper digitization. Continue the documented unfinished H + H2/water conditions instead of replacing
their registered campaigns with a short default input.

## Complete task inventory

Only verified original-system information is recorded. Where the accessible source did not recover the
specific UI biological model, a complete constrained-dynamics application protocol or the six TREQ
reaction identities, the gap is stated explicitly; no guessed alanine/water/NH3 substitute is labeled original.


### Equilibrium, dynamics, rates and spectroscopy reproduction tasks

#### task-pimd

**Primitive distinguishable-particle PIMD — P0**

Papers: [cw81](references.md#cw81), [pr84](references.md#pr84), [pile10](references.md#pile10).

**Original system.** Foundations: quantum/classical ring isomorphism; the early Parrinello–Rahman application is an F center (excess electron) in molten KCl. That electronic model is not a bundled nuclear benchmark.

**Compute.** First obtain harmonic position variance, kinetic/potential/total energy and P convergence against analytic quantum values. Then compare water RDF and energy under a specified water model. Report effective sample sizes and independent-seed uncertainty.

**Existing entry.** [examples/quantum_statistics/harmonic_validation.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/quantum_statistics/harmonic_validation.py); [benchmarks/tier2_literature/path_integral.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/tier2_literature/path_integral.py); [docs/native-ensembles.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/native-ensembles.md)

**Gaps and boundaries.** Do not call a harmonic regression a reproduction of the molten-KCl paper. Exact original Hamiltonians and parameter files must be matched for a molecular reproduction.

[Implementation and scope](references.md#pimd).

#### task-rpmd

**RPMD real-time correlations — P0**

Papers: [rpm04](references.md#rpm04), [kubo57](references.md#kubo57).

**Original system.** Craig–Manolopoulos low-dimensional harmonic/anharmonic model correlation functions.

**Compute.** Compute C_qq^K(t), its frequency spectrum, short-time error and P/dt convergence against independent quantum diagonalization; include P=1 and exactly solvable harmonic limits.

**Existing entry.** [benchmarks/trpmd/harmonic_spectrum.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/trpmd/harmonic_spectrum.py); [benchmarks/tier2_literature/spectroscopy.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/tier2_literature/spectroscopy.py)

**Gaps and boundaries.** Nonlinear observables and long-time quantum coherence are not generally exact. The existing quartic script uses the Cao–Voth model and an independently calculated Kubo reference, not digitized Craig–Manolopoulos data.

[Implementation and scope](references.md#rpmd).

#### task-rpm-rate

**Thermal RPMD bimolecular reaction rate — P1**

Papers: [bennett1977](references.md#bennett1977), [chandler1978](references.md#chandler1978), [rate05](references.md#rate05), [rate05b](references.md#rate05b), [bimol09](references.md#bimol09), [rpmrate13](references.md#rpmrate13), [bkmp21996](references.md#bkmp21996).

**Original system.** 2005: Eckart/model reactions. 2009: H + H₂, Cl + HCl and F + H₂; the original article has errata, which matter for conventions and reference values.

**Compute.** Compute W(ξ), barrier position/height, k_QTST, κ(t) plateau, final k(T), isotope ratios where applicable, P/dt/R∞/umbrella-width/window-overlap/parent-child-time convergence and independent replica intervals.

**Existing entry.** [benchmarks/h3/rpmd.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/h3/rpmd.py); [benchmarks/h3/convergence.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/h3/convergence.py); [docs/literature-validation.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/literature-validation.md); [docs/science.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/science.md)

**Gaps and boundaries.** Use the current validation inventory: H+H₂ has a successful fixed-P128 comparison, while P256 and umbrella-domain checks remain separate gates. Do not promote fixed-P agreement to full convergence. Corrected F+H₂ reference/PES/electronic conventions must be resolved separately.

[Implementation and scope](references.md#rpm-rate).

#### task-direct-rpmd

**Direct-trajectory thermal RPMD (dRPMD) — P1/P2**

Papers: [drpmd18](references.md#drpmd18).

**Original system.** The originating application is H₂ + H₃⁺ → H₃⁺ + H₂, resolving direct proton-hop and complex-mediated exchange through H₅⁺.

**Compute.** Compute reaction probability versus impact parameter/energy, integral cross sections, thermal k(T), direct/complex channel fractions and collision-lifetime distributions. Demonstrate b_max, launch-separation, propagation-time and censoring convergence.

**Existing entry.** [benchmarks/h3/drpmd.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/h3/drpmd.py); [docs/science.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/science.md); [docs/validation/drpmd-optimization-2026-10-30/README.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/validation/drpmd-optimization-2026-10-30/README.md)

**Gaps and boundaries.** The bundled H+H₂ case is a practical surrogate. Reproducing the original mechanism transition requires the H₅⁺ PES, isotope/channel labelling and original classifier/conditions; those must be supplied and verified.

[Implementation and scope](references.md#direct-rpmd).

#### task-state-rpmd

**Approximate harmonic initial-state-selected NE-RPMD — P1**

Papers: [nerp16](references.md#nerp16), [state20](references.md#state20), [state21](references.md#state21), [state22](references.md#state22).

**Original system.** State-selected method papers: Mu/H/D + H₂(v=0,1); polyatomic extension H + CH₄/CHD₃ and F + CHD₃ with C–H-stretch excitation.

**Compute.** Compute state-specific σ_vj(Ecoll), thresholds, b-dependent probabilities, initial total/vibrational/rotational energy distributions and isolated-reactant stability; converge β, P, dt, b_max and trajectory counts; compare QCT under the same energy definition.

**Existing entry.** [benchmarks/h3/state_selected_reproduction.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/h3/state_selected_reproduction.py); [benchmarks/h3/audit_state_selected_initial.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/h3/audit_state_selected_initial.py); [benchmarks/h3/STATE_SELECTED_REPRODUCTION.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/h3/STATE_SELECTED_REPRODUCTION.md); [examples/state_selected/ne_rpmd.inp](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/state_selected/ne_rpmd.inp)

**Gaps and boundaries.** Initialization implements a harmonic approximation, not an exact arbitrary anharmonic eigenstate. Exact original PES, state definition and temperature-like β must match. Existing reduced trajectory budgets cannot be labelled full paper statistics.

[Implementation and scope](references.md#state-rpmd).

#### task-qct

**Harmonic QCT and full-PES action-matched diatomic QCT — P1**

Papers: [qct65](references.md#qct65), [state20](references.md#state20), [state22](references.md#state22).

**Original system.** Foundational QCT: H + H₂ on a simple barrier PES. Modern NE-RPMD papers compare with QCT for atom–diatom or methane scattering.

**Compute.** Compute σ_vj(Ecoll), vibrational action and isolated-energy drift; compare harmonic and action-matched initial energy distributions and scattering under separately labelled fixed-collision-energy and fixed-total-energy protocols.

**Existing entry.** [examples/state_selected/qct.inp](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/state_selected/qct.inp); [benchmarks/h3/state_preparation_study.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/h3/state_preparation_study.py); [benchmarks/h3/STATE_PREPARATION_STUDY.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/h3/STATE_PREPARATION_STUDY.md)

**Gaps and boundaries.** Action quantization is semiclassical, not exact quantum eigenstate preparation. Full-PES action initialization is an H₃ diatomic benchmark extension, not a general polyatomic production method. A 1965-paper reproduction requires its historical PES, rather than substituting BKMP2 silently.

[Implementation and scope](references.md#qct).

#### task-state-switch

**Experimental smooth preparation-to-PES switching diagnostic — P2**

Papers: [state20](references.md#state20), [state22](references.md#state22).

**Original system.** No originating paper for this exact switching variant was identified. The implemented experiment uses H + H₂.

**Compute.** Measure work injected by switching, final reactant energy/action distributions, isolated-reactant stability and σ(E) sensitivity to switching schedule/time. Preserve separate energy labels and compare with harmonic and full-PES-action controls.

**Existing entry.** [benchmarks/h3/state_preparation_study.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/h3/state_preparation_study.py); [benchmarks/h3/STATE_PREPARATION_STUDY.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/h3/STATE_PREPARATION_STUDY.md)

**Gaps and boundaries.** Do not attribute this variant to Marjollet et al. or present it as an established correction that prepares pure quantum states. It is a diagnostic research extension.

[Implementation and scope](references.md#state-switch).

#### task-integrators

**Velocity Verlet, normal-mode BAB and Cayley BCB — P0**

Papers: [vv82](references.md#vv82), [pile10](references.md#pile10), [cayley19](references.md#cayley19).

**Original system.** Verlet reference: small water clusters. Cayley paper: harmonic, weakly anharmonic and quartic one-dimensional oscillators; PILE paper supplies normal-mode splitting context.

**Compute.** Plot stable-trajectory fraction, energy error/drift and reversibility error versus dt and P on the paper's one-dimensional models; verify P=1 and free-ring limits. Report force-evaluation cost separately from accuracy.

**Existing entry.** [docs/integrators.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/integrators.md); [src/pyrpmd/dynamics.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/src/pyrpmd/dynamics.py); [benchmarks/tier2_literature/path_integral.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/tier2_literature/path_integral.py)

**Gaps and boundaries.** Cayley stability does not imply zero time-step bias or exact anharmonic sampling. Match each paper's potential and β before reproducing a figure; routine harmonic regressions are distinct.

[Implementation and scope](references.md#integrators).

#### task-stochastic-integrators

**BAOAB and Cayley BCOCB — P0/P1**

Papers: [baoab13](references.md#baoab13), [bcocb20](references.md#bcocb20), [cayley19](references.md#cayley19).

**Original system.** BAOAB: model oscillators and alanine dipeptide. BCOCB: harmonic/anharmonic models and 32-molecule q-TIP4P/F water at 298 K, density 0.998 g cm−3.

**Compute.** Compare finite-dt configurational covariance, kinetic-energy estimates, autocorrelation time and stable fraction across P/dt. For original water target compare kinetic-energy bead convergence and IR time-step sensitivity with the identical force field.

**Existing entry.** [docs/integrators.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/integrators.md); [docs/native-ensembles.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/native-ensembles.md); [benchmarks/trpmd/harmonic_spectrum.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/trpmd/harmonic_spectrum.py)

**Gaps and boundaries.** Current Torch BCOCB rejects MTS inner_steps>1. Thermostat choice and timestep error must be stated; different splitting sequences should not be conflated.

[Implementation and scope](references.md#stochastic-integrators).

#### task-classical-baths

**Andersen, Cartesian Langevin and stochastic velocity rescaling — P0**

Papers: [and80](references.md#and80), [langevin07](references.md#langevin07), [svr07](references.md#svr07).

**Original system.** The source studies introduce/test canonical thermostats using model/Lennard-Jones systems; the SVR paper includes TIP4P water phases.

**Compute.** Verify kinetic-energy chi-square statistics and mean/variance, harmonic configurational covariance, autocorrelation/equilibration versus coupling time, and invariant distributions across dt. Include zero-energy and momentum-DOF conventions.

**Existing entry.** [docs/andersen-sampling.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/andersen-sampling.md); [docs/native-ensembles.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/native-ensembles.md); [src/pyrpmd/native_ensemble.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/src/pyrpmd/native_ensemble.py)

**Gaps and boundaries.** Sampling thermostats alter time correlations: use the production-dynamics protocol appropriate to the observable. PyRPMD's global discrete Andersen schedule is a documented implementation variant, not identical to every collision scheme in the original paper.

[Implementation and scope](references.md#classical-baths).

#### task-pile

**PILE-L and PILE-G normal-mode thermostats — P0/P2**

Papers: [pile10](references.md#pile10), [svr07](references.md#svr07).

**Original system.** Original PILE efficiency tests include water and hydrogen in palladium.

**Compute.** Compute bead/mode covariance, quantum kinetic/potential energy, integrated autocorrelation time and effective independent samples per force call as P and friction vary. Compare PILE-L/G with the original thermostat baselines.

**Existing entry.** [docs/native-ensembles.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/native-ensembles.md); [docs/integrators.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/integrators.md)

**Gaps and boundaries.** An equilibrium distribution check does not establish the molecular acceleration reported in the paper. Original water/Pd PES and sampling settings must be matched; PILE production is not automatically RPMD real-time dynamics.

[Implementation and scope](references.md#pile).

#### task-nhc

**Nosé–Hoover chain thermostat — P0**

Papers: [nhc92](references.md#nhc92), [yoshida1990](references.md#yoshida1990).

**Original system.** Original harmonic-oscillator and free-particle canonical/ergodicity tests.

**Compute.** Compute joint coordinate–momentum distribution, kinetic-energy distribution, extended-energy drift and equilibration versus chain length/coupling/substeps; compare analytic oscillator statistics.

**Existing entry.** [docs/native-ensembles.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/native-ensembles.md); [src/pyrpmd/thermostats.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/src/pyrpmd/thermostats.py)

**Gaps and boundaries.** Global-chain settings must be converged; deterministic thermostatting alone is not an ergodicity guarantee. Bond-constraint support is not provided for this route.

[Implementation and scope](references.md#nhc).

#### task-gle

**Cartesian and normal-mode GLE baths — P0**

Papers: [gle09](references.md#gle09), [pile10](references.md#pile10).

**Original system.** Original colored-noise thermostat: harmonic-frequency response and selective control of dynamical time scales.

**Compute.** Compare sampled joint momentum/auxiliary covariance with the Lyapunov target; measure position/energy autocorrelation versus oscillator frequency and matrix parameters; check canonical physical variance.

**Existing entry.** [docs/native-ensembles.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/native-ensembles.md); [src/pyrpmd/native_ensemble.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/src/pyrpmd/native_ensemble.py)

**Gaps and boundaries.** Generic GLE matrices are not automatically quantum-correct or optimally tuned. The parameterization and physical/auxiliary covariance conventions must accompany any reproduction.

[Implementation and scope](references.md#gle).

#### task-piglet

**PIGLET with externally calibrated matrices — P2**

Papers: [piglet12](references.md#piglet12).

**Original system.** Original first-principles liquid-water structural and nuclear kinetic-energy/momentum-distribution study.

**Compute.** With the original calibrated bath, compare kinetic energy and static observables against fully converged PIMD versus P and computational cost; independently verify mode covariance and centroid classical temperature.

**Existing entry.** [docs/native-ensembles.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/native-ensembles.md); [src/pyrpmd/native_ensemble.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/src/pyrpmd/native_ensemble.py)

**Gaps and boundaries.** External temperature/bead-specific matrices and the original electronic-structure setup are required. PIGLET NPT is rejected. The paper's transient anisotropic Gaussian momentum estimator is not implemented by selecting this bath.

[Implementation and scope](references.md#piglet).

#### task-ffl

**Fast-forward Langevin rescale thermostat — P0/P2**

Papers: [ffl18](references.md#ffl18).

**Original system.** Original tests: one-dimensional harmonic motion, liquid water and Lennard-Jones polymers.

**Compute.** Compare canonical distributions and integrated potential-energy autocorrelation versus damping; reproduce strong-damping sampling efficiency against ordinary Langevin, reporting force-call-normalized cost.

**Existing entry.** [docs/native-ensembles.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/native-ensembles.md); [src/pyrpmd/native_ensemble.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/src/pyrpmd/native_ensemble.py)

**Gaps and boundaries.** State which FFL variant is used. These are equilibrium-sampling comparisons, not evidence for unperturbed physical transport coefficients.

[Implementation and scope](references.md#ffl).

#### task-constraints

**Per-bead SHAKE/RATTLE bond constraints — P0**

Papers: [shake77](references.md#shake77), [rattle83](references.md#rattle83).

**Original system.** SHAKE original: constrained n-alkane molecular dynamics. RATTLE: velocity-form constrained dynamics.

**Compute.** Measure maximum bond and velocity-constraint residual, reversibility, constrained energy drift and dt/tolerance convergence. For physical replication, use the original molecular force field and thermodynamic state.

**Existing entry.** [examples/materials/rigid_water.inp](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/materials/rigid_water.inp); [docs/bond-constraints.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/bond-constraints.md)

**Gaps and boundaries.** Supported route is fixed-cell velocity Verlet with NVE/Andersen and Python fallback; native/Cayley/PILE/barostat combinations are rejected. The supplied zero-potential rigid-water example tests kinematics only, not realistic water or a quantum rigid-rotor/Fixman treatment.

[Implementation and scope](references.md#constraints).

#### task-rpc-mts

**Ring-polymer contraction and reversible multiple time stepping — P1/P2**

Papers: [rpc08](references.md#rpc08), [respa92](references.md#respa92), [mts16](references.md#mts16), [refrpc16](references.md#refrpc16).

**Original system.** RPC original: flexible liquid water with component-specific bead contraction. Modern real/imaginary-time multilevel demonstration: MP2 with DFT reference for the Zundel cation.

**Compute.** Compute force/energy agreement at Pc=P, observable bias versus contracted beads Pc, energy drift versus inner/outer dt, and wall time/expensive force-call savings. Molecular targets require matched force decomposition and electronic methods.

**Existing entry.** [docs/rpc-mts.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/rpc-mts.md); [src/pyrpmd/potentials/contraction.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/src/pyrpmd/potentials/contraction.py); [src/pyrpmd/multitime.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/src/pyrpmd/multitime.py)

**Gaps and boundaries.** Python opt-in APIs, not blanket acceleration of every workflow. Resonance and contraction errors require independent convergence. A cheap harmonic split tests algebra but does not reproduce water/MP2 accuracy.

[Implementation and scope](references.md#rpc-mts).

#### task-pacmd

**Partially adiabatic centroid molecular dynamics (PA-CMD) — P1**

Papers: [cmd93](references.md#cmd93), [cmd94](references.md#cmd94), [pacmd06](references.md#pacmd06).

**Original system.** PA-CMD comparison paper: liquid para-H₂ at two state points and ortho-D₂ at one. Existing package benchmark instead uses Cao–Voth's 1993 quartic potential V=q²/2+0.1q⁴ at β=5.

**Compute.** For the quartic pilot compute centroid Kubo Cqq(t), peak positions/widths and static moments against independent diagonalization/finite-P transfer matrices; converge P, adiabatic frequency Ω, dt, sampling and trajectory length. Original molecular target adds velocity correlations and diffusion.

**Existing entry.** [benchmarks/tier2_literature/spectroscopy.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/tier2_literature/spectroscopy.py); [benchmarks/tier2_literature/spectroscopy_static.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/tier2_literature/spectroscopy_static.py); [examples/spectroscopy/pacmd.inp](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/spectroscopy/pacmd.inp)

**Gaps and boundaries.** The quartic paper's plotted real quantum correlation is not the independently calculated Kubo reference used here. Full molecular PA-CMD needs the original liquid interaction and state points. Curvature/nonlinear-observable limitations remain.

[Implementation and scope](references.md#pacmd).

#### task-fcmd

**Fast centroid molecular dynamics by force matching — P1**

Papers: [cmd94](references.md#cmd94), [fcmd05](references.md#fcmd05).

**Original system.** Original f-CMD study: liquid para-H₂ and ortho-D₂; static structure and dynamical/diffusion comparisons with direct CMD.

**Compute.** Report held-out centroid-force error, force conservativity, static centroid distribution, Cqq(t)/velocity correlation, spectral or diffusion error, and speedup against direct PA-CMD under the same temperature and training distribution.

**Existing entry.** [benchmarks/tier2_literature/spectroscopy.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/tier2_literature/spectroscopy.py); [src/pyrpmd/cmd.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/src/pyrpmd/cmd.py); [docs/tier2-spectroscopy.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/tier2-spectroscopy.md)

**Gaps and boundaries.** The supplied local quadratic basis is not a general molecular force field. New nonlinear observables need conditional property models. QCMD, hybrid CMD and fast QCMD are not implemented by these exports/IBI helpers.

[Implementation and scope](references.md#fcmd).

#### task-trpmd

**TRPMD dynamics and relative IR spectra — P1**

Papers: [trpmd14](references.md#trpmd14), [rpm04](references.md#rpm04), [kubo57](references.md#kubo57), [water09](references.md#water09).

**Original system.** Original TRPMD tests include harmonic and Morse OH, methane and liquid q-TIP4P/F water. The water comparison uses 216 molecules at 300 K; the Morse oscillator offers a smaller first molecular target.

**Compute.** Compute dipole Cμμ(t), IR peak position/width/intensity ratios and friction dependence; show removal of ring-polymer resonances versus RPMD while quantifying broadening; converge P/dt/time/window/replicas. Harmonic regression precedes Morse OH, then water.

**Existing entry.** [benchmarks/trpmd/harmonic_spectrum.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/trpmd/harmonic_spectrum.py); [benchmarks/water/README.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/water/README.md); [docs/trpmd-spectroscopy.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/trpmd-spectroscopy.md)

**Gaps and boundaries.** TRPMD is approximate and can broaden peaks. Output is relative, without absolute electromagnetic/volume/refractive-index factors. Water09 PA-CMD and TRPMD14 spectra are different targets; selecting TRPMD does not reproduce PA-CMD.

[Implementation and scope](references.md#trpmd).

#### task-trpmd-rates

**TRPMD rate-comparison mode — P2**

Papers: [trpmd14](references.md#trpmd14), [trprate15](references.md#trprate15).

**Original system.** Original assessment: H+H₂, D+MuH, H+CH₄ and F+H₂, comparing rates above and below crossover.

**Compute.** Compare κ(t) and final k(T) versus internal friction at temperatures on both sides of Tc, with matched PMF and independent uncertainty. Separate friction bias from recrossing statistics.

**Existing entry.** [docs/trpmd-spectroscopy.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/trpmd-spectroscopy.md); [docs/science.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/science.md); existing RPMD rate workflow with explicitly selected TRPMD configuration

**Gaps and boundaries.** TRPMD friction is not a controlled error estimate for exact quantum rates, especially below crossover. Do not use thermostat-insensitive equilibrium tests as rate validation.

[Implementation and scope](references.md#trpmd-rates).

#### task-raman

**Relative isotropic/anisotropic Raman spectra — P0/P2**

Papers: [kubo57](references.md#kubo57), [raman21](references.md#raman21).

**Original system.** Kubo provides the original linear-response basis. Dundas et al. supplies the explicit invariant convention in the documentation, with acetone in molecular environments; it is not the historical origin of Raman scattering theory.

**Compute.** First test a synthetic harmonic polarizability model with analytic peaks, tensor rotations, depolarization ratio and temperature/laser factors. A molecular reproduction then needs the original PES, polarizability model and environment treatment.

**Existing entry.** [examples/spectroscopy/README.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/spectroscopy/README.md); [examples/spectroscopy/harmonic_properties.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/spectroscopy/harmonic_properties.py); [docs/tier2-spectroscopy.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/tier2-spectroscopy.md)

**Gaps and boundaries.** This is a relative correlation-spectrum implementation, not Dundas et al.'s polarizable-embedding electronic-structure method. A unique original-paper molecular reproduction cannot be claimed from a formula citation alone; absolute experimental cross sections require further prefactors.

[Implementation and scope](references.md#raman).

#### task-sfg

**Relative SFG susceptibility from time correlations — P0/P2**

Papers: [sfg02](references.md#sfg02), [kubo57](references.md#kubo57).

**Original system.** Original time-dependent theory: water–vapor interfacial SFG.

**Compute.** Compute tensor χijk(ω), signed Imχ and coherent |χ|²; verify causal transform/sign with analytic signals. Molecular target: matched water interface, dipole/polarizability surfaces, orientational convention and polarization components.

**Existing entry.** [examples/spectroscopy/README.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/spectroscopy/README.md); [docs/tier2-spectroscopy.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/tier2-spectroscopy.md); [src/pyrpmd/analysis/spectroscopy.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/src/pyrpmd/analysis/spectroscopy.py)

**Gaps and boundaries.** The package supplies a relative response kernel; interface/lab-frame axes, Fresnel/local-field factors, beam geometry, normalization and electronic background are caller responsibilities. Do not claim the entire Morita–Hynes water model is implemented.

[Implementation and scope](references.md#sfg).

#### task-ti-pcn

**Takahashi–Imada trace action and pCN sampling — P0**

Papers: [ti84](references.md#ti84), [pcn13](references.md#pcn13), [cw81](references.md#cw81).

**Original system.** TI original test: harmonic-oscillator trace convergence. pCN is a function-space MCMC construction, not a particular molecular force field.

**Compute.** For a confined harmonic oscillator compute partition-function differences and E/Cv versus P, comparing primitive O(P−2) and TI O(P−4) trace behavior; measure acceptance, autocorrelation and independent samples per cost.

**Existing entry.** [examples/quantum_statistics/harmonic_validation.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/quantum_statistics/harmonic_validation.py); [docs/quantum-statistics.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/quantum-statistics.md)

**Gaps and boundaries.** TI fourth-order accuracy applies to the trace and corresponding thermodynamic derivatives, not automatically off-diagonal density matrices or arbitrary structural observables. It is not a modified physical potential for RPMD. Current API excludes PBC, exchange, frozen atoms and mass-dependent PES.

[Implementation and scope](references.md#ti-pcn).

#### task-thermo-estimators

**Quantum kinetic energy, total energy and heat capacity — P0/P2**

Papers: [yam05](references.md#yam05), [ti84](references.md#ti84), [cw81](references.md#cw81).

**Original system.** Yamamoto original: (H₂)₂₂ and Ne₁₃ quantum clusters with fourth-order propagators.

**Compute.** Compute E(T), Cv(T), kinetic and potential contributions with analytic oscillator oracles; compare derivative step sizes, P extrapolation, estimator variance and autocorrelation. Original cluster target needs the precise pair potentials and cluster ensemble.

**Existing entry.** [examples/quantum_statistics/harmonic_validation.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/quantum_statistics/harmonic_validation.py); [docs/quantum-statistics.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/quantum-statistics.md)

**Gaps and boundaries.** Report the action and estimator together; physical kinetic energy is not fictitious ring-polymer momentum temperature. Cluster heat capacities and phase behavior cannot be validated by a single harmonic run.

[Implementation and scope](references.md#thermo-estimators).

#### task-isotopes

**Isotope free-energy and fractionation estimators — P1/P2**

Papers: [iso13](references.md#iso13), [iso14](references.md#iso14).

**Original system.** 2013: H/D and oxygen isotope effects in liquid water/vapor. 2014: direct fractionation estimators demonstrated for Zundel cation and simple hydrocarbons.

**Compute.** Compute ΔF(m→m′), ln α between environments and kinetic-energy mass integrals. Compare direct/FEP and intermediate-mass integration, effective sample size, log-weight variance and confidence intervals; use harmonic mass dependence as the first oracle.

**Existing entry.** [examples/quantum_statistics/harmonic_validation.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/quantum_statistics/harmonic_validation.py); [docs/quantum-statistics.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/quantum-statistics.md)

**Gaps and boundaries.** Poor-overlap reweighting is not rescued by a small formal standard error. Current trace-action restrictions limit direct reuse for a periodic liquid-water reproduction; original PES/phase conventions and mass-dependent corrections must be supplied.

[Implementation and scope](references.md#isotopes).

#### task-momentum

**Primitive open-path one-component momentum distribution — P0**

Papers: [cw81](references.md#cw81), [ceperley95](references.md#ceperley95), [mom10](references.md#mom10).

**Original system.** Foundational off-diagonal density-matrix relation; Lin et al. is a related original displaced-path formulation tested on one-dimensional models, water and ice, whose free-energy algorithm is not implemented here.

**Compute.** Compare open-end separation distribution, Fourier n(p), normalization, symmetry and second moment with analytic harmonic momentum variance; converge P, endpoint sampling, histogram/KDE width and Fourier range, preserving negative/noisy diagnostic values.

**Existing entry.** [examples/quantum_statistics/harmonic_validation.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/quantum_statistics/harmonic_validation.py); [docs/quantum-statistics.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/quantum-statistics.md)

**Gaps and boundaries.** Do not call this Lin et al.'s displaced-path free-energy estimator. It is primitive and one-component, not a full molecular three-dimensional distribution or the PIGLET TAG estimator; no clipping/renormalization should hide undersampling.

[Implementation and scope](references.md#momentum).

#### task-suzuki-chin

**Finite-difference Suzuki–Chin path integrals — P0/P2**

Papers: [suz95](references.md#suz95), [chin97](references.md#chin97), [sc16](references.md#sc16).

**Original system.** Kapil–Behler–Ceriotti: liquid water and ice with a neural-network potential trained to dispersion-corrected hybrid density-functional data.

**Compute.** First compare harmonic even-bead position moments/physical potential against exact finite-P and continuum values; verify finite-difference force convergence. Original target: RDF and thermodynamic observables versus P and force-call cost under the identical NN potential.

**Existing entry.** [examples/path_integral/suzuki_chin.inp](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/path_integral/suzuki_chin.inp); [benchmarks/tier2_literature/path_integral.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/tier2_literature/path_integral.py)

**Gaps and boundaries.** Current route is confined/nonperiodic and serial, with physical even-bead potential reporting but no complete SC total-energy/Cv/pressure estimator. Thus original water/ice reproduction requires additional coverage and the original potential.

[Implementation and scope](references.md#suzuki-chin).

#### task-bosons

**Quadratic-scaling bosonic exchange PIMD — P1**

Papers: [bose19](references.md#bose19), [bose23](references.md#bose23).

**Original system.** Original papers: trapped two-dimensional ideal/interacting bosons; 2023 also demonstrates large-system scaling and periodic helium. Existing package benchmark targets the published ideal two-dimensional harmonic-trap setting.

**Compute.** Compute symmetric density and potential/virial-derived total energy versus N, β and P; compare exact canonical ideal-boson recurrence and small-N explicit checks. Measure wall-time scaling and errors per independent replica.

**Existing entry.** [examples/path_integral/bosonic.inp](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/path_integral/bosonic.inp); [benchmarks/tier2_literature/path_integral.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/tier2_literature/path_integral.py); [benchmarks/tier2_literature/README.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/tier2_literature/README.md)

**Gaps and boundaries.** The representative-cycle potential need not equal a full factorial sum pointwise; compare symmetric ensemble observables. No periodic winding/superfluid estimator. Existing P=36 model agreement uses different thermostat and shorter trajectories than the paper and is not a complete original statistical campaign.

[Implementation and scope](references.md#bosons).

#### task-pigs

**Primitive positive-trial path-integral ground state (PIGS) — P0/P2**

Papers: [pigs00](references.md#pigs00).

**Original system.** Original PIGS applications: He₂Cl₂ cluster and liquid/solid ⁴He.

**Compute.** Compute midpoint position moments and physical potential energy versus projection length, imaginary-time spacing and trial width; compare analytic harmonic ground-state moments. Original helium target adds energy/structure with the original pair potentials and trial ansatz.

**Existing entry.** [examples/path_integral/pigs.inp](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/path_integral/pigs.inp); [benchmarks/tier2_literature/path_integral.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/tier2_literature/path_integral.py)

**Gaps and boundaries.** The auxiliary β=Pτ parameterizes projection and is not a physical temperature. Current workflow has no combined bosonic-exchange/PBC production route and no complete original helium energy estimator; do not claim helium reproduction from harmonic examples.

[Implementation and scope](references.md#pigs).

#### task-rpi

**Ring-polymer instanton thermal rate — P1**

Papers: [rpi09](references.md#rpi09), [rpi16](references.md#rpi16).

**Original system.** 2009 connection paper: symmetric/asymmetric one-dimensional Eckart barrier tests. 2016 paper supplies first-principles instanton-rate derivation.

**Compute.** Compute Tc, optimized path/action, gradient residual, Hessian signature, fluctuation prefactor and k(T); converge beads, optimization tolerance and finite-difference Hessian; compare one-dimensional Eckart against numerical quantum transmission integration away from crossover.

**Existing entry.** [examples/instanton/eckart.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/instanton/eckart.py); [examples/instanton/low_temperature.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/instanton/low_temperature.py); [examples/instanton/cross_checks.json](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/instanton/cross_checks.json); [docs/instanton.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/instanton.md)

**Gaps and boundaries.** Optimization success is not physical convergence; reports retain physical_converged=False. Dense Hessian size is bounded. A molecular claim also needs the matching PES, reactant partition functions, symmetry/rotation factors and bead-converged temperatures.

[Implementation and scope](references.md#rpi).

#### task-splitting

**Symmetric bound-well instanton tunnelling splitting — P1/P2**

Papers: [split11](references.md#split11).

**Original system.** Original ring-polymer splitting paper: full-dimensional HO₂ and malonaldehyde.

**Compute.** First use a one-dimensional symmetric double well: compare instanton action, fluctuation prefactor and ground-doublet Δ against independent Hamiltonian diagonalization; converge projection time, beads and optimization.

**Existing entry.** [examples/instanton/low_temperature.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/instanton/low_temperature.py); [docs/instanton.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/instanton.md); [docs/low-temperature.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/low-temperature.md)

**Gaps and boundaries.** Current splitting code rejects free translational/rotational modes and is not a gas-phase rovibrational splitting or multiwell multiplet implementation. HO₂/malonaldehyde paper replication is therefore an extension target, not a presently turnkey reproduction.

[Implementation and scope](references.md#splitting).

#### task-low-t-checks

**Harmonic TST, Wigner and Eckart cross-checks — P0**

Papers: [eyr35](references.md#eyr35), [wig32](references.md#wig32), [eck30](references.md#eck30).

**Original system.** Original theoretical activated-complex/weak-tunnelling and analytical-barrier papers; Eckart is an analytically defined barrier rather than a specific reactive molecule.

**Compute.** Compute classical/quantum harmonic k_TST, κ_Wigner, κ_Eckart and corrected k(T) against direct quadrature; verify high-temperature limit, detailed energy convention, forward/reverse barrier inputs and sensitivity near Tc.

**Existing entry.** [examples/instanton/cross_checks.json](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/instanton/cross_checks.json); [examples/instanton/low_temperature.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/instanton/low_temperature.py); [docs/low-temperature.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/low-temperature.md)

**Gaps and boundaries.** Wigner is a weak-tunnelling expansion, not a deep-tunnelling method; neither Wigner nor a fitted one-dimensional Eckart barrier establishes multidimensional accuracy or provides a uniform crossover correction.

[Implementation and scope](references.md#low-t-checks).

### Free energy, enhanced sampling and reaction paths reproduction tasks

#### task-samp-umbrella

**Umbrella sampling / 1D–3D FES — P1**

Papers: [samp-us1974](references.md#samp-us1974), [samp-us1977](references.md#samp-us1977).

**Original system.** The 1974/1977 papers studied the Lennard-Jones fluid, including the vapor-liquid coexistence region; the 1974 work used an inverse-twelve/soft-sphere fluid as its free-energy reference.

**Compute.** For the original systems, calculate free-energy differences over temperature and density. First validate this package on analytic 1D/2D/3D harmonic FESs and sampled-support coverage, then the classical OGRe double well.

**Existing entry.** [examples/fes/harmonic2d.inp](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/fes/harmonic2d.inp); [benchmarks/fes_methods/benchmark.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/fes_methods/benchmark.py)

**Gaps and boundaries.** Recover the complete LJ parameters, reference path and reference free energy from the original papers. The supplied harmonic examples are implementation regressions, not reproductions of the original LJ calculations.

[Implementation and scope](references.md#samp-umbrella).

#### task-samp-mbar

**MBAR free energies, histogram FES, overlap diagnostics — P0**

Papers: [samp-mbar2008](references.md#samp-mbar2008).

**Original system.** Optical-tweezer measurements of a DNA hairpin under different constant forces; reconstruction of the potential of mean force along molecular extension.

**Compute.** Calculate state free-energy offsets, target weights, the PMF and overlap matrix. Compare against PyMBAR using exactly the same umbrella samples, then check sampling-length and bin-size convergence.

**Existing entry.** [benchmarks/fes_methods/solver_check.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/fes_methods/solver_check.py); [benchmarks/h3/solver_check.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/h3/solver_check.py)

**Gaps and boundaries.** The original DNA measurement data are not supplied with the package. Recover the dataset, applied forces and unit conventions separately.

[Implementation and scope](references.md#samp-mbar).

#### task-samp-ui

**Gaussian umbrella integration — P0**

Papers: [samp-ui2005](references.md#samp-ui2005), [samp-uierr2006](references.md#samp-uierr2006).

**Original system.** The 2005 paper reports an analytic potential function and a biological system. The accessible primary abstract does not name the biological system, so its identity has not been guessed.

**Compute.** Calculate the mean force, PMF and barrier. On identical samples, compare UI/MBAR barriers, Gaussian window fits and support for any tail extrapolation.

**Existing entry.** [benchmarks/h3/convergence.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/h3/convergence.py); [benchmarks/h3/solver_check.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/h3/solver_check.py); [benchmarks/fes_methods/benchmark.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/fes_methods/benchmark.py)

**Gaps and boundaries.** Obtain the 2005 full text to recover the analytic potential parameters and biological model. H+H2/BKMP2 is an existing application benchmark, not the original UI-paper system.

[Implementation and scope](references.md#samp-ui).

#### task-samp-adaptive

**Adaptive umbrella placement and coverage planning — P0**

Papers: [samp-mbar2008](references.md#samp-mbar2008), [samp-us1977](references.md#samp-us1977), [samp-ogre2023](references.md#samp-ogre2023).

**Original system.** No single original paper describes the complete PyRPMD planner. The comparator OGRe paper includes 1D/2D analytic models, proton hopping in zeolites and interlayer stacking in COF-5.

**Compute.** Start with the OGRe Eq.13 double well and Eq.14 Ackley surface at P=1 and 300 K. Report FES/RMSE, coverage, overlap, window count, total PES calls and independent-seed uncertainty; include all pilot cost.

**Existing entry.** `python benchmarks/fes_methods/ogre_benchmark.py build --output runs/ogre`; `python benchmarks/fes_methods/ogre_benchmark.py run --model double_well --method adaptive --seed 4101 --output runs/ogre`

**Gaps and boundaries.** Ackley requires a specified normalization domain; the repository wall is part of that model definition. For P>1, a nonharmonic centroid FES is not the bare potential. These calculations do not reproduce the full OGRe planner.

[Implementation and scope](references.md#samp-adaptive).

#### task-samp-standard-meta

**Standard metadynamics — P1**

Papers: [samp-meta2002](references.md#samp-meta2002).

**Original system.** The original paper studied NaCl dissociation in water and alanine-dipeptide conformational changes in aqueous solution.

**Compute.** For NaCl, calculate an FES using ion separation and solvent-coordination information. For alanine, calculate F(phi,psi), basin free-energy differences, barriers and convergence histories. First check bias forces and finite-time FESs on the OGRe double well.

**Existing entry.** [examples/metadynamics/standard.inp](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/metadynamics/standard.inp); [benchmarks/fes_methods/ogre_benchmark.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/fes_methods/ogre_benchmark.py)

**Gaps and boundaries.** Prepare the same force field, water model and CVs as the original application. Ordinary biased simulation time does not directly provide a physical rate.

[Implementation and scope](references.md#samp-standard-meta).

#### task-samp-wt-meta

**Well-tempered metadynamics — P1**

Papers: [samp-wt2008](references.md#samp-wt2008).

**Original system.** Alanine dipeptide was used to demonstrate free-energy reconstruction in the original paper.

**Compute.** Calculate F(phi,psi), basin differences and barrier errors versus simulation time, bias factor and deposition parameters. Use independent seeds; hill count alone is not convergence evidence.

**Existing entry.** [examples/metadynamics/well_tempered_2d.inp](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/metadynamics/well_tempered_2d.inp); [benchmarks/fes_methods/metadynamics_compare.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/fes_methods/metadynamics_compare.py)

**Gaps and boundaries.** Match the molecular force field, temperature and solvent treatment to the full paper/SI. The supplied harmonic2D example is an implementation regression.

[Implementation and scope](references.md#samp-wt-meta).

#### task-samp-walkers

**Shared-bias multiple-walker metadynamics — P1**

Papers: [samp-walkers2006](references.md#samp-walkers2006), [samp-wt2008](references.md#samp-wt2008).

**Original system.** The original paper studied threading/unthreading of tetracationic cyclobis(paraquat-p-phenylene) and 1,5-dihydroxynaphthalene in acetonitrile, plus Langevin models in 1D/2D/3D spherical wells.

**Compute.** Calculate the 2D FES in center separation and solvent coordination, unthreading/rethreading barriers, FES error and wall time/total simulated time versus walker count. First perform an equal-budget OGRe double-well regression; estimate uncertainty across independent complete shared histories.

**Existing entry.** [docs/enhanced-sampling.md#shared-multiple-walkers](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/enhanced-sampling.md#shared-multiple-walkers); [docs/enhanced-sampling.md#shared-bias-multiple-walkers](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/enhanced-sampling.md#shared-bias-multiple-walkers)

**Gaps and boundaries.** Section 3 and Fig.3 use 300 K and 1–32 walkers; molecular simulation details are delegated to the original paper's ref.35 and must be recovered. Its WHAM-superposition postprocessing is not implemented merely by sharing a bias. Shared walkers are not independent replicates.

[Implementation and scope](references.md#samp-walkers).

#### task-samp-grid-meta

**Grid-accelerated metadynamics and FES projection — P1**

Papers: [samp-meta2002](references.md#samp-meta2002), [samp-wt2008](references.md#samp-wt2008).

**Original system.** Use the systems of the standard/WT-metadynamics papers. There is no independent original paper for the PyRPMD grid engineering.

**Compute.** Compare direct and grid bias energies, forces and FESs under the same trajectory parameters while refining grid spacing. Check nonuniform-bin volume factors and missing-support masks in marginal free energies.

**Existing entry.** [docs/grid-metadynamics.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/grid-metadynamics.md); `pyrpmd project RUN --axes 2 --output NEW_DIRECTORY`

**Gaps and boundaries.** Grid acceleration and projection consistency do not independently establish physical accuracy. Projected basin weights are not dynamical branching fractions.

[Implementation and scope](references.md#samp-grid-meta).

#### task-samp-opes

**OPES_METAD / frozen-bias reweighting — P0**

Papers: [samp-opes2020](references.md#samp-opes2020).

**Original system.** The original paper and SI include a double-well model, alanine dipeptide and alanine tetrapeptide.

**Compute.** Begin with double-well comparisons of KDE, bias, forces and normalization Z. Calculate frozen-bias FESs, ESS, coverage and basin free energies before moving to alanine.

**Existing entry.** [examples/opes/harmonic_2d.inp](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/opes/harmonic_2d.inp); [benchmarks/tier2_literature/sampling.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/tier2_literature/sampling.py)

**Gaps and boundaries.** Kernel compression and shared OPES walkers are absent. Wrapped Gaussian kernels differ from PLUMED's truncated kernels. Cross-code comparisons must match initialization constants, physical kernel widths, normalization and bias gauge.

[Implementation and scope](references.md#samp-opes).

#### task-samp-opes-explore

**OPES-explore and optional adaptive widths — P1**

Papers: [samp-explore2022](references.md#samp-explore2022), [samp-opes2020](references.md#samp-opes2020).

**Original system.** The original paper uses the Müller potential, alanine dipeptide and alanine tetrapeptide to study exploration and FES convergence with incomplete CVs.

**Compute.** Calculate interbasin transitions, FES error and coverage versus time. Separately compare metad/explore modes and fixed/adaptive widths. Faster exploration does not imply FES convergence.

**Existing entry.** `pyrpmd run examples/opes/adaptive_explore.inp --output runs/adaptive-explore`; `python benchmarks/tier2_literature/sampling.py opes --output runs/tier2-sampling`

**Gaps and boundaries.** tier2_literature/sampling.py already includes the published Müller model and x/y CV comparisons, but its default budget is not the original 200-million-step budget. Its adaptive-width rule is a package-specific comparison, not a full reproduction of PLUMED OPES.

[Implementation and scope](references.md#samp-opes-explore).

#### task-samp-reus

**Same-temperature replica-exchange umbrella sampling — P1**

Papers: [samp-reus2000](references.md#samp-reus2000).

**Original system.** The original multidimensional exchange paper uses alanine trimer capped by acetyl and N-methyl groups, an all-atom AMBER potential and distance-dependent dielectric; the O1–H5 distance is the reaction coordinate.

**Compute.** Calculate the distance PMF, helical/turn/extended populations, exchange acceptance, walker round trips and joint window autocorrelation. Compare with ordinary umbrella sampling at equal total cost.

**Existing entry.** `pyrpmd run examples/fes/replica_exchange_2d.inp --output runs/reus2d`

**Gaps and boundaries.** The supplied example is harmonic; the original peptide force field and geometry need separate preparation. Exchanged trajectories serve equilibrium sampling and cannot be used as physical RPMD dynamical correlation times.

[Implementation and scope](references.md#samp-reus).

#### task-samp-remd

**Temperature / Hamiltonian replica-exchange API — P1**

Papers: [samp-remd1999](references.md#samp-remd1999), [samp-reus2000](references.md#samp-reus2000).

**Original system.** The 1999 REMD paper uses the Met-enkephalin pentapeptide; the 2000 multidimensional extension uses capped alanine trimer.

**Compute.** Calculate energy distributions and conformational populations in temperature/Hamiltonian slots, exchange detailed balance and round trips. For P>1, additionally verify spring-action exchange factors and exact harmonic distributions.

**Existing entry.** pyrpmd.replica_exchange.ReplicaExchange (Python API); [docs/tier2-sampling.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/tier2-sampling.md)

**Gaps and boundaries.** Ring-polymer exchange at P>1 applies the classical exchange principle to this package's measure. Original molecular potentials, temperature ladders and initial states require separate preparation.

[Implementation and scope](references.md#samp-remd).

#### task-samp-biased-clock

**Audited biased-time acceleration and censored rate inference — P1**

Papers: [samp-imeta2013](references.md#samp-imeta2013), [samp-flooding2022](references.md#samp-flooding2022).

**Original system.** The OPES-flooding paper uses a 2D double well, gas-phase alanine dipeptide and chignolin folding/unfolding in explicit water. The infrequent-metadynamics paper provides an additional alanine route.

**Compute.** First compare unbiased and rescaled first-passage-time distributions, maximum-likelihood rates, censoring fractions and KS diagnostics in a double well. Vary deposition interval, bias barrier and transition-region exclusion.

**Existing entry.** pyrpmd.biased_kinetics.rescale_biased_time / estimate_biased_rate

**Gaps and boundaries.** Native flooding exclusion is not implemented. The caller must provide bias and transition_bias for every integration interval and evidence that the transition region is unbiased. Sparse FES snapshots cannot supply this clock.

[Implementation and scope](references.md#samp-biased-clock).

#### task-samp-tis

**Two-way canonical transition-interface sampling — P1**

Papers: [samp-tis2003](references.md#samp-tis2003), [samp-tis2005](references.md#samp-tis2005), [samp-tps1998](references.md#samp-tps1998), [samp-shoot2025](references.md#samp-shoot2025).

**Original system.** The original 2003 TIS paper studies isomerization of a diatomic molecule immersed in a simple fluid.

**Compute.** Calculate conditional crossing probabilities, effective positive flux, their product rate, path lengths/acceptance and mechanism mixing across independent path chains. Increase the path cutoff and compare against direct MD or an independently known model.

**Existing entry.** `python examples/sampling/transition_interface.py --samples 200 --burn-in 50`

**Gaps and boundaries.** The supplied example is a separable double well, not the original fluid model. Automated flux runs, stochastic one-way shooting and complete molecular-rate certification are absent.

[Implementation and scope](references.md#samp-tis).

#### task-samp-retis

**Positive-ensemble replica-exchange TIS — P1**

Papers: [samp-retis2007](references.md#samp-retis2007), [samp-tis2003](references.md#samp-tis2003).

**Original system.** The original 2007 RETIS paper studies DNA denaturation using the Peyrard–Bishop–Dauxois model.

**Compute.** First compare TIS/RETIS conditional probabilities, autocorrelation, walker labels and total cost in the same double well. The original-paper route requires a PBD potential and DNA-denaturation rates and sampling-efficiency comparisons.

**Existing entry.** pyrpmd.retis.ReplicaExchangeTIS (Python API)

**Gaps and boundaries.** Do not claim complete equivalence to original RETIS: the minus-interface flux ensemble, persistent checkpointing and MPI orchestration are absent.

[Implementation and scope](references.md#samp-retis).

#### task-samp-committor

**Fixed-full-ring RPMD configurational committor — P0**

Papers: [samp-comm2000](references.md#samp-comm2000), [samp-cp1934](references.md#samp-cp1934).

**Original system.** The primary molecular application by Bolhuis et al. examines alanine dipeptide in vacuum and water and tests whether phi/psi provide an adequate reaction coordinate. It does not introduce this package's ring-polymer extension.

**Compute.** First calculate p_B approximately 1/2 at a symmetric double-well barrier configuration and 0/1 at basin boundaries. Vary shots, timestep, horizon and bead count; then calculate P(p_B) on candidate alanine dividing surfaces.

**Existing entry.** `python examples/sampling/committor.py --shots 64 --beads 4 --output runs/committor-example`

**Gaps and boundaries.** For P>1, distinguish a fixed full ring from a centroid-conditioned distribution. Collapsing every bead onto the same centroid is not the equilibrium conditional ensemble. Keep unresolved shots; p_B_given_resolved is not the unconditional committor.

[Implementation and scope](references.md#samp-committor).

#### task-samp-pathcv

**Aligned RMSD and reference-path collective variables — P1**

Papers: [samp-kabsch1976](references.md#samp-kabsch1976), [samp-path2007](references.md#samp-path2007).

**Original system.** The 2007 path-CV paper studies alanine-dipeptide conformational transitions. Kabsch 1976 is a geometric optimal-rotation algorithm and has no associated chemical dynamics system.

**Compute.** Calculate s/z and full-ring Jacobians, verify rotation/translation invariance, and reconstruct F(s,z) and the path PMF. Test robustness to path-node count and lambda.

**Existing entry.** pyrpmd.advanced_cv.RMSD / PathCV; [docs/tier2-sampling.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/tier2-sampling.md)

**Gaps and boundaries.** Prepare the original reference path and molecular potential separately. Implementing CVs for a supplied path is not the complete global path-optimization protocol of the original paper.

[Implementation and scope](references.md#samp-pathcv).

#### task-samp-string

**Euclidean mean-force / restrained adaptive strings — P1**

Papers: [samp-string2002](references.md#samp-string2002), [samp-stringcv2006](references.md#samp-stringcv2006).

**Original system.** The 2002 string paper includes Lennard-Jones cluster rearrangements and magnetic-film switching. The 2006 CV-string paper studies alanine dipeptide using two versus four dihedral angles.

**Compute.** First verify normal-force residuals, node count and reparameterization on an analytic curved valley. For the 2006 route, calculate the MFEP, free-energy barrier and committor width as CV choice changes.

**Existing entry.** pyrpmd.adaptive_string.adaptive_mean_force_string; [docs/manual/adaptive-nrpmd.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/manual/adaptive-nrpmd.md)

**Gaps and boundaries.** A PES gradient yields an MEP only. The package should not be described as a complete arbitrary-nonlinear-CV MFEP or isocommittor construction. The original four-angle route requires the missing metric terms.

[Implementation and scope](references.md#samp-string).

#### task-samp-prfo

**Partitioned rational-function saddle optimization — P0**

Papers: [samp-prfo1985](references.md#samp-prfo1985).

**Original system.** The 1985 paper uses two analytic model surfaces, including the Cerjan–Miller model; its Fig.3 shows the minimum and saddles at ±(1,0).

**Compute.** Calculate saddle positions/energies, Hessian negative-mode counts, gradient residuals and sensitivity to initial guesses. Vary Hessian finite-difference spacing, then calculate the electronic barrier and imaginary frequency for H+H2/BKMP2.

**Existing entry.** `python benchmarks/transition_state_search/analytic.py --output runs/ts/analytic.json`; [benchmarks/transition_state_search/h3.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/transition_state_search/h3.py)

**Gaps and boundaries.** Match the original analytic model definitions and energy scaling. Optimization iteration counts depend on this implementation's control rules and need not reproduce the original counts.

[Implementation and scope](references.md#samp-prfo).

#### task-samp-cineb

**CI-NEB with improved tangent and FIRE relaxation — P0**

Papers: [samp-nebtangent2000](references.md#samp-nebtangent2000), [samp-cineb2000](references.md#samp-cineb2000), [samp-fire2006](references.md#samp-fire2006).

**Original system.** The original CI-NEB paper studies CH4 dissociative adsorption on Ir(111) and H2 on Si(100). The improved-tangent paper additionally includes Si exchange diffusion and an Al addimer on Al(100).

**Compute.** First verify the Müller–Brown MEP, barrier, highest-image true force, NEB force and convergence with image count/initial path; then use BKMP2. Reproducing the original slabs requires a separate DFT/ASE route.

**Existing entry.** `python -m pyrpmd prepare examples/rate/from_endpoints.inp --pilot --output preparation.json`; [benchmarks/transition_state_search/analytic.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/transition_state_search/analytic.py)

**Gaps and boundaries.** The original slab Hamiltonians/DFT and periodic fixed-layer setup are outside the current gas-phase prepare workflow. Original-system reproduction needs additional implementation; H3 is not a substitute for those original systems.

[Implementation and scope](references.md#samp-cineb).

#### task-samp-connectivity

**Covalent-radius connectivity and channel inference — P0**

Papers: [samp-radii2008](references.md#samp-radii2008).

**Original system.** The original parameter paper derives covalent radii from crystal-structure statistics for most elements through atomic number 96. It is a data source, not one molecular-dynamics test system.

**Compute.** Check radii, units and isotope handling against the original table. Validate inferred connectivity and threshold sensitivity on known reactant/product geometries.

**Existing entry.** `pyrpmd prepare`; [src/pyrpmd/covalent.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/src/pyrpmd/covalent.py)

**Gaps and boundaries.** The threshold, two-sided descents and equivalent-channel combination are package choices. The Cordero data paper does not establish correct reaction identification for arbitrary structures.

[Implementation and scope](references.md#samp-connectivity).

#### task-samp-metric

**General nonlinear-coordinate constrained-parent mass metric — P0**

Papers: [samp-bluemoon1989](references.md#samp-bluemoon1989), [samp-blue1998](references.md#samp-blue1998).

**Original system.** The original papers derive ensemble relations for general constrained reaction coordinates. The complete 1998 chemical application protocol has not been recovered, so no unverified molecule is assigned.

**Compute.** First test rate invariance under s, a*s and nonlinear monotonic f(s) on the same analytic potential; check mass metrics and constrained-parent weights. Then test a general bond-difference coordinate.

**Existing entry.** [examples/rate/general_coordinate.inp](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/rate/general_coordinate.inp); [tests/test_general_coordinate_rates.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/tests/test_general_coordinate_rates.py); [docs/general-coordinate-rates.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/general-coordinate-rates.md)

**Gaps and boundaries.** The nonlinear-CV mass metric cannot be omitted. Agreement of projected PMFs does not guarantee agreement of unbiased dynamical rates.

[Implementation and scope](references.md#samp-metric).

#### task-samp-rate-postprocess

**2D flux integration and signed multichannel postprocessing — P1**

Papers: [samp-blue1998](references.md#samp-blue1998), [bennett1977](references.md#bennett1977), [chandler1978](references.md#chandler1978), [rate05](references.md#rate05), [bimol09](references.md#bimol09).

**Original system.** The existing physical regression uses the two atom-labelled exchange exits of H+H2/BKMP2. It is not the original system of a separate new-method paper.

**Compute.** Calculate consistency of 1D/2D static flux from the same samples, kappa_i(t), channel rates and branching fractions using a shared-parent-chain bootstrap. Check complete dividing-surface support and entrance normalization.

**Existing entry.** `python benchmarks/h3/rate_postprocess.py --profile quick --workers 2 --output runs/h3-rate-postprocess`

**Gaps and boundaries.** Product-basin Boltzmann weights are not dynamical branching fractions. Overlapping transition-surface fluxes cannot be added unconditionally, and exit labels do not supply an extra factor of two.

[Implementation and scope](references.md#samp-rate-postprocess).

#### task-samp-statistics

**Correlation diagnostics and synchronized circular-block bootstrap — P0**

Papers: [samp-cblock1992](references.md#samp-cblock1992), [samp-geyer1992](references.md#samp-geyer1992).

**Original system.** The statistical papers do not specify a PyRPMD chemical system. AR(1) data with known covariance and analytic harmonic time series provide independent calibration cases.

**Compute.** Compare standard-error coverage, block-length sensitivity and independent-seed variance. Preserve a common resampling clock across REUS windows and a common parent chain across product channels.

**Existing entry.** [src/pyrpmd/analysis/fes_statistics.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/src/pyrpmd/analysis/fes_statistics.py); [docs/replica-exchange.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/replica-exchange.md)

**Gaps and boundaries.** Reporting g/ESS does not establish mixing. Do not present 5g, 20 blocks or a fixed overlap threshold as universal guarantees from the cited papers.

[Implementation and scope](references.md#samp-statistics).

#### task-samp-plumed

**PLUMED centroid-bias bridge — P1**

Papers: [samp-plumed2009](references.md#samp-plumed2009), [samp-plumed2014](references.md#samp-plumed2014).

**Original system.** The original software papers contain several examples. Validate the bridge with a specifically defined algorithm and model; the software name is not itself a physical method.

**Compute.** Compare CVs, bias, forces and one-step trajectories between this package and external PLUMED on the same model, then reproduce the original system of the chosen sampling algorithm.

**Existing entry.** `python examples/plumed/centroid_metadynamics.py --steps 200 --output runs/plumed-centroid`; `python benchmarks/tier2_literature/sampling.py plumed --output runs/tier2-sampling`

**Gaps and boundaries.** External-history management and automated resume have explicit limitations. An unconfigured PLUMED action is not an implemented PyRPMD feature.

[Implementation and scope](references.md#samp-plumed).

### Nonadiabatic methods and matrix models reproduction tasks

#### task-na-nrpmd-focused

**Focused nonequilibrium MMST NRPMD — P1**

Papers: [na-mm1979](references.md#na-mm1979), [na-stock1997](references.md#na-stock1997), [na-nrpmd2013](references.md#na-nrpmd2013), [na-focused2019](references.md#na-focused2019), [na-mint2018](references.md#na-mint2018).

**Original system.** 2019 paper: one-dimensional three-state Morse photodissociation models, plus one-mode two-state spin-boson models. The 2013 equilibrium coupled-harmonic model belongs to the separate signed-thermal row.

**Compute.** Electronic populations P_i(t), matching the paper's vertically excited nuclear distribution and focused MMST estimator; Three-state Morse dissociation branches and long-time populations; coherent oscillations in the one-mode spin-boson model; Converge timestep, bead count and independent nuclear trajectories separately; retain signed estimators and statistical errors

**Existing entry.** `pyrpmd run examples/nonadiabatic/population.inp --output runs/nrpmd`; `PYTHONPATH=src python benchmarks/nonadiabatic/population_dynamics.py --smoke --output population_smoke.json`

**Gaps and boundaries.** The two existing entries are teaching/synthetic-LVC regressions, not the 2019 parameter sets. Implement the three-state Morse parameters from Table I or the corresponding one-mode model with its initial distribution. Check the offdiagonal matrix-element convention: a paper parameter Delta/2 is not the same input as DiscreteSpinBoson.tunnelling=Delta.

[Implementation and scope](references.md#na-nrpmd-focused).

#### task-na-mint

**MInt mapping subflow integrator — P0**

Papers: [na-mint2018](references.md#na-mint2018).

**Original system.** The cited paper develops a mixed quantum-classical IVR; package reuse is at the level of the MInt Hamiltonian subflow.

**Compute.** Exact constant-matrix electronic rotation and nuclear momentum impulse at fixed geometry; Time reversal and timestep convergence with noncommuting matrix gradients; energy drift diagnoses integration, not the physical accuracy of approximate dynamics

**Existing entry.** NRPMDDynamics / SMNRPMDDynamics Python APIs; the normal NRPMD examples exercise this integrator.

**Gaps and boundaries.** Do not list the full MQC-IVR results as reproducible: the package does not supply that algorithm's IVR phases, amplitudes and prefactors.

[Implementation and scope](references.md#na-mint).

#### task-na-thermal-nrpmd

**Signed SEO thermal NRPMD and correlation/rate estimators — P1**

Papers: [na-nrpmd2013](references.md#na-nrpmd2013), [na-stock1997](references.md#na-stock1997), [na-mint2018](references.md#na-mint2018).

**Original system.** Richardson–Thoss 2013 Fig. 1(c), Eq. 15: H=p²/2+x²/2+[[x,1],[1,-x]], with m=hbar=omega=kappa=beta=Delta=1 and alpha=0, as explicitly transcribed in the repository's paper benchmark.

**Compute.** Kubo position autocorrelation C_xx(t); Electronic correlation with an initial SEO projector and final MMST population; Average sign, effective independent samples/between-chain uncertainty, bead and timestep convergence, and an independent sinc-DVR comparison; For a rate calculation, additionally verify reactant normalization, a correlation plateau and the plateau fitting interval

**Existing entry.** `python examples/nonadiabatic/published_nrpmd_benchmark.py --plan`; `python examples/nonadiabatic/published_nrpmd_benchmark.py --workers 3 --potential-split lowest_adiabatic --output runs/published-nrpmd --published-data examples/nonadiabatic/data/richardson_thoss_2013_fig1c.csv`; `pyrpmd nonadiabatic-rate examples/nonadiabatic/thermal_rate.inp --output runs/thermal-nonadiabatic`

**Gaps and boundaries.** The existing Fig. 1(c) entry is the most direct paper-reproduction starting point, but still needs independent convergence studies. Curves from the default trace split must be labeled as a different splitting approximation. The short thermal_rate.inp run cannot establish a physically converged rate; complex-matrix phase reweighting is not implemented.

[Implementation and scope](references.md#na-thermal-nrpmd).

#### task-na-adaptive-thermal

**Adaptive restrained-path signed NRPMD composition — P3**

Papers: [na-nrpmd2013](references.md#na-nrpmd2013), [samp-stringcv2006](references.md#samp-stringcv2006), [samp-mbar2008](references.md#samp-mbar2008).

**Original system.** NRPMD's original coupled-harmonic equilibrium model is a component check. The package's confined two-state and electrode examples are new composite validation systems, not systems from a single original adaptive-NRPMD paper.

**Compute.** First reproduce the 2013 NRPMD thermal correlation, then verify restrained free energies for a known reaction coordinate; After freezing the path, use independent production samples for free-energy differences/flux correlations; report signs, window overlap and statistical uncertainty; Check that fixed-path and adaptive-path results agree within their uncertainties

**Existing entry.** [docs/manual/adaptive-nrpmd.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/manual/adaptive-nrpmd.md); [docs/adaptive-electrochemical-string-pilot.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/adaptive-electrochemical-string-pilot.md)

**Gaps and boundaries.** Cite each component separately; do not invent a single origin paper for the composite workflow or claim a validated electrochemical rate from a short adaptation trajectory.

[Implementation and scope](references.md#na-adaptive-thermal).

#### task-na-sm-nrpmd

**Focused two-state SM-NRPMD — P2**

Papers: [na-sm2021](references.md#na-sm2021).

**Original system.** 2021 paper includes a low-temperature one-dimensional two-state model and seven harmonic spin-boson cases. For cases I–VII, H=p²/2+R²/2+[[R+epsilon,Delta],[Delta,-R-epsilon]], m=hbar=omega=beta=1; (Delta,epsilon)=(10,0),(4,0),(1,2),(1,0.5),(1,0),(0.1,1.5),(0.1,0).

**Compute.** Electronic populations under focused initialization and conservation of spin length; For the paper's equilibrium C_RR(t) and C_nn(t), use its Gamma_W initial density, weights and estimators together

**Existing entry.** `pyrpmd run examples/nonadiabatic/extensions/sm_nrpmd.inp --output runs/sm-nrpmd`

**Gaps and boundaries.** The existing example verifies focused propagation. Current initialization cannot directly reproduce the main equilibrium-correlation figures: add Gamma_W sampling and the matching estimators before claiming quantitative reproduction of those results.

[Implementation and scope](references.md#na-sm-nrpmd).

#### task-na-mash

**Classical-nuclei MASH and experimental centroid extension — P1**

Papers: [na-mash2023](references.md#na-mash2023), [na-mashrates2024](references.md#na-mashrates2024).

**Original system.** Original MASH: Tully scattering, spin-boson and pyrazine models. The 2024 rate follow-up uses an electron-transfer spin-boson model represented by a Brownian-oscillator reaction coordinate and an Ohmic bath.

**Compute.** State-resolved Tully I-III reflection/transmission probabilities versus incident momentum at P=1; Electronic populations and coherences using the required MASH |S_z| population sampling and observable weights; Extend to electron-transfer rates in the reaction-coordinate spin-boson model, compared with Marcus/exact references

**Existing entry.** `python examples/nonadiabatic/adiabatic_on_the_fly.py --method mash --beads 1`

**Gaps and boundaries.** The existing entry is a two-state interface demonstration. Add the paper's scattering/ET parameters and ensemble analysis; a generic LVC class does not provide the published multimode pyrazine parameters. Do not label the experimental P>1 MASHDynamics as CA-RP-MASH or claim implementation of a later reversible event integrator.

[Implementation and scope](references.md#na-mash).

#### task-na-carp-mash

**CA-RP-MASH (CLI method rp-mash) — P2**

Papers: [na-carp2025](references.md#na-carp2025), [na-mash2023](references.md#na-mash2023), [na-mashrates2024](references.md#na-mashrates2024).

**Original system.** Cao–Suarez–Kretchmer 2025 v1: Tully models, a three-mode pyrazine model, and spin-boson electron transfer. The published source being followed is explicitly the preprint version cited above.

**Compute.** State-resolved Tully transmission/reflection and population time series, starting with the P=1 limit; Spin-boson electron-transfer rates and nuclear quantum effects with the paper's preparation and bath discretization; Correct operator-pair weights W_CC=3, W_CP=2, W_PP=2|S_z(0)|, together with event-tolerance/max_step convergence

**Existing entry.** `pyrpmd run examples/nonadiabatic/extensions/rp_mash.inp --output runs/rp-mash`

**Gaps and boundaries.** The existing input is a teaching model. Add the published pyrazine/bath parameters, independent equilibrium or scattering preparation, and rate analysis. Arbitrary harmonic-ring population curves are not a reproduction of the paper's thermal rates. The current implementation uses native=False.

[Implementation and scope](references.md#na-carp-mash).

#### task-na-rpsh-ca

**FSSH / centroid-approximation RPSH — P1**

Papers: [na-tully1990](references.md#na-tully1990), [na-rpsh2012](references.md#na-rpsh2012), [na-rpsh2017](references.md#na-rpsh2017), [na-rpshdm2023](references.md#na-rpshdm2023).

**Original system.** Tully1990: three one-dimensional two-state scattering models. Shushkov–Li–Tully2012: two-state scattering with thermal total/state-selected rates. Shakib–Huo2017: Tully low-momentum reflection/transmission, with further models in the supporting information.

**Compute.** Four final-channel reflection/transmission probabilities versus p0; Save active-surface fractions, amplitude populations and electronic coherences separately; none is a substitute for the others; To reproduce the 2012 thermal rates, include incident-flux normalization and thermal averaging instead of calling a final-state fraction a rate

**Existing entry.** `pyrpmd run examples/nonadiabatic/rpsh_population.inp --output runs/rpsh`; `python examples/nonadiabatic/adiabatic_on_the_fly.py --method rpsh --beads 1`

**Gaps and boundaries.** The existing inputs are short population demonstrations. Full paper scattering requires a momentum scan, incident wavepacket or corresponding ring initial density, channel classification and sufficient propagation time. Tully III has nonvanishing asymptotic coupling on one side, so vanishing coupling is not a general stopping criterion.

[Implementation and scope](references.md#na-rpsh-ca).

#### task-na-rpsh-ba

**Bead-average RPSH (RPSH-BA) — P1**

Papers: [na-rpshdm2023](references.md#na-rpshdm2023), [na-rpsh2012](references.md#na-rpsh2012), [na-tully1990](references.md#na-tully1990).

**Original system.** Shakya et al. 2023 compares CA, BA, and DM for Tully's extended-coupling model (Model C/III), including p0=20 and 30 atomic units, mass 2000, initial centroid x0=-15, width parameter 2, and 64 beads in the reported study.

**Compute.** Compare electronic densities, coherences and state-resolved reflection using the same physical preparation as CA/DM; Report both active-surface statistics and electronic amplitude estimators

**Existing entry.** `pyrpmd run examples/nonadiabatic/extensions/rpsh_ba.inp --output runs/rpsh-ba`

**Gaps and boundaries.** The example is not the full paper ensemble. Copy the width, temperature and sampling definitions from the original equations: a width parameter of 2 must not be silently identified with another interface's Gaussian sigma convention.

[Implementation and scope](references.md#na-rpsh-ba).

#### task-na-rpsh-dm

**Density-matrix RPSH (RPSH-DM) — P1**

Papers: [na-rpshdm2023](references.md#na-rpshdm2023).

**Original system.** Same Tully III/Model C settings as the BA row, directly from the 2023 CA/BA/DM comparison.

**Compute.** rho_ij(t), Tr(rho^2), and coherence decay, alongside channel-resolved reflection/transmission; Verify phase/unitary-basis invariance separately, distinguishing bead mixing from numerical phase errors

**Existing entry.** `pyrpmd run examples/nonadiabatic/extensions/rpsh_dm.inp --output runs/rpsh-dm`

**Gaps and boundaries.** Total populations alone do not test the paper's decoherence claims: include its density-matrix observables. Correct the existing guides' first-author initial from P. Shakya to Y. Shakya.

[Implementation and scope](references.md#na-rpsh-dm).

#### task-na-edc

**Energy-based decoherence correction (EDC) — P2**

Papers: [na-edc2007](references.md#na-edc2007).

**Original system.** Granucci–Persico2007: azobenzene photoisomerization and two deliberately designed model problems.

**Compute.** Internal consistency between electronic amplitude populations and active-state statistics; Decoherence timescales and reaction branching in the paper's models; Tully III can serve as an initial package regression substitute

**Existing entry.** RPSHDynamics(..., decoherence='edc', ...) and advanced CA/BA options documented in docs/nonadiabatic-hopping-extensions.md

**Gaps and boundaries.** Tully III is a practical substitute regression system, not the original 2007 paper's system. Reproducing azobenzene requires consistent electronic-structure theory, initial conditions and potential data.

[Implementation and scope](references.md#na-edc).

#### task-na-soc-rpsh

**Complex/SOC RPSH and local geometric force — P2**

Papers: [na-soc2019](references.md#na-soc2019), [na-tully1990](references.md#na-tully1990), [na-rpsh2012](references.md#na-rpsh2012).

**Original system.** Miao–Bellonzi–Subotnik2019: two-dimensional two-state complex Hamiltonian with theta(x)=pi/2[erf(Bx)+1], phi(y)=Wy and flat adiabatic energies ±A. The first illustrated case uses A=0.1, B=3, W=0.3.

**Compute.** First reproduce transverse momentum deflection and state-resolved scattering in the paper at P=1; Compare with and without Berry force against a two-dimensional wavepacket reference, and check constant-phase invariance; Treat the P>1 ring-polymer extension as a subsequent method comparison, not as the original 2019 calculation

**Existing entry.** `pyrpmd run examples/nonadiabatic/extensions/soc_rpsh.inp --output runs/soc-rpsh`

**Gaps and boundaries.** The current example is a synthetic complex-LVC model, not molecular intersystem crossing. The paper's two-dimensional potential and quantum benchmark need separate implementation; the package's SplitOperator1D cannot supply that reference.

[Implementation and scope](references.md#na-soc-rpsh).

#### task-na-dboc-rpsh

**Projected DBOC correction for RPSH — P2**

Papers: [na-dboc2025](references.md#na-dboc2025).

**Original system.** Limbu et al. 2025: Debye spin-boson bath discretized into 100 modes, symmetric/asymmetric cases, 300 and 30 K, and scans of reorganization strength and bath response time; compared with HEOM.

**Compute.** Donor population P_D(t) with and without DBOC, compared with HEOM; Use the paper's mixed quantum/classical population estimator: active-state diagonal terms plus amplitude coherences; Converge beads, timestep, bath discretization and the finite-difference displacement for DBOC gradients separately

**Existing entry.** `pyrpmd run examples/nonadiabatic/extensions/rpsh_dboc.inp --output runs/rpsh-dboc`

**Gaps and boundaries.** The short example is not the paper's Debye-bath/HEOM study. AdvancedRPSHDynamics.density_matrix explicitly returns amplitude density rather than the paper's mixed population estimator. Implement that observable for a paper-curve comparison; do not substitute |c|^2 or a generic saved density.

[Implementation and scope](references.md#na-dboc-rpsh).

#### task-na-qd-interface

**Quasi-diabatic NRPMD and adiabatic on-the-fly provider — P2**

Papers: [na-qd2018](references.md#na-qd2018), [na-qd2019](references.md#na-qd2019), [na-nrpmd2013](references.md#na-nrpmd2013), [na-mint2018](references.md#na-mint2018), [na-tully1990](references.md#na-tully1990).

**Original system.** 2018 QD paper: model/condensed-phase demonstrations with its own nonadiabatic propagators. 2019 on-the-fly paper: ethylene excited-state photodeactivation using CASSCF electronic data and QD-PLDM/SQC.

**Compute.** First use a complete analytic two-state model to check QD/fixed-diabatic NRPMD equivalence, phase invariance, overlap singular values and energy consistency; For a real molecule, record S1 populations, ground-state recovery and conical-intersection pathway branching against references with the same electronic theory and preparation

**Existing entry.** `python examples/nonadiabatic/adiabatic_on_the_fly.py --method qd-nrpmd --beads 1`

**Gaps and boundaries.** The current example is an analytic two-state interface demonstration. Reproducing the original ethylene study requires matching CASSCF data, overlaps/NACs, preparation and PLDM/SQC propagation. Current NRPMD curves are a different-method comparison on that system. A generic Python provider does not automatically supply a portable electronic-structure restart.

[Implementation and scope](references.md#na-qd-interface).

#### task-na-mv-rpmd

**Mapping-variable RPMD (MV-RPMD) — P2**

Papers: [na-mv2013](references.md#na-mv2013).

**Original system.** Ananth2013 Table I: four one-dimensional coupled harmonic two-state models, V0=R²/2, V11=R+c, V22=-R, V12=Delta, beta=M=k=a=1, with (c,Delta)=(0,0.1),(0,1),(0,10),(3,0.1).

**Compute.** Position autocorrelation C_RR(t) compared with exact DVR; Stationarity of equilibrium nuclear/electronic statistics, average sign and independent-sample uncertainty; Assess differences between finite-bead electronic discretizations after controlling bead convergence

**Existing entry.** `python examples/nonadiabatic/method_models.py`; MVRPMDPotential, MVRPMDSampler, and MVRPMDDynamics APIs

**Gaps and boundaries.** method_models.py is a teaching entry. Add all four published parameter sets, thermal sampling, correlation analysis and a DVR benchmark. Differences between finite-bead short-time electronic approximations must not be misclassified as implementation errors.

[Implementation and scope](references.md#na-mv-rpmd).

#### task-na-kc-rpmd

**Kinetically constrained RPMD (KC-RPMD) — P3**

Papers: [na-kc2014](references.md#na-kc2014).

**Original system.** Menzeleev–Bell–Miller2014: model A is one-dimensional unbound two-state scattering with temperature-dependent rates; model B is condensed-phase electron transfer with a solvent coordinate and a harmonic bath, scanning coupling and driving force into normal/inverted regimes.

**Compute.** Temperature-dependent rate k(T) for model A; Model B rates versus electronic coupling K and driving force, spanning nonadiabatic and adiabatic limits; Crossing calibration, auxiliary-coordinate free energy and a flux-side transmission plateau computed separately

**Existing entry.** `python examples/nonadiabatic/method_models.py`; KCCalibration, KCRPMDPotential, KCRPMDSampler and KCRPMDDynamics APIs

**Gaps and boundaries.** The existing demonstration is not a quantitative paper-rate reproduction. Supply physically consistent crossing averages, eta and auxiliary mass; a normalized rate benchmark with the paper's parameters remains to be built. Do not tune a/b or auxiliary mass merely to fit a reference curve.

[Implementation and scope](references.md#na-kc-rpmd).

#### task-na-isomorphic

**Two-state isomorphic potential with Ehrenfest dynamics — P0/P2**

Papers: [na-iso2018](references.md#na-iso2018).

**Original system.** Tao–Shushkov–Miller2018 studies a shifted quartic two-state model, a three-state donor–bridge–acceptor harmonic model, and two-state scattering; reported dynamical applications use QCLE/CMD or SH-RP-iso.

**Compute.** First reproduce the per-ring electronic Boltzmann trace identity, isomorphic coupling and analytic/finite-difference gradients for the paper's two-state potential; Reproduce paper correlation functions or state-resolved rates only after implementing the corresponding QCLE or iso-SH propagator

**Existing entry.** `python examples/nonadiabatic/method_models.py`; IsomorphicPotential and IsomorphicEhrenfestDynamics APIs

**Gaps and boundaries.** The present Ehrenfest curves must not be called reproductions of the paper's QCLE/iso-SH curves. Its three-state donor-bridge-acceptor model also exceeds the current two-state implementation. No implicit thermal sampler is supplied.

[Implementation and scope](references.md#na-isomorphic).

#### task-na-tully-models

**Tully I/II/III analytic matrix potentials — P0/P1**

Papers: [na-tully1990](references.md#na-tully1990).

**Original system.** The three one-dimensional two-state models from Tully1990: single avoided crossing, dual avoided crossing, and extended coupling with reflection. The conventional benchmark nuclear mass is 2000 atomic units.

**Compute.** Check potential matrices and derivatives against the original equations before momentum scans and four-channel scattering probabilities; Compare exact wavepackets, P=1 FSSH/MASH and P>1 RPSH under the same physical preparation

**Existing entry.** TullyModelOne/Two/Three APIs and documented saved-workflow model identities in docs/nonadiabatic-models.md

**Gaps and boundaries.** Available potential classes do not constitute a complete scattering benchmark. Specify incident distributions, width conventions, stopping regions and unclassified/trapped trajectories. Model III's offdiagonal coupling tends to 2b on one side, so coupling cannot be required to vanish in every asymptotic region.

[Implementation and scope](references.md#na-tully-models).

#### task-na-generic-matrix-models

**Generic LVC / discrete spin-boson matrix models and harmonic ring preparation — P1**

Papers: [na-focused2019](references.md#na-focused2019), [na-sm2021](references.md#na-sm2021).

**Original system.** Parameter-dependent: one-mode models from the cited NRPMD/SM-NRPMD papers are useful initial targets. No bundled generic LVC parameter set is automatically the paper's pyrazine/FMO Hamiltonian; a discrete bath also needs an explicitly chosen spectral-density discretization.

**Compute.** Compare initial harmonic-ring covariance with analytic finite-P values, declaring active coordinates and treatment of zero-frequency spectators; After selecting a paper's complete matrix parameters, compute its corresponding populations or equilibrium correlations; Confirm that tunnelling is the offdiagonal matrix element, with no hidden factor of one half

**Existing entry.** [examples/nonadiabatic/population.inp](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/nonadiabatic/population.inp); [benchmarks/nonadiabatic/population_dynamics.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/nonadiabatic/population_dynamics.py); sample_harmonic_ring API

**Gaps and boundaries.** No single original-model citation applies to all arbitrary LVC parameters. An uncoupled harmonic initial density represents vertical switching/excitation, not coupled equilibrium; free spectators also lack a normalizable canonical position distribution.

[Implementation and scope](references.md#na-generic-matrix-models).

#### task-na-nah2-pes

**NaH2 LEPS-5G coupled diabatic potential — P0/P3**

Papers: [na-nah21992](references.md#na-nah21992), [na-nah2err1994](references.md#na-nah2err1994).

**Original system.** Na(3p 2P)+H2 electronic quenching to Na(3s 2S)+H2 or reaction to NaH+H.

**Compute.** First verify reactant/product asymptotic energies, adiabatic surface cuts, derivatives/couplings and the crossing region; Then specify collision energy and the H2 rovibrational initial state to calculate quenching/reactive branches or state-resolved scattering cross sections

**Existing entry.** `pyrpmd run examples/nonadiabatic/extensions/nah2_sm_nrpmd.inp --output runs/nah2-sm-nrpmd`

**Gaps and boundaries.** The current molecular example is short propagation, not a reproduction of the original scattering study. Match PES parameters/erratum, state preparation and trajectory statistics; separately assess nondegenerate hopping algorithms near the crossing region.

[Implementation and scope](references.md#na-nah2-pes).

#### task-na-quantum-reference

**Finite-domain matrix split-operator quantum reference — P0/P1**

Papers: [na-feit1982](references.md#na-feit1982).

**Original system.** The Feit–Fleck–Steiger paper establishes the spectral method. The repository's present comparison is a synthetic one-mode two-state LVC vertical switch with beta=2; this is a package regression system, not a claimed reproduction of a particular figure from the 1982 paper.

**Compute.** Cross-check analytic Rabi oscillations, free-Gaussian translation/spreading and an independent dense-Hamiltonian exponential; Populations P_i(t) for the target one-dimensional matrix model, with separate timestep/grid/box sensitivities and boundary probability; Report NRPMD method differences separately from finite-bead and sampling errors; agreement of approximate NRPMD with quantum dynamics is not an implementation-correctness requirement

**Existing entry.** `PYTHONPATH=src python benchmarks/nonadiabatic/population_dynamics.py --smoke --output population_smoke.json`

**Gaps and boundaries.** --smoke uses only 16 trajectories, 8 beads and short propagation, and does not establish physical convergence. Avoid periodic wraparound in scattering calculations; the two-dimensional SOC paper requires a separate multidimensional quantum reference.

[Implementation and scope](references.md#na-quantum-reference).

### Materials, electrochemistry, potentials and interfaces reproduction tasks

#### task-bzp

**Isotropic BZP NPT PIMD — P1**

Papers: [bzp2009](references.md#bzp2009), [ipi2014](references.md#ipi2014).

**Original system.** BZP: Lennard-Jones liquid/solid near the triple point. i-PI: supercritical water at 750 K and 10 GPa.

**Compute.** Mean and variance of volume, pressure, volume autocorrelation, timestep and piston-relaxation dependence; compare P=1 before PIMD.

**Existing entry.** [examples/ensembles/native_npt.toml](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/ensembles/native_npt.toml); [docs/materials-npt.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/materials-npt.md)

**Gaps and boundaries.** Existing ideal-gas controls are substitutes. Obtain the paper potential and full thermodynamic protocol; NPT trajectories are not diffusion production.

[Implementation and scope](references.md#bzp).

#### task-mtk

**Flexible-cell MTTK/MTK PIMD — P2**

Papers: [mtk1994](references.md#mtk1994), [nptpimd1999](references.md#nptpimd1999).

**Original system.** Constant-pressure PIMD paper: periodic model particle, liquid/solid para-hydrogen and liquid butane.

**Compute.** Cell/volume distribution, pressure tensor and equilibrium energies; compare identical cells and reference-engine equations.

**Existing entry.** [docs/native-ensembles.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/native-ensembles.md); [examples/ensembles/native_npt.toml](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/ensembles/native_npt.toml)

**Gaps and boundaries.** Paper force fields and full inputs must be supplied; masked cell motion does not establish a slab NPT ensemble.

[Implementation and scope](references.md#mtk).

#### task-rgb

**Raiteri–Gale–Bussi stress coupling — P3**

Papers: [rgb2011](references.md#rgb2011).

**Original system.** Pure and Y-doped BaZrO3 proton diffusion with an EVB force field.

**Compute.** First cell/strain statistics under imposed stress; only then proton residence times and diffusion with the original EVB model.

**Existing entry.** [docs/native-ensembles.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/native-ensembles.md)

**Gaps and boundaries.** The paper's EVB material model is not bundled; external-stress coupling alone does not reproduce its diffusion result.

[Implementation and scope](references.md#rgb).

#### task-scr

**Stochastic cell rescaling — P1**

Papers: [scr2020](references.md#scr2020).

**Original system.** Lennard-Jones fluid; the original paper also tests a membrane with an anisotropic variant.

**Compute.** Volume mean/variance and relaxation versus timestep and coupling time; use isotropic liquid for the supported method.

**Existing entry.** [docs/native-ensembles.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/native-ensembles.md)

**Gaps and boundaries.** Do not use P>1 or call this the paper's anisotropic membrane algorithm; finite-step bias must be measured.

[Implementation and scope](references.md#scr).

#### task-diffusion

**RPMD MSD/VACF diffusion and tensor analysis — P2**

Papers: [pd2024](references.md#pd2024), [ni2012](references.md#ni2012).

**Original system.** Ni(100)/H EAM (2012); flexible Pd(111)/H with a DFT-trained neural potential (2024).

**Compute.** D(T), MSD, VACF, site-hop rates and multi-hop correlation; compare fixed/mobile substrate and surface/subsurface channels.

**Existing entry.** [examples/materials/diffusion.inp](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/materials/diffusion.inp); [benchmarks/materials_reference/README.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/materials_reference/README.md); [benchmarks/water/README.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/water/README.md)

**Gaps and boundaries.** Ni EAM4 is presently a static reference adapter; Pd neural weights/protocol must be obtained. The NiAlH pilot uses a different model.

[Implementation and scope](references.md#diffusion).

#### task-site-surface

**Basin-normalized site and surface rates — P2**

Papers: [sds2025](references.md#sds2025), [pd2024](references.md#pd2024), [bazro2008](references.md#bazro2008).

**Original system.** SDS 2025: H+H2, CH3NC isomerization, H recombination on Pt(111), NO desorption on Pd(111). BaZrO3 2008 is a centroid-TST comparison, not RPMD recrossing.

**Compute.** Basin partition, PMF barrier, flux factor, k_QTST, kappa(t), final directed rate; compare coordinate/normalization choices.

**Existing entry.** [examples/materials/site_rate.inp](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/materials/site_rate.inp); [examples/surface/diffusion.inp](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/surface/diffusion.inp); [docs/surface-catalysis.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/surface-catalysis.md)

**Gaps and boundaries.** Surface molecular PES and exact reactant-state convention must match; no automatic multi-site kinetic Monte Carlo network.

[Implementation and scope](references.md#site-surface).

#### task-nialh-model

**NiAlH EAM pilot potential — P2**

Papers: [nialh1995](references.md#nialh1995).

**Original system.** Hydrogen trapping at Ni dislocations and grain boundaries.

**Compute.** Verify forces/units and defect binding energies before a transport campaign.

**Existing entry.** [benchmarks/materials/README.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/materials/README.md)

**Gaps and boundaries.** The small Ni(100) pilot is not the original defect calculation or the 2012 EAM4 diffusion model.

[Implementation and scope](references.md#nialh-model).

#### task-pcet

**Local golden-rule electrochemical PCET — P1**

Papers: [pcet2008](references.md#pcet2008), [pypcet2025](references.md#pypcet2025), [gold2019](references.md#gold2019).

**Original system.** 2008 electrode/solute PCET model systems; 2019 application is TEAH+/TEAD+ discharge on Au in acetonitrile.

**Compute.** Forward/reverse k(U,T), H/D KIE, vibronic-channel contributions; converge proton grid, states and electrode quadrature.

**Existing entry.** [examples/pcet/harmonic_scan.inp](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/pcet/harmonic_scan.inp); [docs/validation/pcet-model-2026-10-30/README.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/validation/pcet-model-2026-10-30/README.md)

**Gaps and boundaries.** pyPCET agreement is an external implementation check. The local model omits the original application's distance averaging, double layer and broader coupling regimes.

[Implementation and scope](references.md#pcet).

#### task-gc

**Grand-canonical shared-electron HMC/MC — P2**

Papers: [hmc1987](references.md#hmc1987), [hastings1970](references.md#hastings1970), [gc2025](references.md#gc2025).

**Original system.** HMC originated in lattice field theory. Related electrode application: 5x5 four-layer Pt(111), 1 ML H, 36 water molecules; GC-(PI)HMC Volmer/Heyrovsky profiles.

**Compute.** Start with analytic charge model: joint q,N distribution, mean/variance of N, acceptance and grand-potential free energy; then classical/quantum interface barriers.

**Existing entry.** [examples/electrochemistry](https://github.com/chen-yu-hao/PyRPMD/tree/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/electrochemistry); [docs/electrochemistry.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/electrochemistry.md)

**Gaps and boundaries.** The complete Sun2025 constrained algorithm, DP-Ne model, charge-plate convention and mixed quantization are not supplied by this sampler.

[Implementation and scope](references.md#gc).

#### task-adiabatic-electrode

**Adiabatic constant-potential scalar and matrix wrappers — P1**

Papers: [gc2025](references.md#gc2025).

**Original system.** No separate original paper for this PyRPMD composition; Sun2025 is physical context, not an identity claim.

**Compute.** Compare analytic optimum N*(R), solver residuals, envelope-theorem forces and isotope/voltage free energies.

**Existing entry.** [examples/electrochemistry/adaptive_constant_potential.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/electrochemistry/adaptive_constant_potential.py); [docs/manual/adaptive-nrpmd.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/manual/adaptive-nrpmd.md)

**Gaps and boundaries.** Independent-bead minima differ from shared-N grand-canonical fluctuations; common charging lacks state-dependent response and is not TP-RPMD.

[Implementation and scope](references.md#adiabatic-electrode).

#### task-her

**Volmer–Heyrovsky–Tafel mean-field postprocessing — P1**

Papers: [gc2025](references.md#gc2025).

**Original system.** No unique paper for these algebraic balances; electrode papers motivate the elementary reactions only.

**Compute.** Coverage theta(U), separate elementary fluxes, H2 TOF, signed current; check steady-state electron/H2 stoichiometry and detailed balance.

**Existing entry.** [docs/electrochemistry.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/electrochemistry.md)

**Gaps and boundaries.** Does not derive rates from PMFs or local PCET, nor add transport, lateral interactions or double-layer physics.

[Implementation and scope](references.md#her).

#### task-efrpmd

**Markovian centroid electronic-friction RPMD — P2**

Papers: [efrpmd2024](references.md#efrpmd2024).

**Original system.** Anderson–Holstein chemisorption model, symmetric/asymmetric electron-transfer potentials.

**Compute.** Nuclear distributions (Fig. 2), population relaxation (Fig. 3), rate versus temperature (Fig. 4), and the asymmetric low-temperature discrepancy (Fig. 5). Match the mean-force model and friction before comparing.

**Existing entry.** [examples/nonadiabatic/friction_surface.inp](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/nonadiabatic/friction_surface.inp); [docs/electronic-friction.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/electronic-friction.md)

**Gaps and boundaries.** Existing corrugated-surface example is a substitute. Build the matching mean-force PES/friction tensor; memory effects and imaginary-time bath renormalization are absent.

[Implementation and scope](references.md#efrpmd).

#### task-pyscf

**PySCF references and on-the-fly force adapter — P0**

Papers: [pyscf2018](references.md#pyscf2018).

**Original system.** Software interface; no unique physical reproduction system. The selected electronic method, basis and functional require their own citations.

**Compute.** Same-geometry energy/gradient/Hessian versus direct PySCF; finite differences, spin/state continuity and convergence.

**Existing entry.** [examples/pesgen/water.toml](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/pesgen/water.toml); [docs/potential-adapters.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/potential-adapters.md)

**Gaps and boundaries.** RHF/STO-3G water is an integration control, not a validated reaction PES.

[Implementation and scope](references.md#pyscf).

#### task-populations

**Hirshfeld charges and Mayer bond orders — P1**

Papers: [hirshfeld1977](references.md#hirshfeld1977), [mayer1983](references.md#mayer1983).

**Original system.** Original papers define molecular charge/bond indices; current data bridge does not reproduce a complete original numerical table.

**Compute.** Charge conservation and grid convergence; compare restricted and open-shell examples against an independent same-convention calculation.

**Existing entry.** [examples/pesgen/water.toml](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/pesgen/water.toml); [docs/manual/pes-generation.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/manual/pes-generation.md)

**Gaps and boundaries.** Different promolecular atom tables yield different charges; do not compare ORCA defaults as identical definitions.

[Implementation and scope](references.md#populations).

#### task-qmdff

**QMDFF generation — P1**

Papers: [qmdff2014](references.md#qmdff2014), [caracal2023](references.md#caracal2023).

**Original system.** Original QMDFF molecular/condensed-phase benchmark suite, parameterized from TPSS-D3/def-TZVP(-f) data; local water is a substitute.

**Compute.** Compare equilibrium geometry, Hessian frequencies, distorted-geometry energies/forces and dissociation behavior before dynamics.

**Existing entry.** [examples/pesgen/water.toml](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/pesgen/water.toml); [docs/manual/pes-generation.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/manual/pes-generation.md)

**Gaps and boundaries.** Requires upstream build and suitable reference theory; recover the chosen original benchmark's SI geometry before claiming reproduction.

[Implementation and scope](references.md#qmdff).

#### task-deevb

**dE-EVB / EVB-QMDFF coupling fit — P2**

Papers: [deevb2015](references.md#deevb2015), [caracal2023](references.md#caracal2023).

**Original system.** Original examples include SN2 Cl−+CH3Br → CH3Cl+Br− and ethene+cyclopentadiene → norbornene (Diels–Alder).

**Compute.** Endpoint limits, coupling, path and off-path energy/force errors, barrier and final RPMD rate relative to the reference PES.

**Existing entry.** [examples/pesgen/nh3_inversion.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/pesgen/nh3_inversion.py); [docs/manual/pes-generation.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/manual/pes-generation.md)

**Gaps and boundaries.** The current neutral-only fitting workflow cannot run the charged SN2 case; use a recovered neutral Diels–Alder protocol for a faithful paper-system study. NH3 is a separate package control.

[Implementation and scope](references.md#deevb).

#### task-treq

**TREQ PES generation — P2**

Papers: [treq2019](references.md#treq2019), [caracal2023](references.md#caracal2023).

**Original system.** Original six-reaction comparison with analytical full-dimensional reference surfaces; the exact selected SI protocol remains to be transcribed.

**Compute.** Path/off-path energies and gradients, finite-difference step convergence, energy drift, then k(T) against the analytical PES.

**Existing entry.** [examples/pesgen/nh3_inversion.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/pesgen/nh3_inversion.py); [docs/manual/pes-generation.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/manual/pes-generation.md)

**Gaps and boundaries.** Experimental bridge; the analytic-force inconsistency documented for the pinned source prevents treating its default as the paper's force implementation.

[Implementation and scope](references.md#treq).

#### task-jaxpip

**Linear PIP / PIP-NN fitting and JaxPIP export — P2**

Papers: [pip2010](references.md#pip2010), [pipnn2013](references.md#pipnn2013), [jaxpip2026](references.md#jaxpip2026).

**Original system.** PIP monomial symmetrization: H3O+ fitted to about 62,000 ab initio energies (Xie–Bowman 2010). PIP-NN 2013: H+H2 and Cl+H2. JaxPIP 2026: 3–9 atom performance tests and OH+CH3OH QCT cross sections/branching (SI Tables S7–S8).

**Compute.** Holdout energy/force errors, permutation invariance, Python/native equality and batch scaling; then scattering cross sections with the same trained surface.

**Existing entry.** [docs/manual/jaxpip-generation.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/manual/jaxpip-generation.md); [examples/pesgen](https://github.com/chen-yu-hao/PyRPMD/tree/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/pesgen)

**Gaps and boundaries.** Small H2 examples do not reproduce OH+CH3OH; obtain exact basis/model weights, train split and QCT initial-state convention.

[Implementation and scope](references.md#jaxpip).

#### task-committee

**Committee uncertainty and acquisition — P2**

Papers: [committee2020](references.md#committee2020).

**Original system.** Condensed water across liquid state points, ice phases and an air/water interface.

**Compute.** Disagreement versus true held-out energy/force error, acquisition enrichment and learning curves with frozen model versions.

**Existing entry.** [examples/ml_lifecycle/committee.inp](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/ml_lifecycle/committee.inp); [docs/tier2-ml-lifecycle.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/tier2-ml-lifecycle.md)

**Gaps and boundaries.** Analytic committee examples test control flow only; external labelled data and training are required.

[Implementation and scope](references.md#committee).

#### task-delta

**Additive delta learning with RPC/MTS — P2**

Papers: [delta2015](references.md#delta2015).

**Original system.** Delta-learning origin: C7H10O2 isomers and organic-molecule thermochemical data, not an RPMD trajectory benchmark.

**Compute.** Holdout corrected energies and force consistency; for the PyRPMD extension compare full versus contracted corrections in the same observable.

**Existing entry.** [examples/ml_lifecycle/delta_rpc.inp](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/ml_lifecycle/delta_rpc.inp); [examples/ml_lifecycle/delta_rpc_mts.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/ml_lifecycle/delta_rpc_mts.py)

**Gaps and boundaries.** Original molecular-property benchmark and dynamics extension are distinct; no trained correction is supplied.

[Implementation and scope](references.md#delta).

#### task-ase

**ASE calculator adapter — P0**

Papers: [ase2017](references.md#ase2017).

**Original system.** External software/model interface; original software paper is attribution, not a unique PyRPMD physics benchmark.

**Compute.** Compare identical configurations directly against the upstream engine: energy, force, virial, units, batching and periodic images; then use a separately validated physical model.

**Existing entry.** [docs/potential-adapters.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/potential-adapters.md)

**Gaps and boundaries.** Optional engine and compatible model bytes must be supplied; this adapter does not recreate upstream training or every published application.

[Implementation and scope](references.md#ase).

#### task-openmm

**OpenMM force adapter — P0**

Papers: [openmm2017](references.md#openmm2017).

**Original system.** External software/model interface; original software paper is attribution, not a unique PyRPMD physics benchmark.

**Compute.** Compare identical configurations directly against the upstream engine: energy, force, virial, units, batching and periodic images; then use a separately validated physical model.

**Existing entry.** [docs/potential-adapters.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/potential-adapters.md)

**Gaps and boundaries.** Optional engine and compatible model bytes must be supplied; this adapter does not recreate upstream training or every published application.

[Implementation and scope](references.md#openmm).

#### task-lammps

**LAMMPS library adapter — P0**

Papers: [lammps1995](references.md#lammps1995).

**Original system.** External software/model interface; original software paper is attribution, not a unique PyRPMD physics benchmark.

**Compute.** Compare identical configurations directly against the upstream engine: energy, force, virial, units, batching and periodic images; then use a separately validated physical model.

**Existing entry.** [docs/potential-adapters.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/potential-adapters.md)

**Gaps and boundaries.** Optional engine and compatible model bytes must be supplied; this adapter does not recreate upstream training or every published application.

[Implementation and scope](references.md#lammps).

#### task-ipi-socket

**i-PI socket adapter — P0**

Papers: [ipi2014](references.md#ipi2014).

**Original system.** External software/model interface; original software paper is attribution, not a unique PyRPMD physics benchmark.

**Compute.** Compare identical configurations directly against the upstream engine: energy, force, virial, units, batching and periodic images; then use a separately validated physical model.

**Existing entry.** [docs/potential-adapters.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/potential-adapters.md)

**Gaps and boundaries.** Optional engine and compatible model bytes must be supplied; this adapter does not recreate upstream training or every published application.

[Implementation and scope](references.md#ipi-socket).

#### task-torch

**Torch/GPU potentials and tensor propagation — P0**

Papers: [torch2019](references.md#torch2019).

**Original system.** External software/model interface; original software paper is attribution, not a unique PyRPMD physics benchmark.

**Compute.** Compare identical configurations directly against the upstream engine: energy, force, virial, units, batching and periodic images; then use a separately validated physical model.

**Existing entry.** [docs/potential-adapters.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/potential-adapters.md)

**Gaps and boundaries.** Optional engine and compatible model bytes must be supplied; this adapter does not recreate upstream training or every published application.

[Implementation and scope](references.md#torch).

#### task-ml-adapters

**MACE / NequIP / DeePMD / fairchem adapters — P0**

Papers: [mace2022](references.md#mace2022), [nequip2022](references.md#nequip2022), [deepmd2018](references.md#deepmd2018), [uma2025](references.md#uma2025).

**Original system.** External software/model interface; original software paper is attribution, not a unique PyRPMD physics benchmark.

**Compute.** Compare identical configurations directly against the upstream engine: energy, force, virial, units, batching and periodic images; then use a separately validated physical model.

**Existing entry.** [docs/potential-adapters.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/potential-adapters.md)

**Gaps and boundaries.** Optional engine and compatible model bytes must be supplied; this adapter does not recreate upstream training or every published application.

[Implementation and scope](references.md#ml-adapters).

#### task-tblite

**GFN1-xTB / GFN2-xTB / IPEA1-xTB via tblite — P0**

Papers: [gfn12017](references.md#gfn12017), [gfn22019](references.md#gfn22019), [ipea2017](references.md#ipea2017).

**Original system.** External software/model interface; original software paper is attribution, not a unique PyRPMD physics benchmark.

**Compute.** Compare identical configurations directly against the upstream engine: energy, force, virial, units, batching and periodic images; then use a separately validated physical model.

**Existing entry.** [docs/potential-adapters.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/potential-adapters.md)

**Gaps and boundaries.** Optional engine and compatible model bytes must be supplied; this adapter does not recreate upstream training or every published application.

[Implementation and scope](references.md#tblite).

#### task-h5md-export

**H5MD, extxyz and ASE trajectory export — P0**

Papers: [h5md2014](references.md#h5md2014), [ase2017](references.md#ase2017).

**Original system.** Data-format paper, not a physical benchmark.

**Compute.** Round-trip coordinates/cells/isotope masses/time/units and compare reanalysed observables to source trajectories.

**Existing entry.** [docs/trajectory-export.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/trajectory-export.md)

**Gaps and boundaries.** Reader support and semantic round-trip must be checked for the selected format.

[Implementation and scope](references.md#h5md-export).

#### task-engineering

**Input, restart, parallel execution, reports and run organization — P0**

Papers: —.

**Original system.** No independent original method paper for these PyRPMD-specific mechanisms. Cite the exact PyRPMD revision plus the underlying method/PES papers.

**Compute.** Backend numerical equivalence, restart identity, units/normalization, uncertainty and archive replay; reuse the same physical benchmark across execution modes.

**Existing entry.** [docs/manual/execution.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/manual/execution.md); [docs/checkpoints.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/checkpoints.md); [docs/manual/analysis-reports.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/manual/analysis-reports.md); [docs/reaction-networks.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/reaction-networks.md)

**Gaps and boundaries.** A successful software test is not a new physical reproduction; do not attach an unrelated paper merely to fill a citation cell.

[Implementation and scope](references.md#engineering).

#### task-analytic-models

**Analytic free, harmonic, Morse, quartic and periodic-cosine models — P0**

Papers: [morse1929](references.md#morse1929).

**Original system.** Morse: diatomic vibrational levels. The other generic test potentials do not specify one original molecular system or a unique originating paper.

**Compute.** Energy/force finite differences, analytic harmonic equilibrium moments, Morse bound levels, exact quadrature FES and coordinate-periodicity checks. Match the energy zero and model parameters.

**Existing entry.** [examples/quantum_statistics/harmonic_validation.py](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/quantum_statistics/harmonic_validation.py); [examples/fes/harmonic2d.inp](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/examples/fes/harmonic2d.inp)

**Gaps and boundaries.** A separable cosine model is not evidence for diffusion in a real material. Arbitrary Morse parameters do not reproduce a particular molecule from the 1929 paper.

[Implementation and scope](references.md#analytic-models).

#### task-rate-studies

**Temperature/isotope scans, Arrhenius fits and convergence studies — P1**

Papers: [bimol09](references.md#bimol09), [iso13](references.md#iso13).

**Original system.** Use H + H2 and its isotopologues on the same identified PES for the rate campaign; use the original isotope-estimator water model for equilibrium fractionation.

**Compute.** k(T), k_H/k_D, fitted activation parameters with residuals, and paired refinement differences for beads, timestep, separation and maximum impact parameter. Predeclare equivalence margins.

**Existing entry.** [docs/rate-studies.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/rate-studies.md)

**Gaps and boundaries.** A good Arrhenius fit does not validate the rates or exclude tunnelling curvature; a statistically insignificant difference is not a convergence/equivalence result.

[Implementation and scope](references.md#rate-studies).

#### task-bkmp2-model

**BKMP2 H3 literature benchmark and external PES provenance — P1**

Papers: [bkmp21996](references.md#bkmp21996), [bimol09](references.md#bimol09), [rpmrate13](references.md#rpmrate13).

**Original system.** BKMP2 1996: H + H2 rotational excitation and H3 PES validation. RPMD-rate papers use H + H2 exchange; these are different target calculations on the same PES family.

**Compute.** First verify the surface, saddle, gradients and isotope masses; then reproduce PMF, k_QTST, kappa(t) and k(T). Reproduce the 1996 state-to-state cross sections only with its own initial states and scattering protocol.

**Existing entry.** [benchmarks/literature/campaign.json](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/benchmarks/literature/campaign.json); [docs/literature-validation.md](https://github.com/chen-yu-hao/PyRPMD/blob/cb60679cad8ea9ce926e5e04bd95ea148f7e5a3d/docs/literature-validation.md)

**Gaps and boundaries.** Pin source bytes/checksum and channel/symmetry conventions. Existing fixed-P agreement leaves the separate parent-time, umbrella-precision and dynamic-bead convergence checks to complete.

[Implementation and scope](references.md#bkmp2-model).
