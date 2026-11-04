# Original papers and implemented features

This inventory groups implemented **scientific methods, models and interfaces**, rather than inventing
a paper for every helper function. It audits commit
[`28cde09`](https://github.com/chen-yu-hao/PyRPMD/commit/28cde092b66707878d99bc45eadf6bdf5950b317) on **2026-11-03**.
The **125 entries and 199 bibliographic records** connect implementation paths to primary sources
and explicit scope boundaries. The [reproduction checklist](reproduction.md) adds the original systems,
observables, existing entry points, missing ingredients and priorities.
Entries for features merged after the previous audit (`cb60679`, 2026-11-02), and entries whose scope those
features changed, were written or revised against this commit; the remaining scope notes carry the earlier
review forward with updated code links.

## How to cite a calculation

Cite the **original algorithms, the actual PES/electronic-structure model, and the upstream software** used,
as well as the exact PyRPMD version/commit. An H + H2/BKMP2 rate calculation needs the Bennett–Chandler
foundation, RPMD rate theory and bimolecular normalization, the BKMP2 surface and the software actually used.
A software paper or a review does not replace the originating algorithm paper.

Cite PyRPMD itself as software, using the metadata in
[CITATION.cff](https://github.com/chen-yu-hao/PyRPMD/blob/main/CITATION.cff) (GitHub's **Cite this repository** button
produces APA and BibTeX from it). Add the version reported by `pyrpmd --version`, which each run also
stores in `manifest.json`, and, for an unreleased checkout, the commit hash from `git rev-parse HEAD`.
PyRPMD has no journal article yet, so do not substitute RPMDrate, Caracal or a method paper for the
software citation.

Sources are deliberately distinguished as original methods, later corrections, models/data, software,
formula references or background. **A citation is not a claim of full implementation or successful
reproduction.** Package-specific compositions cite their components; an adapter does not implement all
of the upstream method; generic controls and engineering utilities have no invented scientific origin.
Respect the source version, erratum and implementation boundary recorded below.

### Workflow checklist

Collect the citation keys for a run from the entries below. Common starting sets:

| Calculation | Catalog entries to cite |
| --- | --- |
| Bimolecular RPMD rate (`rate`) | [rpm-rate](#rpm-rate), [integrators](#integrators), [samp-umbrella](#samp-umbrella), [samp-ui](#samp-ui) or [samp-mbar](#samp-mbar), plus the PES |
| Direct collisions (`drpmd`) | [direct-rpmd](#direct-rpmd), [integrators](#integrators), plus [state-rpmd](#state-rpmd) or [qct](#qct) for selected initial states |
| Free-energy surfaces (`fes`) | The chosen method: [samp-umbrella](#samp-umbrella), [samp-standard-meta](#samp-standard-meta), [samp-wt-meta](#samp-wt-meta), [samp-opes](#samp-opes), [samp-ves](#samp-ves), [samp-abf](#samp-abf) or [samp-steered](#samp-steered) |
| Equilibrium PIMD (`pimd`) | [pimd](#pimd), the thermostat ([pile](#pile), [gle](#gle), [piglet](#piglet), [nhc](#nhc)) and, for NPT, the barostat ([bzp](#bzp), [mtk](#mtk), [scr](#scr)) |
| Replica exchange and tempering | [samp-reus](#samp-reus) or [samp-tempering](#samp-tempering) |
| Path sampling (`path_sampling`) | [samp-tis](#samp-tis), [samp-retis](#samp-retis), [samp-infretis](#samp-infretis), [samp-pptis](#samp-pptis) or [samp-ffs](#samp-ffs) |
| Nonadiabatic populations (`nonadiabatic`) | The selected method entry, such as [na-nrpmd-focused](#na-nrpmd-focused), [na-carp-mash](#na-carp-mash), [na-ba-rp-mash](#na-ba-rp-mash), [na-ms-mash](#na-ms-mash), [na-iesh](#na-iesh) or [na-rpsh-ca](#na-rpsh-ca), plus the model |
| Electrochemical PCET (`pcet`) | [pcet](#pcet), plus the electrode and environment model |
| External force engines and ML potentials | The adapter entry ([ase](#ase), [openmm](#openmm), [lammps](#lammps), [ipi-socket](#ipi-socket), [ml-adapters](#ml-adapters)) and the model paper and checkpoint |

Machine-readable records and BibTeX are available as repository files
[method-references.json](../method-references.json) and
[references.bib](../references.bib). Both pages and the BibTeX file are generated from the JSON by
`python tools/reference_catalog.py --write`.

## Equilibrium, dynamics, rates and spectroscopy

### pimd

**Primitive distinguishable-particle PIMD**

Sources: [cw81](#cw81), [pr84](#pr84), [pile10](#pile10)

Closed distinguishable-particle ring-polymer equilibrium sampling. Imaginary-time beads are a statistical representation; fictitious sampling time is not exact quantum dynamics.

Code: [src/pyrpmd/dynamics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/dynamics.py), [src/pyrpmd/engine.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/engine.py), [src/pyrpmd/path_integral.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/path_integral.py), [src/pyrpmd/workflows/native_pimd.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/native_pimd.py)

[Reproduction task](reproduction.md#task-pimd).

### rpmd

**RPMD real-time correlations**

Sources: [rpm04](#rpm04), [kubo57](#kubo57)

Physical-mass ring-polymer propagation approximates Kubo-transformed equilibrium correlation functions; the classical P=1 and harmonic linear-observable limits provide checks.

Code: [src/pyrpmd/dynamics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/dynamics.py), [src/pyrpmd/propagation.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/propagation.py), [src/pyrpmd/analysis/spectroscopy.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/spectroscopy.py)

[Reproduction task](reproduction.md#task-rpmd).

### rpm-rate

**Thermal RPMD bimolecular reaction rate**

Sources: [bennett1977](#bennett1977), [chandler1978](#chandler1978), [rate05](#rate05), [rate05b](#rate05b), [bimol09](#bimol09), [rpmrate13](#rpmrate13), [bkmp21996](#bkmp21996)

Bennett–Chandler style separation of a centroid free-energy/QTST factor and a dynamical recrossing transmission coefficient, with bimolecular entrance-channel normalization.

Code: [src/pyrpmd/workflows/rate.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/rate.py), [src/pyrpmd/workflows/rate_pmf.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/rate_pmf.py), [src/pyrpmd/workflows/rate_recrossing.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/rate_recrossing.py), [src/pyrpmd/analysis/rate_pmf.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/rate_pmf.py)

[Reproduction task](reproduction.md#task-rpm-rate).

### direct-rpmd

**Direct-trajectory thermal RPMD (dRPMD)**

Sources: [drpmd18](#drpmd18)

Thermally sampled incoming ring polymers followed by direct collision trajectories, cross-section/rate estimation and channel classification.

Code: [src/pyrpmd/workflows/drpmd.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/drpmd.py), [src/pyrpmd/workflows/drpmd_sampling.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/drpmd_sampling.py), [src/pyrpmd/workflows/drpmd_classification.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/drpmd_classification.py), [src/pyrpmd/analysis/scattering.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/scattering.py)

[Reproduction task](reproduction.md#task-direct-rpmd).

### state-rpmd

**Approximate harmonic initial-state-selected NE-RPMD**

Sources: [nerp16](#nerp16), [state20](#state20), [state21](#state21), [state22](#state22)

Cold harmonic ring preparation plus centroid vibrational excitation and optional j=0 angular-momentum projection. Auxiliary β controls state preparation and is distinct from collision energy or a measured thermal temperature.

Code: [src/pyrpmd/initial_states.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/initial_states.py), [src/pyrpmd/workflows/state_selected.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/state_selected.py)

[Reproduction task](reproduction.md#task-state-rpmd).

### qct

**Harmonic QCT and full-PES action-matched diatomic QCT**

Sources: [qct65](#qct65), [state20](#state20), [state22](#state22)

Production QCT uses P=1 harmonic state preparation. The H₃ benchmark additionally solves a one-dimensional full-PES classical action condition J(E)=v+1/2 and samples a time-uniform orbit phase.

Code: [src/pyrpmd/initial_states.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/initial_states.py), [src/pyrpmd/workflows/state_selected.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/state_selected.py), [benchmarks/h3/diatomic_action.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/benchmarks/h3/diatomic_action.py), [benchmarks/h3/state_preparation_study.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/benchmarks/h3/state_preparation_study.py)

[Reproduction task](reproduction.md#task-qct).

### state-switch

**Experimental smooth preparation-to-PES switching diagnostic**

Sources: [state20](#state20), [state22](#state22)

Repository-specific diagnostic that changes the preparation Hamiltonian smoothly before scattering; the cited papers supply the surrounding harmonic NE-RPMD context only.

Code: [benchmarks/h3/diatomic_switch.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/benchmarks/h3/diatomic_switch.py), [benchmarks/h3/state_preparation_study.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/benchmarks/h3/state_preparation_study.py)

[Reproduction task](reproduction.md#task-state-switch).

### integrators

**Velocity Verlet, normal-mode BAB and Cayley BCB**

Sources: [vv82](#vv82), [pile10](#pile10), [cayley19](#cayley19)

Deterministic physical-mass ring-polymer integrators: direct Verlet, analytic free-ring normal-mode BAB, and strongly stable Cayley BCB.

Code: [src/pyrpmd/dynamics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/dynamics.py), [src/pyrpmd/propagation.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/propagation.py), [src/pyrpmd/torch_backend.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/torch_backend.py)

[Reproduction task](reproduction.md#task-integrators).

### stochastic-integrators

**BAOAB and Cayley BCOCB**

Sources: [baoab13](#baoab13), [bcocb20](#bcocb20), [cayley19](#cayley19)

Symmetric Langevin splittings. In BCOCB each C half-step is the square root of the full-step Cayley map, not Cayley evaluated at dt/2.

Code: [src/pyrpmd/dynamics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/dynamics.py), [src/pyrpmd/torch_backend.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/torch_backend.py), [src/pyrpmd/native_ensemble.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/native_ensemble.py)

[Reproduction task](reproduction.md#task-stochastic-integrators).

### classical-baths

**Andersen, Cartesian Langevin and stochastic velocity rescaling**

Sources: [and80](#and80), [langevin07](#langevin07), [svr07](#svr07)

Andersen global Maxwell redraw with an explicit discrete collision clock; exact OU Cartesian Langevin bath; global canonical stochastic velocity rescaling including sign and zero-energy branches.

Code: [src/pyrpmd/thermostats.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/thermostats.py), [src/pyrpmd/native_ensemble.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/native_ensemble.py), [src/pyrpmd/dynamics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/dynamics.py)

[Reproduction task](reproduction.md#task-classical-baths).

### pile

**PILE-L and PILE-G normal-mode thermostats**

Sources: [pile10](#pile10), [svr07](#svr07)

PILE-L uses local mode friction proportional to ring frequency; PILE-G combines internal-mode PILE with global centroid stochastic velocity rescaling. Internal-mode friction follows the ring frequencies; centroid coupling is set separately, rather than inferred from its zero normal-mode frequency.

Code: [src/pyrpmd/dynamics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/dynamics.py), [src/pyrpmd/native_ensemble.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/native_ensemble.py)

[Reproduction task](reproduction.md#task-pile).

### nhc

**Nosé–Hoover chain thermostat**

Sources: [nhc92](#nhc92), [yoshida1990](#yoshida1990)

Global Nosé–Hoover chain with a three-stage Suzuki–Yoshida composition; Python and native routes share mass/DOF conventions.

Code: [src/pyrpmd/thermostats.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/thermostats.py), [src/pyrpmd/native_ensemble.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/native_ensemble.py)

[Reproduction task](reproduction.md#task-nhc).

### gle

**Cartesian and normal-mode GLE baths**

Sources: [gle09](#gle09), [pile10](#pile10)

Extended-variable exact OU update with supplied stable drift and covariance matrices; Cartesian GLE, normal-mode GLE and normal-mode GLE plus global centroid SVR are available.

Code: [src/pyrpmd/native_ensemble.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/native_ensemble.py)

[Reproduction task](reproduction.md#task-gle).

### piglet

**PIGLET with externally calibrated matrices**

Sources: [piglet12](#piglet12)

NVT normal-mode GLE using externally supplied P-dependent calibrated matrices, matched β and parameter provenance; no built-in calibration table is invented.

Code: [src/pyrpmd/native_ensemble.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/native_ensemble.py)

[Reproduction task](reproduction.md#task-piglet).

### ffl

**Fast-forward Langevin rescale thermostat**

Sources: [ffl18](#ffl18)

Native rescale variant updates momentum magnitude while retaining its direction; it is one variant of the fast-forward Langevin family.

Code: [src/pyrpmd/native_ensemble.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/native_ensemble.py)

[Reproduction task](reproduction.md#task-ffl).

### constraints

**Per-bead SHAKE/RATTLE bond constraints**

Sources: [shake77](#shake77), [rattle83](#rattle83)

Position and tangent-momentum constraints on every bead, using the Verlet old-gradient SHAKE displacement and matching momentum impulse followed by RATTLE projection.

Code: [src/pyrpmd/bond_constraints.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/bond_constraints.py), [src/pyrpmd/dynamics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/dynamics.py)

[Reproduction task](reproduction.md#task-constraints).

### rpc-mts

**Ring-polymer contraction and reversible multiple time stepping**

Sources: [rpc08](#rpc08), [respa92](#respa92), [mts16](#mts16), [refrpc16](#refrpc16)

Low-mode bead contraction for a slow correction and two-level r-RESPA fast/slow force propagation. Reference correction means Vslow=Vtarget−Vreference, not a second full target potential.

Code: [src/pyrpmd/potentials/contraction.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/potentials/contraction.py), [src/pyrpmd/multitime.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/multitime.py), [src/pyrpmd/propagation.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/propagation.py)

[Reproduction task](reproduction.md#task-rpc-mts).

### pacmd

**Partially adiabatic centroid molecular dynamics (PA-CMD)**

Sources: [cmd93](#cmd93), [cmd94](#cmd94), [pacmd06](#pacmd06)

Dedicated PACMDDynamics uses fictitious internal-mode masses and an exact-free BAOAB propagator, retaining physical centroid mass. It is distinct from adding friction to an ordinary physical-mass RPMD trajectory.

Code: [src/pyrpmd/cmd.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/cmd.py), [src/pyrpmd/workflows/spectroscopy.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/spectroscopy.py)

[Reproduction task](reproduction.md#task-pacmd).

### fcmd

**Fast centroid molecular dynamics by force matching**

Sources: [cmd94](#cmd94), [fcmd05](#fcmd05)

Exports PIMD centroid/mean physical bead forces and fits a conservative local quadratic centroid force model; temperature is part of model provenance. IBI is a separate update building block.

Code: [src/pyrpmd/cmd.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/cmd.py)

[Reproduction task](reproduction.md#task-fcmd).

### qcmd

**Experimental f-QCMD for bent triatomic molecules**

Sources: [qcmd2019](#qcmd2019), [fqcmd2021](#fqcmd2021), [cmd94](#cmd94)

Bead-averaged bond/angle quasi-centroids for nonperiodic, fixed-topology bent triatomics; iterative Boltzmann inversion of separable correction potentials; classical spectroscopy dynamics with method="f-qcmd" and beads=1. A constrained-PIMD API samples fixed mean internal coordinates for equilibrium research. Real-time adiabatic QCMD and h-CMD are not implemented and fail explicitly.

Code: [src/pyrpmd/qcmd.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/qcmd.py), [src/pyrpmd/_qcmd_geometry.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/_qcmd_geometry.py), [src/pyrpmd/_qcmd_sampling.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/_qcmd_sampling.py), [src/pyrpmd/workflows/spectroscopy.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/spectroscopy.py)

[Reproduction task](reproduction.md#task-qcmd).

### trpmd

**TRPMD dynamics and relative IR spectra**

Sources: [trpmd14](#trpmd14), [rpm04](#rpm04), [kubo57](#kubo57), [water09](#water09)

Internal-mode friction with an unthermostatted centroid; relative IR is derived from dipole Kubo correlations with the βω² convention and no extra harmonic quantum correction.

Code: [src/pyrpmd/dynamics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/dynamics.py), [src/pyrpmd/native_ensemble.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/native_ensemble.py), [src/pyrpmd/analysis/spectroscopy.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/spectroscopy.py), [src/pyrpmd/workflows/spectroscopy.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/spectroscopy.py)

[Reproduction task](reproduction.md#task-trpmd).

### trpmd-rates

**TRPMD rate-comparison mode**

Sources: [trpmd14](#trpmd14), [trprate15](#trprate15)

Optional internally thermostatted rate trajectories supply a friction-sensitivity comparison; rate dynamics must state the selected friction protocol.

Code: [src/pyrpmd/dynamics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/dynamics.py), [src/pyrpmd/workflows/rate_recrossing.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/rate_recrossing.py)

[Reproduction task](reproduction.md#task-trpmd-rates).

### raman

**Relative isotropic/anisotropic Raman spectra**

Sources: [kubo57](#kubo57), [raman21](#raman21)

Polarizability-tensor correlations decomposed into isotropic and traceless anisotropic invariants; VV=45iso+4aniso, VH=3aniso, total=45iso+7aniso. Optional Kubo-to-Stokes and laser-frequency factors are explicit.

Code: [src/pyrpmd/analysis/spectroscopy.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/spectroscopy.py), [src/pyrpmd/properties.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/properties.py)

[Reproduction task](reproduction.md#task-raman).

### sfg

**Relative SFG susceptibility from time correlations**

Sources: [sfg02](#sfg02), [kubo57](#kubo57)

Causal dipole–polarizability response R(t)=−βdCμα(t)/dt and complex susceptibility using the documented positive-frequency transform; signed Imχ and coherent replica averaging are retained.

Code: [src/pyrpmd/analysis/spectroscopy.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/spectroscopy.py), [src/pyrpmd/properties.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/properties.py)

[Reproduction task](reproduction.md#task-sfg).

### ti-pcn

**Takahashi–Imada trace action and pCN sampling**

Sources: [ti84](#ti84), [pcn13](#pcn13), [cw81](#cw81)

Primitive or TI closed-path trace action sampled by Gaussian-preserving pCN internal-mode proposals plus centroid Metropolis moves. MC steps are not dynamical time.

Code: [src/pyrpmd/path_integral.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/path_integral.py)

[Reproduction task](reproduction.md#task-ti-pcn).

### thermo-estimators

**Quantum kinetic energy, total energy and heat capacity**

Sources: [yam05](#yam05), [ti84](#ti84), [cw81](#cw81)

Thermodynamic β derivatives of the chosen trace action, kinetic estimators and optional scaled-coordinate finite-difference virial estimation; correlated-sample uncertainty is handled by blocks.

Code: [src/pyrpmd/quantum_statistics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/quantum_statistics.py)

[Reproduction task](reproduction.md#task-thermo-estimators).

### isotopes

**Isotope free-energy and fractionation estimators**

Sources: [iso13](#iso13), [iso14](#iso14)

Thermodynamic and scaled-coordinate FEP mass substitutions, two-environment fractionation ratios and kinetic-energy thermodynamic integration over mass, with overlap diagnostics.

Code: [src/pyrpmd/quantum_statistics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/quantum_statistics.py)

[Reproduction task](reproduction.md#task-isotopes).

### momentum

**Primitive open-path one-component momentum distribution**

Sources: [cw81](#cw81), [ceperley95](#ceperley95), [mom10](#mom10)

Direct primitive open-chain endpoint-separation sampling and Fourier estimation of one Cartesian component of n(p), with KDE/block-jackknife diagnostics.

Code: [src/pyrpmd/path_integral.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/path_integral.py), [src/pyrpmd/quantum_statistics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/quantum_statistics.py)

[Reproduction task](reproduction.md#task-momentum).

### suzuki-chin

**Finite-difference Suzuki–Chin path integrals**

Sources: [suz95](#suz95), [chin97](#chin97), [sc16](#sc16)

Even-bead alternating Suzuki–Chin weights with force-gradient terms evaluated through symmetric finite differences along F/m. Python supports α in [0,1]; CLI currently restricts α=0.

Code: [src/pyrpmd/pimd_methods.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/pimd_methods.py), [src/pyrpmd/workflows/path_integral.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/path_integral.py), [src/pyrpmd/suzuki_chin.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/suzuki_chin.py)

[Reproduction task](reproduction.md#task-suzuki-chin).

### bosons

**Quadratic-scaling bosonic exchange PIMD**

Sources: [bose19](#bose19), [bose23](#bose23)

Feldman–Hirshberg O(N²+PN) recurrence for equal-mass exchange groups, using representative cycles and symmetric observables.

Code: [src/pyrpmd/pimd_methods.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/pimd_methods.py), [src/pyrpmd/workflows/path_integral.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/path_integral.py)

[Reproduction task](reproduction.md#task-bosons).

### pigs

**Primitive positive-trial path-integral ground state (PIGS)**

Sources: [pigs00](#pigs00)

Odd-bead open chain with endpoint half-potential weights and a positive Gaussian or supplied positive trial wavefunction; midpoint observables estimate the projected ground state.

Code: [src/pyrpmd/pimd_methods.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/pimd_methods.py), [src/pyrpmd/workflows/path_integral.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/path_integral.py)

[Reproduction task](reproduction.md#task-pigs).

### rpi

**Ring-polymer instanton thermal rate**

Sources: [rpi09](#rpi09), [rpi16](#rpi16)

Stationary imaginary-time ring optimization, semiclassical fluctuation determinant/prefactor, harmonic-reactant normalization and negative/zero-mode checks; multidimensional gas-rate handling includes declared rigid-body external modes.

Code: [src/pyrpmd/instanton.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/instanton.py), [src/pyrpmd/low_temperature.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/low_temperature.py)

[Reproduction task](reproduction.md#task-rpi).

### splitting

**Symmetric bound-well instanton tunnelling splitting**

Sources: [split11](#split11)

Single-kink semiclassical splitting for an isolated symmetric bound or explicitly constrained doublet.

Code: [src/pyrpmd/tunnelling.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/tunnelling.py), [src/pyrpmd/low_temperature.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/low_temperature.py)

[Reproduction task](reproduction.md#task-splitting).

### low-t-checks

**Harmonic TST, Wigner and Eckart cross-checks**

Sources: [eyr35](#eyr35), [wig32](#wig32), [eck30](#eck30)

Classical and quantum harmonic-TST baselines, leading Wigner tunnelling correction and thermally integrated asymmetric Eckart transmission, with explicitly named ZPE/barrier conventions.

Code: [src/pyrpmd/low_temperature.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/low_temperature.py)

[Reproduction task](reproduction.md#task-low-t-checks).

## Free energy, enhanced sampling and reaction paths

### samp-umbrella

**Umbrella sampling / 1D–3D FES**

Sources: [samp-us1974](#samp-us1974), [samp-us1977](#samp-us1977)

Static harmonic umbrella biases with full CV Jacobians and explicit ring scaling; US origins are classical Monte Carlo and supply the importance-sampling principle, not an identical PyRPMD dynamics engine.

Code: [src/pyrpmd/bias.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/bias.py), [src/pyrpmd/workflows/fes.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/fes.py)

[Reproduction task](reproduction.md#task-samp-umbrella).

### samp-mbar

**MBAR free energies, histogram FES, overlap diagnostics**

Sources: [samp-mbar2008](#samp-mbar2008)

Direct SciPy implementation of MBAR; bins are used only after MBAR to form CV-density FES. Do not rename this as a separately implemented WHAM solver. Bootstrap is the package uncertainty route; original asymptotic covariance is not thereby claimed.

Code: [src/pyrpmd/analysis/mbar.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/mbar.py), [src/pyrpmd/analysis/fes.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/fes.py), [src/pyrpmd/analysis/umbrella.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/umbrella.py)

[Reproduction task](reproduction.md#task-samp-mbar).

### samp-ui

**Gaussian umbrella integration**

Sources: [samp-ui2005](#samp-ui2005), [samp-uierr2006](#samp-uierr2006)

One-dimensional Gaussian window approximation ported mathematically from RPMDrate; it is not the later multidimensional or higher-order UI algorithms.

Code: [src/pyrpmd/analysis/umbrella.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/umbrella.py), [src/pyrpmd/analysis/rate_pmf.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/rate_pmf.py)

[Reproduction task](reproduction.md#task-samp-ui).

### samp-adaptive

**Adaptive umbrella placement and coverage planning**

Sources: [samp-mbar2008](#samp-mbar2008), [samp-us1977](#samp-us1977), [samp-ogre2023](#samp-ogre2023)

PyRPMD-specific normalized MBAR-overlap graph, midpoint/coverage insertion, fixed stiffness, independent pilot then fresh production. OGRe is a benchmark/model comparison, not the implemented planner; OGRe also uses confinement/consistency and stiffness refinement.

Code: [src/pyrpmd/workflows/adaptive_umbrella.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/adaptive_umbrella.py), [src/pyrpmd/analysis/fes_statistics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/fes_statistics.py)

[Reproduction task](reproduction.md#task-samp-adaptive).

### samp-standard-meta

**Standard metadynamics**

Sources: [samp-meta2002](#samp-meta2002)

History-dependent Gaussian bias and finite-time -V free-energy estimate; original 2002 coarse-grained algorithm is the method origin, while deposition occurs directly in the package MD workflow.

Code: [src/pyrpmd/bias.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/bias.py), [src/pyrpmd/workflows/metadynamics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/metadynamics.py), [src/pyrpmd/analysis/metadynamics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/metadynamics.py)

[Reproduction task](reproduction.md#task-samp-standard-meta).

### samp-wt-meta

**Well-tempered metadynamics**

Sources: [samp-wt2008](#samp-wt2008)

Exponentially tempered Gaussian hills; F approaches -gamma/(gamma-1) V. No time-dependent c(t) trajectory reweighting is implemented in bias_fes.

Code: [src/pyrpmd/bias.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/bias.py), [src/pyrpmd/workflows/metadynamics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/metadynamics.py), [src/pyrpmd/analysis/metadynamics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/metadynamics.py)

[Reproduction task](reproduction.md#task-samp-wt-meta).

### samp-walkers

**Shared-bias multiple-walker metadynamics**

Sources: [samp-walkers2006](#samp-walkers2006), [samp-wt2008](#samp-wt2008)

Shared Gaussian (MetaD) or kernel (OPES) histories with walkers_mode=serial, mpi-sync (one walker per MPI rank, rank-ordered synchronous batches, one atomic ensemble checkpoint) or async-files (append-only per-walker logs, eventually merged). Walkers sharing one bias are statistically dependent; synchronization and restart specifics are project engineering.

Code: [src/pyrpmd/shared_metadynamics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/shared_metadynamics.py), [src/pyrpmd/workflows/shared_metadynamics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/shared_metadynamics.py), [src/pyrpmd/walker_exchange.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/walker_exchange.py), [src/pyrpmd/workflows/_distributed_meta_mpi.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/_distributed_meta_mpi.py), [src/pyrpmd/workflows/_distributed_meta_async.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/_distributed_meta_async.py), [src/pyrpmd/workflows/_distributed_opes_mpi.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/_distributed_opes_mpi.py), [src/pyrpmd/workflows/_distributed_opes_async.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/_distributed_opes_async.py)

[Reproduction task](reproduction.md#task-samp-walkers).

### samp-grid-meta

**Grid-accelerated metadynamics and FES projection**

Sources: [samp-meta2002](#samp-meta2002), [samp-wt2008](#samp-wt2008)

Grid interpolation, derivatives, and Boltzmann marginalization implement numerical representations of the existing bias/FES; no separate invented original paper. Domain-masked marginalization is not a literature hill-filtering procedure.

Code: [src/pyrpmd/bias.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/bias.py), [src/pyrpmd/analysis/projection.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/projection.py)

[Reproduction task](reproduction.md#task-samp-grid-meta).

### samp-opes

**OPES_METAD / frozen-bias reweighting**

Sources: [samp-opes2020](#samp-opes2020)

Uncompressed normalized wrapped-Gaussian KDE and kernel-center Z; fixed bandwidth default. Optional frozen-bias histogram reweighting uses exp(beta V), not a time-dependent c(t) correction.

Code: [src/pyrpmd/opes.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/opes.py), [src/pyrpmd/workflows/opes.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/opes.py)

[Reproduction task](reproduction.md#task-samp-opes).

### samp-opes-explore

**OPES-explore and optional adaptive widths**

Sources: [samp-explore2022](#samp-explore2022), [samp-opes2020](#samp-opes2020)

Explore uses unweighted sampled-target KDE and prefactor (gamma-1)/beta. Optional Welford/Scott-ESS widths are an explicit PyRPMD rule, not PLUMED diffusion-window bandwidth adaptation.

Code: [src/pyrpmd/opes.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/opes.py), [src/pyrpmd/workflows/opes.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/opes.py)

[Reproduction task](reproduction.md#task-samp-opes-explore).

### samp-ves

**Variationally enhanced sampling (VES) on centroid CVs**

Sources: [samp-ves2014](#samp-ves2014), [samp-wtves2015](#samp-wtves2015)

fes.method="ves" optimizes a tensor-product Legendre linear bias on one or two nonperiodic centroid CVs by averaged stochastic gradient descent with the full covariance Hessian. Uniform or well-tempered targets use Gauss–Legendre quadrature; frozen-bias production and reweighting write the common FESResult. Not a PLUMED VES bridge; other basis sets, periodic CVs, barostats and replica exchange are not combined with it.

Code: [src/pyrpmd/ves.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/ves.py), [src/pyrpmd/workflows/ves.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/ves.py)

[Reproduction task](reproduction.md#task-samp-ves).

### samp-abf

**Centroid ABF and extended-system ABF with CZAR**

Sources: [samp-abf2001](#samp-abf2001), [samp-abf2008](#samp-abf2008), [samp-abfproj2015](#samp-abfproj2015), [samp-eabf2017](#samp-eabf2017)

fes.method="abf"/"eabf" on one or two nonperiodic centroid CVs with full-row-rank Jacobians. The local mean force uses the full-ring Jacobian and a mass-weighted dual field with a linear ramp; 1D edge integration or 2D least-squares Helmholtz projection gives the FES. eABF biases harmonically coupled fictitious variables and reconstructs the FES with CZAR. Requires all atoms mobile and a canonical thermostat; fixed atoms, periodic cells/CVs, pressure coupling, replica exchange and the rate-flux measure are rejected.

Code: [src/pyrpmd/abf.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/abf.py), [src/pyrpmd/workflows/abf.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/abf.py)

[Reproduction task](reproduction.md#task-samp-abf).

### samp-steered

**Steered centroid MD and nonequilibrium work estimators**

Sources: [samp-jarzynski1997](#samp-jarzynski1997), [samp-crooks1999](#samp-crooks1999), [samp-bar1976](#samp-bar1976), [samp-shadow2013](#samp-shadow2013)

fes.method="steered" moves a harmonic centroid restraint forward and/or backward and estimates the restrained endpoint free-energy difference with Jarzynski or the Crooks/Bennett acceptance-ratio equation. Work uses H_P/P; total work is protocol work plus finite-step shadow work, and bath heat is excluded. No FESResult grid is produced; Hummer–Szabo unbiased-PMF reconstruction is not implemented.

Code: [src/pyrpmd/steered.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/steered.py), [src/pyrpmd/workflows/steered.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/steered.py)

[Reproduction task](reproduction.md#task-samp-steered).

### samp-reus

**Same-temperature replica-exchange umbrella sampling**

Sources: [samp-reus2000](#samp-reus2000)

Fixed-window, same-temperature 1D/2D REUS swaps full rings; sampling measure includes P*umbrella energy at beta/P. No automatic temperature exchange in this workflow.

Code: [src/pyrpmd/workflows/replica_exchange.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/replica_exchange.py)

[Reproduction task](reproduction.md#task-samp-reus).

### samp-remd

**Temperature / Hamiltonian replica-exchange API**

Sources: [samp-remd1999](#samp-remd1999), [samp-reus2000](#samp-reus2000)

Serial Python engine; ring exchange evaluates the target-temperature spring action and rescales momenta. The resumable [tempering] input workflow with checkpoints and MPI transport is the separate samp-tempering entry.

Code: [src/pyrpmd/replica_exchange.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/replica_exchange.py)

[Reproduction task](reproduction.md#task-samp-remd).

### samp-tempering

**Resumable parallel tempering and Hamiltonian replica-exchange workflow**

Sources: [samp-pt1996](#samp-pt1996), [samp-remd1999](#samp-remd1999), [samp-reus2000](#samp-reus2000), [samp-mbar2008](#samp-mbar2008)

The [tempering] table for task="pimd" and selected task="fes" inputs: explicit or geometric temperature ladders and Hamiltonian ladders. Exchanges evaluate all-state reduced potentials, including each target-temperature spring action, and rescale momenta by sqrt(beta_i/beta_j). Per-state archives, acceptance, round trips, state-space diffusion and MBAR overlap; PIMD estimators are recomputed for every target state. Serial, process and MPI transports share one ordering; independent of same-temperature [replica_exchange] REUS.

Code: [src/pyrpmd/workflows/tempering.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/tempering.py), [src/pyrpmd/replica_exchange.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/replica_exchange.py)

[Reproduction task](reproduction.md#task-samp-tempering).

### samp-biased-clock

**Audited biased-time acceleration and censored rate inference**

Sources: [samp-imeta2013](#samp-imeta2013), [samp-flooding2022](#samp-flooding2022)

Postprocessing API integrates interval-resolved exp(beta*(V-V_TS)); censored exponential exposure and optional refitted KS bootstrap. Not a native infrequent-MetaD or OPES-flooding production workflow.

Code: [src/pyrpmd/biased_kinetics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/biased_kinetics.py)

[Reproduction task](reproduction.md#task-samp-biased-clock).

### samp-tis

**Two-way canonical transition-interface sampling**

Sources: [samp-tis2003](#samp-tis2003), [samp-tis2005](#samp-tis2005), [samp-tps1998](#samp-tps1998), [samp-shoot2025](#samp-shoot2025)

Python API; deterministic real-mass ring or P=1 dynamics, variable-length paths with finite cap, fresh uniform shooting point and full canonical momentum redraw. Separate matching effective flux required.

Code: [src/pyrpmd/transition_interface.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/transition_interface.py)

[Reproduction task](reproduction.md#task-samp-tis).

### samp-retis

**Replica-exchange TIS with optional minus ensemble**

Sources: [samp-retis2007](#samp-retis2007), [samp-tis2003](#samp-tis2003)

Exchanges interface ensembles sharing identical dynamics; repeated/rejected paths remain in observations. minus=true adds [0-] and [0-]<->[0+] exchanges and estimates the effective flux internally from residence times; otherwise a matching external flux is required. Persistent checkpoints and task="path_sampling" are available; asynchronous infinite swaps are the separate samp-infretis entry.

Code: [src/pyrpmd/retis.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/retis.py), [src/pyrpmd/workflows/path_sampling.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/path_sampling.py)

[Reproduction task](reproduction.md#task-samp-retis).

### samp-pptis

**Partial-path TIS (PPTIS) and REPPTIS**

Sources: [samp-pptis2004](#samp-pptis2004), [samp-repptis2023](#samp-repptis2023), [samp-tis2003](#samp-tis2003)

Local three-interface path ensembles with the memory-reduction recurrence for crossing probabilities; REPPTIS adds replica exchange between neighbouring path ensembles. Python API and task="path_sampling" with persistent checkpoints. The memory-loss assumption must be tested against interface spacing and full TIS.

Code: [src/pyrpmd/pptis.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/pptis.py), [src/pyrpmd/workflows/path_sampling.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/path_sampling.py)

[Reproduction task](reproduction.md#task-samp-pptis).

### samp-ffs

**Direct forward-flux sampling (FFS)**

Sources: [samp-ffs2005](#samp-ffs2005), [samp-ffs2009](#samp-ffs2009)

Interface-by-interface splitting for an explicitly supplied stationary, time-homogeneous Markov step(state, rng) callback: flux from basin-A history, trial-based conditional probabilities, persistent checkpoints and task="path_sampling". Needs no reverse dynamics; the sampler does not choose a thermostat or physical process.

Code: [src/pyrpmd/ffs.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/ffs.py), [src/pyrpmd/workflows/path_sampling.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/path_sampling.py)

[Reproduction task](reproduction.md#task-samp-ffs).

### samp-infretis

**Asynchronous infinite-swap RETIS (∞RETIS)**

Sources: [samp-infretis2022](#samp-infretis2022), [samp-infretis2024](#samp-infretis2024), [samp-retis2007](#samp-retis2007)

exchange="infinite" assigns free paths to free positive ensembles with exact permanent-ratio marginals, computed by a quadratic-cost recurrence for nested interfaces. An asynchronous process executor with ensemble locks and ticket-ordered commits gives exact restart; the minus ensemble is a separate block. Full-Maxwell two-way shooting is kept; the 2024 subtrajectory moves, MPI and bead/GPU worker pools are not implemented.

Code: [src/pyrpmd/infinite_retis.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/infinite_retis.py), [src/pyrpmd/parallel/path_executor.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/parallel/path_executor.py), [src/pyrpmd/workflows/path_sampling.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/path_sampling.py)

[Reproduction task](reproduction.md#task-samp-infretis).

### samp-committor

**Fixed-full-ring RPMD configurational committor**

Sources: [samp-comm2000](#samp-comm2000), [samp-cp1934](#samp-cp1934)

Independent canonical momentum shots conditioned on every bead position, deterministic unbiased dynamics, A/B first hits plus unresolved shots. Classical committor at P=1; approximate RPMD committor at P>1. Censoring envelope is a project composition of exact binomial limits.

Code: [src/pyrpmd/committor.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/committor.py)

[Reproduction task](reproduction.md#task-samp-committor).

### samp-pathcv

**Aligned RMSD and reference-path collective variables**

Sources: [samp-kabsch1976](#samp-kabsch1976), [samp-path2007](#samp-path2007)

Weighted proper Kabsch alignment and smooth s/z reference-path variables with explicit scale. Dihedral, coordination and periodic affine CVs are mathematical building blocks; TorchScript supplies derivatives but no learned-CV training method.

Code: [src/pyrpmd/advanced_cv.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/advanced_cv.py)

[Reproduction task](reproduction.md#task-samp-pathcv).

### samp-mlcolvar

**Learned centroid CVs from mlcolvar (Deep-LDA / Deep-TICA) via TorchScript**

Sources: [samp-mlcolvar2023](#samp-mlcolvar2023), [samp-deeplda2020](#samp-deeplda2020), [samp-deeptica2021](#samp-deeptica2021)

Training happens in the external mlcolvar library; PyRPMD evaluates the exported CPU float64 TorchScript graph (preprocessing, network and projection) with TorchScriptCV, including Cartesian derivatives and ring-bead scaling. Optional dependencies; deployment checks do not establish CV quality.

Code: [src/pyrpmd/advanced_cv.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/advanced_cv.py)

[Reproduction task](reproduction.md#task-samp-mlcolvar).

### samp-string

**Euclidean mean-force / restrained adaptive strings**

Sources: [samp-string2002](#samp-string2002), [samp-stringcv2006](#samp-stringcv2006)

Fixed-endpoint Euclidean gradient relaxation and insertion; a supplied consistent free-energy gradient is required. Signed restrained extension uses scaled Euclidean geometry; general nonlinear-CV metric/conditional terms of a full MFEP are not all implemented.

Code: [src/pyrpmd/adaptive_string.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/adaptive_string.py), [src/pyrpmd/adaptive_nrpmd.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/adaptive_nrpmd.py)

[Reproduction task](reproduction.md#task-samp-string).

### samp-ft-string

**Finite-temperature string in centroid CV space**

Sources: [samp-fts2005](#samp-fts2005), [samp-stringcv2006](#samp-stringcv2006)

Restrained centroid sampling with the averaged CV mass metric, block diagnostics, fixed-endpoint path updates and exact iteration-boundary replay, reusing the existing arc-length reparameterization. Fixed image count and endpoints; nonperiodic centroid CVs with full-rank derivatives. At finite restraint stiffness it targets a smoothed free-energy path.

Code: [src/pyrpmd/finite_temperature_string.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/finite_temperature_string.py), [src/pyrpmd/adaptive_string.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/adaptive_string.py)

[Reproduction task](reproduction.md#task-samp-ft-string).

### samp-prfo

**Partitioned rational-function saddle optimization**

Sources: [samp-prfo1985](#samp-prfo1985)

Fresh finite-difference Hessians, separate uphill/downhill augmented eigenproblems and capped trial steps. Local molecular Cartesian saddle search; two-sided descent is a connectivity diagnostic, not a certified quantum-chemistry IRC.

Code: [src/pyrpmd/workflows/saddle.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/saddle.py), [src/pyrpmd/workflows/prepare.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/prepare.py)

[Reproduction task](reproduction.md#task-samp-prfo).

### samp-cineb

**CI-NEB with improved tangent and FIRE relaxation**

Sources: [samp-nebtangent2000](#samp-nebtangent2000), [samp-cineb2000](#samp-cineb2000), [samp-fire2006](#samp-fire2006)

Energy-weighted tangent, climbing image and FIRE optimizer. Current end-to-end rate preparation is molecular Cartesian/nonperiodic, so the original slab calculations are not directly covered by that workflow.

Code: [src/pyrpmd/workflows/neb.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/neb.py)

[Reproduction task](reproduction.md#task-samp-cineb).

### samp-connectivity

**Covalent-radius connectivity and channel inference**

Sources: [samp-radii2008](#samp-radii2008)

Cordero H–Cm tabulated radii plus package bond_scale=1.25 heuristic and user overrides. This is approximate graph classification, not an electronic bond-order model.

Code: [src/pyrpmd/covalent.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/covalent.py), [src/pyrpmd/workflows/prepare.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/prepare.py)

[Reproduction task](reproduction.md#task-samp-connectivity).

### samp-metric

**General nonlinear-coordinate constrained-parent mass metric**

Sources: [samp-bluemoon1989](#samp-bluemoon1989), [samp-blue1998](#samp-blue1998)

Constrained-to-conditional weighting for nonlinear centroid coordinates; reparametrization invariance tests. The package rate estimator is an RPMD combination, not identical reproduction of every blue-moon thermodynamic-integration variant.

Code: [src/pyrpmd/workflows/rate_constraints.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/rate_constraints.py), [src/pyrpmd/cv.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/cv.py), [src/pyrpmd/cv_periodic.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/cv_periodic.py)

[Reproduction task](reproduction.md#task-samp-metric).

### samp-rate-postprocess

**2D flux integration and signed multichannel postprocessing**

Sources: [samp-blue1998](#samp-blue1998), [bennett1977](#bennett1977), [chandler1978](#chandler1978), [rate05](#rate05), [bimol09](#bimol09)

Project extension of flux-over-population / Bennett–Chandler RPMD rate theory; no standalone original paper for this exact offline data pipeline. CV-density and rate-flux measures require different factors.

Code: [src/pyrpmd/analysis/rate_postprocess.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/rate_postprocess.py), [src/pyrpmd/analysis/rate_io.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/rate_io.py), [src/pyrpmd/analysis/bond_flux.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/bond_flux.py), [src/pyrpmd/analysis/bond_rate.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/bond_rate.py), [src/pyrpmd/analysis/entrance.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/entrance.py)

[Reproduction task](reproduction.md#task-samp-rate-postprocess).

### samp-statistics

**Correlation diagnostics and synchronized circular-block bootstrap**

Sources: [samp-cblock1992](#samp-cblock1992), [samp-geyer1992](#samp-geyer1992)

Circular block resampling is Politis–Romano; positive-pair correlation truncation is a related diagnostic. PyRPMD chooses CV/squared-CV/circular-harmonic probes, ceil(5g) blocks with N/2 cap, and joint windows/channels for correlated ensembles. These safeguards are project policies, not universal published constants.

Code: [src/pyrpmd/analysis/fes_statistics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/fes_statistics.py), [src/pyrpmd/analysis/recrossing.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/recrossing.py)

[Reproduction task](reproduction.md#task-samp-statistics).

### samp-plumed

**PLUMED centroid-bias bridge**

Sources: [samp-plumed2009](#samp-plumed2009), [samp-plumed2014](#samp-plumed2014)

External kernel/interface, not an internal implementation of every PLUMED method. Cite the PLUMED software and every selected CV/bias algorithm in the actual plumed.dat.

Code: [src/pyrpmd/plumed.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/plumed.py), [src/pyrpmd/workflows/biased_sampling.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/biased_sampling.py)

[Reproduction task](reproduction.md#task-samp-plumed).

## Nonadiabatic methods and matrix models

### na-nrpmd-focused

**Focused nonequilibrium MMST NRPMD**

Sources: [na-mm1979](#na-mm1979), [na-stock1997](#na-stock1997), [na-nrpmd2013](#na-nrpmd2013), [na-focused2019](#na-focused2019), [na-mint2018](#na-mint2018)

One mapping vector per nuclear bead, any finite number of electronic states and real/complex Hermitian fixed-diabatic matrices. Focused radii sqrt(3) for the initially occupied state and 1 otherwise, random angles, signed unclipped MMST populations. This is not a Gamma/SEO thermal initial state, PLDM, SQC, or the independent mapping-bead multiplicity generalized in the 2017 analysis.

Code: [src/pyrpmd/nonadiabatic.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/nonadiabatic.py), [src/pyrpmd/nonadiabatic_ensemble.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/nonadiabatic_ensemble.py), [src/pyrpmd/workflows/nonadiabatic.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/nonadiabatic.py)

[Reproduction task](reproduction.md#task-na-nrpmd-focused).

### na-mint

**MInt mapping subflow integrator**

Sources: [na-mint2018](#na-mint2018)

Exact electronic rotation at fixed nuclear geometry with the integrated nuclear momentum impulse, composed with the free-ring subflow. This cites an integrator component, not the complete MQC-IVR calculation in Church et al.

Code: [src/pyrpmd/nonadiabatic.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/nonadiabatic.py), [src/pyrpmd/spin_mapping.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/spin_mapping.py)

[Reproduction task](reproduction.md#task-na-mint).

### na-thermal-nrpmd

**Signed SEO thermal NRPMD and correlation/rate estimators**

Sources: [na-nrpmd2013](#na-nrpmd2013), [na-stock1997](#na-stock1997), [na-mint2018](#na-mint2018)

Real symmetric matrices only; sample the absolute SEO-projected initial weight and retain its initial sign in every correlation. Thermal weight is not exp(-beta/P * NRPMD energy). Library default scalar/traceless splitting differs from the 2013 paper's lowest-adiabatic-surface split; the dedicated published benchmark implements that split explicitly. Single-state scalar dynamics reduces to ordinary RPMD. Finite-time rate output is not automatically a converged formal rate.

Code: [src/pyrpmd/thermal_nrpmd.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/thermal_nrpmd.py), [src/pyrpmd/workflows/nonadiabatic_rate.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/nonadiabatic_rate.py)

[Reproduction task](reproduction.md#task-na-thermal-nrpmd).

### na-adaptive-thermal

**Adaptive restrained-path signed NRPMD composition**

Sources: [na-nrpmd2013](#na-nrpmd2013), [samp-stringcv2006](#samp-stringcv2006), [samp-mbar2008](#samp-mbar2008)

Package composition of signed SEO thermal sampling, restrained-path adaptation, free-energy reweighting, and independent production with the path frozen. There is no verified single original paper presenting the complete PyRPMD adaptive/electrode workflow.

Code: [src/pyrpmd/adaptive_nrpmd.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/adaptive_nrpmd.py), [src/pyrpmd/adaptive_string.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/adaptive_string.py), [src/pyrpmd/thermal_nrpmd.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/thermal_nrpmd.py)

[Reproduction task](reproduction.md#task-na-adaptive-thermal).

### na-sm-nrpmd

**Focused two-state SM-NRPMD**

Sources: [na-sm2021](#na-sm2021)

One two-state W-spin of radius sqrt(3)/2 per bead, focused u_z=±1/sqrt(3) with random azimuth, real or complex Hermitian matrices, and exact spin/impulse subflows. No general SU(N) implementation and no Gamma_W equilibrium trace/phase sampler.

Code: [src/pyrpmd/spin_mapping.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/spin_mapping.py), [src/pyrpmd/nonadiabatic_registry.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/nonadiabatic_registry.py)

[Reproduction task](reproduction.md#task-na-sm-nrpmd).

### na-mash

**Classical-nuclei MASH and experimental centroid extension**

Sources: [na-mash2023](#na-mash2023), [na-mashrates2024](#na-mashrates2024)

MASHDynamics implements real two-state spin-hemisphere deterministic hopping through the adiabatic interface. P=1 is the original classical-nuclei setting. Its P>1 centroid extension is explicitly experimental and distinct from RPMASHDynamics. Hops are located at step endpoints; the documented hop-position error requires time-step convergence.

Code: [src/pyrpmd/mash.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/mash.py), [src/pyrpmd/adiabatic.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/adiabatic.py)

[Reproduction task](reproduction.md#task-na-mash).

### na-carp-mash

**CA-RP-MASH (CLI method rp-mash)**

Sources: [na-carp2025](#na-carp2025), [na-mash2023](#na-mash2023), [na-mashrates2024](#na-mashrates2024)

One spin and one common active adiabatic surface for the whole ring; bead-local adiabatic nuclear forces and centroid electronic dynamics. Real symmetric two-state matrices with nonzero gaps. SciPy DOP853 event-root handling resolves spin equator crossings, with full-ring mass-metric hop energy rescaling. BA-RP-MASH and multistate MASH are separate entries (na-ba-rp-mash, na-ms-mash); complex-SOC MASH is absent. General correlations use operator-pair weights; population output is an active-surface indicator, not a generic spin-density population.

Code: [src/pyrpmd/rp_mash.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/rp_mash.py), [src/pyrpmd/nonadiabatic_registry.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/nonadiabatic_registry.py)

[Reproduction task](reproduction.md#task-na-carp-mash).

### na-ba-rp-mash

**BA-RP-MASH (CLI method rp-mash-ba)**

Sources: [na-carp2025](#na-carp2025), [na-mash2023](#na-mash2023)

Two-state bead approximation: one spin for the whole ring in a common adiabatic representation, beadwise matrices, gradients, gaps and derivative couplings, and bead-averaged electronic evolution. Separate state, workflow name and scientific identity from CA-RP-MASH (rp-mash); not multistate or complex-SOC MASH.

Code: [src/pyrpmd/rp_mash_ba.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/rp_mash_ba.py), [src/pyrpmd/nonadiabatic_registry.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/nonadiabatic_registry.py)

[Reproduction task](reproduction.md#task-na-ba-rp-mash).

### na-ms-mash

**Original multistate MASH (CLI method multistate-mash)**

Sources: [na-msmash2023](#na-msmash2023), [na-mash2023](#na-mash2023)

One normalized complex mapping vector, active adiabatic surface with the largest mapping population, and the Runeson–Manolopoulos signed observable estimator for real Hamiltonians with nondegenerate levels; one bead only. The original 2023 formulation, not unSMASH or later initialization variants, and distinct from the two-state estimator even for two states.

Code: [src/pyrpmd/mash.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/mash.py), [src/pyrpmd/workflows/nonadiabatic_multistate.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/nonadiabatic_multistate.py)

[Reproduction task](reproduction.md#task-na-ms-mash).

### na-rpsh-ca

**FSSH / centroid-approximation RPSH**

Sources: [na-tully1990](#na-tully1990), [na-rpsh2012](#na-rpsh2012), [na-rpsh2017](#na-rpsh2017), [na-rpshdm2023](#na-rpshdm2023)

RPSHDynamics is the real fixed-diabatic centroid route; AdvancedRPSHDynamics(method='ca') exposes extended CA options; direct AdiabaticRPSHDynamics accepts electronic-structure data. Each ring has a common active surface. P=1 is the classical-nuclei FSSH limit. Rescaling, frustrated-hop policy and initial distribution are explicit conventions and must match the target paper.

Code: [src/pyrpmd/surface_hopping.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/surface_hopping.py), [src/pyrpmd/advanced_surface_hopping.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/advanced_surface_hopping.py), [src/pyrpmd/adiabatic.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/adiabatic.py)

[Reproduction task](reproduction.md#task-na-rpsh-ca).

### na-rpsh-ba

**Bead-average RPSH (RPSH-BA)**

Sources: [na-rpshdm2023](#na-rpshdm2023), [na-rpsh2012](#na-rpsh2012), [na-tully1990](#na-tully1990)

A single electronic amplitude vector driven by bead-averaged energies and velocity-coupling terms, with one common active surface. This is not an independent surface label or wavefunction per bead.

Code: [src/pyrpmd/advanced_surface_hopping.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/advanced_surface_hopping.py)

[Reproduction task](reproduction.md#task-na-rpsh-ba).

### na-rpsh-dm

**Density-matrix RPSH (RPSH-DM)**

Sources: [na-rpshdm2023](#na-rpshdm2023)

Each bead has an electronic amplitude vector; bead densities are transported to a common centroid basis and averaged. The ring still has a common active surface. The reduced density can lose purity from bead averaging. No additional EDC is allowed in this DM variant.

Code: [src/pyrpmd/advanced_surface_hopping.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/advanced_surface_hopping.py)

[Reproduction task](reproduction.md#task-na-rpsh-dm).

### na-edc

**Energy-based decoherence correction (EDC)**

Sources: [na-edc2007](#na-edc2007)

Inactive electronic amplitudes are damped using energy-gap-dependent times and the active component is renormalized with its phase retained. RPSH uses explicit centroid or ring energy conventions. Supported in CA/BA, not as an extra DM correction.

Code: [src/pyrpmd/surface_hopping.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/surface_hopping.py), [src/pyrpmd/advanced_surface_hopping.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/advanced_surface_hopping.py)

[Reproduction task](reproduction.md#task-na-edc).

### na-soc-rpsh

**Complex/SOC RPSH and local geometric force**

Sources: [na-soc2019](#na-soc2019), [na-tully1990](#na-tully1990), [na-rpsh2012](#na-rpsh2012)

Complex Hermitian fixed-diabatic Hamiltonians; complex NAC is reduced to a real rescaling direction using Miao et al. method 2. Optional antisymmetric Berry-force update uses a Cayley rotation. This is a local nondegenerate correction, not a full geometric-phase interference treatment or a degenerate/non-Abelian SOC algorithm.

Code: [src/pyrpmd/advanced_surface_hopping.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/advanced_surface_hopping.py), [src/pyrpmd/adiabatic.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/adiabatic.py)

[Reproduction task](reproduction.md#task-na-soc-rpsh).

### na-dboc-rpsh

**Projected DBOC correction for RPSH**

Sources: [na-dboc2025](#na-dboc2025)

Included-state projected DBOC sum |d_ij|²/(2m) with finite-difference gradients of the scalar correction; it modifies both nuclear forces and hopping gaps. Truncated-state and near-degeneracy limits must be explicit. Accuracy improvement is regime dependent.

Code: [src/pyrpmd/advanced_surface_hopping.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/advanced_surface_hopping.py)

[Reproduction task](reproduction.md#task-na-dboc-rpsh).

### na-iesh

**Classical IESH and centroid RP-IESH (CLI methods iesh, rp-iesh)**

Sources: [na-iesh2009](#na-iesh2009), [na-iesh2023](#na-iesh2023), [na-rpsh2012](#na-rpsh2012), [na-rpsh2017](#na-rpsh2017)

One spinless Slater determinant of independent electrons in a discretized metal band; nuclei move on the bare energy plus occupied orbital energies with fewest-switches single-excitation hops. RP-IESH shares one determinant across the ring and propagates electrons at the centroid. Real Hamiltonians with nondegenerate levels; one-particle density populations; not a GPU or native full-trajectory solver.

Code: [src/pyrpmd/iesh.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/iesh.py), [src/pyrpmd/rp_iesh.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/rp_iesh.py), [src/pyrpmd/workflows/nonadiabatic_multistate.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/nonadiabatic_multistate.py)

[Reproduction task](reproduction.md#task-na-iesh).

### na-qd-interface

**Quasi-diabatic NRPMD and adiabatic on-the-fly provider**

Sources: [na-qd2018](#na-qd2018), [na-qd2019](#na-qd2019), [na-nrpmd2013](#na-nrpmd2013), [na-mint2018](#na-mint2018), [na-tully1990](#na-tully1990)

AdiabaticResult/AdiabaticPythonPotential carry energies, gradients, NACs and overlaps. QDNRPMDDynamics transports mapping coordinates through local bases, with closest-unitary overlap transport and offdiagonal gradient reconstruction. This combines the QD representation strategy with NRPMD; it does not implement the original papers' PLDM/SQC algorithms. Declared continuous-gauge endpoint-NAC transport is an approximation when true overlaps are absent.

Code: [src/pyrpmd/adiabatic.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/adiabatic.py)

[Reproduction task](reproduction.md#task-na-qd-interface).

### na-excited-state

**Online excited-state providers and overlap-derived couplings**

Sources: [na-hst1994](#na-hst1994), [na-npi2014](#na-npi2014), [na-npi2016](#na-npi2016), [pyscf2018](#pyscf2018)

ExcitedStateProvider supplies total state energies, positive gradients, derivative couplings, adjacent-frame overlaps, state tracking and a restartable electronic context to existing adiabatic consumers (for example AdiabaticRPSHDynamics). Couplings from overlaps use the Hammes-Schiffer–Tully finite difference or the multistate generalized NPI after Löwdin orthogonalization. PySCF TDA/TDDFT is tested; ORCA, Q-Chem and OpenMolcas file adapters are parser- and mock-tested only. It does not select a dynamics method.

Code: [src/pyrpmd/potentials/excited_state](https://github.com/chen-yu-hao/PyRPMD/tree/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/potentials/excited_state)

[Reproduction task](reproduction.md#task-na-excited-state).

### na-mv-rpmd

**Mapping-variable RPMD (MV-RPMD)**

Sources: [na-mv2013](#na-mv2013)

Real fixed-diabatic multistate signed projected-Wigner trace product; sample absolute weight and retain signs. The package uses exact finite-bead electronic matrix exponentials, unlike the original primitive short-time approximation. Nonseparable electronic flow uses implicit midpoint; trace nodes/sign crossings are rejected. CPU model API.

Code: [src/pyrpmd/mv_rpmd.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/mv_rpmd.py), [src/pyrpmd/_matrix_path.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/_matrix_path.py)

[Reproduction task](reproduction.md#task-na-mv-rpmd).

### na-kc-rpmd

**Kinetically constrained RPMD (KC-RPMD)**

Sources: [na-kc2014](#na-kc2014)

Original primitive two-state construction, positive electronic coupling, explicit a/b parameters and crossing calibration eta/auxiliary mass. Current limits include 0<a<=1, b>0, P>=2 and beta*K/P<1. No later KC variants are implicitly implemented.

Code: [src/pyrpmd/kc_rpmd.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/kc_rpmd.py), [src/pyrpmd/_matrix_path.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/_matrix_path.py)

[Reproduction task](reproduction.md#task-na-kc-rpmd).

### na-isomorphic

**Two-state isomorphic potential with Ehrenfest dynamics**

Sources: [na-iso2018](#na-iso2018)

Real two-state trace-matched whole-ring isomorphic Hamiltonian and an explicit Ehrenfest consumer with a single normalized electronic wavefunction. The sum-bead convention carries the factor P explicitly. This is neither the paper's QCLE dynamics nor iso-surface-hopping, and is not a general multistate implementation.

Code: [src/pyrpmd/isomorphic.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/isomorphic.py), [src/pyrpmd/_matrix_path.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/_matrix_path.py)

[Reproduction task](reproduction.md#task-na-isomorphic).

### na-tully-models

**Tully I/II/III analytic matrix potentials**

Sources: [na-tully1990](#na-tully1990)

Analytic value/gradient implementations TullyModelOne, TullyModelTwo and TullyModelThree, corresponding to the original models. Nuclear mass is a system/sampling choice, not encoded by choosing a potential class.

Code: [src/pyrpmd/potentials/nonadiabatic_models.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/potentials/nonadiabatic_models.py)

[Reproduction task](reproduction.md#task-na-tully-models).

### na-generic-matrix-models

**Generic LVC / discrete spin-boson matrix models and harmonic ring preparation**

Sources: [na-focused2019](#na-focused2019), [na-sm2021](#na-sm2021)

LinearVibronicCoupling and DiscreteSpinBoson are generic parameter containers with analytic gradients, not a uniquely published model. Harmonic preparation samples an explicitly declared finite-P uncoupled harmonic initial density; it does not infer the coupled system's thermal density. Cite the parameter source for the selected application.

Code: [src/pyrpmd/potentials/nonadiabatic_models.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/potentials/nonadiabatic_models.py), [src/pyrpmd/nonadiabatic_sampling.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/nonadiabatic_sampling.py)

[Reproduction task](reproduction.md#task-na-generic-matrix-models).

### na-nah2-pes

**NaH2 LEPS-5G coupled diabatic potential**

Sources: [na-nah21992](#na-nah21992), [na-nah2err1994](#na-nah2err1994)

Historical neutral NaH2 two-state doublet potential for the two lowest 2A' surfaces, with analytic matrix derivatives. Cite the physical PES and erratum separately from any ChemPotPy software/adaptor credit.

Code: [src/pyrpmd/potentials/nah2_leps.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/potentials/nah2_leps.py)

[Reproduction task](reproduction.md#task-na-nah2-pes).

### na-quantum-reference

**Finite-domain matrix split-operator quantum reference**

Sources: [na-feit1982](#na-feit1982)

SplitOperator1D uses a uniform periodic Fourier grid, exact pointwise Hermitian matrix exponentials and symmetric second-order splitting. Supports pure states and canonical harmonic mixtures. It has no absorbing boundary or general multidimensional propagator. A converged numerical reference must control box, grid, time step and thermal truncation independently.

Code: [src/pyrpmd/analysis/quantum_reference.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/quantum_reference.py)

[Reproduction task](reproduction.md#task-na-quantum-reference).

## Materials, electrochemistry, potentials and interfaces

### bzp

**Isotropic BZP NPT PIMD**

Sources: [bzp2009](#bzp2009), [ipi2014](#ipi2014)

Centroid-only dilation with physical virial and PIMD measure correction; all-mobile 3D periodic cells.

Code: [src/pyrpmd/barostat.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/barostat.py), [src/pyrpmd/native_ensemble.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/native_ensemble.py)

[Reproduction task](reproduction.md#task-bzp).

### mtk

**Flexible-cell MTTK/MTK PIMD**

Sources: [mtk1994](#mtk1994), [nptpimd1999](#nptpimd1999)

Upper-triangular strain-rate matrix and lower-triangular row-vector cell; hydrostatic pressure, optional cell mask.

Code: [src/pyrpmd/native_ensemble.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/native_ensemble.py)

[Reproduction task](reproduction.md#task-mtk).

### rgb

**Raiteri–Gale–Bussi stress coupling**

Sources: [rgb2011](#rgb2011)

Reference-cell strain-energy model with explicit symmetric external stress.

Code: [src/pyrpmd/native_ensemble.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/native_ensemble.py)

[Reproduction task](reproduction.md#task-rgb).

### scr

**Stochastic cell rescaling**

Sources: [scr2020](#scr2020)

Isotropic P=1 only; square-root-volume stochastic proposal without Metropolis correction.

Code: [src/pyrpmd/native_ensemble.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/native_ensemble.py)

[Reproduction task](reproduction.md#task-scr).

### diffusion

**RPMD MSD/VACF diffusion and tensor analysis**

Sources: [pd2024](#pd2024), [ni2012](#ni2012)

Unwrapped centroids, Einstein and Green–Kubo estimates, replica/block uncertainty; material applications supplement the foundational RPMD citation.

Code: [src/pyrpmd/analysis/diffusion.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/diffusion.py), [src/pyrpmd/workflows/materials.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/materials.py)

[Reproduction task](reproduction.md#task-diffusion).

### site-surface

**Basin-normalized site and surface rates**

Sources: [sds2025](#sds2025), [pd2024](#pd2024), [bazro2008](#bazro2008)

Single-surface reactant-basin normalization; nonlinear all-mobile site coordinates and restricted fixed-substrate surface coordinates have different support.

Code: [src/pyrpmd/workflows/site_rate.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/site_rate.py), [src/pyrpmd/workflows/surface.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/surface.py), [src/pyrpmd/workflows/surface_source.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/surface_source.py)

[Reproduction task](reproduction.md#task-site-surface).

### nialh-model

**NiAlH EAM pilot potential**

Sources: [nialh1995](#nialh1995)

External pinned potential table used for an integration pilot.

Code: [benchmarks/materials/README.md](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/benchmarks/materials/README.md)

[Reproduction task](reproduction.md#task-nialh-model).

### pcet

**Local golden-rule electrochemical PCET**

Sources: [pcet2008](#pcet2008), [pypcet2025](#pypcet2025), [gold2019](#gold2019)

One grid-resolved proton, fixed donor geometry, constant electronic coupling, wide-band metal and classical Gaussian environment.

Code: [src/pyrpmd/pcet.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/pcet.py), [src/pyrpmd/workflows/pcet.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/pcet.py)

[Reproduction task](reproduction.md#task-pcet).

### gc

**Grand-canonical shared-electron HMC/MC**

Sources: [hmc1987](#hmc1987), [hastings1970](#hastings1970), [gc2025](#gc2025)

Project composition: nuclear HMC plus bounded continuous shared-N Metropolis moves; MC sweep count has no physical-time meaning.

Code: [src/pyrpmd/electrochemistry/sampling.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/electrochemistry/sampling.py)

[Reproduction task](reproduction.md#task-gc).

### adiabatic-electrode

**Adiabatic constant-potential scalar and matrix wrappers**

Sources: [gc2025](#gc2025)

Project approximation: minimize E(R,N)-mu*N independently per geometry; matrix version adds the same scalar charging energy to every diabatic state.

Code: [src/pyrpmd/electrochemistry/potentials.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/electrochemistry/potentials.py), [src/pyrpmd/electrochemistry/nonadiabatic.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/electrochemistry/nonadiabatic.py), [src/pyrpmd/electrochemistry/diabatic.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/electrochemistry/diabatic.py)

[Reproduction task](reproduction.md#task-adiabatic-electrode).

### her

**Volmer–Heyrovsky–Tafel mean-field postprocessing**

Sources: [gc2025](#gc2025)

Project mass-action balance using externally supplied forward/reverse elementary coefficients and ideal independent sites.

Code: [src/pyrpmd/electrochemistry/kinetics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/electrochemistry/kinetics.py)

[Reproduction task](reproduction.md#task-her).

### efrpmd

**Markovian centroid electronic-friction RPMD**

Sources: [efrpmd2024](#efrpmd2024)

Bead-averaged positive-semidefinite friction acts on centroid; shared random impulse leaves internal modes unthermostatted.

Code: [src/pyrpmd/electronic_friction.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/electronic_friction.py)

[Reproduction task](reproduction.md#task-efrpmd).

### pyscf

**PySCF references and on-the-fly force adapter**

Sources: [pyscf2018](#pyscf2018)

Delegates RHF/UHF/RKS/UKS energies, gradients and selected Hessians to external PySCF.

Code: [src/pyrpmd/pesgen/reference.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/pesgen/reference.py), [src/pyrpmd/potentials/adapters.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/potentials/adapters.py)

[Reproduction task](reproduction.md#task-pyscf).

### populations

**Hirshfeld charges and Mayer bond orders**

Sources: [hirshfeld1977](#hirshfeld1977), [mayer1983](#mayer1983)

Neutral spherical HF stockholder references; unrestricted Mayer includes spin density.

Code: [src/pyrpmd/pesgen/reference.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/pesgen/reference.py)

[Reproduction task](reproduction.md#task-populations).

### qmdff

**QMDFF generation**

Sources: [qmdff2014](#qmdff2014), [caracal2023](#caracal2023)

Calls source-pinned upstream Caracal QMDFF generator using equilibrium QM data.

Code: [src/pyrpmd/pesgen/workflow.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/pesgen/workflow.py), [src/pyrpmd/pesgen/formats.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/pesgen/formats.py)

[Reproduction task](reproduction.md#task-qmdff).

### deevb

**dE-EVB / EVB-QMDFF coupling fit**

Sources: [deevb2015](#deevb2015), [caracal2023](#caracal2023)

Connects endpoint QMDFFs with upstream fitted sp/1g coupling and supplied path energies.

Code: [src/pyrpmd/pesgen/workflow.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/pesgen/workflow.py)

[Reproduction task](reproduction.md#task-deevb).

### treq

**TREQ PES generation**

Sources: [treq2019](#treq2019), [caracal2023](#caracal2023)

Upstream reaction-path/quadratic expansion construction; public PyRPMD bridge explicitly uses fourth-order finite-difference forces.

Code: [src/pyrpmd/pesgen/workflow.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/pesgen/workflow.py), [src/pyrpmd/pesgen/build.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/pesgen/build.py)

[Reproduction task](reproduction.md#task-treq).

### jaxpip

**Linear PIP / PIP-NN fitting and JaxPIP export**

Sources: [pip2010](#pip2010), [pipnn2013](#pipnn2013), [jaxpip2026](#jaxpip2026)

Explicit train/validation/test data, energy/force loss, upstream JaxPIP descriptors and native ONNX export; exporting a model is separate from validating its PES.

Code: [src/pyrpmd/pesgen/jaxpip_fit.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/pesgen/jaxpip_fit.py), [src/pyrpmd/pesgen/jaxpip_export.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/pesgen/jaxpip_export.py), [src/pyrpmd/pesgen/jaxpip_data.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/pesgen/jaxpip_data.py)

[Reproduction task](reproduction.md#task-jaxpip).

### committee

**Committee uncertainty and acquisition**

Sources: [committee2020](#committee2020)

Mean of fixed providers, population-standard-deviation disagreement, offline acquisition, and ActiveLearningLoop: sample with a fixed committee, select threshold-crossing candidates within label budgets, label, retrain and install the next committee. Disagreement is not a calibrated error bound; no disagreement bias is applied during sampling.

Code: [src/pyrpmd/potentials/lifecycle.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/potentials/lifecycle.py), [src/pyrpmd/active_learning.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/active_learning.py)

[Reproduction task](reproduction.md#task-committee).

### delta

**Additive delta learning with RPC/MTS**

Sources: [delta2015](#delta2015)

Adds a supplied correction model to a fixed reference and exposes a fast/slow force split; RPC/MTS have separate method citations.

Code: [src/pyrpmd/potentials/lifecycle.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/potentials/lifecycle.py), [src/pyrpmd/potentials/contraction.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/potentials/contraction.py)

[Reproduction task](reproduction.md#task-delta).

### ase

**ASE calculator adapter**

Sources: [ase2017](#ase2017)

Energy, force, periodic cell and stress conversion; force-consistent electronic free energy.

Code: [src/pyrpmd/potentials/optional.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/potentials/optional.py)

[Reproduction task](reproduction.md#task-ase).

### openmm

**OpenMM force adapter**

Sources: [openmm2017](#openmm2017)

External force evaluation, virtual sites and periodic virials; PyRPMD owns nuclear dynamics.

Code: [src/pyrpmd/potentials/openmm.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/potentials/openmm.py)

[Reproduction task](reproduction.md#task-openmm).

### lammps

**LAMMPS library adapter**

Sources: [lammps1995](#lammps1995)

Per-context external library force and cell evaluation; not a port of LAMMPS algorithms.

Code: [src/pyrpmd/potentials/lammps.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/potentials/lammps.py)

[Reproduction task](reproduction.md#task-lammps).

### ipi-socket

**i-PI socket adapter**

Sources: [ipi2014](#ipi2014)

Socket force protocol with explicit cell and units.

Code: [src/pyrpmd/potentials/socket.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/potentials/socket.py)

[Reproduction task](reproduction.md#task-ipi-socket).

### torch

**Torch/GPU potentials and tensor propagation**

Sources: [torch2019](#torch2019)

Autodifferentiated energies/forces and tensor versions of the cited physical integrators.

Code: [src/pyrpmd/potentials/optional.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/potentials/optional.py), [src/pyrpmd/torch_backend.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/torch_backend.py)

[Reproduction task](reproduction.md#task-torch).

### ml-adapters

**MACE / NequIP / DeePMD / fairchem adapters**

Sources: [mace2022](#mace2022), [nequip2022](#nequip2022), [deepmd2018](#deepmd2018), [uma2025](#uma2025)

Calls external inference engines; model paper and exact checkpoint are additional citations, not an in-repo reimplementation.

Code: [src/pyrpmd/potentials/adapters.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/potentials/adapters.py)

[Reproduction task](reproduction.md#task-ml-adapters).

### tblite

**GFN1-xTB / GFN2-xTB / IPEA1-xTB via tblite**

Sources: [gfn12017](#gfn12017), [gfn22019](#gfn22019), [ipea2017](#ipea2017)

Delegates the named electronic Hamiltonian to tblite; charge, multiplicity and electronic temperature are explicit.

Code: [src/pyrpmd/potentials/adapters.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/potentials/adapters.py)

[Reproduction task](reproduction.md#task-tblite).

### h5md-export

**H5MD, extxyz and ASE trajectory export**

Sources: [h5md2014](#h5md2014), [ase2017](#ase2017)

H5MD format provenance plus ASE interoperable formats; no new dynamics method.

Code: [src/pyrpmd/trajectory_export.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/trajectory_export.py)

[Reproduction task](reproduction.md#task-h5md-export).

### engineering

**Input, restart, parallel execution, reports and run organization**

Sources: —

Project engineering: TOML/Python schemas, hashes, restart, C ABI/f2py, atomic block propagation, process/MPI/OpenMP/GPU scheduling, reports and rate-matrix assembly.

Code: [src/pyrpmd/input.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/input.py), [src/pyrpmd/checkpoint.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/checkpoint.py), [src/pyrpmd/parallel](https://github.com/chen-yu-hao/PyRPMD/tree/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/parallel), [src/pyrpmd/trajectory.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/trajectory.py), [src/pyrpmd/analysis/reporting.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/reporting.py), [src/pyrpmd/reactions.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/reactions.py), [src/pyrpmd/workflows/network.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/network.py), [src/pyrpmd/cli.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/cli.py), [src/pyrpmd/runner.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/runner.py), [src/pyrpmd/input_schema.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/input_schema.py), [src/pyrpmd/batched.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/batched.py), [src/pyrpmd/batched_nonadiabatic.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/batched_nonadiabatic.py), [src/pyrpmd/trajectory_batch.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/trajectory_batch.py), [src/pyrpmd/electronic_backends.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/electronic_backends.py), [src/pyrpmd/geometry.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/geometry.py)

[Reproduction task](reproduction.md#task-engineering).

### analytic-models

**Analytic free, harmonic, Morse, quartic and periodic-cosine models**

Sources: [morse1929](#morse1929)

MorsePairPotential uses the Morse functional form. Free, harmonic, separable quartic and periodic-cosine providers are explicitly specified analytic controls; they are not fitted molecular or metal potentials.

Code: [src/pyrpmd/potentials/builtin.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/potentials/builtin.py)

[Reproduction task](reproduction.md#task-analytic-models).

### rate-studies

**Temperature/isotope scans, Arrhenius fits and convergence studies**

Sources: [bimol09](#bimol09), [iso13](#iso13)

Study orchestration and regression around already cited rate and isotope methods. No separate original paper is claimed for this package-specific scan/report workflow.

Code: [src/pyrpmd/studies.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/studies.py)

[Reproduction task](reproduction.md#task-rate-studies).

### bkmp2-model

**BKMP2 H3 literature benchmark and external PES provenance**

Sources: [bkmp21996](#bkmp21996), [bimol09](#bimol09), [rpmrate13](#rpmrate13)

BKMP2 is an externally sourced H3 PES used by the benchmark, not a new potential fitted by PyRPMD. Cite its original paper in addition to the rate method and adapter/software source.

Code: [src/pyrpmd/potentials/native.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/potentials/native.py), [src/pyrpmd/potentials/cabi.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/potentials/cabi.py)

[Reproduction task](reproduction.md#task-bkmp2-model).

## Bibliography

The role notes identify the relationship to the implementation, including background-only sources.

### cw81

D. Chandler; P. G. Wolynes. **Exploiting the isomorphism between quantum theory and classical statistical mechanics of polyatomic fluids**. *J. Chem. Phys. 74, 4078–4095* (1981). DOI: [10.1063/1.441588](https://doi.org/10.1063/1.441588).

Role: Ring-polymer statistical-mechanical foundation.

### pr84

M. Parrinello; A. Rahman. **Study of an F center in molten KCl**. *J. Chem. Phys. 80, 860–867* (1984). DOI: [10.1063/1.446740](https://doi.org/10.1063/1.446740).

Role: Early path-integral molecular-dynamics application.

### rpm04

I. R. Craig; D. E. Manolopoulos. **Quantum statistics and classical mechanics: Real time correlation functions from ring polymer molecular dynamics**. *J. Chem. Phys. 121, 3368–3373* (2004). DOI: [10.1063/1.1777575](https://doi.org/10.1063/1.1777575).

Role: Original method.

### rate05

I. R. Craig; D. E. Manolopoulos. **Chemical reaction rates from ring polymer molecular dynamics**. *J. Chem. Phys. 122, 084106* (2005). DOI: [10.1063/1.1850093](https://doi.org/10.1063/1.1850093).

Role: Original method.

### rate05b

I. R. Craig; D. E. Manolopoulos. **A refined ring polymer molecular dynamics theory of chemical reaction rates**. *J. Chem. Phys. 123, 034102* (2005). DOI: [10.1063/1.1954769](https://doi.org/10.1063/1.1954769).

Role: 2005 RPMD rate-theory refinement.

### bimol09

R. Collepardo-Guevara; Y. V. Suleimanov; D. E. Manolopoulos. **Bimolecular reaction rates from ring polymer molecular dynamics**. *J. Chem. Phys. 130, 174713* (2009). DOI: [10.1063/1.3127145](https://doi.org/10.1063/1.3127145).

Role: Bimolecular extension; consult its errata and corrected conventions.

### rpmrate13

Y. V. Suleimanov; J. W. Allen; W. H. Green. **RPMDrate: Bimolecular chemical reaction rates from ring polymer molecular dynamics**. *Comput. Phys. Commun. 184, 833–840* (2013). DOI: [10.1016/j.cpc.2012.10.017](https://doi.org/10.1016/j.cpc.2012.10.017).

Role: Implementation/workflow source; also cite the underlying rate-method papers.

### drpmd18

Y. V. Suleimanov; A. Aguado; S. Gómez-Carrasco; O. Roncero. **A Ring Polymer Molecular Dynamics Approach to Study the Transition between Statistical and Direct Mechanisms in the H₂ + H₃⁺ → H₃⁺ + H₂ Reaction**. *J. Phys. Chem. Lett. 9, 2133–2137* (2018). DOI: [10.1021/acs.jpclett.8b00783](https://doi.org/10.1021/acs.jpclett.8b00783).

Role: Original method.

### qct65

M. Karplus; R. N. Porter; R. D. Sharma. **Exchange Reactions with Activation Energy. I. Simple Barrier Potential for (H, H₂)**. *J. Chem. Phys. 43, 3259–3287* (1965). DOI: [10.1063/1.1697301](https://doi.org/10.1063/1.1697301).

Role: Foundational quasiclassical-trajectory application.

### nerp16

R. Welsch; K. Song; Q. Shi; S. C. Althorpe; T. F. Miller III. **Non-equilibrium dynamics from RPMD and CMD**. *J. Chem. Phys. 145, 204118* (2016). DOI: [10.1063/1.4967958](https://doi.org/10.1063/1.4967958).

Role: Original method.

### state20

A. Marjollet; R. Welsch. **Nuclear quantum effects in state-selective scattering from ring polymer molecular dynamics**. *J. Chem. Phys. 152, 194113* (2020). DOI: [10.1063/5.0004179](https://doi.org/10.1063/5.0004179).

Role: Original method.

### state21

A. Marjollet; R. Welsch. **State-selective cross sections from ring polymer molecular dynamics**. *Int. J. Quantum Chem. 121, e26447* (2021). DOI: [10.1002/qua.26447](https://doi.org/10.1002/qua.26447).

Role: Detailed state-selected methodology and tests; first published online 2020.

### state22

A. Marjollet; L. Inhester; R. Welsch. **Initial state-selected scattering for the reactions H + CH₄/CHD₃ and F + CHD₃ employing ring polymer molecular dynamics**. *J. Chem. Phys. 156, 044101* (2022). DOI: [10.1063/5.0076216](https://doi.org/10.1063/5.0076216).

Role: Polyatomic extension; harmonic initialization equations used by the implementation.

### cayley19

R. Korol; N. Bou-Rabee; T. F. Miller III. **Cayley modification for strongly stable path-integral and ring-polymer molecular dynamics**. *J. Chem. Phys. 151, 124103* (2019). DOI: [10.1063/1.5120282](https://doi.org/10.1063/1.5120282).

Role: Original method.

### bcocb20

R. Korol; J. L. Rosa-Raíces; N. Bou-Rabee; T. F. Miller III. **Dimension-free path-integral molecular dynamics without preconditioning**. *J. Chem. Phys. 152, 104102* (2020). DOI: [10.1063/1.5134810](https://doi.org/10.1063/1.5134810).

Role: Original method.

### vv82

W. C. Swope; H. C. Andersen; P. H. Berens; K. R. Wilson. **A computer simulation method for the calculation of equilibrium constants for the formation of physical clusters of molecules: Application to small water clusters**. *J. Chem. Phys. 76, 637–649* (1982). DOI: [10.1063/1.442716](https://doi.org/10.1063/1.442716).

Role: Velocity-Verlet implementation reference.

### and80

H. C. Andersen. **Molecular dynamics simulations at constant pressure and/or temperature**. *J. Chem. Phys. 72, 2384–2393* (1980). DOI: [10.1063/1.439486](https://doi.org/10.1063/1.439486).

Role: Collision thermostat; PyRPMD uses a global discrete collision-clock variant.

### nhc92

G. J. Martyna; M. L. Klein; M. Tuckerman. **Nosé–Hoover chains: The canonical ensemble via continuous dynamics**. *J. Chem. Phys. 97, 2635–2643* (1992). DOI: [10.1063/1.463940](https://doi.org/10.1063/1.463940).

Role: Original method.

### pile10

M. Ceriotti; M. Parrinello; T. E. Markland; D. E. Manolopoulos. **Efficient stochastic thermostatting of path integral molecular dynamics**. *J. Chem. Phys. 133, 124104* (2010). DOI: [10.1063/1.3489925](https://doi.org/10.1063/1.3489925).

Role: Original method.

### svr07

G. Bussi; D. Donadio; M. Parrinello. **Canonical sampling through velocity rescaling**. *J. Chem. Phys. 126, 014101* (2007). DOI: [10.1063/1.2408420](https://doi.org/10.1063/1.2408420).

Role: Original method.

### gle09

M. Ceriotti; G. Bussi; M. Parrinello. **Langevin Equation with Colored Noise for Constant-Temperature Molecular Dynamics Simulations**. *Phys. Rev. Lett. 102, 020601* (2009). DOI: [10.1103/PhysRevLett.102.020601](https://doi.org/10.1103/PhysRevLett.102.020601).

Role: Original method.

### piglet12

M. Ceriotti; D. E. Manolopoulos. **Efficient First-Principles Calculation of the Quantum Kinetic Energy and Momentum Distribution of Nuclei**. *Phys. Rev. Lett. 109, 100604* (2012). DOI: [10.1103/PhysRevLett.109.100604](https://doi.org/10.1103/PhysRevLett.109.100604).

Role: PIGLET thermostat; paper's transient anisotropic Gaussian momentum estimator is not implemented here.

### ffl18

M. Hijazi; D. M. Wilkins; M. Ceriotti. **Fast-forward Langevin dynamics with momentum flips**. *J. Chem. Phys. 148, 184109* (2018). DOI: [10.1063/1.5029833](https://doi.org/10.1063/1.5029833).

Role: Current native implementation uses the rescale variant.

### langevin07

G. Bussi; M. Parrinello. **Accurate sampling using Langevin dynamics**. *Phys. Rev. E 75, 056707* (2007). DOI: [10.1103/PhysRevE.75.056707](https://doi.org/10.1103/PhysRevE.75.056707).

Role: Langevin splitting/OU integration reference.

### baoab13

B. Leimkuhler; C. Matthews. **Robust and efficient configurational molecular sampling via Langevin dynamics**. *J. Chem. Phys. 138, 174102* (2013). DOI: [10.1063/1.4802990](https://doi.org/10.1063/1.4802990).

Role: BAOAB construction; ring-polymer extensions also require PILE/BCOCB references.

### shake77

J.-P. Ryckaert; G. Ciccotti; H. J. C. Berendsen. **Numerical integration of the Cartesian equations of motion of a system with constraints: molecular dynamics of n-alkanes**. *J. Comput. Phys. 23, 327–341* (1977). DOI: [10.1016/0021-9991(77)90098-5](https://doi.org/10.1016/0021-9991(77)90098-5).

Role: Original method.

### rattle83

H. C. Andersen. **Rattle: A 'velocity' version of the shake algorithm for molecular dynamics calculations**. *J. Comput. Phys. 52, 24–34* (1983). DOI: [10.1016/0021-9991(83)90014-1](https://doi.org/10.1016/0021-9991(83)90014-1).

Role: Original method.

### rpc08

T. E. Markland; D. E. Manolopoulos. **An efficient ring polymer contraction scheme for imaginary time path integral simulations**. *J. Chem. Phys. 129, 024105* (2008). DOI: [10.1063/1.2953308](https://doi.org/10.1063/1.2953308).

Role: Original method.

### respa92

M. Tuckerman; B. J. Berne; G. J. Martyna. **Reversible multiple time scale molecular dynamics**. *J. Chem. Phys. 97, 1990–2001* (1992). DOI: [10.1063/1.463137](https://doi.org/10.1063/1.463137).

Role: Original method.

### mts16

V. Kapil; J. VandeVondele; M. Ceriotti. **Accurate molecular dynamics and nuclear quantum effects at low cost by multiple steps in real and imaginary time: Using density functional theory to accelerate wavefunction methods**. *J. Chem. Phys. 144, 054111* (2016). DOI: [10.1063/1.4941091](https://doi.org/10.1063/1.4941091).

Role: Original method.

### refrpc16

O. Marsalek; T. E. Markland. **Ab initio molecular dynamics with nuclear quantum effects at classical cost: Ring polymer contraction for density functional theory**. *J. Chem. Phys. 144, 054112* (2016). DOI: [10.1063/1.4941093](https://doi.org/10.1063/1.4941093).

Role: Reference-potential correction application.

### cmd93

J. Cao; G. A. Voth. **A new perspective on quantum time correlation functions**. *J. Chem. Phys. 99, 10070–10073* (1993). DOI: [10.1063/1.465512](https://doi.org/10.1063/1.465512).

Role: Early centroid-dynamics paper; quartic model in existing benchmark.

### cmd94

J. Cao; G. A. Voth. **The formulation of quantum statistical mechanics based on the Feynman path centroid density. II. Dynamical properties**. *J. Chem. Phys. 100, 5106–5117* (1994). DOI: [10.1063/1.467176](https://doi.org/10.1063/1.467176).

Role: Original method.

### pacmd06

T. D. Hone; P. J. Rossky; G. A. Voth. **A comparative study of imaginary time path integral based methods for quantum dynamics**. *J. Chem. Phys. 124, 154103* (2006). DOI: [10.1063/1.2186636](https://doi.org/10.1063/1.2186636).

Role: Original method.

### fcmd05

T. D. Hone; S. Izvekov; G. A. Voth. **Fast centroid molecular dynamics: A force-matching approach for the predetermination of the effective centroid forces**. *J. Chem. Phys. 122, 054105* (2005). DOI: [10.1063/1.1836731](https://doi.org/10.1063/1.1836731).

Role: Original method.

### qcmd2019

G. Trenins; M. J. Willatt; S. C. Althorpe. **Path-integral dynamics of water using curvilinear centroids**. *J. Chem. Phys. 151, 054109* (2019). DOI: [10.1063/1.5100587](https://doi.org/10.1063/1.5100587).

Role: Original quasi-centroid molecular dynamics (QCMD).

### fqcmd2021

T. Fletcher; A. Zhu; J. E. Lawrence; D. E. Manolopoulos. **Fast quasi-centroid molecular dynamics**. *J. Chem. Phys. 155, 231101* (2021). DOI: [10.1063/5.0076704](https://doi.org/10.1063/5.0076704).

Role: Original fast QCMD with a separable correction potential.

### trpmd14

M. Rossi; M. Ceriotti; D. E. Manolopoulos. **How to remove the spurious resonances from ring polymer molecular dynamics**. *J. Chem. Phys. 140, 234116* (2014). DOI: [10.1063/1.4883861](https://doi.org/10.1063/1.4883861).

Role: Original method.

### trprate15

T. J. H. Hele; Y. V. Suleimanov. **Should thermostatted ring polymer molecular dynamics be used to calculate thermal reaction rates?**. *J. Chem. Phys. 143, 074107* (2015). DOI: [10.1063/1.4928599](https://doi.org/10.1063/1.4928599).

Role: Assessment of TRPMD rate limitations, not a new accuracy guarantee.

### water09

S. Habershon; T. E. Markland; D. E. Manolopoulos. **Competing quantum effects in the dynamics of a flexible water model**. *J. Chem. Phys. 131, 024501* (2009). DOI: [10.1063/1.3167790](https://doi.org/10.1063/1.3167790).

Role: q-TIP4P/F potential and liquid-water dynamical benchmark.

### kubo57

R. Kubo. **Statistical-Mechanical Theory of Irreversible Processes. I. General Theory and Simple Applications to Magnetic and Conduction Problems**. *J. Phys. Soc. Jpn. 12, 570–586* (1957). DOI: [10.1143/JPSJ.12.570](https://doi.org/10.1143/JPSJ.12.570).

Role: Original linear-response/correlation foundation.

### raman21

K. O. H. M. Dundas; M. T. P. Beerepoot; M. Ringholm; S. Reine; R. Bast; N. H. List; J. Kongsted; K. Ruud; J. M. H. Olsen. **Harmonic Infrared and Raman Spectra in Molecular Environments Using the Polarizable Embedding Model**. *J. Chem. Theory Comput. 17, 3599–3617* (2021). DOI: [10.1021/acs.jctc.0c01323](https://doi.org/10.1021/acs.jctc.0c01323).

Role: Formula/convention source for Raman invariants; not the historical origin of Raman theory, and polarizable embedding is not implemented.

### sfg02

A. Morita; J. T. Hynes. **A Theoretical Analysis of the Sum Frequency Generation Spectrum of the Water Surface. II. Time-Dependent Approach**. *J. Phys. Chem. B 106, 673–685* (2002). DOI: [10.1021/jp0133438](https://doi.org/10.1021/jp0133438).

Role: Original dipole–polarizability time-correlation SFG response framework; online 2001.

### ti84

M. Takahashi; M. Imada. **Monte Carlo Calculation of Quantum Systems. II. Higher Order Correction**. *J. Phys. Soc. Jpn. 53, 3765–3769* (1984). DOI: [10.1143/JPSJ.53.3765](https://doi.org/10.1143/JPSJ.53.3765).

Role: Original method.

### yam05

T. M. Yamamoto. **Path-integral virial estimator based on the scaling of fluctuation coordinates: Application to quantum clusters with fourth-order propagators**. *J. Chem. Phys. 123, 104101* (2005). DOI: [10.1063/1.2013257](https://doi.org/10.1063/1.2013257).

Role: Original method.

### iso13

M. Ceriotti; T. E. Markland. **Efficient methods and practical guidelines for simulating isotope effects**. *J. Chem. Phys. 138, 014112* (2013). DOI: [10.1063/1.4772676](https://doi.org/10.1063/1.4772676).

Role: Original method.

### iso14

B. Cheng; M. Ceriotti. **Direct path integral estimators for isotope fractionation ratios**. *J. Chem. Phys. 141, 244112* (2014). DOI: [10.1063/1.4904293](https://doi.org/10.1063/1.4904293).

Role: Original method.

### mom10

L. Lin; J. A. Morrone; R. Car; M. Parrinello. **Displaced Path Integral Formulation for the Momentum Distribution of Quantum Particles**. *Phys. Rev. Lett. 105, 110602* (2010). DOI: [10.1103/PhysRevLett.105.110602](https://doi.org/10.1103/PhysRevLett.105.110602).

Role: Related original momentum-distribution method; displaced-path free-energy estimator is not implemented.

### ceperley95

D. M. Ceperley. **Path integrals in the theory of condensed helium**. *Rev. Mod. Phys. 67, 279–355* (1995). DOI: [10.1103/RevModPhys.67.279](https://doi.org/10.1103/RevModPhys.67.279).

Role: Review/background for off-diagonal density matrices, explicitly not an original-method claim.

### pcn13

S. L. Cotter; G. O. Roberts; A. M. Stuart; D. White. **MCMC Methods for Functions: Modifying Old Algorithms to Make Them Faster**. *Statistical Science 28, 424–446* (2013). DOI: [10.1214/13-STS421](https://doi.org/10.1214/13-STS421).

Role: pCN Gaussian-preserving Metropolis proposals.

### sc16

V. Kapil; J. Behler; M. Ceriotti. **High order path integrals made easy**. *J. Chem. Phys. 145, 234103* (2016). DOI: [10.1063/1.4971438](https://doi.org/10.1063/1.4971438).

Role: Finite-difference Suzuki–Chin force implementation.

### suz95

M. Suzuki. **Hybrid exponential product formulas for unbounded operators with possible applications to Monte Carlo simulations**. *Phys. Lett. A 201, 425–428* (1995). DOI: [10.1016/0375-9601(95)00266-6](https://doi.org/10.1016/0375-9601(95)00266-6).

Role: Higher-order factorization foundation.

### chin97

S. A. Chin. **Symplectic integrators from composite operator factorizations**. *Phys. Lett. A 226, 344–348* (1997). DOI: [10.1016/S0375-9601(97)00003-0](https://doi.org/10.1016/S0375-9601(97)00003-0).

Role: Force-gradient factorization foundation.

### bose19

B. Hirshberg; V. Rizzi; M. Parrinello. **Path integral molecular dynamics for bosons**. *Proc. Natl. Acad. Sci. USA 116, 21445–21449* (2019). DOI: [10.1073/pnas.1913365116](https://doi.org/10.1073/pnas.1913365116).

Role: Original method.

### bose23

Y. M. Y. Feldman; B. Hirshberg. **Quadratic scaling bosonic path integral molecular dynamics**. *J. Chem. Phys. 159, 154107* (2023). DOI: [10.1063/5.0173749](https://doi.org/10.1063/5.0173749).

Role: Quadratic recurrence implemented by PyRPMD.

### pigs00

A. Sarsa; K. E. Schmidt; W. R. Magro. **Path integral ground state method**. *J. Chem. Phys. 113, 1366–1371* (2000). DOI: [10.1063/1.481926](https://doi.org/10.1063/1.481926).

Role: Original method.

### rpi09

J. O. Richardson; S. C. Althorpe. **Ring-polymer molecular dynamics rate-theory in the deep-tunneling regime: Connection with semiclassical instanton theory**. *J. Chem. Phys. 131, 214106* (2009). DOI: [10.1063/1.3267318](https://doi.org/10.1063/1.3267318).

Role: Original method.

### rpi16

J. O. Richardson. **Derivation of instanton rate theory from first principles**. *J. Chem. Phys. 144, 114106* (2016). DOI: [10.1063/1.4943866](https://doi.org/10.1063/1.4943866).

Role: Later first-principles derivation of instanton rate theory.

### split11

J. O. Richardson; S. C. Althorpe. **Ring-polymer instanton method for calculating tunneling splittings**. *J. Chem. Phys. 134, 054109* (2011). DOI: [10.1063/1.3530589](https://doi.org/10.1063/1.3530589).

Role: Original method.

### eck30

C. Eckart. **The Penetration of a Potential Barrier by Electrons**. *Phys. Rev. 35, 1303–1309* (1930). DOI: [10.1103/PhysRev.35.1303](https://doi.org/10.1103/PhysRev.35.1303).

Role: Original method.

### wig32

E. Wigner. **Über das Überschreiten von Potentialschwellen bei chemischen Reaktionen**. *Z. Phys. Chem. B 19, 203–216* (1932). DOI: [10.1515/zpch-1932-1920](https://doi.org/10.1515/zpch-1932-1920).

Role: Original method.

### eyr35

H. Eyring. **The Activated Complex in Chemical Reactions**. *J. Chem. Phys. 3, 107–115* (1935). DOI: [10.1063/1.1749604](https://doi.org/10.1063/1.1749604).

Role: Original method.

### yoshida1990

H. Yoshida. **Construction of higher order symplectic integrators**. *Physics Letters A 150, 262–268* (1990). DOI: [10.1016/0375-9601(90)90092-3](https://doi.org/10.1016/0375-9601(90)90092-3).

Role: Original higher-order symmetric composition coefficients.

### samp-us1974

G. M. Torrie; J. P. Valleau. **Monte Carlo free energy estimates using non-Boltzmann sampling: Application to the sub-critical Lennard-Jones fluid**. *Chemical Physics Letters 28, 578–581* (1974). DOI: [10.1016/0009-2614(74)80109-0](https://doi.org/10.1016/0009-2614(74)80109-0).

Role: Original method.

### samp-us1977

G. M. Torrie; J. P. Valleau. **Nonphysical sampling distributions in Monte Carlo free-energy estimation: Umbrella sampling**. *Journal of Computational Physics 23, 187–199* (1977). DOI: [10.1016/0021-9991(77)90121-8](https://doi.org/10.1016/0021-9991(77)90121-8).

Role: Original method.

### samp-mbar2008

M. R. Shirts; J. D. Chodera. **Statistically optimal analysis of samples from multiple equilibrium states**. *Journal of Chemical Physics 129, 124105* (2008). DOI: [10.1063/1.2978177](https://doi.org/10.1063/1.2978177).

Role: Original method.

### samp-ui2005

J. Kästner; W. Thiel. **Bridging the gap between thermodynamic integration and umbrella sampling provides a novel analysis method: “Umbrella integration”**. *Journal of Chemical Physics 123, 144104* (2005). DOI: [10.1063/1.2052648](https://doi.org/10.1063/1.2052648).

Role: Original method.

### samp-uierr2006

J. Kästner; W. Thiel. **Analysis of the statistical error in umbrella sampling simulations by umbrella integration**. *Journal of Chemical Physics 124, 234106* (2006). DOI: [10.1063/1.2206775](https://doi.org/10.1063/1.2206775).

Role: Statistical analysis / window design background.

### samp-ogre2023

S. Borgmans; S. M. J. Rogge; L. Vanduyfhuys; V. Van Speybroeck. **OGRe: Optimal Grid Refinement Protocol for Accurate Free Energy Surfaces and Its Application in Proton Hopping in Zeolites and 2D COF Stacking**. *Journal of Chemical Theory and Computation 19, 9032–9048* (2023). DOI: [10.1021/acs.jctc.3c01028](https://doi.org/10.1021/acs.jctc.3c01028).

Role: Benchmark-model source and algorithm comparison; not PyRPMD planner identity.

### samp-meta2002

A. Laio; M. Parrinello. **Escaping free-energy minima**. *PNAS 99, 12562–12566* (2002). DOI: [10.1073/pnas.202427399](https://doi.org/10.1073/pnas.202427399).

Role: Original method.

### samp-wt2008

A. Barducci; G. Bussi; M. Parrinello. **Well-Tempered Metadynamics: A Smoothly Converging and Tunable Free-Energy Method**. *Physical Review Letters 100, 020603* (2008). DOI: [10.1103/PhysRevLett.100.020603](https://doi.org/10.1103/PhysRevLett.100.020603).

Role: Original method.

### samp-walkers2006

P. Raiteri; A. Laio; F. L. Gervasio; C. Micheletti; M. Parrinello. **Efficient Reconstruction of Complex Free Energy Landscapes by Multiple Walkers Metadynamics**. *Journal of Physical Chemistry B 110, 3533–3539* (2006). DOI: [10.1021/jp054359r](https://doi.org/10.1021/jp054359r).

Role: Original method.

### samp-opes2020

M. Invernizzi; M. Parrinello. **Rethinking Metadynamics: From Bias Potentials to Probability Distributions**. *Journal of Physical Chemistry Letters 11, 2731–2736* (2020). DOI: [10.1021/acs.jpclett.0c00497](https://doi.org/10.1021/acs.jpclett.0c00497).

Role: Original method.

### samp-explore2022

M. Invernizzi; M. Parrinello. **Exploration vs Convergence Speed in Adaptive-Bias Enhanced Sampling**. *Journal of Chemical Theory and Computation 18, 3988–3996* (2022). DOI: [10.1021/acs.jctc.2c00152](https://doi.org/10.1021/acs.jctc.2c00152).

Role: Original method.

### samp-ves2014

O. Valsson; M. Parrinello. **Variational Approach to Enhanced Sampling and Free Energy Calculations**. *Phys. Rev. Lett. 113, 090601* (2014). DOI: [10.1103/PhysRevLett.113.090601](https://doi.org/10.1103/PhysRevLett.113.090601).

Role: Original variationally enhanced sampling functional.

### samp-wtves2015

O. Valsson; M. Parrinello. **Well-Tempered Variational Approach to Enhanced Sampling**. *J. Chem. Theory Comput. 11, 1996–2002* (2015). DOI: [10.1021/acs.jctc.5b00076](https://doi.org/10.1021/acs.jctc.5b00076).

Role: Well-tempered target distribution for VES.

### samp-abf2001

E. Darve; A. Pohorille. **Calculating free energies using average force**. *J. Chem. Phys. 115, 9169–9183* (2001). DOI: [10.1063/1.1410978](https://doi.org/10.1063/1.1410978).

Role: Original adaptive biasing force (average force) method.

### samp-abf2008

E. Darve; D. Rodríguez-Gómez; A. Pohorille. **Adaptive biasing force method for scalar and vector free energy calculations**. *J. Chem. Phys. 128, 144120* (2008). DOI: [10.1063/1.2829861](https://doi.org/10.1063/1.2829861).

Role: Multidimensional ABF mean-force formula with first derivatives.

### samp-abfproj2015

H. Alrachid; T. Lelièvre. **Long-time convergence of an adaptive biasing force method: Variance reduction by Helmholtz projection**. *SMAI J. Comput. Math. 1, 55–82* (2015). DOI: [10.5802/smai-jcm.4](https://doi.org/10.5802/smai-jcm.4).

Role: Helmholtz projection of the estimated mean force used for 2D integration.

### samp-eabf2017

A. Lesage; T. Lelièvre; G. Stoltz; J. Hénin. **Smoothed Biasing Forces Yield Unbiased Free Energies with the Extended-System Adaptive Biasing Force Method**. *J. Phys. Chem. B 121, 3676–3685* (2017). DOI: [10.1021/acs.jpcb.6b10055](https://doi.org/10.1021/acs.jpcb.6b10055).

Role: eABF and the CZAR free-energy estimator.

### samp-jarzynski1997

C. Jarzynski. **Nonequilibrium Equality for Free Energy Differences**. *Phys. Rev. Lett. 78, 2690–2693* (1997). DOI: [10.1103/PhysRevLett.78.2690](https://doi.org/10.1103/PhysRevLett.78.2690).

Role: Original nonequilibrium work equality.

### samp-crooks1999

G. E. Crooks. **Entropy production fluctuation theorem and the nonequilibrium work relation for free energy differences**. *Phys. Rev. E 60, 2721* (1999). DOI: [10.1103/PhysRevE.60.2721](https://doi.org/10.1103/PhysRevE.60.2721).

Role: Original work fluctuation theorem behind the bidirectional estimator.

### samp-bar1976

C. H. Bennett. **Efficient estimation of free energy differences from Monte Carlo data**. *J. Comput. Phys. 22, 245–268* (1976). DOI: [10.1016/0021-9991(76)90078-4](https://doi.org/10.1016/0021-9991(76)90078-4).

Role: Original acceptance-ratio (BAR) estimator used for bidirectional work.

### samp-shadow2013

D. A. Sivak; J. D. Chodera; G. E. Crooks. **Using Nonequilibrium Fluctuation Theorems to Understand and Correct Errors in Equilibrium and Nonequilibrium Simulations of Discrete Langevin Dynamics**. *Phys. Rev. X 3, 011007* (2013). DOI: [10.1103/PhysRevX.3.011007](https://doi.org/10.1103/PhysRevX.3.011007).

Role: Shadow-work interpretation of finite-timestep Langevin error.

### samp-remd1999

Y. Sugita; Y. Okamoto. **Replica-exchange molecular dynamics method for protein folding**. *Chemical Physics Letters 314, 141–151* (1999). DOI: [10.1016/S0009-2614(99)01123-9](https://doi.org/10.1016/S0009-2614(99)01123-9).

Role: Original method.

### samp-reus2000

Y. Sugita; A. Kitao; Y. Okamoto. **Multidimensional replica-exchange method for free-energy calculations**. *Journal of Chemical Physics 113, 6042–6051* (2000). DOI: [10.1063/1.1308516](https://doi.org/10.1063/1.1308516).

Role: Original method.

### samp-pt1996

K. Hukushima; K. Nemoto. **Exchange Monte Carlo Method and Application to Spin Glass Simulations**. *J. Phys. Soc. Jpn. 65, 1604–1608* (1996). DOI: [10.1143/JPSJ.65.1604](https://doi.org/10.1143/JPSJ.65.1604).

Role: Original exchange Monte Carlo (parallel tempering) method.

### samp-imeta2013

P. Tiwary; M. Parrinello. **From Metadynamics to Dynamics**. *Physical Review Letters 111, 230602* (2013). DOI: [10.1103/PhysRevLett.111.230602](https://doi.org/10.1103/PhysRevLett.111.230602).

Role: Original method.

### samp-flooding2022

D. Ray; N. Ansari; V. Rizzi; M. Invernizzi; M. Parrinello. **Rare Event Kinetics from Adaptive Bias Enhanced Sampling**. *Journal of Chemical Theory and Computation 18, 6500–6509* (2022). DOI: [10.1021/acs.jctc.2c00806](https://doi.org/10.1021/acs.jctc.2c00806).

Role: Acceleration-clock applicability; native OPES-flooding exclusion not implemented.

### samp-tis2003

T. S. van Erp; D. Moroni; P. G. Bolhuis. **A novel path sampling method for the calculation of rate constants**. *Journal of Chemical Physics 118, 7762–7774* (2003). DOI: [10.1063/1.1562614](https://doi.org/10.1063/1.1562614).

Role: Original method.

### samp-tis2005

T. S. van Erp; P. G. Bolhuis. **Elaborating transition interface sampling methods**. *Journal of Computational Physics 205, 157–181* (2005). DOI: [10.1016/j.jcp.2004.11.003](https://doi.org/10.1016/j.jcp.2004.11.003).

Role: Detailed path/flux measure and extensions.

### samp-retis2007

T. S. van Erp. **Reaction rate calculation by parallel path swapping**. *Physical Review Letters 98, 268301* (2007). DOI: [10.1103/PhysRevLett.98.268301](https://doi.org/10.1103/PhysRevLett.98.268301).

Role: Original method.

### samp-pptis2004

D. Moroni; P. G. Bolhuis; T. S. van Erp. **Rate constants for diffusive processes by partial path sampling**. *J. Chem. Phys. 120, 4055–4065* (2004). DOI: [10.1063/1.1644537](https://doi.org/10.1063/1.1644537).

Role: Original partial-path TIS memory-reduction method.

### samp-repptis2023

W. Vervust; D. T. Zhang; T. S. van Erp; A. Ghysels. **Path sampling with memory reduction and replica exchange to reach long permeation timescales**. *Biophysical Journal 122, 2960–2972* (2023). DOI: [10.1016/j.bpj.2023.02.021](https://doi.org/10.1016/j.bpj.2023.02.021).

Role: Original REPPTIS formulation.

### samp-ffs2005

R. J. Allen; P. B. Warren; P. R. ten Wolde. **Sampling Rare Switching Events in Biochemical Networks**. *Phys. Rev. Lett. 94, 018104* (2005). DOI: [10.1103/PhysRevLett.94.018104](https://doi.org/10.1103/PhysRevLett.94.018104).

Role: Original forward flux sampling method.

### samp-ffs2009

R. J. Allen; C. Valeriani; P. R. ten Wolde. **Forward flux sampling for rare event simulations**. *J. Phys.: Condens. Matter 21, 463102* (2009). DOI: [10.1088/0953-8984/21/46/463102](https://doi.org/10.1088/0953-8984/21/46/463102).

Role: FFS review; stochastic-splitting construction followed by the implementation.

### samp-infretis2022

S. Roet; D. T. Zhang; T. S. van Erp. **Exchanging Replicas with Unequal Cost, Infinitely and Permanently**. *J. Phys. Chem. A 126, 8878–8886* (2022). DOI: [10.1021/acs.jpca.2c06004](https://doi.org/10.1021/acs.jpca.2c06004).

Role: Original asynchronous infinite-swap replica exchange (∞RETIS) algorithm.

### samp-infretis2024

D. T. Zhang; L. Baldauf; S. Roet; A. Lervik; T. S. van Erp. **Highly parallelizable path sampling with minimal rejections using asynchronous replica exchange and infinite swaps**. *Proc. Natl. Acad. Sci. USA 121, e2318731121* (2024). DOI: [10.1073/pnas.2318731121](https://doi.org/10.1073/pnas.2318731121).

Role: Later ∞RETIS developments; its subtrajectory moves are not implemented.

### samp-tps1998

P. G. Bolhuis; C. Dellago; D. Chandler. **Sampling ensembles of deterministic transition pathways**. *Faraday Discussions 110, 421–436* (1998). DOI: [10.1039/A801266K](https://doi.org/10.1039/A801266K).

Role: Deterministic trajectory sampling antecedent.

### samp-shoot2025

S. Falkner; A. Coretti; B. Peters; P. G. Bolhuis; C. Dellago. **Revisiting shooting point Monte Carlo methods for transition path sampling**. *Journal of Chemical Physics 163, 034105* (2025). DOI: [10.1063/5.0261744](https://doi.org/10.1063/5.0261744).

Role: Acceptance-rule analysis for shooting indices.

### samp-comm2000

P. G. Bolhuis; C. Dellago; D. Chandler. **Reaction coordinates of biomolecular isomerization**. *PNAS 97, 5877–5882* (2000). DOI: [10.1073/pnas.100127697](https://doi.org/10.1073/pnas.100127697).

Role: Primary molecular committor application; not a new exact-quantum committor algorithm.

### samp-cp1934

C. J. Clopper; E. S. Pearson. **The use of confidence or fiducial limits illustrated in the case of the binomial**. *Biometrika 26, 404–413* (1934). DOI: [10.1093/biomet/26.4.404](https://doi.org/10.1093/biomet/26.4.404).

Role: Exact binomial interval; PyRPMD composes two bounds into a censoring envelope.

### samp-path2007

D. Branduardi; F. L. Gervasio; M. Parrinello. **From A to B in free energy space**. *Journal of Chemical Physics 126, 054103* (2007). DOI: [10.1063/1.2432340](https://doi.org/10.1063/1.2432340).

Role: Original method.

### samp-mlcolvar2023

L. Bonati; E. Trizio; A. Rizzi; M. Parrinello. **A unified framework for machine learning collective variables for enhanced sampling simulations: mlcolvar**. *J. Chem. Phys. 159, 014801* (2023). DOI: [10.1063/5.0156343](https://doi.org/10.1063/5.0156343).

Role: software; external training library for learned CVs.

### samp-deeplda2020

L. Bonati; V. Rizzi; M. Parrinello. **Data-Driven Collective Variables for Enhanced Sampling**. *J. Phys. Chem. Lett. 11, 2998–3004* (2020). DOI: [10.1021/acs.jpclett.0c00535](https://doi.org/10.1021/acs.jpclett.0c00535).

Role: Original Deep-LDA collective variables.

### samp-deeptica2021

L. Bonati; G. Piccini; M. Parrinello. **Deep learning the slow modes for rare events sampling**. *Proc. Natl. Acad. Sci. USA 118, e2113533118* (2021). DOI: [10.1073/pnas.2113533118](https://doi.org/10.1073/pnas.2113533118).

Role: Original Deep-TICA collective variables.

### samp-kabsch1976

W. Kabsch. **A solution for the best rotation to relate two sets of vectors**. *Acta Crystallographica A 32, 922–923* (1976). DOI: [10.1107/S0567739476001873](https://doi.org/10.1107/S0567739476001873).

Role: Weighted optimal rotation underpinning RMSD; not a molecular dynamics method.

### samp-string2002

W. E; W. Ren; E. Vanden-Eijnden. **String method for the study of rare events**. *Physical Review B 66, 052301* (2002). DOI: [10.1103/PhysRevB.66.052301](https://doi.org/10.1103/PhysRevB.66.052301).

Role: Original method.

### samp-stringcv2006

L. Maragliano; A. Fischer; E. Vanden-Eijnden; G. Ciccotti. **String method in collective variables: Minimum free energy paths and isocommittor surfaces**. *Journal of Chemical Physics 125, 024106* (2006). DOI: [10.1063/1.2212942](https://doi.org/10.1063/1.2212942).

Role: CV-string theoretical background; general nonlinear-CV metric terms not fully implemented.

### samp-fts2005

W. E; W. Ren; E. Vanden-Eijnden. **Finite Temperature String Method for the Study of Rare Events**. *J. Phys. Chem. B 109, 6688–6693* (2005). DOI: [10.1021/jp0455430](https://doi.org/10.1021/jp0455430).

Role: Original finite-temperature string method.

### samp-prfo1985

A. Banerjee; N. Adams; J. Simons; R. Shepard. **Search for stationary points on surfaces**. *Journal of Physical Chemistry 89, 52–57* (1985). DOI: [10.1021/j100247a015](https://doi.org/10.1021/j100247a015).

Role: Original method.

### samp-nebtangent2000

G. Henkelman; H. Jónsson. **Improved tangent estimate in the nudged elastic band method for finding minimum energy paths and saddle points**. *Journal of Chemical Physics 113, 9978–9985* (2000). DOI: [10.1063/1.1323224](https://doi.org/10.1063/1.1323224).

Role: Original method.

### samp-cineb2000

G. Henkelman; B. P. Uberuaga; H. Jónsson. **A climbing image nudged elastic band method for finding saddle points and minimum energy paths**. *Journal of Chemical Physics 113, 9901–9904* (2000). DOI: [10.1063/1.1329672](https://doi.org/10.1063/1.1329672).

Role: Original method.

### samp-fire2006

E. Bitzek; P. Koskinen; F. Gähler; M. Moseler; P. Gumbsch. **Structural Relaxation Made Simple**. *Physical Review Letters 97, 170201* (2006). DOI: [10.1103/PhysRevLett.97.170201](https://doi.org/10.1103/PhysRevLett.97.170201).

Role: Original method.

### samp-radii2008

B. Cordero; V. Gómez; A. E. Platero-Prats; M. Revés; J. Echeverría; E. Cremades; F. Barragán; S. Alvarez. **Covalent radii revisited**. *Dalton Transactions, 2832–2838* (2008). DOI: [10.1039/B801115J](https://doi.org/10.1039/B801115J).

Role: Published parameter data; connectivity threshold is a PyRPMD heuristic.

### samp-bluemoon1989

E. A. Carter; G. Ciccotti; J. T. Hynes; R. Kapral. **Constrained reaction coordinate dynamics for the simulation of rare events**. *Chemical Physics Letters 156, 472–477* (1989). DOI: [10.1016/S0009-2614(89)87314-2](https://doi.org/10.1016/S0009-2614(89)87314-2).

Role: Constrained-to-conditional ensemble reweighting antecedent.

### samp-blue1998

M. Sprik; G. Ciccotti. **Free energy from constrained molecular dynamics**. *Journal of Chemical Physics 109, 7737–7744* (1998). DOI: [10.1063/1.477419](https://doi.org/10.1063/1.477419).

Role: Mass-metric / constrained-ensemble relation.

### samp-plumed2009

M. Bonomi; D. Branduardi; G. Bussi; C. Camilloni; D. Provasi; P. Raiteri; D. Donadio; F. Marinelli; F. Pietrucci; R. A. Broglia; M. Parrinello. **PLUMED: A portable plugin for free-energy calculations with molecular dynamics**. *Computer Physics Communications 180, 1961–1972* (2009). DOI: [10.1016/j.cpc.2009.05.011](https://doi.org/10.1016/j.cpc.2009.05.011).

Role: External software interface; cite selected PLUMED algorithms separately.

### samp-plumed2014

G. A. Tribello; M. Bonomi; D. Branduardi; C. Camilloni; G. Bussi. **PLUMED 2: New feathers for an old bird**. *Computer Physics Communications 185, 604–613* (2014). DOI: [10.1016/j.cpc.2013.09.018](https://doi.org/10.1016/j.cpc.2013.09.018).

Role: External software interface.

### samp-cblock1992

D. N. Politis; J. P. Romano. **A Circular Block-Resampling Procedure for Stationary Data**. *In Exploring the Limits of Bootstrap, R. LePage and L. Billard (eds.), Wiley, pp. 263–270* (1992). [Source / author version](https://statistics.stanford.edu/technical-reports/circular-block-resampling-procedure-stationary-data).

Role: Original circular-block resampling; linked 1991 technical-report precursor, no DOI asserted.

### samp-geyer1992

C. J. Geyer. **Practical Markov Chain Monte Carlo**. *Statistical Science 7, 473–483* (1992). DOI: [10.1214/ss/1177011137](https://doi.org/10.1214/ss/1177011137).

Role: Positive-sequence variance-estimation background; local pairing and observables are implementation choices.

### samp-wham1992

S. Kumar; J. M. Rosenberg; D. Bouzida; R. H. Swendsen; P. A. Kollman. **The weighted histogram analysis method for free-energy calculations on biomolecules. I. The method**. *Journal of Computational Chemistry 13, 1011–1021* (1992). DOI: [10.1002/jcc.540130812](https://doi.org/10.1002/jcc.540130812).

Role: Historical/comparison background, not an independent WHAM solver in this code.

### na-mm1979

H.-D. Meyer and W. H. Miller. **A classical analog for electronic degrees of freedom in nonadiabatic collision processes**. *The Journal of Chemical Physics 70, 3214–3223* (1979). DOI: [10.1063/1.437910](https://doi.org/10.1063/1.437910).

Role: Original classical electronic mapping antecedent; the complete classical S-matrix calculation is not implemented merely by using MMST variables.

### na-stock1997

G. Stock and M. Thoss. **Semiclassical Description of Nonadiabatic Quantum Dynamics**. *Physical Review Letters 78, 578–581* (1997). DOI: [10.1103/PhysRevLett.78.578](https://doi.org/10.1103/PhysRevLett.78.578). [Source / author version](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.78.578).

Role: Quantum mapping / SEO antecedent, not evidence that the package implements the full semiclassical IVR method.

### na-nrpmd2013

J. O. Richardson and M. Thoss. **Communication: Nonadiabatic ring-polymer molecular dynamics**. *The Journal of Chemical Physics 139, 031102* (2013). DOI: [10.1063/1.4816124](https://doi.org/10.1063/1.4816124).

Role: Original NRPMD and its SEO-projected thermal construction; direct origin for the published Fig. 1(c) benchmark.

### na-nrpmd2017

J. O. Richardson, P. Meyer, M.-O. Pleinert, and M. Thoss. **An analysis of nonadiabatic ring-polymer molecular dynamics and its application to vibronic spectra**. *Chemical Physics 482, 124–134* (2017). DOI: [10.1016/j.chemphys.2016.09.036](https://doi.org/10.1016/j.chemphys.2016.09.036). [Source / author version](https://www.sciencedirect.com/science/article/abs/pii/S0301010416305717).

Role: Theoretical analysis and generalized independent mapping/nuclear bead construction; background only for features beyond one mapping vector per nuclear bead.

### na-focused2019

S. N. Chowdhury and P. Huo. **State dependent ring polymer molecular dynamics for investigating excited nonadiabatic dynamics**. *The Journal of Chemical Physics 150, 244102* (2019). DOI: [10.1063/1.5096276](https://doi.org/10.1063/1.5096276). [Source / author version](https://huogroup.github.io/pdf/jcp8.pdf).

Role: Focused nonequilibrium electronic preparation and population estimator used by the public NRPMD path, especially Eqs. 20–23.

### na-mint2018

M. S. Church, T. J. H. Hele, G. S. Ezra, and N. Ananth. **Nonadiabatic semiclassical dynamics in the mixed quantum-classical initial value representation**. *The Journal of Chemical Physics 148, 102326* (2018). DOI: [10.1063/1.5005557](https://doi.org/10.1063/1.5005557). [Source / author version](https://arxiv.org/abs/1709.07474).

Role: MInt exact mapping/nuclear-momentum subflow component; the paper's full MQC-IVR algorithm is not implemented by this substep.

### na-sm2021

D. Bossion, S. N. Chowdhury, and P. Huo. **Non-adiabatic ring polymer molecular dynamics with spin mapping variables**. *The Journal of Chemical Physics 154, 184106* (2021). DOI: [10.1063/5.0051456](https://doi.org/10.1063/5.0051456). [Source / author version](https://huogroup.github.io/pdf/jcp11.pdf).

Role: Original SM-NRPMD; the implemented focused two-state equations do not include the paper's full Gamma_W equilibrium initialization.

### na-mash2023

J. R. Mannouch and J. O. Richardson. **A mapping approach to surface hopping**. *The Journal of Chemical Physics 158, 104111* (2023). DOI: [10.1063/5.0139734](https://doi.org/10.1063/5.0139734). [Source / author version](https://arxiv.org/abs/2212.11773).

Role: Original two-state MASH with deterministic spin-hemisphere surface selection.

### na-mashrates2024

J. E. Lawrence, J. R. Mannouch, and J. O. Richardson. **Recovering Marcus Theory Rates and Beyond without the Need for Decoherence Corrections: The Mapping Approach to Surface Hopping**. *The Journal of Physical Chemistry Letters 15, 707–716* (2024). DOI: [10.1021/acs.jpclett.3c03197](https://doi.org/10.1021/acs.jpclett.3c03197). [Source / author version](https://pmc.ncbi.nlm.nih.gov/articles/PMC10823533/).

Role: Population sampling/ET-rate follow-up used in interpreting MASH and CA-RP-MASH observables.

### na-carp2025

Z. Cao, V. A. Suarez, and J. S. Kretchmer. **A ring polymer molecular dynamics extension of the mapping approach to surface hopping: Incorporation of nuclear quantum effects**. *ChemRxiv, version 1 (7 December 2025), preprint* (2025). DOI: [10.26434/chemrxiv-2025-xps2g](https://doi.org/10.26434/chemrxiv-2025-xps2g). [Source / author version](https://client.prod.orp.cambridge.org/engage/chemrxiv/article-details/692ba382a10c9f5ca169b8de).

Role: Implemented CA-RP-MASH algorithm and operator-pair weights; record the implemented preprint version explicitly.

### na-msmash2023

J. E. Runeson; D. E. Manolopoulos. **A multi-state mapping approach to surface hopping**. *J. Chem. Phys. 159, 094115* (2023). DOI: [10.1063/5.0158147](https://doi.org/10.1063/5.0158147).

Role: Original multistate MASH and its signed observable estimator.

### na-tully1990

J. C. Tully. **Molecular dynamics with electronic transitions**. *The Journal of Chemical Physics 93, 1061–1071* (1990). DOI: [10.1063/1.459170](https://doi.org/10.1063/1.459170).

Role: Original fewest-switches algorithm and Tully's three two-state scattering models.

### na-rpsh2012

P. Shushkov, R. Li, and J. C. Tully. **Ring polymer molecular dynamics with surface hopping**. *The Journal of Chemical Physics 137, 22A549* (2012). DOI: [10.1063/1.4766449](https://doi.org/10.1063/1.4766449).

Role: Original RPSH proposal, with thermal scattering-rate tests.

### na-rpsh2017

F. A. Shakib and P. Huo. **Ring Polymer Surface Hopping: Incorporating Nuclear Quantum Effects into Nonadiabatic Molecular Dynamics Simulations**. *The Journal of Physical Chemistry Letters 8, 3073–3080* (2017). DOI: [10.1021/acs.jpclett.7b01343](https://doi.org/10.1021/acs.jpclett.7b01343). [Source / author version](https://pubs.acs.org/doi/10.1021/acs.jpclett.7b01343).

Role: Nonequilibrium RPSH benchmark and centroid/full-ring rescaling convention discussion; Tully models and low-momentum scattering.

### na-iesh2009

N. Shenvi; S. Roy; J. C. Tully. **Nonadiabatic dynamics at metal surfaces: Independent-electron surface hopping**. *J. Chem. Phys. 130, 174107* (2009). DOI: [10.1063/1.3125436](https://doi.org/10.1063/1.3125436).

Role: Original independent-electron surface hopping.

### na-iesh2023

J. Gardner; D. Corken; S. M. Janke; S. Habershon; R. J. Maurer. **Efficient implementation and performance analysis of the independent electron surface hopping method for dynamics at metal surfaces**. *J. Chem. Phys. 158, 064101* (2023). DOI: [10.1063/5.0137137](https://doi.org/10.1063/5.0137137).

Role: IESH implementation analysis and discretized metal model followed by the implementation.

### na-rpshdm2023

Y. Shakya, R. Welsch, L. Inhester, and R. Santra. **Capturing electronic decoherence in quantum-classical dynamics using the ring-polymer-surface-hopping–density-matrix approach**. *Physical Review A 107, 062810* (2023). DOI: [10.1103/PhysRevA.107.062810](https://doi.org/10.1103/PhysRevA.107.062810). [Source / author version](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.107.062810).

Role: Explicit CA, bead-average (BA), and bead-electronic density-matrix (DM) formulations; first author is Yashoj Shakya, initial Y., not P.

### na-dboc2025

D. K. Limbu, S. Bhusal, D. M. Castañeda-Bagatella, and F. A. Shakib. **Diagonal Born–Oppenheimer corrections in condensed-phase ring polymer surface hopping**. *The Journal of Chemical Physics 163, 234102* (2025). DOI: [10.1063/5.0301249](https://doi.org/10.1063/5.0301249).

Role: DBOC-RPSH extension and condensed-phase spin-boson benchmark; not a guarantee of accuracy improvement in every regime.

### na-soc2019

G. Miao, N. Bellonzi, and J. E. Subotnik. **An extension of the fewest switches surface hopping algorithm to complex Hamiltonians and photophysics in magnetic fields: Berry curvature and ‘magnetic’ forces**. *The Journal of Chemical Physics 150, 124101* (2019). DOI: [10.1063/1.5088770](https://doi.org/10.1063/1.5088770). [Source / author version](https://arxiv.org/pdf/1902.05115).

Role: Complex derivative-coupling rescaling direction (method 2) and local geometric force; ring-polymer use is an adaptation.

### na-edc2007

G. Granucci and M. Persico. **Critical appraisal of the fewest switches algorithm for surface hopping**. *The Journal of Chemical Physics 126, 134114* (2007). DOI: [10.1063/1.2715585](https://doi.org/10.1063/1.2715585).

Role: Energy-based decoherence correction, especially the amplitude damping prescription in Eq. 17; centroid/ring energies are implementation conventions.

### na-qd2018

A. Mandal, S. S. R. K. C. Yamijala, and P. Huo. **Quasi-Diabatic Representation for Nonadiabatic Dynamics Propagation**. *Journal of Chemical Theory and Computation 14, 1828–1840* (2018). DOI: [10.1021/acs.jctc.7b01178](https://doi.org/10.1021/acs.jctc.7b01178). [Source / author version](https://pubs.acs.org/doi/10.1021/acs.jctc.7b01178).

Role: QD local-basis/overlap strategy; its PLDM calculation is not implemented merely by reusing QD transport.

### na-qd2019

W. Zhou, A. Mandal, and P. Huo. **Quasi-Diabatic Scheme for Nonadiabatic On-the-Fly Simulations**. *The Journal of Physical Chemistry Letters 10, 7062–7070* (2019). DOI: [10.1021/acs.jpclett.9b02747](https://doi.org/10.1021/acs.jpclett.9b02747). [Source / author version](https://pubs.acs.org/doi/10.1021/acs.jpclett.9b02747).

Role: On-the-fly QD demonstration with ethylene; package QD-NRPMD is a composition with a different dynamical approximation.

### na-hst1994

S. Hammes-Schiffer; J. C. Tully. **Proton transfer in solution: Molecular dynamics with quantum transitions**. *J. Chem. Phys. 101, 4657–4667* (1994). DOI: [10.1063/1.467455](https://doi.org/10.1063/1.467455).

Role: Finite-difference time-derivative coupling from adjacent-frame overlaps.

### na-npi2014

G. A. Meek; B. G. Levine. **Evaluation of the Time-Derivative Coupling for Accurate Electronic State Transition Probabilities from Numerical Simulations**. *J. Phys. Chem. Lett. 5, 2351–2356* (2014). DOI: [10.1021/jz5009449](https://doi.org/10.1021/jz5009449).

Role: Original norm-preserving interpolation of overlap-based couplings.

### na-npi2016

A. Jain; E. Alguire; J. E. Subotnik. **An Efficient, Augmented Surface Hopping Algorithm That Includes Decoherence for Use in Large-Scale Simulations**. *J. Chem. Theory Comput. 12, 5256–5268* (2016). DOI: [10.1021/acs.jctc.6b00673](https://doi.org/10.1021/acs.jctc.6b00673).

Role: Multistate generalized NPI time-derivative matrix used by the overlap-coupling path; its A-FSSH decoherence is not implemented here.

### na-mv2013

N. Ananth. **Mapping variable ring polymer molecular dynamics: A path-integral based method for nonadiabatic processes**. *The Journal of Chemical Physics 139, 124102* (2013). DOI: [10.1063/1.4821590](https://doi.org/10.1063/1.4821590). [Source / author version](https://arxiv.org/abs/1309.4476).

Role: Original MV-RPMD signed mapping partition function and model dynamics.

### na-kc2014

A. R. Menzeleev, F. Bell, and T. F. Miller III. **Kinetically constrained ring-polymer molecular dynamics for non-adiabatic chemical reactions**. *The Journal of Chemical Physics 140, 064103* (2014). DOI: [10.1063/1.4863919](https://doi.org/10.1063/1.4863919). [Source / author version](https://arxiv.org/abs/1401.3033).

Role: Original primitive two-state KC-RPMD, crossing calibration and auxiliary-coordinate dynamics.

### na-iso2018

X. Tao, P. Shushkov, and T. F. Miller III. **Path-integral isomorphic Hamiltonian for including nuclear quantum effects in non-adiabatic dynamics**. *The Journal of Chemical Physics 148, 102327* (2018). DOI: [10.1063/1.5005544](https://doi.org/10.1063/1.5005544). [Source / author version](https://arxiv.org/abs/1709.06722).

Role: Trace-matched isomorphic Hamiltonian; the package implements an explicit two-state Ehrenfest consumer, not all QCLE/SH dynamics in the paper.

### na-nah21992

P. Halvick and D. G. Truhlar. **A new diabatic representation of the coupled potential energy surfaces for Na(3p 2P)+H2 → Na(3s 2S)+H2 or NaH+H**. *The Journal of Chemical Physics 96, 2895–2909* (1992). DOI: [10.1063/1.461986](https://doi.org/10.1063/1.461986).

Role: Original coupled NaH2 diabatic potential; physical origin is independent of the selected propagation method.

### na-nah2err1994

P. Halvick and D. G. Truhlar. **Erratum: A new diabatic representation of the coupled potential energy surfaces for Na(3p 2P)+H2 → Na(3s 2S)+H2 or NaH+H**. *The Journal of Chemical Physics 100, 4718* (1994). DOI: [10.1063/1.467282](https://doi.org/10.1063/1.467282).

Role: Required correction accompanying the original NaH2 potential citation.

### na-feit1982

M. D. Feit, J. A. Fleck Jr., and A. Steiger. **Solution of the Schrödinger equation by a spectral method**. *Journal of Computational Physics 47, 412–433* (1982). DOI: [10.1016/0021-9991(82)90091-2](https://doi.org/10.1016/0021-9991(82)90091-2). [Source / author version](https://www.sciencedirect.com/science/article/pii/0021999182900912).

Role: Fourier spectral/split-operator quantum propagation antecedent; the exactness claim requires finite-box/grid/time convergence.

### bzp2009

G. Bussi; T. Zykova-Timan; M. Parrinello. **Isothermal-isobaric molecular dynamics using stochastic velocity rescaling**. *J. Chem. Phys. 130, 074101* (2009). DOI: [10.1063/1.3073889](https://doi.org/10.1063/1.3073889). [Source / author version](https://arxiv.org/abs/0901.0779).

Role: method.

### mtk1994

G. J. Martyna; D. J. Tobias; M. L. Klein. **Constant pressure molecular dynamics algorithms**. *J. Chem. Phys. 101, 4177–4189* (1994). DOI: [10.1063/1.467468](https://doi.org/10.1063/1.467468).

Role: method.

### nptpimd1999

G. J. Martyna; A. Hughes; M. E. Tuckerman. **Molecular dynamics algorithms for path integrals at constant pressure**. *J. Chem. Phys. 110, 3275–3290* (1999). DOI: [10.1063/1.478193](https://doi.org/10.1063/1.478193).

Role: method.

### rgb2011

P. Raiteri; J. D. Gale; G. Bussi. **Reactive force field simulation of proton diffusion in BaZrO3 using an empirical valence bond approach**. *J. Phys.: Condens. Matter 23, 334213* (2011). DOI: [10.1088/0953-8984/23/33/334213](https://doi.org/10.1088/0953-8984/23/33/334213). [Source / author version](https://arxiv.org/abs/1104.0773).

Role: method.

### scr2020

M. Bernetti; G. Bussi. **Pressure control using stochastic cell rescaling**. *J. Chem. Phys. 153, 114107* (2020). DOI: [10.1063/5.0020514](https://doi.org/10.1063/5.0020514). [Source / author version](https://arxiv.org/abs/2006.09250).

Role: method.

### ipi2014

M. Ceriotti; J. More; D. E. Manolopoulos. **i-PI: A Python interface for ab initio path integral molecular dynamics simulations**. *Comput. Phys. Commun. 185, 1019–1026* (2014). DOI: [10.1016/j.cpc.2013.10.027](https://doi.org/10.1016/j.cpc.2013.10.027). [Source / author version](https://arxiv.org/abs/1402.1045).

Role: software / PIMD formulation.

### pd2024

Y. Kataoka; J. Haruyama; O. Sugino; M. Shiga. **Predictive evaluation of hydrogen diffusion coefficient on Pd(111) surface by path integral simulations using neural network potential**. *Phys. Rev. Research 6, 043224* (2024). DOI: [10.1103/PhysRevResearch.6.043224](https://doi.org/10.1103/PhysRevResearch.6.043224). [Source / author version](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.6.043224).

Role: application / estimator convention.

### ni2012

Y. V. Suleimanov. **Surface Diffusion of Hydrogen on Ni(100) from Ring Polymer Molecular Dynamics**. *J. Phys. Chem. C 116, 11141–11153* (2012). DOI: [10.1021/jp302453z](https://doi.org/10.1021/jp302453z). [Source / author version](https://pubs.acs.org/doi/10.1021/jp302453z).

Role: application.

### bazro2008

Q. Zhang; G. Wahnström; M. E. Björketun; S. Gao; E. Wang. **Path Integral Treatment of Proton Transport Processes in BaZrO3**. *Phys. Rev. Lett. 101, 215902* (2008). DOI: [10.1103/PhysRevLett.101.215902](https://doi.org/10.1103/PhysRevLett.101.215902).

Role: application; centroid-TST, not a full RPMD recrossing reproduction.

### sds2025

C. Li; L. Zhang; B. Jiang; H. Guo. **Unified ring polymer molecular dynamics rate calculations for reactions with separable and non-separable reactants**. *J. Chem. Phys. 163, 024105* (2025). DOI: [10.1063/5.0273941](https://doi.org/10.1063/5.0273941). [Source / author version](https://arxiv.org/abs/2503.22964).

Role: method.

### nialh1995

J. E. Angelo; N. R. Moody; M. I. Baskes. **Trapping of hydrogen to lattice defects in nickel**. *Modelling Simul. Mater. Sci. Eng. 3, 289–307* (1995). DOI: [10.1088/0965-0393/3/3/001](https://doi.org/10.1088/0965-0393/3/3/001).

Role: potential model.

### pcet2008

C. Venkataraman; A. V. Soudackov; S. Hammes-Schiffer. **Theoretical Formulation of Nonadiabatic Electrochemical Proton-Coupled Electron Transfer at Metal–Solution Interfaces**. *J. Phys. Chem. C 112, 12386–12397* (2008). DOI: [10.1021/jp802171y](https://doi.org/10.1021/jp802171y). [Source / author version](https://pubs.acs.org/doi/10.1021/jp802171y).

Role: method.

### gold2019

Z. K. Goldsmith; Y. C. Lam; A. V. Soudackov; S. Hammes-Schiffer. **Proton Discharge on a Gold Electrode from Triethylammonium in Acetonitrile: Theoretical Modeling of Potential-Dependent Kinetic Isotope Effects**. *J. Am. Chem. Soc. 141, 1084–1090* (2019). DOI: [10.1021/jacs.8b11826](https://doi.org/10.1021/jacs.8b11826).

Role: application; broader model than local workflow.

### pypcet2025

P. Hutchison; K. Cui; J. Zhong; S. Hammes-Schiffer. **Tutorial on computing nonadiabatic proton-coupled electron transfer rate constants**. *J. Chem. Phys. 163, 091501* (2025). DOI: [10.1063/5.0284337](https://doi.org/10.1063/5.0284337). [Source / author version](https://pmc.ncbi.nlm.nih.gov/articles/PMC12646331/).

Role: software / comparison; not original 2008 electrode theory.

### gc2025

M. Sun; B. Jin; X. Yang; S. Xu. **Probing nuclear quantum effects in electrocatalysis via a machine-learning enhanced grand canonical constant potential approach**. *Nature Communications 16, 3600* (2025). DOI: [10.1038/s41467-025-58871-7](https://doi.org/10.1038/s41467-025-58871-7). [Source / author version](https://pmc.ncbi.nlm.nih.gov/articles/PMC12000419/).

Role: related grand-canonical algorithm / application.

### efrpmd2024

R.-H. Bi; W. Dou. **Electronic friction near metal surface: Incorporating nuclear quantum effect with ring polymer molecular dynamics**. *J. Chem. Phys. 160, 074110* (2024). DOI: [10.1063/5.0187646](https://doi.org/10.1063/5.0187646). [Source / author version](https://arxiv.org/abs/2311.08779).

Role: method.

### hmc1987

S. Duane; A. D. Kennedy; B. J. Pendleton; D. Roweth. **Hybrid Monte Carlo**. *Physics Letters B 195, 216–222* (1987). DOI: [10.1016/0370-2693(87)91197-X](https://doi.org/10.1016/0370-2693(87)91197-X). [Source / author version](https://www.sciencedirect.com/science/article/pii/037026938791197X).

Role: method.

### hastings1970

W. K. Hastings. **Monte Carlo sampling methods using Markov chains and their applications**. *Biometrika 57, 97–109* (1970). DOI: [10.1093/biomet/57.1.97](https://doi.org/10.1093/biomet/57.1.97).

Role: method.

### pyscf2018

Q. Sun et al.. **PySCF: the Python-based simulations of chemistry framework**. *WIREs Comput. Mol. Sci. 8, e1340* (2018). DOI: [10.1002/wcms.1340](https://doi.org/10.1002/wcms.1340). [Source / author version](https://arxiv.org/abs/1701.08223).

Role: software.

### hirshfeld1977

F. L. Hirshfeld. **Bonded-atom fragments for describing molecular charge densities**. *Theoretica Chimica Acta 44, 129–138* (1977). DOI: [10.1007/BF00549096](https://doi.org/10.1007/BF00549096).

Role: method.

### mayer1983

I. Mayer. **Charge, bond order and valence in the AB initio SCF theory**. *Chemical Physics Letters 97, 270–274* (1983). DOI: [10.1016/0009-2614(83)80005-0](https://doi.org/10.1016/0009-2614(83)80005-0).

Role: method.

### qmdff2014

S. Grimme. **A General Quantum Mechanically Derived Force Field (QMDFF) for Molecules and Condensed Phase Simulations**. *J. Chem. Theory Comput. 10, 4497–4514* (2014). DOI: [10.1021/ct500573f](https://doi.org/10.1021/ct500573f). [Source / author version](https://pubs.acs.org/doi/10.1021/ct500573f).

Role: method.

### deevb2015

B. Hartke; S. Grimme. **Reactive force fields made simple**. *Phys. Chem. Chem. Phys. 17, 16715–16718* (2015). DOI: [10.1039/C5CP02580J](https://doi.org/10.1039/C5CP02580J). [Source / author version](https://pubs.rsc.org/en/content/articlehtml/2015/cp/c5cp02580j).

Role: method.

### treq2019

J. Steffen. **A new class of reaction path based potential energy surfaces enabling accurate black box chemical rate constant calculations**. *J. Chem. Phys. 150, 154105* (2019). DOI: [10.1063/1.5092589](https://doi.org/10.1063/1.5092589).

Role: method.

### caracal2023

J. Steffen. **Caracal: A Versatile Ring Polymer Molecular Dynamics Simulation Package**. *J. Chem. Theory Comput. 19, 5334–5355* (2023). DOI: [10.1021/acs.jctc.3c00568](https://doi.org/10.1021/acs.jctc.3c00568). [Source / author version](https://pubs.acs.org/doi/10.1021/acs.jctc.3c00568).

Role: software / upstream PES implementation.

### pipnn2013

B. Jiang; H. Guo. **Permutation invariant polynomial neural network approach to fitting potential energy surfaces**. *J. Chem. Phys. 139, 054112* (2013). DOI: [10.1063/1.4817187](https://doi.org/10.1063/1.4817187).

Role: method.

### jaxpip2026

J. Li; K. Song; H. Guo; J. Li. **Efficient Tensorized Evaluation of Permutation Invariant Polynomials for Representing Potential Energy Surfaces**. *J. Chem. Theory Comput. 22, 9940–9953* (2026). DOI: [10.1021/acs.jctc.6c01659](https://doi.org/10.1021/acs.jctc.6c01659). [Source / author version](https://pubs.acs.org/doi/10.1021/acs.jctc.6c01659).

Role: method / software.

### committee2020

C. Schran; K. Brezina; O. Marsalek. **Committee neural network potentials control generalization errors and enable active learning**. *J. Chem. Phys. 153, 104105* (2020). DOI: [10.1063/5.0016004](https://doi.org/10.1063/5.0016004).

Role: method.

### delta2015

R. Ramakrishnan; P. O. Dral; M. Rupp; O. A. von Lilienfeld. **Big Data Meets Quantum Chemistry Approximations: The Δ-Machine Learning Approach**. *J. Chem. Theory Comput. 11, 2087–2096* (2015). DOI: [10.1021/acs.jctc.5b00099](https://doi.org/10.1021/acs.jctc.5b00099). [Source / author version](https://pubs.acs.org/doi/10.1021/acs.jctc.5b00099).

Role: method.

### ase2017

A. H. Larsen et al.. **The atomic simulation environment—a Python library for working with atoms**. *J. Phys.: Condens. Matter 29, 273002* (2017). DOI: [10.1088/1361-648X/aa680e](https://doi.org/10.1088/1361-648X/aa680e).

Role: software.

### openmm2017

P. Eastman et al.. **OpenMM 7: Rapid development of high performance algorithms for molecular dynamics**. *PLoS Computational Biology 13, e1005659* (2017). DOI: [10.1371/journal.pcbi.1005659](https://doi.org/10.1371/journal.pcbi.1005659). [Source / author version](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005659).

Role: software.

### lammps1995

S. Plimpton. **Fast Parallel Algorithms for Short-Range Molecular Dynamics**. *J. Comput. Phys. 117, 1–19* (1995). DOI: [10.1006/jcph.1995.1039](https://doi.org/10.1006/jcph.1995.1039). [Source / author version](https://www.sciencedirect.com/science/article/pii/S002199918571039X).

Role: software / parallel algorithm.

### torch2019

A. Paszke et al.. **PyTorch: An Imperative Style, High-Performance Deep Learning Library**. *NeurIPS 32* (2019). DOI: [10.48550/arXiv.1912.01703](https://doi.org/10.48550/arXiv.1912.01703). [Source / author version](https://arxiv.org/abs/1912.01703).

Role: software; DOI is the author preprint.

### mace2022

I. Batatia; D. P. Kovács; G. N. C. Simm; C. Ortner; G. Csányi. **MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields**. *NeurIPS 35* (2022). DOI: [10.48550/arXiv.2206.07697](https://doi.org/10.48550/arXiv.2206.07697). [Source / author version](https://proceedings.neurips.cc/paper_files/paper/2022/hash/4a36c3c51af11ed9f34615b81edb5bbc-Abstract-Conference.html).

Role: external model; DOI is the author preprint.

### nequip2022

S. Batzner et al.. **E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials**. *Nature Communications 13, 2453* (2022). DOI: [10.1038/s41467-022-29939-5](https://doi.org/10.1038/s41467-022-29939-5). [Source / author version](https://www.nature.com/articles/s41467-022-29939-5).

Role: external model.

### deepmd2018

L. Zhang; J. Han; H. Wang; R. Car; W. E. **Deep Potential Molecular Dynamics: A Scalable Model with the Accuracy of Quantum Mechanics**. *Phys. Rev. Lett. 120, 143001* (2018). DOI: [10.1103/PhysRevLett.120.143001](https://doi.org/10.1103/PhysRevLett.120.143001). [Source / author version](https://arxiv.org/abs/1707.09571).

Role: external model.

### uma2025

B. M. Wood et al.. **UMA: A Family of Universal Models for Atoms**. *arXiv:2506.23971 (v1 2025; v2 2026)* (2025). [Source / author version](https://arxiv.org/abs/2506.23971).

Role: external model preprint; cite the selected checkpoint and manuscript version.

### gfn12017

S. Grimme; C. Bannwarth; P. Shushkov. **A Robust and Accurate Tight-Binding Quantum Chemical Method for Structures, Vibrational Frequencies, and Noncovalent Interactions of Large Molecular Systems Parametrized for All spd-Block Elements (Z = 1–86)**. *J. Chem. Theory Comput. 13, 1989–2009* (2017). DOI: [10.1021/acs.jctc.7b00118](https://doi.org/10.1021/acs.jctc.7b00118).

Role: external electronic method.

### gfn22019

C. Bannwarth; S. Ehlert; S. Grimme. **GFN2-xTB—An Accurate and Broadly Parametrized Self-Consistent Tight-Binding Quantum Chemical Method with Multipole Electrostatics and Density-Dependent Dispersion Contributions**. *J. Chem. Theory Comput. 15, 1652–1671* (2019). DOI: [10.1021/acs.jctc.8b01176](https://doi.org/10.1021/acs.jctc.8b01176).

Role: external electronic method.

### ipea2017

V. Ásgeirsson; C. A. Bauer; S. Grimme. **Quantum chemical calculation of electron ionization mass spectra for general organic and inorganic molecules**. *Chemical Science 8, 4879–4895* (2017). DOI: [10.1039/C7SC00601B](https://doi.org/10.1039/C7SC00601B). [Source / author version](https://pubs.rsc.org/en/content/articlehtml/2017/sc/c7sc00601b).

Role: IPEA-xTB origin; PyRPMD does not implement QCEIMS.

### h5md2014

P. de Buyl; P. H. Colberg; F. Höfling. **H5MD: A structured, efficient, and portable file format for molecular data**. *Comput. Phys. Commun. 185, 1546–1553* (2014). DOI: [10.1016/j.cpc.2014.01.018](https://doi.org/10.1016/j.cpc.2014.01.018).

Role: file format.

### bennett1977

C. H. Bennett. **Molecular Dynamics and Transition State Theory: The Simulation of Infrequent Events**. *Algorithms for Chemical Computations, ACS Symposium Series 46, 63–97* (1977). DOI: [10.1021/bk-1977-0046.ch004](https://doi.org/10.1021/bk-1977-0046.ch004).

Role: Original classical reactive-flux / transition-state factorization; book chapter.

### chandler1978

D. Chandler. **Statistical mechanics of isomerization dynamics in liquids and the transition state approximation**. *J. Chem. Phys. 68, 2959–2970* (1978). DOI: [10.1063/1.436049](https://doi.org/10.1063/1.436049).

Role: Original classical reactive-flux and plateau theory.

### bkmp21996

A. I. Boothroyd; W. J. Keogh; P. G. Martin; M. R. Peterson. **A refined H3 potential energy surface**. *J. Chem. Phys. 104, 7139–7152* (1996). DOI: [10.1063/1.471430](https://doi.org/10.1063/1.471430). [Source / author version](https://www.cita.utoronto.ca/~boothroy/bkmp2.html).

Role: Original BKMP2 model and author source.

### morse1929

P. M. Morse. **Diatomic Molecules According to the Wave Mechanics. II. Vibrational Levels**. *Phys. Rev. 34, 57–64* (1929). DOI: [10.1103/PhysRev.34.57](https://doi.org/10.1103/PhysRev.34.57).

Role: Original Morse potential and analytic vibrational levels.

### pip2010

Z. Xie; J. M. Bowman. **Permutationally Invariant Polynomial Basis for Molecular Energy Surface Fitting via Monomial Symmetrization**. *J. Chem. Theory Comput. 6, 26–34 (online 2009)* (2010). DOI: [10.1021/ct9004917](https://doi.org/10.1021/ct9004917).

Role: Original monomial-symmetrization PIP basis method.
