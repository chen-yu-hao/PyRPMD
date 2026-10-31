# PySCF reference data and Caracal PES generation

`python -m pyrpmd.pesgen` implements an **optional offline molecular workflow**:

```text
endpoint structures + supplied path
 -> PySCF energy / gradient / Hessian / Hirshfeld / Mayer data
 -> original Caracal QMDFF generator and dE-EVB optimizer or TREQ setup
 -> source-pinned native PES bundle
 -> existing PyRPMD JobSpec and C++ trajectory loop
```

It does not implement a generic harmonic approximation and call it QMDFF or TREQ.
The Fortran energy expressions and fitting tools come from Caracal commit
`7f95ce9818b7f6d81d7cb3b634f360373c090820`. Core Caracal is MIT-licensed; its
bundled dependencies retain their own notices. PySCF and Caracal are not imported
or installed by ordinary PyRPMD runs. This feature is independent of the proposed
on-the-fly PySCF adapter: here electronic structure is evaluated **during fitting**, not at every dynamics step.

## Installation and a real water example

The first implementation requires Linux, a C++17 compiler, GNU Fortran/OpenMPI,
BLAS/LAPACK, FFTW headers, Python development/linker configuration, and optional
PySCF. For example, on a Debian/Ubuntu host:

```bash
sudo apt-get install gfortran g++ libopenmpi-dev openmpi-bin libfftw3-dev liblapack-dev python3-dev
python -m pip install -e '.[pesgen,test]'
git clone https://github.com/Trebonius91/Caracal.git external/Caracal
git -C external/Caracal checkout 7f95ce9818b7f6d81d7cb3b634f360373c090820
python -m pyrpmd.pesgen build-native external/Caracal runs/caracal-native
python -m pyrpmd.pesgen generate examples/pesgen/water.toml --output runs/water-pes
python -m pyrpmd.pesgen generate examples/pesgen/water.toml --output runs/water-pes --resume
```

Build destinations must be new and outside the source checkout. The build copies
and hashes upstream sources, applies an archived safety patch, and records the
compiler commands and executable hashes. MPI is a build/runtime dependency of
upstream tools; `evbopt.x` initializes MPI even for a serial fit. Build serially:
upstream Fortran module dependencies are not safe under `make -j`.

The water example optimizes a starting geometry with RHF/STO-3G, calculates the
full analytic Hessian and populations, generates an actual QMDFF, and validates
that its native adapter can evaluate the fitted geometry. **STO-3G is a small
integration test, not a recommended reference level for reaction kinetics.**

## Reference conventions

`PySCFSettings` supports molecular RHF, UHF, RKS and UKS, an explicit named
all-electron basis, charge and `spin = N_alpha - N_beta`. Restricted methods
require spin zero. DFT requires an explicit `xc`; `grid_level` controls its
quadrature. The fitted workflow initially accepts neutral molecules only;
the standalone reference API also accepts charges.

Internal positions are bohr, energies hartree, positive gradients hartree/bohr,
and Cartesian Hessians hartree/bohr² in atom-major x/y/z order. XYZ input is
angstrom. The Hessian is analytic unless `hessian="finite_difference"` is
explicitly selected; that alternative differentiates analytic gradients with
`hessian_step` in bohr. Unsupported analytic Hessians fail rather than silently
changing methods. Only converged, finite SCF results are committed. SCF stability
and electronic-state continuity are **not** established by cold starts; provenance
records these limitations, and unrestricted references include spin diagnostics.

Hirshfeld populations use neutral spherical HF atomic densities in the same AO
basis as the molecule, with an independently controlled `population_grid_level`.
Both density integration and total charge are checked. This is an explicit
stockholder convention, **not asserted numerically identical to ORCA's atomic
reference tables**. Mayer bond orders include the unrestricted spin-density term.
Caracal applies its CM5 correction and empirical charge scaling once; we supply
Hirshfeld, not already-CM5, charges. Files using Caracal's ORCA-format readers
state explicitly that the numerical data came from PySCF, not an ORCA calculation.
The hashed JSON reference is retained alongside the interchange files.

## dE-EVB and TREQ

Use `method="de_evb"` or `method="treq"`, two ordered endpoint XYZ files, and a
multi-frame `path` with the same atom ordering, charge and spin. The endpoints
must pass a gradient/negative-curvature check; `optimize_endpoints=true` runs a
minimum optimization. **This is not a transition-state optimizer or IRC finder.**
The path may be supplied by a separate quantum-chemistry or reaction-path tool.

For dE-EVB, `[fit] coupling="sp"` (default) or `"1g"` selects the original Caracal
coupling fit. Absolute endpoint energies determine QMDFF shifts. The path's
energies and generated QMDFFs are passed to `evbopt.x`; fitted `evb_pars.dat` and
logs remain available. No post-fit offset is applied to hide training error.

TREQ additionally requires `coordinates` containing exactly `3N-6` nonredundant
internal coordinates: a bond is `i j`, an angle `i j k`, a torsion `0 i j k l`,
and an out-of-plane coordinate `1 i j k l`, with **one-based atom indices**.
Linear reference geometries, redundant coordinate sets and automatic coordinate
selection are outside the initial scope. Coordinate count/indices are checked;
that alone does not certify conditioning throughout configuration space.

`hessian_indices` is a sorted unique list of **zero-based path-frame indices**;
by default every frame gets a Hessian. Selected frames must include both ends,
the maximum-energy frame and at least one interior frame. The workflow writes
Caracal's full `grad_hess.dat`. `[fit]` accepts `exp_coefficient=35`,
`mid_total=0.6` and `mid_transition=0.1`; they are model controls, not universal
accuracy settings.

A complete, inexpensive PySCF-to-TREQ example is:

```bash
python examples/pesgen/nh3_inversion.py --native-build runs/caracal-native --output runs/nh3-treq
```

It uses a nine-point constrained NH₃ inversion scan, not an optimized IRC.
`--method de_evb` exercises the other route with the same reference level.

### TREQ force qualification

A check on the pinned CF₃O + CH₄ example found an upstream analytical-force
component inconsistent with finite differences of its own energy. Therefore the
public TREQ workflow currently uses **explicit fourth-order central numerical
energy derivatives inside Fortran**, controlled by `force_step` (default
`1e-4 bohr`). The energy representation is unchanged. This costs `1+12N`
upstream evaluations per structure; it is not advertised as a fast analytic force.
The original analytic mode remains available only through the lower-level bundle
builder for diagnostic comparisons. TREQ results are marked `experimental` and
require explicit pilot opt-in when handed to a dynamics job. Check step-size
convergence and energy conservation for the system being studied.

## Native dynamics and isolation

The bundle contains `libpes.so`, `caracal_worker.x`, source model text, manifests
and third-party notices. The C++ C ABI v1 library embeds the model text and checks
the companion worker's identity at creation. Each context creates one persistent
Fortran process in its own temporary directory. A compiled socket protocol carries
batches; the worker evaluates frames serially. No Python callback, shell command
or process creation occurs per integration step. This isolates Caracal's mutable
Fortran globals, so independent trajectory processes do not share them.

The TREQ path search is global for every geometry, eliminating a previous bead's
local-neighborhood history. The adapter serializes access per context, rejects
inherited contexts after fork, limits transactions, detects worker exits/timeouts,
and reaps workers when contexts close. This first adapter is nonperiodic; it does
not provide stress, virials, GPU kernels or intra-batch bead parallelism. It adds
native-process communication overhead rather than claiming a universal speedup.

## Validation, restart and hand-off

`references/` contains atomic, checksum-verified QM cache entries. Inputs, source
code, QM version and native build identity are fingerprinted. Resume refuses
changed science or corrupted completed artifacts. A file lock prevents two
processes from building into the same run. Failed attempts retain logs; only a
complete fitted/evaluated bundle is marked complete.

`summary.json` separates training errors from optional held-out errors. Supply
independent geometries and explicit bounds to make that check available:

```toml
[validation]
geometries = "heldout.xyz"
energy_rmse_max = 0.001       # hartree; illustrative bound, choose for your problem
force_rmse_max = 0.002        # hartree/bohr
```

Duplicated training geometries are rejected. Passing these user-selected bounds
is not a global transferability or physical-rate validation. Results retain
`validated_physical_accuracy=false` and `converged_rate=false`.

Hand the bundle to an existing complete PyRPMD input, without inventing reaction
coordinates or normalization:

```bash
python -m pyrpmd.pesgen run-job runs/water-pes existing-job.inp --output runs/water-dynamics --allow-unvalidated
```

`--allow-unvalidated` is an explicit **pilot** option required without passing
held-out checks and always for experimental TREQ. Remove it for qualified
nonexperimental jobs. Molecular identity, charge/spin and lack of PBC are checked
by the existing JobSpec and native ABI. The same API is available as
`pyrpmd.pesgen.run_generated_job(...)`; `generate(...)` and `build_native(...)`
provide the offline steps separately.

## Safety patches and references

The pinned source patch accepts a genuinely present all-zero Hirshfeld section,
initializes its accumulator, rejects missing charges, preserves total charge when
scaling CM5 charges, and moves the 12 MB `ff_clean` index workspace from the stack
to an allocated array. No QMDFF, EVB or TREQ energy equation is replaced.

References: Grimme, QMDFF, DOI **10.1021/ct500573f**; Hartke and Grimme, dE-EVB,
DOI **10.1039/C5CP02580J**; Steffen, TREQ, DOI **10.1063/1.5092589**; Steffen,
Caracal, DOI **10.1021/acs.jctc.3c00568**. PySCF analytic derivative APIs are
documented in its [SCF](https://pyscf.org/pyscf_api_docs/pyscf.scf.html) and
[Hessian](https://pyscf.org/pyscf_api_docs/pyscf.hessian.html) references.
