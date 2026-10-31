# Implementation and user-guide audit — 2026-10-31

## Result and review scope

This review found reproducible boundary, persistence and documentation defects,
plus two focused opportunities to avoid repeated work. The accompanying change
fixes those defects, preserves the existing scientific conventions, and records
larger migrations separately below.

**Reviewed baseline:** [`c4fa743f552c1d71db35fab791a2e63409bc1c3e`](https://github.com/chen-yu-hao/PyRPMD/commit/c4fa743f552c1d71db35fab791a2e63409bc1c3e)
on `main`. The baseline has 1,084 tracked files. Implementation review covered
110 source/bridge files under `src/pyrpmd` (about 40,354 lines) and 13 native
source/header files (about 2,863 lines), together with the test, example,
packaging and documentation entry points. This is a repository-wide engineering
review using source tracing, independent reproductions and regression tests;
it is not a new convergence study for every physical method or external PES.

| Area | What was inspected |
| --- | --- |
| Public interfaces | CLI dispatch, TOML/Python `JobSpec`, units, defaults, override/preflight behavior, package exports |
| Numerical core | Nuclear states, Cayley/normal-mode/Verlet maps, scalar/batched/MTS propagation, thermostat/barostat transactions, native observations |
| Potentials and geometry | Physical result contracts, C ABI v1/v2 and native binding lifetime, Python providers, cache/RPC paths, fixed atoms, periodic geometry and CVs |
| Scientific extensions | Initial-state boundaries, complex electronic vs real nuclear data, nonadiabatic trial states, path-action and quantum-statistics interfaces |
| Workflows | FES, gas rate/recrossing, dRPMD, adaptive umbrellas, metadynamics, OPES, replica exchange, materials/native PIMD, surface/site rates, networks, PCET and studies |
| Data and analysis | Run identity, JSON/NPZ publication, checkpoint checksums and semantic checks, trajectory chunks, projection, saved-rate adapters, uncertainty and report provenance |
| Execution and distribution | Process/MPI/root-stage organization, provider cleanup, thread limits, build configuration, source/wheel contents, test and documentation workflows |
| Usage guides | Both READMEs, paired manuals, feature specifications and examples; 136 Markdown files checked for local/own-repository link targets |

Existing open work was checked before editing. [PR #44](https://github.com/chen-yu-hao/PyRPMD/pull/44)
already owns H3 state-preparation numerical/analysis hardening; its changed files
are excluded from this implementation. [PR #39](https://github.com/chen-yu-hao/PyRPMD/pull/39)
documents unresolved physical normalization work for the 2D QTST comparison.
Those scientific questions remain attached to their existing reviews.

## To-do list implemented in this change

P1 denotes a result-integrity or silently accepted numerical-data defect; P2
denotes a reliability, performance or user-contract correction.

| Done | ID | Priority | Finding and resulting behavior | Main implementation |
| --- | --- | --- | --- | --- |
| [x] | A01 | P1 | Nuclear coordinates, momenta, masses and native physical arrays could silently discard imaginary components during float conversion. Validate real numeric data first; preserve complex electronic amplitudes and explicit boolean PBC/cell masks. Validate real scalar beta, dt and friction controls too. | [`dynamics.py`](../../src/pyrpmd/dynamics.py), [`batched.py`](../../src/pyrpmd/batched.py), [`native_ensemble.py`](../../src/pyrpmd/native_ensemble.py) |
| [x] | A02 | P1 | Scalar propagation could commit an infinite time or wrap a NumPy integer step. Normalize state clocks, validate live clocks, and check the next/requested time before forces, momentum resampling or state commits. | [`dynamics.py`](../../src/pyrpmd/dynamics.py) |
| [x] | A03 | P1 | Finite native q/p did not imply finite energy, pressure or squared-radius reductions. Reject nonfinite defined observations inside the block transaction; restore state/RNG and discard partial streams, including late failures. Reject final-time overflow even when no observation is scheduled. | [`thermal.hpp`](../../native/thermal.hpp) |
| [x] | A04 | P2 | Malformed nested input tables leaked `AttributeError`, and invalid native policies failed after creating run output. Validate them during resolution. Reject nonfinite/invalid timesteps, overflowing duration conversion and negative durations before rounding. | [`input.py`](../../src/pyrpmd/input.py), [`units.py`](../../src/pyrpmd/units.py) |
| [x] | A05 | P2 | FES and dRPMD JSON writers reused predictable temporary paths and had inconsistent failure cleanup. Share one unique-temporary, flush/fsync, replace-and-cleanup primitive while retaining each format's conversion policy. | [`io.py`](../../src/pyrpmd/io.py), [`fes.py`](../../src/pyrpmd/workflows/fes.py), [`drpmd.py`](../../src/pyrpmd/workflows/drpmd.py) |
| [x] | A06 | P1 | Saved-rate postprocessing bypassed an NPZ file's checksum envelope. Use `load_checkpoint` for all imported NPZ payloads in this adapter; retain legacy unsealed NPZ support. | [`rate_io.py`](../../src/pyrpmd/analysis/rate_io.py) |
| [x] | A07 | P2 | `project_run(..., axes=0)` normalized the axis for calculation but iterated the original integer while writing outputs, leaving an incomplete directory. Use the normalized retained-axis tuple for every output. | [`projection.py`](../../src/pyrpmd/analysis/projection.py) |
| [x] | A08 | P2 | Completed recrossing children constructed a PES before reading their already-complete checkpoint. Verify and return saved children before acquiring the provider, for both scalar and batched replay. | [`rate.py`](../../src/pyrpmd/workflows/rate.py) |
| [x] | A09 | P2 | NumPy free-ring propagation rebuilt the same mode coefficients at every step. Cache the full and half maps per integrator context without changing FFTs, arithmetic ordering or random draws. | [`dynamics.py`](../../src/pyrpmd/dynamics.py) |
| [x] | A10 | P2 | User guides disagreed with implemented commands, defaults and method boundaries. Synchronize the two editions and point to canonical extension guides. | READMEs and [`manual`](../manual)/[`Chinese manual`](../zh/manual) |

### Reproductions and compatibility decisions

**Real nuclear data and clocks (A01–A02).** A complex array such as
`q = np.ones((1, 1, 3), dtype=complex) * 1j` became zero-valued real coordinates
in the old constructor. A scalar trajectory at `time=1e308`, `dt=1e308` could
advance to `time=inf`. These are rejected before advancing physical state.
NumPy integer clocks are converted to Python integers, avoiding accidental
signed-integer wrap. Deterministic zero and negative timesteps remain supported
for identity/reversibility checks; thermostatted sampling requires positive dt.

The stricter array contract intentionally rejects boolean, string and object
arrays as physical q/p/mass data, as well as complex arrays. Ordinary integer
and floating arrays remain accepted. Nonadiabatic electronic amplitudes remain
complex. Physical boolean masks retain their separate mask contract.

**Native observations (A03).** A free trajectory with finite momentum
`p=1e200` previously returned an infinite kinetic energy/Hamiltonian. A very
weak two-bead ring can also have finite energy but an overflowing squared
radius. Regression tests cover failure on the first observation and after two
valid rows, for memory and streamed output, and compare the complete state
digest including RNG and cached forces. Molecular volume and pressure remain
intentionally undefined (`NaN`) when no cell exists; this allowance applies
only to those two columns.

**Input preflight (A04).** Nested `potential.params`, optional initializer
parameters, thermostat parameters, outcome parameters and bond-cutoff tables
are checked before workflow initialization. The existing Python
`potential.initializer_params=None` default is preserved. Native policies accept
booleans, `None` in Python and `"required"`; arbitrary strings and numeric
boolean lookalikes fail explicitly. Time conversion reports `ValueError`
instead of leaking overflow exceptions, and a tiny negative duration cannot
round into an accepted zero-step request.

**Atomic metadata (A05).** Two synchronized writers previously collided on
the same FES or per-process dRPMD temporary filename. Tests now verify that
each writer publishes a complete document and removes its own temporary after
failure. This is atomic file replacement, not permission to run two simulations
against one output directory. Generic/dRPMD JSON still reject nonfinite values;
FES still represents unavailable estimates as `null`. Key ordering, Unicode
escaping and newline policies are preserved. All new durability work precedes
replacement, preserving the native-PIMD journal's commit/rollback contract.
The shared writer uses private temporary-file permissions (`0600` on POSIX);
FES/dRPMD previously created files according to the process umask. This aligns
them with the existing generic writer and is a file-permission compatibility
change for workflows that rely on group-readable JSON.

**Checked saved data (A06–A07).** A sealed FES edited while retaining its old
checksum envelope previously passed through the saved-rate adapter. The shared
loader now rejects it before replacing an existing analysis. Tests also corrupt
PMF/recrossing members not consumed directly by the estimator, ensuring that
the entire sealed payload is verified. Ordinary legacy NPZ files remain
readable; their lack of an integrity envelope is not upgraded into a checksum
guarantee. Projection tests read back NPZ, CSV and JSON and independently
integrate the Boltzmann density with bin widths, including reordered axes.

**Avoiding repeated work (A08–A09).** Completed-child replay retains all
existing physics-fingerprint, PMF-method and dividing-surface checks and makes
zero provider-construction calls in the regression tests. Free-ring coefficient
caching is bounded to the full/half maps for the current timestep and scheme.
Independent dense Cayley, matrix-exponential and matrix-square-root checks cover
multiple bead counts and positive, negative and zero dt. The benchmark retains
the original formula as a reference and requires exactly equal q/p elements after every
comparison. It measures a kernel; expensive PES evaluation can dominate a full
chemistry trajectory.

### Measured coefficient-cache benefit

The recorded microbenchmark uses two integrator schemes, full/half maps, four
ring sizes, 2,000 steps per path and five alternating repeats. All 16
configurations passed exact elementwise endpoint comparison with
`numpy.testing.assert_array_equal`; this is not a byte-level signed-zero check.
The table summarizes the four map configurations at each size. A ratio greater
than one means the cached kernel was faster in this run.

| Beads / atoms | Median speedup | Range across the four maps |
| --- | --- | --- |
| 1 / 1 | 1.544× | 1.211–1.847× |
| 8 / 3 | 1.315× | 1.110–1.446× |
| 32 / 3 | 1.182× | 1.059–1.237× |
| 64 / 50 | 1.096× | 0.847–1.169× |

The median across all 16 configurations is 1.205×. Timing noise was substantial:
the large Cayley half-map measured 0.847× in the original run. One targeted
follow-up (1,000 steps, nine repeats) measured 1.013× with exact elementwise
equality, but also contained a 2.431-second cached repeat against a 0.514-second
median. Both timing records are retained. These measurements support a benefit
for the small/moderate configurations; they do not establish stable acceleration
for large arrays or an end-to-end chemistry speedup.

## Guide corrections

| Topic | Correct documented behavior |
| --- | --- |
| Task and command discovery | Eleven workflows; all fourteen `pyrpmd` subcommands indexed, including OPES, reports, rate postprocessing, projection and studies |
| FES dimensionality | Umbrella `fes` supports one or two CVs. Metadynamics supports one to three. No new three-dimensional umbrella implementation is implied. |
| Production defaults | Thermal pilot production defaults to 200 fs, except metadynamics and enabled adaptive umbrellas at 1000 fs. |
| Production preset | Requires selected explicit fields and task-specific controls; nested defaults may remain. Resolved input and convergence checks determine the actual production protocol. |
| Sampling | FES supports Andersen or PILE-L; REUS has its own fixed-window/checkpoint rules. Metadynamics also supports NHC. |
| Materials engines | Distinguish the default workflow engine from opt-in native PIMD, including thermostat/barostat limits and journal/chunk output layout. |
| Periodic providers | Cell-aware Torch and C ABI v2 are supported under their declared contracts; C ABI v1 and RPMDrate/f2py remain molecular. |
| Energy estimators | Fixed atoms remain in the PES; physical kinetic estimators count mobile atoms. NPT still requires all atoms mobile. |
| Diffusion | Ordinary production is unthermostatted; optional EF-RPMD exchanges energy with a physical electronic bath. |
| Site rates | Document the supported nonlinear centroid-coordinate mass-metric route and link its method restrictions. |
| Plotting installation | `.[plots]` supplies report rendering; `.[analysis]` additionally installs PyMBAR. |

## Follow-up to-do list

These items need separate migration, measurement or scientific evidence. They
are intentionally open rather than being marked complete by this maintenance PR.

- [ ] **B01 — P2: Bound accumulated-history checkpoint I/O.** FES,
  metadynamics, OPES, rate parents and replica exchange repeatedly serialize
  their full history. At fixed cadence, cumulative serialized history grows
  quadratically with run length. Evaluate the existing `TrajectoryStream`
  design with backward-compatible checkpoints, exact restart/RNG comparisons,
  interrupted-write recovery and scaling measurements.
- [ ] **B02 — P2: Align older semantic checkpoint validators.** FES/parent
  completion paths should check sample counts, array shapes, finite records
  and step/time relationships as explicitly as newer OPES/materials paths.
  Checksums establish stored-byte integrity; they do not establish scientific
  consistency. Preserve or explicitly migrate legacy payloads.
- [ ] **B03 — P2: Publish related analysis files as one generation.** FES
  NPZ/HDF5/CSV/JSON outputs are currently published sequentially. Stage a whole
  generation or add a commit marker so an interrupted reanalysis cannot expose
  files from different generations. Atomic JSON alone does not solve this.
- [ ] **B04 — P3: Extract cohesive shared validation/I/O utilities.** Other
  workflows import helpers from the large FES module, and numeric/count/clock
  validation is duplicated. Move one contract at a time with compatibility
  aliases and unchanged fingerprints, rather than changing all call sites at
  once.
- [ ] **B05 — P3: Maintain capability metadata once.** Task lists, engine
  capabilities and documentation summaries currently live in several places.
  A shared registry could drive `info` and guide indexes, including the native
  NPT modes absent from the current general materials summary.
- [ ] **B06 — P3: Reduce guide duplication and automate example classification.**
  Keep README entry points concise and detailed contracts in canonical guides.
  Maintain a list of lightweight JobSpec examples distinct from native-ensemble,
  PES-generation and provider-module schemas. Pair English/Chinese entry points
  semantically, beyond matching chapter filenames.
- [ ] **B07 — P3: Measure redundant copies at checkpoint read boundaries.**
  The verified payload view already returns owned arrays. Remove additional
  full copies only where ownership is proven and peak-memory measurements
  justify it; preserve copies for slices/views and mutable states.
- [ ] **B08 — P3: Extend completed-task fast paths.** Some FES,
  metadynamics and site-rate checks still construct providers before discovering
  completion. Separate pure validation first so startup savings never skip
  state/fixed-coordinate checks.
- [ ] **B09 — P3: Profile large non-power-of-two native rings.** The native
  normal-mode path has a direct quadratic fallback for those sizes. Measure
  realistic workloads before adding an alternative FFT plan or dispatch rule.
- [ ] **B10 — P3: Simplify native bindings and constants deliberately.**
  Consolidate legacy/canonical native PES descriptors while preserving plugin
  leases. Reuse the canonical kcal/kJ conversion constants in Python adapters
  with an explicit roundoff/provenance note. Reformat dense native code in a
  separate behavior-preserving review.
- [ ] **B11 — Existing scientific work:** finish the independent H3
  preparation review in PR #44 and the physical normalization/convergence work
  recorded in PR #39 before expanding the corresponding scientific claims.

## Validation and reproduction

Validation receipts and the coefficient microbenchmark are stored in the
adjacent [`implementation-guide-audit-2026-10-31`](implementation-guide-audit-2026-10-31)
directory. Final counts and commands are recorded in `validation.json`.

| Gate | Result |
| --- | --- |
| Original implementation, native kernels enabled | 3,804 passed; 115 skipped; 264 subtests passed; zero failures/errors |
| Final implementation, rebuilt native kernels and real BKMP2 | **4,056 passed; 96 skipped; 272 subtests passed; zero failures/errors** |
| English/Chinese HTML and single-page manuals | Four strict `sphinx -W --keep-going` builds passed; integrated HTML builds repeated after source integration |
| Guide contracts | All 14 command names indexed; 29 TOML/INP and 9 Python JobSpec examples parsed; 173 repository Markdown files checked with zero missing local/own-repository file targets |
| Installed distribution | Sdist built; native wheel built from it and installed outside the checkout; both extensions present; 123 runtime/native source hashes and 110 wheel package-source hashes verified |
| Installed workflows | Five example families plus P=3 dRPMD with normal-mode BAB and velocity-Verlet completed and resumed with identical summaries; native numerical parity gates passed |
| Coefficient microbenchmark | All 16 configurations passed exact elementwise q/p comparison; timing limits and the slower observation are reported above |

The additional passes combine new regression cases with previously skipped
real-BKMP2 checks. Suite wall times are not used as a performance comparison.
The final 96 skips require optional providers, source-pinned models, native ONNX,
or other unavailable dependencies; local MPI integration was not exercised.
The BKMP2 build records its pinned source hash and compiler. Neither the
downloaded external PES source nor its binary is redistributed by this change.
Package verification covers Linux x86_64 / CPython 3.12, and its recorded runtime
hashes match the final worktree; later audit-receipt-only edits can postdate the
package build. Example smoke checks retain their `converged: false` diagnostics.

```bash
python -m pip install -e '.[test,docs,ase]' build
python benchmarks/h3/build_pes.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m sphinx -W --keep-going -b html docs/zh docs/_build/html/zh
python -m sphinx -W --keep-going -b singlehtml docs docs/_build/singlehtml
python -m sphinx -W --keep-going -b singlehtml docs/zh docs/_build/singlehtml/zh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python benchmarks/efficiency/free_ring_coefficients.py \
  --steps 2000 --repeats 5 --output free-ring-coefficients.json
python -m build
git diff --check
```

Source/wheel installation should be exercised outside the checkout with
`tests/package_smoke.py`, as in the existing package workflow. MPI, optional
external electronic-structure/ML providers, native ONNX and real-model data
require their declared dependencies; a skipped check is not a passing check.

Scientific invariants retained by this maintenance work include the distinction
between the ring Hamiltonian and thermodynamic energy estimators, parent-level
recrossing resampling, source-validation-based rate withholding, distinct
CV-density/flux/basin normalizations, replica-aware transport correlations,
and source provenance. Successful examples and microbenchmarks establish
implementation behavior, not physical convergence for a named reaction.
