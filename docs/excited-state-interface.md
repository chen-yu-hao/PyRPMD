# Online excited-state electronic-structure interface

PyRPMD's excited-state provider layer connects on-the-fly electronic-structure
calculations to the existing local-adiabatic nonadiabatic APIs. It supplies
state energies, positive nuclear gradients, derivative couplings, adjacent-frame
wavefunction overlaps, state-tracking diagnostics, and an opaque restartable
electronic context. The implementation lives in
`pyrpmd.potentials.excited_state`.

The layer does not select a nonadiabatic dynamics method and does not change the
hopping or mapping equations. Wrap an `ExcitedStateProvider` in
`AdiabaticPythonPotential` and pass it to an existing consumer such as
`AdiabaticRPSHDynamics` or `QDNRPMDDynamics`.

## Contract and units

Every backend solves one geometry with shape `[N, 3]`; the provider batches
these calls over the leading bead/frame dimension. Internal quantities are
strictly atomic units:

| Quantity | Shape | Convention |
| --- | --- | --- |
| Coordinates | `[B,N,3]` | bohr |
| State energies | `[B,S]` | total energies in hartree; `S` includes the ground state |
| Nuclear gradients | `[B,S,N,3]` | positive `dE/dR` in hartree/bohr |
| Derivative couplings | `[B,S,S,N,3]` | `d_ij = <i|grad_R j>` in bohr^-1, anti-Hermitian in state indices |
| Overlaps | `[B,S,S]` | `O_ij = <old_i|new_j>` |

The provider advertises `requires_atomic_units=True`. Consequently,
`AdiabaticPythonPotential` rejects an additional non-atomic unit conversion
instead of silently applying it twice. A backend must never return forces in
place of gradients or excitation energies in place of total state energies.

A minimal analytic example is:

```python
from pyrpmd.adiabatic import AdiabaticPythonPotential
from pyrpmd.potentials.excited_state import (
    AvoidedCrossingBackend,
    ExcitedStateProvider,
)

provider = ExcitedStateProvider(
    AvoidedCrossingBackend(natoms=1),
    symbols=["H"],
    nstates=2,
    coupling_method="analytic",
)
potential = AdiabaticPythonPotential(provider)
```

Close the adapter, or use the provider as a context manager, to release backend
and scratch resources.

## State tracking, phases, and overlap-derived couplings

Adjacent wavefunctions have arbitrary phases and electronic-structure programs
may reorder roots. The provider therefore performs the following operations for
every successful old-to-new frame pair:

1. compute or receive the raw overlap `O_ij = <old_i|new_j>`;
2. reject a retained subspace whose singular values differ too far from one;
3. assign current roots by deterministic maximum absolute overlap;
4. multiply each current ket by a phase that makes its assigned diagonal overlap
   positive real;
5. apply the same permutation and phase transformation to gradients, analytic
   derivative couplings, oscillator strengths, and transition dipoles.

Near-degenerate energy pairs and ambiguous assignments are reported in the
frame diagnostics and message. They are not converted into a false convergence
claim. A singular or poorly connected retained subspace fails explicitly and
usually indicates that the nuclear step must be reduced or more states retained.

If a backend provides analytic derivative couplings, set
`coupling_method="analytic"` and those vectors are used directly. Otherwise the
provider supports two overlap connections after state/phase alignment. With
`U` equal to the polar-unitary overlap,

```text
HST: A = (U - U^H) / 2
NPI: A = log(U)       (principal, anti-Hermitian matrix logarithm)
```

`time_derivative_coupling` returns `A / dt`. `path_derivative_couplings` lifts
`A` to the minimum-norm Cartesian tensor satisfying

```text
sum_A d_ij,A * DeltaR_A = A_ij.
```

Only the component parallel to the sampled displacement is determined by two
frames; perpendicular components are set to zero. This path lift is useful for
short-step path-following tests, but it is not equivalent to a complete analytic
NAC vector for arbitrary momentum rescaling. Production calculations must test
timestep, retained-state, and coupling-convention convergence.

The HST finite-difference expression follows Hammes-Schiffer and Tully,
[J. Chem. Phys. 101, 4657 (1994)](https://doi.org/10.1063/1.467455). Meek and
Levine introduced norm-preserving interpolation for overlap-based TDCs,
[J. Phys. Chem. Lett. 5, 2351 (2014)](https://doi.org/10.1021/jz5009449).
The implemented multistate principal-logarithm form is the generalized NPI
construction of Jain, Alguire, and Subotnik,
[J. Chem. Theory Comput. 12, 5256 (2016)](https://doi.org/10.1021/acs.jctc.6b00673),
after polar/Löwdin orthogonalization of the retained overlap.

## PySCF TDA/TDDFT backend

Install the existing optional dependency group:

```bash
python -m pip install -e '.[pyscf]'
```

`PySCFTDDFTBackend` supports restricted, closed-shell RHF/TDA and RKS/TDA or
RKS/TDDFT with PySCF 2.x. `nstates` includes the ground state. The backend
returns total state energies, PySCF analytic gradients when available,
oscillator strengths, transition dipoles, and adjacent-geometry auxiliary
wavefunction overlaps.

```python
from pyrpmd.adiabatic import AdiabaticPythonPotential
from pyrpmd.potentials.excited_state import ExcitedStateProvider, PySCFTDDFTBackend

backend = PySCFTDDFTBackend(
    symbols=["H", "H"],
    nstates=2,
    basis="sto-3g",
    xc=None,                         # RHF; use an XC string for RKS
    tda=True,
    initial_nac="finite_difference",
    transition_properties=True,
)
provider = ExcitedStateProvider(
    backend,
    symbols=["H", "H"],
    nstates=2,
    coupling_method="npi",
    subspace_tolerance=0.1,
)
potential = AdiabaticPythonPotential(provider)
```

For TDA, overlaps are determinant overlaps of normalized CIS-like singlet
expansions from the PySCF `X` amplitudes. The cross-geometry AO overlap includes
orbital relaxation and moving atom-centered basis functions. For an adjacent
SCF calculation, the previous restricted density is projected from the old AO
basis to the current AO basis with PySCF's basis-projection routine; projection
failure is explicit, and this initial-guess policy is part of the scientific
identity. Full TDDFT uses an explicit auxiliary `X+Y` determinant
representation; that approximation is part of the scientific identity and is
not advertised as an analytic TDDFT NAC. `max_determinant_pairs` is a hard cost
bound: exceeding it fails rather than switching silently to a cheaper overlap
approximation.

With `initial_nac="finite_difference"`, the first frame obtains a full Cartesian
coupling tensor from centered finite differences of state-aligned overlaps. This
costs `6N` additional electronic-structure evaluations. With
`initial_nac="unavailable"`, the first frame has no coupling unless the caller
does not require it. Subsequent frames can use analytic backend NACs or the HST/
NPI path lift described above.

The optional properties are stored on each `ExcitedStateFrameContext` as
`oscillator_strengths` and `transition_dipoles`. They are intentionally not
added to `AdiabaticResult`, preserving the existing dynamics contract.

Current PySCF restrictions are deliberate and fail explicitly: no unrestricted
or open-shell references, no spin-orbit coupling, no QM/MM embedding, no complex
response amplitudes in the overlap implementation, and no automatic fallback
when a requested TD gradient or transition property is unavailable.

## Scratch, previous wavefunctions, and restart

`ExcitedStateProvider` creates a unique process/provider instance directory, a
unique call directory, and one child directory per bead. This prevents
same-named electronic-structure files from colliding across beads or processes.
Available cleanup policies are:

| Policy | Successful call | Failed call |
| --- | --- | --- |
| `always` | remove | remove |
| `success` | remove | retain for diagnosis |
| `never` | retain | retain |

A frame context contains the previous geometry, backend payload, raw-to-tracked
permutation, phases, optional transition properties, optional time-derivative
coupling, and diagnostics. Backends use the payload for an SCF guess and for
wavefunction overlap. No hidden global "last frame" is used, so interleaved
trajectories do not share electronic history.

`provider.export_context(result.context)` creates a JSON-compatible,
checksummed snapshot. `provider.import_context(snapshot)` validates the checksum
and exact scientific identity before reconstructing the payload. The caller is
responsible for placing this snapshot in its own atomic checkpoint. The provider
itself introduces no RNG state.

## Scientific identity and statistics

The provider fingerprint includes all options that can alter numerical results:
state count, atom symbols, coupling and tracking tolerances, timestep used for
reported time-derivative couplings, backend program version and method settings,
resolved basis/ECP hashes, input-template hash, overlap model, and external
command/environment hashes where applicable. Scratch location and cleanup policy
are operational and do not alter the scientific identity.

`provider.statistics()` returns call/frame/failure counts, wall time, backend
time, and counts by stable `ExcitedStateStatus` name. Timings are diagnostic,
not a physical convergence metric.

## External file backends

Three strict file-oriented adapters are supplied:

| Backend | Native target | Default coordinate handling | Validation in this change |
| --- | --- | --- | --- |
| `ORCAExcitedStateBackend` | ORCA TDDFT/TDA | explicit `Bohrs` keyword | recorded-output parser and mock subprocess only; executable untested |
| `QChemExcitedStateBackend` | Q-Chem TDDFT/TDA | `INPUT_BOHR TRUE` | recorded-output parser and mock subprocess only; executable untested |
| `OpenMolcasExcitedStateBackend` | OpenMolcas RASSCF | XYZ comment contains `bohr` | recorded-output parser and mock subprocess only; executable untested |

These classes invoke an argument vector with `shell=False`; a shell command
string is rejected. Commands may use `{input}`, `{output}`, `{scratch}`, and
`{previous}` placeholders. Previous wavefunction artifacts are restored under
`previous/` and selected new artifacts are copied into the in-memory context up
to `max_artifact_bytes`, so scratch cleanup does not destroy the next-frame
wavefunction. User-supplied basis and ECP files are read and hashed at backend
construction, then the exact hashed bytes are staged by basename into every
call directory. Managed-filename collisions are rejected before execution.

Native program output varies substantially across versions and methods. The
adapters therefore parse stable wrapper markers for all-root gradients and
optional transition properties:

```text
PYRPMD GRADIENT STATE 0
1   gx   gy   gz
2   gx   gy   gz

PYRPMD PROPERTY STATE 1 F 0.123 MU 0.1 0.2 0.3
```

Gradient state labels are zero based. Atom labels may be consistently zero or
one based. Values are positive gradients in hartree/bohr and transition dipoles
in atomic units. A site-specific wrapper is expected to collect the requested
all-root data and emit these records. The default templates are starting points,
not claims that an unmodified external executable will produce every required
root gradient in one invocation.

An optional `overlap_command` may run a program-specific overlap tool such as a
wfoverlap-style helper. It must write a finite square overlap matrix with the
same `O_ij = <old_i|new_j>` convention. If neither native overlap nor an overlap
helper is supplied, the provider fails when overlap/coupling is required.
`initial_nac="finite_difference"` is also available for file backends, but is
expensive and must be validated for the concrete wrapper and wavefunction files.

The external input templates, basis and ECP file contents, command vector,
program version, timeout, retained-artifact limit, and a hash of explicit
environment values enter the backend identity. Environment values themselves
are not exposed in the identity.

## Failure semantics

Backends raise `BackendFailure` with a stable status such as
`SCF_NOT_CONVERGED`, `EXCITED_STATE_NOT_CONVERGED`, `STATE_COUNT_MISMATCH`,
`GRADIENT_UNAVAILABLE`, `OVERLAP_UNAVAILABLE`, `STATE_TRACKING_FAILED`,
`EXTERNAL_PROGRAM_FAILED`, `OUTPUT_PARSE_ERROR`, `SCRATCH_ERROR`, or `TIMEOUT`.
The provider leaves energies and gradients as `NaN`, omits context for the
failed bead, and preserves the message. It never returns a zero coupling,
changes precision/device, changes method, or reuses stale data as a silent
fallback.

`validate_adiabatic_result` then raises before a dynamics integrator accepts a
partially failed frame. Thus no half-valid electronic step is committed by the
existing adiabatic propagators.

## Validation and example

Run the focused suite with the optional PySCF dependency:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m pytest -q tests -k "excited_state or adiabatic or pyscf"
```

Run the deterministic public-API example:

```bash
python examples/excited_state/toy_rpsh.py --steps 20 --dt 0.02
```

The toy avoided crossing has independent analytic energy, gradient, NAC, and
overlap references. The short trajectory checks interface operation and energy
drift only; its output deliberately reports `physical_converged: false`.
Detailed commands, raw summaries, hardware, and known validation gaps are in
`docs/validation/excited-state-interface-2026-11-03/README.md`.

## References

Catalog entry: [na-excited-state](manual/references.md#na-excited-state). Also cite the electronic-structure
method, functional and basis set used by the backend.

- S. Hammes-Schiffer and J. C. Tully, "Proton transfer in solution: Molecular
  dynamics with quantum transitions," *J. Chem. Phys.* **101**, 4657 (1994),
  [doi:10.1063/1.467455](https://doi.org/10.1063/1.467455).
- G. A. Meek and B. G. Levine, "Evaluation of the time-derivative coupling for
  accurate electronic state transition probabilities from numerical
  simulations," *J. Phys. Chem. Lett.* **5**, 2351 (2014),
  [doi:10.1021/jz5009449](https://doi.org/10.1021/jz5009449).
- A. Jain, E. Alguire, and J. E. Subotnik, "An efficient, augmented surface
  hopping algorithm that includes decoherence for use in large-scale
  simulations," *J. Chem. Theory Comput.* **12**, 5256 (2016),
  [doi:10.1021/acs.jctc.6b00673](https://doi.org/10.1021/acs.jctc.6b00673).
- Q. Sun *et al.*, "PySCF: the Python-based simulations of chemistry
  framework," *WIREs Comput. Mol. Sci.* **8**, e1340 (2018),
  [doi:10.1002/wcms.1340](https://doi.org/10.1002/wcms.1340).
