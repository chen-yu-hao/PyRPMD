---
orphan: true
---
# Native thermostat and barostat trajectories

`pyrpmd-native` and `pyrpmd.native_ensemble.NativeEnsemble` are an explicit,
opt-in C++17 trajectory driver. One `run(steps)` call executes all steps without
Python callbacks or GIL acquisition. The engine owns positions, momenta, cell,
cell momenta, GLE auxiliaries, force caches, the random generator and its cached
Gaussian draw. Python prepares configuration, launches trajectories and writes
checkpoints at block boundaries. It does not run the integrator or generate
per-step noise.

This driver does **not** silently replace the existing `pyrpmd run` materials,
PMF, metadynamics, constrained-parent or reactive-child workflows. Those retain
their current integrators, checkpoint formats and dispatch. Migrating their
biases, constraints and product classifiers is separate work. In particular,
thermostatted sampling must not be relabeled as ordinary RPMD real-time dynamics.

## Build and run

Build normally with `python -m pip install -e '.[test]'`. Both `_native` and
`_thermal` are compiled. `PYRPMD_NO_NATIVE=1` disables both; the native driver
then fails explicitly instead of falling back to Python. CMake also builds both.

```bash
pyrpmd-native examples/ensembles/native_nvt.toml --output runs/native-nvt
pyrpmd-native examples/ensembles/native_npt.toml --output runs/native-npt
pyrpmd-native examples/ensembles/native_npt.toml --output runs/native-npt \
  --resume runs/native-npt/checkpoint.npz
```

The examples are numerical controls (harmonic oscillator and ideal gas), not
validated material models. On resume, the input duration is an **additional**
propagation length. The command writes `trajectory-START-END.npy`, a checkpoint
and a JSON summary with `physical_converged=false`.

Python can retain one engine across as many native blocks as needed:

```python
import numpy as np
from pyrpmd.native_ensemble import NativeEnsemble

# All numbers in this low-level API are atomic units.
engine = NativeEnsemble(
    q=np.zeros((32, 2, 3)), masses=[1837.0, 1837.0],
    beta=1052.6, dt=4.1341,
    potential={"kind": "free"}, cell=np.eye(3) * 20,
    thermostat={"mode": "pile-l", "gamma_centroid": 0.0002419},
    barostat={"mode": "bzp", "pressure": 3.3989e-9,
              "tau": 41341.0, "gamma": 0.0002419},
    seed=123,
)
result = engine.run(1_000_000, stride=1000,
                    trajectory="trajectory.npy", record_centroids=True)
engine.save("checkpoint.npz")
```

There is no Python time-step loop in this example. A cheap built-in model is
used only to make it runnable. Production use requires an appropriate native
potential and convergence studies.

## Thermostats

| `mode` | Native algorithm and relevant fields |
| --- | --- |
| `none` | Deterministic propagation; no thermostat |
| `andersen` | Whole-momentum Maxwell redraw every `interval` steps on the absolute clock |
| `langevin` | Cartesian exact OU update; `gamma` |
| `pile-l` | Normal-mode OU, internal friction `2*pile_scale*omega_k`, `gamma_centroid` |
| `pile-g` | PILE internal modes plus global centroid SVR; `tau`, `pile_scale` |
| `svr` | Bussi stochastic velocity rescaling, including the sign and zero-energy cases; `tau` |
| `gle` | Cartesian extended OU; explicit drift/covariance matrices in `gle` |
| `nm-gle` | Extended OU separately in each real normal mode |
| `nm-gle-g` | Normal-mode GLE plus centroid SVR; `gle`, `tau` |
| `piglet` | Per-mode GLE with supplied calibrated matrices and provenance; NVT only |
| `trpmd` | Internal-mode PILE with exactly zero centroid friction; `pile_scale`; NVT only |
| `ffl` | Fast-forward Langevin **rescale variant**: preserve direction, update speed; `gamma` |

No deterministic Nose-Hoover-chain thermostat is provided by this change.
Defaults of the numeric API are in atomic units, not physical recommendations.
NVT uses BCOCB for `integrator="cayley-bcb"` and BAOAB for
`integrator="normal-mode-bab"`. The Cayley half map is the square root of the
full-step map, not a Cayley map evaluated at half the time step. Velocity Verlet
is rejected by this driver. Fixed atoms are supported for NVT and excluded from
thermostatting; every fixed bead must be coincident and have zero momentum.

The Hamiltonian convention remains the sum over beads with inverse temperature
`beta/P`. Momentum variance is therefore `m*P/beta`. The ring frequency is
`omega_k = 2*P/beta*sin(pi*k/P)` in atomic units. These are fictitious sampling
momenta, not independent physical nuclei at temperature `P*T`.

### GLE and PIGLET matrices

For mass- and temperature-normalized physical/auxiliary momenta `y`, the native
update is `y' = T*y + S*xi`, where setup computes

```text
T = exp(-A*h)
S*S.T = C - T*C*T.T
```

`h=dt` for NVT and `h=dt/2` for the two NPT thermostat operations. SciPy is used
once at initialization; all draws and matrix-vector operations in the trajectory
are C++. The matrices have shapes `[s,s]` or `[P,s,s]`; `s` includes the physical
momentum, and is limited to 64. Matrix ordering is centroid, cosine/sine pairs,
then the Nyquist mode for even P. `C` is dimensionless, normalized by `P/beta`;
`A` uses inverse atomic time. Canonical GLE requires `C[0,0]=1` and zero physical/
auxiliary covariance. Drift stability, covariance symmetry and positive
semidefiniteness of continuous and discrete diffusion are checked.

```python
thermostat = {
    "mode": "nm-gle",
    "gle": {"A": [[1.0, 2.0], [-2.0, 1.0]], "C": [[1.0, 0.0], [0.0, 1.0]]},
}
```

This is a canonical **test matrix**, not a calibrated PIGLET set. `piglet`
requires explicitly per-mode `A`, matching `beta` and `beads`, and a nonempty
`provenance` string. There are no invented parameter presets. A provenance
string records the source but does not certify its validity. Tests verify matrix
routing, covariance and restart; they do **not** establish reduced-bead PIGLET
accuracy for a molecular observable. Obtain parameter sets appropriate to the
temperature, bead count and target observables, document their units/normalization
and compare against converged conventional PIMD. PIGLET and TRPMD NPT are rejected.

## Barostats

| `mode` | Implementation and restrictions |
| --- | --- |
| `bzp` / `isotropic` | Isotropic Bussi-Zykova-Parrinello PIMD, explicit `pressure`, `tau`, optional piston `gamma` |
| `mttk` / `mtk` / `flexible` | Upper-triangular flexible-cell PIMD, hydrostatic `pressure`, `tau`, `gamma`, optional `cell_mask` |
| `rgb` / `anisotropic` | Raiteri-Gale-Bussi external-stress coupling, explicit symmetric `stress` and `reference_cell` |
| `scr` / `stochastic-rescaling` | Isotropic stochastic cell rescaling, **P=1 only**; `pressure`, `tau`, positive `compressibility` |

NPT requires three-dimensional PBC and all atoms mobile. The piston bath is
Langevin OU. Fixed-atom/slab NPT is not implemented; a flexible-cell mask is not a
claim of a validated slab ensemble. For MTTK/RGB, input **row lattice vectors must
form a lower-triangular cell** with positive diagonals. The strain-rate/momentum
matrix is upper triangular. `cell_mask` follows that upper-triangular convention;
its inactive components are kept zero. No hidden coordinate rotation is performed.

NPT uses a symmetric OBABO splitting, not the NVT BCOCB sampling scheme. Internal
ring modes do not scale with volume: the cell propagator acts on the centroid.
The pressure force includes the physical virial, the centroid kinetic term, the
internal-coordinate force correction and the ensemble-measure term. For BZP,
`W = 3*N*(P/beta)*tau^2`; flexible-cell components use
`W = N*(P/beta)*tau^2`. The coupled force/piston kick includes the integrated
centroid kinetic-pressure polynomial. Cell exponentials and `sinh(x)/x` use
scaling/Taylor/squaring, including zero/repeated eigenvalues and pure shear.

SCR uses the square-root-volume stochastic proposal and a symmetric drift/rescale
map. It has finite-time-step bias and is **not** Metropolized. Reduce the time step
and check the distribution; successful execution is not proof of exact NPT
sampling at arbitrary time steps. RGB uses the reference-cell strain-energy
model; it is not an interchangeable hydrostatic-pressure shortcut.

`sc-isotropic` / `suzuki-chin` raises `NotImplementedError`. Implementing that name
without a Suzuki-Chin action and the corresponding force/virial derivatives
would give the wrong ensemble. High-order path integrals remain outside this PR.

## Native potentials and the callback boundary

Three analytic models are included: `free`, nonperiodic `harmonic`, and a
fractional-coordinate `periodic_cosine` control. Their purpose is testing.

Existing C ABI v1 and Fortran `BIND(C)` batch bindings can be leased once for an
entire nonperiodic trajectory. Pass the provider or native binding plus a stable
`potential_identity` identifying its model and immutable parameters. Their v1
interfaces do not accept cell/virial data, so they cannot run NPT.

Periodic compiled engines can use `NativePeriodicPES` and the synchronous
callback in `native/include/pyrpmd/thermal_pes.h`:

```python
from pyrpmd.native_ensemble import NativePeriodicPES
pes = NativePeriodicPES("libmaterial.so", "evaluate_thermal", natoms=128)
```

The callback gets a batch of beads, unwrapped coordinates, current row-vector
cell and PBC flags, and fills physical energies, forces and configurational
virials. Units are bohr/hartree. Virials are minus the derivative under a common
Cartesian affine strain of coordinates and cell, including explicit cell
dependence and excluding kinetic, spring and bias terms. Missing virials are
errors, not silently zero. The new callback is a small **thermal-engine bridge**,
not a replacement or version bump of the existing public PES ABI.

The wrapper pins the shared library, serializes access to its context, checks
owner lifetime at block boundaries and hashes the library for restart identity.
Callbacks must be trusted compiled, synchronous evaluators whose returned values
do not depend on call history. A nonzero context requires an owner and immutable
model identity. Rollback cannot undo hidden state in an external evaluator; such
stateful evaluators are unsupported. Native callbacks must not call Python.

Python-only ASE, Torch and custom Python potentials are rejected on this path;
wrapping them in a C++ time-step loop would still return to Python for forces.
There is no native socket, GPU model or MPI scheduler in this driver yet. It does
not claim every PyRPMD potential can execute without Python.

## Output, restart and units

`run(..., trajectory=...)` streams NPY records directly from C++, using O(P*N)
state/workspace storage rather than retaining all output frames. The returned
array is memory-mapped. Omitting `trajectory` returns an in-memory array whose
size grows with requested frames. Observations follow absolute step multiples of
`stride`; there is no extra off-clock final sample. A native/PES/I/O failure
rolls back the entire block, including RNG, force cache, auxiliaries and cell.
After a successful native block, failure to publish the final output filename
(e.g. a filesystem race) does not undo the completed trajectory. Existing output
files are never overwritten by `run`.

`save/load` uses checksummed, atomic NPZ checkpoints without pickle. Changing
integrator, matrices, potential identity, fixed positions or other physics is
rejected; changing the seed for a restart is allowed because saved RNG state is
restored. The C++ RNG and Gaussian cache have a different sequence from the old
NumPy path. Bitwise restart is tested within the same standard-library/backend;
RNG ABI fingerprints reject incompatible C++ standard-library implementations.
Python signals are handled after a native block returns, so choose checkpoint
block lengths with interruption latency in mind. There are no Python checkpoints
inside one continuous C++ block.

The explicit input schema is `native-ensembles/1`, not the legacy task schema.
JSON and TOML accept physical units for temperature, masses, positions/cell units,
time steps, thermostat frequencies, pressure (`Pa`, `bar`, `atm`, `MPa`, `GPa`,
`kbar`) and inverse pressure. Numeric inputs remain canonical atomic values.
`gle.A_unit` is converted once at initialization. Unknown or irrelevant fields
are errors rather than ignored settings.

## Validation and scope

Run `pytest tests/test_native_ensembles.py`. Tests cover deterministic comparison
with the original integrators, all thermostat restart streams, native-PES error
rollback, compiled calls and GIL release, streaming output, pressure units,
unit-aware CLI restart, fixed atoms, TRPMD centroid conservation, scalar and
augmented GLE controls, finite-bead harmonic equilibrium, ideal-gas NPT volume
mean/variance, independent matrix-exponential cell steps and NPT time reversal.
These validate implementation properties, **not chemical rates, real-material
accuracy, arbitrary flexible-cell thermodynamics or calibrated PIGLET speedups**.

`benchmarks/efficiency/native_ensembles.py` compares one native block with repeated
one-step calls to the same native engine, using identical final states. It
measures call overhead on a cheap harmonic PES, not a general ML/ab-initio speedup.
See the [validation summary](validation/native-ensembles-2026-10-31/README.md).

## Method sources

Equations and conventions were checked against the official i-PI implementation:
`ipi/engine/barostats.py`, blob `98cf6e6c38366d9580cb48b69e36e29a48e1bb76`,
[official input reference](https://docs.ipi-code.org/input-tags.html) and
[method references](https://docs.ipi-code.org/features.html). The native port
retains the MIT notice in `native/thermal-NOTICE.md`.

Relevant original works include PILE (DOI 10.1063/1.3489925), stochastic velocity
rescaling (10.1063/1.2408420), BZP (10.1063/1.3073889), constant-pressure PIMD
(10.1063/1.478193), RGB (10.1088/0953-8984/23/33/334213), stochastic cell rescaling
(10.1063/5.0020514), and TRPMD (10.1063/1.4883861).
