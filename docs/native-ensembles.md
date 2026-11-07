---
orphan: true
---
# Native thermostat and barostat trajectories

`pyrpmd.native_ensemble.NativeEnsemble` is the persistent C++17 executor used
by the composable `Dynamics` interface. Its direct API and `pyrpmd-native` CLI
remain available. With native PES bindings, one `run(steps)` call executes all
steps without Python callbacks or GIL acquisition. The executor owns positions,
momenta, cell, cell momenta, GLE/NHC auxiliaries, force caches, the random generator
and its cached Gaussian draw. An explicitly selected Python PES bridge uses the
same native operators and reports its force callbacks separately.

The standard `pyrpmd run` PIMD workflow can explicitly select this driver with
`sampling.engine="native"`; see [native PIMD](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/native-pimd-workflow.md). Historical
inputs retain their integrators, checkpoints and random streams. The common
`Dynamics` interface also exposes these thermostats and barostats to task
adapters. Thermostatted sampling must not be relabeled as ordinary RPMD real-time
dynamics; reaction production remains unthermostatted unless the chosen method
explicitly calls for a thermostat.

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
| `piglet` | Per-mode GLE with supplied calibrated matrices and provenance; NVT or isotropic BZP NPT with a canonical centroid |
| `trpmd` | Internal-mode PILE with exactly zero centroid friction; `pile_scale`; NVT only |
| `ffl` | Fast-forward Langevin **rescale variant**: preserve direction, update speed; `gamma` |
| `nhc` | Global Nosé–Hoover chain with three-stage Suzuki–Yoshida composition; `tau` (alias `tdamp`), `chain_length`, `substeps` |

Defaults of the numeric API are in atomic units, not physical recommendations.
NHC uses a symmetric bath/2–deterministic-map–bath/2 split, with the same chain
masses and mobile-atom degree count as the existing Python `NoseHooverChain`.
Its positions and momenta are stored as the two rows of checkpoint `aux`.
The common `Dynamics` API retains the existing pure-Python NHC bath when
`_thermal` is unavailable. That fallback has the separate
`pyrpmd.python-nhc-bath.v1` checkpoint format and remains Python when restored,
even if an extension has since been installed. Neither implementation consumes
random numbers for deterministic chain propagation.
Other NVT thermostats use BCOCB for `integrator="cayley-bcb"` and BAOAB for
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
and compare against converged conventional PIMD. PIGLET supports isotropic BZP
NPT when the centroid physical momentum has unit normalized variance and zero
covariance with its auxiliary variables. The Python API and native kernel
both check that condition. TRPMD and flexible-cell PIGLET NPT remain unsupported.

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

### Meaning of the reported pressure

The `pressure` column of `NativeEnsemble.run()` uses the **instantaneous
centroid momenta**. With a pressure controller this is its feedback pressure;
the same column is a diagnostic when the cell is fixed. It does not replace
the kinetic contribution with the configured temperature.

For an all-mobile, fully periodic system, let `pbar_i` denote the average
momentum over the P beads, and define
`R = Tr(sum_j W_j) - sum_ji (q_ji-qbar_i) dot F_ji`. Then

$$
p_{\mathrm{control}}=
\frac{P\sum_i |\bar p_i|^2/m_i+R}{3PV},\qquad
p_{\mathrm{CV},T}=\frac{N}{\beta V}+\frac{R}{3PV}.
$$

Their pointwise difference is

$$
p_{\mathrm{control}}-p_{\mathrm{CV},T}
=\frac{\sum_i |\bar p_i|^2/m_i-3N/\beta}{3V}.
$$

The usual equality of ensemble means follows from canonical centroid kinetic
moments conditional on coordinates and volume. A PIGLET centroid **bath covariance** satisfying the
required canonical conditions does not establish this property for the
coupled anharmonic steady state, because its internal-mode baths are
noncanonical. Check both pressure definitions, timestep refinement and
sampling uncertainty. Agreement of piston feedback with the target alone
does not establish agreement of the configured-temperature estimator. The
BZP stationary piston force instead contains the volume-weighted combination
`V*(p_control-p_external) + 1/beta`.

SCR uses the square-root-volume stochastic proposal and a symmetric drift/rescale
map. It has finite-time-step bias and is **not** Metropolized. Reduce the time step
and check the distribution; successful execution is not proof of exact NPT
sampling at arbitrary time steps. RGB uses the reference-cell strain-energy
model; it is not an interchangeable hydrostatic-pressure shortcut.

The barostat names `sc-isotropic` / `suzuki-chin` do not infer a high-order action.
Use an explicit `SuzukiChinPotential` with the Python `BZPBarostat` or the legacy
SC-NPT input workflow described in [path-integral actions](tier2-path-integral.md).
Direct `NativeEnsemble` SC providers remain unsupported because its observation
rows use primitive estimators. SC observations must use the method-specific
energy/heat-capacity/pressure functions rather than reinterpret those rows.

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
model identity. The wrapper's bound cell is synchronized after accepted native
blocks, imports and checkpoint restores, so Python reference evaluations see the
same geometry. Binding publication and failures restore the prior cell alongside
the native state. Rollback cannot undo hidden state in an external evaluator;
such stateful evaluators are unsupported. Native callbacks must not call Python.

The direct API continues to reject ordinary Python providers unless
`allow_python=True` is explicitly selected. This option uses a separate copied
buffer bridge, never the trusted native callback ABI. Supply a stable
`potential_identity`; for variable-cell dynamics the provider must implement
`set_cell(cell_bohr)`, and every periodic evaluation must return the physical
per-bead configurational virial. Cartesian `q dot F` is not substituted for a
missing periodic virial. Provider cells are restored along with the complete
native state if a block fails.

The returned backend is `native-operators-python-pes` and
`python_step_callbacks` counts actual Python force evaluations in that block.
Native bindings retain `backend="native-cxx17"` and zero Python callbacks.
There is no native socket, GPU model or MPI scheduler in this executor. A Python
bridge does not imply that an ASE or Torch force evaluation has become native.

### Operators and backend changes

`apply_thermostat(q, p, dt, step=...)` applies only the selected bath to external
momenta. It keeps auxiliary variables and the full RNG state, leaves coordinates
unchanged, and makes no force call. The optional logical step preserves Andersen
collision timing. GLE requires the exact duration used to prepare its transition
matrices. A standalone bath rejects a barostat.

`copy_thermostat_from(other)` transfers compatible bath variables and RNG state
between executors while retaining the target's physical state, forces and cell.
This supports common-engine backend changes without restarting the thermostat.
Masses, temperature, bead count, frozen positions and thermostat parameters are
validated; GLE transition matrices must also match. For NHC, full and half-step
executors can share chain state because the duration is supplied to the operator.

`set_state(q, p, step)` explicitly imports a new physical state and reevaluates
all force/energy/virial caches. It retains the bath and piston state, and restores
the original state on a rejected input or failed potential evaluation. Checkpoint
`restore` remains the checksummed operation for resuming an identical model.

## Output, restart and units

`run(..., trajectory=...)` streams NPY records directly from C++, using O(P*N)
state/workspace storage rather than retaining all output frames. The returned
array is memory-mapped. Omitting `trajectory` returns an in-memory array whose
size grows with requested frames. Observations follow absolute step multiples of
`stride` relative to optional `sample_start`; there is no extra off-clock final
sample. `record_moments=True` with `record_centroids=True` additionally records
centroid velocities and per-atom squared radii of gyration in C++. A native/PES/I/O failure
rolls back the entire block, including RNG, force cache, auxiliaries and cell.
A failure to publish or read the final output file also restores the pre-block
state. Publication is exclusive and never overwrites an existing output. A
concurrent writer's file is not removed during rollback.

`save/load` uses checksummed, atomic NPZ checkpoints without pickle. Changing
integrator, matrices, potential identity, fixed positions or other physics is
rejected; changing the seed for a restart is allowed because saved RNG state is
restored. The C++ RNG and Gaussian cache have a different sequence from the old
NumPy path. Bitwise restart is tested within the same standard-library/backend;
RNG ABI fingerprints reject incompatible C++ standard-library implementations.
Python signals are handled after a native block returns, so choose checkpoint
block lengths with interruption latency in mind. There are no Python checkpoints
inside one continuous C++ block.

The common block API uses `run_prefix` when its contract is to retain completed
steps after a failure. This shares the native loop with `run`, but checkpoints
physical state, bath auxiliaries, RNG and force cache before each step and
accepts its observations in the same transaction. The result contains only
accepted records plus `completed` and `error`; the failed step is absent. A
Python periodic provider's cell is restored to the accepted prefix as well.
Prefix observations are in memory; file-streaming `run` remains block-atomic.

`run_prefix(..., observe=mask, record_energy=True, check_energy=False,
energy_parts=False, ring_radii=False)` selects the common observation contract.
The binary mask has one entry per requested step; only selected rows are returned.
Energy is checked on observed rows when requested, or initially and after every
step when `check_energy=True`. Unrequested thermal estimators and ring radii are
not computed or validated. This allows finite physical states to advance without
unrelated squared-moment overflow. Results include `centroid_momenta` directly
and optional `energy_parts` with physical potential/P, centroid kinetic,
noncentroid kinetic/P and spring/P. Other thermal sample columns are undefined
(NaN). Calling `run_prefix` without these options retains the full thermal
observation schema and its absolute stride clock.

The explicit input schema is `native-ensembles/1`, not the legacy task schema.
JSON and TOML accept physical units for temperature, masses, positions/cell units,
time steps, thermostat frequencies, pressure (`Pa`, `bar`, `atm`, `MPa`, `GPa`,
`kbar`) and inverse pressure. Numeric inputs remain canonical atomic values.
`gle.A_unit` is converted once at initialization. Unknown or irrelevant fields
are errors rather than ignored settings.

## Validation and scope

Run the native ensemble, NHC, Python bridge, operator-switching and prefix
suites: `pytest tests/test_native_ensembles.py tests/test_native_nhc.py
tests/test_native_python_bridge.py tests/test_native_operator_switching.py
tests/test_native_prefix.py tests/test_composed_native_observations.py`.
Tests cover deterministic comparison
with the original integrators, all thermostat restart streams, native-PES error
rollback, compiled calls and GIL release, streaming output, pressure units,
unit-aware CLI restart, fixed atoms, TRPMD centroid conservation, scalar and
augmented GLE controls, finite-bead harmonic equilibrium, ideal-gas NPT volume
mean/variance, independent matrix-exponential cell steps and NPT time reversal.
NHC tests compare chain variables with the established Python implementation,
check second-order extended-energy error, exact restart, frozen atoms and failure
rollback. Bridge tests compare native/Python-PES cell trajectories and complete
restart state; backend-switch tests verify preservation of one bath/RNG stream.
These validate implementation properties, **not chemical rates, real-material
accuracy, arbitrary flexible-cell thermodynamics or calibrated PIGLET speedups**.

`benchmarks/efficiency/native_ensembles.py` compares one native block with repeated
one-step calls to the same native engine, using identical final states. It
measures call overhead on a cheap harmonic PES, not a general ML/ab-initio speedup.
See the [validation summary](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/native-ensembles-2026-10-31/README.md).

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

## Standard PIMD workflow

For the opt-in `task="pimd"` route, streaming checkpoint journals and native
normal-mode fusion, see [Native PIMD workflow](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/native-pimd-workflow.md).

## Versioned random streams and CPU threading

Existing inputs keep `mt19937_64` with the C++ standard library's normal
sampler. Its draw order, serialized Gaussian cache, checkpoint schema and
fingerprint remain unchanged. Explicit `rng="mt19937_64"` selects that same
contract. Random draws on this path remain serial; independent transforms,
kicks and built-in per-bead PES evaluations can use OpenMP.

A new run can opt into `philox-v1` in its thermostat specification:

```python
engine = NativeEnsemble(
    q, masses, beta, dt, thermostat={"mode": "pile-l", "rng": "philox-v1"},
    potential={"kind": "harmonic", "k": 0.5}, seed=49,
)
```

The standard input route accepts the option for the native PIMD engine:

```toml
[sampling]
engine = "native"
thermostat = "pile-l"

[sampling.thermostat_params]
rng = "philox-v1"
```

`pyrpmd-native` inputs put `rng = "philox-v1"` directly in their
`[thermostat]` table. Other standard workflow engines reject this option
before provider construction or output creation. Python `Dynamics` accepts
it in an explicit thermostat mapping; the compiled thermal extension is
required for this stream.

The independent implementation in `native/philox.h` follows Philox4x32-10
(Salmon et al., SC'11, DOI 10.1145/2063384.2063405). The seed and the physical
step plus a monotonically increasing bath-event counter select a stream;
bead/mode, degree of freedom, purpose and auxiliary component select draws.
The event origin retains all 128 Philox output bits, and the seed remains the
key for the coordinate draw. Two 52-bit uniforms strictly inside `(0, 1)` produce a Box--Muller pair.
Neither thread count nor OpenMP scheduling enters this mapping. PILE-G/SVR
uses a sum of independent squared normals for its chi-square draw. The
checkpoint stores the seed and event counter, so restoring into an engine
constructed with another seed continues the saved stream. There is no hidden
noise buffer or cached half-pair on the Philox path.

The `philox-v1` ABI and thermostat option enter the physics fingerprint;
loading a Philox checkpoint into a legacy engine is rejected. This version
also fixes the packed two-real-FFT rounding path used by fused PILE-L/TRPMD
for power-of-two rings with P >= 8 and at least 8 atoms. It implements the
same C-O-C map, but its rounding need not match the legacy four-transform
path. Replay is bitwise within the same floating-point build and libm;
CPU/compiler/FMA changes are not a cross-build bitwise guarantee. The legacy
path keeps the original scalar operation order. No default is switched.

Set `Dynamics(..., threads=n)` or `NativeEnsemble(..., threads=n)` explicitly;
both default to one thread, even when `OMP_NUM_THREADS` is unset or larger.
The native input schema also accepts `threads`. Composed thermal operators
inherit `Dynamics`' thread count, including the Python propagation loop.
Workflow paths using `Dynamics` inherit `parallel.threads_per_rank`; the native
PIMD workflow retains its existing one-thread-per-trajectory restriction.
Requesting more than one thread requires an OpenMP build. Thread counts are
execution settings and do not change physics fingerprints or checkpoints.
Large arrays use static
OpenMP loops with per-thread FFT/GLE scratch; small systems stay serial.
Legacy energy and kinetic reductions retain their serial order. Large
`philox-v1` observations sum each bead independently, then sum beads in order;
its SVR reductions use fixed 2048-element blocks and sum block totals in
index order. Both allow ulp differences from the legacy sum. Built-in PES
reductions retain atom order within each independent bead. Thread-count
changes therefore do not reorder sums. Arbitrary external PES callbacks keep
their existing threading contract.

`Dynamics.propagate()` and `propagate_block()` use private binary rollback
tokens. Tokens preserve the original executor and lazy bath selection, RNG,
auxiliaries, cell, cached forces and the Python generator. Live state transfer
reuses array buffers and does not produce JSON or checkpoint digests. Native
code prepares the initial-energy terms in one traversal; NumPy still performs
their reductions in the original order. Public
`operator_state_dict()`, `state()`, `save()` and checkpoint readers retain
their existing persistent format. A failed whole block restores its initial
state; a prefix API still retains only fully accepted steps.

Repeated harmonic-provider checks compare detached copies of the original
parameter arrays before expanding them over atoms. Unchanged inputs reuse the
same binding. Changed values, shapes or provider capabilities still follow the
existing validation and binding-identity checks.

## Reproducing thermal performance and compatibility checks

Use the same interpreter, compiler flags, CPU affinity and observation options
for both checkouts. Install each checkout before measuring it, and select it
with `PYTHONPATH=/absolute/checkout/src` when using a shared environment. The
script reports its imported source, revision, compiler and every wall time.
Startup and warm-up are excluded; JSON distinguishes the public `Dynamics`
path from direct `NativeEnsemble` measurements.

```bash
python -m pip install -e . --no-build-isolation
OPENBLAS_NUM_THREADS=1 python benchmarks/efficiency/thermal_throughput.py \
  --baseline --threads 1 --profile --output baseline.json
OPENBLAS_NUM_THREADS=1 python benchmarks/efficiency/thermal_throughput.py \
  --rng philox-v1 --threads 1 --profile --output candidate.json
for threads in 1 2 4; do
  OPENBLAS_NUM_THREADS=1 python benchmarks/efficiency/thermal_throughput.py \
    --api native --rng philox-v1 --threads "$threads" --block-steps 8 --output native-$threads.json
done
c++ -O3 -std=c++17 -fopenmp -I native benchmarks/efficiency/thermal_kernels.cpp -o /tmp/thermal-kernels
/tmp/thermal-kernels 4096 32 philox-v1 1
OPENBLAS_NUM_THREADS=1 python benchmarks/efficiency/thermal_statistics.py --rng philox-v1 --threads 4 --output statistics.json
```

The component benchmark measures generic four-transform helpers, the actual
middle map, PES, kicks, noise generation/application, observations, state
copies, and a STREAM-style triad. Component timings overlap and must not be
added to claim a whole-run speedup. The full matrix uses canonical synthetic
GLE/PIGLET matrices to measure implementation cost; it does not validate a
calibrated PIGLET model or real-material dynamics.

`benchmarks/efficiency/thermal_contract.py --write /tmp/thermal-reference
--output reference.json` records legacy checkpoints in `main@801953d`.
Run the same script with `--compare /tmp/thermal-reference --output replay.json`
in the candidate using identical build flags. Keep NPZ references outside Git;
the report records their SHA256 values and exact replay results.


### PR #115 follow-up: tile-local scratch and compact observations

The follow-up [validation record](validation/pr115-thermal-refactor/README.md)
checks a tile-local thermal path without changing the default RNG, checkpoint
format, or floating-point map. The common `Dynamics.propagate()` fast path now
requests only its public position/energy records; energy checks still run on
every step. Direct `NativeEnsemble` estimator output is unchanged, and older
extensions retain the previous observer through a capability check. Rebuild
the native extension to enable the compact records. Performance measurements
and remaining platform/CI limits are reported separately from earlier evidence.
