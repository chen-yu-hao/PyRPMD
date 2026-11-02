# ML potential lifecycle: committees, active learning and additive corrections

The lifecycle API keeps a fixed potential during a trajectory. A committee
provides mean energies and forces together with disagreement scores;
`ActiveLearningMonitor` collects high-disagreement geometries for subsequent
labelling. The separate `ActiveLearningLoop` labels, trains and resamples between
segments with versioned datasets and committees. `DeltaLearningPotential` adds
a learned correction to a reference
potential and exposes the existing ring-polymer contraction (RPC) fast/slow
split. These pieces can use the same nested potential specifications as the
ordinary Torch, MACE, NequIP, DeePMD, fairchem or Python providers.

The runnable examples in [`examples/ml_lifecycle/`](https://github.com/chen-yu-hao/PyRPMD/tree/main/examples/ml_lifecycle)
use inexpensive analytic models so they require no downloaded weights:

```bash
pyrpmd run examples/ml_lifecycle/committee.inp --output runs/committee
pyrpmd run examples/ml_lifecycle/delta_rpc.inp --output runs/delta_rpc
python examples/ml_lifecycle/delta_rpc_mts.py
python examples/ml_lifecycle/closed_loop.py --output runs/closed-loop
```

These short pilots verify configuration and data flow. They do not establish
equilibration, a calibrated prediction error, or contraction convergence.

## Committee mean and uncertainty

`CommitteePotential(members)` requires at least two separate deterministic
providers, each obeying the ordinary independent-configuration PES contract.
The members must use the same atom order, energy zero and units. The mean PES
is integrated; uncertainty does not add a force or change the thermostat.

For `M` members and bead `b`, the reported energy score is the population
standard deviation of the **total configuration energy**,

```{math}
\sigma_{E,b}=\sqrt{M^{-1}\sum_m(E_{m,b}-\bar E_b)^2}.
```

For atom `i`, the force score is the square root of the sum of the three
Cartesian population variances,

```{math}
\sigma_{F,b,i}=\sqrt{\sum_{a=x,y,z}M^{-1}
  \sum_m(F_{m,b,i,a}-\bar F_{b,i,a})^2},\qquad
\sigma_{F,b}^{\max}=\max_i\sigma_{F,b,i}.
```

The denominator is `M`, not `M-1`. Energy disagreement is extensive and can
depend strongly on system size; it is not divided by the atom count. Force
disagreement has units of force, not force squared. Neither score is a
calibrated bound on error: identically biased or insufficiently diverse
models can agree while all being wrong. Select model diversity and acquisition
thresholds using independent reference data.

```python
import numpy as np
from pyrpmd import CommitteePotential
from pyrpmd.potentials import HarmonicPotential

committee = CommitteePotential([
    HarmonicPotential(k=0.002),
    HarmonicPotential(k=0.0024),
])
q = np.zeros((8, 2, 3))       # [P,N,3], bohr
result = committee.evaluate_uncertainty(q)
mean_forces = result.mean.forces       # [P,N,3], hartree/bohr
energy_std = result.energy_std         # [P], hartree
force_std = result.force_std           # [P,N], hartree/bohr
maximum = result.max_force_std         # [P], hartree/bohr
member_energies = result.member_energies  # [M,P], hartree
```

`evaluate(q)` returns the mean `PESResult`. `uncertainty_at(q)` can reuse an
exactly matching coordinate evaluation. All returned arrays own their data;
modifying a returned force or diagnostic cannot corrupt the cached prediction.
Virials are averaged only when every member provides one. Python constructors
borrow providers by default; `owns_members=True` makes `close()` release every
owned provider, including when another close raises an exception. The input
factory owns the providers it constructs.

## Acquisition from an input file

The top-level committee is an ordinary PES specification:

```toml
[potential]
provider = "committee"

[[potential.members]]
provider = "builtin"
name = "harmonic"
params = {k = 0.002}

[[potential.members]]
provider = "builtin"
name = "harmonic"
params = {k = 0.0024}

[active_learning]
interval_steps = 5
force_threshold = "0.00005 hartree/bohr"
energy_threshold = "0.0001 hartree"
max_frames = 32
directory = "active_learning"
```

Replace each complete member table with a supported ML provider to use trained
models. At least one strictly positive threshold is required. A bead is
selected when **either** specified score is greater than or equal to its
threshold. Scores are computed on accepted states every `interval_steps`,
including equilibration; this interval is independent of trajectory output
sampling. The dynamics uses exactly the committee mean whether monitoring is
enabled or disabled.

Input monitoring currently supports the legacy fixed-cell `pimd` and
`diffusion` workflows. Barostats, the native ensemble engine, additional
path-integral actions and PLUMED-driven workflows are rejected when this
monitor is requested. The Python monitor remains independently callable when
a caller explicitly controls its state, clock and provenance.

Each trajectory has a separate directory,
`RUN/active_learning/replica_0000/`, unless another directory is supplied.
A relative directory is resolved against `RUN`; an absolute directory is used
directly, with the replica suffix still appended. Each inspected step is an
atomic, checksummed `step_000000000005.npz` record. It contains:

- The complete ring shape, step, time, geometry digest and model identity.
- Scores for every bead and the indices of the selected beads.
- Selected coordinates, committee mean energies and forces, and member
  energy predictions, all in atomic units.
- `labels_available=False`, explicitly distinguishing predictions from
  reference electronic-structure labels.

`max_frames` limits the number of selected **bead geometries per replica**,
not the number of complete rings. If a step crosses the remaining allowance,
beads are selected in increasing index order. Empty records still preserve
the inspection history when no threshold is crossed or the cap is reached.
The input-file monitor initiates no quantum-chemistry calculation, model fitting
or automatic model replacement. The separate Python closed-loop driver below
connects those stages while keeping every sampling segment on a fixed PES.

Replaying an existing step is idempotent only when its shape, geometry, time,
model identity and acquisition settings agree. The cap is recovered from
validated records at restart. Checkpoints retain collection provenance and
validate that every committed inspection step exists, including completed
resumes that do not reconstruct the PES. Deleting or altering a committed
record is an error; starting a new collection requires a new run directory.

The direct API accepts atomic-unit thresholds:

```python
from pyrpmd import ActiveLearningMonitor

monitor = ActiveLearningMonitor(
    committee, "candidate_records",
    force_threshold=5e-5, interval_steps=5, max_frames=32,
    model_identity="sha256-or-versioned-identity-of-my-complete-committee",
)
monitor.observe(q, step=5, time=10.0)
```

The default direct-API identity is `caller_managed`; provide a stable identity
when the collection must survive model replacement or a process restart. Input
workflows fingerprint the nested potential specification, system and explicit
local model files. Nested model paths resolve relative to the input file,
including local fairchem `.pt`/`.pth`/`.ckpt`/`.bin` paths. Registered fairchem
model names remain names. Indirect imports, remotely served weights and named
model downloads must also be pinned by the surrounding environment; the file
fingerprint is not a package lock or a remote-server attestation.

## Reference labelling and versioned datasets

`pyrpmd.potentials.lifecycle.label_candidates` supplies the next step after
collection. It accepts the ordinary `evaluate(float64[B,N,3])` reference-provider
contract, including the existing PySCF, tblite and i-PI adapters. Coordinates,
energies and forces stay in bohr, hartree and hartree/bohr; forces are never
scaled by bead count. Bounded batching submits each chunk once. Providers may
implement their own native batch or a serial sequence internally.

```python
from pyrpmd.potentials.lifecycle import LabelledDataset, label_candidates

labels = label_candidates(
    reference_provider, candidate_positions,
    reference_identity=reference_specification,
    batch_size=32,
    provenance={"source": "replica_0000/step_000000000005.npz"},
)
labels.save("labels.npz")
labels = LabelledDataset.load("labels.npz")
training_data = labels.successful()
print(labels.reference_hash, labels.sha256)
```

`reference_specification` is a nonempty JSON mapping or string supplied by the
caller. For an electronic-structure calculation, include the atom order,
method, basis/ECP, charge, multiplicity, convergence settings, package versions,
and hashes of external parameter files. Pin defaults as well as overrides. A
provider name or socket address alone does not identify the scientific method.
`reference_hash` hashes this declaration; the dataset `sha256` additionally
covers geometries, labels, statuses, messages and per-frame provenance. These
are content hashes independent of archive timestamps. They establish recorded
provenance and file integrity, not an attestation of a remote reference server.

Datasets own immutable arrays. `append(other)` returns a new dataset and rejects
a different reference identity or atom count. `successful()` returns only rows
with `status == 0`. Failed rows retain their positions, status and diagnostic
message, while their labels are canonical NaNs. A provider that returns
per-frame statuses preserves successful frames in the same batch. A provider
that raises an exception fails the entire submitted batch; the exception type
and message are retained for every affected geometry. The labeler does not
retry each frame, substitute another method or change the device. Malformed
returned shapes and successful-but-nonfinite values raise explicit errors.
Process interruptions propagate and leave the last committed state available.

Exact continuation requires an unchanged numerical/software environment and
independent reference calculations without hidden scientific state. Generic
callbacks must pin their code and dependency versions in their declared
identities; the MACE codec also records its dependency versions automatically.
Electronic warm starts and declared stateful or coupled-frame
providers are rejected. Use `warm_start=False` for PySCF and tblite. An i-PI
reference additionally requires `independent_frames=True`, a complete
`driver_identity` mapping or string and a `driver_sha256` of the pinned driver
configuration/program in its reference identity. This declaration is the
caller's audited promise that reconnection uses the same independent-frame
calculation. It does not permit a stateful external driver. See the adapter's
own socket lifecycle rules after a connection failure.

`label_candidates` alone returns an in-memory dataset. To preserve progress
between batches, either save after each call or use the iterative driver below,
which commits each bounded labelling batch automatically. Providers remain
owned by the caller.

## Iterative sampling, labelling and training

`pyrpmd.active_learning.ActiveLearningLoop` runs the complete cycle:

1. Sample a segment using a fixed committee.
2. Evaluate committee disagreement on the returned candidate configurations.
3. Rank threshold-crossing candidates, remove exact geometry duplicates and
   select within the round and total label budgets.
4. Label bounded reference batches and retain both successes and failures.
5. Train on all successful labels, persist the resulting model version, and
   install a new committee for the next sampling segment.

The loop uses the same population energy and maximum per-atom force scores as
`CommitteePotential`. Either specified score crossing its threshold is enough
for selection. Ranking uses the largest score divided by its configured
threshold; equal scores keep candidate order. Geometry deduplication is exact,
not a chemical similarity criterion. The threshold is assessed over the full
sampled batch, including frames excluded by a budget or already labelled.

```python
from pyrpmd.active_learning import ActiveLearningLoop, SamplingBatch

# sample(committee, rng, state) -> SamplingBatch(positions, new_state)
# train(dataset, previous_models) -> sequence of independent PES providers
loop = ActiveLearningLoop(
    "learning_run", initial_models,
    reference=reference_provider,
    reference_identity=reference_specification,
    sampler=sample, sampler_identity=sampling_specification,
    trainer=train, trainer_identity=training_specification,
    model_codec=codec,
    force_threshold=5e-5, energy_threshold=1e-4,
    max_rounds=8, max_labels=256, labels_per_round=32,
    label_batch_size=8, seed=37,
)
summary = loop.run()
```

The callback interface is deliberately independent of a particular dynamics or
training package. A sampler can drive PIMD/RPMD and return flattened
`[trajectory × bead, atom, 3]` configurations, provided all candidates use the
same atom order, cell and reference method. The built-in demonstrator below
uses classical one-dimensional Metropolis sampling. The loop does not add a
new input-file task or silently alter an existing trajectory workflow.

The sampling callback receives the committee, the loop-owned NumPy RNG and a
detached explicit state. Return every variable needed to continue: positions,
momenta when relevant, thermostat state, counters and any adaptive state.
State containers can contain ordinary JSON values and supported numerical
arrays; arbitrary Python object/pickle state is not accepted. Use only the
supplied RNG or explicitly persisted RNG states. The callback must not change
the committee during the segment. Changing a committee changes the sampling
Hamiltonian; equilibration and production-data selection after each update
remain the sampler's responsibility.

The training callback has exactly the interface
`train(dataset, previous_models) -> models`. It receives the successful subset
of the accumulated dataset and the preceding model versions. Return at least
two independent providers with the same atom order, energy zero and units. Do
not mutate the preceding models, and make any random training reproducible
from the declared seeds and inputs. The full dataset, including failures,
remains accessible through `loop.dataset` for audit. `loop.committee` returns a
new owned committee; use `with loop.committee as committee:` when inspecting it
so decoded model resources are closed. Training a model does not
by itself make it physically validated.

The model codec supplies `encode(model)` and `decode(payload)`, plus a JSON
`identity`. Encoding returns a mapping of named numerical or text arrays. The
loop hashes their names, shapes, dtypes and content and writes them through the
common atomic, checksummed NPZ container. Decoding reconstructs the exact
provider without downloading changing weights or silently casting precision.
The example implements this with five polynomial coefficients; the MACE codec
below records an architecture and a numerical state dictionary. Arbitrary
pickle objects are not a model interchange format in this API.

### Stopping criteria and honest result semantics

`max_labels` limits newly attempted reference frames, including failed labels;
rows supplied through `initial_dataset` are not charged to that new-call budget.
`labels_per_round` limits a round's acquisition and `label_batch_size` controls
its persistent chunks. Exhausting a budget does not imply low disagreement.
If selected configurations fail, their records still consume the budget and
are not silently retried on subsequent rounds. A run with no useful new
labels or no new unique candidates stops with a reason available in its result.

`uncertainty_converged` is true only when a complete sampled batch is below all
configured thresholds. The result keeps `converged=False` and
`physical_converged=False`: limited model diversity, shared model bias or
unsampled regions can produce small disagreement alongside substantial error.
Model accuracy, equilibrium sampling, bead/time-step convergence and reaction
rate convergence require separate reference checks. The supplied demo reports
true reference error and disagreement side by side rather than interpreting
one as the other.

### Interruption and exact continuation

Each sampling stage, labelling batch and trained committee has an atomic commit.
The checkpoint records the current stage and round, accumulated datasets,
selection, model versions, sampler state and RNG state. Dataset/model hashes
are checked before continuation. The scientific identity includes reference,
sampler, trainer and codec declarations, initial data/models, thresholds,
budgets, batch settings and seeds. An identity mismatch, missing committed
artifact or corrupted payload is an error. Existing input-file identities and
old monitoring checkpoints remain unchanged.

```python
# Stop after a bounded number of committed transitions, including inside a round.
loop.run(max_steps=3)

# In a new process, rebuild callbacks with the same declarations.  Omit initial
# models by passing None: persisted model payloads reconstruct the committee.
resumed = ActiveLearningLoop(
    "learning_run", None,
    reference=reference_provider,
    reference_identity=reference_specification,
    sampler=sample, sampler_identity=sampling_specification,
    trainer=train, trainer_identity=training_specification,
    model_codec=codec,
    force_threshold=5e-5, energy_threshold=1e-4,
    max_rounds=8, max_labels=256, labels_per_round=32,
    label_batch_size=8, seed=37,
)
summary = resumed.run()
```

A process can stop at any round. Already committed reference batches are not
submitted again. If interruption occurs inside an uncommitted callback, that
stage is replayed from its preceding committed inputs and RNG state. This
requires deterministic callbacks and stateless independent references; it is
not an epoch-level training checkpoint or a checkpoint of a remote quantum
chemistry process. Long-running samplers should choose manageable segment
sizes, and long-running trainers may implement their own internal restart
mechanism under the same declared identity. Use one writer per run directory.

This reference driver retains a complete dataset snapshot after each label
batch, so cumulative snapshot storage grows quadratically with the number of
labels at fixed batch size. Set explicit budgets and batch sizes accordingly;
large campaigns should use an external dataset/training service behind their
callbacks. Models are reconstructed and verified at stage boundaries, which
can dominate very short sampling segments with expensive model constructors.

## Runnable analytic-reference closed-loop pilot

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python examples/ml_lifecycle/closed_loop.py --output runs/closed-loop
# An interrupted pilot can be continued with exactly the same arguments:
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python examples/ml_lifecycle/closed_loop.py --output runs/closed-loop --resume
# A deliberate pause for restart inspection:
python examples/ml_lifecycle/closed_loop.py --output runs/paused-loop --max-steps 2
python examples/ml_lifecycle/closed_loop.py --output runs/paused-loop --resume
```

The independent analytic reference is
`V=.5*(x*x-1)**2+.5*(y*y+z*z)`. Three quartic models learn its energies and forces
through different fixed-prior ridge penalties. Augmented least squares avoids
forming the normal equations. The sampler uses 128 vectorised Metropolis
walkers with `beta=2`, a Gaussian proposal of `0.35 bohr`, and an explicit hard
box `-1.8 <= x <= 1.8 bohr` with `y=z=0`. Up to four configurations are acquired per
round and reference calls are committed in batches of two.

`diagnostics.json` compares before/after true energy and force errors with
committee disagreement on both the final sampled configurations and 1024
independent quantiles of the reference Boltzmann distribution. The diagnostic
reference calls are additional cheap analytic checks and never enter the
training dataset or its label budget. The report includes elapsed time,
platform, available logical CPUs and thread environment settings. It retains
`physical_converged=False`. A reference run of the default seed completed five
training rounds plus a final threshold-check segment, using 17 successful
labels. Its independent reference-distribution checks were:

| Quantity | Initial committee | Final committee |
| --- | ---: | ---: |
| Energy RMSE (hartree) | 0.348111 | 0.00312698 |
| Force-vector RMSE (hartree/bohr) | 0.0762803 | 0.000523456 |
| RMS force disagreement (hartree/bohr) | 0.388955 | 0.0136556 |

The measured loop time was 1.285 s on Linux x86_64 with nine visible logical CPUs,
`OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`; this includes small atomic
checkpoint writes and excludes the final held-out diagnostics. Timings are
machine-dependent. The final error/disagreement ratio is about 0.0383 on this
held-out distribution, showing that the two numerical quantities are distinct.
Neither this deliberately learnable quartic nor small disagreement on its
finite box certifies a molecular potential or rate.

The RPMD/MTP active-learning work of
[Novikov, Suleimanov and Shapeev (2018)](https://doi.org/10.1039/C8CP06037A)
is a precedent for iterating reference data and models within rate
calculations. Its extrapolation-based acquisition and reaction-rate validation
are distinct from this committee-diagnostic pilot.

## Optional small CPU MACE reference hook

`pyrpmd.active_learning_mace` supplies `MACETrainer`, `MACEModelPotential` and
`MACEModelCodec`. Dependencies are imported only when this path is requested;
the polynomial pilot and core loop need no MACE, Torch or downloaded weights.

```bash
python -m pip install -e '.[mace]'
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_active_learning_mace.py
```

```python
from pyrpmd.active_learning_mace import MACETrainer, MACEModelCodec

trainer = MACETrainer(
    symbols=["H", "H"], seeds=(17, 29), epochs=20,
    channels=4, radial_channels=8, num_bessel=3,
    cutoff=6.0, learning_rate=1e-3, fine_tune=True,
    device="cpu", dtype="float64", num_threads=1,
)
models = trainer.initial_models()
# Use models, trainer, trainer.identity and MACEModelCodec() in the loop above.
# Direct fitting uses the same callback contract:
models = trainer(successful_reference_dataset, models)
```

This reference implementation builds genuine small MACE models and performs a
full graph batch energy/force training pass per epoch. It supports fixed
composition nonperiodic molecules on CPU in float64. Other devices, precisions
and unsupported combinations fail explicitly. Public training data, cutoff and
atomic-energy options use atomic units; the MACE boundary explicitly converts
to angstrom and eV and converts returned forces back with the derivative's
length factor. Fine-tuning copies the preceding model weights, preserving the
previous committee. Each call starts a new Adam optimizer; optimizer moments
do not carry between rounds. Training seeds, architecture, loss weights,
optimizer policy and dependency versions are part of the trainer identity.
The numerical loss is `energy_weight * mean(dE_eV**2) + force_weight *
mean(dF_eV_per_angstrom**2)`: these weights refer to the converted MACE
residuals, while public datasets and returned PES quantities remain in atomic
units.

The codec stores and validates architecture, ordered state tensors and package
versions without pickle. Resumed training requires the declared runtime
versions; it does not silently migrate weights or precision. This is a compact
CPU reference implementation, not a foundation-model download service,
production distributed trainer or general MACE checkpoint converter. Pin the
training environment and use the upstream MACE tooling when those facilities
are needed.

## Additive learning and RPC

`DeltaLearningPotential(reference, correction)` implements
`V = V_reference + delta_V`, summing both energy and force. The correction
model must already predict the target-minus-reference difference in the same
units and energy convention. Passing a full target model as `correction`
would instead integrate reference plus target. For an explicit target PES,
the existing `DifferencePotential(target, reference)` constructs the required
difference.

The input form supports the same nested model specifications:

```toml
[potential]
provider = "delta"
contracted_beads = 1

[potential.reference]
provider = "builtin"
name = "harmonic"
params = {k = 0.002}

[potential.correction]
provider = "builtin"
name = "harmonic"
params = {k = 0.0002}
```

Omitting `contracted_beads` evaluates both components on all beads as an
ordinary physical PES. Supplying it constructs a `SplitPotential`; the full
bead count comes from the input. With `P` beads, `Pc` contracted beads and the
centroid-preserving projection `T`, the sampled potential part is

```{math}
U_{\rm RPC}(q)=\sum_{j=1}^{P}V_{\rm reference}(q_j)
 +\frac{P}{P_c}\sum_{a=1}^{P_c}\Delta V((Tq)_a),\qquad
F_j=F_{{\rm reference},j}+\frac{P}{P_c}\sum_a T_{aj}F_{\Delta,a}.
```

The lifted forces differentiate this complete ring energy. `Pc=P` is the exact
identity, while `Pc=1` evaluates the correction at the centroid. The cheap
reference remains evaluated on all `P` beads. `Pc<P` changes the Hamiltonian;
its accuracy must be checked by increasing `Pc`, independently of full bead
count and time-step convergence. The correction must be smooth enough for the
chosen projection. See [RPC and MTS](rpc-mts.md) for the even-cutoff convention,
periodic-coordinate requirements, numerical checks and original references.

Ordinary fixed-cell input workflows evaluate the supplied complete ring and
retain their existing integrator. RPC does not implicitly enable multiple
time stepping. The explicit Python composition is:

```python
from pyrpmd import DeltaLearningPotential
from pyrpmd.multitime import MTSDynamics

model = DeltaLearningPotential(reference, learned_correction)
split = model.split(nbeads=32, contracted_beads=3)
dyn = MTSDynamics(split, masses, beta, nbeads=32, dt=2.0, inner_steps=4)
# dyn.propagate(state, 100): 100 outer steps; four fast/ring substeps each.
```

The split borrows the model's providers. Do not close them before propagation
ends. `MTSDynamics` uses a deterministic two-level r-RESPA scheme; thermostat
and persistent restart integration are outside that API.

### Meaning of reported energies

For a beta-independent RPC potential, the materials workflow reports
`U_RPC/P`, the ordinary primitive kinetic estimator, and

```{math}
K_{\rm cv}=\frac{3N_{\rm mobile}}{2\beta}
 -\frac{1}{2P}\sum_{j,i\in\mathrm{mobile}}
 (q_{ji}-\bar q_i)\cdot F_{ji}.
```

These estimate thermodynamics of the **contracted partition function**.
`internal_energy_centroid_virial` is `U_RPC/P + K_cv`; it does not recover a
full-bead evaluation of the original target PES. The workflow evaluates the
complete ring, never a singleton centroid through a full-ring-only provider.
Summary estimators and exported trajectory metadata carry
`estimator_scope="rpc_contracted_hamiltonian"` whenever `Pc<P`, together with
the bead counts, `approximate_hamiltonian=True` and
`full_target_recomputed=False`. The same scope applies to any transport from
the contracted dynamics.

The distributed output energy rows only reconstruct the sum of `U_RPC`;
they are not physical bead energies. The kinetic centroid-virial expression
does not by itself supply a pressure virial. Contracted providers now export the
affine strain derivative of `U_RPC` when both physical components provide their
virials, and the legacy materials BZP workflow supports this contracted NPT
ensemble when both providers implement `set_cell`. See [RPC virial and
pressure](rpc-mts.md#virial-and-pressure-of-the-contracted-hamiltonian) for the
normalization and limitations. A complete set of trained weights or a
numerically stable pilot alone does not certify thermodynamic or dynamical
accuracy.

## Validation

`tests/test_ml_lifecycle.py` compares committee moments with independent member
predictions; mutates every outward array to check cache ownership; verifies
thresholds, frame caps, shape-aware replay and completed-resume provenance;
and compares monitored versus unmonitored trajectories and RNG states exactly.
Delta tests cover the full-bead identity, finite differences of the contracted
total energy, the independent harmonic centroid-virial expression and exported
Hamiltonian scope. Existing `tests/test_rpc_mts.py` provides the contraction
and multiple-time-step numerical checks.

`tests/test_active_learning_labeling.py` checks analytic reference labels, batched
failures, immutable datasets, provider restrictions and content-hash integrity.
`tests/test_active_learning.py` exercises round/batch continuation, deterministic
RNG restoration, budgets, identity checks and model/dataset tamper detection.
`tests/test_active_learning_example.py` checks independent finite-difference
forces, multi-round improvement on sampled and held-out distributions, and exact
continuation of the polynomial pilot. The MACE path is tested separately in
`tests/test_active_learning_mace.py`, with optional imports guarded so a minimal
installation does not require MACE.
