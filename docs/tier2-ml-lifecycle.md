# ML potential lifecycle: committees, acquisition and additive corrections

The lifecycle API keeps a fixed potential during a trajectory. A committee
provides mean energies and forces together with disagreement scores;
`ActiveLearningMonitor` collects high-disagreement geometries for subsequent
labelling. `DeltaLearningPotential` adds a learned correction to a reference
potential and exposes the existing ring-polymer contraction (RPC) fast/slow
split. These pieces can use the same nested potential specifications as the
ordinary Torch, MACE, NequIP, DeePMD, fairchem or Python providers.

The runnable examples in [`examples/ml_lifecycle/`](https://github.com/chen-yu-hao/PyRPMD/tree/main/examples/ml_lifecycle)
use inexpensive analytic models so they require no downloaded weights:

```bash
pyrpmd run examples/ml_lifecycle/committee.inp --output runs/committee
pyrpmd run examples/ml_lifecycle/delta_rpc.inp --output runs/delta_rpc
python examples/ml_lifecycle/delta_rpc_mts.py
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
No quantum-chemistry calculation, model fitting or automatic model replacement
is initiated. Labelling, training and validation are separate subsequent jobs.

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
