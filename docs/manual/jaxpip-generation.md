# PySCF to JaxPIP potential generation

This optional offline route shares the checked PySCF reference format with
[Caracal PES generation](pes-generation.md). It fits the actual upstream
JaxPIP `PolynomialLinearModel` or `PolynomialNeuralNetwork`, preserves its
Equinox checkpoint format, then uses the existing native ONNX bridge. It does
not convert QMDFF into PIP or replace any molecular-dynamics integrator.

The initial implementation is stacked on the PES-generation change (#35).
Native export/runtime reuse #34 rather than copying its code. The integration
workflow pins bridge commit `5c7c55aa7622dbadc37c47a8ed65c6b666eade02` and JaxPIP
commit `a18c8cca1e795202194d4082919ebd71722df125`.

## Optional environment

Collecting new references needs PySCF. Fitting needs JAX, JaxPIP and Equinox;
SciPy supplies the dense solvers. Export additionally needs the bridge's pinned
export environment. None is required when a prepared native PES is used in MD.
No Caracal installation is needed for this route.

```bash
git clone https://github.com/chen-yu-hao/PyRPMD.git external/onnx-bridge
git -C external/onnx-bridge checkout 5c7c55aa7622dbadc37c47a8ed65c6b666eade02
python -m pip install -e '.[test]' 'pyscf==2.14.0'
python -m pip install -r external/onnx-bridge/tools/requirements-jaxpip-export.txt
```

Compile `external/onnx-bridge/native/onnx` using its README. The resulting
`libpyrpmd_onnx.so` is supplied explicitly; this workflow does not download or
build runtime libraries implicitly.

## 1. Collect or reuse reference data

```toml
schema = "pyrpmd.jaxpip-data/1"
[geometries]
train = "train.xyz"
validation = "validation.xyz"
test = "test.xyz"
[qm]
method = "RHF"
basis = "sto-3g"
conv_tol = 1e-12
```

XYZ coordinates are in angstrom. Internally every reference uses bohr, hartree
and **positive energy gradients**. Forces for fitting are their negatives.
New PySCF calls request only energies and gradients, not expensive Hessians or
population analyses. The existing PES-generation atomic cache is reused.

```bash
python -m pyrpmd.pesgen jaxpip-data data.toml --output runs/pip-data
python -m pyrpmd.pesgen jaxpip-data data.toml --output runs/pip-data --resume
```

To reuse checked reference files from a Caracal preparation instead, replace
`[geometries]` and `[qm]` with:

```toml
[references]
train = ["references/train0.json", "references/train1.json"]
test = ["references/test0.json"]
```

Only one mode is permitted. Imported data retain their original PySCF settings
and version. All points must share atom ordering, electronic state, method,
basis and reference convention, and report converged SCF. At least two training
points are required. Validation and test splits are optional but explicit.

The collector rejects cross-split duplicates using element-pair distance
multisets, including rigid transformations and atom permutations. This
conservative check can also reject distinct homometric structures. It does not
prove statistical independence: split whole independent trajectories or path
families before collection, rather than adjacent frames from one trajectory.

Input/implementation fingerprints control resume. Every reference and completed
manifest is checksummed. Changed inputs and corrupted artifacts are rejected;
concurrent writers cannot share an output directory.

## 2. Fit a linear PIP or PIP-NN

```toml
schema = "pyrpmd.jaxpip-fit/1"
dataset = "runs/pip-data"
basis = "basis.json.gz"
basis_symbols = ["H", "H"]
[fit]
kind = "linear"              # or "nn"
alpha = 1.0                  # in the declared model length convention
position_unit = "bohr"       # or "angstrom"
energy_unit = "hartree"      # or "eV"
energy_scale = 0.01          # always hartree
force_scale = 0.01           # always hartree/bohr
force_weight = 1.0
ridge = 1e-10                # linear only; constant coefficient is unpenalized
hidden_layers = [16, 16]     # NN only
activation = "tanh"          # NN: tanh or isru
seed = 0
max_steps = 500
[acceptance]
energy_rmse_max = 3e-5       # hartree, user-selected domain accuracy bound
force_rmse_max = 3e-4        # hartree/bohr, per Cartesian component
```

```bash
python -m pyrpmd.pesgen jaxpip-fit fit.toml --output runs/pip-fit
```

Supply an upstream JaxPIP JSON basis with a leading constant. The declared
`basis_symbols` must exactly match the dataset order. Integer exponent range and
atom count are checked; deterministic training probes test identical-element
permutation invariance. These probes are not a proof of a basis's full algebraic
symmetry. Basis generation/selection is outside this change.

Linear fitting uses augmented SVD least squares for weighted energies and
forces, with optional ridge regularization. PIP-NN fitting uses L-BFGS-B and
JAX parameter derivatives of the energy/force objective. Only network layers
are optimized: its descriptor and train-derived scaler remain fixed. Force
labels enter through the actual model derivatives, not a separate force NN.

Both losses are per-component means normalized by the declared scales. All
model-unit combinations convert energies, coordinates and forces consistently.
The training energy offset is restored in the linear model's constant term.
Neither validation nor test labels affect scaling, fitting, stopping or model
selection. They are evaluated only after fitting. No test-set re-zeroing is
performed. Repeated manual hyperparameter selection on test results would still
constitute leakage; reserve a new test set for that use case.

`metrics.json` reports separate train/validation/test errors in atomic units and
the optimizer's actual termination status. Exhausting the NN step budget is not
reported as optimizer convergence. Dense fitting has a `max_design_elements`
guard (default 20 million); large datasets need another fitting implementation.
Completed fits can be checked with `--resume`; interrupted fits require a fresh
output directory and do not claim optimizer-state resume.

## 3. Export with the existing bridge

```bash
python -m pyrpmd.pesgen jaxpip-export runs/pip-fit \
  --exporter external/onnx-bridge/tools/export_jaxpip_onnx.py \
  --output runs/pip-native
```

The explicit exporter is trusted executable code. It runs in a separate Python
process so converter instrumentation cannot affect fitting. The bridge checks
JAX/ONNX energy-force parity, force finite differences and dynamic batch sizes
1, 3 and 32. This wrapper verifies source hashes, atom/state/unit contracts and
the exported file digest before committing a portable bundle. Failures retain
logs and do not publish a completed manifest. Existing outputs are not replaced.
Export parity is distinct from quantum-reference accuracy.

## 4. Use an existing PyRPMD job

```bash
python -m pyrpmd.pesgen jaxpip-run runs/pip-native job.inp \
  --library /absolute/path/libpyrpmd_onnx.so --output runs/pip-dynamics
```

The Python API `native_potential_spec(directory, library)` returns the ordinary
`provider="cabi"` mapping. `run_jaxpip_job` replaces the potential in a supplied
JobSpec and calls the existing runner. Without passing user-defined bounds on
an explicit test split, pilots require `--allow-unvalidated`. Passing those
bounds is not validation of global transferability or a physical RPMD rate;
`physical_validation` remains false in every generated model report.

Inference stays in the existing batched ONNX C++ plugin and C++ propagation
loop. JAX and PySCF are offline only. Independent model contexts, thread control,
CPU/CUDA choice and runtime error handling belong to the bridge. CPU is the
validated target here. There is no new GPU-resident integrator, automatic
multi-trajectory packing, periodic virial or nonadiabatic model in this change.

## Reproducible pilot

`examples/pesgen/jaxpip/h2_pipeline.py` runs actual H2 RHF/STO-3G references,
linear or neural-network fitting, and optionally export plus 100 native steps
at eight beads. With `--exporter` and `--library`, it compares native/reference
propagation and profiles zero Python PES callbacks inside the native block.
The 16/4/4-point bond scan is an interface/local-fit example, not a global
reactive PES, high-level electronic-structure benchmark or converged RPMD rate.

JaxPIP source and methods: [CQPES/JaxPIP](https://github.com/CQPES/JaxPIP),
Li et al., *JCTC* 2026, DOI `10.1021/acs.jctc.6c01659`. Cite the upstream method
and your electronic-structure level when using fitted potentials.
