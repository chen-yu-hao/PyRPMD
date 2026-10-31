# PySCF datasets, JaxPIP fitting and native dynamics

This optional molecular workflow connects explicit reference-data splits to a
JaxPIP linear PIP or PIP neural-network model, the repository's ONNX exporter,
and the existing C ABI potential provider. PySCF supplies reference energies and
gradients offline. Production force evaluation uses the ONNX Runtime C++ plugin.

The output is a fitted model for the supplied molecular domain. Every fit and
export retains `physical_validation=false`; passing an error threshold or an
export check does not establish a converged RPMD observable or a global reactive PES.
The separate [Caracal workflow](pes-generation.md) uses the same reference-data
conventions for QMDFF/dE-EVB/TREQ.

## Install the offline tools

Run these commands from the repository root in a separate environment:

```bash
python -m venv .venv-jaxpip-offline
. .venv-jaxpip-offline/bin/activate
python -m pip install -e . PySCF==2.14.0
python -m pip install -r tools/requirements-jaxpip-export.txt
```

The [pinned dependency file](https://github.com/chen-yu-hao/PyRPMD/blob/main/tools/requirements-jaxpip-export.txt)
contains JaxPIP, JAX, Equinox and the ONNX export stack. These are **offline
fitting/export dependencies**, not ordinary PyRPMD dynamics dependencies. Dataset
locking currently uses POSIX `fcntl`. Native execution separately needs the ONNX
Runtime C/C++ SDK and the compiled plugin; its Python wheel alone is insufficient.
Follow the [native ONNX build guide](https://github.com/chen-yu-hao/PyRPMD/blob/main/native/onnx/README.md).

A small executable pilot generates its own H₂ bond-scan splits and basis:

```bash
python examples/pesgen/jaxpip/h2_pipeline.py --kind linear --output runs/h2-linear
python examples/pesgen/jaxpip/h2_pipeline.py --kind nn --output runs/h2-nn
```

The RHF/STO-3G scan demonstrates the interfaces. It does not establish a general
H₂ potential, independent chemical coverage or production-level reference accuracy.

## Collect explicit reference splits

Prepare ordered multi-frame XYZ files and save this example as `data.toml`:

```toml
schema = "pyrpmd.jaxpip-data/1"

[geometries]
train = "train.xyz"
validation = "validation.xyz"
test = "test.xyz"

[qm]
method = "RHF"
basis = "sto-3g"
charge = 0
spin = 0
conv_tol = 1e-12
```

```bash
python -m pyrpmd.pesgen jaxpip-data data.toml --output runs/jaxpip-data
```

XYZ coordinates are in **angstrom**. Saved references use bohr, hartree and
**positive energy gradients** in hartree/bohr; training converts gradients to
forces with the negative sign. This route requests energies and gradients only.
The reference engine also supports UHF/RKS/UKS; DFT needs an explicit `xc`.
`spin` means `N_alpha - N_beta`, and the model multiplicity is `spin + 1`.
All frames must share ordered elements, electronic state, PySCF version and settings.
SCF convergence is required, but it does not prove state continuity or stability.

| Split | Role in this implementation |
| --- | --- |
| `train` | Required, with at least two points; determines fitted parameters, NN scaling and optimizer stopping. |
| `validation` | Optional diagnostic report; not used for early stopping or checkpoint selection. |
| `test` | Optional diagnostic report; required for the explicit error gate used by native handoff. |

Choose independent trajectory or reaction-path groups before fitting. The
collector rejects cross-split duplicates using element-pair distance signatures,
including rigid transforms and identical-element permutations. This conservative
check can also reject homometric structures; it does not detect every near-duplicate
or establish statistical independence. Keep the final test split untouched when
choosing models or thresholds.

Existing checksum-verified `ReferencePoint` JSON files can be imported with
`[references] train=["reference-a.json", "reference-b.json"]` and optional
`validation`/`test` lists instead of `[geometries]` and `[qm]`.

## Fit a linear PIP or PIP-NN

Supply a JaxPIP `.json` or `.json.gz` basis for the exact ordered composition.
The basis is a descriptor, not a trained PES. The upstream model classes are
`PolynomialLinearModel` and `PolynomialNeuralNetwork`, with their Equinox
checkpoint format preserved. This H,H example assumes a matching
diatomic basis; replace both the basis and `basis_symbols` for another molecule.
Save as `fit.toml`:

```toml
schema = "pyrpmd.jaxpip-fit/1"
dataset = "runs/jaxpip-data"
basis = "basis.json"
basis_symbols = ["H", "H"]

[fit]
kind = "linear"
position_unit = "bohr"
energy_unit = "hartree"
force_weight = 1.0
ridge = 1e-10

[acceptance]
energy_rmse_max = 3e-5
force_rmse_max = 3e-4
```

```bash
python -m pyrpmd.pesgen jaxpip-fit fit.toml --output runs/jaxpip-fit
```

Paths in both TOML files are relative to the containing input file. For PIP-NN,
set `kind="nn"`, with controls such as `hidden_layers=[16,16]`,
`activation="tanh"` or `"isru"`, `seed=0` and `max_steps=500`. The descriptor's
`alpha` uses the declared model length convention. The basis must start with a
constant; integer exponent range and atom count are checked. Deterministic
training probes test identical-element permutation invariance, without proving
the basis's full algebraic symmetry. Basis generation/selection is outside this workflow.

Linear fitting uses augmented SVD least squares for weighted energies and forces
with optional ridge regularization; the constant coefficient is unpenalized.
NN fitting uses L-BFGS-B and JAX derivatives of the energy/force objective. Only
network layers are optimized; the descriptor and training-derived scaler stay
fixed. Force labels enter through actual energy derivatives, not a separate force NN.
Both losses are per-component means normalized by the declared scales. The
training energy offset is restored in the linear model's constant term; test
energies are never independently re-zeroed. Neither route trains on validation
or test points, or selects a checkpoint using their labels.

Model units can independently be `bohr`/`angstrom` and `hartree`/`eV`.
`energy_scale`, `force_scale`, reported errors and `[acceptance]` bounds remain
in **hartree and hartree/bohr**, regardless of model units. The displayed bounds
are illustrative; choose them for the intended molecular domain before fitting.

Inspect `metrics.json`, including each split's RMSE/max error and optimizer
diagnostics. `within_requested_test_bounds=true` requires a nonempty test split,
both explicit positive bounds, and both test RMSEs within those bounds.
The fit also saves `model.eqx`, the matching basis, `probe.npy`, and a hashed
`model.json` manifest. The export probes come from training geometries and test
numerical conversion, not held-out chemical accuracy.

Exhausting the NN step budget is not reported as optimizer convergence. Dense
fitting has a `max_design_elements` guard, defaulting to 20 million; larger
problems need another fitting implementation. Repeated manual tuning against
test results is still leakage; reserve a new test set after that kind of model selection.

## Export and run through the native provider

Use the exporter shipped in this checkout:

```bash
python -m pyrpmd.pesgen jaxpip-export runs/jaxpip-fit \
  --exporter tools/export_jaxpip_onnx.py --output runs/jaxpip-native
python -m pyrpmd.pesgen jaxpip-run runs/jaxpip-native existing-job.inp \
  --library /absolute/path/libpyrpmd_onnx.so --device cpu \
  --output runs/jaxpip-dynamics
```

The export subprocess verifies dynamic float64 energy/force outputs, numerical
JAX/ONNX agreement at batches 1, 3 and 32, independent-frame behavior and force
finite differences. The wrapper checks hashes, units, ordered atoms and electronic
state against the fitted manifest. Successful output contains `model.onnx`,
`export-report.json`, fit artifacts, logs and `export.json`.

`jaxpip-run` replaces the potential section of an existing valid nonperiodic
molecular job; it does not choose a reaction coordinate or normalization. Match
its atom order, charge and multiplicity to the model. To require supported C++
propagation, put `loop_backend="native"` and `block_steps=256` in `[dynamics]`.
Unsupported Python CV/bias combinations then fail explicitly. CUDA requires a
matching GPU SDK and device dependencies; CPU validation does not qualify it.

Native handoff refuses missing or failing test bounds unless the user explicitly
adds `--allow-unvalidated` for a **pilot**. A passing gate still leaves
`physical_validation=false`: check extrapolation, asymptotes, relevant channels,
trajectory energy conservation and numerical/physical convergence separately.
The Python APIs are `collect_dataset`, `fit_jaxpip`, `export_jaxpip`,
`native_potential_spec` and `run_jaxpip_job` in `pyrpmd.pesgen`'s `jaxpip_*` modules.
The handoff and runtime paths do not import JAX or PySCF.

## Resume and preserve provenance

`jaxpip-data --resume` reuses interrupted reference work only with identical
inputs/settings. `jaxpip-fit --resume` reuses a complete, matching fit; it is not
an optimizer checkpoint continuation. Changed data, settings, dependency identity
or corrupted artifacts are rejected. Incomplete fit and export outputs require
a fresh directory; export never overwrites an existing bundle. Preserve the
dataset, fit/export manifests and logs with any subsequent dynamics results.

The [recorded local validation](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/jaxpip-generation-2026-10-31/README.md)
retains the author's environment and numerical evidence. The H₂ example uses
16/4/4 train/validation/test points. With `--exporter` and `--library`, it also
compares 100 native/reference steps at eight beads and checks for zero Python PES
callbacks inside the native block. These checks qualify interfaces, not a physical rate.

The [combined-source review](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/jaxpip-generation-2026-10-31/REVIEWED-INTEGRATION.md)
records all 44 tests passing and both real model exports completing through
native propagation with the current repository exporter.

JaxPIP source and methods: [CQPES/JaxPIP](https://github.com/CQPES/JaxPIP),
Li et al., *JCTC* 2026, DOI `10.1021/acs.jctc.6c01659`. Cite the upstream method
and the electronic-structure level used to generate your reference data.
