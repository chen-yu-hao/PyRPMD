# PySCF/JaxPIP generation validation — 2026-10-31

Local environment: Python 3.13.5, PySCF 2.14.0, JaxPIP 1.0.0 from pinned
`a18c8cca1e795202194d4082919ebd71722df125`, JAX/jaxlib 0.9.0.1,
Equinox 0.13.2, NumPy 2.3.5 and SciPy 1.17.0. C++ extension compiled locally.

- Focused dataset, fitting and export-wrapper suite: **44 passed**.
- Full local suite: **1,610 passed, 34 skipped, 153 subtests passed**, zero
  failures. Four warnings were emitted by existing tests. Skips include absent
  ASE, optional Caracal/MPI/real-PES/reference solvers and BLAS thread support.
- Actual H2 RHF/STO-3G bond-scan references: 16 training, four validation and
  four test geometries. Both actual upstream model classes trained and reloaded.
- Linear test energy RMSE: `3.4835907640525133e-7` hartree; force RMSE:
  `2.6706386578741493e-6` hartree/bohr per component.
- PIP-NN test energy RMSE: `8.271872341345815e-6` hartree; force RMSE:
  `8.001336118470982e-5` hartree/bohr per component. The NN exhausted its 250
  iteration budget; its optimizer success is **false**, not silently promoted.

See `local-h2.json` for split errors and optimizer diagnostics. Synthetic
fixtures in unit tests are explicitly labeled and are not quantum calculations.
Export-wrapper tests use a stand-in only to exercise subprocess, integrity and
failure contracts, not to establish real ONNX numerical accuracy.

The local environment lacks ONNX Runtime/jax2onnx. The dedicated **PySCF to
JaxPIP PES** Actions workflow installs actual pinned runtimes, compiles the
existing PR #34 C++ plugin, exports both freshly trained models and runs 100
steps at eight beads with native/reference parity and zero Python PES callbacks.
Remote results are reported in the PR after inspection, not inferred from the
local stand-in tests. CI retains environment versions, references, fitted
checkpoints, ONNX graphs, logs and reports as artifacts.

## Reproduce

Follow `docs/manual/jaxpip-generation.md` to install the optional environment
and native bridge. Then run:

```bash
python -m pytest -q tests/test_jaxpip_pesgen.py tests/test_jaxpip_export_link.py
python examples/pesgen/jaxpip/h2_pipeline.py --output runs/h2-linear --kind linear
python examples/pesgen/jaxpip/h2_pipeline.py --output runs/h2-nn --kind nn
```

Add `--exporter path/to/tools/export_jaxpip_onnx.py --library path/to/libpyrpmd_onnx.so`
for the full native path. No global PES transferability, rate convergence, GPU
speedup or large-dataset scaling is claimed.
