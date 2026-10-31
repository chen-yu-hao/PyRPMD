# JaxPIP generation integration review — 2026-10-31

The combined implementation passed all **44 dataset, fitting and export-wrapper
tests**, with no skipped tests, and both real H₂ pipelines completed through
native propagation. [Machine-readable results](review-integration.json) record
the tested source digests, package versions, export checks and model diagnostics.
The author's earlier [local record](README.md) and `local-h2.json` are preserved.

## Integration change

The generation workflow originally checked out the ONNX bridge separately at
`59315cd20af727a60077b327e66df30ff3ae8e02`. That exporter passed the linear model
but failed the PIP-NN symbolic batch conversion. Author commit
`ae490fbef2a36daebfedad1e6a89e7f6be99c8a7` updates that dependency to the reviewed
symbolic-Pad compatibility fix `5c7c55aa7622dbadc37c47a8ed65c6b666eade02`.

The integrated workflow uses the exporter, pinned Python requirements and native
plugin from the same checked-out repository. It therefore tests the actual
combined source, including that compatibility fix. The author's latest commit
remains an ancestor. Exporter and plugin changes also trigger this workflow.
No force equations, float64 contract, finite-difference check, parity tolerance
or dynamic-batch check was changed or disabled.

## Measured results

Both calculations use actual H₂ RHF/STO-3G reference data with 16 training,
four validation and four test geometries. The native comparison uses eight
beads and 100 integration steps.

| Quantity | Linear PIP | PIP-NN |
| --- | ---: | ---: |
| Test energy RMSE, hartree | 3.4835907640525133e-7 | 8.271872341345815e-6 |
| Test force RMSE, hartree/bohr per component | 2.6706386578741493e-6 | 8.001336118470982e-5 |
| Maximum ONNX energy discrepancy, model units | 0 | 0 |
| Maximum ONNX force discrepancy, model units | 0 | 6.661338147750939e-16 |
| Force finite-difference discrepancy, model units | 1.0670170302873316e-10 | 2.1923868276374492e-10 |
| Dynamic batch sizes checked | 1, 3, 32 | 1, 3, 32 |
| Python PES callbacks inside native block | 0 | 0 |
| Maximum native/reference coordinate discrepancy, bohr | 0 | 0 |
| Maximum native/reference momentum discrepancy, atomic units | 0 | 0 |

The neural-network optimizer exhausted its 250-iteration budget and retains
`success=false`. Both models pass the requested test-set RMSE bounds; neither
that gate nor export/native parity establishes a globally transferable reactive
potential or a converged physical RPMD rate. `physical_validation=false` remains
explicit throughout the reports. The separately reported validation-set errors
are not used for fitting, stopping or model selection.

The test environment used Python 3.12.14, PySCF 2.14.0, JAX/jaxlib 0.9.0,
JaxPIP 1.0.0 at the repository's pinned upstream revision, Equinox 0.13.2,
jax2onnx 0.17.1, ONNX 1.20.1 and ONNX Runtime 1.24.1. The native SDK matched
the SHA256 already required by the workflow. The complete version list is in
the JSON record.

## Reproduce

Install the optional environment and build the native plugin as described in
the [generation guide](../../manual/jaxpip-generation.md), then run from the
repository root:

```bash
python -m pytest -q tests/test_jaxpip_pesgen.py tests/test_jaxpip_export_link.py
python examples/pesgen/jaxpip/h2_pipeline.py --output runs/h2-linear-review \
  --kind linear --exporter tools/export_jaxpip_onnx.py \
  --library build/onnx/libpyrpmd_onnx.so
python examples/pesgen/jaxpip/h2_pipeline.py --output runs/h2-nn-review \
  --kind nn --exporter tools/export_jaxpip_onnx.py \
  --library build/onnx/libpyrpmd_onnx.so
```

Use fresh output directories. The dedicated workflow runs the same commands
and retains references, model checkpoints, export logs and native reports as
an artifact, including when a stage fails.
