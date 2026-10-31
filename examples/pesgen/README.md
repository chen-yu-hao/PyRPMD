# Real PySCF / Caracal fitting examples

Build the pinned native tools as described in `docs/manual/pes-generation.md`.
Run from the repository root:

```bash
python -m pyrpmd.pesgen generate examples/pesgen/water.toml --output runs/water-pes
python -m pyrpmd.pesgen generate examples/pesgen/water.toml --output runs/water-pes --resume
python examples/pesgen/nh3_inversion.py --native-build runs/caracal-native --output runs/nh3-treq
python examples/pesgen/nh3_inversion.py --native-build runs/caracal-native --method de_evb --output runs/nh3-evb
```

Water uses an actual optimized PySCF RHF/STO-3G reference, Hirshfeld populations,
Mayer bond orders and the full Hessian. NH3 uses two optimized/reflected minima
and a nine-point constrained inversion scan. This scan is not an optimized IRC.
TREQ uses explicit native fourth-order numerical forces and is marked experimental.
These are end-to-end software checks, not production PES accuracy/rate benchmarks.

Artifacts include checksummed electronic-structure reference JSON, original
Caracal input/outputs and logs, `bundle/libpes.so`, a persistent native worker,
model/build provenance, training errors and an explicit unvalidated status.
A known complete JobSpec can consume the fitted potential through
`python -m pyrpmd.pesgen run-job`; unvalidated examples require `--allow-unvalidated`.
