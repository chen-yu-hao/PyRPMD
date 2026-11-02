# PR 56 continuation: executed numerical evidence

## Provenance and execution

Direct GitHub cloning was unavailable in the execution environment. Source was
recovered through the GitHub connector from Actions artifact **13075469909**
(`packages-ubuntu-24.04-py3.12`, run 41288734976). The ZIP SHA256 was verified:
`0547e1d6f174de5e0a89783502e06e9311f81b6bff70a634fd38dfd221e30f35`.
The artifact belongs to PR head `291b35e97c3e4b9b0dc1cadc4c9a6828c126bdac`;
build provenance identifies GitHub's synthetic merge
`24f7ca8049acb298066ff47913a831022f722b43`. The source tarball, not the Python
3.12 wheel, was used. Both native extensions were rebuilt locally on Python
3.13.5; NumPy 2.3.5 and SciPy 1.17.0 were used.

Actual local results after the implementation changes:

| Run | Result |
|---|---|
| Original three string tests, before fixes | 3 passed |
| Existing nonadiabatic/electrode focused baseline with native extensions | 281 passed |
| Revised string tests | 19 passed |
| New electrode/matrix tests | 10 passed |
| Full suite | **4383 passed, 138 skipped, 1 failed; 357 subtests passed** |

The failure is
`tests/test_report_render_validation.py::test_render_never_initializes_a_requested_gui_backend`.
It also fails in isolation. A subprocess containing only Matplotlib, with
`MPLBACKEND=TkAgg` and no DISPLAY, returns `Agg` and loads pyplot when querying
`get_backend()` **before any PyRPMD import**. This reproduces an environment
backend-selection incompatibility. The unrelated rendering code/test was not
modified or skipped to make the run appear green. Thus **the full run is not
all-green**. Optional ASE, external PES, PySCF, PLUMED and other reference tests
account for skips; see the full delivered log for individual reasons. Remote CI
for these new commits is not asserted by this local record.

[evidence.json](evidence.json) stores source/log hashes, parameters and results.
The delivered validation bundle additionally includes full stdout, the isolated
failure/probe and full coherent-population curves. All numbers below are actual
outputs, not predictions about a future run.

## Corrected string diagnostics

A one-update reproduction returned a residual of 3.6380343755449944 for a path
whose independently recomputed residual was 3.6016848764945277. The implementation
now evaluates the returned geometry and can recognize convergence on the final
allowed update. The analytic valley test converges in 27 updates, with residual
6.235496893996949e-6 and maximum transverse-coordinate error 3.113583869337604e-7.
This is a deterministic Euclidean-gradient test, not a sampled quantum PMF.

## Restricted constant-potential / NRPMD checks

The adapter is derived only for an electrode scalar common to all local diabatic
states. See [assumptions and exclusions](../../adaptive-electrochemical-string-pilot.md).
For the explicitly recorded convex synthetic charging model:

| Check | Maximum absolute error |
|---|---:|
| Optimized electron number versus analytic minimum | 2.1094e-15 |
| dE/dN - mu, hartree | 2.7756e-15 |
| Full matrix gradients versus finite differences, hartree/bohr | 1.8950e-12 |
| Constant-matrix coherent population versus analytic quantum result | 8.5487e-15 |

Independent DOP853 integration of the **same approximate Hamilton equations**
provides a numerical integrator reference for four-bead coupled motion:

| dt, atomic time | Max phase-space component error | Max energy drift, hartree |
|---:|---:|---:|
| 0.04 | 3.8997170e-3 | 5.4920659e-4 |
| 0.02 | 9.7795766e-4 | 1.3730968e-4 |
| 0.01 | 2.4467904e-4 | 3.4337865e-5 |

Halving dt reduces errors by approximately four. DOP853 agreement is not a
comparison to exact quantum nuclear dynamics; the coherent electronic limit is
not a dissipative reaction rate. These limitations are explicit in the JSON.

## Fresh PCET comparison with archived author-code output

The existing golden-rule PCET module was executed afresh (1601 grid points,
24 states). Its reference is the archived 512-point FGH/24-state result from
Hammes-Schiffer group's pyPCET commit
`031a679eab50e210aa7f76698ef5f302de31a460`. The external code was **not rerun**
in this continuation. Its paper is Hutchison et al., J. Chem. Phys. 163, 091501
(2025), DOI 10.1063/5.0284337. This is a matched synthetic-model author-code
comparison, not a transcription of a published experimental rate table.

Across 12 forward/reverse rates, the largest absolute relative difference is
**0.0620790441%**. Across six H/D KIEs it is **0.0035823538%**. At U-Uref=0 V:

| Quantity, s^-1 | Fresh PyRPMD | Archived pyPCET | Relative difference |
|---|---:|---:|---:|
| H reduction | 143575.347690 | 143522.588511 | +0.0367602% |
| H oxidation | 193246.193978 | 193179.301406 | +0.0346272% |
| D reduction | 92655.925892 | 92619.073971 | +0.0397887% |
| D oxidation | 115257.496912 | 115213.472787 | +0.0382109% |

The archive intentionally retains the reference's rounded constants. Raw
cross-code differences also include numerical discretization; this continuation
does not assign a unique error budget to these causes. Crucially, **these rates
are not outputs of the new electrode/NRPMD adapter**.

## Reproduction and acceptance status

```bash
python -m pip install -e . --no-build-isolation --no-deps
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python benchmarks/electrochemical_string/validate.py --output runs/electrode-string.json
```

The benchmark exits nonzero if its explicit numerical checks fail and preserves
the JSON for inspection. There is still no sampled quantum mean-force connection
from the electrode/NRPMD ensemble to the adaptive string, no state-dependent
reservoir dynamics, and no converged combined rate/HEOM comparison.
`physical_converged=false`, `combined_rate_available=false`, and
`all_tests_green=false` are retained rather than treating component tests as
end-to-end scientific validation.
