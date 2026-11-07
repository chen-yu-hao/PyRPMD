# Three-regime quantum-correction validation

This validation exercises the response-level quantum-correction workflow in a
three-dimensional coupled harmonic model under three deliberately different
conditions:

| Regime | Force-constant matrix | `beta` | Observable | Fock truncation |
| --- | --- | ---: | --- | ---: |
| Zero-point dominated | `diag(0.64, 1.00, 1.44)` | 8.0 | quadratic, full `I` / subspace `diag(1,0,0)` | `6^3` |
| Quantum resonance | `[[1,.12,0],[.12,1.04,.02],[0,.02,1.6]]` | 2.5 | linear, full `[1,1,0]` / subspace `[1,0,0]` | `7^3` |
| Strong coupling | `[[1,.65,.50],[.65,1.5,.60],[.50,.60,1.8]]` | 1.5 | quadratic, full `I` / subspace `diag(1,0,0)` | `8^3` |

For each case, the quantum reference is a finite normal-mode Fock-space Kubo
sum and the RPMD reference is the analytic classical Gaussian correlation for
the same harmonic model.  The intermediate response uses the matched-subspace
inverse-response defect; an independent full-system quantum response is used
only as an audit reference.  The full-subspace algebraic limit is checked
separately.

## Reproduce

From the repository root:

```bash
PYTHONPATH=src python examples/three_regime_quantum_correction_report.py \
  --output-dir output/three-regime-report
```

The command writes a JSON summary, three PNG figures, and a five-page PDF
report.  The report is generated from the same values as the JSON summary.

## Results

| Regime | Normal frequencies | Thermal omitted bound | Relative inverse defect | Corrected vs full quantum relative RMS | Full-subspace max abs error | Dominant response peaks |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Zero-point dominated | 0.8000, 1.0000, 1.2000 | `2.10e-17` | 3.545 | 3.658 | `2.67e-15` | 1.569, 1.511, 1.627 |
| Quantum resonance | 0.9477, 1.0682, 1.2652 | `7.06e-08` | `1.61e-06` | `8.02e-05` | `1.99e-15` | 1.063, 1.012, 1.115 |
| Strong coupling | 0.7397, 1.0476, 1.6296 | `1.43e-04` | 0.0269 | 0.330 | `3.97e-15` | 1.459, 1.525, 1.392 |

The resonance case has nearly degenerate low modes and is the cleanest test of
the response workflow: the harmonic linear quantum and classical references
nearly coincide, the inverse defect is `1.61e-6`, and the corrected response
tracks the independent quantum response to `8.02e-5` relative RMS.  The
zero-point case is intentionally the most difficult: at low temperature the
quadratic response is dominated by ground-state fluctuations, and the selected
one-coordinate defect is not transferable to the full observable.  The strong
coupling case gives an intermediate failure mode: the off-diagonal force
constants produce a visible quantum/RPMD mismatch, while the one-coordinate
subspace leaves a relative corrected-response error of about 0.33.

The full-subspace error stays below `4e-15` in all cases.  This is the expected
algebraic cancellation check and is separate from the physical transferability
of the intermediate subspace.

## Interpretation and limits

The report is a controlled numerical audit, not a claim of a generally exact
reaction-rate correction.  All three runs retain `physical_converged=False`.
The Fock cutoff, time window, frequency grid, damping, and inverse/output
ridge need convergence scans before using the workflow for production data.
The current generic `DVRSystemHEOM` reference remains one-dimensional; this
three-dimensional report uses a finite normal-mode Fock reference to test the
response layer and its diagnostics in a genuinely coupled 3D model.
