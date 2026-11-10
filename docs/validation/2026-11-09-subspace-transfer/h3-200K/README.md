# 3D H + H₂ (BKMP2), 200 K, P = 128

Inputs: `benchmarks/h3_subspace/rate_200K.inp` and `correct_200K.toml`. Results are summarized in
`../README.md` (section "3D H + H₂ at 200 K").

- `rate_summary.json`: RPMD rate, κ, PMF and warnings (k_RPMD = 7.38e−19 cm³ molecule⁻¹ s⁻¹).
- `C_S_1d.json`, `C_S_2d.json`, `subspace_pipeline.json`: subspace corrections (2.06 and 2.66) and the pipeline record.
- `run/`: PMF (`rate_pmf.*`), recrossing curve (`recrossing.*`), centroid-path moments, mean path and resolved input.

The umbrella samples (87 MB) and checkpoints (9.5 MB) are not stored in the repository.
