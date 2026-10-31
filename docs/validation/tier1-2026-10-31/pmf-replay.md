# Offline replay of the two H3 pilot PMFs

The compact payloads preserve the existing short-run CV samples exactly. They
allow the PMF-support and estimator diagnostic to be reproduced without
loading BKMP2, resampling momenta, or propagating any trajectory.
The binary sample payloads were generated and replayed locally but are not
included in this public code change. The filenames and hashes below identify
those artifacts. The packing command can create equivalent inputs from a
freshly generated pilot run.

| Payload | Streams | Samples per stream | Total scalar CV samples | Bytes |
| --- | ---: | ---: | ---: | ---: |
| `broad-pilot-pmf-samples.npz` | 222 | 200 | 44,400 | 331,034 |
| `stiff-pilot-pmf-samples.npz` | 222 | 200 | 44,400 | 324,934 |

Each payload stores the concatenated samples, stream lengths, umbrella centers
and stiffnesses, window/replica labels, original histogram edges/mask, and saved
effective sample counts. Its JSON metadata contains the original resolved
input, PMF, summary and HDF5 filenames and SHA-256 hashes. Every payload array
also has a dtype, shape and SHA-256 over its C-order bytes. The replay validates
these before analysis. Full q/p checkpoints, PES energies and original MBAR
weights are omitted; MBAR is solved again from the saved samples and biases.

## Exact replay

Given the two generated compact payloads, from the installed repository root:

```bash
python benchmarks/published_rates/diagnose_pilot_pmf.py \
  --replay docs/validation/tier1-2026-10-31/broad-pilot-pmf-samples.npz \
  --replay docs/validation/tier1-2026-10-31/stiff-pilot-pmf-samples.npz \
  --output /tmp/pmf-estimator-replay.json
```

The committed output is `pmf-estimator-diagnostic.json`. The script uses the
recorded physical beta, umbrella stiffnesses and `rate_flux` measure. Gaussian
UI uses 401 points from the reference at zero to the largest umbrella center,
1.05. Both estimators hold the original dividing surface fixed at 0.99225.
MBAR uses the original 100-bin histogram edges, `min_count=5`, overlap threshold
`1e-6`, and zero bootstrap replicates. It therefore tests the point estimator
and support, without generating a new uncertainty interval. Separate UI
replays use the first half, second half and each replica's original samples.

The two sample-payload SHA-256 values are:

```text
9a3877e4273399d509a1c5837dcf57c7728822359fbf56475d8679b847c0b969  broad-pilot-pmf-samples.npz
347fb3ee1e3f9156a2ff70899960ed9c9cfbe02f7771cd59a6c6d9d15041e184  stiff-pilot-pmf-samples.npz
```

To regenerate compact payloads from the original run directories, use:

```bash
python benchmarks/published_rates/diagnose_pilot_pmf.py \
  --run broad-pilot /path/to/h3-published-pilot/run \
  --run stiff-pilot /path/to/h3-published-pilot-stiff/run \
  --pack-directory /tmp/h3-pmf-payloads \
  --output /tmp/pmf-estimator-from-originals.json
```

Original file paths are recorded as provenance. Repacking a run under different
absolute paths changes metadata and archive hashes, while unchanged sample
arrays retain their recorded array hashes and numerical diagnostics.

## Recorded outcome

Both full-sample UI replays reproduce their saved barriers exactly. The weak
0.09500435-hartree restraint has seven masked bins around the barrier, and the
formal MBAR replay returns `ready=false`, `status=dividing_surface_masked`.
Its UI barrier is a Gaussian model value without observed histogram support
at the selected dividing surface.

For the 30-hartree restraint, MBAR returns a supported barrier of
0.02593567335344639 hartree, versus UI's 0.026014030385333054 hartree. The
difference is only −0.00007835703188666457 hartree, changing the QTST rate by
approximately 8.6%. It does not explain the approximately 8000-fold QTST
discrepancy from the published reference. The stiff-pilot replicas give UI
barriers of 0.022881959706822413 and 0.030500985736493354 hartree at the same
surface. Of 222 streams, 213 have estimated effective sample counts below 20.
The broad pilot has 222 of 222 below 20 and no sampled barrier support.

These are stationarity and support diagnostics on existing correlated data.
They establish neither physical convergence nor that sampling is the only
possible source of disagreement. The target-implied barrier in the JSON is
derived from the published QTST value and this calculation's reactant
prefactor; it is not a separately published barrier value. The independently
checked umbrella normalization is documented in
[`umbrella-normalization.md`](https://github.com/chen-yu-hao/PyRPMD/blob/feat/tier1-dynamics-efficiency-20261031/benchmarks/published_rates/umbrella-normalization.md).
