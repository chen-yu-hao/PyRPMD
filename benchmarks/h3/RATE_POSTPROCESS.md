# H + H2: two-dimensional rate postprocessing regression

This benchmark tests the complete offline rate postprocessor using actual
BKMP2 ring-polymer sampling and signed, paired child trajectories. It does
not compare a short calculation against a literature rate or claim physical
convergence.

## Run and reproduce

From the repository root, with the native extension and GNU Fortran installed:

```bash
python benchmarks/h3/build_pes.py
python benchmarks/h3/rate_postprocess.py \
  --output runs/h3-rate-postprocess-quick --workers 2 --profile quick
python benchmarks/h3/rate_postprocess.py \
  --output runs/h3-rate-postprocess-pilot --workers 2 --profile pilot

pyrpmd rate-postprocess runs/h3-rate-postprocess-pilot/run \
  --spec runs/h3-rate-postprocess-pilot/rate-postprocess-spec.json \
  --output runs/h3-rate-postprocess-pilot/offline-reanalysis.json
```

`build_pes.py --source /path/to/h3_bkmp2.f` accepts the exact checksum-pinned
offline source. No PES source is redistributed by this benchmark. Repeating
the benchmark in the same output directory validates and reuses compatible
umbrella/parent checkpoints; the child calculations are deterministically
recomputed. Use a fresh directory when changing the physical settings.

The final `pyrpmd rate-postprocess` command only reads saved data; it performs
no dynamics and does not load a PES. The specification and all input files are
hashed into the output provenance. The benchmark itself calls this same file
adapter as an end-to-end check.

## What is sampled

The two coordinates are `(xi, r12)`, where `xi` is the centroid RPMDrate union
coordinate and `r12` is the original H1-H2 bond distance. The two equivalent
exchange arrangements are explicitly included in `xi`:

```
s1 = r12 - min(r01, r02)
```

The sampling measure is deliberately **`rate_flux` with `flux_axis=0`**.
The existing `FluxMetricBias` applies to `xi` only. The second coordinate is a
transverse observable; its thermal velocity must not be multiplied in again.
A generic `(r01,r12)` density FES is a different measure and cannot be passed
to this formula by relabelling it.

Umbrellas span `xi=-0.05..1.1`. The incoming reference is `xi=0`, corresponding
to the free-reactant sphere at `Rinf=10 bohr`, with prefactor
`4*pi*Rinf**2/sqrt(2*pi*beta*mu)`. The test dividing surface is held at `xi=1`.
Both points are exact histogram-bin centers. The static postprocessor takes
the ratio of the full transverse flux integrals and multiplies by this
known incoming prefactor. No additional factor of two is applied.

The histogram has **23 xi bins and two unequal transverse bins** with edges
`[0, 1.55, 4] bohr`. This intentionally inexpensive histogram tests the
distinction between probability density and probability mass. It is not a
high-resolution two-dimensional free-energy map. Each endpoint slice must
have support in both transverse bins; unsupported bins cause rejection.
The finite transverse domain and its unobserved tails are not assumed
converged. The report gives observed out-of-domain sample and weight fractions.

| Setting | quick | pilot |
| --- | ---: | ---: |
| Temperature | 1500 K | 1500 K |
| Beads | 8 | 16 |
| Time step | 0.1 fs | 0.1 fs |
| Umbrella windows | 48 | 72 |
| Replicas per window | 2 | 2 |
| Equilibration per stream | 20 fs | 100 fs |
| Production per stream | 100 fs | 300 fs |
| Sampling interval | 0.5 fs | 0.5 fs |
| Independent parent chains | 2 | 2 |
| Parent configurations | 32 | 64 |
| Paired child trajectories | 128 | 256 |
| Child duration | 80 fs | 80 fs |

The profiles change both bead number and sampling. Their difference is not
an isolated bead-convergence test.

## Independent checks and product labels

1. A direct one-dimensional histogram of the **same MBAR target weights**
   computes the `xi=1`/`xi=0` marginal ratio without using the free-energy
   integration code. It must agree with the two-dimensional integral to
   floating-point precision. This is an independent algebraic check, not
   independent physical sampling.
2. Actual constrained parents spawn Maxwell momenta `+p` and `-p`. Every
   numerator retains its signed initial flux weight. The two time-dependent
   indicators partition the union product side according to the shorter
   forming bond. Their sum must reproduce the existing scalar recrossing
   estimator at every time point.
3. Labels `H0H1+H2` and `H0H2+H1` mean two **atom-labelled arrangements** of
   the same chemical reaction H + H2, not two different product species.
   No 50:50 averaging is imposed.
4. Each product-side endpoint is separately checked for a forming bond
   shorter than 2.5 bohr, a third-atom-to-diatom-COM separation greater than
   5 bohr, and outward motion. These diagnostics do not establish sustained
   product commitment. The branches remain finite-time union-side estimates.
5. Endpoint energy errors use `abs(delta H)/(P*kB*T)`, avoiding dependence
   on the absolute PES energy zero.

The short study disables PMF bootstrap. Reported dynamic errors hold the
static prefactor fixed; they do not constitute complete rate uncertainty.
Two independent parent chains also provide weak information about branch
mixing, even if a tail plateau is detected. The machine-readable reports
retain `converged=false`.

## Outputs

- `run/fes.npz`, `run/fes.h5`: joint FES, target weights, masks, raw CV streams,
  measure metadata and checkpoints.
- `static-rate.json`: finite-domain TST estimate and normalization.
- `dynamics/channel_recrossing.npz`: parent-grouped signed channel numerators,
  common denominator, time grid and matched surface identity.
- `rate-postprocess-spec.json`: explicit offline normalization and channel
  artifact specification.
- `rate-postprocess.json`: public offline file-adapter result and source hashes.
- `benchmark.json`: numerical comparisons, actual dynamics, commitment
  diagnostics and scientific limitations.

The checked local quick/pilot results are recorded in
[`docs/validation/rate-postprocess-2026-10-31/h3-summary.json`](../../docs/validation/rate-postprocess-2026-10-31/h3-summary.json).
