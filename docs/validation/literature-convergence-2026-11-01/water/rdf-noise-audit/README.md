# Analytic review of the proposed RDF maximum-difference gate

**This audit does not measure liquid-water convergence.** It tests whether the
sample count alone justifies a newly proposed rule that every radial-distribution
bin must differ by at most 0.1 between two windows. No real trajectory, RDF curve,
diffusion coefficient, or paper comparison is read by the audit program. The
original proposal remains byte-for-byte unchanged and unadopted.

The result is a concrete multiplicity warning. In a uniform independent-Poisson
count model with the proposed sampling, the largest single-bin false-alarm
probability is only 2.77%, yet the probability of at least one exceedance is
41.99% across 185 independent bins in one chain. Requiring all eight independent
chains to avoid every exceedance raises the familywise false-alarm probability
to 98.72%. **These are conditional probabilities for the specified toy model;
they are not calibrated false-alarm rates for liquid water.**

## Calculation and assumptions

The unadopted design proposes two fixed 100 ps windows, sampled every 0.5 ps,
with 216 oxygen atoms in the frozen cubic box of side 18.642822871774577 Å.
The assessed range from 2.3 to 6.0 Å contains 185 bins of width 0.02 Å. For a
shell with edges `r_lo` and `r_hi`, define its unordered-pair exposure as

```text
A = F × N(N−1)/2 × [4π/3 × (r_hi³−r_lo³)] / V.
```

This retains the finite-particle `N−1` factor. In the analytic fixture,
each window has an independent Poisson count with mean `A × g`, and its RDF
estimate is count divided by `A`. The variance of the difference between two
window estimates is therefore `2g/A`. The difference of the two Poisson counts
has a Skellam distribution; the exact strict-tail probability under that model
is `2 × Skellam.sf(floor(0.1A), A×g, A×g)`. The implementation follows the
[official SciPy Skellam documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skellam.html).

The familywise product additionally assumes independence across bins and chains.
The saved report also gives union bounds, which do not require that extra
independence but still require the stated marginal count model. No trajectory
autocorrelation time or effective sample count has been estimated.

| Excluded analytic fixture | Standard deviation of a two-window bin difference | Largest single-bin false alarm | At least one exceedance in 185 independent bins | At least one exceedance in eight such independent chains |
| --- | ---: | ---: | ---: | ---: |
| `g=1`, 200 independent frames/window | 0.01759–0.04561 | 2.7746% | 41.9877% | 98.7172% |
| `g=1`, 20 independent frames/window | 0.05563–0.14425 | 50.0307% | approximately 100% | approximately 100% |
| Synthetic `g=3` in every assessed shell, 200 independent frames/window | 0.03047–0.07901 | 20.3831% | 99.9930% | approximately 100% |

The 20-frame row illustrates tenfold lower independent-count exposure. It is
not a claim that the liquid has 20 effective frames. The `g=3` row illustrates
the occupancy dependence of absolute RDF noise; it is not a physically validated
or fitted liquid RDF. Real water has depleted core bins, peaks, pair dependencies,
correlated neighboring bins, and time correlations. These can change the actual
false-alarm probability in either direction. In particular, the `g=1` row is
**not a lower bound for water**.

## Consequence for the proposed extension

The new [version 2 proposal](../equilibration-extension-proposal-v2.json) keeps
all eight seeds, the additional 400 ps duration, the exact 100 ps branching
checkpoint, the frozen thermostat/model/time step, and the original temperature,
potential-energy, OH-length, and HOH-angle thresholds. It retains the full RDF
curves and maximum-bin differences as descriptive diagnostics. It does not adopt
the new maximum-bin tolerance as a mandatory equilibrium gate.

Any mandatory RDF-equivalence criterion must first have a justified, predeclared
assessment of block and simultaneous uncertainty and an independent review.
Fixed 5/10/20 ps block summaries can expose sensitivity to block size, but an
insufficiently precise result must remain unresolved. A confidence interval that
includes zero does not establish equivalence. Neither RDF policy nor preparation
length may be selected using current diffusion values. The coordination-difference
threshold remains a separate unadopted proposal requiring uncertainty review.

No long extension has been launched. Matching the paper's 100 ps preparation
duration did not establish equilibrium for every realized chain: seed 1102
failed the original, separately fixed potential-energy stationarity criterion.
That original failure is retained and is not reclassified by this audit.

## Reproduction and evidence

Run from the repository root into a new output directory:

```sh
python benchmarks/water/analyze_rdf_sampling_noise.py --output /tmp/water-rdf-noise-audit
python -m pytest -q tests/test_water_rdf_sampling_noise.py
```

[report.json](report.json) contains every assumption, exact probability,
source/proposal hash, and interpretation limit. [bin-noise.csv](bin-noise.csv)
contains all 185 bin calculations for each of the three fixtures. Eight focused
tests check unordered-pair normalization, the two-window variance, a strict
integer-tail identity, multiplicity, zero-occupancy shells, and invalid designs.
The generator refuses to run if the original proposal's hash changes. This is
an analytic design review, not a simulation or a new equilibrium pass result.
