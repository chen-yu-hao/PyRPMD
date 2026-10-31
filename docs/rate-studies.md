# Temperature, isotope and convergence studies

`pyrpmd study` runs a collection of existing `rate`, `site_rate`, `surface_rate`
or thermal `drpmd` calculations. Each case has an independent seed, its own
resolved input and the existing restartable trajectory checkpoints. The study
adds Arrhenius fits, explicitly oriented kinetic isotope effects and numerical
equivalence checks. It reuses the existing potentials, parallel workers and
physical rate normalization.

## Inspect, execute and resume

```bash
pyrpmd study examples/studies/site_rate.json --plan
pyrpmd study examples/studies/site_rate.json --output runs/isotope-study
pyrpmd study examples/studies/site_rate.json --output runs/isotope-study --resume
```

The example resolves **72 independent cases**: three temperatures, two
isotopologues, four numerical settings, and three replicas. It is a periodic
cosine demonstration, not a physically calibrated diffusion calculation.
Inspect the plan and set a sampling budget appropriate to the actual system.
The `drpmd.json` example exercises all four convergence axes using a free
scattering model. Its true reaction rate is zero; logarithmic Arrhenius and KIE
fits are therefore deliberately unavailable. Replace its base job with a
validated reaction model for scientific calculations.

Study files may be JSON or TOML. `job` is a path relative to the study file,
or an inline ordinary job mapping. Python callers may also pass a `JobSpec`.

```json
{
  "schema_version": 1,
  "job": "reaction.inp",
  "temperatures": ["200 K", "250 K", "300 K", "400 K"],
  "isotopologues": {
    "H": [],
    "D": [{"atom": 0, "mass": "2.01410177812 amu"}]
  },
  "kie_reference": "H",
  "convergence": {
    "beads": [16, 32, 64],
    "timestep": ["0.1 fs", "0.05 fs", "0.025 fs"],
    "separation": ["10 angstrom", "12 angstrom", "15 angstrom"]
  },
  "replicates": 3,
  "design": "one_at_a_time",
  "relative_tolerance": 0.05,
  "absolute_tolerance": 0.0,
  "confidence_z": 1.959963984540054
}
```

The base job in this declaration must be a gas-phase `rate` or `drpmd` job,
because local basin rates have no gas-phase separation parameter. Add `bmax`
only for `drpmd`. Atom indices are zero based. Isotope masses change nuclear
masses while preserving element identities and the same Born–Oppenheimer PES.
Numeric quantities follow PyRPMD canonical units: kelvin for temperature,
atomic time for timestep, bohr for separation/bmax, electron masses for mass.
Explicit strings are recommended in user input.

## Controlled numerical refinements

| Axis | Applies to | Increasing accuracy direction |
| --- | --- | --- |
| `beads` | All four rate tasks | More beads |
| `timestep` | All four rate tasks | Smaller step |
| `separation` | `rate` | Larger RPMDrate asymptotic surface $R_\infty$ |
| `separation` | `drpmd` | Larger parent COM and collision launch distance $R_0$ |
| `bmax` | Thermal `drpmd` | Larger impact-parameter cutoff |

These directions define the comparison grid, not a guarantee of monotone
convergence. By default each axis varies with all other axes at their finest
settings. Duplicate reference cases are avoided. `design="factorial"` runs
the Cartesian product so interactions can be investigated; the automatic
summary still reports refinements along the finest reference slices.

Time-step changes preserve physical equilibration, production, parent
separation, child duration, thermostat interval, observation interval,
classification persistence/check interval and output cadence. All durations
must remain integer numbers of steps. Incompatible grids fail before dynamics.
Changing the bead count keeps the physical temperature fixed. Increasing $R_0$
translates intact fragments at the parent stage as well as at collision launch;
isotope COM shifts use the updated masses. The product classification exit
radius remains fixed to isolate the entrance-distance test. A longer approach
distance may require a larger **base** maximum propagation time; unresolved
collisions remain withheld, never interpreted as nonreactive events.

Seeds are derived from a study seed and stable case parameters. Replicas and
different temperatures/isotopes/refinements are independent; identical random
numbers are not assumed to produce correlated error cancellation. Atom masses,
all scientific settings and referenced input/plugin files enter the resume
identity. Changed inputs cannot reuse old results. The coordinator executes
cases serially; the ordinary process/GPU/bead parallelism within each case is
retained. MPI coordination, adaptive-window studies and fixed-energy
cross-section Arrhenius fits are explicitly rejected.

## Rates, fits and isotope effects

The study reads only each workflow's accepted primary rate field. A withheld
rate is recorded as unavailable, even if a diagnostic `rate_estimate` is
present. Zero or negative estimates are excluded from logarithmic fits and
cannot establish relative convergence. At the finest numerical settings,

$$\ln k(T)=\ln A-\frac{E_a}{k_B T}.$$

The fit returns $\ln A$, $A$ in the original rate units, $E_a$ in hartree and
kJ/mol, covariance and log-rate residuals. When complete positive rate standard
errors are available it uses weighted least squares with
$\sigma_{\ln k}\simeq\sigma_k/k$. Otherwise it reports an unweighted fit with
residual-based uncertainty; two unweighted points have no estimable residual
variance. Low-temperature tunnelling can make rates non-Arrhenius. Inspect the
residuals and fit range rather than interpreting a fitted slope as a universal
barrier; RPMD isotope studies also require separate bead convergence for the
different masses, as illustrated by
[Allen et al., JCP 138, 221103 (2013)](https://doi.org/10.1063/1.4811329).

`kie_reference="H"` reports $k_H/k_D$ for each other named isotopologue.
Uncertainty uses independent-rate propagation in log space. The reported
95% interval is a first-order approximation, not a replacement for a joint
bootstrap when rate distributions are broad. Numerically unrepresentable
fits/ratios are marked unavailable without discarding the other results.

Python helpers are available independently:

```python
from pyrpmd import arrhenius_fit, kinetic_isotope_effect, plan_rate_study, run_rate_study

fit = arrhenius_fit([300, 400, 500], [1e4, 3e4, 6e4], rate_unit="s^-1")
kie = kinetic_isotope_effect(4.0, 2.0, numerator_error=0.4, denominator_error=0.1)
plan = plan_rate_study("examples/studies/site_rate.json")
```

## Convergence decisions and output

The automatic test requires

$$
|k_{\mathrm{fine}}-k_{\mathrm{coarse}}|
+z\sqrt{\sigma_{\mathrm{fine}}^2+\sigma_{\mathrm{coarse}}^2}
\leq \mathrm{atol}+\mathrm{rtol}|k_{\mathrm{fine}}|.
$$

This is a conservative equivalence criterion: large uncertainty prevents a
pass. It does not equate an insignificant difference with convergence. With
three or more levels the last two adjacent refinement pairs are checked; two
levels provide only one comparison. Replicas are combined with equal weights.
The reported mean variance is the maximum of the between-replica variance of
the mean and the propagated within-run mean variance when available. dRPMD
percentile intervals are not converted silently into standard errors; use
independent study replicas for its convergence uncertainty.

`numerically_converged_on_tested_grid` is true only when every requested test
passes. `physical_converged` remains false: this grid does not by itself test
PMF domain/bins, simulation length, recrossing time, equilibration, reaction
classification or PES adequacy. Bead/time-step/$R_\infty$ studies are standard
RPMD validation requirements; see the original
[RPMDrate paper](https://doi.org/10.1016/j.cpc.2012.10.017).

The directory contains `study.json` (plan and identity), one ordinary run
directory per case, `progress.json`, `results.json` (rates, fits, KIE and each
comparison), and `rates.csv`. A failed case leaves earlier checkpoints and
progress available for an identical-plan resume.
