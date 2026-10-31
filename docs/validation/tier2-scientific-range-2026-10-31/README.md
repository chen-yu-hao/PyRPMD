# Tier 2 scientific-range validation — 2026-10-31

This dossier records local software regression and analytic-model validation of
the scientific extensions built on `main` commit
`347facf9b57d281ba7b3c78bc6a06dcc0c138149`. The combined test result is
**1,615 passed, 23 skipped, and 153 subtests passed**. Documentation and package
builds succeeded. All scientific examples retain `physical_converged=false`:
they validate algorithms and workflow behavior, not production reaction rates.

[Machine-readable summary](summary.json) includes numerical results, source and
evidence hashes, scope limits, and build/test records.

## Delivered scope

| Area | Implemented and checked | Scientific boundary |
| --- | --- | --- |
| General site rates | Smooth scalar centroid CVs, varying mass metric, constrained-parent blue-moon weighting, joint metric/transmission bootstrap | All-mobile molecular or periodic site systems; explicit finite reactant basin and continuous local CV interval |
| Coordinated studies | Temperature/isotope cases, Arrhenius fits, oriented KIEs, bead/timestep/separation/impact-cutoff refinements, restartable case execution | Equivalence on tested refinements does not prove global physical convergence |
| Thermal instanton | Cartesian action/gradient/Hessian, folded periodic orbit, crossover and mode checks | Multidimensional path/action only; quantitative prefactor currently requires exactly one active Cartesian coordinate and an explicit reactant partition function |
| Fewer-bead equilibrium sampling | Primitive and Takahashi–Imada (TI) Metropolis paths with consistent thermodynamic derivatives | Fourth-order closed-trace accuracy; no general PMF/density improvement, PIGLET implementation, or real-time RPMD speedup is claimed |
| Quantum observables | Energy, heat capacity, kinetic energy, mass substitution/fractionation, primitive open-path momentum marginal | Sampling, overlap, finite differences, momentum bandwidth/range and bead convergence remain separate checks |

The method derivations and API boundaries are documented in
[general-coordinate rates](../../general-coordinate-rates.md),
[rate studies](../../rate-studies.md), [instanton](../../instanton.md), and
[quantum statistics](../../quantum-statistics.md).

## Software and build validation

| Check | Recorded result | Evidence |
| --- | --- | --- |
| Final suite excluding local socket tests | 1,613 passed; 23 skipped; 2 deselected; 153 subtests passed; 134.85 s | [Final test log](pytest-non-socket.txt) |
| Two local socket tests, executed separately with socket access | 2 passed; 0.54 s | [Socket test log](pytest-socket.txt) |
| Sphinx 8.2.3 HTML build | Succeeded | [Build log](documentation-build.txt) |
| Source distribution and CPython 3.12 Linux wheel | Built; all four new scientific modules present; import smoke outside the checkout passed | [Package inspection and hashes](package-build.json), [wheel smoke](wheel-import-smoke.json) |

The 23 skips concern optional real BKMP2 and EAM checks, unavailable ASE/PyTorch
runtimes, and an independent optional PMF reference solver. They are enumerated
in the final test log. The package build occurred before this documentation-only
dossier was added; its archive hashes identify those exact build products.
Setuptools emitted its nonfatal direct-`setup.py` deprecation warning.

Scientific regressions include independent surface quadrature for a varying
metric, invariance under a smooth monotone CV reparameterization, joint
metric/transmission cancellation, nonlinear parent/child restart, analytic
instanton derivatives, asymmetric instanton paths, harmonic trace derivatives,
scaled isotope reweighting, and open-path covariance/Fourier normalization.

## Nonlinear-coordinate workflow pilot

The executed input is
[`examples/rate/general_coordinate.inp`](../../../examples/rate/general_coordinate.inp).
It uses a bond-difference centroid coordinate with a nonconstant mass metric in
an artificial three-atom harmonic confinement. Each mass is **2 electron masses**,
the temperature is **105,258.34 K**, and there are two beads. These deliberately
nonchemical parameters make a short numerical regression possible.

| Output | Recorded value |
| --- | ---: |
| QTST estimate | $1.8855838621\times10^{16}$ s$^{-1}$ |
| Transmission coefficient | 0.8797034113 |
| Finite-time rate estimate | $1.6587545558\times10^{16}$ s$^{-1}$ |
| Parent configurations / child trajectories | 8 / 32 |
| Maximum child endpoint energy error | $9.1810\times10^{-5}\,k_BT$ per bead |
| Complete rate uncertainty | Unavailable: PMF bootstrap disabled |

The constrained ensemble has measure

$$
\delta(s-s^*)\sqrt{z}\,e^{-\beta_P H_P},
\qquad z=\nabla s^T M^{-1}\nabla s.
$$

The implementation therefore reweights child contributions by $z^{-1/2}$
and estimates the surface metric and transmission jointly. The full numerical
snapshot is [general-coordinate-result.json](general-coordinate-result.json).
Its `converged` and `physical_converged` flags are false. A finite-time plateau
and a small energy error do not establish PMF-bin, bead, timestep, parent-chain,
or trajectory-length convergence. This model rate is not a chemical prediction.

## Temperature, isotope and convergence studies

Both example study definitions were successfully resolved:

- [Site/isotope example](../../../examples/studies/site_rate.json): **72 cases**.
- [dRPMD example](../../../examples/studies/drpmd.json): **48 cases**.

These two complete study grids were **planned, not executed**. Their role here
is input/workflow validation. Smaller automated regressions exercise actual
execution and restart. The dRPMD example uses free scattering; its zero reaction
rate cannot be logarithmically fitted to an Arrhenius law or used for a KIE.

Regression checks verify independent reproducible case seeds, unchanged physical
durations and sampling/thermostat/classification cadences under timestep changes,
consistent isotope masses and parent/launch COM placement, and fixed product
classification radius during separation scans. Only primary accepted rate
fields are consumed. Withheld estimates, unit mismatches and unresolved KIE
uncertainty cannot silently become successful fits or stop preservation of other
completed results.

The automatic refinement criterion is

$$
|k_{\rm coarse}-k_{\rm fine}|+
z_\alpha\,\mathrm{SE}(k_{\rm coarse}-k_{\rm fine})
\leq \mathrm{atol}+\mathrm{rtol}|k_{\rm fine}|.
$$

This tests numerical equivalence with the stated uncertainty assumptions. A noisy
failure to detect a difference is not reported as convergence.

## Instanton: discretization and semiclassical error

[`examples/instanton/eckart.py`](../../../examples/instanton/eckart.py) compares a
discrete ring-polymer instanton with an analytic continuum instanton and an
independent thermal integral of the exact quantum Eckart transmission. The
potential is $V(x)=0.01\,\mathrm{sech}^2(0.5x)$ hartree, the mass is 1822 electron
masses, and $T=T_c/2=41.6273466$ K.

| Beads | Action $S/\hbar$ | Absolute action error | 1D rate/velocity (m/s) |
| ---: | ---: | ---: | ---: |
| 32 | 56.81259374 | 0.08059973 | $7.6945165\times10^{-22}$ |
| 64 | 56.87316325 | 0.02003022 | $7.2374644\times10^{-22}$ |
| 128 | 56.88819309 | 0.00500038 | $7.1285080\times10^{-22}$ |
| Continuum instanton | 56.89319347 | 0 | $7.0924595\times10^{-22}$ |
| Exact quantum thermal integral | — | — | $7.5693984\times10^{-22}$ |

The primitive action error decreases approximately fourfold per bead doubling.
Every accepted discrete orbit has one negative full-ring Hessian mode. The
128-bead rate remains 0.5083% above the continuum instanton result. The continuum
instanton itself is **6.3009% below the exact quantum rate**. The closer quantum
agreement at 32 beads is error cancellation, not evidence that fewer beads are
more accurate. The quoted units describe a one-dimensional flux velocity and
must not be interpreted as a three-dimensional bimolecular coefficient.

Complete values and exact-transmission references are retained in
[instanton-results.json](instanton-results.json).

## TI trace convergence and quantum observables

The complete existing
[harmonic validation JSON](../../../examples/quantum_statistics/harmonic_validation.json)
is linked rather than duplicated. It records 30,000 production MC attempts,
3,000 burn-in attempts, stride 5 and 6,000 retained configurations per sampled
phase at $T=197.3593905$ K. Phase A has $\beta\omega=8$; phase B has the softer
harmonic confinement described in the example.

### Fewer beads for the closed trace

For one Cartesian harmonic mode at $\beta\omega=8$:

| Action | Beads | Absolute error in $\log Z$ |
| --- | ---: | ---: |
| Primitive | 128 | $6.51192770\times10^{-4}$ |
| TI | 16 | $3.34897081\times10^{-4}$ |

TI uses **eight times fewer beads** and has about 0.5143 times the trace error
in this analytic comparison. This is a bead-count/error comparison, not a measured
wall-clock speedup. It does not establish fewer-bead accuracy for arbitrary
coordinate histograms, PMFs, or real-time RPMD transmission coefficients.

### Heat-capacity variance reduction and remaining uncertainty

The scaled-coordinate estimator evaluates the temperature derivative through
paired finite differences of the complete primitive or TI action, including the
coordinate Jacobian. The example compares $h=0.001$ and $h/2=0.0005$, at a cost
of four additional PES evaluations per saved configuration.

| Phase | Thermodynamic $C_V/k_B$ ± conditional SE | Scaled-coordinate $C_V/k_B$ ± conditional SE | Estimated variance reduction |
| --- | ---: | ---: | ---: |
| A | −0.04436 ± 0.41951 | −0.18600 ± 0.22985 | 3.33× |
| B | 0.22926 ± 0.43073 | 0.79431 ± 0.09700 | 19.72× |

These variance factors are the squared ratios of the reported conditional
standard errors on the same stored samples. They do not include the extra PES
cost and are not universal efficiency factors. Phase A's low-temperature heat
capacity remains **statistically unresolved**; its exact quantum value is
0.0644521 and its finite-16-bead TI value is 0.0837391. Negative noisy estimates
are retained rather than clipped. For phase B the corresponding reference values
are 0.912262 and 0.912896. The $h$-versus-$h/2$ heat differences are only
$3.10\times10^{-7}$ and $1.91\times10^{-8}$, respectively, which addresses
finite-difference sensitivity without eliminating sampling error.

### Fractionation and momentum

The sampled phase fractionation is $\log\alpha_{A/B}=1.6071933\pm0.0209432$
with conditional independent-phase uncertainty, versus the exact harmonic value
1.6395343. Reweighting ESS diagnostics are retained; they cannot reveal unsampled
distribution tails.

The momentum marginal uses **primitive, one-component-open paths with 32 links**,
not fictitious PIMD momenta. Its integral on the supplied finite grid is 1.0033833,
with negative noise area 0.0086347. Neither normalization nor negative values are
silently repaired. Open-path sampling, momentum range and kernel bandwidth require
separate convergence checks.

## Reproduction entry points

From a normal installed checkout with the test dependencies available:

```bash
python -m pytest -q
python examples/instanton/eckart.py
python examples/quantum_statistics/harmonic_validation.py
pyrpmd run examples/rate/general_coordinate.inp --output runs/general-coordinate
pyrpmd study examples/studies/site_rate.json --plan
pyrpmd study examples/studies/drpmd.json --plan
```

Use a fresh output directory for the nonlinear-coordinate example or the ordinary
explicit resume option. The full suite includes optional-runtime and local-socket
checks; the recorded environment limitations are listed above.
