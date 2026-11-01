# Spectroscopy: sources, parameter trace and comparison contract

## Primary model source

J. Cao and G. A. Voth, *A new perspective on quantum time correlation
functions*, **J. Chem. Phys. 99**, 10070–10073 (1993),
[DOI 10.1063/1.465512](https://doi.org/10.1063/1.465512).
The [author-hosted full text](https://web.mit.edu/jianshucaogroup/pdfdir/JCP99-10070-1993.pdf)
was read, and the equation was checked visually on PDF page 3, journal page
10072. Equation (18) specifies

\[
V(q)=\frac{q^2}{2}+\frac{g}{24}q^4,
\qquad g=2.4,\quad m=\hbar=1,\quad\beta=5.
\]

The adjacent paragraph uses 100 harmonic-oscillator states for the exact
calculation and reports a rounded 12% increase of the ground-state energy.
These are the parameter source and the one directly reported numerical
comparison used here. The paper's Fig. 1 is a **real quantum correlation**;
our main dynamical comparison is a **Kubo-transformed position correlation**.
No figure was digitized, and none of our plotted curves is described as data
extracted from that figure.

## PA-CMD and force-matching context

- T. D. Hone, P. J. Rossky and G. A. Voth, *J. Chem. Phys.* **124**, 154103
  (2006), [DOI 10.1063/1.2186636](https://doi.org/10.1063/1.2186636).
  The [primary abstract](https://pubmed.ncbi.nlm.nih.gov/16674214/) was checked.
  Its model comparisons concern liquid para-hydrogen and ortho-deuterium.
- T. D. Hone, S. Izvekov and G. A. Voth, *J. Chem. Phys.* **122**, 054105
  (2005), [DOI 10.1063/1.1836731](https://doi.org/10.1063/1.1836731).
  The [primary abstract](https://pubmed.ncbi.nlm.nih.gov/15740308/) was checked:
  centroid forces are obtained by force matching standard PIMD data.
- C. Wu, R. Li and K. Yu, *Front. Mol. Biosci.* **9**, 851311 (2022),
  [DOI 10.3389/fmolb.2022.851311](https://doi.org/10.3389/fmolb.2022.851311).
  The publisher's [full text](https://www.frontiersin.org/journals/molecular-biosciences/articles/10.3389/fmolb.2022.851311/full),
  especially “Path-Integral Based Methods” and “Computational Details,” was
  read for the light internal masses and conditional centroid-force contract.

The original Hone full texts could not be retrieved in this environment.
Their numerical molecular settings are therefore **not** inferred or claimed
to be reproduced. This benchmark applies the implemented PA-CMD and restricted
quadratic force-matching tools to the fully specified Cao–Voth model. It does
not reproduce liquid H₂/D₂, a water force field, or measured molecular spectra.

## Independently calculated quantum reference

The script builds the Hamiltonian in an oscillator basis and repeats the
calculation with 40, 60, 100 and 160 states. The position operator is padded
before taking its fourth power and projecting, so truncation does not corrupt
the matrix elements at the top of the retained basis. The final reference uses
160 states. A separate test solves the real-space tridiagonal Hamiltonian on
two grids and applies second-order Richardson extrapolation.

Let `p_n` be normalized Boltzmann populations and `Delta_nm=E_m-E_n` for m>n.
The independently computed Kubo correlation is

\[
C_K(t)=\sum_n p_n|q_{nn}|^2+
\sum_{m>n}\frac{2(p_n-p_m)}{\beta\Delta_{nm}}
|q_{nm}|^2\cos(\Delta_{nm}t).
\]

The quantum reference passes the linear-position sum rule
`-C_K''(0)=1/(beta*m)`. The equal-time Kubo variance differs from the ordinary
quantum position variance; both are saved in `exact-basis-convergence.csv`.
Transition frequencies and Kubo/relative-IR line weights are retained in
`exact-transition-lines.csv`.

The additional `spectroscopy_static.py` calculation evaluates a symmetric
primitive transfer matrix on grids of 301 and 501 points in [-8,8]. With
`tau=beta/P`, its kernel is the Gaussian free propagator multiplied by endpoint
half-potentials. A small force field adds `-f*q` to the potential.
`Z(f)=Tr(T_f**P)` and `d²logZ/df² / beta²` give the **finite-P centroid variance**.
Symmetric fields 0.01, 0.005 and 0.0025 with Richardson extrapolation expose
field-discretization error; the two grids expose quadrature error. Six bead
counts separate primitive-action convergence from the PA-CMD sampling error.
The ordinary bead variance is evaluated as an operator insertion in the same
trace. This deterministic calculation does not use an MD trajectory or its
force implementation.

## Canonical initial ensemble and statistical meaning

Every initial ring is an independent rejection sample. The proposal is the
exact finite-P harmonic-ring Gaussian, whose normal-mode variance is

\[
\operatorname{Var}(Q_k)=\frac{P}{\beta(1+\omega_k^2)},
\qquad\omega_k=\frac{2P}{\beta}\sin(k\pi/P).
\]

Accepting with probability `exp[-beta/P * 0.1 * sum(q_b**4)]` gives precisely
the finite-P anharmonic ring distribution. The positive quartic term makes the
rejection bound rigorous. This initialization avoids uncertain equilibration
and does not imply that a finite-P distribution equals the continuum quantum
distribution. Fictitious momenta are drawn by the public PA-CMD engine.

The atom and Cartesian axes hold independent copies of the one-dimensional
Hamiltonian. Each master seed contains 96 copies in the full run. No coupling
is added between these copies. The eight master seeds define the statistical
groups; the reported standard error is the sample standard deviation of their
means divided by sqrt(8). Plot bands use the corresponding 95% Student-t
factor. These bands describe Monte Carlo sampling variability, not uncertainty
in the approximate CMD model. Common master seeds make timestep/Ω comparisons
paired; `paired-sensitivity.csv` retains the paired errors explicitly.

The position correlation uses each independent trajectory's initial centroid
and its later centroid. It does not count every time origin as an independent
experiment. Production leaves the centroid unthermostatted and thermostats
internal modes with `gamma_k=2*friction_scale*Omega`, `friction_scale=0.5`.
There are no discarded transient trajectories or selected favorable seeds.

## Spectrum, finite record and fit interpretation

The linear model dipole is `mu=q`. Actual trajectories are analyzed by the
public `relative_ir_spectrum` API. The quantum comparator is the **expected
finite-record periodogram**, calculated from the exact transition lines with
the same demeaning, Hann taper, sampling cadence, FFT size and `beta*omega**2`
factor. The resulting peak can differ from the unbroadened transition gap.
Comparing both through the same window avoids confusing this finite-record
shift with a dynamics error. Zero padding interpolates the frequency grid and
does not add physical resolution.

`ir_peak_frequency` is the interpolated maximum in the explicitly declared
fundamental region [0.7,1.7]. `phase_fit_frequency_t15` is a separate descriptive
least-squares sine/cosine phase frequency over t≤15; it is not treated as an
exact quantum line. Correlation RMSE and the normalized spectral shape error
are retained alongside the peak to expose dephasing/line-shape differences.

The f-CMD check trains the shipped **quadratic** conservative fitter on
independent finite-P bead-force samples, tests it on a separately drawn
hold-out set, and evaluates the fitted quadratic's correlation analytically.
This isolates the force basis and training uncertainty from time integration.
Its noisy instantaneous held-out force RMSE includes conditional force
fluctuations; it is not purely PMF model error. The binned held-out conditional
forces show the missing anharmonic curvature directly. This is a restricted
force-matching check, not a claim that a quadratic reproduces a general
anharmonic or molecular centroid PMF.

## Reproduction

From the repository root, with the standard scientific Python dependencies
and Matplotlib available:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python \
  benchmarks/tier2_literature/spectroscopy.py \
  --output docs/validation/tier2-literature-2026-11-01/spectroscopy

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python \
  benchmarks/tier2_literature/spectroscopy_static.py \
  --output docs/validation/tier2-literature-2026-11-01/spectroscopy

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python -m pytest -q \
  tests/test_spectroscopy_literature.py tests/test_tier2_spectroscopy.py \
  tests/test_spectroscopy_workflow.py
```

`parameters.json` records seeds, all numerical settings, code SHA256 hashes
and the checkout's HEAD. The untracked working-tree benchmark's own hash is
recorded, so HEAD is not incorrectly presented as identifying uncommitted code.
`--quick` only checks execution and explicitly marks its output as smoke data;
it is not the full evidence in this directory. No physical-convergence flag is
set by a successful numerical run.
