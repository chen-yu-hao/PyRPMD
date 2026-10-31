# State-selected H + H2 scattering: reproduction protocol

This benchmark compares H + H2(v = 0, 1; j = 0) exchange cross sections on BKMP2
with Figures 5(b) and 6(b) of Marjollet and Welsch,
[IJQC 121, e26447 (2021)](https://doi.org/10.1002/qua.26447).
The preceding method paper is [JCP 152, 194113 (2020)](https://doi.org/10.1063/5.0004179).
The driver follows the **2021 triatomic preparation**, with explicit choices where
the article does not specify an executable protocol. It does not mix the two papers' beta rules.
Read the [PES provenance](SOURCE.md), [pointwise comparison](reproduction_results/paper_comparison.csv),
and [case results](reproduction_results/all_case_results.csv) together.
Agreement and disagreement must be assessed point by point; completion of the runs is not a claim of full reproduction.

## Fixed calculation matrix

| Setting | This calculation |
| --- | --- |
| Surface and mass | BKMP2; each H has mass 1.00782503223 u |
| Harmonic reference | Equilibrium bond 1.4014719755049045 bohr; frequency 4407 cm^-1 |
| v = 0 total-energy labels | 0.55, 0.70, 0.90, 1.10 eV; bmax = 3.5 bohr |
| v = 1 total-energy labels | 0.90, 1.10 eV; bmax = 4.0 bohr |
| Main sample size | 4096 trajectories per point and method: six NE-RPMD and six QCT cases |
| Beads | NE-RPMD P = 32; harmonic QCT control P = 1 |
| Propagation | Native C++ normal-mode BAB, dt = 0.02 fs; no thermostat |
| Time budget | Initially 250 fs; unresolved trajectories continue to 1000 fs |
| Random identity | Seed 20261031; main NE IDs start at 10000, QCT IDs at 20000 |

The paper uses 40,000 trajectories per point and beta-dependent bead counts.
Fixed P = 32 is our numerical choice, checked against P = 64 at selected points;
it is not a universal bead requirement or an exact transcription of the paper's bead rounding.
The article's modified velocity-Verlet description does not establish that its splitting
is identical to our exact-free-ring normal-mode BAB integrator.

## Initial preparation and energy definitions

The driver samples the harmonic ring ensemble, adds a random-phase centroid
displacement/momentum with nominal excitation v*hbar*omega, and switches to the full PES.
Zero-point energy is already represented by the harmonic ring ensemble; it is not added twice.
H2 has one molecular vibrational mode. Its ring Fourier modes are separate coordinates.
Unlike the general `initial_ensemble` sampler, this triatomic driver removes H2's
beadwise COM position and momentum, and retains only the incoming atom's free-ring
internal motion along the incident laboratory z axis. No rigid-body rotational ring
modes are included. The molecular centroid has j = 0 and an isotropic orientation.
The general sampler retains three-dimensional noncentroid COM rings for both fragments;
these benchmark-specific projections do not change its defaults.

For NE-RPMD, a draw with centroid H-H distance below 1.078 bohr is rejected before
launch and resampled. This conditions the joint phase/coordinate/momentum ensemble;
the accepted excitation phase need not remain uniform. `preparation_draws` is recorded
per launched trajectory; total preparation rejections are sum(draws - 1).
These rejected preparations are distinct from failed or unresolved launched collisions.
The centroid cutoff does **not** forbid individual beads from having shorter bonds.
The QCT control uses an unconditioned harmonic (v + 1/2) action and uniform phase;
it is not the article's external full-PES QCT reference and is not a P = 1 thermal ensemble.

Use atomic units in the following equations (hbar = 1), with Omega corresponding to 4407 cm^-1:

```text
Ecol = Etot_label - (v + 1/2) Omega
beta0 = (2/pi) * 4.5 / (Ecol + v Omega)
beta_minus = log[(Omega - 0.01 Etot_label)/(0.01 Etot_label)] / Omega
beta = min(1500, max(beta0, beta_minus))
```

This retains the **printed minus sign** in 2021 Eq. (27); beta is in Hartree^-1.
The usual harmonic thermal-excess inequality would give a plus sign instead.
We therefore document the printed convention without claiming that it enforces an exact 1% bound.
Beta sets the preparation distribution; it is not a physical gas temperature here.
`--etot` is the nominal total-energy label, not the collision energy or a measured
energy of every ring. Launch momenta give precisely Ecol and zero total centroid momentum.

Three different energy diagnostics must remain distinct:

- The nominal internal label is (v + 1/2) Omega; conditioning and anharmonicity can shift the ensemble mean.
- The primitive internal estimator is K + mean(V - Vmin) - S, where K and the positive
  spring energy S are bead-averaged. Its trajectory values can be broad or negative.
- The propagated Hamiltonian per bead is H_P/P = K + mean(V) + S. Its drift measures
  integration error; conservation does not prove conservation of the primitive estimator or a quantum state.

The audit compares harmonic and BKMP2 potentials relative to the **same well minimum**,
removes the molecular launch COM motion, and separately measures the residual full-PES
interaction between the separated reactants. Centroid energy is another diagnostic,
not a substitute for either the primitive estimator or H_P/P.

## Launch, stopping, and cross sections

Main runs set the axial separation to d = 10 + Rg(A) + max[Rg(B), Rg(C)] bohr,
then add a transverse impact parameter sampled uniformly in b^2 and its azimuth.
The article writes its entrance distance as a vector norm; our `radial` sensitivity
instead sets z = sqrt(d^2 - b^2). This exposes an axial/radial implementation ambiguity.
The absolute bmax values, exit cutoff, persistence time, and hard time cap below
are our explicit choices; they are not all numerical settings supplied by the article.

An outcome requires exactly one centroid bond below 2.5 bohr, the other atom at
least 10 bohr from the diatom COM and moving outward, and a persistent label for 1 fs.
The original H1H2 bond is nonreactive; either exchanged bond is reactive.
Both exchange channels are counted directly, with no extra factor of two.
Checks are spaced by 0.2 fs; native blocks perform the integration and outcome checks.
Still-unresolved and failed trajectories remain in N, with endpoints/status retained.
For area A = pi(bmax^2 - bmin^2), sigma = A*Nreactive/N, with exact binomial 95% intervals.
Unknown outcomes additionally give identified bounds A*[Nreactive, Nreactive + Nunknown]/N
and a widened interval. They are never silently assigned to the nonreactive population.

## Reproduce from the repository root

Install the native kernel and GNU Fortran, build the pinned PES, then compute the
independent j = 0 radial Schrodinger reference on BKMP2. This reference uses grid
and box convergence checks; it does not replace the scattering calculation.

```bash
python -m pip install -e '.[test]' matplotlib reportlab
export PYTHONPATH=src:.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p runs/state_selected_reproduction/literature_reference
cp benchmarks/h3/reproduction_results/literature_reference/* runs/state_selected_reproduction/literature_reference/
python benchmarks/h3/build_pes.py
python benchmarks/h3/pes.py --check --output runs/state_selected_reproduction/pes-check.json
python benchmarks/h3/h2_reference.py --output-prefix runs/state_selected_reproduction/h2_reference
python benchmarks/h3/plot_h2_reference.py
python benchmarks/h3/state_selected_reproduction.py --v 0 --etot .55 .70 .90 1.10 --beads 32 --samples 4096 --workers 8 --seed 20261031 --id-offset 10000 --bmax 3.5 --output runs/state_selected_reproduction/main
python benchmarks/h3/state_selected_reproduction.py --v 1 --etot .90 1.10 --beads 32 --samples 4096 --workers 8 --seed 20261031 --id-offset 10000 --bmax 4.0 --output runs/state_selected_reproduction/main
python benchmarks/h3/state_selected_reproduction.py --method qct --v 0 --etot .55 .70 .90 1.10 --samples 4096 --workers 8 --seed 20261031 --id-offset 20000 --bmax 3.5 --output runs/state_selected_reproduction/main
python benchmarks/h3/state_selected_reproduction.py --method qct --v 1 --etot .90 1.10 --samples 4096 --workers 8 --seed 20261031 --id-offset 20000 --bmax 4.0 --output runs/state_selected_reproduction/main
```

Keep source and binaries fixed during a run; preserve recorded hashes, configurations,
and trajectory ledgers. Existing case ledgers are not overwritten; use fresh case directories
for a rerun. Workers change scheduling, not trajectory RNG identities.
The separate sensitivity cases use the following fixed workloads:

```bash
python benchmarks/h3/state_selected_sensitivity.py --kind timestep --v 0 --etot .55 --samples 256 --id-offset 10000 --bmax 3.5 --output runs/state_selected_reproduction/sensitivity/timestep-v0-Et0.550
python benchmarks/h3/state_selected_sensitivity.py --kind timestep --v 1 --etot 1.10 --samples 256 --id-offset 10000 --bmax 4.0 --output runs/state_selected_reproduction/sensitivity/timestep-v1-Et1.100
python benchmarks/h3/state_selected_sensitivity.py --kind beads --v 0 --etot .55 --samples 2048 --id-offset 30000 --bmax 3.5 --output runs/state_selected_reproduction/sensitivity/beads-v0-Et0.550
python benchmarks/h3/state_selected_sensitivity.py --kind beads --v 1 --etot .90 --samples 1024 --id-offset 30000 --bmax 4.0 --output runs/state_selected_reproduction/sensitivity/beads-v1-Et0.900
python benchmarks/h3/state_selected_sensitivity.py --kind beads --v 1 --etot 1.10 --samples 512 --id-offset 30000 --bmax 4.0 --output runs/state_selected_reproduction/sensitivity/beads-v1-Et1.100
python benchmarks/h3/state_selected_sensitivity.py --kind radial --v 0 --etot .55 --samples 256 --id-offset 10000 --bmax 3.5 --output runs/state_selected_reproduction/sensitivity/radial-v0-Et0.550
python benchmarks/h3/state_selected_sensitivity.py --kind radial --v 1 --etot 1.10 --samples 256 --id-offset 10000 --bmax 4.0 --output runs/state_selected_reproduction/sensitivity/radial-v1-Et1.100
python benchmarks/h3/state_selected_sensitivity.py --kind annulus --v 0 --etot 1.10 --samples 1024 --id-offset 40000 --bmin 3.5 --bmax 4.5 --output runs/state_selected_reproduction/sensitivity/annulus-v0-Et1.100
python benchmarks/h3/state_selected_sensitivity.py --kind annulus --v 1 --etot .90 --samples 1024 --id-offset 40000 --bmin 4.0 --bmax 5.0 --output runs/state_selected_reproduction/sensitivity/annulus-v1-Et0.900
python benchmarks/h3/state_selected_sensitivity.py --kind annulus --v 1 --etot 1.10 --samples 1024 --id-offset 40000 --bmin 4.0 --bmax 5.0 --output runs/state_selected_reproduction/sensitivity/annulus-v1-Et1.100
python benchmarks/h3/audit_state_selected_initial.py --samples 8192 4096 4096 --output runs/state_selected_reproduction/initial_audit.json
python benchmarks/h3/analyze_state_selected_reproduction.py --input runs/state_selected_reproduction
python benchmarks/h3/render_reproduction_report.py --input runs/state_selected_reproduction --font /usr/share/fonts/truetype/wqy/wqy-zenhei.ttc
```

`timestep` selects 0.01 fs; `beads` selects P = 64. Timestep/radial checks reuse a
subset of main trajectory IDs for paired comparisons. P changes the RNG stream;
annulus IDs are disjoint. Do not pool sensitivity or overlapping pilot runs into main N.
The six preliminary pilots (768 trajectories) recorded the older driver hash
`3f7787f8891e69e573990773e7f539cef0757d0224d6f8be20cd865f2d35cdba`.
That exact draft source was not retained; exact-source replay is scoped to the formal
and sensitivity runs. Pilot outcomes are archived separately and never used in the main estimates.
Zero observed annulus reactions give a finite upper bound, not proof that all larger b are irrelevant.
The zero-propagation audit fixes three accepted sample counts in order (v, Etot/eV):
(0, 0.55), (0, 0.90), (1, 1.10), with seed 20261031 and IDs starting at 900000.
Its low-variance mean is the analytic **conditional finite-P harmonic mean** plus
the sampled paired BKMP2-minus-harmonic potential difference; its SEM is over trajectories.
This control variate diagnoses preparation bias without changing a trajectory's energy.

Analysis needs the supplied `marjollet_2021_rpmd_fig5b_fig6b_digitized.csv` and
`marjollet_2021_requested_energy_interpolation.csv` under the run root's `literature_reference/`.
It writes `paper_comparison.csv`, `all_case_results.csv`, `analysis.json`, figures, and an HTML report.
The optional PDF command requires ReportLab and a readable CJK TrueType font;
the shown path is the WenQuanYi Zen Hei TTC used for this report. Pass `--font` with
the corresponding local path on another system, or use the self-contained HTML report.
Graph-reading tolerances (0.01 Angstrom^2; 0.003/0.002 eV for v = 0/1) are separate
from Monte Carlo confidence intervals; author sampling and interpolation errors are not fully known.
The initial audit is at t = 0, not a reproduction of the article's 50,000-trajectory energy-time curves.
These are approximate state preparations: no strict quantum pure state, arbitrary j,
product quantum-state assignment, or guarantee against long-flight vibrational energy redistribution.
