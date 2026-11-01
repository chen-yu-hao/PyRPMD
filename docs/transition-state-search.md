# From reactant and product geometries to an RPMD rate

Gas-phase bimolecular rate preparation accepts either an approximate transition
state or a pair of reactant/product geometries. Numerical preparation can refine
the former with partitioned rational-function optimization (P-RFO), or find a
saddle candidate with climbing-image nudged elastic band (CI-NEB) and then refine
it with P-RFO. The resulting transition-state geometry becomes the reference for
the existing reaction coordinate, umbrella sampling, and recrossing workflow.

This is preparation for the molecular bimolecular bond-exchange rate workflow.
A successful search must still pass the saddle and reaction-channel checks; it
does not establish convergence of a rate calculation.

## Run the endpoint example

From the repository root, with PyRPMD installed:

```bash
python -m pyrpmd prepare examples/rate/from_endpoints.inp --pilot --output preparation.json
python -m pyrpmd run examples/rate/from_endpoints.inp --output runs/from-endpoints
```

The first command performs numerical preparation and writes its diagnostics and
geometries to JSON. Without `--pilot`, `prepare` validates and describes the job
without running the numerical search. `run` performs the necessary preparation
before sampling and refuses to continue if preparation is unresolved.

The example uses [`toy_exchange.py`](../examples/rate/toy_exchange.py), an analytic
teaching model of A + BC exchanging to AB + C. The two XYZ files preserve the
three atoms' order and put both geometries in the same Cartesian frame. They
represent the two finite-separation asymptotes, with one short bond at each end.
Neither an input TS nor an explicit list of bond changes is required.

The H labels provide isotope masses and connectivity radii. This potential is
not a fitted H3 potential or a physical H + H2 rate benchmark. The short sampling
and recrossing durations are pilot settings, matching the existing TS example;
production work requires convergence studies in beads, time step, sampling,
reactant separation, and recrossing plateau.

## Supply reactant and product structures

Use both endpoint fields in `[system]`:

```toml
[system]
reactant = "reactant.xyz"
product = "product.xyz"

[potential]
provider = "python"
factory = "my_potential.py:build"
```

Paths are relative to the input file. XYZ coordinates use angstrom by default.
Each endpoint must contain the complete system with the same number, identities,
and ordering of atoms. The atoms in row `i` must correspond across the two files,
including atoms of the same element. Supply a physically meaningful geometry in
each basin, already aligned to the same Cartesian frame. The search keeps the
endpoints fixed; it does not minimize, align, or reorder them automatically.

Inline endpoint arrays can instead share `symbols` and `geometry_unit`:

```toml
[system]
symbols = ["H", "H", "H"]
geometry_unit = "bohr"
reactant = [[-3.733333333333333, 0, 0], [1.266666666666667, 0, 0], [2.466666666666667, 0, 0]]
product = [[-2.466666666666667, 0, 0], [-1.266666666666667, 0, 0], [3.733333333333333, 0, 0]]
```

These are the teaching example's endpoint coordinates. A single endpoint,
inconsistent atom identities, nonfinite coordinates, or coincident adjacent band
images is invalid. Straight Cartesian interpolation can pass through atomic
collisions or an irrelevant reaction path. In that case, supply an informed
`initial_path` rather than relying on a larger iteration budget.

## Refine an approximate transition state

Existing TS inputs continue to use `system.transition_state`:

```toml
[system]
transition_state = "ts_guess.xyz"

[reaction.preparation]
refine = true
gradient_tolerance = 1e-5
max_steps = 100
max_step = 0.2
hessian_step = 1e-4
```

When the reaction is inferred numerically, an imperfect TS is refined by default
instead of being rejected solely because its initial force is too large.
P-RFO follows an uphill mode while relaxing the other directions. Each iteration
uses a fresh finite-difference Hessian, limits each atom's displacement, and
checks the trial gradient against the local Hessian model. This is a local
saddle search; a good initial guess matters. A numerical Hessian requires `6N`
PES force evaluations for `N` atoms, in addition to ordinary and trial
evaluations.

An existing input with a complete explicit reaction definition retains its
direct preparation path unless refinement is requested with `refine = true`.
Set `refine = false` to validate a supplied TS without P-RFO. For endpoint inputs,
that setting still runs CI-NEB and the final saddle checks, but skips P-RFO.

| P-RFO setting in `reaction.preparation` | Default | Meaning |
| --- | --- | --- |
| `refine` | Enabled for numerical inference | Permit P-RFO refinement |
| `gradient_tolerance` | `1e-5` | Maximum absolute force component, hartree/bohr |
| `max_steps` | `100` | Maximum refinement steps |
| `max_step` | `0.2` | Maximum displacement of any atom per step, bohr |
| `hessian_step` | `1e-4` | Central finite-difference displacement, bohr |
| `negative_eigenvalue_tolerance` | `1e-8` | Magnitude threshold for resolved negative mass-weighted Hessian eigenvalues, in atomic units |

Bare numeric lengths are in bohr; length strings such as `"0.2 bohr"` are also
accepted. Tolerances must be finite and positive. Iteration counts must be
integers. Force convergence alone does not make a minimum or a higher-order
saddle an accepted TS.

## Configure CI-NEB

Endpoint inputs automatically run CI-NEB before TS validation or refinement:

```toml
[reaction.preparation.neb]
images = 7
spring_constant = 0.1
max_steps = 1000
gradient_tolerance = 1e-4
max_step = 0.1
climb_after = 20
```

| Setting | Default | Meaning |
| --- | --- | --- |
| `images` | `7` | Total image count, including both fixed endpoints; at least three |
| `spring_constant` | `0.1` | Parallel spring constant, hartree/bohr squared |
| `max_steps` | `1000` | Maximum band-optimization steps |
| `gradient_tolerance` | `1e-4` | Maximum absolute band-force and true climbing-image-force component, hartree/bohr |
| `max_step` | `0.1` | Maximum displacement of any atom in one update, bohr |
| `climb_after` | `20` | Number of ordinary NEB updates before activating the climbing image; zero is allowed |
| `initial_path` | Linear interpolation | Optional numeric array of shape `[images, atoms, 3]`, in bohr, including both endpoints |

The band uses an energy-weighted tangent, perpendicular PES forces, and springs
along the path. After warmup, the highest interior image climbs by reversing its
parallel PES force and removing its spring force. FIRE optimizes the resulting
forces. Convergence requires an active climbing image, small band and true
climbing-image forces, and negative curvature along the climbing tangent.
An exhausted warmup or optimization is reported as unconverged.

The P-RFO stage receives the highest image and its tangent, then checks the full
Hessian. CI-NEB's `max_force` describes projected band forces, so it is distinct
from the TS's physical-force diagnostic. More images can resolve a curved path
better but increase PES cost; they do not correct an incorrect atom mapping.

## Choose convergence settings from measured checks

The reproducible [analytic benchmark](../benchmarks/transition_state_search/analytic.py)
checks published Cerjan–Miller and Müller–Brown stationary points, independently
solved analytical Hessians, and an optional independent ASE implementation.
The lower Müller–Brown barrier did not converge within 1,000 updates with five
images in either implementation. Seven, thirteen, and twenty-one images worked
for the tested paths. Begin with at least the default seven images, then check
path resolution for the particular reaction; seven is not a universal guarantee.

The [BKMP2 benchmark](../benchmarks/transition_state_search/h3.py) tests the actual
H + H2 surface with endpoint reversal, rotation and translation, a curved initial
path, and perturbed TS guesses. For a tighter numerical reference on that surface,
the following settings reproduced the published saddle geometry and electronic
barrier while reducing the force below `1e-7` hartree/bohr:

```toml
[reaction.preparation]
gradient_tolerance = 1e-7
hessian_step = 1e-4

[reaction.preparation.neb]
images = 7
gradient_tolerance = 1e-5
```

Force convergence is not a frequency accuracy guarantee. The default force
tolerance produced a roughly `0.55 cm^-1` error in this surface's imaginary
frequency relative to an independent energy-curvature calculation; tightening
the force reduced that discrepancy below `0.03 cm^-1` for the cases tested.
Compare Hessian displacements, for example `1e-4` and `3e-5` bohr, when frequencies
matter. These frequency references were computed independently; they are not
quoted paper values.

Neither stationary-point agreement nor a passing software test establishes a
converged rate. The [endpoint rate study](../benchmarks/transition_state_search/rates.py)
separately compares the 300 K classical and 128-bead RPMD results with the
RPMDrate paper and records independent-seed, time-step, sampling, PMF-estimator,
and recrossing diagnostics. A matching central value from a short run is
insufficient when those diagnostics remain unsettled.

## Inspect preparation before sampling

The preparation report records the accepted or last attempted `transition_state`
geometry, refinement status and diagnostics, and, for endpoint searches, the
`neb` path and its image energies. Report geometries are in bohr, energies in
hartree, and forces in hartree/bohr. Check `reaction_preparation.ready` and its
diagnostics before interpreting the result as a usable rate input.

A numerical preparation becomes ready only after these checks succeed:

1. The requested CI-NEB and P-RFO searches converge.
2. The TS meets the force tolerance and has exactly one resolved negative
   mass-weighted Hessian eigenvalue.
3. Downhill traces along the unstable mode identify a usable reaction channel
   with the required two reactant fragments and paired bond changes, or a valid
   explicit reaction definition resolves the channel.

If multiple entrance directions remain possible, specify the intended reactant
fragments, choose `reaction.entry_side`, or provide the explicit reaction. The
downhill traces are numerical connectivity diagnostics, not a converged
quantum-chemistry intrinsic reaction coordinate. Inspect the reported endpoint
structures and inferred bonds for the chemistry of interest.

The search is Cartesian and unconstrained, and the rate integration is for
finite gas-phase molecular systems. It does not add support for periodic cells,
fixed atoms, arbitrary unimolecular reactions, barrierless reactions, or a
sequence of multiple barriers. Both CI-NEB and P-RFO are local methods: a
converged result does not prove that the path is globally lowest or that all
competing channels have been found.

## Connectivity radii and overrides

Automatic bond inference uses tabulated covalent radii from H through Cm.
Deuterium and tritium use the H radius. A bond is inferred when its length is
less than `bond_scale` times the sum of the two radii; the default scale is
`1.25`. These radii describe approximate connectivity rather than a bond-order
or electronic-structure calculation.

Override individual symbols when appropriate:

```toml
[reaction.preparation.covalent_radii]
H = 0.31
Fe = "1.32 angstrom"
```

Bare numeric radius overrides are in angstrom, unlike the bohr convention for
optimizer lengths. Unit strings make the radius unit explicit. Elements without
a tabulated radius require an explicit positive radius override for automatic
connectivity, or an explicit reaction definition. No arbitrary default radius
is silently substituted for heavier elements.

## Method references

- A. Banerjee, N. Adams, J. Simons, and R. Shepard, *Search for stationary points
  on surfaces*, J. Phys. Chem. **89**, 52–57 (1985).
  [doi:10.1021/j100247a015](https://doi.org/10.1021/j100247a015).
- G. Henkelman and H. Jonsson, *Improved tangent estimate in the nudged elastic
  band method for finding minimum energy paths and saddle points*, J. Chem.
  Phys. **113**, 9978–9985 (2000).
  [doi:10.1063/1.1323224](https://doi.org/10.1063/1.1323224).
- G. Henkelman, B. P. Uberuaga, and H. Jonsson, *A climbing image nudged elastic
  band method for finding saddle points and minimum energy paths*, J. Chem.
  Phys. **113**, 9901–9904 (2000).
  [doi:10.1063/1.1329672](https://doi.org/10.1063/1.1329672).
- E. Bitzek, P. Koskinen, F. Gahler, M. Moseler, and P. Gumbsch, *Structural
  relaxation made simple*, Phys. Rev. Lett. **97**, 170201 (2006).
  [doi:10.1103/PhysRevLett.97.170201](https://doi.org/10.1103/PhysRevLett.97.170201).
- B. Cordero et al., *Covalent radii revisited*, Dalton Trans., 2832–2838 (2008).
  [doi:10.1039/B801115J](https://doi.org/10.1039/B801115J).
