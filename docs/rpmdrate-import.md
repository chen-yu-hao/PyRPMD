---
orphan: true
---

# Importing RPMDrate rate inputs

`pyrpmd import-rpmdrate` records a trusted RPMDrate Python input and writes a
schema-2 `rate.inp`, `ts.xyz`, and `import_report.json`. It does not load a PES,
initialize a PES, run dynamics, or import the `rpmdrate` package. Temperature is
in kelvin; the bead count is an explicit command-line argument.

```bash
pyrpmd import-rpmdrate examples/rpmdrate_import/input.py \
  --temperature 1000 --beads 4 --output runs/rpmdrate-import
pyrpmd validate runs/rpmdrate-import/rate.inp
pyrpmd prepare runs/rpmdrate-import/rate.inp
```

The destination must be nonexistent or empty. Static validation and preparation
do not require the PES. A subsequent numerical preparation or simulation does.
Use `--pes-module /absolute/path/to/PES.so` to select the compiled f2py module
that the converted job will use. Conversion does not inspect or execute it.
Without this option, a unique local Python/shared-library file is recorded by
absolute path; otherwise the original importable module name is retained. That
module must be on the simulation process's Python search path. Initializers
are enabled only when the input explicitly imports `initialize_potential`.

The input API is pinned to
[chen-yu-hao/RPMDrate at 710670c](https://github.com/chen-yu-hao/RPMDrate/tree/710670c6f753b1159c0147c817e82998c2cc8486).
The bundled example is independently written, uses small conversion-test
budgets, and does not establish a physical rate.

## Mapping and units

| RPMDrate input | PyRPMD output | Conversion |
| --- | --- | --- |
| `reactants.atoms` | XYZ symbols | Atom order retained |
| `reactant1Atoms`, `reactant2Atoms` | `reaction.reactant_fragments` | Subtract one from every index; require a complete partition |
| `Rinf` | `reaction.rinf` | Explicit length quantity converted to bohr |
| `transitionState.geometry` | `ts.xyz`, `system.geometry_unit` | Coordinates converted to bohr; orientation retained |
| `formingBonds`, `breakingBonds` | Reaction bond lists | Subtract one; reject duplicates and invalid indices |
| `equivalentTransitionState` | `reaction.equivalent_paths` | Ordered bond correspondences define atom permutations; `symmetry_factor=1` avoids double counting |
| `get_potential`, `initialize_potential` | `potential` with `provider="rpmdrate"` | Module and symbol references only |
| `conductUmbrellaSampling.dt` | `dynamics.timestep` | Time in atomic units |
| `Window.xi`, `Window.kforce` | Explicit `umbrella.windows` | Center and individual normalized force constant |
| Window times, `trajectories` | Window-local `sampling` | Exact step counts and `chains_per_window`; every production step is observed |
| `thermostat('Andersen', samplingTime=...)` | Umbrella and parent thermostat tables | Fixed global refresh with an explicit step interval |
| `computePotentialOfMeanForce.bins` | `umbrella.analysis.integration_points` | UI grid point count; also used as the histogram bin count |
| `xi_min`, `xi_max` | `umbrella.analysis.ranges` | Histogram range; the UI range limitation below is reported separately |
| Recrossing times | `recrossing.parent`, `recrossing.children` | Equilibration, draw interval and child duration |
| `childTrajectories`, `childrenPerSampling` | Parent configurations, child pairs | One parent chain; configurations = total / children; pairs = children / 2 |
| `xi_current` | `rate.dividing_surface` | Explicit coordinate; omission selects the PMF maximum |
| `computeRateCoefficient()` | `task="rate"` | Standard rate workflow |
| `random_seed` | `seed` | Integer retained; this does not reproduce RPMDrate's random stream |

The XYZ file explicitly uses **bohr**, declared by `system.geometry_unit` in
`rate.inp`. Tools that assume all XYZ coordinates are in angstrom must convert
it before use. Supported length/time units follow PyRPMD's quantity parser,
including bohr, angstrom, nm, fs and ps. Unknown or dimensionally inconsistent
units are errors. Atom indices, trajectory counts and time-step alignment are
checked before files are written.

The pinned Fortran routine `add_umbrella_potential` adds
`0.5 * kforce * (xi - xi_window)**2` to **each bead**. Therefore

$$
a_i = kforce_i/T,\qquad K_i=a_iT\quad\text{(hartree)}.
$$

For `kforce=0.1*T` at 1000 K, the output coefficient is 0.1 hartree/K and the
resolved spring is **100 hartree**. Do not insert another bead factor or a
Boltzmann constant. The report includes each source spring, normalized
coefficient and resolved window stiffness. Because the Python script was
evaluated at the supplied temperature, rerun the importer when changing T;
arbitrary Python temperature dependence cannot be inferred from one execution.

## Unsupported settings and explicit replacements

By default, any unmapped setting causes a nonzero exit status and **no output
files**, including no report file. The error lists every recorded unmapped
setting and its proposed replacement. After reviewing those differences:

```bash
pyrpmd import-rpmdrate /path/to/RPMDrate/examples/H+H2/input.py \
  --temperature 1000 --beads 4 --allow-unmapped --output runs/h3-import
```

Each replacement appears in both the report's `unmapped` list and a top-of-file
`# UNMAPPED:` comment. These outputs are structurally valid starting points,
not equivalent sampling protocols. The four pinned upstream examples all
require this option:

- `generateUmbrellaConfigurations` performs sequential single-bead seeding in
  RPMDrate. PyRPMD has no equivalent schema-2 stage. Its windows instead start
  from the supplied TS and use their declared equilibration. All requested
  seeding parameters remain in the report; they are not folded into production.
- RPMDrate passes the final geometry of a window trajectory to its next
  trajectory. PyRPMD's independent chains preserve the requested count and
  per-trajectory times but do not preserve that correlation.
- Rate umbrella/constrained-parent sampling does not support the GLE matrices.
  Explicit permission selects the existing Andersen default: fixed global
  refresh every 10 steps in both stages. Likewise, omitted Andersen
  `samplingTime` means `floor(sqrt(stage steps))` upstream, which cannot be
  represented by one continuous PyRPMD clock. Stage-boundary clock differences
  are reported when an explicit refresh interval does not divide burn-in.
- PyRPMD UI integrates from zero to the largest window center. An upstream
  `xi_min`/`xi_max` pair differing from those endpoints is unmapped: recording
  a histogram range does not change the UI integration grid. Even for matched
  endpoints, the upstream midpoint PMF and PyRPMD grid differ in discretization.

The report also identifies unsupported window rejection bounds, PMF window
subsets, RPMDrate trajectory-save formats, and differing stage timesteps. This
converter emits one common timestep. An odd `childrenPerSampling`, incomplete
child batch, malformed geometry, repeated stage declaration, or ambiguous PES
reference is rejected even with `--allow-unmapped`. dRPMD inputs and output or
checkpoint migration are outside this command's scope.

PyRPMD uses its own RNG, isotope-mass table, Cayley integrator, TS-validation
policy and statistical estimators. These are not bitwise RPMDrate replay.
The report's `defaults` records the selected seed, masses, integrator,
preparation policy, symmetry factor and first-parent-draw convention. In
PyRPMD that draw follows one full draw interval after equilibration; upstream
draws immediately. `all_settings_mapped` concerns the supported input fields;
RPMDrate dynamics and the external PESs
are **未实测 (not run)** in this conversion validation.

## Trust and import lifetime

An input is **trusted executable Python**, using the same trust model as
PyRPMD's Python inputs. It can perform arbitrary side effects if instructed
to do so; this is not a sandbox. Read a downloaded script before conversion.
The injected RPMDrate functions only record calls, and the injected PES entry
points raise if called. No actual PES module is used, even if one with that
name is already present in `sys.modules`.

Ordinary explicit `from PES import get_potential, initialize_potential` and
`import PES` forms are supported, including aliases and dotted module names.
NumPy and standard-library helpers remain available. Relative, wildcard,
`importlib`/`runpy` dynamic imports, local PES functions, and non-PES third-party
helper imports are rejected; use explicit PES imports. This import handling
does not claim to restrict deliberately indirect execution in trusted code.

A context manager restores `sys.modules` bindings and `sys.path` on success
and failure, including newly imported helpers and replaced or deleted bindings.
Convert in a process with no concurrent imports. The importer cannot undo
arbitrary changes a trusted script makes to existing module objects or files.

## Report and validation

`PyRPMD.rpmdrate-import.v1` records the source path, pinned input API commit,
temperature/beads, each intercepted call and its arguments, mapped fields,
unit conversions, PES references, defaults, unmapped items, resolved geometry,
reaction and individual window plans. It is not a restart artifact.

The reproducible upstream check is
`examples/rpmdrate_import/validate_upstream.py`. It exercises the real import,
validate and prepare CLI routes on all four pinned examples. H3 is compared to
an independently handwritten configuration preserving the upstream atom order,
geometry and Rinf. Existing BKMP2 benchmark inputs are used only for topology
comparison: their Rinf is 10 bohr, whereas the upstream H3 input uses about
30.000158 bohr. Conversion never substitutes the benchmark value.
