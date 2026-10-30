# Multiple entrances and nonequivalent transition states

`task = "network"` runs a set of explicit gas-phase bimolecular RPMD rate
problems. Each transition-state reference executes the complete existing
Bennett--Chandler calculation: preparation, umbrella PMF, constrained parents,
paired child trajectories and the recrossing factor. Different TS geometries
have their own reference bond distances. They are not treated as equivalent
atom permutations of the first geometry.

All entries share the same ordered atomic identities, isotope masses, charge
and multiplicity. An entrance specifies its own two-fragment atom partition,
`rinf` and electronic statistical weight. A TS can use its own PES and symmetry
factor. This supports several entrance partitions of the same atomic system;
unrelated stoichiometries or different isotope inventories require separate
jobs.

## Why the output is a matrix

For entrance `e` and dividing-surface reference `a`, the existing rate estimator
is used independently:

\[
k_{e,a}=g_{e,a}\,\sigma_{e,a}\,
\frac{4\pi R_{\infty,e}^2}{\sqrt{2\pi\beta\mu_e}}
\exp[-\beta\Delta W_{F,e,a}]\,\kappa_{e,a}.
\]

Here `g` is `electronic_weight` and `sigma` is `symmetry_factor`. Neither weight
is an inferred entrance population. Each path retains its own reactant
normalization, PMF measure, transmission coefficient and uncertainty.

The supported aggregation is **`separate`**. The result contains
`matrix[entrance][channel]["transition_states"][ts_id]`, including each complete
path result. `total_rate_cm3_s` and every cell's `summed_rate_cm3_s` are `null`.
An unsupported aggregation mode is rejected before sampling.

This is a scientific restriction. A converged RPMD rate can be invariant under
moving a dividing surface: two different TS references can therefore be two
estimates of the same flux. Adding them would count it twice. Likewise, rates
normalized to different entrance reactant populations are not a single rate.
Channel names and optional product fragment declarations label the matrix;
they do **not** provide a product-basin classifier or prove that fluxes are
disjoint. A physical basin/flux partition and its matching estimator are needed
before adding channel contributions. This release does not implement that
additional estimator, nor does it simulate nonadiabatic transitions between
the per-path PESs.

The normalization follows the existing RPMDrate-style bimolecular workflow.
Primary sources for the underlying distinction are Craig and Manolopoulos,
*J. Chem. Phys.* **123**, 034102 (2005),
[doi:10.1063/1.1954769](https://doi.org/10.1063/1.1954769), and the
[RPMDrate input documentation](https://greengroup.mit.edu/rpmdrate-input/).
Separate matrix reporting is PyRPMD's conservative composition of those
single-reaction calculations, not a new multi-channel rate-theory formula.

## Input

Use a Python `build()` mapping or the following TOML table structure. Atom
indices are zero based, and all referenced geometries use the same atom order.
The usual temperature, beads, dynamics, sampling, umbrella, analysis,
recrossing, parallel and output sections are shared defaults.

```toml
task = "network"
temperature = "1000 K"
beads = 32

[system]
geometry = "identity-and-reference.xyz"

[potential]
provider = "cabi"
library = "libshared-pes.so"

[network]
aggregation = "separate"

[[network.entrances]]
id = "a_bc"
fragments = [[0], [1, 2]]
rinf = "12 angstrom"
electronic_weight = 1.0

[[network.channels]]
id = "ab_c"
fragments = [[0, 1], [2]]

[[network.transition_states]]
id = "tight_ts"
entrance = "a_bc"
channel = "ab_c"
transition_state = "tight-ts.xyz"
forming_bonds = [[0, 1]]
breaking_bonds = [[1, 2]]
symmetry_factor = 1.0

[[network.transition_states]]
id = "other_ts"
entrance = "a_bc"
channel = "ab_c"
transition_state = "other-ts.xyz"
forming_bonds = [[0, 1]]
breaking_bonds = [[1, 2]]

[network.transition_states.potential]
provider = "cabi"
library = "libother-pes.so"
```

Each TS must specify either `transition_state` (an XYZ filename) or `geometry`
(a filename or numeric `[N,3]` array), and explicit paired `forming_bonds` and
`breaking_bonds`. Numeric geometries default to bohr; XYZ files default to
angstrom. `geometry_unit` overrides that default. If `system.geometry` is
absent, the first TS in sorted ID order supplies the reference geometry; array
geometries still require `system.symbols`. Files and Python factories are
resolved relative to the input file.

PES inheritance is **global → entrance → TS**, with a complete replacement of
the `potential` table at each narrower scope. An override must state its
`provider` or `type`; it does not silently inherit old parameters from a
different provider. Thus the global potential can be omitted when every path
obtains a complete PES from an entrance or TS.

`dynamics`, `sampling`, `umbrella`, `analysis`, `rate` and `recrossing` tables
can also be provided in an entrance or TS. Their fields override the shared
table fields in that order. Every resulting path is passed through the regular
`task="rate"` `JobSpec` validator, including units, field names, integrator
capabilities and sampling rules. A global `reaction` table is rejected to avoid
competing fragment/TS definitions.

Execution settings (`parallel`, `dynamics.loop_backend` and
`dynamics.block_steps`) belong in the shared job tables. Without an explicit
parallel mode, network task transport defaults to serial; each PES still uses
its own configured CPU/CUDA device. Set the shared layout explicitly when using
trajectory, bead or GPU task parallelism.

IDs are unique letters/digits/underscores/hyphens, up to 64 characters. Unused
entrances/channels, unknown references, incomplete atom partitions and changed
isotope identities are errors. Identical dividing-surface definitions on the
same PES are rejected, including rigidly translated/rotated copies with the
same reference bond differences. This detects a known duplicate; passing the
check does not establish that all other surfaces are disjoint. Do not assign
an additional symmetry factor to an `equivalent_paths` list that already
counts those exchange paths.

## Run, inspect and resume

```bash
pyrpmd prepare examples/network/two_entrances.py --pilot
pyrpmd run examples/network/two_entrances.py --output runs/network-example
pyrpmd analyze runs/network-example
pyrpmd resume runs/network-example
```

The runnable example executes two entrances with two distinct analytic
teaching saddles/PESs each. Its artificial masses, high temperature and short
trajectories are intended for integration checks, not molecular kinetics.

Paths live in `paths/<entrance-id>/<ts-id>/`. Each contains the standard PMF,
parent/child checkpoints, recrossing arrays and summary, plus a path identity
manifest. `network-progress.json` records completed paths after every stage.
On resume, completed paths are reused and an interrupted path resumes its
existing stage checkpoints. Changing a reference/PES file or scientific path
configuration is rejected. Logical ID-derived seeds are unaffected by input
list reordering or the number of worker processes. Changing execution-only
block size/backend settings uses the existing native compatibility rules.

The implementation processes paths in a fixed order and uses the existing
process/MPI executors within each path. It does not create nested worker pools.
All MPI ranks enter the same path sequence. This preserves the existing
collective-stage semantics while keeping each entrance normalization separate.

## Validation scope

`tests/test_reaction_network.py` checks distinct entrance reduced masses and
distances, two actual analytic one-negative-mode saddles, per-scope PES
inheritance, invalid identities/references, duplicate surface rejection,
logical-ID stability, dependency fingerprints, ordinary-rate compatibility,
and a numerical two-entrance/four-path PMF + recrossing calculation. The latter
is interrupted after one path and resumes only unfinished paths, preserving
completed results. These tests do not establish physical rate convergence,
channel exclusivity or real multi-node scaling.
