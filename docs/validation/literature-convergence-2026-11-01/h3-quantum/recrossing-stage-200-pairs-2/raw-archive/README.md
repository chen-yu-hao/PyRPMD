# Local immutable archive of the base 200 fs quantum child cohort

The completed eight-chain, two-pair child data are preserved in one local archive.
All **465 members** were verified by exact size and SHA256. All **400 raw NPZ
chunks** were reloaded from both the original files and archived member bytes
with their checkpoint integrity envelopes. The 16 timestep/seed runs bind to
the unchanged complete-cohort analysis; every original gate outcome is retained.

| Artifact | Bytes | SHA256 |
| --- | ---: | --- |
| `pyrpmd-h3-quantum-recrossing-stage-200-pairs-2-2026-11-01.tar.gz` | 324,627,489 | `d3205ebc4a5210af59d03ebdbf0737708278d83f1e4cb540a18818213d78cea8` |
| Adjacent `.members.json` sidecar | 184,927 | `2bee3fcf59f5bbdfc8f92dda422164512c251aeb2ec83e58c519016b105c199f` |

Both files are under `/workspace/scratch/a5e4820daedd/`, created exclusively and
set read-only after verification. No upload or external persistence is claimed.
The member inventory records every archive-relative name, original local path,
size, hash and category. Uncompressed member bytes total
**325,713,873**, including **325,019,087** raw NPZ bytes.

## What was preserved and checked

- All eight base seeds, both 0.1 and 0.05 fs timestep runs, all 200 parents per
  seed, momentum-pair indices 0 and 1, and all 200 fs child endpoints and saved
  flux observations. This is 6,400 children per timestep, 38.4 million recorded
  native integration steps, and eight independent parent chains.
- Exact settings, report and chunk identities from the complete analysis;
  483 independently reviewed scientific-input file hashes; all eight external
  parent checkpoint integrity envelopes.
- Exact replay of every stored signed-flux numerator and denominator; identical
  initial momentum and equivalent-channel hashes across paired timesteps; all
  archived array descriptors identical to their original-file descriptors.
- Frozen protocol, base scheduling and completion records, analysis output and
  execution receipt, analysis log, precision-extension decision and launch
  metadata. The final base terminal observation is explicitly partial/truncated
  and does not establish a full stdout transcript or exact launch timestamp.

The raw arrays and their whole NPZ bytes were not recompressed individually or
resaved. Only the enclosing tar stream was compressed. Every original included
file was rehashed after archive verification. The bounded packaging process
used 16.500 CPU seconds and peaked at
41,868 KiB RSS; it ran no scientific simulation.

## Scope remains the completed first cohort

The active extra-pair batch, its live output, water and umbrella data, external
PES source/libraries, compiled native code and Git-backed Python source are
excluded. Parent raw arrays remain in the separately verified equilibrium
archive, SHA256
`051ca8d016c722b740bc340f07387b72cf14c3379c21394a171dd34f083e944d`.
Their exact checkpoint/settings/report/domain identities and required software
are listed in [external-prerequisites.json](external-prerequisites.json).

The κ relative 95% half-width remains 0.03272199946508305, above the registered
0.03 limit. The fixed-P gate flag remains false. This packaging result changes
neither that scientific outcome nor the continuation decision. The two paired
timesteps and individual children are not counted as independent chains.

[archive-scope.md](archive-scope.md) gives extraction and exact analysis replay
instructions, dependency paths, and the distinction between saved keyed momenta
and mutable RNG state. Extract into a new directory and replay analysis into a
new output path; do not overwrite original registered data.

## New compact evidence only

- [Archive provenance](archive-provenance.json)
- [All-member index](member-inventory.json)
- [Raw chunk index and array identities](raw-data-index.json)
- [Reload and identity verification](verification.json)
- [Exact executed packaging source](package_stage.py)
- [Execution receipt](execution.json) and [output](package.log)
- [Separate publishable inventory](publishable-files.json)

This new inventory is independent of the frozen stage evidence and equilibrium
inventories. It includes only this compact metadata, instructions and packaging
source. Neither raw archive nor any NPZ is intended for Git publication here.
