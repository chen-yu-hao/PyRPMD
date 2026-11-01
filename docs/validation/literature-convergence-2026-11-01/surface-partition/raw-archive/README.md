# Frozen quantum equilibrium and parent archive

The local archive preserves every file in the frozen surface-partition scope,
including failed controls and excluded calibrations. Its creation and verification
performed no new sampling, PES evaluations, real-time propagation or rate analysis.
No external upload was attempted. The original 243-file publication inventory and
all original source/data bytes remain unchanged.

## Artifact identities

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `pyrpmd-h3-quantum-equilibrium-2026-11-01.tar.gz` | 83,259,073 | `051ca8d016c722b740bc340f07387b72cf14c3379c21394a171dd34f083e944d` |
| `pyrpmd-h3-quantum-equilibrium-2026-11-01.members.json` | 93,147 | `3d0ed87a1197b2b14cc769a572a2e486a5f939d041532e8280e1d6d14eae6b7b` |

Both artifacts are local to `/workspace/scratch/a5e4820daedd/`. The adjacent JSON
is identical to [member-inventory.json](member-inventory.json). The archive has
**279 regular file members**, whose uncompressed contents total **83,926,399
bytes**. Member names are repository-relative, without a leading archive directory:

- 243 frozen source, text and image entries, totaling 956,079 bytes.
- Their unchanged `controls/publishable-files.json` inventory, 67,878 bytes.
- Exactly 35 original raw NPZ files, totaling 82,902,442 bytes.

The scope-authorizing inventory SHA-256 is
`f68456bddc3e340911ae91590507a2769ae0a17d67eba983325870b15ba87b68`.
The new archive metadata is separate and is not added to that frozen inventory.
No external BKMP2 source, PES library, compiled native extension, live child data,
umbrella campaign or unlisted repository file is included.

## Retained data categories

[raw-data-index.json](raw-data-index.json) gives every raw file's exact archive
member name, SHA-256, byte count, category and report/settings links. For checkpoints
it records the verified settings fingerprint, domain-normalization hash, RNG-array
identity and parent-array identities where present.

| Category | Raw NPZ files | Interpretation |
| --- | ---: | --- |
| Primary TS equilibrium and parents | 8 | Eight independent chains; all 1,600 registered parents retained |
| Primary reactant normalization controls | 4 | Four independent chains; no TS parent snapshots |
| Registered bead/domain controls | 16 | Four seeds for each of four control cases |
| Initial calibration | 2 | Excluded from primary inference; exact earlier sampler source snapshot retained |
| Efficiency calibration | 2 | Excluded from primary inference; its five TS snapshots are not primary parents |
| Direct-importance pilots | 2 | Excluded; concentrated weights and original diagnostics retained |
| Short-distance finite-grid diagnostic | 1 | Original grid and energies; not a sampling or convergence claim |

The P64/0.5-bohr control **does not pass** the registered full-interval equivalence
test. P256/0.5-bohr and P128/0.4- and 0.6-bohr cases pass their individual tests.
The aggregate all-control decision remains **false**. Successful file-integrity
checks do not change any scientific gate or convert these equilibrium results
into a complete RPMD rate.

## Verification actually performed

[verification.json](verification.json) records the completed checks:

- All 279 archived member byte strings match their original SHA-256 and sizes;
  all 279 original inputs were rehashed after packing.
- All 32 checkpoint NPZ containers have valid original v2 per-array integrity
  envelopes. Every array was read with `allow_pickle=False`; report-to-NPZ,
  settings-fingerprint and reference-domain identities match. The remaining
  three NPZ files have no original envelope; their complete contents, ZIP CRCs
  and hashes were verified against their original reports.
- All 72 identities in the immutable primary manifest and all 96 identities
  in the control manifest match the archived bytes. All 12 primary and 16
  control chains have their registered complete observations.
- The eight primary TS arrays each have shape `(200,128,3,3)` in bohr and
  completed sweep indices `2100,2200,...,22000`. Every primary parent was
  extracted to a new temporary `.npy`, reloaded, and checked against its original
  dtype, shape and C-order array-byte hash. All 16 coordinate/index array
  roundtrips passed; the temporary files were then removed.

The archive was created exclusively, verified before publication of its local
filename, and made read-only together with its adjacent inventory. Packaging
uses sorted PAX tar members with fixed ownership/timestamps and gzip timestamp
zero. Cryptographic identities are the authority; read-only mode is an additional
protection against accidental writes. The complete executed script hash,
environment and repository identity are in
[archive-provenance.json](archive-provenance.json). The original checkout HEAD
was `3d096d5cfc5709f14a1ad5e3d184b108eef9f20e` at both start and finish; this is not
an assertion that all evidence was committed at that HEAD.

## Verify and extract the exact parents

Run from a repository checkout containing this new archive metadata, with Python
and NumPy available. Neither PyRPMD nor a potential library is imported by the
verification/extraction tool. Paths below are the executed local artifact paths;
they can be replaced with the locations of unchanged copies.

```bash
sha256sum /workspace/scratch/a5e4820daedd/pyrpmd-h3-quantum-equilibrium-2026-11-01.tar.gz
sha256sum /workspace/scratch/a5e4820daedd/pyrpmd-h3-quantum-equilibrium-2026-11-01.members.json

python docs/validation/literature-convergence-2026-11-01/surface-partition/raw-archive/package_archive.py \
  --archive /workspace/scratch/a5e4820daedd/pyrpmd-h3-quantum-equilibrium-2026-11-01.tar.gz \
  --verify /workspace/scratch/a5e4820daedd/pyrpmd-h3-quantum-equilibrium-2026-11-01.members.json

python docs/validation/literature-convergence-2026-11-01/surface-partition/raw-archive/package_archive.py \
  --archive /workspace/scratch/a5e4820daedd/pyrpmd-h3-quantum-equilibrium-2026-11-01.tar.gz \
  --verify /workspace/scratch/a5e4820daedd/pyrpmd-h3-quantum-equilibrium-2026-11-01.members.json \
  --extract-parents ./fresh-quantum-parents
```

Compare the first two hashes with the table above. The second command validates
every archive member and checkpoint/report relationship. The final command repeats
verification, refuses an existing output directory, and writes two arrays per TS
seed plus `parent-array-identities.json`. Seeds are 721301, 721303, 721307, 721309,
721313, 721317, 721321 and 721323. It selects all primary snapshots in stored
order, without thinning, resampling, RNG draws or coordinate transformations.

These are static equilibrium, flux-metric-weighted Cartesian configurations on
`r01=r12<=r02`; their total center-of-mass free modes were integrated out. They
are not real-time trajectories and contain no Maxwell momentum samples. An exact
subsequent child experiment also needs its separately registered channel,
momentum/RNG and propagation protocol. Those experiments are outside this archive.

To unpack the complete scope, first verify the archive, then extract into a fresh
directory, preserving repository-relative paths:

```bash
mkdir ./fresh-quantum-equilibrium
tar -xzf /workspace/scratch/a5e4820daedd/pyrpmd-h3-quantum-equilibrium-2026-11-01.tar.gz \
  -C ./fresh-quantum-equilibrium
```

Do not recompress checkpoint NPZ files or rerun a completed producer/analyzer in
place: whole-file identities bind the checkpoint, settings, report and domain
JSON, so physically identical arrays alone are insufficient to preserve the
recorded experiment. Perform any additional analysis in a new output directory.

## Statistical replay and executable prerequisites

The archive directly reconstructs the complete static parents and retained BAR
observations. It is a scoped evidence bundle, not a standalone installation of
the project. `archive-provenance.json` identifies the seven registered source
dependencies outside its 243-file scope: `mixing_sampling.py`, `pes.py`,
`classical_tst.py`, `constants.py`, `analysis/fes.py`, `checkpoint.py` and `io.py`.
Obtain these from the matching PyRPMD repository history and verify their hashes
before executable replay. The included final sampler and older calibration-v1
snapshot have different registered identities; keep each tied to its original
settings. Replaying checkpoint restart additionally requires its recorded runtime
and external BKMP2 source/library identity. The archive does not supply or claim
redistribution permission for that potential.

For a read-only primary BAR replay with the matching repository dependencies,
load each primary checkpoint through `pyrpmd.checkpoint.load_checkpoint` with
`require_integrity=True`, then call:

```python
analysis = benchmarks.h3.surface_partition.analyze_samples(
    saved["delta_energy"],
    settings["lambdas"],
    settings["beta"],
    reference["log_reference_over_classical"],
    reference_domain,
)
```

All four JSON inputs and the full `delta_energy` arrays are retained. Compare
`log_quantum_classical_ratio`, its first/second-half values, BAR overlaps and
sample counts against the original report; do not write over it. Independent
seed chains remain the uncertainty units. The primary manifest identifies the
eight TS and four reactant chains, and the controls protocol identifies all
16 control chains. Aggregate QTST comparison also needs the independent
`h3-quantum/diatom-transfer.json` oracle referenced by SHA-256 in
`primary-summary.json`, plus the domain-matched oracles referenced by the control
protocol. Those reference dependencies, and real-time recrossing results, are
outside this scoped archive.

## New publishable metadata

[publishable-files.json](publishable-files.json) lists only the new small script,
log, README and JSON receipts in this directory, with exact sizes and hashes;
add that inventory itself separately. The archive, adjacent member inventory
outside the repository, and all 35 raw NPZ files remain local. This packaging
operation performed no Git mutation.
