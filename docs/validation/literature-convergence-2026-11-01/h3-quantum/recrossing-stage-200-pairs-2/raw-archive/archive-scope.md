# Immutable base H3 quantum child data: 200 fs, two momentum pairs

This archive contains the complete **base pair batch 0**: eight independent
parent chains, 200 saved configurations per chain, two Maxwell momentum pairs
per configuration, both antithetic signs, and both 0.1 and 0.05 fs timesteps.
Each child ends at 200 fs. There are 16 run directories and 400 integrity-bound
NPZ chunks, representing 6,400 children at each timestep and 38.4 million native
steps in total. The timesteps use identical parent configurations, keyed momenta
and equivalent channels; they are paired controls, not independent replicas.

The full cohort was completed and analyzed before the separate extra-pair
extension. Its registered kappa precision gate failed: relative 95% half-width
0.03272199946508305 exceeds 0.03. The original report and every failed or passed
gate are retained unchanged. Archiving does not promote a validation flag.

## Content and exclusions

`recrossing-primary/` holds only its original protocol, base schedule, completion
receipt and the eight base `seed-*/dt-*/horizon-200/` directories. Each directory
contains its unchanged settings, report, kappa curve and all 25 chunks.

`recrossing-analysis-200/` preserves the complete analysis, time curve, exact
execution receipt, precision-extension decision and the separately recorded
extension launch receipt. That launch receipt is metadata explaining what
happened after the base result; no extra-pair trajectory is included.

The base analysis log and `recrossing-base-final-tool-output.json` are retained.
The latter contains only the remaining incremental, truncated terminal output
returned after completion, together with the observed exit status. It is **not**
a complete stdout transcript or an exact launch-time record. The reconstructed
base command in `reference/external-prerequisites.json` is explicitly labeled
as a reconstruction from the frozen schedule.

`reference/` includes the relevant normalization data and protocols, source and
runtime identities, parent file references and the independent review's exact
input hashes. Git-backed Python code, external PES Fortran, model libraries and
native binaries are not included. The active extra-pair batch, its live output,
umbrella campaigns, water data and unrelated calibrations are excluded.

## Parent and software prerequisites

The static parents are already preserved in
`pyrpmd-h3-quantum-equilibrium-2026-11-01.tar.gz`, SHA256
`051ca8d016c722b740bc340f07387b72cf14c3379c21394a171dd34f083e944d`.
Its eight TS endpoint arrays contain 1,600 configurations in bohr; the chain is
the uncertainty unit. Parent raw files are not duplicated in this child archive.
The external prerequisite index records exact parent checkpoint, settings,
report and domain-file hashes for every chain. Their original whole-file bytes
are required: rewriting an NPZ or regenerating a report can invalidate the
child source identity even if some physical arrays stay unchanged.

Restore the required source files from the associated PyRPMD repository change,
checking every listed SHA256. The per-file hashes are authoritative; an observed
working-tree HEAD alone does not identify all executed files. Rebuilding actual
child dynamics also requires the separately obtained BKMP2 source/model and the
recorded native build. The archive contains their identities, not their code or
compiled binaries. Source locations, versions, native-library SHA and model
build-receipt identity are listed in `reference/external-prerequisites.json`.

## Extraction and replay

Verify the archive SHA256 against its separate provenance receipt. Use the
adjacent `.members.json` sidecar to verify every member's path, size and hash.
Extract into a **new empty directory**; never extract over active simulations or
overwrite a registered checkpoint. All members are regular, relative paths.

For example, with the archive and sidecar paths supplied explicitly:

```python
import hashlib
import json
from pathlib import Path
import tarfile

archive_path = Path("pyrpmd-h3-quantum-recrossing-stage-200-pairs-2-2026-11-01.tar.gz")
index_path = archive_path.with_name(archive_path.name.removesuffix(".tar.gz") + ".members.json")
destination = Path("replayed-base-200fs")
destination.mkdir(exist_ok=False)
expected = {row["member"]: row for row in json.loads(index_path.read_text())["files"]}
seen = set()
with tarfile.open(archive_path, "r|gz") as archive:
    for member in archive:
        assert member.isfile() and member.name in expected and member.name not in seen
        assert not Path(member.name).is_absolute() and ".." not in Path(member.name).parts
        data = archive.extractfile(member).read()
        row = expected[member.name]
        assert len(data) == row["bytes"]
        assert hashlib.sha256(data).hexdigest() == row["sha256"]
        target = destination / member.name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
        seen.add(member.name)
assert seen == set(expected)
```

To replay the analysis, first restore the exact parent files to the paths below
the source checkout's `docs/validation/literature-convergence-2026-11-01/`, plus
the recorded classical reference and finite-P diatom reference. The independent
equilibrium archive and associated Git evidence supply these prerequisites.
Then, from that checkout, run the frozen analyzer with a **new** output path:

```bash
PYTHONPATH=src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python benchmarks/h3/quantum_recrossing_analyze.py \
  --data /absolute/path/to/replayed-base-200fs/recrossing-primary \
  --output /absolute/path/to/new-base-analysis --horizon 200
```

The default analysis uses only batch 0 and requires all eight chains; do not add
the extra-pair option or use available-only selection for this replay. Compare
numerical fields, gate outcomes and input identities with the archived report;
new creation timestamps and output paths will naturally differ. It is not
necessary to rerun propagation to replay the saved signed-flux observations.

Each raw chunk retains the initial ordinary Maxwell momenta, chosen channel,
parent and pair indices, per-time surfaces, denominators and numerators, observed
minimum pair distances, every-step energy extrema, and final q/p/step/time. The
keyed RNG is reconstructed from the exact seed/parent/pair rule in the frozen
source; there is no mutable child RNG state to infer from execution order.

The local packaging verification reloads all 400 original and archived NPZ
chunks with their integrity envelopes, validates exact settings fingerprints,
checks time/parent/pair/step inventories, replays saved signed-flux numerator and
denominator arrays exactly, and verifies paired-timestep momentum/channel hashes.
It also binds all chunks to the completed reports and independent review inputs.
