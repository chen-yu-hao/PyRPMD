---
orphan: true
---

# Validation-data archive plan

This is an additive preservation plan. T01 inventories the checkout and proposes
an archive layout; it does not upload, move, remove, recompress or rewrite any
existing evidence. Existing Git commits, source archives and scientific run
identities retain their current meaning. The commands below only inspect source
files or create new inventory reports outside the scanned tree.

## Establish a reproducible inventory

From a quiescent checkout, run:

```bash
python tools/data_manifest.py --root . --min-bytes 1048576 --directory-depth 2 > ../pyrpmd-data-manifest.json
python tools/data_manifest.py --root docs/validation --directory-depth 1 > ../pyrpmd-validation-manifest.json
git rev-parse HEAD
git status --short
git count-objects -vH
```

Keep the exact commit, dirty/untracked status, command, and manifest together in
the review record. The scanner reads the **worktree**, including untracked
files; it does not claim to measure only Git-tracked data. Use a clean checkout
for a release inventory. The JSON contains a SHA-256 and byte count for every
included regular file, totals by directory and file type, and a descending list
of files at least `--min-bytes` bytes large. Directory groups are disjoint: the
first `--directory-depth` parent components identify a group, and root-level
files belong to `.`. File types use the lower-case final suffix; suffixless
files use `(no suffix)`. Every file contributes once to each summary.

The default exclusions are `.git`, `.venv`, `.venv-*`, `venv`, `__pycache__`,
`.pytest_cache`, `.mypy_cache`, `.ruff_cache`, `pytest-of-*`, `build`, `dist`,
`_build`, `*.egg-info`, `*.pyc`, `*.pyo`, `*.so`, `*.pyd` and `*.o`.
They remove ordinary build/cache noise, **not** a Git ignore policy. The output
records every pruned entry and matching pattern. Review these entries before
treating a manifest as complete. `--exclude PATTERN` adds a case-sensitive glob;
patterns containing `/` match root-relative POSIX paths, while other patterns
match basenames at any depth. `--no-default-excludes` disables the defaults.
To inspect the Git object store separately, use `--root .git`; the root itself
is not subject to child-entry exclusions.

Symlinks (including broken links), Windows directory junctions and other reparse
points, and nonregular files are recorded under `skipped` and are never followed.
A root that is itself a symlink, junction or other reparse point is rejected.
Unreadable or changing files fail with a
nonzero status instead of producing a partial manifest. The scanner checks file
metadata before and after hashing through each API separately; this accommodates
Windows path/descriptor differences in creation/change times and executable
permission bits without ignoring a mutation during hashing. It is not a
filesystem snapshot and cannot make a concurrently changing directory tree
consistent. Reads may update
filesystem access times. The scanner does not write any source file, contact a
service, inspect NPZ/PT contents, or execute archived code. Its only output is
JSON on stdout (errors go to stderr). Keep redirected output **outside** the
scanned tree so it cannot inventory its own output. No absolute paths or
timestamps enter successful JSON; names are escaped as ASCII-compatible UTF-8,
and paths use `/` on every platform.

### Baseline at `8cc402fb12391e4fb644d3e186e5e58d1699759a`

A separate `git ls-files` inventory of tracked regular files on 2026-11-02 gave:

| Scope | Files | Bytes |
| --- | ---: | ---: |
| Entire tracked checkout | 4,767 | 196,786,009 |
| `docs/validation` | 3,331 | 170,677,840 |
| Validation NPZ files | 134 | 88,560,213 |
| Validation JSON files | 1,857 | 41,868,982 |
| Validation files at least 1 MiB | 21 | 56,131,938 |

Across the whole tracked checkout there were 1,971 JSON, 332 log, 134 NPZ and
12 PT files. The two largest files were
`docs/validation/adaptive-nrpmd-refinement-2026-11-01/Up0_6_P4/source.npz`
(11,119,290 bytes) and the adjacent `Um0_6_P4/source.npz` (11,112,778 bytes).
These are byte totals, not compressed-archive sizes or disk-block usage.
The fresh clone's local `.git` files occupied 122,229,332 bytes; that figure
depends on clone/pack state and is not a portable repository-size guarantee.
Recompute all figures when selecting an archive release.

## Destination and release unit

Prefer a versioned Zenodo dataset for citable, completed campaign evidence, with
an independently maintained data repository as an optional browseable mirror.
An alternative is a dedicated data repository as the primary store, with pinned
commits and separately published release archives. The owner chooses the host,
maintainers, visibility and retention responsibility before any upload.

Use a campaign or coherent validation release as the unit of preservation.
Keep raw observations, model files, protocols, audit logs, analysis code,
environment metadata, and the resulting tables/figures together. Similar files
from failed or superseded runs are still evidence; record their status instead
of deduplicating by filename or treating them as scientific successes.

Each archive release should include:

- A human-readable overview stating the scientific question, provenance,
  limitations and interpretation of every campaign status.
- The exact PyRPMD commit and original repository-relative paths, with the
  existing run identities, RNG/checkpoint information and source hashes intact.
- A per-file SHA-256/byte manifest and a checksum for each packaged archive.
- Original licenses, third-party notices, authorship and required model/data
  redistribution permissions; do not assume the code's license covers data.
- Instructions for restoring the original relative layout into an empty,
  separate staging directory and running the documented analysis commands.

Zenodo supports separate version and concept DOIs. Pin a **version DOI** for
reproduction; a concept DOI resolves to the evolving latest version. See the
[official DOI-versioning guide](https://zenodo.org/help/versioning) and
[version-management instructions](https://help.zenodo.org/docs/deposit/manage-versions/).
Zenodo recommends packing collections of more than 20 files into archives; see
[file-management guidance](https://help.zenodo.org/docs/deposit/manage-files/).
Use a few campaign ZIP files rather than thousands of loose uploads, preserving
member bytes and relative paths. Keep the SHA-256 manifest beside the ZIP so a
reader can inspect it before extraction. A DOI alone is not a byte-integrity
check: service policy permits limited file corrections after publication, so
verify the project manifest on every retrieval.

## Additive rollout and verification gates

1. **Freeze a candidate inventory.** Select the source commit and a quiescent
   checkout. Review `skipped`, record both the full-worktree and validation-only
   manifests, and classify each candidate as essential fixture, documentation,
   raw evidence, model, or reproducible derivative. Classifications do not
   change files.
2. **Build new copies in staging.** Package selected campaign directories into a
   new destination outside the checkout. Retain all original files. Check every
   member name for absolute paths, `..`, symlinks and platform-specific path
   collisions before extraction. Compare the staged bytes and hashes with the
   source inventory; test restore into a second empty directory.
3. **Prepare the destination record.** Add authors, licenses, provenance,
   dataset version, source commit, limitations, and the manifest. The owner
   reviews the concrete archive and metadata before publishing. A data-repository
   mirror must refer to a specific commit/release, never only `main`.
4. **Verify an independent retrieval.** Retrieve the published copy into a new
   empty directory, check archive and member SHA-256 values, and replay the
   campaign's small analysis/reference checks from the restored paths. Record
   exact commands, environment, test results, bytes and retrieval time. Do not
   execute PT/pickle payloads merely to verify file integrity.
5. **Add a repository catalog.** In a later reviewed change, add a small catalog
   mapping every original path to its dataset/version identifier, archive URL,
   archive checksum, member path, member checksum and size. Include the source
   commit, license and coverage classification. Original data remain in place
   during this rollout.
6. **Direct new bulk results to the archive.** After successful retrieval and
   review, store future campaign payloads in the chosen data destination and
   commit only the catalog, small fixtures and scientific summary to the code
   repository. A future package-size change may narrow source-archive inclusion
   after the tests below pass. This phase limits growth without altering Git
   history or removing existing checked-in evidence.

For each gate, keep the old source and verified copies available. If retrieval,
hash verification, link checking, or replay fails, keep the current repository
links and packaging rules; repair the candidate copy or catalog in a follow-up.
No destructive operation is part of these steps.

## Link strategy

Keep human-facing summary pages at stable repository paths. Add a catalog link
and version DOI beside the existing local links before changing any consumer.
When a later consumer is ready, rewrite its data link using the explicit catalog
mapping: old repository path to pinned archive/member. For images needed by the
manual, retain small local figures or serve reviewed copies at stable URLs;
do not replace a Markdown image URL with a ZIP member that browsers cannot render.

Audit relative Markdown links, GitHub `blob`/`raw` links, JSON fields, example
inputs and Python paths separately. Preserve the relative directory tree inside
archives so existing analysis scripts can work in a restored staging checkout.
Do not edit old scientific manifests, source hashes or run identities to point
at a new host: put location metadata in the external catalog. Historical Git
URLs stay valid because commits are unchanged. A future catalog checker should
verify missing mappings, duplicate paths, case collisions and broken URLs; CI
must compare content hashes rather than trusting a successful HTTP status.

## Source archives and tests

T01 adds only the inventory tool to `MANIFEST.in`; it does not narrow existing
data rules. The manifest currently includes broad `docs` patterns and explicit
exceptions for particular NPZ files, raw sampling evidence and PT models. A
blanket exclusion of JSON, NPZ or PT would remove required reference evidence.

Concrete dependencies that must be reviewed before any packaging change include:

| Consumer | Existing dependency |
| --- | --- |
| `tests/test_water_campaign.py` | Water numerical report and source-audit JSON under `docs/validation`. |
| `tests/test_water_native_reference.py` | Water classical-pilot `endpoint.npz`. |
| `tests/test_qtip4pf_benchmark.py` | Archived water-pilot validation results. |
| `tests/test_h3_p256_parent_archive.py` | `archive_parents.py` stored under the H3 validation tree. |

In a follow-up, classify all such consumers by inspection and test collection.
Keep small deterministic unit-test fixtures, required analysis helpers, licenses
and catalog metadata in the source distribution. Keep library tests offline and
independent of service availability. For archive-dependent campaigns, add an
explicit CI preparation step that retrieves an exact dataset version, verifies
hashes, and restores the expected paths in a fresh test checkout before pytest.
Missing required data or a checksum mismatch must fail that campaign job, not
silently skip it. The `campaign` marker only selects tests; it is not permission
to reduce the default Linux test suite.

Before approving any later source-package exclusions, build an sdist, inspect
its members, install it in a fresh environment, run the library tests and the
full campaign suite with verified staged data, and build the manual with
warnings treated as errors. Compare collected/pass/skip counts with the recorded
baseline. Source-size savings must be measured on the built archive; copying
data to another host does not itself shrink existing Git history or clones.

## Decisions for the repository owner

- Choose Zenodo, a dedicated data repository, or both; designate maintainers,
  archive visibility, licenses and a retention/mirroring policy.
- Approve campaign boundaries and the exact keep/archive catalog, including
  failed/superseded evidence and binary-model redistribution rights.
- Decide how future CI obtains data (public or authenticated), its size/time
  budget and cache policy. Cache keys must include the pinned manifest digest.
- Choose when new campaign outputs stop entering the code repository and whether
  a later source-distribution exclusion change is acceptable after replay checks.
- Decide separately whether existing tracked data should ever be removed from a
  future tree, and whether Git history should ever be rewritten. The default is
  **neither**. History rewriting would change commit identities, affect clones,
  signatures and links, and needs a separately scoped decision; this plan does
  not supply or execute such commands.

There is no immediate Git-size reduction claim. The delivered inventory enables
review; verified additive archives and a policy for new outputs limit subsequent
growth while preserving today's repository and scientific evidence.
