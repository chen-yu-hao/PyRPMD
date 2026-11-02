# Complete four-pair H3 increment: raw data and replay

This local archive adds only the completed second momentum-pair batch, IDs 2 and 3, to the immutable first batch, IDs 0 and 1. It contains no external PES source, potential library, compiled native extension or repository source code. No umbrella or water campaign data are included. No upload or external persistence is asserted.

## Exact scope

The increment contains all 400 successful raw NPZ chunks from the extra batch, plus its 16 complete settings/report/curve triples, original and completed schedule, recorded stdout, observed normal-exit receipt, and the complete four-pair analysis. The fixed cohort has seeds 721301, 721303, 721307, 721309, 721313, 721317, 721321 and 721323, both 0.1-fs and 0.05-fs timesteps, and a 200-fs horizon. Each run has 200 parents and 25 chunks of eight parents. Two momenta per parent are paired antithetically, yielding 6,400 additional trajectories per timestep.

The union with the base archive has 800 raw chunks, 12,800 trajectories at each timestep and 76,800,000 recorded native integration steps. It uses the same eight parent-chain checkpoint/settings/report/domain identities and the same 1,600 saved parent configurations. Parent configurations and the two timestep calculations are not additional independent uncertainty units. The uncertainty unit remains one complete parent chain; the combined result has eight such units.

The first-stage precision failure is retained in the base archive. The combined result passes its eleven registered fixed-P primary criteria. Its supplementary first/second-parent-half equivalence diagnostic still fails. This archive preserves both facts and does not demonstrate dynamic bead convergence or validate the production MD umbrella sampler.

## Required immutable archives

1. `pyrpmd-h3-quantum-recrossing-stage-200-pairs-2-2026-11-01.tar.gz`
   SHA256 `d3205ebc4a5210af59d03ebdbf0737708278d83f1e4cb540a18818213d78cea8`, 324,627,489 bytes.
2. `pyrpmd-h3-quantum-equilibrium-2026-11-01.tar.gz`
   SHA256 `051ca8d016c722b740bc340f07387b72cf14c3379c21394a171dd34f083e944d`, 83,259,073 bytes.
3. This increment and its sibling `.members.json` SHA/size inventory.

`reference/increment-external-prerequisites.json` binds the actual archive, parent, scientific source, native runtime, analysis and model identities. The original build/source provenance remains external. Re-running forces requires the separately obtained BKMP2 source/library and the registered runtime; raw statistical replay evaluates no potential and propagates no state. The independent replay still verifies the recorded source/native provenance against the original review checkout. Code is obtained from the repository, by path and exact source hash, rather than duplicated in this data archive.

## Extraction and exact raw replay

Create a fresh empty directory. Verify the SHA256 and size of all three archive files and their inventories before extracting. Extract the base and increment into the same `campaign/` root. Their scientific data paths are disjoint: the increment uses `recrossing-primary/extra-pair-batch-1/`, `recrossing-analysis-200-four-pairs/` and distinct metadata filenames. Do not overwrite any existing member or recompress any NPZ.

From the parent archive, extract the whole five files `checkpoint.npz`, `settings.json`, `report.json`, `reference.json` and `reference-domain.json` for each of the eight `surface-partition/primary/ts/seed-<seed>` directories, preserving their repository-relative paths under a separate `repository/` directory. Also extract the parent `primary-protocol.json`. These are the exact static parents, not regenerated samples.

The repository implementation used for independent replay is:

```
docs/validation/literature-convergence-2026-11-01/h3-quantum/recrossing-stage-200-pairs-4/independent-review/replay_complete.py
```

Its identity is recorded in the prerequisites. From the matching review checkout, with NumPy/BLAS threads fixed to one, run it with:

```bash
python path/to/replay_complete.py \
  --data-root /absolute/fresh/campaign \
  --parent-evidence /absolute/fresh/repository/docs/validation/literature-convergence-2026-11-01/surface-partition \
  --reference-root /absolute/fresh/campaign/reference \
  --output /absolute/fresh/replay-result
```

The reader hashes and decodes the same immutable bytes. It checks every NPZ integrity envelope and complete report inventory; reconstructs keyed channels and all momentum arrays; derives surface gradients, flux norms and velocities from the original parents; verifies each stored signed numerator and positive denominator; and combines all four pair contributions within each chain before computing its transmission ratio. It rebuilds base and combined statistics, same-chain log-partition/transmission covariance, all registered criteria, the supplementary parent-half diagnostic and the complete 200-point mean/CI curves. It makes zero PES or dynamics calls.

The package verification actually performs this extraction into one new campaign root and reruns that independent reader. It compares all original and extracted raw/member identities and the complete numerical summaries. The resulting extraction receipt and compact evidence are outside the archive to avoid a self-referential archive checksum. Parent and base data are references, not duplicated raw payload in this increment.

## Timing and limitations

The exit receipt is an observation of the dispatcher returning exit code zero at its recorded UTC time. It is not the exact launch time or final native-step time. The saved stdout file and the empty final tool response are separately identified; no complete terminal transcript is invented.

The nominal eight-chain t intervals describe this experiment. They do not separately correct finite-chain ratio/geometric-mean bias or conditional-extension coverage, and they exclude PES/model uncertainty. The sampled minimum-distance series is recorded at 1-fs boundaries. Whole-file hashes and exact checkpoint bytes are part of the parent/child provenance; semantically equivalent JSON rewriting or NPZ recompression would break that identity.
