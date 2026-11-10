# PyRPMD subspace-correction run data (2026-11)

Raw run directories that are too large for the main branch. Reproducible inputs live on `main`.

| Archive | Contents | Input on main |
| --- | --- | --- |
| `h3-200K-P128/h3-200K-P128-b20.tgz.part*` | 3D H + H₂ (BKMP2), 200 K, P = 128: umbrella samples, checkpoints, centroid path, PMF, recrossing, subspace corrections | `benchmarks/h3_subspace/rate_200K.inp`, `correct_200K.toml` |
| `clhbr-300K-production/clhbr-300K-production.tgz` | Cl + HBr (BrHCl LEPS1), 300 K, P = 16, with centroid path and subspace corrections | `benchmarks/clhbr_subspace/rate_production.inp`, `correct.toml` |

Restore:

    cat h3-200K-P128/h3-200K-P128-b20.tgz.part* | tar xz -C benchmarks/h3_subspace/runs
    tar xzf clhbr-300K-production/clhbr-300K-production.tgz -C runs
    sha256sum -c SHA256SUMS

Then e.g. `pyrpmd subspace-correct benchmarks/h3_subspace/correct_200K.toml` reuses the saved C_S (`output.reuse = true`), or change `[subspace]`/`[quantum]` to recompute from the stored run.
