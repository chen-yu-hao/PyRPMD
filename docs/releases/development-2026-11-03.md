# Development documentation integration (2026-11-03)

This is a **documentation integration note for changes already merged into `main`**, not an announcement of a new package version. The repository still identifies its manual with the existing 0.2.0 release. Exact reproducibility requires recording the commit, potential and scientific inputs.

## Added to the public entry points

- Explicit `sampling.engine="torch"` for fixed-cell PIMD and diffusion, with an atomic batched checkpoint. See [Torch workflows](../torch-materials-workflows.md). The CPU benchmark does not demonstrate CUDA performance.
- Bofill/Hessian-supported molecular instanton workflows and normal-regime golden-rule instanton validation. See [instanton methods](../instanton.md); reference-model agreement does not establish physical convergence.
- Fermionic sign reweighting, periodic winding choices and all-bead BZP pressure coupling for compatible exchange workflows. See [path-integral methods](../tier2-path-integral.md); signed observables need a sufficient average sign.
- RETIS minus-ensemble flux, REPPTIS and direct FFS Python APIs. See [interface sampling](../transition-interface-sampling.md). No TOML `path_sampling` workflow or infinite-exchange schedule is claimed.
- P=1 independent-electron surface hopping and classical multistate MASH. See [nonadiabatic scope](../nonadiabatic-iesh-multistate.md); neither is ring-polymer IESH.
- Budgeted, restartable `pyrpmd precision plan/run/status` for supported thermal-rate designs. See [precision-driven rates](../rate_precision.md); budget completion is not rate acceptance.
- Shared native CV implementations, Windows restart identity guards and deterministic scheduler tests. See [development](../development.md) and [roadmap](../roadmap.md).

## Compatibility and limits

No legacy input is intentionally assigned a new scientific identity by this documentation change. Existing checkpoint and numerical implementations are unchanged. Unsupported methods remain explicitly rejected. The legacy/native/Torch PIMD engine-convergence work is tracked separately in PR #93 and **is not described here as already merged**. There is no claim of cross-engine bitwise checkpoint conversion.

The main branch also retains experimental f-QCMD; h-CMD and a publication-level water spectrum are not claimed. The GPU performance task still has no actual CUDA measurement. See [f-QCMD](../qcmd.md) and [performance evidence](../t19-performance.md).

## Documentation verification

The English and Chinese README files now link first to installation and quickstart, then task-specific capability guides. The English and Chinese core manual chapters retain matching file names. Documentation CI builds both Sphinx sites with `-W --keep-going` and runs the repository-local Markdown link checker; external links are not checked by this offline job.

For prior 0.2.0 functionality and compatibility, see [0.2.0 notes](0.2.0.md). For original papers, system and observables, see the [source catalog](../manual/references.md) and [reproduction checklist](../manual/reproduction.md).
