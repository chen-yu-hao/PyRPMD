# Workflow stability and compatibility

The executable registry is [`src/pyrpmd/stability.py`](https://github.com/chen-yu-hao/PyRPMD/blob/main/src/pyrpmd/stability.py).
It classifies a workflow's supported interface and validation scope. A label
does not change `ready`, `converged`, uncertainty, failed-history counts or a
withheld rate in a calculation result. Method-specific overrides take priority;
for example TRPMD spectroscopy is stable while f-QCMD is research.

| Level | Required evidence and contract |
| --- | --- |
| stable | Independent analytic or published-model checks, regression and interruption/restart tests, documented ensembles and units, explicit input/output schemas and failure limits. Maintained public fields follow the compatibility policy below. |
| beta | End-to-end implementation and targeted tests exist; coverage, external backend support, or interface design still has stated limits. Minor releases may change documented beta fields with release notes. |
| research | A model-specific approximation, prototype, or unsettled statistical procedure. Parameters and outputs may change. A run must identify the selected method and retain its limitations and raw diagnostics. |

The stable core covers `rate`, thermal single-surface `drpmd`, umbrella `fes`,
independent frozen-bias metadynamics, primitive distinguishable-particle `pimd`,
centroid `diffusion`, `site_rate`, and RPMD/TRPMD `spectroscopy`. Their label
applies within the registry's scope. Shared walkers, OPES, other FES methods,
surface/network/PCET workflows and external PLUMED are beta. RP-IESH, f-QCMD,
signed thermal NRPMD rates, subspace corrections and reactive-path research
ensembles retain research labels. A research feature cannot be made stable by
embedding it inside a stable task.

## Compatibility policy

Starting with the next published 0.2.x release, stable public input fields,
their physical meanings, documented summary keys and units remain compatible
across minor releases. New optional fields and new summary keys may be added;
readers should tolerate unknown output keys. A removal or incompatible rename
requires a documented replacement, a warning for at least one minor release,
and a major-version boundary. Beta/research changes are recorded separately.

Stable checkpoints remain readable across minor releases. Resume must preserve
the recorded integrator, ensemble, random stream and scientific identity; an
implementation that cannot preserve the stored convention must provide an
explicit migration path or an actionable refusal. Loading a checkpoint never
silently changes its scientific meaning. Bitwise identity between different
compilers, CPU architectures or backends is not part of the interface promise.

These criteria describe release maintenance, not the numerical uncertainty of
an individual run. Use [How to converge a result](../convergence.md) for that.

## Developer and validation records

User-facing workflows start in the [English manual](../index.md) or
[Chinese manual](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/zh/index.md). Their installation, input, potentials,
materials, execution and analysis chapters are the supported language paths;
the older standalone Chinese science notes remain historical references.

The following existing records are preserved at their original paths so
published links and scientific provenance remain usable:

- [Validation history](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation.md), [literature inventory](../literature-validation.md)
  and [method references](../manual/references.md).
- [Roadmap and open work](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/roadmap.md), [task records](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/tasks/README.md),
  [periodic-materials audit](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/periodic-materials-audit.md) and
  [implementation review](../implementation-review-2026-11-03.md).
- [Archive migration plan](../data-archive-plan.md) and
  [prepared release/archive inventory](archive-migration.md).
- [Bilingual contribution guide](https://github.com/chen-yu-hao/PyRPMD/blob/main/CONTRIBUTING.md) and
  [unreleased changes](https://github.com/chen-yu-hao/PyRPMD/blob/main/CHANGELOG.md).

Historical test counts and published-model comparisons apply to the commits
and environments stated in each record. They are not current-build results.
Moving raw files or retiring draft pull requests is a separate reviewed change;
this cleanup preserves both.
