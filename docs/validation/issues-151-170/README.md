# Issues 151–170 implementation record

The implementation starts from `f410b97` and addresses the twelve open issues
selected on 2026-11-10. Existing PR implementations were integrated and repaired
where independent numerical checks or regression tests found problems.

| Issue | Implemented behavior | Remaining acceptance |
| --- | --- | --- |
| 151 | Mainline PMF boundaries, log weights, ESS and frozen-production gates verified | cn53 final run |
| 153 | Runtime xi branch guard; continuation; stability ladder; read-only estimator comparison; all walker failure records and recovery | Low-temperature effectiveness/failure-frequency campaign |
| 155 | Small-curvature effective mass shared by both rate halves; cache tied to model/path/settings | Full curvature-family and H3 transfer study |
| 156 | Harmonic/RPH/DVR normal energies, including correct soft-mode DVR boxes | Full physical model comparison |
| 157 | Instanton, bead-weighted and least-action path cores; projected covariance and exact centroid fallback | H3 path-source transfer comparison |
| 158 | Coupled channels as U†HU of the existing Cartesian 2D model; full-basis matrix/flux/partition/rate oracle; pipeline entry | General Frenet metric rate remains unsupported; large H3/channel convergence study |
| 165 | Five analytic tutorials passed at declared budgets; reactive thermal dRPMD tutorial; scheduled reference comparisons | Full three-method exchange campaign prepared for CI |
| 166 | Bounded parent checkpoints, append-only records, sparse allocation identity and compact summaries | cn53 final run |
| 167 | Stage backend/count plans, honest unknowns, runtime counters, affine fractional CV and full constant-friction native blocks | cn53 final run |
| 168 | Dimensional input warnings, resource readiness, electron parity, compatible rate aliases and four task starters | cn53 final run |
| 169 | Stability registry, concise bilingual READMEs, contribution/citation/changelog, fast/campaign CI split, archive inventory and wheel preparation | Release upload, raw-data relocation and historical PR disposition |
| 170 | Specific same-sample UI/MBAR bias diagnostic; width/moments in outputs; narrow defaults saved explicitly with legacy restart compatibility | Scheduled quantitative exchange comparison |

[Local evidence](local-validation.json) reports 10,059 distinct passing test
nodes across the full first run and targeted corrections, with 242 skips and
no unresolved failed nodes. This is explicitly not one uninterrupted full-suite
pass on the final commit. New issue-specific tests and real short BKMP2 runs
supplement the library suite. The five analytic tutorials used unchanged seeds,
budgets and predeclared criteria; all five passed. The reactive short run
observed four products among 256 complete thermal collisions.

[Sampling details](issue151-153.md) and [channel mathematics](issue158.md)
describe the new behavior and its scientific limits. Native claims are supported
by measured counters and callback profiling. No short test marks a production
rate or subspace correction physically converged. cn53 remains a separate
runtime gate; a queued task is not reported as a completed validation.

The prepared wheel has not been uploaded as a release. The validation archive
inventory is additive; original data and the main checkout's existing local
changes were preserved. No new environment or dependency installation was used.
