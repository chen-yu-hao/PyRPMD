# Review status and execution boundary

The [independent method review](independent-method-review.json) found no blocker
in the mathematical design after two clarifications. It binds the final
[proposal](proposal.json), [derivation](README.md) and complete saved-data
[diagnosis](saved-pilot-diagnosis.json). The review performed no new PES call,
dynamics step or random draw.

The explicit `calibration.eigenaxis_order` rule is authoritative for the new
probe geometry: diagonalize the Hessian in the declared scaled coordinates,
order its eigenvalues ascending, and use
`mean_x +/- S*eigenvector_i/sqrt(eigenvalue_i)` in the original chart. These are
unit Mahalanobis displacements; the shorthand “covariance eigenaxes” elsewhere
does not mean unscaled Cartesian principal axes. The original-chart covariance
is `S*H_y^-1*S.T`.

The entry-separation diagnostic compares the same 16 selected draws per seed
at both separations. The main original-domain denominator still uses all 128
entry draws per seed. The two estimates must not be substituted for one another.

**The proposed 4,000-geometry experiment is not implemented or authorized to
run.** Its 800 calibration, 3,072 fresh diagnostic and 128 entry-control
geometries are proposed maximum counts, not completed observations. Required
next steps are isolated implementation, analytic/synthetic and lifecycle tests,
independent implementation review, source/runtime freezing, and a separate
concrete resource adoption. No existing experiment, physical source, input,
target domain, PES or previous publication inventory was changed.

The only executed new computation was the read-only saved-data diagnosis.
[diagnostic-validation.json](diagnostic-validation.json) records its input and
source hashes and the retained initial status-schema mismatch. A successful
proposal-efficiency screen would still leave physical-domain reliability,
unseen tails, full separation convergence, absolute normalization, quantum
corrections and transmission coefficients unresolved.
