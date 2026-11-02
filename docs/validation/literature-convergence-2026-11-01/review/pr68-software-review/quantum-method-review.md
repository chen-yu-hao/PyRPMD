# Independent review of the frozen quantum recrossing method

## Outcome and scope

No factor-of-two, bead-temperature, sign, channel-count, physical-unit, or same-chain covariance defect was found in the reviewed frozen implementation. The existing 21 self-contained tests passed in 1.31 seconds. The driver and all its registered core/native dependencies match the child protocol; the analyzer matches its separate analysis prespecification. No running-primary report, trajectory, or partial rate was inspected.

This review covers source, registered protocol and synthetic tests. It does not declare the running campaign converged or establish literature agreement. The original six-file software/source review inventory was left unchanged.

## Flux normalization, sign and units

Write the centroid surface gradient as a, and g = sum(a_i^2/m_i). Each bead momentum is drawn with variance m_i P/beta. Therefore the mean bead momentum has variance m_i/beta and the surface velocity v = a dot p_centroid/m has variance g/beta, independently of P. The code uses fs = sqrt(g/(2 pi beta)), so v/fs is a normal variable with variance 2 pi and E[abs(v)/fs] = 2.

The parent configuration law already contains delta(s1) fs(q). Dividing the flux weight by fs removes that configurational weight. For the antithetic pair, time-reversal symmetry of the Maxwell law gives

    E[(v/fs) (H(s_plus(t)) - H(s_minus(t)))] = 2 kappa(t).

Dividing by the sampled abs(v)/fs normalization therefore estimates kappa with no additional bead or factor-of-two correction. At an infinitesimal positive time, the numerator equals the denominator for each pair. The existing finite-difference test also checks that the implemented velocity has the sign of actual centroid surface motion.

The RateSurface at xi=1 is s1 in bohr. Positive s1 describes products for the registered forming/breaking bonds. Its two equivalent channels use a mass-preserving permutation. The classical TST integral already includes both channels; kappa is dimensionless and introduces no further channel factor. Maxwell and channel RNG keys are separate; neither timestep, job order nor restart count enters them.

The classical TST scalar supplies cm^3 molecule^-1 s^-1. Both quantum/classical partition corrections and kappa are dimensionless. The driver converts femtoseconds to atomic time once at Dynamics construction. The analysis applies no further rate conversion.

## Fixed cohort, pairing and uncertainty

The protocol fixes eight unique seeds, 200 parents per seed, and the exact saved parent sweep indices. The driver binds the full parent settings, checkpoint integrity, domain normalization and report hashes. Native execution and unchanged source are required. Analysis normally refuses any incomplete seed cohort; available-only mode cannot make the complete-cohort acceptance criterion true.

Both timesteps use the same keyed initial momenta and channel choices. The analyzer compares their hashes and reconstructs the signed numerator and absolute-flux denominator directly from each integrity-checked chunk. Plateau and timestep comparisons are differences within each seed, followed by a Student-t interval across seeds. They are not comparisons treating time points or individual children as independent observations.

For each seed the full log-rate is log(k_classical_TST) + log(Z_TS_quantum/Z_TS_classical) - log(Z_R_quantum/Z_R_classical) + log(kappa). Combining these terms before the eight-seed interval preserves the surface-partition/transmission covariance. The code checks the variance identity explicitly. The reported central full rate is the geometric mean of the seed rates; arithmetic and geometric transmission means are separately labelled. Multiplying the displayed arithmetic kappa by geometric QTST need not equal that geometric full-rate estimate.

## Extension safeguard

The executable restricts choices to the registered 200/400 fs horizons and pair batches 0 or 0+1. A 400 fs extension restores the exact completed 200 fs states and prefix. A new pair batch uses global pair indices 2 and 3; existing draws are unchanged.

However, neither driver nor analyzer itself authorizes an extension by checking a prior all-eight-seed decision report. The study owner confirmed that no extension has launched and committed to the following external safeguard: first retain the complete primary all-eight-seed analysis and its source/raw hashes; then write an immutable decision receipt identifying the exact failed registered plateau or precision gate, next horizon/batch and every seed/timestep, before continuation. Partial diagnostics cannot trigger continuation. The final evidence must contain this receipt if an extension occurs. This is a procedural requirement, not a source change made by this review.

## Interpretation limits retained

- Ordinary eight-seed Student-t intervals are nominal sampling intervals, conditional on adequate parent equilibration and the reported estimator. They are not exact distribution-free or sequentially adjusted confidence guarantees after conditional extensions.
- The per-seed estimator uses a finite-sample ratio, and the final estimate averages its logarithm. Neither ratio bias nor log-transformation bias is separately corrected. Adequate effective sampling is still required; no problem is inferred from a partial campaign value.
- The combined fixed-P acceptance flag requires component point estimates within the registered paper tolerance, plus separate precision/numerical gates. It does not require the entire new confidence interval to lie in the paper tolerance band; that is a separately reported diagnostic.
- Minimum pair distances are observed at 1 fs boundaries. Zero observed crossings does not prove the absence of between-frame crossings. The full-H error screen is not a bound on rate bias.
- These child tests do not establish continuum-bead or domain convergence of kappa. Separate equilibrium controls do not replace dynamic controls, and the result does not validate the production umbrella sampler.
- The analyzer validates the deterministic reactant Hamiltonian explicitly. The completed classical TST reference is read as a scalar and its file hash is recorded, rather than revalidating all its physical metadata. Preserve its independently audited immutable artifact with the final report.

## Reproducible bounded verification

Only tests/test_h3_quantum_recrossing_analysis.py and tests/test_h3_quantum_surface_recrossing.py were run, with native/BLAS thread limits of one. They use synthetic geometry, keyed Maxwell draws and temporary checkpoint fixtures; no external-PES dynamics or campaign analysis is launched. See quantum-method-review-test-receipt.json for the exact command, source hashes and protocol checksums, and quantum-method-review-tests.xml/log for the 21-pass result.
