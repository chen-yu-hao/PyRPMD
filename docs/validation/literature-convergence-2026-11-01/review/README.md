# Independent sampler review and geometric Andersen correction

The fixed 5 fs global momentum refresh used in the earlier H3 P128 campaign
locks closely to several internal ring-mode phases. This leaves some
configuration modes cold despite repeated correct Maxwell momentum draws.
The mechanism is derived analytically, reproduced using the current `Dynamics`
implementation, and observed directly in the saved H3 reactant endpoints.
An optional geometric collision clock now removes the fixed-period alignment
while preserving legacy fixed-schedule replay.

This establishes a real sampling pathology and a tested correction. It does not
determine the sign or size of the complete H3 rate error, and it does not certify
agreement of a molecular rate with the literature.

## Analytic and numerical evidence

For a free mode, one Cayley step has rotation angle
`theta = 2 atan(omega*dt/2)`. After `M` deterministic steps and a complete
momentum refresh, the missing position variance is multiplied by
`cos(M*theta)^2`. From collapsed beads the variance ratio after `n` intervals is
`1 - cos(M*theta)^(2n)`.

At P128, 300 K, hydrogen mass, dt = 0.1 fs and M = 50, mode 37 has a variance
relaxation time of 4.169 ps. Its predicted variance ratios after 500 and 1500 fs
are about 0.113 and 0.302. The first diagnostic runs used 32 independent free
particles with three Cartesian components each:

| Control | Mode 37 at 500 fs | Mode 37 at 1500 fs |
| --- | ---: | ---: |
| Fixed 50-step global refresh | 0.10894 | 0.30159 |
| Independent uniform waits from 25 to 75 steps | 1.17323 | 0.93669 |
| PILE-L | 1.02211 | Not run |

The randomized uniform waits are a diagnostic control, separate from the new
production geometric distribution. These are finite-ensemble variance estimates,
not exact values; a complex Fourier mode has 192 independent real components
in this experiment, giving roughly 10% relative sampling uncertainty for its
variance. The exact formulas, all mode measurements, source hashes and the pCN
Gaussian-prior check are in [free-mode-review.json](free-mode-review.json).
The reproduction script is
[review_free_modes.py](../../../../benchmarks/h3/review_free_modes.py).

The separate production-workflow test calls the actual umbrella sampler for
5000 steps with a centroid-only bias, which leaves the internal free modes
unchanged. It uses the new geometric mean interval of 50 steps:

| Actual umbrella workflow, after 500 fs | Mode 13 variance / equilibrium | Mode 37 variance / equilibrium |
| --- | ---: | ---: |
| Fixed schedule | 0.375736 | 0.106354 |
| Geometric schedule | 1.007251 | 0.985377 |

These values are captured as properties in the final JUnit receipt and repeated
in [andersen-gate.json](andersen-gate.json). The exact final q/p/RNG checkpoints
are retained as `workflow-fixed-checkpoint.npz` and
`workflow-geometric-checkpoint.npz`; their hashes are in the manifest.

### Evidence from the actual earlier H3 calculations

The read-only endpoint review selected every window with center <= 0.1 from
the two previously audited long H3 runs. This geometric selection gives 64
independently seeded task endpoints after 1500 fs. All selected incoming-atom
bead distances from either H2 atom exceed 26.13 bohr. Every file hash matched
the earlier audit's input-checkpoint receipt.

| Internal free coordinate | Mode 37 variance / equilibrium | Between-endpoint standard error |
| --- | ---: | ---: |
| Incoming H atom | 0.301293 | 0.019112 |
| H2 center of mass | 0.307899 | 0.024153 |

Both agree with the approximately 0.302 fixed-clock prediction. No dynamics or
PES evaluations were performed for this check. Individual checkpoint hashes,
the two runs separately, mode 13 results and the analytic predictions are in
[saved-h3-modes.json](saved-h3-modes.json). Reproduce it with
[review_saved_modes.py](../../../../benchmarks/h3/review_saved_modes.py) and
the archived original checkpoints. The remaining physical approximation is the
negligible but nonzero interaction at finite reactant separation.

## Production implementation

The new setting is `sampling.andersen_schedule="geometric"`. It draws a positive
integer wait with probability `p(1-p)^(n-1)`, where
`p = 1/andersen_interval`, immediately after each global momentum refresh.
The schedule applies to umbrella FES, adaptive umbrella stages and gas-phase
rate parent sampling. All other explicit uses fail during input/preflight or at
the actual specialized workflow entrypoint, before dispatch or output creation.
The [user guide](../../../andersen-sampling.md) documents the distribution and
scope.

The absolute next collision step is saved alongside the existing random
generator state. Native blocks stop at this deadline. On a failed block, accepted
steps and observations remain committed. A refresh followed by zero successful
steps restores the previous momenta, RNG and collision deadline before saving.
Native block size, checkpoint boundaries and interrupted restart do not redraw
pending waits.

An omitted setting retains the old fixed schedule without adding configuration
or payload fields. The two fixtures
[andersen_legacy_window.json](../../../../tests/data/andersen_legacy_window.json)
and
[andersen_legacy_parent.json](../../../../tests/data/andersen_legacy_parent.json)
were captured from the pre-change implementation at commit
`274f8289fcc2392278ba9ed894bdae88a92f8b01`. They independently verify old
fingerprints, momentum/RNG sequences, observations and completed-checkpoint
replay. Explicitly changing the setting changes the scientific fingerprint and
requires a new run directory.

## Final verification

The final focused gate passed **318 tests and 14 subtests**, with no skips or
failures, in **107.49 s** pytest time. It covers:

- the new collision distribution, old fixtures and malformed clock rejection;
- native/Python umbrella agreement for all three supported integrators;
- interrupted umbrella and constrained-parent replay across block-size changes;
- initial, refresh-boundary and partial-block PES failures;
- Python observation rollback and adaptive pilot/production wiring;
- the existing FES block, rate/MBAR, PILE-L, input, adaptive and native-override
  compatibility suites.

The command ran with one CPU affinity and one BLAS/OpenMP thread. Peak child RSS
was 234784 KiB (about 229 MiB). All 29 captured code/test/configuration hashes
were unchanged across the gate. JUnit counts its 14 subtests as additional test
cases, so its raw `tests=332` corresponds to the pytest summary above.

Receipts:

- [andersen-gate.json](andersen-gate.json): command, environment, hashes and
  numeric free-mode result.
- [andersen-gate.log](andersen-gate.log) and
  [andersen-gate.xml](andersen-gate.xml): final pytest output.
- [andersen-docs.json](andersen-docs.json) and
  [andersen-docs.log](andersen-docs.log): strict English documentation build with
  `sphinx -n -W --keep-going -b html`.
- [manifest.json](manifest.json): exact owned files and evidence checksums.

The earlier 55-case dedicated run is also retained as `andersen-tests.log/xml`;
the final gate supersedes it. A new complete repository suite was not run for
this scoped correction. The prior full-suite result belongs to the earlier
source snapshot and is not relabeled as evidence for this change.

The NPZ observations and checkpoints are raw evidence, separate from the small
text receipts. Their availability in a published archive should be checked
against the manifest rather than inferred from Git's ignore rules.

## Scientific review boundaries

The review found no missing bead-number, mass, equivalent-channel or flux-metric
normalization factor in the inspected H3 production expressions. The existing
force/flux umbrella weight and centroid momentum convention agree with the
independent classical coarea calculation. Slow orbital orientation alone does
not establish slow relative H2 orientation, and the P1 problem cannot be caused
by internal ring modes. Separate exact classical work identifies finite-sample
PMF uncertainty, non-Gaussian umbrella-integration bias and inadequate barrier
support under weak umbrellas; that work is documented by the classical campaign
owner rather than being attributed to this collision-clock mechanism.

Background primary sources are
[Korol, Bou-Rabee and Miller, JCP 151, 124103 (2019)](https://arxiv.org/abs/1907.07941)
for the Cayley map and its integration properties, and the
[NumPy geometric distribution documentation](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.geometric.html)
for the discrete wait law. The refresh-period relaxation derivation and endpoint
measurements above are independent calculations made during this review.
