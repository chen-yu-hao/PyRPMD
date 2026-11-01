# Can the original coarse trajectories replace newly paired coarse branches?

**The requested exact correspondence condition fails for all eight chains. No study that reuses all 25 original coarse segments while generating only fine-step branches is proposed or adopted here.** The original trajectories remain valid records of their own experiment; this check does not make them identical to a fresh CPU-context continuation.

The comparison is separate from the [completed 64 ps timestep control](../timestep-paired-control/completed-original-cohort/README.md), whose two failed equivalence checks remain unchanged. No diffusion estimate was recomputed or selected during this feasibility check.

## Exact cohort and time alignment

[plan.json](plan.json) was written before reading the comparison arrays. It fixes seeds 1101–1108, original NVE segment 0, the new `dt025` branch and every one of their 801 observations. Both histories use dt = 0.25 fs and a recording stride of 20 steps, so their actual NVE cadence is **5 fs**, from 0 through 4,000 fs. The original absolute step 400,000 corresponds to control step 0. Both saved `segment_step` arrays equal 0, 20, …, 16,000 exactly. The original preparation cadence is 100 fs; no 10 fs grid or interpolation is used in this comparison.

All input arrays came from the previously verified, separately extracted original and paired-control archives. Their original manifests and selected chunk hashes were rechecked before and after the operation. Every chain's four full-precision checkpoints at 1, 2, 3 and 4 ps were also compared. No frame, seed, molecule, coordinate transform or alternate alignment was selected after inspecting differences.

Original molecular observations were stored as float32, whereas the numerical control retained float64. A direct float64-versus-float32 mismatch alone would be inconclusive about trajectory identity. The declared check therefore also casts each control observation to the **original stored dtype** and compares every element exactly. The full-precision position and momentum checkpoints supply a separate comparison unaffected by that observation-format difference. No equality tolerance was widened.

## Shared start, different saved continuations

All eight original 100 ps source coordinates match the control's saved initial coordinates. The saved RNG state before the pending draw matches, and the saved RNG state after the draw matches every original first-segment checkpoint. At time zero, the COM positions, COM velocities and oxygen coordinates agree after conversion to the original float32 storage. The complete canonical momentum immediately after the original draw was not separately checkpointed; the evidence for that draw is the matching before/after RNG states, frozen sampling code and initial molecular velocities. This check generated no new random numbers.

The later saved trajectories differ. Seven COM traces first fail exact equality at the first 5 fs observation; seed 1103 first fails at 10 fs. All 32 full-float64 position/momentum checkpoint comparisons fail exact equality.

| Seed | First unequal COM record, fs | Maximum initial force-cache difference, kJ/mol/nm | Maximum coordinate difference at 1 ps, bohr | Maximum coordinate difference at 4 ps, bohr |
| --- | ---: | ---: | ---: | ---: |
| 1101 | 5 | 0.00137499 | 0.0875950 | 10.2815 |
| 1102 | 5 | 0.00170898 | 0.0460684 | 9.64992 |
| 1103 | 10 | 0.00158691 | 0.0459437 | 15.2768 |
| 1104 | 5 | 0.00136821 | 0.150181 | 9.10472 |
| 1105 | 5 | 0.00146484 | 0.0181753 | 10.8110 |
| 1106 | 5 | 0.00139109 | 0.547050 | 7.91234 |
| 1107 | 5 | 0.00170898 | 0.122345 | 13.8113 |
| 1108 | 5 | 0.00134277 | 1.98378 | 10.9680 |

These are direct, labelled-atom, unwrapped coordinate differences; no minimum-image transformation or best-fit alignment was applied. They are not diffusion coefficients or statistical timestep errors. All exact values, initial checks, force differences, observation dtypes and checkpoint results appear in [report.json](report.json). [all-time-differences.csv](all-time-differences.csv) retains all 6,408 seed/time rows.

The initial recomputed forces differ at the already documented CPU force-cache precision scale. Later divergence is consistent with amplification of those small numerical differences during chaotic liquid dynamics. This saved-data comparison does not isolate every source of that amplification, and it does not show that either same-step trajectory is an incorrect equilibrium sample. It does show that the old coarse record cannot be represented as the exact new coarse branch merely by rounding storage precision.

## Consequence and limits

Root's conditional reuse rule required exact correspondence before proposing an all-eight, all-25-segment design that would launch only fine-step branches. That condition is false. Accordingly, no such registration, branch selection, resource request or physical execution is created. The current control is not enlarged by appending the original coarse records, and its failures are not recalculated away.

The distinction matters: a small initial CPU roundoff difference can break future trajectory identity while remaining consistent with the same physical model. Establishing a new statistical coupling or a new timestep study would require a separately declared design; this identity check supplies no permission to treat approximate correspondence as exact.

The read-only execution exited zero in session 14201, using 1.291 CPU seconds, 1.336 wall seconds and 85,160 KiB peak RSS on CPU 4 with one numerical thread. [check_saved.py](check_saved.py) consumes no PES evaluation, Context, RNG draw or integration step. The original protocol, original data, completed control and all frozen scientific sources remain unchanged.
