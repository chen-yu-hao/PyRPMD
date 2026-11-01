# One completed-cohort lease retirement and later failed postcondition

The recorded root operation followed the separate lifecycle review and the
actual normal exit of the owned sampler with all eight workers drained. It
moved the exact stale lease to a new quarantine path using Linux
`renameat2(RENAME_NOREPLACE)`. The immediate receipt correctly records return
zero, absence at the original path and preserved directory/owner inodes.

A later observation found a new directory and owner inode at the old path,
with the identical owner bytes; the quarantined original remained intact.
Therefore **durable absence was not established**. See the independent
[later postcondition review](../stale-lease-postcondition-review/independent-review.json).
The cause remains unexplained. There was no second retirement. PID 4 is not
used to infer process liveness. The already admitted read-only complete-cohort
analysis finished normally; no new sampler was admitted to this campaign.

The separate owned-directory probe used no campaign data or lock. Its other
process observed the synthetic rename, and the original path remained absent
through that process's exit. This limited negative probe did not reproduce
the campaign anomaly and does not identify its cause.

The source, immediate receipt, original owner record, synthetic observations
and actual terminal response are retained byte for byte. The new reviewed
[archive policy](../archive-preparation-v2/README.md) explicitly binds the
anomaly and requires a root no-writer attestation for read-only archival.
It does not silently treat the original campaign as unlocked. Any later
sampling must use a separately adopted new isolated verified restore.
