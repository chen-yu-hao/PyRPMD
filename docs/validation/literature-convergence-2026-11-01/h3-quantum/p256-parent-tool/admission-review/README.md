# Final admission review of the P=256 parent tool

This additive revision resolves the independent review finding recorded in `revision.json`. A stop signal or wall deadline reached while guarding inputs, forming a command, or journaling the first worker must not admit another worker in the same scheduling loop. The source now checks both immediately before choosing a pending seed and immediately before `Popen`. Every seed that was never launched remains in the complete `unstarted_seeds` list even if command construction fails.

Only `dispatch_jobs` changed relative to the initial reviewed candidate. The scientific producer, fixed eight-chain design, expected settings, observer schedule, RNG, estimator and output schema are unchanged. Exact old source/test copies are in `initial-source/`. The earlier 22-file inventory remains an immutable historical record; its two source entries correspond to these preserved bytes, not the current source paths.

The authoritative current gate is `final/synthetic-gate.json`: **60 passed, zero skipped, 4.37 seconds**, one CPU, zero external PES evaluations or real dynamics. Four additional inert-process fixtures cover a SIGTERM after the first admission, a guard crossing the deadline, SIGINT while forming a command, and a failed command construction. The actual read-only preflight passes all 39 existing identities and creates no parent output.

`execution-authorization.schema.json` and `parent-output-contract.json` here carry the final driver identity. Their semantics are unchanged from the initial versions; use these copies for the execution authorization and future child adapter. The driver remains `benchmarks/h3/p256_parent_campaign.py`. The separate review receipt and root's observed resource decision are still required before any parent sampling. No such campaign has been launched by this tool's author.

`publishable-files-v2.json` one level above is the complete current publication inventory; it also retains all earlier candidate evidence. The source/test freeze is bound by `freeze.json`. Failed attempts and earlier tested code remain available for inspection; nothing was silently replaced in a prior receipt or manifest.
