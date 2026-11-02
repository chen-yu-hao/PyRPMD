# Independent review of the P256 parent campaign tool

No remaining blocker was found in the final bound source. This is a source,
protocol and bounded fixture review; it is not authorization to launch workers
or a claim of physical convergence.

The reviewed driver SHA256 is
`7934c1d7f695ce43337d11171cbad8c46ff95311490b755bf4191a56e00c0da3`.
The test source SHA256 is
`57e78795ecda9fa754fef5a6ca2180e7b4f262ada7b763b76c8f8281506d7634`.
The [complete review receipt](independent-review.json) binds the fixed design,
all input receipts, both resolved findings and the operational limits.

Two scheduler findings were corrected before launch: stop/deadline checks now
run at each admission boundary, and a known nonzero exit from an already owned
worker prevents another launch. Failed children remain owned until reaped.
The [initial findings](initial-findings.json) are retained independently of the
implementation owner's historical inventories.

The owner recorded 61 passing tests using a small analytic sampler and inert
processes, with no BKMP2 evaluation. The reviewer independently ran four selected
inert-process checks: signal at first admission, an already-exited worker failure,
repeated signals during journal-error draining, and retaining exclusion when
normal drainage cannot be confirmed. All four passed in 1.05 seconds; 57 cases
were deselected. These are four repeated checks from the 61-case suite, not four
additional distinct cases. Exact [log](focused-pytest.log) and
[JUnit](focused-pytest.xml) are included.

The reviewer also executed the read-only preflight: all 39 source, runtime,
model and original-state guards passed. It created no output root and launched
no worker, potential evaluation, parent sampler or physical trajectory.

The historical four chains must reproduce every one of the 16 non-observation
checkpoint payload arrays byte for byte. The source adds only deterministic
snapshot observation. All eight fixed chains, 200 observations each, complete
identity and geometry checks, and every registered sampling flag are required
for child eligibility. The independent uncertainty units remain eight chains.
The observed 1600 configurations are correlated, and the sampling flags do not
provide mathematical accuracy bounds.

Root must separately record actual drained-queue evidence and current resource
allocation. No other dispatcher or direct call to the underlying producer may
write these new output directories. The scheduler may terminate and reap only
its own process groups on failure or deadline, preserving atomic checkpoints;
unconfirmed drainage retains the exclusive lease. No parent, child, seed,
count or budget extension was launched or authorized by this review.
