# Optional rendering-test dependency

This additive bundle preserves the exact original six-case test source and its
sealed publication inventory. Only the integration checkout test is revised:
`pytest.importorskip("matplotlib")` is inserted inside the single figure-rendering
case. The five numerical/provenance cases still run without plotting extras.
With the `plots` extra installed, all six cases still execute. No simulation,
analysis or plotting calculation changes.

The [supersession record](supersession.json) binds the old inventory, original
test hash and historical snapshot to the new integration test hash. The original
running checkout and the old frozen inventory remain unchanged. The patch is
[retained verbatim](rendering-portability.patch). The old six-pass evidence
remains an execution record for the original source. Root owns the separate
auxiliary v2 compatibility gate for this revised test; this bundle does not
claim that later execution has already passed.
