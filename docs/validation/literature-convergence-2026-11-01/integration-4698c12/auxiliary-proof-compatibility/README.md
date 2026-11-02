# Preserved first auxiliary audit attempt

The Python 3.12 run collected 42 distinct cases: 41 passed, one skipped, and
no test failed or errored. The complete gate nevertheless **failed** because
its session-finish runtime audit assumed a pytest module key that was absent.
Python 3.13 and 3.14 were not started. The skipped resource-cap case explicitly
requires CPU 6, while this wrapper was pinned to CPU 8. All recorded source,
native-product and ABI-link guards remained unchanged.

This directory preserves the actual logs, JUnit, commands, driver and failed
receipt. It is not counted as a 42-case passing gate. The later v2 attempt
uses collected test module objects and the required CPU affinity. No new
physical evaluation or rate result was produced by this attempt.

See [assessment](initial-attempt-assessment.json),
[actual runtime receipt](py312/compatibility.json), and
[actual process exit](py312/launch-receipt.json).
