# Preserved current-head Python 3.14 full-suite attempt

The ordinary unfiltered project selection at commit `36ec289` finished with
**7,268 passed, 17 failed, 145 skipped and 486 passing subtests** in 568.29
seconds wall time. It is a failed complete attempt. JUnit records 7,916
records including subtests; those are not advertised as 7,916 distinct
primary tests. Optional dependency/model skips remain explicit.

All tracked source, runtime/model, dependency and ABI-link guards remained
unchanged. Thirteen failures arose before local socket creation because
of sandbox permissions, one from the wrapper's overly long Unix socket
path, and two from limiting a four-thread BLAS fixture to two allowed CPUs.
The remaining wall-timeout fixture inherited pytest's accumulated CPU clock;
that test-only order dependence requires correction without changing the
production pilot's CPU budget.

See [complete report](report.json), [actual process receipt](execution.json),
[all failures](pytest.log), [JUnit](pytest.xml), and
[explicit classification](failure-classification.json). Subsequent fixes or
environment corrections cannot overwrite this first attempt.
