# Cross-Python check of the corrected wall fixture

The same integration-only wall-supervisor test correction passed on the existing Python 3.12 and 3.13 interpreters. Both executions emulate a 1,000-second outer pytest CPU age, use the fixture's fixed 0.1-second synthetic CPU clock, and retain its real monotonic timeout and child cleanup/retry checks. They ran sequentially with one CPU and one library thread; no dependencies, production source or original runtime files changed.

| Interpreter | Selected case | Result | Pytest time |
| --- | ---: | --- | ---: |
| Python 3.12 | 1 | Passed, no skips | 1.096 s |
| Python 3.13 | 1 | Passed, no skips | 1.086 s |

The [cross-Python receipt](cross-python.json) binds the exact corrected test, unchanged production/original-source guards, commands, runtimes, logs and JUnit files. The existing [Python 3.14 check and exact supersession](../hch4-wall-fixture-supersession/supersession.json) remain immutable in their separate bundle. These are repeated executions of **one already counted distinct test**. They do not increase the earlier 596-case method-test union or assert a new complete-suite result. No PES evaluation or physical dynamics occurred.
