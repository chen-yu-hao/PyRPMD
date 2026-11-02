# MC continuation ownership checks across Python runtimes

Python3.12 executes all76 cases successfully, including real owned-process
capture, kernel resource caps, signal-safe cleanup and complete-cohort checks.
The observed local conda-forge3.13 and3.14 builds expose neither `os.pidfd_open`
nor `signal.pidfd_send_signal`: each passes69 applicable cases and explicitly
skips the same7 ownership cases. These skips are unavailable coverage, not
passes. Full ownership compatibility on those interpreters is not claimed.

The initial unmarked3.13 run is retained:71 nominal passes and5 failures.
Two error-path cases had also passed for the wrong reason before their injected
event was reached. They are among the seven now capability-marked cases.
The source-guard case, which stops before any child starts, remains runnable.
All test bodies and the active3.12 scientific source/test bytes are unchanged.

The separate101-case repair gate passed without skips on all three runtimes.
Combining nonoverlapping new passing identities with the prior796 gives872
on3.12 and865 on each local conda interpreter. Repeated executions do not
inflate these counts. The new original scientific run uses the fully tested
3.12 environment; no physical work occurred in these integration checks.

Python documents the platform-specific interfaces at
[os.pidfd_open](https://docs.python.org/3.13/library/os.html#os.pidfd_open) and
[signal.pidfd_send_signal](https://docs.python.org/3.13/library/signal.html#signal.pidfd_send_signal).
The local attribute observations establish this build's availability; they
do not imply the same absence on GitHub's official Python builds.
