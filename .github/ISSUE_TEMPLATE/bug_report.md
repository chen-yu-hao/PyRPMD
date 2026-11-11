---
name: Bug report
about: Reproduce an incorrect result, crash, restart failure or interface regression
title: ""
labels: ""
assignees: ""
---

**Expected and observed behavior**
Describe the concrete difference and the first failing stage.

**Reproduction**
Provide the smallest input, exact command, and public PES/model reference needed
to reproduce it. Include units, atom order, seed and whether this is a fresh run,
resume or extension.

**Environment**
Paste `pyrpmd info`, the release/commit, execution backend and relevant native
step/fallback counters. Include the error and bounded log tail.

**Artifacts**
Include the relevant resolved input, summary and failure record. Preserve
original checkpoints; remove private paths or credentials before sharing.
