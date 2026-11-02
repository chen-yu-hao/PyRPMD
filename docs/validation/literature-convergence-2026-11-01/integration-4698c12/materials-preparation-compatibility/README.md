# Nickel static reference and water preparation coordinator: integration gate

**45 newly added cases pass on each of Python 3.12, 3.13 and 3.14, with
zero failures, errors or skips.** The union with the unchanged earlier
270-case evidence contains **315 distinct passing case identities per
interpreter**. These are selected method and interface checks, not a new
execution of the full repository suite.

| Interpreter | New cases | Failures / errors / skips | Pytest wall time | Combined distinct cases |
| --- | ---: | ---: | ---: | ---: |
| Python 3.12.14 | 45 | 0 / 0 / 0 | 4.342 s | 315 |
| Python 3.13.18 | 45 | 0 / 0 / 0 | 4.572 s | 315 |
| Python 3.14.10 | 45 | 0 / 0 / 0 | 4.410 s | 315 |

The new selection has ten real single-bead EAM4 static cases and 35 water
preparation-coordinator fixtures. The signal fixtures use real bounded Python
stub processes; they do not integrate water. The EAM4 checks use the exact
already-built author-attributed library with SHA256
`b1673de4eb3d0673a49d22a9a6521095d0f2848338857ae10ee7adc9a97ab2f1`.
No model, dependency or extension module was rebuilt or installed.

## Exact tested state

The integration checkout began at
`67d55e9ce745a2e18906d7aff9327c7b8646a2cb`, with two independently frozen
bundles copied byte for byte:

- [Ni/H EAM4 source and static evidence](../../other-reactions/materials-source-index/publishable-files.json).
- [Water preparation coordinator and its review](../../water/preparation-coordinator-gate/publication-inventory.json).

The root also updated the Ni entry in `benchmarks/literature/campaign.json`
to link the new static evidence while retaining `not_reproduced`. This is the
only change relative to the previous gate's overlapping source guards, and
both old and new hashes are explicitly recorded. No active experiment source
changed.

The test process and a separate child interpreter verify their actual PyRPMD
native imports. All 832 integrated files, 596 original-checkout files,
15 external model/ABI files, six previously documented ABI discovery links
and each interpreter's installed-distribution inventory remain unchanged
through the gate. The six ABI links were not created, moved or removed here.
Each interpreter ran sequentially with one CPU affinity and all numerical
thread limits set to one.

## Evidence and counting

The [selection](selection.json), [guards](guard-before.json),
[execution driver](run_checks.py), [summary](summary.json) and
[case identity union](case-coverage.json) bind the scope and results.
The setup and summary scripts are preserved as byte copies of the executed
workspace scripts. Commands and start/end observations are retained in
[launches.json](launches.json) and the per-interpreter records.

| Interpreter | Actual gate receipt | JUnit |
| --- | --- | --- |
| Python 3.12 | [receipt](py312/compatibility.json) | [45 cases](py312/pytest.xml) |
| Python 3.13 | [receipt](py313/compatibility.json) | [45 cases](py313/pytest.xml) |
| Python 3.14 | [receipt](py314/compatibility.json) | [45 cases](py314/pytest.xml) |

The three new runs contain 135 interpreter-specific executions of the same
45 test identities. They are not 135 different scientific tests. The prior
270 cases were not rerun; their identity sets and checksummed receipts remain
in [the preceding gate](../new-stage-compatibility/README.md). Its historical
failed fixtures and missing-model skips remain preserved with their separate
successful resolutions. They are not silently replaced by this new pass.

This gate does not establish Ni diffusion, water equilibration or literature
agreement. It does not adopt or launch the separate 400 ps preparation
continuation. Remote GitHub Actions status is a separate observation.
