# Portability and regression audit — 1 November 2026

This audit records the software tests behind the Tier 3 work in
[PR #62](https://github.com/chen-yu-hao/PyRPMD/pull/62). Scientific comparisons and
convergence evidence belong to the other parts of this literature audit.

## Source versions and corrections

| Commit | Change relevant to this audit |
| --- | --- |
| `45763a7ab84100b376e2e5e5df06a8013a82659b` | Original ecosystem, distribution, and validation implementation |
| `b3e229309e6db4dba08f58a75e387b7b5628f169` | Reopen an exported trajectory with a writable handle before file synchronization on Windows; add a three-format regression |
| `4ba870dd7121a31e0e0987a90b194fec3f788ffd` | Pass explicit MPI oversubscription options to CI launchers and preserve all Python matrix jobs after an individual failure |

The observed CI failures were investigated independently:

1. **New Windows export defect.** The CPython 3.11 Windows wheel compiled and
   imported, but the installed smoke test reached `os.fsync` on a file reopened
   read-only. Windows rejected that descriptor with `OSError: [Errno 9]`.
   Commit `b3e2293` changes the reopen mode to `rb+`. Its regression emulates
   the writable-descriptor requirement for extxyz, H5MD, and ASE trajectory
   outputs. See [the failing job](https://github.com/chen-yu-hao/PyRPMD/actions/runs/41311481524/job/124863388143)
   and [the retained failure excerpt](github-windows-export-failure.log).
2. **Existing MPI launcher configuration defect.** The original CPU workflow
   used `OMPI_MCA_rmaps_base_oversubscribe=1`. The `ubuntu-latest` runner now
   supplies Ubuntu 26.04 with Open MPI 5.0.10 and PRRTE. After all Python tests
   passed, its four-rank launch failed at process allocation. Comparing the
   workflow at base `2f907ba` and `45763a7` shows that only the Python version
   matrix had changed; the old launcher configuration predated Tier 3. Commit
   `4ba870d` uses the public `--map-by :OVERSUBSCRIBE` option in `tests.yml`
   and `fused-native.yml`. It also sets `fail-fast: false` in the CPU matrix.
   See [the failing job](https://github.com/chen-yu-hao/PyRPMD/actions/runs/41311481235/job/124862619022)
   and [its unit-test/launcher excerpt](github-cpu312-pytest-mpi.log).
3. **Runner-dependent UCX transport failure.** At `4ba870d`, the ecosystem job
   reached LAMMPS initialization and MPICH's `MPI_Init_thread`, then UCX tried
   to create an unsupported RDMA queue on the Azure runner's `mana_0` device.
   The process aborted before the engine or quantum-science tests began.
   The same ecosystem job had passed at `45763a7`; the failure depends on
   the runner's exposed network devices. The follow-up workflow change sets
   `UCX_TLS=tcp,sm,self` for this single-host job. See
   [the failing job](https://github.com/chen-yu-hao/PyRPMD/actions/runs/41312690867/job/124866205959)
   and [the exact runtime-gate excerpt](github-ucx-runtime-failure.log).

The MPI correction changes CI process placement. The Windows correction changes
the handle used to synchronize an already-written temporary export. Neither
correction changes the integration equations, forces, trajectory values, or
constraint equations.

## What ran successfully on the original commit

[The API snapshot](github-45763a7.json) records each workflow and job separately,
including incomplete and cancelled jobs. Successful jobs remain attributable to
their tested commit even when a later source update cancels other jobs in the
same workflow.

| Check | Observed result at `45763a7` | Evidence |
| --- | --- | --- |
| Full CPython 3.12 unit suite | 4,452 passed, 140 skipped, 357 subtests passed; subsequent MPI launch failed | [Job and excerpt](github-cpu312-pytest-mpi.log) |
| Conda Linux packages | CPython 3.11, 3.12, 3.13, and 3.14 builds, imports, dependency checks, and installed numerical/export/CLI smoke tests passed | [Conda matrix](github-conda-matrix.log), [run](https://github.com/chen-yu-hao/PyRPMD/actions/runs/41311481435) |
| Portable macOS arm64 wheels | All four CPython 3.11–3.14 wheels and their installed smoke tests passed | [macOS wheel log](github-macos-wheel-matrix.log), [job](https://github.com/chen-yu-hao/PyRPMD/actions/runs/41311481524/job/124863388158) |
| Portable manylinux x86-64 wheels | All four CPython 3.11–3.14 wheel builds and installed smoke tests passed | [job](https://github.com/chen-yu-hao/PyRPMD/actions/runs/41311481524/job/124863388219) |
| Installed source/platform wheel workflow | Ubuntu CPython 3.11–3.14 and macOS CPython 3.12 passed in the push workflow | [run](https://github.com/chen-yu-hao/PyRPMD/actions/runs/41311421108) |
| Optional engines and readers | Ecosystem interoperability workflow passed | [run](https://github.com/chen-yu-hao/PyRPMD/actions/runs/41311481345) |
| Documentation | Strict documentation workflow passed | [run](https://github.com/chen-yu-hao/PyRPMD/actions/runs/41311481360) |

The portable release verification job was skipped after the Windows failure.
These original results therefore do not certify a complete twelve-wheel release.
No package publication was attempted.

## Final combined local regression

The combined audit working tree based on `4ba870d` passed the full test suite
under **CPython 3.12.14: 4,527 passed, 97 skipped, and 365 subtests passed**, with
no failures or errors, in 739.88 seconds. The command returned zero. The run
included the real BKMP2 library, OpenMM, LAMMPS, ASE, MDAnalysis, and the local
socket protocol tests. All **450 recorded source files** had identical SHA-256
hashes before and after the run.

The [full-suite manifest](full-suite.json) identifies the base commit, the dirty
working tree, every tested source hash, dependency versions, compiled-library
hashes, command, and output hashes. The [log](full-suite.log) and
[JUnit report](full-suite.xml) retain the original results. The 932 warnings
include NumPy array-shape deprecations in ASE and a Matplotlib colormap pending
deprecation in the report renderer. This evidence describes the recorded source
snapshot; subsequent changes need their own focused checks.

## Direct CPython 3.13 and 3.14 numerical coverage

Fresh conda-forge interpreters were created in isolated workspace environments.
The source was compiled into a native platform wheel and installed independently
for each interpreter. The tests imported those installed packages. All 112
installed Python modules matched the `b3e2293` source by SHA-256, both compiled
extensions loaded, and native capability checks reported OpenMP and all three
integrators. The package modules and selected test files remained unchanged
during each run.

The selected **28 test modules** cover dynamics validation, analytical
path-integral and quantum-statistical checks, thermostats, all three integrators,
native/Python agreement and native ownership/lifetime contracts, periodic cells,
checkpoints, bond constraints and unsupported-workflow guards, and all three
trajectory formats. They provide numerical/runtime coverage beyond installation
smoke tests.

| Interpreter | Initial sandbox run | Local-socket rerun | Resolved result |
| --- | --- | --- | --- |
| CPython 3.13.18 | 978 passed; 64 subtests passed; 10 skipped; 2 denied socket creation | Both socket tests passed in 0.68 s | 980 selected test cases passed, plus 64 subtests |
| CPython 3.14.10 | 978 passed; 64 subtests passed; 10 skipped; 2 denied socket creation | Both socket tests passed in 0.81 s | 980 selected test cases passed, plus 64 subtests |

The original logs and JUnit reports retain the two `PermissionError` results.
Only those two cases were rerun with local socket access. They exercise the real
i-PI protocol, including the periodic cell inverse and virial. The results above
combine these explicitly recorded invocations; they are not a claim that the
initial sandbox invocation returned success.

The ten skips per interpreter were six optional real BKMP2 workflow cases, two
optional PyTorch cases, and two optional MDAnalysis reader cases. ASE was
installed, so its trajectory writer/reader and the Windows file-descriptor
regressions ran. This selection does not establish full-suite coverage on these
interpreters or validate optional engines on every Python version. The full
remote Python matrix remains a separate gate.

| Interpreter | Exact dependencies, commands and source hashes | Test outputs |
| --- | --- | --- |
| 3.13 | [Manifest](python313-regression.json) | [Initial log](python313-regression.log), [JUnit](python313-regression.xml), [socket log](python313-sockets.log), [socket JUnit](python313-sockets.xml) |
| 3.14 | [Manifest](python314-regression.json) | [Initial log](python314-regression.log), [JUnit](python314-regression.xml), [socket log](python314-sockets.log), [socket JUnit](python314-sockets.xml) |

The source manifests contain the exact pytest selection. To repeat it, install
the referenced commit with `python -m pip install '.[test,ase]'` under the desired
interpreter, set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`, and run the recorded
pytest test paths. Optional dependencies determine the documented skips.

## MPI failure reproduction and correction

Open MPI 5.0.11 and mpi4py 4.1.2 were installed in the isolated CPython 3.13
environment. A host declaration of `localhost:2` deliberately limited the
allocation to two slots, independent of the machine's physical CPU count.

| Controlled experiment | Result |
| --- | --- |
| Four ranks, legacy `OMPI_MCA_rmaps_base_oversubscribe=1` | Exit 1 before the Python program starts: insufficient slots |
| Four ranks, `--map-by :OVERSUBSCRIBE`, `tests/mpi_integration.py` | Passed ordered tasks, collective failure handling, continuous parents, restart and routing checks |
| Five ranks, `--map-by :OVERSUBSCRIBE`, `tests/mpi_dynamic_integration.py` | Passed dynamic scheduling, failure cleanup, serialization failures and static/dynamic workflow equivalence |

[The reproduction manifest](mpi-launcher-regression.json) records the exact
commands, environment versions and output hashes. Logs preserve the
[legacy failure](mpi-legacy-oversubscribe-failure.log),
[four-rank pass](mpi-explicit-oversubscribe-pass.log), and
[five-rank pass](mpi-dynamic-oversubscribe-pass.log). These are single-host
communication and process-placement checks. Actionlint 1.7.12 and
`git diff --check` also passed for the workflow change.

The correction follows the official
[Open MPI 5 MCA documentation](https://docs.open-mpi.org/en/v5.0.x/mca.html),
[mpirun placement options](https://docs.open-mpi.org/en/v5.0.10/man-openmpi/man1/mpirun.1.html),
and [mpi4py runtime installation guidance](https://mpi4py.readthedocs.io/en/stable/install.html).

## UCX transport validation

The [official OpenUCX FAQ](https://openucx.readthedocs.io/en/master/faq.html#which-transports-does-ucx-use)
documents `UCX_TLS` transport selection. `tcp`, `sm`, and `self` enable TCP,
shared memory, and communication within a process. Applying this selection to
the ecosystem job keeps its actual MPI communication while avoiding the
runner's unusable RDMA transport. The setting is confined to CI; user simulations
retain their own runtime configuration.

With the same MPICH 5.0.2, mpi4py 4.1.2, and LAMMPS 2025.7.22.4.0 versions as the
failed remote job, the explicit transport selection passed two real Lennard-Jones
energy/force/virial checks (`metal` and `real` units) and the two-rank LAMMPS
`COMM_SELF` isolation/post-close communication check. The
[validation receipt](ucx-transport-validation.json),
[force-test log](ucx-lammps-forces.log), and
[MPI log](ucx-lammps-mpi.log) record these commands and results.

The local machine does not expose the Azure `mana_0` device. The remote log is
the evidence for that hardware initialization failure; the local checks establish
that the chosen replacement transports work with the real engines. A subsequent
remote ecosystem run remains necessary to close that CI gate.

## Corrected-head CI and outstanding gates

The [14:12 UTC snapshot](github-4ba870d.json) records the corrected `4ba870d`
workflow runs and focused job/step states. The fused-native job passed 4,475
tests and 365 subtests, with 120 skips, then passed the actual four-rank MPI
integration and FES/rate workflow/resume checks using the explicit launcher
option. Its source-distribution rebuild also passed 36 tests. See
[the job](https://github.com/chen-yu-hao/PyRPMD/actions/runs/41312690979/job/124866208162)
and [the retained log excerpt](github-fused-mpi-pass.log). This job used Ubuntu
24.04; it is separate from the pending CPU matrix on `ubuntu-latest`.

The corrected-head conda package run and all five installed source/platform
wheel jobs also passed: Ubuntu CPython 3.11–3.14 and macOS CPython 3.12.

At the same observation, `4ba870d` had queued CPU matrix jobs and a pending
portable wheel workflow. The relevant runs are
[CPU tests](https://github.com/chen-yu-hao/PyRPMD/actions/runs/41312690874) and
[portable wheels](https://github.com/chen-yu-hao/PyRPMD/actions/runs/41312691014).
The preceding portable run for the Windows correction `b3e2293` had completed
its macOS wheels, while its Linux and Windows jobs remained queued. A fixed
Windows runtime pass has therefore not yet been observed. Those queued states
are pending evidence, not passes. Later snapshots, if present in this directory,
retain their observation times and tested SHAs.

At 13:59 UTC, the `4ba870d` ecosystem run also required the UCX workflow correction
described above. Local validation of that correction passed; its remote rerun
had not yet occurred.

A 14:16 UTC check also found the same UCX/MANA initialization failure in the
push-triggered ecosystem job for `4ba870d`. Its
[runtime-gate excerpt](github-ucx-push-runtime-failure.log) and
[job](https://github.com/chen-yu-hao/PyRPMD/actions/runs/41312685089/job/124866188164)
confirm that this second run likewise stopped before the scientific tests.
Both failures concern the runner's MPI initialization. The locally validated
transport correction still requires its first remote run. At that same 14:16
check, the CPU matrix and the preceding corrected Windows wheel job remained
queued.
