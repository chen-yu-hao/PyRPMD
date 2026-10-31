# Troubleshooting and diagnostics

When something fails, first keep the complete error, the input, `pyrpmd info`, the code commit and the
run directory. Locate the problem by layer: input → model → integration/sampling → statistical
analysis → restart.

## Installation and commands

| Symptom | Check and fix |
| --- | --- |
| `pyrpmd` not found | Activate the virtual environment it was installed into; check `python -m pip show PyRPMD` |
| C++/OpenMP build fails | Check for a C++17 compiler and OpenMP; if needed build in a clean environment with `PYRPMD_NO_OPENMP=1` or `PYRPMD_NO_NATIVE=1` |
| Installed, but an integrator is unavailable | Look at `pyrpmd info`; after updating the source, rebuild and rule out an old `.so` |
| pytest reports skipped tests | Tests are skipped when optional dependencies (ASE, BKMP2, an EAM model, PyMBAR, Torch, matplotlib, …) are missing; a skip does not mean that path is verified |
| `native="required"` fails | The extension does not provide the required integrator; if the compatible path is enough, recreate the task with the default setting |
| `loop_backend="native"` is rejected | Check whether the PES, CVs, dynamic bias, periodic/fixed atoms, … support the fused path; usually use `auto` |

## Input and units

| Symptom | Check and fix |
| --- | --- |
| `Output directory is not empty` | Use a new `--output` for a new calculation; continue the same one with `pyrpmd resume DIR` or `run INPUT --resume` |
| Override errors such as `... overrides; beads is not applicable` | The task does not accept that command-line override; PCET accepts only `--temperature` and `--output` |
| Unknown field or invalid task combination | Read the current `--help` and the section overview in [Input](input.md); do not delete the error while keeping the old physical assumption |
| XYZ, factory or model not found | Paths inside an input are relative to the input file; CLI overrides are relative to the start directory |
| Time is not a multiple of the step | Use explicit integer step counts, or choose a step that divides the target time |
| Initial structure is huge or atoms are too close | Check inline bohr vs. XYZ angstrom and the explicit `geometry_unit` / `cell_unit` |
| Atom index out of range | All indices start at 0; coordinates, masses, classifiers and model element order must agree |
| Periodic provider rejected | Use an interface that supports cell/PBC; adding a cell does not make a molecular potential periodic |
| Fixed atoms rejected in a materials task | `pimd` / `diffusion` / `site_rate` currently require all atoms to be mobile; use the applicable surface/FES workflow for fixed layers |

Resolve input errors before creating a formal output. For `--check-pes` with an external program,
start the required driver or prepare the calculator's model files first; plain `validate` does not
guarantee that the external program is available.

## Potentials and dynamics

| Symptom | Check and fix |
| --- | --- |
| Force finite-difference check fails | Check the negative-gradient sign, Å/bohr and eV/hartree, atom order and model smoothness first |
| `NaN/Inf` or a non-zero provider status | Check whether the failing configuration leaves the model domain; never continue with zero energy/forces |
| Energy drift or unstable ring oscillations | Shorten the time step and check forces/constraints/bias; velocity-Verlet is also limited by the highest free-ring frequency |
| Different results with another integrator | The integrator is a scientific setting; do a separate time-step convergence; it is not just a performance switch |
| Fixed atoms move away in the initial checkpoint | Check the input system and the saved state; do not mix checkpoints of different slab identities |

Fluctuating thermostatted PIMD energies are part of sampling. Judge the numerical stability of
thermostat-free RPMD production, the physical PIMD energy estimators and the ring Hamiltonian
separately.

## Free energies, rates and diffusion

| Symptom | Interpretation and next step |
| --- | --- |
| FES `disconnected_overlap` | The windows cannot define one global normalization; add bridging windows, adjust stiffness or sample longer |
| Many masks or empty cells | The saved samples do not support these regions; reduce unnecessary grid resolution or sample more, keeping the mask |
| Missing bootstrap interval | Read the per-draw failures and insufficient-sample messages; never substitute a zero error |
| Recrossing coefficient has no plateau | Add independent parents and longer children and review the dividing surface; a finite-time value is not automatically a plateau |
| dRPMD reports no formal thermal rate | Check bmax, classification rules, unresolved/failed trajectories and the input diagnostics |
| Diffusion coefficient is `null` | Possibly still ballistic or no credible linear fit window; read the diagnostics first and extend the physical production time |
| An ordinary PCET rate shows 0 | Read the log rate; it may be numerical underflow; relative numerical convergence checks are still needed |
| Metadynamics refuses a bootstrap reanalysis | The bias-estimate interface does not support a non-zero bootstrap; do not derive rates from biased time |

Do not just relax thresholds to make a diagnostic pass. First understand whether the corresponding
sampling assumption holds and the data are sufficient; see [Scientific conventions](science.md).

## Restart and files

A new `run` never overwrites an existing non-empty directory. Use `resume DIR` to continue and a new
directory for new scientific conditions. A physics-fingerprint mismatch means the system, method or
dependencies changed; a container verification failure means the state cannot be trusted. Neither can
be bypassed by editing hashes.

An old materials branch implicitly used minimum images for ordinary geometric CVs; the integrated
release requires explicit `periodic_*` kinds. Old periodic-CV runs with this semantic ambiguity cannot
be continued directly under the new convention. Choose the explicit CV that matches the original
physical definition and start a new run; old molecular main-line runs and already explicit periodic
coordinates are not renamed by this change.

Reanalysis updates derived results. Before comparing binning or fitting schemes, save the original
summary and result tables. On parallel file systems, checkpoint guarantees still depend on the real
semantics of atomic rename/fsync; keep the complete directory and environment records.

## Filing a reproducible issue

Provide a minimal input, a shareable potential or a teaching substitute, the exact command, the error
traceback, Python/PyRPMD/native capabilities, CPU/GPU/MPI topology, the stage of the first failure and
a reproducible commit. Do not submit only a final figure while omitting masked regions or diagnostics.
The source entry points and testing methods are in the [Developer guide](development.md).
