# Troubleshooting and diagnostics

When something fails, keep the full error, the input, the output of `pyrpmd info`, the code commit
and the run directory. Work through the layers in order: input → model → integration and sampling →
statistical analysis → restart.

## Installation and commands

| Symptom | Fix |
| --- | --- |
| `pyrpmd` not found | Activate the environment it was installed into; check `python -m pip show PyRPMD` |
| C++/OpenMP build fails | Install a C++17 compiler with OpenMP, or build in a clean environment with `PYRPMD_NO_OPENMP=1` or `PYRPMD_NO_NATIVE=1` |
| An integrator is missing from `pyrpmd info` | Reinstall after updating the source so no old `.so` is loaded |
| pytest reports skipped tests | Optional dependencies (ASE, BKMP2, an EAM model, PyMBAR, Torch, matplotlib, …) are missing; install them to run those tests |
| `native="required"` fails | The extension lacks the requested integrator; rebuild it, or use the default `native=true` in a new task |
| `loop_backend="native"` is rejected | The PES, CVs, dynamic bias or periodic/fixed atoms need the Python loop; use `auto` |

## Input and units

| Symptom | Fix |
| --- | --- |
| `Output directory is not empty` | Use a new `--output` for a new calculation; continue an existing one with `pyrpmd resume DIR` or `run INPUT --resume` |
| `... overrides; beads is not applicable` and similar | The task does not take that override; PCET accepts only `--temperature` and `--output` |
| Unknown field or invalid task combination | Check `--help` and the section table in [Input](input.md) |
| XYZ, factory or model not found | Paths inside an input are relative to the input file; CLI paths are relative to the current directory |
| Time is not a multiple of the step | Give step counts, or choose a step that divides the time |
| Atoms far apart or overlapping | Check inline bohr against XYZ angstrom, and `geometry_unit` / `cell_unit` |
| Atom index out of range | Indices start at 0; coordinates, masses, classifiers and the model must use one atom order |
| Periodic provider rejected | Use a provider that supports cell/PBC (ASE, periodic Python plugin, i-PI) |
| Fixed atoms rejected in `pimd` / `diffusion` / `site_rate` | These tasks need all atoms mobile; use surface `fes` or `surface_rate` for fixed layers |

For `--check-pes` with an external program, start the i-PI driver or prepare the calculator's model
files first.

## Potentials and dynamics

| Symptom | Fix |
| --- | --- |
| Force finite-difference check fails | Check the force sign, Å/bohr and eV/hartree, atom order and smoothness of the model |
| `NaN`/`Inf` or a non-zero provider status | Check whether the failing configuration is inside the model's domain |
| Energy drift or unstable ring oscillations | Shorten the time step and check forces, constraints and bias; velocity-Verlet is limited by the highest free-ring frequency |
| Results change with the integrator | Each integrator has its own time-step error; converge each in the time step |
| Fixed atoms move in the initial checkpoint | Check that the input system and the checkpoint belong to the same slab |

Thermostatted PIMD energies fluctuate as part of sampling. Check the stability of thermostat-free
RPMD, the PIMD energy estimators and the ring Hamiltonian separately.

## Free energies, rates and diffusion

| Symptom | Next step |
| --- | --- |
| FES `disconnected_overlap` | Add bridging windows, adjust the stiffness or sample longer |
| Many masked or empty cells | Reduce the grid resolution or sample more |
| No bootstrap interval | Read the per-draw failures and sample-size messages in the bootstrap diagnostics |
| No recrossing plateau | Add independent parents, lengthen the children and review the dividing surface |
| dRPMD reports no thermal rate | Set an explicit `bmax` and classification rules, and check unresolved/failed trajectories |
| Diffusion coefficient is `null` | Read `analysis_status`; extend production into the diffusive regime |
| PCET rate shows 0 | Read the log rate; the rate underflowed |
| Metadynamics rejects a bootstrap | Metadynamics reanalysis takes `--bootstrap 0` |

## Restart and files

`run` never overwrites a non-empty directory. Use `resume DIR` to continue and a new directory for
new scientific settings. A physics-fingerprint mismatch means the system, method or dependencies
changed; a container verification failure means the saved state is damaged.

Ordinary geometric CVs (`distance`, `angle`, `bond_difference`) use unwrapped geometry; minimum
images need the explicit `periodic_*` kinds. Runs from the earlier materials branch, where ordinary
CVs used minimum images, cannot be resumed under this convention; restart them with the matching
explicit CV.

Reanalysis overwrites derived results, so save the original summary and tables before comparing
binning or fitting schemes. On parallel file systems, checkpoint safety depends on atomic rename and
fsync; keep the complete run directory and environment record.

## Reporting an issue

Include a minimal input, a shareable potential or a model substitute, the exact command, the
traceback, the output of `pyrpmd info`, the CPU/GPU/MPI layout, the stage where the run failed and
the commit. Figures should show masked regions and diagnostics. Source layout and testing are in the
[Developer guide](development.md).
