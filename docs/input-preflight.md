# Input preflight and first runs

`pyrpmd validate input.inp` separates two outcomes: `valid` describes the input
schema and physical settings; `ready` also requires the selected runtime
resources. `resources.available=false` lists each missing MPI, PLUMED or Torch
requirement and a setup hint. `prepare` and `validate --check-pes` use the same
resource probe before numerical preparation.

The probe does not launch MPI ranks, construct a PES, allocate a Torch tensor or
read a PLUMED input into its kernel. It checks MPI library loading without
initializing MPI, the active launcher rank count, PLUMED API compatibility, and
the requested CPU/CUDA devices. A serial input does not import these optional
backends. Availability does not establish sampling convergence.

Bare numeric dimensional quantities keep their existing meaning: atomic units,
except temperature in kelvin. For example, `dynamics.timestep = 0.5` means about
0.012094 fs. Input warnings name the field and show its interpreted unit and a
conversion; use `timestep = "0.5 fs"` to request half a femtosecond. Exact step
counts remain integers without unit warnings. Explicit coordinate/CV unit
selectors and fields ending in `_au` retain their declared conventions. Opaque
provider parameters retain the provider's own unit contract. A numeric
`umbrella.force_constant` uses the documented k/T convention and is multiplied
by temperature; it is not a physical stiffness by itself.

Duration errors identify the duration field, whether its value was a default,
and the time step used for divisibility. Electron-count/multiplicity mismatches
produce an input warning; analytic teaching models remain usable. PES metadata
mismatch errors show both declared values.

`site_rate` now publishes `rate_s_inv`, `rate_estimate_s_inv`,
`rate_standard_error_s_inv` and `k_qtst_s_inv`, with matching interval and
conditional-error aliases. The original `_s` spellings remain for this release,
including their `null` values for withheld rates.

Create a portable first run with an installed package:

```sh
pyrpmd init first-run --task fes
pyrpmd init first-pimd --task pimd
pyrpmd init first-rate --task rate
pyrpmd init first-drpmd --task drpmd
```

Each command writes `quickstart.inp` and its required local assets into a new or
empty directory and prints validation, run, analyze and resume commands. The
rate starter uses the analytic exchange teaching PES; the dRPMD starter uses
free scattering. These short examples demonstrate the workflows. Reference
and convergence tutorials are documented separately in `examples/tutorials`.
