# q-TIP4P/F model and reproducible water pilot

This directory adds a real flexible-water potential and a small periodic
integration test. **It does not report a converged liquid diffusion coefficient
or a reproduced liquid IR spectrum.**

`qtip4pf.py` independently implements the equations and nine parameters of
Habershon, Markland and Manolopoulos, *J. Chem. Phys.* **131**, 024501 (2009),
[doi:10.1063/1.3167790](https://doi.org/10.1063/1.3167790),
[author manuscript](https://arxiv.org/abs/1011.1047), Eqs. (1)–(4), Table I.
It creates an OpenMM `System` with quartic O–H stretches, harmonic bends,
oxygen Lennard-Jones interactions, virtual M charges and optional PME. The
separate NumPy cluster-energy implementation supplies an independent
finite-difference oracle for the OpenMM forces, including M-site force transfer.

The model is **flexible**. Constraining its O–H bonds creates a different model
and removes the stretching vibration. PyRPMD coordinates contain physical
O,H,H nuclei per molecule; the provider constructs the massless M sites. Keep
each molecule whole when supplying coordinates or evaluating its neutral dipole.

## Run and replay the software pilot

From the source checkout, with the OpenMM extra installed:

```bash
python -m pip install -e '.[openmm,test]'
python -m pytest -q tests/test_qtip4pf_benchmark.py
python benchmarks/water/pilot.py --output runs/water-pilot
python benchmarks/water/replay.py runs/water-pilot
```

The pilot starts eight randomly oriented molecules on a reproducible grid at
298 K and 0.997 g/cm³. It runs 64 PILE-L sampling steps and 256 unthermostatted
RPMD steps with four beads and a 0.05 fs timestep. Its small box requires a
2.5 Å nonbonded cutoff. The starting grid is not equilibrated liquid; these
parameters deliberately keep the software test inexpensive.

The output contains the serialized OpenMM model, molecular centroid positions
and velocities, bead dipoles, energy history, final physical state, a diffusion
diagnostic, a relative spectrum and SHA256 evidence. `software_checks_passed`
tests the independent model oracle, finite output and a predefined energy-drift
bound. `physical_converged` and `literature_reproduced` remain false. Replay
checks artifact hashes and recomputes the estimators from the saved arrays.

## Published targets and the method distinction

[`references.json`](references.json) transcribes three table entries with their
units and source locations. The 216-molecule finite-box diffusion targets at
298 K are:

| Model/isotope and dynamics | D (Å²/ps) | Reported standard error | Source |
| --- | ---: | ---: | --- |
| q-TIP4P/F H₂O, RPMD | 0.221 | 0.001 | Table III |
| q-TIP4P/F D₂O, RPMD | 0.172 | 0.001 | Table III |
| q-TIP4P/F H₂O, classical MD | 0.199 | 0.002 | Table IV |

These are **literature numbers**, not new PyRPMD outputs. The source uses 32
beads, a 9 Å cutoff and Ewald electrostatics; H₂O/D₂O densities differ. Its
IR spectra use **partially adiabatic centroid molecular dynamics (PA-CMD)**
(Section III.C and Figure 5). An RPMD/TRPMD spectrum can therefore be a useful
cross-method comparison, but cannot pass as a PA-CMD implementation reproduction.
No source IR curve has been digitized here.

## Construct a source-sized model

The factory is reusable with the new OpenMM provider:

```python
from benchmarks.water.qtip4pf import make_system, initial_grid
from pyrpmd.constants import ANGSTROM_TO_AU
from pyrpmd.potentials.openmm import OpenMMPotential
import numpy as np

q, side = initial_grid(cells_per_axis=6)  # 216 molecules; still unrelaxed
model = make_system(molecules=216, box_angstrom=side, cutoff_angstrom=9.0)
potential = OpenMMPotential(
    model, ["O", "H", "H"] * 216,
    cell=np.eye(3) * side * ANGSTROM_TO_AU, pbc=[True] * 3,
    platform="Reference",
)
```

Equilibrate and inspect structure before starting dynamical production. For
D₂O set `hydrogen_mass_da=D_MASS_DA` consistently in the grid, OpenMM model and
PyRPMD masses, and set the D₂O density. OpenMM model masses do not replace the
masses used by PyRPMD's propagator.

Before a quantitative comparison, match isotope masses, cutoff/tail treatment,
Ewald/PME accuracy, finite box, estimator and boundary conditions. This
implementation defaults to no dispersion correction or switching. PME accuracy,
bead count, integration step and sampling length need explicit tests. The
factory evaluates the full potential on every bead; it does not reproduce the
paper's ring-polymer contraction scheduling.

Use independent production replicas, archived VACF integrals and unwrapped
molecular-centroid MSD, with a stable fit/tail. A finite 216-molecule reference
must not be confused with an infinite-size extrapolation. For IR also test
record length, sampling cadence, window and thermostat dependence. The current
relative-IR estimator has no absolute absorbance prefactor.

The [campaign and evidence protocol](../literature/README.md) records these
requirements; [`docs/literature-validation.md`](../../docs/literature-validation.md)
tracks all requested benchmark families.
