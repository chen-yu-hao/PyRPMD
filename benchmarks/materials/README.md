# Fixed-cell materials validation

## Real Ni-H EAM integration pilot

`ni_h_eam.py` exercises the new periodic ASE/PIMD/RPMD transport path with an
actual tabulated Ni-H potential. It builds a 3×3×3 Ni(100) slab (27 Ni atoms),
places one H at a hollow site, and runs H/D with one and four beads at 600 K.
All atoms move; the centre of mass of the Ni atoms is the diffusion reference.
Production trajectories are unthermostatted. The default records contain
64 sampling steps and 160 production steps, with a 0.1 fs time step.

This is a short **implementation pilot**, not a physically converged surface
diffusion calculation. It does not reproduce the PES/lattice model of
Suleimanov's 2012 H/Ni(100) RPMD study, does not establish a quantum enhancement
factor, and does not provide converged H/D diffusion coefficients. Slab size,
coverage, layer motion, equilibration, bead number, time step and independent
sampling all require further study. `physical_converged` remains false.

Install the source checkout with its ASE extra, then run:

```bash
python -m pip install -e '.[ase]'
python benchmarks/materials/ni_h_eam.py \
  --model /absolute/path/NiAlH_jea.eam.alloy \
  --download \
  --output /absolute/path/new-ni-h-pilot
```

`--download` is explicit and only used when the requested file is absent.
Existing or downloaded bytes must match both the fixed length and SHA256.
Nothing is fetched by the normal test suite. `--quick` reduces the run to four
equilibration and sixteen production steps per case for an integration smoke
check; the resulting records are deliberately insufficient for diffusion fits.

The runner verifies energy/force agreement with an independent direct ASE
calculator, periodic translation invariance, selected finite-difference forces,
finite PIMD/NVE output and energy drift. Its machine-readable `report.json`
records dependencies, potential provenance, source file hashes, finite-step
results and limitations. Each case has the standard PyRPMD input, checkpoint,
unwrapped-centroid trajectory and analysis files.

### Archived integration result

[`ni_h_eam_validation.json`](ni_h_eam_validation.json) records a full four-case
pilot run on 2026-10-30. All source files in its calculation fingerprint stayed
unchanged during the run. Each case saved 81 frames over 16 fs of production;
all finite-state, ring-spread and finite-step energy checks passed.

| Isotope | Beads | Maximum NVE energy drift per bead (hartree) | Mean ring radius squared (bohr²) |
| --- | ---: | ---: | ---: |
| H | 1 | 1.6704e-7 | 0 |
| D | 1 | 1.1438e-7 | 0 |
| H | 4 | 1.8282e-7 | 0.039323 |
| D | 4 | 1.6073e-7 | 0.019857 |

Direct ASE/adapter energy and force errors were below 7e-16 in their respective
atomic units. The largest selected finite-difference force error was
4.44e-10 eV/angstrom. All four diffusion analyses returned
`ballistic_or_drifting`, with accepted diffusion coefficients and sampling
errors unavailable. The short, incompletely equilibrated ring-size observations
are not an equilibrium isotope-effect measurement.

Only the compact report is archived in the repository. Its case paths are
relative to the generated output directory; rerunning the command above
produces the corresponding trajectories, input records and checkpoints.

### Potential provenance

The table is **not redistributed in PyRPMD**. The opt-in downloader pins:

- Official source: [LAMMPS NiAlH_jea.eam.alloy](https://github.com/lammps/lammps/blob/5f0288a471841037872e673f93abe6af41490a6a/potentials/NiAlH_jea.eam.alloy).
- Commit: `5f0288a471841037872e673f93abe6af41490a6a`.
- SHA256: `545b0f79207ae519b7f355451812acd999bd6ed095f40a6bcfa9aa5258a38388`.
- Length: 291035 bytes; DYNAMO setfl format, LAMMPS metal units (eV, angstrom).
- Model publication: J. E. Angelo, N. R. Moody and M. I. Baskes,
  “Trapping of hydrogen to lattice defects in nickel,” *Modelling Simul.
  Mater. Sci. Eng.* **3**, 289–307 (1995),
  [doi:10.1088/0965-0393/3/3/001](https://doi.org/10.1088/0965-0393/3/3/001).
- [NIST registry and model history](https://www.ctcms.nist.gov/potentials/entry/1995--Angelo-J-E-Moody-N-R-Baskes-M-I--Ni-Al-H/).
- [LAMMPS EAM format/units](https://docs.lammps.org/pair_eam.html).

For opt-in automated coverage with a previously obtained matching model:

```bash
PYRPMD_NIALH_POTENTIAL=/absolute/path/NiAlH_jea.eam.alloy \
  python -m pytest -q tests/test_materials_eam.py
```

The opt-in test checks the real adapter and one minimal periodic workflow.
The four-case runner is the reproducible H/D and bead-count integration pilot.

## Finite-bead harmonic equilibrium reference

[`harmonic_validation.py`](harmonic_validation.py) and its
[`recorded result`](harmonic_validation.json) provide a separate analytic
sampling check. Four independent P=16 streams at beta=4 give a centroid-virial
internal energy of 1.549981 ± 0.005625 hartree, where the error is the standard
error of the four replica means. The exact finite-P value is 1.544545 hartree;
the difference is 0.97 estimated standard errors. The continuum quantum value,
1.555972 hartree, is a different reference and includes the bead-discretization
limit. This synthetic equilibrium check does not validate material transport,
time-correlation functions, or the accuracy of the Ni-H potential.
