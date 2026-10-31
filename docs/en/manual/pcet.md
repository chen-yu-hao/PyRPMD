# Electrochemical PCET: a local golden-rule model

`task="pcet"` computes proton-coupled electron transfer (PCET) for an already formed local
electrode–reactant complex. It solves the one-dimensional quantum proton spectrum in two electronic
states and combines vibronic overlaps, thermal populations, environmental reorganization and the
electrode Fermi distribution into first-order reduction and oxidation rates.

This is a separate weak-coupling model workflow. It has no ring-polymer beads or real-time
propagation, does not use the gas-phase RPMD asymptotic prefactor, and does not interpret its
results directly as a current density. Understanding this normalization is the starting point for
choosing inputs and comparing with experiment.

## Run the teaching scan

From the repository root:

```bash
pyrpmd validate examples/pcet/harmonic_scan.inp
pyrpmd run examples/pcet/harmonic_scan.inp --output runs/manual-pcet
pyrpmd analyze runs/manual-pcet
pyrpmd resume runs/manual-pcet
```

The [complete input](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/pcet/harmonic_scan.inp)
uses two harmonic wells with different positions and curvatures and scans 250/300/350 K, three
electrode potentials and the two isotopes H/D, 18 conditions in total. It corresponds to no real
molecule or electrode material; it is meant for learning the model definition, the output and the
numerical checks.

After the run you can read every scan point directly:

```python
import json
from pathlib import Path

summary = json.loads(Path("runs/manual-pcet/results/summary.json").read_text())
print(summary["completed_scan_points"])
print(summary["numerical_convergence"])

for row in summary["rates"]:
    print(row["temperature_K"], row["potential_V"], row["isotope"],
          row["reduction_rate_s"], row["oxidation_rate_s"])

for entry in summary["kinetic_isotope_effects"]:
    print(entry["temperature_K"], entry["potential_V"],
          entry["reduction_KIE"], entry["oxidation_KIE"])
```

`resume` verifies the run identity and the checksum of every completed scan point, then reuses those
results and continues with the remaining conditions. It does not extend any dynamics trajectory.
Create a new run directory when you change physical parameters, potential curves or numerical
settings. `analyze` reads the saved summary; PCET does not support the FES `--recompute` unbiasing
analysis.

## What the model contains

| Component | Current model |
| --- | --- |
| Electronic states | One pair of local `oxidized` / `reduced` states |
| Quantum nucleus | One transferring nucleus, H, D or T on a common 1D coordinate |
| Proton potential | Harmonic or tabulated potential in each electronic state |
| Coupling | Constant electronic coupling independent of the proton coordinate and electron energy |
| Environment | A single positive reorganization energy, classical Gaussian environment |
| Electrode | Ideal wide-band metal reservoir with equilibrium Fermi occupation and constant density of states |
| Normalization | `local_first_order`, rates in `s^-1` |

Temperatures, potentials and isotopes are scanned as a Cartesian product. H/D/T use the same pair of
potential curves; only the mass of the transferring nucleus changes before the spectra, populations
and overlaps are recomputed. No mass-ratio correction of the rate is assumed. The automatic KIE
requires both H and D at the same condition and is defined as $k_H/k_D$, separately for reduction
and oxidation.

This isotope substitution does not change the solvent, the reorganization energy or acid–base
equilibria. It therefore does not include every experimental isotope effect that an H₂O/D₂O
exchange may cause.

## Model input: units and fields

PCET needs no `[system]`, `[potential]`, `beads`, `[dynamics]` or `[sampling]` and calls no
molecular PES plugin. It uses the top-level temperature and a separate `[pcet]` model:

| Field | Meaning |
| --- | --- |
| `temperature` | One temperature or a list, e.g. `["250 K", "300 K"]` |
| `pcet.method` | `"golden_rule"` |
| `pcet.normalization` | `"local_first_order"` |
| `pcet.isotopes` | e.g. `["H", "D"]` |
| `pcet.driving_force` | Well-minimum free-energy difference reduced−oxidized at the reference potential |
| `pcet.reorganization_energy` | Positive environmental reorganization energy, e.g. `"0.7 eV"` |
| `pcet.electronic_coupling` | Constant electronic coupling, e.g. `"0.001 eV"` |
| `pcet.electrode.model` | `"wide_band"` |
| `pcet.electrode.density_of_states` | Electronic density of states, e.g. `"1 eV^-1"` |
| `pcet.electrode.potential_reference` | Required label of the potential scale, e.g. `"SHE"` |
| `reference_potential_V` | Reference potential of the driving force, numbers in V |
| `potentials_V` | List of scanned potentials, numbers in V |
| `pcet.proton.grid_min` / `grid_max` | Common 1D proton coordinate domain of both electronic states |
| `pcet.proton.grid_points` | Number of uniform grid points including both ends |
| `pcet.proton.states` | Number of vibrational eigenstates kept per state |

The last two potential fields in the table belong to the `pcet.electrode` subtable. The density of
states must include all electronic states the model actually counts; the program adds no implicit
spin-degeneracy factor.

### Harmonic and tabulated potentials

The following subtables define $V(x)=\tfrac12k(x-x_0)^2$:

```toml
[pcet.proton.oxidized]
kind = "harmonic"
center = "-0.2 angstrom"
force_constant = "4 eV/angstrom^2"

[pcet.proton.reduced]
kind = "harmonic"
center = "0.2 angstrom"
force_constant = "5 eV/angstrom^2"
```

Both potentials must use the same physical coordinate and origin. Two potentials whose coordinates
were shifted independently cannot be used for Franck–Condon overlaps until their relative position
is restored.

If you have 1D energy scans of both states, use tabulated potentials:

```toml
[pcet.proton.oxidized]
kind = "tabulated"
file = "oxidized.csv"
coordinate_unit = "angstrom"
energy_unit = "eV"
```

The other electronic state uses a `reduced` subtable with the same structure. A file needs at least
five rows and two columns (coordinate and energy) with strictly increasing coordinates; CSV files are
comma separated, other suffixes whitespace separated, and `#` starts a comment. Paths are relative
to the input file. Both tables must cover the whole solver domain; the program interpolates linearly
and never extrapolates, and a well minimum must not lie on the grid boundary.

## Well-minimum zero and potential sign

`driving_force` is the environmental free-energy difference between the well minima at the reference
potential:

$$
\Delta G_{\min,\mathrm{ref}}=G_{\min,R}-G_{\min,O}.
$$

It excludes the zero-point and thermal vibrational free energy of the transferring proton. The solver
removes the minimum value from each proton potential, so the eigenvalues are measured from the
respective minimum and include the quantum zero-point energy. If an external calculation gives a
total reaction free energy that already contains this proton vibrational contribution, do not enter
it as `driving_force`; it would be counted twice.

With e the positive elementary charge, the potential correction is

$$
\Delta G_{\min}(U)=\Delta G_{\min,\mathrm{ref}}+e(U-U_{\mathrm{ref}}).
$$

A more positive potential raises the reduction free-energy change of this model and makes reduction
less favorable. Do not add the same potential correction to the driving force again. An arbitrary
global energy constant of a tabulated potential cannot replace the difference between the two state
minima either.

`potential_reference` only records the name of the scale; it does not convert SHE/RHE, pH or vacuum
levels. If you change the scale, also perform the corresponding physical conversion of the numbers.
The input currently has no automatic pH, proton-donor activity or acid–base speciation model.

## From vibrational states to rates

On the common coordinate domain, the solver uses Dirichlet boundaries to solve

$$
\left[-\frac{\hbar^2}{2m}\frac{d^2}{dx^2}+V_a(x)\right]\chi_n^a(x)
=E_n^a\chi_n^a(x),\qquad a\in\{O,R\}.
$$

The overlap of the two states $S_{\mu\nu}=\langle\chi^O_\mu|\chi^R_\nu\rangle$ gives the vibronic
coupling $V_{\mu\nu}=V_{\mathrm{el}}S_{\mu\nu}$. The calculation sums over all kept initial/final
state pairs, weights them by the thermal population of the initial state and integrates with the
electrode electron-energy kernel. The rate is proportional to $\rho|V_{\mathrm{el}}|^2$, a checkable
scaling of the weak-coupling model.

The same model satisfies

$$
\frac{k_{\mathrm{red}}}{k_{\mathrm{ox}}}
=e^{-\beta\Delta G_{\min}(U)}\frac{Z_R}{Z_O},
\qquad Z_a=\sum_n e^{-\beta E_n^a}.
$$

Different well curvatures change the ratio of proton partition functions, so forward and reverse
rates cannot be compared through $e^{-\beta\Delta G_{\min}}$ alone. The implementation uses the
detailed-balance relation of paired kernels to compute very small reverse rates stably; its
detailed-balance residual is an internal consistency check, not an independent validation of the
physical model.

## Output files and numerical convergence

| File | Contents |
| --- | --- |
| `input/resolved.json`, `input/model.json` | Canonical input, grid and both potentials; no fictitious molecular System is created |
| `rates.csv` | Temperature, potential, scale, isotope, driving force, forward/reverse rates and log rates |
| `proton_spectra.json` | Masses, energy levels, energy origin and boundary-probability diagnostics |
| `proton_wavefunctions.npz` | Grid, potential, levels and discrete wavefunctions per isotope/electronic state |
| `pcet_details.json` | Populations per point, vibronic pair contributions and detailed numerical checks |
| `results/summary.json` | Scan summary, `rates` and `kinetic_isotope_effects` |
| `checkpoints/pcet-*.json` | Each completed scan point with identity and integrity information |

Wavefunction columns are discretely normalized with `sum(abs(psi)**2)=1`. To convert to a continuous
wavefunction density, account for the grid spacing; the squared file array is not a density per unit
length.

`reduction_rate_s` / `oxidation_rate_s` are in `s^-1`. `log_reduction_rate_s` /
`log_oxidation_rate_s` are the natural logarithms of those numbers. When an ordinary float underflows
to zero, read the finite log value; that differs from an exact zero rate caused by zero coupling. With
exactly zero coupling the H/D ratio is undefined.

`pcet.numerics.convergence_check=true` compares the original grid M with the refined grid `2*M-1`,
increases the number of kept states, and checks the thermal populations of initial states and the
boundary/high-state contributions weighted by the actual reaction channels. Non-zero rates are
compared by the difference of log rates, so a large absolute tolerance cannot hide a huge relative
error of a tiny rate.

| Numerical field | Meaning |
| --- | --- |
| `quadrature_rtol` / `quadrature_atol` | Tolerances of the log-scaled adaptive electrode integral, default `1e-9` / `1e-11` |
| `relative_tolerance` | Relative-error threshold for the grid and state-count comparison |
| `absolute_tolerance_s` | Absolute tolerance for exact zero rates; does not replace the relative check of non-zero rates |
| `numerical_convergence` | Whether the executed numerical checks passed |
| `physical_applicability_unverified` | Stays true; passing numerics does not validate the physical applicability of the model |

Independently enlarge the coordinate domain, refine the grid, add states, tighten the integration
tolerances and check each temperature, potential and isotope. Refining the grid inside a finite box
does not remove the error of a wavefunction truncated by the box, and a low population of
high-energy initial states does not prove that high-energy final states do not contribute.

## Comparing with real electrochemistry

The output is normalized to one local complex. Currents, heterogeneous rates or area-normalized
quantities additionally need the complex concentration/coverage, interface area, reaction network
and mass transport. The current model has no explicit electrode/solvent sampling, constant-potential
electron-number sampling, double layer, donor–acceptor distance distribution or strongly coupled
solvent-controlled kinetics.

The full equations and method sources are in the
[PCET topic](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/pcet.md) *(Chinese)*. An
independent comparison of the fixed teaching model with the authors' public pyPCET is recorded in
the [model validation record](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/pcet-model-2026-10-30/README.md);
it validates the implementation of that model and does not reproduce a real electrode experiment.
