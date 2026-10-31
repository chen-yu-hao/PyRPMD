# Electrochemical PCET: a local golden-rule model

`task="pcet"` computes proton-coupled electron transfer (PCET) rates for a local electrode–reactant
complex. It solves the one-dimensional quantum proton spectrum in two electronic states and combines
vibronic overlaps, thermal populations, environmental reorganization and the electrode Fermi
distribution into first-order reduction and oxidation rates.

PCET is a separate model workflow without ring-polymer propagation. Its rates are first-order rates
in `s^-1` for one local complex.

## Run the example scan

From the repository root:

```bash
pyrpmd validate examples/pcet/harmonic_scan.inp
pyrpmd run examples/pcet/harmonic_scan.inp --output runs/manual-pcet
pyrpmd analyze runs/manual-pcet
pyrpmd resume runs/manual-pcet
```

The [input](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/pcet/harmonic_scan.inp) defines
two harmonic wells with different positions and curvatures and scans 250/300/350 K, three electrode
potentials and the isotopes H and D: 18 conditions in about 2 s.

Read every scan point from the summary:

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

`resume` verifies the run identity and the checksum of each completed scan point, reuses them and
computes the remaining conditions. Start a new run directory when you change the model or numerical
settings. `analyze` reads the saved summary.

## The model

| Component | Model |
| --- | --- |
| Electronic states | One pair of local `oxidized` / `reduced` states |
| Quantum nucleus | One transferring H, D or T on a common 1D coordinate |
| Proton potential | Harmonic or tabulated, one per electronic state |
| Coupling | Constant electronic coupling |
| Environment | Classical Gaussian environment with one reorganization energy |
| Electrode | Wide-band metal with equilibrium Fermi occupation and constant density of states |
| Normalization | `local_first_order`, rates in `s^-1` |

Temperatures, potentials and isotopes are scanned as a Cartesian product. All isotopes share the
same pair of potential curves; PyRPMD recomputes spectra, populations and overlaps for each mass.
The KIE, $k_H/k_D$, is computed for reduction and oxidation whenever a condition includes both H and
D. The solvent, reorganization energy and acid–base equilibria are the same for every isotope.

## Input fields and units

PCET uses the top-level temperature and a `[pcet]` model; it takes no `[system]`, `[potential]`,
`beads`, `[dynamics]` or `[sampling]`.

| Field | Meaning |
| --- | --- |
| `temperature` | One temperature or a list, e.g. `["250 K", "300 K"]` |
| `pcet.method` | `"golden_rule"` |
| `pcet.normalization` | `"local_first_order"` |
| `pcet.isotopes` | e.g. `["H", "D"]` |
| `pcet.driving_force` | Free-energy difference between the well minima, reduced − oxidized, at the reference potential |
| `pcet.reorganization_energy` | Positive environmental reorganization energy, e.g. `"0.7 eV"` |
| `pcet.electronic_coupling` | Electronic coupling, e.g. `"0.001 eV"` |
| `pcet.electrode.model` | `"wide_band"` |
| `pcet.electrode.density_of_states` | Electronic density of states, e.g. `"1 eV^-1"` |
| `pcet.electrode.potential_reference` | Label of the potential scale, e.g. `"SHE"` (required) |
| `reference_potential_V` | Potential at which `driving_force` is given, in V |
| `potentials_V` | Scanned potentials, in V |
| `pcet.proton.grid_min` / `grid_max` | Common 1D proton coordinate domain |
| `pcet.proton.grid_points` | Uniform grid points including both ends |
| `pcet.proton.states` | Vibrational eigenstates kept per electronic state |

The two potential fields above belong to the `pcet.electrode` subtable. The density of states
includes every electronic state the model counts; no spin factor is added.

### Harmonic and tabulated potentials

These subtables define $V(x)=\tfrac12k(x-x_0)^2$:

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

Both potentials share one coordinate and origin, because the Franck–Condon overlaps depend on their
relative position.

For 1D energy scans of both states, use tabulated potentials:

```toml
[pcet.proton.oxidized]
kind = "tabulated"
file = "oxidized.csv"
coordinate_unit = "angstrom"
energy_unit = "eV"
```

The `reduced` subtable has the same structure. A table has at least five rows of coordinate and
energy with strictly increasing coordinates; `.csv` files are comma separated, other files whitespace
separated, and `#` starts a comment. Paths are relative to the input file. Each table covers the full
solver domain, is interpolated linearly, and has its minimum inside the grid.

## Energy zero and potential sign

`driving_force` is the environmental free-energy difference between the well minima at the
reference potential:

$$
\Delta G_{\min,\mathrm{ref}}=G_{\min,R}-G_{\min,O}.
$$

The solver subtracts each potential's minimum, so eigenvalues are measured from the bottom of their
well and include the proton zero-point energy. Enter the minimum-to-minimum difference here; a total
reaction free energy that already contains the proton vibrational free energy would count it twice.
The absolute energy offset of a tabulated potential has no effect.

With e the elementary charge, the potential dependence is

$$
\Delta G_{\min}(U)=\Delta G_{\min,\mathrm{ref}}+e(U-U_{\mathrm{ref}}).
$$

A more positive potential makes reduction less favorable. PyRPMD applies this shift itself, so give
`driving_force` at `reference_potential_V`. `potential_reference` is a label; convert potentials
between scales such as SHE, RHE or vacuum before entering them.

## From vibrational states to rates

On the common domain, the solver uses Dirichlet boundaries to solve

$$
\left[-\frac{\hbar^2}{2m}\frac{d^2}{dx^2}+V_a(x)\right]\chi_n^a(x)
=E_n^a\chi_n^a(x),\qquad a\in\{O,R\}.
$$

The overlaps $S_{\mu\nu}=\langle\chi^O_\mu|\chi^R_\nu\rangle$ give the vibronic couplings
$V_{\mu\nu}=V_{\mathrm{el}}S_{\mu\nu}$. The rate sums over all kept initial and final states, weighted
by the thermal population of the initial state and integrated over the electrode electron energies.
It scales as $\rho|V_{\mathrm{el}}|^2$.

Forward and reverse rates satisfy

$$
\frac{k_{\mathrm{red}}}{k_{\mathrm{ox}}}
=e^{-\beta\Delta G_{\min}(U)}\frac{Z_R}{Z_O},
\qquad Z_a=\sum_n e^{-\beta E_n^a},
$$

so different well curvatures enter through the proton partition functions. PyRPMD uses paired
kernels to compute very small reverse rates stably and reports the detailed-balance residual as a
consistency check.

## Output and numerical convergence

| File | Contents |
| --- | --- |
| `input/resolved.json`, `input/model.json` | Resolved input, grid and both potentials |
| `rates.csv` | Temperature, potential, scale, isotope, driving force, forward/reverse rates and log rates |
| `proton_spectra.json` | Masses, energy levels, energy origin and boundary-probability diagnostics |
| `proton_wavefunctions.npz` | Grid, potential, levels and discrete wavefunctions per isotope and state |
| `pcet_details.json` | Populations, vibronic pair contributions and numerical checks per point |
| `results/summary.json` | Scan summary, `rates` and `kinetic_isotope_effects` |
| `checkpoints/pcet-*.json` | One per completed scan point, with identity and integrity data |

Wavefunction columns are normalized as `sum(abs(psi)**2)=1`; divide by the grid spacing to get a
density per unit length.

`reduction_rate_s` and `oxidation_rate_s` are in `s^-1`; `log_reduction_rate_s` and
`log_oxidation_rate_s` are their natural logarithms. Rates too small for a float appear as 0 with a
finite log rate. With exactly zero coupling the rates are zero and the KIE is undefined.

`pcet.numerics.convergence_check=true` compares the grid M with a refined grid `2*M-1`, adds states,
and checks the thermal populations of the initial states and the boundary and high-state
contributions of the reacting channels. Non-zero rates are compared through their log rates, so tiny
rates are held to the same relative tolerance as large ones.

| Numerical field | Meaning |
| --- | --- |
| `quadrature_rtol` / `quadrature_atol` | Tolerances of the log-scaled electrode integral, default `1e-9` / `1e-11` |
| `relative_tolerance` | Relative tolerance for the grid and state comparison |
| `absolute_tolerance_s` | Tolerance for exactly zero rates |
| `numerical_convergence` | Whether the numerical checks passed |
| `physical_applicability_unverified` | Always `true`; applicability to a real system is assessed outside PyRPMD |

Converge each temperature, potential and isotope by enlarging the coordinate domain, refining the
grid, adding states and tightening the integration tolerances. Enlarge the domain separately from
refining the grid, since a box that is too small truncates the wavefunctions.

## From local rates to measured quantities

Rates are normalized to one local complex. Currents, heterogeneous rate constants and
area-normalized quantities follow from them with the complex coverage, interface area, reaction
network and mass transport.

The equations and method sources are in the
[PCET topic](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/pcet.md) *(Chinese)*. On the
harmonic example model, PyRPMD matches the Hammes-Schiffer group's pyPCET to within 0.062% for all 12
forward/reverse rates; see the
[model validation record](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/pcet-model-2026-10-30/README.md).
