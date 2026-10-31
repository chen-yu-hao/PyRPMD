# Electrochemical PCET: displaced-harmonic model

`harmonic_scan.inp` computes reduction and oxidation rate constants in **s⁻¹** for one local
electrode–reactant complex with one transferring proton, two electronic states and weak vibronic
coupling. In about two seconds it scans three temperatures and three electrode potentials for
both H and D, and reports H/D kinetic isotope effects and per-vibronic-pair contributions.

```bash
pyrpmd validate examples/pcet/harmonic_scan.inp
pyrpmd run examples/pcet/harmonic_scan.inp --output runs/pcet-harmonic
pyrpmd analyze runs/pcet-harmonic
```

The electrode is an ideal wide-band electron reservoir and the environment has a classical
Gaussian reorganization energy. On this harmonic model at 300 K, PyRPMD matches a pinned pyPCET
commit to within 0.062% across 12 forward and reverse rates
([validation record](../../docs/validation/pcet-model-2026-10-30/README.md)).

> **Scope.** The displaced harmonic wells are artificial parameters chosen so that isotopic
> substitution visibly changes the proton spectrum, overlaps and partition functions. They do not
> describe H + H₂, a specific molecular PCET reaction or a real electrode material.

## Conventions

- **Driving force.** `driving_force = "0.0 eV"` is the reduced-minus-oxidized minimum free-energy
  difference at `reference_potential_V`, excluding the transferring proton's zero-point and
  thermal vibrational contributions. Each proton potential is referenced to its own minimum, so
  do not add the zero-point energy to `driving_force` again.
- **Potential.** A more positive voltage raises the reduction driving free energy by
  `e * (U - U_ref)`.
- **Reference electrode.** `potential_reference = "SHE"` is a recorded label; the example does not
  calibrate to SHE, convert to RHE or add pH dependence.
- **Directions.** Both directions share the electronic coupling and density of states, with Fermi
  occupation for reduction and empty electrode states for oxidation.

## Output

| File | Contents |
| --- | --- |
| `rates.csv` | Rate table for every condition |
| `proton_spectra.json` | Proton vibrational spectra |
| `pcet_details.json` | Individual vibrational-pair contributions |
| `results/summary.json` | Summary and numerical diagnostics |

For the supplied parameters, the 300 K slice gives approximately:

| U (V, declared SHE label) | H reduction (s⁻¹) | H oxidation (s⁻¹) | Reduction H/D KIE | Oxidation H/D KIE |
| --- | ---: | ---: | ---: | ---: |
| −0.2 | 4.534 × 10⁶ | 2.665 × 10³ | 1.539 | 1.665 |
| 0.0 | 1.435 × 10⁵ | 1.932 × 10⁵ | 1.550 | 1.677 |
| +0.2 | 1.976 × 10³ | 6.091 × 10⁶ | 1.460 | 1.580 |

Because the two electronic states have different force constants, their proton free energies
differ, so reduction and oxidation rates differ at the reference voltage even with zero
`driving_force`. Raising the voltage suppresses reduction and favors oxidation, as the sign
convention requires. These are deterministic model reference values.

## Numerical checks

Across the 18-point scan, the largest grid-refinement change in a rate is about 0.035%, and going
from 12 to 24 retained states changes rates by less than 2 × 10⁻¹¹ relative. All configured
numerical checks pass. The automatic check holds the physical coordinate interval fixed; study the
grid extent separately when you adapt the model.

These checks validate the specified model and its discretization. The summary keeps
`physical_applicability_unverified = true` because weak coupling, Gaussian solvent response and
the parameters themselves are your modeling choices to justify for a chemical system. Before
relying on a new model, repeat with a wider and finer proton grid, more retained states and
tighter adaptive-quadrature tolerances.

The workflow computes golden-rule rates; it does not propagate RPMD trajectories, sample an
explicit constant-potential interface, average over a donor–acceptor distance distribution or
predict current density. The [PCET method, input and scope](../../docs/pcet.md) gives the
equations and output interpretation.
