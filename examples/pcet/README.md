# Electrochemical PCET: displaced-harmonic teaching model

`harmonic_scan.inp` runs a one-proton, two-state, weak-coupling PCET model for
one local electrode–reactant complex. It scans three temperatures and three
electrode potentials for both H and D, and reports reduction and oxidation
rate constants in **s⁻¹**. The electrode is an ideal wide-band electron
reservoir; the environment has a classical Gaussian reorganization energy.

```bash
pyrpmd validate examples/pcet/harmonic_scan.inp
pyrpmd run examples/pcet/harmonic_scan.inp --output runs/pcet-harmonic
pyrpmd analyze runs/pcet-harmonic
```

The displaced harmonic wells have different force constants. They provide a
small deterministic example in which isotopic substitution changes the
proton vibrational spectrum, overlaps and the relative proton partition
functions. **These parameters are artificial. They do not describe H + H₂,
any particular molecular PCET reaction, or a real electrode material.**

`driving_force = "0.0 eV"` is the reduced-minus-oxidized minimum free-energy
difference at `reference_potential_V`. It excludes the transferring proton's
zero-point and thermal vibrational contributions. Each proton potential is
referenced to its own minimum; adding its zero-point energy again to
`driving_force` would count it twice. A more positive voltage raises the
reduction driving free energy by `e * (U - U_ref)` under this convention.

`potential_reference = "SHE"` records a reference label. The example does not
compute a real SHE calibration, convert to RHE, or infer pH dependence.
The two directions use the same electronic coupling and density of states,
with Fermi occupation for reduction and empty electrode states for oxidation.

Inspect `rates.csv`, `proton_spectra.json`, `pcet_details.json` and
`results/summary.json` for the rate table, proton spectra, individual vibrational-pair
contributions and numerical diagnostics. Repeat with a wider/finer proton
grid, more retained states and tighter adaptive-quadrature tolerances before relying on a
model result. Passing numerical checks validates this specified model and
the checked discretizations; it does not validate weak coupling, Gaussian
solvent response or the parameters for a chemical system.

This workflow does not propagate RPMD trajectories, sample an explicit
constant-potential interface, average over a donor–acceptor distance
distribution, or predict current density. See the
[PCET method, input and limitations](../../docs/pcet.md) for the equations,
output interpretation and supported scope.

## Illustrative output

For the supplied artificial parameters, the 300 K slice gives approximately:

| U (V, declared SHE label) | H reduction (s⁻¹) | H oxidation (s⁻¹) | Reduction H/D KIE | Oxidation H/D KIE |
| --- | ---: | ---: | ---: | ---: |
| −0.2 | 4.534 × 10⁶ | 2.665 × 10³ | 1.539 | 1.665 |
| 0.0 | 1.435 × 10⁵ | 1.932 × 10⁵ | 1.550 | 1.677 |
| +0.2 | 1.976 × 10³ | 6.091 × 10⁶ | 1.460 | 1.580 |

The asymmetric force constants produce different proton free energies in
the two electronic states, so reduction and oxidation need not have equal
rates at the reference voltage even when `driving_force` is zero. Increasing
voltage suppresses reduction and favors oxidation with the declared sign
convention. These are deterministic model reference values, not experimental
rate predictions.

In the checked 18-point scan, the largest grid-refinement change in a rate
was about 0.035%, and increasing from 12 to 24 states changed rates by less
than 2 × 10⁻¹¹ relatively. All configured numerical checks passed; the output
still has `physical_applicability_unverified = true`. The automatic check
keeps the physical coordinate interval fixed, so an independent grid-extent
study remains separate.
