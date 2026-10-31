# Constant-potential electrode models and surface current

`pyrpmd.electrochemistry` connects electron-number-dependent potentials to the existing
periodic-system, fixed-substrate, FES and propagation interfaces. Choose the physical model
explicitly before composing the following components.

| Calculation | Entry point | Interpretation |
| --- | --- | --- |
| Adiabatically relaxed electron number | `AdiabaticElectrodePotential` with `task="fes"` | A grand-potential minimum for each independent nuclear configuration |
| Ring polymer with one shared electron number | `GrandCanonicalSampler` | Joint HMC/MC equilibrium samples; sweep counts have no physical-time meaning |
| Surface behavior from supplied Volmer, Heyrovsky and Tafel rates | `steady_state_her` | Steady hydrogen coverage, hydrogen turnover and signed current density |

The existing [`task="pcet"`](pcet.md) computes local weak-coupling golden-rule rates.
Its rates in s⁻¹ require consistent site, activity and standard-state definitions before
they can serve as surface kinetic coefficients.

## Run the examples

From the repository root:

```bash
pyrpmd validate examples/electrochemistry/fes2d.inp --check-pes
pyrpmd run examples/electrochemistry/fes2d.inp --output runs/electrode-fes
pyrpmd resume runs/electrode-fes
python examples/electrochemistry/sample_grand_canonical.py
python examples/electrochemistry/her_scan.py
```

The periodic charging model is analytic and is not fitted to an electrode material.
The FES run is a short software check. The HER scan uses separately supplied teaching
rate coefficients; it does not extract them from the FES.

`ElectronReservoir` requires an explicit voltage reference and reference electron
chemical potential:

$$
\mu_e(U)=\mu_{e,\mathrm{ref}}-e(U-U_\mathrm{ref}).
$$

No SHE/RHE conversion or electronic-energy calibration is inferred. The model interface
`evaluate(q, electron_numbers)` returns `ElectronicPESResult` with energy, nuclear forces
and the positive derivative `denergy_dnumber`. All quantities use atomic units; electron
number may be a model-defined continuous excess number.

## Two electron-number models

The adiabatic adapter defines
\(\Omega(q)=\min_N[E(q,N)-\mu_eN]\) independently for every configuration. Each nuclear
bead evaluates this effective surface separately, preserving the existing PES batch and
parallel-execution contracts. The electronic branch must be smooth and strictly convex,
with an interior minimum in the specified domain. Failed or out-of-domain solutions raise
an error.

The shared-number sampler targets a different equilibrium distribution:

$$
\rho(q,N)\propto\exp\left[-\frac{\beta}{P}H_P(q;N)+\beta\mu_eN\right].
$$

It reuses `Dynamics` for fixed-N nuclear HMC proposals, followed by electron-number MC.
Electron moves use the mean bead energy; the nuclear HMC acceptance exponent is
\(-\beta\Delta H_P/P\). Diagnostics count Monte Carlo sweeps. This API does not implement
extended-electron TP-RPMD real-time dynamics or directly produce a recrossing coefficient.

## From elementary rates to current

`HERRates` takes six forward/reverse coefficients. `steady_state_her` also requires proton
activity, hydrogen activity and molar site density. Current is measured in A/m², with
anodic current positive and cathodic current negative. At steady state,
\(j=-2F\Gamma_{\mathrm{site}}\mathrm{TOF}_{\mathrm H_2}\); away from steady state the
adsorbed-hydrogen accumulation term must be retained.

Periodic proton bond differences can be used as FES coordinates. Surface-rate calculations
must obey the coordinate and mass-metric requirements in the
[materials and surfaces chapter](periodic.md); the electrode wrapper does not change them.

The [electrochemistry reference](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/electrochemistry.md)
contains the full equations, Python API, source attribution and scope.
