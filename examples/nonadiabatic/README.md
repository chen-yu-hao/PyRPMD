# Two-state nonadiabatic exchange

This example runs the complete NRPMD chain in about 15 seconds: focused electronic
initialization, matrix-PES propagation, chemical product classification and signed electronic
populations.

> **Scope.** The labels describe H + HD → H₂ + D, but the potential is an artificial teaching
> model, not a fitted H3 surface: two diabatic states with an arbitrary constant gap and a
> localized coupling. Eight trajectories and two short parent chains exercise the workflow; they
> are not a physical prediction.

From the repository root, after installing PyRPMD:

```bash
pyrpmd validate examples/nonadiabatic/exchange.inp --check-pes
pyrpmd run examples/nonadiabatic/exchange.inp --workers 2 --output runs/nrpmd-exchange
pyrpmd resume runs/nrpmd-exchange --block-steps 32
pyrpmd analyze runs/nrpmd-exchange
```

## Reading the output

This is a fixed-collision-energy experiment, so eligible final estimates are cross sections, not
thermal rates. Products may not appear in so small a sample, and long-lived encounters can stay
unresolved. Final electronic cross sections are reported once every trajectory has resolved
without failure. `resume` recovers interrupted work; it does not extend completed trajectories
beyond the original time budget.

## Toward a real calculation

For a state-selected thermal collision experiment, set `sampling = "thermal_flux"` and remove
`energy`. Converge beads, time step, parent equilibration and independence, impact cutoff,
separation, asymptotic coupling tolerance, trajectory duration and classification settings. See
[the method and interface](../../docs/nonadiabatic.md).
