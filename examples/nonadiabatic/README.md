# Two-state nonadiabatic exchange demonstration

This small example connects focused electronic initialization, matrix-PES
propagation, chemical product classification and signed electronic populations.
The labels describe H + HD -> H2 + D, but **the potential is an artificial teaching
model, not a fitted H3 potential or a physical prediction**. The two diabatic
states have an arbitrary constant gap and a localized, nonzero coupling.

From the repository root, after installing PyRPMD:

```bash
pyrpmd validate examples/nonadiabatic/exchange.inp --check-pes
pyrpmd run examples/nonadiabatic/exchange.inp --workers 2 --output runs/nrpmd-exchange
pyrpmd resume runs/nrpmd-exchange --block-steps 32
pyrpmd analyze runs/nrpmd-exchange
```

This is a fixed-collision-energy experiment. Its eligible final estimates have
cross-section units, not thermal-rate units. Eight trajectories and two short
parent chains are sufficient only to exercise the workflow. Products need not
appear in this tiny sample. Long-lived encounters may remain unresolved; final
electronic cross sections are withheld when unresolved or failed trajectories
remain. `resume` recovers interrupted work; it does not extend completed
trajectories past the original time budget.

Use `thermal_flux` and remove `energy` to configure a state-selected thermal
collision experiment. A meaningful result also requires convergence in beads,
timestep, parent equilibration and independence, impact cutoff, separation,
asymptotic coupling tolerance, trajectory duration and classification settings.
See [the method and interface](../../docs/nonadiabatic.md).
