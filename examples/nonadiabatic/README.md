# Nonadiabatic model demonstrations

All examples here use synthetic, uncalibrated models. Their diagnostics verify
the interfaces and numerical controls; they do not establish a converged rate,
diffusion coefficient, or chemical prediction.

| Example | Entry point | Scope |
| --- | --- | --- |
| `exchange.inp` | `pyrpmd run` | Focused-state NRPMD direct collisions |
| `soc_and_rpsh.py` | `python` | Complex Hermitian SOC-NRPMD and real centroid RPSH |
| `thermal_rates.py` | `python` | Signed SEO thermal sampling and finite-time nuclear/electronic rate curves |
| `method_models.py` | `python` | Original KC-RPMD, MV-RPMD and iso-Ehrenfest model APIs |
| `friction_surface.inp` | `pyrpmd run` | Markovian centroid EF-RPMD diffusion over a fixed analytic slab |

Run the new examples from the repository root after installing PyRPMD:

```bash
python examples/nonadiabatic/soc_and_rpsh.py
python examples/nonadiabatic/thermal_rates.py
python examples/nonadiabatic/method_models.py
pyrpmd validate examples/nonadiabatic/friction_surface.inp --check-pes
pyrpmd run examples/nonadiabatic/friction_surface.inp --output runs/ef-rpmd-surface
pyrpmd resume runs/ef-rpmd-surface
```

The friction example uses an illustrative relaxation rate and an analytic PES;
its Cu labels do not imply first-principles copper friction or a fitted surface.
The thermal example deliberately reports no rate constant without a verified
plateau. See [electronic friction](../../docs/electronic-friction.md),
[thermal NRPMD](../../docs/thermal-nrpmd.md),
[SOC and RPSH](../../docs/nonadiabatic-soc-rpsh.md), and
[KC/MV/isomorphic methods](../../docs/nonadiabatic-methods.md) for the equations,
assumptions and validation requirements.

## Two-state exchange

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
