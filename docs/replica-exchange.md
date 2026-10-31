# Replica-exchange umbrella sampling

`task="fes"` supports same-temperature replica-exchange umbrella sampling
(REUS) in one or two CV dimensions. PILE-L is also available for ordinary
fixed-window and adaptive FES sampling. Andersen remains the default.
Exchanged trajectory time cannot be used as physical RPMD time for rates,
recrossing factors or branching fractions.

## Input and execution

Add this section to a fixed-window FES input:

```toml
[replica_exchange]
enabled = true
interval_steps = 50
# Optional zero-based graph connecting all windows:
# pairs = [[0, 1], [1, 2]]
```

Windows share their temperature, bead count, masses, physical potential and
boundary conditions. The measure must be `cv_density`. Adaptive window
insertion, temperature exchange, different bead counts or potentials, and
exchange of history-dependent biases are not supported.

Without `pairs`, a deterministic minimum spanning tree connects the centers
using `beta * delta.T @ K @ delta`. Periodic CVs use wrapped displacements.
An explicit graph is required above 2048 windows. Self-pairs, duplicates,
out-of-range indices and disconnected graphs are rejected. Graph connectivity
does not establish distributional overlap.

`sampling.replicas` counts independent complete exchange ensembles. Existing
process/MPI scheduling distributes these ensembles; windows within each
ensemble currently propagate sequentially. Bead-parallel potential evaluation
is reused. No window-level distributed exchange or fixed speedup is claimed.

```bash
pyrpmd validate examples/fes/replica_exchange_2d.inp
pyrpmd run examples/fes/replica_exchange_2d.inp --output runs/reus2d
pyrpmd resume runs/reus2d
pyrpmd analyze runs/reus2d --recompute --bins 12 12 --bootstrap 10
```

The example uses a four-bead harmonic model, nine windows and two independent
ensembles. It is a method check and retains `converged=false`. Python callers
continue to use `run_fes` and `reanalyze_fes`.

## Exchange measure

Each sweep visits every graph edge once in a configuration-independent random
order. A proposal swaps complete q/p rings. For physical umbrella energies W:

\[
\log R_{ij}=-\beta[W_i(s_j)+W_j(s_i)-W_i(s_i)-W_j(s_j)],
\qquad A_{ij}=\min(1,e^{\log R_{ij}}).
\]

PyRPMD adds P W to the sum-beads Hamiltonian and samples it at beta/P. There is
no additional bead factor in this exponent. Spring, kinetic and physical
potential terms cancel; momenta need no temperature rescaling. Window clocks
and RNG streams stay at their windows.

Samples belong to the active umbrella window. Observation precedes a coincident
exchange. `fes.h5` records `exchange_group` and `walker_ids`; `exchange.json`
records edge attempts, acceptances and walker/window visits, including
exchanges during equilibration.

## PILE-L

```toml
[dynamics]
integrator = "cayley-bcb"
loop_backend = "auto"
timestep = "0.1 fs"

[sampling]
thermostat = "pile-l"
gamma_centroid = "0.02 fs^-1"
equilibration = "20 fs"
production = "100 fs"
interval = "0.2 fs"
replicas = 2
```

Cayley uses BCOCB; exact normal-mode BAB uses BAOAB. Both reuse
`Dynamics.sample_step` and the native free/OU/free substep. The optional
`sampling.integrator` must match. Centroid friction defaults to `0.01 fs^-1`;
bare low-level API values use atomic units.

PILE-L cannot use velocity-Verlet or a strict `loop_backend="native"` outer
loop. With `auto`, native substeps remain available. Periodic cells and fixed
atoms retain ordinary FES potential/CV contracts. Other tasks keep their own
thermostat restrictions.

## Correlated errors and restart

Exchange couples the windows in an ensemble. The MBAR point estimate is
unchanged, while bootstrap draws use identical circular time-block indices
across its windows. Different ensembles are resampled independently. The
analysis API exposes this as `bootstrap_groups`. Grouped streams must be
synchronized and equally long. The largest estimated CV correlation time in
each group determines its common block length. Existing insufficient-block
diagnostics remain in effect.

CV autocorrelation can miss hidden slow motion. Acceptance rates and walker
visits do not establish equilibration. Independent starting families and
checks of sampling duration, block length, beads and timestep remain necessary.

Each `checkpoints/exchange_0000.npz` atomically saves all window states, clocks
and RNG, the exchange RNG, samples, labels and exchange history. A failure in
one window or partway through a sweep rolls back the entire synchronization
event. An interrupted write leaves an old or new complete checkpoint.
Completed resume and offline analysis do not construct a potential.

For the direct Python workflow, checkpoint identity uses the mathematical
parameters of supported, unmodified built-in CV objects. Custom, overridden,
periodic or surface CV objects must be supplied through a serializable CV
mapping/factory specification. Periodic and surface mappings retain their
ordinary FES support. This prevents object memory addresses from invalidating
restart after a copy or worker-process transfer.

## References

- Sugita, Kitao and Okamoto, *Multidimensional replica-exchange method for
  free-energy calculations*, JCP **113**, 6042 (2000),
  [doi:10.1063/1.1308516](https://doi.org/10.1063/1.1308516),
  [author manuscript](https://arxiv.org/abs/cond-mat/0009120).
- Ceriotti et al., *Efficient stochastic thermostatting of path integral
  molecular dynamics*, JCP **133**, 124104 (2010),
  [doi:10.1063/1.3489925](https://doi.org/10.1063/1.3489925),
  [author manuscript](https://arxiv.org/abs/1009.1045).

This independently written implementation does not copy external source code.
