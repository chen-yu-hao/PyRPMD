# Expensive potentials and dynamics extensions

These additions target repeated model setup and force evaluation cost. Existing
scalar RPMD, PILE-L sampling, checkpoint identities and physical bead masses
remain the defaults. Choose an approximation explicitly and test its convergence
for the intended observable.

## Reuse a potential within a worker

Gas-phase/surface recrossing parents and children, periodic site-rate tasks, and
umbrella windows now borrow one potential for each identical potential/System
specification within a worker's task batch. In bead mode that also keeps the
spawned bead process pool alive across tasks. The worker closes its resources
after the batch, including when a task raises. A caught failed lease is evicted
and closed immediately; nested leases receive separate instances.

The actual provider object is returned, preserving native binding dispatch.
Systems with different masses, geometry, atom order, charge, multiplicity, cell
or boundary conditions receive distinct providers. No instance is shared between
processes or between separate `run_tasks` calls. Device selection stays local to
each worker. Bead workers call plugin `close()` during normal process shutdown.

The default is `parallel.reuse_potentials=true`. To retain per-task construction:

```toml
[parallel]
reuse_potentials = false
```

History-dependent plugins must declare `metadata['stateful']=True` or
`metadata['reuse_across_tasks']=False`. The bead wrapper asks the actual initialized
plugin for this capability. A supplied live Python object or configuration that
cannot be safely snapshotted is not automatically cached. Changing batch size or
this resource setting does not change the scientific checkpoint identity.

## Batch independent trajectories

For a frame-independent PES, multiple rings can share one `[K*P,N,3]` force call.
The ring propagator retains `P` beads and treats the trajectories as independent
atom blocks, so springs never connect different trajectories. The
[batching guide](batched-dynamics.md) describes masks, failure handling, model
capabilities and the public `BatchedDynamics` API.

Gas-phase and fixed-surface recrossing can select a maximum number of child
**pairs** per batch:

```toml
[recrossing]
child_batch_pairs = 8
```

This launches up to 16 opposite-momentum trajectories together. The default, 1,
keeps the scalar path. Pair seeds, signed flux weights, filenames and checkpoint
contents retain their established scientific meaning; repacking pending pairs
does not redraw their momenta. Completed pair checkpoints can be replayed by the
scalar path. This option needs `dynamics.loop_backend='auto'` or `'python'` because
Python orchestrates batched PES calls; the free-ring kernel can still be native.
Batched recrossing requires `parallel.bead_workers=1`. Use process trajectory
workers or MPI trajectory groups (one rank per group) for concurrent batches;
keep `child_batch_pairs=1` for bead-process or MPI-bead execution.

Use batching when PES launch cost or GPU utilization dominates. For a cheap
analytic/Fortran PES, the existing fully fused scalar native loop can be faster.
Cross-trajectory GPU batching does not imply fully device-resident integration:
the present public PES contract returns host arrays. Stateful, ring-coupled and
undeclared custom providers are rejected. The `site_rate` checkpoint workflow,
dRPMD production runner and thermostatted umbrella runner do not yet dispatch
trajectory batches; the latter two can use the low-level batch API in custom
drivers. Existing trajectory parallelism continues to work.

## RPC and multiple time stepping

[RPC/MTS](rpc-mts.md) adds explicit Python APIs `RingPolymerContraction`,
`DifferencePotential`, `SplitPotential` and `MTSDynamics`. The slow term is a
declared physical correction, evaluated on a contracted ring and lifted with the
chain-rule normalization. The fast term and springs retain the full ring.
These APIs are not enabled implicitly in rate or sampling inputs. Validate the
contraction bead count, force split and outer timestep against the full potential;
fewer calls alone do not establish an accurate rate.

## TRPMD and spectra

[TRPMD/spectroscopy](trpmd-spectroscopy.md) provides internal-mode Langevin dynamics,
velocity/dipole/flux correlations, power spectra and relative IR intensity. All
these use physical bead masses. PA-CMD mass scaling is not implemented.

For a gas-phase or fixed-surface rate sensitivity calculation:

```toml
[recrossing]
dynamics = "trpmd"
friction_scale = 1.0
child_batch_pairs = 1
```

Internal mode friction is `gamma_k = 2*friction_scale*omega_k`, while the centroid
has exactly zero friction. `friction_scale=0` reproduces deterministic RPMD without
thermostat random draws. Equilibrium parent sampling is unchanged. TRPMD children
use the established unconstrained Maxwell momenta and independent stochastic
noise for the two signs of each pair.

The result is labelled `trpmd_sensitivity_estimate` and reports
`trpmd_rate_estimate_cm3_s` (or `_s_inv` for the fixed surface). The ordinary formal
`rate_cm3_s`/`rate_s_inv` stays null: this is a friction sensitivity estimate, not
an RPMD prediction. Endpoint energy changes include thermostat heat and cannot
by themselves diagnose integration error. Converge timestep, sampling
and friction dependence separately. `site_rate`, velocity-Verlet and batched
TRPMD children are rejected explicitly.

## Reproducible validation

The repository includes regression tests for native/NumPy equivalence, provider
lifetime, per-pair random streams, failure masks, RPC force differentiation,
time reversal, thermostat equilibrium covariance, correlation normalization and
spectral peaks. Reproducible model performance scripts are:

```bash
PYTHONPATH=src python benchmarks/batched_dynamics.py --help
PYTHONPATH=src python benchmarks/rpc_mts.py --help
PYTHONPATH=src python benchmarks/trpmd/harmonic_spectrum.py --help
```

Model checks, real-PES software regression and agreement with published converged
molecular rates are separate levels of evidence. Published-rate references and
comparison rules are tracked under `benchmarks/published_rates`; a finite pilot
result is never promoted to a physical-validation pass merely because it is near
a reference number.

The [2026-10-31 validation record](validation/tier1-2026-10-31/README.md)
contains software checks, measured setup/batching costs, and the unsuccessful
short H + H₂ literature-comparison pilots with their sampling diagnostics.
