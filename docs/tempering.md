---
orphan: true
---

# Parallel temperature and Hamiltonian replica exchange

PyRPMD can run resumable parallel tempering (PT) and Hamiltonian replica
exchange (HREMD) from TOML for equilibrium `task="pimd"` and selected
`task="fes"` calculations. This workflow uses a new `[tempering]` table. It is
independent of `[replica_exchange]`, which retains its existing meaning:
same-temperature exchange among fixed umbrella windows (REUS).

Replica exchange improves thermodynamic-state mixing; it does not turn PIMD
sampling time into physical dynamics. Every result therefore retains
`converged=false` and `physical_converged=false`. Acceptance, round trips,
diffusion in state space, and MBAR overlap are diagnostics rather than evidence
of bead, timestep, equilibration, or production convergence.

## Temperature ladders

An explicit ladder accepts unit-bearing temperatures:

```toml
[tempering]
mode = "temperature"                 # alias: "pt"
temperatures = ["300 K", "450 K", "700 K", "1400 K"]
exchange_interval_steps = 20
pairing = "odd_even"                 # or "adjacent"
seed = 3702
```

A generated ladder is geometric and includes both endpoints:

```toml
[tempering]
temperature_min = "150 K"
temperature_max = "1200 K"
replicas = 6
spacing = "geometric"
exchange_interval = "2 fs"           # alternative to exact steps
```

The ladder must be strictly monotone and contain at least two states. Each
state has its own inverse temperature and ring-polymer spring action. On an
accepted move from state i to j, momenta are multiplied by
`sqrt(beta_i / beta_j)`; coordinates and walker RNG state travel together.

For walker configurations x_i and x_j, PyRPMD evaluates all target reduced
configurational potentials

\[
u_k(x)=\frac{\beta_k}{P}\left[U_{P,\mathrm{spring}}(x;\beta_k)
+H_k(x)\right]
\]

and accepts an exchange with

\[
\log R=u_i(x_i)+u_j(x_j)-u_i(x_j)-u_j(x_i),\qquad
A=\min(1,e^{\log R}).
\]

The temperature dependence of the spring term is therefore not cancelled or
approximated by a classical potential-energy expression.

## Hamiltonian ladders

HREMD uses the ordinary job temperature and a strictly monotone, nonnegative lambda ladder:

```toml
[tempering]
mode = "hamiltonian"                 # alias: "hremd"
lambdas = [0.25, 0.5, 1.0]
target = "bias"                      # potential, bias, or split_slow
exchange_interval_steps = 20
```

`target="potential"` scales the complete PES energy, forces, and virial.
`target="bias"` scales the selected FES bias and requires `task="fes"`.
`target="split_slow"` scales only the slow member of a `SplitPotential`; it
currently requires `parallel.bead_workers=1`. Lambda scaling never adds a bead
factor to forces and does not silently replace a provider, precision, or
device.

## Supported combinations

The workflow deliberately uses a small whitelist:

| Task/method | Supported tempering | Additional conditions |
| --- | --- | --- |
| `pimd` | PT; PES or split-slow HREMD | fixed cell; legacy engine; PILE-L or fixed Andersen schedule |
| `fes`, umbrella | PT; PES, bias, or split-slow HREMD | exactly one base umbrella state; the lambda ladder supplies replicas |
| `fes`, metadynamics | PT; PES or split-slow HREMD | direct hills, `freeze_after_steps`, no shared/multiple walkers or static walls; exchanges begin after freeze and relaxation |

Barostats, nonlinear bond constraints, active learning, specialized
`[path_integral]` workflows, adaptive umbrella placement, trajectory batching,
Torch/native ensemble engines, OPES/ABF/eABF/steered FES, and simultaneous
REUS `[replica_exchange]` are rejected before output creation. HREMD scaling of
an evolving metadynamics bias is also rejected.

## Parallel execution

Each thermodynamic replica is one trajectory group. For an MPI ladder of G
states, set `parallel.backend="mpi"` and `parallel.trajectory_groups=G`; the
existing G x B task/bead topology remains available for ordinary PES targets.
A static four-state run is:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  mpiexec -n 4 python -m pyrpmd run \
  examples/tempering/double_well.inp --output runs/t37
```

Every group writes a generation-scoped atomic stage file on the shared run
filesystem and returns only its identity, clocks, and reduced-energy scalars to
the coordinator. The coordinator validates the complete generation, applies
exchange proposals in deterministic order, and atomically commits one ensemble
checkpoint. Serial, process, and MPI transports execute the same ordering and
RNG streams; the four-rank integration test compares MPI files bit for bit with
a serial run.

`pairing="odd_even"` alternates non-overlapping even and odd edges.
`pairing="adjacent"` performs a deterministic left-to-right nearest-neighbor
sweep, so a walker may move more than one state in one sweep.

## Checkpoint and resume

`checkpoints/tempering.npz` is the only committed ensemble state. It contains
all q/p rings, every replica RNG, the exchange RNG, walker-to-state
permutation, acceptance counters, propagation metadata, FES bias histories,
samples, and exchange history. The file uses the common checksummed atomic NPZ
container. A changed scientific option changes the run identity; execution-only
parallel settings do not.

A crash before any stage is published replays the last committed generation
exactly. If all stage files are present, resume validates and commits that
complete generation. A strict subset is not a valid smaller ensemble and is
rejected rather than silently discarded or propagated.

```bash
python -m pyrpmd resume runs/t37
python -m pyrpmd analyze runs/t37 --recompute
```

Offline reanalysis reads only the authenticated ensemble checkpoint and does
not construct or evaluate the PES.

## Output and estimators

One trajectory archive per thermodynamic state is written under `samples/` as
`temperature_NNNN.npz` or `hamiltonian_NNNN.npz`. Full rings, centroids, the
walker identity, all-state reduced potentials, target-state estimators, CVs,
steps, times, and explicit atomic units are stored after reorganizing samples
by state rather than by walker.

`results/tempering.json` reports adjacent acceptance counts/rates, completed
endpoint round trips, mean-square state displacement versus exchange lag,
direct state estimates, and MBAR overlap/free energies. PIMD estimators are
recomputed for every MBAR target state because beta appears explicitly in the
primitive kinetic term, centroid-virial kinetic term, spring action, and
well-tempered bias. Reusing a source-temperature estimator at another target
temperature would be incorrect.

The short double-well and umbrella inputs in `examples/tempering/` are analytic
method checks, not production molecular benchmarks. Validation evidence and
reproduction commands are recorded in
`docs/validation/tempering-2026-11-03/README.md`.

## References

- Y. Sugita and Y. Okamoto, “Replica-exchange molecular dynamics method for
  protein folding,” *Chemical Physics Letters* **314**, 141–151 (1999),
  DOI `10.1016/S0009-2614(99)01123-9`.
- Y. Sugita, A. Kitao, and Y. Okamoto, “Multidimensional replica-exchange
  method for free-energy calculations,” *J. Chem. Phys.* **113**, 6042 (2000),
  DOI `10.1063/1.1308516`.
- M. R. Shirts and J. D. Chodera, “Statistically optimal analysis of samples
  from multiple equilibrium states,” *J. Chem. Phys.* **129**, 124105 (2008),
  DOI `10.1063/1.2978177`.

This implementation was written independently and does not copy third-party
source code.
