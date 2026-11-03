```{include} _enhanced-sampling-base.inc
```

## Distributed multiple-walker MetaD and OPES

`walkers_mode` selects how a shared history is transported between walkers:

| value | execution | bias semantics | persistence |
| --- | --- | --- | --- |
| `serial` | existing single-process behavior | unchanged | unchanged historical checkpoint |
| `mpi-sync` | exactly one walker per MPI rank | one rank-ordered, synchronous batch | one atomic ensemble checkpoint |
| `async-files` | independent processes or MPI ranks | locally visible history, eventually merged | one atomic checkpoint and append-only log per walker |

The option is accepted in `[metadynamics]` and `[opes]`. Omitting it does not
insert a default into the resolved input and therefore does not change an
existing run identity. An explicit non-serial value is part of the scientific
fingerprint and cannot resume a serial checkpoint.

### Synchronous MPI

MetaD requires `shared_walkers = true`; `sampling.replicas` must equal the MPI
world size. At every deposition clock, each rank contributes one CV center with
`allgather`. All ranks evaluate the complete center array against the same
pre-commit bias, then commit in rank order. This is the existing serial
`SharedMetadynamicsBias.deposit_walkers` transaction, so a deterministic CPU
run has the same walker trajectories, hill records, observations, momenta and
RNG states as the serial coordinator. A failed rank is reported collectively;
the step, RNG and history are rolled back before an atomic checkpoint is
written.

```toml
[fes.sampling]
replicas = 4

[fes.metadynamics]
shared_walkers = true
walkers_mode = "mpi-sync"
mode = "well_tempered"
sigma = ["0.25 bohr", "0.25 bohr"]
height = "0.02 kcal/mol"
pace = 20
bias_factor = 8.0
```

Launch exactly four ranks:

```bash
mpiexec -n 4 python -m pyrpmd run input.inp --output runs/meta-mpi
```

The authoritative checkpoint is
`runs/meta-mpi/checkpoints/shared_metadynamics_mpi.npz`. The former serial
shared checkpoint remains `shared_metadynamics.npz` and its schema is not
changed.

OPES uses the same rank-count rule. A synchronous batch computes every kernel
weight from the common pre-update bias and commits kernels in rank order:

```toml
[fes.sampling]
replicas = 4

[fes.opes]
walkers_mode = "mpi-sync"
sigma = ["0.20 bohr", "0.20 bohr"]
barrier = "8 kcal/mol"
pace = 20
bias_factor = 10.0
```

Its checkpoint is `checkpoints/opes_shared_mpi.npz`.

### Asynchronous files

Each process owns one walker ID and one file under
`RUN/walkers/metadynamics/` or `RUN/walkers/opes/`. It may be launched by MPI,
or as independent scheduler jobs with explicit IDs:

```bash
PYRPMD_WALKER_ID=0 PYRPMD_WALKERS=4 python -m pyrpmd run input.inp --output runs/async &
PYRPMD_WALKER_ID=1 PYRPMD_WALKERS=4 python -m pyrpmd run input.inp --output runs/async &
PYRPMD_WALKER_ID=2 PYRPMD_WALKERS=4 python -m pyrpmd run input.inp --output runs/async &
PYRPMD_WALKER_ID=3 PYRPMD_WALKERS=4 python -m pyrpmd run input.inp --output runs/async &
wait
```

Use `walkers_mode = "async-files"` in the selected method table. The shared
directory has a checksummed manifest binding it to the method, walker count and
run fingerprint. `walker-NNNN.lock` rejects two live writers with the same ID.
Each writer appends checksum-protected JSON records only to
`walker-NNNN.jsonl`; readers sort the committed event set by
`(step, time, walker, sequence)` before rebuilding the numeric history.

A newline is the commit boundary. A complete malformed record or checksum
mismatch is corruption and stops the run. An unterminated final record is not
consumed and does not advance the saved byte offset; after taking the writer
lock, its owner truncates that crash tail before appending again. Every walker
checkpoint stores all reader offsets, duplicate guards, the next local
sequence, dynamics RNG and a possible `pending_record`. Deposition uses two
phases:

1. atomically checkpoint the post-step state and exact prepared record;
2. idempotently ensure that record is in the owner log, merge committed logs,
   clear `pending_record`, and atomically checkpoint again.

Consequently a crash before, during or after publication neither drops nor
duplicates a kernel. Other walkers do not wait for the failed process. The last
walker to finish writes the aggregate FES; an earlier finisher reports
`walker_complete_pending_merge`. Per-walker checkpoints are
`shared_metadynamics_async_NNNN.npz` and `opes_shared_async_NNNN.npz`.

Asynchronous MetaD preserves the height and `bias_before` calculated from the
writer's locally visible history. It never re-tempers a remote hill during
merge. Therefore it is eventually consistent and statistically comparable to
the synchronous algorithm, but it is not trajectory-wise or bitwise identical.

### Shared OPES kernel and normalization rules

Distributed OPES currently requires fixed bandwidth, KDE `cv_density`
analysis and no frozen-bias phase. Every kernel is retained: compression is
`none`, memory is linear in kernel count, and rebuilding the exact shared
normalization is quadratic in kernel count. This is explicit rather than a
silent approximation.

For kernel centers \(s_i\), log weights \(\ell_i\), and normalized kernel
\(K\), the shared estimate is

\[
P(s)=\frac{\sum_j e^{\ell_j}K(s,s_j)}{\sum_j e^{\ell_j}},
\qquad
Z=\frac{1}{N}\sum_i P(s_i).
\]

`log_z` is reconstructed from all uncompressed kernels in deterministic event
order. In `metad` mode, a synchronous batch uses
`log_weight = beta * V_before` with one common pre-commit history; asynchronous
writers use their locally visible `V_before`. In `explore` mode every shared
kernel has zero log weight. The checkpoint records the complete event set and
rejects a supplied `log_z` that is not exactly reproduced.

### Restrictions and interpretation

Distributed MetaD does not combine with the Hermite bias grid,
`multiple_walkers = true`, frozen-bias production, or non-legacy sampler
variants. Distributed OPES does not support adaptive widths, compression,
frozen-bias reweighting, or `rate_flux`. These combinations fail explicitly;
there is no serial or precision fallback.

The summaries report `converged = false` and `physical_converged = false`.
One shared history couples the walkers, so walker spread is not an independent
replica uncertainty. MPI equality tests establish deterministic implementation
parity, and the asynchronous toy campaign establishes only statistical
compatibility for its specified model and budget. Neither establishes CV
adequacy, equilibrium, bead convergence, timestep convergence, or a molecular
free-energy error bar.
