```{include} _plumed_legacy.inc
```

## Replica and bead-resolved extension

Existing inputs that omit `plumed.mode`, `plumed.walkers`, and
`sampling.replicas` continue to use the original single-kernel centroid path
and its checkpoint identities. The extension is selected only when at least one
new control is present or `sampling.replicas > 1`.

```toml
[sampling]
replicas = 4

[plumed]
mode = "centroid"          # centroid or beads
walkers = "independent"    # independent, files, or mpi
restart = "files"          # required for automatic continuation
```

Every replica owns a separate PLUMED kernel, log, and working directory:

```text
RUN/plumed/replica-0000/
RUN/plumed/replica-0001/
...
RUN/logs/plumed-0000.log
RUN/checkpoints/plumed-replica-0000.npz
```

Relative `FILE=` paths in PLUMED input are therefore replica-local. PyRPMD
rejects absolute paths, parent-directory escapes, Windows drive paths, and
user-supplied `WALKERS_*` or `MULTIPLE_WALKERS` options. The workflow owns the
replica IDs and communicator topology.

### Coordinate modes

`mode = "centroid"` retains the established convention. PLUMED sees the `N`
physical atoms at the ring-polymer centroid, its physical bias is
`W(q_centroid)`, and the sum-beads Hamiltonian receives `P W`. The same
centroid bias force is applied to every bead.

`mode = "beads"` exposes `P*N` synthetic atoms in bead-major order:

```text
PLUMED atom (bead, atom) = bead * N + atom + 1
```

PLUMED's returned bias is added once to the ring-polymer Hamiltonian and each
synthetic-atom force is mapped directly to the corresponding bead, without a
bead factor. The run writes the complete one-based mapping to
`results/plumed-atom-map.json`. ENERGY-dependent PLUMED actions are rejected in
this mode because the PES energy-force chain for the synthetic atoms is not
available.

The atom map is scientific state. Changing `mode`, bead count, replica count,
walker method, or configured PLUMED program changes run identity. Existing
inputs that do not select the extension retain their old identity.

### Independent replicas and native walkers

The supported walker matrix is:

| `plumed.walkers` | PLUMED actions | Transport | Required topology |
| --- | --- | --- | --- |
| `independent` | any action accepted by the bridge | no shared bias | serial, process, or static MPI groups |
| `files` | `METAD`, `PBMETAD` | PLUMED `WALKERS_N/ID/DIR/RSTRIDE` | one static process/MPI group per replica |
| `mpi` | `METAD`, `PBMETAD`, `OPES_METAD`, `OPES_METAD_EXPLORE`, `OPES_EXPANDED`; VES optimizers | PLUMED multi-sim communicator | static MPI, `trajectory_groups = replicas`, no coordinator ranks |

File-sharing walkers use `RUN/plumed/shared` and commit PLUMED updates in
walker-ID order. This deterministic schedule avoids concurrent append races
and does not create a second worker pool: propagation still runs exclusively
inside PyRPMD's existing trajectory groups. `walker_read_stride` is admitted
only for file walkers.

MPI walkers bind two communicators before PLUMED `init`: `MPI.COMM_SELF` for the
single process that owns each kernel and a leaders-only multi-sim communicator
for inter-walker collectives. PES bead/rank workers remain in the existing
PyRPMD group and are not entered into the PLUMED communicator. This follows
PLUMED's `setMPIComm`/`setMPImultiSimComm` contract.

A four-rank invocation is:

```bash
mpiexec -n 4 pyrpmd run examples/plumed/mpi-ves-2d.inp \
  --output runs/plumed-ves-2d
```

The PLUMED build must include the actions used by the input. In particular, the
VES example requires a PLUMED build configured with the optional `ves` module.
PyRPMD never substitutes a different action or silently falls back to
independent walkers.

### External-files-v2 restart

For every extended mode, automatic continuation requires:

```toml
[plumed]
restart = "files"
```

At each committed checkpoint PyRPMD atomically stores coordinates, momenta,
simulation clock, complete NumPy RNG state, samples, PLUMED API version, and an
external-file manifest. Every regular file below the replica directory (and
`plumed/shared` for file walkers) is recorded with:

- normalized relative path and ownership scope;
- committed byte offset;
- SHA-256 of the committed prefix;
- the exact prefix bytes required to truncate a crash tail.

Resume is deliberately strict. It rejects a missing file, shortened file,
changed prefix, replacement symlink, non-regular file, extra untracked file,
replica-manifest disagreement, changed input/model hash, changed scientific
settings, or changed PLUMED API version. Extra bytes after a verified prefix
are treated as an interrupted post-checkpoint append and are atomically
truncated. After PLUMED restart initialization, the saved endpoint is replayed
once to prime PLUMED's update clock; PyRPMD verifies unchanged bias/forces and
then removes only the replayed output tail while keeping PLUMED's open inodes.

`FLUSH STRIDE=1` is injected when absent so checkpointed offsets describe
flushed state. The NPZ is still the transaction commit point. A failed PLUMED
command never advances it, and the failure report distinguishes the last MD
step accepted in memory from the last complete restart point.

The external-files-v2 profile is separate from the older audited
`fixed-hills-v1` profile described above. Old checkpoints continue to resume
through the legacy path; they are not rewritten or reinterpreted.

### Examples and validation

- `examples/plumed/file-walkers-metad.inp` is a four-process shared-HILLS
  metadynamics smoke example.
- `examples/plumed/mpi-ves-2d.inp` and `ves-2d.dat` define a two-dimensional,
  four-walker VES calculation on the analytic double well. It is a workflow and
  restart demonstration, not a converged physical free-energy result.
- `docs/validation/plumed-bridge-2026-11-03/report.md` records the independent
  finite-difference, restart, process-walker, MPI, and real-PLUMED checks used
  for this implementation.

Primary PLUMED contracts used here are documented in the PLUMED 2.10 pages for
[METAD](https://www.plumed.org/doc-v2.10/user-doc/html/_m_e_t_a_d.html),
[PBMETAD](https://www.plumed.org/doc-v2.10/user-doc/html/_p_b_m_e_t_a_d.html),
[OPES_METAD](https://www.plumed.org/doc-v2.10/user-doc/html/_o_p_e_s__m_e_t_a_d.html),
[VES_LINEAR_EXPANSION](https://www.plumed.org/doc-v2.10/user-doc/html/_v_e_s__l_i_n_e_a_r__e_x_p_a_n_s_i_o_n.html),
and
[OPT_AVERAGED_SGD](https://www.plumed.org/doc-v2.10/user-doc/html/_o_p_t__a_v_e_r_a_g_e_d__s_g_d.html).
