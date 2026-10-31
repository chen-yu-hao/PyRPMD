# Potential adapters and concurrent force evaluation

These examples exercise software interfaces; their short model calculations are
not converged predictions for a material or reaction.

## Runnable examples

```bash
pip install -e '.[test,tblite,pyscf]'
OMP_NUM_THREADS=1 python examples/potentials/on_the_fly.py tblite
OMP_NUM_THREADS=1 python examples/potentials/on_the_fly.py pyscf

# Install a suitable CPU/CUDA PyTorch wheel first.
pyrpmd validate examples/potentials/periodic_torch.inp --check-pes
pyrpmd run examples/potentials/periodic_torch.inp --output runs/periodic-torch

# Periodic compiled model with analytic force and strain virial:
cc -O3 -std=c11 -shared -fPIC -I native/include \
  native/examples/periodic_cosine_pes.c -lm -o /tmp/libperiodic_pes.so
pyrpmd pes inspect /tmp/libperiodic_pes.so

# Two trajectory groups, two bead ranks each, one coordinator:
mpiexec -n 5 pyrpmd run examples/drpmd/mpi_dynamic.py --output runs/mpi-dynamic
```

C ABI v2 contract and Fortran `ISO_C_BINDING` layout are documented in
[`native/ABI.md`](../../native/ABI.md). The generic Torch example demonstrates
explicit cell/PBC handling and autograd virials; it is not a transferable PES.

## Model selection

All model paths below are placeholders to replace with your own applicable
checkpoint. MACE, NequIP and fairchem reuse the official ASE calculator; they
currently evaluate one structure at a time. DeePMD and generic Torch have native
batch paths. See the [adapter guide](../../docs/potential-adapters.md) for exact
options, dependency versions and validation limits.

```toml
[potential]
provider = "mace"
model = "models/material.model"
device = "cpu"
params = {default_dtype = "float64"}
```

```toml
[potential]
provider = "nequip"
model = "models/ase-model.nequip.pt2"
device = "cpu"
```

```toml
[potential]
provider = "deepmd"
model = "models/frozen-model.pb"
# Device selection is configured in the DeePMD backend, outside PyRPMD GPU mode.
```

```toml
[potential]
provider = "fairchem"
model = "models/uma-checkpoint.pt"
device = "cpu"
params = {task_name = "omat", inference_settings = "batch"}
```

For fairchem molecular calculations use the model's `omol` task and specify
System charge/multiplicity. Model access and accepted inference presets depend on
the installed fairchem version. Named ML precision is the model's own precision;
conversion to PyRPMD float64 arrays does not add numerical information.

## Concurrent i-PI clients

```toml
[potential]
provider = "ipi"
params = {host = "127.0.0.1", port = 31415, clients = 4, timeout = 60.0}

[parallel]
backend = "serial"
bead_workers = 1
```

Start four identical force drivers connecting to port 31415. Each receives one
bead at a time using the ordinary i-PI protocol, then the next pending bead.
PyRPMD returns complete ordered batches. All declared clients must connect before
the first batch. An incomplete/invalid exchange closes the pool and fails the
batch. Driver configuration must honor the cell and partial-PBC INIT contract.
Different PyRPMD task slots require distinct server addresses; port offsets are
not assigned automatically.
