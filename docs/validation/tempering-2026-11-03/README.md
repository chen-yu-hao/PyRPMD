---
orphan: true
---

# T37 parallel-tempering validation — 2026-11-03

This record validates the implementation contract for temperature and
Hamiltonian replica exchange. It is not evidence that an arbitrary molecular
calculation is physically converged. All workflow summaries remain
`converged=false` and `physical_converged=false`.

## Environment

- Linux x86_64 virtual machine, AMD EPYC 9V74, 5 visible single-thread cores.
- Python 3.13.5, PyRPMD 0.2.0, NumPy 2.3.5, SciPy 1.17.0, pytest 9.0.2.
- `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` for reported tests and timings.
- Local source tree based on `main` tree
  `a1fdad8f90b775788f4459c93ac281af946561b9` (`ceee59c`).
- The sandbox had no `mpiexec` or `mpi4py`; the real four-rank equality test is
  `tests/test_tempering_mpi.py` and runs where the project's MPI test extra is
  installed. No local MPI result is claimed here.

The environment could not contact package indexes during isolated build setup,
so the editable test install used the already installed build dependencies:

```text
python -m pip install -e '.[test]' --no-build-isolation
Successfully built pyrpmd
Successfully installed pyrpmd-0.2.0
```

## Pre-change baseline

Before implementation, the task-focused selection was:

```text
$ OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python -m pytest -q tests -k 'replica_exchange or tempering'
124 passed, 9 skipped, 11693 deselected in 29.91s
```

## Post-change task-focused regression

After implementation, the same repository selection plus the new tempering
tests reported:

```text
$ OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python -m pytest -q tests/test_tempering.py tests/test_tempering_mpi.py \
      tests/test_replica_exchange.py tests/test_parallel.py tests/test_checkpoint.py \
      tests/test_fes.py tests/test_metadynamics.py tests/test_runner.py \
      tests/test_batched_ml_deepmd.py tests/test_batched_ml_fairchem.py \
      tests/test_batched_ml_mace.py tests/test_batched_ml_metatomic.py \
      tests/test_batched_ml_nequip.py tests/test_batched_ml_workflow.py \
      tests/test_onnx_native.py tests/test_pyscf_multistate.py \
      tests/test_water_preparation_extension.py
143 passed, 10 skipped, 11693 deselected in 31.04s
```

The ten skips were optional-package guards, including the MPI equality test in
this non-MPI sandbox. The direct tempering module selection separately reported
`19 passed in 10.73s`.

A wheel and source distribution were built locally. Both contain
`src/pyrpmd/workflows/tempering.py` and the two checked-in tempering examples.

## Independent analytic checks

`tests/test_tempering.py` contains four references that do not reuse the
workflow's propagated result as their oracle:

1. **Finite-P harmonic Gaussian.** For every target temperature, the exact
   covariance is the inverse cyclic spring-plus-harmonic precision matrix. The
   potential, primitive kinetic, centroid-virial kinetic, and internal-energy
   expectations are reconstructed by covariance traces and compared with
   `evaluate_thermodynamic_states` at relative tolerance `2e-13`.
2. **Classical harmonic PT acceptance.** For three Cartesian degrees of
   freedom, the exact two-temperature mean acceptance is
   `2 I_{1/(1+beta_cold/beta_hot)}(3/2,3/2)`. A fixed-seed 180,000-pair draw
   agrees within `2.5e-3`.
3. **Permutation detailed balance.** A three-walker, three-state discrete model
   is sampled for 120,000 odd/even sweeps. The chi-square statistic is
   `1.1538622198362165` with p-value `0.9492263320879439` against the exact
   Boltzmann permutation distribution.
4. **Transactional restart.** An injected failure after a committed generation
   is resumed; every ensemble-checkpoint and thermodynamic-state trajectory
   array is bitwise identical to an uninterrupted run. A deliberately partial
   staged generation is rejected.

The tests also cover bias-scaled umbrella HREMD, frozen direct-hill
metadynamics PT, strict unsupported-combination failures, conservative output,
state-regrouped trajectories, process/serial bitwise equality, and offline
reanalysis without a PES.

## Double-well mixing campaign

The symmetric analytic model in `examples/tempering/double_well.inp` starts all
rings in the negative well. Symmetry gives the exact positive-well probability
`1/2` and positive/negative occupancy ratio `1`, including at finite P. The
short campaign deliberately uses a light validation mass to expose a clear
mixing contrast; it is not a molecular benchmark.

Parallel tempering was run with P=8, temperatures 300/450/700/1400 K, 200
warm-up steps, 4000 production steps, a 10-step sample stride, and 20-step
exchange interval. The same model was run at 300 K without tempering.

```text
                              PT, 300 K state     single 300 K
samples                       400                 400
fraction x > 0                0.55                0.05
mean x / bohr                -0.0299836          -0.4446813
exact fraction x > 0          0.50                0.50
```

PT adjacent acceptance counts were `38/105`, `33/105`, and `17/105` (rates
0.3619, 0.3143, and 0.1619), with five completed endpoint round trips. The
single-temperature chain remained predominantly in its initial well. The PT
low-state result is within 0.05 of the exact symmetry value, whereas the
single-temperature error is 0.45. This establishes improved mixing for this
finite campaign, not general equilibration.

Reproduction used the checked-in input with only the execution backend changed
to serial because MPI was unavailable locally:

```text
PT wall time:      16.72 s; max RSS 132876 KiB
single-T wall time: 2.67 s; max RSS 127756 KiB
```

Machine-readable values and checkpoint SHA-256 hashes are in
`double-well-summary.json`.

## Four-rank equality protocol

`tests/test_tempering_mpi.py` first writes a serial reference, launches

```text
mpiexec -n 4 python tests/mpi_tempering_smoke.py INPUT.json OUTPUT
```

with four temperature states and one trajectory group per rank, then compares
the complete ensemble checkpoint and every state trajectory array bit for bit.
The workers publish generation stage files and return only clocks, identities,
and reduced-energy scalars. The root validates and commits the same proposal
order and exchange RNG stream as serial execution.

## Known limits

- Fixed-cell, canonical legacy dynamics only; no barostat, nonlinear bond
  constraints, active learning, trajectory batching, specialized
  `[path_integral]`, or Torch/native ensemble engine.
- FES whitelist: one base umbrella state, or direct-hill metadynamics frozen
  before exchange. OPES, ABF/eABF, steered sampling, shared/multiple-walker
  metadynamics, static metadynamics walls, and adaptive umbrellas are rejected.
- HREMD targets the full PES, the FES bias, or a `SplitPotential` slow term.
  Slow-term scaling currently requires one bead worker.
- A strict subset of replica stage files is rejected rather than treated as a
  smaller ensemble. A complete staged generation can be authenticated and
  committed on resume.
- No REST/solute-tempering implementation is included.
