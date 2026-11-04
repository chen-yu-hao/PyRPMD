# T26 auxiliary-operator validation — 2026-11-03

## Scope

This record covers the Python auxiliary-degree-of-freedom protocol and the NRPMD adapter. It does not claim physical convergence, production accuracy, GPU performance, NPT support, or validation of other nonadiabatic methods.

## Environment

- Host: Linux x86_64 container, kernel 6.18.57
- CPU presented to the container: AMD EPYC 9V74, 5 logical CPUs
- Python: 3.13.5
- Validation thread settings: `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`
- Baseline commit: `ceee59cd5f702165b56c5e541a503d8810fb9111`

## Independent numerical checks

`tests/test_auxiliary_operators.py` contains the following references:

1. **Legacy map reference:** 10,000 NRPMD steps are propagated twice from identical arrays. One path uses the new auxiliary core; the other spells out the former `_mint(dt/2)`, free-ring, `_mint(dt/2)` loop. Nuclear coordinates, momenta, both mapping arrays, clock, and final total energy are required to be bitwise equal.
2. **Periodic force reference:** a periodic analytic two-state Hamiltonian is compared with a centered finite difference of the auxiliary energy and with a full lattice translation.
3. **RPC/MTS reference:** an analytic harmonic split potential with one-bead contraction is propagated for 3,000 steps and checked for bounded total-energy error.
4. **Preparation reference:** Andersen momenta are compared with the analytic variance \(mP/\beta\); Andersen and PILE-L both verify bitwise-frozen mapping arrays.
5. **Persistence reference:** version-2 operator state restores RNG and mapping arrays; a legacy RNG-only state remains readable.
6. **Fail-fast reference:** NPT is rejected before PES calls, provider cell updates, or output-directory creation.

## Commands

The final pull-request description records the exact output of every command actually executed. The required commands are:

```bash
python -m pip install -e '.[test]'
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q -m "not campaign"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q tests -k "nrpmd or nonadiabatic or propagation_stack or engine"
python -m pyrpmd run examples/nonadiabatic/exchange.inp --output runs/t26-exchange
python -m pip install -e '.[docs]'
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

No command is represented here as passing until its raw result has been captured. The connected execution environment initially lacked direct Git transport, so validation is delegated to the branch CI when local source materialization is unavailable.

## Interpretation

Passing these tests establishes implementation-level equivalence and the stated analytic checks. It does not establish convergence in bead count, timestep, trajectory count, electronic-state truncation, or model fidelity. Any result produced by these paths must retain conservative `physical_converged=false` semantics unless a separate convergence study supports a stronger claim.
