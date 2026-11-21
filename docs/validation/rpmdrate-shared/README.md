# Shared RPMDrate coordinate validation — 2026-11-08

## Delivered implementation

Native definition and workflow interfaces were read at PyRPMD main
`ffa14f247e453e9626b2e3fb3c288da0ca753d2e`. New opt-in modules add:

- An adapter to the same native xi and RateSurface, including centroid semantics,
  prepared-TS binding, derivative/mass checks and coordinate fingerprints.
- A locally invertible full Cartesian chart with xi as the exact first axis.
- Explicit constrained 1--3D kinetic/PES Hamiltonian construction and finite-basis
  Kubo flux-side propagation with the SAME native side projector.
- A matched-block normalized active quantum/RPMD defect and conditional bootstrap.
- Native `run_rate` orchestration without replacing default PMF, parent or child code.

The correction keeps the native QTST factor but does not claim the quantum
active defect is exact for a coupled molecular environment. The default finite
slice model fixes omitted coordinates; a validated dynamic-bath backend remains
an explicit scientific requirement.

## Executed tests

```
45 passed, 1 skipped
```

The skipped test requires native `pyrpmd.cv`, unavailable in this isolated
patch-only environment. Native source interfaces were verified via the GitHub
connector, but a complete repository clone/install was unavailable. The actual
native coordinate parity test and native PMF/recrossing production execution
must be run in the target checkout. Orchestration was tested with explicit mocks,
including a changed/prepared system and preservation of native data on failure.

The local coordinate oracle is an independent implementation of the documented
formula, used only by tests and the explicitly flagged validation fallback.
Production modules always use the native coordinate and do not import the oracle.

## Numerical quantum kernel calculation

Physical potential: `V(x)=sech(x)^2`; beta=4, moving physical mass=1,
Dirichlet box x in [-12,12], 4-point Gauss cell integration. The same physical
Hamiltonian was represented in the nonlinear RPMDrate xi. The calculation did
not use a PMF as a quantum potential. The full scientific input is
`benchmarks/rpmdrate_shared/inputs/barrier.json`.

| Nodes | Finite-time Q rate, window [2,4] | Analytic scattering integral / finite-box Qr | Relative difference |
|---:|---:|---:|---:|
| 161 | 0.00133800494 | 0.00137135077 | -2.4316% |
| 321 | 0.00133980905 | 0.00137526890 | -2.5784% |
| 481 | 0.00134029320 | 0.00137589889 | -2.5878% |

Rates in this table are inverse atomic time for the finite-box 1D reference,
**not** cm^3 molecule^-1 s^-1. The analytic reference independently integrates
Eckart transmission over thermal collision energy. Reactant partition values
and the complete finite-time curves are retained.

The [2,4] window was declared before inspecting the result. Later fixed-window
sensitivity checks give -0.5252% for [3,5], -0.2709% for [4,6], and -0.3155% for
[5,7] on the finest grid. They do not replace the primary result or establish an
irreversible plateau in a closed finite basis. The [4,6] curve still varies by
about 3.05% peak-to-peak relative to the analytic reference.

The scalar consistency check is <=8.89e-16; the quantum Cfs(0) is zero. The
independent 1024 P=1 positive-flux launches give the expected no-recrossing
control. Neither check is a chemical-rate or a quantum/RPMD bath validation.

## Not done / publication status

- No full native molecular PMF/recrossing job was run in this environment.
- No general dynamic quantum-bath influence model is generated automatically.
- No molecular accuracy improvement or physically converged rate is claimed.
- `physical_converged=False` and `rate_cm3_s=None` remain deliberate outputs.
- No remote PR or CI run was created. The session's connector exposes read
  actions only; repository push permissions do not add a write tool. A complete
  patch and a guarded publish script are included for a write-enabled checkout.

## Reproduction

See `benchmarks/rpmdrate_shared/README.md`. Exact local commands:

```
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_rpmdrate_shared.py
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python benchmarks/rpmdrate_shared/barrier_validation.py --formula-oracle
python -m compileall -q src tests benchmarks
```

In the real PyRPMD checkout, remove `--formula-oracle` and run the actual native
parity/integration checks before considering merge.
