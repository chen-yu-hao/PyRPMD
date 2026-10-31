# Experimental nonadiabatic collisions: NRPMD

NRPMD is an experimental extension of `task="drpmd"`, **not a new task**. It adds electronic mapping
variables to every nuclear bead, propagates electrons and nuclei together on the full diabatic
potential-energy matrix, and combines this with the existing chemical product classifiers to estimate
electronic-state-resolved contributions.

The current problem is a collision launched from a specified incoming diabatic electronic state that
is decoupled at long range. It does not build an electronic thermal-equilibrium ensemble and provides
no nonadiabatic PMF or Bennett–Chandler equilibrium rate. The initial-state, impact-parameter and
classification conventions of ordinary dRPMD still apply; see the [gas-phase workflows](workflows.md).

## Run the complete example

From the repository root, using a new output directory:

```bash
pyrpmd validate examples/nonadiabatic/exchange.inp --check-pes
pyrpmd run examples/nonadiabatic/exchange.inp --workers 2 --output runs/manual-nrpmd
pyrpmd resume runs/manual-nrpmd --block-steps 32
pyrpmd analyze runs/manual-nrpmd
```

The [complete input](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/nonadiabatic/exchange.inp)
uses four beads, two electronic states, two parent chains and eight fixed-collision-energy
trajectories. H + HD → H₂ + D is only a classification label; the
[matrix potential](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/nonadiabatic/matrix_pes.py)
is a teaching model with a non-zero local coupling, not a fitted real H₃ PES. A few trajectories may
yield no product, or remain unresolved within the time budget.

The example sets `collision.sampling="fixed_energy"` and, when the reporting conditions are met,
outputs cross sections in `bohr^2`, not thermal rates. Switching to `thermal_flux` and removing
`energy` gives a thermal collision distribution for the selected incoming electronic state; that is
still not an electronic thermal-equilibrium rate.

## Configuration and the matrix PES

Use the following potential and electronic settings in an ordinary dRPMD input:

```toml
[potential]
provider = "diabatic_python"
factory = "matrix_pes.py:make"
derivative = "gradient"

[nonadiabatic]
method = "nrpmd"
nstates = 2
initial_state = 0
state_labels = ["entrance", "excited"]
initialization = "focused"
asymptotic_coupling_tolerance = "1e-8 hartree"
```

`initial_state` starts at 0; labels refer to a fixed diabatic basis, not to adiabatic states sorted by
energy at each step. The factory returns an object with `evaluate(q)` or a callable, and the result
uses `pyrpmd.nonadiabatic.DiabaticResult(matrix, gradients)`:

| Data | Canonical shape and unit |
| --- | --- |
| Nuclear coordinates q | `[B,N,3]`, bohr |
| Full energy matrix | `[B,K,K]`, hartree |
| Positive gradient of the matrix | `[B,K,K,N,3]`, hartree/bohr |

B is the batch size of this call and K the number of electronic states. Both outputs must be real,
finite and symmetric in the electronic indices; every diagonal element and coupling needs a gradient.
This interface requires the **positive gradient**, unlike the negative-gradient force common for
scalar potentials. If the provider declares coordinate/energy units, the adapter converts them.

Put a common state-independent energy into every diagonal element; do not add a second scalar
potential on top, and do not multiply or divide by the bead number. Adiabatic energy gradients or
derivative couplings cannot be fed directly into this diabatic energy/gradient interface.
`validate --check-pes` checks the matrix and runs per-element finite-difference gradient diagnostics.

## Mapping variables and signed estimators

The nuclear variables are `q,p[P,N,3]` and the electronic variables `mapping_q,mapping_p[P,K]`.
Parents are sampled on the diagonal potential of the incoming state; each launch draws independent
phases for every bead and electronic state, and focused initialization sets the population of the
incoming state to 1 and the others to 0. The collision production stage is unthermostatted.

With electronic variables x and y, the state-population estimator of a trajectory is

$$
n_s(t)=\frac1P\sum_b\frac{x_{bs}(t)^2+y_{bs}(t)^2-1}{2}.
$$

The total should equal 1 within rounding, but a single state estimate may be below 0 or above 1. It is
not a probability assigning one surface to a single trajectory: do not clip it to `[0,1]` and do not
replace it with argmax.

The joint contribution of chemical channel α and electronic state s is `w * I_alpha * n_s`,
normalized by the total number of original launched trajectories; the sum over electronic states
recovers the chemical contribution. All channels share the parent bootstrap draws to preserve their
joint covariance; signed estimators do not use binomial/Wilson intervals.

## Stopping conditions, output and restart

At launch, the off-diagonal couplings of all beads must be below `asymptotic_coupling_tolerance`.
Even after geometric separation, a product or a returning reactant is accepted only when the
electronic decoupling condition holds; otherwise the trajectory continues to the original time
budget. The coupling threshold, separation distance and propagation time all need convergence checks.

| Output | What to check |
| --- | --- |
| `nonadiabatic` in `results/summary.json` | `final_estimates_available`, state labels, population-sum error, channel results |
| `results/electronic_channels.csv` | Signed observed contributions and any available thermal rates/cross sections |
| `nonadiabatic.channels` | `signed_observed_contributions`, `rate_cm3_molecule_s` or `cross_section_bohr2` |
| `parent_cluster_bootstrap_covariance` | Joint error information ordered by `covariance_columns` |

When failed/unresolved trajectories exist, no final electronic rate or cross section is published,
but the observed contributions are kept; unresolved signed contributions cannot use ordinary 0–1
probability bounds. `experimental=true` and `physically_validated=false` state the model limits
explicitly.

Checkpoints store the nuclear and mapping arrays, clocks, classification persistence state and RNG. A
failed step rolls back both electronic and nuclear state. `resume` recovers interrupted work and does
not extend unresolved trajectories that already reached the original time budget; changing the
maximum collision time needs a new calculation.

## Current limits

Matrix propagation uses CPU NumPy, either serially or with local trajectory processes; the nuclear
free ring can use the existing compiled kernels. MPI, bead-worker splitting, GPU, fused native
electronic propagation, periodic/fixed-atom systems and general complex Hermitian spin–orbit coupling
are not supported. Use `auto` or `python` for `loop_backend`.

Separate singlet/triplet PESs cannot produce intersystem crossing without an appropriate coupling.
Real applications need a justified diabatic basis, the full matrix and its gradients, and validation
of the asymptotic states, energy and mapping norm, bead number, time step, parent correlation,
collision parameters and product commitment. The full method and sources are in the
[NRPMD topic](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/nonadiabatic.md) and the
[example notes](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/nonadiabatic/README.md).
