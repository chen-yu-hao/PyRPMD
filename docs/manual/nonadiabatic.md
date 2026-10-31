# Experimental nonadiabatic collisions: NRPMD

NRPMD is an experimental mode of `task="drpmd"`. It adds electronic mapping variables to every
nuclear bead, propagates electrons and nuclei together on the full diabatic potential-energy matrix,
and combines the electronic populations with the chemical product classifiers to give
electronic-state-resolved contributions.

Collisions start from a chosen incoming diabatic electronic state that is decoupled at long range.
Initial conditions, impact parameters and classification follow ordinary dRPMD; see the
[gas-phase workflows](workflows.md).

## Run the example

From the repository root:

```bash
pyrpmd validate examples/nonadiabatic/exchange.inp --check-pes
pyrpmd run examples/nonadiabatic/exchange.inp --workers 2 --output runs/manual-nrpmd
pyrpmd resume runs/manual-nrpmd --block-steps 32
pyrpmd analyze runs/manual-nrpmd
```

The [input](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/nonadiabatic/exchange.inp)
runs four beads, two electronic states, two parent chains and eight trajectories at a fixed
collision energy. The [matrix potential](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/nonadiabatic/matrix_pes.py)
is an artificial two-state model with a local coupling; channels are labeled as H + HD → H₂ + D.

With `collision.sampling="fixed_energy"`, the result is a cross section in `bohr^2`. Switching to
`thermal_flux` and removing `energy` gives a thermal collision distribution from the selected
incoming state.

## Configuration and the matrix PES

Add the following to an ordinary dRPMD input:

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

`initial_state` is 0-based, and labels name states of the fixed diabatic basis. The factory returns
an object with `evaluate(q)`, or a callable, that returns
`pyrpmd.nonadiabatic.DiabaticResult(matrix, gradients)`:

| Data | Shape and unit |
| --- | --- |
| Nuclear coordinates q | `[B,N,3]`, bohr |
| Energy matrix | `[B,K,K]`, hartree |
| Gradient of the matrix | `[B,K,K,N,3]`, hartree/bohr |

B is the batch size and K the number of electronic states. Both arrays are real, finite and
symmetric in the electronic indices, with gradients for every diagonal element and coupling. The
matrix interface takes **positive gradients** (scalar potentials return forces). If the provider
declares coordinate or energy units, the adapter converts them.

Put any state-independent energy on every diagonal element; the matrix is the complete potential.
`validate --check-pes` checks the matrix and finite-differences each element's gradient.

## Mapping variables and signed estimators

Nuclei carry `q,p[P,N,3]` and electrons `mapping_q,mapping_p[P,K]`. Parents are sampled on the
diagonal potential of the incoming state. Each launch draws independent phases for every bead and
state; focused initialization gives the incoming state population 1 and the others 0. Collisions
run without a thermostat.

With electronic variables x and y, the population estimator of a trajectory is

$$
n_s(t)=\frac1P\sum_b\frac{x_{bs}(t)^2+y_{bs}(t)^2-1}{2}.
$$

The populations sum to 1 within rounding; individual values can fall outside $[0,1]$. Averages over
trajectories give the state-resolved contributions.

The joint contribution of chemical channel α and electronic state s is `w * I_alpha * n_s`,
normalized by the total number of launched trajectories; summing over states recovers the chemical
contribution. All channels share the parent bootstrap draws, which gives their joint covariance.

## Stopping, output and restart

At launch, the off-diagonal couplings of every bead must be below `asymptotic_coupling_tolerance`. A
product or returning reactant counts as resolved when it is both geometrically separated and
electronically decoupled; otherwise the trajectory runs to the time budget. Converge the coupling
threshold, separation distance and propagation time.

| Output | Contents |
| --- | --- |
| `nonadiabatic` in `results/summary.json` | `final_estimates_available`, state labels, population-sum error, channel results |
| `results/electronic_channels.csv` | Signed observed contributions and thermal rates or cross sections |
| `nonadiabatic.channels` | `signed_observed_contributions`, `rate_cm3_molecule_s` or `cross_section_bohr2` |
| `parent_cluster_bootstrap_covariance` | Joint error information ordered by `covariance_columns` |

If any trajectory failed or remains unresolved, final electronic rates or cross sections are left
empty and the observed contributions are still reported. Summaries carry `experimental=true` and
`physically_validated=false`.

Checkpoints store the nuclear and mapping arrays, clocks, classification persistence and RNG state,
and a failed step rolls back electrons and nuclei together. `resume` finishes interrupted work within
the original time budget; a longer maximum collision time needs a new run.

## Supported configurations

NRPMD runs on CPU with NumPy, serially or with local trajectory processes; the nuclear free ring can
use the compiled kernels. Use `loop_backend="auto"` or `"python"`. MPI, bead-worker splitting, GPU,
fused native electronic propagation, periodic or fixed-atom systems and complex Hermitian spin–orbit
coupling are not available in this mode.

For a real system, provide a diabatic basis with the full matrix and its gradients, including the
couplings between states, and converge the asymptotic states, the energy and mapping norm, the bead
number, the time step, parent correlation, collision parameters and product commitment. Method and
sources are in the [NRPMD topic](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/nonadiabatic.md)
and the [example notes](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/nonadiabatic/README.md).
