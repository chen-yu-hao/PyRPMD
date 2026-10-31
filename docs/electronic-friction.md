# Electronic friction for surface RPMD

`ElectronicFrictionDynamics` adds a physical electronic heat bath to a scalar
or electronic mean-force potential. It implements **Markovian centroid
EF-RPMD**: a bead-averaged friction tensor acts on the centroid, and each bead
receives the same random momentum increment. The existing real-mass nuclear
integrators, fixed-atom handling, potential adapters and diffusion checkpoints
are reused.

The method choice follows Bi and Dou, *J. Chem. Phys.* **160**, 074110 (2024),
[DOI: 10.1063/5.0187646](https://doi.org/10.1063/5.0187646), Sec. II C.
This approximation is intended for a rapidly relaxing electronic environment.
It omits the bath-induced change to the imaginary-time distribution and the
memory effects addressed by Trenins and Rossi, *Phys. Rev. Lett.* **134**,
226201 (2025), [DOI: 10.1103/PhysRevLett.134.226201](https://doi.org/10.1103/PhysRevLett.134.226201).
Consequently, this implementation does not predict the friction-induced
suppression of tunnelling described by the latter theory. A full path-integral
GLE is a distinct future method, not an alternative switch in this interface.

## Run the fixed-slab example

From the repository root:

```bash
pyrpmd validate examples/nonadiabatic/friction_surface.inp --check-pes
pyrpmd run examples/nonadiabatic/friction_surface.inp --output runs/friction-surface
pyrpmd resume runs/friction-surface
pyrpmd analyze runs/friction-surface --recompute
```

The example has eight beads, one mobile H and four fixed substrate atoms in a
periodic analytic corrugation. Its 100 fs production time and illustrative
friction rate are software demonstrations, not a calibrated Cu diffusion
prediction. The output retains `physical_converged=false`.

Add this section to a `task="diffusion"` input:

```toml
[electronic_friction]
method = "centroid"
provider = "constant"
gamma = "0.01 fs^-1"
```

`gamma` is a relaxation rate. Scalar values apply to all Cartesian components;
arrays `[N,3]` or `[3*N]` specify diagonal component rates. Numeric entries use
inverse atomic time; strings carry explicit units. The physical tensor is
`eta = diag(mass * gamma)`. For a nondiagonal constant tensor, replace `gamma`
with `tensor = [[...], ...]` in **electron mass / atomic time**, with Cartesian
order `x0,y0,z0,x1,y1,z1,...`. Exactly one of `gamma` and `tensor` is required.

Equilibration still uses the declared nuclear sampling thermostat on the same
PES. The production phase applies the electronic bath and does not apply that
sampling thermostat. Initial sampling, trajectories, observations and random
number state use the established restart mechanism. Different friction
parameters change the scientific checkpoint identity.

This section is currently accepted only by `diffusion`. Other protocols use
the Python propagator below. It is not silently applied to an existing
Bennett–Chandler workflow, whose NVE diagnostics and rate normalization have
different assumptions. Serial, trajectory-process and MPI replica scheduling
use the ordinary materials workflow; electronic friction orchestration runs
in Python on the trajectory worker. `loop_backend="native"` is rejected.

## Python interface and spatially varying tensors

```python
from pyrpmd import State, ElectronicFrictionDynamics, ConstantElectronicFriction

friction = ConstantElectronicFriction.from_rates("0.01 fs^-1", system.masses)
dynamics = ElectronicFrictionDynamics(
    potential, system.masses, beta, nbeads=8, dt=4.0,
    friction=friction, seed=123,
    fixed_atoms=system.fixed_atoms, fixed_positions=system.geometry,
)
state = State(initial_q, initial_p)
dynamics.step(state)
result = dynamics.propagate(state, 20, sample_interval=5)
```

`potential` and `friction` remain caller-owned. The usual scalar potential
must already contain the appropriate electronic mean-force surface. Adding a
friction tensor does not construct a metal PES or an electronic free energy.

A Python provider implements:

```python
from pyrpmd import FrictionResult

class ElectronicTensor:
    metadata = {"supports_pbc": True}

    def bind_system(self, system):
        self.cell = system.cell.copy()
        self.pbc = tuple(system.pbc)

    def evaluate(self, q):
        # q[B,N,3] in bohr; evaluate eta at every bead geometry.
        eta = model_friction_tensor(q, self.cell, self.pbc)
        return FrictionResult(eta)  # eta[B,3*N,3*N], mass / atomic time
```

Use `provider="python"`, `factory="friction.py:make"` and optional `params`
in the input. The factory is called once for each replica. A periodic
provider must explicitly implement `bind_system` and declare `supports_pbc`.
An optional `close()` releases its resources.

Every evaluated tensor must be finite, symmetric and positive semidefinite.
The full tensor is validated before extracting its mobile-atom principal
submatrix; negative damping and antisymmetric/Berry-force tensors are not
accepted. `FrictionResult` can also carry per-frame status and messages;
failed electronic calculations stop the step. The constructor, unit conversion,
matrix averaging and OU update never replace an invalid tensor with zero.

## Noise normalization and integrator

Let $\bar p=P^{-1}\sum_b p_b$, $M$ be the diagonal Cartesian mass matrix,
and $\bar\eta=P^{-1}\sum_b\eta(q_b)$. In mass-weighted centroid momenta
$u=M^{-1/2}\bar p$, define $\Gamma=M^{-1/2}\bar\eta M^{-1/2}$. For a frozen
tensor over duration $h$, the update is

$$
u' = e^{-\Gamma h}u + \xi,\qquad
\langle\xi\xi^T\rangle=\beta^{-1}(I-e^{-2\Gamma h}).
$$

The implementation diagonalizes the symmetric $\Gamma$, uses `expm1` for
small damping, and adds $M^{1/2}(u'-u)$ to **every** bead. Thus the stationary
centroid momentum covariance is $M/\beta$, independently of bead count.
Internal-mode momenta are unchanged by this update. Generating independent
bead kicks at the physical temperature would give the wrong centroid noise.

The complete step is `O/2 – B/2 – free ring – B/2 – O/2`. The middle three
operations use the existing Cayley, exact-normal-mode, or stable Verlet
integrator. Tensor evaluation occurs at each OU endpoint. At zero friction,
the trajectory is exactly the ordinary RPMD trajectory and consumes no noise.
Fixed coordinates stay identical on every bead, with exactly zero momenta.
Failed force/tensor/observation calls restore both the state and the RNG for
the affected `step` or transactional `propagate` block.

`sample_step`, `thermalize` and the deterministic `propagate_block` API are
not used on this class. They reject calls that could accidentally replace or
combine the physical bath with a sampling thermostat. Prepare configurations
with a separate `Dynamics` sampler, then use `step` or `propagate`.

## Interpreting results

The diffusion output identifies `phase="centroid_ef_rpmd_production"` and
records the tensor/noise convention. Its `nve_energy_diagnostics` is null:
energy exchange with the electronic bath is physical. Quantitative checks
include timestep convergence, independent replicas, the fluctuation–dissipation
relation, relaxation spectra and independent references. A constant input
rate is a model parameter; it is not a computed first-principles friction.

The tests independently verify the OU drift/covariance using a matrix
exponential in physical momenta, bead-independent centroid temperature,
zero-friction trajectories, mobile/fixed degree counting, and exact restart.
They establish implementation consistency rather than real-surface accuracy.
