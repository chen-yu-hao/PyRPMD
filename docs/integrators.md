# Selectable RPMD integrators

The same real-mass ring-polymer Hamiltonian is used by every integrator:

\[
H_P=\sum_{j=0}^{P-1}\left[
\sum_a\frac{p_{ja}^2}{2m_a}+V(q_j)
+\sum_a\frac{m_a\omega_P^2}{2}|q_{ja}-q_{j+1,a}|^2\right],
\qquad\omega_P=P/\beta.
\]

Here units are atomic units, so \(\hbar=1\); the equilibrium measure is
\(\exp[-(\beta/P)H_P]\). No method rescales the physical force by `1/P`,
changes the atom masses, or thermostats a real-time collision/recrossing child.
The existing [scientific conventions](science.md) continue to apply.

## Configuration and public API

```toml
[dynamics]
integrator = "normal-mode-bab"
timestep = "0.05 fs"
loop_backend = "auto"
block_steps = 256

[sampling]
integrator = "normal-mode-bab-andersen"
thermostat = "andersen"
```

`dynamics.integrator` has three canonical values:

| Name | Complete deterministic time step | Free ring treatment |
| --- | --- | --- |
| `cayley-bcb` (default) | Physical half-kick, Cayley ring step, physical half-kick | Implicit-midpoint/Cayley map of kinetic plus spring Hamiltonian |
| `normal-mode-bab` | Physical half-kick, exact normal-mode ring step, physical half-kick | Exact harmonic rotation of every cyclic normal mode |
| `velocity-verlet` | Total-force half-kick, drift, total-force half-kick | Explicit physical plus ring-spring force |

`normalize_integrator()` accepts `cayley`, `normal_mode`, `velocity_verlet`
and documented short names `bcb`, `bab`, `vv`; it returns one of the canonical
names. BAOAB/BCOCB are sampling algorithms and are not accepted as real-time
integrator names.

```python
from pyrpmd.dynamics import Dynamics, State, native_capabilities

dynamics = Dynamics(potential, masses, beta, beads, dt,
                    integrator="normal-mode-bab", loop_backend="auto")
dynamics.step(state)                       # deterministic, no thermostat
block = dynamics.propagate(state, 100)      # whole-block transaction
prefix = dynamics.propagate_block(state, 100)  # accepted-prefix transaction
print(dynamics.integrator_name)
print(native_capabilities()["integrators"])
```

The established `CayleyIntegrator` class name remains available for compatibility;
its optional `scheme=` argument selects the actual free-ring map. Its `scheme`
property is read-only. Construct a new `Dynamics` object to change integrator,
time step, temperature or bead count.

## Mathematical maps and timestep choice

The normal-mode frequencies are
\(\omega_k=2(P/\beta)\sin(\pi k/P)\). An exact normal-mode free step of duration
\(h\) applies

\[
q_k'=\cos(\omega_k h)q_k+\frac{\sin(\omega_k h)}{m\omega_k}p_k,
\quad
p_k'=\cos(\omega_k h)p_k-m\omega_k\sin(\omega_k h)q_k.
\]

The zero-frequency centroid uses its continuous limit `q += h*p/m`, `p` unchanged.
Both odd and even bead counts, including the even-P Nyquist mode, are supported.
Exact integration of the free ring does not make the complete split integration
of a nonlinear physical PES exact. Physical-force errors and splitting resonances
still require timestep convergence tests.

For velocity Verlet, the total force contains

\[
F^{\mathrm{spring}}_{ja}
=-m_a(P/\beta)^2(2q_{ja}-q_{j-1,a}-q_{j+1,a}).
\]

Implementation keeps the physical half-kicks outside the free-ring helper;
inside it are spring half-kick, drift and spring half-kick. The two kicks at
the same position commute, giving exactly full-Hamiltonian velocity Verlet.
The `P=2` cyclic graph includes both neighbour bonds, while `P=1` has zero
spring force. All three integrators reduce to ordinary physical velocity
Verlet at `P=1`.

Velocity Verlet requires `abs(dt)*max(omega_k) < 2` even for a free ring.
Construction rejects a timestep at or above that spring stability boundary.
This is a necessary check only: a stiff physical PES or bias can require a
smaller timestep. Cayley and exact normal modes preserve the free-ring harmonic
energy for arbitrarily large finite steps in exact arithmetic; that property
does not remove the physical-PES timestep restriction.

The default remains Cayley BCB so an input which does not select a new
integrator retains its existing deterministic map.

## Native and Python execution

All three deterministic schemes are implemented in the C++17 context used by
both native block APIs, including their supported static biases, observations,
fragment projections and centroid-RATTLE constraints. The GIL is released for
the native block. A custom Python PES/CV/bias keeps its existing compatibility
path; choosing a different integrator does not bypass a Python callback.

The NumPy exact-normal-mode reference uses FFTs. The dependency-free C++ exact
normal-mode implementation precomputes circulant cosine/sine kernels and uses
`O(P^2*N)` operations per free-ring step, `O(P*N)` work space and `O(P)` kernel
storage. Cayley and explicit Verlet are `O(P*N)`. The native normal-mode option
is a correctness and capability choice; it is not advertised as faster than
Cayley or an FFT implementation at large P. Compare actual PES workloads before
choosing an integration method for performance.

`native="required"` requires the selected compiled integrator. An old extension
which only advertises Cayley cannot execute another scheme accidentally:
ordinary `auto` uses the matching Python reference and records the unavailable
native capability; strict native execution fails before propagation. Rebuild
the extension after updating source.

## Sampling and real-time dynamics

The production workflows use Andersen sampling with the selected deterministic
integrator. Randomisation is still scheduled on the continuous parent clock;
changing a block size cannot shift thermostat, observation or checkpoint events.
Collision and recrossing child production remains unthermostatted.

The low-level `sample_step()` additionally exposes two explicit PILE-L samplers:

| Dynamics integrator | `sample_step(method=...)` | Ring half maps |
| --- | --- | --- |
| `cayley-bcb` | `bcocb` | Square root of the full Cayley map, **not** Cayley at half the timestep |
| `normal-mode-bab` | `baoab` | Exact normal-mode rotations for half the timestep |
| `velocity-verlet` | Unsupported; use Andersen | No implicit substitution of another ring map |

Omitting `method` selects the matching sampler for Cayley or normal modes.
An explicit mismatch raises before changing the state. The OU step uses real
orthogonal normal modes, internal friction `gamma_k=2*omega_k`, user-provided
positive `gamma_centroid` in inverse atomic time, and variance `m*P/beta`.
These sampling APIs execute in Python/NumPy and reject `loop_backend="native"`;
their deterministic subkernels may still use C++. They are not an instruction
to thermostat physical RPMD correlation functions or reactive child trajectories.
On a failed stochastic step, both q/p/clock and the NumPy RNG state are restored.

The RATTLE position correction for a centroid surface has the same response
`dt/m` under all three deterministic maps: internal spring forces sum to zero,
and exact/Cayley centroid motion is a free drift. The existing old-normal
position solve and final new-normal momentum projection therefore apply without
changing the constraint measure. This does not claim support for arbitrary
per-bead holonomic constraints or a stochastic constrained BAOAB algorithm.

## Restart identity and validation

Integrator selection remains in the scientific configuration fingerprint;
`loop_backend` and `block_steps` alone are execution-only fields. Resuming a
trajectory with a different map must be rejected. Same-backend chunking and
Andersen split/resume preserve accepted states and RNG order. Different
floating-point backends are compared numerically rather than promised to be
bitwise identical on long chaotic trajectories.

`tests/test_integrators.py` checks against an independent dense Hamiltonian
matrix exponential, explicit graph-Laplacian forces, and a scalar Langevin
reference. Tests cover classical P=1, odd/even rings, second-order convergence,
time reversal, free-ring energy, the Verlet stability boundary, native/reference
block observations, exact chunking and continuous Andersen sampling, stale
extensions, strict native failures, and RNG rollback after a failed stochastic
force evaluation. These are numerical implementation tests, not a convergence
claim for a chemical rate coefficient.
