# Nonadiabatic model Hamiltonians

The standard model library supplies the same full-matrix potential contract to
NRPMD, RPSH and the other model dynamics APIs. It includes Tully's three
two-state models, a general linear vibronic-coupling (LVC) model, and an explicit
finite spin-boson bath. These are model Hamiltonians, not fitted molecular
potentials or a selection of dynamics and initial conditions.

```python
import numpy as np
from pyrpmd.potentials.nonadiabatic_models import TullyModelOne

model = TullyModelOne()
q = np.zeros((8, model.natoms, 3))
q[:, 0, 0] = np.linspace(-4.0, 4.0, len(q))
result = model.evaluate(q)
assert result.matrix.shape == (8, 2, 2)
assert result.gradients.shape == (8, 2, 2, 1, 3)
```

## Coordinates, units and the matrix contract

`evaluate(q)` accepts a finite real array `[B,N,3]` and returns
`DiabaticResult(matrix, gradients)`. `matrix` has shape `[B,S,S]` and
`gradients` has shape `[B,S,S,N,3]`. Gradients are **positive** derivatives
`dV/dq`, including derivatives of every coupling. They are not nuclear forces
or adiabatic derivative couplings.

All energies are hartree, positions bohr, masses electron masses and time atomic
time, with `hbar=1`. The returned matrix is the complete diabatic potential,
including any common harmonic energy on every diagonal. Do not add that scalar
energy again or rescale the matrix by the number of beads.

An active coordinate index is `3*atom + axis`, where axis `0,1,2` denotes x,y,z.
For example, indices `[0,4]` select atom 0's x and atom 1's y. `natoms` is the
exact expected number of atoms, including any inactive atoms. LVC and spin-boson
infer the smallest compatible atom count when `natoms=None`.

These are physical Cartesian coordinate conventions. A mass-weighted normal
coordinate or a dimensionless vibrational coordinate must be converted together
with its coefficients, reference position and kinetic energy before use.
Changing only the coordinate values does not perform this transformation.

**Inactive coordinates are free spectator coordinates:** every matrix
derivative with respect to them is zero. A reduced-dimensional propagation can
initialize all their bead positions and momenta to zero, which leaves them at
zero. For a canonical configurational distribution, unbounded free spectators
are not normalizable. A sampler must explicitly restrict the model to its
active coordinates or supply a physically defined confinement; the generic
thermal NRPMD sampler does not infer a reduced coordinate measure from these
model objects.

## Tully models I–III

The defaults are the original atomic-unit parameters in Tully (1990),
Eqs. (21), (23) and (24) [1]. State labels below use zero-based array indices.

| Class | Model | Parameters |
| --- | --- | --- |
| `TullyModelOne` | Simple avoided crossing | `a=0.01, b=1.6, c=0.005, d=1.0` |
| `TullyModelTwo` | Dual avoided crossing | `a=0.10, b=0.28, c=0.015, d=0.06, e0=0.05` |
| `TullyModelThree` | Extended coupling with reflection | `a=0.0006, b=0.10, c=0.90` |

For Model I:

$$
V_{00}=a\,\operatorname{sgn}(x)(1-e^{-b|x|}),\quad
V_{11}=-V_{00},\quad V_{01}=V_{10}=ce^{-dx^2}.
$$

For Model II:

$$
V_{00}=0,\quad V_{11}=-ae^{-bx^2}+e_0,\quad
V_{01}=V_{10}=ce^{-dx^2}.
$$

For Model III:

$$
V_{00}=a,\quad V_{11}=-a,\quad
V_{01}=V_{10}=
\begin{cases}be^{cx},&x<0,\\b(2-e^{-cx}),&x\ge0.\end{cases}
$$

The first derivatives are evaluated continuously at zero. By default `x` is
`q[:,0,0]`. Every parameter is a keyword; for example
`TullyModelTwo(c=0.02, coordinate=4, natoms=2)` embeds the model along atom 1's y.
The potential does not set a nuclear mass. Tully's calculations used a nuclear
mass of 2000 electron masses [1]; pass the intended mass explicitly to the dynamics.

Model III retains coupling `2*b` as `x` tends to positive infinity. Its
asymptotic adiabatic and diabatic states therefore differ. A scattering stopping
rule that requires all diabatic off-diagonal elements to vanish is inappropriate
for its transmitted branch. Use a model-specific stopping rule and a clearly
specified observation basis when designing scattering benchmarks.

## General linear vibronic coupling

```python
from pyrpmd.potentials.nonadiabatic_models import LinearVibronicCoupling

h0 = np.array([[0.0, 0.02j], [-0.02j, 0.10]])
linear = np.array([
    [[0.03, 0.0], [0.0, -0.03]],
    [[0.0, 0.01], [0.01, 0.0]],
])
model = LinearVibronicCoupling(
    h0, linear,
    harmonic=[0.04, 0.08],
    reference=[0.0, 1.0],
    coordinates=[0, 4],
    natoms=2,
)
```

For `D` active coordinates and `S` electronic states, define
`Q[d] = q.flat[coordinates[d]] - reference[d]` separately for each batch frame.
The implemented potential is

$$
V(Q)=H_0+\sum_{d=1}^{D}G_d Q_d
+\frac12\sum_{d=1}^{D}k_dQ_d^2 I_S.
$$

| Parameter | Shape | Meaning |
| --- | --- | --- |
| `h0` | `[S,S]` | Constant real symmetric or complex Hermitian matrix, hartree |
| `linear` | `[D,S,S]` | Hermitian matrix derivatives `G[d]`, hartree/bohr |
| `harmonic` | `[D]` | Nonnegative common force constants `k[d]`, hartree/bohr²; default zero |
| `reference` | `[D]` | Coordinate origins, bohr; default zero |
| `coordinates` | D distinct indices | Active Cartesian coordinates; default `0,1,...,D-1` |
| `natoms` | Positive integer or `None` | Exact atom count or inference from the active indices |

`D` and `S` are positive integers. A coordinate-independent matrix is expressed
with at least one all-zero linear matrix and zero harmonic force constants.
The model supports common quadratic confinement, diagonal linear shifts, and
off-diagonal linear couplings. It does not include state-specific quadratic
terms or a nonlinear coordinate transformation.

`harmonic` contains force constants, not frequencies. For a physical coordinate
with mass `m` and angular frequency `omega`, use `k=m*omega**2`. The caller must
use the same physical masses in the kinetic-energy propagator.

Hermiticity is checked with an absolute tolerance of `1e-12` hartree or
hartree/bohr, respectively. Only differences within that tolerance are
symmetrized. Large common diagonal offsets do not relax this check. Complex
Hermitian models are supported by the SOC-capable NRPMD propagator; APIs that
require real matrices, including the current RPSH and thermal NRPMD interfaces,
retain that restriction.

## Explicit discrete spin-boson bath

```python
from pyrpmd.potentials.nonadiabatic_models import DiscreteSpinBoson

bath = DiscreteSpinBoson(
    frequencies=[0.01, 0.03],
    couplings=[0.002, -0.003],
    masses=[1.0, 2.0],
    bias=0.005,
    tunnelling=0.002,
)
assert bath.coordinates == (0, 3)
assert bath.natoms == 2
# Use bath.nuclear_masses, not the D-dimensional mode-mass array, in Dynamics.
```

The convention is

$$
U(Q)=\frac12\sum_d m_d\omega_d^2Q_d^2,\qquad
V(Q)=
\begin{pmatrix}
U(Q)+\epsilon+\sum_d c_dQ_d & \Delta\\
\Delta^* & U(Q)-\epsilon-\sum_d c_dQ_d
\end{pmatrix}.
$$

`bias=epsilon` is **half** the diabatic energy gap at `Q=0`.
`tunnelling=Delta` is the off-diagonal matrix element, with no extra factor
of one half. It may be complex. The `frequencies`, `couplings` and mode
`masses` arrays all have length `D`; masses default to one. Frequencies and
masses must be strictly positive. Couplings are real and may have either sign.
No energy counterterm, reorganization-energy shift or spectral-density
discretization is inserted.

The default embedding gives each oscillator its own artificial atom and uses
that atom's x coordinate: `coordinates=[0,3,...,3*(D-1)]`, `natoms=D`.
`nuclear_masses[N]` is provided for the existing Cartesian dynamics API.
For an explicit embedding, any modes on different axes of the same atom must
have the same mass, because the dynamics gives each atom one Cartesian mass.
Unused atoms have mass one in this convenience array. For example, two modes
of equal mass can be embedded in one atom with `coordinates=[0,1]`.

The diagonal surfaces have minima at
`Q[d] = -couplings[d]/(masses[d]*frequencies[d]**2)` for state 0 and the
opposite displacement for state 1. Their minimum energies include the common
lowering `-sum(couplings**2/(2*masses*frequencies**2))`. These identities make
the energy convention and absence of a hidden counterterm directly checkable.

An explicit finite bath is not an automatically converged representation of an
Ohmic, Debye or other continuum bath. Comparisons must record the frequencies,
couplings, masses, initial bath preparation, electronic state and observation
time. The model does not create an initial thermal distribution.

## Stable model identities

Model parameters are copied on construction and stored in immutable arrays.
Scalar parameters and the coordinate layout cannot be reassigned. `to_dict()`
returns a detached, versioned identity containing all numerical model
parameters. Complex values are stored as separate real and imaginary data.

```python
import json
from pyrpmd.potentials.nonadiabatic_models import model_from_dict

encoded = json.dumps(bath.to_dict(), allow_nan=False)
restored = model_from_dict(json.loads(encoded))
assert restored.to_dict() == bath.to_dict()
```

`model_from_dict` accepts only these five builtin model names and schema
version 1. Unknown or missing keys and versions are rejected; model restoration
does not import a Python factory or execute serialized code. The ordinary
`metadata` property contains the same identity under `model_identity`, together
with the provider's units, state count and atom count.

The model identity covers the potential. It does not cover a complete run:
temperature, nuclear masses, initial conditions, integrator, timestep,
electronic preparation, RNG state and observations remain part of the dynamics
and workflow configuration.

## Checks and scope

```bash
python -m pytest tests/test_nonadiabatic_models.py -q
```

The checks compare every matrix gradient with independent finite differences,
including complex off-diagonal derivatives and arbitrary active-coordinate
embeddings. They also test published Tully values and asymptotes, unitary basis
covariance, the scalar LVC limit, spin-boson minima and mass conventions,
immutability, strict coordinate validation and JSON reconstruction.

These checks establish model and interface consistency. A dynamics benchmark
also needs converged initial sampling, time integration, beads or bath size,
trajectory statistics, and an independently specified quantum reference where
available. Agreement of an integrator with a local identity is not a validation
of a nonadiabatic rate or real molecular reaction.

## Reference

1. J. C. Tully, “Molecular dynamics with electronic transitions,”
   *J. Chem. Phys.* **93**, 1061–1071 (1990),
   [doi:10.1063/1.459170](https://doi.org/10.1063/1.459170).
   The original models and default parameters appear in Eqs. (21), (23), (24);
   the nuclear mass is specified at the start of Section IV.
