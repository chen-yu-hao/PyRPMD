# Coupled electronic states in a quantum subspace

**English** · [中文](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/zh/multistate-subspace.md)

> **Status: research.** Python API only (`pyrpmd.analysis.multistate_subspace`). The
> `subspace-correct` command and `[rate.subspace_correction]` still build single-surface
> models. The exact rate is exact only for the supplied subspace Hamiltonian; how well a
> 1–3D subspace with $S$ retained states represents a molecule must be checked case by case.

The [subspace rate correction](subspace-correction.md) treats one Born–Oppenheimer surface.
This page extends the exact half of that correction to reactions that pass through
**several coupled electronic states**: avoided crossings and conical intersections,
spin–orbit-mediated intersystem crossing (complex couplings), and charge-transfer crossings
along a reaction path.

## 1. Subspace Hamiltonian and exact rate

The model is a diabatic $S\times S$ Hermitian matrix on a 1–3D tensor sinc-DVR grid:

$$
H_S = \sum_a T_a \otimes \mathbb{1}_S + V(\mathbf{s}), \qquad
V_{ij}(\mathbf{s}) = V_{ji}^*(\mathbf{s}).
$$

The thermal rate follows Miller–Schwartz–Tromp with an absorbing potential
$\varepsilon(s_0)\otimes\mathbb{1}_S$ at both ends of axis 0, exactly as the single-surface
dense route:

$$
k\,Q_r = \operatorname{Re}\operatorname{Tr}[F_\beta X], \qquad
H^\dagger X - X H = iF, \qquad F = i[H_S, h], \qquad h = \theta(s_0 - s_0^\ddagger)\otimes\mathbb{1}_S .
$$

The dividing surface is nuclear and counts every electronic state. The flux absorbed at the
product end is resolved into electronic **channels** $c$ (adiabatic states, ordered by energy,
or diabatic states) with the identity

$$
\int_0^\infty e^{iH^\dagger t} F e^{-iHt}\,dt = -h + \sum_c Y_c, \qquad
H^\dagger Y_c - Y_c H = i B_c, \qquad B_c = 2\,\varepsilon_P^{1/2} P_c\, \varepsilon_P^{1/2},
$$

so $k_c Q_r = \operatorname{Re}\operatorname{Tr}[F_\beta Y_c]$. All channel solves share one
complex Schur form. The channel sum reproduces the total flux up to
$\operatorname{Tr}[F_\beta h]$; the residual is reported as `channel_sum_residual`.

$Q_r$ is thermal over **all modelled electronic states**: for `reactant = "asymptotic"` it is
$\sqrt{m_0/2\pi\beta}\,\operatorname{Tr}_{\perp,\mathrm{el}}\, e^{-\beta H_\perp(s_\mathrm{ref})}$,
and for `reactant = "well"` it is $\operatorname{Tr}[e^{-\beta H_S}(1-h)]$. If an asymptotic
electronic state that does not react is left out of the model (for example a spin component),
add its partition function yourself.

## 2. Using it with an RPMD rate

`nonadiabatic_subspace_factor` returns

$$
\eta_\mathrm{NA} = \frac{k_\mathrm{QM}(H_S^{\,\mathrm{multi}})}{k_\mathrm{QM}(H_S^{(0)})},
$$

where $H_S^{(0)}$ is the same subspace on the lowest adiabat (no diagonal Born–Oppenheimer
correction). $\eta_\mathrm{NA}$ contains nonadiabatic transitions, the diagonal correction,
geometric-phase effects around a conical intersection, and thermal population of the other
modelled states. Multiplying it into the existing single-surface correction gives

$$
k \approx k_\mathrm{RPMD}^{(0)}(\text{full}) \times
\frac{k_\mathrm{QM}(H_S^{(0)})}{k_\mathrm{RPMD}(H_S^{(0)})} \times \eta_\mathrm{NA}
= k_\mathrm{RPMD}^{(0)}(\text{full}) \times
\frac{k_\mathrm{QM}(H_S^{\,\mathrm{multi}})}{k_\mathrm{RPMD}(H_S^{(0)})} .
$$

The full-dimensional RPMD rate stays on the ground-state surface, so no multistate
ring-polymer method is needed. This **assumes** that the electronic coupling acts within the
subspace coordinates and that the remaining modes follow the lowest adiabat. Use the same
subspace coordinates, grid and absorber for $H_S^{\,\mathrm{multi}}$ as for the
single-surface correction. When the coupling depends strongly on a mode outside $S$, enlarge
$S$ rather than trusting the factorization.

## 3. Minimal example

```python
import numpy as np
from pyrpmd.analysis import (exact_multistate_rate, multistate_model_from_function,
                             nonadiabatic_subspace_factor)

def crossing(points, height=0.04, width=2.0, coupling=0.004):
    s = points[..., 0]
    out = np.zeros(s.shape + (2, 2))
    out[..., 0, 0] = 0.5 * height * (1 + np.tanh(s / width))   # reactant diabat
    out[..., 1, 1] = 0.5 * height * (1 - np.tanh(s / width))   # product diabat
    out[..., 0, 1] = out[..., 1, 0] = coupling * np.exp(-(s / 4.0) ** 2)
    return out

model = multistate_model_from_function(crossing, [np.linspace(-12, 12, 241)],
                                       masses=2000.0, dividing_surface=0.0)
beta = 1.0 / (1000.0 * 3.166811563e-6)
result = exact_multistate_rate(model, beta)        # channels="adiabatic" by default
print(result.rate, result.branching)               # total rate, product-channel branching
print(nonadiabatic_subspace_factor(model, beta)["factor"])
```

Complex couplings (for example spin–orbit matrix elements) are accepted as complex Hermitian
matrices. `model.adiabatic_surface(k)` and `model.diabatic_surface(i)` return ordinary
`SubspaceModel` objects for single-surface comparisons.

## 4. Diabatic models from electronic-structure data

Along a one-dimensional path, `propagation_diabatization(energies, overlaps)` builds $V(s)$
from adiabatic energies `energies[N, S]` and raw adjacent-frame overlaps
`overlaps[N-1, S, S]`, with $O_n[i,j] = \langle\phi_i(s_n)|\phi_j(s_{n+1})\rangle$. This is the
`BackendResult.overlap` convention of the
[excited-state interface](excited-state-interface.md): rows are in the previous frame's raw order
and columns in the current frame's raw order. Each overlap is replaced by its polar unitary
factor $W_n$, and the adiabatic-to-diabatic transformation is propagated as
$A_{n+1} = W_n^\dagger A_n$, anchored to $A=\mathbb{1}$ at frame `anchor`. Root reordering and
sign or phase changes are absorbed exactly. The result is exact for a complete retained
electronic subspace. A singular value below `min_singular_value` (default 0.9) is rejected,
because it signals leakage to states that were not retained or a step that is too large.
Anchor the transformation where the diabatic coupling should vanish, usually at the reactant
asymptote.

On 2D and 3D grids, the derivative coupling has a non-removable (curl) part, so propagation
along one path is not a diabatization. Supply a diabatic model directly instead, such as a
fitted linear or quadratic vibronic-coupling model or a regularized diabatization.

## 5. Diagnostics and convergence

| Diagnostic | Meaning / action |
| --- | --- |
| `channel_sum_residual` | Channel fluxes minus total flux minus `flux_side_t0`, relative to the flux; should be at round-off level |
| `flux_side_t0` | $\operatorname{Tr}[F_\beta h]$; zero for an exact thermal flux operator |
| `electronic.adiabatic_gaps` | Minimum adjacent adiabatic gap over the grid and on the dividing-surface line; a near-zero gap marks a conical intersection or seam that the grid must resolve |
| `minimum_adiabatic_gap_in_product_absorber`, `channel_resolution_ambiguous` | Degenerate adiabats inside the product absorber make individual adiabatic channel projectors arbitrary (the total is unaffected); use `channels="diabatic"` |
| `partition.reference_energy_levels` | Electronic and transverse levels in $Q_r$; check that the retained states cover the thermally populated reactant manifold |
| `solver_plan.dense_memory_mb` | Dense memory estimate; `MultistateRouteUnavailable` is raised above `dense_limit` basis functions |

Converge the grid spacing (the kinetic energy at the upper adiabats near a crossing can be
large), the box and the absorber (`cap_width`, `cap_strength`), and the **number of retained
electronic states**. `dense_limit` counts grid points times states. A two-state 2D grid is
limited to roughly $28\times28$ points at the default of 1600.

## 6. Validation in the test suite

`tests/test_multistate_subspace.py` checks:

- one state reproduces the single-surface dense rate to $10^{-12}$ (1D asymptotic, 1D well and 2D);
- uncoupled states add the single-surface fluxes, channel by channel;
- constant real and complex electronic rotations, and a constant coupling phase, leave the
  total rate and the adiabatic channel rates invariant;
- weak coupling gives golden-rule scaling, $k_{0\to1}\propto|V_{01}|^2$ (ratio $4.002$ for a
  doubled coupling);
- strong coupling converges to the exact rate on the lowest adiabat with its diagonal
  Born–Oppenheimer correction (deviation $1.0\times10^{-3}$ at a Massey parameter of about 6);
- a 2D model conical intersection is located by the gap diagnostic, and its rate is
  rotation-invariant with a closed channel sum;
- propagation diabatization of scrambled, sign-flipped adiabatic data recovers the diabatic
  model up to the anchor-frame rotation to $10^{-12}$, so the rate is unchanged.

## 7. Not yet available

- An iterative route (Chebyshev Boltzmann operator plus a matrix split-operator propagation)
  for coupled-state grids beyond the dense limit.
- Integration with `subspace-correct` and `[rate.subspace_correction]`: building the excited
  states along the reaction path from an excited-state provider, with gates on gaps and overlap
  completeness.
- Diabatization in 2D and 3D, and reactant electronic-state-selected rates. Rates are thermal
  over the modelled states.
- Bound, condensed-phase charge transfer. This needs a dissipative environment rather than
  absorbing boundaries, for example a multistate extension of the DVR-HEOM reference in
  `pyrpmd.analysis.dvr_heom`. Compare with the golden-rule, Marcus and thermal NRPMD tools
  listed in [Direct nonadiabatic data and rates](tier2-nonadiabatic.md).
