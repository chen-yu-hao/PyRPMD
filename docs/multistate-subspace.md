# Coupled electronic states in a quantum subspace

**English** · [中文](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/zh/multistate-subspace.md)

> **Status: research.** Python API (`pyrpmd.analysis.multistate_subspace`) and, for the 1D
> path model, the `[electronic]` table of `subspace-correct` and
> `[rate.subspace_correction.electronic]` (section 3). The exact rate is exact only for the
> supplied subspace Hamiltonian; how well a 1–3D subspace with $S$ retained states represents a
> molecule must be checked case by case.

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

## 3. Conical intersections along a reaction path: `subspace-correct`

Add an `[electronic]` table to the `subspace-correct` TOML (or
`[rate.subspace_correction.electronic]` to a `rate` input). The pipeline then computes
$\eta_\mathrm{NA}$ for its 1D path model and applies $C_S\,\eta_\mathrm{NA}$:

```toml
[source]
rate_run = "rate_output"

[subspace]
dimension = 1                    # coupled states are implemented for the 1D path model

[electronic]
source = "backend"               # "backend", "python" or "file"
object = "es_backend.py:make"    # factory returning an ExcitedStateBackend
arguments = { basis = "6-31g*" } # keyword arguments of the factory
nstates = 2                      # retained adiabatic states, ground state included
state = 0                        # adiabat of the RPMD rate run
points = 64                      # electronic geometries along the path ("grid": every model point)
```

**Construction.** The geometries are placed evenly in arc length on the *same* mass-weighted
path spline as the 1D model $V_S^{(0)}(s)$. The source returns adiabatic energies and
adjacent-frame overlaps at each geometry. `propagation_diabatization`, anchored at the reactant
end, turns them into $V_d(s)$, which is splined onto the model grid. Each grid point is then
shifted by $V_S^{(0)}(s) - E_\mathrm{state}(s)$, so adiabat `state` of the coupled model **is**
the single-surface model of $C_S$. The electronic structure contributes only the gaps and the
state mixing, and every electronic state shares the transverse (zero-point) term of
$V_S^{(0)}$. Two consequences follow:

- The Born–Oppenheimer denominator of $\eta_\mathrm{NA}$ equals the quantum numerator of $C_S$.
  `born_oppenheimer_consistency` reports the relative difference, which is at round-off level.
- Systematic errors of the electronic-structure method in the absolute ground-state energy do
  not enter $\eta_\mathrm{NA}$; only its gaps and couplings do.

**Sources.**

| `source` | `object` / `file` | Returns per geometry |
| --- | --- | --- |
| `"backend"` | an `ExcitedStateBackend` class or factory, called with `arguments` (PySCF TDA/TDDFT, ORCA, Q-Chem, OpenMolcas, `FileProgramBackend`, the toy backend) | `BackendResult.energies` and `.overlap` (or `backend.overlap(...)`); geometries are passed as `[atoms, 3]` bohr |
| `"python"` | a callable `f(geometry, **arguments)`; the geometry is the flat coordinate vector of the rate run | `{"energies": [S], "vectors": [basis, S]}` (orthonormal state coefficients in a fixed basis, any sign and order) or `{"diabatic": [S, S]}` |
| `"file"` | an `.npz` with `geometries`, `energies` and `overlaps` (or `diabatic`) | read as is; if the file is missing or its geometries differ, the pipeline writes `electronic_1d_request.npz` (`arcs`, `geometries`) and stops with status `electronic_data_required` |

Choose an electronic-structure method that describes the states that meet. Linear-response TDDFT from the ground state gives the wrong dimensionality for S0/S1 intersections. Use a multireference method (for example the OpenMolcas backend) or spin-flip TDDFT for those, and keep the retained states the same along the whole path.

The `file` source is meant for expensive calculations: run once to obtain the request, compute
the electronic structure elsewhere at exactly those geometries, then run again. `backend` and
`python` results are cached in `electronic_1d.npz` and reused with `output.reuse = true` when the
geometries and the source identity match. A relative `file.py:object` or `file` is resolved
against the directory of the TOML specification.

**Gates** (policy `"reject"` or `"allow"`, set in `[electronic]`):

| Gate | Fails when | Default |
| --- | --- | --- |
| `electronic_leakage` | the smallest singular value of an adjacent-frame overlap is below `min_singular_value` (0.9): states leave the retained subspace | `leakage = "reject"` |
| `electronic_resolution` | a cubic spline through every other electronic point misses the omitted points by more than `max_interpolation_error_kt` (0.1 kT) | `resolution = "reject"` |
| `ground_state_mismatch` | adiabat `state` differs in shape from the rate PES by more than `max_ground_state_mismatch_kt` (1 kT) within `mismatch_window_kt` (30 kT) of the path minimum | `ground_state_mismatch = "allow"` (recorded) |

When `[electronic]` is present, only dimensions that have both a valid $C_S$ and a valid
$\eta_\mathrm{NA}$ can be applied. 2D models are recorded as `unsupported`. The report
`subspace_pipeline.json` adds `nonadiabatic` and `applied_factors` (`single_surface`,
`nonadiabatic`, `total`). The other outputs are `multistate_1d.npz` (the coupled-state grid) and
`rate_quantum_corrected.json`, whose `correction.diagnostics` keep both factors.

**What a 1D path model captures.** The ground-state path passes the intersection at some
distance and sees an avoided crossing there. The 1D model treats Landau–Zener-like transitions
driven by the derivative coupling along the path, the diagonal Born–Oppenheimer correction, and
tunnelling through the crossing region. It does **not** represent derivative couplings
perpendicular to the path, geometric-phase effects of encircling the intersection, or
transverse frequencies that differ between electronic states. If the thermally relevant paths
pass close to the intersection compared with the transverse zero-point width, the 1D factor is
only a first estimate.

## 4. Minimal Python example

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

## 5. Diabatic models from electronic-structure data

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

## 6. Diagnostics and convergence

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

## 7. Validation in the test suite

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

`tests/test_subspace_electronic.py` runs the pipeline on a 2D model whose diabats cross with a
coupling $\lambda(y-y_c)$, which puts a conical intersection 2.5 transverse widths from the
ground-state path. It checks:

- the electronic geometries reproduce the 1D model's own potential to $10^{-14}$;
- with `points = "grid"`, $\eta_\mathrm{NA}$ from scrambled adiabatic data equals the factor
  built from the exact diabatic matrix to $10^{-9}$. With 64 splined points the difference is
  below $5\times10^{-4}$, and the Born–Oppenheimer consistency is at round-off level;
- $\eta_\mathrm{NA}$ rises towards one as the gap opens: 0.06, 0.21 and 0.85 at 300 K for a
  coupling of one half, one and four times the reference coupling;
- the electronic cache is reused together with a reused $C_S$;
- the `file` source writes the geometry request and then reproduces the `python` result;
- leaking overlaps reject the factor unless `leakage = "allow"`;
- the toy excited-state backend (with a forced phase flip) is diabatized exactly through the
  `backend` source.

## 8. Not yet available

- **2D coupled-state models in the pipeline.** These need three things: a plane containing the
  path tangent and the coupling (branching) vector at the closest approach to the intersection;
  2D diabatization; and an iterative route for grids beyond the dense limit (a Chebyshev
  Boltzmann operator with matrix split-operator propagation).
- State-dependent transverse terms, for example zero-point energies on each diabat.
- Reactant electronic-state-selected rates. Rates are thermal over the modelled states.
- Bound, condensed-phase charge transfer. This needs a dissipative environment rather than
  absorbing boundaries, for example a multistate extension of the DVR-HEOM reference in
  `pyrpmd.analysis.dvr_heom`. Compare with the golden-rule, Marcus and thermal NRPMD tools
  listed in [Direct nonadiabatic data and rates](tier2-nonadiabatic.md).
