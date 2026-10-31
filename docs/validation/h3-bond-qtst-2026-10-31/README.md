# H + H2: physical comparison of two-bond RPMD PMF and independent 1D rate

This branch **does not** claim a converged 2D bimolecular rate. The existing
1500 K, 16-bead BKMP2 PMF is a coordinate-probability density
F(r01,r12), not RPMDrate's flux-weighted F(xi). The finite 1.15–5.55 bohr
umbrella region does not contain a correctly normalized free-reactant basin.

## Independently calculated 1D reference

At 1500 K, 16 beads, 0.1 fs, Rinf=10 bohr, real BKMP2 PES:

| Quantity | Candidate value |
| --- | ---: |
| 1D QTST | 3.564611014987932e-11 cm^3 molecule^-1 s^-1 |
| signed RPMD recrossing kappa | 0.9163598330163885 |
| 1D total rate | 3.26646635446272e-11 cm^3 molecule^-1 s^-1 |
| bootstrap standard error | 9.486504725957814e-12 cm^3 molecule^-1 s^-1 |

Independent [H3 rate GitHub Actions](https://github.com/chen-yu-hao/PyRPMD/actions/runs/41175919954)
with raw archive artifact `h3-rate`. Its source is **not converged**.
The new [4x two-bond PMF](https://github.com/chen-yu-hao/PyRPMD/actions/runs/41192380353)
has 307,200 saved CV samples and 1,888 supported cells on its 44x44 grid.
The PMF itself is **not converged**.

## Implemented and checked

- `bond_flux.bond_diagonal_flux_metric(q,masses,beta)` computes the exact
  projected inverse mass and half-Maxwell normal speed for centroid
  sigma=r01-r12, **including the bond angle**, without a spurious bead factor.
  Local mass metric does not imply its full conditional average along the curve.
- `bond_rate.bond_2d_qtst(...)` calls the existing generic 2D curve
  integral **only if** the same-ensemble physical entrance flux, complete
  reference and target sections, identity, and their conditional positive
  normal speed are supplied. These cannot be obtained from the current
  finite PMF alone.
- `bond_rate.bond_2d_signed_rate(...)` combines **matched** signed,
  parent-grouped RPMD recrossing data with the corresponding normalized
  2D TST factor, using the existing joint-channel bootstrap. It does not
  generate new trajectories or substitute the old xi transmission factor.
- Analytic checks cover exact line-length/velocity normalization, source
  identity, gradient finite differences, angle dependence, bead-number
  invariance, and signed two-channel conservation.

The input-dependent missing physical 2D rate is recorded as `null`, not
as zero. The independent 1D benchmark `xi` dividing surface includes both
equivalent exits, while `r01=r12` alone does not. The third bond r02,
full product commitment, orientation and missing entrance bridge matter.

## Remaining work before a genuine numerical 2D-vs-1D rate comparison

1. Sample or calibrate an unbiased, **same-ensemble** thermodynamic bridge
   between the 10-bohr free entrance flux and the *complete* bond-coordinate
   dividing surface, preserving Jacobians and both atom permutations.
2. Sample constrained **canonical** RPMD parents along that complete surface
   and obtain the **conditional** normal-speed average. Existing endpoint
   checkpoints are not canonical constrained samples.
3. Propagate new positive/negative momentum children for that exact surface
   with **no umbrella or thermostat**, retaining signed flux, return,
   both product channels, unresolved outcomes and chain grouping.
4. Validate time plateaus, PMF bootstrap covariance, adequate independent
   parents, step-size, bead and entrance-radius convergence before reporting
   2D k(T) or a relative deviation from the independent 1D calculation.

Do not spend extra sampling solely on grey nonreactive bins. However,
a physically necessary section carrying flux must have full support;
a rate surface cannot skip masked pieces and still be called complete.
