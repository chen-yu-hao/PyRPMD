# Adaptive reaction-path pilot (mean-force string)

This optional low-level API does **not** yet implement a coupled
constant-potential nonadiabatic RPMD workflow. It is intentionally separated
from physical rate reporting.

`adaptive_mean_force_string(images, mean_force_gradient)` relaxes a string
orthogonally to its tangent, redistributes images by arc length, and inserts
images at bends. The callback must yield a converged **free-energy mean-force
gradient** in the same CV coordinates. An ordinary potential-energy gradient
produces only a minimum-energy path. For nonlinear CVs one must handle the
metric, Jacobian and constrained ensemble carefully.

Existing components relevant to future integration:

- `pyrpmd.electrochemistry`: adiabatic per-bead electronic-number
  minimization and shared-number grand-canonical equilibrium sampling;
- `pyrpmd.thermal_nrpmd`: SEO-projected equilibrium mapping preparation and
  approximate nonadiabatic real-time flux/population rates;
- `pyrpmd.workflows.adaptive_umbrella`: window adaptation/MBAR diagnostics.

**Do not** sum a separately minimized electrode grand potential onto a
diabatic NRPMD matrix without deriving the joint electronic Hamiltonian:
this would risk electron double counting and wrong forces. A physically
consistent model must specify the electrode reservoir charge, diabatic
electron transfer states, charging energy, and their derivatives together.

## Next validation gates

1. Check a two-dimensional analytically integrable potential of mean force
   and gradients (including CV metric), then compare with umbrella+MBAR.
2. Check the constant-potential stationary condition `dE/dN = mu` and
   force finite differences under a model with known convex charging energy.
3. Derive and test a grand-canonical nonadiabatic partition function and
   electronic-state populations against direct quantum diagonalization
   for a two-state spin-boson/Marcus model.
4. Compare real-time rates at each voltage against a numerically exact
   HEOM/MCTDH result using the **identical** bath, discretization,
   temperature and driving force; report uncertainties and convergence.
5. Only then consider the combined workflow validated. No published
   numerical rate can be compared before the model parameters are pinned.

Primary methods: Richardson & Thoss, JCP **139**, 031102 (2013),
doi:10.1063/1.4816124; Maragliano et al., JCP **125**, 024106 (2006),
doi:10.1063/1.2212942; Reed et al., JCP **126**, 084704 (2007),
doi:10.1063/1.2464084; Xia & Xiao, JCTC **19**, 5168 (2023),
doi:10.1021/acs.jctc.3c00237.
