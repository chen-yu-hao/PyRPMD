# Scientific concepts, units and convergence

This chapter defines the quantities PyRPMD computes, the units and array conventions, and the
procedure for converging a result. The inputs and commands for each task are in the
[workflow guide](workflows.md), [Materials and surfaces](periodic.md) and [PCET](pcet.md).

## Tasks and their results

| Goal | Task | Result and normalization |
| --- | --- | --- |
| Thermal rate of two gas-phase reactants | `rate` | PMF, transmission coefficient and bimolecular rate in `cm^3 molecule^-1 s^-1` |
| Explicit collisions with product classification | `drpmd` | Thermal rate or fixed-energy cross section, set by the launch distribution |
| Equilibrium free energy along chosen CVs | `fes` | Probability density per CV volume element, and its free energy |
| Separate rates for several entrances and TSs | `network` | Rate matrix indexed by entrance, channel and TS |
| History-dependent bias on 1D–3D CVs | `metadynamics` | Finite-time bias free-energy estimate and HILLS |
| Probability-based enhanced sampling | `opes` | Adaptive probability bias and optional frozen-bias reweighting |
| Equilibrium nuclear quantum statistics at fixed volume or pressure | `pimd` | Physical energy estimators, ring radius of gyration; cells and volumes for NPT |
| RPMD transport, optionally with electronic friction | `diffusion` | Centroid MSD, VACF, diffusion tensor and diagnostics |
| A selected exit from a material site or surface basin | `site_rate` / `surface_rate` | Basin-normalized first-order rate in `s^-1` |
| Local electrochemical proton–electron transfer | `pcet` | Reduction/oxidation rates and isotope effects in `s^-1` |

`pcet` solves one-dimensional proton eigenstates and evaluates a golden-rule rate; it has no RPMD
propagation. All other dynamics tasks share one ring-polymer convention with real atomic masses.
`network` reports each TS separately (`aggregation="separate"`).

## Atomic units and array conventions

PyRPMD computes in atomic units. The low-level Python API takes numbers in atomic units; input files
also accept strings with units, which are dimension-checked.

| Quantity | Internal unit / shape | Recommended input |
| --- | --- | --- |
| Positions, cell | bohr; `q[P,N,3]`, `cell[3,3]` | `geometry_unit="angstrom"`, `cell_unit="angstrom"` |
| Energy | hartree | `"0.7 eV"`, `"1 kcal/mol"` |
| Force | hartree/bohr, the negative gradient of the physical potential | PES output `[B,N,3]` |
| Mass | electron masses | `masses_amu` or isotope masses with `amu` |
| Time | atomic time | `"0.1 fs"`, `"2 ps"` |
| Temperature | K on input; internally `beta=1/(k_B T)` | `temperature="300 K"` |
| Momentum | atomic units, `p[P,N,3]` | Sampled from mass and temperature |
| PES batch | coordinates `[B,N,3]`, energies `[B]` | B is the number of frames per call |
| D-dimensional CV | `value[D]`, `jacobian[D,P,N,3]` | Custom CVs return the full ring-coordinate Jacobian |

Atom indices start at 0. XYZ coordinates default to angstrom and inline coordinates to bohr. Each
cell row is a lattice vector, with `r = s @ cell`; see [Periodic structures](periodic.md).

A potential returns the **physical potential energy and forces** of each bead. The propagator adds
the ring springs, the bead scaling and any bias. Units, error handling and the optional virial are
specified in the
[potential interface specification](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/potentials.md) *(partly Chinese)*.

## The ring-polymer Hamiltonian

For P beads and N atoms PyRPMD uses the summed Hamiltonian

$$
H_P=\sum_{j=0}^{P-1}\left[
\sum_i\frac{|p_{ji}|^2}{2m_i}+V(q_j)
+\sum_i\frac{m_i\omega_P^2}{2}|q_{ji}-q_{j-1,i}|^2
\right],\qquad \omega_P=\frac{P}{\beta}.
$$

With $\hbar=1$ the equilibrium distribution is $\exp[-(\beta/P)H_P]$, so one momentum component
has Maxwell variance $m_iP/\beta$. For P=1 the springs vanish and deterministic propagation is
classical molecular dynamics. Increasing P converges the path-integral discretization.

The centroid is $\bar q_i=P^{-1}\sum_j q_{ji}$. For a nonlinear function s, the two estimators

$$
s_{\mathrm{centroid}}=s(\bar q),\qquad
s_{\mathrm{bead\ average}}=\frac{1}{P}\sum_j s(q_j)
$$

generally differ; choose `estimator="centroid"` or `"bead_average"` for the quantity you need. The
Jacobian of a centroid CV contains the factor $1/P$.

For a physical umbrella $W(s)=\tfrac12(s-c)^TK(s-c)$, the summed Hamiltonian gains $P W$ and the
bias force on the ring is $-P J^T K(s-c)$. The reduced MBAR bias of the generic FES is therefore
$\beta W$. When CVs of different dimensions are combined, give the centers and stiffness matrix in
matching units.

## Sampling and real-time propagation

PIMD, umbrella windows and constrained parent chains sample equilibrium distributions with a
thermostat. Ordinary dRPMD collisions, RPMD recrossing children and `diffusion` production use
deterministic propagation without thermostat or bias; their time is physical time. Diffusion can
instead select a physical Markovian electronic-friction bath through `[electronic_friction]`;
those trajectories exchange energy with the bath and are not judged by NVE energy conservation.

| Deterministic integrator | Free ring | Converge with respect to |
| --- | --- | --- |
| `cayley-bcb` | Cayley map (default) | Time step: physical-force and splitting error |
| `normal-mode-bab` | Exact normal-mode rotation | Time step: physical-force and splitting error |
| `velocity-verlet` | Explicit springs and forces | Time step, within the stability limit of the highest ring frequency |

Converge every method in the time step. For Verlet the free ring is stable for
$|\Delta t|\max\omega_k<2$; stiff bonds or biases tighten the limit. See the
[integrator notes](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/integrators.md).

Reaction and FES workflows default to Andersen sampling matched to the integrator. FES also supports PILE-L;
materials `pimd` / `diffusion` use PILE-L by default with Cayley or normal-mode dynamics:
`bcocb` with Cayley, `baoab` with the normal-mode integrator, and Andersen with Verlet.
Metadynamics additionally supports Nosé–Hoover chains. `native=true` enables compiled substeps;
`loop_backend="native"` requires the whole loop to run fused. The separate native PIMD engine
offers additional thermostats. The rules for periodic and fixed-atom systems are in the
[periodic task guide](periodic.md).

## Free-energy measures and rate normalization

The generic FES is a density free energy on the volume element of the chosen CV:

$$
F(s)=-\beta^{-1}\log\rho(s)+C.
$$

PyRPMD divides the histogram probability mass by the bin volume to get the density. A coordinate
transformation changes the density measure, so a distance FES includes its geometric degeneracy.

The gas-phase `rate` uses a dedicated PMF with the flux-mass metric and a bimolecular normalization
of two distant fragments. Local site and surface rates use a position density with an explicit basin
probability and give a first-order rate. Each rate uses the PMF, denominator and units of its own
workflow; see [gas-phase rates](workflows.md) and [basin rates](periodic.md).

`pimd` reports physical energy estimators. With all atoms mobile, the potential energy is the bead
average of the physical potential and the centroid-virial kinetic energy is

$$
K_{\mathrm{cv}}=\frac{3N}{2\beta}
-\frac{1}{2P}\sum_{ji}(q_{ji}-\bar q_i)\cdot F_{ji}.
$$

The internal energy estimate is $\langle V\rangle_P+K_{\mathrm{cv}}$. `ring_polymer_hamiltonian`
is reported as a propagation diagnostic. With fixed atoms, replace N by the number of mobile
atoms and restrict the kinetic-energy sum to them. The PES still includes every atom; fixed atoms
are external coordinates with identical bead positions and zero momenta. NPT requires all atoms
to be mobile.

## Converging a result

Build a result up in five layers and keep the complete input and output of every step:

1. **Model and interface.** Check elements, isotopes, masses, charge, spin and the PES definition,
   and finite-difference the forces. For periodic systems, also check the cell, periodic
   translations and partial PBC.
2. **Numerical parameters.** Vary the bead number, time step and grid/bin settings one at a time
   and compare the energy, free energy, rate or D you report.
3. **Equilibration and coverage.** Extend thermalization, add independent replicas and lengthen
   production; check slow CV modes, window overlap, effective sample sizes and the support of basins
   and dividing surfaces.
4. **Task-specific parameters.** Gas phase: `rinf` or the initial separation, `bmax`, classification
   and the maximum collision time. Local rates: the basin and the recrossing plateau. Diffusion: the
   long-time linear regime, reference frame, system size and fitting window. PCET: the coordinate
   domain, grid, number of states and the weak-coupling model.
5. **Statistical errors.** Use independent replicas, or block and parent bootstraps that keep the
   correlations of overlapping time origins, sibling trajectories from one parent and beads of one
   ring.

Error bars give the statistical uncertainty of the sampled problem. Missing errors are reported as
`null`, unsampled bins stay masked, and the bounds from unresolved dRPMD trajectories are reported
separately from confidence intervals.

### Assessing adaptive umbrella results

The [input defaults](input.md#defaults) are short starting budgets. In particular, P=32, a
0.1 fs time step, 1 ps adaptive production, ESS 20 and 20 bootstrap draws do not specify an
accuracy for an unknown reaction or material. `preset="production"` does not establish
convergence. Adaptive placement holds the umbrella stiffness fixed, refines the supplied domain,
then discards the pilot data and starts separately equilibrated production with a new seed.
It cannot discover unspecified basins or certify slow variables outside the selected CVs.

In `RUN/results/summary.json`, read `adaptive_umbrella` for FES or `pmf.adaptive_umbrella`
for gas-phase rates. Gas-phase runs also retain these diagnostics under `adaptive_umbrella`
in `RUN/rate_pmf.json` and `RUN/umbrella/summary.json`. Read the result in three stages:

1. **Pilot selection.** `adaptive_umbrella.pilot_target_met` means the requested local overlap,
   ESS and optional coverage criteria passed in the pilot. The older `target_met` field is a
   compatibility alias for this pilot result.
2. **Production sampling.** `production_target_met` checks those criteria again on the independent
   production streams. Inspect `production_overlap_connected_at_target`,
   `production_effective_samples_target_met` and `production_coverage`; omitted coverage criteria
   do not imply a completely sampled domain. A failed check is reported with warnings and does
   not automatically add sampling or remove the computed point estimate.
3. **Scientific convergence.** Compare independent seeds and initial states, longer equilibration
   and production, P versus 2P, and the time step versus half the time step. Refine windows and
   histogram bins, and check the reference-to-barrier interval, asymptotic separation and
   recrossing plateau where applicable. Accept differences only within a declared error budget.

For a gas-phase rate, the default `rate.pmf_method="ui"` fits Gaussian window distributions.
Local skewness, multiple modes and unsupported Gaussian tails can bias its PMF even when its
mean and variance appear stable. UI retains model estimates at endpoints without observed
histogram support and records a warning; this is not measured support. With
`rate.pmf_method="mbar"`, `compare_ui=true` (the default for MBAR) provides a paired comparison
on the same production samples. Histogram MBAR checks supported bins throughout the interval
from the reactant reference to the dividing surface, but also needs bin refinement and adequate
sampling. Agreement between estimators on one trajectory cannot expose a shared unsampled basin.
Generic `fes` uses MBAR with the CV-density measure; it is not interchangeable with the
`rate_flux` PMF of the gas-phase rate workflow.

Uncertainty is a separate check. Increase the number of independent samples and verify the
stability of the block-bootstrap interval; merely increasing the number of bootstrap draws
cannot repair insufficient sampling. Rate-PMF bootstrap draws that fail support or estimation
are counted as rejected. If any draw is rejected, the primary barrier/rate error and confidence
interval are unavailable (`null`, or NaN for curve arrays). `supported_draw_*` results describe
only the accepted subset, conditional on support and a valid fit; they are not unconditional
confidence intervals. Check requested, attempted, valid and rejected counts and block status.
Generic FES applies the same distinction per bin: its primary `standard_error` is NaN unless
all requested draws support that bin and correlated blocks are available.
`supported_draw_standard_error` and `bootstrap_valid_per_bin` retain the conditional estimate
and support count. The point-estimate mask and these uncertainty diagnostics answer different
questions; a finite error bar still does not prove physical convergence.

Choose an error target for the observable. For the static rate factor
$k_{\rm QTST}\propto\exp(-\beta\Delta F)$, an error magnitude $|\delta F|$ permits a multiplicative
rate deviation up to $\exp(\beta|\delta F|)$. Limiting that factor to 1.10 requires
$|\delta F|\le k_BT\log(1.10)$, about **0.0568 kcal/mol at 300 K**. This example allocates all
of a 10% rate budget to the barrier; transmission-coefficient, PES, time-step, bead and RPMD
approximation errors still need their own assessment. A universal default cannot supply this
budget for all systems.

Method background: [Kästner and Thiel (2006)](https://doi.org/10.1063/1.2206775) explains the
Gaussian UI approximation and correlated uncertainty propagation;
[Shirts and Chodera (2008)](https://doi.org/10.1063/1.2978177) derives MBAR;
[PyMBAR's timeseries guidance](https://pymbar.readthedocs.io/en/latest/timeseries.html) describes
equilibration and correlation diagnostics. The
[RPMDrate input examples](https://greengroup.mit.edu/rpmdrate-input/) are illustrative parameter
sets, not universal precision guarantees. See also the
[OpenMM RPMD guide](https://docs.openmm.org/latest/userguide/library/09_rpmd_plugin.html) for bead
convergence and the distinction between equilibrium path integrals and approximate real-time RPMD.

Summaries carry `converged` or `physical_converged`, which stay `false` for a single run; PCET also
reports `numerical_convergence` and `physical_applicability_unverified`. The
[scientific conventions](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/science.md) *(Chinese)*,
the [MBAR rate PMF](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/mbar-rate-pmf.md) *(Chinese)*
and the task topics give the theory, implementation and method sources.
