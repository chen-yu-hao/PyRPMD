# Direct electronic-structure data and nonadiabatic rates

PyRPMD can propagate local adiabatic electronic data without constructing a
global diabatic potential. The new APIs are `AdiabaticPythonPotential`,
`QDNRPMDDynamics`, `AdiabaticRPSHDynamics`, and two-state `MASHDynamics`.
The separate `pyrpmd nonadiabatic-rate` command connects the existing signed
thermal NRPMD sampler to its analytical population-flux estimator.

An executable direct-data example runs all three dynamics methods against an
analytical provider:

```bash
python examples/nonadiabatic/adiabatic_on_the_fly.py --method qd-nrpmd --beads 4
python examples/nonadiabatic/adiabatic_on_the_fly.py --method rpsh
python examples/nonadiabatic/adiabatic_on_the_fly.py --method mash
```

## Electronic-structure provider contract

An external program or Python quantum-chemistry calculation supplies a callback
with the following interface:

```python
from pyrpmd.adiabatic import AdiabaticPythonPotential, AdiabaticResult

class ElectronicStructure:
    def evaluate(self, q, *, previous_q=None, previous_context=None):
        # Run your electronic-structure program at each geometry in q.
        # previous_context may contain its previous wavefunction objects.
        energies, gradients, nacs, overlaps, wavefunctions = calculate(
            q, previous_q, previous_context
        )
        return AdiabaticResult(
            energies, gradients, nacs, overlaps, context=wavefunctions
        )

provider = AdiabaticPythonPotential(ElectronicStructure())
```

`calculate` is the user's electronic-structure bridge, not a bundled solver.
The interface makes an actual calculation at each requested geometry. It does
not reconstruct a molecular PES from prerecorded populations, silently compute
NACs by finite differences, or claim that a particular quantum-chemistry package
has been validated. A process-local opaque context lets the bridge reuse its
wavefunctions without the dynamics engine inspecting them.

| Quantity | Shape | Definition and default units |
| --- | --- | --- |
| `q`, `previous_q` | `[B,N,3]` | Nuclear coordinates, bohr |
| `energies` | `[B,S]` | Adiabatic energies, hartree |
| `gradients` | `[B,S,N,3]` | Positive energy gradients, hartree/bohr |
| `couplings` | `[B,S,S,N,3]` | Derivative couplings (d_{ij}=\langle i|\nabla j\rangle), bohr⁻¹ |
| `overlaps` | `[B,S,S]` | (O_{ij}=\langle i(q_\mathrm{old})|j(q_\mathrm{new})\rangle), dimensionless |
| `status` | `[B]` | Optional nonnegative integers; any nonzero frame fails |
| `context` | Arbitrary | Previous wavefunction handle or provider-owned context |

The first call has no previous geometry/context and needs no overlap. A later
overlap must connect precisely the supplied previous frame to the returned
frame. Provider calls for the ring beads and centroid are distinct; do not infer
the previous frame from a global “last evaluation” variable. This matters for
parallel trajectories and for a ring with more than one bead.

`units={"length": "angstrom", "energy": "eV"}` converts both current and previous
coordinates, energies, gradients and inverse-length NACs consistently. Energies
and gradients are real; NACs must be anti-Hermitian in their electronic indices.
Result arrays are detached from provider buffers and stored read-only. Failed
calculations, nonfinite values and malformed arrays raise before state commit.
The provider remains responsible for its own external side effects.

Overlap transport uses (c_\mathrm{new}=O^\dagger c_\mathrm{old}). A polar
decomposition removes roundoff-sized loss of orthogonality; the default rejects
singular-value deviations larger than `overlap_tolerance=1e-6`. Increasing that
tolerance is an explicit retained-subspace approximation. Significant norm loss
usually calls for a smaller timestep or more electronic states.

If overlaps are unavailable, `continuous_gauge=True` explicitly permits a
second-order endpoint NAC connection approximation. The provider must then keep
electronic phases consistent. The default refuses this approximation without
the declaration. Genuine overlaps correctly transport independent sign changes;
the tests exercise such sign changes during a trajectory.

## QD-NRPMD

```python
from pyrpmd.adiabatic import QDNRPMDDynamics
from pyrpmd.dynamics import State

dynamics = QDNRPMDDynamics(
    provider, masses, beta, nbeads, dt, nstates, native=False
)
state = dynamics.initialize(State(q, p), initial_state=0, seed=13)
for _ in range(steps):
    dynamics.step(state)
    populations = dynamics.populations(state)
```

The mapping variables are stored in each bead's current adiabatic basis. Each
step applies a half MInt flow, a free-ring flow, overlap transport, and a second
half MInt flow. The force matrix in that local basis includes

\[
G_{ii}=\nabla E_i,\qquad G_{ij}=(E_j-E_i)d_{ij}\quad(i\ne j).
\]

Dropping the off-diagonal terms would give the wrong mapping force even though
the energy matrix is diagonal. For a complete electronic subspace and exact
overlaps, changing bases at the free-ring endpoints is algebraically equivalent
to the existing fixed-diabatic MInt integrator. Tests compare coordinates,
momenta, transformed mapping amplitudes and Hamiltonians for 60 steps and obtain
agreement at floating-point precision. Other tests compare the reconstructed
operator gradients against independent finite differences.

This is a local-basis implementation of focused-state NRPMD. Its populations are
signed MMST estimators and are never clipped. They refer to the instantaneous
adiabatic labels. It does not prepare an SEO thermal ensemble, and arbitrary
focused-state trajectories do not define a thermal rate. The local-basis
construction follows the quasi-diabatic propagation idea of
[Mandal, Yamijala and Huo, JCTC (2018)](https://doi.org/10.1021/acs.jctc.7b01178)
and [Zhou, Mandal and Huo, JPCL (2019)](https://doi.org/10.1021/acs.jpclett.9b02747).

## Direct adiabatic RPSH

`AdiabaticRPSHDynamics` uses energy-ordered surfaces, beadwise adiabatic nuclear
forces, and centroid electronic amplitudes. A phase-aligned overlap logarithm
provides average time-derivative couplings; energies are interpolated during
electronic substeps. Outgoing fewest-switches probabilities are integrated from
the population currents. A probability sum exceeding one fails transactionally
and requests a smaller timestep.

```python
from pyrpmd.adiabatic import AdiabaticRPSHDynamics

dynamics = AdiabaticRPSHDynamics(
    provider, masses, beta, nbeads, dt, nstates,
    seed=7, electronic_substeps=8, native=False,
    frustrated_hops="reverse",
)
state = dynamics.initialize(State(q, p), active_surface=0)
dynamics.step(state)
print(state.active_surface, dynamics.populations(state), dynamics.last_hop)
```

Hops shift every bead momentum by the same vector along the centroid NAC,
leaving internal ring momenta unchanged. The gap used for rescaling is the mean
bead energy gap, so the complete sum-bead Hamiltonian is conserved at the hop.
The `keep` and `reverse` frustrated-hop policies are explicit.

This implementation requires real NACs and nondegenerate, increasing-energy
surface labels. It rejects complex-SOC momentum rescaling and exact degeneracies
instead of choosing an arbitrary real direction. It has no implicit decoherence
correction. Saved electronic frames must match the nuclear coordinates; manually
changing q requires reinitialization. The QC basis context is process local and
is not a portable restart format.

## Two-state MASH

```python
from pyrpmd.mash import MASHDynamics

dynamics = MASHDynamics(provider, masses, beta, nbeads, dt, native=False)
state = dynamics.initialize(State(q, p), active_surface=0, seed=19)
dynamics.step(state)
```

MASH chooses the active surface from the spin hemisphere, equivalently the
larger of the two electronic amplitude populations. No random number is used
to decide a hop. Pure initial populations sample the corresponding hemisphere
with density proportional to (|S_z|). Accepted hops rescale along the NAC.
Frustrated hops reflect the nuclear NAC component and (S_z), preserving the
transverse spin components. `populations(state)` returns surface indicators;
the continuous coefficients remain available as `state.coefficients`.

These choices follow
[Mannouch and Richardson, JCP 158, 104111 (2023)](https://doi.org/10.1063/5.0139734).
The implementation switches at the end of a finite step. It rejects multiple
resolved hemisphere crossings within one step, and the timestep must converge
both the electronic motion and crossing dynamics. It does not implement the
reversible/event-localized algorithms of
[Geuther, Asnaashari and Richardson, JCTC 21, 2179 (2025)](https://doi.org/10.1021/acs.jctc.4c01684).

The [independent event comparison](https://github.com/chen-yu-hao/PyRPMD/blob/codex/tier2-review-fixes-20261101/docs/validation/pr63-independent-review/nonadiabatic_ml_report.md)
quantifies this limitation with constant adiabatic energies, a constant NAC,
and an analytically located single crossing. Across six timestep refinements,
the switched nuclear coordinate has **first-order error**, even though its
energy is conserved to the reported precision at every timestep. The error is
exactly the displacement caused by delaying the hop to the step endpoint.
The separate electronic propagator converges at second order on a prescribed
smooth nuclear path; that result does not establish second-order accuracy of
the complete trajectory across a hop. Production studies must therefore check
timestep dependence of populations and crossing observables directly.

`P=1` is the primary two-state implementation. `P>1` uses an explicitly
experimental centroid extension with ring-energy-conserving rescaling; it is
not advertised as a validated implementation of a particular published
RP-MASH formulation. Multistate and complex-SOC MASH are not included.

## Thermal electronic-rate CLI

```bash
pyrpmd nonadiabatic-rate examples/nonadiabatic/thermal_rate.inp \
    --output runs/thermal-nonadiabatic
```

The command reads TOML/`.inp`, JSON, or a trusted Python file whose `build()`
returns a mapping. Its small schema is independent of the existing gas-phase
rate input; all numeric physical quantities use atomic units. The example has
explicit masses, q, beta, bead/state counts, a `diabatic_python` factory, sampler
settings, propagation settings and reactant/product electronic-state indices.
Relative factory files resolve against the input's directory.

The workflow samples the absolute SEO-projected thermal distribution, retains
each parent's sign and electronic projector weight, and propagates the existing
scalar/traceless `ThermalNRPMDDynamics`. It evaluates the analytical derivative
of the electronic population correlation and divides signed numerator sums by
the signed reactant normalization. It does not fit an exponential to an
arbitrary excited-state trajectory.

Outputs include:

- `thermal-parents.npz`: every q/p and mapping sample, sign, initial electronic
  population estimator and sample index, with integrity checks.
- `population-rate.npz`: time, per-parent signed rate numerators, reactant
  weights, signs and population correlations.
- `results/summary.json`: sign and normalization checks, optional explicitly
  selected plateau diagnostics, and conditional block uncertainty.
- `manifest.json`: resolved input, provider-file digest, physics fingerprint
  and completion status.

Use `rate.plateau_start`, `rate.plateau_stop` and `rate.block_size` only when the
propagation interval and parent-correlation assumptions have been examined.
An unresolved sign or normalization leaves the rate unavailable. A candidate
plateau is distinct from a converged chemical rate; `formal_rate_s_inv` remains
null and `physical_converged` remains false. Repeat independent chains and check
time, bead count, timestep, state definitions and block-size sensitivity.

This rate workflow uses the established real-diabatic SEO thermal implementation
described in [the thermal NRPMD guide](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/thermal-nrpmd.md). A thermal on-the-fly
QD ensemble, multistate MASH and portable QC wavefunction restart remain future
work. The bundled example is an executable model smoke check, not a molecular
rate benchmark.
