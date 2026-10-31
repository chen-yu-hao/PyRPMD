# Experimental nonadiabatic RPMD for direct collisions

PyRPMD supports a state-selected MMST mapping-variable NRPMD implementation for
`task = "drpmd"`. Each nuclear bead has its own electronic mapping coordinates.
The matrix potential couples their evolution to the nuclear motion. Existing
chemical outcome classifiers are combined with signed electronic population
estimators to resolve chemical and electronic contributions together.

This interface is experimental. Its analytic and model tests check implementation
and numerical behavior; they do not establish physical accuracy for a real
chemical reaction. A single chosen incident electronic state is prepared at a
decoupled reactant asymptote. This is **not an implementation of an electronic
thermal-equilibrium PMF or nonadiabatic Bennett–Chandler rate estimator**.
`rate`, `fes` and other equilibrium tasks reject this focused nonadiabatic configuration.
For electronic thermal preparation and finite-time rate correlations, use the
separate [thermal NRPMD API](thermal-nrpmd.md). Additional model methods are
described in [KC/MV/isomorphic RPMD](nonadiabatic-methods.md),
[complex SOC and RPSH](nonadiabatic-soc-rpsh.md), and
[electronic-friction dynamics](electronic-friction.md).

## Hamiltonian and conventions

All internal quantities are atomic units, including hbar = 1. With P nuclear
beads, K electronic states, real nuclear masses and beta = 1/(kBT), the propagated
sum-beads Hamiltonian is

\[
H_P=\sum_{b=1}^{P}\left[
\sum_a\frac{p_{ba}^2}{2m_a}
+\sum_a\frac{m_a\omega_P^2}{2}(q_{ba}-q_{b-1,a})^2
+\frac12\left(z_b^\dagger V(q_b)z_b-\mathrm{Tr}\,V(q_b)\right)
\right],\qquad \omega_P=P/\beta.
\]

`q` and `p` have shape `[P, natoms, 3]`; `mapping_q` (x) and `mapping_p` (y)
have shape `[P, K]`, with $z=x+iy$. The real Hamiltonian uses a Hermitian
matrix; the expression includes the imaginary-coupling cross term automatically.
The provider supplies the **full** diabatic energy matrix,
including any common state-independent energy on every diagonal. The code does
not add a second scalar potential or rescale its matrix by P. `energy()` reports
this propagated Hamiltonian, not a quantum thermodynamic internal-energy
estimator.

The method follows the NRPMD Hamiltonian of Richardson–Thoss [1]. Its use here
with state-selected initial populations follows the focused initialization and
population estimator described by Chowdhury–Huo [2]. Nuclear parent chains sample
the chosen diagonal potential using the existing constrained-fragment equilibrium
workflow. At each collision launch, independently uniform phases theta are drawn
for every bead and electronic state:

\[
x_{bs}=\sqrt{1+2\delta_{s,s_0}}\cos\theta_{bs},\qquad
y_{bs}=\sqrt{1+2\delta_{s,s_0}}\sin\theta_{bs}.
\]

Thus every bead starts with population one in `initial_state` and zero in the
other states. The trajectory estimator is

\[
n_s(t)=\frac1P\sum_b\frac{x_{bs}(t)^2+y_{bs}(t)^2-1}{2}.
\]

The per-bead total mapping norm is conserved by electronic rotation and implies
`sum(n_s) = 1`, up to numerical roundoff. Individual estimates can be negative or
greater than one. They are not probabilities for assigning one surface to one
trajectory. They are retained without clipping or `argmax` assignment. The
limitations of Cartesian mapping populations, including inverted effective
surfaces, require attention in physical applications [4].

## Propagation

One step consists of a half potential substep, a full free-ring substep, and a
half potential substep. The free-ring map is selected with the existing
`cayley-bcb`, `normal-mode-bab` or `velocity-verlet` setting. Production motion is
unthermostatted. Only parent sampling uses its configured thermostat.

The potential substep uses the momentum-integral (MInt) construction [3]. At
fixed nuclear coordinates, diagonalize the Hermitian matrix as
`V = U diag(lambda) U.conj().T`. For `z = x + i*y`, electronic evolution is exactly
`z(t) = exp(-i*V*t) z(0)`. Nuclear momentum must change during this rotation:

\[
\Delta p_a=-\frac12\int_0^h\mathrm{Re}[z(t)^\dagger
(\partial_a V)z(t)]\,dt+\frac h2\mathrm{Tr}(\partial_a V).
\]

The implementation integrates this force in the eigenbasis. For a gap
`d = lambda_i - lambda_j`, the integral factor is
`h * exp(i*d*h/2) * sinc(d*h/2)`, with mathematical `sinc(x)=sin(x)/x`.
This expression is continuous at degenerate eigenvalues. It avoids approximating
the force by a frozen electronic population. The symmetric composition is
reversible and second order; MInt supplies an exact potential Hamiltonian map.

A failed propagation step rolls back nuclear and electronic variables together.
Checkpoints include both mapping arrays and nuclear coordinates, momenta, clocks,
classification persistence and parent RNG state. Recovery requires the same
scientific configuration and provider dependencies. Changing the maximum collision
time is a different computation; ordinary `resume` does not extend a completed
`unresolved` trajectory.

## Matrix potential interface

A Python factory returns an object with `evaluate(q)` or a callable. The canonical
contract is

```python
from pyrpmd.nonadiabatic import DiabaticResult

# q: real coordinates [batch, natoms, 3], bohr
return DiabaticResult(
    matrix,       # [batch, K, K], hartree
    gradients,    # [batch, K, K, natoms, 3], positive dV/dq
)
```

Both arrays must be finite real or complex numerical arrays. The matrix and its
nuclear derivatives must be Hermitian in the two electronic indices, including
real diagonal elements. Every diagonal and coupling must
have its nuclear gradient; energies alone are insufficient. Do not provide
adiabatic energy gradients in this interface without an appropriate diabatic
transformation, and do not mix derivative couplings with diabatic off-diagonal
energies.

Minimal configuration:

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

State indices are zero based. Labels identify a fixed diabatic basis, not
instantaneously sorted adiabatic eigenvalues. The full runnable input and a
nonzero-coupling teaching PES are in
[`examples/nonadiabatic`](../examples/nonadiabatic/README.md).

```bash
pyrpmd validate examples/nonadiabatic/exchange.inp --check-pes
pyrpmd run examples/nonadiabatic/exchange.inp --workers 2 --output runs/nrpmd-exchange
pyrpmd resume runs/nrpmd-exchange --block-steps 32
pyrpmd analyze runs/nrpmd-exchange
```

## Collision stopping and estimates

The maximum off-diagonal coupling over **all** beads must be below
`asymptotic_coupling_tolerance` at launch. A geometrically separated product or
returning reactant is accepted only when the electronic states also meet that
criterion. Otherwise propagation continues to the original time budget. The
coupling tolerance, separation and propagation duration require convergence
checks: a small instantaneous coupling is a practical stopping criterion, not a
proof that later recoupling is impossible.

For chemical channel alpha, the electronic contribution is `I_alpha * n_s`.
Its rate or cross-section estimate is the sample mean of `w * I_alpha * n_s`,
using the same collision sampling weight w and the same original trajectory
count as the chemical estimate. Fixed-energy sampling produces cross sections;
thermal-flux and Maxwell sampling retain their respective rate weights.
Electronic-state contributions sum to the corresponding chemical contribution.
This identity is also useful for detecting lost mapping data or inconsistent
normalization.

Parent-chain bootstrap draws are shared by every chemical/electronic channel.
The resulting joint covariance retains their statistical dependence. Signed
mapping estimates do not use binomial/Wilson uncertainty intervals. More
independent parent chains and correlation checks remain necessary for useful
error bars.

Unresolved and failed trajectories remain in the original denominator. Final
electronic rates/cross sections are withheld if any such trajectories remain;
their signed missing contributions cannot be bounded by ordinary zero-to-one
population bounds. Chemical diagnostics and observed electronic contributions
remain available. Inspect `results/summary.json` under `nonadiabatic` and
`results/electronic_channels.csv`.

## Supported scope and physical validation

The electronic matrix propagation uses NumPy on CPU. Serial execution and local
trajectory processes are supported, and the nuclear free-ring map may use the
existing native kernel. MPI, bead-worker matrix evaluation, GPU matrix propagation
and fused native electronic propagation are not supported. This version uses
real symmetric and complex Hermitian spin-orbit matrices in a fixed diabatic
basis. The exact electronic substep preserves complex phase information in
both electronic rotation and the integrated nuclear impulse. Coordinate-dependent
basis changes require the appropriate derivative terms from the provider.

For H + HO2 or another real reaction, supply a justified diabatic basis, the full
coupled matrix and its gradients. Independent singlet and triplet PESs do not
produce intersystem crossing without the appropriate coupling. Validate forces,
asymptotic states, energy/norm behavior, numerical convergence and product
commitment before interpreting state-resolved rates. Comparing suitable model
cases with exact quantum dynamics is a separate physical validation step.

## References

1. J. O. Richardson and M. Thoss, “Communication: Nonadiabatic ring-polymer
   molecular dynamics,” *J. Chem. Phys.* **139**, 031102 (2013).
   [DOI: 10.1063/1.4816124](https://doi.org/10.1063/1.4816124).
2. S. N. Chowdhury and P. Huo, “State dependent ring polymer molecular dynamics
   for investigating excited nonadiabatic dynamics,” *J. Chem. Phys.* **150**,
   244102 (2019). [DOI: 10.1063/1.5096276](https://doi.org/10.1063/1.5096276);
   [author preprint](https://arxiv.org/abs/1903.06953), especially Eqs. 20–23.
3. M. S. Church, T. J. H. Hele, G. S. Ezra and N. Ananth, “Nonadiabatic
   semiclassical dynamics in the mixed quantum-classical initial value
   representation,” *J. Chem. Phys.* **148**, 102326 (2018).
   [DOI: 10.1063/1.5005557](https://doi.org/10.1063/1.5005557);
   [author preprint](https://arxiv.org/abs/1709.07474).
4. J. O. Richardson, P. Meyer, M.-O. Pleinert and M. Thoss, “An analysis of
   nonadiabatic ring-polymer molecular dynamics and its application to vibronic
   spectra,” *Chem. Phys.* **482**, 124–134 (2017).
   [author preprint](https://arxiv.org/abs/1609.00644).

These papers motivate the mathematical method; their benchmarks are not claims
of validation for the artificial exchange example or for PyRPMD on a real PES.
