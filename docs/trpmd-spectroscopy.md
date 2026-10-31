# TRPMD, correlation functions, and relative spectra

PyRPMD provides explicit internal-mode TRPMD propagation and Python analysis
functions for centroid velocities, bead-averaged dipoles, signed flux
correlations, power spectra, and relative IR spectra. The default `Dynamics.step`
remains deterministic real-mass RPMD. Existing `sample_step` calls still use
PILE-L with a thermostatted centroid for equilibrium sampling.

## Thermostat definition and propagation

The code samples the sum-bead Hamiltonian at inverse temperature `beta/P`, using
physical masses on every bead. In orthonormal real ring modes,

$$
\omega_k=\frac{2P}{\beta}\sin\frac{\pi k}{P},\qquad
\gamma_0=0,\qquad \gamma_k=2\lambda\omega_k\quad(k>0).
$$

Here `lambda` is the dimensionless `friction_scale`. The momentum OU update has
stationary variance `m*P/beta`; the centroid receives neither damping nor random
noise. This follows the TRPMD definition of Rossi, Ceriotti, and Manolopoulos
([J. Chem. Phys. **140**, 234116 (2014)](https://doi.org/10.1063/1.4883861),
[author manuscript](https://arxiv.org/abs/1406.1074)).

```python
# `state` is an equilibrium ring-polymer State, with physical-mass momenta.
# `dyn` is an existing Dynamics object.
dyn.trpmd_step(state, friction_scale=1.0)

# Equivalent explicit selection, also useful for general step dispatchers:
dyn.sample_step(state, thermostat="trpmd", friction_scale=1.0)
```

`gamma_centroid=0` is mandatory for TRPMD; a nonzero supplied value raises an
error. The default `friction_scale=1` retains the existing PILE-L internal-mode
frictions while removing centroid friction. It is a clear reference setting,
not a universally optimal spectroscopy parameter. Compare, for example, `0`,
`0.5`, and `1` at a converged time step. `friction_scale=0` and `P=1` use exactly
the existing deterministic step and do not consume thermostat random numbers.
For ordinary PILE-L the centroid default remains `1` inverse atomic time, and
the internal damping/noise convention is unchanged.

`normal-mode-bab` uses BAOAB with exact free-ring half rotations;
`cayley-bcb` uses BCOCB with the square root of the full Cayley map. They converge
to the same TRPMD equations as the time step decreases, but their finite-step
errors differ. The native free/OU/free kernel is reused where available;
otherwise the existing NumPy normal-mode implementation is used. Set
`loop_backend="auto"` or `"python"`: fully fused native stochastic blocks are
not implemented. Velocity Verlet is not supported by this thermostat API.
For analysis of these integrators see Rosa-Raíces et al.,
[J. Chem. Phys. **154**, 024106 (2021)](https://doi.org/10.1063/5.0036954),
[open full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC7796824/).

Every TRPMD step commits coordinates, momenta, and clock only after successful
propagation. A failed PES evaluation restores the thermostat RNG as well. Fixed
atoms retain identical bead positions and zero momenta. Equilibrium preparation
remains the caller's responsibility: internal-mode thermostats alone cannot
thermalize an uncoupled harmonic centroid. Prepare independent equilibrium
states using PILE-L/Andersen (or an exact distribution), then switch to TRPMD
for the saved real-time production segment.

## Observables and correlation functions

```python
from pyrpmd.analysis.spectroscopy import (
    centroid_velocities, bead_averaged_dipoles,
    velocity_autocorrelation, dipole_autocorrelation,
    time_correlation, flux_correlation,
    power_spectrum, relative_ir_spectrum,
)

# p: [replica, frame, P, N, 3]; masses: [N], physical masses in atomic units.
# dipoles: [replica, frame, P, 3], from a dipole surface on EACH bead.
# dt_saved: time between saved frames, in atomic time.
vacf = velocity_autocorrelation(p, masses, dt_saved, max_lag=1000)
dipole_acf = dipole_autocorrelation(dipoles, dt_saved, max_lag=1000)
ir = relative_ir_spectrum(dipoles, dt_saved, beta=beta, window="hann")
```

Single-replica inputs may omit the replica axis. The velocity estimator is
the centroid momentum divided by the physical mass. Its correlation is the
average of per-atom three-dimensional dot products. No center-of-mass drift
removal, atom selection, or mass weighting is performed implicitly; apply those
choices to the observable deliberately.

The dipole estimator is `mean_beads(mu(q_bead))`. Evaluating a nonlinear dipole
surface only at `q_centroid` gives a different estimator. Both velocity and
dipole functions report RPMD/TRPMD approximations to Kubo correlations. The
analysis routines cannot inspect the dynamics that generated imported arrays;
passing a fully thermostatted PIMD sampling trajectory does not make its clock
physical. General nonlinear observables do not inherit the exact harmonic
linear-observable result.

For an arbitrary scalar/vector observable, `time_correlation(values, dt,
other=None)` accepts `[time]`, `[time,component]`, or
`[replica,time,component]`. The trailing component axis is required for scalar
replicas. It computes

$$
C_{AB}(\ell)=\frac{1}{R}\sum_{r=1}^{R}
\frac{1}{T-\ell}\sum_{t=0}^{T-\ell-1}
\sum_d A_{rtd}B_{r,t+\ell,d}.
$$

The FFT is zero-padded to avoid time wraparound. Independent records are kept
separate: no time origin crosses a replica boundary. `method="direct"` provides
an independent quadratic-time check. `demean=False` preserves the full
correlation; `demean=True` removes each replica's finite-record mean. By default
the maximum lag is half the record. The result contains the lag times, per-replica
curves, mean, number of contributing origins, and standard error across replicas.
A single replica has `standard_error=None`; time origins and Cartesian
components are not counted as independent error samples. Replica error bars do
not include bead, time-step, PES, or approximate-dynamics errors.

`flux_correlation(flux, dt, side=side)` computes a signed equilibrium flux-side
correlation; omit `side` for flux-flux. Scalar input has shape `[time]` or
`[replica,time]`, with `side` in `[0,1]`. The caller supplies the flux observable,
including the dividing-surface delta/bin measure. The function does not infer a
rate constant, reactant normalization, entrance volume, or channel branching,
and does not replace the constrained-parent, paired-child estimator.

## Spectral convention and scope

`power_spectrum` returns a one-sided, nonnegative modified periodogram, with
equal averaging over independent replicas. For each Cartesian component,

$$
S(f)=\frac{\Delta t}{\sum_t w_t^2}
\left|\operatorname{FFT}(w_t A_t)\right|^2,
$$

with a factor of two at positive frequencies except an included Nyquist bin.
Vector component powers are summed. `window="hann"` is the default;
`window="boxcar"` or `None` retains a rectangular window. Parseval's relation
then yields the corresponding window-weighted mean square. `demean=True` removes
constant offsets. Zero padding via `nfft` refines the displayed frequency grid;
the record duration still controls frequency resolution.

The returned `frequency_au` is cycles per atomic time;
`angular_frequency_au=2*pi*frequency_au`. Hz and cm⁻¹ axes are provided. The
numeric `power` density is still per cycles/atomic-time on all these axes:
transforming its integral to a different frequency unit requires the matching
Jacobian. This prevents an axis conversion from silently changing a physical
normalization.

`relative_ir_spectrum` applies `beta*omega**2` to the bead-dipole Kubo spectrum
and, by default, normalizes its peak to one. It preserves the unnormalized
relative array and the scale used. The returned `relative_ir_standard_error`
uses that same common peak scale: it is conditional on the fitted normalization
and does not propagate uncertainty in the peak value or location. Use
`normalize=False` to compare the unnormalized replica uncertainty directly.
No additional harmonic quantum correction
factor is applied. Absolute absorbance/cross sections require a specified
electromagnetic convention, volume or molecular concentration, and possibly a
refractive index; they are not supplied by this routine. Finite-record leakage
is amplified by `omega**2`, so examine cadence, record length, and window
dependence before interpreting weak high-frequency bands. TRPMD damping can
broaden spectra and cannot reproduce arbitrary exact quantum line shapes.

## Rates, PA-CMD, and validation boundaries

TRPMD rate sensitivity is useful, but increasing friction is not a general
accuracy correction. Hele and Suleimanov found nearly friction-independent
rates above the crossover and frequently decreasing rates below it; agreement
with exact quantum results depends on the reaction and friction
([J. Chem. Phys. **143**, 074107 (2015)](https://doi.org/10.1063/1.4928599),
[author manuscript](https://arxiv.org/abs/1506.07460)). Compare a common equilibrium
ensemble, matched stochastic sampling budgets, and resolved transmission
plateaus. Friction spread does not constitute a rigorous uncertainty bound.

PA-CMD is **not implemented** by setting `friction_scale`: it requires a changed
normal-mode mass matrix and a separate adiabatic-limit/convergence treatment.
This implementation always retains real masses.

The regression suite checks free-centroid momentum conservation, the identical
classical/zero-friction limits, finite-friction internal Gibbs covariance,
harmonic centroid invariance, native/NumPy agreement, failure/RNG rollback,
FFT versus direct correlations, record boundaries, nonlinear dipole averaging,
signed flux ordering, Parseval normalization, and a known spectral peak.

For a complete small propagation-to-spectrum example:

```bash
PYTHONPATH=src python benchmarks/trpmd/harmonic_spectrum.py --output /tmp/trpmd.json
```

This benchmark draws exact Gaussian harmonic ring-polymer initial states and
compares `friction_scale=0,0.5,1`. It checks the common centroid motion and a
spectral peak within one finite-record frequency bin. It is an analytic model
validation, not a converged molecular spectrum or a literature reaction-rate
reproduction.
