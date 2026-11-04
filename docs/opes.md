# OPES probability-based sampling

`task = "fes"` with `[fes] method = "opes"` runs native Python OPES with one or two collective variables.
The old `task="opes"` remains a deprecated alias. Common input nesting, the
`FESResult` file contract and legacy checkpoint continuation are described in
[enhanced sampling](enhanced-sampling.md); method parameters may be placed in
`[fes.opes]` instead of the legacy `[opes]` table.
It reuses PyRPMD's ring dynamics, potential adapters, CV Jacobians, task
scheduler and atomic checkpoints. Each `sampling.replicas` stream has an
independent probability estimate and random stream.

## Method and conventions

The default is **OPES_METAD with fixed bandwidth and no kernel compression**.
Optional `mode="explore"` follows
[Invernizzi and Parrinello (2022)](https://doi.org/10.1021/acs.jctc.2c00152),
and `adaptive_sigma=true` enables an explicit package-specific width rule.
Both options are described in
[extended sampling](tier2-sampling.md#opes-explore-and-adaptive-widths).
In the default `mode="metad"`, the method reconstructs an unbiased probability
density and derives the bias from that estimate. For physical inverse
temperature `beta` and fixed width `sigma`,

\[
 P_n(s)=\frac{\sum_{k=1}^n w_k G_\sigma(s-s_k)}{\sum_{k=1}^n w_k},
 \qquad w_k=\exp[\beta V_{k-1}(s_k)],
\]

\[
 Z_n=\frac1n\sum_{i=1}^n P_n(s_i),\qquad
 V_n(s)=\frac{1-1/\gamma}{\beta}
 \log\left[\frac{P_n(s)}{Z_n}+\epsilon\right],\qquad
 \log\epsilon=-\frac{\beta\,\text{barrier}}{1-1/\gamma}.
\]

The probability/bias equations follow
[Invernizzi and Parrinello (2020)](https://doi.org/10.1021/acs.jpclett.0c00497).
The arithmetic kernel-center estimate of `Z` follows the
[PLUMED OPES normalization algorithm](https://www.plumed.org/coverage-v2.10/coverage/opes/OPESmetad.cpp.gcov.html).
It is a finite-sample estimate of normalization over explored CV space,
not an exact geometric volume integral.

Kernels integrate to one. Periodic CV dimensions use a wrapped Gaussian,
including smooth derivatives across the periodic boundary. This differs from
PLUMED's truncated-kernel implementation. Optional adaptive widths use a
Welford/Scott-ESS rule, which differs from PLUMED's diffusion-window rule.
Kernel compression is not implemented, and numerical identity with PLUMED is
not claimed. Walkers that share one OPES kernel history are available through
`walkers_mode`; see
[distributed multiple-walker MetaD and OPES](enhanced-sampling.md#distributed-multiple-walker-metad-and-opes).
Force evaluation, each update and stored
history grow linearly with the number of kernels. Freezing the bias also
stops that growth. The incremental `Z` update avoids a complete pairwise
recalculation at every deposition; restart validation checks the saved
pairwise normalization in bounded memory.

`barrier` is a physical energy. The bias has lower bound `-barrier`; this
is not a guarantee that its total maximum-to-minimum range equals `barrier`.
New runs use the empty-density initial bias `-barrier`, giving the first
METAD kernel log-weight `-beta*barrier`. This initialization is stored in
bias state schema 2; recognized schema-1 histories retain their original
zero-initial-bias convention on restart. See the
[initialization and restart details](tier2-sampling.md#opes-explore-and-adaptive-widths)
before comparing histories or another implementation.
The ring Hamiltonian receives `P*V`. METAD probability weights use `beta*V`;
explore mode uses unit kernel weights to estimate the sampled target density.
Updates occur after completed integration steps. The two kicks in a step
see the same probability history, and observations on update steps precede
the update. A force evaluation never changes history.

## Run the example

```bash
pyrpmd validate examples/opes/harmonic_2d.inp
pyrpmd run examples/opes/harmonic_2d.inp --output runs/opes2d
pyrpmd resume runs/opes2d
pyrpmd analyze runs/opes2d --recompute --bins 10 10 --bootstrap 0
```

The example is a short harmonic model check. It retains `converged=false`.

| Input | Meaning |
| --- | --- |
| `opes.cvs` | CV names; defaults to all configured CVs. |
| `opes.mode` | `"metad"` (default) or `"explore"`; their KDE weights and bias prefactors differ. |
| `opes.sigma` | Required positive initial width per CV; fixed unless `adaptive_sigma=true`. Explicit units are supported; numeric values use the declared CV input units. |
| `opes.sigma_au` | Alternative initial widths in canonical CV units. |
| `opes.adaptive_sigma` | Enable the explicit Welford/Scott-ESS width rule; default false. |
| `opes.sigma_min` / `opes.sigma_min_au` | Optional positive per-CV width floors for adaptive mode, no greater than the initial widths; default one percent of `sigma`. Use input CV units or canonical CV units, respectively. |
| `opes.barrier` | Required positive physical energy; explicit energy units supported. |
| `opes.barrier_au` | Alternative barrier in hartree. |
| `opes.bias_factor` | Finite `gamma > 1`; default 10. |
| `opes.pace` | Positive update interval in production steps; default 100. |
| `opes.freeze_after_steps` | Optional production-relative adaptation duration. Must be a multiple of `pace`. The final update occurs at this step. |
| `opes.freeze_equilibration_steps` | Additional fixed-bias relaxation to discard for histogram analysis; default 0. |

The workflow accepts the canonical thermostats in the
[common propagation stack](propagation-stack.md), including Andersen, PILE-L/G,
NHC, SVR and canonical GLE. Configure their parameters through
`sampling.thermostat_params`; new stateful modes checkpoint their auxiliary
variables and RNG with the probability history. A failed force evaluation or
probability update restores both states before resuming.

OPES currently requires a nonperiodic system with mobile atoms. A periodic **CV** is supported through its declared
period; periodic atomic cells are a separate unsupported workflow case.
The full native propagation loop cannot include history updates, but native
free-ring substeps remain available. Independent complete histories use the
existing process/MPI/bead execution choices. Combining OPES with umbrellas,
replica exchange, MetaD or another bias is rejected.

## Free-energy estimates

With `analysis.method = "kde"`, `mode="metad"` reports `-log(P_n)/beta`
from the final unbiased-density estimate. In `mode="explore"`, the KDE estimates
the sampled target density and reports `-gamma*log(P_target)/beta`; this is a
physical FES only after convergence to the well-tempered target. Finite kernel
widths smooth both estimates, and adaptation creates additional finite-time
error. The regularized OPES bias is **not** converted into a FES using the
unregularized well-tempered relation near its barrier floor.

`analysis.method = "reweighted_histogram"` requires `freeze_after_steps` and
uses only samples strictly after the freeze plus the requested relaxation.
This estimator is available in both modes.
The retained configurations share a fixed bias. Histogram weights are
`exp(beta*V)` and are normalized in log space separately for each independent
history, so arbitrary between-history bias offsets cannot change their
relative contribution. Neither `Z` nor a time-dependent `c(t)` is subtracted
from these weights. The
[PLUMED OPES guide](https://www.plumed.org/doc-v2.10/user-doc/html/_o_p_e_s__m_e_t_a_d.html)
also describes static-bias reweighting and cautions against using its `c(t)`
diagnostic as a reweighting correction.

Samples before freezing remain in checkpoints for inspection but do not enter
the frozen histogram. An adequate fixed-bias relaxation duration still needs
to be established for the system. Independent estimates share a gauge only
on their common sampled support; unsupported bins are masked. Replica spread
is descriptive, standard errors remain undefined, and the reported
weight-only effective sample size does not account for autocorrelation.
Bootstrap uncertainty and automatic convergence certification are absent.
Biased simulation time does not provide an RPMD rate.

## Python and restart

```python
from pyrpmd.cv import CVStack, Cartesian
from pyrpmd.opes import OPESBias

cv = CVStack([Cartesian(0, "x")])
bias = OPESBias(cv, sigma=[0.3], beta=1000.0,
                barrier=0.01, bias_factor=8.0)
# Call only after a successful dynamics step:
bias.update(cv.value(state.q), step=state.step, time=state.time)
energy, force = bias.energy_force(state.q)
saved = bias.state_dict()
restored = OPESBias(cv, [0.3], 1000.0, barrier=0.01,
                    bias_factor=8.0, state=saved)
```

The direct workflow accepts built-in CV objects and fingerprints their
mathematical parameters. Custom workflow CVs must use a serializable
`kind = "python"` factory specification so checkpoints never depend on a
Python object's memory address. The lower-level `OPESBias` API accepts any
CV with the usual value/Jacobian contract.

Each workflow checkpoint commits coordinates, momenta, RNG, observations,
clocks and probability state together. Restart preserves the normalization
accumulators exactly, validates their consistency, and rejects changed
physics. A failed integration/update event rolls back the whole event.
Completed resumes and saved-data reanalysis do not instantiate a potential.
The exported `fes.npz`, `fes.csv` and `summary.json` are derived results;
the checkpoint is the authoritative restart state.

## Original method references

- M. Invernizzi and M. Parrinello, *Rethinking Metadynamics: From Bias Potentials
  to Probability Distributions*, J. Phys. Chem. Lett. **11**, 2731–2736 (2020),
  [doi:10.1021/acs.jpclett.0c00497](https://doi.org/10.1021/acs.jpclett.0c00497).
- M. Invernizzi and M. Parrinello, *Exploration vs Convergence Speed in
  Adaptive-Bias Enhanced Sampling*, J. Chem. Theory Comput. **18**, 3988–3996
  (2022), [doi:10.1021/acs.jctc.2c00152](https://doi.org/10.1021/acs.jctc.2c00152).
