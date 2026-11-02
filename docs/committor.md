# Fixed-ring committor analysis

`pyrpmd.committor.CommittorSampler` launches independent, unbiased,
unthermostatted RPMD trajectories from one **fixed full-ring geometry**, drawing
canonical bead momenta each time. It estimates B-before-A probability for the
supplied RPMD dynamics. At P=1 it is a classical configurational committor.
At P>1 it is a committor of RPMD's approximate real-time dynamics, **not** an
exact quantum committor. It is not a rate coefficient or a new TIS sampler.

This first version exposes only a Python API (no new `task="committor"`).

## Minimal use

```python
from pyrpmd import CommittorSampler
from pyrpmd.cv import Cartesian

# dynamics: existing unbiased, deterministic Dynamics
# q: full positions with shape [P,N,3], in atomic units
sampler = CommittorSampler(
    dynamics, Cartesian(0, "x"),
    basin_a=-0.8, basin_b=0.8,
    max_steps=10000, seed=321,
)
result = sampler.sample(q, n_shots=200)
print(result.summary(confidence_level=0.95))
```

An existing scalar CV or a pure callable `coordinate(q)` can be used.
Derivatives are not required. A is `s<=basin_a`, B is `s>=basin_b`.
Periodic scalar CVs must be replaced with an ordered, continuous basin
coordinate; the code does not infer basin definitions or their metastability.

Run the analytical demonstration and targeted regression tests:

```bash
python examples/sampling/committor.py --shots 64 --beads 4 --output runs/committor-example
python -m pytest -q tests/test_committor.py tests/test_transition_interface.py
```

The example writes JSON statistics, scalar shot records and a NumPy RNG state.
These are demonstration outputs, not a general workflow checkpoint.
The model is a one-particle double well; its collapsed barrier ring is NOT an
equilibrium conditional distribution of internal bead modes, and therefore
cannot validate quantum rates or barrier physics.

## Exact conditioning and momentum measure

Each shot **holds every bead position fixed at launch**. Under PyRPMD's
sum-beads Hamiltonian sampled at inverse temperature `beta/P`, the variance
of each *mobile* Cartesian bead momentum component is `mass*P/beta`.
Fixed atoms stay pinned with zero momenta. The Dynamics RNG is untouched.
No thermostat, bias, centroid-only velocity replacement, or COM constraint is
applied during a shot.

**A centroid-conditioned committor is different:** it additionally requires
proper equilibrium sampling of internal ring coordinates conditioned on the
centroid. Repeating a centroid across beads does not reproduce this measure.
Sample separate full-ring configurations separately; pooling their output as
IID shots of one fixed geometry is not valid without additional statistics.

Only unmodified, built-in deterministic `Dynamics` is accepted. Cayley BCB,
normal-mode BAB and velocity Verlet remain supported with fixed atoms.
Composed thermostat, barostat and force-splitting plans, bond constraints,
patched stepping methods and potentials declaring history/time/state
dependence are rejected before momentum sampling. The code cannot detect a
hidden time dependence or static restraint
inside an arbitrary user PES; supply a pure, unbiased time-independent PES.
Convergence of bead count, timestep, horizon and state definitions remains
the caller's responsibility.

## First-hit and censoring conventions

Membership is checked at step zero and **after each complete MD step,
including the final permitted step**. On the first A/B hit propagation stops.
There is no interpolation between timesteps. A starting state inside a basin
has exact boundary committor 0/1 and needs no random draw or PES evaluation.

If the maximum number of steps is reached without entering either basin, the
shot is **unresolved**, not an A or B event. The result contains one record per
attempt (`-1=A`, `0=unresolved`, `+1=B`); PES/CV errors propagate rather
than getting mislabeled as censored shots. Failed shots restore their own
sampling RNG; earlier completed shots are not undone.

| Output | Interpretation |
|---|---|
| `n_A`, `n_B`, `n_unresolved` | Counts, including unresolved shots |
| `p_B` | B proportion when *all* shots are resolved, otherwise `None` |
| `p_B_by_deadline` | B-by-deadline frequency among all attempts |
| `p_B_given_resolved` | Descriptive; **not** the unconditional committor |
| `empirical_p_B_bounds` | `[n_B/n, (n_B+n_unresolved)/n]` |
| `p_B_confidence_envelope` | Conservative exact-binomial coverage envelope |
| `converged` | Always false: numerical/physical convergence is not certified |

For a nominal `1-alpha` interval, the lower side is a Clopper–Pearson lower
bound on the B-by-deadline probability; the upper side is a Clopper–Pearson
upper bound on the not-A-by-deadline probability. Each uses a two-sided
`1-alpha` exact binomial interval, allocating at most `alpha/2` error
to each tail; their union gives conservative coverage of the *eventual*
B-before-A probability without assuming that unresolved shots are missing at
random. If all shots are unresolved the bounds are `[0,1]`.
This envelope covers *independent canonical momentum sampling at one fixed
configuration*, not uncertainty from the potential, bead count, timestep or
choice of basins. The `converged` flag remains false in all cases.

## Starting from MetaD configurations

In the metadynamics input, use:

```toml
[output]
store_configurations = true
```

Obtain a complete saved ring configuration from a validated checkpoint
(`sample_configurations[sample,bead,atom,xyz]`) and start the shot sampler
with a separate *unbiased* `Dynamics` matching the PES, masses, beta, bead
count, fixed-atom positions and periodic conventions. Do not reuse MetaD
velocities or its bias history. MetaD configurations are drawn from a biased
distribution; shooting from them does not turn the MetaD ensemble into an
unbiased transition-state distribution or a rate.

## Continuation and memory

`sample(..., on_shot=lambda index, shot: ...)` receives each completed shot
(including unresolved shots), with an owned final `State` that can be saved
to extend unresolved trajectories without redrawing momenta. Callbacks must
not mutate sampler or PES. Callback exceptions stop the run, and arbitrary
external side effects are not transactional. The sampler retains only scalar
records, not all trajectories.

For further *independent shots from the same full configuration*,
restore `sampler.rng.bit_generator.state = result.rng_state` before sampling.
Keep the same PES, beta, masses, bead count, integrator, timestep, fixed atoms,
basins and horizon, and restore to the same NumPy bit-generator implementation.
The API has no cross-version or cross-process checkpoint guarantees.

## References

- Swenson et al., *OpenPathSampling*, JCTC 15, 813–831 (2019),
  [doi:10.1021/acs.jctc.8b00626](https://doi.org/10.1021/acs.jctc.8b00626).
- [SciPy binomtest exact confidence intervals](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).
