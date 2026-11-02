# Mixed sampling, fragment PILE, and BCOCB-RESPA

This opt-in extension separates the sampling map from the real-time ring map.
It is based on main `4698c1202d0726bce7d63742e7cd31a6f137a847`, not the old
unmerged review patch. Existing defaults, explicit legacy pairings, random
streams and checkpoint identities remain unchanged unless a new option is used.
It does not silently put existing input workflows on a GPU.

## Available routes

| Route | Explicit selection | Implementation and scope |
| --- | --- | --- |
| Isotropic PIMD NPT | `sampling.integrator="bcocb"` with normal-mode dynamics, or `"baoab"` with Cayley dynamics | Python BZP wrapper or rebuilt native BZP middle map; full 3D PBC, all atoms mobile, PILE-L |
| Real-time TRPMD | `Dynamics.trpmd_step(method="bcocb"/"baoab")`; `spectroscopy.integrator`; `recrossing.integrator` | Selected physical ring map plus internal-mode OU; centroid undamped |
| Thermal dRPMD parents | `sampling.thermostat="pile-l"`, `sampling.integrator="bcocb"/"baoab"` | Mass-metric linear fragment-centroid COM constraints; checkpointed continuous parent chain |
| Multiple inner steps | `TorchDynamics(...integrator="cayley-bcocb",inner_steps=M)` or `MTSDynamics`/`Dynamics` with `sampling_integrator="bcocb"` and explicit force splitting | Symmetric BCOCB-RESPA extension, including additive RPC; independent OU noise at every inner step |

The last row is an explicit Python force-decomposition API, not a new generic
fast/slow `.inp` grammar. Components must supply an actual additive energy
split. No force masking or implicit identification of expensive components is
performed. CPU `sampling_integrator="baoab"` also selects nested BAOAB-RESPA.
`Dynamics.integrator_name` continues to identify its configured real-time map;
`PropagationPlan.ring_integrator` identifies the selected sampling kernel.

Examples that exercise actual public frontends are
`examples/ensembles/pimd_mixed_npt.inp`, `examples/drpmd/fragment_pile.inp`,
`examples/spectroscopy/mixed_trpmd.inp`, and
`examples/sampling/bcocb_mts.py`. They use analytic potentials and short runs;
they are mechanics examples, not converged chemistry calculations.

## Isotropic NPT

```toml
[dynamics]
integrator = "normal-mode-bab"
timestep = "0.1 fs"

[sampling]
engine = "native"
thermostat = "pile-l"
integrator = "bcocb"
gamma_centroid = "0.01 fs^-1"

[barostat]
mode = "bzp"
pressure = "100 bar"
tau = "100 fs"
gamma = "0.01 fs^-1"
```

The explicit native pressure map is a palindrome: piston OU half-step, piston
pressure half-kick, centroid/cell dilation half-step, fixed-volume particle
BCOCB (or BAOAB), reverse dilation/pressure/bath half-steps. Forces and physical
virials are refreshed after cell changes. This is not achieved by changing a
label on the older coupled-kick OBABO pressure map. The native configuration
contains `middle_npt=true` and the checkpoint fingerprint records the changed
map. Old configurations omit that key and retain the old pressure algorithm.
The standard workflow enters this new map for an explicit mixed pairing;
previously accepted paired NPT inputs retain their legacy map/restart identity.
The low-level `NativeEnsemble(sampling_integrator=...)` always explicitly
requests the new middle map, even for a matching base kernel.
A stale native extension lacking `MIDDLE_BZP_API` is rejected with rebuild
instructions rather than silently executing a different map.

Only centroids are scaled by the BZP dilation. The particle-mode thermostat
and cell/piston RNGs participate in the same rollback transaction. Python and
native implementations retain their existing piston inertia conventions:
`3*N/beta*tau^2` versus `3*N*P/beta*tau^2`, respectively. Comparisons must match
actual inertia (native tau = legacy tau/sqrt(P)), not assume nominal time
constants are interchangeable. No existing parameter is reinterpreted.

The new explicit middle pressure map does **not** enable anisotropic MTTK/RGB,
SCR, fixed atoms, constrained bonds, or split/RPC pressure without a validated
physical virial. Those are distinct algorithms, not omitted validation guards.
Existing supported legacy pressure paths are unchanged. TRPMD/PIGLET with
pressure control remain disallowed. A second external piston cannot be applied
to a Dynamics object that already owns pressure-control operators.

## TRPMD and sampling-to-dynamics handoff

```python
sampling.sample_step(state, method="bcocb", gamma_centroid=gamma)
state.p[:] = sampling.sample_momenta()  # independent Maxwell launch momenta
production.trpmd_step(state, method="baoab", friction_scale=0.5)
```

The selected map is used even at zero friction; it does not fall back to a
contradictory base integrator. At P=1, or zero internal friction, no thermostat
randomness is consumed. Ordinary `step()` remains unthermostatted and uses the
original base map. Changing methods does not mutate the base integrator or
bypass custom integration hooks.

For spectroscopy, `spectroscopy.integrator` is allowed only with
`method="trpmd"`. `sampling.integrator` selects equilibration separately. A new
mixed BCOCB equilibration path refreshes Maxwell momenta exactly once at the
phase boundary, before its first production checkpoint. Failure at that boundary
rolls back the draw; resume from the committed boundary does not draw again.
The diffusion workflow has the same once-only transactional handoff. PA-CMD
fictitious-mass sampling remains BAOAB; it is not silently treated as real-mass
BCOCB.

For rates, `recrossing.dynamics="trpmd"` plus an explicit
`recrossing.integrator="bcocb"/"baoab"` selects the child splitting. Ordinary
RPMD children remain thermostat-free. TRPMD recrossing is still an explicitly
labeled friction-sensitive approximation, not an assertion of rigorous RPMD
rate equivalence. Changing the child splitting invalidates checkpoint identity.

## Fragment-constrained dRPMD PILE

```toml
[dynamics]
integrator = "normal-mode-bab"

[sampling]
thermostat = "pile-l"
integrator = "bcocb"
gamma_centroid = "0.1 fs^-1"
```

Only each fragment's mass-weighted centroid COM is fixed. Rotations,
vibrations and internal bead fluctuations remain mobile; these are not rigid
bond constraints. The anchors come from the original fragment geometry and
are unchanged across checkpoint blocks. Position and momentum projections use
the mass metric, and apply after every physical kick, free-ring substep and OU
increment. For these linear constraints the constrained mass/Jacobian factor
is constant; no position-dependent Fixman correction is introduced.

`Dynamics.thermalize(...thermostat="pile-l", method="bcocb",
fragment_reference=reference, resample=False)` exposes the low-level route.
It retains completed steps on later failure, while each completed step is
atomic. The dRPMD parent worker additionally restores its checkpoint transaction
including operator/bath state, clocks and launch index after publication or
launch failures.

BCOCB endpoint momenta are not assumed Maxwellian. Each collision launch uses
an independent stream `[seed,6,trajectory_id]` to redraw real-mass Maxwell
momenta on a **copy** of the parent, then removes fragment COM momentum.
Existing collision geometry/flux sampling keeps stream `[seed,2,trajectory_id]`.
The saved parent and its thermostat stream are not disturbed. Legacy Andersen
inputs retain their original launch and parent RNG sequence.

## Multiple-time-step BCOCB

Let h=dt/M and let S(h) be the **principal square root of the full-h Cayley map**,
not `cay(h*A/2)`. The new extension is

```
Bslow(dt/2) [Bfast(h/2) S(h) O(h) S(h) Bfast(h/2)]**M Bslow(dt/2)
```

Each O uses a fresh independent noise field and duration h. Using dt inside
the inner loop would overdamp by M; reusing the same noise would give an
incorrect covariance. Fast/slow endpoint reuse is unchanged: S outer steps in
one block use S*M+1 fast evaluations and S+1 slow evaluations. The Torch model
receives all trajectories and beads in one flattened call per explicit term.
RFFT and RPMDrate-compatible half-complex layouts obey the same covariance.
On any Torch block failure the entire state, clock and generator roll back.

```python
from pyrpmd.multitime import MTSDynamics

engine = MTSDynamics(
    split_potential, masses, beta, P, dt, inner_steps=4,
    integrator="normal-mode-bab", thermostat="pile-l",
    sampling_integrator="bcocb", seed=17,
)
```

Existing implicit force-split thermostat settings retain their outer-bracketed
scheme and old restart key. Select the new `sampling_integrator` explicitly;
it is not silently inferred from an old checkpoint. The Torch opt-in name
`cayley-bcocb` is already explicit, so `inner_steps>1` activates the extension.
At M=1, additive fast/slow kicks combine to the original BCOCB method. With no
thermostat it reduces to reversible deterministic RESPA.

**This extension is not the single-level harmonic-exact BCOCB theorem.** Outer
slow kicks retain finite-step bias and possible MTS resonance. RPC also changes
the Hamiltonian. Converge both timesteps and contracted bead counts, and report
observables relative to the intended Hamiltonian rather than just a stable run.

## Independent validation

`benchmarks/mixed_sampling_validation.py` builds matrix exponentials, Cayley
solves, general square roots, mass-metric constraint projectors and discrete
Lyapunov solutions without reading production coefficients. Actual conditional
mean/noise maps are measured by injecting coordinate, momentum and independent
noise basis vectors. Tests include both normal-mode layouts, odd/even/classical
bead limits, coupled unequal masses, force-call counts, second-order error,
reversibility, preserved legacy stream identities, and failed force/provider,
observation, publication and restart boundaries.

Five reproducible numerical sections are available (one output per command):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python \
  benchmarks/mixed_sampling_validation.py --section npt --output npt.json
# Other sections: periodic-npt, mts, fragments, quartic. "all" runs all five.
```

All intervals below are **pointwise** 95% intervals over independent replicas
or independent trajectory time means, not over correlated beads/saved times.
The same seed is deliberately reused between methods or contraction levels;
those comparisons are correlated, not extra independent replications.

* **BZP volume law.** For two free particles, beta=2 and pressure=0.1,
  `V ~ Gamma(N+1,1/(beta*p))`, so E[V]=15 and E[V²]=300. At P=4, dt=0.04,
  64 replicas give 14.992812 [14.911875,15.073749], and second moment
  300.375060 [297.185850,303.564271]. All P=1/4, dt=0.04/0.02 volume and
  second-moment intervals cover their references. Internal beads do not add
  extra powers of V. Both splittings share this free-particle volume path.
* **Nonzero periodic forces.** A P=1 fractional cosine potential has an
  independent modified-Bessel partition-function reference: E[U]=0.757156585,
  E[V]=15. At dt=0.04, 64 replicas give U=0.757312738
  [0.752698828,0.761926649], V=15.031466415
  [14.965573669,15.097359161]. Halving dt is also compatible. At P=1 the two
  ring schemes coincide; this test cannot distinguish their internal modes.
  Nonzero force/virial propagation at P=3 is separately compared stepwise
  between Python and native implementations with matched piston inertia.
* **Constrained PILE cold start.** P=4, beta=3, masses [1,2,5], fragments
  [[0,1],[2]], harmonic stiffness 0.6, dt=0.04, duration24, 128 replicas.
  BCOCB q²=0.310739929 [0.281042944,0.340436914], independent finite-time
  prediction 0.304550069. Momentum variance is likewise compatible; constraint
  residuals and BAOAB values are retained in the machine-readable summary.
* **Harmonic RPC/MTS sampling.** P=8, beta=4, mass1.7, fast/slow stiffness
  0.9/0.1, Pc=1/3/8, M=4, dt=0.16/0.08/0.04, 4096 cold-start trajectories,
  time40. Three of nine means (all dt=0.16 with common noise) lie just outside
  their pointwise 95% intervals against the exact discrete covariance; maximum
  discrepancy is 2.0283 estimated standard errors. This is reported, not
  concealed by changing seeds or widening regression tolerances. Independent
  deterministic/noise matrix tests pass. The exact map itself has finite-step
  bias: for Pc=P it is about 0.0228%, 0.00569%, 0.00142%, respectively, relative
  to the contracted Gibbs q². Do not call the MTS map harmonic-unbiased.
* **Quartic MTS observable.** Vfast=x²/2, Vslow=x⁴/4-x²/2, P=16, beta=1,
  Pc=P. Independent transfer-operator grids256/512 give K=0.581077818.
  256 trajectories, equilibration16 plus production64, give centroid-virial
  K=0.580837621 [0.579465933,0.582209308] at (dt,M)=(0.02,1);
  0.581103540 [0.579723207,0.582483873] at (0.08,4); and
  0.580845731 [0.579538885,0.582152578] at (0.04,4). These are compatible at
  the reported precision, not a proof of zero nonlinear bias.

See [numerical summary and full-data checksum](../benchmarks/torch_literature/mixed_sampling_summary.json).
The downloadable review bundle retains full replica summaries and test logs.
No CUDA execution, trained chemistry, liquid-water/metal observables, hardware
speedup, or universally stable NPT/MTS timestep is certified by these tests.

## References and provenance

BCOCB: Korol et al., J. Chem. Phys. 152, 104102 (2020),
DOI 10.1063/1.5134810. Isotropic stochastic pressure control: Bussi,
Zykova-Timan and Parrinello, J. Chem. Phys. 130, 074101 (2009),
DOI 10.1063/1.3073889. Symmetric force splitting: Tuckerman, Berne and Martyna,
J. Chem. Phys. 97, 1990 (1992), DOI 10.1063/1.463137.
The middle-BZP and nested BCOCB-RESPA compositions here are explicitly defined
and independently tested extensions, not claimed verbatim implementations of
all algorithms or numerical figures in those papers.


## Validation receipt (2026-11-02)

Tested source: the changes in this PR on pinned main `4698c1202d0726bce7d63742e7cd31a6f137a847`.
Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, Torch 2.10.0+cpu, float64,
one numerical thread; native extensions rebuilt from source. The base source
archive came from GitHub Actions run41328753813/artifact13095804018, whose zip
SHA-256 is `d6ba31a91fb5a7145e586c1ad7fb1897d8aff5c7542e952771ecf321f1fc9be2`.
Missing already-committed regression fixtures were recovered through GitHub
and verified against their original Git blob hashes; no reference values were
invented. MANIFEST.in now includes those fixtures and the new numerical harness.

| Executed local check | Result |
| --- | --- |
| Three new regression modules | 199 passed, 1 CUDA skip |
| Complete source suite | 7424 passed, 1 failed, 179 skipped, 478 subtests passed |
| Rebuilt/extracted source archive; new tests, old BCOCB and published-NRPMD tests | 368 passed, 2 CUDA skips |
| Three standard-input examples, each validate/run/resume | All nine commands returned success |
| Public MTSDynamics example | 200 outer steps, time8; 801 fast and 201 slow evaluations |

The one complete-suite failure is the existing GUI-backend isolation check.
A fresh subprocess importing only Matplotlib, with `MPLBACKEND=TkAgg`, reports
`backend="agg"`, `pyplot_before_backend_query=true`, and
`pyrpmd_imported=false`. This is independent evidence of the environment
violating the test's prerequisite, not a reason to hide or remove the failure.
The full local suite is **not** reported all-green. Other skips require
optional PESs/dependencies, MPI configurations or CUDA. The original incomplete
source-archive run (7419 passed, two failures) is retained in the review bundle;
its missing-reference failure was repaired by restoring the exact upstream CSV
and including it in the manifest, then rerunning the full suite.

The remote main branch advanced during testing. The counts above belong to
the pinned tested base plus this patch, not untested intervening commits.
Review PR merge-ref CI separately. Existing real-chemistry CI paths do not
constitute CUDA or new-splitting chemistry validation. Nothing is merged by
this submission.
