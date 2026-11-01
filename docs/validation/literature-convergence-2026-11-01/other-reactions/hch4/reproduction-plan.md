# H + CH₄: paper extraction and preregistered next calculation

**Status: protocol registered; no H + CH₄ equilibrium campaign or thermal rate
has been calculated.** The electronic PES and native/Python propagation have
been checked. A short P=128 energy test at the paper time step failed the
predeclared absolute energy-span gate, and that result remains visible.

The machine-readable protocol is
[reproduction_protocol.json](../../../../../benchmarks/hch4/reproduction_protocol.json).
The [source-reading receipt](reproduction-sources.json) pins the official source
files used for conventions. The [native verification](native/README.md) records
actual propagation, the retained failure and measured cost. These additions
do not change the earlier [PES validation](README.md) or any H₃ evidence.

## What is actually in the paper

The target is Suleimanov, Allen and Green, *Computer Physics Communications*
**184**, 833–840 (2013),
[DOI 10.1016/j.cpc.2012.10.017](https://doi.org/10.1016/j.cpc.2012.10.017).
The accessible [author manuscript](https://www.researchgate.net/publication/279840192_RPMDrate_Bimolecular_chemical_reaction_rates_from_ring_polymer_molecular_dynamics)
contains Tables 1 and 2 and the working rate equations. The manuscript's upload
date is not the journal publication date. Its H + CH₄ surface is the pinned
CBE/PES-2008 model, not a different Jordan–Gilbert benchmark.

Table 2 contains one temperature for this example: **300 K**. All rates below
are **cm³ molecule⁻¹ s⁻¹**. The listed surfaces are the paper's PMF maxima.

| Calculation | Beads | ξ‡ | κ | TST/QTST | Full rate |
|---|---:|---:|---:|---:|---:|
| Classical | 1 | 0.999 | 0.762 | 6.53 × 10⁻²² | 4.98 × 10⁻²² |
| RPMD | 128 | 1.007 | 0.659 | 1.71 × 10⁻¹⁹ | 1.13 × 10⁻¹⁹ |

The product of the rounded factors need not equal the rounded full rate
exactly. Section 4 gives an approximate **5% statistical accuracy**, without
publishing a confidence-interval definition. That statement is retained as
paper precision, not recast as a 95% interval.

Table 1 specifies R∞=30 bohr, one paired forming/breaking bond, four equivalent
product channels, an Andersen thermostat, 111 umbrellas from −0.05 through
1.05 with spacing 0.01 and stiffness 30 hartree, and a 0.1 fs time step. Each
window has 200 trajectories, each with 20 ps equilibration and 100 ps sampling.
The original UI range is −0.02…1.02 on 5000 points. Its recrossing protocol uses
20 ps constrained-parent equilibration, parent configurations every 2 ps,
100 child trajectories per configuration and 100000 children in total, each
lasting 50 fs. The reported 48 processors and historical 3600 processor hours
describe that implementation and hardware, not a modern runtime guarantee.

## Distribution details that must not be silently merged with the table

The exact source is
[GreenGroup/RPMDrate commit 8c1828d](https://github.com/GreenGroup/RPMDrate/tree/8c1828d2302aacf09a15faf5c7fc50a18d4c113e).
Its [H + CH₄ input](https://github.com/GreenGroup/RPMDrate/blob/8c1828d2302aacf09a15faf5c7fc50a18d4c113e/examples/H%2BCH4/input.py)
provides atom order, geometry and equivalent bond definitions. Inspection
finds several small or protocol-level differences:

| Detail | Exact distributed behavior | Proposed choice |
|---|---|---|
| Last window | `arange(-.05,1.05,.01)` produces 110 centers ending approximately 1.04 | The Table 1 inclusive 111-center grid, stated explicitly |
| R∞ | `30*.52918` Å becomes 30.000158117920183 bohr using its length constant | Table value 30 bohr |
| Isotope masses | H=1.00782503207 u, C=12 u | Those explicit masses; no inferred element-average or proton mass |
| Mass/time/temperature constants | Legacy conversion constants | Current PyRPMD conversion constants, recorded in resolved input; legacy PES constants remain unchanged |
| Andersen default | Whole-system momentum refresh after `floor(sqrt(nsteps))` steps if no interval is supplied | Geometric waiting times with mean 50 fs, explicitly a different sampling schedule |
| Rate printout | Console final rate is per mole; saved rate file prints both per molecule and per mole | Compare per-molecule values with Table 2 |

For the nominal upstream 20 ps and 100 ps calls, the default collision intervals
are 44.7 fs and 100 fs respectively. A retried shorter call can have another
interval. The modern geometric schedule avoids a fixed refresh clock while
retaining the intended canonical target; this is not a claim of identical
finite sampling. The integrator is current Cayley BCB, so time-step controls
are essential. At 300 K the upstream β is 1052.5822153578245 hartree⁻¹, versus
1052.5834160132904 with current constants. Its u-to-electron-mass factor is
1822.888395490959, versus 1822.8884862827601. None of these constants is adjusted
using the rate table.

## Mapping to the current PyRPMD equations

Use six flexible atoms in **H,C,H,H,H,H** order, with the incoming H last.
The two reactant fragments are `[0,1,2,3,4]` and `[5]`. The reference forming
bond is `(3,5)` and breaking bond `(1,3)`, all zero based. The additional
permutations exchange atom 3 with methane atoms 4, 2 and 0. Carbon and the
incoming atom remain fixed under those relabelings. The official input geometry
is preserved in `geometries.json`.

Writing h for each methane H, the scalar equation checked independently is

\[
s_0=R_\infty-|R_{\rm H}-R_{\rm CH_4}|,\qquad
s_1=\max_h\left[(r_f^\ddagger-r_b^\ddagger)-(r_{{\rm H},h}-r_{{\rm C},h})\right],
\qquad \xi=\frac{s_0}{s_0-s_1}.
\]

The centroid coordinate, gradient and each equivalent reference surface were
verified against direct bond/COM algebra. The rate sampling measure includes

\[
f_s=\sqrt{\frac{\nabla\xi\,M^{-1}\nabla\xi}{2\pi\beta}}.
\]

For P beads, the biased ring Hamiltonian receives
`P*U(ξ) - P*log(fs)/β`, with `U=30*(ξ-center)^2/2` in hartree. This produces the
same flux-weighted centroid PMF needed by the bimolecular prefactor

\[
k_{\rm QTST}=\frac{4\pi R_\infty^2}{\sqrt{2\pi\beta\mu}}
\exp\{-\beta[W(\xi^\ddagger)-W(0)]\},\qquad
\mu=\frac{m_{\rm H}m_{\rm CH_4}}{m_{\rm H}+m_{\rm CH_4}}.
\]

All four channels are already in the maximum-over-paths coordinate. Therefore
`symmetry_factor=1` and `electronic_weight=1`; **no extra factor four** is
applied. These chemical pathways are not a nuclear permutation-exchange
calculation. There are no bond constraints or frozen atoms in this reaction.

The existing Python provider can load `benchmarks.hch4.pes:build`. Its public
`fortran-batch-v1` binding now also supports strict native deterministic blocks,
including the four-path flux/umbrella force and constrained-parent propagation.
The legacy PES energy/gradient conversion mismatch remains documented and
unchanged. The initial electronic barrier and force checks are not rate or
ensemble validation.

## Registered sequence and stopping rules

Four seeds are fixed before sampling: **2026111101, 2026111103, 2026111107,
2026111109**. The native timing seed, 2026111199, is excluded. All seeds and
failed outcomes are retained.

1. **Executable pilot, not run.** Use the full table grid, one trajectory per
   window per seed, 200 fs equilibration and 800 fs production, retaining the
   CV every 2 fs. Four short constrained parents per seed and 32 children per
   seed exercise the complete interface if the PMF support is sufficient.
   This stage is only a failure/overlap/autocorrelation/throughput screen. Its
   short equilibration and tiny child population cannot validate the paper.
2. **Equilibrium and analysis review.** Inspect time splits, autocorrelation,
   CH₄ bond lengths/bends, relative orientation, channel identity and ring-mode
   variance. Current umbrella initialization rigidly separates fragments from
   the TS, preserves initially distorted methane bonds and starts collapsed
   rings; these are initial configurations, not thermal samples. Equivalent
   atom relabeling does not supply independent physical orientations. A pilot
   with inadequate relaxation must trigger a documented initialization/warmup
   revision before any primary result.
3. **Reference-size proposal.** Partition 200 trajectories/window into four
   independent seed groups of 50, using the paper's 20 ps+100 ps lengths. Use
   four parent chains per seed with 63,63,62,62 configurations per chain in the
   four groups, respectively; 50 antithetic pairs/configuration gives exactly
   100000 children overall. Independent parent chains add more equilibration
   than the original single-parent example. This is a proposed budget, not an
   executable one-shot production setting.
4. **Internal accuracy before comparison.** Require independent-seed and
   hierarchical block intervals: ≤2% relative 95% half-width separately for
   QTST and κ, ≤3% for their product. Compare time halves using both a ≤3%
   drift bound and a three-standard-error bound. Require connected overlap,
   endpoint support and no silently discarded unsupported bootstrap draw.
   Four seeds alone may be insufficient for those bounds; failure remains
   failure until a separately registered fixed refinement is completed.
5. **Discretization and dynamical controls.** Compare P=64,128,256 and dt=0.1,
   0.05 fs; P128-vs-P256 and half-dt full-rate changes must be resolved below
   2%. Test the 30–50 fs child plateau, extending the same children to 100 and
   then 200 fs if needed. Paired refinements are not extra independent replicas.
   R∞=20/40 bohr and surface shifts ±0.005 are additional invariance controls,
   with QTST and κ recomputed on the same surface.

The primary component comparison fixes the independently published surfaces
0.999 (classical) and 1.007 (quantum). The measured PMF maximum is reported
separately. It is never moved to bring the final rate nearer the reference.
Only after internal gates pass is a point estimate within the paper's stated
5% precision called agreement. A reference outside our own narrow interval
must still be reported explicitly.

MBAR is the primary current estimator; Gaussian umbrella integration is a
diagnostic. If the large campaign requires histogram WHAM, validate it against
full-sample MBAR on manageable data and require <1% rate change on bin
refinement before primary use. Changing histogram resolution to follow the
paper point is prohibited.

## Measured budget and the production-analysis blocker

The [native probe](native/verification.json) used actual CBE energies/forces and
the four-channel flux/umbrella dynamics at ξ=0,0.9,1.0. Its measured CPU costs
exclude thermostat/checkpoint/task dispatch and equilibrium analysis.

| Work | P=1 | P=128 |
|---|---:|---:|
| Native biased block cost | 17.5–24.5 µs/step | 0.957–1.061 ms/step |
| Four-seed short pilot: 4.44 million umbrella steps | 0.0216–0.0302 CPU h | 1.18–1.31 CPU h |
| Table budget: 26.64 billion umbrella steps | 130–181 CPU h | 7085–7854 CPU h |

These are local-block extrapolations, not statistical forecast intervals or
wall-clock promises. The P128 reference budget corresponds to roughly 148–164
hours at an ideal 48 available cores before overhead. A halved time step doubles
the step count; exact cost per step must be remeasured. Children add 50 million
unconstrained steps, and parent sampling/controls add further work.

The full table budget has **22.2 billion production CV observations** if every
step is retained. One float64 CV alone uses 177.6 GB. Dense MBAR over 22200
replica states and all samples would require about 3.94 PB for just one matrix;
merging identical umbrellas to 111 states still leaves about 19.7 TB. The pilot
is small enough for the existing workflow, but the full proposal needs staged
sampling, bounded-memory analysis, an explicit raw-data retention plan and
separately measured resource requirements before launch. Source-like UI can
accumulate moments cheaply, but that is not sufficient evidence that its
Gaussian approximation is accurate.

## Generate an input without starting a calculation

Run from the repository root after the external pinned PES is built:

```bash
python -m benchmarks.hch4.reproduction_input \
  --library /path/to/libcbe2008.so --beads 128 --seed 2026111101 \
  --output-directory /path/to/hch4-pilot-P128 \
  --write-input /path/to/hch4-pilot-P128.py
```

This command writes and resolves a standard PyRPMD input; it does not simulate.
After resource review, the normal execution command is
`python -m pyrpmd run /path/to/hch4-pilot-P128.py`. `preset="production"` in the
generated mapping means defaults must be explicit; it does **not** certify the
small pilot as a production scientific result. The original generated P1/P128
files are retained in [native](native/) as static interface evidence.

F + H₂ remains a separate preparation task. The official Werner catalog
distinguishes SW1 and spin-orbit-corrected HSW, but the renewed search did not
yield the scientific erratum's corrected numeric table. The receipt records
that retrieval limit; no corrected rates or model correspondence are guessed.
