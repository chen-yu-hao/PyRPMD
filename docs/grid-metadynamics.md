# Grid metadynamics and fixed substrates

`task="metadynamics"` supports fixed atoms and fixed periodic cells, with the
same PES and CV contracts as surface umbrella sampling. An optional bias grid
makes each force evaluation independent of the number of deposited hills.
Existing inputs continue to sum their Gaussian history directly.

## Input

Add a nested table to the ordinary metadynamics settings:

```toml
[metadynamics]
cvs = ["lateral", "height"]
mode = "well_tempered"
sigma = ["0.25 bohr", "0.25 bohr"]
height = "0.02 kcal/mol"
pace = 20
bias_factor = 8.0

[metadynamics.grid]
ranges = [["-4 bohr", "4 bohr"], ["-3 bohr", "3 bohr"]]
bins = [160, 120]
```

`ranges` contains one increasing pair per CV. Explicit unit strings are
converted to that CV's canonical units. Bare numbers use its declared input
unit, as `sigma` does. Python callers with already canonical values can use
`ranges_au` instead; the two spellings are mutually exclusive.

`bins` counts **intervals**, with at least two intervals in each dimension. It
may be a scalar shared by all dimensions or one integer per CV. When omitted,
the spacing is at most `sigma/5`, rounded to an integer interval count. There
are `bins+1` nodes on a nonperiodic axis and `bins` nodes on a periodic axis;
the periodic endpoint is shared with the first node. The grid is restricted
to one or two CVs and at most one million total nodes.

The bias grid and `analysis.bins` have different purposes. The former controls
the forces used during propagation; the latter controls only the exported FES
and its sample-support mask. If analysis ranges are omitted, they default to
the bias domain. Explicit nonperiodic analysis ranges must fit inside it.

Run the complete fixed-substrate example from the repository root:

```bash
pyrpmd validate examples/metadynamics/grid_fixed_slab.inp --check-pes
pyrpmd run examples/metadynamics/grid_fixed_slab.inp --output runs/grid-slab
pyrpmd resume runs/grid-slab
pyrpmd analyze runs/grid-slab --recompute --bins 24 24 --bootstrap 0
```

This example uses an analytic periodic cosine and harmonic confinement. It
tests the propagation, bias and restart interfaces; it is not a Cu-H model or
a converged surface free-energy calculation.

## Fixed atoms and periodic geometry

Set `system.fixed_atoms` to distinct zero-based atom indices. On every bead,
their coordinates stay exactly at the input reference geometry and their
momenta stay exactly zero, including after Andersen momentum resampling.
Forces on mobile atoms still include interactions with the fixed substrate.
At least one atom must remain mobile. Restart also checks fixed positions and
momenta when the checkpoint is already complete.

Periodic cells require a periodic-capable PES: the existing ASE, periodic
Python and i-PI contracts, or the analytic periodic models. Ordinary distance
and angle CVs retain their unwrapped-coordinate meaning; use `periodic_*`
CVs for explicit minimum-image geometry. Surface projections remain
continuous and unwrapped. Spatial PBC does not automatically give a CV a
period: a CV must explicitly expose its physical `periods` through the Python
CV interface to use wrapped hills and periodic grid interpolation.

A periodic bias axis must span exactly one declared CV period. Values are
reduced into that interval for lookup and deposition. For a nonperiodic axis,
leaving the grid raises an error and rolls back the attempted step. There is
no extrapolation, clipping, automatic domain growth or implicit wall. Choose
a domain that covers the intended trajectory; changing it changes the bias
approximation and requires a new scientific run.

## Interpolation and cost

At every grid node PyRPMD accumulates the Gaussian bias and its first
derivatives. In two dimensions it also accumulates the mixed derivative.
All nodes receive every hill, including its tails. Periodic hills use the
existing smooth wrapped-Gaussian image sum, not a minimum-image Gaussian.

For a grid interval of length `h`, with local coordinate `t` between zero and
one, the one-dimensional interpolation is

\[
V(t)=h_{00}(t)V_0+h_{01}(t)V_1
     +h\,h_{10}(t)V'_0+h\,h_{11}(t)V'_1,
\]

where the four cubic Hermite basis functions are
`2t³−3t²+1`, `−2t³+3t²`, `t³−2t²+t` and `t³−t²`.
The two-dimensional version is their tensor product. The gradient is the
analytic derivative of this same interpolated energy, followed by the CV's
full ring Jacobian. Values and forces are continuous across grid-cell edges
and the periodic seam. As for the direct bias, `H_P` receives `P*V` and the
well-tempered deposition exponent uses the physical `V` without that factor.

| Operation | Cost for dimension D and G grid nodes |
| --- | --- |
| Bias energy and force at one configuration | At most `4^D` interpolation terms; independent of hill count |
| Deposit one hill | `O(G)`; every node updated |
| Grid memory | `2^D * G` floating-point values |
| Read a valid grid checkpoint | `O(G + number of hills)` including history validation |
| Rebuild a missing grid cache | `O(G * number of hills)` in original deposition order |

The grid trades interpolation error and fixed memory for faster long-history
lookups. A short history or a very large grid can be faster with the direct
backend. Increase `bins` and compare forces, thermodynamic distributions and
FES before treating the grid approximation as converged. The default spacing
is a starting point, not an accuracy certificate. The HILLS audit list and
sample/checkpoint payloads still grow with the run; this feature bounds bias
lookup work, not every allocation in the metadynamics workflow.

Cubic Hermite interpolation does not preserve the positivity of a Gaussian
sum between nodes. A coarse grid can therefore have negative bias undershoots.
They are retained in the applied energy, its force and the recorded
`bias_before`; no clipping is applied. Well-tempered heights use that same
signed interpolated bias in the exponential, so an undershoot can make a hill
slightly taller than the nominal height. Refine the grid until such
interpolation errors are immaterial for the intended observables. The direct
Gaussian backend retains its nonnegative-bias validation.

## Deposition, checkpoints and analysis

The two kicks in an integration step see the same bias. After a successful
step, observables record the bias **before** any hill scheduled at that step;
the new hill affects the next step. The well-tempered hill height is evaluated
from the applied interpolated bias, so its update is consistent with the force
approximation.

The atomic checkpoint stores state, RNG, sample clocks, hills and the grid
cache together. The cache records its interpolation algorithm, domain, bin
counts, widths, periods, a hash of the exact hill history, and a separate hash
of the coefficient values in canonical float64 order. Direct Python cache
restoration verifies both hashes; workflow storage also uses the checkpoint
container's per-array integrity checks. A hill-only checkpoint
with the same grid configuration can be reconstructed by replaying the hills.
Switching an existing direct run to a grid changes its scientific identity.
Failed deposition transactions
restore the accepted history and rebuild that prefix in the original order,
avoiding the roundoff introduced by subtracting a failed hill. Successful
steps do not copy the history or grid for rollback.

`HILLS.csv` remains a regenerable audit view. Offline FES analysis evaluates
the same interpolated bias used during dynamics. Summary fields
`bias_backend`, `bias_grid` and `fixed_atoms` record the chosen implementation.
The existing finite-time FES qualifications remain: biased trajectory time
does not establish a reaction rate, no evolving-bias reweighting is added, and
independent replicas do not share their hills.

The Python `deposit()` result owns its center list; changing the returned
record cannot alter the deposited history. Treat `bias.hills` as a read-only
audit view. Use `deposit()` for new hills and construct a new bias from a
modified history when deliberately changing it; mutating that internal list
in place bypasses grid updates.

## Method references

- Barducci, Bussi and Parrinello, *Well-Tempered Metadynamics: A Smoothly
  Converging and Tunable Free-Energy Method*, Phys. Rev. Lett. **100**, 020603
  (2008), [doi:10.1103/PhysRevLett.100.020603](https://doi.org/10.1103/PhysRevLett.100.020603).
- The [PLUMED METAD documentation](https://www.plumed.org/doc-v2.10/user-doc/html/_m_e_t_a_d.html)
  describes grid acceleration and the `sigma/5` default-spacing convention.
  PyRPMD implements its own tensor Hermite interpolant and preserves its own
  physical-height HILLS convention; it does not import PLUMED grid files.
