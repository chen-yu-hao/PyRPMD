# F + H₂ author-source static regression, 2026-11-01

**The pinned SW1 and HSW distributions pass their bounded static checks.**
The corrected RPMD rate table and its exact PES/electronic conventions are
still unresolved. This record supplies no thermal-rate or paper-reproduction
claim, and does not replace the earlier
[corrected-reference access record](../fh2-corrected-reference-check-2026-11-01.json).

The original [author catalog](https://www.itheoc.uni-stuttgart.de/research/werner/research/h2f/potential/)
explicitly separates SW1 without spin–orbit correction from HSW with it.
One ordinary public GET per archive succeeded; no authentication or user
data was sent. The downloaded archives, original readmes, parameter files
and compiler products remain separate local assets. This repository holds
their identities and the newly written replay runner.

| Archive | Bytes | SHA256 |
| --- | ---: | --- |
| SW1 | 30,720 | `dcabcc090fb39206d3dd269adf700afde49918f8646665aeccdbb65cdc9441da` |
| HSW | 81,920 | `12a71daa7063a0fa44689236955ac3e1a4e1942b8fb3f7f19ef0324edd6cdee1` |

No source, fit coefficient or unit convention was changed. Ten
predeclared nonsingular triangle geometries and three finite-difference
spacings were evaluated in isolated processes. Both the original
energy-only SW1 routine and its separately implemented analytic-distance
derivative routine execute. HSW's unchanged author verification program
is also compiled and run.

| Check | Actual maximum difference | Fixed limit |
| --- | ---: | ---: |
| SW1 two energy implementations | 2.776 × 10⁻¹⁷ hartree | 5 × 10⁻¹² hartree |
| SW1 analytic distance gradient, all three finite-difference spacings | 4.666 × 10⁻⁹ hartree/bohr | 2 × 10⁻⁷ hartree/bohr |
| H exchange, both energies | 1.443 × 10⁻¹⁵ hartree | 5 × 10⁻¹² hartree |
| H exchange, SW1 gradient | 2.803 × 10⁻¹⁵ hartree/bohr | 5 × 10⁻¹¹ hartree/bohr |
| SW1 component between the two archives | 0 at all ten points | 5 × 10⁻¹² hartree |
| HSW written author decimal targets | 2.850 × 10⁻¹¹ hartree | 5 × 10⁻¹⁰ hartree |

The two HSW targets deserve an explicit precision qualification:

| Distances, bohr | Author-written decimal, hartree | Unchanged compiled result, hartree |
| --- | ---: | ---: |
| 2, 2, 2 | 0.03412633992079376 | 0.034126339949292234 |
| 4, 1.5, 4.5 | 0.001839160872060410 | 0.0018391609005588814 |

Both differences match the default-real rounding of the source's `0.0006`
offset on promotion to double precision. The unchanged author's `verify.f`
also rounds its `ftrue` literals to default real before assignment, so its
printed targets differ from the decimal strings in the source. The
[original stdout](hsw-author-verify.stdout.txt),
[HSW values and rounding](hsw-results.json), and
[rounding diagnostic](legacy-literal-rounding.json) retain both facts.
No energy offset or tolerance was adjusted after seeing the result.

The complete fixed-gate arithmetic, all command arguments, exact executed
source/library identities and timings are in [report.json](report.json).
Build plus validation used **1.960445 child CPU seconds** and **2.332971
wall seconds**, with one CPU affinity and one numerical thread. There are
no trajectories or thermal-rate samples, and the report explicitly retains
`paper_reproduction=false`.

Reproduction instructions and the executable source are in
[benchmarks/fh2_reference](../../../../../benchmarks/fh2_reference/README.md).
The [source manifest](../../../../../benchmarks/fh2_reference/sources.json),
[pre-evaluation protocol](../../../../../benchmarks/fh2_reference/static_protocol.json),
[fetch receipt](fetch-receipt.json), and [SW1 gradients](sw1-results.json)
make the tested scope concrete. No explicit redistribution license file
was present in either archive; third-party source and binary files are
excluded from the publication inventory.

**Limits:** the static adapter uses GNU Fortran's original symbols and
COMMON storage. It is not a production PyRPMD provider or a thread-safe
multi-model engine. Only SW1 analytic distance derivatives were checked;
HSW forces, branch/spline derivative continuity, global PES behavior and
RPMD dynamics remain untested. The two source variants must remain separate
when a corrected thermal-rate reference becomes available.
