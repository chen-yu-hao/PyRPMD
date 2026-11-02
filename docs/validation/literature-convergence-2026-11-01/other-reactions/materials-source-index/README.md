# Pd and Ni literature-source audit — 2026-11-01

**A useful Ni implementation is now available and statically checked, but neither
requested diffusion paper is ready for a matched numerical reproduction.** The
remaining obstacles are specific original model/protocol and reference-data gaps;
no diffusion result has been calculated or substituted in this audit.

## Requested papers and confirmed scope

| Target | Confirmed from its original publisher abstract | Still missing for an executable comparison |
| --- | --- | --- |
| Calvo and Costa, JCTC **6**, 508–516 (2010), [10.1021/ct900554r](https://doi.org/10.1021/ct900554r) | Bulk Pd and clusters up to 512 Pd atoms, 3% loading, explicit many-body interactions, velocity-autocorrelation diffusion estimator | Exact potential implementation; definition of loading; individual geometries and H counts; temperature grid; masses; bead counts; equilibration, thermostat and production settings; numerical diffusion values/units/errors |
| Suleimanov, JPC C **116**, 11141–11153 (2012), [10.1021/jp302453z](https://doi.org/10.1021/jp302453z) | Ni(100), EAM interactions, low-temperature hopping treatment, crossover around 70 K, comparison of lattice motion/recrossing and high-temperature Einstein/Green–Kubo estimates | Original lattice and boundary details, fixed/mobile/quantized partitions, exact temperature grid and masses, bead conventions, sampling/dynamics settings, numerical tables or digitizable original figures with units/errors |

The ACS full-text routes previously returned HTTP 403 and were not retried.
Accessible publisher text remains abstract-level. HAL DOI queries for both
papers and a Pd title query returned zero hits. Author/institutional routes and
the original papers' ResearchGate records did not supply the missing full text.
This records the inspected routes; it does not prove that no lawful copy exists.
See [source-index.json](source-index.json) and the exact public API receipts.

## Ni: a primary-code connection to the 2012 paper

The [official Cambridge OPTIM distribution](https://www-wales.ch.cam.ac.uk/OPTIM/)
provides a public source archive. Its `nimetpot.f90` identifies EAM4 Ni(100)/H,
credits Yury Suleimanov, and explicitly cites the requested 2012 article. The
inspected release's exact source identity is therefore materially stronger than
a generic EAM label or a later citing paper's model description.

| Source component | Bytes | SHA256 |
| --- | ---: | --- |
| [Cambridge public archive](https://www-wales.ch.cam.ac.uk/svn.tar.bz2) | 122758544 | `8463fbbbab3f613c1daa021df99cf89791be3ca61c7f3b9c2f5307e484bf8029` |
| `wales/OPTIM/source/nimetpot.f90` | 43812 | `b294b106cac20038eb941037b63f35dcce374ed40559d68217f5751b90d514a6` |
| Unchanged standalone suffix beginning at byte 6150 | 37662 | `07abd148d739f60a065791f1459bd20ac6434264ffd69af39334937011036384` |
| Archive root `COPYING` (GPL version 3 text) | 35148 | `f51a732c36a182d9489bc3f074b24aff519214acd72207a70aa24ca171e022f2` |

The [NIMET manual](https://www-wales.ch.cam.ac.uk/OPTIM.doc/node36.html) describes
an example with 392 Ni in four layers, lateral periodicity and a fixed bottom
layer. Those are **OPTIM documentation facts**, not verified 2012 RPMD settings.
The downloaded release is later than the paper, so original-executable byte
identity remains unestablished. Source and license bytes are preserved outside
Git. Only metadata, our wrapper/build/check tooling, and compact numerical
receipts are publishable here.

Model labels require care. A
[primary QI review](https://onlinelibrary.wiley.com/doi/10.1155/2012/483504)
distinguishes EAM4 for Ni(100) surface diffusion from EAM6 for subsurface/bulk
transport. The [NIHEAM7 manual](https://www-wales.ch.cam.ac.uk/OPTIM.doc/node37.html)
also documents changes and differences between printed formulas and original
code for later variants. In the downloaded archive, the misleadingly named
`nihEAM6pot.f90` begins with an EAM7 wrapper and is not the selected EAM4 source.
No later-paper EAM6 label overrides the explicit attribution in `nimetpot.f90`.

## Actual static validation, with its physical limits

The isolated [build and replay tooling](../../../../../benchmarks/materials_reference/README.md)
extracts the exact source suffix, without coefficient edits, and fixes `nb=1`.
It uses eV and angstrom directly. Tests use an excluded finite Ni fragment and
explicitly specified boundaries, not a guessed paper slab. The protocol was
written before execution. The full numerical result is
[static-check/report.json](static-check/report.json).

| Check | Observed result |
| --- | --- |
| Three active-coordinate central-difference spacings | Maximum error 3.83e-8 eV/angstrom; finest main-case error at most 2.42e-9 |
| Energy-only versus energy/gradient routes | Identical energy in all four main cases |
| Translation, nonperiodic rotation, periodic lattice shifts | Maximum energy error 1.07e-14 eV; gradient error 6.67e-15 eV/angstrom |
| Mobile/fixed partition checks | Active gradients agree; upstream zero fixed rows retained |
| Smoothing/cutoff probes at 4.9, 5.1, 9.9 and 10.1 angstrom | All predeclared derivative checks pass |
| Standard-collected opt-in tests | 10 passed; no skips in this configured execution |
| Resource use | Build 1.424 CPU s; validation and tests 0.592 CPU s; one thread |

The fixed–fixed Ni pair-energy constant is omitted by the original routine, and
its fixed-Ni derivatives are zero by convention. Different fixed partitions
therefore have different energy zeros. The checks differentiate only active
coordinates and do not treat fixed rows as physical forces. A stale energy-only
comment mentions atomic units; the actual conversion calls are disabled and the
source header/constants use eV/angstrom. The unchanged debug stdout is retained.

An initial command failed because `/usr/bin/time` was absent; no compiler or
scientific calculation started in that attempt. Its receipt is preserved. The
subsequent unchanged build and numerical tests succeeded using Python resource
accounting. No scientific failure was removed, no parameter was tuned, and no
dynamics, diffusivity, mixed-bead or paper-reproduction claim follows from these
static results.

## Pd: two candidate families must not be conflated

NIST distributes a Zhou–Zimmerman–Wong–Hoyt 2008 Pd-H EAM model, but no retrieved
source establishes that Calvo and Costa used it. One bounded request for the
public table bytes returned HTTP 403; no table hash was invented and no retry was
made. The web-rendered table is not an exact byte-level model artifact.

A [different 2008 Calvo author-uploaded article](https://www.researchgate.net/publication/233598801_Desorption_and_Absorption_Dynamics_of_Hydrogen_in_Palladium_Nanoclusters)
describes a many-body-alloy potential derived from Zhong/Li/Tomanek, with Pd–Pd
parameters replaced by Rey and colleagues' values. This is a useful provenance
lead, not proof that the same coefficients, cutoff or model were used in the
2010 RPMD diffusion paper. Its desorption/scattering protocol and dense hydride
clusters must not be substituted for the requested dilute diffusion calculation.

## Existing pilot and next scientific gate

The existing [NiAlH pilot](../../../../../benchmarks/materials/README.md) uses a
different pinned Angelo–Moody–Baskes model, a 27-Ni three-layer slab, one H, and
short P=1/P=4 records at 600 K. Its saved report explicitly declares no converged
diffusivity and no reproduction of Suleimanov 2012. This audit leaves it unchanged.

Before a matched campaign, obtain the original numerical tables/figures and
model/protocol sections through a lawful manuscript, supporting information or
user-supplied copy. Bind the exact potential and geometry, distinguish frozen,
classical and quantized metal cases, register the paper's diffusion estimator
and units, and only then choose an original comparison row. Specify independent
seeds, time/block uncertainty and bead/time-step/finite-size convergence before
looking at the simulated diffusion values. The new EAM4 static checks satisfy
one implementation prerequisite; they do not satisfy those physical gates.

[protocol-gaps.json](protocol-gaps.json) contains explicit nulls for unresolved
paper settings and empty numerical-target lists. [publishable-files.json](publishable-files.json)
is the complete frozen publication inventory, excluding all third-party source,
license files, compiler outputs, libraries and downloaded archives. Their exact
identities and local locations are in [external-assets.json](external-assets.json).
