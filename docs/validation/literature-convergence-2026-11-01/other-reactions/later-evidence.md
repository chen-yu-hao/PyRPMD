# Later evidence for methane, fluorine, hydroxyl and metal systems

The [initial source inventory](README.md) remains an immutable historical
record. The following independent additions document what became available
later; none supplies a completed thermal-rate or diffusion reproduction.

## H + CH₄

The [independent reference feasibility review](../review/hch4-reference-feasibility/README.md)
checks finite-separation normalization, all four abstraction channels, the
surface metric and the molecular-coordinate Jacobian. Its separate
[bounded pilot design](../review/hch4-reference-feasibility/pilot-design/README.md)
limits exploratory model calls and prohibits publishing a converted rate
from the pilot. The existing CBE2008 electronic checks and failed P128
0.1 fs energy screen remain part of the evidence.

## F + H₂

The [author-source static check](fh2-author-static/README.md) pins the public
SW1 and HSW archives and builds their unchanged routines. Ten fixed
geometries pass energy, SW1 analytic distance-gradient, hydrogen-exchange,
and original HSW verification checks. The small original Fortran literal
rounding differences are recorded without changing coefficients or energy
shifts. HSW has an energy-only check; no production dynamics is claimed.

No explicit redistribution license file was found. Third-party source,
parameters and binaries remain outside Git. The
[corrected-reference audit](fh2-corrected-reference-check-2026-11-01.json)
still lacks the scientific erratum's numerical benchmark and exact
surface/electronic convention; the newly retrieved source code does not
resolve that separate problem.

## OH + H₂

The [source-consistency check](oh-source-consistency/README.md) confirms the
matching 300 K uncorrected NN1 value and preserves the unresolved 1000 K
disagreement between the 2017 and 2018 manuscripts. The exact NN1 code and
weights were not obtained, and no alternative surface was substituted.

## Ni and Pd

The [materials source index](materials-source-index/README.md) identifies an
author-attributed Ni(100)/H EAM4 implementation in the official Cambridge
OPTIM distribution. Its header cites the 2012 paper. The unchanged
single-bead routines pass static checks of gradients, invariances,
fixed/mobile Ni partitions and cutoff behavior, with ten collected tests.
The external source, license and compiled artifact have explicit identities.

This does not supply a Ni diffusion coefficient. The complete original
protocol and numerical targets remain unresolved; current OPTIM slab
examples are not assumed to be the paper's configuration. The earlier
NiAlH pilot uses a different potential. The Pd audit records model leads
without substituting an unmatched public Pd-H table. Both original
numerical target lists remain empty.
