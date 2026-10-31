# Third party code and scientific references

## RPMDrate

Reference: https://github.com/GreenGroup/RPMDrate

User Cayley and thermal-dRPMD implementation used in this port:
https://github.com/chen-yu-hao/RPMDrate/tree/9a44d6b378d8bf835d67e1b3b7f78b27bdc84a78

Copyright (c) 2012 Joshua W. Allen, William H. Green, Yury V. Suleimanov.
Additional Cayley/thermal-dRPMD changes are attributed to the contributors of
chen-yu-hao/RPMDrate at the pinned commit above. Adaptations are identified in
the relevant PyRPMD modules.

## Caracal

Reference: https://github.com/Trebonius91/Caracal/tree/7f95ce9818b7f6d81d7cb3b634f360373c090820

Copyright (c) 2023 Julien Steffen; Stefan Grimme (QMDFF code).
The rate-workflow organization and recrossing sampling are informed by
`src/calc_rate.f90`, `src/umbrella.f90`, `src/recross.f90` and `src/verlet.f90`.
This package does not redistribute Caracal's bundled QMDFF, xTB or ML models.

## BKMP2 physical H3 benchmark

The optional `benchmarks/h3` suite fetches the original reactive H3 model from
`chen-yu-hao/RPMDrate`, commit
`19ba2602bad5b2145b0d0ba0716d7ba3d90a1003`,
`examples/H+H2/h3_bkmp2.f`. The 61,792-byte source is verified against SHA256
`6fa37803d0d1db4f43e5bc4bfccd5e26585f01116dae9dc4f6d0d28d5d71d237`.

Model authors: Arnold I. Boothroyd, William J. Keogh, Peter G. Martin and
Michael R. Peterson. Reference: *A Refined H3 Potential Energy Surface*,
J. Chem. Phys. 104, 7139–7152 (1996), https://doi.org/10.1063/1.471430.
Author distribution and independent numerical test:
https://www.cita.utoronto.ca/~boothroy/bkmp2.html.

No explicit redistribution license was identified in this PES source.
The fetched Fortran file and its compiled library are **not bundled** with
PyRPMD or its validation artifacts, and are **not relicensed under PyRPMD's
MIT license**. The newly written fetch/build script, C-interoperable wrapper,
adapter and benchmark drivers are PyRPMD code. Source provenance, precise
compiler flags and the build checksum are recorded separately. Users should
acknowledge the PES authors when using this model for scientific work.

## MIT license applying to the above MIT-licensed code

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Scientific citations

- Suleimanov, Allen and Green, *RPMDrate*, CPC 184, 833–840 (2013),
  https://doi.org/10.1016/j.cpc.2012.10.017
- Steffen, *Caracal*, JCTC 19, 5334–5355 (2023),
  https://doi.org/10.1021/acs.jctc.3c00568
- Korol, Bou-Rabee and Miller, *Cayley modification*, JCP 151, 124103 (2019),
  https://doi.org/10.1063/1.5120282
- Suleimanov et al., *Direct RPMD trajectories*, JPCL 9, 2133–2137 (2018),
  https://doi.org/10.1021/acs.jpclett.8b00783


## Optional native Caracal PES generation

`pyrpmd.pesgen` builds a user-supplied Caracal checkout pinned to
`7f95ce9818b7f6d81d7cb3b634f360373c090820`; upstream source and compiled binaries are
not shipped in the PyRPMD wheel. New orchestration, IPC wrapper and reference-data
code is PyRPMD MIT code. The archived safety patch changes charge-input handling
and an allocation, not the original energy equations. Caracal's core copyright
(Julien Steffen; Stefan Grimme for QMDFF) and MIT notice, and all separately bundled
third-party license/header notices, remain applicable. Generated native bundles
include collected notices and identify the full upstream source commit; they do
not relicense dependency code under PyRPMD's license. PySCF is an optional separately
installed dependency. Cite the reference electronic-structure method and software
as well as QMDFF (10.1021/ct500573f), dE-EVB (10.1039/C5CP02580J), TREQ
(10.1063/1.5092589), and Caracal (10.1021/acs.jctc.3c00568), as applicable.
