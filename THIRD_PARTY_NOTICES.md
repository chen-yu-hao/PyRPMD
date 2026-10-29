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

