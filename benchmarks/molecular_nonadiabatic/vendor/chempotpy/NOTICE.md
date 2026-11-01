# ChemPotPy/POTLIB NaH₂ 5G source notice

These files are retained from the authors' public
[ChemPotPy repository](https://github.com/shuyinan/chempotpy), commit
`4716d22e87ae01770cad9ec995c4a02a8e479296`, under its Apache License 2.0.
The original license is provided as `LICENSE`. The Fortran and catalog files
are unmodified; the original copyright, references and source comments remain.

| Local file | Upstream path | SHA-256 |
| --- | --- | --- |
| `NaH2_LEPS_5G_1992_DPEM.f` | `chempotpy/NaH2/NaH2_LEPS_5G_1992_DPEM.f` | `b98e2cb6a7355d868346b2a901441998705f9dead1dd233146285fd618d08f61` |
| `NaH2_catalog.py` | `chempotpy/NaH2/__init__.py` | `42255f8154ba5d1834c65c064f2afec758176de9c0eed924b7db2ce1f98c1c5d` |
| `LICENSE` | `LICENSE` | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |

The original Fortran Git blob SHA is
`fc9a0c3cf6af0707303508fabc11b7dbb02c3877`.

The PyRPMD NumPy translation is in
`src/pyrpmd/potentials/nah2_leps.py`. Its changes are explicit: vectorization,
chain-rule forward differentiation, Cartesian derivative conversion, fixed
atom-order restoration, and Python exceptions for the original singular domain.
It preserves the fit parameters, historical conversion constants, 5G long-range
term, coupling sign and common energy zero shift. It introduces no new fit.

Scientific sources to cite when using this PES:

- P. Halvick and D. G. Truhlar, *J. Chem. Phys.* **96**, 2895–2909 (1992),
  [doi:10.1063/1.461986](https://doi.org/10.1063/1.461986).
- Their erratum, *J. Chem. Phys.* **100**, 4718 (1994),
  [doi:10.1063/1.467282](https://doi.org/10.1063/1.467282).
- Y. Shu, Z. Varga, D. Zhang and D. G. Truhlar, *J. Phys. Chem. A* **127**,
  9635–9640 (2023), [doi:10.1021/acs.jpca.3c05899](https://doi.org/10.1021/acs.jpca.3c05899).
- [POTLIB online library](https://comp.chem.umn.edu/potlib/), including its
  author list and requested attribution.
