# PyRPMD manuscript for Computer Physics Communications

`pyrpmd-cpc.tex` is a draft Computer Programs in Physics article built on the
CPC `elsarticle` template, with its Program Summary. Build it with

```bash
cd paper
pdflatex pyrpmd-cpc && bibtex pyrpmd-cpc && pdflatex pyrpmd-cpc && pdflatex pyrpmd-cpc
```

| Path | Content |
| --- | --- |
| `pyrpmd-cpc.tex`, `references.bib` | Manuscript and bibliography |
| `figures/` | Figures drawn by `scripts/make_figures.py` |
| `data/` | Compact records behind every number, table and figure |
| `scripts/run_all.sh` | Reruns all calculations (about one hour on four cores) |
| `scripts/collect.py`, `make_figures.py` | Reduce run directories to `data/`, then draw `figures/` |
| `scripts/integrators.py`, `throughput.py`, `parallel.py` | Energy-conservation, backend-timing and worker-scaling measurements |

## Provenance of the numerical results

Calculations run for the manuscript (release 0.2.0, GCC 13.3, Python 3.13,
NumPy 2.5, four x86-64 cores): the five analytic tutorials, the bead-number
scan, the Takahashi–Imada trace convergence, the integrator scan, the Eckart
instanton, the model D + H₂ rate by dRPMD and by Bennett–Chandler (UI and
MBAR), and all timings in the Performance section except the MBAR solver.

Results quoted from validation records already in the repository:

| Result | Record |
| --- | --- |
| H + H₂ / BKMP2 at 300 K, Table 4 and κ(t) in Fig. 4b | `docs/validation/literature-convergence-2026-11-01/` |
| PCET against pyPCET | `docs/validation/pcet-model-2026-10-30/` |
| NRPMD against Richardson and Thoss (2013) | `docs/validation/published-nrpmd-2013-2026-11-01/` |
| Streaming MBAR solver timings | `docs/validation/mbar-native-2026-11-10/` |

## Before submission

The author list, affiliations, corresponding address, acknowledgements and the
generative-AI declaration are marked `TODO(authors)` in the source.
