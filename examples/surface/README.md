# Surface examples

Directed hop rates and free-energy surfaces on fixed periodic substrates, from an analytic model
to a real many-atom ASE potential. The full definitions and scope are in the
[surface catalysis notes](../../docs/surface-catalysis.md).

| File | Purpose |
| --- | --- |
| `diffusion.inp` | 8-bead periodic analytic model: first-order, basin-normalized hop rate with real recrossing |
| `fes2d.inp` | Hop/height 2D free-energy surface on the same model |
| `toy_surface.py` | Analytic periodic Python PES bound explicitly to the cell and PBC, with forces you can check |
| `ase_emt.py` | ASE-built Cu(111) with a Cu adatom: EMT potential, fixed bottom layer and mobile surface layer, FES |

From the repository root:

```bash
python -m pip install -e '.[test,ase]'
pyrpmd run examples/surface/diffusion.inp --output runs/surface-diffusion
pyrpmd run examples/surface/fes2d.inp --output runs/surface-fes2d
pyrpmd run examples/surface/ase_emt.py --output runs/surface-ase
```

> **Scope.** All configurations use the pilot preset and show the workflow, not converged
> physics or quantitative predictions for a target catalyst. For a research calculation, use a
> validated PES and converge the bead number, time step, slab size and thickness, fixed layers,
> basin boundaries, sampling, histograms and the recrossing plateau. The analytic model counts a
> single hop exit and applies no site-degeneracy factor.
