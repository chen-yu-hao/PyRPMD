# Free-energy surfaces in one to three CV dimensions

These examples reconstruct a centroid free-energy surface from umbrella sampling and check it
against an exact answer.

```bash
pyrpmd run examples/fes/harmonic2d.inp --output runs/fes2d          # 2D centroid surface
pyrpmd run examples/fes/custom_input.py --output runs/fes-custom    # Python-defined 2D CV
pyrpmd run examples/fes/adaptive_harmonic.inp --output runs/fes-adaptive
pyrpmd run examples/fes/harmonic3d.inp --output runs/fes3d          # unified nested input, 3D MBAR
```

The harmonic potential has the exact centroid surface `F(x,y) = 0.001*(x*x+y*y) + constant`
hartree, so you can compare the reconstruction with the reference directly. These short runs are
pilots; the [convergence chapter](../../docs/manual/science.md#converging-a-result) explains how to
scale them up.

## How the surface is built

Every umbrella adds **P times** its physical bias energy to the sum-over-beads Hamiltonian. MBAR
removes the known biases, and the output surface is a density with respect to the declared CV
coordinates, including bin volume. Bins without sampling support are masked; if the window
overlap graph is disconnected, the global surface is masked because component offsets cannot be
identified. Correlated circular-block bootstrap runs independently inside each stream.

The shared `FESResult` layout is `fes.npz`, `fes.csv`, `fes.json` and
`summary.json`, with grid, free energy, `uncertainty`, its statistical kind,
units and masks. Umbrella `fes.h5` additionally carries stream diagnostics.
`harmonic3d.inp` demonstrates `[fes] method="umbrella"` with common nested
sections and `[fes.umbrella]`. The older examples retain supported top-level
sections for compatibility regressions. See [enhanced sampling](../../docs/enhanced-sampling.md)
for MetaD/OPES under the same entry point and their different uncertainty meaning.

## Custom CVs

A Python CV receives a read-only static state view and returns `value[D]` and
`jacobian[D,P,N,3]` in canonical atomic/radian/dimensionless units. The Jacobian is the derivative
of the **whole chosen estimator**, including its bead factors. CVs are static functions of the
coordinates with explicit gradients. Importable factories work in process and MPI workers.
