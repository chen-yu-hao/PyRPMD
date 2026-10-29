# One or two collective-variable dimensions

Run `pyrpmd run examples/fes/harmonic2d.inp` for a two-dimensional centroid
surface, or `pyrpmd run examples/fes/custom_input.py` for a Python-defined
two-component CV and its full ring-polymer Jacobian. The harmonic potential
has the exact centroid surface `F(x,y) = 0.001*(x*x+y*y) + constant` hartree,
which provides a useful sampling/convergence reference.

Every umbrella adds **P times** its physical bias energy to the sum-beads
Hamiltonian. MBAR removes the known umbrella biases; the output surface is
a density with respect to the declared CV coordinates, including bin volume.
Unsupported bins are masked. If the window overlap graph is disconnected,
the global surface is masked because component offsets cannot be identified.
`fes.h5`, `fes.npz`, and `fes.csv` carry samples, diagnostics, standard errors,
and the mask. Correlated circular-block bootstrap is performed independently
inside each stream. The short examples are pilots and do not claim convergence.

Python CVs receive a read-only static state view. They return `value[D]` and
`jacobian[D,P,N,3]` in canonical atomic/radian/dimensionless units. This is
the derivative of the **whole chosen estimator**, including its bead factors.
Time-dependent CVs and implicit numerical gradients are not part of this
protocol. Importable factories work in process and MPI workers.
