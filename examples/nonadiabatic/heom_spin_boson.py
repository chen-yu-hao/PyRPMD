"""CPU Drude--Lorentz HEOM population reference and two-axis convergence scan.

Run with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python this_file.py.
Every physical quantity is in atomic units.  H=(epsilon*sigma_z+delta*sigma_x)/2
and Q=sigma_z.  To compare with potentials.nonadiabatic_models.DiscreteSpinBoson,
use bias=epsilon/2 and tunnelling=delta/2, and discretize
J(w)=(pi/2)*sum(c_k**2/(m_k*w_k)*dirac(w-w_k)).  The initial bath here is the
free thermal bath, not a displaced surface-equilibrium preparation.
With Q=sigma_z, the surface-to-surface reorganization energy is 4*lambda,
where lambda=integral J(w)/(pi*w) dw is the API's reorganization_energy.

The largest computed hierarchy is a finite comparison, not an exact answer.
This script reports every cutoff and never sets physical_converged=True.
"""

import argparse
from hashlib import sha256
import json
import os
import platform
from time import perf_counter

import numpy as np
import scipy
from scipy.integrate import quad

from pyrpmd.analysis.heom import SpinBosonHEOM


def _exact_dephasing(time, *, epsilon, lam, gamma, beta):
    """Gaussian influence functional integrated over the continuum spectrum."""
    values = []
    cutoff = 2000.
    for t in time:
        def integrand(omega):
            if omega == 0:
                return 2*lam*t*t/(np.pi*beta*gamma)
            return (2*lam*gamma/(np.pi*omega*(omega*omega+gamma*gamma))
                    * 2*np.sin(omega*t/2)**2/np.tanh(beta*omega/2))
        integral, error = quad(integrand, 0., cutoff, epsabs=2e-10, epsrel=2e-9, limit=3000)
        if error > 2e-9:
            raise RuntimeError("independent spectral integral did not reach its error target")
        values.append(.5*np.exp(-4*integral-1j*epsilon*t))
    return np.asarray(values), 2*lam*gamma/(np.pi*cutoff**2)


def run(*, duration=10., points=101, depths=(2, 4, 6, 8), matsubara=(0, 1, 2, 4, 6)):
    """Return JSON-compatible finite-reference curves and convergence diagnostics."""
    if isinstance(duration, bool) or not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be positive and finite")
    if isinstance(points, bool) or not isinstance(points, int) or points < 2:
        raise ValueError("points must be an integer >=2")
    for values, name in ((depths, "depths"), (matsubara, "matsubara")):
        if (len(values) < 2 or any(isinstance(x, bool) or not isinstance(x, int) or x < 0
                                   for x in values) or any(b <= a for a, b in zip(values, values[1:]))):
            raise ValueError(f"{name} must contain at least two strictly increasing nonnegative integers")
    params = dict(epsilon=.4, delta=1., reorganization_energy=.15, cutoff=1., beta=1.,
                  coupling_scale=1., terminator=True)
    time = np.linspace(0., duration, points)
    rho = np.array([[1., 0.], [0., 0.]])
    reference_pair = (depths[-1], matsubara[-1])
    pairs = list(dict.fromkeys([reference_pair] + [(depth, matsubara[-1]) for depth in depths]
                               + [(depths[-1], terms) for terms in matsubara]))
    calculations = {}
    for depth, terms in pairs:
        start = perf_counter()
        model = SpinBosonHEOM(**params, max_depth=depth, matsubara_terms=terms)
        result = model.propagate(rho, time)
        calculations[depth, terms] = (model, result, perf_counter()-start)
    reference = calculations[reference_pair][1]

    def scan(which):
        rows, previous = [], None
        sequence = ([(depth, matsubara[-1]) for depth in depths] if which == "depth"
                    else [(depths[-1], terms) for terms in matsubara])
        for depth, terms in sequence:
            model, result, elapsed = calculations[depth, terms]
            population = result.populations[:, 0]
            rows.append({"max_depth": depth, "matsubara_terms": terms,
                         "ado_count": model.ado_count,
                         "max_population_change_from_previous": (None if previous is None else
                                                                  float(np.max(abs(population-previous)))),
                         "max_population_difference_from_largest":
                             float(np.max(abs(population-reference.populations[:, 0]))),
                         "final_population_0": float(population[-1]),
                         "max_trace_error": result.max_trace_error,
                         "max_hermiticity_error": result.max_hermiticity_error,
                         "min_density_eigenvalue": result.min_density_eigenvalue,
                         "wall_seconds": elapsed, "run_identity": model.run_identity})
            previous = population
        return rows

    # This nontrivial dissipative analytic limit is independent of both HEOM
    # hierarchy assembly and the Matsubara formula used by the implementation.
    dephasing_time = np.array([0., .5, 1., 2., 4.])
    expected, spectral_tail_bound = _exact_dephasing(
        dephasing_time, epsilon=.3, lam=.15, gamma=1., beta=1.)
    dephasing = SpinBosonHEOM(.3, 0., .15, 1., 1., max_depth=8,
                              matsubara_terms=4, terminator=True).propagate(
                                  np.ones((2, 2))/2, dephasing_time)
    experiment = {"schema": "pyrpmd.heom_spin_boson_example.v1", "model": params,
                  "initial_density": rho.tolist(), "initial_time": 0.,
                  "initial_bath": "factorized_free_thermal", "observation_times": time.tolist(),
                  "depths": list(depths), "matsubara_terms": list(matsubara)}
    experiment_identity = sha256(json.dumps(experiment, sort_keys=True, separators=(",", ":"),
                                             allow_nan=False).encode()).hexdigest()
    return {
        "method": "scaled_sparse_Drude_Lorentz_HEOM", "units": "atomic",
        "experiment_identity": experiment_identity, "experiment": experiment,
        "physical_converged": False,
        "interpretation": "largest finite L,K is a comparison; scans do not prove global convergence",
        "model": params, "initial_density": rho.tolist(),
        "initial_bath": "factorized_free_thermal",
        "reference_max_depth": reference_pair[0], "reference_matsubara_terms": reference_pair[1],
        "time_au": time.tolist(), "reference_populations": reference.populations.tolist(),
        "hierarchy_scan": scan("depth"), "matsubara_scan": scan("matsubara"),
        "analytic_dephasing": {
            "max_depth": 8, "matsubara_terms": 4, "terminator": True,
            "max_coherence_absolute_error": float(np.max(abs(dephasing.density[:, 0, 1]-expected))),
            "spectral_integral_tail_bound": spectral_tail_bound,
            "time_au": dephasing_time.tolist(),
            "exact_coherence_real": expected.real.tolist(),
            "exact_coherence_imag": expected.imag.tolist()},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "scipy": scipy.__version__, "platform": platform.platform(),
                    "logical_cpus": os.cpu_count(),
                    "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "unset"),
                    "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "unset")},
        "references": ["https://doi.org/10.1063/5.0011599",
                       "https://doi.org/10.1103/PhysRevResearch.5.013181"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=10.)
    parser.add_argument("--points", type=int, default=101)
    args = parser.parse_args()
    print(json.dumps(run(duration=args.duration, points=args.points), indent=2, allow_nan=False))
