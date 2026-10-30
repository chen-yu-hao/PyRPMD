"""Independent equilibrium oracle for the actual umbrella propagation maps.

For a harmonic Hamiltonian, Andersen sampling is a linear Gaussian Markov
chain. Solving its discrete Lyapunov equation tests the sampled centroid
distribution without a stochastic pass/fail tolerance or a long trajectory.
This covers the integrator, physical force, P-scaled bias and momentum
temperature together, including both execution backends.
"""

from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
from scipy.linalg import solve_discrete_lyapunov

from pyrpmd.bias import UmbrellaBias
from pyrpmd.cv import Cartesian, CVStack
from pyrpmd.dynamics import Dynamics, State, native_capabilities
from pyrpmd.potentials import HarmonicPotential
from pyrpmd.potentials.cabi import CABIPotential


def stationary_centroid_variance(beads, scheme, native, dt=0.04, library=None):
    beta, mass, physical_k, bias_k = 1.7, 2.3, 0.7, 1.1
    if library is None:
        potential = HarmonicPotential(physical_k)
    else:
        physical_k = 1.0
        potential = CABIPotential(library)
    dynamics = Dynamics(
        potential, [mass], beta, beads, dt,
        native=native, integrator=scheme,
        loop_backend="native" if library is not None else "python",
    )
    cv = CVStack([Cartesian(0, "x")])
    bias = UmbrellaBias(cv, [0.0], [[bias_k]])

    def propagate(q, p):
        state = State(np.zeros((beads, 1, 3)), np.zeros((beads, 1, 3)))
        state.q[:, 0, 0], state.p[:, 0, 0] = q, p
        dynamics.propagate(
            state, 7, bias=bias, sample_interval=7, record_centroids=False,
        )
        return state.q[:, 0, 0].copy()

    basis, zero = np.eye(beads), np.zeros(beads)
    a = np.column_stack([propagate(e, zero) for e in basis])
    b = np.column_stack([propagate(zero, e) for e in basis])
    # Independently derived sum-beads Hamiltonian thermostat convention.
    noise_covariance = (mass * beads / beta) * b @ b.T
    q_covariance = solve_discrete_lyapunov(a, noise_covariance)
    centroid = np.ones(beads) / beads
    actual = float(centroid @ q_covariance @ centroid)
    continuous = 1.0 / (beta * (physical_k + bias_k))
    # Velocity-Verlet's exact configurational invariant for this linear mode;
    # every supported scheme has the same free centroid map.
    discrete = continuous / (1.0 - dt * dt * (physical_k + bias_k) / (4 * mass))
    if library is not None:
        potential.close()
    return actual, continuous, discrete


@pytest.mark.parametrize("beads", [1, 4, 16])
@pytest.mark.parametrize("scheme", ["cayley-bcb", "normal-mode-bab", "velocity-verlet"])
@pytest.mark.parametrize("native", [False, True])
def test_umbrella_equilibrium_centroid_has_physical_temperature(beads, scheme, native):
    if native and not native_capabilities()["available"]:
        pytest.skip("native extension unavailable")
    actual, continuous, discrete = stationary_centroid_variance(beads, scheme, native)
    assert actual == pytest.approx(discrete, rel=2e-11, abs=1e-12)
    assert abs(actual / continuous - 1.0) < 4e-4


@pytest.mark.parametrize("scheme", ["cayley-bcb", "normal-mode-bab", "velocity-verlet"])
def test_umbrella_timestep_error_decreases_quadratically(scheme):
    coarse, exact, _ = stationary_centroid_variance(8, scheme, False, dt=0.08)
    fine, _, _ = stationary_centroid_variance(8, scheme, False, dt=0.04)
    assert (coarse - exact) / (fine - exact) == pytest.approx(4.0, rel=0.002)


@pytest.fixture(scope="module")
def harmonic_library(tmp_path_factory):
    cc = shutil.which("cc")
    if cc is None or not native_capabilities()["available"]:
        pytest.skip("C compiler and native extension required for fused oracle")
    root = Path(__file__).resolve().parents[1]
    target = tmp_path_factory.mktemp("pmf-equilibrium") / "harmonic.so"
    subprocess.run([
        cc, "-shared", "-fPIC", "-O2", "-std=c11",
        "-I", str(root / "native/include"),
        str(root / "native/examples/harmonic_pes.c"), "-o", str(target),
    ], check=True, capture_output=True)
    return target


@pytest.mark.parametrize("beads", [1, 4, 16])
@pytest.mark.parametrize("scheme", ["cayley-bcb", "normal-mode-bab", "velocity-verlet"])
def test_fused_umbrella_has_correct_equilibrium_measure(harmonic_library, beads, scheme):
    actual, continuous, discrete = stationary_centroid_variance(
        beads, scheme, True, library=harmonic_library,
    )
    assert actual == pytest.approx(discrete, rel=2e-11, abs=1e-12)
    assert abs(actual / continuous - 1.0) < 4e-4
