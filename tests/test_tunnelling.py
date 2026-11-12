"""Open-kink action, determinant and ground-doublet benchmarks."""
import math

import numpy as np
import pytest
from scipy.linalg import eigh_tridiagonal

from pyrpmd.constants import WAVENUMBER_TO_AU
from pyrpmd.instanton import InstantonError
from pyrpmd.potentials.base import PESResult
from pyrpmd.tunnelling import tunnelling_splitting

MASK = np.array([[True, False, False]])
MASS = 1822.


class Quartic:
    def __init__(self, height=0.01, bath=0.0, bend=0.0, offset=0.0):
        self.height, self.bath, self.bend, self.offset = height, bath, bend, offset

    def evaluate(self, q):
        x, y = q[:, 0, 0], q[:, 0, 1]
        u = y-self.bend*(1-x*x)
        force = np.zeros_like(q)
        force[:, 0, 0] = -4*self.height*x*(x*x-1)-2*self.bath*self.bend*x*u
        force[:, 0, 1] = -self.bath*u
        return PESResult(self.height*(x*x-1)**2+0.5*self.bath*u*u+self.offset, force)


def setup(p=128, time_factor=24, *, mass=MASS, height=0.01, bath=0.0, bend=0.0, offset=0.0):
    omega = math.sqrt(8*height/mass)
    duration = time_factor/omega
    q = np.zeros((p+1, 1, 3))
    q[:, 0, 0] = np.tanh(omega*np.linspace(-duration/2, duration/2, p+1)/2)
    q[0, 0, 0], q[-1, 0, 0] = -1, 1
    q[:, 0, 1] = bend*(1-q[:, 0, 0]**2)
    mask = np.array([[True, bool(bath), False]])
    return Quartic(height, bath, bend, offset), [mass], q, duration, mask


def solve(*args, **kwargs):
    pes, mass, path, duration, mask = setup(*args, **kwargs)
    return tunnelling_splitting(pes, mass, path, duration, equivalent_wells=True, active_mask=mask)


def test_quartic_action_and_full_prefactor_converge_to_analytic_instanton():
    omega = math.sqrt(0.08/MASS)
    action = 4/3*math.sqrt(2*MASS*0.01)
    # det'(L_inst)/det(L_harmonic)=1/(12*omega^2), independently known.
    exact = 2*omega*math.sqrt(6*action/math.pi)*math.exp(-action)
    errors = []
    for p in (64, 128, 256):
        result = solve(p, time_factor=28)
        errors.append(abs(result["splitting_hartree"]/exact-1))
        assert result["action_hbar"] == pytest.approx(action, rel=0.002)
        assert result["splitting_wavenumber_cm^-1"] == pytest.approx(result["splitting_hartree"]/WAVENUMBER_TO_AU)
    assert errors[0] > 3*errors[1] > 9*errors[2]
    assert errors[2] < 0.002


def test_duration_and_grid_convergence_are_distinct():
    a = solve(128, time_factor=20)
    b = solve(160, time_factor=25)  # Keep dtau fixed, extend the tails.
    assert a["splitting_hartree"] == pytest.approx(b["splitting_hartree"], rel=0.002)
    assert a["nintervals"] != b["nintervals"]


def test_separable_multidimensional_bath_cancels_in_determinant_ratio():
    single = solve()
    bath = solve(bath=0.02)
    assert single["splitting_hartree"] == pytest.approx(bath["splitting_hartree"], rel=1e-8)
    bent = solve(256, time_factor=40, bath=0.02, bend=0.4)
    assert bent["gradient_max"] < 1e-8
    assert np.max(np.asarray(bent["path_bohr"])[:, 0, 1]) > 0.1
    assert bent["splitting_hartree"] < single["splitting_hartree"]


def test_energy_reference_and_isotope_effect():
    base = solve()
    shifted = solve(offset=0.13)
    heavy = solve(mass=2*MASS)
    assert shifted["splitting_hartree"] == pytest.approx(base["splitting_hartree"], rel=1e-8)
    assert heavy["splitting_hartree"] < 0.1*base["splitting_hartree"]


def test_independent_schrodinger_diagonalization_is_close_not_declared_exact():
    # The leading instanton has genuine semiclassical error, unlike the grid
    # error tested above. A deeper well gives an independent quantum benchmark.
    height = 0.04
    x = np.linspace(-2.5, 2.5, 4001)[1:-1]
    dx = 5/4000
    kinetic = 1/(2*MASS*dx*dx)
    levels = eigh_tridiagonal(2*kinetic+height*(x*x-1)**2,
                             np.full(len(x)-1, -kinetic), select='i', select_range=(0, 1))[0]
    exact = levels[1]-levels[0]
    approximate = solve(256, time_factor=28, height=height)["splitting_hartree"]
    assert approximate == pytest.approx(exact, rel=0.1)
    assert abs(approximate/exact-1) > 0.01  # Do not conflate numerical and physical convergence.


def test_endpoint_hessians_can_be_reused():
    pes, mass, q, duration, mask = setup()
    out = tunnelling_splitting(pes, mass, q, duration, equivalent_wells=True,
                              active_mask=mask, endpoint_hessians=([ [.08] ], [ [.08] ]))
    assert out["splitting_hartree"] == pytest.approx(solve()["splitting_hartree"], rel=1e-6)


def test_distinct_and_equivalent_stationary_wells_are_required():
    pes, mass, q, duration, mask = setup()
    with pytest.raises(ValueError, match="equivalent_wells"):
        tunnelling_splitting(pes, mass, q, duration, active_mask=mask)
    with pytest.raises(ValueError, match="distinct"):
        tunnelling_splitting(pes, mass, np.repeat(q[:1], len(q), axis=0), duration,
                            equivalent_wells=True, active_mask=mask)
    nonstationary = q.copy()
    nonstationary[0, 0, 0] += 0.1
    with pytest.raises(InstantonError, match="stationary"):
        tunnelling_splitting(pes, mass, nonstationary, duration, equivalent_wells=True, active_mask=mask)
    with pytest.raises(InstantonError, match="duration"):
        tunnelling_splitting(pes, mass, q, duration/100, equivalent_wells=True, active_mask=mask)
    with pytest.raises(InstantonError, match="zero mode"):
        tunnelling_splitting(pes, mass, q, duration, equivalent_wells=True)  # Free y/z not a bound model.


def test_asymmetric_wells_cannot_be_accepted_by_assertion_alone():
    class Tilted(Quartic):
        def evaluate(self, q):
            value = super().evaluate(q)
            x = q[:, 0, 0]
            # Bias with zero derivatives at +/-1, but unequal endpoint energies.
            value.energy += 0.001*(3*x-x**3)
            value.forces[:, 0, 0] -= 0.003*(1-x*x)
            return value
    _, mass, q, duration, mask = setup()
    with pytest.raises(InstantonError, match="equal electronic energies"):
        tunnelling_splitting(Tilted(), mass, q, duration, equivalent_wells=True, active_mask=mask)
