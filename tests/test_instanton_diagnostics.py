"""Independent determinant checks and auditable production diagnostics."""

import json
import math

import numpy as np
import pytest

from pyrpmd.low_temperature import HarmonicState
from pyrpmd.potentials.base import PESResult
from pyrpmd.tunnelling import tunnelling_splitting


@pytest.mark.parametrize("p", [1, 8, 31, 128])
def test_closed_partition_matches_independent_cyclic_matrix_determinant(p):
    mass, beta, curvature = 1800.0, 2300.0, 0.017
    state = HarmonicState([[0., 0., 0.]], [mass], -0.3, [[curvature]],
                          [[True, False, False]], electronic_degeneracy=2)
    # Build the finite-difference Gaussian integral directly, including the
    # coincident neighbors for P=1 and P=2, independently of the closed form.
    dtau = beta / p
    operator = np.eye(p) * (2 + dtau**2 * curvature / mass)
    for j in range(p):
        operator[j, (j - 1) % p] -= 1
        operator[j, (j + 1) % p] -= 1
    sign, logdet = np.linalg.slogdet(operator)
    assert sign == 1
    diagnostics = state.partition_diagnostics(beta, nbeads=p)
    assert diagnostics["log_partition"] == pytest.approx(math.log(2) - logdet / 2, abs=2e-11)
    assert diagnostics["electronic_energy_hartree"] == -0.3
    assert diagnostics["normalization"] == "dimensionless"
    assert diagnostics["positive_frequencies_au"] == pytest.approx([math.sqrt(curvature/mass)])
    assert diagnostics["discretization"] == "primitive_harmonic"
    json.dumps(diagnostics, allow_nan=False)


def test_gas_partition_reports_geometric_zero_modes_and_rotor_normalization():
    masses = np.array([1500., 2200.])
    distance, curvature, beta = 2.2, 0.04, 1700.
    geometry = np.array([[0., 0., 0.], [distance, 0., 0.]])
    direction = np.array([-1., 0., 0., 1., 0., 0.])
    state = HarmonicState(geometry, masses, -0.2, curvature*np.outer(direction, direction),
                          rigid_body=True, symmetry_number=2, electronic_degeneracy=3)
    reduced_mass = 1 / (1/masses[0] + 1/masses[1])
    diag = state.partition_diagnostics(beta, nbeads=32)
    assert diag["removed_rigid_modes"] == 5
    assert diag["log_translation_partition"] == pytest.approx(1.5*math.log(masses.sum()/(2*math.pi*beta)))
    assert diag["log_rotation_partition"] == pytest.approx(math.log(reduced_mass*distance**2/beta))
    assert diag["log_electronic_degeneracy"] == pytest.approx(math.log(3))
    assert diag["positive_frequencies_au"] == pytest.approx([math.sqrt(curvature/reduced_mass)])
    assert diag["normalization"] == "per_bohr^3"
    assert diag["rigid_bead_powers_cancelled"] is True
    assert diag["log_partition"] == pytest.approx(sum(diag[key] for key in (
        "log_vibrational_partition", "log_translation_partition", "log_rotation_partition",
        "log_electronic_degeneracy")))


def test_partition_remains_finite_for_large_bead_count_and_low_temperature():
    state = HarmonicState([[0., 0., 0.]], [1822.], 0., [[0.08]], [[True, False, False]])
    continuum = state.log_partition(1e9)
    discrete = state.log_partition(1e9, nbeads=10**12)
    assert math.isfinite(discrete)
    assert discrete == pytest.approx(continuum, rel=1e-10)
    with pytest.raises(ValueError, match="boolean"):
        state.partition_diagnostics(10., classical=1)


class AnalyticQuartic:
    def __init__(self):
        self.calls = self.hessian_calls = 0

    def evaluate(self, q):
        self.calls += 1
        x = q[:, 0, 0]
        forces = np.zeros_like(q)
        forces[:, 0, 0] = -0.04*x*(x*x-1)
        return PESResult(0.01*(x*x-1)**2, forces)

    def hessian(self, q):
        self.hessian_calls += 1
        hessian = np.zeros((len(q), 3, 3), dtype=np.float64)
        hessian[:, 0, 0] = 0.04*(3*q[:, 0, 0]**2-1)
        return hessian


def test_splitting_and_harmonic_preparation_use_analytic_provider_and_report_cost():
    pes = AnalyticQuartic()
    mask = np.array([[True, False, False]])
    state = HarmonicState.from_potential(pes, [[-1., 0., 0.]], [1822.], active_mask=mask)
    assert pes.calls == pes.hessian_calls == 1
    assert state.hessian[0, 0] == pytest.approx(0.08)
    omega = math.sqrt(0.08/1822.)
    duration = 28/omega
    q = np.zeros((129, 1, 3))
    q[:, 0, 0] = np.tanh(omega*np.linspace(-duration/2, duration/2, len(q))/2)
    q[0, 0, 0], q[-1, 0, 0] = -1., 1.
    pes.calls = pes.hessian_calls = 0
    report = tunnelling_splitting(pes, [1822.], q, duration, equivalent_wells=True,
                                  active_mask=mask)
    counters = report["optimizer_diagnostics"]
    assert counters["force_calls"] == pes.calls
    assert counters["hessian_provider_calls"] == pes.hessian_calls
    assert counters["finite_difference_hessian_evaluations"] == 0
    assert report["time_step_au"] == duration/128
    assert report["ground_state_duration_parameter"] == pytest.approx(28)
    assert report["endpoint_positive_frequencies_au"] == [[omega], [omega]]
    s0 = 4/3*math.sqrt(2*1822.*0.01)
    analytic = 2*omega*math.sqrt(6*s0/math.pi)*math.exp(-s0)
    assert report["splitting_hartree"] == pytest.approx(analytic, rel=0.006)
    json.dumps(report, allow_nan=False)
