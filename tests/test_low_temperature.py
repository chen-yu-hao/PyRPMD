"""Analytic limits, independent quadrature and invariances, not snapshot rates."""
from dataclasses import replace
import io
import json
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.spatial.transform import Rotation

from pyrpmd.constants import ATOMIC_TIME_S, BOHR_M, KB_AU
from pyrpmd.instanton import InstantonError, RingPolymerAction, instanton_seed, optimize_instanton
from pyrpmd.low_temperature import (HarmonicState, eckart_correction, eckart_transmission,
                                    harmonic_cross_checks, instanton_rate, main,
                                    print_cross_checks)
from pyrpmd.potentials.base import PESResult

MASS = np.array([1822.0])
MASK = np.array([[True, False, False]])
ORIGIN = np.zeros((1, 3))


class Eckart:
    def __init__(self, bath=0.0, offset=0.0):
        self.bath, self.offset, self.calls = bath, offset, 0

    def evaluate(self, q):
        self.calls += 1
        t = np.tanh(0.5*q[:, 0, 0])
        f = np.zeros_like(q)
        f[:, 0, 0] = 0.01*t*(1-t*t)
        f[:, 0, 1] = -self.bath*q[:, 0, 1]
        return PESResult(0.01*(1-t*t)+0.5*self.bath*q[:, 0, 1]**2+self.offset, f)


def state(curvature=0.01, energy=0.0, geometry=ORIGIN, **options):
    return HarmonicState(geometry, MASS, energy, [[curvature]], MASK, **options)


def eckart_orbit(p=64, *, bath=0.0, offset=0.0, masses=MASS):
    pes = Eckart(bath=bath, offset=offset)
    omega = 0.5*np.sqrt(0.02/masses[0])
    temperature = 0.5*omega/(2*np.pi*KB_AU)
    mask = np.array([[True, bool(bath), False]])
    seed = instanton_seed(ORIGIN, [[1., 0., 0.]], p, 2.5)
    result = optimize_instanton(pes, masses, temperature, seed, active_mask=mask,
                               retain_hessian=True)
    action = RingPolymerAction(pes, masses, temperature, active_mask=mask)
    hessian = np.diag([0.01, bath]) if bath else [[0.01]]
    reactant = HarmonicState(ORIGIN, masses, offset, hessian, mask)
    return pes, action, result, reactant


def test_cached_stationary_hessian_requires_only_one_pes_call():
    pes = Eckart()
    ts = HarmonicState.from_potential(pes, ORIGIN, MASS, hessian=[[-0.005]], active_mask=MASK)
    assert pes.calls == 1
    report = harmonic_cross_checks(300., ts, [state()])
    assert pes.calls == 1
    assert report["eckart"]["available"] is False
    with pytest.raises(InstantonError, match="stationary"):
        HarmonicState.from_potential(pes, [[1., 0., 0.]], MASS, active_mask=MASK)


@pytest.mark.parametrize("temperature", [30., 300., 3000.])
def test_tst_and_wigner_have_independent_closed_form(temperature):
    beta = 1/(KB_AU*temperature)
    omega = math.sqrt(0.01/MASS[0])
    barrier_omega = math.sqrt(0.005/MASS[0])
    report = harmonic_cross_checks(temperature, state(-0.005, 0.01), [state()])
    classical = omega/(2*np.pi)*math.exp(-beta*0.01)/ATOMIC_TIME_S
    quantum = 2*math.sinh(beta*omega/2)/(2*np.pi*beta)*math.exp(-beta*0.01)/ATOMIC_TIME_S
    assert report["classical_tst"]["rate"] == pytest.approx(classical)
    assert report["harmonic_tst"]["rate"] == pytest.approx(quantum)
    assert report["wigner"]["rate"] == pytest.approx(quantum*(1+(beta*barrier_omega)**2/24))
    assert report["crossover_temperature_kelvin"] == pytest.approx(barrier_omega/(2*np.pi*KB_AU))
    assert report["physical_converged"] is False


def test_product_zpe_convention_and_nonpositive_eckart_barriers():
    ts, reactant, product = state(-0.005, 0.01), state(), state(0.02, 0.002)
    r = harmonic_cross_checks(300., ts, [reactant], products=[product])
    e = r["eckart"]
    assert e["forward_barrier_hartree"] == pytest.approx(0.01-0.5*np.sqrt(0.01/MASS[0]))
    assert e["reverse_barrier_hartree"] == pytest.approx(0.008-0.5*np.sqrt(0.02/MASS[0]))
    c = harmonic_cross_checks(300., ts, [reactant], products=[product], tunnelling_baseline="classical_tst")
    assert c["eckart"]["forward_barrier_hartree"] == pytest.approx(0.01)
    assert c["eckart"]["reverse_barrier_hartree"] == pytest.approx(0.008)
    assert c["wigner"]["rate"] == pytest.approx(c["classical_tst"]["rate"]*c["wigner"]["kappa"])
    bad = harmonic_cross_checks(300., ts, [reactant], products=[state(energy=0.02)])
    assert not bad["eckart"]["available"]


@pytest.mark.parametrize("height", [0.01, 1e-5])
@pytest.mark.parametrize("energy_ratio", [0.01, 0.5, 1.0, 2.0, 10.0])
def test_eckart_transmission_against_sech_squared_solution(height, energy_ratio):
    # Independent textbook scattering solution for V=height*sech^2(a*x).
    mass, a = MASS[0], 0.5
    energy = height*energy_ratio
    omega = a*math.sqrt(2*height/mass)
    x = math.pi*math.sqrt(2*mass*energy)/a
    d2 = 2*mass*height/a**2-0.25
    barrier_term = (math.cosh(math.pi*math.sqrt(d2))**2 if d2 >= 0
                    else math.cos(math.pi*math.sqrt(-d2))**2)
    expected = math.sinh(x)**2/(math.sinh(x)**2+barrier_term)
    assert eckart_transmission(energy, height, height, omega) == pytest.approx(expected, rel=1e-11)


def test_eckart_integral_threshold_detailed_balance_and_low_temperature():
    omega = 0.5*math.sqrt(0.02/MASS[0])
    f, r, temperature = 0.01, 0.007, 300.
    assert eckart_transmission(f-r, f, r, omega) == 0
    assert eckart_transmission(-1, f, r, omega) == 0
    beta = 1/(KB_AU*temperature)
    integral = quad(lambda e: eckart_transmission(e, f, r, omega)*math.exp(-beta*e),
                    f-r, 0.15, epsabs=1e-15, points=[f])[0]
    result = eckart_correction(temperature, f, r, omega)
    assert result["kappa"] == pytest.approx(beta*math.exp(beta*f)*integral, rel=1e-8)
    assert result["log_kappa"] == pytest.approx(eckart_correction(temperature, r, f, omega)["log_kappa"], abs=1e-10)
    assert eckart_correction(30000., f, r, omega)["kappa"] == pytest.approx(1, abs=0.001)
    deep = eckart_correction(1., f, r, omega)
    assert deep["log_kappa"] > 1000 and deep["kappa"] is None
    json.dumps(deep, allow_nan=False)


@pytest.mark.parametrize("p", [1, 2, 16, 128])
def test_discrete_harmonic_partition_matches_closed_product(p):
    beta = 8000.
    omega = math.sqrt(0.01/MASS[0])
    expected = -math.log(2*math.sinh(p*math.asinh(beta*omega/(2*p))))
    assert state().log_partition(beta, nbeads=p) == pytest.approx(expected, abs=1e-11)


@pytest.mark.parametrize("p", [1, 8, 64])
def test_free_atom_partition_cancels_bead_translation_factors(p):
    atom = HarmonicState(ORIGIN, MASS, 0., np.zeros((3, 3)), rigid_body=True)
    expected = 1.5*math.log(MASS[0]/(2*np.pi*2000.))
    assert atom.log_partition(2000., nbeads=p) == pytest.approx(expected, abs=1e-12)
    assert atom.spectrum()[2] == 3


def molecular_state(q, masses, symmetry=1, electronic=1):
    # Independent rotationally invariant network of equilibrium distance springs.
    h = np.zeros((q.size, q.size))
    for i in range(len(q)):
        for j in range(i):
            u = (q[i]-q[j])/np.linalg.norm(q[i]-q[j])
            g = np.zeros_like(q)
            g[i], g[j] = u, -u
            h += 0.01*np.outer(g.ravel(), g.ravel())
    return HarmonicState(q, masses, 0., h, rigid_body=True,
                         symmetry_number=symmetry, electronic_degeneracy=electronic)


@pytest.mark.parametrize("q,masses,rigid", [
    (np.array([[-1., 0, 0], [1., 0, 0]]), [1822., 1822.], 5),
    (np.array([[0., 0, 0], [2., 0, 0], [0., 2, 0]]), [1822., 2000., 16000.], 6),
])
def test_linear_and_nonlinear_rotors_and_frame_invariance(q, masses, rigid):
    beta = 500.
    molecule = molecular_state(q, masses)
    eigen, _, count = molecule.spectrum()
    assert count == rigid
    mass = np.asarray(masses)
    x = q-np.average(q, axis=0, weights=mass)
    inertia = np.eye(3)*np.sum(mass[:, None]*x*x)-(x.T*mass)@x
    moments = np.linalg.eigvalsh(inertia)
    rotor = (2*moments[-1]/beta if rigid == 5
             else math.sqrt(math.pi*np.prod(moments))*(2/beta)**1.5)
    expected = 1.5*math.log(sum(masses)/(2*np.pi*beta))+math.log(rotor)
    expected -= np.log(beta*np.sqrt(eigen)).sum()
    assert molecule.log_partition(beta, classical=True) == pytest.approx(expected)
    shifted = molecular_state(q@Rotation.from_rotvec([.5, -.3, .2]).as_matrix()+[17, -5, 3], masses)
    assert shifted.log_partition(beta) == pytest.approx(molecule.log_partition(beta), abs=1e-11)
    assert molecular_state(q, masses, 2, 3).log_partition(beta)-molecule.log_partition(beta) == pytest.approx(math.log(1.5))


def test_multidimensional_determinant_recovers_analytic_1d_instanton():
    _, action, result, reactant = eckart_orbit(128)
    output = instanton_rate(action, result, [reactant])
    b = 2*np.pi**2*MASS[0]/0.5**2
    exact_action = 2*np.pi/0.5*np.sqrt(2*MASS[0]*0.01)-b/action.beta
    exact_log_prefactor = 0.5*math.log(2*b/action.beta**3/(2*np.pi))
    ratio = math.exp(output["log_prefactor_au"]-result.action_hbar-(exact_log_prefactor-exact_action))
    assert ratio == pytest.approx(1, rel=0.006)
    assert output["log_prefactor_au"] == pytest.approx(exact_log_prefactor, abs=0.0002)
    assert output["negative_modes"] == 1
    assert output["physical_converged"] is False


def test_separable_transverse_fluctuations_cancel_and_hessian_cache_is_reused():
    pes, action, result, reactant = eckart_orbit(bath=0.007)
    pes.calls = 0
    multi = instanton_rate(action, result, [reactant])
    assert pes.calls == 1  # Stationarity/energy check, no finite differences.
    assert "action_hessian" not in result.to_dict()
    _, a, r, s = eckart_orbit()
    single = instanton_rate(a, r, [s])
    assert multi["log_rate_au"] == pytest.approx(single["log_rate_au"], abs=1e-8)
    assert multi["log_fluctuation_partition"] != pytest.approx(single["log_fluctuation_partition"])
    json.dumps(multi, allow_nan=False)


def test_energy_origin_and_cyclic_phase_invariance():
    _, a, r, s = eckart_orbit()
    expected = instanton_rate(a, r, [s])["log_rate_au"]
    _, a2, r2, s2 = eckart_orbit(offset=0.017)
    assert instanton_rate(a2, r2, [s2])["log_rate_au"] == pytest.approx(expected, abs=1e-9)
    rolled = replace(r, path_bohr=np.roll(r.path_bohr, 7, axis=0), action_hessian=None)
    assert instanton_rate(a, rolled, [s])["log_rate_au"] == pytest.approx(expected, abs=1e-9)


class RadialDoubleWell:
    def evaluate(self, q):
        vector = q[:, 1]-q[:, 0]
        radius = np.linalg.norm(vector, axis=1)
        x = radius-3
        force = np.zeros_like(q)
        force[:, 1] = -0.04*(x*(x*x-1))[:, None]*vector/radius[:, None]
        force[:, 0] = -force[:, 1]
        return PESResult(0.01*(x*x-1)**2, force)


def test_full_molecular_rate_rigid_modes_frame_and_degeneracy():
    pes, masses, p = RadialDoubleWell(), [1822., 1822.], 64
    temperature = 0.5*np.sqrt(0.04/911)/(2*np.pi*KB_AU)
    radius = 3+0.9*np.cos(2*np.pi*(np.arange(p)+0.5)/p)
    seed = np.zeros((p, 2, 3))
    seed[:, 0, 0], seed[:, 1, 0] = -radius/2, radius/2
    r = optimize_instanton(pes, masses, temperature, seed, retain_hessian=True)
    a = RingPolymerAction(pes, masses, temperature)
    s = HarmonicState.from_potential(pes, [[-1., 0, 0], [1., 0, 0]], masses, rigid_body=True)
    out = instanton_rate(a, r, [s], rigid_body=True)
    assert out["removed_rigid_modes"] == 5
    assert out["rate_unit"] == "s^-1"
    rotation = Rotation.from_rotvec([.3, .2, -.4]).as_matrix()
    shift = np.array([8., -3., 5.])
    changed = replace(r, path_bohr=r.path_bohr@rotation+shift, action_hessian=None)
    translated = HarmonicState.from_potential(pes, s.geometry_bohr@rotation+shift, masses, rigid_body=True)
    assert instanton_rate(a, changed, [translated], rigid_body=True)["log_rate_au"] == pytest.approx(out["log_rate_au"], abs=2e-7)
    doubled = instanton_rate(a, r, [s], rigid_body=True, reaction_degeneracy=2, electronic_degeneracy=3, symmetry_number=2)
    assert doubled["rate"] == pytest.approx(3*out["rate"])
    # Two free atoms: verify a bimolecular NUMBER-density rate in m^3/s.
    atoms = [HarmonicState(ORIGIN, [m], 0., np.zeros((3, 3)), rigid_body=True) for m in masses]
    bimolecular = instanton_rate(a, r, atoms, rigid_body=True)
    assert bimolecular["rate_unit"] == "m^3/s"
    assert bimolecular["log_rate_si"]-bimolecular["log_rate_au"] == pytest.approx(math.log(BOHR_M**3/ATOMIC_TIME_S))


def test_prefactor_rejects_underresolved_stale_and_failed_orbits():
    _, a, r, s = eckart_orbit(32)
    with pytest.raises(InstantonError, match="phase mode"):
        instanton_rate(a, r, [s])
    _, a, r, s = eckart_orbit()
    with pytest.raises(InstantonError, match="converged"):
        instanton_rate(a, replace(r, converged=False), [s])
    with pytest.raises(InstantonError, match="temperature"):
        instanton_rate(RingPolymerAction(a.potential, MASS, 2*a.temperature_kelvin, active_mask=MASK), r, [s])
    with pytest.raises(ValueError, match="masses"):
        instanton_rate(a, r, [HarmonicState(ORIGIN, 2*MASS, 0, [[.01]], MASK)])
    with pytest.raises(ValueError, match="action_hessian"):
        instanton_rate(a, r, [s], action_hessian=np.eye(5))
    with pytest.raises(ValueError, match="retain_hessian"):
        optimize_instanton(a.potential, MASS, a.temperature_kelvin, r.path_bohr, active_mask=MASK, retain_hessian=1)


@pytest.mark.parametrize("curvatures,expected", [([0.], "zero mode"), ([-0.01], "negative modes")])
def test_invalid_reactant_modes_are_not_silently_dropped(curvatures, expected):
    with pytest.raises(InstantonError, match=expected):
        state(curvatures[0]).log_partition(1000.)


@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf"), True])
def test_temperature_validation(value):
    with pytest.raises(ValueError):
        harmonic_cross_checks(value, state(-.005, .01), [state()])


def test_report_prints_missing_data_and_logarithmic_underflow(tmp_path, capsys):
    report = harmonic_cross_checks(.001, state(-.005, .01), [state()])
    assert report["classical_tst"]["rate"] == 0
    assert math.isfinite(report["classical_tst"]["log_rate_si"])
    output = io.StringIO()
    print_cross_checks(report, file=output)
    assert "classical_tst" in output.getvalue() and "eckart: unavailable" in output.getvalue()
    assert "weak-tunnelling" in output.getvalue()
    data = {"temperature_kelvin": 300., "transition_state": {
        "geometry_bohr": [[0, 0, 0]], "masses": [1822.], "energy_hartree": .01,
        "hessian": [[-.005]], "active_mask": [[True, False, False]]},
        "reactants": [{"geometry_bohr": [[-1, 0, 0]], "masses": [1822.],
                       "energy_hartree": 0., "hessian": [[.01]], "active_mask": [[True, False, False]]}]}
    path = tmp_path/"cross_checks.json"
    path.write_text(json.dumps(data))
    assert main([str(path), "--json"]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["classical_tst"]["rate_unit"] == "s^-1"
    assert main([str(path)]) == 0
    assert "eckart: unavailable" in capsys.readouterr().out
    path.write_text('{}')
    with pytest.raises(SystemExit) as error:
        main([str(path)])
    assert error.value.code == 2


def test_legacy_one_dimensional_rate_and_path_json_remain_compatible():
    from pyrpmd.instanton import instanton_rate_1d
    pes, a, result, _ = eckart_orbit(64)
    qr = math.sqrt(MASS[0]/(2*math.pi*a.beta))
    old = instanton_rate_1d(pes, MASS, a.temperature_kelvin, result.path_bohr,
                           active_mask=MASK, reactant_partition=qr, normalization="per_bohr")
    assert old["rate_unit"] == "m/s"
    assert old["instanton"]["scope"] == "stationary_action_path_not_a_multidimensional_rate"
    assert "action_hessian" not in old["instanton"]
    assert "rate" not in result.to_dict()
    json.dumps(old, allow_nan=False)
