"""Independent literature/scattering checks for the reproducible T09 example."""
import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest


_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "instanton_production_validation", _ROOT/"examples"/"instanton"/"validate_production.py")
validation = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(validation)


@pytest.mark.parametrize("beta_ratio,sci,quantum", [
    (2., 3.50e3, 4.29e3), (4., 3.27e12, 3.96e12),
])
def test_exact_scattering_quadrature_reproduces_published_eckart_table(beta_ratio, sci, quantum):
    # Karandashev et al. (2017), DOI 10.1063/1.4996339, Table II:
    # alpha=pi*sqrt(2*m*a^2*V0)=12. Values are rounded to 3 significant figures.
    height = 12**2/(2*math.pi**2)
    omega = math.sqrt(2*height)
    temperature = omega/(2*math.pi*validation.KB_AU*beta_ratio)
    reference = validation.eckart_reference(height, 1., 1., temperature)
    assert reference["continuum_sci_tunnelling_factor"] == pytest.approx(sci, rel=0.0015)
    assert reference["exact_quantum_tunnelling_factor"] == pytest.approx(quantum, rel=0.0015)
    assert reference["quadrature_relative_error_estimate"] < 1e-9
    assert reference["omitted_tail_relative_upper_bound"] < 1e-15


def test_bead_scan_converges_to_sci_and_preserves_exact_quantum_error():
    report = validation.eckart_scan((32, 64, 128))
    errors = [row["relative_error_vs_continuum_sci"] for row in report["rows"]]
    assert 0 < errors[2] < errors[1] < errors[0]
    assert errors[2] < 0.006
    # Finer beads remove primitive discretization error, not the SCI approximation.
    assert abs(report["rows"][-1]["relative_error_vs_exact_quantum"]) > 0.05
    assert report["reference"]["continuum_sci_over_exact"] == pytest.approx(0.93699117, abs=1e-8)
    json.dumps(report, allow_nan=False)


def test_measured_optimizer_calls_include_finite_differences_and_final_hessian():
    report = validation.optimizer_benchmark(nbeads=32, natoms=2, repeats=1)
    old, new, analytic_old, analytic_new = report["rows"]
    for row in report["rows"]:
        assert row["converged"]
        assert row["gradient_max"] <= report["gradient_tolerance"]
        assert row["action_hbar"] == pytest.approx(old["action_hbar"], abs=1e-9)
        assert row["force_frames"] >= row["force_calls"]
    assert old["force_calls"] > new["force_calls"] > analytic_new["force_calls"]
    assert old["hessian_calls"] == new["hessian_calls"] == 0
    assert analytic_old["hessian_calls"] >= analytic_new["hessian_calls"] >= 2


class MorseRadialOracle:
    def radial(self, distances):
        r, other1, other2 = distances
        assert other1 == other2 == 99.
        exponential = math.exp(-1.2*(r-1.4))
        return 0.2*(1-exponential)**2-0.2, np.array([0.48*(1-exponential)*exponential, 0., 0.])


def test_isolated_h2_adapter_matches_radial_force_and_rotationally_invariant_spectrum():
    potential = validation.IsolatedBKMP2Hydrogen(MorseRadialOracle())
    q = np.array([[[-0.4, -0.3, -0.5], [0.4, 0.3, 0.5]]])
    value = potential.evaluate(q)
    for index in np.ndindex(q.shape):
        plus, minus = q.copy(), q.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        difference = (potential.evaluate(plus).energy[0]-potential.evaluate(minus).energy[0])/(2e-6)
        assert value.forces[index] == pytest.approx(-difference, abs=2e-10)
    np.testing.assert_allclose(value.forces.sum(axis=1), 0., atol=1e-15)
    geometry = np.array([[-0.7, 0., 0.], [0.7, 0., 0.]])
    state = validation.HarmonicState.from_potential(potential, geometry, [2., 2.], rigid_body=True)
    spectrum, imaginary, removed = state.spectrum()
    assert removed == 5
    assert imaginary == 0
    # Reduced mass=1 and Morse curvature=2*D*a^2=0.576.
    assert spectrum[0] == pytest.approx(0.576, rel=1e-7)


def test_atomic_report_failure_preserves_previous_file(tmp_path, monkeypatch):
    path = tmp_path/"validation.json"
    path.write_text('{"previous": true}\n')
    original = path.read_bytes()

    def fail(report, stream, **options):
        stream.write('{"partial":')
        raise OSError("simulated interrupted write")

    with monkeypatch.context() as patch:
        patch.setattr(validation.json, "dump", fail)
        with pytest.raises(OSError, match="interrupted"):
            validation._atomic_json(path, {"result": 1})
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
    validation._atomic_json(path, {"result": 1})
    assert json.loads(path.read_text()) == {"result": 1}


def test_optional_real_bkmp2_stationary_points_match_author_and_literature_oracles():
    library = Path(os.environ.get("PYRPMD_BKMP2_LIBRARY", str(_ROOT/"benchmarks"/"h3"/"build"/"libbkmp2.so")))
    if not library.is_file():
        pytest.skip("optional real BKMP2 library is not built")
    potential, masses, ts, reactants, _, report = validation.h3_system(library)
    assert report["h2_bond_bohr"] == pytest.approx(1.4014719755, abs=1e-7)
    # Independent author-published energy checked by the benchmark source too.
    assert potential.radial([1.757, 1.757, 3.514])[0] == pytest.approx(-0.159176000, abs=5e-10)
    assert report["crossover_temperature_kelvin"] == pytest.approx(345., abs=1.)
    assert ts.spectrum(negative_modes=1)[2] == 5
    assert [state.spectrum()[2] for state in reactants] == [3, 5]
    assert len(masses) == 3


def test_optional_real_bkmp2_full_cartesian_rate_matches_published_instanton():
    library = Path(os.environ.get("PYRPMD_BKMP2_LIBRARY", str(_ROOT/"benchmarks"/"h3"/"build"/"libbkmp2.so")))
    if not library.is_file():
        pytest.skip("optional real BKMP2 library is not built")
    report = validation.h3_scan(library, temperatures=[250.], bead_counts=[32, 64, 128])
    scan = report["temperatures"][0]
    rates = [row["rate_cm3_s"] for row in scan["rows"]]
    assert rates[0] < rates[1] < rates[2]
    # Primary SCI Table III is 4.3e-17 cm^3/s. Compare the same semiclassical
    # method/PES, not the separately tabulated exact quantum rate 3.67e-17.
    assert rates[-1] == pytest.approx(4.3e-17, rel=0.02)
    assert all(row["removed_rigid_modes"] == 5 for row in scan["rows"])
    # This real finite-P orbit has a small negative *phase* curvature. The
    # optimizer must identify and remove that geometric mode explicitly.
    modes = scan["rows"][0]["optimizer_diagnostics"]
    assert modes["raw_negative_modes"] == 2
    assert modes["phase_projection_applied"]
    assert modes["projected_negative_modes"] == 1
    assert scan["numerical_converged"]
    assert not scan["discretization_converged"]
