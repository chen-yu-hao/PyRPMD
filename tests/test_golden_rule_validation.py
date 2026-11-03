"""The reproducible validation entry really executes the public workflow."""
from pathlib import Path
import runpy

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1]/"examples/instanton/validate_golden_rule.py"


def test_matrix_model_parameters_match_reference_hamiltonian():
    module = runpy.run_path(str(SCRIPT))
    arguments, parameters = module["build_case"](3, (4, 8, 16))
    q = np.zeros((2, 3, 3))
    q[1, :, 0] = 0.3
    values = [surface.evaluate(q) for surface in arguments["potential"]]
    displacement = np.asarray(parameters["bath"]["displacements"])
    mass = np.asarray(parameters["bath"]["masses"])
    omega = np.asarray(parameters["bath"]["frequencies"])
    expected0 = np.sum(0.5*mass*omega**2*(q[:, :, 0]+displacement)**2, axis=1)
    expected1 = np.sum(0.5*mass*omega**2*(q[:, :, 0]-displacement)**2, axis=1)-parameters["bias"]
    reference = arguments["energy_reference_hartree"]
    np.testing.assert_allclose(values[0].energy-reference, expected0, rtol=1e-13, atol=1e-16)
    np.testing.assert_allclose(values[1].energy-reference, expected1, rtol=1e-13, atol=1e-16)


def test_real_validation_case_separates_three_errors_and_preserves_costs():
    module = runpy.run_path(str(SCRIPT))
    result = module["run_case"](2, (4, 8, 16), relative_tolerance=0.2)
    study = result["study"]
    assert study["number_of_optimizations"] == 7
    assert study["numerical_converged"] and not study["physical_converged"]
    assert result["finite_bath_analytic_sci"]["exact_quantum_rate"] is None
    assert abs(result["bead_error_vs_finite_bath_sci"]) < 0.03
    product = np.prod([1+result[name] for name in (
        "bead_error_vs_finite_bath_sci", "bath_error_vs_continuous_sci", "continuous_sci_error_vs_exact_gr")])
    assert product-1 == pytest.approx(result["combined_error_vs_continuous_exact_gr"], abs=1e-13)
    assert result["total_workflow_cost"]["physical_hessian_evaluations"] >= 14
    assert all("path0_bohr" not in row["orbit"] for row in study["rows"])
    assert all(row["orbit"]["coordinate_spectrum_summary"]["minimum"] > 0 for row in study["rows"])


def test_validation_cli_writes_strict_atomic_report(tmp_path, capsys):
    import json
    module = runpy.run_path(str(SCRIPT))
    target = tmp_path/"validation.json"
    module["main"](["--modes", "1", "--intervals", "4", "8", "16",
                    "--relative-tolerance", "0.2", "--skip-classical-limit", "--output", str(target)])
    report = json.loads(target.read_text())
    assert report["schema"] == "PyRPMD.golden-rule-validation.v1"
    assert report["provenance"]["source_sha256"]["src/pyrpmd/golden_rule.py"]
    assert report["classical_limit_case"] is None
    assert json.loads(capsys.readouterr().out)["cases"][0]["modes"] == 1
