"""Exercise the executable integration example and its unavailable-result path."""

import json
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "electrochemistry" / "adaptive_constant_potential.py"


def test_example_cli_classical_limit_runs_full_sampling_and_unbiased_dynamics(tmp_path):
    # P=1 is the deliberately inexpensive classical limit. The default P=2
    # example uses the identical API with a signed quantum-nuclear ensemble.
    output = tmp_path / "adaptive-electrode.json"
    completed = subprocess.run(
        [sys.executable, str(EXAMPLE), "--quick", "--beads", "1",
         "--potential-V", "-0.3", "--output", str(output)],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text())
    assert report["beads"] == 1
    assert report["rate_constant"] is None
    assert report["sampling_iterations_have_physical_time"] is False
    summary = report["sampling_summary"]
    assert summary["production_path_frozen"] is True
    assert summary["pilot_samples_used_in_MBAR"] is False
    centers = np.array(report["frozen_centers_bohr"])
    np.testing.assert_allclose(centers[[0, -1]], [[-1, 0], [1, 0]], atol=1e-15)
    assert np.max(np.abs(centers[:, 1])) > 1e-3
    assert report["charge_at_frozen_centers"]["maximum_chemical_potential_residual_hartree"] < 1e-9
    realtime = report["real_time"]
    assert realtime["available"] is True
    assert realtime["bias_applied"] is False
    assert realtime["thermostat_applied"] is False
    assert realtime["same_constant_potential_as_sampling"] is True
    assert realtime["time_au"][-1] == 0.12
    assert len(realtime["signed_centroid_mean_bohr"]) == 13
    assert np.isfinite(realtime["signed_MMST_population_mean"]).all()
    assert np.isfinite(realtime["maximum_absolute_hamiltonian_drift_hartree"])
    ancestry = realtime["parent_sampling_diagnostics"]
    assert ancestry["static_bias"] is False
    assert ancestry["source_static_bias"] is True
    assert len(ancestry["source_window_indices"]) == len(ancestry["source_sample_indices"]) == 8


def test_example_does_not_launch_from_unresolved_reweighting(monkeypatch):
    import pyrpmd.adaptive_nrpmd as adaptive

    def blocked(*args, **kwargs):
        raise AssertionError("an unavailable ensemble must not be used for observables or dynamics")

    unavailable = SimpleNamespace(
        unavailable_reasons=["average_sign_below_threshold"],
        expectation=blocked, resample_parents=blocked,
    )
    result = SimpleNamespace(
        centers=np.array([[-1.0, 0.0], [1.0, 0.0]]), history=[],
        reweighted=unavailable,
    )
    monkeypatch.setattr(adaptive, "run_adaptive_nrpmd_path", lambda *args, **kwargs: result)
    run = runpy.run_path(str(EXAMPLE))["run"]
    report = run(quick=True)
    assert report["signed_equilibrium_CV_mean_bohr"] is None
    assert report["real_time"]["available"] is False
    assert report["real_time"]["unavailable_reasons"] == ["average_sign_below_threshold"]
