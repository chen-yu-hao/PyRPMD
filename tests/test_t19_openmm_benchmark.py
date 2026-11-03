"""T19 benchmark-report and optional CUDA water campaign checks."""

import json

import numpy as np
import pytest

from pyrpmd.potentials.openmm import OpenMMPotential


def _cuda_device():
    mm = pytest.importorskip("openmm")
    try:
        platform = mm.Platform.getPlatformByName("CUDA")
    except Exception:
        pytest.skip("OpenMM CUDA platform is unavailable")
    model = mm.System()
    model.addParticle(1.)
    integrator = mm.VerletIntegrator(.001)
    try:
        context = mm.Context(model, integrator, platform, {"DeviceIndex": "0", "Precision": "double"})
    except Exception as exc:
        pytest.skip(f"OpenMM CUDA device is unavailable: {exc}")
    del context, integrator
    return "cuda:0"


def test_cuda_rpmd_virtual_site_pme_matches_reference():
    device = _cuda_device()
    from benchmarks.water.qtip4pf import initial_grid, make_system
    from pyrpmd.constants import ANGSTROM_TO_AU

    q0, box = initial_grid(cells_per_axis=2)
    model = make_system(molecules=8, box_angstrom=box, cutoff_angstrom=.45 * box,
                        ewald_tolerance=1e-7)
    q = q0[None] + np.random.default_rng(41).normal(scale=.002, size=(8, len(q0), 3))
    options = {"cell": np.eye(3) * box * ANGSTROM_TO_AU}
    with OpenMMPotential(model, ["O", "H", "H"] * 8, platform="Reference", **options) as ref:
        expected = ref.evaluate(q)
    with OpenMMPotential(model, ["O", "H", "H"] * 8, device=device,
                         batch_mode="rpmd", **options) as resident:
        result = resident.evaluate(q)
        # Platforms can select different PME grids at the same Ewald tolerance.
        np.testing.assert_allclose(result.energy, expected.energy, atol=2e-7, rtol=2e-6)
        np.testing.assert_allclose(result.forces, expected.forces, atol=2e-7, rtol=2e-6)
        np.testing.assert_allclose(resident.evaluate(q[::-1]).forces,
                                   result.forces[::-1], atol=1e-11, rtol=1e-9)


def test_t19_openmm_missing_cuda_records_every_shape_without_fallback(monkeypatch, tmp_path):
    from benchmarks.efficiency import t19_openmm as bench

    monkeypatch.setattr(bench, "hardware", lambda: {"openmm_platforms": ["Reference", "CPU"]})
    monkeypatch.setattr(bench, "run_case", lambda **_: pytest.fail("CPU fallback was attempted"))
    output = tmp_path / "report.json"
    assert bench.main(["--output", str(output)]) == 2
    report = json.loads(output.read_text())
    assert report["claims_cuda_measurement"] is False
    assert len(report["cases"]) == 9
    assert all(case["status"] == "unmeasured" for case in report["cases"])
    assert {(case["beads"], case["atoms"]) for case in report["cases"]} == {
        (p, n) for p in (8, 32, 64) for n in (64, 512, 4096)}


def test_t19_openmm_reference_benchmark_records_samples_and_errors():
    pytest.importorskip("openmm")
    from benchmarks.efficiency.t19_openmm import run_case

    result = run_case(model_name="harmonic", beads=2, atoms=4, device="cpu",
                      platform="Reference", workers=(1,), warmup=1, repeats=2, rounds=2)
    assert set(result["modes"]) == {"serial", "contexts-1", "rpmd"}
    for mode in result["modes"].values():
        assert mode["status"] == "measured"
        assert len(mode["seconds_per_batch_samples"]) == 4
        assert mode["reference_errors"]["energy_max_abs_error"] == 0.
        assert mode["reference_errors"]["forces_max_abs_error"] == 0.
        assert mode["speedup_over_serial"] > 0.
    assert result["modes"]["rpmd"]["resident_contexts"] == 1
    assert result["modes"]["rpmd"]["resident_rpmd_copies"] == 2

    assert result["mode_orders"][1] == result["mode_orders"][0][::-1]
    assert result["reference_evaluation_seconds"] > 0.
    assert result["model_parameters"]["spring_constant_kj_mol_nm2"] == 12.
