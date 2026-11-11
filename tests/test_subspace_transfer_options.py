"""Combined projection selectors in the transfer-study execution path."""

from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.subspace_theory import plan, transfer


@pytest.mark.parametrize("curvature_only", [False, True])
def test_transfer_matrix_combines_selectors_without_artifact_collisions(curvature_only):
    points = plan.points(curvature_only=curvature_only)
    assert len({point["name"] for point in points}) == len(points)
    assert len(points) <= 256  # GitHub Actions matrix limit.
    selected = [point for point in points if point["family"] == "curvature"
                and point["parameter"] == 0.3 and point["barrier"] == "symmetric"
                and point["temperature"] == 300.0 and point["truncation"] == 0.07]
    assert {(point["curvature_mass"], point["zero_point"]) for point in selected} == {
        (mass, zero_point) for mass in ("none", "small") for zero_point in ("harmonic", "rph", "dvr")
    }
    assert any(point["family"] == "frequency" for point in points) is not curvature_only


def test_transfer_workers_use_the_same_combined_projected_hamiltonian(monkeypatch):
    models = {}

    def quantum(model, *args, **kwargs):
        models["quantum"] = model
        return SimpleNamespace(rate=1.0, diagnostics={})

    def rpmd(model, *args, **kwargs):
        models["rpmd"] = model
        return SimpleNamespace(rate=1.0, log_rate_standard_error=0.0, qtst_rate=1.0,
                               transmission=1.0, transmission_standard_error=0.0, diagnostics={})

    monkeypatch.setattr(transfer, "exact_quantum_rate", quantum)
    monkeypatch.setattr(transfer, "ring_polymer_rate", rpmd)
    common = ("1d", "curvature", 0.3, "symmetric", 500.0, "adiabatic", "small", "rph")
    transfer._quantum(common)
    transfer._rpmd(common + (8, {}, 17))
    exact_model, rpmd_model = models["quantum"], models["rpmd"]
    assert exact_model.provenance["curvature"] == "small"
    assert exact_model.provenance["zero_point"] == "rph"
    assert exact_model.provenance["effective_mass_ratio_min"] < 1.0
    np.testing.assert_array_equal(exact_model.axes[0], rpmd_model.axes[0])
    np.testing.assert_array_equal(exact_model.values, rpmd_model.values)
    assert exact_model.dividing_surface == rpmd_model.dividing_surface
    assert exact_model.reactant_reference == rpmd_model.reactant_reference
