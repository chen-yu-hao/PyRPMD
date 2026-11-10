"""Regression coverage for issue #167 backend transparency and projections."""

import numpy as np

from pyrpmd.block_propagation import cv_descriptor as block_cv_descriptor
from pyrpmd.cv_periodic import FractionalCoordinate
from pyrpmd.propagation import combine_propagation_stats, cv_descriptor
from pyrpmd.runner import _append_backend_warnings, _summary_propagation
from pyrpmd.system import System


def test_fractional_cv_lowering_preserves_affine_origin_for_both_native_abis():
    system = System(
        ("H",), [2.0], [[0.0, 0.0, 0.0]],
        cell=np.array([[5.0, 0.0, 0.0], [0.7, 6.0, 0.0], [0.2, 0.4, 7.0]]),
        pbc=(True, True, True),
    )
    cv = FractionalCoordinate(0, 0, system.cell, system.pbc, system.masses, origin=0.13)
    expected = np.asarray(cv._affine_gradient()).reshape(-1)

    direct = cv_descriptor(cv, system.natoms)["primitive"]
    tagged = block_cv_descriptor(cv, system.natoms)

    np.testing.assert_allclose(direct["linear"], expected)
    assert direct["constant"] == -0.13
    np.testing.assert_allclose(tagged["weights"], expected)
    assert tagged["constant"] == -0.13


def test_summary_propagation_combines_nested_streams_without_mutating_records():
    first = {"native_steps": 3, "python_steps": 2, "fallback_reasons": {"cv": 2}}
    second = {"native_steps": 4, "python_steps": 1, "fallback_reasons": {"cv": 1, "pes": 1}}
    result = _summary_propagation({"streams": [{"propagation": first}, {"propagation": second}]})
    assert result["propagation"] == {
        "native_steps": 7, "python_steps": 3,
        "fallback_reasons": {"cv": 3, "pes": 1},
    }
    assert first["fallback_reasons"] == {"cv": 2}
    assert combine_propagation_stats([]) == {
        "native_steps": 0, "python_steps": 0, "fallback_reasons": {},
    }


def test_large_python_backend_plan_adds_one_actionable_warning():
    config = {"_warnings": []}
    _append_backend_warnings(config, [{
        "stage": "umbrella", "backend": "python", "planned_steps": 1_000_001,
        "reason": "CV kind 'fractional' has no native kernel",
    }])
    assert len(config["_warnings"]) == 1
    assert "1,000,001" in config["_warnings"][0]
    assert "fractional" in config["_warnings"][0]
