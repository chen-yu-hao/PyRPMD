"""Regression gates found by the independent Tier 2 scientific review."""

import numpy as np
import pytest

from pyrpmd.potentials.base import PESResult
from pyrpmd.potentials.builtin import HarmonicPotential
from pyrpmd.potentials.contraction import DifferencePotential, SplitPotential
from pyrpmd.potentials.lifecycle import CommitteePotential, DeltaLearningPotential
from pyrpmd.potentials.python import PythonPotential


class _MarkedProvider:
    def __init__(self, marker, location):
        self.calls = 0
        self.metadata = {}
        if location == "attribute":
            setattr(self, marker, True)
        elif location == "metadata":
            self.metadata[marker] = True
        else:
            self.metadata["capabilities"] = {marker: True}

    def evaluate(self, q):
        self.calls += 1
        return PESResult(np.full(len(q), float(self.calls)), np.zeros_like(q))


@pytest.mark.parametrize("marker", ["time_dependent", "history_dependent", "stateful"])
@pytest.mark.parametrize("location", ["attribute", "metadata", "capabilities"])
@pytest.mark.parametrize("wrapper", ["committee", "delta", "difference", "split"])
def test_composed_potential_rejects_dynamic_member_before_evaluation(marker, location, wrapper):
    member = _MarkedProvider(marker, location)
    static = HarmonicPotential()
    with pytest.raises(ValueError, match="deterministic position-only"):
        if wrapper == "committee":
            CommitteePotential([static, member])
        elif wrapper == "delta":
            DeltaLearningPotential(static, member)
        elif wrapper == "difference":
            DifferencePotential(static, member)
        else:
            SplitPotential(static, member, 4, 1)
    assert member.calls == 0


@pytest.mark.parametrize("marker", ["time_dependent", "history_dependent", "stateful"])
@pytest.mark.parametrize("callback_only", [False, True])
def test_python_adapter_cannot_hide_dynamic_member_attributes(marker, callback_only):
    member = _MarkedProvider(marker, "attribute")
    if callback_only:
        def callback(q):
            return member.evaluate(q)
        setattr(callback, marker, True)
        wrapped = PythonPotential(callback)
    else:
        wrapped = PythonPotential(member)
    with pytest.raises(ValueError, match="deterministic position-only"):
        CommitteePotential([HarmonicPotential(), wrapped])
    assert member.calls == 0


def test_static_plain_python_callback_remains_supported():
    def callback(q):
        return PESResult(.5 * np.sum(q*q, axis=(1, 2)), -q)
    committee = CommitteePotential([PythonPotential(callback), HarmonicPotential()])
    q = np.arange(12, dtype=float).reshape(2, 2, 3) / 10
    result = committee.evaluate_uncertainty(q)
    np.testing.assert_allclose(result.mean.forces, -q)
    np.testing.assert_array_equal(result.max_force_std, 0.)


@pytest.mark.parametrize("value", [np.complex64(2+3j), np.complex128(2+3j),
                                  np.complex128(2+0j), 2+3j])
@pytest.mark.parametrize("section,key", [(None, "beta"), ("dynamics", "dt"),
    ("sampling", "nuclear_step"), ("sampling", "centroid_step"),
    ("rate", "relative_tolerance"), ("rate", "min_average_sign")])
def test_thermal_rate_rejects_complex_physical_controls_before_factory(section, key, value):
    from pyrpmd.workflows.nonadiabatic_rate import resolve_nonadiabatic_rate_config

    config = {"masses": [1.], "q": [[0., 0., 0.]], "beta": 2., "nstates": 2,
              "potential": {"factory": "unimportable_review_sentinel:build"}}
    if section is None:
        config[key] = value
    else:
        config[section] = {key: value}
    with pytest.raises(ValueError, match="positive finite number"):
        resolve_nonadiabatic_rate_config(config)
