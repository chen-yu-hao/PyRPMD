"""Provider diagnostics and opt-in identity without optional QC dependencies."""

import numpy as np
import pytest

from pyrpmd.adiabatic import (
    AdiabaticPythonPotential, AdiabaticResult, ElectronicFrame,
    validate_adiabatic_result,
)
from pyrpmd.potentials.base import PotentialError


def _result(**kwargs):
    return AdiabaticResult(np.zeros((2, 2)), np.zeros((2, 2, 1, 3)), **kwargs)


def test_failed_frame_reports_status_and_quantum_chemistry_diagnostic():
    result = _result(status=np.array([0, 3]), messages=("", "CASSCF did not converge"))
    # Failed frames may contain NaN sentinels: expose the actual solver failure.
    result.energies[1] = np.nan
    with pytest.raises(PotentialError, match=r"1 \(status=3: CASSCF did not converge\)"):
        validate_adiabatic_result(result, (2, 1, 3))


@pytest.mark.parametrize("messages", ["ab", ("",), ("", 3), np.array(["", ""])])
def test_messages_are_one_string_per_frame(messages):
    with pytest.raises(PotentialError, match="messages"):
        validate_adiabatic_result(_result(messages=messages), (2, 1, 3))


def test_messages_survive_adapter_and_frame_without_aliasing():
    messages = ["cache hit", "fresh calculation"]

    def callback(q, **kwargs):
        return _result(context="opaque", messages=messages)

    q = np.zeros((2, 1, 3))
    frame = ElectronicFrame(q, AdiabaticPythonPotential(callback).evaluate(q))
    messages[0] = "changed"
    assert frame.result.messages == ("cache hit", "fresh calculation")
    assert frame.result.context == "opaque"


def test_legacy_result_positional_context_and_metadata_are_preserved():
    result = AdiabaticResult(np.zeros((2, 2)), np.zeros((2, 2, 1, 3)),
                             None, None, None, "legacy context")
    assert validate_adiabatic_result(result, (2, 1, 3)).context == "legacy context"
    provider = AdiabaticPythonPotential(lambda q, **kwargs: result)
    assert provider.metadata == {
        "provider": "adiabatic_python", "execution_device": "cpu",
        "thread_safety": "context_serial", "continuous_gauge": False,
    }
    with pytest.raises(ValueError, match="to_dict"):
        provider.to_dict()


class _IdentifiedProvider:
    def __init__(self):
        self.configuration = {"type": "test", "options": {"basis": "basis-a"}}

    def to_dict(self):
        return self.configuration

    def evaluate(self, q, **kwargs):
        return _result()


def test_opt_in_identity_detaches_and_captures_units_gauge_and_provider_options():
    source = _IdentifiedProvider()
    adapter = AdiabaticPythonPotential(source)
    original = adapter.to_dict()
    assert adapter.metadata["scientific_identity"] == original
    source.configuration["options"]["basis"] = "basis-b"
    assert original["provider"]["options"]["basis"] == "basis-a"
    assert adapter.to_dict() != original
    changed_units = AdiabaticPythonPotential(source, units={"length": "angstrom"})
    changed_gauge = AdiabaticPythonPotential(source, continuous_gauge=True)
    assert changed_units.to_dict() != adapter.to_dict()
    assert changed_gauge.to_dict() != adapter.to_dict()


@pytest.mark.parametrize("configuration", [None, [], {"value": np.nan}, {"value": object()}])
def test_invalid_scientific_identity_is_rejected(configuration):
    source = _IdentifiedProvider()
    source.configuration = configuration
    with pytest.raises(ValueError, match="identity"):
        AdiabaticPythonPotential(source)


def test_atomic_unit_provider_rejects_accidental_unit_conversion():
    source = _IdentifiedProvider()
    source.metadata = {"requires_atomic_units": True}
    AdiabaticPythonPotential(source)
    with pytest.raises(PotentialError, match="atomic units"):
        AdiabaticPythonPotential(source, units={"energy": "eV"})
    with pytest.raises(PotentialError, match="atomic units"):
        AdiabaticPythonPotential(source, units={"length": "angstrom"})
