"""Input contracts for the device-resident JaxPIP rate route."""

import pytest

from pyrpmd.input_schema import InputError
from pyrpmd.input_tasks import validate_task_sections
from pyrpmd.potentials import validate_periodic_spec


def _rate(provider="jaxpip"):
    return {
        "task": "rate",
        "sampling": {"engine": "torch", "device": "cuda:0"},
        "potential": {"provider": provider, "basis": "basis.json", "model": "model.eqx"},
    }


def test_jaxpip_gpu_rate_contract_is_accepted_without_importing_accelerator():
    validate_periodic_spec(_rate()["potential"])
    assert validate_task_sections(_rate()) == (False, False, False)


def test_jaxpip_gpu_rate_requires_the_direct_jaxpip_provider():
    with pytest.raises(InputError, match="provider='jaxpip'"):
        validate_task_sections(_rate("cabi"))


def test_jaxpip_spec_rejects_periodic_systems_before_model_load():
    with pytest.raises(Exception, match="nonperiodic"):
        validate_periodic_spec(
            _rate()["potential"],
            {"cell": [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]],
             "pbc": [True, True, True]},
        )


@pytest.mark.parametrize("field", ["allow_unvalidated", "max_frames_per_call"])
def test_jaxpip_runtime_guards_are_checked_before_model_load(field):
    config = _rate()["potential"]
    config[field] = "invalid" if field == "allow_unvalidated" else 0
    with pytest.raises(Exception, match="JaxPIP"):
        validate_periodic_spec(config)
