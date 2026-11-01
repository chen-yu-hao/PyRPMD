"""Public input reaches the same canonical sampling validator as direct calls."""

import copy
from pathlib import Path
import tomllib

import pytest

from pyrpmd.input import InputError, JobSpec
from pyrpmd.units import parse_quantity
from pyrpmd.workflows.sampling import fes_sampler


def configuration():
    root = Path(__file__).resolve().parents[1]
    value = tomllib.loads((root / "examples/fes/harmonic2d.inp").read_text())
    value["sampling"].pop("integrator", None)
    value["sampling"].pop("gamma_centroid", None)
    return value


@pytest.mark.parametrize("mode,params", [
    ("nhc", {"tau": "20 fs", "chain_length": 4}),
    ("nose-hoover-chain", {"tdamp": "20 fs", "substeps": 2}),
    ("svr", {"tau": "30 fs"}),
    ("langevin", {"gamma": "0.01 fs^-1"}),
    ("pile-g", {"tau": "15 fs", "pile_scale": 1.2}),
    ("ffl", {"gamma": "0.01 fs^-1"}),
    ("gle", {"gle": {"A": [[1.0]], "C": [[1.0]], "A_unit": "fs^-1"}}),
    ("nm-gle", {"gle": {"A": [[1.0]], "C": [[1.0]]}}),
    ("nm-gle-g", {"gle": {"A": [[1.0]], "C": [[1.0]]}, "tau": "30 fs"}),
])
def test_fes_input_accepts_canonical_operator_parameters_without_mutation(mode, params):
    config = configuration()
    config["sampling"].update(thermostat=mode, thermostat_params=params)
    config["parallel"] = {"trajectory_batch_size": 3}
    before = copy.deepcopy(config)
    resolved, _ = JobSpec.from_mapping(config).resolve()
    assert config == before
    expected = "obcbo" if mode in ("nhc", "nose-hoover-chain") else "bcocb"
    assert resolved["sampling"]["integrator"] == expected
    sampler = fes_sampler(resolved)
    assert sampler.operator_checkpoint
    assert sampler.thermostat_spec["mode"] == ("nhc" if mode == "nose-hoover-chain" else mode)
    assert resolved["sampling"]["thermostat_params"] == params
    if mode == "svr":
        assert sampler.thermostat_spec["tau"] == parse_quantity("30 fs", "time")


@pytest.mark.parametrize("mode", ["none", "trpmd", "piglet"])
def test_input_rejects_noncanonical_umbrella_measure(mode):
    config = configuration()
    config["sampling"]["thermostat"] = mode
    with pytest.raises(InputError, match="canonical"):
        JobSpec.from_mapping(config).resolve()


def test_native_pile_input_is_validated_by_capability_at_execution():
    config = configuration()
    config["sampling"]["thermostat"] = "pile-l"
    config["dynamics"]["loop_backend"] = "native"
    config["native"] = True
    resolved, _ = JobSpec.from_mapping(config).resolve()
    assert fes_sampler(resolved).operator_checkpoint
    config["native"] = False
    with pytest.raises(InputError, match="cannot disable"):
        JobSpec.from_mapping(config).resolve()
