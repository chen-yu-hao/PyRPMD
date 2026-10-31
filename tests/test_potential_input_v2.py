"""Resolve public potential and scheduler inputs without loading optional engines."""
from unittest.mock import patch

import numpy as np
import pytest

from pyrpmd.input import InputError, JobSpec
from pyrpmd.potentials import make_potential, validate_periodic_spec
from test_input_cli import small_collision


def test_dynamic_mpi_input_has_one_coordinator_by_default():
    config = small_collision()
    config["parallel"] = {"backend": "mpi", "scheduling": "dynamic",
                          "trajectory_groups": 2, "bead_workers": 2}
    resolved, _ = JobSpec.from_mapping(config).resolve()
    assert resolved["parallel"]["coordinator_ranks"] == 1
    assert resolved["parallel"]["trajectory_groups"] == 2


@pytest.mark.parametrize("version", [True, "2", 0, 3])
def test_invalid_cabi_versions_rejected_before_loading(version):
    config = small_collision()
    config["potential"] = {"provider": "cabi", "abi_version": version, "library": "missing.so"}
    with pytest.raises(InputError, match="abi_version"):
        JobSpec.from_mapping(config).resolve()


def test_periodic_contract_preflight_is_lazy():
    system = {"cell": np.eye(3) * 10, "pbc": [True] * 3}
    for provider in ("cabi", "native", "mace", "nequip", "deepmd", "fairchem", "tblite"):
        validate_periodic_spec({"provider": provider}, system)
    validate_periodic_spec({"provider": "torch", "periodic": True}, system)
    with pytest.raises(Exception, match="periodic=true"):
        validate_periodic_spec({"provider": "torch"}, system)
    with pytest.raises(Exception, match="v1"):
        validate_periodic_spec({"provider": "cabi", "abi_version": 1}, system)
    with pytest.raises(Exception, match="periodic"):
        validate_periodic_spec({"provider": "pyscf"}, system)
    with pytest.raises(Exception, match="partial PBC"):
        validate_periodic_spec({"provider": "deepmd"}, {**system, "pbc": [True, True, False]})


def test_fairchem_registered_name_and_relative_file_are_distinct(tmp_path):
    config = small_collision()
    config["potential"] = {"provider": "fairchem", "model": "uma-s-1p2p1", "params": {"task_name": "omol"}}
    resolved, _ = JobSpec.from_mapping(config, base_directory=tmp_path).resolve()
    assert resolved["potential"]["model"] == "uma-s-1p2p1"
    config["potential"]["model"] = "models/local.pt"
    resolved, _ = JobSpec.from_mapping(config, base_directory=tmp_path).resolve()
    assert resolved["potential"]["model"] == str(tmp_path / "models/local.pt")


@pytest.mark.parametrize("kind", ["mace", "nequip", "deepmd", "fairchem", "tblite", "xtb", "pyscf"])
def test_named_provider_factory_receives_system_identity(kind):
    system = {"symbols": ["D", "H"], "charge": -1, "multiplicity": 2}
    with patch("pyrpmd.potentials.adapters.make_named_potential") as factory:
        make_potential({"provider": kind, "model": "checkpoint", "params": {"setting": 7}}, system)
        kwargs = factory.call_args.kwargs
        assert factory.call_args.args == (kind,)
        assert kwargs["symbols"] == ["D", "H"]
        assert kwargs["charge"] == -1
        assert kwargs["multiplicity"] == 2
        assert kwargs["params"] == {"setting": 7}
        assert kwargs["device"] == "cpu"


def test_deepmd_top_level_units_are_forwarded():
    units = {"positions": "bohr", "energy": "hartree"}
    with patch("pyrpmd.potentials.adapters.make_named_potential") as factory:
        make_potential({"provider": "deepmd", "model": "weights.pt", "units": units})
        assert factory.call_args.kwargs["params"]["units"] == units
    with pytest.raises(ValueError, match="fixed backend units"):
        make_potential({"provider": "pyscf", "units": units})
