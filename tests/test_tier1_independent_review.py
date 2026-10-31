"""Independent regression checks for RPC component-contract boundaries."""

import numpy as np
import pytest

from pyrpmd.potentials import FreePotential, HarmonicPotential
from pyrpmd.potentials.contraction import DifferencePotential, SplitPotential
from pyrpmd.potentials.python import PythonPotential


@pytest.mark.parametrize("metadata,match", [
    ({"coupled_frames": True}, "independent physical"),
    ({"capabilities": {"stateful": True}}, "deterministic"),
    ("not-a-mapping", "metadata must be a mapping"),
    ({"capabilities": "not-a-mapping"}, "capabilities must be a mapping"),
])
def test_rpc_rejects_declared_coupling_and_malformed_capabilities(metadata, match):
    provider = FreePotential()
    provider.metadata = metadata
    with pytest.raises(ValueError, match=match):
        SplitPotential(provider, FreePotential(), 4, 1)
    with pytest.raises(ValueError, match=match):
        DifferencePotential(FreePotential(), provider)


@pytest.mark.parametrize("parameter", ["k", "center"])
def test_rpc_rejects_harmonic_parameters_bound_to_frame_indices(parameter):
    params = {parameter: np.arange(4.)[:, None, None]}
    provider = HarmonicPotential(**params)
    with pytest.raises(ValueError, match="frame axis"):
        SplitPotential(provider, FreePotential(), 4)
    with pytest.raises(ValueError, match="frame axis"):
        SplitPotential(FreePotential(), provider, 4, 1)
    with pytest.raises(ValueError, match="frame axis"):
        SplitPotential(FreePotential(), PythonPotential(provider), 4, 1)


def test_atom_and_cartesian_dependent_harmonic_parameters_remain_supported():
    provider = HarmonicPotential(k=np.array([[.1, .2, .3], [.4, .5, .6]]))
    split = SplitPotential(provider, FreePotential(), 4, 1)
    q = np.ones((4, 2, 3))
    np.testing.assert_array_equal(split.evaluate(q).forces, provider.evaluate(q).forces)
