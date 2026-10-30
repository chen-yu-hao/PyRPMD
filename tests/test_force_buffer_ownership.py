"""Force composition and propagation must preserve provider-owned buffers."""

import numpy as np
import pytest

from pyrpmd.dynamics import Dynamics, State, native_capabilities
from pyrpmd.potentials import PESResult


class ConstantProvider:
    def __init__(self):
        self.forces = np.arange(1., 7.).reshape(1, 2, 3)
        self.forces.flags.writeable = False

    def evaluate(self, q):
        return PESResult(np.zeros(1), self.forces)


class ConstantBias:
    def energy_force(self, q):
        return 0., np.full_like(q, 0.5)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("biased,fixed", [(False, False), (True, False), (False, True), (True, True)])
def test_steps_preserve_reused_readonly_provider_forces(native, biased, fixed):
    if native and not native_capabilities()["available"]:
        pytest.skip("native extension unavailable")
    provider = ConstantProvider()
    expected_force = provider.forces.copy()
    if biased:
        expected_force += 0.5
    if fixed:
        expected_force[:, 0] = 0.
    dynamics = Dynamics(
        provider, [2., 2.], beta=1., nbeads=1, dt=0.2, native=native,
        fixed_atoms=(0,) if fixed else (), fixed_positions=np.zeros((2, 3)),
    )
    state = State(np.zeros((1, 2, 3)), np.zeros((1, 2, 3)))
    original = provider.forces.copy()
    for _ in range(2):
        dynamics.step(state, ConstantBias() if biased else None)
    np.testing.assert_array_equal(provider.forces, original)
    assert not provider.forces.flags.writeable
    np.testing.assert_allclose(state.p, 0.4 * expected_force, rtol=1e-14)
    np.testing.assert_allclose(state.q, 0.04 * expected_force, rtol=1e-14)
