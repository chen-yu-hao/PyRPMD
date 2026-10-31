"""Finite-difference diagnostics must retain values from reused PES buffers."""

import numpy as np
import pytest

from pyrpmd.potentials.base import PESResult, check_potential


class ReusedHarmonicPotential:
    """A valid provider whose selected outputs are borrowed until the next call."""

    def __init__(self, shape, *, reuse_energy=True, reuse_forces=True,
                 readonly=False, force_scale=1.0):
        self.energy = np.empty(shape[0])
        self.forces = np.empty(shape)
        self.reuse_energy = reuse_energy
        self.reuse_forces = reuse_forces
        self.readonly = readonly
        self.force_scale = force_scale

    def evaluate(self, q):
        self.energy[:] = 0.5 * np.sum(q * q, axis=(1, 2))
        self.forces[:] = -self.force_scale * q
        energy = self.energy.view() if self.reuse_energy else self.energy.copy()
        forces = self.forces.view() if self.reuse_forces else self.forces.copy()
        if self.readonly:
            energy.flags.writeable = forces.flags.writeable = False
        return PESResult(energy, forces)


@pytest.mark.parametrize("reuse_energy,reuse_forces", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("readonly", [False, True])
def test_force_check_accepts_correct_reused_buffers(reuse_energy, reuse_forces, readonly):
    # Multiple frames and atoms exercise every retained force component.
    q = np.arange(1.0, 19.0).reshape(3, 2, 3) / 10.0
    potential = ReusedHarmonicPotential(
        q.shape, reuse_energy=reuse_energy, reuse_forces=reuse_forces,
        readonly=readonly,
    )
    result = check_potential(potential, q)
    assert result["max_absolute_force_error"] < 2e-10
    assert result["rms_force_error"] < 1e-10
    np.testing.assert_array_equal(q, np.arange(1.0, 19.0).reshape(3, 2, 3) / 10.0)


def test_force_check_rejects_zero_forces_with_reused_energy():
    # If E(q+h) is overwritten by E(q-h), an incorrect zero-force provider
    # appears to pass exactly. Check both reported error statistics analytically.
    q = np.array([[[0.2, 0.4, 0.6]], [[-0.3, -0.5, -0.7]]])
    potential = ReusedHarmonicPotential(q.shape, force_scale=0.0)
    result = check_potential(potential, q)
    assert result["max_absolute_force_error"] == pytest.approx(0.7, abs=1e-10)
    assert result["rms_force_error"] == pytest.approx(np.sqrt(np.mean(q * q)), abs=1e-10)


def test_force_check_accepts_reused_result_object():
    class ReusedResultPotential(ReusedHarmonicPotential):
        def __init__(self, shape):
            super().__init__(shape)
            self.result = PESResult(self.energy, self.forces)

        def evaluate(self, q):
            super().evaluate(q)
            return self.result

    q = np.array([[[0.2, 0.4, 0.6]]])
    result = check_potential(ReusedResultPotential(q.shape), q)
    assert result["max_absolute_force_error"] < 1e-10
