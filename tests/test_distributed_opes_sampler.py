"""Thermostat-clock parity for distributed OPES propagation."""
from __future__ import annotations

import numpy as np

from pyrpmd.workflows.opes import _DistributedFESSampler


class Delegate:
    thermostat = "andersen"
    operator_checkpoint = False

    def __init__(self):
        self.steps = 0

    def step(self, dynamics, state, bias):
        self.steps += 1
        state.step += 1


class Dynamics:
    def __init__(self):
        self.refreshes = 0

    def sample_momenta(self):
        self.refreshes += 1
        return np.full((2,), 3.0)


class State:
    def __init__(self, step):
        self.step = step
        self.p = np.zeros(2)


def test_distributed_opes_preserves_serial_andersen_refresh_clock():
    delegate = Delegate()
    sampler = _DistributedFESSampler(delegate, 3)
    dynamics = Dynamics()
    due = State(0)
    sampler.step(dynamics, due, None)
    np.testing.assert_array_equal(due.p, [3.0, 3.0])
    assert dynamics.refreshes == 1

    not_due = State(1)
    sampler.step(dynamics, not_due, None)
    np.testing.assert_array_equal(not_due.p, [0.0, 0.0])
    assert dynamics.refreshes == 1
    assert delegate.steps == 2
