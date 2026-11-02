"""RPC cell derivatives, centroid pressure and component ownership contracts."""

import numpy as np
import pytest

from pyrpmd.barostat import BZPBarostat
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.potentials import FreePotential, HarmonicPotential, PESResult, PotentialError
from pyrpmd.potentials.contraction import DifferencePotential, SplitPotential
from pyrpmd.system import System


class CellPotential:
    """Pair quartic + fractional cosine + A/V, with an analytic strain virial.

    The fractional term is affine invariant but has nonzero Cartesian forces;
    replacing its cell derivative by q.F would therefore fail these checks.
    """

    def __init__(self, pair=.03, cosine=.17, cell_energy=.8):
        self.pair, self.cosine, self.cell_energy = pair, cosine, cell_energy
        self.cell = np.array([[3.1, .2, -.1], [.1, 3.8, .2], [0., .3, 4.2]])

    def set_cell(self, cell):
        self.cell = np.array(cell, copy=True)

    def evaluate(self, q):
        inv = np.linalg.inv(self.cell)
        volume = abs(np.linalg.det(self.cell))
        phase = 2 * np.pi * (q @ inv)
        r = q[:, 0] - q[:, 1]
        r2 = np.sum(r*r, axis=1)
        pair_force = -self.pair * r2[:, None] * r
        force = -2 * np.pi * self.cosine * np.sin(phase) @ inv.T
        force[:, 0] += pair_force
        force[:, 1] -= pair_force
        energy = (self.pair / 4 * r2*r2
                  + self.cosine * np.sum(1 - np.cos(phase), axis=(1, 2))
                  + self.cell_energy / volume)
        virial = (np.einsum("ba,bc->bac", pair_force, r)
                  + np.eye(3)[None] * self.cell_energy / volume)
        return PESResult(energy, force, virial=virial)


@pytest.mark.parametrize("p,c", [(1, 1), (8, 1), (8, 2), (8, 3), (8, 4), (8, 8), (9, 5)])
def test_rpc_virial_matches_all_affine_cell_derivatives(p, c):
    fast, slow = CellPotential(), CellPotential(.05, .07, .6)
    split = SplitPotential(fast, slow, p, c)
    cell = fast.cell.copy()
    split.set_cell(cell)
    q = np.random.default_rng(146).normal(scale=.2, size=(p, 2, 3))
    value = split.evaluate(q)
    expected = (fast.evaluate(q).virial.sum(axis=0)
                + p/c * slow.evaluate(split.contraction.contract(q)).virial.sum(axis=0))
    np.testing.assert_allclose(value.virial.sum(axis=0), expected, atol=2e-15)
    finite_difference = np.empty((3, 3))
    epsilon = 2e-6
    for a, b in np.ndindex(3, 3):
        energies = []
        for sign in (-1, 1):
            transform = np.eye(3)
            transform[a, b] += sign * epsilon
            split.set_cell(cell @ transform.T)
            energies.append(split.evaluate(q @ transform.T).energy.sum())
        finite_difference[a, b] = -(energies[1] - energies[0]) / (2 * epsilon)
    split.set_cell(cell)
    np.testing.assert_allclose(value.virial.sum(axis=0), finite_difference, rtol=3e-7, atol=3e-9)


@pytest.mark.parametrize("c", [1, 2, 3, 8])
def test_rpc_bzp_pressure_matches_centroid_only_cell_derivative(c):
    p, beta = 8, 3.7
    fast, slow = CellPotential(), CellPotential(.02, .06, .5)
    split = SplitPotential(fast, slow, p, c)
    cell = fast.cell.copy()
    system = System(("H", "H"), [2., 5.], [[.2, .3, .4], [.8, .5, .6]], cell=cell)
    piston = BZPBarostat(split, system, beta, p, .01, 2.)
    rng = np.random.default_rng(17)
    q = system.geometry[None] + rng.normal(scale=.14, size=(p, 2, 3))
    state = State(q, rng.normal(size=q.shape))
    pressure = piston.pressure(state)
    epsilon = 2e-6
    energies = []
    for sign in (-1, 1):
        scale = np.exp(sign * epsilon)
        split.set_cell(cell * scale)
        shifted = q + q.mean(axis=0, keepdims=True) * (scale - 1)
        energies.append(split.evaluate(shifted).energy.sum())
    split.set_cell(cell)
    derivative = -(energies[1] - energies[0]) / (2 * epsilon)
    kinetic = p * np.sum(state.p.mean(axis=0)**2 / system.masses[:, None])
    expected = (kinetic + derivative) / (3*p*abs(np.linalg.det(cell)))
    assert pressure == pytest.approx(expected, rel=5e-8, abs=1e-11)
    assert split.metadata["virial_scope"] == "rpc_contracted_hamiltonian"


def test_full_bead_target_minus_reference_recovers_target_virial():
    reference, target = CellPotential(), CellPotential(.08, .19, 1.4)
    split = SplitPotential(reference, DifferencePotential(target, reference), 6)
    cell = reference.cell * 1.13
    split.set_cell(cell)
    np.testing.assert_array_equal(target.cell, cell)
    np.testing.assert_array_equal(reference.cell, cell)
    q = np.random.default_rng(137).normal(size=(6, 2, 3))
    result = split.evaluate(q)
    expected = target.evaluate(q)
    np.testing.assert_allclose(result.virial, expected.virial, atol=1e-13)
    np.testing.assert_allclose(result.energy, expected.energy, atol=1e-13)
    np.testing.assert_allclose(result.forces, expected.forces, atol=1e-13)


def test_contracted_virial_does_not_alias_shared_provider_buffers():
    storage = np.empty((8, 3, 3))

    class Reusing(CellPotential):
        def evaluate(self, q):
            result = super().evaluate(q)
            storage[:len(q)] = result.virial
            result.virial = storage[:len(q)]
            return result

    fast, slow = Reusing(), Reusing(.02, .06, .5)
    q = np.random.default_rng(147).normal(size=(8, 2, 3))
    split = SplitPotential(fast, slow, 8, 3)
    fast_value = fast.evaluate(q).virial.copy()
    slow_value = slow.evaluate(split.contraction.contract(q)).virial.copy()
    expected = fast_value + slow_value.mean(axis=0)
    result = split.evaluate(q)
    slow.evaluate(q * 1.2)
    np.testing.assert_allclose(result.virial, expected, atol=1e-14)
    assert not np.shares_memory(result.virial, storage)


def test_missing_component_virial_stays_missing_and_cannot_drive_pressure():
    class NoVirial(CellPotential):
        def evaluate(self, q):
            result = super().evaluate(q)
            result.virial = None
            return result

    for fast, slow in ((NoVirial(), CellPotential()), (CellPotential(), NoVirial())):
        split = SplitPotential(fast, slow, 4, 1)
        system = System(("H", "H"), [2., 3.], np.zeros((2, 3)), cell=fast.cell)
        q = np.zeros((4, 2, 3))
        assert split.evaluate(q).virial is None
        piston = BZPBarostat(split, system, 3., 4, .01, 2.)
        with pytest.raises(PotentialError, match="virial"):
            piston.pressure(State(q, q.copy()))


def test_cell_setter_preflight_and_failed_update_restore_components():
    fast = CellPotential()
    original = fast.cell.copy()
    split = SplitPotential(fast, HarmonicPotential(), 4, 1)
    with pytest.raises(PotentialError, match="every.*set_cell"):
        split.set_cell(original * 1.1)
    np.testing.assert_array_equal(fast.cell, original)

    class FailsOnce(CellPotential):
        fail = False

        def set_cell(self, cell):
            super().set_cell(cell)
            if self.fail:
                self.fail = False
                raise RuntimeError("cell failure")

    slow = FailsOnce()
    split = SplitPotential(fast, slow, 4, 1)
    split.set_cell(original)
    slow.fail = True
    with pytest.raises(RuntimeError, match="cell failure"):
        split.set_cell(original * 1.1)
    np.testing.assert_array_equal(fast.cell, original)
    np.testing.assert_array_equal(slow.cell, original)


def test_rpc_npt_runs_with_a_physical_virial():
    provider = SplitPotential(FreePotential(), FreePotential(), 4, 1)
    system = System(("H", "H"), [2., 3.], [[.2, .3, .4], [.8, .5, .6]], cell=np.eye(3)*3)
    dynamics = Dynamics(provider, system.masses, 4., 4, .01, native=False, seed=61)
    piston = BZPBarostat(provider, system, 4., 4, .01, 2.)
    state = State(np.repeat(system.geometry[None], 4, axis=0), dynamics.sample_momenta())
    for _ in range(10):
        piston.step(state, dynamics)
    assert state.step == 10
    assert np.isfinite(piston.pressure(state))
    assert piston.volume > 0


def test_nested_delta_rpc_npt_input_preserves_pressure_scope(tmp_path):
    from pyrpmd.constants import KB_AU
    from pyrpmd.input import JobSpec
    from pyrpmd.workflows.materials import run_materials

    raw = {
        "task": "pimd", "temperature": 1/(4*KB_AU), "beads": 4,
        "seed": 17, "native": False,
        "system": {"symbols": ["H", "H"], "masses": [2., 3.],
                   "geometry": [[.2, .3, .4], [.8, .5, .6]],
                   "cell": (np.eye(3)*3).tolist()},
        "potential": {"provider": "delta", "reference": {"provider": "free"},
                      "correction": {"provider": "free"}, "contracted_beads": 1},
        "dynamics": {"integrator": "normal-mode-bab", "timestep": .01},
        "sampling": {"equilibration_steps": 2, "production_steps": 8,
                     "sample_interval": 2, "thermostat": "pile-l"},
        "barostat": {"method": "bzp", "pressure": .01, "time_constant": 2.},
        "output": {"checkpoint_steps": 4},
    }
    config, system = JobSpec.from_mapping(raw).resolve()
    result = run_materials(config, system, tmp_path)
    assert result["estimator_scope"] == "rpc_contracted_hamiltonian"
    assert result["ring_polymer_contraction"]["pressure_scope"] == "rpc_contracted_hamiltonian"
    assert result["ring_polymer_contraction"]["per_bead_physical_virials_available"] is False
    assert result["ensemble"] == "npt"
