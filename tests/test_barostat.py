"""Independent pressure, ensemble, integration and restart checks for BZP."""

import copy
import importlib.util

import numpy as np
import pytest

from pyrpmd.barostat import BZPBarostat, validate_bzp_snapshot
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.potentials import FreePotential, PESResult, PotentialError, PythonPotential
from pyrpmd.potentials.cache import StepEvaluationCache
from pyrpmd.system import System


class FractionalWell:
    """Periodic cosine plus A/V: affine strain leaves cosines unchanged."""

    metadata = {"periodic": True}

    def __init__(self, height=.15, cell_energy=.3):
        self.height, self.cell_energy = height, cell_energy
        self.calls = 0
        self.fail_at = None

    def evaluate(self, q, *, cell, pbc):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("injected force failure")
        volume = abs(np.linalg.det(cell))
        inverse = np.linalg.inv(cell)
        phase = 2 * np.pi * (q @ inverse)
        energies = self.height * np.sum(1 - np.cos(phase), axis=(1, 2)) + self.cell_energy / volume
        forces = -2 * np.pi * self.height * np.sin(phase) @ inverse.T
        virials = np.repeat((np.eye(3) * self.cell_energy / volume)[None], len(q), axis=0)
        return PESResult(energies, forces, virial=virials)


def make(beads=4, dt=.02, *, free=False, friction=0.0, seed=91):
    cell = np.array([[3., 0, 0], [.3, 3.3, 0], [.1, -.2, 3.7]])
    system = System(("H", "H"), [2., 3.], [[.3, .2, -.1], [.8, -.4, .6]], cell=cell)
    obj = None if free else FractionalWell()
    provider = StepEvaluationCache(FreePotential() if free else PythonPotential(obj, cell=cell))
    dynamics = Dynamics(provider, system.masses, 4., beads, dt, native=False, seed=seed,
                        loop_backend="python", integrator="normal-mode-bab")
    barostat = BZPBarostat(provider, system, 4., beads, pressure_au=.02, tau_au=2., friction_au=friction)
    rng = np.random.default_rng(41)
    q = np.repeat(system.geometry[None], beads, axis=0) + rng.normal(0, .04, (beads, 2, 3))
    state = State(q, dynamics.sample_momenta())
    barostat.momentum = .2
    return barostat, dynamics, state, system, obj


def test_centroid_pressure_matches_independent_cell_energy_derivative():
    barostat, dynamics, state, _, _ = make()
    original = barostat.snapshot()
    epsilon = 2e-6
    centroid = state.q.mean(axis=0)
    energies = []
    for sign in (-1, 1):
        scale = np.exp(sign * epsilon)
        barostat.restore({**original, "cell": (np.array(original["cell"]) * scale).tolist()})
        q = state.q + centroid[None] * (scale - 1)
        energies.append(dynamics.potential.evaluate(q).energy.sum())
    barostat.restore(original)
    minus_log_derivative = -(energies[1] - energies[0]) / (2 * epsilon)
    kinetic = state.q.shape[0] * np.sum(state.p.mean(axis=0)**2 / dynamics.masses[:, None])
    expected = (kinetic + minus_log_derivative) / (3 * state.q.shape[0] * barostat.volume)
    assert barostat.pressure(state) == pytest.approx(expected, rel=1e-9, abs=1e-10)


@pytest.mark.parametrize("beads", [1, 4, 16])
def test_free_gas_pressure_is_bead_normalized(beads):
    barostat, _, state, _, _ = make(beads, free=True)
    velocity = np.array([[.3, -.1, .2], [-.2, .4, .1]])
    state.p[:] = velocity * barostat.masses[:, None]
    expected = np.sum(barostat.masses[:, None] * velocity**2) / (3 * barostat.volume)
    assert barostat.pressure(state) == pytest.approx(expected)


def test_dilation_scales_centroids_and_preserves_internal_modes():
    barostat, _, state, _, _ = make()
    q, p, cell = state.q.copy(), state.p.copy(), barostat.cell
    dt = .1
    factor = np.exp(dt * barostat.momentum / barostat.mass)
    barostat._dilate(state, dt)
    np.testing.assert_allclose(state.q.mean(axis=0), q.mean(axis=0) * factor, atol=1e-14)
    np.testing.assert_allclose(state.p.mean(axis=0), p.mean(axis=0) / factor, atol=1e-14)
    np.testing.assert_allclose(state.q - state.q.mean(axis=0), q - q.mean(axis=0), atol=1e-14)
    np.testing.assert_allclose(state.p - state.p.mean(axis=0), p - p.mean(axis=0), atol=1e-14)
    np.testing.assert_allclose(barostat.cell, cell * factor, atol=1e-14)


@pytest.mark.parametrize("integrator", ["cayley-bcb", "normal-mode-bab", "velocity-verlet"])
def test_deterministic_map_reversibility(integrator):
    barostat, dynamics, state, _, _ = make(dt=.005)
    dynamics = Dynamics(dynamics.potential, dynamics.masses, dynamics.beta, dynamics.nbeads,
                        dynamics.dt, integrator=integrator, native=False, loop_backend="python")
    start, piston = state.copy(), barostat.snapshot()
    for _ in range(80):
        barostat.step(state, dynamics, thermostat=None)
    state.p *= -1
    barostat.momentum *= -1
    for _ in range(80):
        barostat.step(state, dynamics, thermostat=None)
    np.testing.assert_allclose(state.q, start.q, atol=2e-12, rtol=1e-12)
    np.testing.assert_allclose(state.p, -start.p, atol=3e-12, rtol=1e-12)
    np.testing.assert_allclose(barostat.cell, piston["cell"], atol=2e-12, rtol=1e-12)
    assert barostat.momentum == pytest.approx(-piston["momentum"], abs=3e-12)


def test_extended_energy_error_is_second_order():
    errors = []
    for dt in (.02, .01, .005):
        barostat, dynamics, state, _, _ = make(dt=dt)
        initial = barostat.extended_energy(state, dynamics)
        excursion = 0.
        for _ in range(round(2.0 / dt)):
            barostat.step(state, dynamics, thermostat=None)
            excursion = max(excursion, abs(barostat.extended_energy(state, dynamics) - initial))
        errors.append(excursion)
    assert 3.8 < errors[0] / errors[1] < 4.2
    assert 3.8 < errors[1] / errors[2] < 4.2


@pytest.mark.parametrize("thermostat", ["pile-l", "andersen"])
def test_checkpoint_resume_and_force_failure_restore_full_state(thermostat):
    barostat, dynamics, state, _, model = make(friction=.5)
    initial, piston = state.copy(), barostat.snapshot()
    rng = copy.deepcopy(dynamics.rng.bit_generator.state)
    stats = copy.deepcopy(dynamics.propagation_stats)
    # Initial pressure succeeds; a physical force call after dilation fails.
    model.fail_at = 3
    with pytest.raises(RuntimeError, match="injected"):
        barostat.step(state, dynamics, thermostat=thermostat)
    np.testing.assert_array_equal(state.q, initial.q)
    np.testing.assert_array_equal(state.p, initial.p)
    assert (state.step, state.time) == (initial.step, initial.time)
    assert barostat.snapshot() == piston
    assert dynamics.rng.bit_generator.state == rng
    assert dynamics.propagation_stats == stats
    np.testing.assert_array_equal(dynamics.potential.cell, piston["cell"])
    model.fail_at = None
    for _ in range(12):
        barostat.step(state, dynamics, thermostat=thermostat)
    midpoint, midpoint_piston = state.copy(), barostat.snapshot()
    midpoint_rng = copy.deepcopy(dynamics.rng.bit_generator.state)
    for _ in range(9):
        barostat.step(state, dynamics, thermostat=thermostat)
    other_b, other_d, _, _, _ = make(friction=.5)
    other_b.restore(midpoint_piston)
    other_d.rng.bit_generator.state = midpoint_rng
    for _ in range(9):
        other_b.step(midpoint, other_d, thermostat=thermostat)
    np.testing.assert_array_equal(midpoint.q, state.q)
    np.testing.assert_array_equal(midpoint.p, state.p)
    assert other_b.snapshot() == barostat.snapshot()


def test_cell_change_invalidates_same_geometry_cache():
    barostat, dynamics, state, _, _ = make()
    state.q[:] = 0.
    with dynamics.potential.step():
        initial = dynamics.potential.evaluate(state.q)
        snapshot = barostat.snapshot()
        barostat.restore({**snapshot, "cell": (2 * barostat.cell).tolist()})
        changed = dynamics.potential.evaluate(state.q)
        np.testing.assert_allclose(changed.energy, initial.energy / 8)
        np.testing.assert_allclose(changed.virial, initial.virial / 8)


def test_missing_virial_is_not_reconstructed_from_coordinates():
    class NoVirial:
        metadata = {"periodic": True}

        def evaluate(self, q, *, cell, pbc):
            return PESResult(np.zeros(len(q)), np.zeros_like(q))

    _, _, state, system, _ = make()
    provider = PythonPotential(NoVirial(), cell=system.cell)
    barostat = BZPBarostat(provider, system, 4., 4, .02, 2.)
    with pytest.raises(PotentialError, match="virial"):
        barostat.pressure(state)


def test_npt_scope_and_checkpoint_guards():
    barostat, dynamics, state, system, _ = make()
    original = barostat.snapshot()
    for changed in ({"momentum": np.nan}, {"method": "mttk"},
                    {"cell": np.diag([2., 3., 4.]).tolist()}, {"extra": 1}):
        with pytest.raises(ValueError):
            barostat.restore({**original, **changed})
        assert barostat.snapshot() == original
    for parameters in ({"pressure_au": True}, {"pressure_au": np.complex128(1)},
                       {"tau_au": 0}, {"friction_au": -1}):
        with pytest.raises(ValueError):
            BZPBarostat(FreePotential(), system, 4., 4, **{
                "pressure_au": .02, "tau_au": 2., **parameters})
    for kwargs in ({"pbc": [True, True, False]}, {"fixed_atoms": (0,)}):
        wrong = System(system.symbols, system.masses, system.geometry, cell=system.cell, **kwargs)
        with pytest.raises(ValueError):
            BZPBarostat(FreePotential(), wrong, 4., 4, .02, 2.)
    dynamics.loop_backend = "native"
    with pytest.raises(ValueError, match="loop_backend"):
        barostat.step(state, dynamics)


def test_bound_python_pes_requires_explicit_cell_update_contract():
    class Bound:
        metadata = {"supports_pbc": True}

        def evaluate(self, q):
            return PESResult(np.zeros(len(q)), np.zeros_like(q), virial=np.zeros((len(q), 3, 3)))

        def bind_system(self, system):
            self.cell = system.cell.copy()

    _, _, _, system, _ = make()
    bound = Bound()
    provider = PythonPotential(bound, cell=system.cell)
    provider.bind_system(system)
    with pytest.raises(PotentialError, match="set_cell"):
        BZPBarostat(provider, system, 4., 4, .02, 2.)
    bound.set_cell = lambda cell: setattr(bound, "cell", cell.copy())
    barostat = BZPBarostat(provider, system, 4., 4, .02, 2.)
    barostat.restore({**barostat.snapshot(), "cell": (system.cell * 1.1).tolist()})
    np.testing.assert_array_equal(bound.cell, barostat.cell)


@pytest.mark.parametrize("bound", [False, True])
def test_python_cell_updates_obey_provider_units_and_bound_au_contract(bound):
    from pyrpmd.constants import BOHR_ANGSTROM, EV_TO_AU

    class AngstromEV:
        metadata = {"supports_pbc": True} if bound else {"periodic": True}

        def bind_system(self, system):
            self.cell_bohr = system.cell.copy()

        def set_cell(self, cell_bohr):
            self.cell_bohr = cell_bohr.copy()

        def evaluate(self, q, *, cell=None, pbc=None):
            lattice = self.cell_bohr * BOHR_ANGSTROM if bound else cell
            self.evaluated_cell_angstrom = lattice.copy()
            self.evaluated_q_angstrom = q.copy()
            energy = 1.0 / abs(np.linalg.det(lattice))
            return PESResult(np.full(len(q), energy), np.zeros_like(q),
                             virial=np.repeat((np.eye(3) * energy)[None], len(q), axis=0))

    _, _, state, system, _ = make()
    obj = AngstromEV()
    provider = PythonPotential(obj, cell=system.cell,
                               units={"positions": "angstrom", "energy": "eV"}, derivative="gradient")
    provider.bind_system(system)
    piston = BZPBarostat(provider, system, 4., 4, .02, 2.)
    piston.restore({**piston.snapshot(), "cell": (system.cell * 1.2).tolist()})
    state.p[:] = 0
    expected = EV_TO_AU / (piston.volume**2 * BOHR_ANGSTROM**3)
    assert piston.pressure(state) == pytest.approx(expected)
    np.testing.assert_allclose(obj.evaluated_cell_angstrom, system.cell * 1.2 * BOHR_ANGSTROM)
    np.testing.assert_allclose(obj.evaluated_q_angstrom, state.q * BOHR_ANGSTROM)
    if bound:
        np.testing.assert_array_equal(obj.cell_bohr, piston.cell)


def test_snapshot_validation_needs_no_provider_and_copies_payload():
    _, _, _, system, _ = make()
    snapshot = {"method": "bzp", "cell": system.cell.tolist(), "momentum": 1.25}
    validated = validate_bzp_snapshot(snapshot, system.cell)
    assert validated == snapshot
    snapshot["cell"][0][0] = 99
    assert validated["cell"] == system.cell.tolist()
    with pytest.raises(ValueError, match="isotropic"):
        validate_bzp_snapshot(snapshot, system.cell)


@pytest.mark.parametrize("storage", ["memory", "stream"])
def test_materials_observed_pressure_failure_restores_piston_and_rng(tmp_path, monkeypatch, storage):
    from pyrpmd.workflows.materials import _sample_replica, _read_stream

    _, _, _, system, _ = make()
    config = {"task": "pimd", "beta": 4., "beads": 4, "dt_au": .02, "native": False,
              "seed": 818, "potential": {"provider": "builtin", "name": "free"},
              "dynamics": {"loop_backend": "python"},
              "barostat": {"pressure": .02, "time_constant": 2., "friction": .5},
              "sampling": {"equilibration_steps": 0, "production_steps": 12,
                           "sample_interval": 1, "replicas": 1, "thermostat": "pile-l"},
              "output": {"checkpoint_steps": 2, "trajectory_storage": storage, "chunk_frames": 2}}
    task = {"config": config, "system": system, "replica": 0}
    reference = _sample_replica({**task, "output": tmp_path / "reference"})
    original = BZPBarostat.pressure

    def fail(self, state):
        if state.step == 4:
            raise RuntimeError("pressure observable failed after accepted step")
        return original(self, state)

    with monkeypatch.context() as context:
        context.setattr(BZPBarostat, "pressure", fail)
        with pytest.raises(RuntimeError, match="pressure observable"):
            _sample_replica({**task, "output": tmp_path / "interrupted"})
    saved = _read_stream(config, system, tmp_path / "interrupted", 0)
    assert saved["elapsed"] == 3
    np.testing.assert_array_equal(saved["barostat"]["cell"], reference["cells"][3])
    resumed = _sample_replica({**task, "output": tmp_path / "interrupted"})
    for key in ("positions", "velocities", "observables", "cells", "volumes", "pressures"):
        np.testing.assert_array_equal(resumed[key], reference[key])
    assert resumed["barostat"] == reference["barostat"]
    assert resumed["rng"] == reference["rng"]


@pytest.mark.parametrize("beads", [1, 4])
def test_free_gas_samples_exact_npt_volume_distribution(beads):
    # Integrating e^-beta*PV over N particle centroids gives
    # p(V) proportional to V^N exp(-beta*P*V), a Gamma(N+1, 1/beta/P).
    # N=2 distinguishes the correct +1/beta correction from its omission
    # or the constrained-COM +2/beta convention. Internal modes do not
    # change the volume exponent; P=4 catches wrong bead normalization.
    n, beta, pressure = 2, 2., .5
    system = System(("H",) * n, np.ones(n), np.zeros((n, 3)), cell=np.eye(3) * 3**(1/3))
    provider = FreePotential()
    dynamics = Dynamics(provider, system.masses, beta, beads, .01, native=False,
                        seed=1234 + beads, loop_backend="python", integrator="normal-mode-bab")
    barostat = BZPBarostat(provider, system, beta, beads, pressure, .4)
    state = State(np.repeat(system.geometry[None], beads, axis=0), dynamics.sample_momenta())
    volumes, pressures = [], []
    for i in range(26000):
        barostat.step(state, dynamics, thermostat="pile-l", gamma_centroid=5.)
        if i >= 2000 and i % 10 == 0:
            volumes.append(barostat.volume)
            pressures.append(barostat.pressure(state))
    expected_mean = (n + 1) / (beta * pressure)
    expected_variance = (n + 1) / (beta * pressure)**2
    assert np.mean(volumes) == pytest.approx(expected_mean, rel=.08)
    assert np.var(volumes) == pytest.approx(expected_variance, rel=.25)
    assert np.mean(pressures) == pytest.approx(pressure, rel=.12)


def test_ipi_cell_update_preserves_wire_orientation():
    from pyrpmd.potentials.socket import IPIPotential

    # No connection is opened: this tests the POSDATA cell payload contract.
    provider = IPIPotential.__new__(IPIPotential)
    provider.pbc = np.ones(3, dtype=bool)
    cell = np.array([[4., 0, 0], [1., 5., 0], [.2, -.4, 6.]])
    provider.set_cell(cell)
    np.testing.assert_array_equal(provider._wire_cell, cell.T)
    np.testing.assert_allclose(provider._wire_inverse @ provider._wire_cell, np.eye(3), atol=1e-14)


@pytest.mark.skipif(importlib.util.find_spec("ase") is None, reason="optional ASE")
def test_ase_cell_update_updates_physical_stress_without_scaling_positions():
    from ase.calculators.lj import LennardJones
    from pyrpmd.potentials.optional import ASEPotential
    from pyrpmd.constants import BOHR_ANGSTROM

    cell = np.eye(3) * 8.
    provider = ASEPotential(["Ar", "Ar"], LennardJones(), cell=cell)
    q = np.array([[[.1, .2, .3], [2.4, .2, .3]]])
    first = provider.evaluate(q)
    provider.set_cell(cell * 1.1)
    second = provider.evaluate(q)
    np.testing.assert_allclose(provider.atoms.cell.array, cell * 1.1 * BOHR_ANGSTROM)
    np.testing.assert_allclose(provider.atoms.positions, q[0] * BOHR_ANGSTROM)
    assert second.virial.shape == first.virial.shape == (1, 3, 3)
    assert np.isfinite(second.virial).all()
