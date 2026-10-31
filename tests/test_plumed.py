"""PLUMED clock, atomic units and full-ring force-contract regressions."""

import copy
import importlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from pyrpmd.constants import AMU_TO_AU, ATOMIC_TIME_S, BOHR_M, KCAL_MOL_TO_AU
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.plumed import PlumedBias
from pyrpmd.potentials.builtin import HarmonicPotential


class FakeKernel:
    """Official command protocol with a harmonic bias in PLUMED input units.

    Conversion is performed in this stand-in independently of the adapter,
    so the tests exercise the direction of every MD conversion factor.
    """

    def __init__(self, *, k=800.0, center=0.03, changing=False, api=10):
        self.values = {}
        self.k, self.center, self.changing, self.api = k, center, changing, api
        self.calls = []
        self.calculations = []
        self.updates = []
        self.finalizations = 0
        self.needs_energy = False
        self.fail = None
        self.stop_after = None

    def cmd(self, command, value=None):
        self.calls.append(command)
        if self.fail == command and command != "update":
            raise RuntimeError("injected kernel failure")
        if command == "getApiVersion":
            value[0] = self.api
        elif command == "isEnergyNeeded":
            value[0] = int(self.needs_energy)
        elif command == "performCalcNoUpdate":
            x = self.values["setPositions"] * self.values["setMDLengthUnits"]
            k = self.k * (1 + 0.25 * len(self.updates) if self.changing else 1)
            d = x - self.center
            self.bias = 0.5 * k * np.sum(d * d) / self.values["setMDEnergyUnits"]
            force = -k * d * self.values["setMDLengthUnits"] / self.values["setMDEnergyUnits"]
            self.values["setForces"][:] += force
            self.values["setVirial"][:] += -x.T @ (-k * d) / self.values["setMDEnergyUnits"]
            self.calculations.append((self.values["setStep"], len(self.updates), x.copy()))
        elif command == "getBias":
            value[0] = self.bias
        elif command == "update":
            self.updates.append(self.values["setStep"])
            if self.stop_after == len(self.updates):
                self.values["setStopFlag"][0] = 1
            if self.fail == command:
                raise RuntimeError("injected partially completed update")
        elif command.startswith("set") or command == "readInputLines":
            self.values[command] = value

    def finalize(self):
        self.finalizations += 1


def bridge(beads=4, *, kernel=None, **kwargs):
    return PlumedBias(
        "x: POSITION ATOM=1\nb: RESTRAINT ARG=x.x AT=0 KAPPA=1",
        [2.0], 3.0, beads, 0.02,
        _kernel=FakeKernel() if kernel is None else kernel, **kwargs,
    )


def trajectory(beads=4, potential=None, seed=114):
    dyn = Dynamics(HarmonicPotential(0.2) if potential is None else potential,
                   [2.0], 3.0, beads, 0.02, seed=seed, native=False)
    q = np.arange(beads * 3, dtype=float).reshape(beads, 1, 3) * 0.01
    return dyn, State(q, dyn.sample_momenta())


@pytest.mark.parametrize("beads", [1, 4, 16])
def test_energy_force_atomic_unit_conversion_and_ring_scaling(beads):
    kernel = FakeKernel()
    with bridge(beads, kernel=kernel) as bias:
        q = np.random.default_rng(2).normal(size=(beads, 1, 3))
        result = bias.evaluate(q)
        eunit = 4.184 / KCAL_MOL_TO_AU
        lunit = BOHR_M / 1e-9
        delta = np.mean(q, axis=0) * lunit - kernel.center
        expected_physical = 0.5 * kernel.k * np.sum(delta**2) / eunit
        assert result.physical_energy == pytest.approx(expected_physical)
        assert result.energy == pytest.approx(beads * expected_physical)
        np.testing.assert_allclose(result.forces,
                                   np.broadcast_to(-kernel.k * delta * lunit / eunit, q.shape))
        numerical = np.zeros_like(q)
        for ix in np.ndindex(q.shape):
            plus, minus = q.copy(), q.copy()
            plus[ix] += 1e-6
            minus[ix] -= 1e-6
            numerical[ix] = -(bias.evaluate(plus).energy - bias.evaluate(minus).energy) / 2e-6
        np.testing.assert_allclose(result.forces, numerical, atol=3e-11, rtol=1e-7)
        assert kernel.values["setMDMassUnits"] == pytest.approx(1 / AMU_TO_AU)
        assert kernel.values["setMDTimeUnits"] == pytest.approx(ATOMIC_TIME_S / 1e-12)
        assert kernel.values["setKbT"] == pytest.approx(1 / 3)
        assert kernel.values["setTimestep"] == 0.02
        np.testing.assert_array_equal(kernel.values["setMasses"], [2.0])
        assert kernel.updates == []
    assert kernel.finalizations == 1


@pytest.mark.parametrize("thermostat", [None, "andersen", "pile-l"])
@pytest.mark.parametrize("beads", [1, 4])
def test_one_update_at_endpoint_and_no_diagnostic_deposition(thermostat, beads):
    kernel = FakeKernel(changing=True)
    dyn, state = trajectory(beads)
    with bridge(beads, kernel=kernel) as bias:
        initial = state.q.copy()
        bias.evaluate(state.q)
        bias.step(dyn, state, thermostat=thermostat)
        assert state.step == bias.completed_step == 1
        assert state.time == 0.02
        assert kernel.updates == [1]
        assert [(s, h) for s, h, _ in kernel.calculations] == [(0, 0), (0, 0), (1, 0)]
        np.testing.assert_allclose(kernel.calculations[1][2], initial.mean(axis=0) * BOHR_M / 1e-9,
                                   rtol=5e-16, atol=0)
        np.testing.assert_allclose(kernel.calculations[2][2], state.q.mean(axis=0) * BOHR_M / 1e-9,
                                   rtol=5e-16, atol=0)
        saved = bias.evaluate(state.q)
        for _ in range(4):
            again = bias.evaluate(state.q)
            np.testing.assert_array_equal(again.forces, saved.forces)
        assert kernel.updates == [1]
        bias.step(dyn, state, thermostat=thermostat)
        assert kernel.updates == [1, 2]
        assert [(s, h) for s, h, _ in kernel.calculations[-2:]] == [(1, 1), (2, 1)]
        assert dyn.propagation_stats["python_steps"] == 2


def test_bridge_static_bias_matches_direct_existing_dynamics():
    kernel = FakeKernel()
    dyn, state = trajectory()
    reference_dyn, reference_state = trajectory()
    eunit, lunit = 4.184 / KCAL_MOL_TO_AU, BOHR_M / 1e-9

    class AnalyticBias:
        def energy_force(self, q):
            d = q.mean(axis=0) * lunit - kernel.center
            return (len(q) * 0.5 * kernel.k * np.sum(d*d) / eunit,
                    np.broadcast_to(-kernel.k*d*lunit/eunit, q.shape).copy())

    with bridge(kernel=kernel) as bias:
        for _ in range(20):
            bias.step(dyn, state)
            reference_dyn.step(reference_state, AnalyticBias())
        np.testing.assert_array_equal(state.q, reference_state.q)
        np.testing.assert_array_equal(state.p, reference_state.p)


def test_cell_mass_charge_buffers_and_outputs_are_owned():
    cell = np.array([[6.0, 0.0, 0.0], [0.4, 7.0, 0.0], [0.2, 0.1, 8.0]])
    charge = np.array([-0.2])
    kernel = FakeKernel()
    with bridge(kernel=kernel, cell=cell, charges=charge) as bias:
        cell[:] = -200.0
        charge[:] = 999.0
        q = np.ones((4, 1, 3))
        result = bias.evaluate(q)
        np.testing.assert_array_equal(kernel.values["setBox"],
                                      [[6, 0, 0], [0.4, 7, 0], [0.2, 0.1, 8]])
        np.testing.assert_array_equal(kernel.values["setCharges"], [-0.2])
        assert np.linalg.norm(result.plumed_virial) > 0
        original = result.forces.copy()
        q[:] = 12.0
        bias.evaluate(q)
        np.testing.assert_array_equal(result.forces, original)
        assert not np.shares_memory(result.forces, kernel.values["setForces"])
    with bridge(cell=np.eye(3), pbc=[False, False, False]) as bias:
        bias.evaluate(np.zeros((4, 1, 3)))
        np.testing.assert_array_equal(bias._kernel.values["setBox"], np.zeros((3, 3)))


def test_pes_failure_restores_state_rng_and_does_not_update():
    class FailingPES(HarmonicPotential):
        def __init__(self):
            super().__init__(0.2)
            self.calls = 0

        def evaluate(self, q):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("injected endpoint PES failure")
            return super().evaluate(q)

    dyn, state = trajectory(potential=FailingPES())
    old_state, old_rng = state.copy(), copy.deepcopy(dyn.rng.bit_generator.state)
    kernel = FakeKernel()
    with bridge(kernel=kernel) as bias:
        with pytest.raises(RuntimeError, match="endpoint PES"):
            bias.step(dyn, state, thermostat="pile-l")
        np.testing.assert_array_equal(state.q, old_state.q)
        np.testing.assert_array_equal(state.p, old_state.p)
        assert state.step == old_state.step == bias.completed_step == 0
        assert dyn.rng.bit_generator.state == old_rng
        assert kernel.updates == []
        # Failure came from PES, not the opaque kernel; this retry is safe.
        bias.step(dyn, state, thermostat="pile-l")
        assert state.step == 1 and kernel.updates == [1]


@pytest.mark.parametrize("command", ["performCalcNoUpdate", "update"])
def test_kernel_failure_poisoning_is_explicit_and_md_state_is_transactional(command):
    dyn, state = trajectory()
    old_state, old_rng = state.copy(), copy.deepcopy(dyn.rng.bit_generator.state)
    kernel = FakeKernel()
    kernel.fail = command
    with bridge(kernel=kernel) as bias:
        with pytest.raises(RuntimeError, match="injected"):
            bias.step(dyn, state, thermostat="andersen")
        np.testing.assert_array_equal(state.q, old_state.q)
        np.testing.assert_array_equal(state.p, old_state.p)
        assert dyn.rng.bit_generator.state == old_rng
        assert state.step == bias.completed_step == 0
        with pytest.raises(RuntimeError, match="failed and cannot be reused"):
            bias.step(dyn, state)
        with pytest.raises(RuntimeError, match="failed and cannot be reused"):
            bias.evaluate(state.q)


def test_unsupported_energy_actions_fail_before_calculation():
    kernel = FakeKernel()
    kernel.needs_energy = True
    with bridge(kernel=kernel) as bias:
        with pytest.raises(ValueError, match="ENERGY-dependent"):
            bias.evaluate(np.zeros((4, 1, 3)))
        assert kernel.calculations == kernel.updates == []


def test_direct_dynamics_misuse_clock_mismatch_and_mutated_trajectory_rejected():
    dyn, state = trajectory()
    with bridge() as bias:
        with pytest.raises(RuntimeError, match="Use PlumedBias.step"):
            dyn.step(state, bias)
        state.step = 2
        with pytest.raises(ValueError, match="clock"):
            bias.step(dyn, state)
        state.step = 0
        state.time = 0.1
        with pytest.raises(ValueError, match="state.time"):
            bias.step(dyn, state)
        state.time = 0
        bias.step(dyn, state)
        state.q[0, 0, 0] += 0.1
        with pytest.raises(ValueError, match="one trajectory"):
            bias.step(dyn, state)


def test_stop_signal_is_observed_after_committing_endpoint():
    kernel = FakeKernel()
    kernel.stop_after = 1
    dyn, state = trajectory()
    with bridge(kernel=kernel) as bias:
        bias.step(dyn, state)
        assert bias.stop_requested and state.step == 1
        with pytest.raises(RuntimeError, match="requested a stop"):
            bias.step(dyn, state)
        assert kernel.updates == [1]


@pytest.mark.parametrize("field", ["q", "p"])
def test_caller_readonly_state_is_rejected_before_irreversible_update(field):
    dyn, state = trajectory()
    getattr(state, field).flags.writeable = False
    kernel = FakeKernel()
    with bridge(kernel=kernel) as bias:
        with pytest.raises(ValueError, match="writable"):
            bias.step(dyn, state)
        assert kernel.calculations == kernel.updates == []
        assert bias.completed_step == state.step == 0


def test_close_waits_for_an_active_kernel_call():
    entered, release, close_started, finalized = (threading.Event() for _ in range(4))

    class BlockingKernel(FakeKernel):
        def cmd(self, command, value=None):
            if command == "performCalcNoUpdate":
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("test did not release the kernel")
            return super().cmd(command, value)

        def finalize(self):
            super().finalize()
            finalized.set()

    kernel = BlockingKernel()
    bias = bridge(kernel=kernel)

    def close():
        close_started.set()
        bias.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        evaluate_future = pool.submit(bias.evaluate, np.zeros((4, 1, 3)))
        try:
            assert entered.wait(5)
            close_future = pool.submit(close)
            assert close_started.wait(5)
            assert not finalized.wait(0.1)
        finally:
            release.set()
        assert np.isfinite(evaluate_future.result(timeout=5).energy)
        close_future.result(timeout=5)
    assert finalized.is_set() and kernel.finalizations == 1


def test_optional_import_missing_api_cleanup_and_context_manager(monkeypatch):
    original = importlib.import_module

    def unavailable(name):
        if name == "plumed":
            raise ImportError("not installed")
        return original(name)

    monkeypatch.setattr(importlib, "import_module", unavailable)
    with pytest.raises(ImportError, match="Optional PLUMED bridge requires"):
        PlumedBias("x: POSITION ATOM=1", [2], 3, 4, 0.02)
    kernel = FakeKernel(api=7)
    with pytest.raises(RuntimeError, match="API >= 8"):
        bridge(kernel=kernel)
    assert kernel.finalizations == 1
    kernel = FakeKernel()
    bias = bridge(kernel=kernel)
    with bias:
        pass
    bias.close()
    assert kernel.finalizations == 1
    with pytest.raises(RuntimeError, match="closed"):
        bias.evaluate(np.zeros((4, 1, 3)))


def test_explicit_native_restart_and_file_input_preserve_clock(tmp_path):
    script = tmp_path / "bias.dat"
    script.write_text("d: DISTANCE ATOMS=1,2\n# retained multiline input\n")
    kernel = FakeKernel()
    with PlumedBias.from_file(script, [2], 3, 4, 0.02, start_step=10,
                              restart=True, _kernel=kernel) as bias:
        assert kernel.values["readInputLines"] == script.read_text()
        assert kernel.values["setRestart"] == 1
        dyn, state = trajectory()
        state.step, state.time = 10, 0.2
        bias.step(dyn, state)
        assert kernel.updates == [11]


@pytest.mark.parametrize("kwargs", [
    {"pbc": [True, False, True], "cell": np.eye(3)},
    {"charges": [float("nan")]},
    {"charges": [1.0, 2.0]},
    {"start_step": True},
    {"restart": "yes"},
])
def test_invalid_bridge_configuration_rejected_before_kernel_commands(kwargs):
    kernel = FakeKernel()
    with pytest.raises(ValueError):
        bridge(kernel=kernel, **kwargs)
    assert kernel.calls == []


def _live_kernel():
    required = os.environ.get("PYRPMD_REQUIRE_PLUMED") == "1"
    try:
        module = importlib.import_module("plumed")
    except ImportError as exc:
        if required:
            pytest.fail(f"PYRPMD_REQUIRE_PLUMED=1 but the Python wrapper is unavailable: {exc}", pytrace=False)
        pytest.skip("optional PLUMED Python wrapper unavailable")
    try:
        kernel = module.Plumed()
    except RuntimeError as exc:
        if "not available" in str(exc):
            if required:
                pytest.fail(f"PYRPMD_REQUIRE_PLUMED=1 but libplumedKernel is unavailable: {exc}", pytrace=False)
            pytest.skip("optional libplumedKernel unavailable")
        raise
    return module, kernel


def test_required_runtime_flag_does_not_hide_missing_dependencies(monkeypatch):
    monkeypatch.setenv("PYRPMD_REQUIRE_PLUMED", "1")

    def missing(name):
        assert name == "plumed"
        raise ImportError("injected missing wrapper")

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(pytest.fail.Exception, match="Python wrapper is unavailable"):
        _live_kernel()

    class NoKernel:
        @staticmethod
        def Plumed():
            raise RuntimeError("PLUMED not available, check your PLUMED_KERNEL environment variable")

    monkeypatch.setattr(importlib, "import_module", lambda name: NoKernel)
    with pytest.raises(pytest.fail.Exception, match="libplumedKernel is unavailable"):
        _live_kernel()


def test_real_optional_plumed_distance_restraint(tmp_path):
    module, kernel = _live_kernel()
    # KAPPA=100 kJ/mol/nm^2, AT=0.1 nm, with no atom/bead index ambiguity.
    with PlumedBias(
        "d: DISTANCE ATOMS=1,2 NOPBC\nb: RESTRAINT ARG=d AT=0.1 KAPPA=100",
        [2.0, 3.0], 4.0, 4, 0.02, log_file=tmp_path / "plumed.log", _kernel=kernel,
    ) as bias:
        q = np.zeros((4, 2, 3))
        q[:, 1, 0] = 3.0
        q[:, 1, 1] = np.array([-0.1, 0.0, 0.2, -0.1])
        result = bias.evaluate(q)
        lunit, eunit = BOHR_M / 1e-9, 4.184 / KCAL_MOL_TO_AU
        delta = 3 * lunit - 0.1
        assert result.energy == pytest.approx(4 * 0.5 * 100 * delta**2 / eunit, rel=1e-10)
        np.testing.assert_allclose(result.forces[:, 1, 0], -100 * delta * lunit / eunit, rtol=1e-10)
        np.testing.assert_allclose(result.forces[:, 0, 0], 100 * delta * lunit / eunit, rtol=1e-10)
        np.testing.assert_allclose(result.forces[:, :, 1:], 0, atol=1e-14)
    hills, colvar = tmp_path / "HILLS", tmp_path / "COLVAR"
    with PlumedBias(
        f"d: DISTANCE ATOMS=1,2 NOPBC\n"
        f"m: METAD ARG=d SIGMA=0.02 HEIGHT=0.1 PACE=2 FILE={hills}\n"
        f"PRINT ARG=d,m.bias STRIDE=1 FILE={colvar}",
        [2.0, 3.0], 4.0, 4, 0.02, log_file=tmp_path / "metad.log", _kernel=module.Plumed(),
    ) as bias:
        dyn = Dynamics(HarmonicPotential(0.2), [2.0, 3.0], 4.0, 4, 0.02, native=False)
        state = State(q, np.zeros_like(q))
        for _ in range(6):
            bias.step(dyn, state)
            for _ in range(3):
                bias.evaluate(state.q)
    hill_rows = [line for line in hills.read_text().splitlines() if line.strip() and not line.startswith("#")]
    printed_rows = [line for line in colvar.read_text().splitlines() if line.strip() and not line.startswith("#")]
    assert len(hill_rows) == 3
    assert len(printed_rows) == 6
