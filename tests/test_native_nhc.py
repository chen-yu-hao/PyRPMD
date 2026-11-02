"""Native chains preserve the established NHC splitting and full restart state."""
import copy

import numpy as np
import pytest

pytest.importorskip("pyrpmd._thermal")
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.native_ensemble import NativeEnsemble, _state_digest, ensemble_from_config
from pyrpmd.potentials import HarmonicPotential
from pyrpmd.thermostats import NoseHooverChain


def settings(beads=4, **kwargs):
    rng = np.random.default_rng(712)
    q, p = rng.normal(scale=.3, size=(2, beads, 3, 3))
    q[:, 2] = [.2, -.1, .3]
    p[:, 2] = 0.
    args = dict(q=q, p=p, masses=[1., 2., 3.], beta=2., dt=.01,
                fixed_atoms=[2], potential={"kind": "harmonic", "k": .7},
                thermostat={"mode": "nhc", "tdamp": .4,
                            "chain_length": 3, "substeps": 2})
    args.update(kwargs)
    return args


@pytest.mark.parametrize("beads", [1, 3, 8])
@pytest.mark.parametrize("integrator", ["cayley-bcb", "normal-mode-bab"])
@pytest.mark.parametrize("chain_length", [1, 3])
def test_native_nhc_matches_existing_chain(beads, integrator, chain_length):
    args = settings(beads, integrator=integrator)
    args["thermostat"]["chain_length"] = chain_length
    native = NativeEnsemble(**args)
    dyn = Dynamics(HarmonicPotential(k=.7), args["masses"], args["beta"],
                   beads, args["dt"], fixed_atoms=[2], fixed_positions=args["q"][0],
                   native=False, integrator=integrator)
    reference = State(args["q"], args["p"])
    bath = NoseHooverChain(dyn, tdamp=.4, chain_length=chain_length, substeps=2)
    for _ in range(80):
        bath.step(reference)
    result = native.run(80)
    state = native.state()
    np.testing.assert_allclose(state["q"], reference.q, atol=3e-12, rtol=3e-12)
    np.testing.assert_allclose(state["p"], reference.p, atol=3e-12, rtol=3e-12)
    np.testing.assert_allclose(state["aux"][0], bath.eta, atol=3e-12, rtol=3e-12)
    np.testing.assert_allclose(state["aux"][1], bath.momentum, atol=3e-12, rtol=3e-12)
    np.testing.assert_array_equal(state["q"][:, 2], args["q"][:, 2])
    np.testing.assert_array_equal(state["p"][:, 2], 0.)
    assert result["python_step_callbacks"] == 0
    assert result["splitting"] == ("OBCBO" if integrator == "cayley-bcb" else "OBABO")


@pytest.mark.parametrize("beads", [1, 3, 8])
def test_native_nhc_exact_restart_and_chain_identity(beads, tmp_path):
    args = settings(beads)
    whole, part = NativeEnsemble(**args), NativeEnsemble(**args)
    expected = whole.run(91, stride=7, record_centroids=True)
    first = part.run(23, stride=7, record_centroids=True)
    part.save(tmp_path / "nhc.npz")
    resumed = NativeEnsemble(**args)
    resumed.load(tmp_path / "nhc.npz")
    last = resumed.run(68, stride=7, record_centroids=True)
    np.testing.assert_array_equal(np.concatenate([first["samples"], last["samples"]]), expected["samples"])
    np.testing.assert_array_equal(np.concatenate([first["centroids"], last["centroids"]]), expected["centroids"])
    assert resumed.state()["digest"] == whole.state()["digest"]
    other_args = copy.deepcopy(args)
    other_args["thermostat"]["substeps"] = 3
    with pytest.raises(ValueError, match="physics"):
        NativeEnsemble(**other_args).restore(part.state())


def test_native_nhc_state_reversal_and_second_order_extended_energy():
    def integrate(dt, nsteps):
        e = NativeEnsemble(**settings(1, dt=dt))
        first = e.state()
        masses = np.array([1., 2., 3.])
        chain_mass = np.array([6., 1., 1.]) * .5 * .4**2

        def energy(state):
            physical = .5*np.sum(state["p"]**2/masses[None, :, None]) + np.sum(state["energy"])
            eta, momentum = state["aux"]
            return physical + .5*(6*eta[0] + eta[1:].sum()) + .5*np.sum(momentum**2/chain_mass)

        initial = energy(first)
        error = 0.
        for _ in range(nsteps):
            e.run(1)
            error = max(error, abs(energy(e.state()) - initial))
        reversed_state = e.state()
        reversed_state["p"] *= -1
        reversed_state["aux"][1] *= -1
        reversed_state["digest"] = _state_digest(reversed_state)
        e.restore(reversed_state)
        e.run(nsteps)
        final = e.state()
        np.testing.assert_allclose(final["q"], first["q"], atol=3e-12)
        np.testing.assert_allclose(final["p"], -first["p"], atol=3e-12)
        np.testing.assert_allclose(final["aux"], first["aux"], atol=3e-12)
        return error

    coarse, fine = integrate(.02, 100), integrate(.01, 200)
    assert coarse < 2e-3
    assert 3.5 < coarse/fine < 4.5


def test_native_nhc_overflow_rolls_back_chain_and_rng():
    e = NativeEnsemble(**settings(1, dt=10., thermostat={"mode": "nhc", "tdamp": .001}))
    before = e.state()
    with pytest.raises((ValueError, RuntimeError), match="nonfinite"):
        e.run(10)
    assert e.state()["digest"] == before["digest"]


@pytest.mark.parametrize("field,value", [("chain_length", 0), ("chain_length", 65),
                                          ("substeps", 0), ("substeps", True),
                                          ("tdamp", 0.), ("tdamp", float("inf"))])
def test_native_nhc_configuration_validation(field, value):
    thermo = {"mode": "nhc", field: value}
    with pytest.raises(ValueError):
        NativeEnsemble(**settings(thermostat=thermo))


def test_nhc_config_parses_damping_units():
    e = ensemble_from_config({"positions": [[0., 0., 0.]], "masses": [1.], "beads": 1,
                              "beta": 2., "timestep": ".001 fs",
                              "thermostat": {"mode": "nhc", "tdamp": "20 fs"}})
    from pyrpmd.units import parse_quantity
    assert e._cfg["tau"] == parse_quantity("20 fs", "time")


@pytest.mark.parametrize("mode", ["nhc", "svr", "gle", "pile-l"])
def test_bath_operator_restart_and_no_force_calls(mode):
    thermo = {"mode": mode}
    if mode == "gle":
        thermo["gle"] = {"A": [[1., 2.], [-2., 1.]]}
    args = settings(3, thermostat=thermo, potential=None)
    a, b = NativeEnsemble(**args), NativeEnsemble(**args)
    pa, pb = args["p"].copy(), args["p"].copy()
    original_q = args["q"].copy()
    for _ in range(5):
        a.apply_thermostat(args["q"], pa)
        b.apply_thermostat(args["q"], pb)
    saved = b.state()
    b = NativeEnsemble(**args)
    b.restore(saved)
    for _ in range(4):
        a.apply_thermostat(args["q"], pa)
        b.apply_thermostat(args["q"], pb)
    np.testing.assert_array_equal(pa, pb)
    np.testing.assert_array_equal(args["q"], original_q)
    assert a.state()["digest"] == b.state()["digest"]
    assert a.state()["force_evaluations"] == 1


def test_bath_operator_gle_duration_mismatch_is_transactional():
    e = NativeEnsemble(**settings(thermostat={"mode": "gle", "gle": {"A": [[1.]]}}))
    before = e.state()
    p = before["p"].copy()
    with pytest.raises(ValueError, match="duration"):
        e.apply_thermostat(before["q"], p, 2*e.dt)
    np.testing.assert_array_equal(p, before["p"])
    assert before["digest"] == e.state()["digest"]


def test_set_state_refreshes_forces_and_retains_bath():
    e = NativeEnsemble(**settings())
    e.run(5)
    before = e.state()
    q, p = before["q"].copy(), before["p"].copy()
    q[:, :2] += .4
    e.set_state(q, p, 11)
    after = e.state()
    np.testing.assert_array_equal(after["q"], q)
    np.testing.assert_allclose(after["force"], -.7*q)
    np.testing.assert_allclose(after["energy"], .35*np.sum(q*q, axis=(1, 2)))
    np.testing.assert_array_equal(after["aux"], before["aux"])
    assert after["rng"] == before["rng"]
    assert after["step"] == 11
    assert after["force_evaluations"] == before["force_evaluations"]+1
    q[:, 2, 0] += 1.
    with pytest.raises(ValueError, match="fixed"):
        e.set_state(q, p, 12)
    assert e.state()["digest"] == after["digest"]
